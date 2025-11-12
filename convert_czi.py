#!/usr/bin/env python3
"""convert_czi.py — CZI → TIFF conversion utility

Highlights vs. previous version
- Single source of truth for config with validated defaults
- Safer Windows/WSL/Docker path mapping helpers
- Clear logging and summarized stats
- Robust dtype handling (uint8/uint16) and normalization
- Optional outputs (per-channel, stack, rgb preview, false-color composite, colored pages, OME-TIFF with colors)
- ImageJ hyperstack metadata support (YXC default)
"""

from __future__ import annotations

import os
import sys
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Optional backend(s) for CZI
try:
    from aicspylibczi import CziFile  # type: ignore
except Exception:
    try:
        from czifile import CziFile  # type: ignore
    except Exception:
        CziFile = None  # will error later with a clear message

from pathlib import Path
from skimage.transform import resize
from skimage import io as skio
from skimage import img_as_ubyte
from tqdm import tqdm

try:
    from src.config_archiver import save_config_snapshot
except Exception:
    # fallback if not available
    def save_config_snapshot(*args, **kwargs):
        pass

try:
    import tifffile as tiff  # richer metadata handling
except Exception:
    tiff = None  # we can still write with skimage as a fallback

CONFIG_PATH = "config_czi.yaml"

# Simple patterns to infer group/region from paths
REGION_TOKENS = ("lArc","lPVH","lDMH", "rArc","rPVH","rDMH", "Arc","5x","DMH", "5x_pro")
COND_TOKENS = ("old","adult", "pos", "neg", "all", "")

_pattern = re.compile(r"Input_(old|adult|pos|neg|all)_(lArc|lPVH|lDMH|rArc|rPVH|rDMH|Arc|5x|DMH|5x_pro)", re.IGNORECASE)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class CziConfig:
    input_root: str = r""
    output_base: str = r""
    channels: List[int] = field(default_factory=lambda: [0,1,2,3])
    target_size: Optional[int] = None                 # max side; None=original size
    per_channel_normalize: bool = True
    overwrite: bool = False

    # Saving options
    save_per_channel: bool = True
    save_stack_tiff: bool = True
    save_rgb_preview: bool = False
    rgb_channels: List[int] = field(default_factory=lambda: [0,1,2])
    dtype: str = "uint8"                             # "uint8"|"uint16"

    # Stack/metadata
    stack_layout: str = "YXC"                        # "YXC" | "CYX" (we write YXC by default)
    imagej_hyperstack: bool = True
    channel_names: List[str] = field(default_factory=list)

    # Color composites
    save_color_composite: bool = True
    channel_colors: List[str] = field(default_factory=lambda: ['#ff0000','#00ff00','#0000ff','#ffff00'])
    save_multichannel_colored_pages: bool = True
    save_ome_tiff_colors: bool = True
    verbose: bool = False

    def validate(self) -> None:
        if self.dtype not in {"uint8","uint16"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        self.channels = [int(c) for c in self.channels if isinstance(c, (int, np.integer)) or str(c).isdigit()]
        self.rgb_channels = [int(c) for c in self.rgb_channels if isinstance(c, (int, np.integer)) or str(c).isdigit()]
        if self.target_size is not None:
            self.target_size = int(self.target_size)
        # normalize layout string
        self.stack_layout = (self.stack_layout or "YXC").upper()
        if self.stack_layout not in {"YXC","CYX"}:
            self.stack_layout = "YXC"

    @classmethod
    def from_yaml(cls, path: Path) -> "CziConfig":
        data: Dict = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        cfg = cls(**{**cls().__dict__, **data})  # start with defaults, overlay file
        cfg.validate()
        return cfg

    def to_yaml(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)


# ------------------------------------------------------------
# Path helpers (Windows ↔︎ WSL ↔︎ Docker)
# ------------------------------------------------------------
def _convert_windows_to_wsl(p: str) -> str:
    if not isinstance(p, str):
        return p
    if len(p) >= 3 and p[1:3] == ':\\':
        drive = p[0].lower()
        rest = p[3:].replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    return p

def _map_path_for_container(p: str) -> str:
    """
    Map incoming paths for the current runtime (WSL):
    - Windows-style paths (e.g. C:\\...) -> /mnt/<drive>/...
    - Leave existing POSIX paths (/mnt/..., /home/..., etc.) unchanged
    Do NOT rewrite to /host_mnt (Docker-specific).
    """
    if not isinstance(p, str):
        return p
    # Windows path → WSL path
    if '\\' in p or (len(p) >= 3 and p[1:3] == ':\\'):
        return _convert_windows_to_wsl(p)
    # Already POSIX → leave as is
    return p


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def find_czi_files(base: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Walk base and group files into out[cond][region] lists."""
    files = []
    for root, _, names in os.walk(base):
        for n in names:
            if n.lower().endswith(".czi"):
                files.append(Path(root) / n)
    out: Dict[str, Dict[str, List[Path]]] = {}
    for czi in files:
        rel = str(czi)
        m = _pattern.search(rel)
        if not m:
            # try to infer from tokens
            parts_lower = [p.lower() for p in Path(rel).parts]
            group = "all"
            for tok in ("old","adult","pos","neg","all"):
                if any(tok in p for p in parts_lower):
                    group = tok; break
            region = "ALL"
            for tok in REGION_TOKENS:
                if any(tok.lower() in p for p in parts_lower):
                    region = tok.upper(); break
        else:
            group, region = m.group(1).lower(), m.group(2).upper()
        out.setdefault(group, {}).setdefault(region, []).append(czi)
    return out


def load_czi_to_CZYX(czi_path: str) -> np.ndarray:
    if CziFile is None:
        raise ImportError("CZI backend missing (install aicspylibczi or czifile)")
    czi = CziFile(czi_path)
    data = czi.read_image()
    if isinstance(data, tuple):
        arr, dims = data
    else:
        arr, dims = data, None

    arr = np.asarray(arr)

    if dims:
        # dims -> list of (axis_name, size)
        axes = [("C" if ax == "A" else ax) for ax, _ in dims]  # treat 'A' samples as channels
        sizes = [size for _, size in dims]
        desired = ("C", "Z", "Y", "X")
        keep = set(desired)

        # Drop axes that are not relevant (keep first index)
        for idx in reversed(range(len(axes))):
            ax_name = axes[idx]
            size = sizes[idx]
            if ax_name not in keep:
                if size > 1:
                    print(f"[WARN] Axis '{ax_name}' (size={size}) collapsed to first index for {Path(czi_path).name}")
                arr = np.take(arr, 0, axis=idx)
                axes.pop(idx)
                sizes.pop(idx)

        if "C" not in axes:
            arr = np.expand_dims(arr, axis=0)
            axes.insert(0, "C")

        # Reorder axes to C, Z, Y, X (dropping missing ones)
        order = [axes.index(ax) for ax in desired if ax in axes]
        arr = np.transpose(arr, axes=order)

        return arr

    # Fallback heuristic when dims metadata is unavailable
    arr = np.squeeze(arr)
    if arr.ndim >= 4:
        axis_sizes = [(i, s) for i, s in enumerate(arr.shape)]
        c_axis = min(axis_sizes, key=lambda kv: kv[1])[0]
        arr = np.moveaxis(arr, c_axis, 0)
    elif arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)
    else:
        arr = arr[None, ...]
    return arr


def ensure_uint(arr: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "uint16":
        if arr.dtype == np.uint16:
            return arr
        if arr.dtype == np.uint8:
            return (arr.astype(np.float32) / 255.0 * 65535.0 + 0.5).astype(np.uint16)
        m = float(arr.max()) or 1.0
        return (arr.astype(np.float32) / m * 65535.0 + 0.5).astype(np.uint16)
    # uint8
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    m = float(arr.max()) or 1.0
    return img_as_ubyte(arr.astype(np.float32) / m)


def normalize_stack(cyx: np.ndarray, as_uint8: bool) -> np.ndarray:
    """Per-channel percentile normalization to 0-255 if requested."""
    out = []
    for i in range(cyx.shape[0]):
        plane = cyx[i].astype(np.float32)
        lo, hi = np.percentile(plane, (2, 99.8))
        if hi <= lo:
            scaled = np.zeros_like(plane, dtype=np.uint8)
        else:
            scaled = np.clip((plane - lo) / (hi - lo), 0, 1) * 255.0
        out.append(scaled.astype(np.uint8))
    return np.stack(out, axis=0) if as_uint8 else np.stack(out, axis=0).astype(np.uint16) * 257


# Color helpers
def _parse_color(cstr: str) -> Tuple[float, float, float]:
    NAMED = {
        'red': (1,0,0), 'green': (0,1,0), 'blue': (0,0,1),
        'yellow': (1,1,0), 'cyan': (0,1,1), 'magenta': (1,0,1),
        'white': (1,1,1), 'orange': (1,0.5,0), 'purple': (0.5,0,0.5)
    }
    if not cstr:
        return (1,1,1)
    s = str(cstr).strip().lower()
    if s in NAMED:
        return NAMED[s]
    if s.startswith('#') and len(s) == 7:
        try:
            r = int(s[1:3], 16) / 255.0
            g = int(s[3:5], 16) / 255.0
            b = int(s[5:7], 16) / 255.0
            return (r,g,b)
        except Exception:
            return (1,1,1)
    return (1,1,1)

def _colorize_plane(plane: np.ndarray, color_str: str, out_dtype: str) -> np.ndarray:
    if plane.dtype == np.uint8:
        norm = plane.astype(np.float32) / 255.0
    elif plane.dtype == np.uint16:
        norm = plane.astype(np.float32) / 65535.0
    else:
        m = float(plane.max()) or 1.0
        norm = plane.astype(np.float32) / m
    r, g, b = _parse_color(color_str)
    rgb = np.stack([norm * r, norm * g, norm * b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    if out_dtype == 'uint16':
        return (rgb * 65535.0 + 0.5).astype(np.uint16)
    return (rgb * 255.0 + 0.5).astype(np.uint8)

def _rgba32_from_rgb(rgb: Tuple[float,float,float]) -> int:
    r,g,b = [int(255*max(0,min(1,x))) for x in rgb]
    a = 255
    return (a<<24) | (r<<16) | (g<<8) | b


# ------------------------------------------------------------
# Core processing
# ------------------------------------------------------------
def process_czi_file(inp_path: str, group: str, target_size: Optional[int], channels: List[int],
                     per_channel_normalize: bool, overwrite: bool, output_base: str, cfg: Optional[dict]=None, verbose: bool=False):
    if CziFile is None:
        return False, None, "missing_czi_backend"

    inp = Path(inp_path)

    # Output folder: <output_base>/<group>/<REGION>/
    region = "ALL"
    lower = inp.as_posix().lower()
    for tok in REGION_TOKENS:
        if tok.lower() in lower:
            region = tok.upper(); break

    out_dir = Path(output_base) / group / region
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if we already wrote a stack file and overwrite is False
    stack_marker = out_dir / f"{inp.stem}_stack.tif"
    if (not overwrite) and stack_marker.exists():
        if verbose:
            print(f"[SKIP] {inp.name} (exists)")
        return True, str(stack_marker), "exists"

    # Load CZI -> (C, Y, X)
    cyx = load_czi_to_CZYX(str(inp))
    # Collapse Z if present (take max)
    if cyx.ndim == 4:
        cyx = cyx.max(axis=1)

    # Subset channels if requested
    if channels:
        keep = [c for c in channels if 0 <= c < cyx.shape[0]]
        if keep:
            cyx = cyx[keep]

    # Resize (max side) if required
    if target_size and target_size > 0:
        h, w = cyx.shape[1:]
        scale = target_size / max(h, w)
        if scale != 1.0:
            new_h, new_w = int(round(h*scale)), int(round(w*scale))
            cyx = np.stack([resize(cyx[i], (new_h, new_w), preserve_range=True, anti_aliasing=True) for i in range(cyx.shape[0])], axis=0)

    # Normalization
    if per_channel_normalize:
        cyx = normalize_stack(cyx, as_uint8=(cfg or {}).get('dtype','uint8')=='uint8')

    # Ensure requested dtype
    out_dtype = (cfg or {}).get('dtype', 'uint8')
    cyx = ensure_uint(cyx, out_dtype)

    if verbose:
        print(f"[LOAD] {inp.name}: C={cyx.shape[0]} HxW={cyx.shape[-2:]} group={group} region={region}")

    # Save options
    save_per_channel = bool((cfg or {}).get('save_per_channel', True))
    save_stack_tiff = bool((cfg or {}).get('save_stack_tiff', True))
    save_rgb_preview = bool((cfg or {}).get('save_rgb_preview', False))
    rgb_map = list((cfg or {}).get('rgb_channels', [0,1,2]))
    save_color_composite = bool((cfg or {}).get('save_color_composite', True))
    channel_colors = list((cfg or {}).get('channel_colors', []))
    save_colored_pages = bool((cfg or {}).get('save_multichannel_colored_pages', True))
    save_ome_colors = bool((cfg or {}).get('save_ome_tiff_colors', True))
    ch_names = list((cfg or {}).get('channel_names', []))

    saved_any = False
    saved_main_path = None

    # 1) per-channel TIFFs
    if save_per_channel:
        for i, plane in enumerate(cyx):
            ch_path = out_dir / f"{inp.stem}_ch{i}.tif"
            if overwrite or (not ch_path.exists()):
                try:
                    skio.imsave(str(ch_path), plane.astype(np.uint16 if out_dtype=='uint16' else np.uint8), check_contrast=False)
                    if verbose: print(f"[SAVE] {ch_path.name}")
                    saved_any = True
                    if saved_main_path is None:
                        saved_main_path = str(ch_path)
                except Exception as e:
                    print(f"[WARN] Save per-channel failed ({ch_path.name}): {e}")

    # 2) multi-channel stack (layout configurable)
    if save_stack_tiff:
        stack_path = out_dir / f"{inp.stem}_stack.tif"
        if overwrite or (not stack_path.exists()):
            try:
                layout = str((cfg or {}).get('stack_layout', 'YXC')).upper()
                if layout not in {'CYX', 'YXC'}:
                    layout = 'YXC'
                if layout == 'CYX':
                    arr = cyx  # (C,Y,X)
                    axes = 'CYX'
                else:
                    arr = np.moveaxis(cyx, 0, -1)  # (Y,X,C)
                    axes = 'YXC'
                if tiff is not None:
                    meta = {'axes': axes}
                    desc = json.dumps({'shape': [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])] })
                    if axes == 'YXC' and arr.shape[-1] == 3:
                        photometric = 'rgb'
                    else:
                        photometric = 'minisblack'
                    tiff.imwrite(str(stack_path), arr, photometric=photometric, metadata=meta, description=desc)
                else:
                    skio.imsave(str(stack_path), arr, check_contrast=False)
                if verbose: print(f"[SAVE] {stack_path.name}")
                saved_any = True
                if saved_main_path is None:
                    saved_main_path = str(stack_path)
            except Exception as e:
                print(f"[WARN] Save stack failed: {e}")

    # 3) 3-channel RGB preview
    if save_rgb_preview and len(cyx) >= 3:
        r_idx, g_idx, b_idx = (rgb_map + [0,0,0])[:3]
        h, w = cyx.shape[1], cyx.shape[2]
        rgb = np.zeros((h, w, 3), dtype=(np.uint16 if out_dtype=='uint16' else np.uint8))
        for chan, dst in ((r_idx, 0), (g_idx, 1), (b_idx, 2)):
            if 0 <= chan < cyx.shape[0]:
                rgb[..., dst] = cyx[chan]
        rgb_path = out_dir / f"{inp.stem}_rgb.tif"
        if overwrite or (not rgb_path.exists()):
            try:
                if tiff is not None:
                    tiff.imwrite(str(rgb_path), rgb, photometric='rgb')
                else:
                    skio.imsave(str(rgb_path), rgb, check_contrast=False)
                if verbose: print(f"[SAVE] {rgb_path.name}")
                saved_any = True
                if saved_main_path is None:
                    saved_main_path = str(rgb_path)
            except Exception as e:
                print(f"[WARN] Save RGB preview failed: {e}")

    # 4) N-way false-color composite (blend)
    if save_color_composite and tiff is not None and cyx.shape[0] > 0:
        try:
            h, w = cyx.shape[1:]
            comp = np.zeros((h, w, 3), dtype=(np.uint16 if out_dtype=='uint16' else np.uint8))
            acc = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(cyx.shape[0]):
                col = channel_colors[i] if i < len(channel_colors) else 'white'
                acc += _colorize_plane(cyx[i], col, 'uint8').astype(np.float32) / 255.0
            acc = np.clip(acc, 0.0, 1.0)
            comp = (acc * (65535.0 if out_dtype=='uint16' else 255.0) + 0.5).astype(np.uint16 if out_dtype=='uint16' else np.uint8)
            color_path = out_dir / f"{inp.stem}_color.tif"
            if overwrite or (not color_path.exists()):
                tiff.imwrite(str(color_path), comp, photometric='rgb')
                if verbose: print(f"[SAVE] {color_path.name}")
                saved_any = True
                if saved_main_path is None:
                    saved_main_path = str(color_path)
        except Exception as e:
            print(f"[WARN] Save color composite failed: {e}")

    # 5) per-channel colored pages (multi-page RGB)
    if save_colored_pages and tiff is not None and cyx.shape[0] > 0:
        try:
            pages = []
            for i in range(cyx.shape[0]):
                col = channel_colors[i] if i < len(channel_colors) else 'white'
                pages.append(_colorize_plane(cyx[i], col, out_dtype))
            pages_arr = np.stack(pages, axis=0)  # (C,H,W,3)
            pages_path = out_dir / f"{inp.stem}_channels_color_pages.tif"
            if overwrite or (not pages_path.exists()):
                tiff.imwrite(str(pages_path), pages_arr, photometric='rgb')
                if verbose: print(f"[SAVE] {pages_path.name}")
                saved_any = True
                if saved_main_path is None:
                    saved_main_path = str(pages_path)
        except Exception as e:
            print(f"[WARN] Save colored pages failed: {e}")

    # 6) OME-TIFF CYX with channel color/name metadata
    if save_ome_colors and tiff is not None and cyx.shape[0] > 0:
        try:
            ome_path = out_dir / f"{inp.stem}_ome.tif"
            C = cyx.shape[0]
            chan_meta = []
            for i in range(C):
                name = (ch_names[i] if i < len(ch_names) and ch_names[i] else f"ch{i}")
                r,g,b = _parse_color(channel_colors[i] if i < len(channel_colors) else 'white')
                rgba = _rgba32_from_rgb((r,g,b))
                chan_meta.append({
                    'Name': name,
                    'Color': int(rgba),
                    'SamplesPerPixel': 1
                })
            arr_cyx = cyx.astype(np.uint16 if out_dtype=='uint16' else np.uint8)
            metadata = {
                'axes': 'CYX',  # keep channel dimension first so X/Y sizes stay consistent with ImageJ expectations
                'Channel': chan_meta,
                'SignificantBits': 16 if out_dtype=='uint16' else 8,
            }
            tiff.imwrite(
                str(ome_path),
                arr_cyx,
                metadata=metadata,
            )
            if verbose: print(f"[SAVE] {ome_path.name}")
            saved_any = True
            if saved_main_path is None:
                saved_main_path = str(ome_path)
        except Exception as e:
            print(f"[WARN] Save OME-TIFF failed: {e}")

    status = "ok" if saved_any else "ok_nochange"
    return True, saved_main_path, status


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CZI → TIFF converter")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to config_czi.yaml")
    parser.add_argument("--input", help="Override input_root", default=None)
    parser.add_argument("--output", help="Override output_base", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")  # NEW
    args = parser.parse_args(argv)

    # Load config
    cfg_path = Path(args.config)
    cfg = CziConfig.from_yaml(cfg_path)

    if args.input:  cfg.input_root  = args.input
    if args.output: cfg.output_base = args.output
    cfg.validate()
    verbose = bool(args.verbose or getattr(cfg, 'verbose', False))  # NEW

    if not cfg.input_root or not cfg.output_base:
        print("[ERROR] input_root/output_base not set. Edit config_czi.yaml or pass --input/--output.")
        return 2

    print("=== CZI CONFIG ===")
    print(f"Input:   {cfg.input_root}")
    print(f"Output:  {cfg.output_base}")
    print(f"Channels: {cfg.channels} | DType: {cfg.dtype} | Normalize: {cfg.per_channel_normalize} | Overwrite: {cfg.overwrite}")
    print(f"Save: per_channel={cfg.save_per_channel}, stack={cfg.save_stack_tiff}, rgb_preview={cfg.save_rgb_preview}, color_comp={cfg.save_color_composite}, pages={cfg.save_multichannel_colored_pages}, ome={cfg.save_ome_tiff_colors}")
    if verbose:
        print("[VERBOSE] Enabled")
    print("===================\n")

    # Map to container-friendly paths if needed
    in_root  = Path(_map_path_for_container(cfg.input_root))
    out_base = Path(_map_path_for_container(cfg.output_base))
    in_root  = in_root.resolve()
    out_base = out_base.resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # Discover files by condition/region
    grouped = find_czi_files(in_root)
    if not grouped:
        print("[WARN] Keine CZI-Dateien gefunden.")
        return 0

    # Save config snapshot for reproducibility
    runtime_params = {
        "input_root": str(in_root),
        "output_base": str(out_base),
        "channels": cfg.channels,
        "target_size": cfg.target_size,
        "per_channel_normalize": cfg.per_channel_normalize,
        "dtype": cfg.dtype,
        "overwrite": cfg.overwrite,
        "save_per_channel": cfg.save_per_channel,
        "save_stack_tiff": cfg.save_stack_tiff,
        "save_rgb_preview": cfg.save_rgb_preview,
        "save_color_composite": cfg.save_color_composite,
        "n_conditions": len(grouped),
        "n_total_files": sum(len(files) for regions in grouped.values() for files in regions.values()),
    }
    config_dict = cfg.__dict__.copy()
    save_config_snapshot(
        output_dir=out_base,
        config_file=cfg_path if cfg_path.exists() else None,
        config_dict=config_dict,
        runtime_params=runtime_params,
        snapshot_name="czi_conversion_config",
    )

    stats: List[str] = []
    for group, regions in grouped.items():
        print(f"\n[{group.upper()}] {len(regions)} Regionen")
        for region, files in regions.items():
            if not files: 
                continue
            print(f"  - {region}: {len(files)} Datei(en)")
            for f in tqdm(files, desc=f"{group}/{region}"):
                try:
                    ok, outp, status = process_czi_file(
                        str(f), group, cfg.target_size, cfg.channels,
                        cfg.per_channel_normalize, cfg.overwrite, str(out_base), cfg.__dict__, verbose=verbose  # NEW
                    )
                    stats.append(status)
                except Exception as e:
                    print(f"[ERROR] {f.name}: {e}")
                    stats.append("error")
                finally:
                    # NEW: signal one file finished (for progress tracking in Flask)
                    try:
                        print(f"[FILE_DONE] {f.name}", flush=True)
                    except Exception:
                        pass

    ok_count      = sum(1 for s in stats if s.startswith("ok"))
    exists_count  = sum(1 for s in stats if s == "exists")
    error_count   = sum(1 for s in stats if s.startswith("error"))
    total         = len(stats)

    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"✅ Erfolgreich/keine Änderung: {ok_count}")
    print(f"⏭️  Übersprungen:              {exists_count}")
    print(f"❌ Fehler:                    {error_count}")
    print(f"📊 Gesamt:                    {total}")
    print(f"\n[DONE] Ausgabe: {out_base}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
