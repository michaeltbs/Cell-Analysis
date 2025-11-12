#!/usr/bin/env python3
"""
analysis_combinations.py ÔÇö Flexible co-expression analysis across channels

Key ideas
---------
- English-only logs and messages.
- Highly configurable via YAML `config_analysis.yaml` (keeps your field names where possible).
- Region selection is per run (optional list of regions).
- Works with different folder layouts via patterns:
    * Default: by-channel layout like  .../ch0/**/_mask.tif, .../ch1/**/_mask.tif
    * You can override per-channel glob pattern in the config.
- Prefers mask TIFFs (binary) for robust overlap computation. If only overlays are available, script can fall back to
  a naive threshold on overlays (set `fallback_from_overlays: true`), but masks are strongly recommended.
- Produces per-combination CSV summaries and optional colored overlay PNGs.

Config (example)
----------------
paths:
  base_results_dir: "./results_det"   # where channel folders (or any layout) live
  output_dir: "./coexpr_out"          # where to write co-expression results

dataset_name: ""                      # optional label (used in CSV naming only)

# Regions are filtered by path tokens (case-insensitive). Omit to include all.
regions_enabled: ["ARC", "DMH", "PVH"]

# Channels available to pick from (UI may generate this)
# - index: integer id (e.g., 0..3)
# - name: optional human name (for CSV headers)
# - enabled: if false, channel is ignored by default combinatorics
# - glob: optional custom glob (relative to base_results_dir) to locate *_mask.tif files for this channel
#         Default is f"ch{index}/**/*_mask.tif"
channels:
  - { index: 0, name: "GAL", enabled: true }
  - { index: 1, name: "POMC", enabled: true }
  - { index: 2, name: "AgRP", enabled: false }
  - { index: 3, name: "cFos", enabled: false }

# Strategy to select combinations
# - If selected_combos is provided, use exactly those (e.g., [[0,1],[0,2,3]])
# - Else, generate all combinations of size in combos_enabled (e.g., [2,3]) from enabled channels.
combos_enabled: [2, 3]
selected_combos: []

# Overlay drawing (optional)
overlay_params:
  dpi: 200
  line_width: 2
  figsize: [12, 12]
  colors_by_k:
    "2": "red"
    "3": "magenta"
    "4": "cyan"

# If true, allows deriving a crude mask from overlay PNGs when no *_mask.tif is found.
# It will search for files like **/*_overlay.png under each channel's root and threshold luminance.
# Not recommended for quantitative work.
use_detection_masks: true             # if true, look for *_mask.tif files first
fallback_from_overlays: true          # if true and no masks found, use *_overlay.png

# Debug verbosity
debug: false

Outputs
-------
- <output_dir>/k2/ch0_ch1/  -> CSV + overlays for each sample
- <output_dir>/k3/ch0_ch1_ch2/ -> ...
- Summary CSV per combination: coexpr_summary.csv
"""
from __future__ import annotations

import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# imaging
try:
    import tifffile as tiff
except Exception:
    tiff = None
from skimage import io as skio
from skimage import measure, morphology, color, exposure, util
from skimage.filters import threshold_otsu

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class Paths:
    base_results_dir: str = ""
    output_dir: str = "./coexpr_out"

@dataclass
class ChannelCfg:
    index: int
    name: str = ""
    enabled: bool = True
    glob: Optional[str] = None  # e.g., "ch0/**/*_mask.tif"

@dataclass
class OverlayParams:
    dpi: int = 200
    line_width: int = 2
    figsize: Sequence[int] = field(default_factory=lambda: (12, 12))
    colors_by_k: Dict[str, str] = field(default_factory=lambda: {"2": "red", "3": "magenta", "4": "cyan"})

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    dataset_name: str = ""
    regions_enabled: Optional[Sequence[str]] = None
    channels: Sequence[ChannelCfg] = field(default_factory=list)
    combos_enabled: Sequence[int] = field(default_factory=lambda: (2,))
    selected_combos: Sequence[Sequence[int]] = field(default_factory=list)
    overlay_params: OverlayParams = field(default_factory=OverlayParams)
    use_detection_masks: bool = True  # NEW: prefer masks over overlays
    fallback_from_overlays: bool = True  # UPDATED: now clearer purpose
    debug: bool = False
    save_composite_figure: bool = False
    test_mode: bool = False
    test_samples_per_channel: int = 0
    test_seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        def map_dc(dc_cls, d):
            if isinstance(d, dict):
                obj = dc_cls()
                for k, v in d.items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
                return obj
            return dc_cls()
        paths = map_dc(Paths, data.get("paths"))
        ovp = map_dc(OverlayParams, data.get("overlay_params"))
        # channels (prefer 'channels', fallback to 'channel_config')
        src_channels = data.get("channels")
        if not isinstance(src_channels, list) or not src_channels:
            src_channels = data.get("channel_config") or []
        chs = []
        for ch in src_channels:
            try:
                idx = int(ch.get("index"))
            except Exception:
                continue
            chs.append(ChannelCfg(index=idx, name=ch.get("name",""), enabled=bool(ch.get("enabled", True)), glob=ch.get("glob")))
        testing_data = data.get("testing") or {}
        test_mode = bool(testing_data.get("enabled", data.get("test_mode", False)))
        raw_samples = testing_data.get("samples_per_channel", data.get("test_samples_per_channel", 0))
        try:
            test_samples = int(raw_samples or 0)
        except Exception:
            test_samples = 0
        raw_seed = testing_data.get("seed", data.get("test_seed"))
        if raw_seed in (None, ""):
            test_seed = None
        else:
            try:
                test_seed = int(raw_seed)
            except Exception:
                test_seed = None

        scf_raw = data.get("save_composite_figure")
        save_fig = True if scf_raw is None else bool(scf_raw)

        return cls(
            paths=paths,
            dataset_name=str(data.get("dataset_name","") or ""),
            regions_enabled=data.get("regions_enabled"),
            channels=chs,
            combos_enabled=data.get("combos_enabled") or [2],
            selected_combos=data.get("selected_combos") or [],
            overlay_params=ovp,
            use_detection_masks=bool(data.get("use_detection_masks", True)),  # NEW
            fallback_from_overlays=bool(data.get("fallback_from_overlays", True)),  # UPDATED default
            save_composite_figure=save_fig,
            debug=bool(data.get("debug", False)),
            test_mode=test_mode,
            test_samples_per_channel=test_samples,
            test_seed=test_seed,
        )

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
IMG_EXTS = (".tif", ".tiff", ".png")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _normalize_region_token(s: str) -> str:
    s = s.replace("\\", "/").replace("\\", "/").lower()
    for token in ("larc","ldmh","lpvh","rarc","rdmh","rpvh","arc","dmh","pvh","mbh"):
        if token in s:
            return token.upper()
    return "ALL"

def _read_mask(path: Path) -> np.ndarray:
    try:
        if tiff and path.suffix.lower() in (".tif",".tiff"):
            arr = tiff.imread(str(path))
        else:
            arr = skio.imread(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path.name}: {e}")
    arr = util.img_as_float32(arr)
    if arr.ndim == 3:
        # assume grayscale overlay; collapse
        arr = color.rgb2gray(arr) if arr.shape[-1] == 3 else arr[...,0]
    # binarize if not already
    if arr.dtype != np.uint8:
        thr = threshold_otsu(arr) if np.any(arr > 0) else 0.0
        arr = (arr > thr).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
    return arr

def _read_overlay_and_threshold(path: Path) -> np.ndarray:
    # Fallback: threshold overlay luminance to approximate mask
    try:
        img = skio.imread(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read overlay {path.name}: {e}")
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    gray = color.rgb2gray(img) if img.ndim == 3 else util.img_as_float32(img)
    thr = threshold_otsu(gray) if np.any(gray > 0) else 0.0
    return (gray > thr).astype(np.uint8)

def _load_channel_preview(base: Path, ch_idx: int, key: str, src_path: Path) -> Optional[np.ndarray]:
    """Load a representative image for the given channel (overlay if available, else source)."""
    try:
        src_name = src_path.name.lower()
        if src_name.endswith((".tif", ".tiff")) and "_mask" in src_name:
            overlay_candidate = base / f"ch{ch_idx}" / "overlays" / f"{key}_overlay.png"
            if overlay_candidate.exists():
                img = skio.imread(str(overlay_candidate))
            else:
                img = tiff.imread(str(src_path)) if tiff else skio.imread(str(src_path))
        else:
            img = skio.imread(str(src_path))
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        return img
    except Exception:
        return None

def _save_composite_figure(images: Sequence[Tuple[Optional[np.ndarray], str]], merged_mask: np.ndarray, out_path: Path, dpi: int) -> None:
    """Create a side-by-side figure of channel previews and co-expression mask."""
    if not images:
        return
    cols = len(images) + 1
    width = max(4, 3 * cols)
    height = 3.5
    fig, axes = plt.subplots(1, cols, figsize=(width, height))
    axes = np.atleast_1d(axes)
    for ax, (img, title) in zip(axes[:-1], images):
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        if img is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="red", fontsize=10)
            continue
        arr = np.asarray(img)
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray")
        else:
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            ax.imshow(arr)
    last_ax = axes[-1]
    last_ax.set_title("Co-expression", fontsize=10)
    last_ax.set_axis_off()
    base_preview = images[0][0] if images and images[0][0] is not None else None
    if base_preview is not None:
        base_arr = np.asarray(base_preview)
        if base_arr.ndim == 2:
            last_ax.imshow(base_arr, cmap="gray")
        else:
            if base_arr.shape[-1] == 4:
                base_arr = base_arr[..., :3]
            last_ax.imshow(base_arr)
        last_ax.imshow(merged_mask, cmap="magma", alpha=0.35)
    else:
        last_ax.imshow(merged_mask, cmap="magma")
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def _glob_channel_files(base: Path, ch: ChannelCfg, use_masks: bool, fallback_overlays: bool) -> Dict[str, Path]:
    """Return mapping key -> path for this channel with robust layout support."""
    out: Dict[str, Path] = {}

    def _add(files: List[Path], mode: str):
        for f in files:
            st = f.stem
            key = st.replace("_mask", "").replace("_overlay", "")
            out[key] = f
        return bool(files)

    # Try masks first if enabled
    if use_masks:
        mask_patterns = [
            ch.glob or f"ch{ch.index}/masks/**/*_mask.tif",
            (ch.glob or f"ch{ch.index}/masks/**/*.tif").replace("*_mask.tif", "*.tif"),
            (ch.glob or f"ch{ch.index}/**/*_mask.tif"),
        ]
        for pat in mask_patterns:
            if out: break
            files = list(base.glob(pat))
            if _add(files, "mask"):
                return out

    # Overlays fallback (or if masks not used)
    if fallback_overlays or not use_masks:
        overlay_patterns = [
            f"ch{ch.index}/overlays/**/*_overlay.png",
            f"ch{ch.index}/overlays/**/*.png",
            f"ch{ch.index}/**/*_overlay.png",
        ]
        for pat in overlay_patterns:
            files = list(base.glob(pat))
            if _add(files, "overlay"):
                return out

    return out

def _merge_masks(mask_list: List[np.ndarray]) -> np.ndarray:
    merged = mask_list[0].astype(bool)
    for m in mask_list[1:]:
        merged &= (m.astype(bool))
    return merged.astype(np.uint8)

def _save_overlay(base_img: np.ndarray, mask: np.ndarray, out_path: Path, color: str, lw: int, dpi: int, figsize: Sequence[int]):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0,0,1,1])
    ax.imshow(base_img, cmap="gray")
    ax.axis("off")
    contours = measure.find_contours(mask.astype(float), 0.5)
    for c in contours:
        ax.plot(c[:,1], c[:,0], color=color, linewidth=lw)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ------------------------------------------------------------
# Core
# ------------------------------------------------------------
def run_coexpression(cfg: Config) -> None:
    base = Path(cfg.paths.base_results_dir)
    out_root = Path(cfg.paths.output_dir)
    if cfg.test_mode:
        out_root = out_root / "__test__"
    _ensure_dir(out_root)

    if cfg.debug:
        print(f'[DEBUG] base_results_dir={base}')
        print(f'[DEBUG] output_dir={out_root}')
        print(f'[DEBUG] regions_enabled={cfg.regions_enabled}')
        print(f'[DEBUG] use_detection_masks={cfg.use_detection_masks}')  # NEW
        print(f'[DEBUG] fallback_from_overlays={cfg.fallback_from_overlays}')

    if cfg.test_mode:
        seed_note = f' (seed={cfg.test_seed})' if cfg.test_seed is not None else ''
        print(f'[TEST] Co-expression test mode: samples_per_channel={cfg.test_samples_per_channel}{seed_note}')

    enabled_regions = set([s.upper() for s in (cfg.regions_enabled or [])])
    def region_allowed(p: Path) -> bool:
        if not enabled_regions:
            return True
        return _normalize_region_token(p.as_posix()) in enabled_regions

    # Determine combinations
    enabled_channels = [ch for ch in cfg.channels if ch.enabled]
    if cfg.selected_combos:
        combos = [tuple(map(int, c)) for c in cfg.selected_combos]
    else:
        # build all combinations for sizes in combos_enabled
        from itertools import combinations
        combos = []
        for k in cfg.combos_enabled:
            for comb in combinations([ch.index for ch in enabled_channels], int(k)):
                combos.append(tuple(comb))

    if cfg.debug:
        print(f"[DEBUG] channels={[(c.index,c.name,c.enabled) for c in cfg.channels]}")
        print(f"[DEBUG] combos_enabled={cfg.combos_enabled}, selected={cfg.selected_combos or 'auto'}")

    # Pre-scan: build a per-channel index {index: {key->path}}
    per_channel_index: Dict[int, Dict[str, Path]] = {}
    for ch in cfg.channels:
        mapping = _glob_channel_files(base, ch, cfg.use_detection_masks, cfg.fallback_from_overlays)  # UPDATED
        per_channel_index[ch.index] = mapping
        if cfg.debug:
            # NEW: show what type of files were found
            if mapping:
                sample_file = next(iter(mapping.values()))
                file_type = "masks" if "_mask" in sample_file.name else "overlays"
                print(f"[DEBUG] Channel {ch.index}: {len(mapping)} {file_type} found")
            else:
                print(f"[DEBUG] Channel {ch.index}: no files found")

    # Build a universe of keys (sample ids) present in at least one channel
    all_keys = set()
    for d in per_channel_index.values():
        all_keys.update(d.keys())
    if cfg.debug:
        print(f"[DEBUG] Total unique sample keys found: {len(all_keys)}")

    if cfg.test_mode:
        limit = cfg.test_samples_per_channel or 2
        if limit <= 0:
            limit = 2
        rng = random.Random(cfg.test_seed)
        if len(all_keys) > limit:
            sampled_keys = rng.sample(sorted(all_keys), limit)
            all_keys = set(sampled_keys)
            print(f"[TEST] Limiting co-expression to {len(all_keys)} sample keys (limit {limit}).")
        # Trim per-channel indices to sampled keys
        for ch_idx, mapping in per_channel_index.items():
            per_channel_index[ch_idx] = {k: v for k, v in mapping.items() if not all_keys or k in all_keys}

    # For each combination, compute overlaps for keys that have ALL required channels present
    for comb in combos:
        k = len(comb)
        color_k = cfg.overlay_params.colors_by_k.get(str(k), "red")
        comb_sorted = tuple(sorted(comb))
        comb_dir_name = "ch" + "_".join(map(str, comb_sorted))
        out_dir = out_root / f"k{k}" / comb_dir_name
        _ensure_dir(out_dir)
        rows = []

        for key in sorted(all_keys):
            # Resolve files for all channels
            fpaths: List[Path] = []
            channel_previews: List[Optional[np.ndarray]] = []
            for ch_idx in comb_sorted:
                fp = per_channel_index.get(ch_idx, {}).get(key)
                if not fp:
                    fpaths = []
                    channel_previews = []
                    break
                fpaths.append(fp)
                channel_previews.append(_load_channel_preview(base, ch_idx, key, fp))
            if not fpaths:
                continue

            # Region filter by path tokens
            if not region_allowed(fpaths[0]):
                continue

            # Read masks (or overlays as fallback), build merged co-expression mask
            masks = []
            base_img = None
            for fp in fpaths:
                # UPDATED: clearer logic for mask vs overlay handling
                is_mask = "_mask" in fp.name and fp.suffix.lower() in (".tif", ".tiff")

                if is_mask:
                    m = _read_mask(fp)
                    masks.append(m)
                    if base_img is None:
                        base_img = (tiff.imread(str(fp)) if tiff else skio.imread(str(fp)))
                        if base_img.ndim == 3:
                            base_img = color.rgb2gray(base_img)
                else:
                    # Overlay fallback
                    m = _read_overlay_and_threshold(fp)
                    masks.append(m)
                    if base_img is None:
                        base_img = skio.imread(str(fp))
                        if base_img.ndim == 3:
                            base_img = color.rgb2gray(base_img)

            merged = _merge_masks(masks)

            # Count connected components as "co-expressing objects"
            labels = measure.label(merged, connectivity=1)
            n_co = int(labels.max())

            # Basic stats
            area_px = int((merged > 0).sum())
            rows.append({
                "sample_key": key,
                "k": k,
                "channels": ",".join(map(str, comb_sorted)),
                "n_coexpressing": n_co,
                "area_px": area_px,
            })

            # Optional overlay output
            ov = out_dir / f"{key}_coexpr_overlay.png"
            _save_overlay(base_img, merged, ov, color=color_k, lw=cfg.overlay_params.line_width, dpi=cfg.overlay_params.dpi, figsize=tuple(cfg.overlay_params.figsize))
            if cfg.save_composite_figure:
                figure_path = out_dir / f"{key}_coexpr_figure.png"
                previews = [(channel_previews[i] if i < len(channel_previews) else None, f"ch{idx}") for i, idx in enumerate(comb_sorted)]
                _save_composite_figure(previews, merged, figure_path, cfg.overlay_params.dpi)

        # Write summary CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / "coexpr_summary.csv", index=False)

        # Combo-level README - UPDATED to show what was used
        mode_note = "detection masks" if cfg.use_detection_masks else "overlay images"
        (out_dir / "README.txt").write_text(
            f"Combination: k={k}, channels={comb_sorted}\n"
            f"Files derived from base: {base}\n"
            f"Analysis mode: {mode_note} (fallback_enabled={cfg.fallback_from_overlays})\n"
            f"Composite figures saved: {cfg.save_composite_figure}\n"
            f"Rows: {len(rows)}\n",
            encoding="utf-8"
        )
        print(f"[INFO] k={k}, channels={comb_sorted} ÔåÆ {len(rows)} samples ÔåÆ {out_dir}")

    print("\n[DONE] Co-expression analysis finished.")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Co-expression combinations analysis")
    p.add_argument("--config", "-c", default="config_analysis.yaml", help="Path to YAML config")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p.add_argument("--test", action="store_true", help="Enable test mode (limit samples)")
    p.add_argument("--test-samples", type=int, default=None, help="Number of sample keys when test mode is enabled")
    p.add_argument("--test-seed", type=int, default=None, help="Optional seed for test sampling")
    args = p.parse_args(argv)
    cfg = Config.from_yaml(Path(args.config))
    if args.verbose:
        cfg.debug = True
    if args.test:
        cfg.test_mode = True
    if args.test_samples is not None:
        cfg.test_samples_per_channel = max(1, int(args.test_samples))
    if args.test_seed is not None:
        cfg.test_seed = int(args.test_seed)
    run_coexpression(cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
