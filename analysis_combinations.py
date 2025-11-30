#!/usr/bin/env python3
"""
analysis_combinations.py — Flexible co-expression analysis across channels

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
  dpi: 300
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
import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set, Mapping
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
from skimage.transform import resize
from skimage.filters import threshold_otsu

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

try:
    from src.config_archiver import save_config_snapshot
except Exception:
    # fallback if not available
    def save_config_snapshot(*args, **kwargs):
        pass

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
    color: str = ""  # Farbe für diesen Kanal (z.B. "#ff0000", "red", etc.)

@dataclass
class OverlayParams:
    dpi: int = 300
    line_width: int = 2
    figsize: Sequence[int] = field(default_factory=lambda: (12, 12))
    colors_by_k: Dict[str, str] = field(default_factory=lambda: {"2": "red", "3": "magenta", "4": "cyan"})
    single_positive_color: str = "gray"
    base_channel_index: Optional[int] = None
    figure_type: str = "overlay"
    even_layout: str = "line"
    overlay_style: str = "mixed"
    fill_alpha: float = 0.45
    save_centroid_heatmap: bool = False
    centroid_marker_size: int = 80

@dataclass
class CoexpressionParams:
    mode: str = "overlap"
    centroid_max_distance: float = 8.0
    blend_base_color: str = "#b9b9b9"
    blend_other_color: str = "#40c4ff"
    blend_overlap_color: str = "#ff4080"
    blend_overlap_fraction: float = 0.02
    blend_channel_colors: Dict[int, str] = field(default_factory=dict)
    centroid_overlap_fraction: float = 0.005
    # Dateiformate für Ausgabe
    blend_output_formats: List[str] = field(default_factory=lambda: ["png"])
    overlay_output_formats: List[str] = field(default_factory=lambda: ["png"])

@dataclass
class CoexprSweepParams:
    enabled: bool = False
    modes: Sequence[str] = field(default_factory=lambda: ("overlap", "either", "intersection"))
    sample_key: str = ""

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    dataset_name: str = ""
    regions_enabled: Optional[Sequence[str]] = None
    channels: Sequence[ChannelCfg] = field(default_factory=list)
    combos_enabled: Sequence[int] = field(default_factory=lambda: (2,))
    selected_combos: Sequence[Sequence[int]] = field(default_factory=list)
    overlay_params: OverlayParams = field(default_factory=OverlayParams)
    coexpression: CoexpressionParams = field(default_factory=CoexpressionParams)
    coexpr_sweep: CoexprSweepParams = field(default_factory=CoexprSweepParams)
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
        # Support flat config structure (from UI)
        if data.get("input_dir"):
            paths.base_results_dir = data.get("input_dir")
        if data.get("output_dir"):
            paths.output_dir = data.get("output_dir")

        ovp = map_dc(OverlayParams, data.get("overlay_params"))
        # Support flat overlay params from UI (overlay_visualization)
        if data.get("overlay_visualization"):
             ovp_flat = map_dc(OverlayParams, data.get("overlay_visualization"))
             for k in vars(ovp_flat):
                 v = getattr(ovp_flat, k)
                 if v is not None and v != "":
                     setattr(ovp, k, v)

        overlay_colors = data.get("overlay_colors") or {}
        if isinstance(overlay_colors, dict):
            for key, val in overlay_colors.items():
                if not isinstance(val, str):
                    continue
                norm_key = str(key).split("-", 1)[0]
                ovp.colors_by_k[str(norm_key)] = val
        cleaned_colors = {}
        for ck, cv in (ovp.colors_by_k or {}).items():
            cleaned_colors[str(ck).split("-", 1)[0]] = cv
        ovp.colors_by_k = cleaned_colors
        try:
            if ovp.base_channel_index in ("", None):
                ovp.base_channel_index = None
            else:
                ovp.base_channel_index = int(ovp.base_channel_index)
        except Exception:
            ovp.base_channel_index = None
        if not getattr(ovp, "single_positive_color", None):
            ovp.single_positive_color = "gray"
        fig_type = (getattr(ovp, "figure_type", None) or overlay_colors.get("figure_type") or data.get("figure_type") or "overlay").strip().lower()
        if fig_type not in ("overlay", "mask"):
            fig_type = "overlay"
        ovp.figure_type = fig_type
        even_layout = (getattr(ovp, "even_layout", None) or overlay_colors.get("even_layout") or data.get("even_layout") or "line").strip().lower()
        if even_layout not in ("line", "grid"):
            even_layout = "line"
        ovp.even_layout = even_layout
        overlay_style = (
            getattr(ovp, "overlay_style", None)
            or overlay_colors.get("overlay_style")
            or data.get("overlay_style")
            or "contour"
        )
        overlay_style_raw = str(overlay_style).strip().lower()
        if "blend" in overlay_style_raw:
            overlay_style = "blend"
        elif overlay_style_raw in ("contour", "filled", "mixed"):
            overlay_style = overlay_style_raw
        else:
            overlay_style = "mixed"
        ovp.overlay_style = overlay_style
        try:
            ovp.fill_alpha = float(getattr(ovp, "fill_alpha", 0.45))
        except Exception:
            ovp.fill_alpha = 0.45
        ovp.fill_alpha = min(max(ovp.fill_alpha, 0.0), 1.0)
        ovp.save_centroid_heatmap = bool(getattr(ovp, "save_centroid_heatmap", False))
        try:
            ovp.centroid_marker_size = int(getattr(ovp, "centroid_marker_size", 80))
        except Exception:
            ovp.centroid_marker_size = 80
        if ovp.centroid_marker_size <= 0:
            ovp.centroid_marker_size = 40
        coexpr_raw = data.get("coexpression") or data.get("coexpr") or {}
        # Merge root keys into coexpr_raw if not present (flat config support)
        for k in ["coexpr_mode", "centroid_max_distance", "blend_base_color", "blend_other_color", "blend_overlap_color", "blend_overlap_fraction", "centroid_overlap_fraction"]:
             if k in data and k not in coexpr_raw:
                 target_k = "mode" if k == "coexpr_mode" else k
                 coexpr_raw[target_k] = data[k]

        coexpr = map_dc(CoexpressionParams, coexpr_raw)
        mode = (getattr(coexpr, "mode", "") or "overlap").strip().lower()
        allowed_modes = ALLOWED_COEXPR_MODES
        if mode not in allowed_modes:
            mode = "overlap"
        coexpr.mode = mode
        try:
            coexpr.centroid_max_distance = float(getattr(coexpr, "centroid_max_distance", 8.0))
        except Exception:
            coexpr.centroid_max_distance = 8.0
        if coexpr.centroid_max_distance < 0:
            coexpr.centroid_max_distance = 0.0
        coexpr.blend_base_color = _sanitize_color(
            coexpr_raw.get("blend_base_color", getattr(coexpr, "blend_base_color", None)),
            "#b9b9b9"
        )
        coexpr.blend_other_color = _sanitize_color(
            coexpr_raw.get("blend_other_color", getattr(coexpr, "blend_other_color", None)),
            "#40c4ff"
        )
        coexpr.blend_overlap_color = _sanitize_color(
            coexpr_raw.get("blend_overlap_color", getattr(coexpr, "blend_overlap_color", None)),
            "#ff4080"
        )
        try:
            overlap_fraction = float(coexpr_raw.get("blend_overlap_fraction", getattr(coexpr, "blend_overlap_fraction", 0.02)))
        except Exception:
            overlap_fraction = 0.02
        if overlap_fraction < 0:
            overlap_fraction = 0.0
        if overlap_fraction > 1:
            overlap_fraction = 1.0
        coexpr.blend_overlap_fraction = overlap_fraction
        try:
            centroid_fraction = float(
                coexpr_raw.get(
                    "centroid_overlap_fraction",
                    getattr(coexpr, "centroid_overlap_fraction", 0.005),
                )
            )
        except Exception:
            centroid_fraction = 0.005
        if centroid_fraction < 0:
            centroid_fraction = 0.0
        if centroid_fraction > 1:
            centroid_fraction = 1.0
        coexpr.centroid_overlap_fraction = centroid_fraction
        
        # channels (prefer 'channels', fallback to 'channel_config')
        # WICHTIG: Muss VOR anderen Initialisierungen kommen!
        src_channels = data.get("channels")
        if not isinstance(src_channels, list) or not src_channels:
            src_channels = data.get("channel_config") or []
        chs = []
        for ch in src_channels:
            try:
                idx = int(ch.get("index"))
            except Exception:
                continue
            ch_color = _sanitize_color(ch.get("color", ""), "")
            chs.append(ChannelCfg(
                index=idx, 
                name=ch.get("name",""), 
                enabled=bool(ch.get("enabled", True)), 
                glob=ch.get("glob"),
                color=ch_color
            ))
        
        # Blend channel colors - Nur aus Channel-Konfiguration
        channel_color_map: Dict[int, str] = {}
        for ch in chs:
            if ch.color:
                channel_color_map[ch.index] = ch.color
        
        coexpr.blend_channel_colors = channel_color_map
        sweep_raw = coexpr_raw.get("sweep") or {}
        sweep = map_dc(CoexprSweepParams, sweep_raw)
        sweep.enabled = bool(sweep_raw.get("enabled", getattr(sweep, "enabled", False)))
        modes_raw = sweep_raw.get("modes", getattr(sweep, "modes", []))
        parsed_modes: List[str] = []
        if isinstance(modes_raw, str):
            parsed_modes = [m.strip().lower() for m in modes_raw.split(",") if m.strip()]
        elif isinstance(modes_raw, (list, tuple, set)):
            for item in modes_raw:
                val = str(item).strip().lower()
                if val:
                    parsed_modes.append(val)
        if not parsed_modes:
            parsed_modes = ["overlap", "either", "intersection"]
        parsed_modes = [m for m in parsed_modes if m in allowed_modes]
        if not parsed_modes:
            parsed_modes = ["overlap"]
        sweep.modes = parsed_modes
        sample_key_raw = sweep_raw.get("sample_key", getattr(sweep, "sample_key", ""))
        sweep.sample_key = str(sample_key_raw or "").strip()
        
        # Channel-Parsing wurde bereits oben durchgeführt (vor blend_channel_colors)
        
        testing_data = data.get("testing") or {}
        
        # Handle test_mode from UI (dict or bool)
        tm_val = data.get("test_mode")
        if isinstance(tm_val, dict):
            # UI sends test_mode as dict
            test_mode = tm_val.get("type", "off") != "off"
            raw_samples = tm_val.get("samples") or tm_val.get("samples_per_channel") or 0
            raw_seed = tm_val.get("seed")
            
            # Also handle sweep params from test_mode dict if present
            if "sweep_modes" in tm_val and tm_val["sweep_modes"]:
                sweep.modes = tm_val["sweep_modes"]
            if "sweep_sample" in tm_val:
                sweep.sample_key = tm_val["sweep_sample"]
        else:
            test_mode = bool(testing_data.get("enabled", tm_val if tm_val is not None else False))
            raw_samples = testing_data.get("samples_per_channel", data.get("test_samples_per_channel", 0))
            raw_seed = testing_data.get("seed", data.get("test_seed"))
        
        try:
            test_samples = int(raw_samples or 0)
        except Exception:
            test_samples = 0

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
            coexpression=coexpr,
            coexpr_sweep=sweep,
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
ALLOWED_COEXPR_MODES = {"overlap", "centroid", "either", "both", "union", "intersection", "all", "blend"}
OVERLAP_MODES = {"overlap", "either", "both", "union", "intersection", "all", "blend"}
CENTROID_MODES = {"centroid", "either", "both", "union", "intersection", "all"}

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

def _extract_background(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr)
    if data.ndim == 3 and data.shape[-1] >= 3:
        data = util.img_as_float32(data)
        return color.rgb2gray(data)
    return data


def _prepare_display(arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.array([])
    data = np.asarray(arr)
    if data.size == 0:
        return data
    if data.ndim == 2:
        data = data.astype(np.float32, copy=False)
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            data = (data - lo) / (hi - lo)
        elif hi not in (0.0, np.nan):
            data = data / hi if hi != 0 else data
        return np.clip(data, 0.0, 1.0)
    data = data.astype(np.float32, copy=False)
    if data.shape[-1] > 3:
        data = data[..., :3]
    for c in range(data.shape[-1]):
        chan = data[..., c]
        lo, hi = float(np.nanmin(chan)), float(np.nanmax(chan))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            data[..., c] = (chan - lo) / (hi - lo)
        elif hi not in (0.0, np.nan):
            data[..., c] = chan / hi if hi != 0 else chan
        else:
            data[..., c] = 0.0
    return np.clip(data, 0.0, 1.0)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=-1)
    elif data.ndim == 3:
        if data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        elif data.shape[-1] > 3:
            data = data[..., :3]
    else:
        data = np.zeros((1, 1, 3), dtype=np.float32)
    return data.astype(np.float32, copy=False)


def _mask_outline(mask_bool: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Create outline of mask with specified thickness."""
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    
    # For thick outlines, dilate first then subtract original
    # This creates an outline AROUND the mask, not inside it
    thickness = max(int(round(thickness)), 1)
    
    if thickness >= 2:
        # Dilate mask to create outer boundary
        dilated = morphology.binary_dilation(mask_bool, morphology.disk(thickness))
        # Also erode slightly for inner boundary
        eroded = morphology.binary_erosion(mask_bool, morphology.disk(max(1, thickness // 2)))
        if not np.any(eroded):
            eroded = mask_bool
        # Outline is dilated minus eroded (thick ring around cells)
        outline = dilated & ~eroded
    else:
        # Thin outline: just erode once
        eroded = morphology.binary_erosion(mask_bool, morphology.disk(1))
        if not np.any(eroded):
            outline = mask_bool
        else:
            outline = mask_bool & ~eroded
    
    if not np.any(outline):
        outline = mask_bool
    return outline


def _apply_colored_outline(
    base_arr: np.ndarray,
    mask_bool: np.ndarray,
    color: str,
    thickness: int = 1,
) -> np.ndarray:
    rgb = _ensure_rgb(base_arr).copy()
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask_bool):
        return np.clip(rgb, 0.0, 1.0)
    color_rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
    outline = _mask_outline(mask_bool, thickness)
    if np.any(outline):
        rgb[outline] = color_rgb
    return np.clip(rgb, 0.0, 1.0)


def _apply_colored_fill(
    base_arr: np.ndarray,
    mask_bool: np.ndarray,
    color: str,
    alpha: float = 0.45,
) -> np.ndarray:
    rgb = _ensure_rgb(base_arr).copy()
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask_bool):
        return np.clip(rgb, 0.0, 1.0)
    alpha = float(alpha)
    if alpha <= 0.0:
        return np.clip(rgb, 0.0, 1.0)
    if alpha > 1.0:
        alpha = 1.0
    color_rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
    rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * color_rgb
    return np.clip(rgb, 0.0, 1.0)


def _line_width_to_thickness(lw: float) -> int:
    value = max(float(lw), 0.0)
    if value <= 1.0:
        return 1
    return max(1, int(round(value / 1.5)))


def _resize_bool(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask) > 0
    if arr.shape == target_shape:
        return arr
    resized = resize(
        arr.astype(np.float32),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized > 0.5


def _normalize_masks(mask_list: Sequence[np.ndarray], base_idx: int) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    base_arr = np.asarray(mask_list[base_idx]) > 0
    target_shape = base_arr.shape
    normalized: List[np.ndarray] = []
    for m in mask_list:
        normalized.append(_resize_bool(m, target_shape))
    return normalized, target_shape


def _slugify_mode(mode: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", (mode or "").lower().strip())
    safe = safe.strip("-")
    return safe or "mode"

def _normalize_sample_key(key: str) -> str:
    return re.sub(r"\s+", "", (key or "").lower().strip())

def _sanitize_color(value: Optional[str], default: str) -> str:
    if not value:
        return default
    val = str(value).strip()
    return val or default

def _color_to_rgb(color: str, default: Tuple[float, float, float]) -> np.ndarray:
    try:
        rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
    except Exception:
        rgb = np.array(default, dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)

REGION_FALLBACK_TOKENS = ("arc", "dmh", "gal", "pvh", "pos", "neg")
CONDITION_POSITIVE_TOKENS = ("pos", "positive", "treated", "stim", "input_pos")
CONDITION_NEGATIVE_TOKENS = ("neg", "negative", "control", "vehicle", "input_neg")

def _resolve_region_tokens(regions: Optional[Iterable[str]]) -> Set[str]:
    tokens = {str(tok).lower().strip() for tok in (regions or []) if str(tok).strip()}
    if not tokens:
        tokens = set(REGION_FALLBACK_TOKENS)
    return tokens

def _sample_family_key(key: str, region_tokens: Iterable[str]) -> str:
    """Collapse region tokens to group ARC/DMH variants of the same animal+slice."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", (key or "").lower())
    tokens = sorted({tok.lower() for tok in region_tokens}, key=len, reverse=True)
    for token in tokens:
        if not token:
            continue
        pattern = rf"(?:(?<=_)|^){re.escape(token)}(?=_|$)"
        cleaned = re.sub(pattern, "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or _normalize_sample_key(key)

def _extract_sample_metadata(key: str, region_tokens: Iterable[str]) -> Dict[str, str]:
    tokens = [tok for tok in re.split(r"[_\-\s]+", key or "") if tok]
    lower_tokens = [tok.lower() for tok in tokens]
    animal = tokens[0].upper() if tokens else ""
    slice_id = ""
    for tok in lower_tokens:
        m = re.search(r"slice\d+", tok)
        if m:
            slice_id = m.group(0).upper()
            break
    region = ""
    for tok in lower_tokens[::-1]:
        if tok in region_tokens:
            region = tok.upper()
            break
    family = _sample_family_key(key, region_tokens)
    return {
        "animal": animal,
        "slice": slice_id,
        "region": region,
        "family": family,
    }


def _load_condition_map(base_dir: Path) -> Dict[str, str]:
    """
    Scans for All_Counts_Master.csv files in base_dir and builds a map:
    - filename_stem -> condition
    """
    mapping = {}
    print(f"[INFO] Loading condition map from {base_dir}...")
    # Find all All_Counts_Master.csv files
    try:
        # 1. Try explicit conditions.csv in base_dir
        explicit_map = base_dir / "conditions.csv"
        if explicit_map.exists():
            try:
                df = pd.read_csv(explicit_map)
                # flexible columns
                key_col = next((c for c in df.columns if c.lower() in ("filename", "file", "key", "sample", "animal", "id")), None)
                cond_col = next((c for c in df.columns if c.lower() in ("condition", "group", "cond")), None)
                
                if key_col and cond_col:
                    for _, row in df.iterrows():
                        k = str(row[key_col]).strip()
                        c = str(row[cond_col]).strip()
                        if k and c:
                            mapping[k.lower()] = c.upper()
                            # Also map stem if it's a path
                            stem = Path(k).stem
                            mapping[stem.lower()] = c.upper()
                print(f"[INFO] Loaded {len(mapping)} entries from conditions.csv")
            except Exception as e:
                print(f"[WARN] Failed to read conditions.csv: {e}")

        # 2. Scan for All_Counts_Master.csv files
        csv_files = sorted(base_dir.rglob("All_Counts_Master.csv"))
        print(f"[INFO] Found {len(csv_files)} All_Counts_Master.csv files.")
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if "filename" in df.columns and "condition" in df.columns:
                    count = 0
                    for _, row in df.iterrows():
                        fname = str(row["filename"])
                        cond = str(row["condition"]).strip()
                        if not cond:
                            continue
                        
                        # Extract stem from filename (e.g. "Arc/G520...tif" -> "G520...")
                        fname_clean = fname.replace("\\", "/")
                        stem = Path(fname_clean).stem
                        mapping[stem.lower()] = cond.upper()
                        
                        # Also map animal ID (first token) if possible
                        parts = stem.split("_")
                        if parts:
                            animal = parts[0]
                            mapping[animal.lower()] = cond.upper()
                        count += 1
                    # print(f"[DEBUG] Loaded {count} entries from {csv_file.name}")
            except Exception as e:
                print(f"[WARN] Failed to read {csv_file}: {e}")
                continue
    except Exception as e:
        print(f"[ERROR] Error loading condition map: {e}")
        pass
    
    print(f"[INFO] Total condition map size: {len(mapping)}")
    return mapping


def _infer_condition_from_path(path: Path, condition_map: Optional[Dict[str, str]] = None, sample_key: str = "") -> str:
    parts = [p.lower() for p in path.parts]
    
    # Check for explicit conditions first
    for cond in ["adult", "old"]:
        if any(cond in part for part in parts):
            return cond.upper()

    for token in CONDITION_POSITIVE_TOKENS:
        if any(token in part for part in parts):
            return "POS"
    for token in CONDITION_NEGATIVE_TOKENS:
        if any(token in part for part in parts):
            return "NEG"
            
    # Fallback: check map
    if condition_map:
        # Try sample key (stem)
        if sample_key:
            sk_lower = sample_key.lower()
            if sk_lower in condition_map:
                return condition_map[sk_lower]
            
            # Try removing region suffix (e.g. _arc, _dmh)
            # We assume the key ends with _region
            for region in REGION_FALLBACK_TOKENS:
                suffix = f"_{region}"
                if sk_lower.endswith(suffix):
                    trimmed = sk_lower[:-len(suffix)]
                    if trimmed in condition_map:
                        return condition_map[trimmed]
            
            # Try animal ID (first token of key)
            tokens = sample_key.split("_")
            if tokens:
                animal = tokens[0].lower()
                if animal in condition_map:
                    return condition_map[animal]

    return ""


def _run_coexpression_mode(
    cfg: Config,
    mode: str,
    out_root: Path,
    sample_filter: Optional[Iterable[str]] = None,
    sweep_label: Optional[str] = None,
) -> None:
    base = Path(cfg.paths.base_results_dir)
    condition_map = _load_condition_map(base)
    _ensure_dir(out_root)

    params = CoexpressionParams(
        mode=(mode or "overlap").lower().strip(),
        centroid_max_distance=cfg.coexpression.centroid_max_distance,
        blend_base_color=cfg.coexpression.blend_base_color,
        blend_other_color=cfg.coexpression.blend_other_color,
        blend_overlap_color=cfg.coexpression.blend_overlap_color,
        blend_overlap_fraction=cfg.coexpression.blend_overlap_fraction,
        blend_channel_colors=dict(cfg.coexpression.blend_channel_colors or {}),
        centroid_overlap_fraction=cfg.coexpression.centroid_overlap_fraction,
        blend_output_formats=list(cfg.coexpression.blend_output_formats or ["png"]),
        overlay_output_formats=list(cfg.coexpression.overlay_output_formats or ["png"]),
    )
    overlap_fraction = max(0.0, min(1.0, cfg.coexpression.blend_overlap_fraction or 0.0))
    centroid_overlap_fraction = max(0.0, min(1.0, cfg.coexpression.centroid_overlap_fraction or 0.0))
    region_tokens = _resolve_region_tokens(cfg.regions_enabled)
    channel_name_map = {ch.index: (ch.name or f"ch{ch.index}") for ch in cfg.channels}
    
    # Debug: Channel-Farben ausgeben
    if cfg.debug or True:  # Temporär immer aktiviert
        print("[DEBUG] Channel Configuration:")
        for ch in cfg.channels:
            print(f"  Channel {ch.index} ({ch.name or 'unnamed'}): color={ch.color or 'None'}")
        print(f"[DEBUG] Effective blend_channel_colors: {params.blend_channel_colors}")
    
    sample_filter_set: Optional[Set[str]] = None
    if sample_filter:
        sample_filter_set = {str(s).strip() for s in sample_filter if str(s).strip()}
        if not sample_filter_set:
            sample_filter_set = None

    mode_slug = _slugify_mode(params.mode)
    snapshot_name = "coexpression_config" if sweep_label is None else f"coexpression_{mode_slug}"

    runtime_params = {
        "test_mode": cfg.test_mode,
        "test_samples_per_channel": cfg.test_samples_per_channel,
        "test_seed": cfg.test_seed,
        "base_results_dir": str(base),
        "output_dir": str(out_root),
        "regions_enabled": list(cfg.regions_enabled) if cfg.regions_enabled else [],
        "use_detection_masks": cfg.use_detection_masks,
        "fallback_from_overlays": cfg.fallback_from_overlays,
        "n_channels": len(cfg.channels),
        "enabled_channels": [ch.index for ch in cfg.channels if ch.enabled],
        "coexpr_mode": params.mode,
        "coexpr_centroid_max_distance": cfg.coexpression.centroid_max_distance,
        "coexpr_blend_overlap_fraction": cfg.coexpression.blend_overlap_fraction,
        "coexpr_centroid_overlap_fraction": cfg.coexpression.centroid_overlap_fraction,
        "coexpr_sweep_label": sweep_label,
        "coexpr_sweep_sample_filter": sorted(sample_filter_set) if sample_filter_set else [],
        "channel_colors": {ch.index: ch.color for ch in cfg.channels if ch.color},  # NEW: Track channel colors
    }
    config_dict = {
        "dataset_name": cfg.dataset_name,
        "channels": [{"index": ch.index, "name": ch.name, "enabled": ch.enabled} for ch in cfg.channels],
        "combos_enabled": list(cfg.combos_enabled),
        "selected_combos": [list(c) for c in cfg.selected_combos] if cfg.selected_combos else [],
        "overlay_params": {
            "dpi": cfg.overlay_params.dpi,
            "line_width": cfg.overlay_params.line_width,
            "figsize": list(cfg.overlay_params.figsize),
            "colors_by_k": dict(cfg.overlay_params.colors_by_k),
            "single_positive_color": cfg.overlay_params.single_positive_color,
            "base_channel_index": cfg.overlay_params.base_channel_index,
            "figure_type": cfg.overlay_params.figure_type,
            "even_layout": cfg.overlay_params.even_layout,
            "overlay_style": cfg.overlay_params.overlay_style,
            "fill_alpha": cfg.overlay_params.fill_alpha,
            "save_centroid_heatmap": cfg.overlay_params.save_centroid_heatmap,
            "centroid_marker_size": cfg.overlay_params.centroid_marker_size,
        },
        "coexpression": {
            "mode": params.mode,
            "centroid_max_distance": cfg.coexpression.centroid_max_distance,
            "blend_base_color": cfg.coexpression.blend_base_color,
            "blend_other_color": cfg.coexpression.blend_other_color,
            "blend_overlap_color": cfg.coexpression.blend_overlap_color,
            "blend_overlap_fraction": cfg.coexpression.blend_overlap_fraction,
            "centroid_overlap_fraction": cfg.coexpression.centroid_overlap_fraction,
            "blend_channel_colors": dict(cfg.coexpression.blend_channel_colors or {}),
            "sweep": {
                "enabled": cfg.coexpr_sweep.enabled,
                "modes": list(cfg.coexpr_sweep.modes),
                "sample_key": cfg.coexpr_sweep.sample_key,
            },
        },
    }
    save_config_snapshot(
        output_dir=out_root,
        config_file=None,
        config_dict=config_dict,
        runtime_params=runtime_params,
        snapshot_name=snapshot_name,
    )

    if cfg.debug:
        print(f'[DEBUG] base_results_dir={base}')
        print(f'[DEBUG] output_dir={out_root}')
        print(f'[DEBUG] regions_enabled={cfg.regions_enabled}')
        print(f'[DEBUG] use_detection_masks={cfg.use_detection_masks}')
        print(f'[DEBUG] fallback_from_overlays={cfg.fallback_from_overlays}')
        print(f'[DEBUG] coexpr_mode={params.mode}')
        if sample_filter_set:
            print(f'[DEBUG] sweep sample filter={sorted(sample_filter_set)}')

    if cfg.test_mode:
        seed_note = f' (seed={cfg.test_seed})' if cfg.test_seed is not None else ''
        print(f'[TEST] Co-expression test mode: samples_per_channel={cfg.test_samples_per_channel}{seed_note} (mode={params.mode})')

    if sweep_label:
        print(f"[INFO] Co-expression sweep '{sweep_label}' using mode='{params.mode}'")
    else:
        print(f"[INFO] Co-expression mode '{params.mode}'")

    enabled_regions = set([s.upper() for s in (cfg.regions_enabled or [])])

    def region_allowed(p: Path) -> bool:
        if not enabled_regions:
            return True
        return _normalize_region_token(p.as_posix()) in enabled_regions

    enabled_channels = [ch for ch in cfg.channels if ch.enabled]
    if cfg.selected_combos:
        combos = [tuple(map(int, c)) for c in cfg.selected_combos]
    else:
        from itertools import combinations
        combos = []
        for k in cfg.combos_enabled:
            for comb in combinations([ch.index for ch in enabled_channels], int(k)):
                combos.append(tuple(comb))

    per_channel_index: Dict[int, Dict[str, Path]] = {}
    for ch in cfg.channels:
        mapping = _glob_channel_files(base, ch, cfg.use_detection_masks, cfg.fallback_from_overlays)
        per_channel_index[ch.index] = mapping
        if cfg.debug:
            if mapping:
                sample_file = next(iter(mapping.values()))
                file_type = "masks" if "_mask" in sample_file.name else "overlays"
                print(f"[DEBUG] Channel {ch.index}: {len(mapping)} {file_type} found")
            else:
                print(f"[DEBUG] Channel {ch.index}: no files found")

    all_keys = set()
    for d in per_channel_index.values():
        all_keys.update(d.keys())
    if cfg.debug:
        print(f"[DEBUG] Total unique sample keys found: {len(all_keys)}")
    key_norm_lookup: Dict[str, str] = {}
    key_to_family: Dict[str, str] = {}
    family_to_keys: Dict[str, Set[str]] = defaultdict(set)
    for original_key in all_keys:
        norm = _normalize_sample_key(original_key)
        key_norm_lookup[norm] = original_key
        family = _sample_family_key(original_key, region_tokens)
        key_to_family[original_key] = family
        family_to_keys[family].add(original_key)

    if sample_filter_set:
        matched = set()
        missing = []
        for requested in sample_filter_set:
            norm = _normalize_sample_key(requested)
            base_key = key_norm_lookup.get(norm)
            if base_key:
                fam = key_to_family.get(base_key)
                if fam:
                    matched.update(family_to_keys.get(fam, {base_key}))
                else:
                    matched.add(base_key)
                continue
            requested_family = _sample_family_key(requested, region_tokens)
            if requested_family in family_to_keys and family_to_keys[requested_family]:
                matched.update(family_to_keys[requested_family])
            else:
                missing.append(requested)
        if missing:
            print(f"[WARN] Sweep sample keys not found: {sorted(missing)}")
        if cfg.debug and matched:
            print(f"[DEBUG] Sweep sample expansion -> {sorted(matched)}")
        all_keys = matched

    if cfg.test_mode and not sample_filter_set:
        limit = cfg.test_samples_per_channel or 2
        if limit <= 0:
            limit = 2
        rng = random.Random(cfg.test_seed)
        if len(all_keys) > limit:
            sampled_keys = rng.sample(sorted(all_keys), limit)
            all_keys = set(sampled_keys)
            print(f"[TEST] Limiting co-expression to {len(all_keys)} sample keys (limit {limit}).")
    for ch_idx, mapping in per_channel_index.items():
        per_channel_index[ch_idx] = {k: v for k, v in mapping.items() if not all_keys or k in all_keys}

    if not all_keys:
        print(f"[WARN] No matching sample keys for mode '{params.mode}'. Skipping.")
        return

    total_rows = 0
    for comb in combos:
        k = len(comb)
        color_k = cfg.overlay_params.colors_by_k.get(str(k), "red")
        comb_sorted = tuple(sorted(comb))
        comb_dir_name = "ch" + "_".join(map(str, comb_sorted))
        out_dir = out_root / f"k{k}" / comb_dir_name
        _ensure_dir(out_dir)
        rows = []

        if (
            cfg.overlay_params.base_channel_index is not None
            and cfg.overlay_params.base_channel_index in comb_sorted
        ):
            base_mask_idx = comb_sorted.index(cfg.overlay_params.base_channel_index)
        else:
            base_mask_idx = 0

        for key in sorted(all_keys):
            fpaths: List[Tuple[int, Path]] = []
            preview_map: Dict[int, Optional[np.ndarray]] = {}
            for ch_idx in comb_sorted:
                fp = per_channel_index.get(ch_idx, {}).get(key)
                if not fp:
                    fpaths = []
                    preview_map = {}
                    break
                fpaths.append((ch_idx, fp))
                preview_map[ch_idx] = _load_channel_preview(base, ch_idx, key, fp)
            if not fpaths:
                continue

            if not region_allowed(fpaths[0][1]):
                continue

            masks: List[np.ndarray] = []
            for _, fp in fpaths:
                is_mask = "_mask" in fp.name and fp.suffix.lower() in (".tif", ".tiff")
                if is_mask:
                    m = _read_mask(fp)
                else:
                    m = _read_overlay_and_threshold(fp)
                masks.append(m)

            normalized_masks, _ = _normalize_masks(masks, base_mask_idx)
            base_mask = normalized_masks[base_mask_idx]
            base_label_cached = None
            base_props_cached = None
            mode_lower = params.mode
            if mode_lower in {"intersection", "all"}:
                effective_overlap_fraction = centroid_overlap_fraction
            else:
                effective_overlap_fraction = overlap_fraction
            current_overlap_fraction = effective_overlap_fraction if mode_lower in OVERLAP_MODES else 0.0
            if params.mode == "blend":
                base_label_cached, base_props_cached = _label_mask(base_mask)
            co_mask = _compute_coexpression_mask(
                normalized_masks,
                base_mask_idx,
                params,
                min_overlap_fraction=current_overlap_fraction,
            )
            blend_overlap_mask = None
            if params.mode == "blend":
                if base_label_cached is None or base_props_cached is None:
                    base_label_cached, base_props_cached = _label_mask(base_mask)
                blend_overlap_mask = _compute_overlap_mask(
                    normalized_masks,
                    base_mask_idx,
                    base_label_cached,
                    base_props_cached or [],
                    min_fraction=current_overlap_fraction,
                )
            condition = _infer_condition_from_path(fpaths[0][1], condition_map, key) if fpaths else ""

            labels = measure.label(co_mask.astype(np.uint8), connectivity=1)
            props = measure.regionprops(labels)
            n_co = len(props)

            area_px = int(co_mask.sum())
            meta = _extract_sample_metadata(key, region_tokens)
            channel_labels = " + ".join(
                [channel_name_map.get(idx, f"ch{idx}") or f"ch{idx}" for idx in comb_sorted]
            )
            rows.append({
                "sample_key": key,
                "animal_id": meta["animal"],
                "region": meta["region"],
                "slice_id": meta["slice"],
                "family_id": meta["family"],
                "condition": condition,
                "k": k,
                "channels": ",".join(map(str, comb_sorted)),
                "channel_labels": channel_labels,
                "n_coexpressing": n_co,
                "area_px": area_px,
                "coexpr_mode": params.mode,
                "sweep_label": sweep_label or "",
            })
            total_rows += 1

            previews_for_plot = [(preview_map.get(idx), f"ch{idx}", idx) for idx in comb_sorted]
            base_preview = None
            if cfg.overlay_params.base_channel_index is not None:
                base_preview = preview_map.get(cfg.overlay_params.base_channel_index)
            if base_preview is None and previews_for_plot:
                base_preview = previews_for_plot[0][0]
            if base_preview is None:
                base_preview = base_mask.astype(np.float32)

            ov = out_dir / f"{key}_coexpr_overlay.png"
            overlay_style = (cfg.overlay_params.overlay_style or "mixed").lower()
            if overlay_style == "blend":
                blend_tif_path = ov.with_suffix(".tif")
                _save_blend_mask(
                    normalized_masks,
                    base_mask_idx,
                    comb_sorted,
                    co_mask,
                    ov,
                    blend_tif_path,
                    params.blend_base_color,
                    params.blend_channel_colors,
                    params.blend_other_color,
                    params.blend_overlap_color,
                    background_img=base_preview,
                    min_overlap_fraction=overlap_fraction,
                    output_formats=params.blend_output_formats,
                )
            else:
                # Verwende multichannel overlay wenn Channel-Farben verfügbar
                if params.blend_channel_colors and len(comb_sorted) > 1:
                    _save_multichannel_overlay(
                        base_preview,
                        normalized_masks,
                        comb_sorted,
                        co_mask,
                        ov,
                        channel_colors=params.blend_channel_colors,
                        overlap_color=params.blend_overlap_color,
                        lw=cfg.overlay_params.line_width,
                        dpi=cfg.overlay_params.dpi,
                        figsize=tuple(cfg.overlay_params.figsize),
                        style=cfg.overlay_params.overlay_style,
                        fill_alpha=cfg.overlay_params.fill_alpha,
                        show_only_coexpressing=True,  # Nur doppelpositive anzeigen
                    )
                else:
                    _save_overlay(
                        base_preview,
                        base_mask,
                        co_mask,
                        ov,
                        multi_color=color_k,
                        lw=cfg.overlay_params.line_width,
                        dpi=cfg.overlay_params.dpi,
                        figsize=tuple(cfg.overlay_params.figsize),
                        style=cfg.overlay_params.overlay_style,
                        fill_alpha=cfg.overlay_params.fill_alpha,
                    )
            if params.mode == "blend" or overlay_style == "blend":
                blend_png = out_dir / f"{key}_coexpr_blend.png"
                blend_tif = out_dir / f"{key}_coexpr_blend.tif"
                overlap_for_blend = blend_overlap_mask if blend_overlap_mask is not None else co_mask
                _save_blend_mask(
                    normalized_masks,
                    base_mask_idx,
                    comb_sorted,
                    overlap_for_blend,
                    blend_png,
                    blend_tif,
                    params.blend_base_color,
                    params.blend_channel_colors,
                    params.blend_other_color,
                    params.blend_overlap_color,
                    background_img=base_preview,
                    min_overlap_fraction=overlap_fraction,
                    output_formats=params.blend_output_formats,
                )
            if cfg.overlay_params.save_centroid_heatmap and params.mode in CENTROID_MODES:
                centroid_png = out_dir / f"{key}_coexpr_centroid.png"
                _save_centroid_heatmap(
                    base_preview,
                    base_mask,
                    props,
                    centroid_png,
                    color_k,
                    cfg.overlay_params.centroid_marker_size,
                    cfg.overlay_params.dpi,
                )
            if cfg.save_composite_figure:
                figure_path = out_dir / f"{key}_coexpr_figure.png"
                _save_composite_figure(
                    previews_for_plot,
                    normalized_masks,
                    co_mask,
                    figure_path,
                    cfg.overlay_params.dpi,
                    multi_color=color_k,
                    line_width=cfg.overlay_params.line_width,
                    base_channel_index=cfg.overlay_params.base_channel_index,
                    base_mask_index=base_mask_idx,
                    figure_type=cfg.overlay_params.figure_type,
                    even_layout=cfg.overlay_params.even_layout,
                )

        if rows:
            df = pd.DataFrame(rows)
            preferred_cols = [
                "sample_key",
                "animal_id",
                "region",
                "slice_id",
                "family_id",
                "condition",
                "k",
                "channels",
                "channel_labels",
                "n_coexpressing",
                "area_px",
                "coexpr_mode",
                "sweep_label",
            ]
            ordered_cols = [c for c in preferred_cols if c in df.columns]
            remaining_cols = [c for c in df.columns if c not in ordered_cols]
            df = df[ordered_cols + remaining_cols]
            df.to_csv(out_dir / "coexpr_summary.csv", index=False)

        mode_note = "detection masks" if cfg.use_detection_masks else "overlay images"
        (out_dir / "README.txt").write_text(
            f"Combination: k={k}, channels={comb_sorted}\n"
            f"Files derived from base: {base}\n"
            f"Analysis mode: {mode_note} (fallback_enabled={cfg.fallback_from_overlays})\n"
            f"Co-expression mode: {params.mode}\n"
            + (f"Sweep label: {sweep_label}\n" if sweep_label else "")
            + f"Composite figures saved: {cfg.save_composite_figure}\n"
            f"Rows: {len(rows)}\n",
            encoding="utf-8"
        )
        print(f"[INFO] mode={params.mode} k={k}, channels={comb_sorted} -> {len(rows)} samples -> {out_dir}")

    print(f"[DONE] Co-expression analysis finished for mode '{params.mode}' (rows={total_rows}).")


def run_coexpression(cfg: Config) -> None:
    base_out_root = Path(cfg.paths.output_dir)
    if cfg.test_mode:
        base_out_root = base_out_root / "__test__"

    base_mode = (cfg.coexpression.mode or "overlap").lower()
    plan: List[Tuple[str, Path, Optional[str], Optional[List[str]]]] = []

    def add_mode(mode_value: str, sweep_label: Optional[str], sample_filter: Optional[List[str]]) -> None:
        mode_norm = (mode_value or "overlap").lower().strip()
        if mode_norm not in ALLOWED_COEXPR_MODES:
            print(f"[WARN] Unsupported co-expression mode '{mode_value}', falling back to 'overlap'.")
            mode_norm = "overlap"
        out_root = base_out_root if sweep_label is None else base_out_root / "__sweep__" / _slugify_mode(sweep_label)
        plan.append((mode_norm, out_root, sweep_label, sample_filter))

    sweep_modes = [str(m).strip() for m in (cfg.coexpr_sweep.modes or []) if str(m).strip()]
    sweep_active = bool(cfg.coexpr_sweep.enabled and sweep_modes)

    if not sweep_active:
        add_mode(base_mode, None, None)

    if sweep_active:
        sample = cfg.coexpr_sweep.sample_key.strip() if cfg.coexpr_sweep.sample_key else ""
        sample_filter = [sample] if sample else None
        for mode_value in sweep_modes:
            label = mode_value.strip() or mode_value
            add_mode(mode_value, label, sample_filter)

    for mode_value, out_root, sweep_label, sample_filter in plan:
        _run_coexpression_mode(cfg, mode_value, out_root, sample_filter=sample_filter, sweep_label=sweep_label)
    master_path = _write_coexpr_master_summary(base_out_root)
    if master_path:
        print(f"[INFO] Combined sweep summary written -> {master_path}")


def _label_mask(mask: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    label_img = measure.label(mask.astype(bool), connectivity=1)
    props = measure.regionprops(label_img)
    return label_img, props


def _compute_overlap_mask(
    masks: Sequence[np.ndarray],
    base_idx: int,
    base_label: np.ndarray,
    base_props: Sequence[Any],
    min_fraction: float = 0.0,
) -> np.ndarray:
    if not base_props:
        return np.zeros_like(masks[base_idx], dtype=bool)
    other_masks = [np.asarray(m, dtype=bool) for i, m in enumerate(masks) if i != base_idx]
    if not other_masks:
        return np.zeros_like(masks[base_idx], dtype=bool)
    co_mask = np.zeros_like(masks[base_idx], dtype=bool)
    for prop in base_props:
        cell_mask = base_label == prop.label
        cell_area = max(int(prop.area), 1)
        match_all = True
        for om in other_masks:
            overlap_pixels = cell_mask & om
            if min_fraction > 0:
                overlap_ratio = float(overlap_pixels.sum()) / cell_area
                if overlap_ratio < min_fraction:
                    match_all = False
                    break
            else:
                if not np.any(overlap_pixels):
                    match_all = False
                    break
        if match_all:
            co_mask[cell_mask] = True
    return co_mask


def _compute_centroid_mask(
    masks: Sequence[np.ndarray],
    base_idx: int,
    base_label: np.ndarray,
    base_props: Sequence[Any],
    max_distance: float,
) -> np.ndarray:
    if not base_props or max_distance < 0:
        return np.zeros_like(masks[base_idx], dtype=bool)
    other_centroids: List[np.ndarray] = []
    for idx, mask in enumerate(masks):
        if idx == base_idx:
            continue
        label_img = measure.label(mask.astype(bool), connectivity=1)
        props = measure.regionprops(label_img)
        centroids = np.array([prop.centroid for prop in props], dtype=np.float32) if props else np.empty((0, 2), dtype=np.float32)
        other_centroids.append(centroids)
    if any(cents.size == 0 for cents in other_centroids):
        return np.zeros_like(masks[base_idx], dtype=bool)
    threshold_sq = float(max_distance) ** 2
    co_mask = np.zeros_like(masks[base_idx], dtype=bool)
    for prop in base_props:
        centroid = np.array(prop.centroid, dtype=np.float32)
        match = True
        for cents in other_centroids:
            diffs = cents - centroid
            dist_sq = np.sum(diffs * diffs, axis=1)
            if not np.any(dist_sq <= threshold_sq):
                match = False
                break
        if match:
            co_mask[base_label == prop.label] = True
    return co_mask


def _compute_coexpression_mask(
    masks: Sequence[np.ndarray],
    base_idx: int,
    params: CoexpressionParams,
    min_overlap_fraction: float = 0.0,
) -> np.ndarray:
    base_mask = np.asarray(masks[base_idx], dtype=bool)
    base_label, base_props = _label_mask(base_mask)
    if not base_props:
        return np.zeros_like(base_mask, dtype=bool)
    mode = (params.mode or "overlap").lower()
    need_overlap = mode in {"overlap", "either", "both", "union", "intersection", "all", "blend"}
    need_centroid = mode in {"centroid", "either", "both", "union", "intersection", "all"}
    overlap_mask = _compute_overlap_mask(
        masks,
        base_idx,
        base_label,
        base_props,
        min_fraction=min_overlap_fraction if need_overlap else 0.0,
    ) if need_overlap else None
    centroid_mask = _compute_centroid_mask(
        masks,
        base_idx,
        base_label,
        base_props,
        max(params.centroid_max_distance, 0.0),
    ) if need_centroid else None
    if mode in {"centroid"}:
        return centroid_mask if centroid_mask is not None else np.zeros_like(base_mask, dtype=bool)
    if mode in {"overlap"}:
        return overlap_mask if overlap_mask is not None else np.zeros_like(base_mask, dtype=bool)
    if mode in {"blend"}:
        return overlap_mask if overlap_mask is not None else np.zeros_like(base_mask, dtype=bool)
    if mode in {"either", "both", "union"}:
        result = np.zeros_like(base_mask, dtype=bool)
        if overlap_mask is not None:
            result |= overlap_mask
        if centroid_mask is not None:
            result |= centroid_mask
        return result
    if mode in {"intersection", "all"}:
        if overlap_mask is None and centroid_mask is None:
            return np.zeros_like(base_mask, dtype=bool)
        if overlap_mask is None:
            return centroid_mask
        if centroid_mask is None:
            return overlap_mask
        return overlap_mask & centroid_mask
    # Fallback
    return overlap_mask if overlap_mask is not None else (
        centroid_mask if centroid_mask is not None else np.zeros_like(base_mask, dtype=bool)
    )


def _write_coexpr_master_summary(base_out_root: Path) -> Optional[Path]:
    print(f"[INFO] Generating master summary from {base_out_root}...")
    summary_paths = sorted(base_out_root.rglob("coexpr_summary.csv"))
    if not summary_paths:
        print("[WARN] No coexpr_summary.csv files found.")
        return None
    print(f"[INFO] Found {len(summary_paths)} summary files.")
    frames: List[pd.DataFrame] = []
    for csv_path in summary_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
            continue
        if df is None or df.empty and not list(df.columns):
            continue
        frames.append(df)
    if not frames:
        print("[WARN] No valid data frames collected.")
        return None
    
    try:
        master = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"[ERROR] Failed to concat frames: {e}")
        return None

    drop_cols = [c for c in ("source_relpath", "source_k", "source_combination", "source_mode_dir") if c in master.columns]
    if drop_cols:
        master = master.drop(columns=drop_cols)
    preferred_cols = [
        "coexpr_mode",
        "sweep_label",
        "sample_key",
        "animal_id",
        "region",
        "slice_id",
        "family_id",
        "condition",
        "k",
        "channels",
        "channel_labels",
        "n_coexpressing",
        "area_px",
    ]
    ordered_cols = [c for c in preferred_cols if c in master.columns]
    remaining_cols = [c for c in master.columns if c not in ordered_cols]
    master = master[ordered_cols + remaining_cols]
    sort_cols = [c for c in ["sweep_label", "coexpr_mode", "animal_id", "slice_id", "channels"] if c in master.columns]
    if sort_cols:
        master = master.sort_values(sort_cols, ignore_index=True)
    out_path = base_out_root / "coexpr_sweep_summary.csv"
    try:
        master.to_csv(out_path, index=False)
        print(f"[INFO] Master summary written to {out_path}")
        return out_path
    except Exception as e:
        print(f"[ERROR] Failed to write master summary: {e}")
        return None

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

def _save_composite_figure(
    images: Sequence[Tuple[Optional[np.ndarray], str, int]],
    mask_list: Sequence[np.ndarray],
    co_mask: np.ndarray,
    out_path: Path,
    dpi: int,
    multi_color: str = "magenta",
    line_width: int = 2,
    base_channel_index: Optional[int] = None,
    base_mask_index: int = 0,
    figure_type: str = "overlay",
    even_layout: str = "line",
) -> None:
    """Create a composite figure using either overlays or raw masks."""
    if figure_type not in ("overlay", "mask"):
        figure_type = "overlay"
    total_masks = len(mask_list)
    if total_masks == 0:
        return

    base_mask_idx = int(np.clip(base_mask_index, 0, total_masks - 1))
    normalized_masks, target_shape = _normalize_masks(mask_list, base_mask_idx)
    base_mask = normalized_masks[base_mask_idx]

    base_preview = None
    if base_channel_index is not None:
        for img, _, idx in images:
            if idx == base_channel_index and img is not None:
                base_preview = img
                break
    if base_preview is None and images and images[0][0] is not None:
        base_preview = images[0][0]

    background = _extract_background(base_preview)
    base_arr = _prepare_display(background)
    use_mask_panels = figure_type == "mask"
    if base_arr.size == 0 or use_mask_panels:
        base_arr = _prepare_display(base_mask.astype(np.float32))
    target_shape = tuple(base_arr.shape[:2]) if base_arr.ndim >= 2 else target_shape
    outline_thickness = _line_width_to_thickness(line_width)
    co_mask_rs = _resize_bool(co_mask, target_shape)
    colored = _apply_colored_outline(base_arr, co_mask_rs, multi_color, outline_thickness)

    # Determine layout
    total_panels = len(images) + 1
    if even_layout == "grid" and total_panels >= 4:
        rows = 2
        cols = int(math.ceil(total_panels / 2))
    else:
        rows = 1
        cols = total_panels
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for panel_idx, (ax, (img, title, _)) in enumerate(zip(axes_flat[: len(images)], images)):
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        if use_mask_panels:
            mask_disp = _prepare_display(normalized_masks[panel_idx].astype(np.float32))
            ax.imshow(mask_disp, cmap="gray", interpolation="bilinear")
            continue
        if img is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="red", fontsize=10)
            continue
        arr = _prepare_display(img)
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", interpolation="bilinear")
        else:
            ax.imshow(arr, interpolation="bilinear")

    last_ax = axes_flat[len(images)]
    last_ax.set_title("Co-expression", fontsize=10)
    last_ax.set_axis_off()
    last_ax.imshow(colored, interpolation="bilinear")

    for ax in axes_flat[len(images) + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    # Use higher DPI for better quality
    effective_dpi = max(dpi, 300)
    fig.savefig(out_path, dpi=effective_dpi, bbox_inches="tight")
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

def _save_overlay(
    base_img: Optional[np.ndarray],
    base_mask: np.ndarray,
    co_mask: np.ndarray,
    out_path: Path,
    multi_color: str,
    lw: int,
    dpi: int,
    figsize: Sequence[int],
    style: str = "contour",
    fill_alpha: float = 0.45,
) -> None:
    """
    Save overlay showing ONLY co-expressing cells on the base channel background.
    Cells are clearly circled with thick contours.
    """
    base_mask = np.asarray(base_mask, dtype=bool)
    co_mask = np.asarray(co_mask, dtype=bool)
    if base_mask.size == 0:
        return
    
    # Get background from base image
    background = _extract_background(base_img)
    display = _prepare_display(background)
    if display.size > 0:
        target_shape = tuple(display.shape[:2])
    else:
        target_shape = base_mask.shape
        display = _prepare_display(base_mask.astype(np.float32))
    
    # Resize co_mask to target shape
    co_mask_rs = _resize_bool(co_mask, target_shape)
    
    # Start with background
    styled = _ensure_rgb(display).copy()
    
    style_norm = (style or "contour").lower()
    has_fill = style_norm in ("filled", "mixed")
    has_contour = style_norm in ("contour", "mixed")
    
    # Use thicker lines for better visibility (minimum 2, scale with config)
    thickness = max(2, _line_width_to_thickness(lw) + 1)
    
    # Apply fill if requested (semi-transparent)
    if has_fill and np.any(co_mask_rs):
        styled = _apply_colored_fill(styled, co_mask_rs, multi_color, alpha=fill_alpha)
    
    # Always draw thick contours for clear cell circling
    if np.any(co_mask_rs):
        # Use thicker outline for visibility
        styled = _apply_colored_outline(styled, co_mask_rs, multi_color, thickness)
    
    # High quality output
    effective_dpi = max(dpi, 300)
    fig = plt.figure(figsize=figsize, dpi=effective_dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(styled, interpolation="bilinear")
    ax.axis("off")
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=effective_dpi)
    plt.close(fig)


def _save_multichannel_overlay(
    base_img: Optional[np.ndarray],
    mask_list: Sequence[np.ndarray],
    combo_channels: Sequence[int],
    co_mask: np.ndarray,
    out_path: Path,
    channel_colors: Mapping[int, str],
    overlap_color: str,
    lw: int,
    dpi: int,
    figsize: Sequence[int],
    style: str = "contour",
    fill_alpha: float = 0.45,
    output_formats: Sequence[str] = ("png",),
    show_only_coexpressing: bool = True,
) -> None:
    """
    Creates overlay showing co-expressing cells clearly circled.
    
    When show_only_coexpressing=True (default):
    - Only co-expressing cells are shown with thick colored circles
    - Background is the base channel image
    
    When show_only_coexpressing=False:
    - Individual channels shown in their colors
    - Co-expressing regions highlighted in overlap_color
    """
    if not mask_list:
        return
    
    background = _extract_background(base_img)
    display = _prepare_display(background)
    if display.size > 0:
        target_shape = tuple(display.shape[:2])
    else:
        target_shape = mask_list[0].shape
        display = _prepare_display(mask_list[0].astype(np.float32))
    
    styled = _ensure_rgb(display).copy()
    style_norm = (style or "contour").lower()
    has_fill = style_norm in ("filled", "mixed")
    has_contour = style_norm in ("contour", "mixed")
    
    # Use thicker lines for better visibility
    thickness = max(2, _line_width_to_thickness(lw) + 1)
    
    # Resize co_mask
    co_mask_rs = _resize_bool(np.asarray(co_mask, dtype=bool), target_shape)
    
    if show_only_coexpressing:
        # ONLY show co-expressing cells - clear and simple
        if has_fill and co_mask_rs.any():
            styled = _apply_colored_fill(styled, co_mask_rs, overlap_color, alpha=fill_alpha)
        if co_mask_rs.any():
            styled = _apply_colored_outline(styled, co_mask_rs, overlap_color, thickness)
    else:
        # Show individual channels + co-expressing overlay
        for idx, mask in enumerate(mask_list):
            if idx >= len(combo_channels):
                continue
            ch_id = combo_channels[idx]
            ch_color = channel_colors.get(ch_id, "#808080")
            
            mask_rs = _resize_bool(np.asarray(mask, dtype=bool), target_shape)
            # Only areas that are NOT co-expressing
            ch_single = mask_rs & ~co_mask_rs
            
            if has_fill and ch_single.any():
                styled = _apply_colored_fill(styled, ch_single, ch_color, alpha=fill_alpha * 0.6)
            if has_contour and ch_single.any():
                styled = _apply_colored_outline(styled, ch_single, ch_color, thickness)
        
        # Draw co-expressing regions in overlap_color
        if has_fill and co_mask_rs.any():
            styled = _apply_colored_fill(styled, co_mask_rs, overlap_color, alpha=fill_alpha)
        if co_mask_rs.any():
            styled = _apply_colored_outline(styled, co_mask_rs, overlap_color, thickness)
    
    # High quality output
    effective_dpi = max(dpi, 300)
    fig = plt.figure(figsize=figsize, dpi=effective_dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(styled, interpolation="bilinear")
    ax.axis("off")
    _ensure_dir(out_path.parent)
    
    # Save in requested formats
    formats = [fmt.lower().strip() for fmt in output_formats] if output_formats else ["png"]
    
    for fmt in formats:
        if fmt == "png":
            out_file = out_path.with_suffix('.png')
            fig.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=effective_dpi)
        elif fmt in ("tiff", "tif"):
            out_file = out_path.with_suffix('.tif')
            try:
                styled_uint8 = (np.clip(styled, 0.0, 1.0) * 255).astype(np.uint8)
                if styled_uint8.ndim == 2:
                    styled_uint8 = np.stack([styled_uint8] * 3, axis=-1)
                tiff.imwrite(str(out_file), styled_uint8, photometric="rgb")
            except Exception:
                pass
        elif fmt in ("jpg", "jpeg"):
            out_file = out_path.with_suffix('.jpg')
            fig.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=effective_dpi, format='jpg')
    
    plt.close(fig)


def _save_blend_mask(
    mask_list: Sequence[np.ndarray],
    base_idx: int,
    combo_channels: Sequence[int],
    overlap_mask: np.ndarray,
    out_png: Path,
    out_tiff: Optional[Path],
    base_color: str,
    channel_color_map: Mapping[int, str],
    other_color: str,
    overlap_color: str,
    background_img: Optional[np.ndarray] = None,
    min_overlap_fraction: float = 0.02,
    output_formats: Sequence[str] = ("png",),
) -> None:
    """
    Erstellt eine Blend-Heatmap mit korrektem Overlap-Threshold.
    
    Logik:
    1. Basis-Zellen (aus base_mask) werden in base_color eingefärbt
    2. Partner-Zellen werden in ihren jeweiligen Farben eingefärbt
    3. Nur Basis-Zellen, deren Overlap mit ALLEN Partner-Kanälen >= min_overlap_fraction ist,
       werden als Mehrfach-Positiv (overlap_color) markiert
    
    output_formats: Liste von Formaten zum Speichern (png, tiff, jpg)
    """
    base_mask = np.asarray(mask_list[base_idx], dtype=bool)
    if base_mask.size == 0:
        return
    
    # Channel IDs zuordnen
    if not combo_channels or len(combo_channels) != len(mask_list):
        combo_channels = list(range(len(mask_list)))
    
    # Channel-Farben direkt verwenden (channel_color_map ist bereits korrekt aufbereitet)
    base_channel_id = combo_channels[base_idx] if base_idx < len(combo_channels) else combo_channels[0]
    
    # Partner-Masken sammeln
    partner_masks: List[Tuple[int, np.ndarray]] = []
    for idx, mask in enumerate(mask_list):
        if idx == base_idx:
            continue
        cur = np.asarray(mask, dtype=bool)
        partner_masks.append((combo_channels[idx], cur))
    
    # Basis-Zellen labeln für zellweise Overlap-Prüfung
    # Use passed overlap_mask if available to avoid recalculation and ensure consistency
    if overlap_mask is not None and np.any(overlap_mask):
        overlap_bool = np.asarray(overlap_mask, dtype=bool)
        # Ensure overlap_bool matches base_mask shape
        if overlap_bool.shape != base_mask.shape:
             overlap_bool = _resize_bool(overlap_bool, base_mask.shape)
    else:
        base_label = measure.label(base_mask.astype(bool), connectivity=1)
        base_props = measure.regionprops(base_label)
        
        # Mehrfach-Positive identifizieren (mit Threshold)
        overlap_bool = np.zeros_like(base_mask, dtype=bool)
        
        for prop in base_props:
            cell_mask = base_label == prop.label
            cell_area = max(int(prop.area), 1)
            
            # Prüfe Overlap mit ALLEN Partner-Kanälen
            is_multi_positive = True
            for ch_idx, partner_mask in partner_masks:
                overlap_pixels = cell_mask & partner_mask
                overlap_ratio = float(overlap_pixels.sum()) / cell_area
                
                if overlap_ratio < min_overlap_fraction:
                    is_multi_positive = False
                    break
            
            if is_multi_positive:
                overlap_bool[cell_mask] = True
    
    # Kategorien erstellen
    base_only = base_mask & ~overlap_bool
    
    # Partner-only: Pixel in Partner-Masken, die NICHT in overlap_bool sind
    other_union = np.zeros_like(base_mask, dtype=bool)
    for _, ch_mask in partner_masks:
        other_union |= ch_mask
    other_only = other_union & ~overlap_bool & ~base_mask
    
    # Label-Image für TIFF (0=Hintergrund, 1=base_only, 2=other_only, 3=overlap)
    label = np.zeros_like(base_mask, dtype=np.uint8)
    label[base_only] = 1
    label[other_only] = 2
    label[overlap_bool] = 3
    
    # Basis-Bild für Hintergrund
    # display = None
    # if background_img is not None:
    #     display = _extract_background(background_img)
    # if display is None or display.size == 0:
    #     display = base_mask.astype(np.float32)
    
    # User requested to remove background channel, so we use a black image
    display = np.zeros_like(base_mask, dtype=np.float32)
    blend = _prepare_display(display)
    target_shape = blend.shape[:2]
    
    # Ensure channel_color_map has int keys to avoid lookup failures
    if channel_color_map:
        channel_color_map = {int(k): v for k, v in channel_color_map.items()}

    # Farben anwenden - Basis-Kanal in seiner Farbe (exakt wie eingestellt, ohne Transparenz)
    # Prioritize channel_color_map, fallback to base_color only if missing
    base_color_value = channel_color_map.get(base_channel_id) or base_color
    base_only_rs = _resize_bool(base_only, target_shape)
    blend = _apply_colored_fill(blend, base_only_rs, base_color_value, alpha=1.0)
    
    # Jeder Partner-Kanal bekommt seine eigene Farbe (exakt wie eingestellt, ohne Transparenz)
    for ch_idx, ch_mask in partner_masks:
        partner_only = ch_mask & ~overlap_bool & ~base_mask
        if not partner_only.any():
            continue
        partner_only_rs = _resize_bool(partner_only, target_shape)
        # Prioritize channel_color_map, fallback to other_color only if missing
        partner_color = channel_color_map.get(ch_idx) or other_color
        blend = _apply_colored_fill(blend, partner_only_rs, partner_color, alpha=1.0)
    
    # Mehrfach-Positive in overlap_color highlighten (exakt wie eingestellt, ohne Transparenz)
    overlap_bool_rs = _resize_bool(overlap_bool, target_shape)
    blend = _apply_colored_fill(blend, overlap_bool_rs, overlap_color, alpha=1.0)
    
    # Speichere in gewählten Formaten
    _ensure_dir(out_png.parent)
    formats = [fmt.lower().strip() for fmt in output_formats] if output_formats else ["png"]
    
    for fmt in formats:
        if fmt == "png":
            out_file = out_png.with_suffix('.png')
            # Use uint8 for PNG to ensure compatibility with PIL/imageio
            blend_uint8 = (np.clip(blend, 0.0, 1.0) * 255).astype(np.uint8)
            skio.imsave(str(out_file), blend_uint8, check_contrast=False)
        elif fmt in ("tiff", "tif"):
            out_file = out_png.with_suffix('.tif')
            # Removed try-except to expose errors
            blend_uint8 = (np.clip(blend, 0.0, 1.0) * 255).astype(np.uint8)
            if blend_uint8.ndim == 2:
                blend_uint8 = np.stack([blend_uint8] * 3, axis=-1)
            tiff.imwrite(str(out_file), blend_uint8, photometric="rgb")
        elif fmt in ("jpg", "jpeg"):
            out_file = out_png.with_suffix('.jpg')
            # Removed try-except to expose errors
            blend_uint8 = (np.clip(blend, 0.0, 1.0) * 255).astype(np.uint8)
            skio.imsave(str(out_file), blend_uint8, check_contrast=False, quality=95)


def _save_centroid_heatmap(
    base_img: Optional[np.ndarray],
    fallback_mask: np.ndarray,
    props: Sequence[Any],
    out_path: Path,
    multi_color: str,
    marker_size: int,
    dpi: int,
) -> None:
    if not props:
        return
    display = _extract_background(base_img)
    if display.size == 0:
        display = _prepare_display(fallback_mask.astype(np.float32))
    coords = [(float(p.centroid[0]), float(p.centroid[1])) for p in props if p.centroid is not None]
    if not coords:
        return
    ys, xs = zip(*coords)
    effective_dpi = max(dpi, 300)
    fig = plt.figure(figsize=(display.shape[1] / effective_dpi, display.shape[0] / effective_dpi), dpi=effective_dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(display, interpolation="bilinear")
    ax.scatter(xs, ys, s=max(marker_size, 10), c=multi_color, alpha=0.8, edgecolors="none")
    ax.axis("off")
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=effective_dpi)
    plt.close(fig)

# ------------------------------------------------------------
# Core
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
