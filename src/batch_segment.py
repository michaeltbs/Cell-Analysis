#!/usr/bin/env python3
"""
batch_segment_v4.py — Detection with Cellpose v4 (CellposeModel) + multi-channel TIFF support

- Works with Cellpose 4+ (falls back to older API if needed)
- Accepts multi-channel images; you can select which channel to segment
- Reads config from YAML (similar to your previous config_det_cpsam.yaml)
- Saves *_mask.tif and optional overlays + a simple All_Counts_Master.csv
- English-only logs

YAML (example)
--------------
paths:
  input_pos: "./data/processed_tiffs/Input_pos"
  input_neg: "./data/processed_tiffs/Input_neg"   # optional
  output_root: "./results_det"

regions_enabled: []  # optional, tokens matched in path

cellpose:
  model_name: "cyto2"        # cyto2|cyto3|cyto|nuclei|cpsam(alias->cyto2)
  use_gpu: false
  channel: 0                 # which image channel to segment on multi-channel inputs
  diameter: 20               # null/0 -> auto
  flow_threshold: 0.4
  cellprob_threshold: 0.0

overlays:
  overlay_mode: "contours"   # contours|masks|both
  contour_color: "lime"
  line_width: 1
  dpi: 300
  figsize: [12, 12]

outputs:
  overlay_suffix: "_overlay.png"
  resume: true
  overwrite: false

Notes
-----
- Multi-channel handling: if image is (Y, X, C) we extract img[..., channel].
  If (C, Y, X) we extract img[channel, ...]. If single-channel, we use it directly.
- Cellpose channels param is set to [0, 0] for grayscale input (cellpose convention).
- A minimal All_Counts_Master.csv is written per condition (pos/neg) + region folder.
"""
from __future__ import annotations

import os
import sys
import csv
import yaml
import math
import time
import random
import re
import traceback
import copy
from pathlib import Path
from typing import Iterable, Tuple, Optional, List, Dict, Any, Set

import numpy as np
import tifffile as tiff

from scipy import ndimage as ndi
from skimage import (
    exposure,
    feature,
    measure,
    morphology,
    segmentation,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x  # no-op fallback

try:
    from src.config_archiver import save_config_snapshot
except Exception:
    # fallback if not available
    def save_config_snapshot(*args, **kwargs):
        pass

try:
    from src.config_utils import (
    apply_magnification_scaling as _apply_magnification_scaling,
    apply_image_aware_scaling,
    compute_effective_scale,
)
except Exception:
    # fallback: define locally if import fails
    _apply_magnification_scaling = None
    apply_image_aware_scaling = None
    compute_effective_scale = None

# -----------------------------
# Config I/O
# -----------------------------
def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# -----------------------------
# Fallback for magnification scaling if import failed
# -----------------------------
if _apply_magnification_scaling is None:
    import copy as _copy
    def _apply_magnification_scaling(cfg: dict) -> dict:
        """Fallback: Return config unchanged if central function not available."""
        return _copy.deepcopy(cfg or {})

if apply_image_aware_scaling is None:
    import copy as _copy
    def apply_image_aware_scaling(cfg: dict, image_shape: tuple = None, resize_max: int = 2048) -> tuple:
        """Fallback: Return config unchanged with default scale info if central function not available."""
        cfg_copy = _copy.deepcopy(cfg or {})
        scale_info = {
            "effective_diameter": cfg_copy.get("cellpose", {}).get("diameter", 30),
            "resize_factor": 1.0,
            "combined_scale": 1.0,
            "quality_warning": None
        }
        return cfg_copy, scale_info

# -----------------------------
# Cellpose v4 model loader
# -----------------------------
def _get_gpu_device():
    """Detect and return the best available GPU device for PyTorch/Cellpose.
    
    Returns:
        tuple: (device_name, is_available) where device_name is 'cuda', 'mps', or None
    """
    try:
        import torch
    except ImportError:
        return None, False
    
    # Check CUDA first (NVIDIA GPUs)
    if torch.cuda.is_available():
        return 'cuda', True
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', True
    
    return None, False


_MODEL_CACHE: Dict[Tuple[str, bool], Any] = {}


def _load_cellpose_model(model_name: str, use_gpu: bool = False):
    """Return a Cellpose model that works on v4+, with fallback for older versions.

    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends.
    Models are cached per (model_type, gpu) pair to avoid reloading on every batch.
    """
    try:
        from cellpose import models as _models
    except Exception as e:
        raise RuntimeError("Cellpose is not installed. Please `pip install cellpose`.") from e

    name = (model_name or "cyto2").lower()
    valid_models = {"cyto3", "cyto2", "cyto", "nuclei", "cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"}
    mtype = name if name in valid_models else "cyto2"

    # Determine GPU device
    gpu_device, gpu_available = _get_gpu_device()

    if use_gpu and not gpu_available:
        print("[WARN] GPU requested but no GPU backend available. Using CPU.")
        use_gpu = False

    cache_key = (mtype, bool(use_gpu))
    if cache_key in _MODEL_CACHE:
        cached = _MODEL_CACHE[cache_key]
        print(f"[INFO] Reusing cached Cellpose model ({mtype}, gpu={use_gpu}).")
        return cached

    # prefer v4 API
    try:
        # Cellpose 3.x+ accepts device parameter
        model_kwargs = {'gpu': bool(use_gpu), 'model_type': mtype}

        # Set device for MPS support (Cellpose 3.x+)
        if use_gpu and gpu_device:
            try:
                import torch
                if gpu_device == 'mps':
                    model_kwargs['device'] = torch.device('mps')
                    print("[INFO] CellposeModel using Apple Silicon MPS GPU acceleration.")
                elif gpu_device == 'cuda':
                    model_kwargs['device'] = torch.device('cuda')
                    print("[INFO] CellposeModel using NVIDIA CUDA GPU acceleration.")
            except Exception:
                pass  # Fall back to default device handling

        model = _models.CellposeModel(**model_kwargs)

        if use_gpu and not getattr(model, "gpu", False):
            print("[WARN] CellposeModel requested GPU but fell back to CPU. Check GPU visibility.")
        elif use_gpu and gpu_device:
            pass  # Already printed device info above
        elif use_gpu:
            print("[INFO] CellposeModel running with GPU acceleration.")
        _MODEL_CACHE[cache_key] = model
        return model
    except (AttributeError, TypeError):
        # fallback for older versions (no device parameter)
        model = _models.Cellpose(gpu=bool(use_gpu), model_type=mtype)
        if use_gpu and not getattr(model, "gpu", False):
            print("[WARN] Cellpose GPU fallback is not active; running on CPU.")
        elif use_gpu:
            print("[INFO] Cellpose fallback running with GPU acceleration.")
        _MODEL_CACHE[cache_key] = model
        return model

# -----------------------------
# Image helpers
# -----------------------------
def _read_image(path: Path) -> np.ndarray:
    img = tiff.imread(str(path))
    if img.ndim <= 3:
        squeezed = np.squeeze(img)
        # True Z-stack: first axis has more planes than a channel axis allows
        # (channels are <= 8, see _resolve_channel_selection). Collapse via
        # max projection so downstream channel detection sees (Y, X) or (C, Y, X).
        if squeezed.ndim == 3 and squeezed.shape[0] > 8:
            return squeezed.max(axis=0)
        return squeezed
    squeezed = np.squeeze(img)
    if squeezed.ndim <= 3:
        if squeezed.ndim == 3 and squeezed.shape[0] > 8:
            return squeezed.max(axis=0)
        return squeezed
    while squeezed.ndim > 3:
        squeezed = squeezed[0]
    return squeezed

_CHANNEL_AXIS_CACHE: Dict[Tuple[Tuple[int, ...], int], Tuple[int, int, bool]] = {}
_CHANNEL_WARNED: Set[Tuple[Tuple[int, ...], int]] = set()


def _resolve_channel_selection(shape: Tuple[int, ...], requested_index: int) -> Tuple[int, int, bool]:
    """
    Determine which axis is the channel axis and return a safe index.
    Returns (axis, resolved_index, adjusted) where 'adjusted' is True if the index was corrected
    (e.g. converting 1-based to 0-based).
    """
    channel_axes = [ax for ax, size in enumerate(shape) if size <= 8]
    if not channel_axes:
        channel_axes = [len(shape) - 1]

    for ax in channel_axes:
        size = shape[ax]
        if size <= 0:
            continue
        idx = int(requested_index)
        if idx < 0:
            idx += size
        if 0 <= idx < size:
            return ax, idx, False
        # no valid index on this axis, keep checking other candidates

    raise ValueError(
        f"Channel index {requested_index} out of range for image shape {shape}. "
        "Please verify the detection channel selection and the TIFF channel count."
    )


def _extract_channel(img: np.ndarray, channel_index: int) -> np.ndarray:
    """Return a 2D array selecting the desired channel from multi-channel arrays."""
    if img.ndim == 2:
        return img
    if img.ndim != 3:
        raise ValueError(f"Unsupported image ndim={img.ndim}; expected 2D or 3D.")

    shape = tuple(int(x) for x in img.shape)
    cache_key = (shape, int(channel_index))
    if cache_key not in _CHANNEL_AXIS_CACHE:
        axis, resolved_idx, adjusted = _resolve_channel_selection(shape, int(channel_index))
        _CHANNEL_AXIS_CACHE[cache_key] = (axis, resolved_idx, adjusted)
    else:
        axis, resolved_idx, adjusted = _CHANNEL_AXIS_CACHE[cache_key]

    if adjusted and cache_key not in _CHANNEL_WARNED:
        print(f"[WARN] channel index {channel_index} adjusted to {resolved_idx} for image shape {shape}.")
        _CHANNEL_WARNED.add(cache_key)

    return np.take(img, indices=resolved_idx, axis=axis)

# -----------------------------
# Image preprocessing helpers
# -----------------------------
def _normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    max_val = float(np.nanmax(arr))
    if not np.isfinite(max_val) or max_val <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / max_val


def _global_intensity_normalize(
    img: np.ndarray,
    background_percentile: float = 5.0,
    signal_percentile: float = 99.5,
    debug: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Globale Intensitäts-Normalisierung ohne lokalen Hintergrund.
    
    Diese Methode ist robust bei dichten Zellclustern, da sie:
    1. Globalen Hintergrund aus den dunkelsten Pixeln schätzt (z.B. 5% Percentil)
    2. Signal-Range global normalisiert
    3. Keine lokale Subtraktion macht (die bei dichten Zellen versagt)
    
    Args:
        img: Input-Bild (2D array)
        background_percentile: Percentil für Hintergrund (niedrig, z.B. 5)
        signal_percentile: Percentil für Signal-Maximum (hoch, z.B. 99.5)
        debug: Wenn True, gibt zusätzliche Debug-Info aus
    
    Returns:
        Tuple von:
        - normalized: Global normalisiertes Bild (0-1)
        - stats: Dictionary mit Statistiken für Filterung
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img, {}
    
    flat = img.flatten()
    
    # 1. Globaler Hintergrund aus niedrigstem Percentil
    # Das sind garantiert Pixel ohne Zellen
    global_bg = float(np.percentile(flat, background_percentile))
    
    # 2. Signal-Maximum aus hohem Percentil (ignoriert Ausreißer)
    global_max = float(np.percentile(flat, signal_percentile))
    
    # 3. Median als robuste Mitte
    global_median = float(np.median(flat))
    
    # 4. Signal-Range
    signal_range = global_max - global_bg
    if signal_range <= 0:
        signal_range = 1.0
    
    # 5. Rausch-Schätzung aus niedrigen Intensitäten
    # Nimm Standardabweichung der dunkelsten 30% als Rausch-Schätzung
    low_threshold = float(np.percentile(flat, 30))
    low_pixels = flat[flat <= low_threshold]
    if low_pixels.size > 10:
        global_noise = float(np.std(low_pixels))
    else:
        global_noise = signal_range * 0.01  # Fallback: 1% des Signals
    global_noise = max(global_noise, 1e-6)
    
    # 6. Normalisierung: Hintergrund auf 0, Signal auf 1
    normalized = (img - global_bg) / signal_range
    normalized = np.clip(normalized, 0, 1).astype(np.float32)
    
    # Statistiken für Filterung zurückgeben
    stats = {
        'global_bg': global_bg,
        'global_max': global_max,
        'global_median': global_median,
        'signal_range': signal_range,
        'global_noise': global_noise,
        # Schwellwerte für Zell-Filterung (als Fraktion des Signal-Range)
        'bg_threshold': (global_median - global_bg) / signal_range,  # Unter diesem Wert ist Hintergrund
    }
    
    if debug:
        print(f"[DEBUG] Global BG (p{background_percentile}): {global_bg:.2f}")
        print(f"[DEBUG] Global Max (p{signal_percentile}): {global_max:.2f}")
        print(f"[DEBUG] Signal Range: {signal_range:.2f}")
        print(f"[DEBUG] Global Noise: {global_noise:.4f}")
        print(f"[DEBUG] BG Threshold: {stats['bg_threshold']:.3f}")
    
    return normalized, stats


def _robust_background_normalize(
    img: np.ndarray, 
    background_radius: int = 50,
    noise_radius: int = 25,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DEPRECATED: Lokale Hintergrund-Normalisierung.
    Problem: Bei dichten Zellclustern wird der Hintergrund zu hell geschätzt.
    
    Verwende stattdessen _global_intensity_normalize() für robustere Ergebnisse.
    Diese Funktion bleibt für Rückwärtskompatibilität erhalten.
    """
    from scipy.ndimage import median_filter, uniform_filter
    
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img, np.zeros_like(img), np.zeros_like(img)
    
    # Sicherstellen dass Radien sinnvoll sind
    h, w = img.shape
    background_radius = min(background_radius, min(h, w) // 4)
    background_radius = max(background_radius, 3)
    noise_radius = min(noise_radius, background_radius // 2)
    noise_radius = max(noise_radius, 3)
    
    # 1. Hintergrund-Schätzung mit großem Median-Filter
    # Median ist robust gegen helle Zellen (Ausreißer)
    background = median_filter(img, size=background_radius)
    
    # 2. Lokale Statistiken für Rausch-Schätzung
    # Verwende Fenster kleiner als Hintergrund für feinere Rausch-Details
    local_mean = uniform_filter(img, size=noise_radius)
    local_sqr_mean = uniform_filter(img ** 2, size=noise_radius)
    local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)
    
    # Minimale Standardabweichung um Division durch 0 zu vermeiden
    # Schätze globales Rauschen als Fallback
    global_noise = float(np.percentile(local_std[local_std > 0], 50)) if np.any(local_std > 0) else 1.0
    local_std = np.maximum(local_std, global_noise * 0.1)
    
    # 3. Signal-Extraktion (Hintergrund-Subtraktion)
    signal = np.clip(img - background, 0, None)
    
    # 4. SNR-Map: Wie stark ist Signal über lokalem Rauschen
    snr_map = signal / local_std
    
    # 5. Signal-Normalisierung für Cellpose (0-1 Range)
    # Verwende 99.5-Percentil um Ausreißer zu ignorieren
    signal_p995 = float(np.percentile(signal, 99.5))
    if signal_p995 > 0:
        signal_normalized = np.clip(signal / signal_p995, 0, 1).astype(np.float32)
    else:
        # Fallback: einfache Normalisierung
        signal_max = float(np.max(signal))
        if signal_max > 0:
            signal_normalized = (signal / signal_max).astype(np.float32)
        else:
            signal_normalized = np.zeros_like(signal, dtype=np.float32)
    
    if debug:
        print(f"[DEBUG] Background range: {background.min():.2f} - {background.max():.2f}")
        print(f"[DEBUG] Signal range: {signal.min():.2f} - {signal.max():.2f}")
        print(f"[DEBUG] SNR range: {snr_map.min():.2f} - {snr_map.max():.2f}")
        print(f"[DEBUG] Global noise estimate: {global_noise:.4f}")
    
    return signal_normalized, snr_map, background

def _preprocess_image(img: np.ndarray, proc_cfg: Dict[str, Any]) -> np.ndarray:
    """Apply optional preprocessing (CLAHE, tophat, contrast stretch)."""
    if not isinstance(proc_cfg, dict) or not proc_cfg:
        return img

    processed = img.astype(np.float32, copy=True)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) first
    # This helps with uneven illumination and improves local contrast
    if proc_cfg.get("clahe", False):
        try:
            # Normalize to 0-1 range for CLAHE
            pmin, pmax = processed.min(), processed.max()
            if pmax > pmin:
                normalized = (processed - pmin) / (pmax - pmin)
                # Apply CLAHE with configurable parameters
                clip_limit = float(proc_cfg.get("clahe_clip_limit", 0.03) or 0.03)
                kernel_size = proc_cfg.get("clahe_kernel_size", None)
                processed = exposure.equalize_adapthist(
                    normalized, 
                    clip_limit=clip_limit,
                    kernel_size=kernel_size
                ).astype(np.float32)
                # Scale back to original range
                processed = processed * (pmax - pmin) + pmin
        except Exception as e:
            print(f"[WARN] CLAHE failed: {e}")
    
    if proc_cfg.get("tophat"):
        radius = int(proc_cfg.get("tophat_radius", 15) or 15)
        radius = max(radius, 1)
        selem = morphology.disk(radius)
        try:
            processed = morphology.white_tophat(processed, selem)
        except Exception:
            # fallback with safe structuring element if original radius fails
            processed = morphology.white_tophat(processed, morphology.disk(max(1, min(radius, 64))))

    if proc_cfg.get("contrast_stretch"):
        percs = proc_cfg.get("stretch_percentiles") or [1, 99]
        try:
            if len(percs) != 2:
                percs = [1, 99]
            p0, p1 = sorted(float(x) for x in percs)
            p0 = float(np.clip(p0, 0.0, 100.0))
            p1 = float(np.clip(p1, 0.0, 100.0))
            if p1 <= p0:
                p0, p1 = 1.0, 99.0
            v0, v1 = np.percentile(processed, (p0, p1))
            if np.isfinite(v0) and np.isfinite(v1) and v1 > v0:
                processed = exposure.rescale_intensity(processed, in_range=(v0, v1))
        except Exception:
            pass

    return processed

def _touches_border(bbox: Tuple[int, ...], shape: Tuple[int, ...]) -> bool:
    """Return True if region bounding box touches image border."""
    if len(bbox) < 4 or len(shape) < 2:
        return False
    min_r, min_c, max_r, max_c = bbox[:4]
    return min_r <= 0 or min_c <= 0 or max_r >= shape[0] or max_c >= shape[1]

def _split_touching_cells(mask: np.ndarray, proc_cfg: Dict[str, Any], min_area: int) -> np.ndarray:
    """Split merged masks using watershed if enabled."""
    if not isinstance(proc_cfg, dict) or not proc_cfg.get("split_touching_cells", True):
        return mask

    binary = mask > 0
    if not np.any(binary):
        return mask

    min_area = int(min_area or 0)
    area_factor = float(proc_cfg.get("split_area_multiplier", 2.5))
    split_min_area = int(proc_cfg.get("split_min_area", 0) or 0)
    if split_min_area <= 0 and min_area > 0:
        split_min_area = int(min_area * area_factor)
    min_distance = max(2, int(proc_cfg.get("split_min_distance", 5) or 5))
    rel_peak_threshold = float(proc_cfg.get("split_rel_peak_threshold", 0.2) or 0.2)
    keep_ridge = bool(proc_cfg.get("split_keep_ridge", True))
    min_area_ratio = float(proc_cfg.get("split_min_area_ratio", 0.0) or 0.0)
    if min_area_ratio < 0:
        min_area_ratio = 0.0

    new_mask = np.zeros_like(mask, dtype=np.int32)
    next_label = 1
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        component = mask == lbl
        area = int(component.sum())
        if area == 0:
            continue
        if split_min_area > 0 and area <= split_min_area:
            new_mask[component] = next_label
            next_label += 1
            continue

        distance = ndi.distance_transform_edt(component)
        max_dist = float(distance.max(initial=0.0))
        if max_dist < max(1.5, min_distance / 2):
            new_mask[component] = next_label
            next_label += 1
            continue

        local_maxi = feature.peak_local_max(
            distance,
            labels=component,
            footprint=np.ones((min_distance, min_distance), dtype=bool),
            exclude_border=False,
        )

        if local_maxi.size == 0:
            new_mask[component] = next_label
            next_label += 1
            continue

        peaks = np.asarray(local_maxi, dtype=int)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            peaks = peaks.reshape(-1, 2)
        if peaks.shape[0] == 0:
            new_mask[component] = next_label
            next_label += 1
            continue
        peaks = np.unique(peaks, axis=0)
        peak_vals = distance[peaks[:, 0], peaks[:, 1]]
        if rel_peak_threshold > 0:
            keep = peak_vals >= rel_peak_threshold * max_dist
            peaks = peaks[keep]
            peak_vals = peak_vals[keep]

        if peaks.shape[0] <= 1:
            new_mask[component] = next_label
            next_label += 1
            continue

        if min_area > 0:
            expected_cells = max(1, int(round(area / max(min_area, 1))))
            if peaks.shape[0] > expected_cells:
                order = np.argsort(peak_vals)[-expected_cells:]
                peaks = peaks[order]

        markers = np.zeros_like(distance, dtype=np.int32)
        markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, peaks.shape[0] + 1, dtype=np.int32)
        ws = segmentation.watershed(-distance, markers, mask=component, watershed_line=keep_ridge)
        unique_ws, counts_ws = np.unique(ws, return_counts=True)
        segments = [(lab, cnt) for lab, cnt in zip(unique_ws, counts_ws) if lab > 0]
        if not segments:
            new_mask[component] = next_label
            next_label += 1
            continue

        min_allowed = split_min_area
        if min_area_ratio > 0:
            min_allowed = max(min_allowed, int(area * min_area_ratio))
        if min_allowed > 0:
            if any(cnt < min_allowed for _, cnt in segments):
                new_mask[component] = next_label
                next_label += 1
                continue

        for ws_lbl, _ in segments:
            new_mask[(ws == ws_lbl) & component] = next_label
            next_label += 1

    if next_label == 1:
        return mask
    return new_mask.astype(np.uint16)

def _apply_filters(
    mask: np.ndarray, 
    gray_norm: np.ndarray, 
    cfg: dict, 
    image_shape: Tuple[int, int],
    snr_map: Optional[np.ndarray] = None,
    global_stats: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply geometric and intensity filters to mask labels based on config.
    
    Args:
        mask: Labeled mask from Cellpose
        gray_norm: Normalized grayscale image (0-1)
        cfg: Configuration dictionary
        image_shape: (height, width) of image
        snr_map: Optional SNR map from robust normalization (deprecated, use global_stats)
        global_stats: Optional stats from _global_intensity_normalize for global filtering
    """
    filters_cfg = cfg.get("filters", {}) or {}
    proc_cfg = cfg.get("processing", {}) or {}
    adv_cfg = cfg.get("advanced_filtering", {}) or {}
    enable_adv = bool(cfg.get("enable_advanced_filtering", False))

    min_area = int(filters_cfg.get("min_area") or 0)
    max_area = int(filters_cfg.get("max_area") or 0)
    min_circ = float(filters_cfg.get("min_circularity") or 0.0)
    max_circ = float(filters_cfg.get("max_circularity") or 0.0)
    max_hole = filters_cfg.get("max_hole_ratio")
    max_hole = float(max_hole) if max_hole is not None else None
    # max_size_fraction: drop masks larger than this fraction of the image
    # (Cellpose 4 default is 0.4 — removes merged/oversized ROIs)
    max_size_fraction = float(filters_cfg.get("max_size_fraction") or 0.0)
    image_pixels = float(max(image_shape[0], 1) * max(image_shape[1], 1))
    intensity_factor = float(filters_cfg.get("intensity_threshold_factor") or 1.0)
    intensity_factor = max(intensity_factor, 1e-3)

    intensity_thresh = proc_cfg.get("intensity_threshold")
    if intensity_thresh is None:
        intensity_thresh = 0.0
    intensity_thresh = float(intensity_thresh) / 100.0
    mean_intensity_threshold = min(1.0, intensity_thresh / intensity_factor) if intensity_thresh > 0 else 0.0

    remove_edge = bool(proc_cfg.get("remove_edge_cells", False))
    
    # Intensity mode: "max" = use max_intensity, "mean" = use mean_intensity, "both" = use both, "snr" = use SNR-map
    intensity_mode = str(adv_cfg.get("intensity_mode", "max") or "max").lower()
    # Threshold for max_intensity filter (percentage of global max, e.g. 0.1 = 10%)
    max_intensity_threshold = float(adv_cfg.get("max_intensity_threshold", 0.1) or 0.1)
    # Minimum local contrast: cell max must exceed cell mean by this factor
    min_local_contrast = float(adv_cfg.get("min_local_contrast", 1.2) or 1.2)
    # SNR threshold for snr mode (minimum mean SNR within cell)
    snr_threshold = float(adv_cfg.get("snr_threshold", 2.0) or 2.0)
    # Use robust normalization with SNR-map
    use_robust_norm = intensity_mode == "snr" or bool(adv_cfg.get("use_robust_normalization", False))

    flat = gray_norm[np.isfinite(gray_norm)]
    if flat.size == 0:
        return mask.astype(np.uint16), {"initial": 0, "kept": 0, "removed": 0}

    bg_percentile = float(adv_cfg.get("abs_floor_percentile", 85.0) or 85.0)
    bg_percentile = float(np.clip(bg_percentile, 0.0, 100.0))
    bg_floor = float(np.percentile(flat, bg_percentile))
    bg_vals = flat[flat <= bg_floor]
    if bg_vals.size == 0:
        bg_vals = flat
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals)) if bg_vals.size else float(np.std(flat))
    if not np.isfinite(bg_std) or bg_std < 1e-6:
        bg_std = 1e-6
    
    # Compute global intensity stats for threshold calculation
    # Use median as reference point instead of min for more robust thresholding
    global_max = float(np.percentile(flat, 99.5))
    global_median = float(np.median(flat))
    # Dynamic range above median (where real signals should be)
    signal_range = global_max - global_median if global_max > global_median else 1.0

    snr_min = float(adv_cfg.get("snr_min", 0.0) or 0.0)

    props = measure.regionprops(mask.astype(np.int32), intensity_image=gray_norm.astype(np.float32, copy=False))
    keep_labels: List[int] = []
    stats = {
        "initial": len(props),
        "removed": 0,
        "removed_area": 0,
        "kept": 0,
        "intensity_mode": intensity_mode,
    }
    
    # Track filter reasons for debugging
    filter_reasons: Dict[str, int] = {
        "min_area": 0, "max_area": 0, "max_size_fraction": 0, "circularity": 0, "hole_ratio": 0,
        "edge": 0, "intensity": 0, "intensity_max": 0, "intensity_mean": 0, 
        "low_contrast": 0, "bg_floor": 0, "snr": 0
    }

    for region in props:
        keep = True
        filter_reason = None
        area = int(region.area)
        
        if min_area and area < min_area:
            keep = False
            filter_reason = "min_area"
        if keep and max_area and area > max_area:
            keep = False
            filter_reason = "max_area"
        if keep and max_size_fraction > 0.0 and image_pixels > 0 and area > image_pixels * max_size_fraction:
            keep = False
            filter_reason = "max_size_fraction"

        perimeter = float(region.perimeter or 0.0)
        # Skip circularity check if disabled (min_circ=0 and max_circ=1 means disabled)
        # Also skip for small cells (<100 pixels) - perimeter is too inaccurate at low magnifications
        circ_enabled = (min_circ > 0.0) or (max_circ > 0.0 and max_circ < 1.0)
        if keep and perimeter > 0.0 and circ_enabled and area >= 100:
            circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
            if min_circ > 0.0 and circularity < min_circ:
                keep = False
                filter_reason = "circularity"
            if max_circ > 0.0 and max_circ < 1.0 and circularity > max_circ:
                keep = False
                filter_reason = "circularity"

        filled_area = int(getattr(region, "filled_area", area))
        hole_ratio = 0.0
        if filled_area > 0:
            hole_ratio = float(filled_area - area) / float(filled_area)
        if keep and max_hole is not None and hole_ratio > max_hole:
            keep = False
            filter_reason = "hole_ratio"

        if keep and remove_edge and _touches_border(region.bbox, image_shape):
            keep = False
            filter_reason = "edge"

        mean_intensity = float(region.mean_intensity or 0.0)
        max_intensity = float(region.max_intensity or 0.0) if hasattr(region, 'max_intensity') else mean_intensity
        
        # Calculate relative max intensity based on signal range (above median)
        # This is more robust than using global_min which can be very low
        rel_max_intensity = (max_intensity - global_median) / signal_range if signal_range > 0 else 0.0
        rel_max_intensity = max(0.0, rel_max_intensity)  # Clamp to 0 if below median
        
        # Local contrast check: max intensity should be significantly higher than mean
        # This filters out flat/uniform regions that are not real cells
        local_contrast = max_intensity / mean_intensity if mean_intensity > 1e-6 else 0.0
        
        # Apply intensity filter based on mode
        if keep and intensity_mode == "global":
            # Global intensity filtering: Uses global stats, robust for dense cell clusters
            # This mode doesn't use local background which fails with many bright cells
            if global_stats:
                # Use global background threshold
                global_bg_thr = global_stats.get('bg_threshold', 0.1)
                global_noise = global_stats.get('global_noise', 0.01)
                signal_range_gs = global_stats.get('signal_range', 1.0)
                
                # Cell must be significantly above background
                # max_intensity is already in 0-1 range (normalized image)
                if max_intensity < global_bg_thr + max_intensity_threshold:
                    keep = False
                    filter_reason = "intensity_max"
                # Local contrast check only if explicitly enabled (>1.0)
                # Disabled by default as it rejects uniformly bright cells
                elif min_local_contrast > 1.05 and local_contrast < min_local_contrast:
                    keep = False
                    filter_reason = "low_contrast"
            else:
                # Fallback to simple relative threshold if no global_stats
                if rel_max_intensity < max_intensity_threshold:
                    keep = False
                    filter_reason = "intensity_max"
        elif keep and intensity_mode == "snr":
            # SNR-based filtering: uses the SNR map from robust normalization
            # More robust against variable background than intensity-based methods
            if snr_map is not None:
                # Get SNR values within this cell
                cell_mask_region = mask == region.label
                cell_snr_values = snr_map[cell_mask_region]
                if cell_snr_values.size > 0:
                    cell_snr_mean = float(np.mean(cell_snr_values))
                    cell_snr_max = float(np.max(cell_snr_values))
                    if cell_snr_max < snr_threshold:
                        keep = False
                        filter_reason = "snr"
                    elif cell_snr_mean < snr_threshold * 0.5:
                        keep = False
                        filter_reason = "snr"
            else:
                # Fallback if no SNR map: use relative intensity
                if rel_max_intensity < max_intensity_threshold:
                    keep = False
                    filter_reason = "intensity_max"
        elif keep and intensity_mode == "max":
            # Use max_intensity for filtering - best for fluorescent signals
            # Cell must have max intensity above threshold
            if rel_max_intensity < max_intensity_threshold:
                keep = False
                filter_reason = "intensity_max"
            # Local contrast check only if explicitly enabled (>1.05)
            elif min_local_contrast > 1.05 and local_contrast < min_local_contrast:
                keep = False
                filter_reason = "low_contrast"
        elif keep and intensity_mode == "mean":
            # Use only mean_intensity (legacy behavior)
            if mean_intensity_threshold > 0.0 and mean_intensity < mean_intensity_threshold:
                keep = False
                filter_reason = "intensity_mean"
        elif keep and intensity_mode == "both":
            # Cell must pass both thresholds
            if rel_max_intensity < max_intensity_threshold or (mean_intensity_threshold > 0.0 and mean_intensity < mean_intensity_threshold):
                keep = False
                filter_reason = "intensity"

        # Advanced filtering (SNR and background floor) - now uses max_intensity when mode is "max"
        if keep and enable_adv:
            check_intensity = max_intensity if intensity_mode in ("max", "snr") else mean_intensity
            if check_intensity <= bg_floor:
                keep = False
                filter_reason = "bg_floor"
            else:
                snr = (check_intensity - bg_mean) / (bg_std + 1e-6)
                if snr < snr_min:
                    keep = False
                    filter_reason = "snr"

        if keep:
            keep_labels.append(region.label)
        else:
            stats["removed"] += 1
            stats["removed_area"] += area
            if filter_reason:
                filter_reasons[filter_reason] = filter_reasons.get(filter_reason, 0) + 1

    if not keep_labels:
        stats["filter_reasons"] = filter_reasons
        return np.zeros_like(mask, dtype=np.uint16), stats

    filtered = np.zeros_like(mask, dtype=np.uint16)
    for new_idx, lbl in enumerate(keep_labels, start=1):
        filtered[mask == lbl] = new_idx
    stats["kept"] = len(keep_labels)
    stats["filter_reasons"] = filter_reasons
    return filtered, stats

# -----------------------------
# Region + file discovery
# -----------------------------
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

TOKEN_SPLIT_RE = re.compile(r'[_\s\-]+')

def _split_tokens_for_sample(text: str) -> List[str]:
    return [tok for tok in TOKEN_SPLIT_RE.split(text) if tok]

def _extract_sample_tokens_from_rel_path(rel_path: Path) -> tuple[str | None, str | None]:
    tokens: List[str] = []
    for part in rel_path.parts:
        tokens.extend(_split_tokens_for_sample(part))
    tokens_lower = [tok.lower() for tok in tokens if tok]
    sample_token = None
    for tok in tokens_lower:
        if any(ch.isdigit() for ch in tok) and len(tok) >= 3:
            sample_token = tok
            break
    if sample_token is None:
        for tok in tokens_lower:
            if any(ch.isdigit() for ch in tok):
                sample_token = tok
                break
    slice_token = None
    for tok in tokens_lower:
        if 'slice' in tok:
            slice_token = tok
            break
    return sample_token, slice_token

def _find_related_sample_paths(all_paths: List[Path], root: Path, reference: Path) -> List[Path]:
    if not isinstance(reference, Path):
        try:
            reference = Path(reference)
        except Exception:
            return []
    if reference.is_absolute():
        try:
            rel_reference = reference.relative_to(root)
        except Exception:
            rel_reference = reference
    else:
        rel_reference = reference
    sample_token, slice_token = _extract_sample_tokens_from_rel_path(rel_reference)
    if not sample_token and not slice_token:
        return []
    matched_by_region: Dict[str, List[Path]] = {}
    for candidate in all_paths:
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            continue
        tokens: List[str] = []
        for part in rel.parts:
            tokens.extend(_split_tokens_for_sample(part))
        tokens_lower = [tok.lower() for tok in tokens if tok]
        if sample_token and sample_token not in tokens_lower:
            continue
        if slice_token and slice_token not in tokens_lower:
            continue
        region = rel.parts[0] if len(rel.parts) > 1 else '__root__'
        matched_by_region.setdefault(region, []).append(candidate)
    if not matched_by_region:
        return []
    selected: List[Path] = []
    for region_name in sorted(matched_by_region.keys()):
        region_candidates = sorted(matched_by_region[region_name], key=lambda p: p.as_posix())
        if region_candidates:
            selected.append(region_candidates[0])
    return selected

def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def _region_from_path(p: Path, region_tokens: List[str]) -> str:
    if not region_tokens:
        name = p.name.strip()
        if name:
            return name
        parent = p.parent.name.strip()
        return parent or "ALL"
    # simple token search in parts
    parts = [s.lower() for s in p.parts]
    for tok in region_tokens:
        t = tok.lower()
        if any(t in part for part in parts):
            return tok
    return "ALL"

# -----------------------------
# Magnification detection from path
# -----------------------------
_MAG_PATTERN = re.compile(r"(\d+)\s*[xX]")

def _detect_magnification_from_path(p: Path, default_reference: float = 10.0) -> float:
    """Detect objective magnification from file path tokens like '5X', '10x', '20X'.
    Falls back to default_reference when nothing is detected.
    Only common values {5, 10, 20, 40} are recognized.
    """
    try:
        text = p.as_posix()
    except Exception:
        text = str(p)
    mags = _MAG_PATTERN.findall(text)
    for m in mags:
        try:
            val = int(m)
        except Exception:
            continue
        if val in (5, 10, 20, 40):
            return float(val)
    return float(default_reference or 10.0)

# -----------------------------
# Overlays
# -----------------------------
def _save_overlay(base_img: np.ndarray, mask: np.ndarray, out_png: Path, color: str = "lime",
                  line_width: int = 1, dpi: int = 300, figsize: Tuple[int,int] = (12,12),
                  mode: str = "contours") -> None:
    import matplotlib.pyplot as plt
    from skimage import measure, color as skcolor

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    # grayscale background
    if base_img.ndim == 2:
        ax.imshow(base_img, cmap="gray", interpolation="nearest")
    else:
        # if someone passes 3D, try to show first channel
        show = base_img[..., 0] if base_img.ndim == 3 else base_img
        ax.imshow(show, cmap="gray", interpolation="nearest")

    if mode in ("masks", "both"):
        ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.3, interpolation="nearest")

    if mode in ("contours", "both"):
        unique_labels = np.unique(mask)
        for lbl in unique_labels:
            if lbl == 0:
                continue
            contours = measure.find_contours(mask == lbl, 0.5)
            for c in contours:
                ax.plot(c[:, 1], c[:, 0], color=color, linewidth=line_width)

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# -----------------------------
# Main segmentation routine
# -----------------------------
def _segment_dir(
    root: Path,
    out_root: Path,
    cfg: dict,
    condition: str,
    regions_enabled: List[str],
    testing_cfg: Optional[Dict[str, Any]],
) -> None:
    cp_base = cfg.get("cellpose", {}) or {}
    ov_base = cfg.get("overlays", {}) or {}
    outp = cfg.get("outputs", {}) or {}

    model_name = cp_base.get("model_name", "cyto2")
    use_gpu = bool(cp_base.get("use_gpu", False))
    ch_index = int(cp_base.get("channel", 0))

    overlay_suffix = ov_base.get("overlay_suffix") or cfg.get("outputs", {}).get("overlay_suffix", "_overlay.png")

    resume = bool(outp.get("resume", True))
    overwrite = bool(outp.get("overwrite", False))

    model = _load_cellpose_model(model_name, use_gpu=use_gpu)
    cp_channels = [0, 0]

    summary_rows: List[Dict[str, Any]] = []

    paths = list(_iter_images(root))
    if not paths:
        print(f"[WARN] No input images found under {root}")
        return

    testing_cfg = testing_cfg or {}
    mode_flag = str((testing_cfg.get("mode") or "").lower())
    selection_map = testing_cfg.get("selected_images") if isinstance(testing_cfg.get("selected_images"), dict) else {}
    selected_image_raw = ""
    if isinstance(selection_map, dict):
        selected_image_raw = selection_map.get(condition)
        if not selected_image_raw:
            selected_image_raw = selection_map.get(condition.lower()) or selection_map.get(condition.upper()) or ""
    selected_image_raw = str(selected_image_raw or testing_cfg.get("selected_image") or "").strip()
    test_enabled = bool(testing_cfg.get("enabled"))
    test_limit = int(testing_cfg.get("samples_per_channel") or 0)
    test_seed = testing_cfg.get("seed")
    if isinstance(test_seed, str) and not test_seed:
        test_seed = None
    total_paths = len(paths)
    selected_path = None
    if selected_image_raw:
        normalized_value = selected_image_raw.replace("\\", "/").strip()
        candidate_path = Path(normalized_value)
        candidate_paths: List[Path] = []
        seen_candidates: Set[str] = set()

        def _add_candidate_path(path_obj: Path):
            if not isinstance(path_obj, Path):
                return
            try:
                resolved = path_obj.resolve()
            except Exception:
                resolved = path_obj
            key = resolved.as_posix()
            if key not in seen_candidates:
                seen_candidates.add(key)
                candidate_paths.append(resolved)

        if candidate_path.is_absolute():
            _add_candidate_path(candidate_path)
        else:
            _add_candidate_path(root / candidate_path)

        parts = [part for part in candidate_path.parts if part not in ('.',)]
        trimmed_parts = list(parts)
        while len(trimmed_parts) > 1:
            trimmed_parts = trimmed_parts[1:]
            _add_candidate_path(root / Path(*trimmed_parts))

        try:
            _add_candidate_path(candidate_path.resolve())
        except Exception:
            pass

        for cand in candidate_paths:
            try:
                cand_resolved = Path(cand).resolve()
            except Exception:
                cand_resolved = cand
            for p in paths:
                try:
                    if os.path.samefile(p, cand_resolved):
                        selected_path = p
                        break
                except Exception:
                    try:
                        if Path(p).resolve() == cand_resolved:
                            selected_path = p
                            break
                    except Exception:
                        continue
            if selected_path:
                break
        if selected_path:
            print(f"[TEST] {condition}: using selected image -> {selected_path}")
            paths = [selected_path]
            test_enabled = True
            test_limit = 1
        else:
            print(f"[WARN] {condition}: selected test image not found -> {selected_image_raw}")

    if test_enabled and test_limit > 0:
        region_buckets: Dict[str, List[Path]] = {}
        for img_path in paths:
            rel_tmp = img_path.relative_to(root)
            region_tmp = _region_from_path(img_path.parent, regions_enabled or [])
            if (not region_tmp or region_tmp.upper() == "ALL") and rel_tmp.parts:
                region_tmp = rel_tmp.parts[0]
            region_buckets.setdefault(region_tmp, []).append(img_path)

        sampled_paths: List[Path] = []
        for region_name, region_paths in region_buckets.items():
            region_paths_sorted = sorted(region_paths, key=lambda p: p.as_posix())
            if len(region_paths_sorted) > test_limit:
                if mode_flag == 'advanced' and test_seed is None:
                    sample = region_paths_sorted[:test_limit]
                else:
                    rng = random.Random()
                    if test_seed is not None:
                        try:
                            rng.seed(f"{test_seed}:{condition}:{region_name}")
                        except Exception:
                            rng.seed()
                    else:
                        rng.seed()
                    sample = rng.sample(region_paths_sorted, test_limit)
                print(
                    f"[TEST] {condition}/{region_name}: limiting to {len(sample)} of {len(region_paths_sorted)} images "
                    f"(samples_per_channel={test_limit})"
                )
            else:
                sample = region_paths_sorted
                print(
                    f"[TEST] {condition}/{region_name}: {len(sample)} image(s) = limit "
                    f"(samples_per_channel={test_limit})"
                )
            sampled_paths.extend(sample)

        if sampled_paths:
            paths = sorted(sampled_paths, key=lambda p: p.as_posix())
        else:
            print(f"[TEST] {condition}: no images selected (check configuration).")
    elif test_enabled and total_paths > 0:
        print(f"[TEST] {condition}: test mode enabled but samples_per_channel={test_limit}, skipping limit.")

    total_to_process = len(paths)
    print(f"\\n==> Condition: {condition} | root={root}")
    for img_path in tqdm(paths, desc=f"{condition}/ALL"):
        try:
            region = _region_from_path(img_path.parent, regions_enabled or [])
            rel = img_path.relative_to(root)
            if (not region or region.upper() == "ALL") and rel.parts:
                region = rel.parts[0]
            parent_parts = list(rel.parts[:-1])
            if parent_parts and parent_parts[0] == region:
                parent_parts = parent_parts[1:]
            out_dir = out_root / condition / region
            for part in parent_parts:
                out_dir /= part
            out_dir.mkdir(parents=True, exist_ok=True)

            mask_tif = out_dir / f"{img_path.stem}_mask.tif"
            overlay_png = out_dir / f"{img_path.stem}{overlay_suffix}"

            masks_arr = None
            cached_mask = False
            if mask_tif.exists() and resume and not overwrite:
                try:
                    masks_arr = np.squeeze(tiff.imread(str(mask_tif)))
                    cached_mask = True
                except Exception:
                    masks_arr = None
                    cached_mask = False

            # Detect per-image magnification and build a scaled config for this image
            scope_cfg = cfg.get("microscope", {}) or {}
            reference_mag = float(scope_cfg.get("reference_magnification", 10.0) or 10.0)
            auto_from_path = bool(scope_cfg.get("auto_from_path", True))
            if auto_from_path:
                current_mag = _detect_magnification_from_path(img_path, default_reference=reference_mag)
            else:
                current_mag = float(scope_cfg.get("current_magnification", reference_mag) or reference_mag)
            cfg_img = copy.deepcopy(cfg)
            img_scope = dict(cfg_img.get("microscope", {}) or {})
            img_scope["reference_magnification"] = reference_mag
            img_scope["current_magnification"] = float(current_mag)
            cfg_img["microscope"] = img_scope
            
            # Read image first to get dimensions for image-aware scaling
            img = _read_image(img_path)
            gray_raw = _extract_channel(img, ch_index).astype(np.float32)
            img_h, img_w = gray_raw.shape
            
            # Get resize_max from config
            cp_base_cfg = cfg_img.get("cellpose", {}) or {}
            resize_max = int(cp_base_cfg.get("resize_max", 2048) or 2048)
            
            # Check if image-aware scaling is enabled (default: True)
            use_image_aware_scaling = img_scope.get("image_aware_scaling", True)
            if isinstance(use_image_aware_scaling, str):
                use_image_aware_scaling = use_image_aware_scaling.lower() in ("true", "1", "yes")
            
            # Apply image-aware scaling if enabled (considers both image size and magnification)
            if use_image_aware_scaling:
                cfg_scaled, scale_info = apply_image_aware_scaling(
                    cfg_img, 
                    image_shape=(img_h, img_w),
                    resize_max=resize_max
                )
            else:
                # Fallback: just use magnification scaling without image-size awareness
                cfg_scaled = copy.deepcopy(cfg_img)
                scale_info = {
                    "effective_diameter": cfg_scaled.get("cellpose", {}).get("diameter", 30),
                    "resize_factor": 1.0,
                    "combined_scale": 1.0,
                    "quality_warning": None
                }

            cp_img = cfg_scaled.get("cellpose", {}) or {}
            proc_img = cfg_scaled.get("processing", {}) or {}
            ov_img = cfg_scaled.get("overlays", {}) or {}
            filters_img = cfg_scaled.get("filters", {}) or {}

            # Log scaling information
            effective_diameter = scale_info["effective_diameter"]
            resize_factor = scale_info["resize_factor"]
            combined_scale = scale_info["combined_scale"]
            quality_warning = scale_info.get("quality_warning")
            
            if abs(resize_factor - 1.0) > 0.01 or abs(current_mag - reference_mag) > 0.1:
                src = "path" if auto_from_path else "manual"
                print(f"[INFO] {img_path.name}: {img_w}x{img_h}px, {current_mag}x (ref {reference_mag}x, {src})")
                print(f"       -> resize_factor={resize_factor:.3f}, effective_diameter={effective_diameter:.1f}px")
                if quality_warning:
                    print(f"[WARN] {img_path.name}: {quality_warning}")
            
            # Debug: log scaled parameters
            if abs(combined_scale - 1.0) > 0.01:
                print(f"[DEBUG] Scaled params: diameter={cp_img.get('diameter'):.1f} min_area={filters_img.get('min_area')} "
                      f"max_area={filters_img.get('max_area')} fg_block={cfg_scaled.get('advanced_filtering',{}).get('foreground_block_size')} "
                      f"tophat_radius={proc_img.get('tophat_radius')}")

            # Prepare per-image parameters
            diameter = effective_diameter
            if diameter == 0:
                diameter = None  # Let Cellpose estimate
            flow_thr = float(cp_img.get("flow_threshold", 0.4))
            cell_thr = float(cp_img.get("cellprob_threshold", 0.0))

            mode = ov_img.get("overlay_mode", "contours")
            color = ov_img.get("contour_color", "lime")
            line_w = int(ov_img.get("line_width", 1))
            dpi = int(ov_img.get("dpi", 300))
            figsize = tuple(ov_img.get("figsize", [12, 12]))

            min_area_for_split = int(filters_img.get("min_area") or 0)

            # Intelligent rescaling for optimal cell detection
            # Now handled by apply_image_aware_scaling - only manual resize if diameter < 3
            max_dimension = max(img_h, img_w)
            target_diameter_min = 3.0
            
            if diameter and diameter < target_diameter_min:
                # Effective diameter is too small after scaling - upscale image
                max_safe_dimension = resize_max  # Use resize_max as the limit
                required_scale = target_diameter_min / diameter
                target_dimension = int(max_dimension * required_scale)
                if target_dimension <= max_safe_dimension:
                    new_h = int(img_h * required_scale)
                    new_w = int(img_w * required_scale)
                    print(f"[INFO] {img_path.name}: upscaling {img_w}x{img_h} -> {new_w}x{new_h} for minimum diameter {target_diameter_min:.1f}")
                    from skimage import transform
                    gray_raw = transform.resize(gray_raw, (new_h, new_w), preserve_range=True, anti_aliasing=True).astype(np.float32)
                    diameter = target_diameter_min
                else:
                    print(f"[WARN] {img_path.name}: diameter {diameter:.1f}px is below optimal but cannot upscale further")
            
            gray_proc = _preprocess_image(gray_raw, proc_img)

            # Check normalization mode
            adv_cfg_img = cfg_scaled.get("advanced_filtering", {}) or {}
            intensity_mode = str(adv_cfg_img.get("intensity_mode", "global") or "global").lower()
            use_robust_norm = intensity_mode == "snr" or bool(adv_cfg_img.get("use_robust_normalization", False))
            use_global_norm = intensity_mode == "global"

            snr_map = None
            background_img = None
            global_stats = None
            
            if use_global_norm:
                # Use global intensity normalization - best for dense cell clusters
                bg_pct = float(adv_cfg_img.get("background_percentile", 5.0) or 5.0)
                sig_pct = float(adv_cfg_img.get("signal_percentile", 99.5) or 99.5)
                gray_norm, global_stats = _global_intensity_normalize(
                    gray_proc,
                    background_percentile=bg_pct,
                    signal_percentile=sig_pct,
                    debug=False
                )
                print(f"[INFO] {img_path.name}: using global normalization (bg_pct={bg_pct}, bg_thr={global_stats.get('bg_threshold', 0):.3f})")
            elif use_robust_norm:
                # Use robust background normalization (local SNR) - may fail with dense cells
                background_radius = int(adv_cfg_img.get("background_radius", 50) or 50)
                noise_radius = int(adv_cfg_img.get("noise_radius", 25) or 25)
                gray_norm, snr_map, background_img = _robust_background_normalize(
                    gray_proc, 
                    background_radius=background_radius,
                    noise_radius=noise_radius,
                    debug=False
                )
                print(f"[INFO] {img_path.name}: using robust normalization (bg_radius={background_radius}, SNR range={snr_map.min():.1f}-{snr_map.max():.1f})")
            else:
                # Standard normalization
                gray_norm = _normalize_image(gray_proc)

            if masks_arr is None:
                # batch_size from config (default 8 per Cellpose docs), applied per-image here
                batch_size = int(cp_img.get("batch_size", 8) or 8)
                masks_arr, _, _ = model.eval(
                    gray_norm,
                    channels=cp_channels,
                    diameter=diameter,
                    flow_threshold=flow_thr,
                    cellprob_threshold=cell_thr,
                    batch_size=batch_size,
                )
                masks_arr = masks_arr.astype(np.uint16, copy=False)

            masks_arr = masks_arr.astype(np.uint16, copy=False)
            needs_processing = overwrite or not cached_mask

            if masks_arr.size and needs_processing:
                orig_count = int(np.max(masks_arr))
                split_mask = _split_touching_cells(masks_arr, proc_img, min_area_for_split)
                if split_mask is not None and split_mask.shape == masks_arr.shape:
                    new_count = int(np.max(split_mask))
                    if new_count > orig_count:
                        print(f"[INFO] {img_path.name}: split touching cells {orig_count}->{new_count}")
                    masks_arr = split_mask

                # Pass global_stats or snr_map to filter function
                filtered_mask, filter_stats = _apply_filters(
                    masks_arr, gray_norm, cfg_scaled, gray_norm.shape, 
                    snr_map=snr_map, global_stats=global_stats
                )
                if filter_stats.get("removed"):
                    reasons = filter_stats.get("filter_reasons", {})
                    reasons_str = ", ".join(f"{k}={v}" for k, v in reasons.items() if v > 0)
                    mode_info = filter_stats.get("intensity_mode", "global")
                    msg = f"[INFO] {img_path.name}: filtered {filter_stats['removed']} cells (kept {filter_stats.get('kept', 0)}, mode={mode_info})"
                    if reasons_str:
                        msg += f" [{reasons_str}]"
                    print(msg)
                masks_arr = filtered_mask

            if needs_processing or not mask_tif.exists():
                tiff.imwrite(str(mask_tif), masks_arr, photometric="minisblack")

            if overwrite or not overlay_png.exists() or not cached_mask:
                try:
                    _save_overlay(gray_norm, masks_arr, overlay_png, color=color, line_width=line_w, dpi=dpi, figsize=figsize, mode=mode)
                except Exception as e:
                    print(f"[WARN] overlay failed for {img_path.name}: {e}")
            
            # Save debug heatmaps if enabled (SNR map and background)
            save_debug_images = bool(adv_cfg_img.get("save_debug_images", False))
            if save_debug_images and snr_map is not None:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Save SNR heatmap
                    snr_png = out_dir / f"{img_path.stem}_snr_heatmap.png"
                    fig, ax = plt.subplots(figsize=figsize, dpi=dpi//2)
                    snr_clipped = np.clip(snr_map, 0, 10)  # Clip for better visualization
                    im = ax.imshow(snr_clipped, cmap='hot', interpolation='nearest')
                    ax.set_axis_off()
                    plt.colorbar(im, ax=ax, label='SNR')
                    ax.set_title(f'SNR Map - {img_path.name}')
                    fig.savefig(snr_png, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
                    
                    # Save background estimate
                    if background_img is not None:
                        bg_png = out_dir / f"{img_path.stem}_background.png"
                        fig, ax = plt.subplots(figsize=figsize, dpi=dpi//2)
                        ax.imshow(background_img, cmap='gray', interpolation='nearest')
                        ax.set_axis_off()
                        ax.set_title(f'Estimated Background - {img_path.name}')
                        fig.savefig(bg_png, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig)
                    
                    print(f"[DEBUG] Saved SNR heatmap: {snr_png.name}")
                except Exception as e:
                    print(f"[WARN] Could not save debug images: {e}")

            labeled = measure.label(masks_arr > 0)
            regions = measure.regionprops(labeled, intensity_image=gray_raw)
            cell_count = len(regions)
            
            # Quality metrics
            area_std = 0.0
            min_cell_area = 0.0
            max_cell_area = 0.0
            diameter_cv = 0.0  # Coefficient of variation for cell sizes
            quality_flag = "OK"
            
            if cell_count > 0:
                areas = np.array([r.area for r in regions], dtype=np.float64)
                mean_ints = np.array([r.mean_intensity for r in regions], dtype=np.float64)
                integ = areas * mean_ints
                mean_area = float(areas.mean())
                area_std = float(areas.std())
                min_cell_area = float(areas.min())
                max_cell_area = float(areas.max())
                mean_intensity = float(mean_ints.mean())
                mean_integrated = float(integ.mean())
                
                # Calculate coefficient of variation (CV) as quality indicator
                # CV > 1.0 suggests highly variable cell sizes (potential detection issues)
                if mean_area > 0:
                    diameter_cv = area_std / mean_area
                
                # Quality warnings based on expected cell areas
                expected_min = int(filters_img.get("min_area", 50) or 50)
                expected_max = int(filters_img.get("max_area", 5000) or 5000)
                
                # Check for potential issues
                if mean_area < expected_min * 0.5:
                    quality_flag = "WARN:too_small"
                    print(f"[QC] {img_path.name}: mean cell area {mean_area:.0f}px² is below expected min ({expected_min}px²)")
                elif mean_area > expected_max * 0.8:
                    quality_flag = "WARN:too_large"
                    print(f"[QC] {img_path.name}: mean cell area {mean_area:.0f}px² approaches max ({expected_max}px²)")
                elif diameter_cv > 1.5:
                    quality_flag = "WARN:high_variance"
                    print(f"[QC] {img_path.name}: high size variance (CV={diameter_cv:.2f})")
                elif cell_count < 3:
                    quality_flag = "WARN:few_cells"
            else:
                mean_area = mean_intensity = mean_integrated = 0.0

            summary_rows.append({
                "filename": str(rel).replace("\\", "/"),
                "condition": condition,
                "region": region,
                "channel": f"ch{ch_index}",
                "cell_count": int(cell_count),
                "mean_area_per_cell": mean_area,
                "area_std": area_std,
                "min_cell_area": min_cell_area,
                "max_cell_area": max_cell_area,
                "mean_intensity_per_cell": mean_intensity,
                "mean_integrated_density_per_cell": mean_integrated,
                "image_width": img_w,
                "image_height": img_h,
                "effective_diameter": effective_diameter,
                "resize_factor": resize_factor,
                "magnification": current_mag,
                "quality_flag": quality_flag,
            })

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    master_csv = out_root / condition / "All_Counts_Master.csv"
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    with master_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "condition",
            "region",
            "channel",
            "cell_count",
            "mean_area_per_cell",
            "area_std",
            "min_cell_area",
            "max_cell_area",
            "mean_intensity_per_cell",
            "mean_integrated_density_per_cell",
            "image_width",
            "image_height",
            "effective_diameter",
            "resize_factor",
            "magnification",
            "quality_flag",
        ])
        for row in summary_rows:
            w.writerow([
                row["filename"],
                row["condition"],
                row["region"],
                row["channel"],
                row["cell_count"],
                row["mean_area_per_cell"],
                row.get("area_std", 0),
                row.get("min_cell_area", 0),
                row.get("max_cell_area", 0),
                row["mean_intensity_per_cell"],
                row["mean_integrated_density_per_cell"],
                row.get("image_width", 0),
                row.get("image_height", 0),
                row.get("effective_diameter", 0),
                row.get("resize_factor", 1.0),
                row.get("magnification", 10.0),
                row.get("quality_flag", "OK"),
            ])

    print(f"   Condition={condition}: processed {total_to_process} images -> wrote {master_csv.name}")
# -----------------------------
# CLI
# -----------------------------
def segment_dirs(config_path: Path) -> None:
    # Load base (unscaled) config; scaling will be applied per-image based on path tokens (e.g., 5X/10X/20X/40X)
    cfg = _load_yaml(config_path)
    paths = cfg.get("paths", {}) or {}
    testing_cfg = cfg.get("testing", {}) or {}

    input_pos = paths.get("input_pos", "")
    input_neg = paths.get("input_neg", "")
    output_root = paths.get("output_root", "./results_det")

    regions_enabled = cfg.get("regions_enabled") or []

    if not input_pos and not input_neg:
        print("[ERROR] No inputs configured (paths.input_pos / paths.input_neg). Nothing to do.")
        return

    out_root = Path(output_root).absolute()
    out_root.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    scope_cfg = cfg.get("microscope", {}) or {}
    runtime_params = {
        "testing_config": testing_cfg,
        "regions_enabled": regions_enabled,
        "magnification_mode": "per-image-from-path",
        "reference_magnification": scope_cfg.get("reference_magnification", 10.0),
    }
    save_config_snapshot(
        output_dir=out_root,
        config_file=config_path,
        config_dict=cfg,
        runtime_params=runtime_params,
        snapshot_name="segmentation_config",
    )

    if input_pos:
        root = Path(input_pos).absolute()
        _segment_dir(root, out_root, cfg, condition="pos", regions_enabled=regions_enabled, testing_cfg=testing_cfg)
    if input_neg:
        root = Path(input_neg).absolute()
        _segment_dir(root, out_root, cfg, condition="neg", regions_enabled=regions_enabled, testing_cfg=testing_cfg)

    print("\n[DONE] Segmentation finished.")

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Batch segmentation with Cellpose v4 (multi-channel)")
    ap.add_argument("--config", required=True, help="Path to config_det_cpsam.yaml")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 2
    try:
        segment_dirs(cfg_path)
    except Exception as e:
        print(f"[FATAL] {e}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())




