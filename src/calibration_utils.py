#!/usr/bin/env python3
"""
calibration_utils.py — Count-based sensitivity level calibration

This module provides functions to automatically find the optimal detection
sensitivity level by comparing detected cell counts against user-provided
ground truth counts.

Workflow:
1. User selects 3-4 representative images from the input folder
2. User provides expected cell counts for each image
3. System runs detection at all 15 sensitivity levels (sequentially)
4. System calculates Mean Absolute Error (MAE) for each level
5. System recommends the level with lowest MAE
6. User can apply the recommended level to the Detection tab

Usage:
    from src.calibration_utils import run_calibration, find_optimal_level

    images_with_counts = [
        {"path": "/data/Input/sample1.tif", "expected": 45},
        {"path": "/data/Input/sample2.tif", "expected": 62},
        {"path": "/data/Input/sample3.tif", "expected": 38},
        {"path": "/data/Input/sample4.tif", "expected": 51},
    ]
    
    results = run_calibration(images_with_counts, sensitivity_levels, callback=progress_callback)
    optimal = find_optimal_level(results)
    print(f"Recommended: Level {optimal['level']} with MAE {optimal['mae']:.1f}")
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Try to import image reading libraries
try:
    import tifffile as tiff
except ImportError:
    tiff = None

try:
    from skimage import io as skio
    from skimage import util as skutil
except ImportError:
    skio = None
    skutil = None

# Try to import Cellpose
try:
    from cellpose import models as _models
except ImportError:
    _models = None


def _read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image file and return as numpy array."""
    path = Path(path)
    if not path.exists():
        print(f"[CALIBRATION] Image not found: {path}")
        return None
    
    try:
        if tiff and path.suffix.lower() in (".tif", ".tiff"):
            arr = tiff.imread(str(path))
        elif skio:
            arr = skio.imread(str(path))
        else:
            print("[CALIBRATION] No image reading library available (tifffile or skimage)")
            return None
        return arr
    except Exception as e:
        print(f"[CALIBRATION] Error reading {path}: {e}")
        return None


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in range [0, 1]."""
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle multi-channel images - take first channel or convert to grayscale
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            # RGB(A) - convert to grayscale
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        elif arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] < arr.shape[1]:
            # Channel-first format (C, H, W) - take first channel
            arr = arr[0]
        else:
            # Take first channel
            arr = arr[..., 0]
    
    # Normalize to [0, 1]
    arr_min = float(np.nanmin(arr))
    arr_max = float(np.nanmax(arr))
    
    if np.isfinite(arr_min) and np.isfinite(arr_max) and arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    elif arr_max != 0:
        arr = arr / arr_max
    
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def evaluate_single_image(
    image_path: str,
    level_params: Dict[str, Any],
    model_type: str = "cyto2",
    use_gpu: bool = False,
    diameter: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run cell detection on a single image with specified sensitivity parameters.
    
    Args:
        image_path: Path to the image file
        level_params: Dictionary with sensitivity parameters:
            - cellprob: cellprob_threshold (-1.0 to 1.0)
            - flow: flow_threshold (-1.0 to 1.0)
            - snr: minimum SNR (not used in basic mode)
            - floor_pct: background floor percentile
            - abs_int: absolute intensity threshold
            - itf: intensity threshold factor
        model_type: Cellpose model type (cyto2, cyto3, cyto, nuclei)
        use_gpu: Whether to use GPU acceleration
        diameter: Expected cell diameter (None = auto)
    
    Returns:
        Dict with:
            - cell_count: Number of detected cells
            - error: Error message if failed, None otherwise
            - duration_ms: Processing time in milliseconds
    """
    if _models is None:
        return {"cell_count": 0, "error": "Cellpose not available", "duration_ms": 0}
    
    start_time = time.time()
    
    try:
        # Read and normalize image
        img = _read_image(Path(image_path))
        if img is None:
            return {"cell_count": 0, "error": f"Could not read image: {image_path}", "duration_ms": 0}
        
        gray_norm = _normalize_image(img)
        
        # Extract parameters
        cellprob_threshold = float(level_params.get("cellprob", 0.0))
        flow_threshold = float(level_params.get("flow", 0.4))
        
        # Load model
        model = _models.CellposeModel(gpu=use_gpu, model_type=model_type)
        
        # Run segmentation
        masks, flows, styles = model.eval(
            gray_norm,
            channels=[0, 0],  # Grayscale
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        
        # Count cells (unique labels excluding background 0)
        unique_labels = np.unique(masks)
        cell_count = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        cell_count = max(0, cell_count)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return {
            "cell_count": cell_count,
            "error": None,
            "duration_ms": duration_ms,
        }
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "cell_count": 0,
            "error": str(e),
            "duration_ms": duration_ms,
        }


def run_calibration(
    images_with_counts: List[Dict[str, Any]],
    sensitivity_levels: List[Dict[str, Any]],
    model_type: str = "cyto2",
    use_gpu: bool = False,
    diameter: Optional[float] = None,
    callback: Optional[Callable[[int, int, str, Dict], None]] = None,
) -> Dict[str, Any]:
    """
    Run calibration by testing all sensitivity levels on all images.
    
    Args:
        images_with_counts: List of dicts with 'path' and 'expected' keys
        sensitivity_levels: List of sensitivity level parameter dicts
        model_type: Cellpose model type
        use_gpu: Whether to use GPU
        diameter: Expected cell diameter (None = auto)
        callback: Progress callback function(current_step, total_steps, message, result)
    
    Returns:
        Dict with:
            - timestamp: ISO timestamp of calibration
            - images: List of image results with actual counts per level
            - levels: List of level results with MAE
            - optimal_level: Index of best level (0-14)
            - optimal_mae: MAE of best level
            - total_duration_ms: Total processing time
    """
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    n_images = len(images_with_counts)
    n_levels = len(sensitivity_levels)
    total_steps = n_images * n_levels
    current_step = 0
    
    # Initialize results structure
    image_results = []
    for img_info in images_with_counts:
        image_results.append({
            "path": img_info["path"],
            "filename": Path(img_info["path"]).name,
            "expected": int(img_info.get("expected", 0)),
            "counts_by_level": {},
            "errors_by_level": {},
        })
    
    level_results = []
    for level_idx, level_params in enumerate(sensitivity_levels):
        level_results.append({
            "index": level_idx,
            "label": level_params.get("label", f"Level {level_idx + 1}"),
            "params": level_params,
            "total_actual": 0,
            "total_expected": 0,
            "mae": 0.0,
            "errors": [],
        })
    
    # Run detection for each image × level combination (sequentially)
    for level_idx, level_params in enumerate(sensitivity_levels):
        level_errors = []
        level_total_actual = 0
        level_total_expected = 0
        
        for img_idx, img_info in enumerate(images_with_counts):
            current_step += 1
            img_path = img_info["path"]
            expected = int(img_info.get("expected", 0))
            
            # Progress callback
            if callback:
                msg = f"Level {level_idx + 1}/{n_levels}, Image {img_idx + 1}/{n_images}"
                callback(current_step, total_steps, msg, None)
            
            # Run evaluation
            result = evaluate_single_image(
                img_path,
                level_params,
                model_type=model_type,
                use_gpu=use_gpu,
                diameter=diameter,
            )
            
            actual = result["cell_count"]
            error = abs(actual - expected)
            
            # Store in image results
            image_results[img_idx]["counts_by_level"][str(level_idx)] = actual
            if result["error"]:
                image_results[img_idx]["errors_by_level"][str(level_idx)] = result["error"]
            
            # Accumulate for level
            level_errors.append(error)
            level_total_actual += actual
            level_total_expected += expected
            
            # Progress callback with result
            if callback:
                callback(current_step, total_steps, msg, {
                    "level_idx": level_idx,
                    "img_idx": img_idx,
                    "expected": expected,
                    "actual": actual,
                    "error": error,
                })
        
        # Calculate MAE for this level
        mae = sum(level_errors) / len(level_errors) if level_errors else float('inf')
        level_results[level_idx]["total_actual"] = level_total_actual
        level_results[level_idx]["total_expected"] = level_total_expected
        level_results[level_idx]["mae"] = mae
        level_results[level_idx]["errors"] = level_errors
    
    # Find optimal level
    optimal_level = 0
    optimal_mae = float('inf')
    for level_idx, level_res in enumerate(level_results):
        if level_res["mae"] < optimal_mae:
            optimal_mae = level_res["mae"]
            optimal_level = level_idx
    
    total_duration_ms = int((time.time() - start_time) * 1000)
    
    return {
        "timestamp": timestamp,
        "images": image_results,
        "levels": level_results,
        "optimal_level": optimal_level,
        "optimal_mae": optimal_mae,
        "total_duration_ms": total_duration_ms,
        "config": {
            "model_type": model_type,
            "use_gpu": use_gpu,
            "diameter": diameter,
            "n_images": n_images,
            "n_levels": n_levels,
        },
    }


def find_optimal_level(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optimal level info from calibration results.
    
    Args:
        results: Results dict from run_calibration
    
    Returns:
        Dict with:
            - level: Level index (0-14)
            - label: Level label string
            - mae: Mean Absolute Error
            - params: Level parameters
    """
    optimal_idx = results.get("optimal_level", 0)
    levels = results.get("levels", [])
    
    if not levels or optimal_idx >= len(levels):
        return {
            "level": 0,
            "label": "Level 1",
            "mae": float('inf'),
            "params": {},
        }
    
    optimal = levels[optimal_idx]
    return {
        "level": optimal_idx,
        "label": optimal.get("label", f"Level {optimal_idx + 1}"),
        "mae": optimal.get("mae", float('inf')),
        "params": optimal.get("params", {}),
    }


def save_calibration_result(
    results: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """
    Save calibration results to JSON file with timestamp.
    
    Args:
        results: Results dict from run_calibration
        output_dir: Directory to save results (will create calibration_history subfolder)
    
    Returns:
        Path to saved file, or None on error
    """
    try:
        output_path = Path(output_dir) / "calibration_history"
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = results.get("timestamp", datetime.now().isoformat())
        # Create filename from timestamp
        ts_clean = timestamp.replace(":", "-").replace("T", "_").split(".")[0]
        filename = f"calibration_{ts_clean}.json"
        filepath = output_path / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[CALIBRATION] Results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"[CALIBRATION] Error saving results: {e}")
        return None


def list_tiff_images(input_dir: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List TIFF images in the input directory.
    
    Args:
        input_dir: Path to input directory
        limit: Maximum number of images to return
    
    Returns:
        List of dicts with path, filename, size_mb
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        return []
    
    images = []
    extensions = (".tif", ".tiff", ".TIF", ".TIFF")
    
    try:
        for f in sorted(input_path.rglob("*")):
            if f.is_file() and f.suffix in extensions:
                try:
                    size_mb = f.stat().st_size / (1024 * 1024)
                except Exception:
                    size_mb = 0
                
                images.append({
                    "path": str(f),
                    "filename": f.name,
                    "relative_path": str(f.relative_to(input_path)),
                    "size_mb": round(size_mb, 2),
                })
                
                if len(images) >= limit:
                    break
    except Exception as e:
        print(f"[CALIBRATION] Error listing images: {e}")
    
    return images


def get_calibration_history(output_dir: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get list of previous calibration results.
    
    Args:
        output_dir: Directory containing calibration_history subfolder
        limit: Maximum number of results to return
    
    Returns:
        List of dicts with path, timestamp, optimal_level, optimal_mae
    """
    history_path = Path(output_dir) / "calibration_history"
    if not history_path.exists():
        return []
    
    results = []
    try:
        for f in sorted(history_path.glob("calibration_*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                results.append({
                    "path": str(f),
                    "filename": f.name,
                    "timestamp": data.get("timestamp", ""),
                    "optimal_level": data.get("optimal_level", 0),
                    "optimal_mae": data.get("optimal_mae", 0),
                    "n_images": data.get("config", {}).get("n_images", 0),
                })
            except Exception:
                continue
    except Exception as e:
        print(f"[CALIBRATION] Error reading history: {e}")
    
    return results


# ============================================================================
# COLOCALIZATION CALIBRATION
# ============================================================================

# 9 calibration levels varying overlap_threshold and centroid_distance
COLOC_CALIBRATION_LEVELS = [
    {"level": 1, "label": "Sehr strikt",    "overlap_pct": 30, "centroid_dist": 4},
    {"level": 2, "label": "Strikt",         "overlap_pct": 20, "centroid_dist": 5},
    {"level": 3, "label": "Strikt-Mittel",  "overlap_pct": 15, "centroid_dist": 6},
    {"level": 4, "label": "Mittel-Strikt",  "overlap_pct": 10, "centroid_dist": 7},
    {"level": 5, "label": "Mittel",         "overlap_pct": 8,  "centroid_dist": 8},
    {"level": 6, "label": "Mittel-Locker",  "overlap_pct": 5,  "centroid_dist": 10},
    {"level": 7, "label": "Locker",         "overlap_pct": 3,  "centroid_dist": 12},
    {"level": 8, "label": "Sehr locker",    "overlap_pct": 2,  "centroid_dist": 15},
    {"level": 9, "label": "Ultra locker",   "overlap_pct": 1,  "centroid_dist": 20},
]


def list_coloc_samples(results_dir: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List available samples (detection results) that can be used for colocalization calibration.
    
    Looks for folders containing mask files in the results directory.
    
    Args:
        results_dir: Path to results directory from detection
        limit: Maximum number of samples to return
    
    Returns:
        List of dicts with sample_name, path, channels found
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    samples = []
    try:
        # Look for detection output structure: results_dir/channel_X/sample_masks.tif
        # Or: results_dir/sample_name/masks/
        for item in sorted(results_path.iterdir()):
            if not item.is_dir():
                continue
            
            # Check if this is a sample folder with mask files
            mask_files = list(item.glob("*_masks*.tif")) + list(item.glob("*_masks*.png"))
            if mask_files:
                samples.append({
                    "sample_name": item.name,
                    "path": str(item),
                    "n_masks": len(mask_files),
                })
            
            if len(samples) >= limit:
                break
    except Exception as e:
        print(f"[COLOC_CALIB] Error listing samples: {e}")
    
    return samples


def evaluate_coloc_params(
    sample_path: str,
    channel_pair: Tuple[int, int],
    overlap_pct: float,
    centroid_dist: float,
    coexpr_mode: str = "overlap"
) -> int:
    """
    Run colocalization analysis on a single sample with specific parameters.
    
    Args:
        sample_path: Path to sample directory with mask files
        channel_pair: Tuple of channel indices to analyze (e.g., (0, 1))
        overlap_pct: Overlap threshold percentage (0-100)
        centroid_dist: Maximum centroid distance in pixels
        coexpr_mode: Analysis mode ("overlap", "centroid", "either")
    
    Returns:
        Number of co-expressing cells detected
    """
    try:
        # Import analysis module
        from src.analysis import compute_overlap, compute_centroid_match
        
        sample_dir = Path(sample_path)
        ch_a, ch_b = channel_pair
        
        # Find mask files for both channels
        mask_a = None
        mask_b = None
        
        # Look for mask files matching channel indices
        for f in sample_dir.glob("*_masks*.tif"):
            fname = f.name.lower()
            if f"ch{ch_a}" in fname or f"channel{ch_a}" in fname or f"c{ch_a}" in fname:
                mask_a = f
            elif f"ch{ch_b}" in fname or f"channel{ch_b}" in fname or f"c{ch_b}" in fname:
                mask_b = f
        
        if mask_a is None or mask_b is None:
            print(f"[COLOC_CALIB] Could not find masks for channels {ch_a} and {ch_b} in {sample_path}")
            return 0
        
        # Read masks
        import tifffile as tiff
        masks_a = tiff.imread(str(mask_a))
        masks_b = tiff.imread(str(mask_b))
        
        # Compute overlap or centroid matching
        overlap_fraction = overlap_pct / 100.0
        
        if coexpr_mode == "centroid":
            coexpr_count = compute_centroid_match(
                masks_a, masks_b,
                max_distance=centroid_dist,
                overlap_threshold=overlap_fraction
            )
        elif coexpr_mode == "either":
            overlap_count = compute_overlap(masks_a, masks_b, min_overlap=overlap_fraction)
            centroid_count = compute_centroid_match(
                masks_a, masks_b,
                max_distance=centroid_dist,
                overlap_threshold=overlap_fraction
            )
            coexpr_count = max(overlap_count, centroid_count)
        else:  # "overlap" mode
            coexpr_count = compute_overlap(masks_a, masks_b, min_overlap=overlap_fraction)
        
        return coexpr_count
        
    except ImportError as e:
        print(f"[COLOC_CALIB] Import error: {e}")
        return 0
    except Exception as e:
        print(f"[COLOC_CALIB] Error evaluating params: {e}")
        traceback.print_exc()
        return 0


def run_coloc_calibration(
    samples_with_counts: List[Dict[str, Any]],
    channel_pair: Tuple[int, int],
    levels: List[Dict[str, Any]] = None,
    coexpr_mode: str = "overlap",
    callback: Optional[Callable[[int, int, str], None]] = None,
    stop_flag: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Run colocalization calibration across all parameter levels.
    
    Args:
        samples_with_counts: List of {"path": str, "expected_count": int}
        channel_pair: Tuple of channel indices (e.g., (0, 1))
        levels: Parameter levels to test (default: COLOC_CALIBRATION_LEVELS)
        coexpr_mode: Analysis mode ("overlap", "centroid", "either")
        callback: Progress callback(current_step, total_steps, message)
        stop_flag: Dict with 'stop' key to signal early termination
    
    Returns:
        Dict with level_results, optimal_level, config info
    """
    if levels is None:
        levels = COLOC_CALIBRATION_LEVELS
    
    n_samples = len(samples_with_counts)
    n_levels = len(levels)
    total_steps = n_samples * n_levels
    current_step = 0
    
    level_results = []
    
    for level in levels:
        if stop_flag and stop_flag.get('stop'):
            print("[COLOC_CALIB] Calibration stopped by user")
            break
        
        level_idx = level["level"]
        label = level["label"]
        overlap_pct = level["overlap_pct"]
        centroid_dist = level["centroid_dist"]
        
        total_detected = 0
        total_expected = 0
        sample_errors = []
        
        for sample in samples_with_counts:
            if stop_flag and stop_flag.get('stop'):
                break
            
            current_step += 1
            if callback:
                callback(current_step, total_steps, f"Level {level_idx} ({label}): {Path(sample['path']).name}")
            
            detected = evaluate_coloc_params(
                sample["path"],
                channel_pair,
                overlap_pct,
                centroid_dist,
                coexpr_mode
            )
            
            expected = sample.get("expected_count", 0)
            error = abs(detected - expected)
            
            total_detected += detected
            total_expected += expected
            sample_errors.append(error)
        
        # Calculate MAE for this level
        mae = sum(sample_errors) / len(sample_errors) if sample_errors else 999999
        
        level_results.append({
            "level": level_idx,
            "label": label,
            "overlap_pct": overlap_pct,
            "centroid_dist": centroid_dist,
            "total_detected": total_detected,
            "total_expected": total_expected,
            "mae": mae,
        })
    
    # Find optimal level (lowest MAE)
    optimal = min(level_results, key=lambda x: x["mae"]) if level_results else None
    
    results = {
        "level_results": level_results,
        "optimal_level": optimal,
        "config": {
            "n_samples": n_samples,
            "channel_pair": list(channel_pair),
            "coexpr_mode": coexpr_mode,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    return results


def save_coloc_calibration_result(results: Dict[str, Any], output_dir: str) -> str:
    """Save colocalization calibration results to JSON."""
    history_dir = Path(output_dir) / "coloc_calibration_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coloc_calib_{timestamp}.json"
    filepath = history_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def get_coloc_calibration_history(output_dir: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get list of previous colocalization calibration results."""
    history_path = Path(output_dir) / "coloc_calibration_history"
    if not history_path.exists():
        return []
    
    results = []
    try:
        for f in sorted(history_path.glob("coloc_calib_*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                optimal = data.get("optimal_level", {})
                config = data.get("config", {})
                results.append({
                    "path": str(f),
                    "filename": f.name,
                    "timestamp": config.get("timestamp", ""),
                    "optimal_level": optimal.get("level", 0),
                    "optimal_label": optimal.get("label", ""),
                    "optimal_mae": optimal.get("mae", 0),
                    "channel_pair": config.get("channel_pair", []),
                    "n_samples": config.get("n_samples", 0),
                })
            except Exception:
                continue
    except Exception as e:
        print(f"[COLOC_CALIB] Error reading history: {e}")
    
    return results


if __name__ == "__main__":
    # Simple test
    print("Calibration Utils - Test Mode")
    print("=" * 40)
    
    # List images in current directory
    images = list_tiff_images(".", limit=5)
    print(f"Found {len(images)} TIFF images")
    for img in images[:3]:
        print(f"  - {img['filename']} ({img['size_mb']:.2f} MB)")
    
    print("\nColocalization Calibration Levels:")
    for lvl in COLOC_CALIBRATION_LEVELS:
        print(f"  Level {lvl['level']}: {lvl['label']} (overlap={lvl['overlap_pct']}%, dist={lvl['centroid_dist']}px)")
