"""
src/calibration/optimizer.py — parameter sweep optimizer for detection tuning.

Given expected cell counts for a few representative images, sweeps
detection parameters and finds the combination with best F1 / MAE balance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tifffile

from src.batch_segment import _load_cellpose_model, _extract_channel, _preprocess_image, _apply_filters


def _read_gray(path: Path, channel: int = 0, resize_max: int | None = None) -> np.ndarray:
    """Read image and extract requested channel as float32. Optionally downscale."""
    from src.batch_segment import _read_image
    img = _read_image(path)
    gray = _extract_channel(img, channel).astype(np.float32)
    if resize_max and max(gray.shape) > resize_max:
        from skimage import transform
        h, w = gray.shape
        scale = resize_max / max(h, w)
        gray = transform.resize(gray, (int(h * scale), int(w * scale)), preserve_range=True, anti_aliasing=True).astype(np.float32)
    return gray


def _count_cells(
    gray: np.ndarray,
    model,
    params: Dict[str, Any],
    base_cfg: Dict[str, Any],
) -> int:
    """Run detection with the given params and return cell count after filtering."""
    try:
        masks, _, _ = model.eval(
            gray,
            channels=[0, 0],
            diameter=params.get("diameter"),
            flow_threshold=float(params.get("flow", 0.4)),
            cellprob_threshold=float(params.get("cellprob", 0.0)),
        )
    except Exception:
        return 0

    masks = masks.astype(np.uint16, copy=False)

    # Build a per-image cfg with the sweep params merged in
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("filters", {})["min_area"] = int(params.get("min_area", 10))
    cfg.setdefault("filters", {})["max_area"] = int(params.get("max_area", 32000))
    cfg.setdefault("filters", {})["min_circularity"] = float(params.get("min_circ", 0.0))
    cfg.setdefault("filters", {})["max_circularity"] = float(params.get("max_circ", 1.0))
    cfg.setdefault("advanced_filtering", {})["intensity_mode"] = str(params.get("intensity_mode", "max"))
    cfg.setdefault("advanced_filtering", {})["max_intensity_threshold"] = float(params.get("max_intensity_threshold", 0.1))
    cfg.setdefault("advanced_filtering", {})["min_local_contrast"] = float(params.get("min_local_contrast", 1.2))
    cfg.setdefault("advanced_filtering", {})["snr_threshold"] = float(params.get("snr_threshold", 2.0))

    filtered, stats = _apply_filters(masks, gray, cfg, (int(gray.shape[0]), int(gray.shape[1])))
    return stats.get("kept", 0)


def _f1_score(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


def run_optimization_sweep(
    images_with_counts: List[Dict[str, Any]],
    model_name: str = "cyto2",
    use_gpu: bool = False,
    channel: int = 0,
    base_cfg: Optional[Dict[str, Any]] = None,
    params_grid: Optional[Dict[str, List[Any]]] = None,
    fast: bool = False,
    resize_max: int = 512,
    callback: Optional[Callable[[int, int, str, Dict], None]] = None,
) -> Dict[str, Any]:
    """
    Sweep detection params to minimize MAE + maximize F1 vs expected counts.

    Args:
        images_with_counts: [{"path": str, "expected": int}, ...]
        model_name: cellpose model type
        use_gpu: gpu flag
        channel: channel index to use
        base_cfg: base config dict (filters etc.)
        params_grid: dict of param -> list of values to sweep
        fast: if True, use a reduced grid (~12 combos) for quick calibration
              on slow hardware (e.g. Mac MPS with cpsam)

    Returns:
        dict with results, best params, best score, timings
    """
    start = time.time()
    base_cfg = base_cfg or {}

    if fast:
        default_grid: Dict[str, List[Any]] = {
            "cellprob": [0.0, 1.0],
            "flow": [0.4],
            "max_intensity_threshold": [0.05, 0.15],
            "min_local_contrast": [1.0, 1.2],
            "min_area": [10],
            "diameter": [None],
        }
    else:
        default_grid: Dict[str, List[Any]] = {
            "cellprob": [-1.0, 0.0, 1.0],
            "flow": [0.4],
            "max_intensity_threshold": [0.05, 0.1, 0.2],
            "min_local_contrast": [1.0, 1.2],
            "min_area": [5, 15, 30],
            "diameter": [None],
        }
    grid = params_grid or default_grid

    # Build cartesian product of params
    import itertools
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = list(itertools.product(*value_lists))

    # Load model ONCE and reuse for all combos (huge speedup on CPU/MPS)
    model = _load_cellpose_model(model_name, use_gpu=use_gpu)

    results = []
    total = len(combos)
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if params.get("diameter") is None:
            params["diameter"] = None

        mae_sum = 0.0
        tp_sum = fp_sum = fn_sum = 0
        per_image = []
        for img_info in images_with_counts:
            expected = int(img_info.get("expected", 0))
            try:
                gray = _read_gray(Path(img_info["path"]), channel, resize_max=resize_max)
                actual = _count_cells(gray, model, params, base_cfg)
            except Exception:
                actual = 0

            tp = min(actual, expected)
            fp = max(0, actual - expected)
            fn = max(0, expected - actual)
            mae_sum += abs(actual - expected)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            per_image.append({
                "path": str(img_info["path"]),
                "expected": expected,
                "actual": actual,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

        n = len(images_with_counts)
        mae = mae_sum / n if n else float("inf")
        f1 = _f1_score(tp_sum, fp_sum, fn_sum)
        # Combine: lower MAE + higher F1. Weight MAE heavily (counts matter).
        score = f1 - 0.1 * mae

        results.append({
            "params": {k: (v if v is not None else "auto") for k, v in params.items()},
            "mae": mae,
            "f1": f1,
            "score": score,
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum,
            "actual_total": tp_sum + fp_sum,
            "expected_total": tp_sum + fn_sum,
            "per_image": per_image,
        })

        if callback:
            callback(i + 1, total, f"params={params}", results[-1])

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0] if results else {}

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": model_name,
        "use_gpu": use_gpu,
        "channel": channel,
        "n_images": len(images_with_counts),
        "n_combos": total,
        "best": best,
        "top_results": results[:20],
        "total_duration_ms": int((time.time() - start) * 1000),
    }
