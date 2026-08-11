"""
src/calibration/train.py — Cellpose fine-tuning wrapper.

Trains a custom model on user-annotated image/mask pairs, following the
official Cellpose training docs:
    https://cellpose.readthedocs.io/en/latest/train.html

Recommended setup for fine-tuning the Cellpose-SAM model:
    python -m cellpose --train --dir ~/images/train/ --test_dir ~/images/test/ \
        --learning_rate 0.00001 --weight_decay 0.1 --n_epochs 100 --train_batch_size 1

Data layout expected:
    train/
        sample1.tif
        sample1_masks.tif     (0 = background, 1,2,... = cells)
        sample2.tif
        sample2_masks.tif
    test/   (optional)
        ...
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tifffile


def validate_training_data(train_dir: str, mask_filter: str = "_masks") -> Dict[str, Any]:
    """
    Check that the training directory has paired image/mask files.

    Returns dict with valid pairs, missing masks, and warnings.
    """
    train_path = Path(train_dir)
    pairs = []
    missing = []
    warnings = []

    if not train_path.exists():
        return {"valid": False, "pairs": [], "missing_masks": [], "warnings": ["Training dir not found"]}

    # find all images that are not masks themselves
    image_exts = (".tif", ".tiff", ".png")
    candidates = sorted(p for p in train_path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)
    for img in candidates:
        if mask_filter in img.name or img.name.endswith("_seg.npy"):
            continue
        mask_path = img.with_name(img.stem + mask_filter + img.suffix)
        if not mask_path.exists():
            # try _masks.tif with same stem
            alt = img.parent / (img.stem + mask_filter + ".tif")
            if alt.exists():
                mask_path = alt
            else:
                missing.append(str(img))
                continue
        pairs.append({"image": str(img), "mask": str(mask_path)})

    # quick mask sanity check
    n_masks_total = 0
    for p in pairs:
        try:
            m = tifffile.imread(p["mask"])
            n = int(np.max(m)) if m.size else 0
            n_masks_total += n
            if n == 0:
                warnings.append(f"{Path(p['mask']).name}: mask is empty (no labels)")
        except Exception as e:
            warnings.append(f"{Path(p['mask']).name}: could not read mask ({e})")

    return {
        "valid": len(pairs) > 0 and len(missing) == 0,
        "pairs": pairs,
        "missing_masks": missing,
        "n_pairs": len(pairs),
        "n_masks_total": n_masks_total,
        "warnings": warnings,
    }


def run_finetune(
    train_dir: str,
    test_dir: Optional[str] = None,
    model_name: str = "cpsam",
    mask_filter: str = "_masks",
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    n_epochs: int = 100,
    train_batch_size: int = 1,
    use_gpu: bool = True,
    model_name_out: str = "cellpose_custom",
    callback: Optional[Callable[[int, int, str, Optional[Dict]], None]] = None,
) -> Dict[str, Any]:
    """
    Run Cellpose fine-tuning as a subprocess.

    Returns dict with return code, stdout tail, and model path.
    """
    start = time.time()

    validation = validate_training_data(train_dir, mask_filter)
    if not validation["valid"]:
        return {
            "success": False,
            "error": f"Invalid training data: {validation['missing_masks']}",
            "validation": validation,
            "duration_s": 0,
        }

    cmd = [
        sys.executable, "-m", "cellpose", "--train",
        "--dir", str(Path(train_dir).resolve()),
        "--mask_filter", mask_filter,
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--n_epochs", str(n_epochs),
        "--train_batch_size", str(train_batch_size),
        "--model_name_out", model_name_out,
    ]
    if test_dir:
        cmd += ["--test_dir", str(Path(test_dir).resolve())]
    if use_gpu:
        cmd.append("--use_gpu")

    if callback:
        callback(0, 1, f"Starting training: {model_name} -> {model_name_out}", None)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=n_epochs * 600,  # generous upper bound
        )
        stdout_tail = proc.stdout[-4000:] + "\n" + proc.stderr[-2000:]
        model_path = Path(train_dir) / "models" / f"{model_name_out}.model"
        success = proc.returncode == 0 and model_path.exists()

        if callback:
            callback(1, 1, "Training finished", {"success": success})

        return {
            "success": success,
            "return_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "model_path": str(model_path) if model_path.exists() else None,
            "validation": validation,
            "duration_s": int(time.time() - start),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Training timed out",
            "duration_s": int(time.time() - start),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "duration_s": int(time.time() - start),
        }
