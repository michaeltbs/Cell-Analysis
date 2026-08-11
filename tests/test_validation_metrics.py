"""
tests/test_validation_metrics.py — unit tests for IoU / precision / recall
against manually or programmatically generated ground-truth masks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from skimage.measure import label

from tests.fixtures.synthetic_cells import create_synthetic_stack
from src.validation.metrics import (
    mask_iou,
    instance_detection_metrics,
    pixel_level_metrics,
)


def test_mask_iou_perfect():
    a = np.zeros((100, 100), dtype=np.uint16)
    a[20:40, 20:40] = 1
    b = a.copy()
    assert mask_iou(a > 0, b > 0) == pytest.approx(1.0, abs=1e-6)


def test_mask_iou_disjoint():
    a = np.zeros((100, 100), dtype=np.uint16)
    a[10:20, 10:20] = 1
    b = np.zeros((100, 100), dtype=np.uint16)
    b[80:90, 80:90] = 1
    assert mask_iou(a > 0, b > 0) == pytest.approx(0.0, abs=1e-6)


def test_instance_metrics_against_ground_truth():
    """Detect instances on synthetic image and compare to known centers."""
    stack, centers = create_synthetic_stack(
        shape=(256, 256), n_cells=20, n_channels=1, seed=7
    )
    img = stack[0]

    # simple threshold + label as baseline "prediction"
    pred = label(img > img.mean() + 0.7 * img.std())

    # build gt mask from centers
    gt = np.zeros_like(img, dtype=np.uint16)
    for _, cy, cx in centers:
        rr, cc = np.ogrid[: gt.shape[0], : gt.shape[1]]
        mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= 6 ** 2
        gt[mask] = 1

    gt_labels = label(gt)
    precision, recall, f1, iou = instance_detection_metrics(pred, gt_labels, iou_threshold=0.3)
    assert recall > 0.6, f"recall too low: {recall}"
    assert precision > 0.4, f"precision too low: {precision}"
    assert f1 > 0.4, f"f1 too low: {f1}"


def test_pixel_metrics():
    gt = np.zeros((100, 100), dtype=bool)
    gt[20:50, 20:50] = True
    pred = gt.copy()
    # add some false positives / negatives
    pred[60:70, 60:70] = True  # false positive
    gt[30:35, 30:35] = False   # false negative
    gt[80:90, 80:90] = True    # extra false negative region

    metrics = pixel_level_metrics(pred, gt)
    assert 0.5 < metrics["pixel_iou"] < 0.9
    assert metrics["pixel_precision"] < 1.0
    assert metrics["pixel_recall"] < 1.0


if __name__ == "__main__":
    test_mask_iou_perfect()
    test_mask_iou_disjoint()
    test_instance_metrics_against_ground_truth()
    test_pixel_metrics()
    print("✅ validation metrics tests passed")
