"""
src/validation/metrics.py — validation metrics for cell segmentation and
co-expression analysis against ground truth masks/cell lists.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Pixel-level intersection-over-union for two binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def pixel_level_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Pixel-level precision, recall, F1, IoU."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _centroids(label_img: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of (row, col) centroids for each labeled instance."""
    props = regionprops(label_img)
    if not props:
        return np.empty((0, 2))
    return np.array([p.centroid for p in props])


def instance_detection_metrics(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Match predicted instances to ground-truth instances via IoU and report
    precision, recall, F1 and mean matched IoU.
    """
    pred_props = regionprops(pred_labels)
    gt_props = regionprops(gt_labels)

    if not gt_props:
        # no ground truth
        precision = 0.0 if pred_props else 1.0
        return precision, 0.0, 0.0, 0.0
    if not pred_props:
        return 0.0, 0.0, 0.0, 0.0

    iou_matrix = np.zeros((len(pred_props), len(gt_props)), dtype=np.float32)
    for i, pp in enumerate(pred_props):
        p_mask = pred_labels == pp.label
        for j, gp in enumerate(gt_props):
            g_mask = gt_labels == gp.label
            iou_matrix[i, j] = mask_iou(p_mask, g_mask)

    # hungarian matching on cost = 1 - iou
    row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)

    tp = int((iou_matrix[row_ind, col_ind] >= iou_threshold).sum())
    fp = len(pred_props) - tp
    fn = len(gt_props) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(iou_matrix[row_ind, col_ind].mean()) if row_ind.size > 0 else 0.0

    return precision, recall, f1, mean_iou


def evaluate_ground_truth_csv(
    pred_csv_path: str,
    gt_csv_path: str,
    group_cols: List[str] | None = None,
) -> Dict[str, float]:
    """
    Compare two count CSVs (prediction vs ground truth) by channel/region.
    Returns MAE and Pearson correlation per grouping key.
    """
    import pandas as pd

    group_cols = group_cols or ["channel", "region"]
    pred = pd.read_csv(pred_csv_path)
    gt = pd.read_csv(gt_csv_path)

    for col in group_cols:
        if col not in pred.columns or col not in gt.columns:
            raise ValueError(f"group column {col} missing from one CSV")

    merged = pd.merge(
        pred, gt, on=group_cols, suffixes=("_pred", "_gt"), how="outer"
    ).fillna(0)
    count_pred = merged["cell_count_pred"].values
    count_gt = merged["cell_count_gt"].values

    mae = float(np.mean(np.abs(count_pred - count_gt)))
    corr = float(np.corrcoef(count_pred, count_gt)[0, 1]) if len(count_pred) > 1 else 0.0

    return {"mae": mae, "pearson_r": corr, "n_groups": len(merged)}
