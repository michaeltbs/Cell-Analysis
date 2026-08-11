"""
src/validation/expression.py — expression percentage and distance-map metrics.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops


def expression_percentage(mask: np.ndarray, intensity_image: np.ndarray | None = None) -> Dict[str, float]:
    """
    Compute percentage of positive/negative pixels or labeled cells.

    If intensity_image is given, percentage is computed over positive pixels
    (pixels > mean). Otherwise over the binary mask.
    """
    mask = mask.astype(bool)
    total = mask.size
    positive = int(mask.sum())
    negative = total - positive

    pct_positive = positive / total * 100.0 if total > 0 else 0.0
    pct_negative = negative / total * 100.0 if total > 0 else 0.0

    result = {
        "total_pixels": total,
        "positive_pixels": positive,
        "negative_pixels": negative,
        "percent_positive": pct_positive,
        "percent_negative": pct_negative,
    }

    if intensity_image is not None:
        mean_intensity = float(np.mean(intensity_image))
        pos_pixels = intensity_image[mask]
        result["mean_intensity_positive"] = float(np.mean(pos_pixels)) if pos_pixels.size else 0.0
        result["mean_intensity_negative"] = float(np.mean(intensity_image[~mask]))
        result["threshold"] = mean_intensity

    return result


def expression_percentage_per_instance(
    labels: np.ndarray,
    intensity_images: Dict[int, np.ndarray],
) -> List[Dict[str, float]]:
    """
    For each labeled cell/instance, compute expression percentage per channel.
    """
    rows = []
    props = regionprops(labels, intensity_image=None)
    for p in props:
        rr, cc = p.coords[:, 0], p.coords[:, 1]
        row = {
            "label": int(p.label),
            "area": int(p.area),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
        }
        for ch_idx, img in intensity_images.items():
            cell_intensity = img[rr, cc]
            if cell_intensity.size:
                row[f"channel_{ch_idx}_mean_intensity"] = float(np.mean(cell_intensity))
                row[f"channel_{ch_idx}_percent_positive"] = float(np.mean(cell_intensity > cell_intensity.mean()) * 100.0)
            else:
                row[f"channel_{ch_idx}_mean_intensity"] = 0.0
                row[f"channel_{ch_idx}_percent_positive"] = 0.0
        rows.append(row)
    return rows


def receptor_distance_map(receptor_mask: np.ndarray, cell_mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute distance transform from each receptor-positive pixel to the nearest cell.

    Returns dict with distance map array and summary statistics.
    """
    receptor = receptor_mask.astype(bool)
    cell = cell_mask.astype(bool)

    # distance from every pixel to nearest non-cell (background) = inside-cell distance inverted
    # we want distance from receptor pixel to nearest cell boundary, so distance_transform_edt on inverted cell
    dist_to_cell = ndi.distance_transform_edt(~cell)
    receptor_distances = dist_to_cell[receptor]

    return {
        "distance_map": dist_to_cell,
        "receptor_distances": receptor_distances,
        "mean_distance": float(np.mean(receptor_distances)) if receptor_distances.size else 0.0,
        "median_distance": float(np.median(receptor_distances)) if receptor_distances.size else 0.0,
        "std_distance": float(np.std(receptor_distances)) if receptor_distances.size else 0.0,
        "max_distance": float(np.max(receptor_distances)) if receptor_distances.size else 0.0,
    }
