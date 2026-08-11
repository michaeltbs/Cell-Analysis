"""
src/calibration/profile.py — intensity profile analysis for detection tuning.

Analyzes the detected cells' intensity distribution to diagnose why bright
cells are missed or weak cells are counted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile
from skimage import measure


def analyze_detection_profile(
    image_path: str,
    mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute per-cell intensity profile from a segmentation mask.

    Args:
        image_path: source image path (for reference)
        mask: labeled mask from Cellpose
        intensity_image: optional intensity image; if None, tries to read image_path

    Returns:
        dict with cell stats, intensity histogram, and diagnostic flags
    """
    if intensity_image is None:
        try:
            intensity_image = tifffile.imread(image_path)
            if intensity_image.ndim == 3:
                # assume channel-first and take first channel
                intensity_image = intensity_image[0]
        except Exception as e:
            raise ValueError(f"Could not read intensity image: {e}") from e

    if mask is None or mask.size == 0 or intensity_image is None:
        return {
            "image": str(image_path),
            "cell_count": 0,
            "cells": [],
            "histogram": [],
            "diagnosis": ["No mask available"],
        }

    props = measure.regionprops(mask.astype(np.int32), intensity_image=intensity_image.astype(np.float32))
    cells = []
    for p in props:
        cells.append({
            "label": int(p.label),
            "area": int(p.area),
            "mean_intensity": float(p.mean_intensity or 0.0),
            "max_intensity": float(getattr(p, "max_intensity", p.mean_intensity) or 0.0),
            "min_intensity": float(getattr(p, "min_intensity", 0.0) or 0.0),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
        })

    if not cells:
        return {
            "image": str(image_path),
            "cell_count": 0,
            "cells": [],
            "histogram": [],
            "diagnosis": ["No cells detected"],
        }

    intensities = np.array([c["max_intensity"] for c in cells], dtype=np.float64)
    means = np.array([c["mean_intensity"] for c in cells], dtype=np.float64)

    # Histogram of max intensities (10 bins from min to max)
    lo, hi = float(intensities.min()), float(intensities.max())
    hist, edges = np.histogram(intensities, bins=10, range=(lo, hi)) if hi > lo else (np.array([len(intensities)]), np.array([lo, hi + 1.0]))
    histogram = [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(hist[i])}
        for i in range(len(hist))
    ]

    diagnosis = []
    p10 = float(np.percentile(intensities, 10))
    p90 = float(np.percentile(intensities, 90))
    mean = float(intensities.mean())
    dynamic_range = (p90 - p10) / (mean + 1e-9) if mean > 0 else 0.0

    if dynamic_range > 3.0:
        diagnosis.append(
            "Hohe Intensitäts-Spreizung: Einheitlicher Threshold schneidet entweder schwache ab oder "
            "lässt starke durch. Kalibrierung der Filter empfohlen."
        )
    if p10 < 0.02 and mean > 0.1:
        diagnosis.append(
            "Viele sehr schwache Zellen nahe dem Hintergrund. Threshold anheben, um False Positives zu reduzieren."
        )
    if p90 > 0.9:
        diagnosis.append(
            "Sehr helle Zellen nahe der Sättigung. Cellpose kann saturierte Zellen übersehen — "
            "Preprocessing prüfen oder Sättigung in Aufnahme vermeiden."
        )
    if not diagnosis:
        diagnosis.append("Verteilung sieht ausgewogen aus. Keine akute Fehlkonfiguration.")

    return {
        "image": str(image_path),
        "cell_count": len(cells),
        "cells": cells,
        "histogram": histogram,
        "stats": {
            "mean_max_intensity": mean,
            "p10_max_intensity": p10,
            "p90_max_intensity": p90,
            "dynamic_range": dynamic_range,
            "min_max_intensity": lo,
            "max_max_intensity": hi,
        },
        "diagnosis": diagnosis,
    }
