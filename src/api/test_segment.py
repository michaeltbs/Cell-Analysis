"""
src/api/test_segment.py — lightweight threshold-based segmentation for local API tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tifffile
from skimage.measure import label, regionprops


def threshold_segment(
    input_dir: str,
    output_root: str,
    threshold: float = 0.5,
    min_area: int = 5,
) -> Dict[str, List[Dict[str, any]]]:
    """
    Fast test-mode segmentation: threshold on mean intensity, label, filter by area.
    Mimics the CSV output format of batch_segment.py for downstream co-expression.

    Also persists labeled masks as <output_root>/Input_<cond>/ALL/<stem>_mask.tif
    (same layout as batch_segment save_masks=True) so downstream metric exports work.
    """
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for condition in ["Input_pos", "Input_neg"]:
        cond_dir = Path(input_dir) / condition
        if not cond_dir.exists():
            continue
        rows = []
        for tiff_path in sorted(cond_dir.glob("*.tiff")) + sorted(cond_dir.glob("*.tif")):
            img = tifffile.imread(str(tiff_path))
            if img.ndim == 3:
                # assume channel last
                img = img[..., 0]
            img = img.astype(np.float32)
            if img.max() > 0:
                img_norm = img / img.max()
            else:
                img_norm = img
            mask = (img_norm > threshold).astype(np.uint8)
            labels = np.asarray(label(mask, connectivity=1))
            props = regionprops(labels)
            for p in props:
                if p.area < min_area:
                    continue
                rows.append({
                    "filename": tiff_path.name,
                    "condition": condition.replace("Input_", ""),
                    "region": "ALL",
                    "label": int(p.label),
                    "area": int(p.area),
                    "centroid_y": float(p.centroid[0]),
                    "centroid_x": float(p.centroid[1]),
                    "mean_intensity": float(np.mean(img[p.coords[:, 0], p.coords[:, 1]])),
                })
            # persist labels (uint16) next to downstream mask layout
            if labels.max() > 0:
                mask_out = out_root / condition / "ALL"
                mask_out.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    mask_out / f"{tiff_path.stem}_mask.tif",
                    labels.astype(np.uint16),
                    photometric="minisblack",
                )
        results[condition] = rows

    return results
