"""src/api/metrics_service.py — expression + distance-map exports for job results.

Runs after detection/co-expression in an upload job and writes:
- expression_percentages_<condition>.csv per condition (+ expression_all.csv)
- distance_maps_stats.csv + heatmap PNGs per image

Input structure (produced by batch_segment / threshold_segment):
    input_tiffs/Input_pos/<stem>.tiff        original multi-channel images
    output_root/Input_pos/<region>/<stem>_mask.tif   labeled masks (uint16)
Conditions are derived from the directory names ("Input_pos" -> "pos").
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from src.config.naming import NamingConfig
from src.validation.expression import receptor_distance_map


def _iter_masks(output_root: Path):
    """Yield (condition, region, stem, mask_path) for every *_mask.tif under output_root.

    Mask layout from batch_segment: <output_root>/Input_<cond>/<region>/<stem>_mask.tif
    (single-region case: region == one subdir, e.g. 'ALL').
    """
    for mask_path in sorted(output_root.rglob("*_mask.tif")):
        rel = mask_path.relative_to(output_root)
        if len(rel.parts) < 2:
            continue
        condition = rel.parts[0]
        if condition.startswith("Input_"):
            condition = condition[len("Input_") :]
        stem = mask_path.name[: -len("_mask.tif")]
        # region = the part right after the condition dir, if present
        region = rel.parts[1] if len(rel.parts) >= 3 else ""
        yield condition, region, stem, mask_path


def _find_original(input_tiffs: Path, condition: str, stem: str, region: str = "") -> Path | None:
    """Locate the original TIFF for a mask, searching recursively.

    Inputs may be flat (<cond>/<stem>.tiff) or region-nested
    (<cond>/<region>/<stem>.tiff). Prefer the region-specific file when given,
    then fall back to any match under the condition dir.
    """
    cond_dir = input_tiffs / f"Input_{condition}"
    if not cond_dir.exists():
        return None

    def _match(p: Path) -> bool:
        return p.stem == stem and p.suffix.lower() in (".tiff", ".tif")

    if region:
        for p in sorted((cond_dir / region).glob("*")) if (cond_dir / region).is_dir() else []:
            if _match(p):
                return p
    for p in sorted(cond_dir.rglob("*")):
        if p.is_file() and _match(p):
            return p
    return None


def _channel_indices(img: np.ndarray) -> List[int]:
    """Channel axis detection: smallest axis with size <= 8 (like batch_segment)."""
    if img.ndim == 2:
        return []
    shape = img.shape
    candidates = [ax for ax, size in enumerate(shape) if size <= 8]
    ax = candidates[0] if candidates else len(shape) - 1
    return list(range(shape[ax]))


def _resize_to(img: np.ndarray, target_shape) -> np.ndarray:
    """Resize a single-channel image to target_shape (H, W) via numpy slicing/zoom."""
    if img.shape == tuple(target_shape):
        return img
    from skimage.transform import resize

    return resize(img, target_shape, order=1, preserve_range=True, anti_aliasing=True)


def _mask_dir_for(output_root: Path, condition: str) -> Path:
    """Masks are written per condition/region; if only one subdir exists use it."""
    cond_root = output_root / f"Input_{condition}"
    if cond_root.is_dir():
        subdirs = [p for p in cond_root.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
    # flat layout fallback
    return cond_root


def export_expression_csvs(
    input_tiffs: Path | str,
    output_root: Path | str,
    naming: NamingConfig | None = None,
) -> Dict[str, Any]:
    """Write per-cell expression statistics CSV per condition (and combined).

    Returns dict with 'pos_csv', 'neg_csv', 'all_csv' file paths.
    """
    input_tiffs = Path(input_tiffs)
    output_root = Path(output_root)
    naming = naming or NamingConfig()

    per_condition_rows: Dict[str, List[Dict[str, Any]]] = {}

    for condition, region, stem, mask_path in _iter_masks(output_root):
        try:
            labels = tifffile.imread(str(mask_path))
            if labels.ndim == 3:
                labels = labels[0]
            labels = labels.astype(np.int64)
        except Exception:
            continue

        orig = _find_original(input_tiffs, condition, stem, region)
        if orig is None:
            continue
        img = tifffile.imread(str(orig))
        if img.ndim == 2:
            continue  # no channels to analyse

        # channel axis
        ch_indices = _channel_indices(img)
        if not ch_indices:
            continue

        # resize intensity planes to mask resolution (resize_max case)
        mask_shape = labels.shape[:2]
        intensity_images: Dict[int, np.ndarray] = {}

        # detect axis for per-channel slices
        shape = img.shape
        ch_axis = ch_indices[0]  # first candidate axis
        # ensure ch_axis is the axis with size == len(ch_indices)
        for ax in range(img.ndim):
            if shape[ax] == len(ch_indices):
                ch_axis = ax
                break

        for idx in ch_indices:
            plane = np.take(img, idx, axis=ch_axis)
            if plane.shape[:2] != mask_shape:
                plane = _resize_to(plane, mask_shape)
            intensity_images[idx] = plane.astype(np.float32)

        # Per-cell expression: fraction of cell pixels above the GLOBAL channel
        # threshold (whole-image mean, matching expression_percentage semantics).
        # Deliberately NOT the per-cell mean (that measures heterogeneity and is
        # ~0% for uniformly bright cells).
        rows = []
        for p in regionprops(labels):
            rr, cc = p.coords[:, 0], p.coords[:, 1]
            row = {
                "label": int(p.label),
                "area": int(p.area),
                "centroid_y": float(p.centroid[0]),
                "centroid_x": float(p.centroid[1]),
            }
            for idx, plane in intensity_images.items():
                cell_px = plane[rr, cc]
                thr = float(plane.mean())
                if cell_px.size:
                    row[f"channel_{idx}_mean_intensity"] = float(np.mean(cell_px))
                    row[f"channel_{idx}_percent_positive"] = float(
                        np.mean(cell_px > thr) * 100.0
                    )
                else:
                    row[f"channel_{idx}_mean_intensity"] = 0.0
                    row[f"channel_{idx}_percent_positive"] = 0.0
            rows.append(row)
        for row in rows:
            row["condition"] = condition
            row["condition_label"] = naming.condition_name(condition)
            row["region"] = region
            row["filename"] = stem
            # rename channel columns to configured names
            for idx in ch_indices:
                ch_name = naming.channel_name(idx)
                for field in ("mean_intensity", "percent_positive"):
                    key = f"channel_{idx}_{field}"
                    if key in row:
                        row[f"{ch_name}_{field}"] = row.pop(key)

        if not rows:
            continue
        per_condition_rows.setdefault(condition, []).extend(rows)

    per_condition: Dict[str, str] = {}
    for condition, rows in per_condition_rows.items():
        df = pd.DataFrame(rows)
        cond_csv = output_root / f"expression_percentages_{condition}.csv"
        df.to_csv(cond_csv, index=False)
        per_condition[condition] = str(cond_csv)

    all_csv = output_root / "expression_all.csv"
    if per_condition:
        frames = [pd.read_csv(p) for p in per_condition.values()]
        pd.concat(frames, ignore_index=True).to_csv(all_csv, index=False)

    return {
        "pos_csv": per_condition.get("pos", ""),
        "neg_csv": per_condition.get("neg", ""),
        "all_csv": str(all_csv) if all_csv.exists() else "",
    }


def export_distance_maps(
    input_tiffs: Path | str,
    output_root: Path | str,
    naming: NamingConfig | None = None,
    receptor_channel: int = 2,
) -> Dict[str, Any]:
    """Compute receptor-to-cell distance statistics per image + heatmap PNGs.

    Receptor mask = pixels above Otsu threshold in the receptor channel
    (restricted to non-background signal). Cell mask = union of labels.
    Returns dict with 'stats_csv' and 'heatmaps' (list of PNG paths).
    """
    input_tiffs = Path(input_tiffs)
    output_root = Path(output_root)
    naming = naming or NamingConfig()

    stats_rows: List[Dict[str, Any]] = []
    heatmaps: List[str] = []
    maps_dir = output_root / "distance_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    for condition, region, stem, mask_path in _iter_masks(output_root):
        try:
            labels = tifffile.imread(str(mask_path))
            if labels.ndim == 3:
                labels = labels[0]
            labels = labels.astype(np.int64)
        except Exception:
            continue
        if labels.size == 0 or int(labels.max()) == 0:
            continue

        orig = _find_original(input_tiffs, condition, stem, region)
        if orig is None:
            continue
        img = tifffile.imread(str(orig))
        if img.ndim == 2:
            continue

        ch_indices = _channel_indices(img)
        if receptor_channel not in ch_indices:
            continue
        ch_axis = next(
            (ax for ax in range(img.ndim) if img.shape[ax] == len(ch_indices)),
            ch_indices[0],
        )
        receptor_plane = np.take(img, receptor_channel, axis=ch_axis).astype(np.float32)
        if receptor_plane.shape[:2] != labels.shape[:2]:
            receptor_plane = _resize_to(receptor_plane, labels.shape[:2])

        # receptor mask: Otsu threshold over the receptor channel
        from skimage.filters import threshold_otsu

        receptor_mask = np.zeros(labels.shape[:2], dtype=bool)
        if receptor_plane.max() > receptor_plane.min():
            thr = threshold_otsu(receptor_plane)
            receptor_mask = receptor_plane > thr
            # restrict to positive signal: at least 10 px, otherwise ignore channel
            if receptor_mask.sum() < 10:
                receptor_mask = np.zeros_like(receptor_mask)

        cell_mask = labels > 0
        stat = receptor_distance_map(receptor_mask, cell_mask)

        stats_rows.append(
            {
                "filename": stem,
                "condition": condition,
                "condition_label": naming.condition_name(condition),
                "region": region,
                "receptor_channel": receptor_channel,
                "receptor_channel_label": naming.channel_name(receptor_channel),
                "receptor_pixels": int(receptor_mask.sum()),
                "mean_distance": float(stat["mean_distance"]),
                "median_distance": float(stat["median_distance"]),
                "std_distance": float(stat["std_distance"]),
                "max_distance": float(stat["max_distance"]),
            }
        )

        if receptor_mask.sum() > 0:
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            dm = stat["distance_map"]
            im = ax.imshow(dm, cmap="viridis")
            ax.set_title(f"{stem} — {naming.channel_name(receptor_channel)} distance")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            png_path = maps_dir / f"{stem}_distance_map.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            heatmaps.append(str(png_path))

    stats_csv = output_root / "distance_maps_stats.csv"
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)

    return {"stats_csv": str(stats_csv) if stats_csv.exists() else "", "heatmaps": heatmaps}
