"""tests/test_metrics_service.py — unit tests for expression + distance-map exports.

Builds tiny synthetic TIFFs (channel-first C,Y,X) and label masks directly with
numpy, so no Cellpose or real segmentation is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.metrics_service import export_expression_csvs, export_distance_maps
from src.config.naming import NamingConfig

HERE = Path(__file__).resolve().parent


def _make_condition(
    cond_dir: Path, stem: str, out_root: Path, n_cells: int = 2, receptor_signal: bool = False
) -> None:
    """Create one input image (under cond_dir) + its label mask (under out_root).

    Mirrors the upload pipeline where masks are written by segmentation into
    <output_root>/Input_<condition>/<region>/ as <stem>_mask.tif.

    Image: (3, 32, 32) uint16.
    - channel 0: cells at full intensity (100), background 0
    - channel 1: half of each cell's pixels at 100, half at 0 -> 50% positive
    - channel 2: all zeros, unless receptor_signal -> spot outside the cells
    Mask: labeled uint16, one box per cell, values 1..n_cells.
    """
    img = np.zeros((3, 32, 32), dtype=np.uint16)
    labels = np.zeros((32, 32), dtype=np.uint16)

    for i in range(n_cells):
        y0, x0 = 4 + i * 10, 4 + i * 10
        # cell box 4x4 -> 16 px
        y1, x1 = y0 + 4, x0 + 4
        img[0, y0:y1, x0:x1] = 100
        # channel 1: upper half of the box positive, lower half negative
        img[1, y0 : y0 + 2, x0:x1] = 100
        img[1, y0 + 2 : y1, x0:x1] = 0
        labels[y0:y1, x0:x1] = i + 1

    if receptor_signal:
        # receptor spot clearly outside all cell boxes (cells end at x<=18)
        img[2, 20:24, 24:28] = 100

    cond_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(cond_dir / f"{stem}.tiff", img)
    mask_dir = out_root / cond_dir.name / "ALL"
    mask_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(mask_dir / f"{stem}_mask.tif", labels)


def _job_output(tmp_path: Path) -> Path:
    """Build the input/out structure like the upload pipeline: returns out_root."""
    input_root = tmp_path / "input_tiffs"
    out_root = tmp_path / "results"
    _make_condition(input_root / "Input_pos", "M001_1", out_root, n_cells=2, receptor_signal=True)
    _make_condition(input_root / "Input_neg", "M002_1", out_root, n_cells=1, receptor_signal=True)
    return out_root


# ---------------------------------------------------------------------------
# Task 2: expression percentage CSV export
# ---------------------------------------------------------------------------


def test_export_expression_csvs_writes_per_condition(tmp_path):
    out_root = _job_output(tmp_path)
    input_root = tmp_path / "input_tiffs"
    naming = NamingConfig()
    naming.channel_names = {0: "PomC", 1: "Glp1r"}

    result = export_expression_csvs(input_root, out_root, naming)

    pos_csv = Path(result["pos_csv"])
    assert pos_csv.exists()
    df = pd.read_csv(pos_csv)
    assert {"label", "area", "centroid_x", "centroid_y", "condition"} <= set(df.columns)
    assert "PomC_mean_intensity" in df.columns
    assert "Glp1r_percent_positive" in df.columns
    assert "PomC_percent_positive" in df.columns
    assert len(df) == 2  # two cells in pos

    # deterministic values: full-intensity channel -> 100% positive
    assert df.loc[df["label"] == 1, "PomC_percent_positive"].iloc[0] == pytest.approx(100.0)
    # channel 1: exactly half of the cell pixels positive
    assert df.loc[df["label"] == 1, "Glp1r_percent_positive"].iloc[0] == pytest.approx(50.0)
    # channel 2 all zeros -> 0%
    assert "Gal_mean_intensity" not in df.columns  # only configured channels exported


def test_export_expression_csvs_uses_condition_labels(tmp_path):
    out_root = _job_output(tmp_path)
    input_root = tmp_path / "input_tiffs"
    naming = NamingConfig()
    naming.condition_names = {"pos": "KO", "neg": "WT"}

    result = export_expression_csvs(input_root, out_root, naming)

    df = pd.read_csv(Path(result["all_csv"]))
    assert {"condition_label"} <= set(df.columns)
    assert set(df["condition_label"].unique()) == {"KO", "WT"}


def test_export_expression_csvs_handles_scale_mismatch(tmp_path):
    """Mask smaller than the image (resize_max case): intensity resized to mask,
    no crash, sane output."""
    input_root = tmp_path / "input_tiffs"
    out_root = tmp_path / "results"
    img = np.zeros((3, 64, 64), dtype=np.uint16)
    img[0, 8:24, 8:24] = 100
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:12, 4:12] = 1
    (input_root / "Input_pos").mkdir(parents=True)
    tifffile.imwrite(input_root / "Input_pos" / "big.tiff", img)
    (out_root / "Input_pos" / "ALL").mkdir(parents=True)
    tifffile.imwrite(out_root / "Input_pos" / "ALL" / "big_mask.tif", labels)

    result = export_expression_csvs(input_root, out_root, NamingConfig())

    df = pd.read_csv(Path(result["pos_csv"]))
    assert len(df) == 1
    assert df["area"].iloc[0] == 64


# ---------------------------------------------------------------------------
# Task 3: distance-map export (stats CSV + heatmap PNG)
# ---------------------------------------------------------------------------


def test_export_distance_maps_writes_stats_and_heatmap(tmp_path):
    out_root = _job_output(tmp_path)
    input_root = tmp_path / "input_tiffs"
    naming = NamingConfig()
    naming.channel_names = {0: "PomC", 1: "Glp1r", 2: "Gal"}

    result = export_distance_maps(input_root, out_root, naming, receptor_channel=2)

    stats = Path(result["stats_csv"])
    assert stats.exists()
    df = pd.read_csv(stats)
    assert {"filename", "condition", "mean_distance", "median_distance", "max_distance"} <= set(df.columns)
    assert len(df) == 2  # 1 pos + 1 neg image

    heatmaps = result["heatmaps"]
    assert len(heatmaps) > 0
    for png in heatmaps:
        assert Path(png).exists() and Path(png).stat().st_size > 0


def test_export_distance_maps_reports_empty_receptor(tmp_path):
    """No receptor signal in any image: stats row with all-zero distances,
    no crash."""
    input_root = tmp_path / "input_tiffs"
    out_root = tmp_path / "results"
    # receptor channel (2) all zeros; cells only in channel 0
    _make_condition(input_root / "Input_pos", "M001_1", out_root, n_cells=1)

    result = export_distance_maps(input_root, out_root, NamingConfig(), receptor_channel=2)

    df = pd.read_csv(Path(result["stats_csv"]))
    assert len(df) == 1
    assert df["mean_distance"].iloc[0] == 0.0
