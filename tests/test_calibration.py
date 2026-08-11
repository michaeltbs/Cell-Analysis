"""
tests/test_calibration.py — tests for calibration profile + optimizer.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tifffile

from src.calibration.profile import analyze_detection_profile
from src.calibration.optimizer import _f1_score


def _make_mask_image(tmp: Path):
    """Create a small synthetic image + mask with known cell positions."""
    img = np.zeros((128, 128), dtype=np.uint16)
    mask = np.zeros((128, 128), dtype=np.uint16)
    # two bright cells
    img[30:40, 30:40] = 40000
    img[80:90, 80:90] = 30000
    mask[30:40, 30:40] = 1
    mask[80:90, 80:90] = 2
    p = tmp / "test_cells.tif"
    tifffile.imwrite(str(p), img)
    mp = tmp / "test_cells_mask.tif"
    tifffile.imwrite(str(mp), mask)
    return p, mp


def test_profile_detects_cells():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_p, mask_p = _make_mask_image(tmp)
        mask = tifffile.imread(str(mask_p))
        result = analyze_detection_profile(str(img_p), mask)
        assert result["cell_count"] == 2
        assert len(result["cells"]) == 2
        assert result["cells"][0]["max_intensity"] > 10000
        assert len(result["histogram"]) > 0


def test_profile_empty_mask():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_p, _ = _make_mask_image(tmp)
        empty = np.zeros((128, 128), dtype=np.uint16)
        result = analyze_detection_profile(str(img_p), empty)
        assert result["cell_count"] == 0


def test_f1_score():
    assert _f1_score(10, 0, 0) == 1.0
    assert _f1_score(5, 5, 5) == 0.5
    assert _f1_score(0, 0, 0) == 0.0


if __name__ == "__main__":
    test_profile_detects_cells()
    test_profile_empty_mask()
    test_f1_score()
    print("✅ calibration tests passed")
