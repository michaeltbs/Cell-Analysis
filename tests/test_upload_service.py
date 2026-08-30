"""tests/test_upload_service.py — upload handling for new microscopy formats."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.upload_service import handle_upload


def test_handle_upload_ome_tiff_zstack_maxprojection(tmp_path):
    """An OME-TIFF with (C, Z, Y, X) must be converted to a CYX stack TIFF
    with max-projection over Z, placed in the pos dir."""
    data = np.zeros((2, 3, 16, 16), dtype=np.uint16)
    data[0, 0, 4:8, 4:8] = 100
    data[0, 2, 4:8, 4:8] = 200  # max for ch0
    data[1, 1, 10:14, 10:14] = 150

    ome_path = tmp_path / "sample.ome.tif"
    tifffile.imwrite(ome_path, data, photometric="minisblack", metadata={"axes": "CZYX"})
    content = ome_path.read_bytes()

    result = handle_upload([("sample.ome.tif", content)], "job-test-1", channels=[0, 1], target_size=None)

    out = Path(result["input_pos"]) / "sample_stack.tif"
    assert out.exists()
    arr = tifffile.imread(out)
    assert arr.shape == (2, 16, 16)  # C, Y, X after max-projection
    assert arr[0, 4, 4] == 200  # max over Z
    assert arr[1, 10, 10] == 150


def test_handle_upload_nd2_neg_placement(tmp_path):
    """ND2 files with 'neg' in the name must land in the neg dir."""
    # fake ND2 content — loader is mocked at the module level
    import src.api.upload_service as us

    fake = np.zeros((1, 2, 8, 8), dtype=np.uint16)
    fake[0, 0, 2:4, 2:4] = 50
    fake[0, 1, 2:4, 2:4] = 90

    orig = us.load_microscopy_to_CZYX
    us.load_microscopy_to_CZYX = lambda p: fake
    try:
        result = handle_upload([("ctrl_neg.nd2", b"fake-nd2-bytes")], "job-test-2", channels=[0], target_size=None)
    finally:
        us.load_microscopy_to_CZYX = orig

    out = Path(result["input_neg"]) / "ctrl_neg_stack.tif"
    assert out.exists()
    arr = tifffile.imread(out)
    assert arr.shape == (1, 8, 8)
    assert arr[0, 2, 2] == 90  # max over Z
