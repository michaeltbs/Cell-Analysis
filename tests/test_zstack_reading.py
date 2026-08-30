"""tests/test_zstack_reading.py — Z-stack handling in batch_segment image reading."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.batch_segment import _read_image


def test_read_image_zstack_maxprojection(tmp_path):
    """A (Z, Y, X) TIFF with Z > 8 planes must collapse to (Y, X) via max
    projection (not be mistaken for a channel axis)."""
    data = np.zeros((12, 32, 32), dtype=np.uint16)
    data[0, 4:8, 4:8] = 100
    data[11, 4:8, 4:8] = 250  # max plane
    p = tmp_path / "zstack.tif"
    tifffile.imwrite(p, data)

    out = _read_image(p)
    assert out.shape == (32, 32)
    assert out[4, 4] == 250


def test_read_image_channel_stack_untouched(tmp_path):
    """A (C, Y, X) stack with C <= 8 must stay 3D (channel axis preserved)."""
    data = np.zeros((3, 32, 32), dtype=np.uint16)
    data[1, 4:8, 4:8] = 100
    p = tmp_path / "channels.tif"
    tifffile.imwrite(p, data)

    out = _read_image(p)
    assert out.shape == (3, 32, 32)
    assert out[1, 4, 4] == 100
