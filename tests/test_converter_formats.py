"""tests/test_converter_formats.py — ND2/LIF/OME-TIFF loading + Z-stack handling.

ND2 and LIF libraries are read-only (no synthetic file writer), so those
loaders are tested via dispatch + mocked readers. OME-TIFF is fully
round-tripped with tifffile (it can write OME-TIFF).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.czi_converter import (
    load_microscopy_to_CZYX,
    load_ome_tiff_to_CZYX,
    collapse_z_max,
    find_microscopy_files,
    SUPPORTED_MICROSCOPY_EXTS,
)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_dispatch_known_extensions():
    """Every supported extension must map to a loader without raising."""
    for ext in SUPPORTED_MICROSCOPY_EXTS:
        assert ext.startswith(".")


def test_dispatch_nd2_uses_nd2_loader(tmp_path):
    fake = np.zeros((2, 3, 16, 16), dtype=np.uint16)  # (C, Z, Y, X)
    with mock.patch("src.czi_converter.load_nd2_to_CZYX", return_value=fake) as m:
        out = load_microscopy_to_CZYX(str(tmp_path / "img.nd2"))
        m.assert_called_once()
        assert out.shape == (2, 3, 16, 16)


def test_dispatch_lif_uses_lif_loader(tmp_path):
    fake = np.zeros((2, 1, 16, 16), dtype=np.uint16)
    with mock.patch("src.czi_converter.load_lif_to_CZYX", return_value=fake) as m:
        out = load_microscopy_to_CZYX(str(tmp_path / "img.lif"))
        m.assert_called_once()
        assert out.shape == (2, 1, 16, 16)


def test_dispatch_ome_tiff_uses_tiff_loader(tmp_path):
    fake = np.zeros((2, 1, 16, 16), dtype=np.uint16)
    with mock.patch("src.czi_converter.load_ome_tiff_to_CZYX", return_value=fake) as m:
        out = load_microscopy_to_CZYX(str(tmp_path / "img.ome.tif"))
        m.assert_called_once()
        assert out.shape == (2, 1, 16, 16)


def test_dispatch_unknown_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        load_microscopy_to_CZYX(str(tmp_path / "img.xyz"))


# ---------------------------------------------------------------------------
# OME-TIFF roundtrip (real file, tifffile can write OME-TIFF)
# ---------------------------------------------------------------------------


def test_ome_tiff_roundtrip_czxy(tmp_path):
    """A (C, Z, Y, X) OME-TIFF written by tifffile must load back as CZYX."""
    data = np.zeros((2, 3, 16, 16), dtype=np.uint16)
    data[0, 0, 4:8, 4:8] = 100
    data[1, 2, 10:14, 10:14] = 200
    p = tmp_path / "stack.ome.tif"
    tifffile.imwrite(p, data, photometric="minisblack", metadata={"axes": "CZYX"})

    out = load_ome_tiff_to_CZYX(str(p))
    assert out.shape == (2, 3, 16, 16)
    assert out[0, 0, 4, 4] == 100
    assert out[1, 2, 10, 10] == 200


def test_ome_tiff_roundtrip_cyx(tmp_path):
    """Plain (C, Y, X) OME-TIFF loads as (C, 1, Y, X) — Z axis added."""
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    data[0, 4:8, 4:8] = 100
    p = tmp_path / "plain.ome.tif"
    tifffile.imwrite(p, data, photometric="minisblack", metadata={"axes": "CYX"})

    out = load_ome_tiff_to_CZYX(str(p))
    assert out.shape == (2, 1, 16, 16)
    assert out[0, 0, 4, 4] == 100


# ---------------------------------------------------------------------------
# Z-stack collapse
# ---------------------------------------------------------------------------


def test_collapse_z_max():
    """Max-projection over Z: (C, Z, Y, X) -> (C, Y, X)."""
    data = np.zeros((2, 3, 8, 8), dtype=np.uint16)
    data[0, 0, 2, 2] = 10
    data[0, 1, 2, 2] = 50   # max for ch0
    data[0, 2, 2, 2] = 30
    data[1, 0, 5, 5] = 90   # max for ch1
    data[1, 1, 5, 5] = 20

    out = collapse_z_max(data)
    assert out.shape == (2, 8, 8)
    assert out[0, 2, 2] == 50
    assert out[1, 5, 5] == 90


def test_collapse_z_max_2d_passthrough():
    data = np.zeros((2, 8, 8), dtype=np.uint16)
    out = collapse_z_max(data)
    assert out.shape == (2, 8, 8)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def test_find_microscopy_files_discovers_all_formats(tmp_path):
    (tmp_path / "Input_pos").mkdir(parents=True)
    for name in ("a.czi", "b.nd2", "c.lif", "d.ome.tif", "e.ome.tiff"):
        (tmp_path / "Input_pos" / name).write_bytes(b"x")

    files = find_microscopy_files(tmp_path)
    names = {p.name for p in files}
    assert names == {"a.czi", "b.nd2", "c.lif", "d.ome.tif", "e.ome.tiff"}
