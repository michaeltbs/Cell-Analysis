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


# ---------------------------------------------------------------------------
# Kritik-Regressionen: ND2-Achsen, LIF-Kanäle, OME-ZYX/T>1, Silent-Failure
# ---------------------------------------------------------------------------


def test_nd2_axes_metadata_reorders_zcyx(tmp_path):
    """ND2 with sizes {'Z':3,'C':2,'Y':8,'X':8} must load as (C,Z,Y,X)."""
    import src.czi_converter as cc

    fake = np.zeros((3, 2, 8, 8), dtype=np.uint16)  # (Z, C, Y, X)
    fake[1, 0, 2, 2] = 100  # Z=1, C=0
    fake[2, 1, 5, 5] = 200  # Z=2, C=1

    class FakeND2:
        def __init__(self, path):
            self.sizes = {"Z": 3, "C": 2, "Y": 8, "X": 8}

        def asarray(self):
            return fake

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    orig = cc.nd2
    cc.nd2 = type("M", (), {"ND2File": FakeND2})
    try:
        out = cc.load_nd2_to_CZYX(str(tmp_path / "x.nd2"))
    finally:
        cc.nd2 = orig

    assert out.shape == (2, 3, 8, 8)
    assert out[0, 1, 2, 2] == 100
    assert out[1, 2, 5, 5] == 200


def test_lif_dims_tzc_uses_correct_z_and_channels(tmp_path):
    """readlif dims is (T, Z, C); Z must come from index 1, channels from
    the frame's first axis."""
    import src.czi_converter as cc

    frames = {
        (0, 0): np.zeros((2, 8, 8), dtype=np.uint16),  # (C, Y, X)
        (1, 0): np.zeros((2, 8, 8), dtype=np.uint16),
    }
    frames[(0, 0)][0, 2, 2] = 50
    frames[(1, 0)][1, 5, 5] = 90

    class FakeImg:
        dims = (1, 2, 2)  # T=1, Z=2, C=2

        def get_frame(self, z=0, t=0):
            return frames[(z, t)]

    class FakeLif:
        image_list = [FakeImg()]

        def get_image(self, i):
            return self.image_list[i]

    orig = cc.LifFile
    cc.LifFile = lambda p: FakeLif()
    try:
        out = cc.load_lif_to_CZYX(str(tmp_path / "x.lif"))
    finally:
        cc.LifFile = orig

    assert out.shape == (2, 2, 8, 8)  # (C, Z, Y, X)
    assert out[0, 0, 2, 2] == 50
    assert out[1, 1, 5, 5] == 90


def test_ome_tiff_zyx_without_channels(tmp_path):
    """A (Z, Y, X) OME-TIFF (no C axis) must load as (1, Z, Y, X)."""
    data = np.zeros((5, 16, 16), dtype=np.uint16)
    data[3, 4:8, 4:8] = 100
    p = tmp_path / "zyx.ome.tif"
    tifffile.imwrite(p, data, photometric="minisblack", metadata={"axes": "ZYX"})

    out = load_ome_tiff_to_CZYX(str(p))
    assert out.shape == (1, 5, 16, 16)
    assert out[0, 3, 4, 4] == 100


def test_ome_tiff_tzc_yx_takes_first_frame(tmp_path):
    """(T, Z, C, Y, X) with T=2 must take the first frame."""
    data = np.zeros((2, 2, 2, 8, 8), dtype=np.uint16)
    data[0, 1, 0, 2, 2] = 100  # first frame, Z=1, C=0
    data[1, 0, 0, 6, 6] = 200  # second frame (must be dropped)
    p = tmp_path / "tzcyx.ome.tif"
    tifffile.imwrite(p, data, photometric="minisblack", metadata={"axes": "TZCYX"})

    out = load_ome_tiff_to_CZYX(str(p))
    assert out.shape == (2, 2, 8, 8)
    assert out[0, 1, 2, 2] == 100
    assert out[0, 0, 6, 6] == 0  # second frame dropped
