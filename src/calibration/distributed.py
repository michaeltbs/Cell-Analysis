"""
src/calibration/distributed.py — Cellpose distributed_eval wrapper for large images.

Official Cellpose solution for big data (larger-than-memory images / z-stacks).
See: https://cellpose.readthedocs.io/en/latest/distributed.html

Usage: converts a big TIFF to a zarr array, runs distributed_eval in blocks,
and returns the segmented labels (zarr path) + bounding boxes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def tiff_to_zarr(tiff_path: str, zarr_path: str, blocksize: int = 512) -> str:
    """
    Convert a (possibly huge) TIFF into a chunked zarr array for distributed eval.
    Reads the image header to get shape without loading everything.
    """
    import tifffile
    import zarr

    src = Path(tiff_path)
    out = Path(zarr_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # read header info
    with tifffile.TiffFile(str(src)) as tf:
        page = tf.pages[0]
        shape = page.shape
        dtype = page.dtype
        ndim = len(shape)

    # For 2D: (Y, X) -> store as (1, Y, X) 3D block-compatible array
    store = zarr.open(str(out), mode="w", shape=shape, dtype=dtype, chunks=(blocksize, blocksize))

    # stream in chunks
    with tifffile.TiffFile(str(src)) as tf:
        for page_idx, page in enumerate(tf.pages):
            arr = page.asarray()
            store[page_idx : page_idx + 1] = arr

    return str(out)


def run_distributed_segmentation(
    input_path: str,
    output_zarr: str,
    model_type: str = "cyto3",
    use_gpu: bool = True,
    blocksize: int = 256,
    overlap: int = 60,
    diameter: Optional[float] = None,
    z_axis: Optional[int] = None,
    channel_axis: Optional[int] = None,
    do_3D: bool = False,
) -> Dict[str, Any]:
    """
    Run distributed_eval on a large image (tiff or zarr).

    Args:
        input_path: path to .tif/.tiff/.zarr
        output_zarr: where segment labels zarr will be written
        model_type: cellpose model (cyto3, cpsam_v2, ...)
        use_gpu: gpu flag
        blocksize: block size for distributed eval
        overlap: overlap between blocks
        diameter: cell diameter
        z_axis: Z axis index for 3D
        channel_axis: channel axis index
        do_3D: run in 3D

    Returns:
        dict with segments_zarr, boxes, duration
    """
    from cellpose.contrib.distributed_segmentation import distributed_eval

    start = time.time()
    src = Path(input_path)

    # zarr input if already zarr, else convert
    if src.suffix.lower() == ".zarr" or src.is_dir():
        input_zarr = str(src)
    else:
        tmp_zarr = str(src.with_suffix(src.suffix + ".input.zarr"))
        input_zarr = tiff_to_zarr(str(src), tmp_zarr, blocksize=blocksize)

    model_kwargs = {"gpu": bool(use_gpu), "model_type": model_type}
    eval_kwargs: Dict[str, Any] = {}
    if diameter:
        eval_kwargs["diameter"] = diameter
    if z_axis is not None:
        eval_kwargs["z_axis"] = z_axis
    if channel_axis is not None:
        eval_kwargs["channel_axis"] = channel_axis
    eval_kwargs["do_3D"] = do_3D

    cluster_kwargs = {
        "n_workers": 1,
        "ncpus": 4,
        "memory_limit": "16GB",
        "threads_per_worker": 1,
    }

    segments, boxes = distributed_eval(
        input_zarr=input_zarr,
        blocksize=tuple([blocksize] * (3 if do_3D else 2)),
        write_path=output_zarr,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=cluster_kwargs,
    )

    return {
        "segments_zarr": output_zarr,
        "n_boxes": len(boxes) if boxes else 0,
        "duration_s": int(time.time() - start),
    }
