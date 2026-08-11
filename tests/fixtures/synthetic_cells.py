"""
tests/fixtures/synthetic_cells.py — generates small, reproducible TIFF stacks
that mimic PomC/Glp1r/Gal channel data with known cell positions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tiff


def _draw_gaussian_spot(img: np.ndarray, cy: int, cx: int, sigma: float, intensity: float) -> None:
    """Draw a soft circular cell onto a 2-D image."""
    y, x = np.ogrid[: img.shape[0], : img.shape[1]]
    dist_sq = ((y - cy) ** 2 + (x - cx) ** 2) / (sigma ** 2)
    spot = intensity * np.exp(-dist_sq)
    img[:] = np.maximum(img, spot)


def create_synthetic_stack(
    shape: Tuple[int, int] = (512, 512),
    n_cells: int = 40,
    n_channels: int = 3,
    seed: int = 42,
    output_path: Path | str | None = None,
    pos_shift: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Create a (C, Y, X) uint16 synthetic TIFF with `n_cells` randomly placed spots.

    Cells are placed consistently across channels with optional translation for one
    condition (e.g. pos vs neg) to simulate registration drift.

    Returns:
        stack: uint16 ndarray shape (C, Y, X)
        centers: list of (channel, cy, cx) for ground truth
    """
    rng = np.random.default_rng(seed)
    cy_min, cy_max = 30, shape[0] - 30
    cx_min, cx_max = 30, shape[1] - 30

    centers: List[Tuple[int, int, int]] = []
    stack = np.zeros((n_channels, *shape), dtype=np.float32)

    for i in range(n_cells):
        cy = int(rng.integers(cy_min, cy_max))
        cx = int(rng.integers(cx_min, cx_max))
        sigma = float(rng.uniform(3.5, 5.5))

        # channel participation per cell:
        # - channel 0: all cells
        # - channel 1: 70% of cells
        # - channel 2: 50% of cells (co-expression subset)
        in_ch0 = True
        in_ch1 = rng.random() < 0.70
        in_ch2 = rng.random() < 0.50

        channel_flags = [in_ch0, in_ch1, in_ch2][:n_channels]
        # pad if n_channels < 3
        channel_flags += [False] * (n_channels - len(channel_flags))

        for ch, present in enumerate(channel_flags):
            if not present:
                continue
            intensity = float(rng.uniform(0.6, 1.0))
            _draw_gaussian_spot(
                stack[ch],
                cy + pos_shift[0],
                cx + pos_shift[1],
                sigma,
                intensity,
            )
            centers.append((ch, cy + pos_shift[0], cx + pos_shift[1]))

    # add low background + poisson-like noise
    for ch in range(n_channels):
        bg = float(rng.uniform(0.02, 0.05))
        noise = rng.normal(0.0, 0.015, shape)
        stack[ch] = stack[ch] + bg + noise
        stack[ch] = np.clip(stack[ch], 0, None)

    # rescale to uint16
    max_val = stack.max()
    if max_val > 0:
        stack = stack / max_val * 65535
    stack = stack.astype(np.uint16)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(output_path, stack, imagej=True)
        # write ground-truth centers
        meta = {"shape": shape, "n_cells": n_cells, "n_channels": n_channels, "centers": centers}
        output_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    return stack, centers


def create_test_dataset(root: Path | str, n_images: int = 4, seed_offset: int = 0) -> Path:
    """
    Create a tiny pos/neg dataset under root/data/processed_tiffs/.
    Returns the root path.
    """
    root = Path(root)
    pos_dir = root / "data" / "processed_tiffs" / "Input_pos"
    neg_dir = root / "data" / "processed_tiffs" / "Input_neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        create_synthetic_stack(
            shape=(512, 512),
            n_cells=40,
            n_channels=3,
            seed=seed_offset + i,
            output_path=pos_dir / f"M{i:03d}_1.tiff",
            pos_shift=(0, 0),
        )
        create_synthetic_stack(
            shape=(512, 512),
            n_cells=40,
            n_channels=3,
            seed=seed_offset + i,
            output_path=neg_dir / f"M{i:03d}_1.tiff",
            pos_shift=(2, 2),  # small shift to simulate drift
        )
    return root


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_data"
    create_test_dataset(out, n_images=2)
    print(f"sample dataset written to {out}")
