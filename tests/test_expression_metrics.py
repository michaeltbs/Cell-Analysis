"""
tests/test_expression_metrics.py — tests for expression percentage + distance map.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.validation.expression import expression_percentage, receptor_distance_map


def test_expression_percentage():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True  # 20x20 = 400 positive out of 10000
    result = expression_percentage(mask)
    assert result["percent_positive"] == 4.0
    assert result["percent_negative"] == 96.0


def test_receptor_distance_map():
    cell = np.zeros((100, 100), dtype=bool)
    cell[45:55, 45:55] = True
    receptor = np.zeros((100, 100), dtype=bool)
    receptor[50, 50] = True  # inside cell
    receptor[70, 70] = True  # outside cell
    result = receptor_distance_map(receptor, cell)
    dists = result["receptor_distances"]
    assert dists.min() == 0.0
    assert dists.max() > 10.0


if __name__ == "__main__":
    test_expression_percentage()
    test_receptor_distance_map()
    print("✅ expression metrics tests passed")
