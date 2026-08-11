"""
tests/test_naming.py — tests for universal naming config.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.naming import NamingConfig, apply_naming_to_csv_rows


def test_channel_naming():
    cfg = NamingConfig.from_dict({
        "channel_names": {0: "PomC", 1: "DREADD", 2: "Fos", 3: "Gal"}
    })
    assert cfg.channel_name(0) == "PomC"
    assert cfg.channel_name(3) == "Gal"
    assert cfg.channel_name(9) == "Channel_9"


def test_csv_row_naming():
    cfg = NamingConfig.from_dict({
        "channel_names": {0: "PomC", 1: "DREADD"},
        "condition_names": {"pos": "CNO", "neg": "Saline"},
    })
    rows = [
        {"channel": 0, "condition": "pos", "region": "Arc", "cell_count": 10},
        {"channel": 1, "condition": "neg", "region": "PVH", "cell_count": 5},
    ]
    out = apply_naming_to_csv_rows(rows, cfg)
    assert out[0]["channel_label"] == "PomC"
    assert out[0]["condition_label"] == "CNO"
    assert out[1]["channel_label"] == "DREADD"
    assert out[1]["condition_label"] == "Saline"


if __name__ == "__main__":
    test_channel_naming()
    test_csv_row_naming()
    print("✅ naming tests passed")
