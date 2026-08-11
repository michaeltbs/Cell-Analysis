"""
tests/test_config_models.py — tests for Pydantic config validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic import ValidationError

from src.api.config_models import DetectionConfig, CoexprConfig, CellposeConfig


def test_detection_config_valid():
    cfg = DetectionConfig.from_dict({
        "paths": {"input_pos": "a", "input_neg": "b", "output_root": "out"},
        "cellpose": {"model_name": "cyto3", "diameter": 20},
    })
    assert cfg.cellpose.model_name == "cyto3"
    assert cfg.cellpose.diameter == 20.0
    assert cfg.paths.input_pos == "a"


def test_detection_config_missing_paths():
    with pytest.raises(ValidationError):
        DetectionConfig.from_dict({"cellpose": {"model_name": "cyto2"}})


def test_cellpose_invalid_model_falls_back():
    cfg = CellposeConfig(model_name="does-not-exist")
    assert cfg.model_name == "cyto2"


def test_coexpr_invalid_mode_falls_back():
    cfg = CoexprConfig(input_dir="a", output_dir="b", coexpr_mode="nonsense")
    assert cfg.coexpr_mode == "overlap"


if __name__ == "__main__":
    test_detection_config_valid()
    test_detection_config_missing_paths()
    test_cellpose_invalid_model_falls_back()
    test_coexpr_invalid_mode_falls_back()
    print("✅ config_models tests passed")
