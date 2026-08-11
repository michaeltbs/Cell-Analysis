"""
tests/test_fastapi_app.py — lightweight tests for FastAPI endpoints.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import yaml

from src.api.fastapi_app import app
from tests.fixtures.synthetic_cells import create_test_dataset
from tests.test_pipeline_service import _write_det_config, _write_coexpr_config


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["backend"] == "fastapi"


def test_detection_endpoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        create_test_dataset(tmp_path, n_images=1, seed_offset=400)
        input_pos = tmp_path / "data" / "processed_tiffs" / "Input_pos"
        input_neg = tmp_path / "data" / "processed_tiffs" / "Input_neg"
        output_root = tmp_path / "results" / "final"
        cfg = _write_det_config(tmp_path, input_pos, input_neg, output_root)

        r = client.post(
            "/pipeline/detection",
            json={
                "config_path": str(cfg),
                "test_mode": True,
                "test_samples": 1,
                "save_masks": False,
                "create_overlays": False,
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["success"]
        assert Path(data["result"]["master_csv"]).exists()


def test_full_pipeline_endpoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        create_test_dataset(tmp_path, n_images=1, seed_offset=500)
        input_pos = tmp_path / "data" / "processed_tiffs" / "Input_pos"
        input_neg = tmp_path / "data" / "processed_tiffs" / "Input_neg"
        output_root = tmp_path / "results" / "final"
        det_cfg = _write_det_config(tmp_path, input_pos, input_neg, output_root)
        coexpr_cfg = _write_coexpr_config(output_root)

        r = client.post(
            "/pipeline/full",
            json={
                "det_config_path": str(det_cfg),
                "coexpr_config_path": str(coexpr_cfg),
                "save_masks": False,
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["success"]
        assert Path(data["result"]["master_csv"]).exists()


if __name__ == "__main__":
    test_health()
    test_version()
    test_detection_endpoint()
    test_full_pipeline_endpoint()
    print("✅ FastAPI tests passed")
