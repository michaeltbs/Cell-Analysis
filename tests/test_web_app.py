"""
tests/test_web_app.py — lightweight tests for the Flask blueprint app.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.web.app import create_app
from tests.fixtures.synthetic_cells import create_test_dataset
from tests.test_pipeline_service import _write_det_config


def test_health():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"


def test_detection_start():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        create_test_dataset(tmp_path, n_images=1, seed_offset=600)
        input_pos = tmp_path / "data" / "processed_tiffs" / "Input_pos"
        input_neg = tmp_path / "data" / "processed_tiffs" / "Input_neg"
        output_root = tmp_path / "results" / "final"
        cfg = _write_det_config(tmp_path, input_pos, input_neg, output_root)

        r = client.post(
            "/api/det/start",
            json={
                "config_path": str(cfg),
                "test_mode": True,
                "test_samples": 1,
                "save_masks": False,
            },
        )
        assert r.status_code == 200, r.data.decode()
        assert r.json["success"]


if __name__ == "__main__":
    test_health()
    test_detection_start()
    print("✅ Flask web app tests passed")
