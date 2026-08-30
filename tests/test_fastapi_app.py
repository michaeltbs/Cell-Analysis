"""
tests/test_fastapi_app.py — lightweight tests for FastAPI endpoints.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import tifffile
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


def _wait_for_job(client, job_id: str, timeout: float = 45.0) -> dict:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("success", "failed"):
            return j
        time.sleep(0.3)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def _synthetic_tif_bytes(tmp_path: Path, stem: str = "sample", n_channels: int = 3) -> bytes:
    """Small (n_channels, 64, 64) TIFF with a bright blob in each channel."""
    import io

    from tests.fixtures.synthetic_cells import create_synthetic_stack

    stack, _ = create_synthetic_stack(
        shape=(64, 64), n_cells=5, n_channels=n_channels, seed=777, output_path=None
    )
    buf = io.BytesIO()
    tifffile.imwrite(buf, stack, imagej=True)
    return buf.getvalue()


def test_upload_job_exports_expression_and_distance_maps():
    """Upload in test_mode runs the pipeline and must write expression CSVs
    + distance-map stats + heatmaps into the job workspace."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tif_bytes = _synthetic_tif_bytes(tmp_path)

        r = client.post(
            "/jobs/upload",
            files=[("files", ("sample.tif", tif_bytes, "image/tiff"))],
            data={
                "test_mode": "true",
                "channel_names": "PomC,Glp1r,Gal",
                "condition_names": "pos=KO,neg=WT",
                "channels": "0,1,2",
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        assert job_id

        job = _wait_for_job(client, job_id, timeout=60.0)
        assert job["status"] == "success", job.get("error")
        result = job["result"]

        assert result["expression"]["pos_csv"], "expression pos_csv missing"
        assert Path(result["expression"]["pos_csv"]).exists()
        assert result["distance_maps"]["stats_csv"], "distance stats csv missing"
        assert Path(result["distance_maps"]["stats_csv"]).exists()
        assert result["distance_maps"]["heatmaps"], "no heatmaps produced"

        # CSV content sanity
        import pandas as pd

        df = pd.read_csv(result["expression"]["pos_csv"])
        assert len(df) > 0
        assert "PomC_percent_positive" in df.columns

        # naming: condition labels flow into the CSVs
        assert result["naming"]["condition_names"] == {"pos": "KO", "neg": "WT"}
        assert set(df["condition_label"].unique()) == {"KO"}


def test_parse_kv_form():
    from src.api.fastapi_app import _parse_kv_form

    assert _parse_kv_form("pos=KO,neg=WT") == {"pos": "KO", "neg": "WT"}
    assert _parse_kv_form("") == {}
    # whitespace tolerant
    assert _parse_kv_form(" pos = KO , neg = WT ") == {"pos": "KO", "neg": "WT"}
    # keys without '=' get positional numbering
    assert _parse_kv_form("ARC,VMH") == {"0": "ARC", "1": "VMH"}


if __name__ == "__main__":
    test_health()
    test_version()
    test_detection_endpoint()
    test_full_pipeline_endpoint()
    print("✅ FastAPI tests passed")
