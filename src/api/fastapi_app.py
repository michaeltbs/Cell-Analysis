"""
src/api/fastapi_app.py — FastAPI backend for the cell analysis pipeline.

Can be started standalone via:
    python -m src.api.fastapi_app

Endpoints:
    GET  /health                  -> health check
    GET  /version                 -> version info
    POST /pipeline/detection      -> run detection
    POST /pipeline/coexpression   -> run co-expression
    POST /pipeline/full           -> detection + co-expression
    POST /jobs/upload             -> upload files and enqueue full pipeline
    GET  /jobs/{job_id}           -> get job status/result
    GET  /jobs                    -> list recent jobs
    GET  /jobs/{job_id}/download  -> download a result file from job workspace
"""
from __future__ import annotations

import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.logging_setup import get_logger
from src.api.pipeline_service import run_detection, run_coexpression, run_full_pipeline
from src.api.jobs import manager
from src.api.upload_service import handle_upload
from src.api.config_models import DetectionConfig, CoexprConfig
from src.config.naming import NamingConfig
import yaml

logger = get_logger(__name__)

app = FastAPI(title="Cell Analysis API", version="0.1.0")

# Allow dashboard served from anywhere (HF Spaces, file://, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionRequest(BaseModel):
    config_path: str
    output_root: Optional[str] = None
    save_masks: bool = True
    create_overlays: Optional[bool] = None
    test_mode: bool = False
    test_samples: Optional[int] = None
    test_seed: Optional[int] = None


class CoexpressionRequest(BaseModel):
    config_path: str


class FullPipelineRequest(BaseModel):
    det_config_path: str
    coexpr_config_path: Optional[str] = None
    save_masks: bool = True


class PipelineResponse(BaseModel):
    success: bool
    result: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, str]:
    return {"version": "0.1.0", "backend": "fastapi"}


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard UI."""
    dashboard = Path(__file__).resolve().parents[2] / "dashboard" / "index.html"
    if dashboard.exists():
        return FileResponse(dashboard, media_type="text/html")
    return FileResponse(Path(__file__).resolve().parents[2] / "README.md", media_type="text/markdown")


@app.post("/pipeline/detection", response_model=PipelineResponse)
def detection(req: DetectionRequest) -> PipelineResponse:
    try:
        # Validate config before running
        cfg_dict = yaml.safe_load(Path(req.config_path).read_text(encoding="utf-8")) or {}
        DetectionConfig.from_dict(cfg_dict)
        result = run_detection(
            cfg_path=req.config_path,
            output_root=req.output_root,
            save_masks=req.save_masks,
            create_overlays=req.create_overlays,
            test_mode=req.test_mode,
            test_samples=req.test_samples,
            test_seed=req.test_seed,
        )
        logger.info("Detection finished: %s", result.get("master_csv"))
        return PipelineResponse(success=True, result=result)
    except Exception as e:
        logger.error("Detection failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/coexpression", response_model=PipelineResponse)
def coexpression(req: CoexpressionRequest) -> PipelineResponse:
    try:
        cfg_dict = yaml.safe_load(Path(req.config_path).read_text(encoding="utf-8")) or {}
        CoexprConfig.from_dict(cfg_dict)
        result = run_coexpression(req.config_path)
        logger.info("Co-expression finished: %s", result.get("output_dir"))
        return PipelineResponse(success=True, result=result)
    except Exception as e:
        logger.error("Co-expression failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/full", response_model=PipelineResponse)
def full_pipeline(req: FullPipelineRequest) -> PipelineResponse:
    try:
        result = run_full_pipeline(
            det_cfg_path=req.det_config_path,
            coexpr_cfg_path=req.coexpr_config_path,
            save_masks=req.save_masks,
        )
        logger.info("Full pipeline finished")
        return PipelineResponse(success=True, result=result)
    except Exception as e:
        logger.error("Full pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _build_det_cfg(
    paths: Dict[str, str],
    naming: NamingConfig,
    model_name: str = "cyto2",
    use_gpu: bool = True,
    resize_max: int = 2048,
) -> str:
    workspace = Path(paths["workspace"])
    cfg = {
        "paths": {
            "input_pos": paths["input_pos"],
            "input_neg": paths["input_neg"],
            "output_root": paths["output_root"],
        },
        "cellpose": {
            "model_name": model_name,
            "use_gpu": use_gpu,
            "diameter": 12,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "save_masks": False,
            "split_regions": False,
            "channel": 0,
            "batch_size": 1,
            "resize_max": resize_max,
        },
        "filters": {
            "min_area": 10,
            "max_area": 32000,
            "min_circularity": 0.0,
            "max_circularity": 1.0,
        },
        "outputs": {
            "base_name": "analysis_results",
            "create_overlays": False,
        },
        "analysis": {"enable_anova": False},
        "testing": {"enabled": False},
    }
    p = workspace / "det_config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


def _build_coexpr_cfg(paths: Dict[str, str], naming: NamingConfig) -> str:
    workspace = Path(paths["workspace"])
    cfg = {
        "input_dir": paths["output_root"],
        "output_dir": str(Path(paths["output_root"]) / "Coex"),
        "channel_config": [
            {
                "index": idx,
                "name": naming.channel_name(idx),
                "enabled": True,
                "color": color,
            }
            for idx, color in [(0, "#ff0000"), (1, "#00ff00"), (2, "#0000ff")]
        ],
        "coexpr_mode": "overlap",
        "overlap_threshold": 30,
        "centroid_overlap_fraction": 0.5,
        "selected_combos": [[0, 1], [0, 2], [1, 2], [0, 1, 2]],
        "combos_enabled": [2, 3],
        "use_masks": False,
        "fallback_from_overlays": True,
        "save_figure": False,
        "test_mode": {"type": "off"},
    }
    p = workspace / "coexpr_config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


@app.post("/jobs/upload")
def upload(
    files: List[UploadFile] = File(...),
    channels: str = Form("0,1,2,3"),
    model_name: str = Form("cyto2"),
    use_gpu: str = Form("true"),
    channel_names: str = Form(""),
    resize_max: int = Form(2048),
    test_mode: str = Form("false"),
) -> Dict[str, Any]:
    job = manager.create("upload+full_pipeline")
    try:
        parsed_channels = [int(c.strip()) for c in channels.split(",") if c.strip().isdigit()]
        if not parsed_channels:
            parsed_channels = [0, 1, 2, 3]

        naming = NamingConfig()
        if channel_names.strip():
            names = {i: n.strip() for i, n in enumerate(channel_names.split(",")) if n.strip()}
            naming.channel_names.update(names)

        file_tuples = []
        for f in files:
            content = f.file.read()
            file_tuples.append((f.filename, content))

        paths = handle_upload(file_tuples, job.id, parsed_channels)
        coexpr_cfg = _build_coexpr_cfg(paths, naming)
        test_enabled = test_mode.lower() in ("true", "1", "yes")

        def _run(_job):
            _job.log.append("Running detection...")
            if test_enabled:
                from src.api.test_segment import threshold_segment
                det_result = threshold_segment(paths["input_tiffs"], paths["output_root"])
            else:
                det_cfg = _build_det_cfg(
                    paths,
                    naming,
                    model_name=model_name,
                    use_gpu=use_gpu.lower() in ("true", "1", "yes"),
                    resize_max=resize_max,
                )
                det_result = run_detection(det_cfg, save_masks=False, create_overlays=False)
            _job.log.append("Running co-expression...")
            coexpr_result = run_coexpression(coexpr_cfg)
            return {
                "detection": det_result,
                "coexpression": coexpr_result,
                "workspace": paths["workspace"],
                "naming": naming.channel_names,
            }

        manager.submit(job, _run)
        return {"success": True, "job_id": job.id, "status": job.status.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/jobs")
def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    return manager.list_jobs(limit=limit)


@app.post("/calibration/profile")
async def calibration_profile(
    files: List[UploadFile] = File(...),
    channels: str = Form("0"),
    test_mode: str = Form("true"),
    model_name: str = Form("cyto2"),
    use_gpu: str = Form("true"),
) -> Dict[str, Any]:
    """Upload 1-3 images and get an intensity profile + diagnosis of the detection."""
    job = manager.create("calibration_profile")
    try:
        parsed_channels = [int(c.strip()) for c in channels.split(",") if c.strip().isdigit()]
        ch = parsed_channels[0] if parsed_channels else 0
        file_tuples: List[tuple[str, bytes]] = []
        for f in files:
            file_tuples.append(((f.filename or "upload.tif"), f.file.read()))
        paths = handle_upload(file_tuples, job.id, parsed_channels or [0, 1, 2, 3])

        def _run(_job):
            from src.api.test_segment import threshold_segment
            _job.log.append("Segmenting images...")
            results = threshold_segment(paths["input_tiffs"], paths["output_root"])
            profiles = []
            for cond, rows in results.items():
                for row in rows[:200]:
                    profiles.append({
                        "filename": row["filename"],
                        "condition": row["condition"],
                        "mean_intensity": row.get("mean_intensity", 0),
                        "area": row.get("area", 0),
                    })
            # group by condition, compute histogram
            conditions = {}
            for p in profiles:
                conditions.setdefault(p["condition"], {"cells": []})["cells"].append(p)
            out = {}
            for cond, data in conditions.items():
                ints = np.array([c["mean_intensity"] for c in data["cells"]], dtype=np.float64) if data["cells"] else np.array([])
                out[cond] = {
                    "cell_count": len(data["cells"]),
                    "mean_intensity": float(ints.mean()) if ints.size else 0.0,
                    "min_intensity": float(ints.min()) if ints.size else 0.0,
                    "max_intensity": float(ints.max()) if ints.size else 0.0,
                    "p10": float(np.percentile(ints, 10)) if ints.size else 0.0,
                    "p90": float(np.percentile(ints, 90)) if ints.size else 0.0,
                    "cells": data["cells"][:50],
                }
            _job.log.append("Profile complete")
            return {"profiles": out, "workspace": paths["workspace"]}

        manager.submit(job, _run)
        return {"success": True, "job_id": job.id, "status": job.status.value}
    except Exception as e:
        logger.error("Calibration profile failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calibration/optimize")
async def calibration_optimize(
    files: List[UploadFile] = File(...),
    expected_counts: str = Form(""),
    channels: str = Form("0"),
    model_name: str = Form("cyto2"),
    use_gpu: str = Form("true"),
    fast: str = Form("true"),
) -> Dict[str, Any]:
    """
    Upload images + expected cell counts (comma separated) and sweep
    detection params to find the best F1/MAE balance.
    """
    job = manager.create("calibration_optimize")
    try:
        parsed_channels = [int(c.strip()) for c in channels.split(",") if c.strip().isdigit()]
        ch = parsed_channels[0] if parsed_channels else 0
        counts = [int(c.strip()) for c in expected_counts.split(",") if c.strip().isdigit()]
        file_tuples: List[tuple[str, bytes]] = []
        for f in files:
            file_tuples.append(((f.filename or "upload.tif"), f.file.read()))
        if len(file_tuples) != len(counts):
            raise HTTPException(status_code=400, detail="expected_counts must have one value per file")
        paths = handle_upload(file_tuples, job.id, parsed_channels or [0, 1, 2, 3])

        # build images_with_counts from uploaded files
        images_with_counts = []
        input_tiffs = Path(paths["input_tiffs"])
        for (fname, _), count in zip(file_tuples, counts):
            target = input_tiffs / "Input_pos" / fname
            if not target.exists():
                target = input_tiffs / "Input_neg" / fname
            images_with_counts.append({"path": str(target), "expected": count})

        def _run(_job):
            from src.calibration.optimizer import run_optimization_sweep
            _job.log.append("Sweeping parameters...")
            result = run_optimization_sweep(
                images_with_counts,
                model_name=model_name,
                use_gpu=use_gpu.lower() in ("true", "1", "yes"),
                channel=ch,
                fast=fast.lower() in ("true", "1", "yes"),
            )
            _job.log.append(f"Optimization done: best score={result['best'].get('score', 0):.3f}")
            return result

        manager.submit(job, _run)
        return {"success": True, "job_id": job.id, "status": job.status.value}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Calibration optimize failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calibration/train")
async def calibration_train(
    train_dir: str = Form(...),
    test_dir: str = Form(""),
    model_name: str = Form("cpsam"),
    n_epochs: int = Form(100),
    learning_rate: float = Form(1e-5),
    use_gpu: str = Form("true"),
    model_name_out: str = Form("cellpose_custom"),
) -> Dict[str, Any]:
    """Validate + start Cellpose fine-tuning on a local training directory."""
    job = manager.create("calibration_train")

    def _run(_job):
        from src.calibration.train import validate_training_data, run_finetune
        _job.log.append("Validating training data...")
        validation = validate_training_data(train_dir)
        if not validation["valid"]:
            _job.log.append(f"Invalid training data: {validation['missing_masks'][:5]}")
            raise ValueError(f"Invalid training data. Missing masks: {validation['missing_masks'][:5]}")
        _job.log.append(f"{validation['n_pairs']} image/mask pairs, {validation['n_masks_total']} masks total")
        _job.log.append(f"Starting fine-tune: {model_name} -> {model_name_out} ({n_epochs} epochs)")
        result = run_finetune(
            train_dir=train_dir,
            test_dir=test_dir or None,
            model_name=model_name,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            use_gpu=use_gpu.lower() in ("true", "1", "yes"),
            model_name_out=model_name_out,
        )
        if result.get("success"):
            _job.log.append(f"Training complete -> {result['model_path']}")
        else:
            _job.log.append(f"Training failed: {result.get('error', result.get('stdout_tail', '')[-500:])}")
        return result

    manager.submit(job, _run)
    return {"success": True, "job_id": job.id, "status": job.status.value}


@app.get("/jobs/{job_id}/download")
def download(job_id: str, file: str) -> FileResponse:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result or "workspace" not in (job.result or {}):
        raise HTTPException(status_code=400, detail="Job has no workspace yet or not finished")

    workspace = Path(job.result["workspace"])
    target = (workspace / file).resolve()
    # security: prevent escaping workspace
    if not str(target).startswith(str(workspace.resolve())):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File {file} not found")

    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type, filename=target.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
