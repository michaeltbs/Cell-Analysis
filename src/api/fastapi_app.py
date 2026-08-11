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

import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.pipeline_service import run_detection, run_coexpression, run_full_pipeline
from src.api.jobs import manager
from src.api.upload_service import handle_upload
from src.config.naming import NamingConfig
import yaml

app = FastAPI(title="Cell Analysis API", version="0.1.0")


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


@app.post("/pipeline/detection", response_model=PipelineResponse)
def detection(req: DetectionRequest) -> PipelineResponse:
    try:
        result = run_detection(
            cfg_path=req.config_path,
            output_root=req.output_root,
            save_masks=req.save_masks,
            create_overlays=req.create_overlays,
            test_mode=req.test_mode,
            test_samples=req.test_samples,
            test_seed=req.test_seed,
        )
        return PipelineResponse(success=True, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/coexpression", response_model=PipelineResponse)
def coexpression(req: CoexpressionRequest) -> PipelineResponse:
    try:
        result = run_coexpression(req.config_path)
        return PipelineResponse(success=True, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/full", response_model=PipelineResponse)
def full_pipeline(req: FullPipelineRequest) -> PipelineResponse:
    try:
        result = run_full_pipeline(
            det_cfg_path=req.det_config_path,
            coexpr_cfg_path=req.coexpr_cfg_path,
            save_masks=req.save_masks,
        )
        return PipelineResponse(success=True, result=result)
    except Exception as e:
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
        det_cfg = _build_det_cfg(
            paths,
            naming,
            model_name=model_name,
            use_gpu=use_gpu.lower() in ("true", "1", "yes"),
            resize_max=resize_max,
        )
        coexpr_cfg = _build_coexpr_cfg(paths, naming)

        def _run(_job):
            _job.log.append("Running detection...")
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
