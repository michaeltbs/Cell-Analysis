"""
src/api/fastapi_app.py — FastAPI backend for the cell analysis pipeline.

Can be started standalone via:
    python -m src.api.fastapi_app

Endpoints:
    POST /pipeline/detection      -> run detection
    POST /pipeline/coexpression   -> run co-expression
    POST /pipeline/full           -> detection + co-expression
    GET  /health                  -> health check
    GET  /version                 -> version info
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.pipeline_service import run_detection, run_coexpression, run_full_pipeline

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
            coexpr_cfg_path=req.coexpr_config_path,
            save_masks=req.save_masks,
        )
        return PipelineResponse(success=True, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
