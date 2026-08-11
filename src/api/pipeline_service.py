"""
src/api/pipeline_service.py — headless service layer for the cell analysis pipeline.

This module exposes the core CZI → Detection → Co-Expression → Statistics flow
as plain Python functions with no Flask dependencies, so it can be reused by:
- a standalone CLI (run_pipeline.py)
- a Flask/FastAPI backend
- a future desktop/mobile app via HTTP API
"""
from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.batch_segment import segment_dirs
from src.coexpression import main as coexpression_main
from src.config_archiver import save_config_snapshot
from src.analysis import (
    calculate_animal_averages,
    calculate_condition_averages,
    create_summary_report,
)
from src.anova_analysis import run_anova_analysis, run_mixed_effects_analysis


def _write_temp_cfg(cfg: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    return Path(tmp.name)


def run_detection(
    cfg_path: Path | str,
    output_root: Path | str | None = None,
    save_masks: bool = True,
    create_overlays: Optional[bool] = None,
    test_mode: bool = False,
    test_samples: Optional[int] = None,
    test_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the detection step (segment_dirs) using a YAML config.

    Returns:
        dict with keys: master_csv, output_root, animal_csv, condition_csv, summary_md
    """
    cfg_path = Path(cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    paths_cfg = cfg.get("paths", {}) or {}
    out_root = Path(output_root).resolve() if output_root else Path(
        paths_cfg.get("output_root", "./results_det")
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("paths", {}).update({
        "input_pos": paths_cfg.get("input_pos", paths_cfg.get("input_old", "")),
        "input_neg": paths_cfg.get("input_neg", paths_cfg.get("input_adult", "")),
        "output_root": str(out_root),
    })
    run_cfg.setdefault("cellpose", {})["save_masks"] = save_masks
    if create_overlays is not None:
        run_cfg.setdefault("outputs", {})["create_overlays"] = create_overlays

    testing_cfg = run_cfg.get("testing", {}) or {}
    if test_mode:
        testing_cfg["enabled"] = True
    if test_samples is not None:
        testing_cfg["samples_per_channel"] = max(1, int(test_samples))
    if test_seed is not None:
        testing_cfg["seed"] = int(test_seed)
    run_cfg["testing"] = testing_cfg

    runtime_params = {
        "save_masks": save_masks,
        "create_overlays": create_overlays,
        "testing_config": testing_cfg,
        "config_file": str(cfg_path),
    }
    save_config_snapshot(
        output_dir=out_root,
        config_file=cfg_path,
        config_dict=run_cfg,
        runtime_params=runtime_params,
        snapshot_name="pipeline_config",
    )

    tmp_cfg_path = _write_temp_cfg(run_cfg)
    try:
        segment_dirs(tmp_cfg_path)
    finally:
        try:
            os.remove(tmp_cfg_path)
        except Exception:
            pass

    master_csv = out_root / "pos" / "All_Counts_Master.csv"
    if not master_csv.exists():
        master_csv = out_root / "All_Counts_Master.csv"
    master_csv = master_csv.resolve()

    base_name = cfg.get("outputs", {}).get("base_name", "analysis_results")
    animal_csv = out_root / f"{base_name}_animal_averages.csv"
    cond_csv = out_root / f"{base_name}_condition_averages.csv"
    summary_md = out_root / f"{base_name}_summary.md"

    if master_csv.exists():
        animal_df = calculate_animal_averages(str(master_csv), str(animal_csv))
        condition_df = calculate_condition_averages(animal_df, str(cond_csv))
        create_summary_report(animal_df, condition_df, str(summary_md))

        if cfg.get("analysis", {}).get("enable_anova", True):
            try:
                run_anova_analysis(str(master_csv), str(out_root))
            except Exception as e:
                print(f"[warn] ANOVA analysis failed: {e}")
        if cfg.get("analysis", {}).get("enable_mixed_effects", False):
            try:
                run_mixed_effects_analysis(str(master_csv), str(out_root))
            except Exception as e:
                print(f"[warn] Mixed-effects analysis failed: {e}")

    return {
        "master_csv": str(master_csv),
        "output_root": str(out_root),
        "animal_csv": str(animal_csv),
        "condition_csv": str(cond_csv),
        "summary_md": str(summary_md),
    }


def run_coexpression(cfg_path: Path | str, force: bool = False) -> Dict[str, Any]:
    """Run co-expression analysis from a YAML config.

    If the master summary CSV already exists and is newer than all input
    files, the run is skipped (cached) unless force=True.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    output_dir = cfg.get("output_dir", cfg.get("paths", {}).get("output_dir", "./coexpr_out"))
    output_dir = Path(output_dir).resolve()

    summary = output_dir / "coexpr_sweep_summary.csv"
    if not force and summary.exists():
        input_dir = Path(cfg.get("input_dir", output_dir))
        newest_input = 0.0
        if input_dir.exists():
            for p in input_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png"):
                    newest_input = max(newest_input, p.stat().st_mtime)
        if newest_input <= summary.stat().st_mtime:
            print(f"[CACHE] Co-expression up-to-date, skipping ({summary})")
            return {"output_dir": str(output_dir), "cached": True}

    rc = coexpression_main(["--config", str(cfg_path)])
    if rc != 0:
        raise RuntimeError(f"Co-expression analysis failed with exit code {rc}")
    return {"output_dir": str(output_dir), "cached": False}


def run_full_pipeline(
    det_cfg_path: Path | str,
    coexpr_cfg_path: Path | str | None = None,
    save_masks: bool = True,
    run_anova: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end: detection + co-expression + statistics.

    Returns a combined result dict.
    """
    det_result = run_detection(det_cfg_path, save_masks=save_masks)
    coexpr_result = {}
    if coexpr_cfg_path:
        coexpr_result = run_coexpression(coexpr_cfg_path)
    return {**det_result, "coexpression": coexpr_result}
