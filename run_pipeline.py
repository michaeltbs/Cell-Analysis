
#!/usr/bin/env python3
"""
run_pipeline.py – CLI wrapper for detection + co-expression summary.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
import yaml

from src.analysis import (
    calculate_animal_averages,
    calculate_condition_averages,
    create_summary_report,
)
from src.batch_segment import segment_dirs
from src.config_archiver import save_config_snapshot


def _print_env() -> None:
    try:
        version = torch.__version__
    except Exception:
        version = "<unknown>"
    print(f"[env] torch={version}")
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    print(f"[env] cuda_available={cuda_ok}")
    if cuda_ok:
        try:
            print(f"[env] device={torch.cuda.get_device_name(0)}")
        except Exception:
            pass


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cell Analysis Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--no-masks", action="store_true", help="Do not save binary masks")
    parser.add_argument("--no-summary", action="store_true", help="Skip markdown summary generation")
    parser.add_argument("--create-overlays", action="store_true", help="Force overlay generation")
    parser.add_argument("--no-overlays", action="store_true", help="Disable overlay generation")
    parser.add_argument("--test", action="store_true", help="Enable test mode (limit samples)")
    parser.add_argument("--test-samples", type=int, default=None, help="Samples per channel in test mode")
    parser.add_argument("--test-seed", type=int, default=None, help="Optional seed for test sampling")
    return parser


def _write_temp_cfg(cfg: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    return Path(tmp.name)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _load_cfg(cfg_path)
    _print_env()

    cellpose_cfg = cfg.get("cellpose", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    final_save_masks = bool(cellpose_cfg.get("save_masks", True)) and not args.no_masks
    split_regions = bool(cellpose_cfg.get("split_regions", False))

    if args.no_overlays:
        create_overlays = False
    elif args.create_overlays:
        create_overlays = True
    elif not final_save_masks:
        create_overlays = True
    else:
        create_overlays = False

    testing_cfg = cfg.get("testing", {}) or {}
    if args.test:
        testing_cfg["enabled"] = True
    if args.test_samples is not None:
        testing_cfg["samples_per_channel"] = max(1, int(args.test_samples))
    testing_cfg.setdefault("enabled", False)
    testing_cfg.setdefault("samples_per_channel", 0)
    if args.test_seed is not None:
        testing_cfg["seed"] = int(args.test_seed)

    print(f"[cfg] save_masks={final_save_masks}")
    print(f"[cfg] split_regions={split_regions}")
    print(f"[cfg] create_overlays={create_overlays}")
    if testing_cfg.get("enabled") and testing_cfg.get("samples_per_channel", 0) > 0:
        seed_info = f" seed={testing_cfg.get('seed')}" if testing_cfg.get('seed') is not None else ''
        print(f"[cfg] test_mode samples_per_channel={testing_cfg['samples_per_channel']}{seed_info}")

    run_cfg = dict(cfg)
    run_cfg.setdefault("paths", {}).update({
        "input_pos": paths_cfg.get("input_pos", paths_cfg.get("input_old", "")),
        "input_neg": paths_cfg.get("input_neg", paths_cfg.get("input_adult", "")),
        "output_root": paths_cfg.get("output_root", "./results_det"),
    })
    run_cfg.setdefault("cellpose", {}).update({
        "save_masks": final_save_masks,
    })
    run_cfg.setdefault("outputs", {})
    run_cfg["testing"] = testing_cfg

    if create_overlays:
        run_cfg.setdefault("outputs", {})["create_overlays"] = True
    if args.no_overlays:
        run_cfg.setdefault("outputs", {})["create_overlays"] = False

    tmp_cfg_path = _write_temp_cfg(run_cfg)
    try:
        # Save config snapshot before starting analysis
        output_root = Path(run_cfg["paths"]["output_root"]).resolve()
        runtime_params = {
            "command_line_args": vars(args),
            "save_masks": final_save_masks,
            "split_regions": split_regions,
            "create_overlays": create_overlays,
            "testing_config": testing_cfg,
            "config_file": str(cfg_path),
        }
        save_config_snapshot(
            output_dir=output_root,
            config_file=cfg_path,
            config_dict=run_cfg,
            runtime_params=runtime_params,
            snapshot_name="pipeline_config",
        )
        
        segment_dirs(tmp_cfg_path)
    finally:
        try:
            os.remove(tmp_cfg_path)
        except Exception:
            pass

    master_csv = Path(run_cfg["paths"]["output_root"]) / "pos" / "All_Counts_Master.csv"
    if not master_csv.exists():
        master_csv = Path(run_cfg["paths"]["output_root"]) / "All_Counts_Master.csv"
    master_csv = master_csv.resolve()

    out_dir = master_csv.parent
    base_name = cfg.get("outputs", {}).get("base_name", "analysis_resSults")

    animal_csv = out_dir / f"{base_name}_animal_averages.csv"
    cond_csv = out_dir / f"{base_name}_condition_averages.csv"

    animal_df = calculate_animal_averages(str(master_csv), str(animal_csv))
    condition_df = calculate_condition_averages(animal_df, str(cond_csv))

    if not args.no_summary:
        summary_path = out_dir / f"{base_name}_summary.md"
        create_summary_report(animal_df, condition_df, str(summary_path))
        print(f"[out] summary written to {summary_path}")

    print("[done] pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
