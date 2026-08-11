"""
src/api/upload_service.py — handles file uploads and prepares configs for HF Space.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.czi_converter import CziConfig, run_czi_conversion


def handle_upload(
    files: List[Tuple[str, bytes]],
    job_id: str,
    channels: List[int] | None = None,
    target_size: int = 2048,
) -> Dict[str, str]:
    """
    Save uploaded files, convert CZI if needed, and return input/output paths.

    Args:
        files: list of (filename, content) tuples
        job_id: unique job id used for workspace
        channels: channels to extract from CZI
        target_size: max side for TIFF conversion

    Returns:
        dict with input_pos, input_neg, output_root paths
    """
    workspace = Path(tempfile.gettempdir()) / "cell_analysis" / job_id
    input_raw = workspace / "input_raw"
    input_tiffs = workspace / "input_tiffs"
    output_root = workspace / "results"
    input_raw.mkdir(parents=True, exist_ok=True)
    input_tiffs.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    pos_dir = input_tiffs / "Input_pos"
    neg_dir = input_tiffs / "Input_neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    czi_files: List[Path] = []
    tiff_files: List[Tuple[str, Path]] = []

    for filename, content in files:
        p = input_raw / filename
        p.write_bytes(content)
        suffix = p.suffix.lower()
        if suffix in (".czi",):
            czi_files.append(p)
        elif suffix in (".tif", ".tiff"):
            # separate pos/neg by filename if present
            if "neg" in filename.lower():
                target = neg_dir / filename
            else:
                target = pos_dir / filename
            shutil.copy2(p, target)
            tiff_files.append((filename, target))

    # convert CZI files
    if czi_files:
        cfg = CziConfig(
            input_root=str(input_raw),
            output_base=str(input_tiffs),
            channels=channels or [0, 1, 2, 3],
            target_size=target_size,
            overwrite=True,
            save_per_channel=False,
            save_stack_tiff=True,
            save_color_composite=False,
            save_rgb_preview=False,
        )
        cfg_path = workspace / "czi_config.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({
                "input_root": cfg.input_root,
                "output_base": cfg.output_base,
                "channels": cfg.channels,
                "target_size": cfg.target_size,
                "overwrite": cfg.overwrite,
                "save_stack_tiff": cfg.save_stack_tiff,
                "save_per_channel": cfg.save_per_channel,
                "save_color_composite": cfg.save_color_composite,
                "save_rgb_preview": cfg.save_rgb_preview,
            }, f)
        run_czi_conversion(str(cfg_path))

    return {
        "workspace": str(workspace),
        "input_pos": str(pos_dir),
        "input_neg": str(neg_dir),
        "output_root": str(output_root),
    }
