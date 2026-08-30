"""
src/api/upload_service.py — handles file uploads and prepares configs for HF Space.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import yaml

from src.czi_converter import (
    CziConfig,
    run_czi_conversion,
    load_microscopy_to_CZYX,
    collapse_z_max,
    SUPPORTED_MICROSCOPY_EXTS,
)


def handle_upload(
    files: List[Tuple[str, bytes]],
    job_id: str,
    channels: List[int] | None = None,
    target_size: int | None = 2048,
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
    other_microscopy: List[Path] = []

    for filename, content in files:
        p = input_raw / filename
        p.write_bytes(content)
        suffix = p.suffix.lower()
        lower_name = p.name.lower()
        if suffix in (".czi",):
            czi_files.append(p)
        elif lower_name.endswith((".nd2", ".lif", ".ome.tif", ".ome.tiff")):
            other_microscopy.append(p)
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

    # convert ND2 / LIF / OME-TIFF: load -> (C,Z,Y,X) -> max-project -> write stack
    for p in other_microscopy:
        try:
            czxy = load_microscopy_to_CZYX(str(p))
            cyx = collapse_z_max(czxy)
            # subset channels
            if channels:
                keep = [c for c in channels if 0 <= c < cyx.shape[0]]
                if keep:
                    cyx = cyx[keep]
            # resize if requested
            if target_size and target_size > 0:
                from skimage.transform import resize

                h, w = cyx.shape[1:]
                scale = target_size / max(h, w)
                if scale != 1.0:
                    new_h, new_w = int(round(h * scale)), int(round(w * scale))
                    cyx = np.stack(
                        [resize(cyx[i], (new_h, new_w), preserve_range=True, anti_aliasing=True)
                         for i in range(cyx.shape[0])],
                        axis=0,
                    )
            # write as CYX stack TIFF into the pos/neg dir based on filename
            target_dir = neg_dir if "neg" in p.name.lower() else pos_dir
            # stem handling: .ome.tif/.ome.tiff are double extensions
            stem = p.name
            for ext in (".ome.tif", ".ome.tiff"):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            else:
                stem = p.stem
            out_name = stem + "_stack.tif"
            tifffile.imwrite(
                target_dir / out_name,
                np.ascontiguousarray(cyx),
                photometric="minisblack",
                metadata={"axes": "CYX"},
            )
            tiff_files.append((out_name, target_dir / out_name))
        except Exception as e:
            print(f"[WARN] Conversion failed for {p.name}: {e}")

    return {
        "workspace": str(workspace),
        "input_pos": str(pos_dir),
        "input_neg": str(neg_dir),
        "output_root": str(output_root),
        "input_tiffs": str(input_tiffs),
    }
