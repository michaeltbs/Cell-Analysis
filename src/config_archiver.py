#!/usr/bin/env python3
"""
config_archiver.py — Utility to save config snapshots for reproducibility

Automatically saves copies of:
- YAML config files
- Runtime parameters (command-line args, computed values)
- Timestamp and metadata

This ensures complete traceability of all analysis runs.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def save_config_snapshot(
    output_dir: str | Path,
    config_file: str | Path | None = None,
    config_dict: Optional[Dict[str, Any]] = None,
    runtime_params: Optional[Dict[str, Any]] = None,
    snapshot_name: str = "config_snapshot",
) -> Path:
    """
    Save a complete snapshot of configuration and runtime parameters.
    
    Args:
        output_dir: Directory where the snapshot will be saved
        config_file: Path to the original YAML config file (will be copied)
        config_dict: Dictionary with config data (will be saved as YAML)
        runtime_params: Dictionary with runtime parameters (args, computed values)
        snapshot_name: Base name for the snapshot files (default: "config_snapshot")
    
    Returns:
        Path to the created snapshot directory
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = output_path / f"{snapshot_name}_{timestamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original config file if provided
    if config_file is not None:
        config_path = Path(config_file)
        if config_path.exists():
            dest = snapshot_dir / f"original_{config_path.name}"
            shutil.copy2(config_path, dest)
            print(f"[config] Saved original config: {dest.relative_to(output_path)}")
    
    # Save config dictionary as YAML
    if config_dict is not None:
        config_yaml = snapshot_dir / "effective_config.yaml"
        with config_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"[config] Saved effective config: {config_yaml.relative_to(output_path)}")
    
    # Save runtime parameters as JSON
    if runtime_params is not None:
        params_json = snapshot_dir / "runtime_params.json"
        # Convert non-serializable objects to strings
        serializable_params = _make_serializable(runtime_params)
        with params_json.open("w", encoding="utf-8") as f:
            json.dump(serializable_params, f, indent=2, ensure_ascii=False)
        print(f"[config] Saved runtime params: {params_json.relative_to(output_path)}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "datetime_iso": datetime.now().isoformat(),
        "python_version": sys.version,
        "snapshot_name": snapshot_name,
        "original_config_file": str(config_file) if config_file else None,
    }
    metadata_json = snapshot_dir / "metadata.json"
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Create a README
    readme = snapshot_dir / "README.txt"
    with readme.open("w", encoding="utf-8") as f:
        f.write(f"Configuration Snapshot\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Snapshot: {snapshot_name}\n\n")
        f.write(f"Files:\n")
        if config_file:
            f.write(f"  - original_{Path(config_file).name}: Original config file\n")
        if config_dict:
            f.write(f"  - effective_config.yaml: Processed/merged config\n")
        if runtime_params:
            f.write(f"  - runtime_params.json: CLI args and computed parameters\n")
        f.write(f"  - metadata.json: Timestamp and system info\n")
        f.write(f"\nThis snapshot allows complete reproducibility of the analysis.\n")
    
    print(f"[config] Config snapshot saved: {snapshot_dir.relative_to(output_path)}")
    return snapshot_dir


def _make_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to serializable forms."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_quick_snapshot(output_dir: str | Path, config_file: str | Path) -> Path:
    """
    Quick snapshot: just copy the config file with timestamp.
    
    Args:
        output_dir: Directory where to save
        config_file: Config file to copy
    
    Returns:
        Path to the copied config file
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = output_path / f"{config_path.stem}_{timestamp}{config_path.suffix}"
    shutil.copy2(config_path, dest)
    
    print(f"[config] Saved config snapshot: {dest.relative_to(output_path)}")
    return dest
