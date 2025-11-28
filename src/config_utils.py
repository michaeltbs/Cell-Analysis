import yaml
import copy
import os
from pathlib import Path
from typing import Any, Dict, Union

def load_yaml(path: Union[str, Path]) -> dict:
    """Safely load a YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def save_yaml(path: Union[str, Path], data: dict) -> None:
    """Safely save a dictionary to a YAML file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        print(f"Failed to save YAML to {path}: {e}")

def deep_update(target: dict, source: dict) -> dict:
    """Recursively update a dictionary."""
    for k, v in source.items():
        if isinstance(v, dict):
            target[k] = deep_update(target.get(k, {}), v)
        else:
            target[k] = v
    return target

def apply_magnification_scaling(cfg: dict) -> dict:
    """Return a config copy with size-dependent parameters scaled for the current objective."""
    new_cfg = copy.deepcopy(cfg or {})
    scope_cfg = new_cfg.get("microscope", {}) or {}

    def _as_float(value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    reference = scope_cfg.get("reference_magnification", 10.0)
    current = scope_cfg.get("current_magnification", scope_cfg.get("magnification", reference))
    reference = _as_float(reference, 10.0)
    current = _as_float(current, reference)
    if reference <= 0:
        reference = 10.0
    if current <= 0:
        current = reference

    scale = max(current / reference, 1e-3)
    area_scale = scale * scale
    
    # If current equals reference, no scaling needed
    if abs(scale - 1.0) < 1e-6:
        scope_cfg["reference_magnification"] = reference
        scope_cfg["current_magnification"] = current
        scope_cfg["scale_factor"] = 1.0
        new_cfg["microscope"] = scope_cfg
        return new_cfg

    scope_cfg["reference_magnification"] = reference
    scope_cfg["current_magnification"] = current
    scope_cfg["scale_factor"] = scale
    new_cfg["microscope"] = scope_cfg

    cp_cfg = new_cfg.get("cellpose")
    if isinstance(cp_cfg, dict):
        # resize_max stays constant - all images normalized to same size
        # diameter scales with magnification (cells appear smaller at lower mag)
        diameter = cp_cfg.get("diameter")
        if isinstance(diameter, (int, float)) and diameter:
            cp_cfg["diameter"] = max(1.0, float(diameter) * scale)

    processing_cfg = new_cfg.get("processing")
    if isinstance(processing_cfg, dict):
        # tophat operates on original image BEFORE resize -> inverse scaling
        radius = processing_cfg.get("tophat_radius")
        if isinstance(radius, (int, float)) and radius:
            processing_cfg["tophat_radius"] = int(max(1, round(float(radius) / scale)))
        # All other processing params operate AFTER resize and scale with magnification
        split_dist = processing_cfg.get("split_min_distance")
        if isinstance(split_dist, (int, float)) and split_dist:
            processing_cfg["split_min_distance"] = int(max(1, round(float(split_dist) * scale)))
        split_area = processing_cfg.get("split_min_area")
        if isinstance(split_area, (int, float)) and split_area:
            processing_cfg["split_min_area"] = int(max(1, round(float(split_area) * area_scale)))

    filters_cfg = new_cfg.get("filters")
    if isinstance(filters_cfg, dict):
        # Filters operate on resized image - scale with magnification
        min_area = filters_cfg.get("min_area")
        if isinstance(min_area, (int, float)) and min_area:
            filters_cfg["min_area"] = int(max(1, round(float(min_area) * area_scale)))
        max_area = filters_cfg.get("max_area")
        if isinstance(max_area, (int, float)) and max_area:
            filters_cfg["max_area"] = int(max(1, round(float(max_area) * area_scale)))

    adv_cfg = new_cfg.get("advanced_filtering")
    if isinstance(adv_cfg, dict):
        # Advanced filtering operates on resized image - scale with magnification
        block = adv_cfg.get("foreground_block_size")
        if isinstance(block, (int, float)) and block:
            block_scaled = int(max(3, round(float(block) * scale)))
            # Ensure odd number
            if block_scaled % 2 == 0:
                block_scaled += 1
            adv_cfg["foreground_block_size"] = block_scaled
        offset = adv_cfg.get("foreground_offset")
        if isinstance(offset, (int, float)) and offset:
            adv_cfg["foreground_offset"] = int(round(float(offset) * scale))

    overlays_cfg = new_cfg.get("overlays")
    if isinstance(overlays_cfg, dict):
        # Overlays operate on resized image - scale with magnification
        line_width = overlays_cfg.get("line_width")
        if isinstance(line_width, (int, float)) and line_width:
            overlays_cfg["line_width"] = int(max(1, round(float(line_width) * scale)))

    return new_cfg
