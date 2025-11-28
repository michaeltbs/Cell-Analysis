import yaml
import copy
import os
from pathlib import Path
from typing import Any, Dict, Union, Tuple, Optional

# ---------------------------------------------------------------------
# Image-Size-Based Scaling Utilities
# ---------------------------------------------------------------------

def compute_effective_scale(
    image_shape: Tuple[int, int],
    resize_max: int = 2048,
    reference_magnification: float = 10.0,
    current_magnification: float = 10.0,
    base_diameter: float = 30.0,
    target_diameter_range: Tuple[float, float] = (10.0, 30.0),
) -> Dict[str, Any]:
    """
    Compute effective scaling factors based on image dimensions, resize behavior, and magnification.
    
    This function solves the problem where different source images (e.g., cropped 5x vs full 10x)
    result in different effective cell sizes after Cellpose resizing.
    
    Key insight: Cellpose's resize_max shrinks large images more than small images,
    which affects the effective cell diameter in the resized image.
    
    Parameters:
    -----------
    image_shape : Tuple[int, int]
        Original image dimensions (height, width)
    resize_max : int
        Cellpose resize_max parameter (default 2048)
    reference_magnification : float
        The magnification at which base_diameter was calibrated (default 10x)
    current_magnification : float
        The magnification of the current image
    base_diameter : float
        The expected cell diameter at reference magnification (in original pixels)
    target_diameter_range : Tuple[float, float]
        The target diameter range for optimal Cellpose performance (default 10-30)
    
    Returns:
    --------
    Dict with:
        - effective_diameter: diameter to use for Cellpose
        - resize_factor: how much the image will be shrunk by Cellpose
        - magnification_scale: scaling factor from magnification difference
        - combined_scale: total effective scale factor for areas and distances
        - quality_warning: optional warning message if parameters are suboptimal
    """
    height, width = image_shape
    max_dim = max(height, width)
    
    # 1. Calculate how much Cellpose will resize the image
    if max_dim > resize_max:
        resize_factor = resize_max / max_dim
    else:
        resize_factor = 1.0
    
    # 2. Calculate magnification-based scaling
    # Lower magnification = cells appear smaller in the image
    if reference_magnification > 0 and current_magnification > 0:
        mag_scale = current_magnification / reference_magnification
    else:
        mag_scale = 1.0
    
    # 3. Calculate effective diameter after resize
    # base_diameter is calibrated for reference_mag on reference image size
    # At different magnification: apparent_diameter = base_diameter * mag_scale
    # After resize: effective_diameter = apparent_diameter * resize_factor
    apparent_diameter = base_diameter * mag_scale
    effective_diameter = apparent_diameter * resize_factor
    
    # 4. Check if diameter is in optimal range
    target_min, target_max = target_diameter_range
    quality_warning = None
    
    if effective_diameter < target_min:
        quality_warning = (
            f"Effective diameter {effective_diameter:.1f}px is below optimal range [{target_min}-{target_max}]. "
            f"Consider using smaller resize_max or higher magnification images."
        )
    elif effective_diameter > target_max:
        quality_warning = (
            f"Effective diameter {effective_diameter:.1f}px is above optimal range [{target_min}-{target_max}]. "
            f"This may cause under-segmentation."
        )
    
    # 5. Combined scale factor for areas and distances (operates in resized space)
    # This accounts for both magnification AND resize effects
    combined_scale = mag_scale * resize_factor
    area_scale = combined_scale * combined_scale
    
    return {
        "effective_diameter": effective_diameter,
        "resize_factor": resize_factor,
        "magnification_scale": mag_scale,
        "combined_scale": combined_scale,
        "area_scale": area_scale,
        "quality_warning": quality_warning,
        "original_shape": (height, width),
        "max_dimension": max_dim,
        "resize_max": resize_max,
    }


def apply_image_aware_scaling(
    cfg: dict,
    image_shape: Tuple[int, int],
    resize_max: Optional[int] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """
    Apply scaling to config based on both image size and magnification.
    
    This is an enhanced version of apply_magnification_scaling that also
    considers the actual image dimensions and how Cellpose will resize them.
    
    Returns:
    --------
    Tuple of (scaled_config, scaling_info)
    """
    new_cfg = copy.deepcopy(cfg or {})
    
    # Get configuration values
    scope_cfg = new_cfg.get("microscope", {}) or {}
    cp_cfg = new_cfg.get("cellpose", {}) or {}
    
    if resize_max is None:
        resize_max = int(cp_cfg.get("resize_max", 2048) or 2048)
    
    reference_mag = float(scope_cfg.get("reference_magnification", 10.0) or 10.0)
    current_mag = float(scope_cfg.get("current_magnification", reference_mag) or reference_mag)
    base_diameter = float(cp_cfg.get("diameter", 30.0) or 30.0)
    
    # Compute effective scaling
    scale_info = compute_effective_scale(
        image_shape=image_shape,
        resize_max=resize_max,
        reference_magnification=reference_mag,
        current_magnification=current_mag,
        base_diameter=base_diameter,
    )
    
    # Apply the combined scale to parameters
    combined_scale = scale_info["combined_scale"]
    area_scale = scale_info["area_scale"]
    
    # Update cellpose diameter with effective value
    if isinstance(cp_cfg, dict):
        cp_cfg["diameter"] = scale_info["effective_diameter"]
        new_cfg["cellpose"] = cp_cfg
    
    # Scale processing parameters
    processing_cfg = new_cfg.get("processing", {})
    if isinstance(processing_cfg, dict):
        split_dist = processing_cfg.get("split_min_distance")
        if isinstance(split_dist, (int, float)) and split_dist:
            processing_cfg["split_min_distance"] = int(max(1, round(float(split_dist) * combined_scale)))
        split_area = processing_cfg.get("split_min_area")
        if isinstance(split_area, (int, float)) and split_area:
            processing_cfg["split_min_area"] = int(max(1, round(float(split_area) * area_scale)))
        new_cfg["processing"] = processing_cfg
    
    # Scale filter parameters
    filters_cfg = new_cfg.get("filters", {})
    if isinstance(filters_cfg, dict):
        min_area = filters_cfg.get("min_area")
        if isinstance(min_area, (int, float)) and min_area:
            filters_cfg["min_area"] = int(max(1, round(float(min_area) * area_scale)))
        max_area = filters_cfg.get("max_area")
        if isinstance(max_area, (int, float)) and max_area:
            filters_cfg["max_area"] = int(max(1, round(float(max_area) * area_scale)))
        new_cfg["filters"] = filters_cfg
    
    # Scale advanced filtering
    adv_cfg = new_cfg.get("advanced_filtering", {})
    if isinstance(adv_cfg, dict):
        block = adv_cfg.get("foreground_block_size")
        if isinstance(block, (int, float)) and block:
            block_scaled = int(max(3, round(float(block) * combined_scale)))
            if block_scaled % 2 == 0:
                block_scaled += 1
            adv_cfg["foreground_block_size"] = block_scaled
        new_cfg["advanced_filtering"] = adv_cfg
    
    # Store scaling info in config
    scope_cfg["effective_scale"] = combined_scale
    scope_cfg["resize_factor"] = scale_info["resize_factor"]
    new_cfg["microscope"] = scope_cfg
    
    return new_cfg, scale_info


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
