#!/usr/bin/env python
"""Test script for image-size-based scaling."""
from src.config_utils import compute_effective_scale, apply_image_aware_scaling

# Test cases: (image_shape, resize_max, ref_mag, cur_mag, base_diameter, description)
tests = [
    ((2000, 3000), 2048, 10.0, 5.0, 30.0, "5x on 3000x2000"),
    ((1024, 1024), 2048, 10.0, 10.0, 30.0, "10x on 1024x1024"),
    ((1024, 1024), 2048, 10.0, 5.0, 30.0, "5x on 1024x1024 (cropped)"),
    ((4000, 4000), 2048, 10.0, 10.0, 30.0, "10x on 4000x4000 (large)"),
    ((512, 512), 2048, 10.0, 5.0, 30.0, "5x on 512x512 (small crop)"),
]

print("=" * 70)
print("Image-Size-Based Scaling Tests")
print("=" * 70)
print()

for shape, resize_max, ref_mag, cur_mag, base_d, desc in tests:
    info = compute_effective_scale(
        image_shape=shape,
        resize_max=resize_max,
        reference_magnification=ref_mag,
        current_magnification=cur_mag,
        base_diameter=base_d,
    )
    
    print(f"{desc}:")
    print(f"  Original: {shape[1]}x{shape[0]}px")
    print(f"  Effective diameter: {info['effective_diameter']:.1f}px")
    print(f"  Resize factor: {info['resize_factor']:.3f}")
    print(f"  Magnification scale: {info['magnification_scale']:.3f}")
    print(f"  Combined scale: {info['combined_scale']:.3f}")
    print(f"  Area scale: {info['area_scale']:.3f}")
    if info['quality_warning']:
        print(f"  ⚠️  {info['quality_warning']}")
    print()

print("=" * 70)
print("Test apply_image_aware_scaling with config")
print("=" * 70)

test_cfg = {
    "microscope": {"reference_magnification": 10.0, "current_magnification": 5.0},
    "cellpose": {"diameter": 30.0, "resize_max": 2048},
    "filters": {"min_area": 50, "max_area": 5000},
    "processing": {"split_min_distance": 5, "split_min_area": 100},
}

scaled_cfg, info = apply_image_aware_scaling(test_cfg, image_shape=(2000, 3000), resize_max=2048)

print(f"Original config:")
print(f"  diameter: {test_cfg['cellpose']['diameter']}")
print(f"  min_area: {test_cfg['filters']['min_area']}")
print(f"  max_area: {test_cfg['filters']['max_area']}")

print(f"\nScaled config (5x on 3000x2000):")
print(f"  diameter: {scaled_cfg['cellpose']['diameter']:.1f}")
print(f"  min_area: {scaled_cfg['filters']['min_area']}")
print(f"  max_area: {scaled_cfg['filters']['max_area']}")

print("\n✅ All tests completed successfully!")
