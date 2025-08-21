import os
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from skimage import io as skio
from skimage.transform import resize as skresize
from tqdm import tqdm

import torch
from cellpose import models

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from skimage.measure import regionprops
    from skimage.color import gray2rgb
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib nicht verfügbar - Overlays werden übersprungen")

from skimage.filters import gaussian, threshold_local
from skimage.morphology import binary_dilation, disk, white_tophat
from scipy.ndimage import binary_fill_holes

def _ensure_dir(p: str):
	os.makedirs(p, exist_ok=True)

def _list_tiffs(folder: str) -> List[str]:
	patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.czi"]
	paths = []
	for pat in patterns:
		paths.extend(glob.glob(os.path.join(folder, pat)))
	return sorted(set(paths), key=lambda p: (os.path.basename(p).lower(), p))

def load_image(image_path: str) -> np.ndarray:
    """Load image based on file format (TIFF, CZI, etc.)"""
    if image_path.lower().endswith('.czi'):
        try:
            from aicspylibczi import CziFile
            czi = CziFile(image_path)
            image_array = czi.read_image()
            
            # Handle different CZI dimensions and create Z-projection
            print(f"CZI Shape: {image_array.shape}")
            if image_array.ndim == 5:  # (T, C, Z, Y, X)
                projected = np.max(image_array[0, 0], axis=0)  # First time, first channel
            elif image_array.ndim == 4:  # (C, Z, Y, X)
                projected = np.max(image_array[0], axis=0)     # First channel
            elif image_array.ndim == 3:  # (Z, Y, X)
                projected = np.max(image_array, axis=0)        # Z-projection
            else:
                projected = image_array.squeeze()
                
            return projected.astype(np.uint16)
        except ImportError:
            raise ImportError("aicspylibczi is not installed. Run: pip install aicspylibczi")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    else:
        # Default to TIFF/PNG loading
        return skio.imread(image_path)

def _preprocess_image(image: np.ndarray, processing: Dict) -> np.ndarray:
    """Robust 8/16-bit normalization, optional contrast stretch, and white-tophat."""
    orig_dtype = image.dtype
    img = image.astype(np.float32, copy=False)

    # Scale based on dtype or value range
    if np.issubdtype(orig_dtype, np.integer):
        maxv = float(np.iinfo(orig_dtype).max)  # 255 or 65535
    else:
        vmax = float(np.nanmax(img)) if img.size else 1.0
        maxv = 65535.0 if vmax > 4096 else (255.0 if vmax > 1.5 else 1.0)

    if maxv > 1.0:
        img = img / maxv

    # Optional: robust contrast stretch (1-99 percentile by default)
    if processing.get("contrast_stretch", True):
        p_low, p_high = processing.get("stretch_percentiles", [1.0, 99.0])
        p1, p99 = np.percentile(img, (p_low, p_high))
        if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
            img = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

    # Optional: white-tophat for background suppression
    if processing.get("tophat", False):
        radius = int(processing.get("tophat_radius", 15))
        if radius > 0:
            selem = disk(radius)
            img = white_tophat(img, selem)
            img = np.clip(img, 0.0, 1.0)

    return img

def _load_and_preprocess(path: str, target_max: int, channel: int = 0, processing: Dict = None) -> Tuple[str, np.ndarray]:
    processing = processing or {}
    img = load_image(path)
    
    # Channel extraction for multi-channel images
    if img.ndim == 3 and img.shape[2] > 1:
        print(f"🔍 Multi-channel image detected: {img.shape}, using channel {channel}")
        if channel < img.shape[2]:
            img = img[:, :, channel]
        else:
            print(f"⚠️ Channel {channel} not available, using channel 0")
            img = img[:, :, 0]
    
    # Apply preprocessing
    img = _preprocess_image(img, processing)
    
    # Resize if necessary
    h, w = img.shape[:2]
    scale = target_max / max(h, w) if max(h, w) > 0 else 1.0
    if scale != 1.0:
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = skresize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True).astype(np.float32)
    return path, img

def _parallel_load(paths: List[str], target_max: int, num_workers: int, channel: int = 0) -> List[Tuple[str, np.ndarray]]:
	results = []
	with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
		futs = {ex.submit(_load_and_preprocess, p, target_max, channel): p for p in paths}
		for fut in tqdm(as_completed(futs), total=len(futs), desc="Load/resize", unit="img"):
			results.append(fut.result())
	order = {p: i for i, p in enumerate(paths)}
	results.sort(key=lambda x: order[x[0]])
	return results

def apply_advanced_filtering(masks: np.ndarray, image: np.ndarray, 
                           advanced_config: dict) -> np.ndarray:
    """Apply advanced filtering based on SNR, intensity, and foreground detection"""
    if masks.max() == 0:
        return masks
    
    props = regionprops(masks, intensity_image=image)
    filtered_masks = masks.copy()
    
    # Get advanced filtering parameters
    snr_min = advanced_config.get('snr_min', 1.2)
    abs_floor_percentile = advanced_config.get('abs_floor_percentile', 75)
    foreground_block_size = advanced_config.get('foreground_block_size', 51)
    foreground_offset = advanced_config.get('foreground_offset', -10)
    
    # Calculate intensity floor from percentile
    intensity_floor = np.percentile(image, abs_floor_percentile)
    
    # Foreground detection using local thresholding
    from skimage.filters import threshold_local
    local_thresh = threshold_local(image, foreground_block_size, offset=foreground_offset)
    foreground_mask = image > local_thresh
    
    # Filter cells
    cells_to_remove = []
    for prop in props:
        cell_id = prop.label
        
        # SNR calculation
        cell_mask = masks == cell_id
        background_region = np.logical_and(~cell_mask, foreground_mask)
        if background_region.sum() > 0:
            bg_std = np.std(image[background_region])
            if bg_std > 0:
                snr = prop.mean_intensity / bg_std
                if snr < snr_min:
                    cells_to_remove.append(cell_id)
                    continue
        
        # Intensity floor check
        if prop.mean_intensity < intensity_floor:
            cells_to_remove.append(cell_id)
            continue
        
        # Foreground overlap check
        cell_foreground_overlap = np.logical_and(cell_mask, foreground_mask).sum()
        cell_total = cell_mask.sum()
        if cell_foreground_overlap / cell_total < 0.5:  # Require 50% overlap
            cells_to_remove.append(cell_id)
    
    # Remove filtered cells
    for cell_id in cells_to_remove:
        filtered_masks[filtered_masks == cell_id] = 0
    
    # Renumber masks
    unique_labels = np.unique(filtered_masks)
    unique_labels = unique_labels[unique_labels > 0]
    renumbered_masks = np.zeros_like(filtered_masks)
    for new_id, old_id in enumerate(unique_labels, 1):
        renumbered_masks[filtered_masks == old_id] = new_id
    
    return renumbered_masks

def apply_filters(masks: np.ndarray, image: np.ndarray, 
                 filters_config: dict, enable_advanced: bool = False,
                 advanced_config: dict = None) -> np.ndarray:
    """Apply all filters to segmentation masks"""
    if masks.max() == 0:
        return masks
    
    # DEBUG: Print filter settings
    print(f"🔍 DEBUG apply_filters:")
    print(f"   Input masks: {masks.max()} cells")
    print(f"   Filters: {filters_config}")
    print(f"   Advanced enabled: {enable_advanced}")
    
    # Apply advanced filtering if enabled
    if enable_advanced and advanced_config:
        masks = apply_advanced_filtering(masks, image, advanced_config)
        print(f"   After advanced filtering: {masks.max()} cells")
        if masks.max() == 0:
            return masks
    
    # Apply basic filters
    props = regionprops(masks, intensity_image=image)
    filtered_masks = masks.copy()
    
    # Get basic filter parameters
    min_area = filters_config.get('min_area', 0)
    max_area = filters_config.get('max_area', float('inf'))
    min_circularity = filters_config.get('min_circularity', 0.0)
    max_circularity = filters_config.get('max_circularity', 1.0)
    max_hole_ratio = filters_config.get('max_hole_ratio', 1.0)
    intensity_threshold_factor = filters_config.get('intensity_threshold_factor', 0.0)
    
    print(f"   Basic filters: area={min_area}-{max_area}, circ={min_circularity}-{max_circularity}")
    
    # Calculate intensity threshold
    mean_intensity = np.mean(image[image > 0])
    intensity_threshold = mean_intensity * intensity_threshold_factor
    print(f"   Intensity: mean={mean_intensity:.3f}, threshold={intensity_threshold:.3f}")
    
    cells_to_remove = []
    area_filtered = 0
    circ_filtered = 0
    intensity_filtered = 0
    
    for prop in props:
        # Area filter
        if not (min_area <= prop.area <= max_area):
            cells_to_remove.append(prop.label)
            area_filtered += 1
            continue
        
        # Circularity filter
        perimeter = prop.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * prop.area / (perimeter ** 2)
            if not (min_circularity <= circularity <= max_circularity):
                cells_to_remove.append(prop.label)
                circ_filtered += 1
                continue
        
        # Hole ratio filter
        if prop.solidity < (1 - max_hole_ratio):
            cells_to_remove.append(prop.label)
            continue
        
        # Intensity filter
        if prop.mean_intensity < intensity_threshold:
            cells_to_remove.append(prop.label)
            intensity_filtered += 1
            continue
    
    print(f"   Filtered: {area_filtered} area, {circ_filtered} circularity, {intensity_filtered} intensity")
    print(f"   Remaining: {len(props) - len(cells_to_remove)}/{len(props)} cells")
    
    # Remove filtered cells
    for cell_id in cells_to_remove:
        filtered_masks[filtered_masks == cell_id] = 0
    
    # Renumber masks
    unique_labels = np.unique(filtered_masks)
    unique_labels = unique_labels[unique_labels > 0]
    renumbered_masks = np.zeros_like(filtered_masks)
    for new_id, old_id in enumerate(unique_labels, 1):
        renumbered_masks[filtered_masks == old_id] = new_id
    
    return renumbered_masks

def calculate_metrics(image: np.ndarray, masks: np.ndarray, 
                     filters: dict = None) -> Tuple[dict, List[int]]:
    """Calculate cell metrics and return valid labels based on basic filters."""
    if masks.max() == 0:
        return {
            'cell_count': 0,
            'mean_area_per_cell': 0.0,
            'mean_intensity_per_cell': 0.0,
            'mean_integrated_density_per_cell': 0.0
        }, []
    
    props = regionprops(masks, intensity_image=image)

    valid_props: List = []
    valid_labels: List[int] = []

    for prop in props:
        if filters:
            # Area
            if prop.area < filters.get('min_area', 0) or prop.area > filters.get('max_area', float('inf')):
                continue
            # Circularity
            perim = prop.perimeter
            circ = (4 * np.pi * prop.area / (perim ** 2)) if perim > 0 else 0.0
            if circ < filters.get('min_circularity', 0.0) or circ > filters.get('max_circularity', 1.0):
                continue
        valid_props.append(prop)
        valid_labels.append(prop.label)

    if not valid_props:
        return {
            'cell_count': 0,
            'mean_area_per_cell': 0.0,
            'mean_intensity_per_cell': 0.0,
            'mean_integrated_density_per_cell': 0.0
        }, []

    areas = [p.area for p in valid_props]
    intensities = [p.mean_intensity for p in valid_props]
    integrated = [p.area * p.mean_intensity for p in valid_props]

    return {
        'cell_count': len(valid_props),
        'mean_area_per_cell': float(np.mean(areas)),
        'mean_intensity_per_cell': float(np.mean(intensities)),
        'mean_integrated_density_per_cell': float(np.mean(integrated)),
    }, valid_labels

def create_overlay_image(image: np.ndarray, masks: np.ndarray, 
                        output_path: str, overlay_config: dict,
                        valid_labels: List[int] = None) -> None:
    """Create overlay using config; only draw valid_labels if provided."""
    if not MATPLOTLIB_AVAILABLE:
        return

    display_img = gray2rgb(image) if image.ndim == 2 else image.copy()
    vmax = float(display_img.max())
    if vmax > 0:
        display_img = (display_img / vmax * 255).astype(np.uint8)

    overlay_mode = overlay_config.get('overlay_mode', 'circles')
    figsize = tuple(overlay_config.get('figsize', (10, 10)))
    dpi = overlay_config.get('dpi', 200)
    line_width = overlay_config.get('line_width', 2)
    circle_radius = overlay_config.get('circle_radius', 5)
    circle_color = overlay_config.get('circle_color', 'red')
    contour_color = overlay_config.get('contour_color', 'blue')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(display_img, cmap='gray')

    props = regionprops(masks)
    if valid_labels is not None:
        props = [p for p in props if p.label in valid_labels]

    if overlay_mode == 'contours':
        from skimage.measure import find_contours
        for p in props:
            cmask = (masks == p.label).astype(np.uint8)
            for contour in find_contours(cmask, 0.5):
                ax.plot(contour[:, 1], contour[:, 0],
                        color=contour_color, linewidth=line_width)
    else:
        for p in props:
            y, x = p.centroid
            circ = Circle((x, y), radius=circle_radius, color=circle_color,
                          fill=False, linewidth=line_width)
            ax.add_patch(circ)

    ax.set_title(f"Valid cells: {len(props)}")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def segment_dirs(
	input_pos: str,
	input_neg: str,
	output_root: str = "results",
	batch_size: int = 8,
	resize_max: int = 1000,
	niter: int = 250,
	flow_threshold: float = 0.4,
	cellprob_threshold: float = 0.75,
	num_workers: int = 4,
	model_name: str = "cpsam",
	save_masks: bool = True,
	create_overlays: bool = False,
	channel: int = 0,
	filters: dict = None,
	overlay_config: dict = None,
	split_regions: bool = False,
	diameter: int = 30,
	enable_advanced_filtering: bool = False,
	advanced_filtering: dict = None,
	processing: dict = None  # Add processing parameter
) -> str:
	pos_files = _list_tiffs(input_pos) if os.path.isdir(input_pos) else []
	neg_files = _list_tiffs(input_neg) if os.path.isdir(input_neg) else []
	all_files = [(p, "pos") for p in pos_files] + [(p, "neg") for p in neg_files]
	if not all_files:
		raise FileNotFoundError("No TIFF files found.")

	# Setup output directories
	if save_masks:
		masks_dir = os.path.join(output_root, "masks")
		_ensure_dir(masks_dir)
	
	if create_overlays and MATPLOTLIB_AVAILABLE:
		overlays_dir = os.path.join(output_root, "overlays")
		_ensure_dir(overlays_dir)
	
	_ensure_dir(output_root)
	master_csv = os.path.join(output_root, "All_Counts_Master.csv")

	model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=model_name)

	rows = []
	
	# Local region splitter to avoid NameError
	def _split_regions_local(image: np.ndarray) -> Dict[str, np.ndarray]:
		h = image.shape[0]
		mid = h // 2
		return {"DMH": image[:mid, :], "ARC": image[mid:, :]}

	# Process each image
	for path, cond in tqdm(all_files, desc="Processing images", unit="img"):
		base = os.path.splitext(os.path.basename(path))[0]
		
		try:
			# Load and preprocess image with processing options
			_, img = _load_and_preprocess(path, resize_max, channel, processing)
			
			if split_regions:
				# Process image regions separately (DMH and ARC)
				regions = _split_regions_local(img)
				
				for region_name, region_img in regions.items():
					# Run cellpose segmentation on region
					with torch.inference_mode():
						masks, flows, styles = model.eval(
							[region_img],
							diameter=diameter,
							channels=[0, 0],
							flow_threshold=flow_threshold,
							cellprob_threshold=cellprob_threshold,
							do_3D=False,
							normalize=True,
							resample=False,
							niter=niter,
							batch_size=1,
							augment=False,
						)
					
					# Handle mask format
					if isinstance(masks, list):
						mask = masks[0]
					else:
						mask = masks
					
					# Apply filters and advanced filtering ONCE
					filtered_mask = apply_filters(
						mask, 
						region_img, 
						filters or {}, 
						enable_advanced_filtering,
						advanced_filtering
					)
					
					# Calculate metrics WITHOUT additional filtering (filters=None)
					metrics, valid_labels = calculate_metrics(region_img, filtered_mask, None)
					
					# Save mask if requested
					if save_masks:
						out_mask_path = os.path.join(masks_dir, f"{base}_{region_name}_mask.tiff")
						skio.imsave(out_mask_path, filtered_mask.astype(np.uint16), check_contrast=False)
					
					# Create overlay if requested
					if create_overlays:
						overlay_path = os.path.join(overlays_dir, f"{base}_{region_name}_overlay.png")
						create_overlay_image(region_img, filtered_mask, overlay_path, overlay_config, valid_labels)
					
					# Add to results
					rows.append({
						"filename": os.path.basename(path),
						"condition": cond,
						"region": region_name,
						"channel": f"channel_{channel}",
						"cell_count": metrics['cell_count'],
						"mean_area_per_cell": metrics['mean_area_per_cell'],
						"mean_intensity_per_cell": metrics['mean_intensity_per_cell'],
						"mean_integrated_density_per_cell": metrics['mean_integrated_density_per_cell'],
					})
			
			else:
				# Process whole image as MBH
				with torch.inference_mode():
					masks, flows, styles = model.eval(
						[img],
						diameter=diameter,
						channels=[0, 0],
						flow_threshold=flow_threshold,
						cellprob_threshold=cellprob_threshold,
						do_3D=False,
						normalize=True,
						resample=False,
						niter=niter,
						batch_size=1,
						augment=False,
					)
				
				if isinstance(masks, list):
					mask = masks[0]
				else:
					mask = masks
				
				# Apply filters and advanced filtering ONCE
				filtered_mask = apply_filters(
					mask, 
					img, 
					filters or {}, 
					enable_advanced_filtering,
					advanced_filtering
				)
				
				# Calculate metrics WITHOUT additional filtering (filters=None)
				metrics, valid_labels = calculate_metrics(img, filtered_mask, None)
				
				# Save mask if requested
				if save_masks:
					out_mask_path = os.path.join(masks_dir, f"{base}_MBH_mask.tiff")
					skio.imsave(out_mask_path, filtered_mask.astype(np.uint16), check_contrast=False)
				
				# Create overlay if requested
				if create_overlays:
					overlay_path = os.path.join(overlays_dir, f"{base}_MBH_overlay.png")
					create_overlay_image(img, filtered_mask, overlay_path, overlay_config, valid_labels)
				
				# Add to results
				rows.append({
					"filename": os.path.basename(path),
					"condition": cond,
					"region": "MBH",
					"channel": f"channel_{channel}",
					"cell_count": metrics['cell_count'],
					"mean_area_per_cell": metrics['mean_area_per_cell'],
					"mean_intensity_per_cell": metrics['mean_intensity_per_cell'],
					"mean_integrated_density_per_cell": metrics['mean_integrated_density_per_cell'],
				})
		
		except Exception as e:
			print(f"❌ Error processing {path}: {e}")
			continue

	# Always write headers even if no rows
	df = pd.DataFrame(
		rows,
		columns=[
			"filename",
			"condition",
			"region",
			"channel",
			"cell_count",
			"mean_area_per_cell",
			"mean_intensity_per_cell",
			"mean_integrated_density_per_cell",
		],
	)
	df.to_csv(master_csv, index=False)
	return master_csv

def main():
	import argparse
	import yaml
	
	def _load_cfg(path):
		with open(path, 'r', encoding='utf-8') as f:
			return yaml.safe_load(f)
	
	parser = argparse.ArgumentParser(description="Cell Analysis Pipeline")
	parser.add_argument("--config", type=str, default="config.yaml")
	parser.add_argument("--no-masks", action="store_true")
	parser.add_argument("--no-summary", action="store_true")
	parser.add_argument("--create-overlays", action="store_true")
	parser.add_argument("--no-overlays", action="store_true")
	args = parser.parse_args()

	cfg = _load_cfg(args.config)
	
	final_save_masks = cfg.get("cellpose", {}).get("save_masks", True) and not args.no_masks
	split_regions = cfg.get("cellpose", {}).get("split_regions", False)
	enable_advanced_filtering = cfg.get("cellpose", {}).get("enable_advanced_filtering", False)
	advanced_filtering = cfg.get("advanced_filtering", {})

	if args.no_overlays:
		create_overlays_final = False
	elif args.create_overlays:
		create_overlays_final = True
	elif not final_save_masks:
		create_overlays_final = True
	else:
		create_overlays_final = False

	master_csv = segment_dirs(
		input_pos=cfg["paths"]["input_pos"],
		input_neg=cfg["paths"]["input_neg"],
		output_root=cfg["paths"]["output_root"],
		batch_size=cfg.get("cellpose", {}).get("batch_size", 8),
		resize_max=cfg.get("cellpose", {}).get("resize_max", 1000),
		niter=cfg.get("cellpose", {}).get("niter", 250),
		flow_threshold=cfg.get("cellpose", {}).get("flow_threshold", 0.4),
		cellprob_threshold=cfg.get("cellpose", {}).get("cellprob_threshold", 0.0),
		num_workers=cfg.get("cpu", {}).get("num_workers", 4),
		model_name=cfg.get("cellpose", {}).get("model_name", "cpsam"),
		save_masks=final_save_masks,
		create_overlays=create_overlays_final,
		channel=cfg.get("cellpose", {}).get("channel", 0),
		filters=cfg.get("filters", {}),
		overlay_config=cfg.get("overlays", {}),
		split_regions=split_regions,
		diameter=cfg.get("cellpose", {}).get("diameter", 30),
		enable_advanced_filtering=enable_advanced_filtering,
		advanced_filtering=advanced_filtering,
		processing=cfg.get("processing", {})  # Add processing parameter
	)

	if not args.no_summary:
		df = pd.read_csv(master_csv)
		print(f"\n📊 ZUSAMMENFASSUNG:")
		print(f"   Bilder verarbeitet: {df['filename'].nunique()}")
		print(f"   Regionen analysiert: {len(df)}")

if __name__ == "__main__":
	main()
