import os
import glob
import numpy as np  # Korrigierter Import
import pandas as pd
from typing import List, Tuple
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

def _ensure_dir(p: str):
	os.makedirs(p, exist_ok=True)

def _list_tiffs(folder: str) -> List[str]:
	patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
	paths = []
	for pat in patterns:
		paths.extend(glob.glob(os.path.join(folder, pat)))
	return sorted(set(paths), key=lambda p: (os.path.basename(p).lower(), p))

def _load_and_preprocess(path: str, target_max: int, channel: int = 0) -> Tuple[str, np.ndarray]:
	img = skio.imread(path)
	
	# Kanal-Extraktion für Multi-Kanal Bilder
	if img.ndim == 3 and img.shape[2] > 1:
		print(f"🔍 Multi-Kanal Bild erkannt: {img.shape}, verwende Kanal {channel}")
		if channel < img.shape[2]:
			img = img[:, :, channel]
		else:
			print(f"⚠️ Kanal {channel} nicht verfügbar, verwende Kanal 0")
			img = img[:, :, 0]
	
	img = img.astype(np.float32)
	if img.max() > 1.5:
		img = img / 255.0
	img = np.clip(img, 0.0, 1.0)
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

def _compute_metrics_with_filters(mask: np.ndarray, img: np.ndarray, filters: dict) -> Tuple[int, float, float, float, int]:
	"""Berechnet Metriken mit geometrischen Filtern"""
	labels = mask.astype(np.int64)
	if labels.max() == 0:
		return 0, 0.0, 0.0, 0.0, 0
	
	# Analysiere alle Regionen
	regions = regionprops(labels)
	
	# Wende Filter an
	valid_regions = []
	filtered_count = 0
	
	for region in regions:
		area = region.area
		
		# Größen-Filter
		if area < filters.get('min_area', 0) or area > filters.get('max_area', float('inf')):
			filtered_count += 1
			continue
		
		# Circularity berechnen: 4π*area / perimeter²
		perimeter = region.perimeter
		if perimeter > 0:
			circularity = 4 * np.pi * area / (perimeter ** 2)
		else:
			circularity = 0
		
		if (circularity < filters.get('min_circularity', 0) or 
		    circularity > filters.get('max_circularity', 1.0)):
			filtered_count += 1
			continue
		
		# Solidity (Kompaktheit): area / convex_area
		solidity = region.solidity
		if (solidity < filters.get('min_solidity', 0) or 
		    solidity > filters.get('max_solidity', 1.0)):
			filtered_count += 1
			continue
		
		valid_regions.append(region)
	
	n_valid_cells = len(valid_regions)
	
	if n_valid_cells == 0:
		return 0, 0.0, 0.0, 0.0, filtered_count
	
	# Berechne Metriken nur für gültige Zellen
	areas = [r.area for r in valid_regions]
	mean_area = float(np.mean(areas))
	
	# Intensitäts-Metriken (nur für gültige Regionen)
	img_gray = img.mean(axis=2) if img.ndim == 3 else img
	
	intensity_sums = []
	for region in valid_regions:
		coords = region.coords
		region_intensities = img_gray[coords[:, 0], coords[:, 1]]
		intensity_sums.append(np.sum(region_intensities))
	
	mean_intensity_per_cell = float(np.mean([sum_int / area for sum_int, area in zip(intensity_sums, areas)]))
	mean_integrated_density = float(np.mean(intensity_sums))
	
	return n_valid_cells, mean_area, mean_intensity_per_cell, mean_integrated_density, filtered_count

def _create_quick_overlay(img: np.ndarray, mask: np.ndarray, output_path: str, max_circles: int = None):
    """Erstellt Overlay für ALLE Zellen (nicht mehr limitiert)"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib nicht verfügbar - Overlay übersprungen")
        return
    
    try:
        vis_img = img.mean(axis=2) if img.ndim == 3 else img
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
        
        regions = regionprops(mask)[:max_circles]  # Nur erste N für Performance
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(vis_img, cmap='gray')
        
        for region in regions:
            if region.area > 10:
                y, x = region.centroid
                radius = np.sqrt(region.area / np.pi)
                circle = Circle((x, y), radius, color='red', fill=False, linewidth=1, alpha=0.7)
                ax.add_patch(circle)
        
        ax.set_title(f'{len(regions)} Zellen (von {mask.max()} gesamt)')
        ax.axis('off')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Overlay-Fehler: {e}")

def _create_advanced_overlay(img: np.ndarray, mask: np.ndarray, output_path: str, 
                           overlay_config: dict = None):
    """Erstellt erweiterte Overlays mit konfigurierbaren Optionen"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib nicht verfügbar - Overlay übersprungen")
        return
    
    # Standard-Konfiguration
    if overlay_config is None:
        overlay_config = {
            'show_all_cells': True,
            'high_resolution': True,
            'circle_color': 'red',
            'circle_alpha': 0.8,
            'line_width': 0.8
        }
    
    try:
        vis_img = img.mean(axis=2) if img.ndim == 3 else img
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
        
        # Alle Regionen analysieren
        regions = regionprops(mask)
        
        # Figur-Größe basierend auf Bildgröße
        img_size = max(img.shape[:2])
        fig_size = min(12, max(8, img_size / 200))  # Dynamische Größe
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        ax.imshow(vis_img, cmap='gray')
        
        drawn_circles = 0
        total_area = 0
        
        for region in regions:
            if region.area > 10:  # Mindestgröße
                y, x = region.centroid
                radius = np.sqrt(region.area / np.pi)
                
                circle = Circle(
                    (x, y), 
                    radius, 
                    color=overlay_config.get('circle_color', 'red'),
                    fill=False, 
                    linewidth=overlay_config.get('line_width', 0.8),
                    alpha=overlay_config.get('circle_alpha', 0.8)
                )
                ax.add_patch(circle)
                drawn_circles += 1
                total_area += region.area
        
        # Erweiterte Titel-Information
        avg_area = total_area / drawn_circles if drawn_circles > 0 else 0
        total_cells = mask.max()
        
        title = f'Alle {drawn_circles} Zellen visualisiert (von {total_cells} gefunden)\n'
        title += f'Durchschnittliche Zellgröße: {avg_area:.1f} Pixel'
        
        ax.set_title(title, fontsize=12, pad=20)
        ax.axis('off')
        
        # DPI basierend auf Konfiguration
        dpi = 150 if overlay_config.get('high_resolution', True) else 100
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  📊 Overlay: {drawn_circles} von {total_cells} Zellen visualisiert (Ø{avg_area:.1f}px)")
        
    except Exception as e:
        print(f"Overlay-Fehler: {e}")

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
	create_overlays: bool = False,  # Neue Option
	channel: int = 0,  # Neuer Parameter
	filters: dict = None,  # Neuer Parameter
	overlay_config: dict = None,  # Neue Overlay-Konfiguration
) -> str:
	pos_files = _list_tiffs(input_pos) if os.path.isdir(input_pos) else []
	neg_files = _list_tiffs(input_neg) if os.path.isdir(input_neg) else []
	all_files = [(p, "pos") for p in pos_files] + [(p, "neg") for p in neg_files]
	if not all_files:
		raise FileNotFoundError("No TIFF files found.")

	# Setup output directories - nur wenn wirklich benötigt
	if save_masks:
		masks_dir = os.path.join(output_root, "masks")
		_ensure_dir(masks_dir)
		print(f"🔍 DEBUG: masks_dir wird erstellt: {masks_dir}")
	else:
		print("🔍 DEBUG: Masken werden NICHT gespeichert")
	
	if create_overlays and MATPLOTLIB_AVAILABLE:
		overlays_dir = os.path.join(output_root, "overlays")
		_ensure_dir(overlays_dir)
		print(f"🔍 DEBUG: overlays_dir wird erstellt: {overlays_dir}")
	elif create_overlays and not MATPLOTLIB_AVAILABLE:
		print("⚠️ Overlays angefordert, aber Matplotlib nicht verfügbar")
		create_overlays = False
	
	_ensure_dir(output_root)
	master_csv = os.path.join(output_root, "All_Counts_Master.csv")

	print(f"CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")

	# Debug: Masken-Speicherung
	print(f"🔍 DEBUG: save_masks = {save_masks}")

	model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=model_name)

	paths = [p for p, _ in all_files]
	print(f"Loading {len(paths)} images with {num_workers} workers (max {resize_max}px, channel {channel})")
	path_img_pairs = _parallel_load(paths, resize_max, num_workers, channel)
	imgs = [img for _, img in path_img_pairs]

	print(f"Inference: model={model_name}, batch_size={batch_size}, niter={niter}, flow_th={flow_threshold}, cellprob_th={cellprob_threshold}")
	all_masks: List[np.ndarray] = []
	for start in tqdm(range(0, len(imgs), batch_size), desc="Inference", unit="batch"):
		end = min(start + batch_size, len(imgs))
		batch = imgs[start:end]
		with torch.inference_mode():
			masks, flows, styles = model.eval(
				batch,
				diameter=None,
				channels=[0, 0],
				flow_threshold=flow_threshold,
				cellprob_threshold=cellprob_threshold,
				do_3D=False,
				normalize=True,
				resample=True,
				niter=niter,
				batch_size=len(batch),
			)
		# Debug: Masken-Ausgabe prüfen
		print(f"🔍 DEBUG: Batch {start//batch_size + 1}: masks type = {type(masks)}")
		if isinstance(masks, np.ndarray):
			print(f"🔍 DEBUG: masks shape = {masks.shape}, dtype = {masks.dtype}")
			masks = [masks]
		elif isinstance(masks, list):
			print(f"🔍 DEBUG: masks list length = {len(masks)}")
			if len(masks) > 0:
				print(f"🔍 DEBUG: first mask shape = {masks[0].shape}, dtype = {masks[0].dtype}")
		all_masks.extend(masks)

	# Debug: Finale Masken
	print(f"🔍 DEBUG: Gesamt {len(all_masks)} Masken erhalten")

	# Standard-Filter falls nicht angegeben
	if filters is None:
		filters = {
			'min_area': 10,
			'max_area': 5000,
			'min_circularity': 0.0,
			'max_circularity': 1.0,
			'min_solidity': 0.0,
			'max_solidity': 1.0
		}
	
	print(f"🔍 Filter aktiviert:")
	print(f"   Größe: {filters.get('min_area', 0)}-{filters.get('max_area', '∞')} Pixel")
	print(f"   Rundheit: {filters.get('min_circularity', 0):.2f}-{filters.get('max_circularity', 1):.2f}")
	print(f"   Kompaktheit: {filters.get('min_solidity', 0):.2f}-{filters.get('max_solidity', 1):.2f}")

	rows = []
	total_filtered = 0
	
	for i, ((path, cond), mask, (_, img)) in enumerate(tqdm(zip(all_files, all_masks, path_img_pairs), total=len(all_files), desc="Save/measure", unit="img")):
		base = os.path.splitext(os.path.basename(path))[0]
		
		# Speichere Maske nur wenn explizit gewünscht
		if save_masks:
			out_mask_path = os.path.join(masks_dir, f"{base}_mask.tiff")
			skio.imsave(out_mask_path, mask.astype(np.uint16), check_contrast=False)
		
		# Erstelle Overlay (Standard wenn Masken deaktiviert)
		if create_overlays:
			overlay_path = os.path.join(overlays_dir, f"{base}_overlay.png")
			_create_advanced_overlay(img, mask, overlay_path, overlay_config)
		
		cell_count, mean_area, mean_intensity, mean_int_density, filtered_count = _compute_metrics_with_filters(mask, img, filters)
		total_filtered += filtered_count
		
		print(f"🔍 {base}: {cell_count} gültige Zellen ({filtered_count} gefiltert)")
		
		rows.append({
			"filename": os.path.basename(path),
			"condition": cond,
			"region": "unknown",
			"channel": f"channel_{channel}",  # Verwende channel Parameter direkt
			"cell_count": cell_count,
			"filtered_count": filtered_count,
			"mean_area_per_cell": mean_area,
			"mean_intensity_per_cell": mean_intensity,
			"mean_integrated_density_per_cell": mean_int_density,
		})

	print(f"\n📊 FILTER-STATISTIK:")
	print(f"   Gesamt gefilterte Objekte: {total_filtered}")
	print(f"   Durchschnitt pro Bild: {total_filtered/len(all_files):.1f}")
	
	df = pd.DataFrame(rows)  # Korrekte Einrückung
	df.to_csv(master_csv, index=False)  # Korrekte Einrückung
	print(f"✅ Saved: {master_csv} ({len(df)} rows)")  # Korrekte Einrückung
	return master_csv  # Korrekte Einrückung