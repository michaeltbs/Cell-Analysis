import os
import argparse
import yaml
import torch

from src.batch_segment import segment_dirs
from src.analysis import calculate_animal_averages, calculate_condition_averages, create_summary_report

def _print_env():
	print(f"PyTorch: {torch.__version__}")
	print(f"CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"GPU: {torch.cuda.get_device_name(0)}")

def _load_cfg(p):
	with open(p, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)

def main():
	parser = argparse.ArgumentParser(description="Cell Analysis Pipeline")
	parser.add_argument("--config", type=str, default="config.yaml")
	parser.add_argument("--no-masks", action="store_true")
	parser.add_argument("--no-summary", action="store_true")
	parser.add_argument("--create-overlays", action="store_true", help="Erstelle Overlay-Bilder mit markierten Zellen")
	parser.add_argument("--no-overlays", action="store_true", help="Keine Overlays erstellen")
	args = parser.parse_args()

	if not os.path.exists(args.config):
		raise FileNotFoundError(f"Konfiguration nicht gefunden: {args.config}")

	cfg = _load_cfg(args.config)
	_print_env()

	# Debug: Konfiguration prüfen
	print(f"🔍 Config Debug:")
	print(f"   save_masks in config: {cfg.get('cellpose', {}).get('save_masks', True)}")
	print(f"   --no-masks flag: {args.no_masks}")
	final_save_masks = cfg.get("cellpose", {}).get("save_masks", True) and not args.no_masks
	print(f"   Finale save_masks: {final_save_masks}")
	
	# Region splitting check
	split_regions = cfg.get("cellpose", {}).get("split_regions", False)
	print(f"   split_regions: {split_regions}")
	
	# Overlay-Logik verbessern
	if args.no_overlays:
		create_overlays_final = False
		print(f"   create_overlays: False (explizit deaktiviert)")
	elif args.create_overlays:
		create_overlays_final = True
		print(f"   create_overlays: True (explizit aktiviert)")
	elif not final_save_masks:
		create_overlays_final = True
		print(f"   create_overlays: True (auto-aktiviert, da keine Masken)")
	else:
		create_overlays_final = False
		print(f"   create_overlays: False (Standard)")

	master_csv = segment_dirs(
		input_pos=cfg["paths"].get("input_pos", cfg["paths"].get("input_old", "")),
		input_neg=cfg["paths"].get("input_neg", cfg["paths"].get("input_adult", "")),
		output_root=cfg["paths"]["output_root"],
		batch_size=cfg.get("cellpose", {}).get("batch_size", 8),
		resize_max=cfg.get("cellpose", {}).get("resize_max", 1000),
		niter=cfg.get("cellpose", {}).get("niter", 250),
		flow_threshold=cfg.get("cellpose", {}).get("flow_threshold", 0.4),
		cellprob_threshold=cfg.get("cellpose", {}).get("cellprob_threshold", 0.0),
		num_workers=cfg.get("cpu", {}).get("num_workers", 4),  # Robuster Zugriff
		model_name=cfg.get("cellpose", {}).get("model_name", "cpsam"),
		save_masks=final_save_masks,
		create_overlays=create_overlays_final,
		channel=cfg.get("cellpose", {}).get("channel", 0),
		filters=cfg.get("filters", {}),  # Neue Filter-Parameter
		overlay_config=cfg.get("overlays", {}),  # Neue Overlay-Konfiguration
		split_regions=split_regions,  # Fehlender Parameter hinzugefügt
		# NEW: advanced filtering
		enable_advanced_filtering=cfg.get("enable_advanced_filtering", False),
		advanced_filtering=cfg.get("advanced_filtering", {}),
		processing=cfg.get("processing", {}),  # Pass processing options
	)

	out_dir = os.path.dirname(master_csv)
	base = cfg.get("outputs", {}).get("base_name", "analysis_results")

	animal_csv = os.path.join(out_dir, f"{base}_animal_averages.csv")
	cond_csv = os.path.join(out_dir, f"{base}_condition_averages.csv")

	animal = calculate_animal_averages(master_csv, animal_csv)
	cond = calculate_condition_averages(animal, cond_csv)

	if not args.no_summary:
		summary = os.path.join(out_dir, f"{base}_summary.md")
		create_summary_report(animal, cond, summary)
		print(f"✅ Summary: {summary}")

	print("✅ Pipeline fertig")

if __name__ == "__main__":
	main()