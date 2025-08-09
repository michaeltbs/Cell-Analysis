import os
import re
import pandas as pd
from typing import Optional

# Schema validation constants
REQUIRED_BASE_COLS = ["filename", "condition", "region", "channel"]
NUMERIC_COLS_RAW = [
	"cell_count",
	"mean_area_per_cell", 
	"mean_intensity_per_cell",
	"mean_integrated_density_per_cell",
]

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
	"""Validate and clean CSV schema"""
	# Ensure required columns exist
	for col in REQUIRED_BASE_COLS:
		if col not in df.columns:
			df[col] = "unknown" if col in ["region", "channel"] else ""
	
	# Clean string columns
	df["filename"] = df["filename"].astype(str)
	df["condition"] = df["condition"].astype(str).str.lower().replace({
		"positive": "pos", "negative": "neg"
	})
	df["region"] = df["region"].astype(str)
	df["channel"] = df["channel"].astype(str)

	# Ensure numeric columns exist
	for col in NUMERIC_COLS_RAW:
		if col not in df.columns:
			df[col] = 0

	# Convert numeric columns safely
	for col in NUMERIC_COLS_RAW:
		df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

	# Remove duplicates
	df = df.drop_duplicates(subset=["filename", "region", "channel"], keep="first")
	return df

def extract_animal_id(filename: str) -> str:
	"""Extract animal ID from filename (e.g., M009_1.tiff -> M009)"""
	match = re.match(r'(M\d+)_\d+', filename)
	return match.group(1) if match else "unknown"

def calculate_animal_averages(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
	"""Calculate per-animal averages from master CSV"""
	# Load CSV
	try:
		df = pd.read_csv(csv_path, engine="pyarrow")
	except Exception:
		df = pd.read_csv(csv_path)
	
	# Validate schema
	df = _ensure_schema(df)
	
	# Extract animal IDs
	df['animal_id'] = df['filename'].apply(extract_animal_id)
	
	# Group by animal, condition, region, channel
	grouped = df.groupby(['animal_id', 'condition', 'region', 'channel'])
	
	# Calculate means
	metrics = ['cell_count', 'mean_area_per_cell', 'mean_intensity_per_cell', 'mean_integrated_density_per_cell']
	averages = grouped[metrics].mean().reset_index()
	
	# Add image counts
	image_counts = grouped.size().reset_index(name='image_count')
	averages = pd.merge(averages, image_counts, on=['animal_id', 'condition', 'region', 'channel'])
	
	# Add standard deviations
	std_dev = grouped[metrics].std().reset_index()
	std_dev.columns = [col if col in ['animal_id', 'condition', 'region', 'channel'] 
					   else f"{col}_std" for col in std_dev.columns]
	averages = pd.merge(averages, std_dev, on=['animal_id', 'condition', 'region', 'channel'])
	
	# Rename columns for clarity
	averages.rename(columns={
		'cell_count': 'avg_cell_count',
		'mean_area_per_cell': 'avg_area_per_cell',
		'mean_intensity_per_cell': 'avg_intensity_per_cell',
		'mean_integrated_density_per_cell': 'avg_integrated_density_per_cell'
	}, inplace=True)
	
	# Sort results
	averages = averages.sort_values(['animal_id', 'channel'])
	
	# Save if requested
	if output_path:
		averages.to_csv(output_path, index=False)
	
	return averages

def calculate_condition_averages(animal_averages: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
	"""Calculate condition-level averages from animal averages"""
	# Group by condition and channel
	grouped = animal_averages.groupby(['condition', 'region', 'channel'])
	
	# Calculate means
	metrics = ['avg_cell_count', 'avg_area_per_cell', 'avg_intensity_per_cell', 'avg_integrated_density_per_cell']
	condition_avgs = grouped[metrics].mean().reset_index()
	
	# Add animal counts
	animal_counts = animal_averages.groupby(['condition', 'region', 'channel'])['animal_id'].nunique().reset_index(name='animal_count')
	condition_avgs = pd.merge(condition_avgs, animal_counts, on=['condition', 'region', 'channel'])
	
	# Add standard deviations
	std_dev = grouped[metrics].std().reset_index()
	std_dev.columns = [col if col in ['condition', 'region', 'channel'] 
					   else f"{col}_std" for col in std_dev.columns]
	condition_avgs = pd.merge(condition_avgs, std_dev, on=['condition', 'region', 'channel'])
	
	# Rename columns
	condition_avgs.rename(columns={
		'avg_cell_count': 'condition_avg_cell_count',
		'avg_area_per_cell': 'condition_avg_area_per_cell',
		'avg_intensity_per_cell': 'condition_avg_intensity_per_cell',
		'avg_integrated_density_per_cell': 'condition_avg_integrated_density_per_cell'
	}, inplace=True)
	
	# Sort results
	condition_avgs = condition_avgs.sort_values(['condition', 'channel'])
	
	# Save if requested
	if output_path:
		condition_avgs.to_csv(output_path, index=False)
	
	return condition_avgs

def create_summary_report(animal_averages: pd.DataFrame, condition_averages: pd.DataFrame, 
						 output_path: str) -> None:
	"""Create markdown summary report"""
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("# Zusammenfassung\n\n")
		f.write(f"- Tier-Mittelwerte: {len(animal_averages)} Einträge\n")
		f.write(f"- Bedingungsmittelwerte: {len(condition_averages)} Einträge\n\n")
		
		f.write("## POS vs. NEG (Zellzahl) je Kanal\n")
		for channel in sorted(condition_averages['channel'].unique()):
			pos_d = condition_averages[
				(condition_averages['condition'] == 'pos') & 
				(condition_averages['channel'] == channel)
			]
			neg_d = condition_averages[
				(condition_averages['condition'] == 'neg') & 
				(condition_averages['channel'] == channel)
			]
			
			if not pos_d.empty and not neg_d.empty:
				pos_c = pos_d['condition_avg_cell_count'].values[0]
				neg_c = neg_d['condition_avg_cell_count'].values[0]
				diff = ((pos_c - neg_c) / neg_c * 100) if neg_c != 0 else float('inf')
				f.write(f"- Kanal {channel}: POS={pos_c:.2f}, NEG={neg_c:.2f}, Δ={diff:.1f}%\n")

def create_summary_csv(csv_path: str, base_output_name: str = "analysis_results", write_markdown: bool = True) -> None:
	"""Wrapper function to create summary CSV and optional markdown report"""
	out_dir = os.path.dirname(csv_path)
	animal_out = os.path.join(out_dir, f"{base_output_name}_animal_averages.csv")
	cond_out = os.path.join(out_dir, f"{base_output_name}_condition_averages.csv")
	animal = calculate_animal_averages(csv_path, animal_out)
	if animal.empty:
		return
	cond = calculate_condition_averages(animal, cond_out)
	if write_markdown:
		md = os.path.join(out_dir, f"{base_output_name}_summary.md")
		create_summary_report(animal, cond, md)