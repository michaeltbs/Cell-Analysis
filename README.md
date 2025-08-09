# Cell Analysis Pipeline

GPU-beschleunigte Segmentierung (Cellpose) und statistische Auswertung.

## Installation
pip install -r requirements.txt

## Konfiguration
Pfade und Parameter in config.yaml anpassen.

## Ausführung
python run_pipeline.py
# ohne Masken
python run_pipeline.py --no-masks
# ohne Summary
python run_pipeline.py --no-summary

## Output
- results/All_Counts_Master.csv
- results/analysis_results_animal_averages.csv
- results/analysis_results_condition_averages.csv
- results/analysis_results_summary.md
- results/masks/*.tiff (optional)

## Struktur
- run_pipeline.py
- src/batch_segment.py
- src/analysis.py
- config.yaml
- requirements.txt