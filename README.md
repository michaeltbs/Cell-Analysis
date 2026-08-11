# Cell Analysis

Web-App und API für automatisierte Zellanalyse von Konfokalmikroskopie-Bildern.
Bachelorarbeit-Projekt: Segmentierung von PomC / Glp1r / Gal Kanälen.

## Pipeline

1. **CZI → TIFF** (`src/czi_converter.py`)
2. **Zell-Detection** (`src/batch_segment.py`, Cellpose/CPSAM)
3. **Co-Expression-Analyse** (`src/coexpression.py`)
4. **Statistik & Export** (`src/analysis.py`, `src/anova_analysis.py`)
5. **Validierung** (`src/validation/metrics.py`, `src/validation/expression.py`)
6. **Benennung** (`src/config/naming.py`)

## Schnelleinstieg

```bash
# 1. Venv aufbauen
python3 -m venv .venv
source .venv/bin/activate

# 2. Pakete installieren
pip install -e ".[dev]"

# 3. Tests laufen lassen
pytest tests/

# 4. FastAPI-Server starten
python3 -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8001
```

## APIs

### FastAPI (empfohlen für App #3)

- `GET  /health`
- `GET  /version`
- `POST /pipeline/detection`
- `POST /pipeline/coexpression`
- `POST /pipeline/full`
- `POST /jobs/upload` — Dateien hochladen + Pipeline als Job starten
- `GET  /jobs/{job_id}` — Job-Status abfragen
- `GET  /jobs` — Jobs auflisten

Beispiel Upload:
```bash
curl -X POST http://localhost:8001/jobs/upload \
  -F "files=@sample.czi" \
  -F "channels=0,1,2,3" \
  -F "model_name=cyto2" \
  -F "use_gpu=true" \
  -F "channel_names=PomC,DREADD,Fos,Gal"
```

### Flask (legacy Web-UI)

```bash
python src/web/app.py
```

Port 5000. Neue Blueprints-Struktur unter `src/web/routes/`.

## Projektstruktur

```
Cell-Analysis/
├── src/
│   ├── api/                 # Headless API-Layer
│   │   ├── pipeline_service.py
│   │   ├── fastapi_app.py
│   │   ├── jobs.py          # async jobs
│   │   └── upload_service.py
│   ├── batch_segment.py     # Zellsegmentierung
│   ├── coexpression.py      # Co-Expression
│   ├── czi_converter.py     # CZI → TIFF
│   ├── analysis.py          # Statistik
│   ├── validation/          # Validierungsmetriken
│   │   ├── metrics.py
│   │   └── expression.py    # expression % + distance-map
│   ├── config/
│   │   └── naming.py        # universelle Benennung
│   └── web/                 # Flask Web-UI (Blueprints)
├── tests/                   # Tests + Fixtures
├── config.yaml              # Detection-Config (Beispiel)
├── config_analysis.yaml     # Co-Expression-Config
├── config_czi.yaml          # CZI-Config
├── config_det_cpsam.yaml    # CPSAM-Config
├── Dockerfile               # HF GPU Space
├── README_HF_SPACE.md       # HF Space Metadata
├── pyproject.toml           # Dependencies
└── ROADMAP.md               # Geplante Features
```

## Konfiguration

Alle Configs nutzen relative Pfade (`./data/...`).
Passe sie an dein System an oder überschreibe per API-Request.

### Wichtige Env-Variablen

- `OPENWEBUI_BASE_URL` — für AI-Assistant (optional)
- `OPENWEBUI_API_KEY` — für AI-Assistant (nicht in Dateien speichern!)
- `HOST_DATA` — Docker-Host-Datenpfad

## Tests

```bash
pytest tests/ -v
```

## Validierung

`src/validation/metrics.py` bietet:
- `mask_iou(a, b)`
- `pixel_level_metrics(pred, gt)`
- `instance_detection_metrics(pred_labels, gt_labels, iou_threshold=0.5)`
- `evaluate_ground_truth_csv(pred_csv, gt_csv)`

`src/validation/expression.py` bietet:
- `expression_percentage(mask, intensity_image=None)`
- `expression_percentage_per_instance(labels, intensity_images)`
- `receptor_distance_map(receptor_mask, cell_mask)`

## Hugging Face GPU Space

```bash
# Space mit diesem Repo verbinden
git remote add hf https://huggingface.co/spaces/DEIN_USERNAME/cell-analysis
git push hf app3-refactor:main
```

In den Space-Settings auf GPU (T4 small) upgraden.

## Docker

CPU:
```bash
docker compose up -d --build
```

GPU (NVIDIA):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Sicherheit

- API-Keys niemals in `config_ai.yaml` oder Git speichern.
- `config_ai.yaml` nutzt `api_key: ""` — Key per `OPENWEBUI_API_KEY` setzen.

## Lizenz

Privat / Bachelorarbeit. Nicht öffentlich ohne Freigabe.
