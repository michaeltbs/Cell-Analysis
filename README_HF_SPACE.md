---
title: Cell Analysis
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Cell Analysis HF Space

GPU-accelerated cell segmentation and co-expression analysis for confocal microscopy.

## Endpoints

- `GET /health`
- `GET /version`
- `POST /pipeline/detection`
- `POST /pipeline/coexpression`
- `POST /pipeline/full`
- `POST /jobs/upload` — upload CZI/TIFF files and run full pipeline asynchronously
- `GET /jobs/{job_id}` — poll job status/result
- `GET /jobs` — list recent jobs

## Hardware

Upgrade this Space to a GPU instance in the Settings tab for faster inference.
Recommended: Nvidia T4 small / medium.

## Usage

Upload CZI or multi-channel TIFF files via `/jobs/upload`, then poll `/jobs/{job_id}` until status is `success`.

Example:
```bash
curl -X POST https://YOUR_SPACE.hf.space/jobs/upload \
  -F "files=@sample.czi" \
  -F "channels=0,1,2,3" \
  -F "model_name=cyto2" \
  -F "use_gpu=true"
```

Response: `{"job_id": "...", "status": "pending"}`

Then poll:
```bash
curl https://YOUR_SPACE.hf.space/jobs/YOUR_JOB_ID
```
