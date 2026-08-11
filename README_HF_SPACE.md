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

## Hardware

Upgrade this Space to a GPU instance in the Settings tab for faster inference.
Recommended: Nvidia T4 small / medium.

## Usage

Upload a CZI or multi-channel TIFF, then call `/pipeline/full` with a detection config.
