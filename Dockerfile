# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# System dependencies for Python + image processing + Cellpose
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install cellpose and python deps (without torch, will be installed with CUDA next)
COPY pyproject.toml ./
RUN python3 -m pip install --no-cache-dir -e ".[dev]" --no-deps \
    || true

# Install CUDA-enabled PyTorch (CUDA 12.1 matches the base image)
RUN python3 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Re-install project deps now that torch is present
RUN python3 -m pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . .

# HF Spaces expects the app to listen on 7860 by default, but we expose 8000
EXPOSE 8000

# Run FastAPI app
CMD ["python3", "-m", "uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
