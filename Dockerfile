# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
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

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project metadata
COPY pyproject.toml ./

# Install project dependencies + CUDA-enabled PyTorch in one layer
# This avoids dependency conflicts between torch CPU and CUDA wheels.
RUN python3 -m pip install --no-cache-dir \
    -e ".[dev]" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# Copy application code
COPY . .

# HF Spaces default port is 7860; we use 8000 consistently
EXPOSE 8000

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run FastAPI app
CMD ["python3", "-m", "uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
