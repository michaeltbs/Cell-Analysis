########################################
# Multi-target Dockerfile: cpu & gpu
########################################

# -------------------- CPU target --------------------
FROM python:3.10-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# System deps for opencv, imagecodecs, skimage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    libjpeg-turbo8 libtiff5 libopenjp2-7 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU PyTorch stack (compatible with py3.10)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . /app

# Create a non-root user to avoid permission issues when host volumes are mounted.
# Build-time args allow overriding UID/GID: `--build-arg USER_ID=1000 --build-arg GROUP_ID=1000`
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup || true \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser || true \
    && chown -R appuser:appgroup /app /root || true

USER appuser

EXPOSE 5000
CMD ["python", "app.py"]


# -------------------- GPU target --------------------
# This stage uses NVIDIA CUDA runtime as base and installs a CUDA-capable PyTorch.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install python3 and system deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential gcc git curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    libjpeg-turbo8 libtiff5 libopenjp2-7 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip refers to python3
RUN ln -s /usr/bin/python3 /usr/bin/python || true

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-enabled PyTorch (adjust version if needed). This uses the official PyTorch CUDA 11.8 wheels.
RUN pip install --no-cache-dir \
    torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy source
COPY . /app

# Create non-root user in GPU image as well
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup || true \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser || true \
    && chown -R appuser:appgroup /app /root || true

USER appuser

EXPOSE 5000
CMD ["python", "app.py"]
