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
# Use an official PyTorch CUDA base image to avoid re-installing large PyTorch wheels
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install system deps required for image processing libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc git curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    libjpeg-turbo8 libtiff5 libopenjp2-7 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (PyTorch is already present in base image)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

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
