FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# System deps for opencv, imagecodecs, skimage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    libjpeg62-turbo libtiff5 libopenjp2-7 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU PyTorch stack (compatible with py3.10)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . /app

EXPOSE 5000
CMD ["python", "app.py"]
