#!/usr/bin/env bash
set -euo pipefail

# start_remote.sh
# Einfaches Hilfsskript zum Bauen und Starten des Projekts mit HOST_DATA und UID/GID-Angleichung.

usage() {
  echo "Usage: $0 [HOST_DATA]"
  echo "If HOST_DATA is not provided, you'll be prompted." 
}

HOST_DATA_ARG="${1:-}"
GPU_FLAG="false"
if [ "$HOST_DATA_ARG" = "--gpu" ] || [ "$HOST_DATA_ARG" = "-g" ]; then
  GPU_FLAG="true"
  HOST_DATA_ARG="${2:-}"
fi
if [ -n "$HOST_DATA_ARG" ]; then
  HOST_DATA="$HOST_DATA_ARG"
else
  if [ -z "${HOST_DATA:-}" ]; then
    read -p "HOST_DATA nicht gesetzt. Pfad zu Host-Datenverzeichnis eingeben (z.B. /srv/cell_data) [./]: " HOST_DATA
    HOST_DATA="${HOST_DATA:-.}"
  else
    HOST_DATA="$HOST_DATA"
  fi
fi

HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"

export HOST_DATA HOST_UID HOST_GID

echo "Using HOST_DATA=$HOST_DATA"
echo "Using HOST_UID=$HOST_UID HOST_GID=$HOST_GID"

echo "Building docker image (this may take a while)..."
docker-compose build --build-arg USER_ID=${HOST_UID} --build-arg GROUP_ID=${HOST_GID}
echo "Starting container(s) with HOST_DATA mounted to /data..."
COMPOSE_FILES="-f docker-compose.yml"
if [ "$GPU_FLAG" = "true" ]; then
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"
fi

echo "Building docker image (this may take a while)..."
docker compose $COMPOSE_FILES build --build-arg USER_ID=${HOST_UID} --build-arg GROUP_ID=${HOST_GID}

# Start container
echo "Starting container(s) with HOST_DATA mounted to /data..."
HOST_DATA=${HOST_DATA} HOST_UID=${HOST_UID} HOST_GID=${HOST_GID} docker compose $COMPOSE_FILES up -d

echo "Containers started. Access the web UI at http://<host>:5000"

echo "To follow logs: docker-compose logs -f"

echo "If you need to stop: docker-compose down"
