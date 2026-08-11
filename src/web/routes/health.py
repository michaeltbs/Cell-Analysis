"""
src/web/routes/health.py — Flask blueprint for health/version endpoints.
"""
from __future__ import annotations

from flask import Blueprint, jsonify

bp = Blueprint("health", __name__, url_prefix="/api")


@bp.get("/health")
def health():
    return jsonify({"status": "ok", "backend": "flask"})


@bp.get("/version")
def version():
    return jsonify({"version": "0.1.0", "backend": "flask"})
