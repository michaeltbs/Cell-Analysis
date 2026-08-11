"""
src/web/routes/detection.py — Flask blueprint for detection routes.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.api.pipeline_service import run_detection

bp = Blueprint("detection", __name__, url_prefix="/api/det")


@bp.route("/start", methods=["POST"])
def start():
    data = request.get_json() or {}
    cfg_path = data.get("config_path", "config.yaml")
    try:
        result = run_detection(
            cfg_path=cfg_path,
            save_masks=data.get("save_masks", True),
            test_mode=data.get("test_mode", False),
            test_samples=data.get("test_samples"),
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/status", methods=["GET"])
def status():
    return jsonify({"running": False})
