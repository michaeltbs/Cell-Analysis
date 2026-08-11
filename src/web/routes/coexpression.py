"""
src/web/routes/coexpression.py — Flask blueprint for co-expression routes.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.api.pipeline_service import run_coexpression

bp = Blueprint("coexpression", __name__, url_prefix="/api/coexpr")


@bp.route("/start", methods=["POST"])
def start():
    data = request.get_json() or {}
    cfg_path = data.get("config_path", "config_analysis.yaml")
    try:
        result = run_coexpression(cfg_path)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/status", methods=["GET"])
def status():
    return jsonify({"running": False})
