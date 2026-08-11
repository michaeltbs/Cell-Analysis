"""
src/web/app.py — minimal Flask app using blueprints.

Replaces the legacy monolithic app.py. The old app.py remains available as
app_legacy.py for reference during transition.
"""
from __future__ import annotations

from flask import Flask, render_template

from src.web.routes import detection, coexpression, health


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.register_blueprint(detection.bp)
    app.register_blueprint(coexpression.bp)
    app.register_blueprint(health.bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
