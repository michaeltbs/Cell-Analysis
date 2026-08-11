"""src/web/routes/__init__.py"""
from .detection import bp as detection_bp
from .coexpression import bp as coexpression_bp
from .health import bp as health_bp

__all__ = ["detection_bp", "coexpression_bp", "health_bp"]
