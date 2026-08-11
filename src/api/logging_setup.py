"""
src/api/logging_setup.py — configure Python logging for the API layer.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the cell_analysis logger."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
        )
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger("cell_analysis")


def get_logger(name: str = "cell_analysis") -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
