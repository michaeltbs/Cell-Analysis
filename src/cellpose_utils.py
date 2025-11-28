import sys
from typing import Optional

def load_cellpose_model(model_name: str, use_gpu: bool = False):
    """Return a Cellpose model that works on v4+, with fallback for older versions."""
    try:
        from cellpose import models as _models
    except Exception as e:
        raise RuntimeError("Cellpose is not installed. Please `pip install cellpose`.") from e

    name = (model_name or "cyto2").lower()
    mtype = name if name in ("cyto3", "cyto2", "cyto", "nuclei") else "cyto2"

    # prefer v4 API
    try:
        model = _models.CellposeModel(gpu=bool(use_gpu), model_type=mtype)
        if use_gpu and not getattr(model, "gpu", False):
            print("[WARN] CellposeModel requested GPU but fell back to CPU. Check CUDA visibility.")
        elif use_gpu:
            print("[INFO] CellposeModel running with GPU acceleration.")
        return model
    except AttributeError:
        # fallback for older versions
        model = _models.Cellpose(gpu=bool(use_gpu), model_type=mtype)
        if use_gpu and not getattr(model, "gpu", False):
            print("[WARN] Cellpose GPU fallback is not active; running on CPU.")
        elif use_gpu:
            print("[INFO] Cellpose fallback running with GPU acceleration.")
        return model
