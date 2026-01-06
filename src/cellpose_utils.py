import sys
from typing import Optional

def _get_gpu_device():
    """Detect and return the best available GPU device for PyTorch/Cellpose.
    
    Returns:
        tuple: (device_name, is_available) where device_name is 'cuda', 'mps', or None
    """
    try:
        import torch
    except ImportError:
        return None, False
    
    # Check CUDA first (NVIDIA GPUs)
    if torch.cuda.is_available():
        return 'cuda', True
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', True
    
    return None, False


def load_cellpose_model(model_name: str, use_gpu: bool = False):
    """Return a Cellpose model that works on v4+, with fallback for older versions.
    
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends.
    """
    try:
        from cellpose import models as _models
    except Exception as e:
        raise RuntimeError("Cellpose is not installed. Please `pip install cellpose`.") from e

    name = (model_name or "cyto2").lower()
    mtype = name if name in ("cyto3", "cyto2", "cyto", "nuclei") else "cyto2"

    # Determine GPU device
    gpu_device, gpu_available = _get_gpu_device()
    
    if use_gpu and not gpu_available:
        print("[WARN] GPU requested but no GPU backend available. Using CPU.")
        use_gpu = False

    # prefer v4 API
    try:
        # Cellpose 3.x+ accepts device parameter
        model_kwargs = {'gpu': bool(use_gpu), 'model_type': mtype}
        
        # Set device for MPS support (Cellpose 3.x+)
        if use_gpu and gpu_device:
            try:
                import torch
                if gpu_device == 'mps':
                    model_kwargs['device'] = torch.device('mps')
                    print("[INFO] CellposeModel using Apple Silicon MPS GPU acceleration.")
                elif gpu_device == 'cuda':
                    model_kwargs['device'] = torch.device('cuda')
                    print("[INFO] CellposeModel using NVIDIA CUDA GPU acceleration.")
            except Exception:
                pass  # Fall back to default device handling
        
        model = _models.CellposeModel(**model_kwargs)
        
        if use_gpu and not getattr(model, "gpu", False):
            print("[WARN] CellposeModel requested GPU but fell back to CPU. Check GPU visibility.")
        elif use_gpu and gpu_device:
            pass  # Already printed device info above
        elif use_gpu:
            print("[INFO] CellposeModel running with GPU acceleration.")
        return model
    except (AttributeError, TypeError):
        # fallback for older versions (no device parameter)
        model = _models.Cellpose(gpu=bool(use_gpu), model_type=mtype)
        if use_gpu and not getattr(model, "gpu", False):
            print("[WARN] Cellpose GPU fallback is not active; running on CPU.")
        elif use_gpu:
            print("[INFO] Cellpose fallback running with GPU acceleration.")
        return model

