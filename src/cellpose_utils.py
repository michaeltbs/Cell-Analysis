import sys
import torch
from typing import Optional, Tuple
from cellpose import models


def _get_gpu_device() -> Tuple[torch.device, bool]:
    """Detect and return the best available GPU device config.
    
    Priority:
    1. Apple Silicon (MPS) - Metal Performance Shaders
    2. NVIDIA (CUDA)
    3. CPU (fallback)
    
    Returns:
        tuple: (device, is_gpu_available)
    """
    # Priorität 1: Apple Silicon (MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("[INFO] Apple Silicon GPU (MPS) erkannt und verfügbar.")
        return torch.device("mps"), True
    
    # Priorität 2: NVIDIA (CUDA)
    if torch.cuda.is_available():
        print(f"[INFO] NVIDIA GPU (CUDA) erkannt: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda"), True
    
    # Fallback: CPU
    print("[WARN] Keine GPU erkannt. Nutze CPU (Performance wird deutlich geringer sein).")
    return torch.device("cpu"), False


def load_cellpose_model(model_name: str, use_gpu: bool = True):
    """Load Cellpose model with explicit device configuration for M3/M2/Apple Silicon.
    
    This function ensures that:
    - MPS (Apple Silicon) is explicitly used when available
    - CUDA (NVIDIA) is used as secondary option
    - CPU fallback is graceful
    
    Args:
        model_name: Model type (cyto2, cyto3, cyto, nuclei, etc.)
        use_gpu: Whether to attempt GPU acceleration
    
    Returns:
        CellposeModel instance configured for the detected device
    """
    model_type = (model_name or "cyto2").lower()
    if model_type not in ("cyto3", "cyto2", "cyto", "nuclei"):
        model_type = "cyto2"
    
    # Device ermitteln
    device, gpu_available = _get_gpu_device()
    
    # Wenn der User GPU will, aber keine da ist -> Fallback
    if use_gpu and not gpu_available:
        print("[WARN] GPU angefordert, aber nicht verfügbar. Fallback auf CPU.")
        device = torch.device("cpu")
        use_gpu = False

    print(f"[INIT] Lade Cellpose Modell '{model_type}' auf Device: {device}")

    try:
        # Für Cellpose 3.0+: Wir übergeben das 'device' Objekt direkt.
        # WICHTIG: Wir setzen gpu=False im Konstruktor, um Cellpose's interne CUDA-Checks 
        # zu umgehen, da wir das Device manuell kontrollieren.
        model = models.CellposeModel(
            gpu=False,  # WICHTIG: False, damit Cellpose nicht nach CUDA sucht
            model_type=model_type,
            device=device
        )
        print(f"[SUCCESS] Cellpose Modell erfolgreich auf {device} geladen.")
        return model
    except TypeError as e:
        # Falls ältere Cellpose-Version kein 'device' Parameter akzeptiert
        print(f"[WARN] Cellpose akzeptiert 'device' Parameter nicht (ältere Version?). Fallback auf gpu-Flag.")
        model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        return model
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden des Modells: {e}")
        try:
            return models.Cellpose(gpu=False, model_type=model_type)
        except Exception as e2:
            print(f"[ERROR] Auch Fallback fehlgeschlagen: {e2}")
            raise

