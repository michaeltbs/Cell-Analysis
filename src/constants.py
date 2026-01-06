import os

DEFAULT_AI_CONFIG = {
    'base_url': os.environ.get('OPENWEBUI_BASE_URL', 'http://localhost:8090'),
    'api_key': os.environ.get('OPENWEBUI_API_KEY', ''),
    'model_id': os.environ.get('OPENWEBUI_MODEL_ID', '5ce5a9ac1cc0'),
    'temperature': float(os.environ.get('OPENWEBUI_TEMPERATURE', 0.2)),
    'max_tokens': int(os.environ.get('OPENWEBUI_MAX_TOKENS', 800)),
}

SENSITIVITY_LEVELS = [
    # Strenge Level (1-5): Für sehr helle, klar definierte Zellen
    {"label": "Level 1 - Ultra Strict", "cellprob": 0.80, "flow": 0.95, "snr": 4.5, "floor_pct": 97, "abs_int": 45, "itf": 0.15, "int_mode": "global", "max_int_thr": 0.50},
    {"label": "Level 2 - Very Strict", "cellprob": 0.72, "flow": 0.88, "snr": 4.0, "floor_pct": 95, "abs_int": 42, "itf": 0.22, "int_mode": "global", "max_int_thr": 0.45},
    {"label": "Level 3 - Strict", "cellprob": 0.64, "flow": 0.80, "snr": 3.5, "floor_pct": 92, "abs_int": 38, "itf": 0.30, "int_mode": "global", "max_int_thr": 0.40},
    {"label": "Level 4 - Semi-Strict", "cellprob": 0.55, "flow": 0.70, "snr": 3.2, "floor_pct": 88, "abs_int": 34, "itf": 0.40, "int_mode": "global", "max_int_thr": 0.35},
    {"label": "Level 5 - Moderate-Strict", "cellprob": 0.48, "flow": 0.62, "snr": 2.9, "floor_pct": 84, "abs_int": 30, "itf": 0.50, "int_mode": "global", "max_int_thr": 0.30},
    # Balanced Level (6-8): Standard-Einstellungen
    {"label": "Level 6 - Balanced", "cellprob": 0.40, "flow": 0.52, "snr": 2.6, "floor_pct": 78, "abs_int": 26, "itf": 0.62, "int_mode": "global", "max_int_thr": 0.25},
    {"label": "Level 7 - Balanced+", "cellprob": 0.32, "flow": 0.44, "snr": 2.4, "floor_pct": 72, "abs_int": 23, "itf": 0.75, "int_mode": "global", "max_int_thr": 0.20},
    {"label": "Level 8 - Moderate", "cellprob": 0.24, "flow": 0.36, "snr": 2.1, "floor_pct": 66, "abs_int": 20, "itf": 0.88, "int_mode": "global", "max_int_thr": 0.16},
    # Sensitive Level (9-12): Für schwächere Signale
    {"label": "Level 9 - Medium-Sensitive", "cellprob": 0.16, "flow": 0.28, "snr": 1.9, "floor_pct": 60, "abs_int": 17, "itf": 1.00, "int_mode": "global", "max_int_thr": 0.12},
    {"label": "Level 10 - Sensitive", "cellprob": 0.08, "flow": 0.20, "snr": 1.6, "floor_pct": 54, "abs_int": 14, "itf": 1.15, "int_mode": "global", "max_int_thr": 0.09},
    {"label": "Level 11 - High Sensitivity", "cellprob": 0.00, "flow": 0.12, "snr": 1.4, "floor_pct": 48, "abs_int": 11, "itf": 1.28, "int_mode": "global", "max_int_thr": 0.06},
    {"label": "Level 12 - Very High", "cellprob": -0.08, "flow": 0.04, "snr": 1.2, "floor_pct": 42, "abs_int": 9, "itf": 1.40, "int_mode": "global", "max_int_thr": 0.04},
    # Ultra-Sensitive Level (13-15): Für sehr schwache Signale
    {"label": "Level 13 - Ultra Sensitive", "cellprob": -0.16, "flow": -0.04, "snr": 1.0, "floor_pct": 36, "abs_int": 7, "itf": 1.55, "int_mode": "global", "max_int_thr": 0.03},
    {"label": "Level 14 - Maximum", "cellprob": -0.24, "flow": -0.12, "snr": 0.8, "floor_pct": 30, "abs_int": 5, "itf": 1.70, "int_mode": "global", "max_int_thr": 0.02},
    {"label": "Level 15 - Absolute Max", "cellprob": -0.32, "flow": -0.20, "snr": 0.5, "floor_pct": 24, "abs_int": 3, "itf": 1.85, "int_mode": "global", "max_int_thr": 0.01},
]

AI_CONFIG_PATH = 'config_ai.yaml'
DET_CONFIG_PATH = 'config.yaml'
DET_CPSAM_CONFIG_PATH = 'config_det_cpsam.yaml'
CZI_CONFIG_PATH = 'config_czi.yaml'
