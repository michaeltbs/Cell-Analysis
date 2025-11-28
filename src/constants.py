import os

DEFAULT_AI_CONFIG = {
    'base_url': os.environ.get('OPENWEBUI_BASE_URL', 'http://localhost:8090'),
    'api_key': os.environ.get('OPENWEBUI_API_KEY', ''),
    'model_id': os.environ.get('OPENWEBUI_MODEL_ID', '5ce5a9ac1cc0'),
    'temperature': float(os.environ.get('OPENWEBUI_TEMPERATURE', 0.2)),
    'max_tokens': int(os.environ.get('OPENWEBUI_MAX_TOKENS', 800)),
}

SENSITIVITY_LEVELS = [
    {"label": "Level 1 - Strict", "cellprob": 0.64, "flow": 0.80, "snr": 3.30, "floor_pct": 92, "abs_int": 35, "itf": 0.36},
    {"label": "Level 2 - Semi-strict", "cellprob": 0.5511, "flow": 0.6978, "snr": 3.03, "floor_pct": 85.8, "abs_int": 31.89, "itf": 0.4889},
    {"label": "Level 3 - Balanced", "cellprob": 0.4622, "flow": 0.5956, "snr": 2.77, "floor_pct": 79.6, "abs_int": 28.78, "itf": 0.6178},
    {"label": "Level 4 - Balanced+", "cellprob": 0.3733, "flow": 0.4933, "snr": 2.50, "floor_pct": 73.3, "abs_int": 25.67, "itf": 0.7467},
    {"label": "Level 5 - Default", "cellprob": 0.2844, "flow": 0.3911, "snr": 2.23, "floor_pct": 67.1, "abs_int": 22.56, "itf": 0.8756},
    {"label": "Level 6 - Relaxed", "cellprob": 0.1956, "flow": 0.2889, "snr": 1.97, "floor_pct": 60.9, "abs_int": 19.44, "itf": 1.0044},
    {"label": "Level 7 - Sensitive", "cellprob": 0.1067, "flow": 0.1867, "snr": 1.70, "floor_pct": 54.7, "abs_int": 16.33, "itf": 1.1333},
    {"label": "Level 8 - High sensitivity", "cellprob": 0.0178, "flow": 0.0844, "snr": 1.43, "floor_pct": 48.4, "abs_int": 13.22, "itf": 1.2622},
    {"label": "Level 9 - Very high sensitivity", "cellprob": -0.0711, "flow": -0.0178, "snr": 1.17, "floor_pct": 42.2, "abs_int": 10.11, "itf": 1.3911},
    {"label": "Level 10 - Ultra sensitive", "cellprob": -0.16, "flow": -0.12, "snr": 0.90, "floor_pct": 36, "abs_int": 7, "itf": 1.52},
]

AI_CONFIG_PATH = 'config_ai.yaml'
DET_CONFIG_PATH = 'config.yaml'
DET_CPSAM_CONFIG_PATH = 'config_det_cpsam.yaml'
CZI_CONFIG_PATH = 'config_czi.yaml'
