from flask import Flask, render_template, request, jsonify, send_file, after_this_request
import requests
import os
import yaml
import subprocess
import threading
import json
from datetime import datetime
from pathlib import Path
import sys
from urllib.parse import quote
import tempfile, zipfile
import copy
import csv  # added
import re
from src.config_archiver import save_config_snapshot
from werkzeug.utils import secure_filename
from src.anova_analysis import run_anova_analysis

app = Flask(__name__)

# Globale Variablen fuer Analyse-Status
analysis_status = {
    'running': False,
    'progress': 0,
    'current_file': '',
    'log': []
}

# Globale Variable fuer Prozesssteuerung
analysis_process = None

# NEU: CZI Conversion Status
czi_status = {
    'running': False,
    'log': [],
    'total': 0,      # NEW
    'done': 0,       # NEW
    'progress': 0    # NEW (0-100)
}
czi_process = None

# NEU: CZI Config Path
CZI_CONFIG_PATH = 'config_czi.yaml'

def _resolve_data_root() -> Path:
    """Determine the writable data root inside the container."""
    raw = (os.environ.get('HOST_DATA') or os.environ.get('CONTAINER_DATA_ROOT') or '').strip()
    candidates: list[str] = []
    if raw:
        if ':' in raw and '\\' in raw:
            drive = raw[0].lower()
            rest = raw[2:].replace('\\', '/')
            candidates.append(f"/host_mnt/{drive}/{rest}")
            candidates.append(f"/mnt/{drive}/{rest}")
        candidates.append(raw)
    candidates.append('/data')
    for cand in candidates:
        try:
            path = Path(cand)
            if path.is_dir():
                return path.resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path.resolve()
        except Exception:
            continue
    return Path('/data').resolve()

# Data exchange base paths
DATA_ROOT = _resolve_data_root()
UPLOAD_SUBDIR = os.environ.get('DATA_UPLOAD_SUBDIR', 'Input') or 'Input'
UPLOAD_ROOT = (DATA_ROOT / UPLOAD_SUBDIR).resolve()
try:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Optional Google Drive integration hints
GOOGLE_DRIVE_EMBED_URL = os.environ.get('GOOGLE_DRIVE_EMBED_URL', '').strip()
GOOGLE_DRIVE_SYNC_PATH = os.environ.get('GOOGLE_DRIVE_SYNC_PATH', '').strip()

# detection state
det_status = { 'running': False, 'log': [] }
det_process = None
DET_CONFIG_PATH = 'config.yaml'
# NEW: separate CPSAM config that matches batch_segment.py schema
DET_CPSAM_CONFIG_PATH = 'config_det_cpsam.yaml'
AI_CONFIG_PATH = 'config_ai.yaml'

SENSITIVITY_LEVELS = [
    {"label": "Level 1 - Strict", "cellprob": 0.64, "flow": 0.80, "snr": 3.30, "floor_pct": 92, "abs_int": 35, "itf": 0.36},
    {"label": "Level 2 - Semi-strict", "cellprob": 0.5511, "flow": 0.6978, "snr": 3.03, "floor_pct": 85.8, "abs_int": 31.89, "itf": 0.4889},
    {"label": "Level 3 - Balanced", "cellprob": 0.4622, "flow": 0.5956, "snr": 2.77, "floor_pct": 79.6, "abs_int": 28.78, "itf": 0.6178},
    {"label": "Level 4 - Balanced+", "cellprob": 0.3733, "flow": 0.4933, "snr": 2.50, "floor_pct": 73.3, "abs_int": 25.67, "itf": 0.7467},
    {"label": "Level 5 - Moderate", "cellprob": 0.2844, "flow": 0.3911, "snr": 2.23, "floor_pct": 67.1, "abs_int": 22.56, "itf": 0.8756},
    {"label": "Level 6 - Medium-high", "cellprob": 0.1956, "flow": 0.2889, "snr": 1.97, "floor_pct": 60.9, "abs_int": 19.44, "itf": 1.0044},
    {"label": "Level 7 - Sensitive", "cellprob": 0.1067, "flow": 0.1867, "snr": 1.70, "floor_pct": 54.7, "abs_int": 16.33, "itf": 1.1333},
    {"label": "Level 8 - High sensitivity", "cellprob": 0.0178, "flow": 0.0844, "snr": 1.43, "floor_pct": 48.4, "abs_int": 13.22, "itf": 1.2622},
    {"label": "Level 9 - Very high sensitivity", "cellprob": -0.0711, "flow": -0.0178, "snr": 1.17, "floor_pct": 42.2, "abs_int": 10.11, "itf": 1.3911},
    {"label": "Level 10 - Ultra sensitive", "cellprob": -0.16, "flow": -0.12, "snr": 0.90, "floor_pct": 36, "abs_int": 7, "itf": 1.52},
]

def _normalize_channel_levels(raw):
    """
    Normalize incoming per-channel sensitivity levels.

    Returns (list_for_config, level_map) where list_for_config is ready to persist.
    """
    level_map: dict[int, int] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            idx = entry.get('index')
            level = entry.get('level')
            try:
                idx_int = int(idx)
                level_int = int(level)
            except Exception:
                continue
            if SENSITIVITY_LEVELS:
                level_int = max(0, min(len(SENSITIVITY_LEVELS) - 1, level_int))
            level_map[idx_int] = level_int
    elif isinstance(raw, dict):
        for key, value in raw.items():
            try:
                idx_int = int(key)
                level_int = int(value)
            except Exception:
                continue
            if SENSITIVITY_LEVELS:
                level_int = max(0, min(len(SENSITIVITY_LEVELS) - 1, level_int))
            level_map[idx_int] = level_int
    normalized = [{'index': idx, 'level': level_map[idx]} for idx in sorted(level_map)]
    return normalized, level_map

DEFAULT_AI_CONFIG = {
    'base_url': os.environ.get('OPENWEBUI_BASE_URL', 'http://localhost:8090'),
    'api_key': os.environ.get('OPENWEBUI_API_KEY', ''),
    'model_id': os.environ.get('OPENWEBUI_MODEL_ID', '5ce5a9ac1cc0'),
    'temperature': float(os.environ.get('OPENWEBUI_TEMPERATURE', 0.2)),
    'max_tokens': int(os.environ.get('OPENWEBUI_MAX_TOKENS', 800)),
}

def load_ai_config() -> dict:
    cfg = DEFAULT_AI_CONFIG.copy()
    if os.path.exists(AI_CONFIG_PATH):
        try:
            with open(AI_CONFIG_PATH, 'r', encoding='utf-8') as f:
                disk_cfg = yaml.safe_load(f) or {}
            if isinstance(disk_cfg, dict):
                for key in ('base_url', 'api_key', 'model_id', 'temperature', 'max_tokens'):
                    val = disk_cfg.get(key)
                    if val not in (None, ''):
                        cfg[key] = val
        except Exception as e:
            print(f"[config] Failed to load AI config: {e}")
    else:
        try:
            sample = {
                'base_url': cfg['base_url'],
                'api_key': '',
                'model_id': cfg['model_id'],
                'temperature': cfg['temperature'],
                'max_tokens': cfg['max_tokens'],
            }
            with open(AI_CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.safe_dump(sample, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception:
            pass
    try:
        cfg['temperature'] = float(cfg.get('temperature', 0.2))
    except Exception:
        cfg['temperature'] = 0.2
    try:
        cfg['max_tokens'] = int(cfg.get('max_tokens', 800))
    except Exception:
        cfg['max_tokens'] = 800
    return cfg


def _prepare_detection_ai_context() -> dict:
    cfg = load_det_config() or {}
    snapshot = copy.deepcopy(cfg)
    try:
        cpsam_full = _load_yaml(DET_CPSAM_CONFIG_PATH)
        if isinstance(cpsam_full, dict) and cpsam_full:
            snapshot['cpsam'] = cpsam_full
    except Exception:
        pass
    log_tail = det_status.get('log', [])[-20:]
    return {
        'detection_config': snapshot,
        'recent_log_tail': log_tail,
        'timestamp': datetime.now().isoformat(),
    }


CONFIG_UPDATE_PATTERN = re.compile(r"`config_update\s*(\{.*?\})\s*`", re.DOTALL)


def _extract_config_update(raw_text: str) -> tuple[str, dict | None]:
    if not raw_text:
        return "", None
    match = CONFIG_UPDATE_PATTERN.search(raw_text)
    if not match:
        return raw_text.strip(), None
    block = match.group(1)
    update_payload = None
    try:
        update_payload = json.loads(block)
    except Exception:
        update_payload = None
    cleaned = CONFIG_UPDATE_PATTERN.sub('', raw_text).strip()
    return cleaned, update_payload


def _call_assistant_chat(messages: list[dict], include_detection: bool = True) -> tuple[str, dict | None]:
    ai_cfg = load_ai_config()
    base_url = (ai_cfg.get('base_url') or '').strip()
    if not base_url:
        raise ValueError('AI base URL is not configured. Set OPENWEBUI_BASE_URL or edit config_ai.yaml.')
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    api_key = (ai_cfg.get('api_key') or '').strip()
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    context_sections: list[str] = []
    if include_detection:
        detection_context = _prepare_detection_ai_context()
        try:
            context_yaml = yaml.safe_dump(detection_context, allow_unicode=True, sort_keys=False)
        except Exception:
            context_yaml = json.dumps(detection_context, indent=2, ensure_ascii=False)
        context_sections.append("Detection Kontext:\n`yaml\n" + context_yaml + "\n`")

    system_lines = [
        (
            "ROLLE: Du bist der integrierte Assistent fuer die Cell Analysis Pipeline. "
            "ANTWORTSPRACHE: Deutsch mit kurzer 'du'-Ansprache. "
            "UMFANG: Unterstuetze bei CZI-Konvertierung, Zellsegmentierung (Cellpose/CPSAM) und Co-Expression-Analyse. "
            "Nutze nur Fakten aus uebergebenem Kontext."
        ),
        (
            "STANDARDSTIL: "
            "- Liefere zwei bis drei praegnante Bullet Points. "
            "- Bei ausdruecklichen Nachfragen nach Erklaerung oder Details ergaenze eine kurze strukturierte Erlaeuterung. "
            "- Fuehre nur relevante Pfade, Parameter oder Logs an."
        ),
        (
            "KONTEXT: "
            "- Verwende detection_config nur fuer vorhandene Schluessel. "
            "- Logs (det_status, analysis_status, czi_status) nur nennen, wenn sie fuer die Antwort wichtig sind. "
            "- Fehlen Informationen, sage dies knapp und frage zielgerichtet nach."
        ),
        (
            "KONFIG-AENDERUNGEN: "
            "- Nutze `config_update {\"target\":\"...\",\"changes\":{...},\"reason\":\"...\"}` in Backticks. "
            "- Gueltige Ziele: det, cpsam, czi, analysis, ai. "
            "- Begruende Aenderungen konservativ; Alternativen ggf. getrennt anbieten."
        ),
        (
            "ZUSTIMMUNG: "
            "- Bitte vor dem Anwenden immer um JA oder NEIN. "
            "- Ohne JA keine Umsetzung; fehlende Pflichtwerte zuerst erfragen."
        ),
        (
            "RISIKEN & PFADREGELN: "
            "- Weise auf moegliche Nebenwirkungen (RAM/VRAM, Laufzeit, Datenueberschreibung) hin. "
            "- Erinnere bei Bedarf an Pfad-Mappings (Windows -> /host_mnt/<drive>/, /mnt/<drive>/, /app/data)."
        ),
        (
            "CHECKLISTE: "
            "- CZI: input_root/output_base, Kanalauswahl, Normalisierung, OME/Stack-Optionen, overwrite. "
            "- Detection: input_tiffs_dir/output_results_dir, channels, thresholds, magnification, GPU/CPU. "
            "- CPSAM: filters, processing, advanced_filtering, overlays, microscope, testing. "
            "- Co-Expression: Ergebnis-Pfade, CSV-Ausgaben, Overlay-Abhaengigkeiten."
        ),
        (
            "ABSCHLUSS: "
            "- Frage am Ende nach JA/NEIN oder ob weitere Details benoetigt werden."
        ),
    ]
    if context_sections:
        system_lines.append("Kontext:\n" + "\n\n".join(context_sections))
    payload_messages = [
        {'role': 'system', 'content': "\n\n".join(system_lines)}
    ]
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content')
        if role not in ('user', 'assistant'):
            continue
        if not isinstance(content, str):
            continue
        payload_messages.append({'role': role, 'content': content})
    if len(payload_messages) == 1:
        raise ValueError('No valid messages provided for assistant chat.')

    payload = {
        'model': ai_cfg.get('model_id') or DEFAULT_AI_CONFIG['model_id'],
        'messages': payload_messages,
        'temperature': float(ai_cfg.get('temperature', 0.2)),
        'max_tokens': int(ai_cfg.get('max_tokens', 800)),
    }

    response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    try:
        raw_text = data['choices'][0]['message']['content']
    except Exception:
        raw_text = json.dumps(data, indent=2, ensure_ascii=False)
    return _extract_config_update(raw_text)
def _call_detection_assistant(user_notes: str | None = None) -> str:
    base_message = "Analysiere die aktuelle Detection-Konfiguration und gib konkrete, priorisierte Empfehlungen."
    if user_notes:
        base_message += f"\n\nBenutzerhinweis: {user_notes}"
    base_message += "\n\nGib Hinweise zu Risiken, Validierungen und moeglichen Parametern zum Anpassen."
    reply_text, _ = _call_assistant_chat([{'role': 'user', 'content': base_message}], include_detection=True)
    return reply_text
def load_config(filename=None):
    """Laedt die Konfiguration aus einer YAML-Datei"""
    config_path = filename if filename else 'config_analysis.yaml'
    # Security check: prevent directory traversal
    if os.path.basename(config_path) != config_path:
        config_path = 'config_analysis.yaml'
        
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_config(config, filename=None):
    """Speichert die Konfiguration in einer YAML-Datei"""
    config_path = filename if filename else 'config_analysis.yaml'
    # Security check
    if os.path.basename(config_path) != config_path:
        config_path = 'config_analysis.yaml'
        
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# NEU: CZI Config Funktionen
def load_czi_config():
    """Laedt CZI-Konvertierungs-Konfiguration"""
    defaults = {
        'input_root': '',
        'output_base': '',
        'channels': [0, 1, 2, 3],
        'target_size': None,
        'per_channel_normalize': True,
        'overwrite': False,
        # NEU: Save options
        'save_per_channel': True,
        'save_stack_tiff': True,
        'save_rgb_preview': False,
        'rgb_channels': [0, 1, 2],
        'dtype': 'uint8',
        # NEW: stack layout + ImageJ hyperstack + channel names
        'stack_layout': 'CYX',
        'imagej_hyperstack': True,
        'channel_names': [],
        # NEW
        'save_color_composite': True,
        'channel_colors': ['#ff0000', '#00ff00', '#0000ff', '#ffff00'],
        # NEW:
        'save_colored_pages': True,
        'save_ome_colors': True,
        'verbose': False,  # NEW
    }
    if os.path.exists(CZI_CONFIG_PATH):
        try:
            with open(CZI_CONFIG_PATH, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
                defaults.update(loaded)
        except Exception:
            pass
    return defaults

def save_czi_config(config):
    """Speichert CZI-Konvertierungs-Konfiguration"""
    with open(CZI_CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_czi_conversion_thread(config):
    """Fuehrt CZI-Konvertierung in separatem Thread aus"""
    global czi_status, czi_process
    try:
        czi_status['running'] = True
        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] CZI Konvertierung gestartet...")

        # Speichere Config temporaer
        save_czi_config(config)

        # Pre-count total .czi files for progress (best-effort)
        try:
            # Map Windows path to WSL if needed
            in_root = config.get('input_root') or ''
            in_root = _map_incoming_path(in_root)  # Convert Windows paths to WSL
            
            total = 0
            if os.path.exists(in_root) and os.path.isdir(in_root):
                for root, _, files in os.walk(in_root):
                    total += sum(1 for fn in files if fn.lower().endswith('.czi'))
            
            czi_status['total'] = total
            czi_status['done'] = 0
            czi_status['progress'] = 0
            
            if total > 0:
                czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Found {total} CZI file(s)")
            else:
                czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] No CZI files found in {in_root}")
                # Don't fail, let the script run and report its own findings
        except Exception as e:
            czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Could not count files: {e}")
            czi_status['total'] = 0
            czi_status['done'] = 0
            czi_status['progress'] = 0

        # Starte convert_czi.py
        cmd = [sys.executable, 'convert_czi.py']
        if bool(config.get('verbose')):
            cmd.append('--verbose')

        czi_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd()
        )

        # Lese Output
        for raw in czi_process.stdout:
            line = raw.strip()
            czi_status['log'].append(line)
            
            # progress marker from convert_czi: "[FILE_DONE] name"
            if line.startswith('[FILE_DONE]'):
                czi_status['done'] = czi_status.get('done', 0) + 1
                total = czi_status.get('total', 0)
                
                # If we didn't count files initially, update total based on progress
                if total == 0 and czi_status['done'] > 0:
                    # Estimate: assume we're halfway through
                    czi_status['total'] = czi_status['done'] * 2
                    total = czi_status['total']
                
                if total > 0:
                    progress = min(100, int(czi_status['done'] * 100 / total))
                    czi_status['progress'] = progress
                    # Add progress to log if not verbose
                    if not config.get('verbose'):
                        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {progress}% ({czi_status['done']}/{total})")
            
            # Keep log size manageable
            if len(czi_status['log']) > 200:
                czi_status['log'] = czi_status['log'][-200:]

        czi_process.wait()

        if czi_process.returncode == 0:
            czi_status['progress'] = 100
            czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Konvertierung erfolgreich! ({czi_status['done']} files)")
        else:
            czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Konvertierung fehlgeschlagen (Code: {czi_process.returncode})")

    except Exception as e:
        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Fehler: {str(e)}")
    finally:
        czi_status['running'] = False
        czi_process = None

def run_analysis_thread(config):
    """Fuehrt die Analyse in einem separaten Thread aus"""
    global analysis_status, analysis_process
    
    try:
        analysis_status['running'] = True
        analysis_status['user_stopped'] = False
        analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Analyse gestartet...")
        
        # Speichere temporaere Konfiguration
        save_config(config)
        
        # Debug-Flag aus Payload lesen
        debug_flag = bool(config.get('debug', False))
        cmd = [sys.executable, 'analysis_combinations.py', '--config', 'config_analysis.yaml']
        if debug_flag:
            cmd.append('-v')

        # Starte Analyse-Skript mit gleichem Interpreter und explizitem Config-Pfad
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd()
        )
        analysis_process = proc
        
        # Lese Output Zeile fuer Zeile
        for line in proc.stdout:
            analysis_status['log'].append(line.strip())
            # Begrenze Log auf letzte 100 Zeilen
            if len(analysis_status['log']) > 100:
                analysis_status['log'] = analysis_status['log'][-100:]
        
        proc.wait()
        
        if proc.returncode == 0:
            analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Analyse erfolgreich abgeschlossen!")
        elif proc.returncode == -15:
            if analysis_status.get('user_stopped'):
                # Logged in stop_analysis already
                pass
            else:
                analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Analyse durch System beendet (SIGTERM)")
        else:
            analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Analyse fehlgeschlagen (Code: {proc.returncode})")
            
    except Exception as e:
        analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Fehler: {str(e)}")
    finally:
        analysis_status['running'] = False
        analysis_status['progress'] = 100
        analysis_process = None

def load_det_config():
    """Laedt Detection-Konfiguration (config.yaml)"""
    if os.path.exists(DET_CONFIG_PATH):
        with open(DET_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
            # ensure new key exists to roundtrip with UI
            cfg.setdefault('det_script_path', '')
            cfg.setdefault('verbose', False)  # NEW
            scope = cfg.setdefault('microscope', {})
            if not isinstance(scope, dict):
                scope = {}
            scope.setdefault('reference_magnification', 10)
            scope.setdefault('current_magnification', scope.get('reference_magnification', 10))
            cfg['microscope'] = scope
            
            # Ensure analysis config exists
            analysis = cfg.setdefault('analysis', {})
            if not isinstance(analysis, dict):
                analysis = {}
            analysis.setdefault('enable_anova', True)
            cfg['analysis'] = analysis
            
            return cfg
    return {
        'input_tiffs_dir': '',
        'output_results_dir': '',
        'channels': [0,1],
        'model_type': 'cyto2',
        'diameter': None,
        'flow_threshold': 0.4,
        'cellprob_threshold': 0.0,
        'min_area': 20,
        'save_overlay': True,
        'overlay_suffix': '_overlay.png',
        'use_gpu': False,
        'batch_size': 8,
        'region_mapping': {},
        # NEW: path to batch_segment.py (optional)
        'det_script_path': '',
        'verbose': False,  # NEW
        'microscope': {
            'reference_magnification': 10,
            'current_magnification': 10,
        },
    }

# Helper: default CPSAM config template (all parameters as requested)
def _default_cpsam_config() -> dict:
    return {
        'paths': {
            'input_pos': "./data/processed_tiffs/Input_pos",
            'input_neg': "./data/processed_tiffs/Input_neg",
            'output_root': "./results_Julia_ch3",
        },
        'conditions': {
            'pos_pattern': "Input_old_",
            'neg_pattern': "Input_adult_",
        },
        'cellpose': {
            'model_name': "cpsam",
            'batch_size': 8,
            'resize_max': 2048,
            'niter': 750,
            'flow_threshold': 0.5,
            'cellprob_threshold': 0.5,
            'diameter': 20,
            'save_masks': False,
            'channel': 3,
            'split_regions': False,
        },
        'filters': {
            'min_area': 80,
            'max_area': 3000,
            'min_circularity': 0.1,
            'max_circularity': 1.0,
            'max_hole_ratio': 0.2,
            'intensity_threshold_factor': 0.3,
        },
        'enable_advanced_filtering': True,
        'advanced_filtering': {
            'intensity_mode': 'max',
            'max_intensity_threshold': 0.1,
            'snr_min': 1.1,
            'abs_floor_percentile': 75,
            'foreground_block_size': 501,
            'foreground_offset': 10,
        },
        'processing': {
            'tophat': True,
            'tophat_radius': 15,
            'contrast_stretch': True,
            'stretch_percentiles': [1, 99],
            'merge_adjacent_cells': False,
            'remove_edge_cells': False,
            'intensity_threshold': 5,
            'split_touching_cells': True,
            'split_min_distance': 6,
            'split_area_multiplier': 2.5,
            'split_min_area': 0,
            'split_rel_peak_threshold': 0.2,
        },
        'overlays': {
            'overlay_mode': 'contours',
            'contour_color': 'lime',
            'line_width': 1,
            'dpi': 300,
            'figsize': [12, 12],
        },
        'outputs': {
            'base_name': "cell_analysis",
        },
        'microscope': {
            'reference_magnification': 10,
            'current_magnification': 10,
        },
    }

def _deep_update(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _build_cpsam_config(ui_cfg: dict, base: dict | None = None) -> dict:
    """
    Map UI detection config -> CPSAM batch_segment.py config.
    Starts from existing config (if provided) else defaults, overlays UI CPSAM block, then applies top-level overrides.
    """
    if isinstance(base, dict) and base:
        cfg = copy.deepcopy(base)
    else:
        cfg = _default_cpsam_config()

    # Ensure required sections exist
    for section in (
        'paths', 'conditions', 'cellpose', 'filters',
        'advanced_filtering', 'processing', 'overlays', 'outputs', 'microscope'
    ):
        cfg.setdefault(section, {})

    # 1) Overlay CPSAM block from UI if present
    cpsam_ui = ui_cfg.get('cpsam') or {}
    if cpsam_ui:
        _deep_update(cfg, cpsam_ui)

    # 2) Paths from top-level UI (preferred)
    inp_root = (ui_cfg.get('input_tiffs_dir') or '').strip()
    out_root = (ui_cfg.get('output_results_dir') or '').strip()
    cond_cfg = cfg.setdefault('conditions', {}) or {}

    def _tokenize_patterns(value):
        tokens: list[str] = []
        if isinstance(value, str):
            for piece in re.split(r'[;,/]+|\s+', value):
                piece = piece.strip().lower()
                if piece:
                    tokens.append(piece)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                tokens.extend(_tokenize_patterns(item))
        return tokens

    def _merge_tokens(primary, fallback):
        merged: list[str] = []
        for tok in list(primary) + list(fallback):
            tok = (tok or '').strip().lower()
            if tok and tok not in merged:
                merged.append(tok)
        return tuple(merged)

    pos_tokens = _merge_tokens(
        _tokenize_patterns(cond_cfg.get('pos_pattern')),
        ('input_pos', 'pos', 'positive')
    )
    neg_tokens = _merge_tokens(
        _tokenize_patterns(cond_cfg.get('neg_pattern')),
        ('input_neg', 'neg', 'negative')
    )

    def _guess_condition_dir(root: str, tokens: tuple[str, ...], fallback_other: str | None = None) -> str:
        if not root:
            return ''
        root_norm = os.path.normpath(root)
        tokens = tuple(tok.lower() for tok in tokens if tok)
        token_set = set(tokens)
        if os.path.isdir(root_norm):
            try:
                entries = list(os.scandir(root_norm))
            except Exception:
                entries = []
            # direct files?
            if any(entry.is_file() and entry.name.lower().endswith(('.tif', '.tiff')) for entry in entries):
                return root_norm
            # token match in immediate subdirs
            for entry in entries:
                if entry.is_dir():
                    name = entry.name.lower()
                    if token_set and any(tok in name for tok in token_set):
                        return entry.path
            # single subdir fallback
            dirs_only = [entry.path for entry in entries if entry.is_dir()]
            if len(dirs_only) == 1:
                return dirs_only[0]
        # fallback: search deeper up to limited depth
        matches_token: list[str] = []
        matches_general: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root_norm):
            if any(fn.lower().endswith(('.tif', '.tiff')) for fn in filenames):
                candidate = dirpath
                try:
                    if fallback_other and os.path.samefile(candidate, fallback_other):
                        continue
                except Exception:
                    pass
                base_name = os.path.basename(candidate).lower()
                if token_set and any(tok in base_name for tok in token_set):
                    matches_token.append(candidate)
                else:
                    matches_general.append(candidate)
        if matches_token:
            return matches_token[0]
        if matches_general:
            return matches_general[0]
        return ''

    if inp_root:
        cfg.setdefault('paths', {})
        # Preserve existing paths from config before attempting auto-detection
        existing_pos = cfg.get('paths', {}).get('input_pos')
        existing_neg = cfg.get('paths', {}).get('input_neg')
        
        pos_dir = _guess_condition_dir(inp_root, pos_tokens)
        neg_dir = _guess_condition_dir(inp_root, neg_tokens, fallback_other=pos_dir or None)
        
        if pos_dir:
            cfg['paths']['input_pos'] = pos_dir
        elif not existing_pos:
            # Only remove if it wasn't already set in base config
            cfg['paths'].pop('input_pos', None)
        # else: keep existing_pos
        
        if neg_dir and (not pos_dir or not os.path.samefile(pos_dir, neg_dir)):
            cfg['paths']['input_neg'] = neg_dir
        elif not existing_neg:
            # Only remove if it wasn't already set in base config
            cfg['paths'].pop('input_neg', None)
        # else: keep existing_neg
    if out_root:
        cfg.setdefault('paths', {})
        cfg['paths']['output_root'] = out_root

    # 3) Core cellpose params from top-level UI take precedence
    cp = cfg.setdefault('cellpose', {})
    model = (ui_cfg.get('model_type') or '').strip() or cp.get('model_name', 'cpsam')
    cp['model_name'] = model
    if isinstance(ui_cfg.get('batch_size'), int):
        cp['batch_size'] = int(ui_cfg['batch_size'])
    if isinstance(ui_cfg.get('diameter'), (int, float)) and ui_cfg['diameter']:
        cp['diameter'] = int(ui_cfg['diameter'])
    if isinstance(ui_cfg.get('flow_threshold'), (int, float)):
        cp['flow_threshold'] = float(ui_cfg['flow_threshold'])
    if isinstance(ui_cfg.get('cellprob_threshold'), (int, float)):
        cp['cellprob_threshold'] = float(ui_cfg['cellprob_threshold'])
    if 'use_gpu' in ui_cfg:
        cp['use_gpu'] = bool(ui_cfg.get('use_gpu'))

    # 4) Filters: min_area from top-level UI if present
    fil = cfg.setdefault('filters', {})
    if isinstance(ui_cfg.get('min_area'), (int, float)):
        fil['min_area'] = int(ui_cfg['min_area'])

    scope_cfg = cfg.setdefault('microscope', {})
    if not isinstance(scope_cfg, dict):
        scope_cfg = {}
        cfg['microscope'] = scope_cfg

    def _coerce_positive_float(val):
        try:
            f = float(val)
            return f if f > 0 else None
        except Exception:
            return None

    ui_scope = ui_cfg.get('microscope') or {}
    if isinstance(ui_scope, dict):
        cur = _coerce_positive_float(ui_scope.get('current_magnification'))
        ref = _coerce_positive_float(ui_scope.get('reference_magnification'))
        auto = ui_scope.get('auto_from_path')
        if ref is not None:
            scope_cfg['reference_magnification'] = ref
        if cur is not None:
            scope_cfg['current_magnification'] = cur
        if isinstance(auto, bool):
            scope_cfg['auto_from_path'] = auto
    scope_cfg.setdefault('reference_magnification', 10)
    scope_cfg.setdefault('current_magnification', scope_cfg.get('reference_magnification', 10))
    scope_cfg.setdefault('auto_from_path', True)

    return cfg

def _save_yaml(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def save_det_config(cfg: dict):
    """Speichert Detection-Konfiguration"""
    with open(DET_CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

def _resolve_script(*candidates: str) -> Path | None:
    """Find a local script by name in likely locations."""
    bases = [
        Path(__file__).parent,                   # /app (Docker) or repo root
        Path(os.getcwd()),                       # current working dir
        Path(__file__).parent / 'scripts',       # optional scripts/
    ]
    for base in bases:
        for name in candidates:
            p = (base / name)
            if p.exists() and p.is_file():
                return p
    return None

# Legacy CSV schema normalizer for All_Counts_Master.csv
def _normalize_master_csvs(base_dir: str, label_map: dict[int, str] | None = None):
    """Normalize All_Counts_Master.csv files and add friendly channel names."""
    if not base_dir or not os.path.isdir(base_dir):
        return
    label_map = label_map or {}
    legacy_order = [
        'filename','condition','region','channel','channel_index','cell_count',
        'mean_area_per_cell','mean_intensity_per_cell','mean_integrated_density_per_cell'
    ]
    synonyms = {
        'file': 'filename', 'image': 'filename', 'image_name': 'filename',
        'region_token': 'region', 'region_label': 'region',
        'channel_id': 'channel', 'channel_index': 'channel',
        'channel_idx': 'channel_index',
        'n_cells': 'cell_count', 'cells': 'cell_count', 'cellcount': 'cell_count',
        'mean_area': 'mean_area_per_cell', 'area_mean': 'mean_area_per_cell',
        'mean_intensity': 'mean_intensity_per_cell', 'intensity_mean': 'mean_intensity_per_cell',
        'mean_integrated_density': 'mean_integrated_density_per_cell',
        'integrated_density_mean': 'mean_integrated_density_per_cell',
    }
    for root, _, files in os.walk(base_dir):
        if 'All_Counts_Master.csv' not in files:
            continue
        fp = os.path.join(root, 'All_Counts_Master.csv')
        ch_idx = None
        try:
            import re
            m = re.search(r'[\\/](?:ch)(\d+)[\\/]', os.path.abspath(root))
            if m:
                ch_idx = int(m.group(1))
        except Exception:
            ch_idx = None
        reg_hint = _infer_region_token(root)
        try:
            with open(fp, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue
                cur_fields = list(reader.fieldnames or [])
                rename = {}
                for name in cur_fields:
                    key = name.strip()
                    low = key.lower()
                    if key in legacy_order:
                        rename[name] = key
                    elif low in synonyms:
                        rename[name] = synonyms[low]
                    else:
                        rename[name] = name
                normalized = []
                for r in rows:
                    tmp = {}
                    for k, v in r.items():
                        nk = rename.get(k, k)
                        tmp[nk] = v
                    nr = {fld: tmp.get(fld, "") for fld in legacy_order}
                    if ch_idx is not None:
                        idx_str = f"channel_{ch_idx}"
                        nr['channel_index'] = idx_str
                        label = label_map.get(ch_idx)
                        if label:
                            nr['channel'] = label
                        elif not nr.get('channel'):
                            nr['channel'] = idx_str
                    else:
                        if nr.get('channel') and not nr.get('channel_index'):
                            nr['channel_index'] = nr['channel']
                    if reg_hint and reg_hint != 'ALL':
                        nr['region'] = reg_hint
                    elif nr.get('region', '').upper() == 'ALL':
                        fname = (nr.get('filename') or '').replace('\\', '/')
                        if '/' in fname:
                            nr['region'] = fname.split('/', 1)[0]
                    normalized.append(nr)
            with open(fp, 'w', encoding='utf-8', newline='') as w:
                writer = csv.DictWriter(w, fieldnames=legacy_order)
                writer.writeheader()
                writer.writerows(normalized)
        except Exception as e:
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: CSV normalize failed at {fp}: {e}")

def _merge_master_csvs(base_dir: str) -> str | None:
    """Merge per-channel CSVs and keep channel labels."""
    import csv
    base = os.path.abspath(base_dir)
    if not os.path.isdir(base):
        return None
    target = os.path.join(base, 'All_Counts_Master.csv')
    legacy_order = [
        'filename','condition','region','channel','channel_index','cell_count',
        'mean_area_per_cell','mean_intensity_per_cell','mean_integrated_density_per_cell'
    ]
    rows = []
    for root, _, files in os.walk(base):
        for fn in files:
            if fn != 'All_Counts_Master.csv':
                continue
            fp = os.path.join(root, fn)
            if os.path.abspath(fp) == os.path.abspath(target):
                continue
            try:
                with open(fp, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append({k: r.get(k, '') for k in legacy_order})
            except Exception:
                continue
    if not rows:
        return None
    rows.sort(key=lambda r: (
        str(r.get('channel_index', '')),
        str(r.get('condition', '')),
        str(r.get('region', '')),
        str(r.get('filename', ''))
    ))
    try:
        with open(target, 'w', encoding='utf-8', newline='') as w:
            writer = csv.DictWriter(w, fieldnames=legacy_order)
            writer.writeheader()
            writer.writerows(rows)
        return target
    except Exception:
        return None

def _collect_level_counts(level_root: str):
    """Read All_Counts_Master.csv files under a level output and aggregate counts per condition."""
    results = []
    aggregate_cells = 0
    aggregate_images = 0
    base = Path(level_root)
    if not base.exists():
        return results
    for condition_dir in base.iterdir():
        if not condition_dir.is_dir():
            continue
        csv_path = condition_dir / "All_Counts_Master.csv"
        if not csv_path.exists():
            continue
        condition = condition_dir.name
        total_cells = 0
        image_count = 0
        try:
            with csv_path.open('r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_count += 1
                    try:
                        total_cells += int(float(row.get('cell_count', 0) or 0))
                    except Exception:
                        continue
        except Exception:
            continue
        aggregate_cells += total_cells
        aggregate_images += image_count
        results.append({
            'condition': condition,
            'images': image_count,
            'total_cells': total_cells,
        })
    if aggregate_images or aggregate_cells:
        results.append({
            'condition': 'ALL',
            'images': aggregate_images,
            'total_cells': aggregate_cells,
        })
    return results

def _write_sweep_summary(ch_root: str, entries) -> str | None:
    """Write a sweep summary CSV for a channel."""
    if not entries:
        return None
    sorted_entries = sorted(
        entries,
        key=lambda x: (
            x.get('level_idx', 0),
            1 if (x.get('condition') or '').upper() == 'ALL' else 0,
            x.get('condition', ''),
        )
    )
    summary_path = Path(ch_root) / "sweep_counts_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "channel",
            "level_index",
            "level_label",
            "condition",
            "images",
            "total_cells",
            "mean_cells_per_image",
        ])
        for entry in sorted_entries:
            images = entry.get('images') or 0
            total = entry.get('total_cells') or 0
            mean = (total / images) if images else 0.0
            writer.writerow([
                f"ch{entry.get('channel')}",
                ((entry.get('level_idx') or 0) + 1) if entry.get('level_idx') is not None else "",
                entry.get('level_label', ''),
                entry.get('condition', ''),
                images,
                total,
                f"{mean:.2f}",
            ])
    return str(summary_path)

def _prune_empty_dirs(root_dir: str, keep: set[str] | None = None):
    """Remove empty directories under root_dir except those in keep."""
    if not root_dir or not os.path.isdir(root_dir):
        return
    keep = keep or set()
    for current, dirs, files in os.walk(root_dir, topdown=False):
        abs_current = os.path.abspath(current)
        if abs_current in keep:
            continue
        if dirs or files:
            continue
        try:
            os.rmdir(current)
        except OSError:
            continue

def _reorganize_outputs(ch_out: str):
    """Move overlays and masks into canonical folders for a channel."""
    if not ch_out or not os.path.isdir(ch_out):
        return
    overlays_dir = os.path.join(ch_out, 'overlays')
    masks_dir = os.path.join(ch_out, 'masks')
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    abs_overlays = os.path.abspath(overlays_dir)
    abs_masks = os.path.abspath(masks_dir)
    for root, _, files in os.walk(ch_out):
        abs_root = os.path.abspath(root)
        if abs_root in (abs_overlays, abs_masks):
            continue
        region = _infer_region_token(root)
        for fn in files:
            src = os.path.join(root, fn)
            low = fn.lower()
            try:
                if low.endswith('.png') and '_overlay' in low:
                    new_name = _with_region_suffix(fn, region, 'overlay')
                    dst = _unique_path(os.path.join(overlays_dir, new_name))
                    if os.path.abspath(src) != os.path.abspath(dst):
                        os.replace(src, dst)
                elif low.endswith(('.tif', '.tiff')) and '_mask' in low:
                    new_name = _with_region_suffix(fn, region, 'mask')
                    dst = _unique_path(os.path.join(masks_dir, new_name))
                    if os.path.abspath(src) != os.path.abspath(dst):
                        os.replace(src, dst)
            except Exception:
                continue

def run_det_thread(cfg: dict):
    """Startet batch_segment.py und streamt Logs (unterstuetzt Multi-Channel)."""
    global det_status, det_process
    try:
        det_status['running'] = True
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Detection gestartet...")

        try:
            existing_cpsam = _load_yaml(DET_CPSAM_CONFIG_PATH)
            if not isinstance(existing_cpsam, dict):
                existing_cpsam = {}
        except Exception:
            existing_cpsam = {}
        cpsam_cfg = _build_cpsam_config(cfg, existing_cpsam)
        # DO NOT scale here - batch_segment.py will scale when loading the config
        # (scaling was applied twice: here + in batch_segment.py -> OOM with large images)
        cpsam_cfg_path = str(Path(DET_CPSAM_CONFIG_PATH).resolve())

        normalized_levels, channel_levels_map = _normalize_channel_levels(
            cfg.get('channel_levels') or cfg.get('det_channel_levels') or []
        )
        if normalized_levels:
            cfg['channel_levels'] = normalized_levels
        else:
            cfg.pop('channel_levels', None)

        save_det_config(cfg)

        labels: dict[int, str] = {}
        for item in cfg.get('channel_labels') or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get('index'))
            except Exception:
                continue
            labels[idx] = (item.get('name') or '').strip()

        raw_testing = cfg.get('testing') if isinstance(cfg.get('testing'), dict) else {}
        raw_mode = raw_testing.get('mode')
        if isinstance(raw_mode, str):
            raw_mode = raw_mode.lower()
        else:
            raw_mode = None
        testing_cfg = {
            'enabled': bool(raw_testing.get('enabled')),
            'samples_per_channel': int(raw_testing.get('samples_per_channel') or 0),
            'mode': raw_mode,
            'levels': copy.deepcopy(raw_testing.get('levels')) if isinstance(raw_testing.get('levels'), (list, tuple)) else None,
        }
        raw_selected_image = raw_testing.get('selected_image')
        raw_selected_images = raw_testing.get('selected_images') if isinstance(raw_testing.get('selected_images'), dict) else {}
        selected_images: dict[str, str] = {}
        if isinstance(raw_selected_images, dict):
            for key, value in raw_selected_images.items():
                cleaned = _clean_relative_path_value(value)
                if cleaned:
                    selected_images[key.lower()] = cleaned
        for key in ('pos', 'neg'):
            alt_val = _clean_relative_path_value(raw_testing.get(f'selected_image_{key}'))
            if alt_val:
                selected_images[key] = alt_val
        single_image = _clean_relative_path_value(raw_selected_image)
        if single_image:
            selected_images.setdefault('pos', single_image)
        seed_val = raw_testing.get('seed')
        if seed_val not in (None, ''):
            testing_cfg['seed'] = str(seed_val)
        advanced_mode = testing_cfg.get('mode') == 'advanced'
        sweep_levels = []
        if advanced_mode:
            levels_val = testing_cfg.get('levels') or []
            if isinstance(levels_val, (list, tuple)):
                for value in levels_val:
                    try:
                        idx = int(value)
                    except Exception:
                        continue
                    if 0 <= idx < len(SENSITIVITY_LEVELS):
                        sweep_levels.append(idx)
            if not sweep_levels:
                sweep_levels = list(range(len(SENSITIVITY_LEVELS)))
            testing_cfg['levels'] = sweep_levels
        if advanced_mode and not testing_cfg.get('seed'):
            if not (selected_images.get('pos') or selected_images.get('neg') or '').strip():
                testing_cfg['seed'] = '__advanced_sweep__'
        if selected_images:
            refined_images: dict[str, str] = {}
            for cond_key, rel_path in selected_images.items():
                refined_val = _refine_selected_image_reference(
                    rel_path,
                    cfg.get('input_tiffs_dir'),
                    cpsam_cfg.get('paths')
                )
                if refined_val:
                    refined_images[cond_key] = refined_val
                elif rel_path:
                    refined_images[cond_key] = rel_path
            if refined_images:
                testing_cfg['selected_images'] = refined_images
                primary = refined_images.get('pos') or next(iter(refined_images.values()))
                if primary:
                    testing_cfg['selected_image'] = primary
        else:
            testing_cfg.pop('selected_images', None)
        if not advanced_mode:
            testing_cfg.pop('levels', None)
        cpsam_cfg['testing'] = copy.deepcopy(testing_cfg)
        testing_enabled = testing_cfg['enabled'] and testing_cfg.get('samples_per_channel', 0) > 0
        if testing_enabled:
            seed_info = f" seed={testing_cfg['seed']}" if testing_cfg.get('seed') else ''
            if advanced_mode:
                det_status['log'].append("[{}] [info] Advanced sweep active: levels={} samples_per_channel={}{}".format(
                    datetime.now().strftime('%H:%M:%S'),
                    len(sweep_levels),
                    testing_cfg['samples_per_channel'],
                    seed_info
                ))
            else:
                det_status['log'].append("[{}] [info] Test mode active: samples_per_channel={}{}".format(
                    datetime.now().strftime('%H:%M:%S'),
                    testing_cfg['samples_per_channel'],
                    seed_info
                ))
        if not testing_enabled and advanced_mode:
            advanced_mode = False
            testing_cfg.pop('levels', None)
            sweep_levels = []

        paths = cpsam_cfg.setdefault('paths', {})
        base_out_raw = (paths.get('output_root') or '').strip() or cfg.get('output_results_dir') or './results_det'
        norm_out = os.path.normpath(base_out_raw)
        head, tail = os.path.split(norm_out)
        if tail.lower().startswith('ch') and tail[2:].isdigit():
            base_out_root = head or norm_out
        else:
            base_out_root = norm_out
        if not base_out_root:
            base_out_root = './results_det'
        if testing_enabled:
            base_out_root = os.path.join(base_out_root, '__test__')
            det_status['log'].append('[{}] [info] Test outputs -> {}'.format(datetime.now().strftime('%H:%M:%S'), base_out_root))
            if advanced_mode:
                base_out_root = os.path.join(base_out_root, '__sweep__')
                det_status['log'].append('[{}] [info] Advanced sweep outputs -> {}'.format(
                    datetime.now().strftime('%H:%M:%S'), base_out_root))

        cp_block = cpsam_cfg.setdefault('cellpose', {})

        selected_channels = cfg.get('det_channel_selection') or []
        if not selected_channels:
            try:
                selected_channels = [int(cp_block.get('channel', 0))]
            except Exception:
                selected_channels = [0]
        selected_channels = [int(ch) for ch in selected_channels if isinstance(ch, (int, float, str))]
        selected_channels = list(dict.fromkeys(selected_channels))
        if not selected_channels:
            selected_channels = [0]

        if channel_levels_map:
            level_notes = []
            for idx, lvl in sorted(channel_levels_map.items()):
                if 0 <= lvl < len(SENSITIVITY_LEVELS):
                    label = SENSITIVITY_LEVELS[lvl].get('label', '').replace('  ', ' ')
                    level_notes.append(f"ch{idx}=L{lvl+1:02d} {label}")
                else:
                    level_notes.append(f"ch{idx}=L{lvl+1:02d}")
            det_status['log'].append(
                "[{}] [info] Per-channel sensitivity -> {}".format(
                    datetime.now().strftime('%H:%M:%S'),
                    '; '.join(level_notes)
                )
            )

        snapshot_dir = None
        try:
            snapshot_root = (Path(base_out_root) / '__config__').resolve()
            snapshot_payload = {
                'ui_config': copy.deepcopy(cfg),
                'cpsam_config': copy.deepcopy(cpsam_cfg),
                'channel_levels_map': dict(sorted(channel_levels_map.items())),
                'selected_channels': list(selected_channels),
            }
            runtime_meta = {
                'started_at': datetime.now().isoformat(),
                'advanced_mode': bool(advanced_mode),
                'testing': copy.deepcopy(testing_cfg),
                'sweep_levels': list(sweep_levels) if advanced_mode else [],
                'base_output': str(base_out_root),
            }
            snapshot_dir = save_config_snapshot(
                snapshot_root,
                config_dict=snapshot_payload,
                runtime_params=runtime_meta,
                snapshot_name='detection_run'
            )
            det_status['log'].append(
                "[{}] [info] Config snapshot gespeichert: {}".format(
                    datetime.now().strftime('%H:%M:%S'),
                    snapshot_dir
                )
            )
        except Exception as snapshot_err:
            det_status['log'].append(
                "[{}] WARN: Config snapshot fehlgeschlagen: {}".format(
                    datetime.now().strftime('%H:%M:%S'),
                    snapshot_err
                )
            )

        tried = []
        script = None
        if cfg.get('det_script_path'):
            sp = Path(cfg['det_script_path'])
            if not sp.is_absolute():
                sp = Path(__file__).parent / sp
            tried.append(str(sp))
            if sp.exists():
                script = sp
            else:
                p2 = Path(cfg['det_script_path'])
                if str(p2) != str(sp):
                    tried.append(str(p2))
                    if p2.exists():
                        script = p2

        if script is None:
            for name in [
                'src/batch_segment.py', 'batch_segment.py', 'scripts/batch_segment.py', 'detection/batch_segment.py',
                'src/batch_segment_v4.py', 'batch_segment_v4.py', 'scripts/batch_segment_v4.py'
            ]:
                tried.append(str((Path(__file__).parent / name).resolve()))
            script = _resolve_script(
                'src/batch_segment.py', 'batch_segment.py', 'scripts/batch_segment.py', 'detection/batch_segment.py',
                'src/batch_segment_v4.py', 'batch_segment_v4.py', 'scripts/batch_segment_v4.py'
            )

        if script is None:
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] batch_segment.py nicht gefunden.")
            det_status['log'].append("  Tried:")
            for entry in tried:
                det_status['log'].append(f"   - {entry}")
            det_status['log'].append("  Tipp: 'Detection Script Path' setzen oder batch_segment.py nach /app kopieren.")
            det_status['running'] = False
            det_process = None
            return

        env = os.environ.copy()
        if bool(cfg.get('use_gpu')):
            visible = (env.get('CUDA_VISIBLE_DEVICES') or '').strip()
            if not visible:
                env['CUDA_VISIBLE_DEVICES'] = '0'
                visible = '0'
            env.pop('CELLPOSE_FORCE_CPU', None)
            det_status['log'].append(
                f"[{datetime.now().strftime('%H:%M:%S')}] GPU requested (CUDA_VISIBLE_DEVICES={visible})"
            )
        else:
            env['CELLPOSE_FORCE_CPU'] = '1'
            det_status['log'].append(
                f"[{datetime.now().strftime('%H:%M:%S')}] GPU disabled for this run (CELLPOSE_FORCE_CPU=1)"
            )
        if bool(cfg.get('verbose')):
            env['VERBOSE'] = '1'
            env['PYTHONUNBUFFERED'] = '1'


        overall_success = True
        completed_channels: list[int] = []

        for ch_idx in selected_channels:
            label_suffix = f" ({labels.get(ch_idx)})" if labels.get(ch_idx) else ""
            ch_root_base = os.path.normpath(os.path.join(base_out_root, f"ch{ch_idx}"))
            channel_level_idx = channel_levels_map.get(ch_idx)
            channel_level_info = None
            simple_level_note = ""
            if channel_level_idx is not None and 0 <= channel_level_idx < len(SENSITIVITY_LEVELS):
                channel_level_info = SENSITIVITY_LEVELS[channel_level_idx]
                label_text = channel_level_info.get('label', '').replace('  ', ' ')
                simple_level_note = f" [Level {channel_level_idx+1:02d} {label_text}]"
            if advanced_mode:
                det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Kanal ch{ch_idx}{label_suffix}: sweep ({len(sweep_levels)} Stufen) -> {ch_root_base}")
            else:
                det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Kanal ch{ch_idx}{label_suffix}: output -> {ch_root_base}{simple_level_note}")

            cp_base = copy.deepcopy(cpsam_cfg.get('cellpose', {}) or {})
            filters_base = copy.deepcopy(cpsam_cfg.get('filters', {}) or {})
            adv_base = copy.deepcopy(cpsam_cfg.get('advanced_filtering', {}) or {})
            proc_base = copy.deepcopy(cpsam_cfg.get('processing', {}) or {})
            testing_base = copy.deepcopy(cpsam_cfg.get('testing') or {})

            variants = []
            if advanced_mode:
                for level_idx in sweep_levels:
                    level_info = SENSITIVITY_LEVELS[level_idx]
                    variant_out = os.path.normpath(os.path.join(ch_root_base, f"L{level_idx+1:02d}"))
                    variants.append({
                        'level_idx': level_idx,
                        'info': level_info,
                        'output_root': variant_out,
                    })
            else:
                variants.append({
                    'level_idx': None,
                    'info': None,
                    'output_root': ch_root_base,
                })

            channel_summary_entries = []
            channel_success = True

            for variant in variants:
                cpsam_cfg['cellpose'] = copy.deepcopy(cp_base)
                cpsam_cfg['filters'] = copy.deepcopy(filters_base)
                cpsam_cfg['advanced_filtering'] = copy.deepcopy(adv_base)
                cpsam_cfg['processing'] = copy.deepcopy(proc_base)
                current_testing = copy.deepcopy(testing_base)

                if advanced_mode and variant['info'] is not None:
                    current_testing['current_level'] = variant['level_idx']
                    current_testing['current_label'] = variant['info']['label']
                else:
                    current_testing.pop('levels', None)
                    if channel_level_info is not None:
                        current_testing['current_level'] = channel_level_idx
                        current_testing['current_label'] = channel_level_info.get('label')
                    else:
                        current_testing.pop('current_level', None)
                        current_testing.pop('current_label', None)
                cpsam_cfg['testing'] = current_testing

                cp_block = cpsam_cfg['cellpose']
                filters_block = cpsam_cfg['filters']
                adv_block = cpsam_cfg['advanced_filtering']
                proc_block = cpsam_cfg['processing']
                paths['output_root'] = variant['output_root']
                cp_block['channel'] = ch_idx

                if advanced_mode and variant['info'] is not None:
                    level_info = variant['info']
                    cp_block['flow_threshold'] = round(level_info['flow'], 4)
                    cp_block['cellprob_threshold'] = round(level_info['cellprob'], 4)
                    filters_block['intensity_threshold_factor'] = round(level_info['itf'], 4)
                    adv_block['snr_min'] = round(level_info['snr'], 2)
                    adv_block['abs_floor_percentile'] = int(round(level_info['floor_pct']))
                    proc_block['intensity_threshold'] = int(round(level_info['abs_int']))
                elif channel_level_info is not None:
                    cp_block['flow_threshold'] = round(channel_level_info['flow'], 4)
                    cp_block['cellprob_threshold'] = round(channel_level_info['cellprob'], 4)
                    filters_block['intensity_threshold_factor'] = round(channel_level_info['itf'], 4)
                    adv_block['snr_min'] = round(channel_level_info['snr'], 2)
                    adv_block['abs_floor_percentile'] = int(round(channel_level_info['floor_pct']))
                    proc_block['intensity_threshold'] = int(round(channel_level_info['abs_int']))

                tmp_cfg_path = None
                try:
                    tmp_handle = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
                    yaml.safe_dump(cpsam_cfg, tmp_handle, default_flow_style=False, allow_unicode=True)
                    tmp_cfg_path = tmp_handle.name
                finally:
                    try:
                        tmp_handle.close()
                    except Exception:
                        pass

                try:
                    cp_view = cpsam_cfg.get('cellpose', {}) or {}
                    fil = cpsam_cfg.get('filters', {}) or {}
                    proc = cpsam_cfg.get('processing', {}) or {}
                    adv = cpsam_cfg.get('advanced_filtering', {}) or {}
                    overlays = cpsam_cfg.get('overlays', {}) or {}
                    scope_cfg = cpsam_cfg.get('microscope') or {}
                    det_status['log'].append(
                        f"  model={cp_view.get('model_name','cpsam')} channel={cp_view.get('channel')} "
                        f"batch={cp_view.get('batch_size')} diameter={cp_view.get('diameter')} use_gpu={cp_view.get('use_gpu')}"
                    )
                    det_status['log'].append(
                        f"  flow_thr={cp_view.get('flow_threshold')} cellprob_thr={cp_view.get('cellprob_threshold')} "
                        f"resize_max={cp_view.get('resize_max')} niter={cp_view.get('niter')}"
                    )
                    if scope_cfg:
                        det_status['log'].append(
                            f"  magnification current={scope_cfg.get('current_magnification','?')}x "
                            f"(reference {scope_cfg.get('reference_magnification','?')}x)"
                        )
                    det_status['log'].append(
                        "  filters: min_area={min_area} max_area={max_area} circ=[{min_circularity},{max_circularity}] "
                        "hole={max_hole_ratio} itf={intensity_threshold_factor}".format(**{
                            'min_area': fil.get('min_area'),
                            'max_area': fil.get('max_area'),
                            'min_circularity': fil.get('min_circularity'),
                            'max_circularity': fil.get('max_circularity'),
                            'max_hole_ratio': fil.get('max_hole_ratio'),
                            'intensity_threshold_factor': fil.get('intensity_threshold_factor')
                        })
                    )
                    det_status['log'].append(
                        f"  advanced: enabled={cpsam_cfg.get('enable_advanced_filtering')} "
                        f"snr_min={adv.get('snr_min')} abs_floor={adv.get('abs_floor_percentile')} "
                        f"fg_block={adv.get('foreground_block_size')} fg_offset={adv.get('foreground_offset')}"
                    )
                    det_status['log'].append(
                        f"  processing: tophat={proc.get('tophat')} radius={proc.get('tophat_radius')} "
                        f"stretch={proc.get('contrast_stretch')} stretch_pct={proc.get('stretch_percentiles')} "
                        f"intensity={proc.get('intensity_threshold')}"
                    )
                    det_status['log'].append(
                        f"  overlays: mode={overlays.get('overlay_mode')} color={overlays.get('contour_color')} "
                        f"line_width={overlays.get('line_width')} dpi={overlays.get('dpi')}"
                    )
                except Exception as log_err:
                    det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Hinweis: Konfigurationsvorschau fehlgeschlagen: {log_err}")

                cfg_to_use = tmp_cfg_path or cpsam_cfg_path
                run_label = ''
                if advanced_mode and variant['info'] is not None:
                    run_label = f" [L{variant['level_idx']+1:02d} {variant['info']['label']}]"
                elif channel_level_info is not None:
                    run_label = f" [L{channel_level_idx+1:02d} {channel_level_info.get('label')}]"
                cmd = [sys.executable, str(script), '--config', cfg_to_use]
                det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Starte Kanal ch{ch_idx}{run_label}: {cmd}")

                det_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(script.parent),
                    env=env
                )
                for line in det_process.stdout:
                    det_status['log'].append(line.rstrip())
                    if len(det_status['log']) > 200:
                        det_status['log'] = det_status['log'][-200:]
                det_process.wait()

                if det_process.returncode == 0:
                    try:
                        out_root = cpsam_cfg.get('paths', {}).get('output_root', '')
                        if out_root:
                            _reorganize_outputs(out_root)
                            _normalize_master_csvs(out_root, label_map=labels)
                            merged_path = _merge_master_csvs(out_root)
                            if merged_path:
                                det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Master CSV saved: {merged_path}")
                                
                                # ANOVA Integration
                                if cfg.get('analysis', {}).get('enable_anova', True):
                                    try:
                                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starte ANOVA Analyse...")
                                        run_anova_analysis(str(merged_path), str(out_root))
                                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ANOVA Analyse abgeschlossen.")
                                    except Exception as anova_err:
                                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: ANOVA Analyse fehlgeschlagen: {anova_err}")

                            _prune_empty_dirs(out_root, keep={os.path.join(out_root, 'overlays'), os.path.join(out_root, 'masks')})
                    except Exception as post_err:
                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: Postprocess fehlgeschlagen (ch{ch_idx}): {post_err}")
                    if advanced_mode and variant['info'] is not None:
                        stats_entries = _collect_level_counts(variant['output_root'])
                        for stat in stats_entries:
                            channel_summary_entries.append({
                                'channel': ch_idx,
                                'level_idx': variant['level_idx'],
                                'level_label': variant['info']['label'],
                                'condition': stat.get('condition'),
                                'images': stat.get('images'),
                                'total_cells': stat.get('total_cells'),
                            })
                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Kanal ch{ch_idx}{label_suffix} [L{variant['level_idx']+1:02d}] abgeschlossen.")
                    else:
                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Kanal ch{ch_idx}{label_suffix}{simple_level_note} abgeschlossen.")
                else:
                    det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Kanal ch{ch_idx}{label_suffix}{simple_level_note} fehlgeschlagen (Code: {det_process.returncode})")
                    overall_success = False
                    channel_success = False
                    break
                det_process = None
                if tmp_cfg_path and os.path.exists(tmp_cfg_path):
                    try:
                        os.remove(tmp_cfg_path)
                    except Exception:
                        pass

            if channel_success:
                completed_channels.append(ch_idx)
                if advanced_mode:
                    summary_path = _write_sweep_summary(ch_root_base, channel_summary_entries)
                    det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ? Kanal ch{ch_idx}{label_suffix} (Advanced Sweep) abgeschlossen.")
                    if summary_path:
                        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Sweep summary saved: {summary_path}")
            else:
                break
        if overall_success and completed_channels:
            suffix = 'e' if len(completed_channels) != 1 else ''
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Detection erfolgreich ({len(completed_channels)} Kanal{suffix}).")
        elif not completed_channels:
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Detection konnte nicht gestartet werden.")
        else:
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Detection vorzeitig beendet.")

    except Exception as e:
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Fehler: {e}")
    finally:
        det_status['running'] = False
        det_process = None

def _safe_join(base: Path, *parts: str) -> Path:
    """Join path parts and ensure the result remains within the base directory."""
    candidate = base.joinpath(*[part for part in parts if part]).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(f"Path escapes base directory: {candidate}")
    return candidate

def _map_incoming_path(p: str) -> str:
    """
    Map incoming paths for the current runtime:
    - Windows-style paths (e.g. C:\\...) -> /mnt/<drive>/... (WSL)
    - Docker volume mappings (heuristic)
    """
    if not p:
        return os.getcwd()
    
    # Normalize slashes
    p_norm = p.replace('\\', '/')
    
    # Heuristic: Map project root D:/Cell/Cell-Analysis to /app
    # This handles the specific Docker volume mount case
    if 'Cell/Cell-Analysis' in p_norm:
        # Try to map to /app
        # e.g. D:/Cell/Cell-Analysis/data -> /app/data
        # or /mnt/d/Cell/Cell-Analysis/data -> /app/data
        
        # Split by the project folder name
        parts = p_norm.split('Cell/Cell-Analysis')
        if len(parts) > 1:
            suffix = parts[1]
            # If suffix starts with /data, we might want to map to /data directly if mounted
            # But /app/data is also valid if . is mounted to /app
            candidate = f"/app{suffix}"
            return candidate

    # Windows path -> WSL path (generic)
    if ':' in p and (len(p) > 1 and p[1] == ':'):
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
        
    return p


def _clean_relative_path_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _refine_selected_image_reference(selected: str, input_root: str | None, paths_cfg: dict | None = None) -> str:
    """
    Normalize a selected test image so it becomes relative to the detected condition directory.
    Removes duplicated leading folders (e.g. 'old/...' when condition path already points to /.../old).
    """
    if not selected:


        return selected
    cleaned = selected.strip().replace('\\', '/')
    if cleaned.startswith('./'):
        cleaned = cleaned[2:]
    if not cleaned:
        return cleaned

    def _coerce_path(val: str | None) -> Path | None:
        if not val:
            return None
        try:
            path_obj = Path(val)
        except Exception:
            return None
        try:
            return path_obj.resolve()
        except Exception:
            return path_obj

    input_root_path = _coerce_path(input_root)
    paths_cfg = paths_cfg or {}
    entries: list[tuple[tuple[str, ...], str | None]] = []
    for key in ('input_pos', 'input_neg'):
        candidate = _coerce_path(paths_cfg.get(key))
        if not candidate:
            continue
        prefix_parts: tuple[str, ...] = ()
        if input_root_path:
            try:
                rel = candidate.relative_to(input_root_path)
                prefix_parts = tuple(part.lower() for part in rel.parts if part not in ('.',))
            except Exception:
                prefix_parts = ()
        if not prefix_parts and candidate.name:
            prefix_parts = (candidate.name.lower(),)
        entries.append((prefix_parts, candidate.name.lower() if candidate.name else None))

    parts = tuple(part for part in Path(cleaned).parts if part not in ('.',))
    if not parts:
        return cleaned
    lowered_parts = tuple(part.lower() for part in parts)

    for prefix_parts, fallback_name in entries:
        if prefix_parts and len(lowered_parts) >= len(prefix_parts) and lowered_parts[:len(prefix_parts)] == prefix_parts:
            trimmed = parts[len(prefix_parts):]
            if trimmed:
                return Path(*trimmed).as_posix()
        elif fallback_name and lowered_parts and lowered_parts[0] == fallback_name:
            trimmed = parts[1:]
            if trimmed:
                return Path(*trimmed).as_posix()
    return cleaned

def _existing_dirs(paths):
    out = []
    seen = set()
    for p in paths:
        if not p:
            continue
        try:
            ap = os.path.abspath(p)
        except Exception:
            continue
        if ap in seen:
            continue
        if os.path.isdir(ap):
            out.append(ap)
            seen.add(ap)
    return out

def _candidate_dirs_from_config():
    """Collect useful directories from all configs (mapped as-is; browse will validate)."""
    cands = []
    try:
        czi = load_czi_config()
        cands += [czi.get('input_root'), czi.get('output_base')]
    except Exception:
        pass
    try:
        det = load_det_config()
        cands += [det.get('input_tiffs_dir'), det.get('output_results_dir')]
    except Exception:
        pass
    try:
        co = load_config()
        paths = (co or {}).get('paths', {})
        cands += [paths.get('base_results_dir'), paths.get('output_dir')]
    except Exception:
        pass
    # add common anchors
    cands += [
        os.getcwd(), '/app', '/', '/host_mnt', '/host_mnt/c', '/mnt', '/mnt/c', '/data'
    ]
    # If user set HOST_DATA (recommendation for remote runs), prefer that anchor too
    host_data = os.environ.get('HOST_DATA')
    if host_data:
        cands.append(host_data)
    # map Windows/WSL variants for each candidate
    mapped = []
    for p in cands:
        if not p:
            continue
        mapped.append(p)
        mapped.append(_map_incoming_path(p))
        # flip /host_mnt<->/mnt if needed
        if isinstance(p, str) and p.startswith('/host_mnt/'):
            mapped.append(p.replace('/host_mnt/', '/mnt/', 1))
        if isinstance(p, str) and p.startswith('/mnt/'):
            mapped.append(p.replace('/mnt/', '/host_mnt/', 1))
    return _existing_dirs(mapped)

def _virtual_roots_payload():
    """Build a virtual root listing of useful anchors."""
    anchors = _candidate_dirs_from_config()
    # Always add container roots
    anchors.extend(['/app', '/data', '/app/data', '/app/results'])
    
    items = []
    seen = set()
    
    for a in anchors:
        if not a: continue
        # Normalize
        try:
            if os.path.exists(a):
                a = os.path.abspath(a)
        except Exception:
            pass
            
        if a in seen: continue
        seen.add(a)
        
        name = a
        if a == '/app':
            name = 'APP (/app)'
        elif a == '/data':
            name = 'DATA (/data)'
        elif a == os.getcwd():
            name = f'WORKDIR ({a})'
        elif a.startswith('/host_mnt/'):
            name = a.replace('/host_mnt/', 'HOST ')
        elif a.startswith('/mnt/'):
            name = a.replace('/mnt/', 'WSL ')
            
        # Only add if it actually exists or is a known root
        if os.path.exists(a) or a in ['/app', '/data']:
             items.append({'name': name, 'path': a, 'type': 'directory', 'isDir': True})
             
    return jsonify({'current_path': '/', 'items': items})

def _infer_region_token(path_str: str) -> str:
    s = (path_str or '').replace('\\','/').lower()
    for tok in ('larc','ldmh','lpvh','rarc','rdmh','rpvh','arc','dmh','pvh','mbh'):
        if f'/{tok}/' in s or s.endswith('/'+tok) or tok in s:
            return tok.upper()
    return 'ALL'

def _with_region_suffix(fn: str, region: str, kind: str) -> str:
    # kind: 'overlay' | 'mask'
    name, ext = os.path.splitext(fn)
    reg = (region or 'ALL').upper()
    # idempotent: if already contains _REGION_kind, keep
    if f"_{reg}_{kind}".lower() in name.lower():
        return fn
    if kind == 'overlay':
        if name.lower().endswith('_overlay'):
            name = name[:-8] + f"_{reg}_overlay"
        else:
            name = f"{name}_{reg}_overlay"
    else:
        if name.lower().endswith('_mask'):
            name = name[:-5] + f"_{reg}_mask"
        else:
            name = f"{name}_{reg}_mask"
    return name + ext

def _unique_path(dst_path: str) -> str:
    base, ext = os.path.splitext(dst_path)
    i = 1
    out = dst_path
    while os.path.exists(out):
        out = f"{base}_{i}{ext}"
        i += 1
    return out

@app.route('/')
def index():
    """Hauptseite"""
    return render_template(
        'index.html',
        google_drive_embed=GOOGLE_DRIVE_EMBED_URL,
        google_drive_sync_hint=GOOGLE_DRIVE_SYNC_PATH,
        upload_root=str(UPLOAD_ROOT),
        data_root=str(DATA_ROOT)
    )

# ========== CZI CONVERSION ENDPOINTS ==========
@app.route('/api/czi/config', methods=['GET'])
def get_czi_config():
    """Liefert CZI-Konfiguration"""
    try:
        config = load_czi_config() or {}
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e), 'config': {}}), 200

@app.route('/api/czi/config', methods=['POST'])
def update_czi_config():
    """Aktualisiert CZI-Konfiguration"""
    config = request.json
    save_czi_config(config)
    return jsonify({'status': 'success', 'message': 'CZI-Konfiguration gespeichert'})

@app.route('/api/czi/start', methods=['POST'])
def start_czi_conversion():
    """Startet CZI-Konvertierung"""
    global czi_status
    
    if czi_status['running']:
        return jsonify({'status': 'error', 'message': 'Konvertierung laeuft bereits'}), 400
    
    config = load_czi_config()
    
    # Validiere Pfade
    if not config.get('input_root') or not config.get('output_base'):
        return jsonify({'status': 'error', 'message': 'Input/Output Pfade fehlen'}), 400
    
    # Starte in Thread
    thread = threading.Thread(target=run_czi_conversion_thread, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'CZI-Konvertierung gestartet'})

@app.route('/api/czi/stop', methods=['POST'])
def stop_czi_conversion():
    """Stoppt CZI-Konvertierung"""
    global czi_process, czi_status
    
    if not czi_status['running']:
        return jsonify({'status': 'error', 'message': 'Keine Konvertierung laeuft'}), 400
    
    if czi_process:
        try:
            czi_process.terminate()
            czi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            czi_process.kill()
        
        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Konvertierung gestoppt")
        czi_status['running'] = False
        czi_process = None
        
        return jsonify({'status': 'success', 'message': 'Konvertierung gestoppt'})
    
    return jsonify({'status': 'error', 'message': 'Prozess konnte not gestoppt werden'}), 500

@app.route('/api/czi/status', methods=['GET'])
def get_czi_status():
    """Liefert CZI-Konvertierungs-Status"""
    return jsonify(czi_status)

@app.route('/api/czi/results', methods=['GET'])
def list_czi_results():
    """Listet konvertierte TIFF-Dateien"""
    config = load_czi_config()
    output_dir = config.get('output_base', '')
    
    if not output_dir or not os.path.exists(output_dir):
        return jsonify({'files': [], 'total_files': 0, 'total_size': 0})
    
    files = []
    total_size = 0
    
    # Suche rekursiv nach .tif/.tiff Dateien
    for root, dirs, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, filename)
                try:
                    size = os.path.getsize(file_path)
                    total_size += size
                    files.append({
                        'name': filename,
                        'path': file_path,
                        'size': size,
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
                        'download': f"/api/czi/download?path={quote(file_path)}"
                    })
                except:
                    continue
    
    # Sortiere nach Datum (neueste zuerst)
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({
        'files': files[:50],  # Erste 50
        'total_files': len(files),
        'total_size': total_size
    })

@app.route('/api/czi/download')
def download_czi_file():
    """Laedt eine konvertierte TIFF-Datei herunter"""
    config = load_czi_config()
    base = os.path.abspath(config.get('output_base', ''))
    req_path = request.args.get('path', '')
    if not base or not req_path:
        return jsonify({'error': 'Pfad fehlt'}), 400

    file_path = os.path.abspath(req_path)
    # Path traversal protection
    if not file_path.startswith(base):
        return jsonify({'error': 'Pfad not erlaubt'}), 403
    if not os.path.exists(file_path):
        return jsonify({'error': 'Datei not gefunden'}), 404

    return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path))

@app.route('/api/czi/download-all')
def download_czi_all_zip():
    # Erstelle ein ZIP mit allen .tif/.tiff Dateien im Output-Verzeichnis
    config = load_czi_config()
    base = os.path.abspath(config.get('output_base', ''))
    if not base or not os.path.isdir(base):
        return jsonify({'status': 'error', 'message': 'Output-Verzeichnis ungueltig'}), 400

    # Temporaere ZIP-Datei
    tmp_dir = tempfile.mkdtemp(prefix='czi_zip_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = os.path.join(tmp_dir, f'czi_converted_{ts}.zip')

    # Packe rekursiv alle TIFFs mit relativen Pfaden
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base):
            for fn in files:
                if not fn.lower().endswith(('.tif', '.tiff')):
                    continue
                full = os.path.join(root, fn)
                # Sicherheit: innerhalb base bleiben
                abs_full = os.path.abspath(full)
                if not abs_full.startswith(base):
                    continue
                rel = os.path.relpath(abs_full, base)
                zf.write(abs_full, arcname=rel)

    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass
        return response

    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

# ========== ANALYSIS ENDPOINTS (existing) ==========
@app.route('/api/config', methods=['GET'])
def get_config():
    """Liefert die aktuelle Konfiguration"""
    filename = request.args.get('file')
    try:
        config = load_config(filename) or {}
        try:
            chan_cfg = config.get('channel_config')
            if not chan_cfg:
                det_cfg = load_det_config() or {}
                labels = det_cfg.get('channel_labels') or []
                if labels:
                    channel_config = []
                    for item in labels:
                        if not isinstance(item, dict):
                            continue
                        try:
                            idx = int(item.get('index'))
                        except Exception:
                            continue
                        channel_config.append({
                            'index': idx,
                            'enabled': True,
                            'name': item.get('name') or ''
                        })
                    if channel_config:
                        config['channel_config'] = channel_config
                        config['channels_limit'] = max(config.get('channels_limit', len(channel_config)), len(channel_config))
        except Exception:
            pass
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e), 'config': {}}), 200

@app.route('/api/config', methods=['POST'])
def update_config():
    """Aktualisiert die Konfiguration"""
    filename = request.args.get('file')
    config = request.json
    save_config(config, filename)
    return jsonify({'status': 'success', 'message': 'Konfiguration gespeichert'})

@app.route('/api/start', methods=['POST'])
def start_analysis():
    """Startet die Analyse"""
    global analysis_status
    
    if analysis_status['running']:
        return jsonify({'status': 'error', 'message': 'Analyse laeuft bereits'}), 400
    
    config = request.json
    
    # Starte Analyse in separatem Thread
    thread = threading.Thread(target=run_analysis_thread, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Analyse gestartet'})

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    """Stoppt die laufende Analyse"""
    global analysis_process, analysis_status
    
    if not analysis_status['running']:
        return jsonify({'status': 'error', 'message': 'Keine Analyse laeuft'}), 400
    
    if analysis_process:
        # Set flag BEFORE terminating to ensure thread sees it if it wakes up fast
        analysis_status['user_stopped'] = True
        
        try:
            analysis_process.terminate()
            analysis_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            analysis_process.kill()
        
        analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Analyse vom Benutzer gestoppt")
        analysis_status['running'] = False
        analysis_process = None
        
        return jsonify({'status': 'success', 'message': 'Analyse gestoppt'})
    
    return jsonify({'status': 'error', 'message': 'Prozess konnte not gestoppt werden'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Liefert den aktuellen Analyse-Status"""
    return jsonify(analysis_status)

@app.route('/api/results', methods=['GET'])
def list_results():
    """Listet verfuegbare Ergebnis-Dateien"""
    config = load_config()
    output_dir = config.get('paths', {}).get('output_dir', 'coexpression_lea2_new')
    
    results = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith('__'):
                continue
            if file.endswith('.csv'):
                file_path = os.path.join(output_dir, file)
                results.append({
                    'name': file,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    return jsonify(results)

@app.route('/api/download/<filename>')
def download_file(filename):
    """Laedt eine Ergebnis-Datei herunter"""
    config = load_config()
    output_dir = config.get('paths', {}).get('output_dir', 'coexpression_lea2_new')
    file_path = os.path.join(output_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'status': 'error', 'message': 'Datei not gefunden'}), 404

@app.route('/api/results/download-all')
def download_all_coexpr_zip():
    # Zip all CSV files in configured coexpression output_dir (recursive)
    config = load_config()
    output_dir = config.get('paths', {}).get('output_dir', '')
    base = os.path.abspath(output_dir or '')
    if not base or not os.path.isdir(base):
        return jsonify({'status': 'error', 'message': 'Output-Verzeichnis ungueltig'}), 400

    tmp_dir = tempfile.mkdtemp(prefix='coexpr_zip_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = os.path.join(tmp_dir, f'coexpr_results_{ts}.zip')

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            if "__test__" in root:
                continue
            for fn in files:
                # include CSVs (results and summaries)
                if not fn.lower().endswith('.csv'):
                    continue
                full = os.path.join(root, fn)
                abs_full = os.path.abspath(full)
                if not abs_full.startswith(base):
                    continue
                rel = os.path.relpath(abs_full, base)
                zf.write(abs_full, arcname=rel)

    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass
        return response

    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

@app.route('/api/config/list', methods=['GET'])
def list_configs():
    """Listet verfuegbare Konfigurationsdateien auf"""
    try:
        files = [f for f in os.listdir('.') if f.startswith('config_analysis') and f.endswith('.yaml')]
        # Sort: config_analysis.yaml first, then others alphabetically
        files.sort(key=lambda x: (x != 'config_analysis.yaml', x))
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e), 'files': []}), 500

@app.route('/api/files/upload', methods=['POST'])
def upload_files():
    """Handle drag-and-drop uploads into the shared data directory."""
    target_subdir = (request.form.get('subdir') or '').strip()
    try:
        target_dir = _safe_join(DATA_ROOT, target_subdir) if target_subdir else UPLOAD_ROOT
    except ValueError:
        return jsonify({'error': 'Invalid target directory'}), 400

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return jsonify({'error': f'Unable to prepare target directory: {exc}'}), 500

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    rel_hints = request.form.getlist('relative_paths') or []

    def _sanitize_relative(rel: str, fallback: str) -> list[str]:
        rel = (rel or '').replace('\\', '/').strip()
        parts = [segment for segment in rel.split('/') if segment and segment not in ('.', '..')]
        if not parts:
            safe = secure_filename(fallback) or fallback
            return [safe]
        safe_parts = []
        for segment in parts:
            safe = secure_filename(segment) or segment
            if safe:
                safe_parts.append(safe)
        if not safe_parts:
            safe_parts = [secure_filename(fallback) or fallback]
        return safe_parts

    saved = []
    for idx, storage in enumerate(files):
        if not storage or not storage.filename:
            continue
        original_name = storage.filename
        safe_name = secure_filename(original_name) or original_name
        if not safe_name:
            continue
        rel_hint = rel_hints[idx] if idx < len(rel_hints) else safe_name
        relative_parts = _sanitize_relative(rel_hint, safe_name)
        dest_path = target_dir.joinpath(*relative_parts)
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return jsonify({'error': f'Unable to prepare folder for {storage.filename}: {exc}'}), 500
        final_path = Path(_unique_path(str(dest_path)))
        try:
            storage.save(final_path)
            stat = final_path.stat()
            saved.append({
                'name': final_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'relative_path': str(final_path.relative_to(DATA_ROOT))
            })
        except Exception as exc:
            return jsonify({'error': f'Failed to save {storage.filename}: {exc}'}), 500

    return jsonify({
        'saved': saved,
        'target': str(target_dir.relative_to(DATA_ROOT)),
        'total': len(saved)
    })

@app.route('/api/files/list', methods=['GET'])
def list_uploaded_files():
    """Return simple file listing for the upload directory."""
    subdir = (request.args.get('subdir') or '').strip()
    try:
        root_dir = _safe_join(DATA_ROOT, subdir) if subdir else UPLOAD_ROOT
    except ValueError:
        return jsonify({'error': 'Invalid directory'}), 400

    try:
        root_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    files = []
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_file():
            continue
        stat = entry.stat()
        files.append({
            'name': entry.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'relative_path': str(entry.relative_to(DATA_ROOT))
        })

    return jsonify({
        'path': str(root_dir.relative_to(DATA_ROOT)),
        'files': files,
        'count': len(files)
    })

@app.route('/api/files/download')
def download_uploaded_file():
    """Download a single file from the shared data directory."""
    rel_path = (request.args.get('path') or '').strip()
    if not rel_path:
        return jsonify({'error': 'Missing path parameter'}), 400
    try:
        file_path = _safe_join(DATA_ROOT, rel_path)
    except ValueError:
        return jsonify({'error': 'Invalid file path'}), 400
    if not file_path.exists() or not file_path.is_file():
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True, download_name=file_path.name)

@app.route('/api/browse', methods=['POST'])
def browse_directory():
    """Durchsucht Verzeichnisse und gibt Struktur zurueck"""
    data = request.json or {}
    raw_path = (data.get('path') or '').strip()
    include_files = bool(data.get('include_files'))

    # Preferred start dir if none provided
    if not raw_path or raw_path in ('.', '~'):
        cands = _candidate_dirs_from_config()
        if cands:
            raw_path = cands[0]
        else:
            raw_path = os.getcwd()

    # Map incoming path (Windows/WSL -> container mount)
    mapped = _map_incoming_path(raw_path)
    path = os.path.abspath(mapped) if mapped else os.getcwd()

    # If path doesn't exist, try alternates and finally return virtual roots
    if not os.path.exists(path) or not os.path.isdir(path):
        alternates = []
        # flip /host_mnt <-> /mnt
        if mapped.startswith('/host_mnt/'):
            alternates.append(mapped.replace('/host_mnt/', '/mnt/', 1))
        if mapped.startswith('/mnt/'):
            alternates.append(mapped.replace('/mnt/', '/host_mnt/', 1))
        # original raw as-is (maybe already container path)
        alternates.append(raw_path)
        # fallback: useful anchors
        alternates += _candidate_dirs_from_config()
        for cand in alternates:
            try:
                cand_abs = os.path.abspath(cand)
            except Exception:
                continue
            if os.path.isdir(cand_abs):
                path = cand_abs
                break
        else:
            # Nothing found: return virtual roots instead of 404
            return _virtual_roots_payload()

    try:
        items = []
        parent = os.path.dirname(path)
        if parent != path:
            items.append({'name': '..', 'path': parent, 'type': 'parent', 'isDir': True})
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)
            try:
                if os.path.isdir(item_path):
                    items.append({'name': item, 'path': item_path, 'type': 'directory', 'isDir': True})
                elif include_files and os.path.isfile(item_path):
                    items.append({'name': item, 'path': item_path, 'type': 'file', 'isDir': False})
            except PermissionError:
                continue
        return jsonify({'current_path': path, 'items': items})
    except PermissionError:
        return jsonify({'error': 'Keine Berechtigung'}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-path', methods=['POST'])
def validate_path():
    """Prueft ob ein Pfad existiert und welchen Typ er hat"""
    data = request.json
    path = data.get('path', '')
    
    if not path:
        return jsonify({'valid': False, 'message': 'Kein Pfad angegeben'})
    
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        return jsonify({
            'valid': False,
            'message': 'Pfad existiert not',
            'path': path
        })
    
    is_dir = os.path.isdir(path)
    
    # Pruefe ob es ein valides Input-Verzeichnis ist (enthaelt ch0, ch1, etc.)
    has_channels = False
    if is_dir:
        try:
            subdirs = os.listdir(path)
            has_channels = any(d.startswith('ch') and d[2:].isdigit() for d in subdirs)
        except:
            pass
    
    return jsonify({
        'valid': True,
        'isDir': is_dir,
        'hasChannels': has_channels,
        'path': path,
        'message': 'Valides Input-Verzeichnis' if has_channels else ('Verzeichnis' if is_dir else 'Datei')
    })

# ===== Detection API =====
def _load_yaml(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

@app.route('/api/det/config', methods=['GET'])
def det_get_config():
    try:
        path_arg = request.args.get('path')
        if path_arg:
            cfg = _load_yaml(path_arg)
            if not isinstance(cfg, dict):
                cfg = {}
        else:
            cfg = load_det_config() or {}
            
        # merge shared regions from co-expression config
        try:
            co = load_config() or {}
            if isinstance(co.get('region_mapping'), dict):
                cfg['region_mapping'] = co['region_mapping']
            if isinstance(co.get('regions_enabled'), list):
                cfg['regions_enabled'] = co['regions_enabled']
        except Exception:
            pass
        # load cpsam nested config so UI can prefill all parameters
        try:
            cpsam_full = _load_yaml(DET_CPSAM_CONFIG_PATH)
            if isinstance(cpsam_full, dict) and cpsam_full:
                cfg['cpsam'] = cpsam_full
                if 'microscope' in cpsam_full and 'microscope' not in cfg:
                    cfg['microscope'] = cpsam_full['microscope']
        except Exception:
            pass
        cfg.setdefault('microscope', {
            'reference_magnification': 10,
            'current_magnification': 10
        })
        return jsonify(cfg)
    except Exception as e:
        import traceback
        error_msg = f"Error loading detection config: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/det/config', methods=['POST'])
def det_set_config():
    cfg = request.get_json(silent=True) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    save_det_config(cfg)
    # also persist shared regions into co-expression config
    try:
        co = load_config() or {}
        if isinstance(cfg.get('region_mapping'), dict):
            co['region_mapping'] = cfg['region_mapping']
        if 'regions_enabled' in cfg:
            re = cfg.get('regions_enabled') or []
            co['regions_enabled'] = [s for s in re if isinstance(s, str)]
        save_config(co)
    except Exception:
        pass
    # NEW: deep-merge UI cpsam into existing file and persist
    try:
        existing = _load_yaml(DET_CPSAM_CONFIG_PATH)  # may be empty
        if not isinstance(existing, dict):
            existing = {}
        ui_cpsam_full = _build_cpsam_config(cfg, existing)      # build full tree from UI
        base = copy.deepcopy(existing) if existing else {}
        merged = _deep_update(base, ui_cpsam_full) if base else copy.deepcopy(ui_cpsam_full)
        for block in ('paths', 'conditions', 'cellpose', 'filters',
                      'advanced_filtering', 'processing', 'overlays', 'outputs', 'microscope'):
            merged.setdefault(block, {})
        # DO NOT scale before saving - keep baseline values
        _save_yaml(DET_CPSAM_CONFIG_PATH, merged)
    except Exception as e:
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: CPSAM-Config Merge fehlgeschlagen: {e}")

    return jsonify({'status':'success'})


@app.route('/api/assistant/detect', methods=['POST'])
def assistant_detect_advice():
    payload = request.get_json(silent=True) or {}
    notes = payload.get('notes') if isinstance(payload, dict) else None
    try:
        ai_response = _call_detection_assistant(notes)
        return jsonify({'status': 'success', 'response': ai_response})
    except ValueError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except requests.HTTPError as exc:
        details = None
        if exc.response is not None:
            try:
                details = exc.response.json()
            except Exception:
                details = exc.response.text
        return jsonify({'status': 'error', 'message': str(exc), 'details': details}), 502
    except requests.RequestException as exc:
        return jsonify({'status': 'error', 'message': f'Connection error: {exc}'}), 502
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500

@app.route('/api/assistant/chat', methods=['POST'])
def assistant_chat():
    payload = request.get_json(silent=True) or {}
    raw_messages = payload.get('messages') if isinstance(payload, dict) else None
    if not isinstance(raw_messages, list) or not raw_messages:
        return jsonify({'status': 'error', 'message': 'messages must be a non-empty list'}), 400
    include_detection = bool(payload.get('include_detection', True))

    sanitized: list[dict] = []
    for entry in raw_messages:
        if not isinstance(entry, dict):
            continue
        role = entry.get('role')
        content = entry.get('content')
        if role not in ('user', 'assistant'):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        sanitized.append({'role': role, 'content': content})
    if not sanitized or sanitized[-1]['role'] != 'user':
        return jsonify({'status': 'error', 'message': 'last message must be a user message'}), 400

    try:
        reply_text, config_update = _call_assistant_chat(sanitized, include_detection=include_detection)
        return jsonify({'status': 'success', 'response': reply_text, 'config_update': config_update})
    except ValueError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except requests.HTTPError as exc:
        details = None
        if exc.response is not None:
            try:
                details = exc.response.json()
            except Exception:
                details = exc.response.text
        return jsonify({'status': 'error', 'message': str(exc), 'details': details}), 502
    except requests.RequestException as exc:
        return jsonify({'status': 'error', 'message': f'Connection error: {exc}'}), 502
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/assistant/apply', methods=['POST'])
def assistant_apply_update():
    payload = request.get_json(silent=True) or {}
    confirmation = str(payload.get('confirmation') or '').strip().upper()
    if confirmation != 'JA':
        return jsonify({'status': 'error', 'message': 'Aenderung nicht bestaetigt (JA erforderlich).'}), 400
    target = payload.get('target')
    if target != 'detection':
        return jsonify({'status': 'error', 'message': 'Unsupported target for config update.'}), 400

    changes = payload.get('changes')
    if not isinstance(changes, dict) or not changes:
        return jsonify({'status': 'error', 'message': 'Changes must be a non-empty object.'}), 400

    reason = payload.get('reason') if isinstance(payload, dict) else None

    try:
        current_cfg = load_det_config() or {}
        updated_cfg = _deep_update(copy.deepcopy(current_cfg), changes)
        save_det_config(updated_cfg)

        try:
            existing_cpsam = _load_yaml(DET_CPSAM_CONFIG_PATH)
            if not isinstance(existing_cpsam, dict):
                existing_cpsam = {}
        except Exception:
            existing_cpsam = {}
        merged_cpsam = _build_cpsam_config(updated_cfg, existing_cpsam)
        # DO NOT scale before saving - keep baseline values
        try:
            _save_yaml(DET_CPSAM_CONFIG_PATH, merged_cpsam)
        except Exception:
            pass

        snapshot_dir = None
        try:
            snapshot_dir = save_config_snapshot(DATA_ROOT / 'config_snapshots', config_dict=updated_cfg, snapshot_name='assistant_detection_update')
        except Exception:
            snapshot_dir = None

        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Assistant applied config update."
        if reason:
            log_msg += f" Grund: {reason}"
        det_status['log'].append(log_msg)

        response_payload = {'status': 'success', 'config': updated_cfg}
        if snapshot_dir:
            response_payload['snapshot'] = str(snapshot_dir)
        return jsonify(response_payload)
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/det/start', methods=['POST'])
def det_start():
    global det_status
    if det_status['running']:
        return jsonify({'status': 'error', 'message': 'Detection is already running'}), 400

    def ensure_nested(target: dict):
        cpsam_block = target.setdefault('cpsam', {})
        cellpose_block = cpsam_block.setdefault('cellpose', {})
        filters_block = cpsam_block.setdefault('filters', {})
        processing_block = cpsam_block.setdefault('processing', {})
        scope_block = cpsam_block.setdefault('microscope', {})
        target.setdefault('microscope', scope_block)
        return cpsam_block, cellpose_block, filters_block, processing_block

    payload = request.get_json(silent=True)
    cfg = load_det_config() or {}
    if not isinstance(cfg, dict):
        cfg = {}

    structured_keys = (
        'input_tiffs_dir', 'output_results_dir', 'channels', 'cpsam',
        'model_type', 'batch_size', 'flow_threshold', 'cellprob_threshold',
        'save_overlay', 'overlay_suffix', 'use_gpu', 'det_script_path', 'microscope'
    )

    def _payload_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ('true', '1', 'yes', 'y', 'on')
        return bool(val)

    if isinstance(payload, dict) and payload:
        if any(key in payload for key in structured_keys):
            try:
                base = copy.deepcopy(cfg) if cfg else {}
            except Exception:
                base = cfg or {}
            try:
                cfg = _deep_update(base, payload) if base else copy.deepcopy(payload)
            except Exception:
                cfg = payload
        else:
            cpsam_block, cellpose_block, filters_block, processing_block = ensure_nested(cfg)
            scope_block = cpsam_block.setdefault('microscope', {})
            cfg.setdefault('microscope', scope_block)

            def _as_bool(val):
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.strip().lower() in ('true', '1', 'yes', 'y', 'on')
                return bool(val)

            if 'model' in payload:
                cfg['model_type'] = payload['model']
            if 'batch' in payload:
                try:
                    cfg['batch_size'] = int(payload['batch'])
                except Exception:
                    pass
            if 'diameter' in payload and payload['diameter'] not in (None, ''):
                try:
                    cfg['diameter'] = float(payload['diameter'])
                except Exception:
                    pass
            if 'flow_threshold' in payload and payload['flow_threshold'] not in (None, ''):
                try:
                    cfg['flow_threshold'] = float(payload['flow_threshold'])
                except Exception:
                    pass
            if 'cellprob_threshold' in payload and payload['cellprob_threshold'] not in (None, ''):
                try:
                    cfg['cellprob_threshold'] = float(payload['cellprob_threshold'])
                except Exception:
                    pass
            if 'channel' in payload:
                try:
                    cellpose_block['channel'] = int(payload['channel'])
                except Exception:
                    pass
            if 'resize_max' in payload and payload['resize_max'] not in (None, ''):
                try:
                    cellpose_block['resize_max'] = int(payload['resize_max'])
                except Exception:
                    pass
            if 'niter' in payload and payload['niter'] not in (None, ''):
                try:
                    cellpose_block['niter'] = int(payload['niter'])
                except Exception:
                    pass
            if 'min_area' in payload and payload['min_area'] not in (None, ''):
                try:
                    filters_block['min_area'] = int(payload['min_area'])
                except Exception:
                    pass
            if 'max_area' in payload and payload['max_area'] not in (None, ''):
                try:
                    filters_block['max_area'] = int(payload['max_area'])
                except Exception:
                    pass
            if 'min_circ' in payload and payload['min_circ'] not in (None, ''):
                try:
                    filters_block['min_circularity'] = float(payload['min_circ'])
                except Exception:
                    pass
            if 'max_circ' in payload and payload['max_circ'] not in (None, ''):
                try:
                    filters_block['max_circularity'] = float(payload['max_circ'])
                except Exception:
                    pass
            if 'max_hole' in payload and payload['max_hole'] not in (None, ''):
                try:
                    filters_block['max_hole_ratio'] = float(payload['max_hole'])
                except Exception:
                    pass
            if 'tophat' in payload:
                processing_block['tophat'] = _as_bool(payload['tophat'])
            if 'tophat_radius' in payload:
                try:
                    processing_block['tophat_radius'] = int(payload['tophat_radius'])
                except Exception:
                    pass
            if 'contrast_stretch' in payload:
                processing_block['contrast_stretch'] = _as_bool(payload['contrast_stretch'])
            # CLAHE preprocessing
            if 'clahe' in payload:
                processing_block['clahe'] = _as_bool(payload['clahe'])
            if 'clahe_clip_limit' in payload:
                try:
                    processing_block['clahe_clip_limit'] = float(payload['clahe_clip_limit'])
                except Exception:
                    pass
            # Resize max for Cellpose
            if 'resize_max' in payload:
                try:
                    cellpose_block = cfg.get('cellpose', {})
                    if not isinstance(cellpose_block, dict):
                        cellpose_block = {}
                    cellpose_block['resize_max'] = int(payload['resize_max'])
                    cfg['cellpose'] = cellpose_block
                except Exception:
                    pass
            if 'current_magnification' in payload:
                try:
                    scope_block['current_magnification'] = float(payload['current_magnification'])
                except Exception:
                    pass
            if 'reference_magnification' in payload:
                try:
                    scope_block['reference_magnification'] = float(payload['reference_magnification'])
                except Exception:
                    pass
            if 'image_aware_scaling' in payload:
                scope_block['image_aware_scaling'] = _as_bool(payload['image_aware_scaling'])
    else:
        ensure_nested(cfg)

    ensure_nested(cfg)

    raw_channels = cfg.get('channels')
    channel_indices: list[int] = []
    label_map: dict[int, str] = {}

    def _add_channel(idx_val, name_val=None):
        try:
            idx_int = int(idx_val)
        except Exception:
            return
        channel_indices.append(idx_int)
        if name_val is not None:
            label_map[idx_int] = str(name_val).strip()

    if isinstance(raw_channels, list):
        for ch in raw_channels:
            if isinstance(ch, dict):
                _add_channel(ch.get('index'), ch.get('name'))
            else:
                _add_channel(ch)
    elif isinstance(raw_channels, str):
        parts = [p.strip() for p in raw_channels.split(',')]
        for part in parts:
            if part:
                _add_channel(part)
    else:
        # fallback default
        for idx in range(4):
            _add_channel(idx)

    channel_indices = sorted(dict.fromkeys(channel_indices))
    cfg['channels'] = channel_indices

    raw_labels = cfg.get('channel_labels')
    if isinstance(raw_labels, list):
        for item in raw_labels:
            if not isinstance(item, dict):
                continue
            idx = item.get('index')
            name = item.get('name')
            if idx is None:
                continue
            try:
                idx_int = int(idx)
            except Exception:
                continue
            label_map.setdefault(idx_int, str(name or '').strip())
    cfg['channel_labels'] = [{'index': idx, 'name': label_map.get(idx, '')} for idx in channel_indices]

    selection = cfg.get('det_channel_selection') or cfg.get('channel_selection') or []
    normalized_selection = []
    if isinstance(selection, list):
        for value in selection:
            try:
                normalized_selection.append(int(value))
            except Exception:
                continue
    elif isinstance(selection, str):
        for part in selection.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                normalized_selection.append(int(part))
            except Exception:
                continue
    cfg['det_channel_selection'] = [idx for idx in channel_indices if idx in dict.fromkeys(normalized_selection)]

    normalized_levels, _ = _normalize_channel_levels(
        cfg.get('channel_levels') or cfg.get('det_channel_levels') or []
    )
    if normalized_levels:
        cfg['channel_levels'] = normalized_levels
    else:
        cfg.pop('channel_levels', None)

    testing_cfg = cfg.get('testing') if isinstance(cfg.get('testing'), dict) else {}
    testing_payload = None
    if isinstance(payload, dict):
        testing_payload = payload.get('testing')
    if isinstance(testing_payload, dict):
        testing_cfg.update(testing_payload)

    mode_raw = testing_cfg.get('mode')
    mode = str(mode_raw).lower() if isinstance(mode_raw, str) else ''
    enabled_flag = _payload_bool(testing_cfg.get('enabled'))
    try:
        samples = max(0, int(testing_cfg.get('samples_per_channel') or 0))
    except Exception:
        samples = 0
    selected_images_cfg = testing_cfg.get('selected_images') if isinstance(testing_cfg.get('selected_images'), dict) else {}
    normalized_selected_images: dict[str, str] = {}
    if isinstance(selected_images_cfg, dict):
        for key, value in selected_images_cfg.items():
            cleaned = _clean_relative_path_value(value)
            if cleaned:
                normalized_selected_images[key.lower()] = cleaned
    for key in ('pos', 'neg'):
        alt_key = f'selected_image_{key}'
        alt_val = _clean_relative_path_value(testing_cfg.get(alt_key))
        if alt_val:
            normalized_selected_images[key] = alt_val
        if alt_key in testing_cfg:
            testing_cfg.pop(alt_key, None)
    single_image_cfg = _clean_relative_path_value(testing_cfg.get('selected_image'))
    if single_image_cfg:
        normalized_selected_images.setdefault('pos', single_image_cfg)
    if normalized_selected_images:
        testing_cfg['selected_images'] = normalized_selected_images
        primary_val = normalized_selected_images.get('pos') or next(iter(normalized_selected_images.values()))
        if primary_val:
            testing_cfg['selected_image'] = primary_val
        else:
            testing_cfg.pop('selected_image', None)
    else:
        testing_cfg.pop('selected_images', None)
        testing_cfg.pop('selected_image', None)

    if mode not in ('disabled', 'simple', 'advanced'):
        mode = 'simple' if enabled_flag else 'disabled'

    sweep_levels: list[int] = []
    if mode == 'advanced':
        enabled_flag = True
        if samples <= 0:
            samples = 1
        levels_raw = testing_cfg.get('levels')
        sanitized_levels: list[int] = []
        if isinstance(levels_raw, (list, tuple)):
            for value in levels_raw:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if 0 <= idx < len(SENSITIVITY_LEVELS):
                    sanitized_levels.append(idx)
        if not sanitized_levels:
            sanitized_levels = list(range(len(SENSITIVITY_LEVELS)))
        sweep_levels = sorted(dict.fromkeys(sanitized_levels))
        testing_cfg['levels'] = sweep_levels
    else:
        if 'levels' in testing_cfg:
            testing_cfg.pop('levels', None)

    if mode == 'simple':
        enabled_flag = enabled_flag and samples > 0
    elif mode == 'disabled':
        enabled_flag = False
        samples = 0

    testing_cfg['mode'] = mode
    testing_cfg['enabled'] = enabled_flag
    testing_cfg['samples_per_channel'] = samples
    cfg['testing'] = testing_cfg

    if not cfg.get('input_tiffs_dir') or not cfg.get('output_results_dir'):
        return jsonify({
            'status': 'error',
            'message': 'Input and output directories are required for detection. Please set both before starting.'
        }), 400

    save_det_config(cfg)

    thread = threading.Thread(target=run_det_thread, args=(cfg,))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'success', 'message': 'Detection started'})

@app.route('/api/det/stop', methods=['POST'])
def det_stop():
    global det_process, det_status
    if not det_status['running']:
        return jsonify({'status':'error','message':'Keine Detection laeuft'}), 400
    if det_process:
        try:
            det_process.terminate()
            det_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            det_process.kill()
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Detection gestoppt")
        det_status['running'] = False
        det_process = None
        return jsonify({'status':'success'})
    return jsonify({'status':'error','message':'Prozess konnte not gestoppt werden'}), 500

@app.route('/api/det/status', methods=['GET'])
def det_status_api():
    return jsonify(det_status)

@app.route('/api/det/results', methods=['GET'])
def det_results():
    cfg = load_det_config()
    base = cfg.get('output_results_dir','')
    if not base or not os.path.isdir(base):
        return jsonify({'csv_files': [], 'overlays_total': 0})
    csv_files = []
    overlays_total = 0

    # count overlays
    for root, _, files in os.walk(base):
        if os.path.basename(root) == 'overlays':
            overlays_total += sum(1 for fn in files if fn.lower().endswith('.png'))

    # Prefer chX top-level CSVs
    try:
        for d in sorted(os.listdir(base)):
            if d.startswith('__'):
                continue
            if not d.startswith('ch') or not d[2:].isdigit():
                continue
            ch_dir = os.path.join(base, d)
            fp = os.path.join(ch_dir, 'All_Counts_Master.csv')
            if os.path.isfile(fp):
                try:
                    csv_files.append({
                        'name': f"{d}/All_Counts_Master.csv",
                        'path': fp,
                        'size': os.path.getsize(fp),
                        'modified': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S'),
                        'download': f"/api/det/download-csv?path={quote(fp)}"
                    })
                except Exception:
                    continue
    except Exception:
        pass

    # Fallback: legacy recursive listing
    if not csv_files:
        for root, _, files in os.walk(base):
            if '__test__' in root:
                continue
            for fn in files:
                if fn == 'All_Counts_Master.csv':
                    fp = os.path.join(root, fn)
                    try:
                        csv_files.append({
                            'name': f"{os.path.basename(os.path.dirname(root))}/{fn}",
                            'path': fp,
                            'size': os.path.getsize(fp),
                            'modified': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S'),
                            'download': f"/api/det/download-csv?path={quote(fp)}"
                        })
                    except:
                        continue

    csv_files.sort(key=lambda x: x['name'])
    return jsonify({'csv_files': csv_files, 'overlays_total': overlays_total})

@app.route('/api/det/download-csv')
def det_download_csv():
    cfg = load_det_config()
    base = os.path.abspath(cfg.get('output_results_dir',''))
    req_path = request.args.get('path','')
    if not base or not req_path:
        return jsonify({'status':'error','message':'Pfad fehlt'}), 400
    file_path = os.path.abspath(req_path)
    if not file_path.startswith(base):
        return jsonify({'status':'error','message':'Pfad not erlaubt'}), 403
    if not os.path.exists(file_path):
        return jsonify({'status':'error','message':'Datei not gefunden'}), 404
    return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path))

@app.route('/api/det/download-all')
def det_download_all_csv():
    cfg = load_det_config()
    base = os.path.abspath(cfg.get('output_results_dir',''))
    if not base or not os.path.isdir(base):
        return jsonify({'status':'error','message':'Output-Verzeichnis ungueltig'}), 400
    tmp_dir = tempfile.mkdtemp(prefix='det_csv_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = os.path.join(tmp_dir, f'detection_csv_{ts}.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            for fn in files:
                if fn != 'All_Counts_Master.csv':
                    continue
                full = os.path.join(root, fn)
                abs_full = os.path.abspath(full)
                if not abs_full.startswith(base):
                    continue
                rel = os.path.relpath(abs_full, base)
                zf.write(abs_full, arcname=rel)
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass
        return response
    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

@app.route('/api/det/overlays/download-all')
def det_download_all_overlays():
    cfg = load_det_config()
    base = os.path.abspath(cfg.get('output_results_dir',''))
    if not base or not os.path.isdir(base):
        return jsonify({'status':'error','message':'Output-Verzeichnis ungueltig'}), 400
    tmp_dir = tempfile.mkdtemp(prefix='det_ov_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = os.path.join(tmp_dir, f'detection_overlays_{ts}.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            if "__test__" in root:
                continue
            if os.path.basename(root) != 'overlays':
                continue
            for fn in files:
                if not fn.lower().endswith('.png'):
                    continue
                full = os.path.join(root, fn)
                abs_full = os.path.abspath(full)
                if not abs_full.startswith(base):
                    continue
                rel = os.path.relpath(abs_full, base)
                zf.write(abs_full, arcname=rel)
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass
        return response
    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))


if __name__ == '__main__':
    # Erstelle templates Ordner falls nicht vorhanden
    os.makedirs('templates', exist_ok=True)

    print("Starting Cell Analysis Web Interface...")
    print("Pipeline: CZI Conversion -> Cell Detection -> Co-Expression")
    print("Open browser at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)





