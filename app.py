from flask import Flask, render_template, request, jsonify, send_file, after_this_request
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

app = Flask(__name__)

# Globale Variablen fÃ¼r Analyse-Status
analysis_status = {
    'running': False,
    'progress': 0,
    'current_file': '',
    'log': []
}

# Globale Variable fÃ¼r Prozesssteuerung
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

# detection state
det_status = { 'running': False, 'log': [] }
det_process = None
DET_CONFIG_PATH = 'config.yaml'
# NEW: separate CPSAM config that matches batch_segment.py schema
DET_CPSAM_CONFIG_PATH = 'config_det_cpsam.yaml'

def load_config():
    """LÃ¤dt die Konfiguration aus config_analysis.yaml"""
    config_path = 'config_analysis.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_config(config):
    """Speichert die Konfiguration in config_analysis.yaml"""
    config_path = 'config_analysis.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# NEU: CZI Config Funktionen
def load_czi_config():
    """LÃ¤dt CZI-Konvertierungs-Konfiguration"""
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
        'stack_layout': 'YXC',
        'imagej_hyperstack': True,
        'channel_names': [],
        # NEW
        'save_color_composite': True,
        'channel_colors': ['#ff0000', '#00ff00', '#0000ff', '#ffff00'],
        # NEW:
        'save_multichannel_colored_pages': True,
        'save_ome_tiff_colors': True,
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
    """FÃ¼hrt CZI-Konvertierung in separatem Thread aus"""
    global czi_status, czi_process
    try:
        czi_status['running'] = True
        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] CZI Konvertierung gestartet...")

        # Speichere Config temporÃ¤r
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
            czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âœ… Konvertierung erfolgreich! ({czi_status['done']} files)")
        else:
            czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âŒ Konvertierung fehlgeschlagen (Code: {czi_process.returncode})")

    except Exception as e:
        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âŒ Fehler: {str(e)}")
    finally:
        czi_status['running'] = False
        czi_process = None

def run_analysis_thread(config):
    """FÃ¼hrt die Analyse in einem separaten Thread aus"""
    global analysis_status, analysis_process
    
    try:
        analysis_status['running'] = True
        analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Analyse gestartet...")
        
        # Speichere temporÃ¤re Konfiguration
        save_config(config)
        
        # Debug-Flag aus Payload lesen
        debug_flag = bool(config.get('debug', False))
        cmd = [sys.executable, 'analysis_combinations.py', '--config', 'config_analysis.yaml']
        if debug_flag:
            cmd.append('-v')

        # Starte Analyse-Skript mit gleichem Interpreter und explizitem Config-Pfad
        analysis_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd()
        )
        
        # Lese Output Zeile fÃ¼r Zeile
        for line in analysis_process.stdout:
            analysis_status['log'].append(line.strip())
            # Begrenze Log auf letzte 100 Zeilen
            if len(analysis_status['log']) > 100:
                analysis_status['log'] = analysis_status['log'][-100:]
        
        analysis_process.wait()
        
        if analysis_process.returncode == 0:
            analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âœ… Analyse erfolgreich abgeschlossen!")
        else:
            analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âŒ Analyse fehlgeschlagen (Code: {analysis_process.returncode})")
            
    except Exception as e:
        analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âŒ Fehler: {str(e)}")
    finally:
        analysis_status['running'] = False
        analysis_status['progress'] = 100
        analysis_process = None

def load_det_config():
    """LÃ¤dt Detection-Konfiguration (config.yaml)"""
    if os.path.exists(DET_CONFIG_PATH):
        with open(DET_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
            # ensure new key exists to roundtrip with UI
            cfg.setdefault('det_script_path', '')
            cfg.setdefault('verbose', False)  # NEW
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
        'advanced_filtering', 'processing', 'overlays', 'outputs'
    ):
        cfg.setdefault(section, {})

    # 1) Overlay CPSAM block from UI if present
    cpsam_ui = ui_cfg.get('cpsam') or {}
    if cpsam_ui:
        _deep_update(cfg, cpsam_ui)

    # 2) Paths from top-level UI (preferred)
    inp_root = (ui_cfg.get('input_tiffs_dir') or '').strip()
    out_root = (ui_cfg.get('output_results_dir') or '').strip()
    if inp_root:
        cfg.setdefault('paths', {})
        cfg['paths']['input_pos'] = os.path.join(inp_root, 'Input_pos')
        cfg['paths']['input_neg'] = os.path.join(inp_root, 'Input_neg')
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

def _merge_master_csvs(base_dir: str):
    """Merge per-channel CSVs and keep channel labels."""
    import csv
    base = os.path.abspath(base_dir)
    if not os.path.isdir(base):
        return
    target = os.path.join(base, 'All_Counts_Master.csv')
    legacy_order = [
        'filename','condition','region','channel','channel_index','cell_count',
        'mean_area_per_cell','mean_intensity_per_cell','mean_integrated_density_per_cell'
    ]
    rows, sources = [], []
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
                sources.append(fp)
            except Exception:
                continue

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
    """Startet batch_segment.py und streamt Logs (unterstützt Multi-Channel)."""
    global det_status, det_process
    try:
        det_status['running'] = True
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Detection gestartet...")

        save_det_config(cfg)

        try:
            existing_cpsam = _load_yaml(DET_CPSAM_CONFIG_PATH)
            if not isinstance(existing_cpsam, dict):
                existing_cpsam = {}
        except Exception:
            existing_cpsam = {}
        cpsam_cfg = _build_cpsam_config(cfg, existing_cpsam)
        cpsam_cfg_path = str(Path(DET_CPSAM_CONFIG_PATH).resolve())

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
        testing_cfg = {
            'enabled': bool(raw_testing.get('enabled')),
            'samples_per_channel': int(raw_testing.get('samples_per_channel') or 0),
        }
        seed_val = raw_testing.get('seed')
        if seed_val not in (None, ''):
            testing_cfg['seed'] = str(seed_val)
        cpsam_cfg['testing'] = testing_cfg
        testing_enabled = testing_cfg['enabled'] and testing_cfg.get('samples_per_channel', 0) > 0
        if testing_enabled:
            seed_info = f" seed={testing_cfg['seed']}" if testing_cfg.get('seed') else ''
            det_status['log'].append("[{}] [info] Test mode active: samples_per_channel={}{}".format(
                datetime.now().strftime('%H:%M:%S'),
                testing_cfg['samples_per_channel'],
                seed_info
            ))

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
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ batch_segment.py nicht gefunden.")
            det_status['log'].append("  Tried:")
            for entry in tried:
                det_status['log'].append(f"   - {entry}")
            det_status['log'].append("  Tipp: 'Detection Script Path' setzen oder batch_segment.py nach /app kopieren.")
            det_status['running'] = False
            det_process = None
            return

        env = os.environ.copy()
        if bool(cfg.get('use_gpu')):
            env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '')
        if bool(cfg.get('verbose')):
            env['VERBOSE'] = '1'
            env['PYTHONUNBUFFERED'] = '1'

        overall_success = True
        completed_channels: list[int] = []

        for ch_idx in selected_channels:
            label_suffix = f" ({labels.get(ch_idx)})" if labels.get(ch_idx) else ""
            ch_out = os.path.normpath(os.path.join(base_out_root, f"ch{ch_idx}"))
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Kanal ch{ch_idx}{label_suffix}: output -> {ch_out}")

            cp_block['channel'] = ch_idx
            paths['output_root'] = ch_out

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
                det_status['log'].append(
                    f"  model={cp_view.get('model_name','cpsam')} channel={cp_view.get('channel')} "
                    f"batch={cp_view.get('batch_size')} diameter={cp_view.get('diameter')} use_gpu={cp_view.get('use_gpu')}"
                )
                det_status['log'].append(
                    f"  flow_thr={cp_view.get('flow_threshold')} cellprob_thr={cp_view.get('cellprob_threshold')} "
                    f"resize_max={cp_view.get('resize_max')} niter={cp_view.get('niter')}"
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
            cmd = [sys.executable, str(script), '--config', cfg_to_use]
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Starte Kanal ch{ch_idx}: {cmd}")

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
                        _merge_master_csvs(out_root)
                        _prune_empty_dirs(out_root, keep={os.path.join(out_root, 'overlays'), os.path.join(out_root, 'masks')})
                except Exception as post_err:
                    det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: Postprocess fehlgeschlagen (ch{ch_idx}): {post_err}")
                det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Kanal ch{ch_idx}{label_suffix} abgeschlossen.")
                completed_channels.append(ch_idx)
            else:
                det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Kanal ch{ch_idx}{label_suffix} fehlgeschlagen (Code: {det_process.returncode})")
                overall_success = False
                break
            det_process = None
            if tmp_cfg_path and os.path.exists(tmp_cfg_path):
                try:
                    os.remove(tmp_cfg_path)
                except Exception:
                    pass

        if overall_success and completed_channels:
            suffix = 'e' if len(completed_channels) != 1 else ''
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Detection erfolgreich ({len(completed_channels)} Kanal{suffix}).")
        elif not completed_channels:
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Detection konnte nicht gestartet werden.")
        else:
            det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Detection vorzeitig beendet.")

    except Exception as e:
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Fehler: {e}")
    finally:
        det_status['running'] = False
        det_process = None

def _map_incoming_path(p: str) -> str:
    """
    Map incoming paths for the current runtime:
    - Windows-style paths (e.g. C:\\...) â†’ /mnt/<drive>/... (WSL)
    - Leave /mnt/... and other POSIX paths unchanged
    Note: Do NOT rewrite to /host_mnt; that is Docker-specific and breaks on native WSL.
    """
    if not p:
        return os.getcwd()
    # Windows path â†’ WSL path
    if '\\' in p and ':' in p:
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    # Already POSIX (/mnt/..., /home/..., etc.) â†’ leave as is
    return p

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
        os.getcwd(), '/app', '/', '/host_mnt', '/host_mnt/c', '/mnt', '/mnt/c'
    ]
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
    items = []
    for a in anchors:
        name = a
        if a == '/app':
            name = 'APP (/app)'
        elif a == os.getcwd():
            name = f'WORKDIR ({a})'
        elif a.startswith('/host_mnt/'):
            name = a.replace('/host_mnt/', 'HOST ')
        elif a.startswith('/mnt/'):
            name = a.replace('/mnt/', 'WSL ')
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
    return render_template('index.html')

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
        return jsonify({'status': 'error', 'message': 'Konvertierung lÃ¤uft bereits'}), 400
    
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
        return jsonify({'status': 'error', 'message': 'Keine Konvertierung lÃ¤uft'}), 400
    
    if czi_process:
        try:
            czi_process.terminate()
            czi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            czi_process.kill()
        
        czi_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âš ï¸ Konvertierung gestoppt")
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
    """LÃ¤dt eine konvertierte TIFF-Datei herunter"""
    config = load_czi_config()
    base = os.path.abspath(config.get('output_base', ''))
    req_path = request.args.get('path', '')
    if not base or not req_path:
        return jsonify({'status': 'error', 'message': 'Pfad fehlt'}), 400

    file_path = os.path.abspath(req_path)
    # Path traversal protection
    if not file_path.startswith(base):
        return jsonify({'status': 'error', 'message': 'Pfad not erlaubt'}), 403
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'Datei not gefunden'}), 404

    return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path))

@app.route('/api/czi/download-all')
def download_czi_all_zip():
    # Erstelle ein ZIP mit allen .tif/.tiff Dateien im Output-Verzeichnis
    config = load_czi_config()
    base = os.path.abspath(config.get('output_base', ''))
    if not base or not os.path.isdir(base):
        return jsonify({'status': 'error', 'message': 'Output-Verzeichnis ungÃ¼ltig'}), 400

    # TemporÃ¤re ZIP-Datei
    tmp_dir = tempfile.mkdtemp(prefix='czi_zip_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = os.path.join(tmp_dir, f'czi_converted_{ts}.zip')

    # Packe rekursiv alle TIFFs mit relativen Pfaden
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
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
    try:
        config = load_config() or {}
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
    config = request.json
    save_config(config)
    return jsonify({'status': 'success', 'message': 'Konfiguration gespeichert'})

@app.route('/api/start', methods=['POST'])
def start_analysis():
    """Startet die Analyse"""
    global analysis_status
    
    if analysis_status['running']:
        return jsonify({'status': 'error', 'message': 'Analyse lÃ¤uft bereits'}), 400
    
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
        return jsonify({'status': 'error', 'message': 'Keine Analyse lÃ¤uft'}), 400
    
    if analysis_process:
        try:
            analysis_process.terminate()
            analysis_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            analysis_process.kill()
        
        analysis_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âš ï¸ Analyse vom Benutzer gestoppt")
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
    """Listet verfÃ¼gbare Ergebnis-Dateien"""
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
    """LÃ¤dt eine Ergebnis-Datei herunter"""
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
        return jsonify({'status': 'error', 'message': 'Output-Verzeichnis ungÃ¼ltig'}), 400

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
            if os.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass
        return response

    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

@app.route('/api/browse', methods=['POST'])
def browse_directory():
    """Durchsucht Verzeichnisse und gibt Struktur zurÃ¼ck"""
    data = request.json or {}
    raw_path = (data.get('path') or '').strip()

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
                is_dir = os.path.isdir(item_path)
                if is_dir:
                    items.append({'name': item, 'path': item_path, 'type': 'directory', 'isDir': True})
            except PermissionError:
                continue
        return jsonify({'current_path': path, 'items': items})
    except PermissionError:
        return jsonify({'error': 'Keine Berechtigung'}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-path', methods=['POST'])
def validate_path():
    """PrÃ¼ft ob ein Pfad existiert und welchen Typ er hat"""
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
    
    # PrÃ¼fe ob es ein valides Input-Verzeichnis ist (enthÃ¤lt ch0, ch1, etc.)
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
    except Exception:
        pass
    return jsonify(cfg)

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
                      'advanced_filtering', 'processing', 'overlays', 'outputs'):
            merged.setdefault(block, {})
        _save_yaml(DET_CPSAM_CONFIG_PATH, merged)
    except Exception as e:
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: CPSAM-Config Merge fehlgeschlagen: {e}")

    return jsonify({'status':'success'})

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
        return cpsam_block, cellpose_block, filters_block, processing_block

    payload = request.get_json(silent=True)
    cfg = load_det_config() or {}
    if not isinstance(cfg, dict):
        cfg = {}

    structured_keys = (
        'input_tiffs_dir', 'output_results_dir', 'channels', 'cpsam',
        'model_type', 'batch_size', 'flow_threshold', 'cellprob_threshold',
        'save_overlay', 'overlay_suffix', 'use_gpu', 'det_script_path'
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
            if 'contrast_stretch' in payload:
                processing_block['contrast_stretch'] = _as_bool(payload['contrast_stretch'])
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

    testing_cfg = cfg.get('testing') if isinstance(cfg.get('testing'), dict) else {}
    testing_payload = None
    if isinstance(payload, dict):
        testing_payload = payload.get('testing')
    if isinstance(testing_payload, dict):
        testing_cfg.update(testing_payload)
    testing_cfg['enabled'] = _payload_bool(testing_cfg.get('enabled'))
    try:
        testing_cfg['samples_per_channel'] = max(0, int(testing_cfg.get('samples_per_channel') or 0))
    except Exception:
        testing_cfg['samples_per_channel'] = 0
    seed_val = testing_cfg.get('seed')
    if seed_val in (None, ''):
        testing_cfg.pop('seed', None)
    else:
        testing_cfg['seed'] = str(seed_val)
    cfg['testing'] = testing_cfg

    if not cfg.get('input_tiffs_dir') or not cfg.get('output_results_dir'):
        return jsonify({'status':'error','message':'Input/Output Pfade fehlen'}), 400

    save_det_config(cfg)

    thread = threading.Thread(target=run_det_thread, args=(cfg,))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'success', 'message': 'Detection started'})

@app.route('/api/det/stop', methods=['POST'])
def det_stop():
    global det_process, det_status
    if not det_status['running']:
        return jsonify({'status':'error','message':'Keine Detection lÃ¤uft'}), 400
    if det_process:
        try:
            det_process.terminate()
            det_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            det_process.kill()
        det_status['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] âš ï¸ Detection gestoppt")
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
        return jsonify({'status':'error','message':'Output-Verzeichnis ungÃ¼ltig'}), 400
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
            if os.exists(zip_path): os.remove(zip_path)
            if os.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass
        return response
    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

@app.route('/api/det/overlays/download-all')
def det_download_all_overlays():
    cfg = load_det_config()
    base = os.path.abspath(cfg.get('output_results_dir',''))
    if not base or not os.path.isdir(base):
        return jsonify({'status':'error','message':'Output-Verzeichnis ungÃ¼ltig'}), 400
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
            if os.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass
        return response
    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

if __name__ == '__main__':
    # Erstelle templates Ordner falls not vorhanden
    os.makedirs('templates', exist_ok=True)
    
    print("ðŸŒ Starting Cell Analysis Web Interface...")
    print("ðŸ“Š Pipeline: CZI Conversion â†’ Cell Detection â†’ Co-Expression")
    print("ðŸ”— Open browser at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)









