import os
import json
import re
import requests
import yaml
from typing import Optional, Tuple, List, Dict
from src.constants import DEFAULT_AI_CONFIG, AI_CONFIG_PATH

CONFIG_UPDATE_PATTERN = re.compile(r"`config_update\s*(\{.*?\})\s*`", re.DOTALL)

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
        # Create default if missing
        try:
            sample = {
                'base_url': cfg['base_url'],
                'api_key': cfg['api_key'],
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

def extract_config_update(raw_text: str) -> Tuple[str, Optional[dict]]:
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

def call_assistant_chat(messages: List[dict]) -> Tuple[str, Optional[dict]]:
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

    payload = {
        'model': ai_cfg.get('model_id') or DEFAULT_AI_CONFIG['model_id'],
        'messages': messages,
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
    return extract_config_update(raw_text)
