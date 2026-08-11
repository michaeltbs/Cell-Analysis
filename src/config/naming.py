"""
src/config/naming.py — universal channel/condition naming config.
Allows users to map arbitrary channel/condition names to internal indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


DEFAULT_CHANNEL_NAMES: Dict[int, str] = {
    0: "Channel_0",
    1: "Channel_1",
    2: "Channel_2",
    3: "Channel_3",
}


@dataclass
class NamingConfig:
    channel_names: Dict[int, str] = field(default_factory=lambda: DEFAULT_CHANNEL_NAMES.copy())
    condition_names: Dict[str, str] = field(default_factory=lambda: {"pos": "Positive", "neg": "Negative"})
    region_names: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "NamingConfig":
        cfg = cls()
        if "channel_names" in data:
            cfg.channel_names = {int(k): str(v) for k, v in data["channel_names"].items()}
        if "condition_names" in data:
            cfg.condition_names = {str(k): str(v) for k, v in data["condition_names"].items()}
        if "region_names" in data:
            cfg.region_names = {str(k): str(v) for k, v in data["region_names"].items()}
        return cfg

    def channel_name(self, index: int) -> str:
        return self.channel_names.get(index, f"Channel_{index}")

    def condition_name(self, key: str) -> str:
        return self.condition_names.get(key.lower(), key)

    def region_name(self, key: str) -> str:
        return self.region_names.get(key.lower(), key)


def apply_naming_to_csv_rows(rows: List[dict], naming: NamingConfig) -> List[dict]:
    """Rewrite channel/condition/region names in CSV-like rows."""
    out = []
    for row in rows:
        new_row = dict(row)
        if "channel" in new_row:
            idx = int(new_row["channel"]) if str(new_row["channel"]).isdigit() else None
            if idx is not None:
                new_row["channel_label"] = naming.channel_name(idx)
        if "condition" in new_row:
            new_row["condition_label"] = naming.condition_name(str(new_row["condition"]))
        if "region" in new_row:
            new_row["region_label"] = naming.region_name(str(new_row["region"]))
        out.append(new_row)
    return out
