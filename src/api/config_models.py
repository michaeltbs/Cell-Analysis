"""
src/api/config_models.py — Pydantic models for validating pipeline configs.
Catches typos and missing keys early (especially for HF Space uploads).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CellposeConfig(BaseModel):
    model_name: str = "cyto2"
    use_gpu: bool = False
    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    save_masks: bool = True
    split_regions: bool = False
    channel: int = 0
    batch_size: int = 8
    resize_max: int = 2048
    min_size: int = 15
    max_size_fraction: float = 0.4

    @field_validator("model_name")
    @classmethod
    def _valid_model(cls, v: str) -> str:
        allowed = {"cyto3", "cyto2", "cyto", "nuclei", "cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"}
        return v if v in allowed else "cyto2"


class FilterConfig(BaseModel):
    min_area: int = 10
    max_area: int = 32000
    min_circularity: float = 0.0
    max_circularity: float = 1.0


class OutputConfig(BaseModel):
    base_name: str = "analysis_results"
    create_overlays: bool = False


class PathsConfig(BaseModel):
    input_pos: str = ""
    input_neg: str = ""
    output_root: str = "./results_det"


class DetectionConfig(BaseModel):
    paths: PathsConfig
    cellpose: CellposeConfig = Field(default_factory=CellposeConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    analysis: Dict[str, bool] = Field(default_factory=lambda: {"enable_anova": True})
    testing: Dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionConfig":
        return cls.model_validate(data)


class ChannelConfig(BaseModel):
    index: int
    name: str = "Channel"
    enabled: bool = True
    color: str = "#ff0000"


class CoexprConfig(BaseModel):
    input_dir: str
    output_dir: str
    channel_config: List[ChannelConfig] = Field(default_factory=list)
    coexpr_mode: str = "overlap"
    overlap_threshold: int = 30
    centroid_overlap_fraction: float = 0.5
    selected_combos: List[List[int]] = Field(default_factory=list)
    combos_enabled: List[int] = Field(default_factory=list)
    use_masks: bool = False
    fallback_from_overlays: bool = True
    save_figure: bool = False
    test_mode: Dict[str, object] = Field(default_factory=lambda: {"type": "off"})

    @field_validator("coexpr_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        allowed = {"overlap", "centroid", "intersection", "union", "blend_heatmap"}
        return v if v in allowed else "overlap"

    @classmethod
    def from_dict(cls, data: dict) -> "CoexprConfig":
        return cls.model_validate(data)
