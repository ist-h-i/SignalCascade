from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

FeatureVector = tuple[float, float, float, float, float, float]
ShapeVector = tuple[float, float, float]


@dataclass(frozen=True)
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class TimeframeFeatureRow:
    timestamp: datetime
    close: float
    shape: ShapeVector
    vector: FeatureVector


@dataclass(frozen=True)
class TrainingExample:
    anchor_time: datetime
    main_sequences: dict[str, list[FeatureVector]]
    overlay_sequences: dict[str, list[FeatureVector]]
    main_shape_targets: dict[str, ShapeVector]
    returns_target: tuple[float, ...]
    overlay_target: int
    current_close: float


@dataclass(frozen=True)
class PredictionResult:
    anchor_time: str
    selected_horizon: int
    position: float
    predicted_closes: dict[str, float]
    uncertainties: dict[str, float]
    overlay_action: str
