from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

FeatureVector = tuple[float, float, float, float, float, float]
ShapeVector = tuple[float, float, float]
RegimeVector = tuple[float, float, float, float, float]


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
    long_mae: tuple[float, ...]
    short_mae: tuple[float, ...]
    long_mfe: tuple[float, ...]
    short_mfe: tuple[float, ...]
    direction_targets: tuple[int, ...]
    direction_weights: tuple[float, ...]
    direction_thresholds: tuple[float, ...]
    direction_mae_thresholds: tuple[float, ...]
    horizon_costs: tuple[float, ...]
    overlay_target: int
    current_close: float
    regime_id: str
    regime_features: RegimeVector
    realized_volatility: float
    trend_strength: float


@dataclass(frozen=True)
class PredictionResult:
    anchor_time: str
    current_close: float
    selected_horizon: int | None
    selected_direction: int
    position: float
    expected_log_returns: dict[str, float]
    predicted_closes: dict[str, float]
    uncertainties: dict[str, float]
    accepted_signal: bool
    selection_probability: float
    selection_score: float
    selection_threshold: float | None
    correctness_probability: float
    hold_probability: float
    hold_threshold: float
    overlay_action: str
    regime_id: str
