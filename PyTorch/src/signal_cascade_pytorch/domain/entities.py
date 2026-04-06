from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from .historical_compatibility import (
    build_prediction_legacy_compatibility,
    prediction_selected_direction,
)

FeatureVector = tuple[float, float, float, float, float, float]
ShapeVector = tuple[float, float, float]
RegimeVector = tuple[float, float, float, float, float]
StateFeatureVector = tuple[float, ...]

TIMEFRAME_FEATURE_NAMES = (
    "ell_log_return",
    "b_real_body",
    "u_upper_shadow",
    "d_lower_shadow",
    "nu_volume_anomaly",
    "zeta_ema_deviation",
)
SHAPE_COMPONENT_NAMES = ("upper_share", "body_share", "lower_share")
REGIME_FEATURE_NAMES = (
    "session_asia",
    "session_london",
    "session_ny",
    "volatility_regime_offset",
    "regime_trend_strength",
)
STATE_FEATURE_NAMES = (
    *REGIME_FEATURE_NAMES,
    "realized_volatility_30m",
    "baseline_volatility_30m",
    "volatility_ratio_offset_30m",
    "anchor_volume_anomaly_30m",
    "anchor_ema_deviation_30m",
)
STATE_VECTOR_COMPONENT_NAMES = ("h_t", "s_t", "z_t", "m_t")
TRAINING_EXAMPLE_CONTRACT_VERSION = 1


def _validate_vector_length(
    values: Sequence[float],
    expected_names: Sequence[str],
    label: str,
) -> None:
    if len(values) != len(expected_names):
        raise ValueError(
            f"{label} must contain {len(expected_names)} values in order "
            f"{tuple(expected_names)}; received {len(values)}."
        )


def named_feature_dict(
    values: Sequence[float],
    expected_names: Sequence[str],
    label: str,
) -> dict[str, float]:
    _validate_vector_length(values, expected_names, label)
    return {
        str(name): float(value)
        for name, value in zip(expected_names, values)
    }


def build_state_feature_vector(
    *,
    regime_features: RegimeVector,
    realized_volatility: float,
    baseline_volatility: float,
    volatility_ratio_offset: float,
    anchor_volume_anomaly: float,
    anchor_ema_deviation: float,
) -> StateFeatureVector:
    state_features: StateFeatureVector = (
        *tuple(float(value) for value in regime_features),
        float(realized_volatility),
        float(baseline_volatility),
        float(volatility_ratio_offset),
        float(anchor_volume_anomaly),
        float(anchor_ema_deviation),
    )
    _validate_vector_length(state_features, STATE_FEATURE_NAMES, "TrainingExample.state_features")
    return state_features


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

    def __post_init__(self) -> None:
        _validate_vector_length(
            self.shape,
            SHAPE_COMPONENT_NAMES,
            "TimeframeFeatureRow.shape",
        )
        _validate_vector_length(
            self.vector,
            TIMEFRAME_FEATURE_NAMES,
            "TimeframeFeatureRow.vector",
        )

    @property
    def feature_map(self) -> dict[str, float]:
        return named_feature_dict(
            self.vector,
            TIMEFRAME_FEATURE_NAMES,
            "TimeframeFeatureRow.vector",
        )


@dataclass(frozen=True)
class TrainingExample:
    anchor_time: datetime
    main_sequences: dict[str, list[FeatureVector]]
    overlay_sequences: dict[str, list[FeatureVector]]
    main_shape_targets: dict[str, ShapeVector]
    state_features: StateFeatureVector
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
    price_scale: float = 1.0

    def __post_init__(self) -> None:
        for scope_name, sequences in (
            ("main_sequences", self.main_sequences),
            ("overlay_sequences", self.overlay_sequences),
        ):
            for timeframe, rows in sequences.items():
                if not rows:
                    raise ValueError(f"TrainingExample.{scope_name}[{timeframe!r}] must not be empty.")
                for index, vector in enumerate(rows):
                    _validate_vector_length(
                        vector,
                        TIMEFRAME_FEATURE_NAMES,
                        f"TrainingExample.{scope_name}[{timeframe!r}][{index}]",
                    )

        for timeframe, shape_target in self.main_shape_targets.items():
            _validate_vector_length(
                shape_target,
                SHAPE_COMPONENT_NAMES,
                f"TrainingExample.main_shape_targets[{timeframe!r}]",
            )

        _validate_vector_length(
            self.regime_features,
            REGIME_FEATURE_NAMES,
            "TrainingExample.regime_features",
        )
        _validate_vector_length(
            self.state_features,
            STATE_FEATURE_NAMES,
            "TrainingExample.state_features",
        )

        if tuple(float(value) for value in self.regime_features) != tuple(
            float(value) for value in self.state_features[: len(REGIME_FEATURE_NAMES)]
        ):
            raise ValueError("TrainingExample.state_features must begin with regime_features.")
        if not math.isclose(
            float(self.trend_strength),
            float(self.regime_features[4]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "TrainingExample.trend_strength must match regime_features['regime_trend_strength']."
            )
        if not math.isclose(
            float(self.trend_strength),
            float(self.state_features[4]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "TrainingExample.trend_strength must match state_features['regime_trend_strength']."
            )
        if not math.isclose(
            float(self.realized_volatility),
            float(self.state_features[5]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "TrainingExample.realized_volatility must match "
                "state_features['realized_volatility_30m']."
            )

    @property
    def state_feature_map(self) -> dict[str, float]:
        return named_feature_dict(
            self.state_features,
            STATE_FEATURE_NAMES,
            "TrainingExample.state_features",
        )


@dataclass(frozen=True)
class PredictionResult:
    anchor_time: str
    current_close: float
    policy_horizon: int
    executed_horizon: int | None
    previous_position: float
    position: float
    trade_delta: float
    no_trade_band_hit: bool
    tradeability_gate: float
    shape_entropy: float
    policy_score: float
    expected_log_returns: dict[str, float]
    predicted_closes: dict[str, float]
    uncertainties: dict[str, float]
    horizon_utilities: dict[str, float]
    horizon_positions: dict[str, float]
    shape_probabilities: dict[str, float]
    regime_id: str
    policy_log_returns: dict[str, float] | None = None
    policy_uncertainties: dict[str, float] | None = None
    inference_context_mode: str = "stateless"
    price_scale: float = 1.0

    @property
    def selected_direction(self) -> int:
        return prediction_selected_direction(self)

    @property
    def legacy_compatibility(self) -> dict[str, object]:
        return build_prediction_legacy_compatibility(self)
