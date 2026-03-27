from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

FeatureVector = tuple[float, float, float, float, float, float]
ShapeVector = tuple[float, float, float]
RegimeVector = tuple[float, float, float, float, float]
StateFeatureVector = tuple[float, ...]


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

    @property
    def proposed_horizon(self) -> int:
        return self.policy_horizon

    @property
    def accepted_horizon(self) -> int | None:
        return self.executed_horizon

    @property
    def selected_horizon(self) -> int:
        return self.policy_horizon

    @property
    def selected_direction(self) -> int:
        mean = self.expected_log_returns.get(str(self.policy_horizon), 0.0)
        if mean > 0.0:
            return 1
        if mean < 0.0:
            return -1
        return 0

    @property
    def accepted_signal(self) -> bool:
        return self.executed_horizon is not None and (
            abs(self.position) > 1e-9 or abs(self.trade_delta) > 1e-9
        )

    @property
    def selection_probability(self) -> float:
        return self.tradeability_gate

    @property
    def selection_score(self) -> float:
        return self.policy_score

    @property
    def selection_threshold(self) -> None:
        return None

    @property
    def threshold_status(self) -> str:
        return "retired"

    @property
    def threshold_origin(self) -> str:
        return "profit_policy"

    @property
    def correctness_probability(self) -> float:
        return self.tradeability_gate

    @property
    def hold_probability(self) -> float:
        return max(0.0, 1.0 - min(abs(self.position), 1.0))

    @property
    def hold_threshold(self) -> float:
        return 0.0

    @property
    def overlay_action(self) -> str:
        return "hold" if self.no_trade_band_hit else "reduce"
