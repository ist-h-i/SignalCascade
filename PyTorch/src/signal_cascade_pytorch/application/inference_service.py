from __future__ import annotations

import math
from datetime import datetime, timedelta

import torch

from .config import TrainingConfig
from .policy_service import apply_selection_policy
from .training_service import examples_to_batch
from ..domain.entities import PredictionResult, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel

PREDICTION_SCHEMA_VERSION = 5
FORECAST_SCHEMA_VERSION = 5


def predict_latest(
    model: SignalCascadeModel,
    examples: list[TrainingExample],
    config: TrainingConfig,
    previous_position: float = 0.0,
) -> PredictionResult:
    return predict_from_example(
        model=model,
        example=examples[-1],
        config=config,
        previous_position=previous_position,
    )


def predict_from_example(
    model: SignalCascadeModel,
    example: TrainingExample,
    config: TrainingConfig,
    previous_position: float = 0.0,
) -> PredictionResult:
    batch = examples_to_batch([example], config)
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["main"],
            batch["overlay"],
            batch["state_features"],
            previous_state=None,
        )

    mean = outputs["mu"][0]
    sigma = outputs["sigma"][0]
    tradeability_gate = float(outputs["tradeability_gate"][0].item())
    decision = apply_selection_policy(
        example=example,
        mean=mean.tolist(),
        sigma=sigma.tolist(),
        config=config,
        previous_position=previous_position,
        tradeability_gate=tradeability_gate,
        shape_probs=outputs["shape_posterior"][0].tolist(),
    )

    predicted_closes = {
        str(horizon): example.current_close * math.exp(float(mean[index].item()))
        for index, horizon in enumerate(config.horizons)
    }
    expected_log_returns = {
        str(horizon): float(mean[index].item())
        for index, horizon in enumerate(config.horizons)
    }
    uncertainties = {
        str(horizon): float(sigma[index].item())
        for index, horizon in enumerate(config.horizons)
    }
    horizon_utilities = {
        str(row["horizon"]): float(row["policy_utility"]) for row in decision["horizon_rows"]
    }
    horizon_positions = {
        str(row["horizon"]): float(row["position"]) for row in decision["horizon_rows"]
    }
    shape_probabilities = {
        str(index): float(value)
        for index, value in enumerate(outputs["shape_posterior"][0].tolist())
    }

    return PredictionResult(
        anchor_time=example.anchor_time.isoformat(),
        current_close=float(example.current_close),
        policy_horizon=int(decision["policy_horizon"]),
        executed_horizon=decision["executed_horizon"],
        previous_position=float(previous_position),
        position=float(decision["position"]),
        trade_delta=float(decision["trade_delta"]),
        no_trade_band_hit=bool(decision["no_trade_band_hit"]),
        tradeability_gate=tradeability_gate,
        shape_entropy=float(outputs["shape_entropy"][0].item()),
        policy_score=float(decision["selection_score"]),
        expected_log_returns=expected_log_returns,
        predicted_closes=predicted_closes,
        uncertainties=uncertainties,
        horizon_utilities=horizon_utilities,
        horizon_positions=horizon_positions,
        shape_probabilities=shape_probabilities,
        regime_id=example.regime_id,
    )


def serialize_prediction_result(prediction: PredictionResult) -> dict[str, object]:
    sigma_t_sq = {
        str(horizon): float(value) ** 2
        for horizon, value in prediction.uncertainties.items()
    }
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "predicted_close_semantics": "median_from_log_return",
        "anchor_time": prediction.anchor_time,
        "current_close": prediction.current_close,
        "policy_horizon": prediction.policy_horizon,
        "executed_horizon": prediction.executed_horizon,
        "q_t_prev": prediction.previous_position,
        "previous_position": prediction.previous_position,
        "q_t": prediction.position,
        "position": prediction.position,
        "q_t_trade_delta": prediction.trade_delta,
        "trade_delta": prediction.trade_delta,
        "no_trade_band_hit": prediction.no_trade_band_hit,
        "g_t": prediction.tradeability_gate,
        "tradeability_gate": prediction.tradeability_gate,
        "shape_entropy": prediction.shape_entropy,
        "selected_policy_utility": prediction.policy_score,
        "policy_score": prediction.policy_score,
        "mu_t": dict(prediction.expected_log_returns),
        "expected_log_returns": dict(prediction.expected_log_returns),
        "median_predicted_close_by_horizon": dict(prediction.predicted_closes),
        "predicted_closes": dict(prediction.predicted_closes),
        "median_predicted_closes": dict(prediction.predicted_closes),
        "sigma_t": dict(prediction.uncertainties),
        "sigma_t_sq": sigma_t_sq,
        "uncertainties": dict(prediction.uncertainties),
        "horizon_utilities": dict(prediction.horizon_utilities),
        "horizon_positions": dict(prediction.horizon_positions),
        "shape_posterior": dict(prediction.shape_probabilities),
        "shape_probabilities": dict(prediction.shape_probabilities),
        "regime_id": prediction.regime_id,
    }


def build_forecast_summary_payload(
    prediction: PredictionResult,
    config: TrainingConfig,
    validation_metrics: dict[str, object] | None = None,
    best_params: dict[str, object] | None = None,
) -> dict[str, object]:
    anchor_close = float(prediction.current_close)
    sigma_t_sq = {
        str(horizon): float(value) ** 2
        for horizon, value in prediction.uncertainties.items()
    }
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "predicted_close_semantics": "median_from_log_return",
        "anchor_time": prediction.anchor_time,
        "anchor_close": anchor_close,
        "policy_horizon": prediction.policy_horizon,
        "executed_horizon": prediction.executed_horizon,
        "q_t_prev": prediction.previous_position,
        "previous_position": prediction.previous_position,
        "q_t": prediction.position,
        "position": prediction.position,
        "q_t_trade_delta": prediction.trade_delta,
        "trade_delta": prediction.trade_delta,
        "no_trade_band_hit": prediction.no_trade_band_hit,
        "g_t": prediction.tradeability_gate,
        "tradeability_gate": prediction.tradeability_gate,
        "shape_entropy": prediction.shape_entropy,
        "selected_policy_utility": prediction.policy_score,
        "policy_score": prediction.policy_score,
        "mu_t": dict(prediction.expected_log_returns),
        "expected_log_returns": dict(prediction.expected_log_returns),
        "median_predicted_close_by_horizon": dict(prediction.predicted_closes),
        "predicted_closes": dict(prediction.predicted_closes),
        "median_predicted_closes": dict(prediction.predicted_closes),
        "sigma_t": dict(prediction.uncertainties),
        "sigma_t_sq": sigma_t_sq,
        "uncertainties": dict(prediction.uncertainties),
        "horizon_utilities": dict(prediction.horizon_utilities),
        "horizon_positions": dict(prediction.horizon_positions),
        "shape_posterior": dict(prediction.shape_probabilities),
        "shape_probabilities": dict(prediction.shape_probabilities),
        "forecast_rows": [
            {
                "horizon_4h": horizon,
                "forecast_time_utc": (
                    datetime.fromisoformat(prediction.anchor_time) + timedelta(hours=4 * horizon)
                ).isoformat(),
                "mu_t": prediction.expected_log_returns[str(horizon)],
                "expected_log_return": prediction.expected_log_returns[str(horizon)],
                "g_t": prediction.tradeability_gate,
                "expected_return_pct": prediction.predicted_closes[str(horizon)] / max(anchor_close, 1e-6) - 1.0,
                "median_predicted_close": prediction.predicted_closes[str(horizon)],
                "predicted_close": prediction.predicted_closes[str(horizon)],
                "sigma_t": prediction.uncertainties[str(horizon)],
                "sigma_t_sq": sigma_t_sq[str(horizon)],
                "uncertainty": prediction.uncertainties[str(horizon)],
                "one_sigma_low_close": anchor_close
                * math.exp(
                    prediction.expected_log_returns[str(horizon)]
                    - prediction.uncertainties[str(horizon)]
                ),
                "one_sigma_high_close": anchor_close
                * math.exp(
                    prediction.expected_log_returns[str(horizon)]
                    + prediction.uncertainties[str(horizon)]
                ),
            }
            for horizon in config.horizons
        ],
        "best_params": None if best_params is None else dict(best_params),
        "validation_metrics": {} if validation_metrics is None else dict(validation_metrics),
    }
