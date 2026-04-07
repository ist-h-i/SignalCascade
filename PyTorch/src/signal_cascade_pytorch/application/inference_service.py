from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import torch

from .config import TrainingConfig
from .policy_service import apply_selection_policy
from .price_scale import DEFAULT_PRICE_SCALE, coerce_positive_price_scale
from .training_service import examples_to_batch, replay_recurrent_context
from ..domain.entities import PredictionResult, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel

PREDICTION_SCHEMA_VERSION = 7
FORECAST_SCHEMA_VERSION = 7


def _resolve_price_scale(value: float | None) -> float:
    resolved = coerce_positive_price_scale(value)
    return resolved if resolved is not None else DEFAULT_PRICE_SCALE


def _to_display_price(raw_price: float, price_scale: float) -> float:
    return float(raw_price) / _resolve_price_scale(price_scale)


def predict_latest(
    model: SignalCascadeModel,
    examples: list[TrainingExample],
    config: TrainingConfig,
    previous_position: float = 0.0,
    latest_example: TrainingExample | None = None,
) -> PredictionResult:
    if not examples and latest_example is None:
        raise ValueError("At least one training example is required for prediction.")

    target_example = latest_example or examples[-1]
    context_examples = [example for example in examples if example.anchor_time < target_example.anchor_time]
    previous_state, carried_previous_position = replay_recurrent_context(
        model=model,
        examples=context_examples,
        config=config,
        initial_previous_position=previous_position,
    )
    return predict_from_example(
        model=model,
        example=target_example,
        config=config,
        previous_position=carried_previous_position,
        previous_state=previous_state,
        inference_context_mode="carry_on",
    )


def predict_from_example(
    model: SignalCascadeModel,
    example: TrainingExample,
    config: TrainingConfig,
    previous_position: float = 0.0,
    previous_state: torch.Tensor | None = None,
    inference_context_mode: str | None = None,
) -> PredictionResult:
    batch = examples_to_batch([example], config)
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["main"],
            batch["overlay"],
            batch["state_features"],
            previous_state=previous_state,
        )

    forecast_mean = outputs.get("forecast_mu", outputs["mu"])[0]
    forecast_sigma = outputs.get("forecast_sigma", outputs["sigma"])[0]
    policy_mean = outputs.get("policy_mu", outputs["mu"])[0]
    policy_sigma = outputs.get("policy_sigma", outputs["sigma"])[0]
    tradeability_gate = float(outputs["tradeability_gate"][0].item())
    decision = apply_selection_policy(
        example=example,
        mean=policy_mean.tolist(),
        sigma=policy_sigma.tolist(),
        config=config,
        previous_position=previous_position,
        tradeability_gate=tradeability_gate,
        shape_probs=outputs["shape_posterior"][0].tolist(),
    )

    predicted_closes = {
        str(horizon): example.current_close * math.exp(float(forecast_mean[index].item()))
        for index, horizon in enumerate(config.horizons)
    }
    expected_log_returns = {
        str(horizon): float(forecast_mean[index].item())
        for index, horizon in enumerate(config.horizons)
    }
    uncertainties = {
        str(horizon): float(forecast_sigma[index].item())
        for index, horizon in enumerate(config.horizons)
    }
    policy_log_returns = {
        str(horizon): float(policy_mean[index].item())
        for index, horizon in enumerate(config.horizons)
    }
    policy_uncertainties = {
        str(horizon): float(policy_sigma[index].item())
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
        price_scale=float(example.price_scale),
        policy_horizon=int(decision["policy_horizon"]),
        executed_horizon=decision["executed_horizon"],
        previous_position=float(previous_position),
        position=float(decision["position"]),
        trade_delta=float(decision["trade_delta"]),
        no_trade_band_hit=bool(decision["no_trade_band_hit"]),
        tradeability_gate=tradeability_gate,
        shape_entropy=float(outputs["shape_entropy"][0].item()),
        policy_score=float(decision["selected_policy_utility"]),
        expected_log_returns=expected_log_returns,
        predicted_closes=predicted_closes,
        uncertainties=uncertainties,
        horizon_utilities=horizon_utilities,
        horizon_positions=horizon_positions,
        shape_probabilities=shape_probabilities,
        regime_id=example.regime_id,
        policy_log_returns=policy_log_returns,
        policy_uncertainties=policy_uncertainties,
        policy_head_tied_to_forecast=bool(outputs["policy_head_tied_to_forecast"].item()),
        overlay_branch_disabled=bool(outputs["overlay_branch_disabled"].item()),
        inference_context_mode=inference_context_mode
        or ("carry_on" if previous_state is not None else "stateless"),
    )


def serialize_prediction_result(prediction: PredictionResult) -> dict[str, object]:
    price_scale = _resolve_price_scale(prediction.price_scale)
    current_close_raw = float(prediction.current_close)
    current_close_display = _to_display_price(current_close_raw, price_scale)
    predicted_closes_raw = dict(prediction.predicted_closes)
    predicted_closes_display = {
        str(horizon): _to_display_price(float(value), price_scale)
        for horizon, value in predicted_closes_raw.items()
    }
    sigma_t_sq = {
        str(horizon): float(value) ** 2
        for horizon, value in prediction.uncertainties.items()
    }
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "predicted_close_semantics": "median_from_log_return",
        "anchor_time": prediction.anchor_time,
        "current_close": prediction.current_close,
        "current_close_raw": current_close_raw,
        "current_close_display": current_close_display,
        "effective_price_scale": price_scale,
        "price_scale": price_scale,
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
        "median_predicted_close_by_horizon": predicted_closes_raw,
        "predicted_closes": predicted_closes_raw,
        "median_predicted_closes": predicted_closes_raw,
        "median_predicted_close_raw_by_horizon": predicted_closes_raw,
        "median_predicted_close_display_by_horizon": predicted_closes_display,
        "median_predicted_closes_raw": predicted_closes_raw,
        "median_predicted_closes_display": predicted_closes_display,
        "sigma_t": dict(prediction.uncertainties),
        "sigma_t_sq": sigma_t_sq,
        "uncertainties": dict(prediction.uncertainties),
        "policy_mu_t": (
            None if prediction.policy_log_returns is None else dict(prediction.policy_log_returns)
        ),
        "policy_sigma_t": (
            None
            if prediction.policy_uncertainties is None
            else dict(prediction.policy_uncertainties)
        ),
        "policy_head_tied_to_forecast": bool(prediction.policy_head_tied_to_forecast),
        "overlay_branch_disabled": bool(prediction.overlay_branch_disabled),
        "overlay_branch_contract": (
            "disabled_in_canonical_path"
            if prediction.overlay_branch_disabled
            else "auxiliary_latent_branch_without_direct_supervision"
        ),
        "display_forecast": {
            "label": "display forecast",
            "mean_key": "mu_t",
            "sigma_key": "sigma_t",
            "mean_by_horizon": dict(prediction.expected_log_returns),
            "sigma_by_horizon": dict(prediction.uncertainties),
        },
        "policy_driver": {
            "label": "policy driver",
            "mean_key": "policy_mu_t",
            "sigma_key": "policy_sigma_t",
            "head_relationship": (
                "tied_to_forecast_head"
                if prediction.policy_head_tied_to_forecast
                else "separate_policy_head"
            ),
            "mean_by_horizon": (
                None if prediction.policy_log_returns is None else dict(prediction.policy_log_returns)
            ),
            "sigma_by_horizon": (
                None
                if prediction.policy_uncertainties is None
                else dict(prediction.policy_uncertainties)
            ),
        },
        "horizon_utilities": dict(prediction.horizon_utilities),
        "horizon_positions": dict(prediction.horizon_positions),
        "shape_posterior": dict(prediction.shape_probabilities),
        "shape_probabilities": dict(prediction.shape_probabilities),
        "inference_context_mode": prediction.inference_context_mode,
        "regime_id": prediction.regime_id,
    }


def build_forecast_summary_payload(
    prediction: PredictionResult,
    config: TrainingConfig,
    validation_metrics: dict[str, object] | None = None,
    best_params: dict[str, object] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    price_scale = _resolve_price_scale(prediction.price_scale)
    anchor_close = float(prediction.current_close)
    anchor_close_display = _to_display_price(anchor_close, price_scale)
    resolved_generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()
    predicted_closes_raw = dict(prediction.predicted_closes)
    predicted_closes_display = {
        str(horizon): _to_display_price(float(value), price_scale)
        for horizon, value in predicted_closes_raw.items()
    }
    sigma_t_sq = {
        str(horizon): float(value) ** 2
        for horizon, value in prediction.uncertainties.items()
    }
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "generated_at": resolved_generated_at_utc,
        "generated_at_utc": resolved_generated_at_utc,
        "predicted_close_semantics": "median_from_log_return",
        "anchor_time": prediction.anchor_time,
        "anchor_close": anchor_close,
        "anchor_close_raw": anchor_close,
        "anchor_close_display": anchor_close_display,
        "effective_price_scale": price_scale,
        "price_scale": price_scale,
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
        "median_predicted_close_by_horizon": predicted_closes_raw,
        "predicted_closes": predicted_closes_raw,
        "median_predicted_closes": predicted_closes_raw,
        "median_predicted_close_raw_by_horizon": predicted_closes_raw,
        "median_predicted_close_display_by_horizon": predicted_closes_display,
        "median_predicted_closes_raw": predicted_closes_raw,
        "median_predicted_closes_display": predicted_closes_display,
        "sigma_t": dict(prediction.uncertainties),
        "sigma_t_sq": sigma_t_sq,
        "uncertainties": dict(prediction.uncertainties),
        "policy_mu_t": (
            None if prediction.policy_log_returns is None else dict(prediction.policy_log_returns)
        ),
        "policy_sigma_t": (
            None
            if prediction.policy_uncertainties is None
            else dict(prediction.policy_uncertainties)
        ),
        "policy_head_tied_to_forecast": bool(prediction.policy_head_tied_to_forecast),
        "overlay_branch_disabled": bool(prediction.overlay_branch_disabled),
        "overlay_branch_contract": (
            "disabled_in_canonical_path"
            if prediction.overlay_branch_disabled
            else "auxiliary_latent_branch_without_direct_supervision"
        ),
        "display_forecast": {
            "label": "display forecast",
            "mean_key": "mu_t",
            "sigma_key": "sigma_t",
            "mean_by_horizon": dict(prediction.expected_log_returns),
            "sigma_by_horizon": dict(prediction.uncertainties),
        },
        "policy_driver": {
            "label": "policy driver",
            "mean_key": "policy_mu_t",
            "sigma_key": "policy_sigma_t",
            "head_relationship": (
                "tied_to_forecast_head"
                if prediction.policy_head_tied_to_forecast
                else "separate_policy_head"
            ),
            "mean_by_horizon": (
                None if prediction.policy_log_returns is None else dict(prediction.policy_log_returns)
            ),
            "sigma_by_horizon": (
                None
                if prediction.policy_uncertainties is None
                else dict(prediction.policy_uncertainties)
            ),
        },
        "horizon_utilities": dict(prediction.horizon_utilities),
        "horizon_positions": dict(prediction.horizon_positions),
        "shape_posterior": dict(prediction.shape_probabilities),
        "shape_probabilities": dict(prediction.shape_probabilities),
        "inference_context_mode": prediction.inference_context_mode,
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
                "median_predicted_close_raw": prediction.predicted_closes[str(horizon)],
                "median_predicted_close_display": predicted_closes_display[str(horizon)],
                "sigma_t": prediction.uncertainties[str(horizon)],
                "sigma_t_sq": sigma_t_sq[str(horizon)],
                "uncertainty": prediction.uncertainties[str(horizon)],
                "policy_mu_t": None
                if prediction.policy_log_returns is None
                else prediction.policy_log_returns[str(horizon)],
                "policy_sigma_t": None
                if prediction.policy_uncertainties is None
                else prediction.policy_uncertainties[str(horizon)],
                "one_sigma_low_close": anchor_close
                * math.exp(
                    prediction.expected_log_returns[str(horizon)]
                    - prediction.uncertainties[str(horizon)]
                ),
                "one_sigma_low_close_display": _to_display_price(
                    anchor_close
                    * math.exp(
                        prediction.expected_log_returns[str(horizon)]
                        - prediction.uncertainties[str(horizon)]
                    ),
                    price_scale,
                ),
                "one_sigma_high_close": anchor_close
                * math.exp(
                    prediction.expected_log_returns[str(horizon)]
                    + prediction.uncertainties[str(horizon)]
                ),
                "one_sigma_high_close_display": _to_display_price(
                    anchor_close
                    * math.exp(
                        prediction.expected_log_returns[str(horizon)]
                        + prediction.uncertainties[str(horizon)]
                    ),
                    price_scale,
                ),
            }
            for horizon in config.horizons
        ],
        "best_params": None if best_params is None else dict(best_params),
        "validation_metrics": {} if validation_metrics is None else dict(validation_metrics),
    }
