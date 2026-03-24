from __future__ import annotations

import math

import torch

from .config import TrainingConfig
from .policy_service import apply_selection_policy
from .training_service import examples_to_batch, restore_return_units
from ..domain.entities import PredictionResult, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel


def predict_latest(
    model: SignalCascadeModel,
    examples: list[TrainingExample],
    config: TrainingConfig,
    selection_policy: dict[str, object] | None = None,
) -> PredictionResult:
    return predict_from_example(model, examples[-1], config, selection_policy)


def predict_from_example(
    model: SignalCascadeModel,
    example: TrainingExample,
    config: TrainingConfig,
    selection_policy: dict[str, object] | None = None,
) -> PredictionResult:
    batch = examples_to_batch([example], config)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["main"], batch["overlay"])

    mean, sigma = restore_return_units(
        outputs["mu"],
        outputs["sigma"],
        batch["return_scale"],
    )
    mean = mean[0]
    sigma = sigma[0]
    direction_probabilities = torch.softmax(outputs["direction_logits"][0], dim=-1)
    overlay_probability = float(torch.sigmoid(outputs["overlay_logits"][0]).item())
    decision = apply_selection_policy(
        example=example,
        mean=mean.tolist(),
        sigma=sigma.tolist(),
        direction_probabilities=direction_probabilities.tolist(),
        overlay_probability=overlay_probability,
        policy=selection_policy,
        config=config,
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

    return PredictionResult(
        anchor_time=example.anchor_time.isoformat(),
        current_close=float(example.current_close),
        selected_horizon=decision["selected_horizon"],
        selected_direction=decision["selected_direction"],
        position=float(decision["position"]),
        expected_log_returns=expected_log_returns,
        predicted_closes=predicted_closes,
        uncertainties=uncertainties,
        accepted_signal=bool(decision["accepted_signal"]),
        selection_probability=float(decision["selection_probability"]),
        selection_score=float(decision["selection_score"]),
        selection_threshold=(
            None
            if decision["selection_threshold"] is None
            else float(decision["selection_threshold"])
        ),
        correctness_probability=float(decision["correctness_probability"]),
        hold_probability=float(decision["hold_probability"]),
        hold_threshold=float(decision["hold_threshold"]),
        overlay_action=str(decision["overlay_action"]),
        regime_id=example.regime_id,
    )
