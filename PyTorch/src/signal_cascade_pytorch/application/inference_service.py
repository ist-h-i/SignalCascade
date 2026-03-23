from __future__ import annotations

import math

import torch

from .config import TrainingConfig
from .dataset_service import OVERLAY_LABELS
from .training_service import examples_to_batch
from ..domain.entities import PredictionResult, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel


def predict_latest(
    model: SignalCascadeModel,
    examples: list[TrainingExample],
    config: TrainingConfig,
) -> PredictionResult:
    latest_example = examples[-1]
    batch = examples_to_batch([latest_example])
    model.eval()
    with torch.no_grad():
        outputs = model(batch["main"], batch["overlay"])

    mean = outputs["mu"][0]
    sigma = outputs["sigma"][0]
    score = mean.abs() / sigma
    best_index = int(torch.argmax(score).item())
    selected_horizon = config.horizons[best_index]
    position = math.tanh(float((mean[best_index] / sigma[best_index]).item()))
    predicted_closes = {
        str(horizon): latest_example.current_close * math.exp(float(mean[index].item()))
        for index, horizon in enumerate(config.horizons)
    }
    uncertainties = {
        str(horizon): float(sigma[index].item())
        for index, horizon in enumerate(config.horizons)
    }
    overlay_index = int(outputs["overlay_logits"][0].argmax().item())

    return PredictionResult(
        anchor_time=latest_example.anchor_time.isoformat(),
        selected_horizon=selected_horizon,
        position=position,
        predicted_closes=predicted_closes,
        uncertainties=uncertainties,
        overlay_action=OVERLAY_LABELS[overlay_index],
    )
