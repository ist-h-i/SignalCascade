from __future__ import annotations

import torch
from torch.nn import functional as functional

from ...domain.timeframes import MAIN_TIMEFRAMES


def total_loss(
    outputs: dict[str, object],
    batch: dict[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    return_loss = heteroscedastic_huber_loss(outputs["mu"], outputs["sigma"], batch["returns"])
    direction_loss = directional_focal_loss(
        outputs["direction_logits"],
        batch["direction_target"],
        batch["direction_weight"],
    )
    shape_loss = main_shape_loss(outputs["shape_predictions"], batch["shape_targets"])
    overlay_loss = overlay_binary_loss(outputs["overlay_logits"], batch["overlay_target"])
    total = return_loss + (0.35 * direction_loss) + (0.25 * shape_loss) + (0.15 * overlay_loss)
    return total, {
        "total": float(total.item()),
        "return_loss": float(return_loss.item()),
        "direction_loss": float(direction_loss.item()),
        "shape_loss": float(shape_loss.item()),
        "overlay_loss": float(overlay_loss.item()),
    }


def heteroscedastic_huber_loss(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.02,
) -> torch.Tensor:
    error = target - mean
    absolute_error = error.abs()
    quadratic = torch.minimum(absolute_error, torch.full_like(absolute_error, delta))
    linear = absolute_error - quadratic
    huber = 0.5 * quadratic.pow(2) + delta * linear
    variance = sigma.pow(2).clamp_min(1e-6)
    return ((huber / variance) + torch.log(sigma)).mean()


def directional_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    probabilities = functional.softmax(logits, dim=-1)
    selected_probabilities = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
    alpha = torch.tensor([1.2, 0.7, 1.2], dtype=logits.dtype, device=logits.device)
    alpha_weight = alpha.gather(0, targets.reshape(-1)).reshape_as(targets)
    focal_factor = torch.pow(1.0 - selected_probabilities, gamma)
    per_target_loss = -torch.log(selected_probabilities) * alpha_weight * focal_factor * weights
    return per_target_loss.mean()


def main_shape_loss(
    shape_predictions: dict[str, torch.Tensor],
    shape_targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = [
        functional.smooth_l1_loss(shape_predictions[timeframe], shape_targets[timeframe])
        for timeframe in MAIN_TIMEFRAMES
    ]
    return torch.stack(losses).mean()


def overlay_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    squeezed_logits = logits.squeeze(-1)
    return functional.binary_cross_entropy_with_logits(squeezed_logits, targets)
