from __future__ import annotations

import torch
from torch.nn import functional as functional

from ...application.config import TrainingConfig
from ...application.policy_service import smooth_policy_distribution
from ...domain.timeframes import MAIN_TIMEFRAMES


def total_loss(
    outputs: dict[str, object],
    batch: dict[str, object],
    config: TrainingConfig,
    previous_position: torch.Tensor,
    warmup_phase: bool = False,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    forecast_mean = outputs.get("forecast_mu", outputs["mu"])
    forecast_sigma = outputs.get("forecast_sigma", outputs["sigma"])
    return_loss = heteroscedastic_huber_loss(
        forecast_mean,
        forecast_sigma,
        batch["returns"],
    )
    shape_loss = main_shape_loss(outputs["main_shape_predictions"], batch["shape_targets"])
    profit_loss, policy_metrics = profit_objective_loss(
        outputs=outputs,
        batch=batch,
        config=config,
        previous_position=previous_position,
    )
    profit_weight = config.profit_loss_weight * (0.35 if warmup_phase else 1.0)
    total = (
        (config.return_loss_weight * return_loss)
        + (config.shape_loss_weight * shape_loss)
        + (profit_weight * profit_loss)
    )
    metrics = {
        "total": float(total.item()),
        "profit_loss": float(profit_loss.item()),
        "return_loss": float(return_loss.item()),
        "shape_loss": float(shape_loss.item()),
        "average_log_wealth": float(policy_metrics["average_log_wealth"].item()),
        "mean_pnl": float(policy_metrics["mean_pnl"].item()),
        "cvar_tail_loss": float(policy_metrics["cvar_tail_loss"].item()),
        "mean_position": float(policy_metrics["mean_position"].item()),
        "log_wealth_clamp_hit_rate": float(policy_metrics["log_wealth_clamp_hit_rate"].item()),
        "shape_entropy": float(outputs["shape_entropy"].mean().item()),
        "forecast_mae": float(torch.mean(torch.abs(batch["returns"] - forecast_mean)).item()),
    }
    return total, metrics, policy_metrics


def profit_objective_loss(
    outputs: dict[str, object],
    batch: dict[str, object],
    config: TrainingConfig,
    previous_position: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_mean = outputs.get("policy_mu", outputs["mu"])
    policy_sigma = outputs.get("policy_sigma", outputs["sigma"])
    policy = smooth_policy_distribution(
        mean=policy_mean,
        sigma=policy_sigma,
        costs=batch["horizon_costs"],
        tradeability_gate=outputs["tradeability_gate"],
        previous_position=previous_position,
        config=config,
    )
    pnl_by_horizon = (
        policy["horizon_positions"] * batch["returns"]
    ) - (batch["horizon_costs"] * policy["turnover"])
    combined_pnl = torch.sum(policy["horizon_weights"] * pnl_by_horizon, dim=-1)
    clamp_hits = (combined_pnl <= -0.95).float()
    clamped_pnl = torch.clamp(combined_pnl, min=-0.95)
    log_wealth = torch.log1p(clamped_pnl)
    cvar_tail = cvar_tail_loss(-combined_pnl, config.cvar_alpha)
    profit_loss = -log_wealth.mean() + (config.cvar_weight * cvar_tail)
    return profit_loss, {
        "combined_pnl": combined_pnl,
        "average_log_wealth": log_wealth.mean(),
        "mean_pnl": combined_pnl.mean(),
        "cvar_tail_loss": cvar_tail,
        "combined_position": policy["combined_position"],
        "mean_position": policy["combined_position"].mean(),
        "combined_utility": policy["combined_utility"],
        "selected_position": policy["selected_position"],
        "selected_horizon_index": policy["selected_horizon_index"],
        "log_wealth_clamp_hit_rate": clamp_hits.mean(),
        "horizon_positions": policy["horizon_positions"],
        "horizon_weights": policy["horizon_weights"],
    }


def cvar_tail_loss(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    flattened = losses.reshape(-1)
    if flattened.numel() == 0:
        return torch.tensor(0.0, dtype=losses.dtype, device=losses.device)
    tail_fraction = max(min(alpha, 1.0), 1e-3)
    tail_count = max(int(torch.ceil(torch.tensor(flattened.numel() * tail_fraction)).item()), 1)
    tail_values, _ = torch.topk(flattened, k=tail_count)
    return tail_values.mean()


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


def main_shape_loss(
    shape_predictions: dict[str, torch.Tensor],
    shape_targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = [
        functional.smooth_l1_loss(shape_predictions[timeframe], shape_targets[timeframe])
        for timeframe in MAIN_TIMEFRAMES
    ]
    return torch.stack(losses).mean()
