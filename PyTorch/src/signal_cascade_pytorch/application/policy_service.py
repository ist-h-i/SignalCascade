from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as functional

from .config import TrainingConfig
from ..domain.entities import TrainingExample


def build_default_policy(config: TrainingConfig) -> dict[str, object]:
    return {
        "mode": "profit_maximization",
        "status": "retired_threshold_policy",
        "selection_score_source": "profit_utility",
        "allow_no_candidate": bool(config.allow_no_candidate),
    }


def build_selection_policy(
    snapshots: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> dict[str, object]:
    return build_default_policy(config)


def build_replay_selection_policy(
    policy: dict[str, object] | None,
    snapshots: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> dict[str, object]:
    replay = dict(policy or build_default_policy(config))
    replay["status"] = "profit_policy_replay"
    return replay


def selection_thresholds_match_config(
    policy: dict[str, object],
    config: TrainingConfig,
) -> bool:
    return True


def apply_selection_policy(
    example: TrainingExample,
    mean: Sequence[float],
    sigma: Sequence[float],
    policy: dict[str, object] | None = None,
    config: TrainingConfig | None = None,
    previous_position: float = 0.0,
    tradeability_gate: float = 1.0,
    shape_probs: Sequence[float] | None = None,
    cost_multiplier: float = 1.0,
    gamma_multiplier: float = 1.0,
    **_: object,
) -> dict[str, object]:
    if config is None:
        raise ValueError("TrainingConfig is required.")
    rows = build_exact_policy_rows(
        mean=mean,
        sigma=sigma,
        costs=example.horizon_costs,
        tradeability_gate=tradeability_gate,
        previous_position=previous_position,
        config=config,
        cost_multiplier=cost_multiplier,
        gamma_multiplier=gamma_multiplier,
    )
    selected_row = max(
        rows,
        key=lambda row: (
            float(row["policy_utility"]),
            abs(float(row["mu_t_tilde"])),
            -int(row["horizon"]),
        ),
    )
    position = float(selected_row["position"])
    trade_delta = position - float(previous_position)
    no_trade = bool(selected_row["no_trade_band"])
    executed_horizon = int(selected_row["horizon"]) if abs(position) > 1e-9 else None
    shape_vector = list(shape_probs or [])
    shape_entropy = _shape_entropy(shape_vector)
    selected_direction = _sign_from_value(float(selected_row["mean"]))

    return {
        "policy_horizon": int(selected_row["horizon"]),
        "proposed_horizon": int(selected_row["horizon"]),
        "executed_horizon": executed_horizon,
        "accepted_horizon": executed_horizon,
        "selected_horizon": int(selected_row["horizon"]),
        "selected_direction": selected_direction,
        "position": position,
        "previous_position": float(previous_position),
        "trade_delta": trade_delta,
        "raw_position": position,
        "pre_threshold_position": position,
        "pre_threshold_eligible": not no_trade,
        "accepted_signal": executed_horizon is not None,
        "no_trade_band_hit": no_trade,
        "selection_probability": float(tradeability_gate),
        "selection_score": float(selected_row["policy_utility"]),
        "selection_score_source": "profit_utility",
        "selection_threshold": None,
        "threshold_status": "retired",
        "threshold_origin": "profit_policy",
        "stored_threshold_compatibility": "retired",
        "threshold_score_source": "profit_utility",
        "precision_infeasible": False,
        "correctness_probability": float(tradeability_gate),
        "hold_probability": max(0.0, 1.0 - min(abs(position), 1.0)),
        "hold_threshold": 0.0,
        "overlay_action": "hold" if no_trade else "reduce",
        "expected_direction": selected_direction,
        "direction_alignment": True,
        "accept_reject_reason": "no_trade_band" if no_trade else "executed",
        "reject_flags": {
            "no_trade_band": no_trade,
            "retired_threshold_policy": True,
        },
        "meta_label": int(
            executed_horizon is not None
            and selected_direction != 0
            and selected_direction == int(example.direction_targets[config.horizons.index(int(selected_row["horizon"]))])
        ),
        "direction_correct": int(
            selected_direction != 0
            and selected_direction == _sign_from_value(
                float(example.returns_target[config.horizons.index(int(selected_row["horizon"]))])
            )
        ),
        "candidate_count": len(rows),
        "strict_candidate_count": len(rows),
        "any_candidate": bool(rows),
        "any_strict_candidate": bool(rows),
        "tradeability_gate": float(tradeability_gate),
        "shape_entropy": shape_entropy,
        "horizon_rows": rows,
        "selected_row": selected_row,
    }


def build_prediction_snapshots(
    examples: Sequence[TrainingExample],
    mean: torch.Tensor,
    sigma: torch.Tensor,
    tradeability_gate: torch.Tensor,
    config: TrainingConfig,
    previous_position: float = 0.0,
) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    running_position = float(previous_position)
    for index, example in enumerate(examples):
        decision = apply_selection_policy(
            example=example,
            mean=mean[index].tolist(),
            sigma=sigma[index].tolist(),
            config=config,
            previous_position=running_position,
            tradeability_gate=float(tradeability_gate[index].item()),
        )
        running_position = float(decision["position"])
        snapshots.append(
            {
                "anchor_time": example.anchor_time.isoformat(),
                "regime_id": example.regime_id,
                "state_features": [float(value) for value in example.state_features],
                "decision": decision,
                "horizons": [dict(row) for row in decision["horizon_rows"]],
            }
        )
    return snapshots


def build_exact_policy_rows(
    mean: Sequence[float],
    sigma: Sequence[float],
    costs: Sequence[float],
    tradeability_gate: float,
    previous_position: float,
    config: TrainingConfig,
    cost_multiplier: float = 1.0,
    gamma_multiplier: float = 1.0,
) -> list[dict[str, object]]:
    path_terms = build_policy_path_terms(
        mean=torch.tensor([list(mean)], dtype=torch.float32),
        sigma=torch.tensor([list(sigma)], dtype=torch.float32),
        costs=torch.tensor([list(costs)], dtype=torch.float32),
        tradeability_gate=torch.tensor([float(tradeability_gate)], dtype=torch.float32),
        previous_position=torch.tensor([float(previous_position)], dtype=torch.float32),
        config=config,
        cost_multiplier=cost_multiplier,
        gamma_multiplier=gamma_multiplier,
    )
    rows: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(config.horizons):
        mean_value = float(path_terms["mean"][0, horizon_index].item())
        sigma_sq_value = float(path_terms["sigma_sq"][0, horizon_index].item())
        sigma_value = math.sqrt(sigma_sq_value)
        cost_value = float(path_terms["costs"][0, horizon_index].item())
        g_t = float(path_terms["g_t"][0, horizon_index].item())
        mu_t_tilde = float(path_terms["mu_t_tilde"][0, horizon_index].item())
        effective_gamma = float(path_terms["effective_gamma"][0, horizon_index].item())
        margin = float(path_terms["margin"][0, horizon_index].item())
        position, no_trade = solve_exact_policy_position_from_terms(
            mu_t_tilde=mu_t_tilde,
            sigma_sq=sigma_sq_value,
            previous_position=previous_position,
            cost=cost_value,
            effective_gamma=effective_gamma,
            q_max=config.q_max,
        )
        utility = policy_utility(
            position=position,
            previous_position=previous_position,
            gated_mean=mu_t_tilde,
            sigma=sigma_value,
            cost=cost_value,
            config=config,
            gamma_multiplier=gamma_multiplier,
        )
        rows.append(
            {
                "horizon": int(horizon),
                "mean": mean_value,
                "sigma": sigma_value,
                "sigma_sq": sigma_sq_value,
                "cost": cost_value,
                "g_t": g_t,
                "mu_t_tilde": mu_t_tilde,
                "gated_mean": mu_t_tilde,
                "effective_gamma": effective_gamma,
                "margin": margin,
                "position": position,
                "predicted_sign": _sign_from_value(mean_value),
                "policy_utility": utility,
                "no_trade_band": no_trade,
            }
        )
    return rows


def solve_exact_policy_position(
    gated_mean: float,
    sigma: float,
    previous_position: float,
    cost: float,
    config: TrainingConfig,
    gamma_multiplier: float = 1.0,
) -> tuple[float, bool]:
    sigma_sq = max(float(sigma) ** 2, config.min_policy_sigma**2)
    effective_gamma = float(config.risk_aversion_gamma) * float(gamma_multiplier)
    return solve_exact_policy_position_from_terms(
        mu_t_tilde=float(gated_mean),
        sigma_sq=sigma_sq,
        previous_position=previous_position,
        cost=cost,
        effective_gamma=effective_gamma,
        q_max=config.q_max,
    )


def solve_exact_policy_position_from_terms(
    mu_t_tilde: float,
    sigma_sq: float,
    previous_position: float,
    cost: float,
    effective_gamma: float,
    q_max: float,
) -> tuple[float, bool]:
    margin = float(mu_t_tilde) - (float(effective_gamma) * float(sigma_sq) * float(previous_position))
    if abs(margin) <= float(cost):
        position = float(previous_position)
        no_trade = True
    elif margin > 0.0:
        position = (float(mu_t_tilde) - float(cost)) / (float(effective_gamma) * float(sigma_sq))
        no_trade = False
    else:
        position = (float(mu_t_tilde) + float(cost)) / (float(effective_gamma) * float(sigma_sq))
        no_trade = False
    position = max(-float(q_max), min(float(q_max), position))
    return position, no_trade


def smooth_policy_distribution(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    costs: torch.Tensor,
    tradeability_gate: torch.Tensor,
    previous_position: torch.Tensor,
    config: TrainingConfig,
    cost_multiplier: float = 1.0,
    gamma_multiplier: float = 1.0,
) -> dict[str, torch.Tensor]:
    path_terms = build_policy_path_terms(
        mean=mean,
        sigma=sigma,
        costs=costs,
        tradeability_gate=tradeability_gate,
        previous_position=previous_position,
        config=config,
        cost_multiplier=cost_multiplier,
        gamma_multiplier=gamma_multiplier,
    )
    effective_gamma = path_terms["effective_gamma"]
    sigma_sq = path_terms["sigma_sq"]
    g_t = path_terms["g_t"]
    mu_t_tilde = path_terms["mu_t_tilde"]
    scaled_costs = path_terms["costs"]
    margin = path_terms["margin"]
    previous_position_matrix = path_terms["previous_position"]
    abs_margin = torch.abs(margin)
    smooth_excess = functional.softplus(
        abs_margin - scaled_costs,
        beta=config.policy_smoothing_beta,
    ) / max(config.policy_smoothing_beta, 1e-6)
    direction = torch.tanh(margin / max(config.policy_abs_epsilon, 1e-6))
    delta_position = direction * smooth_excess / (
        effective_gamma * sigma_sq
    )
    raw_position = previous_position_matrix + delta_position
    horizon_positions = config.q_max * torch.tanh(raw_position / max(config.q_max, 1e-6))
    turnover = torch.sqrt(
        (horizon_positions - previous_position_matrix).pow(2)
        + config.policy_abs_epsilon
    )
    utilities = (
        mu_t_tilde * horizon_positions
        - (0.5 * effective_gamma * sigma_sq * horizon_positions.pow(2))
        - (scaled_costs * turnover)
    )
    horizon_weights = functional.softmax(utilities * config.policy_smoothing_beta, dim=-1)
    combined_position = torch.sum(horizon_weights * horizon_positions, dim=-1)
    combined_utility = torch.sum(horizon_weights * utilities, dim=-1)
    selected_horizon_index = torch.argmax(utilities, dim=-1)
    selected_position = torch.gather(
        horizon_positions,
        dim=-1,
        index=selected_horizon_index.unsqueeze(-1),
    ).squeeze(-1)
    selected_utility = torch.gather(
        utilities,
        dim=-1,
        index=selected_horizon_index.unsqueeze(-1),
    ).squeeze(-1)
    selected_no_trade = torch.gather(
        (abs_margin <= scaled_costs),
        dim=-1,
        index=selected_horizon_index.unsqueeze(-1),
    ).squeeze(-1)
    return {
        "g_t": g_t,
        "mu_t_tilde": mu_t_tilde,
        "gated_mean": mu_t_tilde,
        "sigma_sq": sigma_sq,
        "margin": margin,
        "horizon_positions": horizon_positions,
        "turnover": turnover,
        "utilities": utilities,
        "horizon_weights": horizon_weights,
        "combined_position": combined_position,
        "combined_utility": combined_utility,
        "selected_horizon_index": selected_horizon_index,
        "selected_position": selected_position,
        "selected_utility": selected_utility,
        "selected_no_trade": selected_no_trade,
    }


def build_policy_path_terms(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    costs: torch.Tensor,
    tradeability_gate: torch.Tensor,
    previous_position: torch.Tensor,
    config: TrainingConfig,
    cost_multiplier: float = 1.0,
    gamma_multiplier: float = 1.0,
) -> dict[str, torch.Tensor]:
    effective_gamma_value = float(config.risk_aversion_gamma) * float(gamma_multiplier)
    sigma_sq = sigma.pow(2).clamp_min(config.min_policy_sigma**2)
    previous_position_matrix = previous_position.unsqueeze(-1).expand_as(mean)
    g_t = tradeability_gate.unsqueeze(-1).expand_as(mean)
    mu_t_tilde = g_t * mean
    scaled_costs = costs * float(cost_multiplier)
    margin = mu_t_tilde - (effective_gamma_value * sigma_sq * previous_position_matrix)
    effective_gamma = torch.full_like(mean, effective_gamma_value)
    return {
        "mean": mean,
        "sigma": sigma,
        "sigma_sq": sigma_sq,
        "costs": scaled_costs,
        "g_t": g_t,
        "mu_t_tilde": mu_t_tilde,
        "effective_gamma": effective_gamma,
        "previous_position": previous_position_matrix,
        "margin": margin,
    }


def policy_utility(
    position: float,
    previous_position: float,
    gated_mean: float,
    sigma: float,
    cost: float,
    config: TrainingConfig,
    gamma_multiplier: float = 1.0,
) -> float:
    sigma_sq = max(float(sigma) ** 2, config.min_policy_sigma**2)
    effective_gamma = float(config.risk_aversion_gamma) * float(gamma_multiplier)
    return (
        (float(gated_mean) * float(position))
        - (0.5 * effective_gamma * sigma_sq * (float(position) ** 2))
        - (float(cost) * abs(float(position) - float(previous_position)))
    )


def implied_direction_probabilities(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    band: torch.Tensor,
) -> torch.Tensor:
    clamped_sigma = sigma.clamp_min(1e-6)
    sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=mean.dtype, device=mean.device))
    z_up = (band - mean) / clamped_sigma
    z_down = (-band - mean) / clamped_sigma
    p_up = 1.0 - (0.5 * (1.0 + torch.erf(z_up / sqrt_two)))
    p_down = 0.5 * (1.0 + torch.erf(z_down / sqrt_two))
    p_flat = (1.0 - p_up - p_down).clamp_min(1e-6)
    probabilities = torch.stack([p_down, p_flat, p_up], dim=-1)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def _sign_from_value(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _shape_entropy(shape_probs: Sequence[float]) -> float:
    if not shape_probs:
        return 0.0
    clipped = [max(float(value), 1e-6) for value in shape_probs]
    entropy = -sum(value * math.log(value) for value in clipped)
    return entropy / math.log(max(len(clipped), 2))


def _precision_lower_bound(
    retrieved_true: int,
    retrieved_total: int,
    z_value: float,
) -> float:
    if retrieved_total <= 0:
        return 0.0
    phat = retrieved_true / retrieved_total
    z_sq = z_value**2
    denominator = 1.0 + (z_sq / retrieved_total)
    center = phat + (z_sq / (2.0 * retrieved_total))
    margin = z_value * math.sqrt(
        ((phat * (1.0 - phat)) / retrieved_total) + (z_sq / (4.0 * (retrieved_total**2)))
    )
    return max(0.0, (center - margin) / denominator)
