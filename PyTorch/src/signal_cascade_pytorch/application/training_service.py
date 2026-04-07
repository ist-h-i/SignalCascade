from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import torch

from .artifact_provenance import STATE_RESET_BOUNDARY_SPEC_VERSION
from .config import TrainingConfig
from .policy_service import apply_selection_policy, policy_utility, smooth_policy_distribution
from ..domain.entities import TrainingExample
from ..domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES
from ..infrastructure.ml.losses import cvar_tail_loss, total_loss
from ..infrastructure.ml.model import SignalCascadeModel
from ..infrastructure.persistence import save_checkpoint


def examples_to_batch(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, object]:
    return {
        "main": {
            timeframe: torch.tensor(
                [example.main_sequences[timeframe] for example in examples],
                dtype=torch.float32,
            )
            for timeframe in MAIN_TIMEFRAMES
        },
        "overlay": {
            timeframe: torch.tensor(
                [example.overlay_sequences[timeframe] for example in examples],
                dtype=torch.float32,
            )
            for timeframe in OVERLAY_TIMEFRAMES
        },
        "shape_targets": {
            timeframe: torch.tensor(
                [example.main_shape_targets[timeframe] for example in examples],
                dtype=torch.float32,
            )
            for timeframe in MAIN_TIMEFRAMES
        },
        "state_features": torch.tensor(
            [example.state_features for example in examples],
            dtype=torch.float32,
        ),
        "returns": torch.tensor(
            [example.returns_target for example in examples],
            dtype=torch.float32,
        ),
        "horizon_costs": torch.tensor(
            [example.horizon_costs for example in examples],
            dtype=torch.float32,
        ),
        "current_close": torch.tensor(
            [example.current_close for example in examples],
            dtype=torch.float32,
        ),
    }


def restore_return_units(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    return_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mean, sigma


def train_model(
    examples: list[TrainingExample],
    config: TrainingConfig,
    output_dir: Path,
) -> tuple[SignalCascadeModel, dict[str, object]]:
    _set_seed(config.seed)
    train_examples, valid_examples = split_examples(examples, config)
    feature_dim = len(examples[0].main_sequences["4h"][0])
    state_feature_dim = len(examples[0].state_features)
    model, fit_summary = _fit_model(
        train_examples=train_examples,
        valid_examples=valid_examples,
        config=config,
        feature_dim=feature_dim,
        state_feature_dim=state_feature_dim,
    )
    checkpoint_audit = fit_summary.get("checkpoint_audit") or _build_checkpoint_audit(
        fit_summary.get("history", []),
        config,
    )
    save_checkpoint(output_dir / "model.pt", model, config)
    validation_metrics = evaluate_model(model, valid_examples, config)
    purged_samples = max(len(examples) - len(train_examples) - len(valid_examples), 0)
    summary = {
        "policy_mode": "shape_aware_profit_maximization",
        "loss_contract": {
            "primary_objective": "profit_objective_log_wealth_minus_cvar",
            "primary_metric": "profit_loss",
            "auxiliary_objectives": {
                "return": "heteroscedastic_huber",
                "shape": "main_shape_smooth_l1",
            },
            "weights": {
                "profit_loss_weight": float(config.profit_loss_weight),
                "warmup_profit_loss_weight": float(config.profit_loss_weight * 0.35),
                "return_loss_weight": float(config.return_loss_weight),
                "shape_loss_weight": float(config.shape_loss_weight),
                "cvar_weight": float(config.cvar_weight),
                "cvar_alpha": float(config.cvar_alpha),
            },
        },
        "train_samples": len(train_examples),
        "validation_samples": len(valid_examples),
        "purged_samples": purged_samples,
        "best_validation_loss": fit_summary["best_validation_loss"],
        "best_selection_score": fit_summary.get(
            "best_selection_score",
            fit_summary["best_validation_loss"],
        ),
        "best_epoch": fit_summary["best_epoch"],
        "best_epoch_by_exact_log_wealth": checkpoint_audit.get("best_epoch_by_exact_log_wealth"),
        "best_epoch_by_exact_log_wealth_minus_lambda_cvar": checkpoint_audit.get(
            "best_epoch_by_exact_log_wealth_minus_lambda_cvar"
        ),
        "best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar": checkpoint_audit.get(
            "best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar"
        ),
        "checkpoint_selection_metric": config.checkpoint_selection_metric,
        "walk_forward_folds": int(config.walk_forward_folds),
        "oof_epochs": int(config.oof_epochs),
        "checkpoint_audit": checkpoint_audit,
        "history": fit_summary["history"],
        "validation_metrics": validation_metrics,
    }
    return model, summary


def evaluate_model(
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    state_reset_mode: str | None = None,
    cost_multiplier: float | None = None,
    gamma_multiplier: float | None = None,
    min_policy_sigma: float | None = None,
) -> dict[str, object]:
    if not examples:
        raise ValueError("At least one example is required for evaluation.")

    model.eval()
    effective_config = (
        replace(config, min_policy_sigma=float(min_policy_sigma))
        if min_policy_sigma is not None
        else config
    )
    resolved_cost_multiplier = (
        float(effective_config.policy_cost_multiplier)
        if cost_multiplier is None
        else float(cost_multiplier)
    )
    resolved_gamma_multiplier = (
        float(effective_config.policy_gamma_multiplier)
        if gamma_multiplier is None
        else float(gamma_multiplier)
    )
    resolved_state_reset_mode = state_reset_mode or config.evaluation_state_reset_mode
    _validate_state_reset_mode(resolved_state_reset_mode)
    previous_state = None
    previous_position = 0.0
    previous_example: TrainingExample | None = None
    pnl_values: list[float] = []
    log_wealth_values: list[float] = []
    drawdown_values: list[float] = []
    mu_errors: list[float] = []
    sigma_errors: list[float] = []
    tradeability_gates: list[float] = []
    shape_entropies: list[float] = []
    policy_scores: list[float] = []
    horizon_counts = {str(horizon): 0 for horizon in config.horizons}
    direction_correct = 0
    total_samples = 0
    turnover = 0.0
    equity = 0.0
    peak_equity = 0.0
    no_trade_hits = 0
    log_wealth_clamp_hits = 0
    exact_smooth_horizon_matches = 0
    exact_smooth_no_trade_matches = 0
    exact_smooth_position_abs_error = 0.0
    exact_smooth_utility_regret = 0.0
    state_reset_count = 0
    session_count = 0
    window_count = 0

    with torch.no_grad():
        for example in examples:
            session_boundary, window_boundary = _state_reset_boundaries(
                current_example=example,
                previous_example=previous_example,
            )
            if previous_example is None or session_boundary:
                session_count += 1
            if previous_example is None or window_boundary:
                window_count += 1
            if _should_reset_recurrent_context(
                state_reset_mode=resolved_state_reset_mode,
                current_example=example,
                previous_example=previous_example,
            ):
                previous_state = None
                previous_position = 0.0
                state_reset_count += 1
            batch = examples_to_batch([example], effective_config)
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
            gate = float(outputs["tradeability_gate"][0].item())
            smooth_policy = smooth_policy_distribution(
                mean=outputs.get("policy_mu", outputs["mu"]),
                sigma=outputs.get("policy_sigma", outputs["sigma"]),
                costs=batch["horizon_costs"],
                tradeability_gate=outputs["tradeability_gate"],
                previous_position=torch.tensor([previous_position], dtype=policy_mean.dtype),
                config=effective_config,
                cost_multiplier=resolved_cost_multiplier,
                gamma_multiplier=resolved_gamma_multiplier,
            )
            decision = apply_selection_policy(
                example=example,
                mean=policy_mean.tolist(),
                sigma=policy_sigma.tolist(),
                config=effective_config,
                previous_position=previous_position,
                tradeability_gate=gate,
                shape_probs=outputs["shape_posterior"][0].tolist(),
                cost_multiplier=resolved_cost_multiplier,
                gamma_multiplier=resolved_gamma_multiplier,
            )

            selected_row = dict(decision["selected_row"])
            selected_horizon = int(selected_row["horizon"])
            selected_index = config.horizons.index(selected_horizon)
            realized_return = float(example.returns_target[selected_index])
            selected_forecast_mean = float(forecast_mean[selected_index].item())
            selected_forecast_sigma = float(forecast_sigma[selected_index].item())
            trade_cost = float(selected_row["cost"]) * abs(float(decision["trade_delta"]))
            pnl = (float(decision["position"]) * realized_return) - trade_cost
            equity += pnl
            peak_equity = max(peak_equity, equity)
            max_drawdown = peak_equity - equity

            pnl_values.append(pnl)
            log_wealth_values.append(math.log1p(max(pnl, -0.95)))
            drawdown_values.append(max_drawdown)
            mu_errors.append(abs(realized_return - selected_forecast_mean))
            sigma_errors.append(abs(abs(realized_return - selected_forecast_mean) - selected_forecast_sigma))
            tradeability_gates.append(gate)
            shape_entropies.append(float(outputs["shape_entropy"][0].item()))
            policy_scores.append(float(decision["selected_policy_utility"]))
            horizon_counts[str(selected_horizon)] += 1
            direction_correct += int(
                _sign_from_value(selected_forecast_mean) == _sign_from_value(realized_return)
            )
            total_samples += 1
            turnover += abs(float(decision["trade_delta"]))
            no_trade_hits += int(bool(decision["no_trade_band_hit"]))
            log_wealth_clamp_hits += int(pnl <= -0.95)

            smooth_selected_index = int(smooth_policy["selected_horizon_index"][0].item())
            smooth_selected_horizon = int(config.horizons[smooth_selected_index])
            smooth_selected_position = float(smooth_policy["selected_position"][0].item())
            smooth_selected_no_trade = bool(smooth_policy["selected_no_trade"][0].item())
            smooth_reference_row = dict(decision["horizon_rows"][smooth_selected_index])
            smooth_reference_utility = policy_utility(
                position=smooth_selected_position,
                previous_position=previous_position,
                gated_mean=float(smooth_reference_row["gated_mean"]),
                sigma=float(smooth_reference_row["sigma"]),
                cost=float(smooth_reference_row["cost"]),
                config=effective_config,
                gamma_multiplier=resolved_gamma_multiplier,
            )
            exact_smooth_horizon_matches += int(selected_horizon == smooth_selected_horizon)
            exact_smooth_no_trade_matches += int(
                bool(decision["no_trade_band_hit"]) == smooth_selected_no_trade
            )
            exact_smooth_position_abs_error += abs(
                float(decision["position"]) - smooth_selected_position
            )
            exact_smooth_utility_regret += max(
                float(selected_row["policy_utility"]) - float(smooth_reference_utility),
                0.0,
            )
            previous_position = float(decision["position"])
            previous_state = outputs["memory_state"].detach()
            previous_example = example

    average_log_wealth = sum(log_wealth_values) / max(total_samples, 1)
    realized_pnl_per_anchor = sum(pnl_values) / max(total_samples, 1)
    max_drawdown = max(drawdown_values, default=0.0)
    no_trade_rate = no_trade_hits / max(total_samples, 1)
    expert_entropy = sum(shape_entropies) / max(total_samples, 1)
    shape_gate_usage = sum(tradeability_gates) / max(total_samples, 1)
    mu_calibration = sum(mu_errors) / max(total_samples, 1)
    sigma_calibration = sum(sigma_errors) / max(total_samples, 1)
    direction_accuracy = direction_correct / max(total_samples, 1)
    cvar_tail = float(
        cvar_tail_loss(
            torch.tensor([-value for value in pnl_values], dtype=torch.float32),
            effective_config.cvar_alpha,
        ).item()
    )
    project_value_score = _project_value_score(
        average_log_wealth=average_log_wealth,
        cvar_tail=cvar_tail,
        max_drawdown=max_drawdown,
        no_trade_rate=no_trade_rate,
        mu_calibration=mu_calibration,
        shape_gate_usage=shape_gate_usage,
    )
    utility_score = _utility_score(
        average_log_wealth=average_log_wealth,
        direction_accuracy=direction_accuracy,
        no_trade_rate=no_trade_rate,
        shape_gate_usage=shape_gate_usage,
    )
    forecast_mae = sum(mu_errors) / max(total_samples, 1)
    checkpoint_selection_score = _checkpoint_selection_score(
        average_log_wealth=average_log_wealth,
        cvar_tail=cvar_tail,
        forecast_mae=forecast_mae,
        sigma_calibration=sigma_calibration,
        exact_smooth_position_mae=exact_smooth_position_abs_error / max(total_samples, 1),
        config=effective_config,
    )

    return {
        "policy_mode": "shape_aware_profit_maximization",
        "average_log_wealth": average_log_wealth,
        "realized_pnl_per_anchor": realized_pnl_per_anchor,
        "turnover": turnover,
        "max_drawdown": max_drawdown,
        "cvar_alpha": effective_config.cvar_alpha,
        "cvar_tail_loss": cvar_tail,
        "no_trade_band_hit_rate": no_trade_rate,
        "log_wealth_clamp_hit_rate": log_wealth_clamp_hits / max(total_samples, 1),
        "expert_entropy": expert_entropy,
        "shape_gate_usage": shape_gate_usage,
        "g_t_mean": shape_gate_usage,
        "mu_calibration": mu_calibration,
        "mu_t_calibration": mu_calibration,
        "sigma_calibration": sigma_calibration,
        "sigma_t_calibration": sigma_calibration,
        "directional_accuracy": direction_accuracy,
        "exact_smooth_horizon_agreement": exact_smooth_horizon_matches / max(total_samples, 1),
        "exact_smooth_no_trade_agreement": exact_smooth_no_trade_matches / max(total_samples, 1),
        "exact_smooth_position_mae": exact_smooth_position_abs_error / max(total_samples, 1),
        "exact_smooth_utility_regret": exact_smooth_utility_regret / max(total_samples, 1),
        "policy_score_mean": sum(policy_scores) / max(total_samples, 1),
        "policy_utility_mean": sum(policy_scores) / max(total_samples, 1),
        "utility_score": utility_score,
        "project_value_score": project_value_score,
        "forecast_mae": forecast_mae,
        "checkpoint_selection_metric": effective_config.checkpoint_selection_metric,
        "checkpoint_selection_score": checkpoint_selection_score,
        "state_reset_mode": resolved_state_reset_mode,
        "state_reset_boundary_spec_version": STATE_RESET_BOUNDARY_SPEC_VERSION,
        "state_reset_count": state_reset_count,
        "session_count": session_count,
        "window_count": window_count,
        "cost_multiplier": resolved_cost_multiplier,
        "gamma_multiplier": resolved_gamma_multiplier,
        "min_policy_sigma": float(effective_config.min_policy_sigma),
        "policy_horizon_distribution": {
            horizon: horizon_counts[horizon] / max(total_samples, 1)
            for horizon in sorted(horizon_counts, key=int)
        },
    }


def split_examples(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    sample_count = len(examples)
    desired_validation = max(1, int(sample_count * (1.0 - config.train_ratio)))
    desired_validation = min(desired_validation, sample_count - 1)
    purge = min(config.purge_examples, max(sample_count - desired_validation - 1, 0))
    train_end = max(1, sample_count - desired_validation - purge)
    validation_start = min(sample_count - 1, train_end + purge)
    train_examples = list(examples[:train_end])
    valid_examples = list(examples[validation_start:])
    if not valid_examples:
        train_examples = list(examples[:-1])
        valid_examples = list(examples[-1:])
    return train_examples, valid_examples


def _split_examples_into_contiguous_folds(
    examples: Sequence[TrainingExample],
    fold_count: int,
) -> list[list[TrainingExample]]:
    total_examples = len(examples)
    if total_examples == 0:
        return []
    resolved_fold_count = max(1, min(int(fold_count), total_examples))
    base_fold_size = total_examples // resolved_fold_count
    remainder = total_examples % resolved_fold_count
    folds: list[list[TrainingExample]] = []
    start_index = 0
    for fold_index in range(resolved_fold_count):
        current_size = base_fold_size + (1 if fold_index < remainder else 0)
        end_index = start_index + current_size
        folds.append(list(examples[start_index:end_index]))
        start_index = end_index
    return [fold for fold in folds if fold]


def _evaluate_blocked_epoch_metrics(
    model: SignalCascadeModel,
    valid_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, float]:
    if len(valid_examples) < max(2, int(config.walk_forward_folds)):
        return {}

    folds = _split_examples_into_contiguous_folds(valid_examples, config.walk_forward_folds)
    if len(folds) <= 1:
        return {}

    fold_metrics = [evaluate_model(model, fold_examples, config) for fold_examples in folds]
    average_log_wealth_values = [
        float(metrics["average_log_wealth"]) for metrics in fold_metrics
    ]
    cvar_values = [float(metrics["cvar_tail_loss"]) for metrics in fold_metrics]
    turnover_values = [float(metrics["turnover"]) for metrics in fold_metrics]
    blocked_average_log_wealth = sum(average_log_wealth_values) / len(average_log_wealth_values)
    blocked_cvar_tail_loss = sum(cvar_values) / len(cvar_values)
    return {
        "validation_blocked_fold_count": float(len(folds)),
        "validation_blocked_average_log_wealth_mean": blocked_average_log_wealth,
        "validation_blocked_cvar_tail_loss_mean": blocked_cvar_tail_loss,
        "validation_blocked_turnover_mean": sum(turnover_values) / len(turnover_values),
        "validation_blocked_objective_log_wealth_minus_lambda_cvar_mean": (
            blocked_average_log_wealth
            - (float(config.cvar_weight) * blocked_cvar_tail_loss)
        ),
    }


def _rolling_epoch_metric_mean(
    history: Sequence[dict[str, float]],
    key: str,
    window: int,
) -> float | None:
    numeric_values = [
        float(row[key])
        for row in history
        if key in row
    ]
    if not numeric_values:
        return None
    resolved_window = max(1, min(int(window), len(numeric_values)))
    return sum(numeric_values[-resolved_window:]) / resolved_window


def _resolve_epoch_selection_score(
    history: list[dict[str, float]],
    config: TrainingConfig,
) -> tuple[float, str]:
    current_row = history[-1]
    blocked_window = max(1, int(config.oof_epochs))

    if config.checkpoint_selection_metric == "exact_log_wealth":
        blocked_mean = _rolling_epoch_metric_mean(
            history,
            "validation_blocked_average_log_wealth_mean",
            blocked_window,
        )
        if blocked_mean is not None:
            current_row["validation_oof_epoch_window"] = float(
                min(blocked_window, len(history))
            )
            return -blocked_mean, "blocked_walk_forward_average_log_wealth_oof_mean"
    if config.checkpoint_selection_metric == "exact_log_wealth_minus_lambda_cvar":
        blocked_objective_mean = _rolling_epoch_metric_mean(
            history,
            "validation_blocked_objective_log_wealth_minus_lambda_cvar_mean",
            blocked_window,
        )
        if blocked_objective_mean is not None:
            current_row["validation_oof_epoch_window"] = float(
                min(blocked_window, len(history))
            )
            return (
                -blocked_objective_mean,
                "blocked_walk_forward_objective_log_wealth_minus_lambda_cvar_oof_mean",
            )

    return float(current_row["validation_selection_score"]), "single_split_exact_metric"


def _fit_model(
    train_examples: Sequence[TrainingExample],
    valid_examples: Sequence[TrainingExample],
    config: TrainingConfig,
    feature_dim: int,
    state_feature_dim: int,
) -> tuple[SignalCascadeModel, dict[str, object]]:
    model = SignalCascadeModel(
        feature_dim=feature_dim,
        state_feature_dim=state_feature_dim,
        hidden_dim=config.hidden_dim,
        state_dim=config.state_dim,
        num_horizons=len(config.horizons),
        shape_classes=config.shape_classes,
        branch_dilations=config.branch_dilations,
        dropout=config.dropout,
        tie_policy_to_forecast_head=config.tie_policy_to_forecast_head,
        disable_overlay_branch=config.disable_overlay_branch,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation_loss = float("inf")
    best_selection_score = float("inf")
    best_epoch = 1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            examples=train_examples,
            config=config,
            optimizer=optimizer,
            warmup_phase=epoch <= config.warmup_epochs,
        )
        valid_metrics = _run_epoch(
            model=model,
            examples=valid_examples,
            config=config,
            optimizer=None,
            warmup_phase=False,
        )
        exact_metrics = evaluate_model(model, valid_examples, config)
        blocked_metrics = _evaluate_blocked_epoch_metrics(model, valid_examples, config)
        history.append(
            {
                "epoch": float(epoch),
                "train_total": train_metrics["total"],
                "validation_total": valid_metrics["total"],
                "train_profit": train_metrics["profit_loss"],
                "validation_profit": valid_metrics["profit_loss"],
                "train_return": train_metrics["return_loss"],
                "validation_return": valid_metrics["return_loss"],
                "train_shape": train_metrics["shape_loss"],
                "validation_shape": valid_metrics["shape_loss"],
                "train_log_wealth": train_metrics["average_log_wealth"],
                "validation_log_wealth": valid_metrics["average_log_wealth"],
                "validation_forecast_mae": valid_metrics.get("forecast_mae", valid_metrics["return_loss"]),
                "validation_exact_log_wealth": exact_metrics["average_log_wealth"],
                "validation_exact_forecast_mae": exact_metrics["forecast_mae"],
                "validation_exact_cvar_tail_loss": exact_metrics["cvar_tail_loss"],
                "validation_exact_sigma_calibration": exact_metrics["sigma_calibration"],
                "validation_exact_position_mae": exact_metrics["exact_smooth_position_mae"],
                "validation_selection_score": exact_metrics["checkpoint_selection_score"],
                **blocked_metrics,
            }
        )
        selection_score, selection_score_source = _resolve_epoch_selection_score(history, config)
        history[-1]["validation_selection_score"] = selection_score
        history[-1]["validation_selection_score_source"] = selection_score_source
        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best_validation_loss = valid_metrics["total"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        "best_validation_loss": best_validation_loss,
        "best_selection_score": best_selection_score,
        "best_epoch": float(best_epoch),
        "checkpoint_audit": _build_checkpoint_audit(history, config),
        "history": history,
    }


def _build_checkpoint_audit(
    history: Sequence[dict[str, float]],
    config: TrainingConfig,
) -> dict[str, object]:
    ranked_rows = [
        row
        for row in history
        if "epoch" in row
        and "validation_exact_log_wealth" in row
        and "validation_selection_score" in row
    ]
    if not ranked_rows:
        return {}

    selection_rank = sorted(
        ranked_rows,
        key=lambda row: (
            float(row["validation_selection_score"]),
            float(row["epoch"]),
        ),
    )
    wealth_rank = sorted(
        ranked_rows,
        key=lambda row: (
            -float(row["validation_exact_log_wealth"]),
            float(row["epoch"]),
        ),
    )
    selected_row = selection_rank[0]
    best_wealth_row = wealth_rank[0]

    selected_epoch = int(float(selected_row["epoch"]))
    audit: dict[str, object] = {
        "selection_metric": config.checkpoint_selection_metric,
        "cvar_weight": float(config.cvar_weight),
        "selected_epoch": float(selected_epoch),
        "best_epoch_by_selection_score": float(selected_epoch),
        "best_epoch_by_exact_log_wealth": float(best_wealth_row["epoch"]),
        "selected_epoch_rank_by_exact_log_wealth": float(
            next(
                index
                for index, row in enumerate(wealth_rank, start=1)
                if int(float(row["epoch"])) == selected_epoch
            )
        ),
        "selected_epoch_exact_log_wealth": float(selected_row["validation_exact_log_wealth"]),
        "best_exact_log_wealth": float(best_wealth_row["validation_exact_log_wealth"]),
        "delta_to_best_exact_log_wealth": float(best_wealth_row["validation_exact_log_wealth"])
        - float(selected_row["validation_exact_log_wealth"]),
    }

    if all("validation_exact_cvar_tail_loss" in row for row in ranked_rows):
        wealth_minus_cvar_rank = sorted(
            ranked_rows,
            key=lambda row: (
                -(
                    float(row["validation_exact_log_wealth"])
                    - (
                        float(config.cvar_weight)
                        * float(row["validation_exact_cvar_tail_loss"])
                    )
                ),
                float(row["epoch"]),
            ),
        )
        best_wealth_minus_cvar_row = wealth_minus_cvar_rank[0]
        selected_wealth_minus_cvar = float(selected_row["validation_exact_log_wealth"]) - (
            float(config.cvar_weight) * float(selected_row["validation_exact_cvar_tail_loss"])
        )
        best_wealth_minus_cvar = float(best_wealth_minus_cvar_row["validation_exact_log_wealth"]) - (
            float(config.cvar_weight)
            * float(best_wealth_minus_cvar_row["validation_exact_cvar_tail_loss"])
        )
        audit.update(
            {
                "best_epoch_by_exact_log_wealth_minus_lambda_cvar": float(
                    best_wealth_minus_cvar_row["epoch"]
                ),
                "selected_epoch_rank_by_exact_log_wealth_minus_lambda_cvar": float(
                    next(
                        index
                        for index, row in enumerate(wealth_minus_cvar_rank, start=1)
                        if int(float(row["epoch"])) == selected_epoch
                    )
                ),
                "selected_epoch_exact_log_wealth_minus_lambda_cvar": selected_wealth_minus_cvar,
                "best_exact_log_wealth_minus_lambda_cvar": best_wealth_minus_cvar,
                "delta_to_best_exact_log_wealth_minus_lambda_cvar": (
                    best_wealth_minus_cvar - selected_wealth_minus_cvar
                ),
            }
        )
    else:
        audit["exact_log_wealth_minus_lambda_cvar_unavailable_reason"] = (
            "validation_exact_cvar_tail_loss_missing_in_history"
        )

    if all(
        "validation_blocked_objective_log_wealth_minus_lambda_cvar_mean" in row
        for row in ranked_rows
    ):
        blocked_rank = sorted(
            ranked_rows,
            key=lambda row: (
                -float(row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"]),
                float(row["epoch"]),
            ),
        )
        best_blocked_row = blocked_rank[0]
        audit.update(
            {
                "best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    best_blocked_row["epoch"]
                ),
                "selected_epoch_rank_by_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    next(
                        index
                        for index, row in enumerate(blocked_rank, start=1)
                        if int(float(row["epoch"])) == selected_epoch
                    )
                ),
                "selected_epoch_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    selected_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"]
                ),
                "best_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    best_blocked_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"]
                ),
                "delta_to_best_blocked_objective_log_wealth_minus_lambda_cvar": (
                    float(best_blocked_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"])
                    - float(selected_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"])
                ),
            }
        )

    return audit


def _run_epoch(
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    warmup_phase: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    batches = 0
    previous_state = None
    previous_position = torch.tensor([0.0], dtype=torch.float32)
    previous_example: TrainingExample | None = None

    for example in examples:
        if _should_reset_recurrent_context(
            state_reset_mode=config.training_state_reset_mode,
            current_example=example,
            previous_example=previous_example,
        ):
            previous_state = None
            previous_position = torch.tensor([0.0], dtype=torch.float32)
        batch = examples_to_batch([example], config)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(
                batch["main"],
                batch["overlay"],
                batch["state_features"],
                previous_state=previous_state,
            )
            loss, metrics, policy_metrics = total_loss(
                outputs=outputs,
                batch=batch,
                config=config,
                previous_position=previous_position,
                warmup_phase=warmup_phase,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        for key, value in metrics.items():
            totals[key] += value
        batches += 1
        previous_state = outputs["memory_state"].detach()
        previous_position = policy_metrics["selected_position"].detach().reshape(1)
        previous_example = example

    if batches == 0:
        raise ValueError("Epoch received no examples.")
    return {key: value / batches for key, value in totals.items()}


def _validate_state_reset_mode(state_reset_mode: str) -> None:
    allowed = {
        "carry_on",
        "reset_each_example",
        "reset_each_session_or_window",
    }
    if state_reset_mode not in allowed:
        raise ValueError(f"Unsupported state reset mode: {state_reset_mode}")


def replay_recurrent_context(
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    initial_previous_position: float = 0.0,
    state_reset_mode: str | None = None,
) -> tuple[torch.Tensor | None, float]:
    resolved_state_reset_mode = state_reset_mode or config.evaluation_state_reset_mode
    _validate_state_reset_mode(resolved_state_reset_mode)
    previous_state = None
    previous_position = float(initial_previous_position)
    previous_example: TrainingExample | None = None
    model.eval()

    with torch.no_grad():
        for example in examples:
            if _should_reset_recurrent_context(
                state_reset_mode=resolved_state_reset_mode,
                current_example=example,
                previous_example=previous_example,
            ):
                previous_state = None
                previous_position = 0.0
            batch = examples_to_batch([example], config)
            outputs = model(
                batch["main"],
                batch["overlay"],
                batch["state_features"],
                previous_state=previous_state,
            )
            policy_mean = outputs.get("policy_mu", outputs["mu"])[0]
            policy_sigma = outputs.get("policy_sigma", outputs["sigma"])[0]
            decision = apply_selection_policy(
                example=example,
                mean=policy_mean.tolist(),
                sigma=policy_sigma.tolist(),
                config=config,
                previous_position=previous_position,
                tradeability_gate=float(outputs["tradeability_gate"][0].item()),
                shape_probs=outputs["shape_posterior"][0].tolist(),
            )
            previous_position = float(decision["position"])
            previous_state = outputs["memory_state"].detach()
            previous_example = example
    return previous_state, previous_position


def _checkpoint_selection_score(
    *,
    average_log_wealth: float,
    cvar_tail: float,
    forecast_mae: float,
    sigma_calibration: float,
    exact_smooth_position_mae: float,
    config: TrainingConfig,
) -> float:
    if config.checkpoint_selection_metric == "validation_total":
        return forecast_mae
    if config.checkpoint_selection_metric == "exact_log_wealth":
        return -float(average_log_wealth)
    if config.checkpoint_selection_metric == "exact_log_wealth_minus_lambda_cvar":
        return -(
            float(average_log_wealth)
            - (float(config.cvar_weight) * float(cvar_tail))
        )
    calibration_error = 0.5 * (float(forecast_mae) + float(sigma_calibration))
    return (
        -float(average_log_wealth)
        + (float(config.checkpoint_selection_forecast_weight) * float(forecast_mae))
        + (float(config.checkpoint_selection_calibration_weight) * calibration_error)
        + (
            float(config.checkpoint_selection_position_gap_weight)
            * float(exact_smooth_position_mae)
        )
    )


def _should_reset_recurrent_context(
    state_reset_mode: str,
    current_example: TrainingExample,
    previous_example: TrainingExample | None,
) -> bool:
    _validate_state_reset_mode(state_reset_mode)
    if previous_example is None:
        return True
    if state_reset_mode == "carry_on":
        return False
    if state_reset_mode == "reset_each_example":
        return True
    session_boundary, window_boundary = _state_reset_boundaries(
        current_example=current_example,
        previous_example=previous_example,
    )
    return session_boundary or window_boundary


def _state_reset_boundaries(
    current_example: TrainingExample,
    previous_example: TrainingExample | None,
) -> tuple[bool, bool]:
    if previous_example is None:
        return True, True
    previous_session = previous_example.regime_id.split("|", maxsplit=1)[0]
    current_session = current_example.regime_id.split("|", maxsplit=1)[0]
    previous_day = previous_example.anchor_time.date()
    current_day = current_example.anchor_time.date()
    anchor_gap = current_example.anchor_time - previous_example.anchor_time
    return (
        previous_session != current_session,
        previous_day != current_day or anchor_gap > timedelta(hours=4),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _project_value_score(
    average_log_wealth: float,
    cvar_tail: float,
    max_drawdown: float,
    no_trade_rate: float,
    mu_calibration: float,
    shape_gate_usage: float,
) -> float:
    wealth_score = _clamp(0.5 + (average_log_wealth * 10.0))
    cvar_score = _clamp(1.0 - (cvar_tail * 8.0))
    drawdown_score = _clamp(1.0 - (max_drawdown * 8.0))
    calibration_score = _clamp(1.0 - (mu_calibration * 12.0))
    activity_score = _clamp(1.0 - no_trade_rate)
    gate_score = _clamp(shape_gate_usage)
    return (
        (0.28 * wealth_score)
        + (0.20 * cvar_score)
        + (0.18 * drawdown_score)
        + (0.14 * calibration_score)
        + (0.10 * activity_score)
        + (0.10 * gate_score)
    )


def _utility_score(
    average_log_wealth: float,
    direction_accuracy: float,
    no_trade_rate: float,
    shape_gate_usage: float,
) -> float:
    wealth_score = _clamp(0.5 + (average_log_wealth * 10.0))
    return (
        (0.40 * wealth_score)
        + (0.25 * _clamp(direction_accuracy))
        + (0.20 * _clamp(1.0 - no_trade_rate))
        + (0.15 * _clamp(shape_gate_usage))
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _sign_from_value(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0
