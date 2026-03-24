from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from math import atan, pi, sqrt
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .policy_service import (
    apply_selection_policy,
    build_prediction_snapshots,
    build_selection_policy,
)
from ..domain.entities import TrainingExample
from ..domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES
from ..infrastructure.ml.losses import total_loss
from ..infrastructure.ml.model import SignalCascadeModel
from ..infrastructure.persistence import save_checkpoint


class TorchExampleDataset(Dataset):
    def __init__(self, examples: Sequence[TrainingExample]) -> None:
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self._examples[index]


def examples_to_batch(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, object]:
    main = {
        timeframe: torch.tensor(
            [example.main_sequences[timeframe] for example in examples],
            dtype=torch.float32,
        )
        for timeframe in MAIN_TIMEFRAMES
    }
    overlay = {
        timeframe: torch.tensor(
            [example.overlay_sequences[timeframe] for example in examples],
            dtype=torch.float32,
        )
        for timeframe in OVERLAY_TIMEFRAMES
    }
    shape_targets = {
        timeframe: torch.tensor(
            [example.main_shape_targets[timeframe] for example in examples],
            dtype=torch.float32,
        )
        for timeframe in MAIN_TIMEFRAMES
    }
    raw_returns = torch.tensor(
        [example.returns_target for example in examples],
        dtype=torch.float32,
    )
    return_scales = torch.tensor(
        [
            [_return_scale(example.realized_volatility, horizon, config) for horizon in config.horizons]
            for example in examples
        ],
        dtype=torch.float32,
    )
    standardized_returns = torch.clamp(
        raw_returns / return_scales,
        min=-config.standardized_return_clip,
        max=config.standardized_return_clip,
    )
    direction_band = torch.tensor(
        [
            [
                float(example.direction_thresholds[horizon_index]) / max(
                    _return_scale(example.realized_volatility, horizon, config),
                    1e-6,
                )
                for horizon_index, horizon in enumerate(config.horizons)
            ]
            for example in examples
        ],
        dtype=torch.float32,
    )
    return {
        "main": main,
        "overlay": overlay,
        "shape_targets": shape_targets,
        "returns": standardized_returns,
        "returns_raw": raw_returns,
        "return_scale": return_scales,
        "direction_band": direction_band,
        "direction_target": torch.tensor(
            [[direction + 1 for direction in example.direction_targets] for example in examples],
            dtype=torch.long,
        ),
        "direction_weight": torch.tensor(
            [example.direction_weights for example in examples],
            dtype=torch.float32,
        ),
        "overlay_target": torch.tensor(
            [float(example.overlay_target) for example in examples],
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
    return_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mean * return_scale, sigma * return_scale


def train_model(
    examples: list[TrainingExample],
    config: TrainingConfig,
    output_dir: Path,
) -> tuple[SignalCascadeModel, dict[str, object]]:
    _set_seed(config.seed)
    train_examples, valid_examples = split_examples(examples, config)
    feature_dim = len(examples[0].main_sequences["4h"][0])
    model, fit_summary = _fit_model(train_examples, valid_examples, config, feature_dim)
    save_checkpoint(output_dir / "model.pt", model, config)

    oof_snapshots = _build_walk_forward_snapshots(examples, config, feature_dim)
    if len(oof_snapshots) < max(8, len(valid_examples) // 2):
        oof_snapshots = _predict_snapshots(model, valid_examples, config)
    selection_policy = build_selection_policy(oof_snapshots, config)
    validation_metrics = evaluate_model(model, valid_examples, config, selection_policy)

    summary = {
        "train_samples": len(train_examples),
        "validation_samples": len(valid_examples),
        "best_validation_loss": fit_summary["best_validation_loss"],
        "best_epoch": fit_summary["best_epoch"],
        "history": fit_summary["history"],
        "validation_metrics": validation_metrics,
        "policy_metrics": selection_policy.get("metrics", {}),
        "selection_policy": selection_policy,
    }
    return model, summary


def evaluate_model(
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    selection_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    if not examples:
        raise ValueError("At least one example is required for evaluation.")

    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_points = 0
    direction_correct = 0
    uncertainty_calibration_error = 0.0
    coverage_1sigma = 0
    overlay_correct = 0
    accepted_signals = 0
    accepted_clean = 0
    realized_value_sum = 0.0
    realized_value_abs_sum = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    turnover = 0.0
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    previous_position = 0.0
    total_direction_brier = 0.0
    direction_brier_points = 0
    realized_values: list[float] = []
    selection_probabilities: list[float] = []
    selection_labels: list[int] = []
    hold_probabilities: list[float] = []
    hold_labels: list[int] = []
    overlay_predictions: list[int] = []
    pre_threshold_value_sum = 0.0
    pre_threshold_abs_value_sum = 0.0
    horizon_counts = {str(horizon): 0 for horizon in config.horizons}
    horizon_to_index = {horizon: index for index, horizon in enumerate(config.horizons)}
    proposed_count = 0
    no_candidate_count = 0
    horizon_audit = {
        str(horizon): {
            "samples": 0,
            "nonflat": 0,
            "up": 0,
            "down": 0,
            "alignment": 0,
            "actionable_edge": 0,
        }
        for horizon in config.horizons
    }

    with torch.no_grad():
        for chunk in _chunk_examples(examples, config.batch_size):
            batch = examples_to_batch(chunk, config)
            outputs = model(batch["main"], batch["overlay"])
            predicted_returns, predicted_sigma = restore_return_units(
                outputs["mu"],
                outputs["sigma"],
                batch["return_scale"],
            )
            direction_probabilities = torch.softmax(outputs["direction_logits"], dim=-1)
            overlay_probabilities = torch.sigmoid(outputs["overlay_logits"].squeeze(-1))
            target_returns = batch["returns_raw"]
            absolute_error = (predicted_returns - target_returns).abs()
            direction_targets = functional.one_hot(batch["direction_target"], num_classes=3).float()

            total_squared_error += float(torch.square(predicted_returns - target_returns).sum().item())
            total_absolute_error += float(absolute_error.sum().item())
            total_points += int(target_returns.numel())
            direction_correct += int(
                (direction_probabilities.argmax(dim=-1) == batch["direction_target"]).sum().item()
            )
            total_direction_brier += float(
                torch.square(direction_probabilities - direction_targets).sum(dim=-1).sum().item()
            )
            direction_brier_points += int(batch["direction_target"].numel())
            uncertainty_calibration_error += float(
                (absolute_error - predicted_sigma).abs().sum().item()
            )
            coverage_1sigma += int((absolute_error <= predicted_sigma).sum().item())

            for example_index, example in enumerate(chunk):
                decision = apply_selection_policy(
                    example=example,
                    mean=predicted_returns[example_index].tolist(),
                    sigma=predicted_sigma[example_index].tolist(),
                    direction_probabilities=direction_probabilities[example_index].tolist(),
                    overlay_probability=float(overlay_probabilities[example_index].item()),
                    policy=selection_policy,
                    config=config,
                )
                for horizon_index, row in enumerate(decision["horizon_rows"]):
                    audit = horizon_audit[str(row["horizon"])]
                    audit["samples"] += 1
                    direction_target = int(example.direction_targets[horizon_index])
                    audit["nonflat"] += int(direction_target != 0)
                    audit["up"] += int(direction_target > 0)
                    audit["down"] += int(direction_target < 0)
                    audit["alignment"] += int(
                        int(row["actionable_sign"]) != 0
                        and _sign_from_value(float(row["mean"])) == int(row["actionable_sign"])
                    )
                    audit["actionable_edge"] += int(float(row["actionable_edge"]) > 0.0)

                if decision["accepted_signal"]:
                    accepted_signals += 1
                    accepted_clean += int(decision["meta_label"])

                overlay_target = int(bool(example.overlay_target))
                overlay_prediction = int(decision["overlay_action"] == "hold")
                overlay_correct += int(overlay_prediction == overlay_target)
                overlay_predictions.append(overlay_prediction)
                hold_labels.append(overlay_target)
                hold_probabilities.append(float(decision["hold_probability"]))
                selection_labels.append(int(decision["meta_label"]))
                selection_probabilities.append(float(decision["selection_probability"]))

                proposed_horizon = decision["proposed_horizon"]
                if proposed_horizon is None:
                    no_candidate_count += 1
                    realized_return = 0.0
                else:
                    proposed_count += 1
                    selected_index = horizon_to_index[int(proposed_horizon)]
                    horizon_counts[str(proposed_horizon)] += 1
                    realized_return = float(example.returns_target[selected_index])
                realized_value = float(decision["position"]) * realized_return
                pre_threshold_value = float(decision["pre_threshold_position"]) * realized_return
                realized_value_sum += realized_value
                realized_value_abs_sum += abs(realized_return)
                pre_threshold_value_sum += pre_threshold_value
                pre_threshold_abs_value_sum += abs(realized_return)
                gross_profit += max(realized_value, 0.0)
                gross_loss += min(realized_value, 0.0)
                realized_values.append(realized_value)
                turnover += abs(float(decision["position"]) - previous_position)
                previous_position = float(decision["position"])
                equity += realized_value
                peak_equity = max(peak_equity, equity)
                max_drawdown = max(max_drawdown, peak_equity - equity)

    total_examples = max(len(examples), 1)
    proposal_coverage = proposed_count / total_examples
    coverage = accepted_signals / max(len(examples), 1)
    selection_precision = accepted_clean / max(accepted_signals, 1)
    overlay_macro_f1 = _binary_macro_f1(overlay_predictions, hold_labels)
    value_capture_ratio = realized_value_sum / max(realized_value_abs_sum, 1e-6)
    downside_per_signal = sum(min(value, 0.0) for value in realized_values) / total_examples
    accepted_value_per_signal = realized_value_sum / max(accepted_signals, 1)
    sparse_signals = accepted_signals < config.selection_min_support
    profit_factor = 0.0 if sparse_signals else _bounded_profit_factor(gross_profit, gross_loss)
    signal_sharpe = 0.0 if sparse_signals else _risk_adjusted_ratio(realized_values)
    signal_sortino = (
        0.0 if sparse_signals else _risk_adjusted_ratio(realized_values, downside_only=True)
    )
    selection_brier_score = _mean_square_error(selection_probabilities, selection_labels)
    selection_calibration_error = _mean_absolute_error(selection_probabilities, selection_labels)
    hold_brier_score = _mean_square_error(hold_probabilities, hold_labels)
    direction_brier_score = total_direction_brier / max(direction_brier_points, 1)
    capture_score = _clamp((value_capture_ratio + 1.0) / 2.0)
    profit_factor_score = _clamp(profit_factor / (1.0 + profit_factor))
    sortino_score = _bounded_score(signal_sortino)
    drawdown_score = _drawdown_score(realized_value_sum, max_drawdown)
    calibration_score = _clamp(1.0 - selection_brier_score)
    alignment_rate = (
        sum(int(audit["alignment"]) for audit in horizon_audit.values())
        / max(sum(int(audit["samples"]) for audit in horizon_audit.values()), 1)
    )
    actionable_edge_rate = (
        sum(int(audit["actionable_edge"]) for audit in horizon_audit.values())
        / max(sum(int(audit["samples"]) for audit in horizon_audit.values()), 1)
    )
    nonflat_rate = (
        sum(int(audit["nonflat"]) for audit in horizon_audit.values())
        / max(sum(int(audit["samples"]) for audit in horizon_audit.values()), 1)
    )
    hold_rate = sum(hold_labels) / total_examples
    pre_threshold_capture = pre_threshold_value_sum / max(pre_threshold_abs_value_sum, 1e-6)
    threshold_calibration_feasible = _policy_has_feasible_threshold(selection_policy)
    precision_feasible = (
        accepted_signals >= config.selection_min_support
        and selection_precision >= config.precision_target
        and threshold_calibration_feasible
    )
    threshold_meta = _selection_threshold_meta(selection_policy)
    utility_score = (
        (0.30 * selection_precision)
        + (0.20 * coverage)
        + (0.15 * capture_score)
        + (0.10 * overlay_macro_f1)
        + (0.10 * (direction_correct / max(total_points, 1)))
        + (0.10 * drawdown_score)
        + (0.05 * calibration_score)
    )
    project_value_score = (
        (0.22 * selection_precision)
        + (0.14 * coverage)
        + (0.16 * capture_score)
        + (0.14 * profit_factor_score)
        + (0.14 * sortino_score)
        + (0.10 * drawdown_score)
        + (0.10 * calibration_score)
    )
    if not precision_feasible:
        utility_score *= 0.5
        project_value_score *= 0.25
    return {
        "return_rmse": sqrt(total_squared_error / max(total_points, 1)),
        "return_mae": total_absolute_error / max(total_points, 1),
        "directional_accuracy": direction_correct / max(total_points, 1),
        "direction_brier_score": direction_brier_score,
        "uncertainty_calibration_error": uncertainty_calibration_error / max(total_points, 1),
        "coverage_at_1sigma": coverage_1sigma / max(total_points, 1),
        "overlay_accuracy": overlay_correct / max(len(examples), 1),
        "overlay_macro_f1": overlay_macro_f1,
        "proposal_count": proposed_count,
        "accepted_count": accepted_signals,
        "proposal_coverage": proposal_coverage,
        "selection_precision": selection_precision,
        "selection_support": accepted_signals,
        "acceptance_precision": selection_precision,
        "acceptance_support": accepted_signals,
        "precision_feasible": precision_feasible,
        "threshold_calibration_feasible": threshold_calibration_feasible,
        "coverage_at_target_precision": coverage,
        "acceptance_coverage": coverage,
        "no_trade_rate": 1.0 - coverage,
        "value_per_signal": realized_value_sum / total_examples,
        "value_per_anchor": realized_value_sum / total_examples,
        "value_per_proposed": (
            None if proposed_count == 0 else realized_value_sum / proposed_count
        ),
        "accepted_value_per_signal": accepted_value_per_signal,
        "value_per_accepted": (
            None if accepted_signals == 0 else realized_value_sum / accepted_signals
        ),
        "downside_per_signal": downside_per_signal,
        "value_capture_ratio": value_capture_ratio,
        "profit_factor": profit_factor,
        "signal_sharpe": signal_sharpe,
        "signal_sortino": signal_sortino,
        "selection_brier_score": selection_brier_score,
        "selection_calibration_error": selection_calibration_error,
        "selector_head_brier_score": selection_brier_score,
        "selector_head_calibration_error": selection_calibration_error,
        "selection_score_source": str(config.selection_score_source),
        "acceptance_score_source": str(config.selection_score_source),
        "allow_no_candidate": bool(config.allow_no_candidate),
        "hold_brier_score": hold_brier_score,
        "hold_rate": hold_rate,
        "turnover": turnover,
        "max_drawdown": max_drawdown,
        "no_candidate_rate": no_candidate_count / total_examples,
        "capture_score": capture_score,
        "profit_factor_score": profit_factor_score,
        "sortino_score": sortino_score,
        "drawdown_score": drawdown_score,
        "calibration_score": calibration_score,
        "alignment_rate": alignment_rate,
        "actionable_edge_rate": actionable_edge_rate,
        "nonflat_rate": nonflat_rate,
        "pre_threshold_capture": pre_threshold_capture,
        "best_selection_lcb": threshold_meta["best_selection_lcb"],
        "support_at_best_lcb": threshold_meta["support_at_best_lcb"],
        "precision_at_best_lcb": threshold_meta["precision_at_best_lcb"],
        "tau_at_best_lcb": threshold_meta["tau_at_best_lcb"],
        "utility_score": utility_score,
        "project_value_score": project_value_score,
        "proposed_horizon_distribution": {
            horizon: horizon_counts[horizon] / total_examples for horizon in sorted(horizon_counts)
        },
        "selected_horizon_distribution": {
            horizon: horizon_counts[horizon] / total_examples for horizon in sorted(horizon_counts)
        },
        "horizon_diagnostics": {
            horizon: {
                "nonflat_rate": int(audit["nonflat"]) / max(int(audit["samples"]), 1),
                "up_rate": int(audit["up"]) / max(int(audit["samples"]), 1),
                "down_rate": int(audit["down"]) / max(int(audit["samples"]), 1),
                "alignment_rate": int(audit["alignment"]) / max(int(audit["samples"]), 1),
                "actionable_edge_rate": int(audit["actionable_edge"]) / max(int(audit["samples"]), 1),
            }
            for horizon, audit in sorted(horizon_audit.items(), key=lambda item: int(item[0]))
        },
    }


def split_examples(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    return _split_examples(examples, config)


def _fit_model(
    train_examples: Sequence[TrainingExample],
    valid_examples: Sequence[TrainingExample],
    config: TrainingConfig,
    feature_dim: int,
) -> tuple[SignalCascadeModel, dict[str, object]]:
    model = SignalCascadeModel(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_horizons=len(config.horizons),
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = DataLoader(
        TorchExampleDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch_examples: examples_to_batch(batch_examples, config),
    )
    valid_loader = DataLoader(
        TorchExampleDataset(valid_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch_examples: examples_to_batch(batch_examples, config),
    )

    best_validation_loss = float("inf")
    best_epoch = 1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, config, optimizer)
        valid_metrics = _run_epoch(model, valid_loader, config)
        history.append(
            {
                "epoch": float(epoch),
                "train_total": train_metrics["total"],
                "validation_total": valid_metrics["total"],
                "train_return": train_metrics["return_loss"],
                "validation_return": valid_metrics["return_loss"],
                "train_direction": train_metrics["direction_loss"],
                "validation_direction": valid_metrics["direction_loss"],
                "train_shape": train_metrics["shape_loss"],
                "validation_shape": valid_metrics["shape_loss"],
                "train_overlay": train_metrics["overlay_loss"],
                "validation_overlay": valid_metrics["overlay_loss"],
                "train_consistency": train_metrics["consistency_loss"],
                "validation_consistency": valid_metrics["consistency_loss"],
            }
        )
        if valid_metrics["total"] < best_validation_loss:
            best_validation_loss = valid_metrics["total"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        "best_validation_loss": best_validation_loss,
        "best_epoch": float(best_epoch),
        "history": history,
    }


def _build_walk_forward_snapshots(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    feature_dim: int,
) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    fold_config = replace(config, epochs=max(1, config.oof_epochs))
    for train_slice, valid_slice in _walk_forward_slices(len(examples), config):
        train_examples = list(examples[train_slice[0] : train_slice[1]])
        valid_examples = list(examples[valid_slice[0] : valid_slice[1]])
        if len(train_examples) < max(config.batch_size, 32) or not valid_examples:
            continue
        model, _ = _fit_model(train_examples, valid_examples, fold_config, feature_dim)
        snapshots.extend(_predict_snapshots(model, valid_examples, config))
    return snapshots


def _predict_snapshots(
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    model.eval()
    snapshots: list[dict[str, object]] = []
    with torch.no_grad():
        for chunk in _chunk_examples(examples, config.batch_size):
            batch = examples_to_batch(chunk, config)
            outputs = model(batch["main"], batch["overlay"])
            direction_probabilities = torch.softmax(outputs["direction_logits"], dim=-1)
            overlay_probabilities = torch.sigmoid(outputs["overlay_logits"].squeeze(-1))
            predicted_returns, predicted_sigma = restore_return_units(
                outputs["mu"],
                outputs["sigma"],
                batch["return_scale"],
            )
            snapshots.extend(
                build_prediction_snapshots(
                    examples=chunk,
                    mean=predicted_returns,
                    sigma=predicted_sigma,
                    direction_probabilities=direction_probabilities,
                    overlay_probabilities=overlay_probabilities,
                    config=config,
                )
            )
    return snapshots


def _split_examples(
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


def _walk_forward_slices(
    sample_count: int,
    config: TrainingConfig,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if sample_count < 96:
        return []

    initial_train = max(int(sample_count * 0.45), 64)
    remaining = sample_count - initial_train
    if remaining <= 0:
        return []

    fold_size = max(1, remaining // max(config.walk_forward_folds, 1))
    slices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for fold_index in range(config.walk_forward_folds):
        validation_start = initial_train + (fold_index * fold_size)
        validation_end = sample_count if fold_index == config.walk_forward_folds - 1 else min(
            sample_count,
            validation_start + fold_size,
        )
        if validation_start >= sample_count or validation_start >= validation_end:
            continue
        train_end = max(1, validation_start - config.purge_examples)
        if train_end <= 1:
            continue
        slices.append(((0, train_end), (validation_start, validation_end)))
    return slices


def _chunk_examples(
    examples: Sequence[TrainingExample],
    batch_size: int,
) -> Sequence[list[TrainingExample]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _run_epoch(
    model: SignalCascadeModel,
    loader: DataLoader,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    batches = 0

    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(batch["main"], batch["overlay"])
            loss, metrics = total_loss(outputs, batch, config)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        for key, value in metrics.items():
            totals[key] += value
        batches += 1

    if batches == 0:
        raise ValueError("Loader did not yield any batches.")
    return {key: value / batches for key, value in totals.items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _precision(retrieved_true: int, retrieved_total: int) -> float:
    return retrieved_true / max(retrieved_total, 1)


def _f1_score(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = _precision(true_positive, true_positive + false_positive)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _binary_macro_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    tp_hold = sum(1 for predicted, label in zip(predictions, labels) if predicted == 1 and label == 1)
    fp_hold = sum(1 for predicted, label in zip(predictions, labels) if predicted == 1 and label == 0)
    fn_hold = sum(1 for predicted, label in zip(predictions, labels) if predicted == 0 and label == 1)
    f1_hold = _f1_score(tp_hold, fp_hold, fn_hold)

    tp_reduce = sum(1 for predicted, label in zip(predictions, labels) if predicted == 0 and label == 0)
    fp_reduce = sum(1 for predicted, label in zip(predictions, labels) if predicted == 0 and label == 1)
    fn_reduce = sum(1 for predicted, label in zip(predictions, labels) if predicted == 1 and label == 0)
    f1_reduce = _f1_score(tp_reduce, fp_reduce, fn_reduce)
    return (f1_hold + f1_reduce) / 2.0


def _mean_square_error(predictions: Sequence[float], labels: Sequence[int]) -> float:
    if not predictions:
        return 0.0
    return sum((float(prediction) - float(label)) ** 2 for prediction, label in zip(predictions, labels)) / len(
        predictions
    )


def _mean_absolute_error(predictions: Sequence[float], labels: Sequence[int]) -> float:
    if not predictions:
        return 0.0
    return sum(abs(float(prediction) - float(label)) for prediction, label in zip(predictions, labels)) / len(
        predictions
    )


def _bounded_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_profit <= 0.0:
        return 0.0
    if abs(gross_loss) <= 1e-6:
        return 10.0
    return min(gross_profit / abs(gross_loss), 10.0)


def _risk_adjusted_ratio(values: Sequence[float], downside_only: bool = False) -> float:
    if not values:
        return 0.0

    mean_value = sum(values) / len(values)
    if downside_only:
        deviation = sqrt(sum(min(value, 0.0) ** 2 for value in values) / len(values))
    else:
        deviation = sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
    if deviation <= 1e-9:
        return 0.0 if mean_value <= 0.0 else 10.0
    return max(-10.0, min((mean_value / deviation) * sqrt(len(values)), 10.0))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _bounded_score(value: float) -> float:
    return _clamp(0.5 + (atan(value) / pi))


def _drawdown_score(total_value: float, max_drawdown: float) -> float:
    if total_value <= 0.0:
        return 0.0
    return total_value / max(total_value + max_drawdown, 1e-6)


def _return_scale(
    realized_volatility: float,
    horizon: int,
    config: TrainingConfig,
) -> float:
    return max((float(realized_volatility) * sqrt(horizon)) + config.return_scale_epsilon, 1e-6)


def _selection_threshold_meta(policy: dict[str, object] | None) -> dict[str, float | None]:
    if not policy:
        return {
            "best_selection_lcb": 0.0,
            "support_at_best_lcb": 0.0,
            "precision_at_best_lcb": 0.0,
            "tau_at_best_lcb": None,
        }

    threshold_meta = dict(policy.get("selection_thresholds", {}).get("meta", {}))
    global_meta = dict(threshold_meta.get("global", {}))
    return {
        "best_selection_lcb": float(global_meta.get("best_selection_lcb", 0.0)),
        "support_at_best_lcb": float(global_meta.get("support_at_best_lcb", 0.0)),
        "precision_at_best_lcb": float(global_meta.get("precision_at_best_lcb", 0.0)),
        "tau_at_best_lcb": (
            None
            if global_meta.get("tau_at_best_lcb") is None
            else float(global_meta.get("tau_at_best_lcb", 0.0))
        ),
    }


def _policy_has_feasible_threshold(policy: dict[str, object] | None) -> bool:
    if not policy:
        return False

    threshold_meta = dict(policy.get("selection_thresholds", {}).get("meta", {}))
    global_meta = dict(threshold_meta.get("global", {}))
    return bool(global_meta.get("feasible"))


def _sign_from_value(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0
