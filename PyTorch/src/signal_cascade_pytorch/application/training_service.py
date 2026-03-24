from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from math import sqrt
from pathlib import Path
from typing import Sequence

import torch
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


def examples_to_batch(examples: Sequence[TrainingExample]) -> dict[str, object]:
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
    return {
        "main": main,
        "overlay": overlay,
        "shape_targets": shape_targets,
        "returns": torch.tensor(
            [example.returns_target for example in examples],
            dtype=torch.float32,
        ),
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


def train_model(
    examples: list[TrainingExample],
    config: TrainingConfig,
    output_dir: Path,
) -> tuple[SignalCascadeModel, dict[str, object]]:
    _set_seed(config.seed)
    train_examples, valid_examples = _split_examples(examples, config)
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
) -> dict[str, float]:
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
    turnover = 0.0
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    previous_position = 0.0
    horizon_to_index = {horizon: index for index, horizon in enumerate(config.horizons)}

    with torch.no_grad():
        for chunk in _chunk_examples(examples, config.batch_size):
            batch = examples_to_batch(chunk)
            outputs = model(batch["main"], batch["overlay"])
            predicted_returns = outputs["mu"]
            predicted_sigma = outputs["sigma"]
            direction_probabilities = torch.softmax(outputs["direction_logits"], dim=-1)
            overlay_probabilities = torch.sigmoid(outputs["overlay_logits"].squeeze(-1))
            target_returns = batch["returns"]
            absolute_error = (predicted_returns - target_returns).abs()

            total_squared_error += float(torch.square(predicted_returns - target_returns).sum().item())
            total_absolute_error += float(absolute_error.sum().item())
            total_points += int(target_returns.numel())
            direction_correct += int(
                (direction_probabilities.argmax(dim=-1) == batch["direction_target"]).sum().item()
            )
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

                if decision["accepted_signal"]:
                    accepted_signals += 1
                    accepted_clean += int(decision["meta_label"])

                overlay_correct += int(
                    (decision["overlay_action"] == "hold") == bool(example.overlay_target)
                )

                selected_index = horizon_to_index[decision["selected_horizon"]]
                realized_return = float(example.returns_target[selected_index])
                realized_value = float(decision["position"]) * realized_return
                realized_value_sum += realized_value
                realized_value_abs_sum += abs(realized_return)
                turnover += abs(float(decision["position"]) - previous_position)
                previous_position = float(decision["position"])
                equity += realized_value
                peak_equity = max(peak_equity, equity)
                max_drawdown = max(max_drawdown, peak_equity - equity)

    coverage = accepted_signals / max(len(examples), 1)
    selection_precision = accepted_clean / max(accepted_signals, 1)
    utility_score = (
        (0.45 * selection_precision)
        + (0.20 * coverage)
        + (0.15 * (realized_value_sum / max(realized_value_abs_sum, 1e-6)))
        + (0.10 * (overlay_correct / max(len(examples), 1)))
        + (0.10 * (direction_correct / max(total_points, 1)))
    )
    return {
        "return_rmse": sqrt(total_squared_error / max(total_points, 1)),
        "return_mae": total_absolute_error / max(total_points, 1),
        "directional_accuracy": direction_correct / max(total_points, 1),
        "uncertainty_calibration_error": uncertainty_calibration_error / max(total_points, 1),
        "coverage_at_1sigma": coverage_1sigma / max(total_points, 1),
        "overlay_accuracy": overlay_correct / max(len(examples), 1),
        "selection_precision": selection_precision,
        "coverage_at_target_precision": coverage,
        "no_trade_rate": 1.0 - coverage,
        "value_per_signal": realized_value_sum / max(len(examples), 1),
        "value_capture_ratio": realized_value_sum / max(realized_value_abs_sum, 1e-6),
        "turnover": turnover,
        "max_drawdown": max_drawdown,
        "utility_score": utility_score,
    }


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
        collate_fn=examples_to_batch,
    )
    valid_loader = DataLoader(
        TorchExampleDataset(valid_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=examples_to_batch,
    )

    best_validation_loss = float("inf")
    best_epoch = 1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer)
        valid_metrics = _run_epoch(model, valid_loader)
        history.append(
            {
                "epoch": float(epoch),
                "train_total": train_metrics["total"],
                "validation_total": valid_metrics["total"],
                "train_return": train_metrics["return_loss"],
                "validation_return": valid_metrics["return_loss"],
                "train_direction": train_metrics["direction_loss"],
                "validation_direction": valid_metrics["direction_loss"],
                "train_overlay": train_metrics["overlay_loss"],
                "validation_overlay": valid_metrics["overlay_loss"],
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
            batch = examples_to_batch(chunk)
            outputs = model(batch["main"], batch["overlay"])
            direction_probabilities = torch.softmax(outputs["direction_logits"], dim=-1)
            overlay_probabilities = torch.sigmoid(outputs["overlay_logits"].squeeze(-1))
            snapshots.extend(
                build_prediction_snapshots(
                    examples=chunk,
                    mean=outputs["mu"],
                    sigma=outputs["sigma"],
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
    split_index = max(1, int(len(examples) * config.train_ratio))
    split_index = min(split_index, len(examples) - 1)
    purge = min(config.purge_examples, max(len(examples) - split_index - 1, 0))
    validation_start = min(len(examples) - 1, split_index + purge)
    train_examples = list(examples[:split_index])
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
            loss, metrics = total_loss(outputs, batch)
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
