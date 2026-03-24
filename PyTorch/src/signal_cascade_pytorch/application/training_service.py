from __future__ import annotations

import random
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
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
        "overlay_target": torch.tensor(
            [example.overlay_target for example in examples],
            dtype=torch.long,
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
    train_examples, valid_examples = _split_examples(examples, config.train_ratio)
    feature_dim = len(examples[0].main_sequences["4h"][0])
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
    best_checkpoint = output_dir / "model.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer)
        valid_metrics = _run_epoch(model, valid_loader)
        epoch_record = {
            "epoch": float(epoch),
            "train_total": train_metrics["total"],
            "validation_total": valid_metrics["total"],
            "train_return": train_metrics["return_loss"],
            "validation_return": valid_metrics["return_loss"],
            "train_overlay": train_metrics["overlay_loss"],
            "validation_overlay": valid_metrics["overlay_loss"],
        }
        history.append(epoch_record)

        if valid_metrics["total"] < best_validation_loss:
            best_validation_loss = valid_metrics["total"]
            save_checkpoint(best_checkpoint, model, config)

    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate_model(model, valid_examples, config)
    summary = {
        "train_samples": len(train_examples),
        "validation_samples": len(valid_examples),
        "best_validation_loss": best_validation_loss,
        "best_epoch": min(history, key=lambda row: row["validation_total"])["epoch"],
        "history": history,
        "validation_metrics": validation_metrics,
    }
    return model, summary


def evaluate_model(
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, float]:
    loader = DataLoader(
        TorchExampleDataset(examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=examples_to_batch,
    )
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_points = 0
    direction_correct = 0
    uncertainty_calibration_error = 0.0
    coverage_1sigma = 0
    overlay_correct = 0
    overlay_confusion = [[0 for _ in range(4)] for _ in range(4)]
    realized_value_sum = 0.0
    realized_value_abs_sum = 0.0
    downside_value_sum = 0.0
    opportunity_count = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["main"], batch["overlay"])
            predicted_returns = outputs["mu"]
            predicted_sigma = outputs["sigma"]
            target_returns = batch["returns"]
            absolute_error = (predicted_returns - target_returns).abs()
            total_squared_error += float(torch.square(predicted_returns - target_returns).sum().item())
            total_absolute_error += float(absolute_error.sum().item())
            total_points += int(target_returns.numel())
            direction_correct += int(
                ((predicted_returns >= 0) == (target_returns >= 0)).sum().item()
            )
            uncertainty_calibration_error += float(
                (absolute_error - predicted_sigma).abs().sum().item()
            )
            coverage_1sigma += int((absolute_error <= predicted_sigma).sum().item())

            predicted_overlay = outputs["overlay_logits"].argmax(dim=1)
            overlay_target = batch["overlay_target"]
            overlay_correct += int((predicted_overlay == overlay_target).sum().item())
            for target_value, predicted_value in zip(
                overlay_target.tolist(),
                predicted_overlay.tolist(),
            ):
                overlay_confusion[int(target_value)][int(predicted_value)] += 1

            edge = predicted_returns.abs() / predicted_sigma.clamp_min(1e-6)
            selected_indices = edge.argmax(dim=1)
            positions = torch.tanh(
                predicted_returns.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
                / predicted_sigma.gather(1, selected_indices.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            )
            realized_returns = target_returns.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
            realized_value = positions * realized_returns
            realized_value_sum += float(realized_value.sum().item())
            realized_value_abs_sum += float(realized_returns.abs().sum().item())
            downside_value_sum += float(torch.minimum(realized_value, torch.zeros_like(realized_value)).sum().item())
            opportunity_count += int(realized_value.numel())

    overlay_total = len(examples)
    macro_f1 = _macro_f1(overlay_confusion)
    value_per_signal = realized_value_sum / max(opportunity_count, 1)
    value_capture_ratio = realized_value_sum / max(realized_value_abs_sum, 1e-6)
    downside_per_signal = downside_value_sum / max(opportunity_count, 1)
    utility_score = (
        (0.45 * value_capture_ratio)
        + (0.25 * (direction_correct / max(total_points, 1)))
        + (0.20 * macro_f1)
        - (0.05 * (total_absolute_error / max(total_points, 1)))
        - (0.05 * (uncertainty_calibration_error / max(total_points, 1)))
    )
    return {
        "return_rmse": sqrt(total_squared_error / max(total_points, 1)),
        "return_mae": total_absolute_error / max(total_points, 1),
        "directional_accuracy": direction_correct / max(total_points, 1),
        "uncertainty_calibration_error": uncertainty_calibration_error / max(total_points, 1),
        "coverage_at_1sigma": coverage_1sigma / max(total_points, 1),
        "overlay_accuracy": overlay_correct / max(overlay_total, 1),
        "overlay_macro_f1": macro_f1,
        "value_per_signal": value_per_signal,
        "downside_per_signal": downside_per_signal,
        "value_capture_ratio": value_capture_ratio,
        "utility_score": utility_score,
    }


def _split_examples(
    examples: list[TrainingExample],
    train_ratio: float,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    split_index = max(1, int(len(examples) * train_ratio))
    split_index = min(split_index, len(examples) - 1)
    return examples[:split_index], examples[split_index:]


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


def _macro_f1(confusion: list[list[int]]) -> float:
    per_class_scores: list[float] = []
    for class_index, row in enumerate(confusion):
        true_positive = row[class_index]
        false_positive = sum(confusion[other][class_index] for other in range(len(confusion))) - true_positive
        false_negative = sum(row) - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        if precision_denominator == 0 or recall_denominator == 0:
            per_class_scores.append(0.0)
            continue
        precision = true_positive / precision_denominator
        recall = true_positive / recall_denominator
        if precision + recall == 0:
            per_class_scores.append(0.0)
            continue
        per_class_scores.append(2 * precision * recall / (precision + recall))
    return sum(per_class_scores) / max(len(per_class_scores), 1)
