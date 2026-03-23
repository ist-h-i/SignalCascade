from __future__ import annotations

import random
from collections import defaultdict
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
    summary = {
        "train_samples": len(train_examples),
        "validation_samples": len(valid_examples),
        "best_validation_loss": best_validation_loss,
        "history": history,
    }
    return model, summary


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
