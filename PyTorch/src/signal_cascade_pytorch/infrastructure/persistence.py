from __future__ import annotations

import json
from pathlib import Path

import torch

from ..application.config import TrainingConfig


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, model: torch.nn.Module, config: TrainingConfig) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
