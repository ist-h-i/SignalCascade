from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.close_anchor import TimeframeParameters
from ..domain.timeframes import HORIZONS


def _default_main_windows() -> dict[str, int]:
    return {"4h": 48, "1d": 21, "1w": 8}


def _default_overlay_windows() -> dict[str, int]:
    return {"1h": 48, "30m": 96}


def _default_timeframe_parameters() -> dict[str, TimeframeParameters]:
    return {
        "30m": TimeframeParameters(ema_window=32, gate_weights=(0.60, 0.30, -0.20)),
        "1h": TimeframeParameters(ema_window=32, gate_weights=(0.60, 0.30, -0.20)),
        "4h": TimeframeParameters(ema_window=24, gate_weights=(0.75, 0.40, -0.25)),
        "1d": TimeframeParameters(ema_window=20, gate_weights=(0.85, 0.45, -0.30)),
        "1w": TimeframeParameters(ema_window=13, gate_weights=(0.95, 0.50, -0.35)),
    }


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    synthetic_bars: int = 10_080
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    dropout: float = 0.1
    train_ratio: float = 0.8
    output_dir: str = "artifacts/demo"
    horizons: tuple[int, ...] = HORIZONS
    main_windows: dict[str, int] = field(default_factory=_default_main_windows)
    overlay_windows: dict[str, int] = field(default_factory=_default_overlay_windows)
    timeframe_parameters: dict[str, TimeframeParameters] = field(
        default_factory=_default_timeframe_parameters
    )

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "synthetic_bars": self.synthetic_bars,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "train_ratio": self.train_ratio,
            "output_dir": self.output_dir,
            "horizons": list(self.horizons),
            "main_windows": dict(self.main_windows),
            "overlay_windows": dict(self.overlay_windows),
            "timeframe_parameters": {
                key: value.to_dict() for key, value in self.timeframe_parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TrainingConfig":
        timeframe_parameters = {
            key: TimeframeParameters.from_dict(value)
            for key, value in dict(payload["timeframe_parameters"]).items()
        }
        return cls(
            seed=int(payload["seed"]),
            synthetic_bars=int(payload["synthetic_bars"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
            train_ratio=float(payload["train_ratio"]),
            output_dir=str(payload["output_dir"]),
            horizons=tuple(int(value) for value in payload["horizons"]),
            main_windows={key: int(value) for key, value in dict(payload["main_windows"]).items()},
            overlay_windows={
                key: int(value) for key, value in dict(payload["overlay_windows"]).items()
            },
            timeframe_parameters=timeframe_parameters,
        )
