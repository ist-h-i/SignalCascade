from __future__ import annotations

import math
from dataclasses import dataclass

from .candlestick import candlestick_shape, ema_series, path_averaged_directional_balance
from .entities import OHLCVBar, TimeframeFeatureRow


@dataclass(frozen=True)
class TimeframeParameters:
    ema_window: int
    gate_weights: tuple[float, float, float]
    gate_bias: float = 0.0
    beta0: float = 0.0
    beta_v: float = 0.05
    beta_x: float = 0.15
    beta_vx: float = 0.05
    epsilon: float = 1e-6

    def to_dict(self) -> dict[str, object]:
        return {
            "ema_window": self.ema_window,
            "gate_weights": list(self.gate_weights),
            "gate_bias": self.gate_bias,
            "beta0": self.beta0,
            "beta_v": self.beta_v,
            "beta_x": self.beta_x,
            "beta_vx": self.beta_vx,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TimeframeParameters":
        return cls(
            ema_window=int(payload["ema_window"]),
            gate_weights=tuple(float(value) for value in payload["gate_weights"]),
            gate_bias=float(payload["gate_bias"]),
            beta0=float(payload["beta0"]),
            beta_v=float(payload["beta_v"]),
            beta_x=float(payload["beta_x"]),
            beta_vx=float(payload["beta_vx"]),
            epsilon=float(payload["epsilon"]),
        )


def build_close_anchor_features(
    bars: list[OHLCVBar],
    parameters: TimeframeParameters,
) -> list[TimeframeFeatureRow]:
    ranges = [bar.high - bar.low for bar in bars]
    local_scales = [
        value + parameters.epsilon for value in ema_series(ranges, parameters.ema_window)
    ]
    shapes = [candlestick_shape(bar, parameters.epsilon) for bar in bars]
    directional_balance = [
        path_averaged_directional_balance(bar, parameters.epsilon) for bar in bars
    ]
    volume_ema = ema_series(
        [max(bar.volume, parameters.epsilon) for bar in bars],
        parameters.ema_window,
    )
    gates = _build_feedback_gates(shapes, parameters)
    l0_values = [
        _build_l0_value(bar.close, local_scales[index], directional_balance[index], gates[index], parameters)
        for index, bar in enumerate(bars)
    ]
    l0_ema = ema_series(l0_values, parameters.ema_window)

    rows: list[TimeframeFeatureRow] = []
    previous_z = 0.0
    for index, bar in enumerate(bars):
        local_scale = local_scales[index]
        z_value = (l0_values[index] - l0_ema[index]) / local_scale
        delta_z = z_value - previous_z
        rho = ranges[index] / local_scale if local_scale > 0 else 0.0
        normalized_volume = (bar.volume / volume_ema[index]) - 1.0 if volume_ema[index] > 0 else 0.0
        rows.append(
            TimeframeFeatureRow(
                timestamp=bar.timestamp,
                close=bar.close,
                shape=shapes[index],
                vector=(
                    z_value,
                    delta_z,
                    directional_balance[index],
                    gates[index],
                    rho,
                    normalized_volume,
                ),
            )
        )
        previous_z = z_value
    return rows


def _build_feedback_gates(
    shapes: list[tuple[float, float, float]],
    parameters: TimeframeParameters,
) -> list[float]:
    gates: list[float] = []
    previous_shape = (0.0, 0.0, 0.0)
    for shape in shapes:
        projection = sum(
            weight * component for weight, component in zip(parameters.gate_weights, previous_shape)
        )
        gates.append(math.tanh(projection + parameters.gate_bias))
        previous_shape = shape
    return gates


def _build_l0_value(
    close: float,
    local_scale: float,
    directional_balance: float,
    gate: float,
    parameters: TimeframeParameters,
) -> float:
    residual = (
        parameters.beta0
        + parameters.beta_v * gate
        + parameters.beta_x * directional_balance
        + parameters.beta_vx * gate * directional_balance
    )
    return close + local_scale * residual
