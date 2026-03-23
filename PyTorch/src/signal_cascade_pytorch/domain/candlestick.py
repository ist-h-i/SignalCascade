from __future__ import annotations

from typing import Sequence

from .entities import OHLCVBar, ShapeVector


def candlestick_shape(bar: OHLCVBar, epsilon: float = 1e-6) -> ShapeVector:
    candle_range = (bar.high - bar.low) + epsilon
    upper_shadow = (bar.high - max(bar.open, bar.close)) / candle_range
    body = (bar.close - bar.open) / candle_range
    lower_shadow = (min(bar.open, bar.close) - bar.low) / candle_range
    return (upper_shadow, body, lower_shadow)


def path_averaged_directional_balance(bar: OHLCVBar, epsilon: float = 1e-6) -> float:
    numerator = bar.close - bar.open
    denominator = (2.0 * (bar.high - bar.low)) + epsilon
    x_value = numerator / denominator
    clipped = max(-0.5, min(0.5, x_value))
    balance = clipped / (1.0 - (clipped * clipped) + epsilon)
    return max(-1.0, min(1.0, 1.5 * balance))


def ema_series(values: Sequence[float], span: int) -> list[float]:
    if span <= 0:
        raise ValueError("EMA span must be positive.")
    alpha = 2.0 / (span + 1.0)
    outputs: list[float] = []
    previous = 0.0
    initialized = False

    for value in values:
        if not initialized:
            previous = value
            initialized = True
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        outputs.append(previous)
    return outputs
