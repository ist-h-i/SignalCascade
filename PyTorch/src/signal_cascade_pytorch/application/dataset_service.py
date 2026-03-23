from __future__ import annotations

import math
from bisect import bisect_right

from .config import TrainingConfig
from .ports import MarketDataSource
from ..domain.close_anchor import build_close_anchor_features
from ..domain.entities import OHLCVBar, TimeframeFeatureRow, TrainingExample
from ..domain.timeframes import ALL_TIMEFRAMES, MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES, resample_bars

OVERLAY_LABELS = ("hold", "reduce", "full_exit", "hard_exit")


def build_training_examples(
    source: MarketDataSource,
    config: TrainingConfig,
) -> list[TrainingExample]:
    base_bars = sorted(source.load_bars(), key=lambda bar: bar.timestamp)
    if len(base_bars) < 512:
        raise ValueError("At least 512 base 30m bars are required to build a training set.")

    bars_by_timeframe = _build_bars_by_timeframe(base_bars)
    features_by_timeframe = {
        timeframe: build_close_anchor_features(
            bars_by_timeframe[timeframe],
            config.timeframe_parameters[timeframe],
        )
        for timeframe in ALL_TIMEFRAMES
    }
    return _build_examples(features_by_timeframe, config)


def _build_bars_by_timeframe(base_bars: list[OHLCVBar]) -> dict[str, list[OHLCVBar]]:
    return {timeframe: resample_bars(base_bars, timeframe) for timeframe in ALL_TIMEFRAMES}


def _build_examples(
    features_by_timeframe: dict[str, list[TimeframeFeatureRow]],
    config: TrainingConfig,
) -> list[TrainingExample]:
    timestamps = {
        timeframe: [row.timestamp for row in rows]
        for timeframe, rows in features_by_timeframe.items()
    }
    bars_4h = features_by_timeframe["4h"]
    max_horizon = config.max_horizon
    examples: list[TrainingExample] = []

    for idx_4h, anchor_row in enumerate(bars_4h):
        if idx_4h < config.main_windows["4h"] - 1 or idx_4h + max_horizon >= len(bars_4h):
            continue

        main_sequences: dict[str, list[tuple[float, ...]]] = {}
        overlay_sequences: dict[str, list[tuple[float, ...]]] = {}
        main_shape_targets: dict[str, tuple[float, float, float]] = {}

        if not _collect_main_sequences(
            anchor_row.timestamp,
            idx_4h,
            features_by_timeframe,
            timestamps,
            config,
            main_sequences,
            main_shape_targets,
        ):
            continue

        if not _collect_overlay_sequences(
            anchor_row.timestamp,
            features_by_timeframe,
            timestamps,
            config,
            overlay_sequences,
        ):
            continue

        idx_30m = _latest_index(timestamps["30m"], anchor_row.timestamp)
        if idx_30m is None or idx_30m + 8 >= len(features_by_timeframe["30m"]):
            continue

        returns_target = tuple(
            math.log(bars_4h[idx_4h + horizon].close / anchor_row.close)
            for horizon in config.horizons
        )
        direction = 1 if returns_target[0] >= 0 else -1
        volatility = _realized_volatility(features_by_timeframe["30m"], idx_30m)
        overlay_target = _overlay_label(features_by_timeframe["30m"], idx_30m, direction, volatility)

        examples.append(
            TrainingExample(
                anchor_time=anchor_row.timestamp,
                main_sequences=main_sequences,
                overlay_sequences=overlay_sequences,
                main_shape_targets=main_shape_targets,
                returns_target=returns_target,
                overlay_target=overlay_target,
                current_close=anchor_row.close,
            )
        )

    if not examples:
        raise ValueError("No training examples could be constructed from the provided data.")
    return examples


def _collect_main_sequences(
    anchor_time,
    idx_4h: int,
    features_by_timeframe: dict[str, list[TimeframeFeatureRow]],
    timestamps: dict[str, list],
    config: TrainingConfig,
    main_sequences: dict[str, list[tuple[float, ...]]],
    main_shape_targets: dict[str, tuple[float, float, float]],
) -> bool:
    for timeframe in MAIN_TIMEFRAMES:
        index = idx_4h if timeframe == "4h" else _latest_index(timestamps[timeframe], anchor_time)
        if index is None:
            return False
        window = config.main_windows[timeframe]
        if index < window - 1 or index + 1 >= len(features_by_timeframe[timeframe]):
            return False
        rows = features_by_timeframe[timeframe][index - window + 1 : index + 1]
        main_sequences[timeframe] = [row.vector for row in rows]
        main_shape_targets[timeframe] = features_by_timeframe[timeframe][index + 1].shape
    return True


def _collect_overlay_sequences(
    anchor_time,
    features_by_timeframe: dict[str, list[TimeframeFeatureRow]],
    timestamps: dict[str, list],
    config: TrainingConfig,
    overlay_sequences: dict[str, list[tuple[float, ...]]],
) -> bool:
    for timeframe in OVERLAY_TIMEFRAMES:
        index = _latest_index(timestamps[timeframe], anchor_time)
        if index is None:
            return False
        window = config.overlay_windows[timeframe]
        if index < window - 1:
            return False
        rows = features_by_timeframe[timeframe][index - window + 1 : index + 1]
        overlay_sequences[timeframe] = [row.vector for row in rows]
    return True


def _latest_index(timestamps: list, anchor_time):
    index = bisect_right(timestamps, anchor_time) - 1
    if index < 0:
        return None
    return index


def _realized_volatility(rows: list[TimeframeFeatureRow], end_index: int, lookback: int = 48) -> float:
    start = max(1, end_index - lookback + 1)
    returns = [
        abs(math.log(rows[index].close / rows[index - 1].close))
        for index in range(start, end_index + 1)
    ]
    if not returns:
        return 1e-4
    return max(sum(returns) / len(returns), 1e-4)


def _overlay_label(
    rows: list[TimeframeFeatureRow],
    anchor_index: int,
    direction: int,
    volatility: float,
) -> int:
    anchor_close = rows[anchor_index].close
    future_rows = rows[anchor_index + 1 : anchor_index + 9]
    path_returns = [direction * math.log(row.close / anchor_close) for row in future_rows]
    worst_path = min(path_returns)
    final_return = path_returns[-1]
    reduce_cut = -1.25 * volatility
    full_exit_cut = -2.0 * volatility
    hard_exit_cut = -3.0 * volatility

    if worst_path <= hard_exit_cut:
        return 3
    if final_return <= full_exit_cut or worst_path <= full_exit_cut:
        return 2
    if worst_path <= reduce_cut or final_return <= reduce_cut * 0.5:
        return 1
    return 0
