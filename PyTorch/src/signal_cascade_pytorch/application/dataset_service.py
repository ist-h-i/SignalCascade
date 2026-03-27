from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

from .config import TrainingConfig
from .ports import MarketDataSource
from ..domain.close_anchor import build_close_anchor_features
from ..domain.entities import OHLCVBar, TimeframeFeatureRow, TrainingExample
from ..domain.timeframes import ALL_TIMEFRAMES, MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES, resample_bars

OVERLAY_LABELS = ("reduce", "hold")


def build_training_examples(
    source: MarketDataSource,
    config: TrainingConfig,
) -> list[TrainingExample]:
    base_bars = sorted(source.load_bars(), key=lambda bar: bar.timestamp)
    return build_training_examples_from_bars(base_bars, config)


def build_latest_inference_example(
    source: MarketDataSource,
    config: TrainingConfig,
) -> TrainingExample:
    base_bars = sorted(source.load_bars(), key=lambda bar: bar.timestamp)
    return build_latest_inference_example_from_bars(base_bars, config)


def build_training_examples_from_bars(
    base_bars: list[OHLCVBar],
    config: TrainingConfig,
) -> list[TrainingExample]:
    _validate_base_bars(base_bars, "training set")
    features_by_timeframe = _build_features_by_timeframe(base_bars, config)
    return _build_examples(features_by_timeframe, config)


def build_latest_inference_example_from_bars(
    base_bars: list[OHLCVBar],
    config: TrainingConfig,
) -> TrainingExample:
    _validate_base_bars(base_bars, "inference example")
    features_by_timeframe = _build_features_by_timeframe(base_bars, config)
    return _build_latest_example(features_by_timeframe, config)


def trim_base_bars_for_training(
    base_bars: list[OHLCVBar],
    config: TrainingConfig,
    min_examples: int = 2,
) -> list[OHLCVBar]:
    _validate_base_bars(base_bars, "training set")

    def is_valid(candidate: list[OHLCVBar]) -> bool:
        return len(build_training_examples_from_bars(candidate, config)) >= min_examples

    return _trim_base_bars(base_bars, is_valid)


def trim_base_bars_for_latest_inference(
    base_bars: list[OHLCVBar],
    config: TrainingConfig,
) -> list[OHLCVBar]:
    _validate_base_bars(base_bars, "inference example")

    def is_valid(candidate: list[OHLCVBar]) -> bool:
        build_latest_inference_example_from_bars(candidate, config)
        return True

    return _trim_base_bars(base_bars, is_valid)


def limit_base_bars_to_lookback_days(
    base_bars: list[OHLCVBar],
    lookback_days: int | None,
) -> list[OHLCVBar]:
    if lookback_days is None or lookback_days <= 0 or not base_bars:
        return list(base_bars)

    ordered = sorted(base_bars, key=lambda bar: bar.timestamp)
    cutoff = ordered[-1].timestamp - timedelta(days=int(lookback_days))
    limited = [bar for bar in ordered if bar.timestamp >= cutoff]
    return limited or ordered


def _trim_base_bars(
    base_bars: list[OHLCVBar],
    is_valid,
) -> list[OHLCVBar]:
    low = 1
    high = len(base_bars)
    best = list(base_bars)

    while low <= high:
        middle = (low + high) // 2
        candidate = base_bars[-middle:]
        try:
            if not is_valid(candidate):
                raise ValueError("Candidate did not satisfy the requested dataset constraints.")
        except ValueError:
            low = middle + 1
            continue
        best = candidate
        high = middle - 1

    return best


def _validate_base_bars(base_bars: list[OHLCVBar], target_name: str) -> None:
    if len(base_bars) < 512:
        raise ValueError(f"At least 512 base 30m bars are required to build a {target_name}.")


def _build_features_by_timeframe(
    base_bars: list[OHLCVBar],
    config: TrainingConfig,
) -> dict[str, list[TimeframeFeatureRow]]:
    bars_by_timeframe = _build_bars_by_timeframe(base_bars)
    return {
        timeframe: build_close_anchor_features(
            bars_by_timeframe[timeframe],
            config.timeframe_parameters[timeframe],
        )
        for timeframe in ALL_TIMEFRAMES
    }


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
    rows_4h = features_by_timeframe["4h"]
    rows_30m = features_by_timeframe["30m"]
    examples: list[TrainingExample] = []

    for idx_4h, anchor_row in enumerate(rows_4h):
        if idx_4h < config.main_windows["4h"] - 1 or idx_4h + config.max_horizon >= len(rows_4h):
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
        if idx_30m is None or idx_30m + 8 >= len(rows_30m):
            continue

        realized_volatility = _realized_volatility(rows_30m, idx_30m, lookback=48)
        baseline_volatility = _realized_volatility(rows_30m, idx_30m, lookback=192)
        trend_strength = abs(anchor_row.vector[5]) + (0.5 * abs(anchor_row.vector[1]))
        regime = _build_regime(anchor_row.timestamp, realized_volatility, baseline_volatility, trend_strength)
        state_features = _build_state_features(
            rows_30m=rows_30m,
            index_30m=idx_30m,
            realized_volatility=realized_volatility,
            baseline_volatility=baseline_volatility,
            trend_strength=trend_strength,
            regime=regime,
        )

        returns_target: list[float] = []
        long_mae_values: list[float] = []
        short_mae_values: list[float] = []
        long_mfe_values: list[float] = []
        short_mfe_values: list[float] = []
        direction_targets: list[int] = []
        direction_weights: list[float] = []
        direction_thresholds: list[float] = []
        direction_mae_thresholds: list[float] = []
        horizon_costs: list[float] = []

        for horizon in config.horizons:
            future_rows = rows_4h[idx_4h + 1 : idx_4h + horizon + 1]
            if len(future_rows) != horizon:
                break
            path_returns = [
                math.log(row.close / anchor_row.close)
                for row in future_rows
            ]
            target_return = path_returns[-1]
            long_mae = max(0.0, max((-value) for value in path_returns))
            short_mae = max(0.0, max(path_returns))
            long_mfe = max(0.0, max(path_returns))
            short_mfe = max(0.0, max((-value) for value in path_returns))
            cost = config.cost_for_horizon(horizon)
            delta = _main_move_threshold(config, horizon, realized_volatility, regime)
            eta = _main_mae_threshold(config, horizon, realized_volatility, regime)
            direction_target = _direction_label(
                target_return=target_return,
                long_mae=long_mae,
                short_mae=short_mae,
                delta=delta,
                eta=eta,
            )
            weight = _direction_weight(
                config=config,
                direction_target=direction_target,
                target_return=target_return,
                cost=cost,
                volatility=realized_volatility,
                horizon=horizon,
                long_mfe=long_mfe,
                long_mae=long_mae,
                short_mfe=short_mfe,
                short_mae=short_mae,
            )

            returns_target.append(target_return)
            long_mae_values.append(long_mae)
            short_mae_values.append(short_mae)
            long_mfe_values.append(long_mfe)
            short_mfe_values.append(short_mfe)
            direction_targets.append(direction_target)
            direction_weights.append(weight)
            direction_thresholds.append(delta)
            direction_mae_thresholds.append(eta)
            horizon_costs.append(cost)

        if len(returns_target) != len(config.horizons):
            continue

        overlay_target = _overlay_label(
            rows=rows_30m,
            anchor_index=idx_30m,
            direction_targets=direction_targets,
            direction_weights=direction_weights,
            volatility=realized_volatility,
            config=config,
        )

        examples.append(
            TrainingExample(
                anchor_time=anchor_row.timestamp,
                main_sequences=main_sequences,
                overlay_sequences=overlay_sequences,
                main_shape_targets=main_shape_targets,
                state_features=state_features,
                returns_target=tuple(returns_target),
                long_mae=tuple(long_mae_values),
                short_mae=tuple(short_mae_values),
                long_mfe=tuple(long_mfe_values),
                short_mfe=tuple(short_mfe_values),
                direction_targets=tuple(direction_targets),
                direction_weights=tuple(direction_weights),
                direction_thresholds=tuple(direction_thresholds),
                direction_mae_thresholds=tuple(direction_mae_thresholds),
                horizon_costs=tuple(horizon_costs),
                overlay_target=overlay_target,
                current_close=anchor_row.close,
                regime_id=regime["id"],
                regime_features=regime["features"],
                realized_volatility=realized_volatility,
                trend_strength=trend_strength,
            )
        )

    if not examples:
        raise ValueError("No training examples could be constructed from the provided data.")
    return examples


def _build_latest_example(
    features_by_timeframe: dict[str, list[TimeframeFeatureRow]],
    config: TrainingConfig,
) -> TrainingExample:
    timestamps = {
        timeframe: [row.timestamp for row in rows]
        for timeframe, rows in features_by_timeframe.items()
    }
    rows_4h = features_by_timeframe["4h"]
    rows_30m = features_by_timeframe["30m"]

    for idx_4h in range(len(rows_4h) - 1, -1, -1):
        anchor_row = rows_4h[idx_4h]
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
            require_future_target=False,
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
        if idx_30m is None:
            continue

        realized_volatility = _realized_volatility(rows_30m, idx_30m, lookback=48)
        baseline_volatility = _realized_volatility(rows_30m, idx_30m, lookback=192)
        trend_strength = abs(anchor_row.vector[5]) + (0.5 * abs(anchor_row.vector[1]))
        regime = _build_regime(anchor_row.timestamp, realized_volatility, baseline_volatility, trend_strength)
        state_features = _build_state_features(
            rows_30m=rows_30m,
            index_30m=idx_30m,
            realized_volatility=realized_volatility,
            baseline_volatility=baseline_volatility,
            trend_strength=trend_strength,
            regime=regime,
        )

        return TrainingExample(
            anchor_time=anchor_row.timestamp,
            main_sequences=main_sequences,
            overlay_sequences=overlay_sequences,
            main_shape_targets=main_shape_targets,
            state_features=state_features,
            returns_target=tuple(0.0 for _ in config.horizons),
            long_mae=tuple(0.0 for _ in config.horizons),
            short_mae=tuple(0.0 for _ in config.horizons),
            long_mfe=tuple(0.0 for _ in config.horizons),
            short_mfe=tuple(0.0 for _ in config.horizons),
            direction_targets=tuple(0 for _ in config.horizons),
            direction_weights=tuple(1.0 for _ in config.horizons),
            direction_thresholds=tuple(
                _main_move_threshold(config, horizon, realized_volatility, regime)
                for horizon in config.horizons
            ),
            direction_mae_thresholds=tuple(
                _main_mae_threshold(config, horizon, realized_volatility, regime)
                for horizon in config.horizons
            ),
            horizon_costs=tuple(config.cost_for_horizon(horizon) for horizon in config.horizons),
            overlay_target=0,
            current_close=anchor_row.close,
            regime_id=regime["id"],
            regime_features=regime["features"],
            realized_volatility=realized_volatility,
            trend_strength=trend_strength,
        )

    raise ValueError("No inference example could be constructed from the provided data.")


def _collect_main_sequences(
    anchor_time,
    idx_4h: int,
    features_by_timeframe: dict[str, list[TimeframeFeatureRow]],
    timestamps: dict[str, list],
    config: TrainingConfig,
    main_sequences: dict[str, list[tuple[float, ...]]],
    main_shape_targets: dict[str, tuple[float, float, float]],
    require_future_target: bool = True,
) -> bool:
    for timeframe in MAIN_TIMEFRAMES:
        index = idx_4h if timeframe == "4h" else _latest_index(timestamps[timeframe], anchor_time)
        if index is None:
            return False
        window = config.main_windows[timeframe]
        if index < window - 1:
            return False
        if require_future_target and index + 1 >= len(features_by_timeframe[timeframe]):
            return False
        rows = features_by_timeframe[timeframe][index - window + 1 : index + 1]
        main_sequences[timeframe] = [row.vector for row in rows]
        shape_index = min(index + 1, len(features_by_timeframe[timeframe]) - 1)
        main_shape_targets[timeframe] = features_by_timeframe[timeframe][shape_index].shape
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


def _latest_index(timestamps: list[datetime], anchor_time: datetime) -> int | None:
    index = bisect_right(timestamps, anchor_time) - 1
    if index < 0:
        return None
    return index


def _realized_volatility(
    rows: list[TimeframeFeatureRow],
    end_index: int,
    lookback: int,
) -> float:
    start = max(1, end_index - lookback + 1)
    returns = [
        abs(math.log(rows[index].close / rows[index - 1].close))
        for index in range(start, end_index + 1)
    ]
    if not returns:
        return 1e-4
    return max(sum(returns) / len(returns), 1e-4)


def _build_regime(
    anchor_time: datetime,
    volatility: float,
    baseline_volatility: float,
    trend_strength: float,
) -> dict[str, object]:
    session_name = _session_name(anchor_time)
    volatility_ratio = volatility / max(baseline_volatility, 1e-6)
    volatility_bin = "high" if volatility_ratio >= 1.1 else "low"
    trend_bin = "trend" if trend_strength >= 0.45 else "range"
    clipped_volatility_ratio = max(-1.0, min(1.0, volatility_ratio - 1.0))

    return {
        "id": f"{session_name}|{volatility_bin}|{trend_bin}",
        "session": session_name,
        "volatility_bin": volatility_bin,
        "trend_bin": trend_bin,
        "features": (
            1.0 if session_name == "asia" else 0.0,
            1.0 if session_name == "london" else 0.0,
            1.0 if session_name == "ny" else 0.0,
            clipped_volatility_ratio,
            trend_strength,
        ),
    }


def _build_state_features(
    rows_30m: list[TimeframeFeatureRow],
    index_30m: int,
    realized_volatility: float,
    baseline_volatility: float,
    trend_strength: float,
    regime: dict[str, object],
) -> tuple[float, ...]:
    anchor_vector = rows_30m[index_30m].vector
    volatility_ratio = realized_volatility / max(baseline_volatility, 1e-6)
    clipped_volatility_ratio = max(-2.0, min(2.0, volatility_ratio - 1.0))
    return (
        *tuple(float(value) for value in regime["features"]),
        float(realized_volatility),
        float(baseline_volatility),
        float(clipped_volatility_ratio),
        float(trend_strength),
        float(anchor_vector[4]),
        float(anchor_vector[5]),
    )


def _session_name(anchor_time: datetime) -> str:
    current_time = anchor_time.astimezone(timezone.utc) if anchor_time.tzinfo else anchor_time
    hour = current_time.hour
    if hour < 8:
        return "asia"
    if hour < 16:
        return "london"
    return "ny"


def _main_move_threshold(
    config: TrainingConfig,
    horizon: int,
    volatility: float,
    regime: dict[str, object],
) -> float:
    multiplier = config.delta_multiplier
    if regime["volatility_bin"] == "high":
        multiplier *= 1.15
    else:
        multiplier *= 0.9
    if regime["trend_bin"] == "trend":
        multiplier *= 0.85
    else:
        multiplier *= 1.05
    if regime["session"] == "asia":
        multiplier *= 1.05
    return config.cost_for_horizon(horizon) + (multiplier * volatility * math.sqrt(horizon))


def _main_mae_threshold(
    config: TrainingConfig,
    horizon: int,
    volatility: float,
    regime: dict[str, object],
) -> float:
    multiplier = config.mae_multiplier
    if regime["volatility_bin"] == "high":
        multiplier *= 1.1
    else:
        multiplier *= 0.95
    if regime["trend_bin"] == "trend":
        multiplier *= 0.9
    else:
        multiplier *= 1.05
    if regime["session"] == "asia":
        multiplier *= 1.05
    return multiplier * volatility * math.sqrt(horizon)


def _direction_label(
    target_return: float,
    long_mae: float,
    short_mae: float,
    delta: float,
    eta: float,
) -> int:
    if target_return >= delta and long_mae <= eta:
        return 1
    if target_return <= -delta and short_mae <= eta:
        return -1
    return 0


def _direction_weight(
    config: TrainingConfig,
    direction_target: int,
    target_return: float,
    cost: float,
    volatility: float,
    horizon: int,
    long_mfe: float,
    long_mae: float,
    short_mfe: float,
    short_mae: float,
) -> float:
    denominator = (volatility * math.sqrt(horizon)) + 1e-6
    excess_move = max(abs(target_return) - cost, 0.0) / denominator
    weight = 1.0 + (config.clean_weight_return_scale * excess_move)
    if direction_target != 0:
        weight += config.clean_weight_bonus

    if direction_target > 0:
        ratio = long_mfe / max(long_mae, 1e-6)
        weight += config.clean_weight_ratio_scale * min(ratio, 4.0)
    elif direction_target < 0:
        ratio = short_mfe / max(short_mae, 1e-6)
        weight += config.clean_weight_ratio_scale * min(ratio, 4.0)

    return max(1.0, min(weight, 6.0))


def _overlay_label(
    rows: list[TimeframeFeatureRow],
    anchor_index: int,
    direction_targets: list[int],
    direction_weights: list[float],
    volatility: float,
    config: TrainingConfig,
) -> int:
    direction = _primary_direction(direction_targets, direction_weights)
    if direction == 0:
        return 0

    future_rows = rows[anchor_index + 1 : anchor_index + 9]
    if len(future_rows) < 8:
        return 0

    anchor_close = rows[anchor_index].close
    path_returns = [direction * math.log(row.close / anchor_close) for row in future_rows]
    final_return = path_returns[-1]
    adverse_excursion = abs(min(min(path_returns), 0.0))
    delta = config.cost_for_horizon(1) + (
        config.overlay_delta_multiplier * volatility * math.sqrt(len(future_rows))
    )
    eta = config.overlay_mae_multiplier * volatility * math.sqrt(len(future_rows))
    return int(final_return >= delta and adverse_excursion <= eta)


def _primary_direction(
    direction_targets: list[int],
    direction_weights: list[float],
) -> int:
    best_score = 0.0
    best_direction = 0
    for direction_target, weight in zip(direction_targets, direction_weights):
        if direction_target == 0:
            continue
        if weight > best_score:
            best_score = weight
            best_direction = direction_target
    return best_direction
