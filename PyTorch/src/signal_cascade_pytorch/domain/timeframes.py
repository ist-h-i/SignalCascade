from __future__ import annotations

from datetime import datetime, timedelta

from .entities import OHLCVBar

TIMEFRAME_TO_MINUTES = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1_440,
    "1w": 10_080,
}
MAIN_TIMEFRAMES = ("4h", "1d", "1w")
OVERLAY_TIMEFRAMES = ("1h", "30m")
ALL_TIMEFRAMES = ("30m", "1h", "4h", "1d", "1w")
HORIZONS = (1, 2, 3, 6, 12, 18, 30)


def resample_bars(base_bars: list[OHLCVBar], timeframe: str) -> list[OHLCVBar]:
    ordered_bars = sorted(base_bars, key=lambda bar: bar.timestamp)
    if timeframe == "30m":
        return ordered_bars

    aggregated: list[OHLCVBar] = []
    current_bucket_end: datetime | None = None
    bucket_bars: list[OHLCVBar] = []

    for bar in ordered_bars:
        bucket_end = close_bucket_end(bar.timestamp, timeframe)
        if current_bucket_end is None or bucket_end != current_bucket_end:
            if bucket_bars:
                aggregated.append(_merge_bucket(bucket_bars, current_bucket_end))
            current_bucket_end = bucket_end
            bucket_bars = [bar]
        else:
            bucket_bars.append(bar)

    if bucket_bars and current_bucket_end is not None:
        aggregated.append(_merge_bucket(bucket_bars, current_bucket_end))
    return aggregated


def close_bucket_end(timestamp: datetime, timeframe: str) -> datetime:
    effective_time = timestamp - timedelta(microseconds=1)
    return bucket_start(effective_time, timeframe) + timeframe_delta(timeframe)


def timeframe_delta(timeframe: str) -> timedelta:
    return timedelta(minutes=TIMEFRAME_TO_MINUTES[timeframe])


def bucket_start(timestamp: datetime, timeframe: str) -> datetime:
    timestamp = timestamp.replace(second=0, microsecond=0)
    if timeframe in {"30m", "1h", "4h"}:
        bucket_minutes = TIMEFRAME_TO_MINUTES[timeframe]
        total_minutes = timestamp.hour * 60 + timestamp.minute
        floor_minutes = total_minutes - (total_minutes % bucket_minutes)
        hour, minute = divmod(floor_minutes, 60)
        return timestamp.replace(hour=hour, minute=minute)
    if timeframe == "1d":
        return timestamp.replace(hour=0, minute=0)
    if timeframe == "1w":
        monday = timestamp - timedelta(days=timestamp.weekday())
        return monday.replace(hour=0, minute=0)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _merge_bucket(bucket_bars: list[OHLCVBar], timestamp: datetime) -> OHLCVBar:
    return OHLCVBar(
        timestamp=timestamp,
        open=bucket_bars[0].open,
        high=max(bar.high for bar in bucket_bars),
        low=min(bar.low for bar in bucket_bars),
        close=bucket_bars[-1].close,
        volume=sum(bar.volume for bar in bucket_bars),
    )
