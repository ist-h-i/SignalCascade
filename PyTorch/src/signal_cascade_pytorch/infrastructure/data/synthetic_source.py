from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ...domain.entities import OHLCVBar


@dataclass
class SyntheticMarketDataSource:
    bar_count: int
    seed: int = 7

    def load_bars(self) -> list[OHLCVBar]:
        rng = random.Random(self.seed)
        timestamp = datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
        price = 100.0
        drift = 0.0
        bars: list[OHLCVBar] = []

        for index in range(self.bar_count):
            if index % 96 == 0:
                drift = rng.gauss(0.0, 0.0025)

            open_price = price
            innovation = rng.gauss(drift, 0.008)
            close_price = max(1.0, open_price * math.exp(innovation))
            upper_wick = max(open_price, close_price) * abs(rng.gauss(0.0, 0.0035))
            lower_wick = max(open_price, close_price) * abs(rng.gauss(0.0, 0.0035))
            high_price = max(open_price, close_price) + upper_wick
            low_price = max(0.5, min(open_price, close_price) - lower_wick)
            volume = max(
                100.0,
                1_000.0 * (1.0 + abs(innovation) * 80.0 + rng.uniform(-0.15, 0.25)),
            )
            bars.append(
                OHLCVBar(
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )
            )
            price = close_price
            timestamp += timedelta(minutes=30)

        return bars
