from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ...domain.entities import OHLCVBar


@dataclass
class CsvMarketDataSource:
    path: Path

    def load_bars(self) -> list[OHLCVBar]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        bars: list[OHLCVBar] = []
        with self.path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = _parse_timestamp(row["timestamp"])
                volume = float(row["volume"]) if row.get("volume") else 0.0
                bars.append(
                    OHLCVBar(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=volume,
                    )
                )
        return sorted(bars, key=lambda bar: bar.timestamp)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)
