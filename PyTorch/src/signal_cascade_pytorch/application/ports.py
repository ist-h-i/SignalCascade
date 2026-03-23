from __future__ import annotations

from typing import Protocol

from ..domain.entities import OHLCVBar


class MarketDataSource(Protocol):
    def load_bars(self) -> list[OHLCVBar]:
        ...
