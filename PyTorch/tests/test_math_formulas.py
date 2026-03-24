from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.dataset_service import _main_mae_threshold, _main_move_threshold
from signal_cascade_pytorch.domain.close_anchor import TimeframeParameters, _build_feedback_gates
from signal_cascade_pytorch.application.policy_service import _precision_lower_bound
from signal_cascade_pytorch.domain.candlestick import candlestick_shape, path_averaged_directional_balance
from signal_cascade_pytorch.domain.entities import OHLCVBar
from signal_cascade_pytorch.infrastructure.ml.losses import heteroscedastic_huber_loss


def _bar(*, open_: float, high: float, low: float, close: float) -> OHLCVBar:
    return OHLCVBar(
        timestamp=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class MathFormulaTests(unittest.TestCase):
    def test_candlestick_shape_uses_normalized_range_components(self) -> None:
        upper_shadow, body, lower_shadow = candlestick_shape(
            _bar(open_=2.0, high=5.0, low=1.0, close=4.0)
        )

        self.assertAlmostEqual(upper_shadow, 0.25, places=6)
        self.assertAlmostEqual(body, 0.5, places=6)
        self.assertAlmostEqual(lower_shadow, 0.25, places=6)

    def test_directional_balance_saturates_to_positive_one_at_clip_boundary(self) -> None:
        value = path_averaged_directional_balance(_bar(open_=0.0, high=1.0, low=0.0, close=1.0))
        self.assertAlmostEqual(value, 1.0, places=5)

    def test_directional_balance_saturates_to_negative_one_at_clip_boundary(self) -> None:
        value = path_averaged_directional_balance(_bar(open_=1.0, high=1.0, low=0.0, close=0.0))
        self.assertAlmostEqual(value, -1.0, places=5)

    def test_directional_balance_handles_zero_range_without_nan(self) -> None:
        value = path_averaged_directional_balance(_bar(open_=1.0, high=1.0, low=1.0, close=1.0))
        self.assertFalse(math.isnan(value))
        self.assertEqual(value, 0.0)

    def test_feedback_gate_starts_at_zero_for_initial_shape(self) -> None:
        gates = _build_feedback_gates(
            shapes=[(0.2, 0.3, 0.5)],
            parameters=TimeframeParameters(ema_window=8, gate_weights=(0.6, 0.3, -0.2)),
        )
        self.assertEqual(gates, [0.0])

    def test_feedback_gate_saturates_under_extreme_projection(self) -> None:
        gates = _build_feedback_gates(
            shapes=[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            parameters=TimeframeParameters(ema_window=8, gate_weights=(10.0, 10.0, 10.0)),
        )
        self.assertGreater(gates[1], 0.999)

    def test_heteroscedastic_huber_loss_matches_closed_form(self) -> None:
        loss = heteroscedastic_huber_loss(
            mean=torch.tensor([[0.0]], dtype=torch.float32),
            sigma=torch.tensor([[2.0]], dtype=torch.float32),
            target=torch.tensor([[3.0]], dtype=torch.float32),
            delta=1.0,
        )

        expected = ((0.5 * (1.0**2)) + (1.0 * 2.0)) / (2.0**2) + math.log(2.0)
        self.assertAlmostEqual(float(loss.item()), expected, places=6)

    def test_wilson_lower_bound_matches_reference_value(self) -> None:
        self.assertAlmostEqual(
            _precision_lower_bound(8, 10, 1.96),
            0.49015684672072346,
            places=9,
        )

    def test_wilson_lower_bound_is_zero_when_support_is_zero(self) -> None:
        self.assertEqual(_precision_lower_bound(0, 0, 1.96), 0.0)

    def test_threshold_formulas_include_regime_adjustments(self) -> None:
        config = TrainingConfig(base_cost=0.01, delta_multiplier=1.35, mae_multiplier=0.95)
        regime = {
            "session": "asia",
            "volatility_bin": "high",
            "trend_bin": "trend",
        }

        delta = _main_move_threshold(config, 4, 0.02, regime)
        eta = _main_mae_threshold(config, 4, 0.02, regime)

        expected_delta = (0.01 * math.sqrt(4)) + (1.35 * 1.15 * 0.85 * 1.05 * 0.02 * math.sqrt(4))
        expected_eta = 0.95 * 1.10 * 0.90 * 1.05 * 0.02 * math.sqrt(4)
        self.assertAlmostEqual(delta, expected_delta, places=9)
        self.assertAlmostEqual(eta, expected_eta, places=9)


if __name__ == "__main__":
    unittest.main()
