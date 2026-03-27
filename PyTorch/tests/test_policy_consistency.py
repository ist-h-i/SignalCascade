from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.diagnostics_service import build_validation_diagnostics
from signal_cascade_pytorch.domain.entities import TrainingExample
from signal_cascade_pytorch.domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES


def _example(anchor_offset_hours: int = 0) -> TrainingExample:
    main_sequences = {
        timeframe: [(0.01, 0.2, 0.1, 0.05, 0.0, 0.3)] * 4 for timeframe in MAIN_TIMEFRAMES
    }
    overlay_sequences = {
        timeframe: [(0.01, 0.2, 0.1, 0.05, 0.0, 0.3)] * 4 for timeframe in OVERLAY_TIMEFRAMES
    }
    main_shape_targets = {timeframe: (0.2, 0.3, 0.5) for timeframe in MAIN_TIMEFRAMES}
    return TrainingExample(
        anchor_time=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc) + timedelta(hours=anchor_offset_hours),
        main_sequences=main_sequences,
        overlay_sequences=overlay_sequences,
        main_shape_targets=main_shape_targets,
        state_features=(1.0, 0.0, 0.0, 0.1, 0.2, 0.02, 0.03, 0.1, 0.25, 0.4),
        returns_target=(0.02, 0.01),
        long_mae=(0.0, 0.0),
        short_mae=(0.0, 0.0),
        long_mfe=(0.0, 0.0),
        short_mfe=(0.0, 0.0),
        direction_targets=(1, 1),
        direction_weights=(1.0, 1.0),
        direction_thresholds=(0.01, 0.01),
        direction_mae_thresholds=(0.01, 0.01),
        horizon_costs=(0.001, 0.001),
        overlay_target=0,
        current_close=100.0,
        regime_id="asia|low|trend",
        regime_features=(1.0, 0.0, 0.0, 0.1, 0.2),
        realized_volatility=0.02,
        trend_strength=0.25,
    )


class _StaticPolicyModel(torch.nn.Module):
    def forward(self, main_sequences, overlay_sequences, state_features, previous_state=None):
        batch_size = state_features.shape[0]
        return {
            "mu": torch.tensor([[0.03, 0.01]], dtype=torch.float32).repeat(batch_size, 1),
            "sigma": torch.tensor([[0.02, 0.04]], dtype=torch.float32).repeat(batch_size, 1),
            "tradeability_gate": torch.tensor([1.0], dtype=torch.float32).repeat(batch_size),
            "shape_entropy": torch.tensor([0.2], dtype=torch.float32).repeat(batch_size),
            "shape_probs": torch.tensor(
                [[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]],
                dtype=torch.float32,
            ).repeat(batch_size, 1),
            "next_state": torch.zeros(batch_size, 4, dtype=torch.float32),
        }


class PolicyConsistencyTests(unittest.TestCase):
    def test_validation_diagnostics_include_exact_vs_smooth_metrics(self) -> None:
        config = TrainingConfig(horizons=(1, 3))
        diagnostics = build_validation_diagnostics(
            model=_StaticPolicyModel(),
            validation_examples=[_example()],
            config=config,
        )

        summary = diagnostics["summary"]
        self.assertIn("exact_smooth_horizon_agreement", summary)
        self.assertIn("exact_smooth_no_trade_agreement", summary)
        self.assertIn("exact_smooth_position_mae", summary)
        self.assertIn("exact_smooth_utility_regret", summary)
        self.assertAlmostEqual(float(summary["exact_smooth_horizon_agreement"]), 1.0, places=6)

        row = diagnostics["policy_summary"][0]
        self.assertIn("smooth_policy_horizon", row)
        self.assertIn("exact_smooth_position_abs_error", row)
        self.assertIn("exact_smooth_utility_regret", row)

        validation_row = diagnostics["validation_rows"][0]
        self.assertIn("smooth_policy_horizon", validation_row)
        self.assertIn("smooth_position", validation_row)
        self.assertIn("smooth_no_trade_band", validation_row)


if __name__ == "__main__":
    unittest.main()
