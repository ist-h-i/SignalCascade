from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.inference_service import predict_from_example, predict_latest
from signal_cascade_pytorch.application.training_service import evaluate_model
from signal_cascade_pytorch.domain.entities import TrainingExample
from signal_cascade_pytorch.domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES


def _example(index: int, regime_id: str) -> TrainingExample:
    main_sequences = {
        timeframe: [(0.01, 0.2, 0.1, 0.05, 0.0, 0.3)] * 4 for timeframe in MAIN_TIMEFRAMES
    }
    overlay_sequences = {
        timeframe: [(0.01, 0.2, 0.1, 0.05, 0.0, 0.3)] * 4 for timeframe in OVERLAY_TIMEFRAMES
    }
    main_shape_targets = {timeframe: (0.2, 0.3, 0.5) for timeframe in MAIN_TIMEFRAMES}
    return TrainingExample(
        anchor_time=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc) + timedelta(hours=4 * index),
        main_sequences=main_sequences,
        overlay_sequences=overlay_sequences,
        main_shape_targets=main_shape_targets,
        state_features=(1.0, 0.0, 0.0, 0.1, 0.25, 0.02, 0.03, 0.1, 0.25, 0.4),
        returns_target=(0.02,),
        long_mae=(0.0,),
        short_mae=(0.0,),
        long_mfe=(0.0,),
        short_mfe=(0.0,),
        direction_targets=(1,),
        direction_weights=(1.0,),
        direction_thresholds=(0.01,),
        direction_mae_thresholds=(0.01,),
        horizon_costs=(0.001,),
        overlay_target=0,
        current_close=100.0,
        regime_id=regime_id,
        regime_features=(1.0, 0.0, 0.0, 0.1, 0.25),
        realized_volatility=0.02,
        trend_strength=0.25,
    )


class _StaticPolicyModel(torch.nn.Module):
    def forward(self, main_sequences, overlay_sequences, state_features, previous_state=None):
        batch_size = state_features.shape[0]
        shape_posterior = torch.tensor(
            [[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]],
            dtype=torch.float32,
        ).repeat(batch_size, 1)
        memory_state = torch.zeros(batch_size, 4, dtype=torch.float32)
        return {
            "mu": torch.tensor([[0.02]], dtype=torch.float32).repeat(batch_size, 1),
            "sigma": torch.tensor([[0.02]], dtype=torch.float32).repeat(batch_size, 1),
            "tradeability_gate": torch.tensor([1.0], dtype=torch.float32).repeat(batch_size),
            "shape_entropy": torch.tensor([0.2], dtype=torch.float32).repeat(batch_size),
            "shape_posterior": shape_posterior,
            "shape_probs": shape_posterior,
            "memory_state": memory_state,
            "next_state": memory_state,
        }


class StatefulEvaluationTests(unittest.TestCase):
    def test_default_primary_state_reset_mode_preserves_carry_on(self) -> None:
        config = TrainingConfig()

        self.assertEqual(config.training_state_reset_mode, "carry_on")
        self.assertEqual(config.evaluation_state_reset_mode, "carry_on")
        self.assertEqual(config.diagnostic_state_reset_modes[0], "carry_on")

    def test_state_reset_modes_change_turnover_semantics(self) -> None:
        config = TrainingConfig(horizons=(1,))
        examples = [
            _example(0, "asia|low|trend"),
            _example(1, "ny|low|trend"),
        ]
        model = _StaticPolicyModel()

        carry_on = evaluate_model(model, examples, config, state_reset_mode="carry_on")
        reset_each_example = evaluate_model(
            model,
            examples,
            config,
            state_reset_mode="reset_each_example",
        )
        reset_each_session = evaluate_model(
            model,
            examples,
            config,
            state_reset_mode="reset_each_session_or_window",
        )

        self.assertEqual(carry_on["state_reset_mode"], "carry_on")
        self.assertEqual(carry_on["state_reset_boundary_spec_version"], 1)
        self.assertEqual(carry_on["state_reset_count"], 1)
        self.assertEqual(carry_on["session_count"], 2)
        self.assertEqual(carry_on["window_count"], 1)
        self.assertLess(float(carry_on["turnover"]), float(reset_each_example["turnover"]))
        self.assertAlmostEqual(
            float(reset_each_example["turnover"]),
            float(reset_each_session["turnover"]),
            places=6,
        )

    def test_predict_latest_replays_carry_on_state_before_latest_example(self) -> None:
        class _CarryAwareModel(torch.nn.Module):
            def forward(self, main_sequences, overlay_sequences, state_features, previous_state=None):
                batch_size = state_features.shape[0]
                prior = 0.0 if previous_state is None else float(previous_state[0, 0].item())
                forecast_mu = torch.tensor([[0.01 + prior]], dtype=torch.float32).repeat(batch_size, 1)
                policy_mu = torch.tensor([[0.02]], dtype=torch.float32).repeat(batch_size, 1)
                sigma = torch.tensor([[0.02]], dtype=torch.float32).repeat(batch_size, 1)
                shape_posterior = torch.tensor(
                    [[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]],
                    dtype=torch.float32,
                ).repeat(batch_size, 1)
                memory_state = torch.tensor([[0.5]], dtype=torch.float32).repeat(batch_size, 1)
                return {
                    "mu": forecast_mu,
                    "sigma": sigma,
                    "forecast_mu": forecast_mu,
                    "forecast_sigma": sigma,
                    "policy_mu": policy_mu,
                    "policy_sigma": sigma,
                    "tradeability_gate": torch.tensor([1.0], dtype=torch.float32).repeat(batch_size),
                    "shape_entropy": torch.tensor([0.2], dtype=torch.float32).repeat(batch_size),
                    "shape_posterior": shape_posterior,
                    "shape_probs": shape_posterior,
                    "memory_state": memory_state,
                    "next_state": memory_state,
                }

        config = TrainingConfig(horizons=(1,))
        examples = [_example(0, "asia|low|trend"), _example(1, "asia|low|trend")]
        prediction = predict_latest(_CarryAwareModel(), examples, config)

        self.assertAlmostEqual(prediction.expected_log_returns["1"], 0.51, places=6)
        self.assertEqual(prediction.inference_context_mode, "carry_on")

    def test_predict_latest_can_target_latest_inference_anchor(self) -> None:
        class _CarryAwareModel(torch.nn.Module):
            def forward(self, main_sequences, overlay_sequences, state_features, previous_state=None):
                batch_size = state_features.shape[0]
                prior = 0.0 if previous_state is None else float(previous_state[0, 0].item())
                forecast_mu = torch.tensor([[0.01 + prior]], dtype=torch.float32).repeat(batch_size, 1)
                policy_mu = torch.tensor([[0.02]], dtype=torch.float32).repeat(batch_size, 1)
                sigma = torch.tensor([[0.02]], dtype=torch.float32).repeat(batch_size, 1)
                shape_posterior = torch.tensor(
                    [[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]],
                    dtype=torch.float32,
                ).repeat(batch_size, 1)
                memory_state = torch.tensor([[0.5]], dtype=torch.float32).repeat(batch_size, 1)
                return {
                    "mu": forecast_mu,
                    "sigma": sigma,
                    "forecast_mu": forecast_mu,
                    "forecast_sigma": sigma,
                    "policy_mu": policy_mu,
                    "policy_sigma": sigma,
                    "tradeability_gate": torch.tensor([1.0], dtype=torch.float32).repeat(batch_size),
                    "shape_entropy": torch.tensor([0.2], dtype=torch.float32).repeat(batch_size),
                    "shape_posterior": shape_posterior,
                    "shape_probs": shape_posterior,
                    "memory_state": memory_state,
                    "next_state": memory_state,
                }

        config = TrainingConfig(horizons=(1,))
        examples = [_example(0, "asia|low|trend"), _example(1, "asia|low|trend")]
        latest_example = _example(2, "asia|low|trend")
        prediction = predict_latest(
            _CarryAwareModel(),
            examples,
            config,
            latest_example=latest_example,
        )

        self.assertEqual(prediction.anchor_time, latest_example.anchor_time.isoformat())
        self.assertAlmostEqual(prediction.expected_log_returns["1"], 0.51, places=6)
        self.assertEqual(prediction.inference_context_mode, "carry_on")

    def test_predict_from_example_marks_stateless_context_by_default(self) -> None:
        config = TrainingConfig(horizons=(1,))
        prediction = predict_from_example(_StaticPolicyModel(), _example(0, "asia|low|trend"), config)

        self.assertEqual(prediction.inference_context_mode, "stateless")


if __name__ == "__main__":
    unittest.main()
