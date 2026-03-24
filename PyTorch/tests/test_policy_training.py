from __future__ import annotations

from dataclasses import asdict
import unittest
from datetime import datetime, timezone

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.policy_service import (
    _augment_snapshot,
    _calibrate_threshold,
    _lookup_selection_threshold,
    build_replay_selection_policy,
    build_default_policy,
)
from signal_cascade_pytorch.application.training_service import examples_to_batch, restore_return_units
from signal_cascade_pytorch.domain.entities import PredictionResult, TrainingExample
from signal_cascade_pytorch.domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES


def _example(
    *,
    returns_target: tuple[float, ...],
    direction_targets: tuple[int, ...],
    direction_thresholds: tuple[float, ...],
    realized_volatility: float = 0.02,
) -> TrainingExample:
    main_sequences = {timeframe: [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] for timeframe in MAIN_TIMEFRAMES}
    overlay_sequences = {
        timeframe: [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] for timeframe in OVERLAY_TIMEFRAMES
    }
    main_shape_targets = {timeframe: (0.0, 0.0, 0.0) for timeframe in MAIN_TIMEFRAMES}
    horizon_costs = tuple(0.001 for _ in returns_target)
    direction_weights = tuple(1.0 for _ in returns_target)
    return TrainingExample(
        anchor_time=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc),
        main_sequences=main_sequences,
        overlay_sequences=overlay_sequences,
        main_shape_targets=main_shape_targets,
        returns_target=returns_target,
        long_mae=tuple(0.0 for _ in returns_target),
        short_mae=tuple(0.0 for _ in returns_target),
        long_mfe=tuple(0.0 for _ in returns_target),
        short_mfe=tuple(0.0 for _ in returns_target),
        direction_targets=direction_targets,
        direction_weights=direction_weights,
        direction_thresholds=direction_thresholds,
        direction_mae_thresholds=tuple(0.0 for _ in returns_target),
        horizon_costs=horizon_costs,
        overlay_target=0,
        current_close=100.0,
        regime_id="asia|low|range",
        regime_features=(1.0, 0.0, 0.0, 0.0, 0.0),
        realized_volatility=realized_volatility,
        trend_strength=0.1,
    )


class PolicyAndTrainingTests(unittest.TestCase):
    def test_sign_aware_score_rejects_misaligned_mean(self) -> None:
        config = TrainingConfig(horizons=(1, 2))
        policy = build_default_policy(config)
        policy["selection_thresholds"]["global"] = 0.5
        snapshot = {
            "anchor_time": "2026-03-24T00:00:00+00:00",
            "regime_id": "asia|low|range",
            "regime_features": [1.0, 0.0, 0.0, 0.0, 0.0],
            "realized_volatility": 0.02,
            "trend_strength": 0.1,
            "overlay_target": 0,
            "overlay_probability": 0.4,
            "horizons": [
                {
                    "horizon": 1,
                    "mean": -0.03,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 1,
                    "true_return": -0.01,
                    "true_direction": -1,
                    "direction_probabilities": [0.1, 0.1, 0.8],
                },
                {
                    "horizon": 2,
                    "mean": 0.02,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 1,
                    "true_return": 0.01,
                    "true_direction": 1,
                    "direction_probabilities": [0.2, 0.1, 0.7],
                },
            ],
        }

        decision = _augment_snapshot(snapshot, policy, config)

        self.assertEqual(decision["selected_horizon"], 2)
        self.assertEqual(decision["proposed_horizon"], 2)
        first_row = next(row for row in decision["horizon_rows"] if int(row["horizon"]) == 1)
        self.assertEqual(float(first_row["actionable_edge"]), 0.0)
        self.assertEqual(float(first_row["score"]), 0.0)

    def test_lookup_selection_threshold_uses_global_scope(self) -> None:
        config = TrainingConfig(horizons=(1, 2))
        policy = build_default_policy(config)
        policy["selection_thresholds"]["global"] = 0.61
        policy["selection_thresholds"]["by_horizon"]["1"] = 0.95

        threshold = _lookup_selection_threshold(policy, "asia|low|range", 1, config)

        self.assertAlmostEqual(float(threshold), 0.61)

    def test_calibrate_threshold_reports_best_lcb_metadata(self) -> None:
        config = TrainingConfig(selection_min_support=2, precision_target=0.95)
        records = [
            {"score": 0.91, "target": 1, "horizon": 1, "regime_id": "r"},
            {"score": 0.89, "target": 1, "horizon": 1, "regime_id": "r"},
            {"score": 0.72, "target": 0, "horizon": 1, "regime_id": "r"},
            {"score": 0.68, "target": 1, "horizon": 1, "regime_id": "r"},
        ]

        bundle = _calibrate_threshold(records, config, anchor_count=len(records))
        meta = dict(bundle["meta"])

        self.assertIn("best_selection_lcb", meta)
        self.assertIn("support_at_best_lcb", meta)
        self.assertIn("precision_at_best_lcb", meta)
        self.assertIn("tau_at_best_lcb", meta)
        self.assertGreaterEqual(float(meta["best_selection_lcb"]), 0.0)
        self.assertTrue(bundle["scan"])

    def test_return_standardization_round_trips_to_raw_units(self) -> None:
        config = TrainingConfig(horizons=(1, 4), standardized_return_clip=6.0)
        example = _example(
            returns_target=(0.01, 0.02),
            direction_targets=(1, -1),
            direction_thresholds=(0.015, 0.03),
            realized_volatility=0.02,
        )

        batch = examples_to_batch([example], config)
        restored_mean, restored_sigma = restore_return_units(
            batch["returns"],
            torch.ones_like(batch["returns"]),
            batch["return_scale"],
        )

        self.assertAlmostEqual(float(restored_mean[0, 0]), 0.01, places=6)
        self.assertAlmostEqual(float(restored_mean[0, 1]), 0.02, places=6)
        self.assertGreater(float(restored_sigma[0, 0]), 0.0)

    def test_accept_reject_reason_reports_missing_threshold(self) -> None:
        config = TrainingConfig(horizons=(1,))
        policy = build_default_policy(config)
        snapshot = {
            "anchor_time": "2026-03-24T00:00:00+00:00",
            "regime_id": "asia|low|range",
            "regime_features": [1.0, 0.0, 0.0, 0.0, 0.0],
            "realized_volatility": 0.02,
            "trend_strength": 0.1,
            "overlay_target": 0,
            "overlay_probability": 0.8,
            "horizons": [
                {
                    "horizon": 1,
                    "mean": 0.03,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 1,
                    "true_return": 0.03,
                    "true_direction": 1,
                    "direction_probabilities": [0.1, 0.1, 0.8],
                }
            ],
        }

        decision = _augment_snapshot(snapshot, policy, config)

        self.assertEqual(decision["accept_reject_reason"], "selection_threshold_missing")
        self.assertTrue(decision["reject_flags"]["selection_threshold_missing"])

    def test_accept_reject_reason_reports_selection_score_below_threshold(self) -> None:
        config = TrainingConfig(horizons=(1,))
        policy = build_default_policy(config)
        policy["selection_thresholds"]["global"] = 0.6
        snapshot = {
            "anchor_time": "2026-03-24T00:00:00+00:00",
            "regime_id": "asia|low|range",
            "regime_features": [1.0, 0.0, 0.0, 0.0, 0.0],
            "realized_volatility": 0.02,
            "trend_strength": 0.1,
            "overlay_target": 0,
            "overlay_probability": 0.8,
            "horizons": [
                {
                    "horizon": 1,
                    "mean": 0.03,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 1,
                    "true_return": 0.03,
                    "true_direction": 1,
                    "direction_probabilities": [0.1, 0.1, 0.8],
                }
            ],
        }

        decision = _augment_snapshot(snapshot, policy, config)

        self.assertEqual(decision["accept_reject_reason"], "selection_score_below_threshold")
        self.assertTrue(decision["reject_flags"]["selection_score_below_threshold"])
        self.assertTrue(decision["reject_flags"]["selector_probability_below_threshold"])
        self.assertEqual(decision["proposed_horizon"], 1)
        self.assertIsNone(decision["accepted_horizon"])
        self.assertEqual(decision["threshold_status"], "ready")
        self.assertEqual(decision["threshold_origin"], "stored_policy")
        self.assertEqual(decision["stored_threshold_compatibility"], "compatible")

    def test_allow_no_candidate_returns_no_candidate_reject_reason(self) -> None:
        config = TrainingConfig(horizons=(1, 2), allow_no_candidate=True)
        policy = build_default_policy(config)
        snapshot = {
            "anchor_time": "2026-03-24T00:00:00+00:00",
            "regime_id": "asia|low|range",
            "regime_features": [1.0, 0.0, 0.0, 0.0, 0.0],
            "realized_volatility": 0.02,
            "trend_strength": 0.1,
            "overlay_target": 0,
            "overlay_probability": 0.8,
            "horizons": [
                {
                    "horizon": 1,
                    "mean": -0.03,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 1,
                    "true_return": -0.03,
                    "true_direction": -1,
                    "direction_probabilities": [0.1, 0.1, 0.8],
                },
                {
                    "horizon": 2,
                    "mean": 0.0,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 0,
                    "true_return": 0.0,
                    "true_direction": 0,
                    "direction_probabilities": [0.1, 0.8, 0.1],
                },
            ],
        }

        decision = _augment_snapshot(snapshot, policy, config)

        self.assertIsNone(decision["selected_horizon"])
        self.assertIsNone(decision["proposed_horizon"])
        self.assertIsNone(decision["accepted_horizon"])
        self.assertEqual(decision["accept_reject_reason"], "no_candidate")
        self.assertTrue(decision["reject_flags"]["no_candidate"])
        self.assertFalse(decision["accepted_signal"])

    def test_score_source_mismatch_marks_threshold_as_missing_with_compatibility(self) -> None:
        config = TrainingConfig(horizons=(1,), selection_score_source="correctness_probability")
        policy = build_default_policy(TrainingConfig(horizons=(1,)))
        policy["selection_thresholds"]["global"] = 0.6
        snapshot = {
            "anchor_time": "2026-03-24T00:00:00+00:00",
            "regime_id": "asia|low|range",
            "regime_features": [1.0, 0.0, 0.0, 0.0, 0.0],
            "realized_volatility": 0.02,
            "trend_strength": 0.1,
            "overlay_target": 0,
            "overlay_probability": 0.8,
            "horizons": [
                {
                    "horizon": 1,
                    "mean": 0.03,
                    "sigma": 0.01,
                    "cost": 0.005,
                    "predicted_sign": 1,
                    "true_return": 0.03,
                    "true_direction": 1,
                    "direction_probabilities": [0.1, 0.1, 0.8],
                }
            ],
        }

        decision = _augment_snapshot(snapshot, policy, config)

        self.assertEqual(decision["threshold_status"], "missing")
        self.assertEqual(decision["threshold_origin"], "none")
        self.assertEqual(decision["stored_threshold_compatibility"], "config_mismatch")
        self.assertIsNone(decision["selection_threshold"])

    def test_build_replay_selection_policy_sets_validation_replay_origin(self) -> None:
        config = TrainingConfig(horizons=(1,))
        policy = build_default_policy(config)
        snapshots = [
            {
                "anchor_time": "2026-03-24T00:00:00+00:00",
                "regime_id": "asia|low|range",
                "regime_features": [1.0, 0.0, 0.0, 0.0, 0.0],
                "realized_volatility": 0.02,
                "trend_strength": 0.1,
                "overlay_target": 0,
                "overlay_probability": 0.8,
                "horizons": [
                    {
                        "horizon": 1,
                        "mean": 0.03,
                        "sigma": 0.01,
                        "cost": 0.005,
                        "predicted_sign": 1,
                        "true_return": 0.03,
                        "true_direction": 1,
                        "direction_probabilities": [0.1, 0.1, 0.8],
                    }
                ],
            }
        ]

        replay_policy = build_replay_selection_policy(policy, snapshots, config)

        self.assertEqual(replay_policy["selection_thresholds"]["origin"], "validation_replay")

    def test_prediction_result_selected_horizon_is_read_side_alias(self) -> None:
        prediction = PredictionResult(
            anchor_time="2026-03-24T00:00:00+00:00",
            current_close=100.0,
            proposed_horizon=3,
            accepted_horizon=None,
            selected_direction=1,
            position=0.0,
            expected_log_returns={"3": 0.01},
            predicted_closes={"3": 101.0},
            uncertainties={"3": 0.02},
            accepted_signal=False,
            selection_probability=0.6,
            selection_score=0.6,
            selection_threshold=0.7,
            threshold_status="missing",
            threshold_origin="none",
            correctness_probability=0.55,
            hold_probability=0.5,
            hold_threshold=0.5,
            overlay_action="hold",
            regime_id="asia|low|range",
        )

        self.assertEqual(prediction.selected_horizon, 3)
        self.assertNotIn("selected_horizon", asdict(prediction))


if __name__ == "__main__":
    unittest.main()
