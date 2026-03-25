from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.diagnostics_service import (
    DIAGNOSTICS_SCHEMA_VERSION,
    export_review_diagnostics,
)
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

    def test_export_review_diagnostics_writes_current_schema_headers(self) -> None:
        config = TrainingConfig(horizons=(1,))
        example = _example(
            returns_target=(0.01,),
            direction_targets=(1,),
            direction_thresholds=(0.015,),
        )
        diagnostics = {
            "validation_rows": [
                {
                    "sample_id": 0,
                    "horizon": 1,
                    "proposed": 1,
                    "accepted": 0,
                }
            ],
            "threshold_scan": [
                {
                    "tau": 0.7,
                    "accepted_count_at_tau": 3,
                    "success_count_at_tau": 2,
                    "precision_at_tau": 2 / 3,
                    "lcb": 0.5,
                    "feasible": False,
                    "proposal_coverage": 0.3,
                    "anchor_coverage": 0.2,
                }
            ],
            "threshold_scan_source": "policy_calibration:validation_replay",
            "horizon_diag": [
                {
                    "horizon": 1,
                    "nonflat_rate": 1.0,
                    "up_rate": 1.0,
                    "down_rate": 0.0,
                    "align_rate": 1.0,
                    "candidate_rate": 1.0,
                    "strict_candidate_rate": 1.0,
                    "actionable_edge_rate": 1.0,
                    "mean_mu": 0.01,
                    "mean_sigma": 0.02,
                    "median_abs_mu": 0.01,
                    "proposed_rate": 1.0,
                    "accepted_rate": 0.0,
                }
            ],
            "proposed_row_count": 1,
            "accepted_row_count": 0,
            "no_candidate_count": 0,
            "no_strict_candidate_count": 0,
            "candidate_but_no_strict_count": 0,
            "any_candidate_rate": 1.0,
            "any_strict_candidate_rate": 1.0,
            "candidate_count_per_anchor": 1.0,
            "strict_candidate_count_per_anchor": 1.0,
            "accept_reject_reason_counts": {"selection_threshold_missing": 1},
            "reject_flag_counts": {"selection_threshold_missing": 1},
            "threshold_status": "missing",
            "threshold_origin": "validation_replay",
            "stored_threshold_compatibility": "config_mismatch",
            "threshold_score_source": "selector_probability",
            "proposed_horizon_summary": {
                "1": {
                    "proposed_count": 1,
                    "accepted_count": 0,
                    "proposed_clean_count": 1,
                    "accepted_clean_count": 0,
                }
            },
        }

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            with (
                patch(
                    "signal_cascade_pytorch.application.diagnostics_service.build_validation_diagnostics",
                    return_value=diagnostics,
                ),
                patch(
                    "signal_cascade_pytorch.application.diagnostics_service._build_dataset_summary",
                    return_value={"kind": "test"},
                ),
                patch(
                    "signal_cascade_pytorch.application.diagnostics_service._build_label_summary",
                    return_value={"count": 1},
                ),
            ):
                summary = export_review_diagnostics(
                    output_dir=output_dir,
                    model=object(),
                    examples=[example],
                    config=config,
                    selection_policy=None,
                    threshold_resolution={
                        "selection_threshold_mode_requested": "none",
                        "selection_threshold_mode_resolved": "none",
                        "stored_threshold_compatibility": "not_applicable",
                    },
                )

            self.assertEqual(summary["diagnostics_schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
            self.assertIn("generated_at_utc", summary)
            self.assertEqual(summary["validation"]["threshold_status"], "disabled")

            with (output_dir / "threshold_scan.csv").open(newline="", encoding="utf-8") as handle:
                threshold_header = next(csv.reader(handle))
            self.assertEqual(
                threshold_header,
                [
                    "tau",
                    "accepted_count_at_tau",
                    "success_count_at_tau",
                    "precision_at_tau",
                    "lcb",
                    "feasible",
                    "proposal_coverage",
                    "anchor_coverage",
                ],
            )

            with (output_dir / "horizon_diag.csv").open(newline="", encoding="utf-8") as handle:
                horizon_header = next(csv.reader(handle))
            self.assertEqual(
                horizon_header,
                [
                    "horizon",
                    "nonflat_rate",
                    "up_rate",
                    "down_rate",
                    "align_rate",
                    "candidate_rate",
                    "strict_candidate_rate",
                    "actionable_edge_rate",
                    "mean_mu",
                    "mean_sigma",
                    "median_abs_mu",
                    "proposed_rate",
                    "accepted_rate",
                ],
            )

            written_summary = json.loads((output_dir / "validation_summary.json").read_text())
            self.assertEqual(
                written_summary["diagnostics_schema_version"],
                DIAGNOSTICS_SCHEMA_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
