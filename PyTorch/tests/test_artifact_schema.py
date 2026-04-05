from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from signal_cascade_pytorch.application.artifact_provenance import (
    build_artifact_source_payload,
    build_subartifact_lineage,
    materialize_artifact_source,
)
from signal_cascade_pytorch.bootstrap import _resolve_diagnostics_output_dir
from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.diagnostics_service import DIAGNOSTICS_SCHEMA_VERSION
from signal_cascade_pytorch.application.inference_service import (
    FORECAST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    build_forecast_summary_payload,
    serialize_prediction_result,
)
from signal_cascade_pytorch.application.report_service import ANALYSIS_SCHEMA_VERSION, generate_research_report
from signal_cascade_pytorch.domain.entities import OHLCVBar, PredictionResult


def _prediction() -> PredictionResult:
    return PredictionResult(
        anchor_time="2026-03-24T00:00:00+00:00",
        current_close=100.0,
        policy_horizon=3,
        executed_horizon=3,
        previous_position=0.0,
        position=0.4,
        trade_delta=0.4,
        no_trade_band_hit=False,
        tradeability_gate=0.8,
        shape_entropy=0.3,
        policy_score=0.05,
        expected_log_returns={"1": 0.01, "3": 0.02},
        predicted_closes={"1": 101.0, "3": 102.0},
        uncertainties={"1": 0.01, "3": 0.02},
        horizon_utilities={"1": 0.01, "3": 0.05},
        horizon_positions={"1": 0.1, "3": 0.4},
        shape_probabilities={"0": 0.4, "1": 0.6},
        regime_id="asia|low|trend",
    )


class ArtifactSchemaTests(unittest.TestCase):
    def test_config_round_trip_materializes_versioned_state_reset_contract(self) -> None:
        config = TrainingConfig()

        payload = config.to_dict()
        restored = TrainingConfig.from_dict(payload)

        self.assertEqual(payload["config_schema_version"], 2)
        self.assertEqual(restored.training_state_reset_mode, "carry_on")
        self.assertEqual(restored.evaluation_state_reset_mode, "carry_on")
        self.assertEqual(
            restored.diagnostic_state_reset_modes,
            ("carry_on", "reset_each_session_or_window", "reset_each_example"),
        )
        self.assertEqual(
            restored.policy_sweep_state_reset_modes,
            ("carry_on", "reset_each_session_or_window"),
        )
        self.assertEqual(restored.policy_sweep_min_policy_sigmas, (5e-5, 1e-4, 2e-4))

    def test_legacy_config_payload_keeps_legacy_state_reset_semantics(self) -> None:
        legacy_payload = {
            "seed": 7,
            "synthetic_bars": 10080,
            "epochs": 6,
            "warmup_epochs": 1,
            "oof_epochs": 3,
            "batch_size": 16,
            "learning_rate": 8e-4,
            "weight_decay": 1e-4,
            "hidden_dim": 48,
            "state_dim": 24,
            "shape_classes": 6,
            "dropout": 0.1,
            "train_ratio": 0.8,
            "base_cost": 6e-4,
            "delta_multiplier": 1.35,
            "mae_multiplier": 0.95,
            "overlay_delta_multiplier": 0.75,
            "overlay_mae_multiplier": 0.7,
            "clean_weight_return_scale": 0.75,
            "clean_weight_bonus": 0.65,
            "clean_weight_ratio_scale": 0.35,
            "return_loss_weight": 0.15,
            "shape_loss_weight": 0.05,
            "profit_loss_weight": 1.0,
            "cvar_weight": 0.2,
            "cvar_alpha": 0.1,
            "risk_aversion_gamma": 3.0,
            "q_max": 1.0,
            "policy_abs_epsilon": 1e-4,
            "policy_smoothing_beta": 15.0,
            "min_policy_sigma": 1e-4,
            "branch_dilations": [1, 2, 4],
            "walk_forward_folds": 3,
            "position_scale": 1.0,
            "allow_no_candidate": False,
            "selection_score_source": "profit_utility",
            "output_dir": "artifacts/demo",
            "horizons": [1, 3],
            "main_windows": {"4h": 48, "1d": 21, "1w": 8},
            "overlay_windows": {"1h": 48, "30m": 96},
            "timeframe_parameters": {
                key: value.to_dict() for key, value in TrainingConfig(horizons=(1, 3)).timeframe_parameters.items()
            },
        }

        restored = TrainingConfig.from_dict(legacy_payload)

        self.assertEqual(restored.training_state_reset_mode, "carry_on")
        self.assertEqual(restored.evaluation_state_reset_mode, "carry_on")
        self.assertEqual(restored.policy_sweep_state_reset_modes, ("carry_on",))
        self.assertEqual(restored.policy_sweep_min_policy_sigmas, (1e-4,))

    def test_artifact_source_payload_materializes_ids_hashes_and_snapshot_metadata(self) -> None:
        config = TrainingConfig(output_dir="artifacts/demo")
        bars = [
            OHLCVBar(
                timestamp=datetime(2026, 3, 24, hour, 0, tzinfo=timezone.utc),
                open=100.0 + hour,
                high=101.0 + hour,
                low=99.0 + hour,
                close=100.5 + hour,
                volume=10.0 + hour,
            )
            for hour in (0, 4)
        ]

        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as parent_dir:
            output_dir = Path(temp_dir)
            parent_artifact_dir = Path(parent_dir)
            (output_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            (parent_artifact_dir / "source.json").write_text(
                json.dumps({"artifact_id": "parent-artifact-id"}),
                encoding="utf-8",
            )
            source_payload = materialize_artifact_source(
                {"kind": "csv", "path": "/tmp/live.csv"},
                output_dir,
                base_bars=bars,
            )
            payload = build_artifact_source_payload(
                source_payload,
                output_dir,
                artifact_kind="diagnostic_replay_overlay",
                parent_artifact_dir=parent_artifact_dir,
                sub_artifacts=build_subartifact_lineage(
                    {
                        "config.json": "generated",
                        "data_snapshot.csv": "generated",
                    },
                    source_artifact_dir=parent_artifact_dir,
                ),
            )

        self.assertEqual(payload["artifact_schema_version"], 2)
        self.assertEqual(payload["artifact_kind"], "diagnostic_replay_overlay")
        self.assertEqual(payload["config_origin"], "explicit_v2")
        self.assertEqual(payload["parent_artifact_id"], "parent-artifact-id")
        self.assertEqual(payload["state_reset_boundary_spec_version"], 1)
        self.assertTrue(str(payload["path"]).endswith("data_snapshot.csv"))
        self.assertIsNotNone(payload.get("artifact_id"))
        self.assertIsNotNone(payload.get("config_sha256"))
        self.assertIsNotNone(payload.get("data_snapshot_sha256"))
        self.assertEqual(payload["sub_artifacts"]["config.json"]["schema_version"], 2)
        self.assertIsNotNone(payload["sub_artifacts"]["data_snapshot.csv"].get("sha256"))

    def test_overlay_output_dir_defaults_to_sibling_and_rejects_source_artifact_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "current"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            resolved_default = _resolve_diagnostics_output_dir(artifact_dir, None)
            self.assertNotEqual(resolved_default, artifact_dir)
            self.assertEqual(resolved_default.name, "current_diagnostic_replay_overlay")

            with self.assertRaises(ValueError):
                _resolve_diagnostics_output_dir(artifact_dir, str(artifact_dir))

    def test_prediction_serializer_adds_schema_and_median_semantics(self) -> None:
        payload = serialize_prediction_result(_prediction())

        self.assertEqual(payload["schema_version"], PREDICTION_SCHEMA_VERSION)
        self.assertEqual(payload["predicted_close_semantics"], "median_from_log_return")
        self.assertEqual(payload["median_predicted_closes"]["3"], 102.0)

    def test_report_analysis_tracks_artifact_versions(self) -> None:
        prediction = _prediction()
        config = TrainingConfig(horizons=(1, 3))
        prediction_payload = serialize_prediction_result(prediction)
        forecast_payload = build_forecast_summary_payload(prediction, config)

        self.assertEqual(forecast_payload["schema_version"], FORECAST_SCHEMA_VERSION)
        self.assertIn("median_predicted_close", forecast_payload["forecast_rows"][0])

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "policy_mode": "shape_aware_profit_maximization",
                        "sample_count": 10,
                        "effective_sample_count": 8,
                        "train_samples": 5,
                        "validation_samples": 3,
                        "purged_samples": 2,
                        "best_validation_loss": -0.1,
                        "best_epoch": 2.0,
                        "validation_metrics": {
                            "average_log_wealth": 0.01,
                            "exact_smooth_horizon_agreement": 1.0,
                            "exact_smooth_no_trade_agreement": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "prediction.json").write_text(
                json.dumps(prediction_payload),
                encoding="utf-8",
            )
            (output_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            (output_dir / "source.json").write_text(
                json.dumps(
                    {
                        "kind": "csv",
                        "path": "/tmp/frozen.csv",
                        "artifact_schema_version": 2,
                        "artifact_kind": "diagnostic_replay_overlay",
                        "artifact_id": "overlay-123",
                        "artifact_dir": str(output_dir),
                        "parent_artifact_dir": "/tmp/parent-artifact",
                        "parent_artifact_id": "parent-123",
                        "config_origin": "explicit_v2",
                        "data_snapshot_sha256": "feedface",
                        "git": {
                            "head": "abc123",
                            "git_commit_sha": "abc123",
                            "git_tree_sha": "cafe1234",
                            "dirty": True,
                            "git_dirty": True,
                            "dirty_patch_sha256": "deadbeef",
                        },
                        "sub_artifacts": {
                            "prediction.json": {"materialization": "copied"},
                            "analysis.json": {"materialization": "regenerated"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                        "primary_state_reset_mode": "reset_each_session_or_window",
                        "stateful_evaluation": {"carry_on": {"average_log_wealth": 0.01}},
                        "policy_calibration_summary": {
                            "row_count": 1,
                            "pareto_optimal_count": 1,
                            "dominated_count": 0,
                            "policy_calibration_rows_sha256": "rows1234",
                            "selection_basis": "pareto_rank_then_average_log_wealth_cvar_tail_loss_turnover_row_key",
                            "selection_rule_version": 2,
                            "selected_row_key": "state_reset_mode=reset_each_session_or_window|cost_multiplier=4|gamma_multiplier=2|min_policy_sigma=0.0001",
                            "selected_row": {
                                "row_key": "state_reset_mode=reset_each_session_or_window|cost_multiplier=4|gamma_multiplier=2|min_policy_sigma=0.0001",
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 4.0,
                                "gamma_multiplier": 2.0,
                                "min_policy_sigma": 0.0001,
                                "average_log_wealth": 0.01,
                            },
                            "best_row": {
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 4.0,
                                "gamma_multiplier": 2.0,
                                "min_policy_sigma": 0.0001,
                                "average_log_wealth": 0.01,
                            },
                        },
                        "policy_calibration_sweep": [
                            {
                                "row_key": "state_reset_mode=reset_each_session_or_window|cost_multiplier=1|gamma_multiplier=1|min_policy_sigma=0.0001",
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 1.0,
                                "gamma_multiplier": 1.0,
                                "min_policy_sigma": 0.0001,
                                "average_log_wealth": 0.01,
                                "turnover": 0.4,
                                "cvar_tail_loss": 0.02,
                                "no_trade_band_hit_rate": 0.0,
                                "dominated": False,
                                "pareto_optimal": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = generate_research_report(output_dir)

        self.assertEqual(analysis["schema_version"], ANALYSIS_SCHEMA_VERSION)
        self.assertEqual(analysis["artifact_versions"]["prediction"], PREDICTION_SCHEMA_VERSION)
        self.assertEqual(analysis["artifact_provenance"]["artifact_kind"], "diagnostic_replay_overlay")
        self.assertEqual(analysis["artifact_provenance"]["artifact_id"], "overlay-123")
        self.assertEqual(analysis["artifact_provenance"]["git_head"], "abc123")
        self.assertEqual(analysis["artifact_provenance"]["git_commit_sha"], "abc123")
        self.assertEqual(analysis["artifact_provenance"]["git_tree_sha"], "cafe1234")
        self.assertEqual(analysis["dataset"]["purged_samples"], 2)
        self.assertIn("carry_on", analysis["stateful_evaluation"])
        self.assertEqual(analysis["policy_calibration_summary"]["pareto_optimal_count"], 1)
        self.assertEqual(analysis["policy_calibration_summary"]["selection_rule_version"], 2)
        self.assertEqual(
            analysis["policy_calibration_summary"]["selected_row"]["state_reset_mode"],
            "reset_each_session_or_window",
        )


if __name__ == "__main__":
    unittest.main()
