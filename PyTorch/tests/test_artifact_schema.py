from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from signal_cascade_pytorch.application.artifact_provenance import (
    build_artifact_source_payload,
    build_subartifact_lineage,
    materialize_artifact_source,
)
from signal_cascade_pytorch.application.current_alias import build_current_alias_metadata
from signal_cascade_pytorch.application.tuning_service import tune_latest_dataset
from signal_cascade_pytorch.bootstrap import (
    _build_artifact_entrypoints,
    _build_artifact_manifest,
    _promote_training_run_to_current,
    _resolve_diagnostics_output_dir,
    _resolve_top_level_report_path,
)
from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.dataset_service import (
    build_latest_inference_example_from_bars,
    build_training_examples_from_bars,
)
from signal_cascade_pytorch.application.diagnostics_service import DIAGNOSTICS_SCHEMA_VERSION
from signal_cascade_pytorch.application.inference_service import (
    FORECAST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    build_forecast_summary_payload,
    serialize_prediction_result,
)
from signal_cascade_pytorch.application.report_service import ANALYSIS_SCHEMA_VERSION, generate_research_report
from signal_cascade_pytorch.domain.entities import (
    OHLCVBar,
    PredictionResult,
    STATE_FEATURE_NAMES,
    STATE_VECTOR_COMPONENT_NAMES,
    TIMEFRAME_FEATURE_NAMES,
    TRAINING_EXAMPLE_CONTRACT_VERSION,
)
from signal_cascade_pytorch.bootstrap import _build_model_from_example
from signal_cascade_pytorch.infrastructure.persistence import save_checkpoint
from signal_cascade_pytorch.interfaces.cli import build_parser


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
        price_scale=100.0,
    )


def _build_csv_bars(*, start: datetime, count: int) -> list[OHLCVBar]:
    bars: list[OHLCVBar] = []
    for index in range(count):
        timestamp = start + timedelta(minutes=30 * index)
        open_price = 1800.0 + (0.2 * index)
        close_price = open_price + (0.08 if index % 2 == 0 else -0.05)
        bars.append(
            OHLCVBar(
                timestamp=timestamp,
                open=open_price,
                high=max(open_price, close_price) + 0.12,
                low=min(open_price, close_price) - 0.12,
                close=close_price,
                volume=1000.0 + float(index % 17),
            )
        )
    return bars


def _write_csv_bars(path: Path, bars: list[OHLCVBar]) -> None:
    rows = ["timestamp,open,high,low,close,volume"]
    rows.extend(
        ",".join(
            (
                bar.timestamp.isoformat(),
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.volume),
            )
        )
        for bar in bars
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class ArtifactSchemaTests(unittest.TestCase):
    def test_training_manifest_matches_source_identity_and_relative_entrypoints(self) -> None:
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

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            for relative_path in (
                "analysis.json",
                "forecast_summary.json",
                "horizon_diag.csv",
                "metrics.json",
                "model.pt",
                "policy_summary.csv",
                "prediction.json",
                "research_report.md",
                "validation_rows.csv",
                "validation_summary.json",
            ):
                (output_dir / relative_path).write_text("{}", encoding="utf-8")

            source_payload = materialize_artifact_source(
                {"kind": "csv", "path": "/tmp/live.csv"},
                output_dir,
                base_bars=bars,
            )
            source = build_artifact_source_payload(
                source_payload,
                output_dir,
                artifact_kind="training_run",
                generated_at_utc="2026-04-05T10:00:00+00:00",
                sub_artifacts=build_subartifact_lineage(
                    {
                        "analysis.json": "regenerated",
                        "config.json": "generated",
                        "data_snapshot.csv": "generated",
                        "forecast_summary.json": "generated",
                        "horizon_diag.csv": "generated",
                        "manifest.json": "generated",
                        "metrics.json": "generated",
                        "prediction.json": "generated",
                        "research_report.md": "regenerated",
                        "source.json": "generated",
                        "validation_rows.csv": "generated",
                        "validation_summary.json": "generated",
                    }
                ),
            )
            (output_dir / "source.json").write_text(json.dumps(source), encoding="utf-8")
            manifest = _build_artifact_manifest(
                artifact_kind="training_run",
                artifact_id=str(source["artifact_id"]),
                parent_artifact_id=None,
                generated_at_utc="2026-04-05T10:00:00+00:00",
                source_payload=source_payload,
                entrypoints=_build_artifact_entrypoints(source_payload, include_model=True),
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["artifact_kind"], source["artifact_kind"])
            self.assertEqual(manifest["artifact_id"], source["artifact_id"])
            self.assertIsNone(manifest["parent_artifact_id"])
            self.assertEqual(manifest["generated_at"], "2026-04-05T10:00:00+00:00")
            self.assertEqual(manifest["generated_at_utc"], "2026-04-05T10:00:00+00:00")
            self.assertEqual(manifest["effective_price_scale"], 1.0)
            self.assertEqual(manifest["price_scale_origin"], "default")
            self.assertFalse(manifest["provider_scale_confirmed"])
            self.assertEqual(manifest["entrypoints"]["source"], "source.json")
            self.assertEqual(manifest["entrypoints"]["config"], "config.json")
            self.assertEqual(manifest["entrypoints"]["validation_summary"], "validation_summary.json")
            self.assertEqual(manifest["entrypoints"]["metrics"], "metrics.json")
            self.assertEqual(manifest["entrypoints"]["prediction"], "prediction.json")
            self.assertEqual(manifest["entrypoints"]["model"], "model.pt")
            self.assertEqual(manifest["entrypoints"]["data_snapshot"], "data_snapshot.csv")
            for relative_path in manifest["entrypoints"].values():
                self.assertFalse(Path(relative_path).is_absolute())
                self.assertTrue((output_dir / relative_path).exists())

    def test_overlay_manifest_matches_source_identity_and_relative_entrypoints(self) -> None:
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
            (output_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            for relative_path in (
                "analysis.json",
                "forecast_summary.json",
                "horizon_diag.csv",
                "metrics.json",
                "policy_summary.csv",
                "prediction.json",
                "research_report.md",
                "validation_rows.csv",
                "validation_summary.json",
            ):
                (output_dir / relative_path).write_text("{}", encoding="utf-8")
            (parent_artifact_dir / "source.json").write_text(
                json.dumps({"artifact_id": "parent-artifact-id"}),
                encoding="utf-8",
            )

            source_payload = materialize_artifact_source(
                {"kind": "csv", "path": "/tmp/live.csv"},
                output_dir,
                base_bars=bars,
            )
            source = build_artifact_source_payload(
                source_payload,
                output_dir,
                artifact_kind="diagnostic_replay_overlay",
                parent_artifact_dir=parent_artifact_dir,
                generated_at_utc="2026-04-05T10:05:00+00:00",
                sub_artifacts=build_subartifact_lineage(
                    {
                        "analysis.json": "regenerated",
                        "config.json": "generated",
                        "data_snapshot.csv": "generated",
                        "forecast_summary.json": "copied",
                        "horizon_diag.csv": "regenerated",
                        "manifest.json": "generated",
                        "metrics.json": "regenerated",
                        "policy_summary.csv": "regenerated",
                        "prediction.json": "copied",
                        "research_report.md": "regenerated",
                        "source.json": "regenerated",
                        "validation_rows.csv": "regenerated",
                        "validation_summary.json": "regenerated",
                    },
                    source_artifact_dir=parent_artifact_dir,
                ),
            )
            (output_dir / "source.json").write_text(json.dumps(source), encoding="utf-8")
            manifest = _build_artifact_manifest(
                artifact_kind="diagnostic_replay_overlay",
                artifact_id=str(source["artifact_id"]),
                parent_artifact_id=str(source["parent_artifact_id"]),
                generated_at_utc="2026-04-05T10:05:00+00:00",
                source_payload=source_payload,
                entrypoints=_build_artifact_entrypoints(source_payload),
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["artifact_kind"], source["artifact_kind"])
            self.assertEqual(manifest["artifact_id"], source["artifact_id"])
            self.assertEqual(manifest["parent_artifact_id"], "parent-artifact-id")
            self.assertEqual(manifest["effective_price_scale"], 1.0)
            self.assertEqual(manifest["price_scale_origin"], "default")
            self.assertFalse(manifest["provider_scale_confirmed"])
            self.assertEqual(manifest["entrypoints"]["source"], "source.json")
            self.assertEqual(manifest["entrypoints"]["config"], "config.json")
            self.assertEqual(manifest["entrypoints"]["validation_summary"], "validation_summary.json")
            self.assertEqual(manifest["entrypoints"]["metrics"], "metrics.json")
            self.assertEqual(manifest["entrypoints"]["prediction"], "prediction.json")
            self.assertNotIn("model", manifest["entrypoints"])
            self.assertEqual(manifest["entrypoints"]["data_snapshot"], "data_snapshot.csv")
            for relative_path in manifest["entrypoints"].values():
                self.assertFalse(Path(relative_path).is_absolute())
                self.assertTrue((output_dir / relative_path).exists())

    def test_config_round_trip_materializes_versioned_state_reset_contract(self) -> None:
        config = TrainingConfig(requested_price_scale=100.0)

        payload = config.to_dict()
        restored = TrainingConfig.from_dict(payload)

        self.assertEqual(payload["config_schema_version"], 7)
        self.assertEqual(restored.training_state_reset_mode, "carry_on")
        self.assertEqual(restored.evaluation_state_reset_mode, "carry_on")
        self.assertEqual(restored.policy_cost_multiplier, 1.0)
        self.assertEqual(restored.policy_gamma_multiplier, 1.0)
        self.assertFalse(restored.tie_policy_to_forecast_head)
        self.assertFalse(restored.disable_overlay_branch)
        self.assertEqual(
            restored.diagnostic_state_reset_modes,
            ("carry_on", "reset_each_session_or_window", "reset_each_example"),
        )
        self.assertEqual(
            restored.policy_sweep_state_reset_modes,
            ("carry_on", "reset_each_session_or_window"),
        )
        self.assertEqual(restored.policy_sweep_min_policy_sigmas, (5e-5, 1e-4, 2e-4))
        self.assertEqual(restored.policy_sweep_q_max_values, (1.0,))
        self.assertEqual(restored.policy_sweep_cvar_weights, (0.2,))
        self.assertEqual(payload["feature_contract_version"], TRAINING_EXAMPLE_CONTRACT_VERSION)
        self.assertEqual(tuple(payload["timeframe_feature_names"]), TIMEFRAME_FEATURE_NAMES)
        self.assertEqual(tuple(payload["state_feature_names"]), STATE_FEATURE_NAMES)
        self.assertEqual(
            tuple(payload["state_vector_component_names"]),
            STATE_VECTOR_COMPONENT_NAMES,
        )
        self.assertEqual(payload["requested_price_scale"], 100.0)
        self.assertEqual(restored.feature_contract_version, TRAINING_EXAMPLE_CONTRACT_VERSION)
        self.assertEqual(restored.timeframe_feature_names, TIMEFRAME_FEATURE_NAMES)
        self.assertEqual(restored.state_feature_names, STATE_FEATURE_NAMES)
        self.assertEqual(restored.state_vector_component_names, STATE_VECTOR_COMPONENT_NAMES)
        self.assertEqual(restored.requested_price_scale, 100.0)

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
        self.assertEqual(restored.policy_cost_multiplier, 1.0)
        self.assertEqual(restored.policy_gamma_multiplier, 1.0)
        self.assertFalse(restored.tie_policy_to_forecast_head)
        self.assertFalse(restored.disable_overlay_branch)
        self.assertEqual(restored.policy_sweep_state_reset_modes, ("carry_on",))
        self.assertEqual(restored.policy_sweep_min_policy_sigmas, (1e-4,))
        self.assertEqual(restored.policy_sweep_q_max_values, (1.0,))
        self.assertEqual(restored.policy_sweep_cvar_weights, (0.2,))

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
        self.assertEqual(payload["effective_price_scale"], 1.0)
        self.assertEqual(payload["price_scale_origin"], "default")
        self.assertFalse(payload["provider_scale_confirmed"])
        self.assertEqual(payload["parent_artifact_id"], "parent-artifact-id")
        self.assertEqual(payload["state_reset_boundary_spec_version"], 1)
        self.assertTrue(str(payload["path"]).endswith("data_snapshot.csv"))
        self.assertIsNotNone(payload.get("artifact_id"))
        self.assertIsNotNone(payload.get("config_sha256"))
        self.assertIsNotNone(payload.get("data_snapshot_sha256"))
        self.assertEqual(payload["sub_artifacts"]["config.json"]["schema_version"], 7)
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

    def test_promote_training_run_to_current_replaces_current_without_mutating_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            source_dir = artifact_root / "reruns" / "accepted_parent"
            current_dir = artifact_root / "current"
            source_dir.mkdir(parents=True, exist_ok=True)
            current_dir.mkdir(parents=True, exist_ok=True)

            (source_dir / "source.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "training_run",
                        "artifact_id": "artifact-123",
                        "git": {"git_commit_sha": "1497a93", "git_dirty": False},
                    }
                ),
                encoding="utf-8",
            )
            (source_dir / "config.json").write_text("{}", encoding="utf-8")
            (source_dir / "validation_summary.json").write_text(
                json.dumps({"generated_at_utc": "2026-04-06T04:40:00+00:00"}),
                encoding="utf-8",
            )
            (source_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (source_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
            (source_dir / "marker.txt").write_text("accepted", encoding="utf-8")
            (current_dir / "source.json").write_text(
                json.dumps({"artifact_kind": "legacy"}),
                encoding="utf-8",
            )
            (current_dir / "old-only.txt").write_text("legacy", encoding="utf-8")

            promoted_dir = _promote_training_run_to_current(artifact_root, source_dir)

            self.assertEqual(promoted_dir.resolve(), current_dir.resolve())
            self.assertTrue((current_dir / "marker.txt").exists())
            self.assertFalse((current_dir / "old-only.txt").exists())
            promoted_source = json.loads((current_dir / "source.json").read_text(encoding="utf-8"))
            promoted_manifest = json.loads((current_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(promoted_source["artifact_kind"], "training_run")
            self.assertEqual(promoted_source["artifact_id"], "artifact-123")
            self.assertEqual(
                promoted_source["current_alias_contract"]["alias_role"],
                "production_current",
            )
            self.assertEqual(
                promoted_source["current_selection_governance"]["selection_mode"],
                "manual_promote_without_session_context",
            )
            self.assertEqual(promoted_manifest["schema_version"], 1)
            self.assertEqual(promoted_manifest["artifact_kind"], "training_run")
            self.assertEqual(promoted_manifest["artifact_id"], "artifact-123")
            self.assertEqual(
                promoted_manifest["generated_at_utc"],
                "2026-04-06T04:40:00+00:00",
            )
            self.assertEqual(
                promoted_manifest["generated_at_utc"],
                promoted_manifest["generated_at"],
            )
            self.assertEqual(promoted_manifest["entrypoints"]["source"], "source.json")
            self.assertTrue((source_dir / "marker.txt").exists())
            source_source = json.loads((source_dir / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(source_source["artifact_id"], "artifact-123")

    def test_build_current_alias_metadata_backfills_forecast_quality_from_candidate_diagnostics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            (artifact_root / "current").mkdir(parents=True, exist_ok=True)
            session_dir = artifact_root / "archive" / "session_20260408T075001Z"
            candidate_dir = session_dir / "candidate_03"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            (candidate_dir / "source.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "training_run",
                        "artifact_id": "artifact-123",
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "prediction.json").write_text(
                json.dumps(
                    {
                        "policy_horizon": 1,
                        "trade_delta": -0.05,
                        "g_t": 0.49,
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "validation": {
                            "average_log_wealth": 0.001,
                            "project_value_score": 0.62,
                            "utility_score": 0.55,
                            "mu_calibration": 0.04,
                            "sigma_calibration": 0.06,
                            "directional_accuracy": 0.61,
                            "max_drawdown": 0.08,
                            "turnover": 1.2,
                            "exact_smooth_position_mae": 0.03,
                            "no_trade_band_hit_rate": 0.07,
                        },
                        "dataset": {
                            "validation_sample_count": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "validation_rows.csv").write_text(
                "\n".join(
                    (
                        "timestamp,mu_t,sigma_t,y_raw,pit,normalized_abs_error,policy_horizon_selected,selected",
                        "2026-04-08T00:00:00+00:00,0.01,0.02,0.015,0.52,0.4,1,1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (session_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "best_candidate": {"candidate": "candidate_03"},
                        "accepted_candidate": {"candidate": "candidate_03"},
                        "production_current_candidate": {"candidate": "candidate_03"},
                        "production_current_selection": {
                            "selection_mode": "accepted_candidate",
                            "selection_rule": "optimization_gate_then_deployment_score",
                            "selection_rule_version": 1,
                            "selection_status": "accepted_and_production_same",
                            "decision_summary": "production current matches accepted candidate",
                            "override_priority_metrics": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "leaderboard.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "candidate": "candidate_03",
                                "blocked_average_log_wealth_mean": 0.0008,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0004,
                                "blocked_turnover_mean": 0.9,
                                "blocked_directional_accuracy_mean": 0.6,
                                "blocked_exact_smooth_position_mae_mean": 0.025,
                                "user_value_score": 0.58,
                                "user_value_chart_fidelity_score": 0.69,
                                "user_value_sigma_band_score": 0.61,
                                "user_value_execution_stability_score": 0.68,
                                "optimization_gate_passed": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            metadata = build_current_alias_metadata(
                artifact_root,
                candidate_dir,
                selection_timestamp_utc="2026-04-08T11:39:12+00:00",
            )

            governance = metadata["current_selection_governance"]
            production_current = governance["production_current"]
            accepted_candidate = governance["accepted_candidate"]
            self.assertIsNotNone(production_current["selected_horizon_forecast_quality_score"])
            self.assertIsNotNone(production_current["all_horizon_forecast_quality_score"])
            self.assertIsNotNone(
                production_current["forecast_quality_score_gap_all_minus_selected"]
            )
            self.assertEqual(
                production_current["selected_horizon_forecast_quality_score"],
                accepted_candidate["selected_horizon_forecast_quality_score"],
            )
            self.assertEqual(
                production_current["all_horizon_forecast_quality_score"],
                accepted_candidate["all_horizon_forecast_quality_score"],
            )
            self.assertEqual(
                governance["paired_frontier"]["delta_production_minus_accepted"][
                    "selected_horizon_forecast_quality_score"
                ],
                0.0,
            )

    def test_promote_training_run_to_current_accepts_overlay_with_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            source_dir = artifact_root / "reruns" / "overlay"
            current_dir = artifact_root / "current"
            source_dir.mkdir(parents=True, exist_ok=True)
            current_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "source.json").write_text(
                json.dumps({"artifact_kind": "diagnostic_replay_overlay", "artifact_id": "overlay-123"}),
                encoding="utf-8",
            )
            (source_dir / "config.json").write_text("{}", encoding="utf-8")
            (source_dir / "validation_summary.json").write_text(
                json.dumps({"generated_at_utc": "2026-04-07T00:00:00+00:00"}),
                encoding="utf-8",
            )
            (source_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (source_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
            (source_dir / "model.pt").write_bytes(b"model")

            promoted_dir = _promote_training_run_to_current(artifact_root, source_dir)

            promoted_source = json.loads((promoted_dir / "source.json").read_text(encoding="utf-8"))
            promoted_manifest = json.loads((promoted_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(promoted_source["artifact_kind"], "diagnostic_replay_overlay")
        self.assertEqual(promoted_manifest["artifact_kind"], "diagnostic_replay_overlay")
        self.assertEqual(promoted_manifest["entrypoints"]["model"], "model.pt")

    def test_promote_training_run_to_current_rejects_non_promotable_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            source_dir = artifact_root / "reruns" / "legacy"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "source.json").write_text(
                json.dumps({"artifact_kind": "legacy"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                _promote_training_run_to_current(artifact_root, source_dir)

    def test_promote_training_run_to_current_rejects_overlay_without_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            source_dir = artifact_root / "reruns" / "overlay"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "source.json").write_text(
                json.dumps({"artifact_kind": "diagnostic_replay_overlay", "artifact_id": "overlay-123"}),
                encoding="utf-8",
            )
            (source_dir / "config.json").write_text("{}", encoding="utf-8")
            (source_dir / "validation_summary.json").write_text(
                json.dumps({"generated_at_utc": "2026-04-07T00:00:00+00:00"}),
                encoding="utf-8",
            )
            (source_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (source_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "model.pt exists"):
                _promote_training_run_to_current(artifact_root, source_dir)

    def test_promote_training_run_to_current_requires_published_diagnostics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            source_dir = artifact_root / "reruns" / "candidate"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "source.json").write_text(
                json.dumps({"artifact_kind": "training_run", "artifact_id": "artifact-123"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "diagnostics unpublished for artifact",
            ):
                _promote_training_run_to_current(artifact_root, source_dir)

    def test_tune_latest_materializes_current_diagnostics_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)
            csv_path = Path(temp_dir) / "xauusd.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "timestamp,open,high,low,close,volume",
                        "2026-03-24T00:00:00+00:00,100,101,99,100.5,10",
                        "2026-03-24T00:30:00+00:00,100.5,101.5,100,101,11",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            validation_metrics = {
                "project_value_score": 0.12,
                "utility_score": 0.08,
                "average_log_wealth": 0.01,
                "realized_pnl_per_anchor": 0.02,
                "cvar_tail_loss": 0.01,
                "max_drawdown": 0.05,
                "no_trade_band_hit_rate": 0.1,
                "mu_calibration": 0.01,
                "sigma_calibration": 0.02,
                "directional_accuracy": 0.6,
            }
            summary = {
                "train_samples": 3,
                "validation_samples": 1,
                "best_validation_loss": -0.1,
                "validation_metrics": validation_metrics,
            }
            diagnostics_summary = {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "generated_at_utc": "2026-04-06T04:40:00+00:00",
                "validation": validation_metrics,
                "blocked_walk_forward_evaluation": {
                    "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                    "state_reset_modes": {
                        "carry_on": {
                            "average_log_wealth_mean": 0.01,
                            "turnover_mean": 0.2,
                            "directional_accuracy_mean": 0.6,
                            "exact_smooth_position_mae_mean": 0.03,
                            "folds": [{"cvar_tail_loss": 0.01}],
                        }
                    },
                },
                "policy_calibration_summary": {
                    "selected_row": {
                        "row_key": "carry_on|cost=1|gamma=1|min_sigma=0.0001|q=1|cvar=0.2",
                    },
                    "applied_runtime_policy": {
                        "row_key": "carry_on|cost=1|gamma=1|min_sigma=0.0001|q=1|cvar=0.2",
                    },
                },
            }

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            def fake_generate_research_report(output_dir, report_path=None):
                (Path(output_dir) / "analysis.json").write_text("{}", encoding="utf-8")
                target_report = (
                    Path(report_path)
                    if report_path is not None
                    else Path(output_dir) / "research_report.md"
                )
                target_report.parent.mkdir(parents=True, exist_ok=True)
                target_report.write_text("# report\n", encoding="utf-8")
                return {}

            with (
                patch(
                    "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                    return_value=[object()],
                ),
                patch(
                    "signal_cascade_pytorch.application.tuning_service.train_model",
                    return_value=(object(), summary),
                ),
                patch(
                    "signal_cascade_pytorch.application.tuning_service.predict_latest",
                    return_value=_prediction(),
                ),
                patch(
                    "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                    side_effect=fake_export_review_diagnostics,
                ),
                patch(
                    "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                    return_value=[
                        {
                            "epochs": 12,
                            "batch_size": 16,
                            "learning_rate": 5e-4,
                            "hidden_dim": 48,
                            "dropout": 0.1,
                            "weight_decay": 5e-5,
                        }
                    ],
                ),
                patch(
                    "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                    side_effect=fake_generate_research_report,
                ),
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"horizons": (1, 3)},
                )

            self.assertTrue(manifest["current_updated"])
            self.assertEqual(manifest["generated_at_utc"], manifest["generated_at"])
            current_dir = artifact_root / "current"
            for relative_path in (
                "horizon_diag.csv",
                "policy_summary.csv",
                "validation_rows.csv",
                "validation_summary.json",
            ):
                self.assertTrue((current_dir / relative_path).exists(), relative_path)

            source_payload = json.loads((current_dir / "source.json").read_text(encoding="utf-8"))
            current_manifest = json.loads((current_dir / "manifest.json").read_text(encoding="utf-8"))
            archived_manifest = json.loads(
                (Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
            )
            self.assertEqual(current_manifest["schema_version"], 1)
            self.assertEqual(current_manifest["artifact_kind"], "training_run")
            self.assertEqual(current_manifest["artifact_id"], source_payload["artifact_id"])
            self.assertIsNone(current_manifest["parent_artifact_id"])
            self.assertEqual(current_manifest["generated_at_utc"], current_manifest["generated_at"])
            self.assertEqual(current_manifest["generated_at_utc"], source_payload["generated_at_utc"])
            self.assertEqual(current_manifest["entrypoints"]["source"], "source.json")
            self.assertEqual(
                source_payload["current_alias_contract"]["alias_role"],
                "production_current",
            )
            self.assertEqual(
                source_payload["current_selection_governance"]["selection_mode"],
                "accepted_candidate",
            )
            self.assertTrue(
                Path(
                    source_payload["current_alias_contract"]["authoritative_paths"]["top_level_report"]
                ).exists()
            )
            self.assertEqual(archived_manifest["generated_at_utc"], archived_manifest["generated_at"])
            for relative_path in (
                "horizon_diag.csv",
                "policy_summary.csv",
                "validation_rows.csv",
                "validation_summary.json",
            ):
                self.assertIn(relative_path, source_payload["sub_artifacts"])

    def test_predict_cli_uses_latest_inference_anchor_for_live_csv(self) -> None:
        bars = _build_csv_bars(
            start=datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
            count=512,
        )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "predict_artifact"
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = Path(temp_dir) / "live_latest.csv"
            _write_csv_bars(csv_path, bars)

            config = TrainingConfig(
                output_dir=str(output_dir),
                horizons=(1,),
                main_windows={"4h": 8, "1d": 3, "1w": 2},
                overlay_windows={"1h": 8, "30m": 16},
            )
            examples = build_training_examples_from_bars(bars, config)
            latest_example = build_latest_inference_example_from_bars(bars, config)
            self.assertLess(examples[-1].anchor_time, latest_example.anchor_time)

            model = _build_model_from_example(examples[0], config)
            save_checkpoint(output_dir / "model.pt", model, config)
            (output_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )

            parser = build_parser()
            args = parser.parse_args(
                [
                    "predict",
                    "--output-dir",
                    str(output_dir),
                    "--csv",
                    str(csv_path),
                ]
            )
            self.assertEqual(args.handler(args), 0)

            prediction_payload = json.loads(
                (output_dir / "prediction.json").read_text(encoding="utf-8")
            )
            forecast_payload = json.loads(
                (output_dir / "forecast_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                prediction_payload["anchor_time"],
                latest_example.anchor_time.isoformat(),
            )
            self.assertEqual(
                forecast_payload["anchor_time"],
                latest_example.anchor_time.isoformat(),
            )
            self.assertNotEqual(
                prediction_payload["anchor_time"],
                examples[-1].anchor_time.isoformat(),
            )
            self.assertEqual(prediction_payload["inference_context_mode"], "carry_on")
            self.assertIsInstance(forecast_payload["generated_at_utc"], str)
            self.assertTrue(forecast_payload["generated_at_utc"])
            self.assertEqual(
                forecast_payload["generated_at"],
                forecast_payload["generated_at_utc"],
            )

    def test_predict_cli_refreshes_current_artifact_contract_outputs(self) -> None:
        bars = _build_csv_bars(
            start=datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
            count=512,
        )

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifacts" / "gold_xauusd_m30"
            current_dir = artifact_root / "current"
            current_dir.mkdir(parents=True, exist_ok=True)
            csv_path = Path(temp_dir) / "live_latest.csv"
            top_level_report = Path(temp_dir) / "report_signalcascade_xauusd.md"
            _write_csv_bars(csv_path, bars)

            config = TrainingConfig(
                output_dir=str(current_dir),
                horizons=(1, 3),
                main_windows={"4h": 8, "1d": 3, "1w": 2},
                overlay_windows={"1h": 8, "30m": 16},
                policy_cost_multiplier=0.5,
                policy_gamma_multiplier=0.5,
                min_policy_sigma=0.0002,
                q_max=1.25,
                cvar_weight=0.1,
            )
            examples = build_training_examples_from_bars(bars, config)
            latest_example = build_latest_inference_example_from_bars(bars, config)
            model = _build_model_from_example(examples[0], config)
            save_checkpoint(current_dir / "model.pt", model, config)
            (current_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            (current_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "history": [],
                        "validation_metrics": {"average_log_wealth": -0.01},
                    }
                ),
                encoding="utf-8",
            )

            selected_row = {
                "state_reset_mode": "reset_each_session_or_window",
                "cost_multiplier": 6.0,
                "gamma_multiplier": 2.0,
                "min_policy_sigma": 0.0001,
                "q_max": 1.0,
                "cvar_weight": 0.2,
            }
            stale_validation_summary = {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "generated_at_utc": "2026-04-07T00:00:00+00:00",
                "primary_state_reset_mode": "carry_on",
                "validation": {"average_log_wealth": -0.02, "directional_accuracy": 0.5},
                "stateful_evaluation": {
                    "carry_on": {
                        "average_log_wealth": -0.02,
                        "directional_accuracy": 0.5,
                    },
                    "reset_each_session_or_window": {
                        "average_log_wealth": 0.01,
                        "directional_accuracy": 0.6,
                    },
                },
                "dataset": {
                    "source": {"kind": "csv", "path": "/tmp/stale.csv"},
                    "source_rows_original": 500,
                    "source_rows_used": 500,
                    "base_bar_count": 500,
                },
                "policy_calibration_summary": {
                    "selected_row": {
                        "row_key": (
                            "state_reset_mode=reset_each_session_or_window|cost_multiplier=6|"
                            "gamma_multiplier=2|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                        ),
                        **selected_row,
                    }
                },
            }
            (current_dir / "validation_summary.json").write_text(
                json.dumps(stale_validation_summary),
                encoding="utf-8",
            )
            (current_dir / "validation_rows.csv").write_text(
                (
                    "sample_id,timestamp,horizon,y_raw,mu_t,sigma_t,pit,normalized_abs_error,"
                    "policy_horizon_selected,selected\n"
                    "0,2026-04-07T00:00:00+00:00,1,0.01,0.012,0.02,0.53,0.10,1,1\n"
                ),
                encoding="utf-8",
            )
            (current_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (current_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
            (current_dir / "source.json").write_text(
                json.dumps(
                    {
                        "kind": "csv",
                        "path": str((current_dir / "data_snapshot.csv").resolve()),
                        "source_origin_path": str(csv_path.resolve()),
                        "artifact_schema_version": 2,
                        "artifact_kind": "training_run",
                        "artifact_id": "current-artifact-123",
                        "artifact_dir": str(current_dir),
                        "generated_at_utc": "2026-04-07T00:00:00+00:00",
                        "sub_artifacts": {
                            "prediction.json": {"materialization": "copied"},
                        },
                        "current_alias_contract": {
                            "alias_role": "production_current",
                            "top_level_report_role": "mirror_of_current_research_report",
                            "authoritative_paths": {
                                "runtime_config": str(current_dir / "config.json"),
                                "prediction": str(current_dir / "prediction.json"),
                                "forecast_summary": str(current_dir / "forecast_summary.json"),
                                "source": str(current_dir / "source.json"),
                                "research_report": str(current_dir / "research_report.md"),
                                "top_level_report": str(top_level_report),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_generate_research_report(output_dir, report_path=None):
                output_dir = Path(output_dir)
                prediction_payload = json.loads(
                    (output_dir / "prediction.json").read_text(encoding="utf-8")
                )
                analysis = {"prediction_anchor_time": prediction_payload["anchor_time"]}
                (output_dir / "analysis.json").write_text(
                    json.dumps(analysis),
                    encoding="utf-8",
                )
                report_text = f"# report\nanchor_time: {prediction_payload['anchor_time']}\n"
                (output_dir / "research_report.md").write_text(report_text, encoding="utf-8")
                if report_path is not None:
                    report_path = Path(report_path)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(report_text, encoding="utf-8")
                return analysis

            parser = build_parser()
            args = parser.parse_args(
                [
                    "predict",
                    "--output-dir",
                    str(current_dir),
                    "--csv",
                    str(csv_path),
                    "--apply-selected-policy-calibration",
                ]
            )
            with patch(
                "signal_cascade_pytorch.bootstrap.generate_research_report",
                side_effect=fake_generate_research_report,
            ):
                self.assertEqual(args.handler(args), 0)

            prediction_payload = json.loads(
                (current_dir / "prediction.json").read_text(encoding="utf-8")
            )
            forecast_payload = json.loads(
                (current_dir / "forecast_summary.json").read_text(encoding="utf-8")
            )
            config_payload = json.loads((current_dir / "config.json").read_text(encoding="utf-8"))
            metrics_payload = json.loads((current_dir / "metrics.json").read_text(encoding="utf-8"))
            validation_summary = json.loads(
                (current_dir / "validation_summary.json").read_text(encoding="utf-8")
            )
            source_payload = json.loads((current_dir / "source.json").read_text(encoding="utf-8"))
            manifest_payload = json.loads((current_dir / "manifest.json").read_text(encoding="utf-8"))
            analysis_payload = json.loads((current_dir / "analysis.json").read_text(encoding="utf-8"))
            report_text = (current_dir / "research_report.md").read_text(encoding="utf-8")

            self.assertEqual(prediction_payload["anchor_time"], latest_example.anchor_time.isoformat())
            self.assertEqual(forecast_payload["anchor_time"], prediction_payload["anchor_time"])
            self.assertEqual(config_payload["policy_cost_multiplier"], 6.0)
            self.assertEqual(config_payload["policy_gamma_multiplier"], 2.0)
            self.assertEqual(config_payload["min_policy_sigma"], 0.0001)
            self.assertEqual(config_payload["q_max"], 1.0)
            self.assertEqual(config_payload["cvar_weight"], 0.2)
            self.assertEqual(
                config_payload["evaluation_state_reset_mode"],
                "reset_each_session_or_window",
            )
            self.assertEqual(validation_summary["primary_state_reset_mode"], "reset_each_session_or_window")
            self.assertEqual(
                validation_summary["validation"],
                validation_summary["stateful_evaluation"]["reset_each_session_or_window"],
            )
            self.assertEqual(
                validation_summary["selection_diagnostics"]["validation"],
                validation_summary["validation"],
            )
            self.assertEqual(
                validation_summary["forecast_quality_scorecards"]["all_horizon"]["sample_count"],
                1,
            )
            self.assertIsNotNone(
                validation_summary["forecast_quality_scorecards"]["selected_horizon"][
                    "quality_score"
                ]
            )
            self.assertEqual(
                validation_summary["runtime_current"]["operating_point"]["cost_multiplier"],
                6.0,
            )
            self.assertEqual(
                validation_summary["runtime_current"]["operating_point_role"],
                "authoritative_runtime_config",
            )
            self.assertTrue(
                validation_summary["runtime_current"]["selection_alignment"][
                    "selected_row_matches_runtime"
                ]
            )
            self.assertEqual(metrics_payload["validation_metrics"], validation_summary["validation"])
            self.assertEqual(source_payload["generated_at_utc"], validation_summary["generated_at_utc"])
            self.assertEqual(manifest_payload["generated_at_utc"], validation_summary["generated_at_utc"])
            self.assertEqual(source_payload["path"], str((current_dir / "data_snapshot.csv").resolve()))
            self.assertEqual(source_payload["source_origin_path"], str(csv_path.resolve()))
            self.assertEqual(
                validation_summary["policy_calibration_summary"]["applied_runtime_policy"]["row_key"],
                validation_summary["policy_calibration_summary"]["selected_row"]["row_key"],
            )
            self.assertTrue(
                validation_summary["policy_calibration_summary"]["selected_row_matches_applied_runtime"]
            )
            self.assertEqual(
                source_payload["current_alias_contract"]["authoritative_paths"]["top_level_report"],
                str(top_level_report),
            )
            self.assertTrue((current_dir / "data_snapshot.csv").exists())
            self.assertEqual(
                top_level_report.read_text(encoding="utf-8"),
                report_text,
            )
            self.assertEqual(analysis_payload["prediction_anchor_time"], prediction_payload["anchor_time"])
            self.assertIn(prediction_payload["anchor_time"], report_text)
            self.assertEqual(
                source_payload["sub_artifacts"]["validation_summary.json"]["materialization"],
                "regenerated",
            )
            self.assertEqual(
                source_payload["sub_artifacts"]["prediction.json"]["materialization"],
                "regenerated",
            )
            self.assertEqual(
                source_payload["sub_artifacts"]["policy_summary.csv"]["materialization"],
                "copied",
            )
            self.assertEqual(
                source_payload["sub_artifacts"]["data_snapshot.csv"]["materialization"],
                "generated",
            )
            self.assertEqual(manifest_payload["entrypoints"]["validation_summary"], "validation_summary.json")
            self.assertEqual(manifest_payload["entrypoints"]["data_snapshot"], "data_snapshot.csv")

    def test_resolve_top_level_report_path_prefers_current_artifact_root_over_stale_authoritative_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            current_dir = Path(temp_dir) / "PyTorch" / "artifacts" / "gold_xauusd_m30" / "current"
            current_dir.mkdir(parents=True, exist_ok=True)
            source_payload = {
                "current_alias_contract": {
                    "authoritative_paths": {
                        "top_level_report": "/tmp/original-repo/PyTorch/report_signalcascade_xauusd.md",
                    }
                }
            }

            resolved = _resolve_top_level_report_path(current_dir, source_payload)

            self.assertEqual(
                resolved,
                current_dir.parent.parent.parent / "report_signalcascade_xauusd.md",
            )

    def test_prediction_serializer_adds_schema_and_median_semantics(self) -> None:
        payload = serialize_prediction_result(_prediction())

        self.assertEqual(payload["schema_version"], PREDICTION_SCHEMA_VERSION)
        self.assertEqual(payload["predicted_close_semantics"], "median_from_log_return")
        self.assertEqual(payload["g_t"], 0.8)
        self.assertEqual(payload["selected_policy_utility"], 0.05)
        self.assertEqual(payload["mu_t"]["3"], 0.02)
        self.assertAlmostEqual(payload["sigma_t_sq"]["3"], 0.0004, places=9)
        self.assertEqual(payload["median_predicted_closes"]["3"], 102.0)
        self.assertEqual(payload["effective_price_scale"], 100.0)
        self.assertEqual(payload["price_scale"], 100.0)
        self.assertEqual(payload["current_close_display"], 1.0)
        self.assertAlmostEqual(payload["median_predicted_close_display_by_horizon"]["3"], 1.02, places=9)
        self.assertEqual(payload["display_forecast"]["label"], "display forecast")
        self.assertEqual(payload["policy_driver"]["label"], "policy driver")
        self.assertEqual(
            payload["policy_driver"]["head_relationship"],
            "separate_policy_head",
        )
        self.assertEqual(
            payload["overlay_branch_contract"],
            "auxiliary_latent_branch_without_direct_supervision",
        )

    def test_report_analysis_tracks_artifact_versions(self) -> None:
        prediction = _prediction()
        config = TrainingConfig(horizons=(1, 3))
        prediction_payload = serialize_prediction_result(prediction)
        forecast_payload = build_forecast_summary_payload(
            prediction,
            config,
            generated_at_utc="2026-04-06T10:00:00+00:00",
        )

        self.assertEqual(forecast_payload["schema_version"], FORECAST_SCHEMA_VERSION)
        self.assertEqual(forecast_payload["generated_at"], "2026-04-06T10:00:00+00:00")
        self.assertEqual(forecast_payload["generated_at_utc"], "2026-04-06T10:00:00+00:00")
        self.assertIn("median_predicted_close", forecast_payload["forecast_rows"][0])
        self.assertIn("median_predicted_close_display", forecast_payload["forecast_rows"][0])
        self.assertIn("mu_t", forecast_payload["forecast_rows"][0])
        self.assertIn("sigma_t_sq", forecast_payload["forecast_rows"][0])
        self.assertEqual(forecast_payload["g_t"], 0.8)
        self.assertEqual(forecast_payload["selected_policy_utility"], 0.05)
        self.assertEqual(forecast_payload["effective_price_scale"], 100.0)
        self.assertAlmostEqual(forecast_payload["anchor_close_display"], 1.0, places=9)
        self.assertEqual(forecast_payload["display_forecast"]["label"], "display forecast")
        self.assertEqual(forecast_payload["policy_driver"]["label"], "policy driver")

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
                        "history": [
                            {
                                "epoch": 2.0,
                                "validation_exact_log_wealth": 0.01,
                                "validation_exact_cvar_tail_loss": 0.02,
                                "validation_selection_score": 0.02,
                            },
                            {
                                "epoch": 3.0,
                                "validation_exact_log_wealth": 0.03,
                                "validation_exact_cvar_tail_loss": 0.04,
                                "validation_selection_score": 0.04,
                            },
                        ],
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
                        "current_alias_contract": {
                            "alias_role": "production_current",
                            "top_level_report_role": "mirror_of_current_research_report",
                            "authoritative_paths": {
                                "runtime_config": str(output_dir / "config.json"),
                                "prediction": str(output_dir / "prediction.json"),
                                "forecast_summary": str(output_dir / "forecast_summary.json"),
                                "source": str(output_dir / "source.json"),
                                "research_report": str(output_dir / "research_report.md"),
                                "top_level_report": "/tmp/report_signalcascade_xauusd.md",
                            },
                            "runtime_contract": {
                                "diagnostic_recommendation_pointer": (
                                    "validation_summary.json.policy_calibration_summary.selected_row"
                                ),
                            },
                            "source_of_truth_summary": "current artifacts are authoritative",
                        },
                        "current_selection_governance": {
                            "selection_mode": "explicit_governance_override",
                            "selection_rule": "optimization_gate_then_blocked_objective",
                            "selection_rule_version": 1,
                            "decision_summary": "override applied",
                            "best_candidate": {"candidate": "candidate_05"},
                            "accepted_candidate": {"candidate": "candidate_05"},
                            "production_current": {"candidate": "candidate_04"},
                            "paired_frontier": {
                                "delta_production_minus_accepted": {
                                    "blocked_turnover_mean": -1.25,
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                        "generated_at_utc": "2026-04-06T04:40:00+00:00",
                        "primary_state_reset_mode": "reset_each_session_or_window",
                        "state_vector_summary": {
                            "shape_posterior_top_class_share": {"1": 1.0},
                        },
                        "stateful_evaluation": {"carry_on": {"average_log_wealth": 0.01}},
                        "blocked_walk_forward_evaluation": {
                            "fold_count": 3,
                            "fold_sample_counts": [1, 1, 1],
                            "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                            "state_reset_modes": {
                                "carry_on": {
                                    "fold_count": 3,
                                    "average_log_wealth_mean": 0.01,
                                    "average_log_wealth_min": 0.009,
                                    "average_log_wealth_max": 0.011,
                                    "turnover_mean": 0.2,
                                    "directional_accuracy_mean": 0.6,
                                    "exact_smooth_position_mae_mean": 0.03,
                                    "folds": [
                                        {
                                            "fold_index": 1,
                                            "sample_count": 1,
                                            "start_timestamp": "2026-04-06T00:00:00+00:00",
                                            "end_timestamp": "2026-04-06T00:00:00+00:00",
                                            "average_log_wealth": 0.009,
                                            "cvar_tail_loss": 0.01,
                                            "turnover": 0.2,
                                            "directional_accuracy": 0.6,
                                            "exact_smooth_position_mae": 0.03,
                                        }
                                    ],
                                }
                            },
                        },
                        "policy_calibration_summary": {
                            "row_count": 1,
                            "pareto_optimal_count": 1,
                            "dominated_count": 0,
                            "policy_calibration_rows_sha256": "rows1234",
                            "selection_basis": (
                                "pareto_rank_then_near_best_blocked_objective_mean_turnover_mean_"
                                "blocked_objective_mean_average_log_wealth_mean_cvar_tail_loss_mean_row_key"
                            ),
                            "selection_rule_version": 4,
                            "selected_row_key": (
                                "state_reset_mode=reset_each_session_or_window|cost_multiplier=4|"
                                "gamma_multiplier=2|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                            ),
                            "selected_row": {
                                "row_key": (
                                    "state_reset_mode=reset_each_session_or_window|cost_multiplier=4|"
                                    "gamma_multiplier=2|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                                ),
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 4.0,
                                "gamma_multiplier": 2.0,
                                "min_policy_sigma": 0.0001,
                                "q_max": 1.0,
                                "cvar_weight": 0.2,
                                "average_log_wealth": 0.01,
                                "blocked_average_log_wealth_mean": 0.009,
                                "blocked_cvar_tail_loss_mean": 0.02,
                                "blocked_turnover_mean": 0.4,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.005,
                            },
                            "best_row": {
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 4.0,
                                "gamma_multiplier": 2.0,
                                "min_policy_sigma": 0.0001,
                                "q_max": 1.0,
                                "cvar_weight": 0.2,
                                "average_log_wealth": 0.01,
                            },
                        },
                        "policy_calibration_sweep": [
                            {
                                "row_key": (
                                    "state_reset_mode=reset_each_session_or_window|cost_multiplier=1|"
                                    "gamma_multiplier=1|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                                ),
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 1.0,
                                "gamma_multiplier": 1.0,
                                "min_policy_sigma": 0.0001,
                                "q_max": 1.0,
                                "cvar_weight": 0.2,
                                "average_log_wealth": 0.01,
                                "objective_log_wealth_minus_lambda_cvar": 0.006,
                                "turnover": 0.4,
                                "cvar_tail_loss": 0.02,
                                "blocked_average_log_wealth_mean": 0.009,
                                "blocked_cvar_tail_loss_mean": 0.02,
                                "blocked_turnover_mean": 0.4,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.005,
                                "no_trade_band_hit_rate": 0.0,
                                "dominated": False,
                                "pareto_optimal": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")

            analysis = generate_research_report(output_dir)

        self.assertEqual(analysis["schema_version"], ANALYSIS_SCHEMA_VERSION)
        self.assertEqual(analysis["artifact_versions"]["prediction"], PREDICTION_SCHEMA_VERSION)
        self.assertEqual(analysis["artifact_provenance"]["artifact_kind"], "diagnostic_replay_overlay")
        self.assertEqual(analysis["artifact_provenance"]["artifact_id"], "overlay-123")
        self.assertEqual(analysis["artifact_provenance"]["git_head"], "abc123")
        self.assertEqual(analysis["artifact_provenance"]["git_commit_sha"], "abc123")
        self.assertEqual(analysis["artifact_provenance"]["git_tree_sha"], "cafe1234")
        self.assertEqual(analysis["dataset"]["purged_samples"], 2)
        self.assertEqual(analysis["training"]["best_epoch_by_exact_log_wealth"], 3.0)
        self.assertEqual(
            analysis["training"]["best_epoch_by_exact_log_wealth_minus_lambda_cvar"],
            3.0,
        )
        self.assertEqual(
            analysis["training"]["checkpoint_audit"]["selected_epoch_rank_by_exact_log_wealth"],
            2.0,
        )
        self.assertIn("carry_on", analysis["stateful_evaluation"])
        self.assertEqual(analysis["blocked_walk_forward_evaluation"]["fold_count"], 3)
        self.assertEqual(
            analysis["blocked_walk_forward_evaluation"]["best_state_reset_mode_by_mean_log_wealth"],
            "carry_on",
        )
        self.assertEqual(analysis["artifact_contract"]["alias_role"], "production_current")
        self.assertEqual(
            analysis["governance"]["selection_mode"],
            "explicit_governance_override",
        )
        self.assertEqual(
            analysis["claim_hardening"]["shape_top_class_collapse"]["dominant_share"],
            1.0,
        )
        self.assertEqual(analysis["policy_calibration_summary"]["pareto_optimal_count"], 1)
        self.assertEqual(analysis["policy_calibration_summary"]["selection_rule_version"], 4)
        self.assertEqual(analysis["forecast"]["g_t"], 0.8)
        self.assertEqual(analysis["forecast"]["selected_policy_utility"], 0.05)
        self.assertEqual(analysis["forecast"]["mu_t"]["3"], 0.02)
        self.assertAlmostEqual(analysis["forecast"]["sigma_t_sq"]["3"], 0.0004, places=9)
        self.assertEqual(
            analysis["policy_calibration_summary"]["selected_row"]["state_reset_mode"],
            "reset_each_session_or_window",
        )

    def test_generate_research_report_backfills_forecast_quality_ranking_diagnostics_from_leaderboard(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "current"
            output_dir.mkdir(parents=True, exist_ok=True)
            session_dir = Path(temp_dir) / "archive" / "session_20260408T075001Z"
            previous_session_dir = Path(temp_dir) / "archive" / "session_20260407T041853Z"
            candidate_01_dir = session_dir / "candidate_01"
            candidate_02_dir = session_dir / "candidate_02"
            candidate_03_dir = previous_session_dir / "candidate_03"
            candidate_04_dir = previous_session_dir / "candidate_04"
            candidate_01_dir.mkdir(parents=True, exist_ok=True)
            candidate_02_dir.mkdir(parents=True, exist_ok=True)
            candidate_03_dir.mkdir(parents=True, exist_ok=True)
            candidate_04_dir.mkdir(parents=True, exist_ok=True)

            config = TrainingConfig(horizons=(1, 3), output_dir=str(output_dir))
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "sample_count": 10,
                        "effective_sample_count": 8,
                        "train_samples": 6,
                        "validation_samples": 2,
                        "purged_samples": 2,
                        "source_rows_original": 10,
                        "source_rows_used": 10,
                        "validation_metrics": {
                            "average_log_wealth": 0.01,
                            "realized_pnl_per_anchor": 0.01,
                            "cvar_tail_loss": 0.02,
                            "max_drawdown": 0.03,
                            "no_trade_band_hit_rate": 0.1,
                            "mu_calibration": 0.04,
                            "sigma_calibration": 0.05,
                            "directional_accuracy": 0.6,
                            "project_value_score": 0.7,
                            "utility_score": 0.65,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "prediction.json").write_text(
                json.dumps(serialize_prediction_result(_prediction())),
                encoding="utf-8",
            )
            (output_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            (output_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                        "generated_at_utc": "2026-04-08T12:00:00+00:00",
                        "stateful_evaluation": {"carry_on": {"average_log_wealth": 0.01}},
                        "blocked_walk_forward_evaluation": {"fold_count": 2},
                        "policy_calibration_summary": {
                            "selected_row_matches_applied_runtime": True,
                        },
                        "forecast_quality_scorecards": {
                            "selected_horizon": {"quality_score": 0.62},
                            "all_horizon": {"quality_score": 0.41},
                            "quality_score_gap_all_minus_selected": -0.21,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
            (output_dir / "source.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "training_run",
                        "artifact_id": "current-artifact",
                        "artifact_dir": str(output_dir),
                        "current_alias_contract": {
                            "alias_role": "production_current",
                            "top_level_report_role": "mirror_of_current_research_report",
                        },
                        "current_selection_governance": {
                            "selection_mode": "accepted_candidate",
                            "selection_rule": "optimization_gate_then_deployment_score",
                            "selection_rule_version": 1,
                            "selection_leaderboard_path": str(session_dir / "leaderboard.json"),
                            "accepted_candidate": {"candidate": "candidate_02"},
                            "best_candidate": {"candidate": "candidate_02"},
                            "production_current": {"candidate": "candidate_02"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "leaderboard.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "candidate": "candidate_02",
                                "optimization_gate_passed": True,
                            },
                            {
                                "candidate": "candidate_01",
                                "optimization_gate_passed": True,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": "session_20260408T075001Z",
                        "generated_at_utc": "2026-04-08T07:50:01+00:00",
                        "leaderboard_path": str(session_dir / "leaderboard.json"),
                        "accepted_candidate": {"candidate": "candidate_02"},
                        "production_current_candidate": {"candidate": "candidate_02"},
                        "selection_status": "accepted_and_production_same",
                        "production_current_selection": {
                            "selection_mode": "accepted_candidate",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (candidate_01_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "forecast_quality_scorecards": {
                            "selected_horizon": {"quality_score": 0.81},
                            "all_horizon": {"quality_score": 0.30},
                            "quality_score_gap_all_minus_selected": -0.51,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (candidate_02_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "forecast_quality_scorecards": {
                            "selected_horizon": {"quality_score": 0.52},
                            "all_horizon": {"quality_score": 0.91},
                            "quality_score_gap_all_minus_selected": 0.39,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (previous_session_dir / "leaderboard.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "candidate": "candidate_03",
                                "optimization_gate_passed": True,
                            },
                            {
                                "candidate": "candidate_04",
                                "optimization_gate_passed": True,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (previous_session_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": "session_20260407T041853Z",
                        "generated_at_utc": "2026-04-07T04:18:53+00:00",
                        "leaderboard_path": str(previous_session_dir / "leaderboard.json"),
                        "accepted_candidate": {"candidate": "candidate_03"},
                        "production_current_candidate": {"candidate": "candidate_04"},
                        "selection_status": "accepted_and_production_diverged",
                        "production_current_selection": {
                            "selection_mode": "deployment_score_override",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (candidate_03_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "forecast_quality_scorecards": {
                            "selected_horizon": {"quality_score": 0.85},
                            "all_horizon": {"quality_score": 0.40},
                            "quality_score_gap_all_minus_selected": -0.45,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (candidate_04_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "forecast_quality_scorecards": {
                            "selected_horizon": {"quality_score": 0.22},
                            "all_horizon": {"quality_score": 0.95},
                            "quality_score_gap_all_minus_selected": 0.73,
                        }
                    }
                ),
                encoding="utf-8",
            )

            analysis = generate_research_report(output_dir)
            report_markdown = (output_dir / "research_report.md").read_text(encoding="utf-8")

        self.assertEqual(
            analysis["forecast_quality_ranking_diagnostics"][
                "selected_horizon_vs_current_spearman_rank_correlation"
            ],
            -1.0,
        )
        self.assertEqual(
            analysis["forecast_quality_ranking_diagnostics"][
                "all_horizon_vs_current_spearman_rank_correlation"
            ],
            1.0,
        )
        self.assertEqual(
            analysis["forecast_quality_ranking_diagnostics"]["accepted_candidate_selected_horizon_rank"],
            2,
        )
        self.assertEqual(analysis["selection_history_summary"]["session_count"], 2)
        self.assertEqual(
            analysis["selection_history_summary"]["accepted_vs_production_divergence_count"],
            1,
        )
        self.assertEqual(
            analysis["selection_history_summary"]["accepted_current_top_match_ratio"],
            1.0,
        )
        self.assertEqual(
            analysis["selection_history_summary"]["accepted_selected_horizon_top_match_ratio"],
            0.5,
        )
        self.assertEqual(
            analysis["selection_history_summary"]["accepted_all_horizon_top_match_ratio"],
            0.5,
        )
        self.assertEqual(
            analysis["selection_history_summary"]["production_current_top_match_ratio"],
            0.5,
        )
        self.assertEqual(
            analysis["selection_history_summary"]["production_selected_horizon_top_match_ratio"],
            0.0,
        )
        self.assertEqual(
            analysis["selection_history_summary"]["production_all_horizon_top_match_ratio"],
            1.0,
        )
        self.assertIn(
            "ranking split Spearman (selected/current / all/current / selected/all): `-1.000000` / `1.000000` / `-1.000000`",
            report_markdown,
        )
        self.assertIn(
            "history sessions / accepted / production / diverged: `2` / `2` / `2` / `1`",
            report_markdown,
        )

    def test_generate_research_report_builds_selection_divergence_scorecard_clusters(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "current"
            output_dir.mkdir(parents=True, exist_ok=True)
            latest_session_dir = Path(temp_dir) / "archive" / "session_20260408T075001Z"
            previous_session_dir = Path(temp_dir) / "archive" / "session_20260407T041853Z"
            for session_dir in (latest_session_dir, previous_session_dir):
                for candidate_name in ("candidate_01", "candidate_02", "candidate_03", "candidate_04"):
                    (session_dir / candidate_name).mkdir(parents=True, exist_ok=True)

            config = TrainingConfig(horizons=(1, 3), output_dir=str(output_dir))
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "sample_count": 10,
                        "effective_sample_count": 8,
                        "train_samples": 6,
                        "validation_samples": 2,
                        "purged_samples": 2,
                        "source_rows_original": 10,
                        "source_rows_used": 10,
                        "validation_metrics": {
                            "average_log_wealth": 0.01,
                            "realized_pnl_per_anchor": 0.01,
                            "cvar_tail_loss": 0.02,
                            "max_drawdown": 0.03,
                            "no_trade_band_hit_rate": 0.1,
                            "mu_calibration": 0.04,
                            "sigma_calibration": 0.05,
                            "directional_accuracy": 0.6,
                            "project_value_score": 0.7,
                            "utility_score": 0.65,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "prediction.json").write_text(
                json.dumps(serialize_prediction_result(_prediction())),
                encoding="utf-8",
            )
            (output_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            (output_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                        "generated_at_utc": "2026-04-08T12:00:00+00:00",
                        "stateful_evaluation": {"carry_on": {"average_log_wealth": 0.01}},
                        "blocked_walk_forward_evaluation": {"fold_count": 2},
                        "policy_calibration_summary": {
                            "selected_row_matches_applied_runtime": True,
                        },
                        "forecast_quality_scorecards": {
                            "selected_horizon": {"quality_score": 0.52},
                            "all_horizon": {"quality_score": 0.10},
                            "quality_score_gap_all_minus_selected": -0.42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
            (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
            (output_dir / "source.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "training_run",
                        "artifact_id": "current-artifact",
                        "artifact_dir": str(output_dir),
                        "current_alias_contract": {
                            "alias_role": "production_current",
                            "top_level_report_role": "mirror_of_current_research_report",
                        },
                        "current_selection_governance": {
                            "selection_mode": "accepted_candidate",
                            "selection_rule": "optimization_gate_then_deployment_score",
                            "selection_rule_version": 1,
                            "selection_leaderboard_path": str(latest_session_dir / "leaderboard.json"),
                            "accepted_candidate": {"candidate": "candidate_03"},
                            "best_candidate": {"candidate": "candidate_03"},
                            "production_current": {"candidate": "candidate_03"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (latest_session_dir / "leaderboard.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "candidate": "candidate_03",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": -0.0003,
                                "blocked_turnover_mean": 1.10,
                                "blocked_exact_smooth_position_mae_mean": 0.011,
                                "max_drawdown": 0.019,
                                "deployment_score": 0.66,
                                "user_value_score": 0.58,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                            {
                                "candidate": "candidate_01",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0015,
                                "blocked_turnover_mean": 0.70,
                                "blocked_exact_smooth_position_mae_mean": 0.040,
                                "max_drawdown": 0.041,
                                "deployment_score": 0.63,
                                "user_value_score": 0.54,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                            {
                                "candidate": "candidate_02",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0020,
                                "blocked_turnover_mean": 0.90,
                                "blocked_exact_smooth_position_mae_mean": 0.055,
                                "max_drawdown": 0.052,
                                "deployment_score": 0.61,
                                "user_value_score": 0.51,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                            {
                                "candidate": "candidate_04",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": -0.0008,
                                "blocked_turnover_mean": 1.30,
                                "blocked_exact_smooth_position_mae_mean": 0.080,
                                "max_drawdown": 0.060,
                                "deployment_score": 0.55,
                                "user_value_score": 0.49,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (latest_session_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": "session_20260408T075001Z",
                        "generated_at_utc": "2026-04-08T07:50:01+00:00",
                        "leaderboard_path": str(latest_session_dir / "leaderboard.json"),
                        "accepted_candidate": {"candidate": "candidate_03"},
                        "production_current_candidate": {"candidate": "candidate_03"},
                        "selection_status": "accepted_and_production_same",
                        "production_current_selection": {
                            "selection_mode": "accepted_candidate",
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_scorecards = {
                "candidate_01": (0.81, 0.70),
                "candidate_02": (0.40, 0.91),
                "candidate_03": (0.52, 0.10),
                "candidate_04": (0.20, 0.60),
            }
            for candidate_name, (selected_score, all_score) in latest_scorecards.items():
                (
                    latest_session_dir / candidate_name / "validation_summary.json"
                ).write_text(
                    json.dumps(
                        {
                            "forecast_quality_scorecards": {
                                "selected_horizon": {"quality_score": selected_score},
                                "all_horizon": {"quality_score": all_score},
                                "quality_score_gap_all_minus_selected": all_score - selected_score,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            (previous_session_dir / "leaderboard.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "candidate": "candidate_03",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0073,
                                "blocked_turnover_mean": 1.4590,
                                "blocked_exact_smooth_position_mae_mean": 0.1957,
                                "max_drawdown": 0.0887,
                                "deployment_score": 0.55,
                                "user_value_score": 0.56,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                            {
                                "candidate": "candidate_04",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0012,
                                "blocked_turnover_mean": 0.2413,
                                "blocked_exact_smooth_position_mae_mean": 0.0188,
                                "max_drawdown": 0.0170,
                                "deployment_score": 0.67,
                                "user_value_score": 0.67,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                            {
                                "candidate": "candidate_01",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0040,
                                "blocked_turnover_mean": 0.90,
                                "blocked_exact_smooth_position_mae_mean": 0.0400,
                                "max_drawdown": 0.0300,
                                "deployment_score": 0.62,
                                "user_value_score": 0.60,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                            {
                                "candidate": "candidate_02",
                                "optimization_gate_passed": True,
                                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0030,
                                "blocked_turnover_mean": 0.70,
                                "blocked_exact_smooth_position_mae_mean": 0.0300,
                                "max_drawdown": 0.0250,
                                "deployment_score": 0.61,
                                "user_value_score": 0.59,
                                "policy_cost_multiplier": 6.0,
                                "evaluation_state_reset_mode": "carry_on",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (previous_session_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": "session_20260407T041853Z",
                        "generated_at_utc": "2026-04-07T04:18:53+00:00",
                        "leaderboard_path": str(previous_session_dir / "leaderboard.json"),
                        "accepted_candidate": {"candidate": "candidate_03"},
                        "production_current_candidate": {"candidate": "candidate_04"},
                        "selection_status": "accepted_and_production_diverged",
                        "production_current_selection": {
                            "selection_mode": "auto_user_value_selection",
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous_scorecards = {
                "candidate_01": (0.80, 0.60),
                "candidate_02": (0.50, 0.55),
                "candidate_03": (0.90, 0.95),
                "candidate_04": (0.10, 0.20),
            }
            for candidate_name, (selected_score, all_score) in previous_scorecards.items():
                (
                    previous_session_dir / candidate_name / "validation_summary.json"
                ).write_text(
                    json.dumps(
                        {
                            "forecast_quality_scorecards": {
                                "selected_horizon": {"quality_score": selected_score},
                                "all_horizon": {"quality_score": all_score},
                                "quality_score_gap_all_minus_selected": all_score - selected_score,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            analysis = generate_research_report(output_dir)
            report_markdown = (output_dir / "research_report.md").read_text(encoding="utf-8")

        current_ranking = analysis["forecast_quality_ranking_diagnostics"]
        self.assertEqual(current_ranking["current_top_candidate"], "candidate_03")
        self.assertEqual(current_ranking["selected_horizon_top_candidate"], "candidate_01")
        self.assertEqual(current_ranking["all_horizon_top_candidate"], "candidate_02")
        self.assertEqual(current_ranking["accepted_candidate_current_rank"], 1)
        self.assertEqual(current_ranking["accepted_candidate_selected_horizon_rank"], 2)
        self.assertEqual(current_ranking["accepted_candidate_all_horizon_rank"], 4)

        divergence_scorecard = analysis["selection_divergence_scorecard"]
        self.assertEqual(divergence_scorecard["session_count"], 2)
        self.assertEqual(divergence_scorecard["full_coverage_session_count"], 2)
        self.assertEqual(divergence_scorecard["partial_coverage_session_count"], 0)
        self.assertEqual(
            divergence_scorecard["cluster_counts"],
            {
                "objective_evaluation_mismatch": 1,
                "stability_override": 1,
            },
        )
        latest_record = next(
            row
            for row in divergence_scorecard["rows"]
            if row["session_id"] == "session_20260408T075001Z"
        )
        previous_record = next(
            row
            for row in divergence_scorecard["rows"]
            if row["session_id"] == "session_20260407T041853Z"
        )
        self.assertEqual(latest_record["failure_mode_cluster"], "objective_evaluation_mismatch")
        self.assertEqual(latest_record["accepted_candidate_all_minus_current_rank"], 3)
        self.assertEqual(previous_record["failure_mode_cluster"], "stability_override")
        self.assertTrue(previous_record["override_flag"])
        self.assertEqual(previous_record["production_current_all_minus_current_rank"], 2)
        self.assertIn(
            "divergence scorecard clusters (all sessions): `objective_evaluation_mismatch=1, stability_override=1`",
            report_markdown,
        )
        self.assertIn(
            "session_20260408T075001Z [full/objective_evaluation_mismatch]",
            report_markdown,
        )
        self.assertIn(
            "session_20260407T041853Z [full/stability_override]",
            report_markdown,
        )

    def test_generate_research_report_requires_published_current_diagnostics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "current"
            output_dir.mkdir(parents=True, exist_ok=True)
            config = TrainingConfig(horizons=(1, 3), output_dir=str(output_dir))
            (output_dir / "metrics.json").write_text(
                json.dumps({"schema_version": 4, "validation_metrics": {}}),
                encoding="utf-8",
            )
            (output_dir / "prediction.json").write_text(
                json.dumps(serialize_prediction_result(_prediction())),
                encoding="utf-8",
            )
            (output_dir / "config.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "diagnostics unpublished for current artifact",
            ):
                generate_research_report(output_dir)


if __name__ == "__main__":
    unittest.main()
