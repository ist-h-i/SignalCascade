from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.diagnostics_service import (
    DIAGNOSTICS_SCHEMA_VERSION,
    export_review_diagnostics,
)
from signal_cascade_pytorch.application.policy_service import (
    apply_selection_policy,
    build_exact_policy_rows,
    build_policy_path_terms,
    smooth_policy_distribution,
    solve_exact_policy_position,
)
from signal_cascade_pytorch.application.training_service import (
    _checkpoint_selection_score,
    _fit_model,
    examples_to_batch,
    train_model,
)
from signal_cascade_pytorch.application.tuning_service import (
    _build_user_value_metrics,
    _build_candidate_parameters,
    _load_parameter_seed,
    _prioritize_quick_mode_candidates,
    _select_production_current_candidate,
    _resolve_session_candidates,
    tune_latest_dataset,
)
from signal_cascade_pytorch.bootstrap import (
    audit_checkpoints_command,
    _emit_cli_compat_warnings,
    _load_config_with_overrides,
    _materialize_replay_artifact,
    _resolve_source_payload,
    tune_latest_command,
)
from signal_cascade_pytorch.domain.entities import (
    PredictionResult,
    STATE_FEATURE_NAMES,
    STATE_VECTOR_COMPONENT_NAMES,
    TIMEFRAME_FEATURE_NAMES,
    TimeframeFeatureRow,
    TrainingExample,
)
from signal_cascade_pytorch.domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES
from signal_cascade_pytorch.infrastructure.ml.model import SignalCascadeModel
from signal_cascade_pytorch.interfaces.cli import build_parser


def _example(
    *,
    returns_target: tuple[float, ...],
    realized_volatility: float = 0.02,
) -> TrainingExample:
    main_sequences = {
        timeframe: [(0.01, 0.2, 0.1, 0.05, 0.0, 0.3)] * 4 for timeframe in MAIN_TIMEFRAMES
    }
    overlay_sequences = {
        timeframe: [(0.01, 0.2, 0.1, 0.05, 0.0, 0.3)] * 4 for timeframe in OVERLAY_TIMEFRAMES
    }
    main_shape_targets = {timeframe: (0.2, 0.3, 0.5) for timeframe in MAIN_TIMEFRAMES}
    horizon_costs = tuple(0.001 for _ in returns_target)
    return TrainingExample(
        anchor_time=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc),
        main_sequences=main_sequences,
        overlay_sequences=overlay_sequences,
        main_shape_targets=main_shape_targets,
        state_features=(1.0, 0.0, 0.0, 0.1, 0.25, 0.02, 0.03, 0.1, 0.25, 0.4),
        returns_target=returns_target,
        long_mae=tuple(0.0 for _ in returns_target),
        short_mae=tuple(0.0 for _ in returns_target),
        long_mfe=tuple(0.0 for _ in returns_target),
        short_mfe=tuple(0.0 for _ in returns_target),
        direction_targets=tuple(1 if value > 0 else -1 if value < 0 else 0 for value in returns_target),
        direction_weights=tuple(1.0 for _ in returns_target),
        direction_thresholds=tuple(0.01 for _ in returns_target),
        direction_mae_thresholds=tuple(0.01 for _ in returns_target),
        horizon_costs=horizon_costs,
        overlay_target=0,
        current_close=100.0,
        regime_id="asia|low|trend",
        regime_features=(1.0, 0.0, 0.0, 0.1, 0.25),
        realized_volatility=realized_volatility,
        trend_strength=0.25,
    )


class PolicyAndTrainingTests(unittest.TestCase):
    def test_exact_policy_hits_no_trade_band_when_margin_is_inside_cost(self) -> None:
        config = TrainingConfig(horizons=(1,))

        position, no_trade = solve_exact_policy_position(
            gated_mean=0.0015,
            sigma=0.02,
            previous_position=0.1,
            cost=0.002,
            config=config,
        )

        self.assertTrue(no_trade)
        self.assertAlmostEqual(position, 0.1, places=6)

    def test_apply_selection_policy_prefers_highest_utility_horizon(self) -> None:
        config = TrainingConfig(horizons=(1, 3))
        example = _example(returns_target=(0.02, 0.03))

        decision = apply_selection_policy(
            example=example,
            mean=(0.01, 0.03),
            sigma=(0.02, 0.02),
            config=config,
            previous_position=0.0,
            tradeability_gate=0.9,
            shape_probs=(0.4, 0.2, 0.1, 0.1, 0.1, 0.1),
        )

        self.assertEqual(decision["policy_horizon"], 3)
        self.assertGreater(float(decision["selected_policy_utility"]), 0.0)
        self.assertNotIn("selection_score", decision)
        self.assertGreater(float(decision["legacy_compatibility"]["selection_score"]), 0.0)

    def test_apply_selection_policy_uses_config_policy_multipliers_by_default(self) -> None:
        example = _example(returns_target=(0.02, 0.03))
        tuned_config = TrainingConfig(
            horizons=(1, 3),
            policy_cost_multiplier=0.5,
            policy_gamma_multiplier=0.5,
            q_max=1.25,
        )
        baseline_config = TrainingConfig(horizons=(1, 3))

        tuned = apply_selection_policy(
            example=example,
            mean=(0.01, 0.03),
            sigma=(0.02, 0.02),
            config=tuned_config,
            previous_position=0.0,
            tradeability_gate=0.9,
            shape_probs=(0.4, 0.2, 0.1, 0.1, 0.1, 0.1),
        )
        baseline = apply_selection_policy(
            example=example,
            mean=(0.01, 0.03),
            sigma=(0.02, 0.02),
            config=baseline_config,
            previous_position=0.0,
            tradeability_gate=0.9,
            shape_probs=(0.4, 0.2, 0.1, 0.1, 0.1, 0.1),
        )

        self.assertGreater(abs(float(tuned["position"])), abs(float(baseline["position"])))

    def test_smooth_policy_distribution_returns_finite_outputs(self) -> None:
        config = TrainingConfig(horizons=(1, 3))
        policy = smooth_policy_distribution(
            mean=torch.tensor([[0.01, 0.02]], dtype=torch.float32),
            sigma=torch.tensor([[0.02, 0.03]], dtype=torch.float32),
            costs=torch.tensor([[0.001, 0.002]], dtype=torch.float32),
            tradeability_gate=torch.tensor([0.8], dtype=torch.float32),
            previous_position=torch.tensor([0.0], dtype=torch.float32),
            config=config,
        )

        self.assertTrue(torch.isfinite(policy["combined_position"]).all())
        self.assertTrue(torch.isfinite(policy["combined_utility"]).all())
        self.assertAlmostEqual(float(policy["horizon_weights"].sum().item()), 1.0, places=6)

    def test_exact_policy_rows_share_path_terms_with_differentiable_path(self) -> None:
        config = TrainingConfig(horizons=(1, 3))
        mean = torch.tensor([[0.01, 0.03]], dtype=torch.float32)
        sigma = torch.tensor([[0.02, 0.04]], dtype=torch.float32)
        costs = torch.tensor([[0.001, 0.002]], dtype=torch.float32)
        tradeability_gate = torch.tensor([0.8], dtype=torch.float32)
        previous_position = torch.tensor([0.1], dtype=torch.float32)

        path_terms = build_policy_path_terms(
            mean=mean,
            sigma=sigma,
            costs=costs,
            tradeability_gate=tradeability_gate,
            previous_position=previous_position,
            config=config,
        )
        rows = build_exact_policy_rows(
            mean=tuple(float(value) for value in mean[0].tolist()),
            sigma=tuple(float(value) for value in sigma[0].tolist()),
            costs=tuple(float(value) for value in costs[0].tolist()),
            tradeability_gate=float(tradeability_gate[0].item()),
            previous_position=float(previous_position[0].item()),
            config=config,
        )

        for index, row in enumerate(rows):
            self.assertAlmostEqual(float(row["g_t"]), float(path_terms["g_t"][0, index].item()), places=6)
            self.assertAlmostEqual(
                float(row["mu_t_tilde"]),
                float(path_terms["mu_t_tilde"][0, index].item()),
                places=6,
            )
            self.assertAlmostEqual(
                float(row["sigma_sq"]),
                float(path_terms["sigma_sq"][0, index].item()),
                places=6,
            )
            self.assertAlmostEqual(
                float(row["margin"]),
                float(path_terms["margin"][0, index].item()),
                places=6,
            )

    def test_examples_to_batch_includes_state_features(self) -> None:
        config = TrainingConfig(horizons=(1, 3))
        batch = examples_to_batch([_example(returns_target=(0.01, 0.02))], config)

        self.assertEqual(tuple(batch["state_features"].shape), (1, 10))
        self.assertEqual(tuple(batch["returns"].shape), (1, 2))
        self.assertEqual(tuple(batch["horizon_costs"].shape), (1, 2))

    def test_training_example_exposes_named_state_feature_contract(self) -> None:
        example = _example(returns_target=(0.01, 0.02))

        self.assertEqual(tuple(example.state_feature_map.keys()), STATE_FEATURE_NAMES)
        self.assertEqual(example.state_feature_map["session_asia"], 1.0)
        self.assertAlmostEqual(
            example.state_feature_map["realized_volatility_30m"],
            example.realized_volatility,
            places=9,
        )

    def test_timeframe_feature_row_exposes_named_bridge_inputs(self) -> None:
        row = TimeframeFeatureRow(
            timestamp=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc),
            close=100.0,
            shape=(0.2, 0.3, 0.5),
            vector=(0.01, 0.2, 0.1, 0.05, 0.0, 0.3),
        )

        self.assertEqual(tuple(row.feature_map.keys()), TIMEFRAME_FEATURE_NAMES)
        self.assertAlmostEqual(row.feature_map["ell_log_return"], 0.01, places=9)
        self.assertAlmostEqual(row.feature_map["zeta_ema_deviation"], 0.3, places=9)

    def test_timeframe_feature_row_rejects_contract_drift(self) -> None:
        with self.assertRaises(ValueError):
            TimeframeFeatureRow(
                timestamp=datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc),
                close=100.0,
                shape=(0.2, 0.3, 0.5),
                vector=(0.01, 0.2, 0.1),
            )

    def test_model_forward_emits_shape_probs_and_state_vector(self) -> None:
        config = TrainingConfig(horizons=(1, 3), shape_classes=4)
        example = _example(returns_target=(0.01, 0.02))
        batch = examples_to_batch([example], config)
        model = SignalCascadeModel(
            feature_dim=len(example.main_sequences["4h"][0]),
            state_feature_dim=len(example.state_features),
            hidden_dim=8,
            state_dim=6,
            num_horizons=len(config.horizons),
            shape_classes=config.shape_classes,
            branch_dilations=(1, 2),
            dropout=0.0,
        )

        outputs = model(batch["main"], batch["overlay"], batch["state_features"])
        state_components = outputs["state_vector_components"]
        expert_outputs = outputs["shape_conditioned_experts"]

        self.assertEqual(tuple(outputs["mu"].shape), (1, 2))
        self.assertEqual(tuple(outputs["sigma"].shape), (1, 2))
        self.assertEqual(tuple(outputs["forecast_mu"].shape), (1, 2))
        self.assertEqual(tuple(outputs["forecast_sigma"].shape), (1, 2))
        self.assertEqual(tuple(outputs["policy_mu"].shape), (1, 2))
        self.assertEqual(tuple(outputs["policy_sigma"].shape), (1, 2))
        self.assertEqual(tuple(outputs["shape_probs"].shape), (1, 4))
        self.assertAlmostEqual(float(outputs["shape_probs"].sum().item()), 1.0, places=5)
        self.assertTrue(torch.equal(outputs["shape_probs"], outputs["shape_posterior"]))
        self.assertEqual(tuple(outputs["state_vector"].shape[:1]), (1,))
        self.assertEqual(tuple(state_components.keys()), STATE_VECTOR_COMPONENT_NAMES)
        self.assertEqual(tuple(outputs["state_vector_component_dims"].keys()), STATE_VECTOR_COMPONENT_NAMES)
        self.assertTrue(torch.equal(outputs["shape_feature"], state_components["h_t"]))
        self.assertTrue(torch.equal(outputs["shape_posterior"], state_components["s_t"]))
        self.assertTrue(torch.equal(outputs["state_projection"], state_components["z_t"]))
        self.assertTrue(torch.equal(outputs["memory_state"], state_components["m_t"]))
        self.assertEqual(tuple(expert_outputs.keys()), ("mu_by_shape", "sigma_by_shape", "mixture_weights"))
        self.assertEqual(tuple(outputs["expert_mu_by_shape"].shape), (1, 2, 4))
        self.assertEqual(tuple(outputs["expert_sigma_by_shape"].shape), (1, 2, 4))
        self.assertEqual(tuple(outputs["main_shape_predictions"].keys()), MAIN_TIMEFRAMES)
        reconstructed_state_vector = torch.cat(
            [state_components[name] for name in STATE_VECTOR_COMPONENT_NAMES],
            dim=1,
        )
        self.assertTrue(torch.allclose(outputs["state_vector"], reconstructed_state_vector))

    def test_model_can_tie_policy_head_to_forecast_head(self) -> None:
        config = TrainingConfig(horizons=(1, 3), shape_classes=4)
        example = _example(returns_target=(0.01, 0.02))
        batch = examples_to_batch([example], config)
        model = SignalCascadeModel(
            feature_dim=len(example.main_sequences["4h"][0]),
            state_feature_dim=len(example.state_features),
            hidden_dim=8,
            state_dim=6,
            num_horizons=len(config.horizons),
            shape_classes=config.shape_classes,
            branch_dilations=(1, 2),
            dropout=0.0,
            tie_policy_to_forecast_head=True,
        )

        outputs = model(batch["main"], batch["overlay"], batch["state_features"])

        self.assertTrue(torch.allclose(outputs["policy_mu"], outputs["forecast_mu"]))
        self.assertTrue(torch.allclose(outputs["policy_sigma"], outputs["forecast_sigma"]))
        self.assertTrue(bool(outputs["policy_head_tied_to_forecast"].item()))

    def test_model_can_disable_overlay_branch(self) -> None:
        config = TrainingConfig(horizons=(1, 3), shape_classes=4)
        example = _example(returns_target=(0.01, 0.02))
        batch = examples_to_batch([example], config)
        model = SignalCascadeModel(
            feature_dim=len(example.main_sequences["4h"][0]),
            state_feature_dim=len(example.state_features),
            hidden_dim=8,
            state_dim=6,
            num_horizons=len(config.horizons),
            shape_classes=config.shape_classes,
            branch_dilations=(1, 2),
            dropout=0.0,
            disable_overlay_branch=True,
        )
        overlay_shifted = {
            timeframe: tensor + torch.full_like(tensor, 5.0)
            for timeframe, tensor in batch["overlay"].items()
        }

        baseline = model(batch["main"], batch["overlay"], batch["state_features"])
        shifted = model(batch["main"], overlay_shifted, batch["state_features"])

        self.assertTrue(torch.allclose(baseline["forecast_mu"], shifted["forecast_mu"]))
        self.assertTrue(torch.allclose(baseline["policy_mu"], shifted["policy_mu"]))
        self.assertTrue(torch.allclose(baseline["shape_probs"], shifted["shape_probs"]))
        self.assertTrue(bool(baseline["overlay_branch_disabled"].item()))

    def test_prediction_result_exposes_legacy_compatibility_view(self) -> None:
        prediction = PredictionResult(
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

        legacy = prediction.legacy_compatibility

        self.assertEqual(legacy["proposed_horizon"], 3)
        self.assertEqual(legacy["accepted_horizon"], 3)
        self.assertTrue(legacy["accepted_signal"])
        self.assertEqual(legacy["selection_probability"], 0.8)
        self.assertNotIn("legacy_compatibility", asdict(prediction))

    def test_export_diagnostics_writes_new_schema_summary(self) -> None:
        config = TrainingConfig(horizons=(1,))
        example = _example(returns_target=(0.01,))
        summary_payload = {
            "validation_rows": [
                {"sample_id": 0, "timestamp": example.anchor_time.isoformat(), "horizon": 1}
            ],
            "policy_summary": [
                {"sample_id": 0, "policy_horizon": 1, "executed_horizon": 1}
            ],
            "horizon_diag": [
                {"horizon": 1, "sample_count": 1, "selection_rate": 1.0}
            ],
            "summary": {
                "anchor_sample_count": 1,
                "executed_trade_count": 1,
                "no_trade_count": 0,
                "average_log_wealth": 0.01,
                "realized_pnl_per_anchor": 0.01,
                "turnover": 0.2,
                "max_drawdown": 0.0,
                "cvar_alpha": 0.1,
                "cvar_tail_loss": 0.0,
                "no_trade_band_hit_rate": 0.0,
                "expert_entropy": 0.4,
                "shape_gate_usage": 0.7,
                "mu_calibration": 0.01,
                "sigma_calibration": 0.01,
                "policy_horizon_distribution": {"1": 1.0},
            },
        }

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch(
                "signal_cascade_pytorch.application.diagnostics_service.build_validation_diagnostics",
                return_value=summary_payload,
            ):
                summary = export_review_diagnostics(
                    output_dir=output_dir,
                    model=None,
                    examples=[example],
                    config=config,
                    checkpoint_audit={
                        "selected_epoch": 12.0,
                        "best_epoch_by_exact_log_wealth": 13.0,
                        "delta_to_best_exact_log_wealth": 0.0065,
                    },
                    selection_policy=None,
                )

            self.assertEqual(summary["diagnostics_schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
            self.assertEqual(summary["policy_mode"], "shape_aware_profit_maximization")
            self.assertEqual(summary["checkpoint_audit"]["best_epoch_by_exact_log_wealth"], 13.0)
            self.assertTrue((output_dir / "policy_summary.csv").exists())
            saved_summary = json.loads((output_dir / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["validation"]["executed_trade_count"], 1)
            self.assertEqual(saved_summary["checkpoint_audit"]["selected_epoch"], 12.0)

    def test_export_diagnostics_acceptance_threshold_mode_alias_maps_to_selection_threshold_mode(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "export-diagnostics",
                "--acceptance-threshold-mode",
                "none",
            ]
        )

        self.assertEqual(args.selection_threshold_mode, "none")

    def test_resolve_source_payload_falls_back_to_artifact_snapshot_without_source_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "data_snapshot.csv").write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
            (output_dir / "config.json").write_text(
                json.dumps({"requested_price_scale": 100.0}),
                encoding="utf-8",
            )

            payload = _resolve_source_payload(
                SimpleNamespace(csv=None, csv_lookback_days=None, price_scale=None),
                output_dir,
            )

        self.assertEqual(payload["kind"], "csv")
        self.assertTrue(str(payload["path"]).endswith("data_snapshot.csv"))
        self.assertEqual(payload["requested_price_scale"], 100.0)
        self.assertEqual(payload["effective_price_scale"], 100.0)

    def test_audit_checkpoints_command_writes_checkpoint_audit_json(self) -> None:
        config = TrainingConfig(
            cvar_weight=0.2,
            checkpoint_selection_metric="exact_log_wealth_minus_lambda_cvar",
        )
        history = [
            {
                "epoch": 12.0,
                "validation_selection_score": 0.07089334746508921,
                "validation_exact_log_wealth": 0.005911248423024029,
                "validation_exact_cvar_tail_loss": 0.0200,
            },
            {
                "epoch": 13.0,
                "validation_selection_score": 0.07189334746508921,
                "validation_exact_log_wealth": 0.01240889878719627,
                "validation_exact_cvar_tail_loss": 0.0140,
            },
            {
                "epoch": 14.0,
                "validation_selection_score": 0.07289334746508921,
                "validation_exact_log_wealth": 0.01140840458293162,
                "validation_exact_cvar_tail_loss": 0.0210,
            },
        ]

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            (output_dir / "metrics.json").write_text(
                json.dumps({"best_epoch": 12.0, "history": history}),
                encoding="utf-8",
            )

            exit_code = audit_checkpoints_command(
                SimpleNamespace(output_dir=str(output_dir), audit_output=None)
            )

            payload = json.loads((output_dir / "checkpoint_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["summary"]["selected_epoch"], 12.0)
        self.assertEqual(payload["summary"]["best_epoch_by_exact_log_wealth"], 13.0)
        self.assertEqual(
            payload["summary"]["best_epoch_by_exact_log_wealth_minus_lambda_cvar"],
            13.0,
        )
        self.assertEqual(payload["top_epochs_by_exact_log_wealth"][0]["epoch"], 13.0)

    def test_audit_checkpoints_command_marks_cvar_unavailable_for_legacy_history(self) -> None:
        config = TrainingConfig()
        history = [
            {
                "epoch": 12.0,
                "validation_selection_score": 0.07089334746508921,
                "validation_exact_log_wealth": 0.005911248423024029,
            },
            {
                "epoch": 13.0,
                "validation_selection_score": 0.07189334746508921,
                "validation_exact_log_wealth": 0.01240889878719627,
            },
        ]

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            (output_dir / "metrics.json").write_text(
                json.dumps({"best_epoch": 12.0, "history": history}),
                encoding="utf-8",
            )

            audit_checkpoints_command(SimpleNamespace(output_dir=str(output_dir), audit_output=None))
            payload = json.loads((output_dir / "checkpoint_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(
            payload["exact_log_wealth_minus_lambda_cvar_unavailable_reason"],
            "validation_exact_cvar_tail_loss_missing_in_history",
        )
        self.assertEqual(payload["summary"]["best_epoch_by_exact_log_wealth"], 13.0)

    def test_cli_compat_warnings_emit_for_deprecated_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "export-diagnostics",
                "--selection-threshold-mode",
                "replay",
                "--selection-score-source",
                "selector_probability",
            ]
        )

        with patch("builtins.print") as print_mock:
            _emit_cli_compat_warnings(args)

        self.assertEqual(print_mock.call_count, 2)

    def test_audit_checkpoints_cli_accepts_audit_output_override(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "audit-checkpoints",
                "--output-dir",
                "artifacts/demo",
                "--audit-output",
                "artifacts/demo/custom_audit.json",
            ]
        )

        self.assertEqual(args.output_dir, "artifacts/demo")
        self.assertEqual(args.audit_output, "artifacts/demo/custom_audit.json")

    def test_train_cli_accepts_checkpoint_selection_metric_override(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "train",
                "--checkpoint-selection-metric",
                "exact_log_wealth",
                "--policy-cost-multiplier",
                "0.5",
                "--policy-gamma-multiplier",
                "2.0",
                "--min-policy-sigma",
                "0.0002",
                "--tie-policy-to-forecast-head",
                "--disable-overlay-branch",
            ]
        )

        self.assertEqual(args.checkpoint_selection_metric, "exact_log_wealth")
        self.assertEqual(args.policy_cost_multiplier, 0.5)
        self.assertEqual(args.policy_gamma_multiplier, 2.0)
        self.assertEqual(args.min_policy_sigma, 0.0002)
        self.assertTrue(args.tie_policy_to_forecast_head)
        self.assertTrue(args.disable_overlay_branch)

    def test_predict_cli_accepts_apply_selected_policy_calibration(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "predict",
                "--apply-selected-policy-calibration",
            ]
        )

        self.assertTrue(args.apply_selected_policy_calibration)

    def test_checkpoint_selection_metric_defaults_to_exact_log_wealth_minus_lambda_cvar(self) -> None:
        config = TrainingConfig()

        self.assertEqual(
            config.checkpoint_selection_metric,
            "exact_log_wealth_minus_lambda_cvar",
        )

    def test_checkpoint_selection_score_can_use_exact_wealth_minus_lambda_cvar(self) -> None:
        config = TrainingConfig(
            checkpoint_selection_metric="exact_log_wealth_minus_lambda_cvar",
            cvar_weight=0.2,
        )

        score = _checkpoint_selection_score(
            average_log_wealth=0.012,
            cvar_tail=0.01,
            forecast_mae=10.0,
            sigma_calibration=10.0,
            exact_smooth_position_mae=10.0,
            config=config,
        )

        self.assertAlmostEqual(score, -0.01, places=9)

    def test_tune_latest_dataset_accepts_tunable_overrides(self) -> None:
        validation_metrics = {
            "project_value_score": 0.4,
            "utility_score": 0.3,
            "average_log_wealth": 0.02,
            "realized_pnl_per_anchor": 0.01,
            "cvar_tail_loss": 0.05,
            "max_drawdown": 0.01,
            "no_trade_band_hit_rate": 0.25,
            "mu_calibration": 0.01,
            "sigma_calibration": 0.02,
            "directional_accuracy": 0.75,
        }
        summary = {
            "best_validation_loss": 0.9,
            "validation_metrics": validation_metrics,
        }
        diagnostics_summary = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "generated_at_utc": "2026-04-06T04:40:00+00:00",
            "validation": validation_metrics,
        }
        prediction = PredictionResult(
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

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                return_value=(object(), summary),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                return_value=prediction,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 6,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "reset_each_session_or_window",
                        "min_policy_sigma": 2e-4,
                        "policy_cost_multiplier": 0.5,
                        "policy_gamma_multiplier": 0.5,
                        "q_max": 1.25,
                        "cvar_weight": 0.1,
                        "tie_policy_to_forecast_head": True,
                        "disable_overlay_branch": True,
                    }
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"epochs": 6, "horizons": (1, 3)},
                )

            self.assertEqual(manifest["best_candidate"]["epochs"], 6)
            self.assertEqual(
                manifest["best_candidate"]["evaluation_state_reset_mode"],
                "reset_each_session_or_window",
            )
            self.assertEqual(manifest["best_candidate"]["policy_cost_multiplier"], 0.5)
            self.assertEqual(manifest["best_candidate"]["policy_gamma_multiplier"], 0.5)
            self.assertEqual(manifest["best_candidate"]["q_max"], 1.25)
            self.assertEqual(manifest["best_candidate"]["cvar_weight"], 0.1)
            self.assertEqual(manifest["optimization_gate"]["status"], "passed")
            self.assertIsNotNone(manifest["accepted_candidate"])
            self.assertTrue(manifest["current_updated"])
            saved_config = json.loads(
                (artifact_root / "current" / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config["epochs"], 6)
            self.assertEqual(saved_config["horizons"], [1, 3])
            self.assertEqual(
                saved_config["evaluation_state_reset_mode"],
                "reset_each_session_or_window",
            )
            self.assertEqual(saved_config["policy_cost_multiplier"], 0.5)
            self.assertEqual(saved_config["policy_gamma_multiplier"], 0.5)
            self.assertEqual(saved_config["q_max"], 1.25)
            self.assertEqual(saved_config["cvar_weight"], 0.1)
            self.assertTrue(saved_config["tie_policy_to_forecast_head"])
            self.assertTrue(saved_config["disable_overlay_branch"])
            best_params = json.loads((artifact_root / "best_params.json").read_text(encoding="utf-8"))
            self.assertEqual(
                best_params["parameters"]["evaluation_state_reset_mode"],
                "reset_each_session_or_window",
            )
            self.assertEqual(best_params["parameters"]["min_policy_sigma"], 2e-4)
            self.assertEqual(best_params["parameters"]["policy_cost_multiplier"], 0.5)
            self.assertEqual(best_params["parameters"]["policy_gamma_multiplier"], 0.5)
            self.assertEqual(best_params["parameters"]["q_max"], 1.25)
            self.assertEqual(best_params["parameters"]["cvar_weight"], 0.1)
            self.assertTrue(best_params["parameters"]["tie_policy_to_forecast_head"])
            self.assertTrue(best_params["parameters"]["disable_overlay_branch"])

    def test_tune_latest_dataset_prefers_candidate_with_better_blocked_objective(self) -> None:
        prediction = PredictionResult(
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
        summary_rows = [
            {
                "best_validation_loss": 0.5,
                "validation_metrics": {
                    "project_value_score": 0.8,
                    "utility_score": 0.6,
                    "average_log_wealth": 0.03,
                    "realized_pnl_per_anchor": 0.02,
                    "cvar_tail_loss": 0.02,
                    "max_drawdown": 0.01,
                    "no_trade_band_hit_rate": 0.2,
                    "mu_calibration": 0.01,
                    "sigma_calibration": 0.01,
                    "directional_accuracy": 0.75,
                },
            },
            {
                "best_validation_loss": 0.6,
                "validation_metrics": {
                    "project_value_score": 0.5,
                    "utility_score": 0.4,
                    "average_log_wealth": 0.02,
                    "realized_pnl_per_anchor": 0.015,
                    "cvar_tail_loss": 0.02,
                    "max_drawdown": 0.01,
                    "no_trade_band_hit_rate": 0.2,
                    "mu_calibration": 0.01,
                    "sigma_calibration": 0.01,
                    "directional_accuracy": 0.75,
                },
            },
        ]
        diagnostics_rows = [
            {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "generated_at_utc": "2026-04-06T04:40:00+00:00",
                "validation": {"average_log_wealth": 0.03},
                "blocked_walk_forward_evaluation": {
                    "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                    "state_reset_modes": {
                        "carry_on": {
                            "average_log_wealth_mean": 0.002,
                            "turnover_mean": 3.0,
                            "directional_accuracy_mean": 0.6,
                            "exact_smooth_position_mae_mean": 0.2,
                            "folds": [
                                {"cvar_tail_loss": 0.05},
                                {"cvar_tail_loss": 0.05},
                                {"cvar_tail_loss": 0.05},
                            ],
                        }
                    },
                },
            },
            {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "generated_at_utc": "2026-04-06T04:40:00+00:00",
                "validation": {"average_log_wealth": 0.02},
                "blocked_walk_forward_evaluation": {
                    "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                        "state_reset_modes": {
                            "carry_on": {
                                "average_log_wealth_mean": 0.02,
                                "turnover_mean": 1.0,
                                "directional_accuracy_mean": 0.7,
                                "exact_smooth_position_mae_mean": 0.03,
                                "folds": [
                                    {"cvar_tail_loss": 0.01},
                                    {"cvar_tail_loss": 0.01},
                                    {"cvar_tail_loss": 0.01},
                                ],
                        }
                    },
                },
            },
        ]

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
            summary_iter = iter(summary_rows)
            diagnostics_iter = iter(diagnostics_rows)

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                diagnostics_summary = next(diagnostics_iter)
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                side_effect=lambda *args, **kwargs: (object(), next(summary_iter)),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                return_value=prediction,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 6,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                    {
                        "epochs": 12,
                        "batch_size": 16,
                        "learning_rate": 5e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"horizons": (1, 3)},
                )

        self.assertEqual(manifest["best_candidate"]["candidate"], "candidate_02")
        self.assertEqual(manifest["accepted_candidate"]["candidate"], "candidate_02")
        self.assertAlmostEqual(
            manifest["best_candidate"]["blocked_objective_log_wealth_minus_lambda_cvar_mean"],
            0.018,
            places=9,
        )

    def test_select_production_current_candidate_prefers_user_value_over_blocked_objective(self) -> None:
        leaderboard = [
            {
                "candidate": "candidate_05",
                "optimization_gate_passed": True,
                "user_value_score": 0.5600,
                "project_value_score": 0.54,
                "average_log_wealth": 0.0075,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0073,
                "blocked_directional_accuracy_mean": 0.7222,
                "mu_calibration": 0.0512,
                "sigma_calibration": 0.0272,
                "blocked_exact_smooth_position_mae_mean": 0.1957,
                "max_drawdown": 0.0887,
                "blocked_turnover_mean": 1.4590,
            },
            {
                "candidate": "candidate_04",
                "optimization_gate_passed": True,
                "user_value_score": 0.6110,
                "project_value_score": 0.6041,
                "average_log_wealth": 0.0013,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0009,
                "blocked_directional_accuracy_mean": 0.7778,
                "mu_calibration": 0.1552,
                "sigma_calibration": 0.1749,
                "blocked_exact_smooth_position_mae_mean": 0.0172,
                "max_drawdown": 0.0142,
                "blocked_turnover_mean": 0.2098,
            },
        ]

        production = _select_production_current_candidate(leaderboard)

        self.assertIsNotNone(production)
        self.assertEqual(production["candidate"], "candidate_04")

    def test_build_user_value_metrics_penalizes_extreme_forecast_swings(self) -> None:
        stable_candidate = {
            "candidate": "candidate_05",
            "average_log_wealth": 0.0075,
            "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0073,
            "blocked_directional_accuracy_mean": 0.7222,
            "mu_calibration": 0.0512,
            "sigma_calibration": 0.0272,
            "blocked_exact_smooth_position_mae_mean": 0.1957,
            "max_drawdown": 0.0887,
            "blocked_turnover_mean": 1.4590,
            "forecast_return_pct_jump_max": 0.0844,
            "forecast_long_horizon_return_pct_abs_max": 0.0467,
        }
        unstable_candidate = {
            "candidate": "candidate_17",
            "average_log_wealth": 0.0019,
            "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0012,
            "blocked_directional_accuracy_mean": 0.8056,
            "mu_calibration": 0.1815,
            "sigma_calibration": 0.1340,
            "blocked_exact_smooth_position_mae_mean": 0.0188,
            "max_drawdown": 0.0170,
            "blocked_turnover_mean": 0.2413,
            "forecast_return_pct_jump_max": 0.4844,
            "forecast_long_horizon_return_pct_abs_max": 0.2430,
        }

        stable_metrics = _build_user_value_metrics(stable_candidate)
        unstable_metrics = _build_user_value_metrics(unstable_candidate)

        self.assertGreater(
            stable_metrics["user_value_forecast_stability_score"],
            unstable_metrics["user_value_forecast_stability_score"],
        )
        self.assertGreater(
            stable_metrics["user_value_score"],
            unstable_metrics["user_value_score"],
        )

    def test_select_production_current_candidate_prefers_stable_curve_when_user_value_is_recomputed(self) -> None:
        stable_candidate = {
            "candidate": "candidate_05",
            "optimization_gate_passed": True,
            "project_value_score": 0.54,
            "average_log_wealth": 0.0075,
            "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0073,
            "blocked_directional_accuracy_mean": 0.7222,
            "mu_calibration": 0.0512,
            "sigma_calibration": 0.0272,
            "blocked_exact_smooth_position_mae_mean": 0.1957,
            "max_drawdown": 0.0887,
            "blocked_turnover_mean": 1.4590,
            "forecast_return_pct_jump_max": 0.0844,
            "forecast_long_horizon_return_pct_abs_max": 0.0467,
        }
        unstable_candidate = {
            "candidate": "candidate_17",
            "optimization_gate_passed": True,
            "project_value_score": 0.6066,
            "average_log_wealth": 0.0019,
            "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.0012,
            "blocked_directional_accuracy_mean": 0.8056,
            "mu_calibration": 0.1815,
            "sigma_calibration": 0.1340,
            "blocked_exact_smooth_position_mae_mean": 0.0188,
            "max_drawdown": 0.0170,
            "blocked_turnover_mean": 0.2413,
            "forecast_return_pct_jump_max": 0.4844,
            "forecast_long_horizon_return_pct_abs_max": 0.2430,
        }
        stable_candidate.update(_build_user_value_metrics(stable_candidate))
        unstable_candidate.update(_build_user_value_metrics(unstable_candidate))

        production = _select_production_current_candidate(
            [unstable_candidate, stable_candidate]
        )

        self.assertIsNotNone(production)
        self.assertEqual(production["candidate"], "candidate_05")

    def test_tune_latest_dataset_updates_current_with_deployment_score_override(self) -> None:
        base_prediction = {
            "anchor_time": "2026-03-24T00:00:00+00:00",
            "current_close": 100.0,
            "previous_position": 0.0,
            "no_trade_band_hit": False,
            "tradeability_gate": 0.8,
            "shape_entropy": 0.3,
            "policy_score": 0.05,
            "expected_log_returns": {"1": 0.01, "3": 0.02},
            "predicted_closes": {"1": 101.0, "3": 102.0},
            "uncertainties": {"1": 0.01, "3": 0.02},
            "horizon_utilities": {"1": 0.01, "3": 0.05},
            "horizon_positions": {"1": 0.1, "3": 0.4},
            "shape_probabilities": {"0": 0.4, "1": 0.6},
            "regime_id": "asia|low|trend",
        }
        predictions = iter(
            [
                PredictionResult(
                    policy_horizon=30,
                    executed_horizon=30,
                    position=0.55,
                    trade_delta=0.45,
                    **base_prediction,
                ),
                PredictionResult(
                    policy_horizon=2,
                    executed_horizon=2,
                    position=0.43,
                    trade_delta=0.21,
                    **base_prediction,
                ),
            ]
        )
        summary_rows = iter(
            [
                {
                    "best_validation_loss": 0.55,
                    "validation_metrics": {
                        "project_value_score": 0.6041,
                        "utility_score": 0.58,
                        "average_log_wealth": 0.0013,
                        "realized_pnl_per_anchor": 0.0012,
                        "cvar_tail_loss": 0.0031,
                        "max_drawdown": 0.0142,
                        "no_trade_band_hit_rate": 0.44,
                        "mu_calibration": 0.1552,
                        "sigma_calibration": 0.0749,
                        "directional_accuracy": 0.7778,
                        "turnover": 0.5869,
                        "exact_smooth_position_mae": 0.0161,
                    },
                },
                {
                    "best_validation_loss": 0.42,
                    "validation_metrics": {
                        "project_value_score": 0.5400,
                        "utility_score": 0.55,
                        "average_log_wealth": 0.0075,
                        "realized_pnl_per_anchor": 0.0077,
                        "cvar_tail_loss": 0.0159,
                        "max_drawdown": 0.0887,
                        "no_trade_band_hit_rate": 0.56,
                        "mu_calibration": 0.0512,
                        "sigma_calibration": 0.0272,
                        "directional_accuracy": 0.6111,
                        "turnover": 2.7757,
                        "exact_smooth_position_mae": 0.0417,
                    },
                },
            ]
        )
        diagnostics_rows = iter(
            [
                {
                    "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                    "generated_at_utc": "2026-04-06T04:40:00+00:00",
                    "validation": {
                        "average_log_wealth": 0.0013,
                        "project_value_score": 0.6041,
                        "utility_score": 0.58,
                        "mu_calibration": 0.1552,
                        "sigma_calibration": 0.0749,
                        "directional_accuracy": 0.7778,
                        "exact_smooth_position_mae": 0.0161,
                        "max_drawdown": 0.0142,
                        "turnover": 0.5869,
                        "no_trade_band_hit_rate": 0.44,
                    },
                    "blocked_walk_forward_evaluation": {
                        "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                        "state_reset_modes": {
                            "carry_on": {
                                "average_log_wealth_mean": 0.0012,
                                "turnover_mean": 0.2098,
                                "directional_accuracy_mean": 0.7778,
                                "exact_smooth_position_mae_mean": 0.0172,
                                "folds": [
                                    {"cvar_tail_loss": 0.0014},
                                    {"cvar_tail_loss": 0.0014},
                                    {"cvar_tail_loss": 0.0014},
                                ],
                            }
                        },
                    },
                },
                {
                    "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                    "generated_at_utc": "2026-04-06T04:40:00+00:00",
                    "validation": {
                        "average_log_wealth": 0.0075,
                        "project_value_score": 0.5400,
                        "utility_score": 0.55,
                        "mu_calibration": 0.0512,
                        "sigma_calibration": 0.0272,
                        "directional_accuracy": 0.6111,
                        "exact_smooth_position_mae": 0.1617,
                        "max_drawdown": 0.0887,
                        "turnover": 2.7757,
                        "no_trade_band_hit_rate": 0.56,
                    },
                    "blocked_walk_forward_evaluation": {
                        "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                        "state_reset_modes": {
                            "carry_on": {
                                "average_log_wealth_mean": 0.0096,
                                "turnover_mean": 1.4590,
                                "directional_accuracy_mean": 0.7222,
                                "exact_smooth_position_mae_mean": 0.0457,
                                "folds": [
                                    {"cvar_tail_loss": 0.0114},
                                    {"cvar_tail_loss": 0.0114},
                                    {"cvar_tail_loss": 0.0114},
                                ],
                            }
                        },
                    },
                },
            ]
        )

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                diagnostics_summary = next(diagnostics_rows)
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                side_effect=lambda *args, **kwargs: (object(), next(summary_rows)),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                side_effect=lambda *args, **kwargs: next(predictions),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 14,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                    {
                        "epochs": 16,
                        "batch_size": 16,
                        "learning_rate": 5e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"horizons": (1, 3)},
                )

            self.assertEqual(manifest["accepted_candidate"]["candidate"], "candidate_02")
            self.assertEqual(manifest["production_current_candidate"]["candidate"], "candidate_01")
            self.assertEqual(
                manifest["production_current_selection"]["selection_mode"],
                "deployment_score_override",
            )
            self.assertEqual(
                manifest["production_current_selection"]["selection_status"],
                "accepted_and_production_diverged",
            )
            current_config = json.loads((artifact_root / "current" / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(current_config["epochs"], 14)
            best_params = json.loads((artifact_root / "best_params.json").read_text(encoding="utf-8"))
            self.assertEqual(best_params["parameters"]["epochs"], 16)
            current_source = json.loads((artifact_root / "current" / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(
                current_source["current_selection_governance"]["selection_mode"],
                "deployment_score_override",
            )
            self.assertEqual(
                current_source["current_selection_governance"]["selection_status"],
                "accepted_and_production_diverged",
            )
            self.assertEqual(
                current_source["current_selection_governance"]["accepted_candidate"]["candidate"],
                "candidate_02",
            )
            self.assertEqual(
                current_source["current_selection_governance"]["production_current"]["candidate"],
                "candidate_01",
            )

    def test_tune_latest_dataset_keeps_current_when_optimization_gate_fails(self) -> None:
        validation_metrics = {
            "project_value_score": 0.1,
            "utility_score": 0.05,
            "average_log_wealth": -0.01,
            "realized_pnl_per_anchor": -0.02,
            "cvar_tail_loss": 0.2,
            "max_drawdown": 0.3,
            "no_trade_band_hit_rate": 0.9,
            "mu_calibration": 0.02,
            "sigma_calibration": 0.03,
            "directional_accuracy": 0.4,
        }
        summary = {
            "best_validation_loss": 1.2,
            "validation_metrics": validation_metrics,
        }
        diagnostics_summary = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "generated_at_utc": "2026-04-06T04:40:00+00:00",
            "validation": validation_metrics,
        }
        prediction = PredictionResult(
            anchor_time="2026-03-24T00:00:00+00:00",
            current_close=100.0,
            policy_horizon=1,
            executed_horizon=None,
            previous_position=0.0,
            position=0.0,
            trade_delta=0.0,
            no_trade_band_hit=True,
            tradeability_gate=0.4,
            shape_entropy=0.3,
            policy_score=0.0,
            expected_log_returns={"1": -0.01, "3": -0.02},
            predicted_closes={"1": 99.0, "3": 98.0},
            uncertainties={"1": 0.02, "3": 0.03},
            horizon_utilities={"1": -0.01, "3": -0.02},
            horizon_positions={"1": 0.0, "3": 0.0},
            shape_probabilities={"0": 0.4, "1": 0.6},
            regime_id="asia|low|trend",
        )

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            current_dir = artifact_root / "current"
            current_dir.mkdir()
            (current_dir / "marker.txt").write_text("keep", encoding="utf-8")
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                return_value=(object(), summary),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                return_value=prediction,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 6,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                    }
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"epochs": 6, "horizons": (1, 3)},
                )

            self.assertEqual(manifest["optimization_gate"]["status"], "failed")
            self.assertIsNone(manifest["accepted_candidate"])
            self.assertFalse(manifest["current_updated"])
            self.assertTrue(manifest["interrupted_tuning"])
            self.assertTrue((current_dir / "marker.txt").exists())
            self.assertFalse((artifact_root / "best_params.json").exists())
            session_manifest = json.loads(
                Path(manifest["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(session_manifest["optimization_gate"]["status"], "failed")

    def test_tune_latest_dataset_respects_candidate_limit(self) -> None:
        validation_metrics = {
            "project_value_score": 0.4,
            "utility_score": 0.3,
            "average_log_wealth": 0.02,
            "realized_pnl_per_anchor": 0.01,
            "cvar_tail_loss": 0.05,
            "max_drawdown": 0.01,
            "no_trade_band_hit_rate": 0.25,
            "mu_calibration": 0.01,
            "sigma_calibration": 0.02,
            "directional_accuracy": 0.75,
        }
        summary = {
            "best_validation_loss": 0.9,
            "validation_metrics": validation_metrics,
        }
        diagnostics_summary = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "generated_at_utc": "2026-04-06T04:40:00+00:00",
            "validation": validation_metrics,
        }
        prediction = PredictionResult(
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

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            train_calls = 0

            def fake_train_model(*_args, **_kwargs):
                nonlocal train_calls
                train_calls += 1
                return object(), summary

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                side_effect=fake_train_model,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                return_value=prediction,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 6,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                    {
                        "epochs": 12,
                        "batch_size": 16,
                        "learning_rate": 5e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                    {
                        "epochs": 18,
                        "batch_size": 16,
                        "learning_rate": 3e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"horizons": (1, 3)},
                    candidate_limit=2,
                )

            leaderboard_payload = json.loads(
                Path(manifest["leaderboard_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(train_calls, 2)
            self.assertEqual(manifest["candidate_limit"], 2)
            self.assertFalse(manifest["quick_mode"])
            self.assertEqual(manifest["generated_candidate_count"], 3)
            self.assertEqual(manifest["evaluated_candidate_count"], 2)
            self.assertEqual(manifest["optimization_gate"]["candidate_count"], 2)
            self.assertEqual(leaderboard_payload["generated_candidate_count"], 3)
            self.assertEqual(leaderboard_payload["evaluated_candidate_count"], 2)
            self.assertEqual(len(leaderboard_payload["results"]), 2)

    def test_load_parameter_seed_prefers_current_config_over_policy_calibration_selected_row(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            current_dir = artifact_root / "current"
            current_dir.mkdir(parents=True)
            (current_dir / "config.json").write_text(
                json.dumps(
                    TrainingConfig(
                        output_dir=str(current_dir),
                        evaluation_state_reset_mode="carry_on",
                        policy_cost_multiplier=6.0,
                        policy_gamma_multiplier=4.0,
                        min_policy_sigma=1e-4,
                        q_max=0.75,
                        cvar_weight=0.2,
                        tie_policy_to_forecast_head=True,
                        disable_overlay_branch=False,
                    ).to_dict()
                ),
                encoding="utf-8",
            )
            (current_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "policy_calibration_summary": {
                            "selected_row": {
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 0.5,
                                "gamma_multiplier": 0.5,
                                "min_policy_sigma": 0.0002,
                                "q_max": 1.25,
                                "cvar_weight": 0.1,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            seed = _load_parameter_seed(artifact_root)

        self.assertEqual(seed["evaluation_state_reset_mode"], "carry_on")
        self.assertEqual(seed["policy_cost_multiplier"], 6.0)
        self.assertEqual(seed["policy_gamma_multiplier"], 4.0)
        self.assertEqual(seed["min_policy_sigma"], 0.0001)
        self.assertEqual(seed["q_max"], 0.75)
        self.assertEqual(seed["cvar_weight"], 0.2)
        self.assertTrue(seed["tie_policy_to_forecast_head"])
        self.assertFalse(seed["disable_overlay_branch"])

    def test_load_config_with_overrides_can_apply_selected_policy_calibration(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "config.json").write_text(
                json.dumps(
                    TrainingConfig(
                        output_dir=str(output_dir),
                        evaluation_state_reset_mode="carry_on",
                        policy_cost_multiplier=1.0,
                        policy_gamma_multiplier=1.0,
                        min_policy_sigma=1e-4,
                        q_max=1.0,
                        cvar_weight=0.2,
                    ).to_dict()
                ),
                encoding="utf-8",
            )
            (output_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "policy_calibration_summary": {
                            "selected_row": {
                                "state_reset_mode": "reset_each_session_or_window",
                                "cost_multiplier": 2.0,
                                "gamma_multiplier": 2.0,
                                "min_policy_sigma": 0.0002,
                                "q_max": 1.25,
                                "cvar_weight": 0.1,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "predict",
                    "--output-dir",
                    str(output_dir),
                    "--apply-selected-policy-calibration",
                ]
            )

            config = _load_config_with_overrides(output_dir, args)

        self.assertEqual(config.evaluation_state_reset_mode, "reset_each_session_or_window")
        self.assertEqual(config.policy_cost_multiplier, 2.0)
        self.assertEqual(config.policy_gamma_multiplier, 2.0)
        self.assertEqual(config.min_policy_sigma, 0.0002)
        self.assertEqual(config.q_max, 1.25)
        self.assertEqual(config.cvar_weight, 0.1)

    def test_tune_cli_accepts_candidate_limit_and_quick_mode(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "tune-latest",
                "--candidate-limit",
                "3",
                "--quick-mode",
                "--warm-start-from-current",
            ]
        )

        self.assertEqual(args.candidate_limit, 3)
        self.assertTrue(args.quick_mode)
        self.assertTrue(args.warm_start_from_current)

    def test_tune_latest_command_forwards_csv_lookback_days(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            manifest = {
                "current_dir": str(artifact_root / "current"),
                "archive_session_dir": str(artifact_root / "archive" / "session_20260407T000000Z"),
                "leaderboard_path": str(artifact_root / "archive" / "session_20260407T000000Z" / "leaderboard.json"),
                "best_candidate": {
                    "candidate": "candidate_01",
                    "best_validation_loss": -0.1,
                    "project_value_score": 0.2,
                    "average_log_wealth": 0.01,
                    "cvar_tail_loss": 0.005,
                    "policy_horizon": 6,
                },
                "accepted_candidate": {
                    "candidate": "candidate_01",
                },
                "production_current_candidate": {
                    "candidate": "candidate_01",
                },
                "optimization_gate": {
                    "status": "passed",
                    "passed_candidate_count": 1,
                    "candidate_count": 1,
                },
            }
            args = SimpleNamespace(
                artifact_root=str(artifact_root),
                csv=None,
                seed=7,
                csv_lookback_days=360,
                candidate_limit=None,
                quick_mode=False,
            )

            with patch(
                "signal_cascade_pytorch.bootstrap.tune_latest_dataset",
                return_value=manifest,
            ) as mocked:
                exit_code = tune_latest_command(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(mocked.call_args.kwargs["lookback_days"], 360)
            self.assertFalse(mocked.call_args.kwargs["warm_start_from_current"])
            self.assertEqual(
                mocked.call_args.kwargs["csv_path"],
                artifact_root.resolve() / "live" / "xauusd_m30_latest.csv",
            )

    def test_tune_latest_command_forwards_warm_start_from_current(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            manifest = {
                "current_dir": str(artifact_root / "current"),
                "archive_session_dir": str(artifact_root / "archive" / "session_20260407T000000Z"),
                "leaderboard_path": str(artifact_root / "archive" / "session_20260407T000000Z" / "leaderboard.json"),
                "best_candidate": {
                    "candidate": "candidate_01",
                    "best_validation_loss": -0.1,
                    "project_value_score": 0.2,
                    "average_log_wealth": 0.01,
                    "cvar_tail_loss": 0.005,
                    "policy_horizon": 6,
                },
                "accepted_candidate": {
                    "candidate": "candidate_01",
                },
                "production_current_candidate": {
                    "candidate": "candidate_01",
                },
                "optimization_gate": {
                    "status": "passed",
                    "passed_candidate_count": 1,
                    "candidate_count": 1,
                },
            }
            args = SimpleNamespace(
                artifact_root=str(artifact_root),
                csv=None,
                seed=7,
                csv_lookback_days=360,
                candidate_limit=None,
                quick_mode=False,
                warm_start_from_current=True,
            )

            with patch(
                "signal_cascade_pytorch.bootstrap.tune_latest_dataset",
                return_value=manifest,
            ) as mocked:
                exit_code = tune_latest_command(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue(mocked.call_args.kwargs["warm_start_from_current"])

    def test_tune_latest_command_reports_current_not_updated_for_quick_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            manifest = {
                "current_dir": str(artifact_root / "current"),
                "archive_session_dir": str(artifact_root / "archive" / "session_20260407T000000Z"),
                "leaderboard_path": str(artifact_root / "archive" / "session_20260407T000000Z" / "leaderboard.json"),
                "best_candidate": {
                    "candidate": "candidate_01",
                    "best_validation_loss": -0.1,
                    "project_value_score": 0.2,
                    "average_log_wealth": 0.01,
                    "cvar_tail_loss": 0.005,
                    "policy_horizon": 6,
                },
                "accepted_candidate": {
                    "candidate": "candidate_01",
                },
                "production_current_candidate": None,
                "current_updated": False,
                "optimization_gate": {
                    "status": "passed",
                    "passed_candidate_count": 1,
                    "candidate_count": 1,
                },
            }
            args = SimpleNamespace(
                artifact_root=str(artifact_root),
                csv=None,
                seed=7,
                csv_lookback_days=360,
                candidate_limit=None,
                quick_mode=True,
            )

            stdout = io.StringIO()
            with patch(
                "signal_cascade_pytorch.bootstrap.tune_latest_dataset",
                return_value=manifest,
            ) as mocked, redirect_stdout(stdout):
                exit_code = tune_latest_command(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue(mocked.called)
            self.assertIn("accepted candidate: candidate_01", stdout.getvalue())
            self.assertIn("production current candidate: none", stdout.getvalue())
            self.assertIn("current updated: False", stdout.getvalue())

    def test_resolve_session_candidates_uses_quick_mode_default_limit(self) -> None:
        candidates = [{"epochs": index} for index in range(6)]

        evaluated, resolved_limit = _resolve_session_candidates(
            candidates,
            candidate_limit=None,
            quick_mode=True,
        )

        self.assertEqual(resolved_limit, len(candidates))
        self.assertEqual(len(evaluated), len(candidates))
        self.assertEqual(evaluated[0]["epochs"], 0)

    def test_tune_latest_dataset_warm_starts_only_compatible_candidates(self) -> None:
        summary_rows = iter(
            [
                {
                    "best_validation_loss": 0.8,
                    "validation_metrics": {
                        "project_value_score": 0.62,
                        "utility_score": 0.55,
                        "average_log_wealth": 0.022,
                        "realized_pnl_per_anchor": 0.010,
                        "cvar_tail_loss": 0.015,
                        "max_drawdown": 0.03,
                        "no_trade_band_hit_rate": 0.1,
                        "mu_calibration": 0.013,
                        "sigma_calibration": 0.021,
                        "directional_accuracy": 0.64,
                    },
                },
                {
                    "best_validation_loss": 0.8,
                    "validation_metrics": {
                        "project_value_score": 0.60,
                        "utility_score": 0.53,
                        "average_log_wealth": 0.020,
                        "realized_pnl_per_anchor": 0.009,
                        "cvar_tail_loss": 0.016,
                        "max_drawdown": 0.035,
                        "no_trade_band_hit_rate": 0.1,
                        "mu_calibration": 0.015,
                        "sigma_calibration": 0.023,
                        "directional_accuracy": 0.62,
                    },
                },
            ]
        )
        diagnostics_template = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "generated_at_utc": "2026-04-06T04:40:00+00:00",
            "validation": {"average_log_wealth": 0.022},
            "blocked_walk_forward_evaluation": {
                "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                "state_reset_modes": {
                    "carry_on": {
                        "average_log_wealth_mean": 0.022,
                        "turnover_mean": 1.1,
                        "directional_accuracy_mean": 0.64,
                        "exact_smooth_position_mae_mean": 0.16,
                        "folds": [
                            {"cvar_tail_loss": 0.013},
                            {"cvar_tail_loss": 0.013},
                            {"cvar_tail_loss": 0.013},
                        ],
                    }
                },
            },
        }
        predictions = iter(
            [
                PredictionResult(
                    anchor_time="2026-03-24T00:00:00+00:00",
                    current_close=100.0,
                    policy_horizon=1,
                    executed_horizon=1,
                    previous_position=0.0,
                    position=0.12,
                    trade_delta=0.12,
                    no_trade_band_hit=False,
                    tradeability_gate=0.8,
                    shape_entropy=0.3,
                    policy_score=0.05,
                    expected_log_returns={"1": 0.01},
                    predicted_closes={"1": 101.0},
                    uncertainties={"1": 0.01},
                    horizon_utilities={"1": 0.01},
                    horizon_positions={"1": 0.12},
                    shape_probabilities={"0": 0.4, "1": 0.6},
                    regime_id="asia|low|trend",
                ),
                PredictionResult(
                    anchor_time="2026-03-24T00:00:00+00:00",
                    current_close=100.0,
                    policy_horizon=1,
                    executed_horizon=1,
                    previous_position=0.0,
                    position=0.12,
                    trade_delta=0.12,
                    no_trade_band_hit=False,
                    tradeability_gate=0.8,
                    shape_entropy=0.3,
                    policy_score=0.05,
                    expected_log_returns={"1": 0.01},
                    predicted_closes={"1": 101.0},
                    uncertainties={"1": 0.01},
                    horizon_utilities={"1": 0.01},
                    horizon_positions={"1": 0.12},
                    shape_probabilities={"0": 0.4, "1": 0.6},
                    regime_id="asia|low|trend",
                ),
            ]
        )

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            current_dir = artifact_root / "current"
            current_dir.mkdir(parents=True)
            (current_dir / "model.pt").write_bytes(b"checkpoint")
            current_config = TrainingConfig(
                output_dir=str(current_dir),
                horizons=(1,),
                hidden_dim=48,
            )
            (current_dir / "config.json").write_text(
                json.dumps(current_config.to_dict()),
                encoding="utf-8",
            )
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
            warm_start_paths: list[str | None] = []

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_template),
                    encoding="utf-8",
                )
                return diagnostics_template

            def fake_train_model(_examples, _config, _output_dir, *, warm_start_checkpoint_path=None):
                warm_start_paths.append(
                    None
                    if warm_start_checkpoint_path is None
                    else str(warm_start_checkpoint_path)
                )
                return object(), next(summary_rows)

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                side_effect=fake_train_model,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                side_effect=lambda *args, **kwargs: next(predictions),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 14,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                    {
                        "epochs": 14,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 64,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    },
                ],
            ):
                tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"horizons": (1,)},
                    candidate_limit=2,
                    warm_start_from_current=True,
                )

        self.assertEqual(warm_start_paths[0], str((current_dir / "model.pt").resolve()))
        self.assertIsNone(warm_start_paths[1])

    def test_quick_mode_prioritizes_structural_ablation_candidates(self) -> None:
        candidates = _build_candidate_parameters(
            {
                "epochs": 14,
                "batch_size": 32,
                "learning_rate": 5e-4,
                "hidden_dim": 32,
                "dropout": 0.2,
                "weight_decay": 2.5e-5,
                "evaluation_state_reset_mode": "carry_on",
                "min_policy_sigma": 1e-4,
                "policy_cost_multiplier": 6.0,
                "policy_gamma_multiplier": 4.0,
                "q_max": 0.75,
                "cvar_weight": 0.2,
                "tie_policy_to_forecast_head": True,
                "disable_overlay_branch": False,
            }
        )

        prioritized = _prioritize_quick_mode_candidates(candidates)
        structural_pairs = [
            (
                bool(candidate["tie_policy_to_forecast_head"]),
                bool(candidate["disable_overlay_branch"]),
            )
            for candidate in prioritized[:4]
        ]

        self.assertEqual(
            structural_pairs,
            [(False, False), (True, False), (False, True), (True, True)],
        )
        self.assertEqual(prioritized[0]["policy_cost_multiplier"], 6.0)
        self.assertEqual(prioritized[0]["policy_gamma_multiplier"], 4.0)

    def test_tune_latest_dataset_marks_quick_mode_as_non_promotable(self) -> None:
        summary_rows = iter(
            [
                {
                    "best_validation_loss": 0.8,
                    "validation_metrics": {
                        "project_value_score": 0.62,
                        "utility_score": 0.55,
                        "average_log_wealth": 0.022,
                        "realized_pnl_per_anchor": 0.010,
                        "cvar_tail_loss": 0.015,
                        "max_drawdown": 0.03,
                        "no_trade_band_hit_rate": 0.1,
                        "mu_calibration": 0.013,
                        "sigma_calibration": 0.021,
                        "directional_accuracy": 0.64,
                    },
                }
            ]
        )

        diagnostics_rows = iter(
            [
                {
                    "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                    "generated_at_utc": "2026-04-06T04:40:00+00:00",
                    "validation": {"average_log_wealth": 0.022},
                    "blocked_walk_forward_evaluation": {
                        "best_state_reset_mode_by_mean_log_wealth": "carry_on",
                        "state_reset_modes": {
                            "carry_on": {
                                "average_log_wealth_mean": 0.022,
                                "turnover_mean": 1.1,
                                "directional_accuracy_mean": 0.64,
                                "exact_smooth_position_mae_mean": 0.016,
                                "folds": [
                                    {"cvar_tail_loss": 0.013},
                                    {"cvar_tail_loss": 0.013},
                                    {"cvar_tail_loss": 0.013},
                                ],
                            }
                        },
                    },
                }
            ]
        )

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "artifact_root"
            artifact_root.mkdir()
            csv_path = artifact_root / "latest.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

            def fake_export_review_diagnostics(*, output_dir, **_kwargs):
                diagnostics_summary = next(diagnostics_rows)
                (output_dir / "validation_rows.csv").write_text("sample_id\n0\n", encoding="utf-8")
                (output_dir / "policy_summary.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "horizon_diag.csv").write_text("horizon\n1\n", encoding="utf-8")
                (output_dir / "validation_summary.json").write_text(
                    json.dumps(diagnostics_summary),
                    encoding="utf-8",
                )
                return diagnostics_summary

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                side_effect=lambda *args, **kwargs: (object(), next(summary_rows)),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_latest",
                return_value=PredictionResult(
                    anchor_time="2026-03-24T00:00:00+00:00",
                    current_close=100.0,
                    policy_horizon=1,
                    executed_horizon=1,
                    previous_position=0.0,
                    position=0.12,
                    trade_delta=0.12,
                    no_trade_band_hit=False,
                    tradeability_gate=0.8,
                    shape_entropy=0.3,
                    policy_score=0.05,
                    expected_log_returns={"1": 0.01},
                    predicted_closes={"1": 101.0},
                    uncertainties={"1": 0.01},
                    horizon_utilities={"1": 0.01},
                    horizon_positions={"1": 0.12},
                    shape_probabilities={"0": 0.4, "1": 0.6},
                    regime_id="asia|low|trend",
                ),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.export_review_diagnostics",
                side_effect=fake_export_review_diagnostics,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_candidate_parameters",
                return_value=[
                    {
                        "epochs": 14,
                        "batch_size": 16,
                        "learning_rate": 8e-4,
                        "hidden_dim": 48,
                        "dropout": 0.1,
                        "weight_decay": 1e-4,
                        "evaluation_state_reset_mode": "carry_on",
                        "min_policy_sigma": 1e-4,
                        "policy_cost_multiplier": 1.0,
                        "policy_gamma_multiplier": 1.0,
                        "q_max": 1.0,
                        "cvar_weight": 0.2,
                        "tie_policy_to_forecast_head": False,
                        "disable_overlay_branch": False,
                    }
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"horizons": (1,)},
                    quick_mode=True,
                )

        self.assertTrue(manifest["quick_mode"])
        self.assertEqual(manifest["selection_status"], "quick_mode_non_promotable")
        self.assertIsNone(manifest["production_current_candidate"])
        self.assertEqual(manifest["accepted_candidate"]["candidate"], "candidate_01")
        self.assertEqual(
            manifest["production_current_selection"]["selection_mode"],
            "quick_mode_non_promotable",
        )
        self.assertEqual(
            manifest["production_current_selection"]["selection_status"],
            "quick_mode_non_promotable",
        )

    def test_materialize_replay_artifact_copies_model_into_overlay(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "artifact"
            overlay_dir = Path(temp_dir) / "overlay"
            artifact_dir.mkdir()
            overlay_dir.mkdir()
            (artifact_dir / "model.pt").write_bytes(b"model")
            (artifact_dir / "metrics.json").write_text(
                json.dumps({"validation_metrics": {"average_log_wealth": 0.01}}),
                encoding="utf-8",
            )
            (artifact_dir / "prediction.json").write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            (artifact_dir / "forecast_summary.json").write_text(
                json.dumps({"best_params": {}}),
                encoding="utf-8",
            )
            config = TrainingConfig(output_dir=str(artifact_dir))
            validation_summary = {
                "generated_at_utc": "2026-04-07T00:00:00+00:00",
                "validation": {"average_log_wealth": 0.02},
            }
            source_payload = {"kind": "csv", "path": "/tmp/source.csv"}

            with patch(
                "signal_cascade_pytorch.bootstrap.generate_research_report",
                return_value={},
            ):
                _materialize_replay_artifact(
                    artifact_dir=artifact_dir,
                    diagnostics_output_dir=overlay_dir,
                    config=config,
                    source_payload=source_payload,
                    validation_summary=validation_summary,
                )

            manifest = json.loads((overlay_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((overlay_dir / "model.pt").exists())
            self.assertEqual(manifest["entrypoints"]["model"], "model.pt")

    def test_build_candidate_parameters_includes_structural_ablation_candidates(self) -> None:
        candidates = _build_candidate_parameters(
            {
                "epochs": 12,
                "batch_size": 16,
                "learning_rate": 5e-4,
                "hidden_dim": 48,
                "dropout": 0.1,
                "weight_decay": 5e-5,
                "evaluation_state_reset_mode": "carry_on",
                "min_policy_sigma": 1e-4,
                "policy_cost_multiplier": 1.0,
                "policy_gamma_multiplier": 1.0,
                "q_max": 1.0,
                "cvar_weight": 0.2,
                "tie_policy_to_forecast_head": False,
                "disable_overlay_branch": False,
            }
        )

        ablation_pairs = {
            (
                bool(candidate["tie_policy_to_forecast_head"]),
                bool(candidate["disable_overlay_branch"]),
            )
            for candidate in candidates
        }

        self.assertIn((False, False), ablation_pairs)
        self.assertIn((True, False), ablation_pairs)
        self.assertIn((False, True), ablation_pairs)
        self.assertIn((True, True), ablation_pairs)

    def test_train_model_summary_marks_profit_objective_as_primary(self) -> None:
        examples = [_example(returns_target=(0.01, 0.02)) for _ in range(4)]
        config = TrainingConfig(horizons=(1, 3), output_dir="artifacts/demo")

        with TemporaryDirectory() as temp_dir:
            with patch(
                "signal_cascade_pytorch.application.training_service._fit_model",
                return_value=(
                    object(),
                    {
                        "best_validation_loss": 0.5,
                        "best_epoch": 2.0,
                        "history": [{"epoch": 1.0, "train_profit": 0.1, "validation_profit": 0.2}],
                    },
                ),
            ), patch(
                "signal_cascade_pytorch.application.training_service.save_checkpoint"
            ), patch(
                "signal_cascade_pytorch.application.training_service.evaluate_model",
                return_value={"average_log_wealth": 0.01, "cvar_tail_loss": 0.02},
            ):
                _, summary = train_model(
                    examples=examples,
                    config=config,
                    output_dir=Path(temp_dir),
                )

        self.assertEqual(
            summary["loss_contract"]["primary_objective"],
            "profit_objective_log_wealth_minus_cvar",
        )
        self.assertEqual(summary["loss_contract"]["primary_metric"], "profit_loss")
        self.assertEqual(
            summary["loss_contract"]["auxiliary_objectives"]["return"],
            "heteroscedastic_huber",
        )
        self.assertEqual(
            summary["loss_contract"]["auxiliary_objectives"]["shape"],
            "main_shape_smooth_l1",
        )

    def test_fit_model_history_includes_exact_risk_and_position_metrics(self) -> None:
        examples = [_example(returns_target=(0.01, 0.02)) for _ in range(4)]
        config = TrainingConfig(horizons=(1, 3), epochs=2, warmup_epochs=1)

        with patch(
            "signal_cascade_pytorch.application.training_service._run_epoch",
            side_effect=[
                {
                    "total": 0.1,
                    "profit_loss": 0.2,
                    "return_loss": 0.3,
                    "shape_loss": 0.4,
                    "average_log_wealth": 0.5,
                    "forecast_mae": 0.6,
                },
                {
                    "total": 0.7,
                    "profit_loss": 0.8,
                    "return_loss": 0.9,
                    "shape_loss": 1.0,
                    "average_log_wealth": 1.1,
                    "forecast_mae": 1.2,
                },
                {
                    "total": 1.3,
                    "profit_loss": 1.4,
                    "return_loss": 1.5,
                    "shape_loss": 1.6,
                    "average_log_wealth": 1.7,
                    "forecast_mae": 1.8,
                },
                {
                    "total": 1.9,
                    "profit_loss": 2.0,
                    "return_loss": 2.1,
                    "shape_loss": 2.2,
                    "average_log_wealth": 2.3,
                    "forecast_mae": 2.4,
                },
            ],
        ), patch(
            "signal_cascade_pytorch.application.training_service.evaluate_model",
            side_effect=[
                {
                    "average_log_wealth": 0.01,
                    "forecast_mae": 0.02,
                    "cvar_tail_loss": 0.03,
                    "sigma_calibration": 0.04,
                    "exact_smooth_position_mae": 0.05,
                    "checkpoint_selection_score": 0.06,
                },
                {
                    "average_log_wealth": 0.07,
                    "forecast_mae": 0.08,
                    "cvar_tail_loss": 0.09,
                    "sigma_calibration": 0.10,
                    "exact_smooth_position_mae": 0.11,
                    "checkpoint_selection_score": 0.12,
                },
            ],
        ):
            _, fit_summary = _fit_model(
                train_examples=examples[:2],
                valid_examples=examples[2:],
                config=config,
                feature_dim=len(examples[0].main_sequences["4h"][0]),
                state_feature_dim=len(examples[0].state_features),
            )

        self.assertEqual(fit_summary["history"][0]["validation_exact_cvar_tail_loss"], 0.03)
        self.assertEqual(fit_summary["history"][0]["validation_exact_sigma_calibration"], 0.04)
        self.assertEqual(fit_summary["history"][0]["validation_exact_position_mae"], 0.05)
        self.assertEqual(fit_summary["history"][1]["validation_exact_cvar_tail_loss"], 0.09)
        self.assertEqual(fit_summary["history"][1]["validation_exact_sigma_calibration"], 0.10)
        self.assertEqual(fit_summary["history"][1]["validation_exact_position_mae"], 0.11)
        self.assertEqual(fit_summary["checkpoint_audit"]["best_epoch_by_selection_score"], 1.0)
        self.assertEqual(fit_summary["checkpoint_audit"]["best_epoch_by_exact_log_wealth"], 2.0)
        self.assertEqual(
            fit_summary["checkpoint_audit"]["best_epoch_by_exact_log_wealth_minus_lambda_cvar"],
            2.0,
        )
        self.assertEqual(
            fit_summary["checkpoint_audit"]["selected_epoch_rank_by_exact_log_wealth"],
            2.0,
        )

    def test_fit_model_prefers_blocked_oof_selection_score_when_available(self) -> None:
        examples = [_example(returns_target=(0.01, 0.02)) for _ in range(6)]
        config = TrainingConfig(
            horizons=(1, 3),
            epochs=2,
            warmup_epochs=1,
            walk_forward_folds=2,
            oof_epochs=2,
            checkpoint_selection_metric="exact_log_wealth_minus_lambda_cvar",
            cvar_weight=0.2,
        )

        with patch(
            "signal_cascade_pytorch.application.training_service._run_epoch",
            side_effect=[
                {
                    "total": 0.1,
                    "profit_loss": 0.2,
                    "return_loss": 0.3,
                    "shape_loss": 0.4,
                    "average_log_wealth": 0.5,
                    "forecast_mae": 0.6,
                },
                {
                    "total": 0.7,
                    "profit_loss": 0.8,
                    "return_loss": 0.9,
                    "shape_loss": 1.0,
                    "average_log_wealth": 1.1,
                    "forecast_mae": 1.2,
                },
                {
                    "total": 1.3,
                    "profit_loss": 1.4,
                    "return_loss": 1.5,
                    "shape_loss": 1.6,
                    "average_log_wealth": 1.7,
                    "forecast_mae": 1.8,
                },
                {
                    "total": 1.9,
                    "profit_loss": 2.0,
                    "return_loss": 2.1,
                    "shape_loss": 2.2,
                    "average_log_wealth": 2.3,
                    "forecast_mae": 2.4,
                },
            ],
        ), patch(
            "signal_cascade_pytorch.application.training_service.evaluate_model",
            side_effect=[
                {
                    "average_log_wealth": 0.08,
                    "forecast_mae": 0.02,
                    "cvar_tail_loss": 0.10,
                    "sigma_calibration": 0.04,
                    "exact_smooth_position_mae": 0.05,
                    "checkpoint_selection_score": 0.50,
                    "turnover": 3.0,
                },
                {
                    "average_log_wealth": 0.03,
                    "cvar_tail_loss": 0.05,
                    "turnover": 2.0,
                },
                {
                    "average_log_wealth": 0.01,
                    "cvar_tail_loss": 0.05,
                    "turnover": 2.5,
                },
                {
                    "average_log_wealth": 0.09,
                    "forecast_mae": 0.03,
                    "cvar_tail_loss": 0.08,
                    "sigma_calibration": 0.05,
                    "exact_smooth_position_mae": 0.06,
                    "checkpoint_selection_score": 0.40,
                    "turnover": 2.5,
                },
                {
                    "average_log_wealth": 0.05,
                    "cvar_tail_loss": 0.05,
                    "turnover": 1.5,
                },
                {
                    "average_log_wealth": 0.04,
                    "cvar_tail_loss": 0.05,
                    "turnover": 1.0,
                },
            ],
        ):
            _, fit_summary = _fit_model(
                train_examples=examples[:2],
                valid_examples=examples[2:],
                config=config,
                feature_dim=len(examples[0].main_sequences["4h"][0]),
                state_feature_dim=len(examples[0].state_features),
            )

        self.assertEqual(
            fit_summary["history"][0]["validation_selection_score_source"],
            "blocked_walk_forward_objective_log_wealth_minus_lambda_cvar_oof_mean",
        )
        self.assertEqual(
            fit_summary["history"][1]["validation_selection_score_source"],
            "blocked_walk_forward_objective_log_wealth_minus_lambda_cvar_oof_mean",
        )
        self.assertAlmostEqual(
            fit_summary["history"][0]["validation_selection_score"],
            -0.01,
            places=9,
        )
        self.assertAlmostEqual(
            fit_summary["history"][1]["validation_selection_score"],
            -0.0225,
            places=9,
        )
        self.assertEqual(fit_summary["best_epoch"], 2.0)


if __name__ == "__main__":
    unittest.main()
