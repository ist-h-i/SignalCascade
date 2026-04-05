from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
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
from signal_cascade_pytorch.application.training_service import examples_to_batch, train_model
from signal_cascade_pytorch.application.tuning_service import tune_latest_dataset
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
        self.assertGreater(float(decision["selection_score"]), 0.0)

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

    def test_prediction_result_keeps_compatibility_properties(self) -> None:
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

        self.assertEqual(prediction.proposed_horizon, 3)
        self.assertEqual(prediction.accepted_horizon, 3)
        self.assertTrue(prediction.accepted_signal)
        self.assertEqual(prediction.selection_probability, 0.8)
        self.assertNotIn("proposed_horizon", asdict(prediction))

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
                    selection_policy=None,
                )

            self.assertEqual(summary["diagnostics_schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
            self.assertEqual(summary["policy_mode"], "shape_aware_profit_maximization")
            self.assertTrue((output_dir / "policy_summary.csv").exists())
            saved_summary = json.loads((output_dir / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["validation"]["executed_trade_count"], 1)

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

            with patch(
                "signal_cascade_pytorch.application.tuning_service.CsvMarketDataSource.load_bars",
                return_value=["bar"],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.build_training_examples_from_bars",
                return_value=[object()],
            ), patch(
                "signal_cascade_pytorch.application.tuning_service._build_latest_example",
                return_value=object(),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.train_model",
                return_value=(object(), summary),
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.predict_from_example",
                return_value=prediction,
            ), patch(
                "signal_cascade_pytorch.application.tuning_service.generate_research_report",
                return_value=None,
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
                    }
                ],
            ):
                manifest = tune_latest_dataset(
                    csv_path=csv_path,
                    artifact_root=artifact_root,
                    config_overrides={"epochs": 6, "horizons": (1, 3)},
                )

            self.assertEqual(manifest["best_candidate"]["epochs"], 6)
            saved_config = json.loads(
                (artifact_root / "current" / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config["epochs"], 6)
            self.assertEqual(saved_config["horizons"], [1, 3])

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


if __name__ == "__main__":
    unittest.main()
