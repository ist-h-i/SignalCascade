from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.diagnostics_service import (
    POLICY_SELECTION_BASIS,
    POLICY_SELECTION_RULE_VERSION,
    _annotate_policy_calibration_sweep,
    _summarize_policy_calibration_sweep,
    export_review_diagnostics,
)
from signal_cascade_pytorch.application.training_service import split_examples
from signal_cascade_pytorch.domain.entities import TrainingExample
from signal_cascade_pytorch.domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES
from signal_cascade_pytorch.interfaces.cli import build_parser


def _example(index: int) -> TrainingExample:
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
        returns_target=(0.02, 0.015),
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
        regime_features=(1.0, 0.0, 0.0, 0.1, 0.25),
        realized_volatility=0.02,
        trend_strength=0.25,
    )


class _StaticPolicyModel(torch.nn.Module):
    def forward(self, main_sequences, overlay_sequences, state_features, previous_state=None):
        batch_size = state_features.shape[0]
        shape_posterior = torch.tensor(
            [[0.45, 0.2, 0.15, 0.1, 0.05, 0.05]],
            dtype=torch.float32,
        ).repeat(batch_size, 1)
        memory_state = torch.zeros(batch_size, 4, dtype=torch.float32)
        return {
            "mu": torch.tensor([[0.03, 0.012]], dtype=torch.float32).repeat(batch_size, 1),
            "sigma": torch.tensor([[0.02, 0.03]], dtype=torch.float32).repeat(batch_size, 1),
            "tradeability_gate": torch.tensor([0.9], dtype=torch.float32).repeat(batch_size),
            "shape_entropy": torch.tensor([0.25], dtype=torch.float32).repeat(batch_size),
            "shape_posterior": shape_posterior,
            "shape_probs": shape_posterior,
            "memory_state": memory_state,
            "next_state": memory_state,
        }


class PolicySweepTests(unittest.TestCase):
    def test_export_review_diagnostics_emits_policy_calibration_sweep(self) -> None:
        config = TrainingConfig(horizons=(1, 3), train_ratio=0.5)
        examples = [_example(index) for index in range(8)]

        with TemporaryDirectory() as temp_dir:
            summary = export_review_diagnostics(
                output_dir=Path(temp_dir),
                model=_StaticPolicyModel(),
                examples=examples,
                config=config,
            )

        sweep = summary["policy_calibration_sweep"]
        self.assertEqual(
            len(sweep),
            len(config.policy_sweep_state_reset_modes)
            * len(config.policy_sweep_cost_multipliers)
            * len(config.policy_sweep_gamma_multipliers)
            * len(config.policy_sweep_min_policy_sigmas)
            * len(config.policy_sweep_q_max_values)
            * len(config.policy_sweep_cvar_weights),
        )
        self.assertIn("average_log_wealth", sweep[0])
        self.assertIn("objective_log_wealth_minus_lambda_cvar", sweep[0])
        self.assertIn("turnover", sweep[0])
        self.assertIn("no_trade_band_hit_rate", sweep[0])
        self.assertIn("min_policy_sigma", sweep[0])
        self.assertIn("q_max", sweep[0])
        self.assertIn("cvar_weight", sweep[0])
        self.assertIn("blocked_walk_forward", sweep[0])
        self.assertIn("blocked_objective_log_wealth_minus_lambda_cvar_mean", sweep[0])
        self.assertIn("dominated", sweep[0])
        self.assertIn("pareto_optimal", sweep[0])
        self.assertTrue(any(not bool(row["dominated"]) for row in sweep))
        self.assertEqual(
            summary["policy_calibration_summary"]["row_count"],
            len(sweep),
        )
        self.assertEqual(
            summary["policy_calibration_summary"]["selection_rule_version"],
            POLICY_SELECTION_RULE_VERSION,
        )
        self.assertIsNotNone(summary["policy_calibration_summary"]["selected_row_key"])
        self.assertEqual(
            summary["policy_calibration_summary"]["selected_row_role"],
            "diagnostic_recommendation_not_applied_runtime_config",
        )
        self.assertEqual(
            summary["policy_calibration_summary"]["applied_runtime_policy_role"],
            "authoritative_runtime_config",
        )
        self.assertEqual(
            summary["policy_calibration_summary"]["applied_runtime_policy"]["state_reset_mode"],
            config.evaluation_state_reset_mode,
        )
        self.assertIsNotNone(summary["policy_calibration_summary"]["policy_calibration_rows_sha256"])
        _, validation_examples = split_examples(examples, config)
        blocked_walk_forward = summary["blocked_walk_forward_evaluation"]
        self.assertEqual(blocked_walk_forward["fold_count"], config.walk_forward_folds)
        self.assertEqual(sum(blocked_walk_forward["fold_sample_counts"]), len(validation_examples))
        self.assertIn("carry_on", blocked_walk_forward["state_reset_modes"])
        self.assertEqual(
            blocked_walk_forward["state_reset_modes"]["carry_on"]["fold_count"],
            config.walk_forward_folds,
        )

    def test_policy_sweep_selection_prefers_best_blocked_objective_mean(self) -> None:
        rows = [
            {
                "row_key": (
                    "state_reset_mode=carry_on|cost_multiplier=1|gamma_multiplier=1|"
                    "min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                ),
                "state_reset_mode": "carry_on",
                "cost_multiplier": 1.0,
                "gamma_multiplier": 1.0,
                "min_policy_sigma": 0.0001,
                "q_max": 1.0,
                "cvar_weight": 0.2,
                "average_log_wealth": 0.012,
                "turnover": 0.40,
                "cvar_tail_loss": 0.02,
                "blocked_average_log_wealth_mean": 0.010,
                "blocked_turnover_mean": 0.40,
                "blocked_cvar_tail_loss_mean": 0.02,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.006,
            },
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
                "average_log_wealth": 0.012,
                "turnover": 0.40,
                "cvar_tail_loss": 0.02,
                "blocked_average_log_wealth_mean": 0.011,
                "blocked_turnover_mean": 0.39,
                "blocked_cvar_tail_loss_mean": 0.02,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.007,
            },
        ]

        forward_summary = _summarize_policy_calibration_sweep(_annotate_policy_calibration_sweep(rows))
        reverse_summary = _summarize_policy_calibration_sweep(
            _annotate_policy_calibration_sweep(list(reversed(rows)))
        )

        self.assertEqual(
            POLICY_SELECTION_BASIS,
            "pareto_rank_then_near_best_blocked_objective_mean_turnover_mean_"
            "blocked_objective_mean_average_log_wealth_mean_cvar_tail_loss_mean_row_key",
        )
        self.assertEqual(
            forward_summary["selected_row_key"],
            reverse_summary["selected_row_key"],
        )
        self.assertEqual(
            forward_summary["policy_calibration_rows_sha256"],
            reverse_summary["policy_calibration_rows_sha256"],
        )
        self.assertEqual(
            forward_summary["selected_row_key"],
            "state_reset_mode=reset_each_session_or_window|cost_multiplier=1|"
            "gamma_multiplier=1|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2",
        )

    def test_policy_sweep_selection_prefers_lower_turnover_within_objective_tolerance(self) -> None:
        rows = [
            {
                "row_key": (
                    "state_reset_mode=reset_each_session_or_window|cost_multiplier=0.5|"
                    "gamma_multiplier=0.5|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                ),
                "state_reset_mode": "reset_each_session_or_window",
                "cost_multiplier": 0.5,
                "gamma_multiplier": 0.5,
                "min_policy_sigma": 0.0001,
                "q_max": 1.0,
                "cvar_weight": 0.2,
                "average_log_wealth": 0.020,
                "turnover": 9.0,
                "cvar_tail_loss": 0.01,
                "blocked_average_log_wealth_mean": 0.020,
                "blocked_turnover_mean": 9.0,
                "blocked_cvar_tail_loss_mean": 0.01,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.018,
            },
            {
                "row_key": (
                    "state_reset_mode=carry_on|cost_multiplier=2|gamma_multiplier=2|"
                    "min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2"
                ),
                "state_reset_mode": "carry_on",
                "cost_multiplier": 2.0,
                "gamma_multiplier": 2.0,
                "min_policy_sigma": 0.0001,
                "q_max": 1.0,
                "cvar_weight": 0.2,
                "average_log_wealth": 0.017,
                "turnover": 2.0,
                "cvar_tail_loss": 0.01,
                "blocked_average_log_wealth_mean": 0.017,
                "blocked_turnover_mean": 2.0,
                "blocked_cvar_tail_loss_mean": 0.01,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean": 0.014,
            },
        ]

        summary = _summarize_policy_calibration_sweep(_annotate_policy_calibration_sweep(rows))

        self.assertEqual(
            summary["best_row"]["row_key"],
            "state_reset_mode=reset_each_session_or_window|cost_multiplier=0.5|"
            "gamma_multiplier=0.5|min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2",
        )
        self.assertEqual(
            summary["selected_row_key"],
            "state_reset_mode=carry_on|cost_multiplier=2|gamma_multiplier=2|"
            "min_policy_sigma=0.0001|q_max=1|cvar_weight=0.2",
        )

    def test_policy_sweep_uses_current_q_max_and_cvar_when_axes_are_not_overridden(self) -> None:
        config = TrainingConfig(
            horizons=(1, 3),
            train_ratio=0.5,
            q_max=1.25,
            cvar_weight=0.4,
        )
        examples = [_example(index) for index in range(8)]

        with TemporaryDirectory() as temp_dir:
            summary = export_review_diagnostics(
                output_dir=Path(temp_dir),
                model=_StaticPolicyModel(),
                examples=examples,
                config=config,
            )

        q_max_values = {float(row["q_max"]) for row in summary["policy_calibration_sweep"]}
        cvar_weights = {float(row["cvar_weight"]) for row in summary["policy_calibration_sweep"]}
        self.assertEqual(q_max_values, {1.25})
        self.assertEqual(cvar_weights, {0.4})

    def test_policy_sweep_includes_current_cost_and_gamma_when_axes_are_not_overridden(self) -> None:
        config = TrainingConfig(
            horizons=(1, 3),
            train_ratio=0.5,
            policy_cost_multiplier=6.0,
            policy_gamma_multiplier=4.0,
        )
        examples = [_example(index) for index in range(8)]

        with TemporaryDirectory() as temp_dir:
            summary = export_review_diagnostics(
                output_dir=Path(temp_dir),
                model=_StaticPolicyModel(),
                examples=examples,
                config=config,
            )

        cost_values = {float(row["cost_multiplier"]) for row in summary["policy_calibration_sweep"]}
        gamma_values = {float(row["gamma_multiplier"]) for row in summary["policy_calibration_sweep"]}
        self.assertIn(6.0, cost_values)
        self.assertIn(4.0, gamma_values)

    def test_parser_accepts_policy_sweep_overrides(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "export-diagnostics",
                "--evaluation-state-reset-mode",
                "reset_each_example",
                "--diagnostic-state-reset-modes",
                "carry_on,reset_each_session_or_window",
                "--policy-sweep-cost-multipliers",
                "0.5,1.0,2.0",
                "--policy-sweep-gamma-multipliers",
                "1.0,2.0",
                "--policy-sweep-min-policy-sigmas",
                "0.00005,0.0001",
                "--policy-sweep-q-max-values",
                "0.75,1.0,1.25",
                "--policy-sweep-cvar-weights",
                "0.1,0.2,0.4",
                "--policy-sweep-state-reset-modes",
                "carry_on,reset_each_session_or_window",
            ]
        )

        self.assertEqual(args.evaluation_state_reset_mode, "reset_each_example")
        self.assertEqual(args.diagnostic_state_reset_modes, "carry_on,reset_each_session_or_window")
        self.assertEqual(args.policy_sweep_cost_multipliers, "0.5,1.0,2.0")
        self.assertEqual(args.policy_sweep_gamma_multipliers, "1.0,2.0")
        self.assertEqual(args.policy_sweep_min_policy_sigmas, "0.00005,0.0001")
        self.assertEqual(args.policy_sweep_q_max_values, "0.75,1.0,1.25")
        self.assertEqual(args.policy_sweep_cvar_weights, "0.1,0.2,0.4")
        self.assertEqual(args.policy_sweep_state_reset_modes, "carry_on,reset_each_session_or_window")


if __name__ == "__main__":
    unittest.main()
