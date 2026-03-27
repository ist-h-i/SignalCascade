from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from signal_cascade_pytorch.application.config import TrainingConfig
from signal_cascade_pytorch.application.inference_service import (
    FORECAST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    build_forecast_summary_payload,
    serialize_prediction_result,
)
from signal_cascade_pytorch.application.report_service import ANALYSIS_SCHEMA_VERSION, generate_research_report
from signal_cascade_pytorch.domain.entities import PredictionResult


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
            (output_dir / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "primary_state_reset_mode": "reset_each_session_or_window",
                        "stateful_evaluation": {"carry_on": {"average_log_wealth": 0.01}},
                        "policy_calibration_summary": {
                            "row_count": 1,
                            "pareto_optimal_count": 1,
                            "dominated_count": 0,
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
        self.assertEqual(analysis["dataset"]["purged_samples"], 2)
        self.assertIn("carry_on", analysis["stateful_evaluation"])
        self.assertEqual(analysis["policy_calibration_summary"]["pareto_optimal_count"], 1)


if __name__ == "__main__":
    unittest.main()
