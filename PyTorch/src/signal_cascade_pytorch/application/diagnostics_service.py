from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from .config import TrainingConfig
from .policy_service import apply_selection_policy, policy_utility, smooth_policy_distribution
from .training_service import (
    _should_reset_recurrent_context,
    evaluate_model,
    examples_to_batch,
    split_examples,
)
from ..domain.entities import OHLCVBar, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel
from ..infrastructure.persistence import save_json

DIAGNOSTICS_SCHEMA_VERSION = 5
POLICY_SELECTION_RULE_VERSION = 2
POLICY_SELECTION_BASIS = "pareto_rank_then_average_log_wealth_cvar_tail_loss_turnover_row_key"


def export_review_diagnostics(
    output_dir: Path,
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    selection_policy: dict[str, object] | None = None,
    threshold_resolution: dict[str, object] | None = None,
    source_payload: dict[str, object] | None = None,
    source_rows_original: int | None = None,
    source_rows_used: int | None = None,
    base_bars: Sequence[OHLCVBar] | None = None,
) -> dict[str, object]:
    _, validation_examples = split_examples(examples, config)
    diagnostics = build_validation_diagnostics(
        model=model,
        validation_examples=validation_examples,
        config=config,
    )
    stateful_evaluation = (
        {
            mode: evaluate_model(
                model=model,
                examples=validation_examples,
                config=config,
                state_reset_mode=mode,
            )
            for mode in config.diagnostic_state_reset_modes
        }
        if model is not None
        else {}
    )
    policy_calibration_sweep = (
        _build_policy_calibration_sweep(
            model=model,
            validation_examples=validation_examples,
            config=config,
        )
        if model is not None
        else []
    )

    validation_rows_path = output_dir / "validation_rows.csv"
    policy_summary_path = output_dir / "policy_summary.csv"
    horizon_diag_path = output_dir / "horizon_diag.csv"
    summary_path = output_dir / "validation_summary.json"

    _write_csv(validation_rows_path, diagnostics["validation_rows"])
    _write_csv(policy_summary_path, diagnostics["policy_summary"])
    _write_csv(horizon_diag_path, diagnostics["horizon_diag"])

    summary = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_mode": "shape_aware_profit_maximization",
        "primary_state_reset_mode": config.evaluation_state_reset_mode,
        "dataset": {
            "sample_count": len(examples),
            "validation_sample_count": len(validation_examples),
            "source": source_payload,
            "source_rows_original": source_rows_original,
            "source_rows_used": source_rows_used,
            "base_bar_count": None if base_bars is None else len(base_bars),
        },
        "validation": diagnostics["summary"],
        "stateful_evaluation": stateful_evaluation,
        "policy_calibration_sweep": policy_calibration_sweep,
        "policy_calibration_summary": _summarize_policy_calibration_sweep(policy_calibration_sweep),
        "paths": {
            "validation_rows_csv": str(validation_rows_path),
            "policy_summary_csv": str(policy_summary_path),
            "horizon_diag_csv": str(horizon_diag_path),
        },
    }
    save_json(summary_path, summary)
    return summary


def build_validation_diagnostics(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, object]:
    model.eval()
    state_reset_mode = config.evaluation_state_reset_mode
    summary = evaluate_model(
        model=model,
        examples=validation_examples,
        config=config,
        state_reset_mode=state_reset_mode,
    )
    validation_rows: list[dict[str, object]] = []
    policy_summary: list[dict[str, object]] = []
    previous_state = None
    previous_position = 0.0
    previous_example: TrainingExample | None = None
    horizon_stats = {
        str(horizon): {
            "count": 0,
            "selected": 0,
            "utility_sum": 0.0,
            "position_sum": 0.0,
            "abs_error_sum": 0.0,
        }
        for horizon in config.horizons
    }
    sample_id = 0

    with torch.no_grad():
        for example in validation_examples:
            if _should_reset_recurrent_context(
                state_reset_mode=state_reset_mode,
                current_example=example,
                previous_example=previous_example,
            ):
                previous_state = None
                previous_position = 0.0
            batch = examples_to_batch([example], config)
            outputs = model(
                batch["main"],
                batch["overlay"],
                batch["state_features"],
                previous_state=previous_state,
            )
            mean = outputs["mu"][0]
            sigma = outputs["sigma"][0]
            gate = float(outputs["tradeability_gate"][0].item())
            entropy = float(outputs["shape_entropy"][0].item())
            smooth_policy = smooth_policy_distribution(
                mean=outputs["mu"],
                sigma=outputs["sigma"],
                costs=batch["horizon_costs"],
                tradeability_gate=outputs["tradeability_gate"],
                previous_position=torch.tensor([previous_position], dtype=mean.dtype),
                config=config,
            )
            decision = apply_selection_policy(
                example=example,
                mean=mean.tolist(),
                sigma=sigma.tolist(),
                config=config,
                previous_position=previous_position,
                tradeability_gate=gate,
                shape_probs=outputs["shape_probs"][0].tolist(),
            )
            selected_row = dict(decision["selected_row"])
            selected_horizon = int(selected_row["horizon"])
            selected_index = config.horizons.index(selected_horizon)
            realized_return = float(example.returns_target[selected_index])
            trade_cost = float(selected_row["cost"]) * abs(float(decision["trade_delta"]))
            pnl = (float(decision["position"]) * realized_return) - trade_cost
            smooth_selected_index = int(smooth_policy["selected_horizon_index"][0].item())
            smooth_selected_horizon = int(config.horizons[smooth_selected_index])
            smooth_selected_position = float(smooth_policy["selected_position"][0].item())
            smooth_selected_no_trade = bool(smooth_policy["selected_no_trade"][0].item())
            smooth_reference_row = dict(decision["horizon_rows"][smooth_selected_index])
            smooth_reference_utility = policy_utility(
                position=smooth_selected_position,
                previous_position=previous_position,
                gated_mean=float(smooth_reference_row["gated_mean"]),
                sigma=float(smooth_reference_row["sigma"]),
                cost=float(smooth_reference_row["cost"]),
                config=config,
            )

            for row in decision["horizon_rows"]:
                horizon_key = str(row["horizon"])
                horizon_index = config.horizons.index(int(row["horizon"]))
                actual_return = float(example.returns_target[horizon_index])
                horizon_stats[horizon_key]["count"] += 1
                horizon_stats[horizon_key]["selected"] += int(int(row["horizon"]) == selected_horizon)
                horizon_stats[horizon_key]["utility_sum"] += float(row["policy_utility"])
                horizon_stats[horizon_key]["position_sum"] += float(row["position"])
                horizon_stats[horizon_key]["abs_error_sum"] += abs(actual_return - float(row["mean"]))
                validation_rows.append(
                    {
                        "sample_id": sample_id,
                        "timestamp": example.anchor_time.isoformat(),
                        "regime_id": example.regime_id,
                        "horizon": int(row["horizon"]),
                        "y_raw": actual_return,
                        "mu_raw": float(row["mean"]),
                        "sigma_raw": float(row["sigma"]),
                        "gated_mean": float(row["gated_mean"]),
                        "cost": float(row["cost"]),
                        "policy_utility": float(row["policy_utility"]),
                        "position_if_chosen": float(row["position"]),
                        "tradeability_gate": gate,
                        "shape_entropy": entropy,
                        "selected_horizon": selected_horizon,
                        "executed_horizon": decision["executed_horizon"],
                        "smooth_policy_horizon": smooth_selected_horizon,
                        "smooth_position": smooth_selected_position,
                        "smooth_no_trade_band": int(smooth_selected_no_trade),
                        "selected": int(int(row["horizon"]) == selected_horizon),
                        "no_trade_band": int(bool(row["no_trade_band"])),
                    }
                )

            policy_summary.append(
                {
                    "sample_id": sample_id,
                    "timestamp": example.anchor_time.isoformat(),
                    "regime_id": example.regime_id,
                    "policy_horizon": decision["policy_horizon"],
                    "executed_horizon": decision["executed_horizon"],
                    "previous_position": float(decision["previous_position"]),
                    "position": float(decision["position"]),
                    "trade_delta": float(decision["trade_delta"]),
                    "no_trade_band_hit": int(bool(decision["no_trade_band_hit"])),
                    "smooth_policy_horizon": smooth_selected_horizon,
                    "smooth_position": smooth_selected_position,
                    "smooth_no_trade_band_hit": int(smooth_selected_no_trade),
                    "exact_smooth_horizon_agreement": int(selected_horizon == smooth_selected_horizon),
                    "exact_smooth_no_trade_agreement": int(
                        bool(decision["no_trade_band_hit"]) == smooth_selected_no_trade
                    ),
                    "exact_smooth_position_abs_error": abs(
                        float(decision["position"]) - smooth_selected_position
                    ),
                    "exact_smooth_utility_regret": max(
                        float(selected_row["policy_utility"]) - float(smooth_reference_utility),
                        0.0,
                    ),
                    "policy_score": float(decision["selection_score"]),
                    "tradeability_gate": gate,
                    "shape_entropy": entropy,
                    "realized_return": realized_return,
                    "realized_pnl": pnl,
                }
            )
            previous_position = float(decision["position"])
            previous_state = outputs["next_state"].detach()
            previous_example = example
            sample_id += 1

    horizon_diag = [
        {
            "horizon": int(horizon),
            "sample_count": int(stats["count"]),
            "selection_rate": int(stats["selected"]) / max(int(stats["count"]), 1),
            "mean_policy_utility": float(stats["utility_sum"]) / max(int(stats["count"]), 1),
            "mean_position": float(stats["position_sum"]) / max(int(stats["count"]), 1),
            "mu_calibration": float(stats["abs_error_sum"]) / max(int(stats["count"]), 1),
        }
        for horizon, stats in sorted(horizon_stats.items(), key=lambda item: int(item[0]))
    ]
    return {
        "validation_rows": validation_rows,
        "policy_summary": policy_summary,
        "horizon_diag": horizon_diag,
        "summary": summary,
    }


def build_validation_snapshots(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    diagnostics = build_validation_diagnostics(model, validation_examples, config)
    return diagnostics["policy_summary"]


def _build_policy_calibration_sweep(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    sweep_rows: list[dict[str, object]] = []
    for state_reset_mode in config.policy_sweep_state_reset_modes:
        for min_policy_sigma in config.policy_sweep_min_policy_sigmas:
            for gamma_multiplier in config.policy_sweep_gamma_multipliers:
                for cost_multiplier in config.policy_sweep_cost_multipliers:
                    metrics = evaluate_model(
                        model=model,
                        examples=validation_examples,
                        config=config,
                        state_reset_mode=state_reset_mode,
                        cost_multiplier=float(cost_multiplier),
                        gamma_multiplier=float(gamma_multiplier),
                        min_policy_sigma=float(min_policy_sigma),
                    )
                    sweep_rows.append(
                        {
                            "row_key": _policy_sweep_row_key(
                                state_reset_mode=state_reset_mode,
                                cost_multiplier=float(cost_multiplier),
                                gamma_multiplier=float(gamma_multiplier),
                                min_policy_sigma=float(min_policy_sigma),
                            ),
                            "state_reset_mode": state_reset_mode,
                            "cost_multiplier": float(cost_multiplier),
                            "gamma_multiplier": float(gamma_multiplier),
                            "min_policy_sigma": float(min_policy_sigma),
                            "average_log_wealth": float(metrics["average_log_wealth"]),
                            "turnover": float(metrics["turnover"]),
                            "cvar_tail_loss": float(metrics["cvar_tail_loss"]),
                            "no_trade_band_hit_rate": float(metrics["no_trade_band_hit_rate"]),
                            "exact_smooth_horizon_agreement": float(
                                metrics["exact_smooth_horizon_agreement"]
                            ),
                            "exact_smooth_no_trade_agreement": float(
                                metrics["exact_smooth_no_trade_agreement"]
                            ),
                            "exact_smooth_position_mae": float(metrics["exact_smooth_position_mae"]),
                            "exact_smooth_utility_regret": float(
                                metrics["exact_smooth_utility_regret"]
                            ),
                        }
                    )
    return _annotate_policy_calibration_sweep(sweep_rows)


def _annotate_policy_calibration_sweep(
    sweep_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    annotated_rows: list[dict[str, object]] = []
    for index, row in enumerate(sweep_rows):
        dominated = any(
            _policy_sweep_row_dominates(other_row, row)
            for other_index, other_row in enumerate(sweep_rows)
            if other_index != index
        )
        enriched = dict(row)
        enriched["dominated"] = dominated
        enriched["pareto_optimal"] = not dominated
        annotated_rows.append(enriched)
    return sorted(
        annotated_rows,
        key=lambda item: (
            bool(item["dominated"]),
            -float(item["average_log_wealth"]),
            float(item["cvar_tail_loss"]),
            float(item["turnover"]),
            str(item["row_key"]),
        ),
    )


def _policy_sweep_row_dominates(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    comparisons = (
        float(left["average_log_wealth"]) >= float(right["average_log_wealth"]),
        float(left["turnover"]) <= float(right["turnover"]),
        float(left["cvar_tail_loss"]) <= float(right["cvar_tail_loss"]),
    )
    strictly_better = (
        float(left["average_log_wealth"]) > float(right["average_log_wealth"])
        or float(left["turnover"]) < float(right["turnover"])
        or float(left["cvar_tail_loss"]) < float(right["cvar_tail_loss"])
    )
    return all(comparisons) and strictly_better


def _summarize_policy_calibration_sweep(
    sweep_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    pareto_rows = [dict(row) for row in sweep_rows if not bool(row.get("dominated", False))]
    selected_row = pareto_rows[0] if pareto_rows else (dict(sweep_rows[0]) if sweep_rows else None)
    return {
        "row_count": len(sweep_rows),
        "pareto_optimal_count": len(pareto_rows),
        "dominated_count": max(len(sweep_rows) - len(pareto_rows), 0),
        "policy_calibration_rows_sha256": _policy_sweep_rows_sha256(sweep_rows),
        "selection_basis": POLICY_SELECTION_BASIS,
        "selection_rule_version": POLICY_SELECTION_RULE_VERSION,
        "selected_row": selected_row,
        "selected_row_key": None if selected_row is None else selected_row.get("row_key"),
        "best_row": selected_row,
    }


def _policy_sweep_row_key(
    *,
    state_reset_mode: str,
    cost_multiplier: float,
    gamma_multiplier: float,
    min_policy_sigma: float,
) -> str:
    return "|".join(
        (
            f"state_reset_mode={state_reset_mode}",
            f"cost_multiplier={cost_multiplier:.12g}",
            f"gamma_multiplier={gamma_multiplier:.12g}",
            f"min_policy_sigma={min_policy_sigma:.12g}",
        )
    )


def _policy_sweep_rows_sha256(sweep_rows: Sequence[dict[str, object]]) -> str | None:
    row_keys = sorted(
        str(row["row_key"])
        for row in sweep_rows
        if row.get("row_key") is not None
    )
    if not row_keys:
        return None
    encoded = json.dumps(row_keys, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
