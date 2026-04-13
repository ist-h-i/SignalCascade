from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from .config import TrainingConfig
from .policy_service import apply_selection_policy, policy_utility, smooth_policy_distribution
from .training_service import (
    _gaussian_pit,
    _probabilistic_calibration_score,
    _sign_from_value,
    _should_reset_recurrent_context,
    evaluate_model,
    examples_to_batch,
    split_examples,
)
from ..domain.entities import OHLCVBar, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel
from ..infrastructure.persistence import save_json

DIAGNOSTICS_SCHEMA_VERSION = 12
POLICY_SELECTION_RULE_VERSION = 4
POLICY_SELECTION_OBJECTIVE_FLOOR_RATIO = 0.7
POLICY_SELECTION_BASIS = (
    "pareto_rank_then_near_best_blocked_objective_mean_turnover_mean_"
    "blocked_objective_mean_average_log_wealth_mean_cvar_tail_loss_mean_row_key"
)


def export_review_diagnostics(
    output_dir: Path,
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    checkpoint_audit: dict[str, object] | None = None,
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
    blocked_walk_forward_evaluation = (
        _build_blocked_walk_forward_evaluation(
            model=model,
            validation_examples=validation_examples,
            config=config,
        )
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

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    dataset_payload = {
        "sample_count": len(examples),
        "validation_sample_count": len(validation_examples),
        "source": source_payload,
        "source_rows_original": source_rows_original,
        "source_rows_used": source_rows_used,
        "base_bar_count": None if base_bars is None else len(base_bars),
    }
    validation_payload = diagnostics["summary"]
    state_vector_summary = diagnostics.get("state_vector_summary", {})
    shape_usage_summary = _build_shape_usage_summary(state_vector_summary)
    policy_horizon_summary = _build_policy_horizon_summary(validation_payload)
    policy_calibration_summary = _summarize_policy_calibration_sweep(
        policy_calibration_sweep,
        config=config,
    )
    forecast_quality_scorecards = _build_forecast_quality_scorecards(
        validation=validation_payload,
        validation_rows=diagnostics["validation_rows"],
        validation_sample_count=len(validation_examples),
    )

    summary = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "policy_mode": "shape_aware_profit_maximization",
        "primary_state_reset_mode": config.evaluation_state_reset_mode,
        "checkpoint_audit": dict(checkpoint_audit or {}),
        "dataset": dataset_payload,
        "validation": validation_payload,
        "state_vector_summary": state_vector_summary,
        "shape_usage_summary": shape_usage_summary,
        "policy_horizon_summary": policy_horizon_summary,
        "stateful_evaluation": stateful_evaluation,
        "blocked_walk_forward_evaluation": blocked_walk_forward_evaluation,
        "policy_calibration_sweep": policy_calibration_sweep,
        "policy_calibration_summary": policy_calibration_summary,
        "forecast_quality_scorecards": forecast_quality_scorecards,
        "selection_diagnostics": _build_selection_diagnostics_payload(
            generated_at_utc=generated_at_utc,
            checkpoint_audit=dict(checkpoint_audit or {}),
            dataset=dataset_payload,
            validation=validation_payload,
            state_vector_summary=state_vector_summary,
            shape_usage_summary=shape_usage_summary,
            policy_horizon_summary=policy_horizon_summary,
            stateful_evaluation=stateful_evaluation,
            blocked_walk_forward_evaluation=blocked_walk_forward_evaluation,
            policy_calibration_summary=policy_calibration_summary,
            forecast_quality_scorecards=forecast_quality_scorecards,
        ),
        "runtime_current": _build_runtime_current_payload(
            generated_at_utc=generated_at_utc,
            config=config,
            dataset=dataset_payload,
            policy_calibration_summary=policy_calibration_summary,
        ),
        "paths": {
            "validation_rows_csv": str(validation_rows_path),
            "policy_summary_csv": str(policy_summary_path),
            "horizon_diag_csv": str(horizon_diag_path),
        },
    }
    save_json(summary_path, summary)
    return summary


def _build_selection_policy_calibration_summary(
    policy_calibration_summary: dict[str, object],
) -> dict[str, object]:
    selection_policy_summary = dict(policy_calibration_summary)
    selection_policy_summary.pop("applied_runtime_policy", None)
    selection_policy_summary.pop("applied_runtime_row_key", None)
    selection_policy_summary.pop("applied_runtime_policy_role", None)
    return selection_policy_summary


def _build_selection_diagnostics_payload(
    *,
    generated_at_utc: str,
    checkpoint_audit: dict[str, object],
    dataset: dict[str, object],
    validation: dict[str, object],
    state_vector_summary: dict[str, object],
    shape_usage_summary: dict[str, object],
    policy_horizon_summary: dict[str, object],
    stateful_evaluation: dict[str, object],
    blocked_walk_forward_evaluation: dict[str, object],
    policy_calibration_summary: dict[str, object],
    forecast_quality_scorecards: dict[str, object],
) -> dict[str, object]:
    return {
        "generated_at_utc": generated_at_utc,
        "checkpoint_audit": checkpoint_audit,
        "dataset": dict(dataset),
        "validation": dict(validation),
        "state_vector_summary": dict(state_vector_summary),
        "shape_usage_summary": dict(shape_usage_summary),
        "policy_horizon_summary": dict(policy_horizon_summary),
        "stateful_evaluation": dict(stateful_evaluation),
        "blocked_walk_forward_evaluation": dict(blocked_walk_forward_evaluation),
        "policy_calibration_summary": _build_selection_policy_calibration_summary(
            policy_calibration_summary
        ),
        "forecast_quality_scorecards": dict(forecast_quality_scorecards),
    }


def _build_runtime_current_payload(
    *,
    generated_at_utc: str,
    config: TrainingConfig,
    dataset: dict[str, object],
    policy_calibration_summary: dict[str, object],
) -> dict[str, object]:
    applied_runtime_policy_payload = policy_calibration_summary.get("applied_runtime_policy")
    applied_runtime_policy = (
        dict(applied_runtime_policy_payload)
        if isinstance(applied_runtime_policy_payload, dict)
        else _build_applied_runtime_policy(config)
    )
    dataset_summary = {
        "sample_count": dataset.get("sample_count"),
        "validation_sample_count": dataset.get("validation_sample_count"),
        "source_rows_original": dataset.get("source_rows_original"),
        "source_rows_used": dataset.get("source_rows_used"),
        "base_bar_count": dataset.get("base_bar_count"),
    }
    return {
        "generated_at_utc": generated_at_utc,
        "operating_point": applied_runtime_policy,
        "operating_point_role": str(
            policy_calibration_summary.get("applied_runtime_policy_role")
            or "authoritative_runtime_config"
        ),
        "state_reset_mode": str(config.evaluation_state_reset_mode),
        "dataset": dataset_summary,
        "selection_alignment": {
            "selected_row_key": policy_calibration_summary.get("selected_row_key"),
            "runtime_row_key": applied_runtime_policy.get("row_key"),
            "selected_row_matches_runtime": policy_calibration_summary.get(
                "selected_row_matches_applied_runtime"
            ),
        },
    }


def _build_forecast_quality_scorecards(
    *,
    validation: dict[str, object],
    validation_rows: Sequence[dict[str, object]],
    validation_sample_count: int | None,
) -> dict[str, object]:
    selected_sample_count = (
        int(validation_sample_count)
        if validation_sample_count is not None
        else _resolve_selected_horizon_sample_count(validation, validation_rows)
    )
    selected_horizon = _build_forecast_quality_scorecard(
        scope="selected_horizon",
        sample_count=selected_sample_count,
        directional_accuracy=_finite_float_or_none(validation.get("directional_accuracy")),
        mu_calibration=_finite_float_or_none(validation.get("mu_calibration")),
        sigma_calibration=_finite_float_or_none(validation.get("sigma_calibration")),
        interval_1sigma_coverage=_finite_float_or_none(validation.get("interval_1sigma_coverage")),
        interval_2sigma_coverage=_finite_float_or_none(validation.get("interval_2sigma_coverage")),
        pit_mean=_finite_float_or_none(validation.get("pit_mean")),
        pit_variance=_finite_float_or_none(validation.get("pit_variance")),
        normalized_abs_error=_finite_float_or_none(validation.get("normalized_abs_error")),
        gaussian_nll=_finite_float_or_none(validation.get("gaussian_nll")),
        probabilistic_calibration_score=_finite_float_or_none(
            validation.get("probabilistic_calibration_score")
        ),
    )
    all_horizon = _build_all_horizon_forecast_quality_scorecard(validation_rows)
    selected_quality_score = _finite_float_or_none(selected_horizon.get("quality_score"))
    all_quality_score = _finite_float_or_none(all_horizon.get("quality_score"))
    return {
        "selected_horizon": selected_horizon,
        "all_horizon": all_horizon,
        "quality_score_gap_all_minus_selected": (
            None
            if selected_quality_score is None or all_quality_score is None
            else all_quality_score - selected_quality_score
        ),
    }


def _resolve_selected_horizon_sample_count(
    validation: dict[str, object],
    validation_rows: Sequence[dict[str, object]],
) -> int | None:
    anchor_sample_count = _finite_float_or_none(validation.get("anchor_sample_count"))
    if anchor_sample_count is not None:
        return int(anchor_sample_count)
    selected_sample_ids = {
        int(sample_id)
        for row in validation_rows
        if _is_selected_horizon_row(row)
        for sample_id in [_finite_float_or_none(row.get("sample_id"))]
        if sample_id is not None
    }
    if selected_sample_ids:
        return len(selected_sample_ids)
    return None


def _build_all_horizon_forecast_quality_scorecard(
    validation_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    directional_correct = 0
    interval_1sigma_hits = 0
    interval_2sigma_hits = 0
    sample_count = 0
    mu_errors: list[float] = []
    sigma_errors: list[float] = []
    pit_values: list[float] = []
    normalized_abs_errors: list[float] = []
    gaussian_nll_values: list[float] = []

    for row in validation_rows:
        actual_return = _finite_float_or_none(row.get("y_raw"))
        forecast_mean = _finite_float_or_none(row.get("mu_t", row.get("mu_raw")))
        forecast_sigma = _finite_float_or_none(row.get("sigma_t", row.get("sigma_raw")))
        if actual_return is None or forecast_mean is None or forecast_sigma is None:
            continue
        sigma_safe = max(forecast_sigma, 1e-6)
        forecast_error = actual_return - forecast_mean
        z_score = forecast_error / sigma_safe
        pit_value = _finite_float_or_none(row.get("pit"))
        if pit_value is None:
            pit_value = _gaussian_pit(z_score)
        normalized_abs_error = _finite_float_or_none(row.get("normalized_abs_error"))
        if normalized_abs_error is None:
            normalized_abs_error = abs(z_score)

        sample_count += 1
        directional_correct += int(
            _sign_from_value(forecast_mean) == _sign_from_value(actual_return)
        )
        interval_1sigma_hits += int(abs(forecast_error) <= sigma_safe)
        interval_2sigma_hits += int(abs(forecast_error) <= 2.0 * sigma_safe)
        mu_errors.append(abs(forecast_error))
        sigma_errors.append(abs(abs(forecast_error) - forecast_sigma))
        pit_values.append(pit_value)
        normalized_abs_errors.append(normalized_abs_error)
        gaussian_nll_values.append(
            0.5 * math.log(2.0 * math.pi * (sigma_safe**2)) + (0.5 * (z_score**2))
        )

    if sample_count == 0:
        return _build_forecast_quality_scorecard(
            scope="all_horizon",
            sample_count=0,
            directional_accuracy=None,
            mu_calibration=None,
            sigma_calibration=None,
            interval_1sigma_coverage=None,
            interval_2sigma_coverage=None,
            pit_mean=None,
            pit_variance=None,
            normalized_abs_error=None,
            gaussian_nll=None,
            probabilistic_calibration_score=None,
        )

    pit_mean = sum(pit_values) / sample_count
    pit_variance = sum((value - pit_mean) ** 2 for value in pit_values) / sample_count
    interval_1sigma_coverage = interval_1sigma_hits / sample_count
    interval_2sigma_coverage = interval_2sigma_hits / sample_count
    return _build_forecast_quality_scorecard(
        scope="all_horizon",
        sample_count=sample_count,
        directional_accuracy=directional_correct / sample_count,
        mu_calibration=sum(mu_errors) / sample_count,
        sigma_calibration=sum(sigma_errors) / sample_count,
        interval_1sigma_coverage=interval_1sigma_coverage,
        interval_2sigma_coverage=interval_2sigma_coverage,
        pit_mean=pit_mean,
        pit_variance=pit_variance,
        normalized_abs_error=sum(normalized_abs_errors) / sample_count,
        gaussian_nll=sum(gaussian_nll_values) / sample_count,
        probabilistic_calibration_score=_probabilistic_calibration_score(
            interval_1sigma_coverage=interval_1sigma_coverage,
            interval_2sigma_coverage=interval_2sigma_coverage,
            pit_mean=pit_mean,
            pit_variance=pit_variance,
        ),
    )


def _build_forecast_quality_scorecard(
    *,
    scope: str,
    sample_count: int | None,
    directional_accuracy: float | None,
    mu_calibration: float | None,
    sigma_calibration: float | None,
    interval_1sigma_coverage: float | None,
    interval_2sigma_coverage: float | None,
    pit_mean: float | None,
    pit_variance: float | None,
    normalized_abs_error: float | None,
    gaussian_nll: float | None,
    probabilistic_calibration_score: float | None,
) -> dict[str, object]:
    return {
        "scope": scope,
        "sample_count": sample_count,
        "directional_accuracy": directional_accuracy,
        "mu_calibration": mu_calibration,
        "sigma_calibration": sigma_calibration,
        "interval_1sigma_coverage": interval_1sigma_coverage,
        "interval_2sigma_coverage": interval_2sigma_coverage,
        "pit_mean": pit_mean,
        "pit_variance": pit_variance,
        "normalized_abs_error": normalized_abs_error,
        "gaussian_nll": gaussian_nll,
        "probabilistic_calibration_score": probabilistic_calibration_score,
        "quality_score": _forecast_quality_score(
            directional_accuracy=directional_accuracy,
            mu_calibration=mu_calibration,
            sigma_calibration=sigma_calibration,
            normalized_abs_error=normalized_abs_error,
            probabilistic_calibration_score=probabilistic_calibration_score,
        ),
    }


def _forecast_quality_score(
    *,
    directional_accuracy: float | None,
    mu_calibration: float | None,
    sigma_calibration: float | None,
    normalized_abs_error: float | None,
    probabilistic_calibration_score: float | None,
) -> float | None:
    components = [
        (
            0.30,
            _clamp01(directional_accuracy) if directional_accuracy is not None else None,
        ),
        (0.25, _inverse_linear_score(mu_calibration, upper_bound=0.20)),
        (0.15, _inverse_linear_score(sigma_calibration, upper_bound=0.20)),
        (0.15, _inverse_linear_score(normalized_abs_error, upper_bound=2.0)),
        (
            0.15,
            _clamp01(probabilistic_calibration_score)
            if probabilistic_calibration_score is not None
            else None,
        ),
    ]
    weighted_components = [
        (weight, component)
        for weight, component in components
        if component is not None
    ]
    if not weighted_components:
        return None
    total_weight = sum(weight for weight, _ in weighted_components)
    if total_weight <= 0.0:
        return None
    return sum(weight * component for weight, component in weighted_components) / total_weight


def _inverse_linear_score(value: float | None, *, upper_bound: float) -> float | None:
    if value is None or upper_bound <= 0.0:
        return None
    return _clamp01(1.0 - (float(value) / float(upper_bound)))


def _clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(float(value), 1.0))


def _is_selected_horizon_row(row: dict[str, object]) -> bool:
    for key in ("policy_horizon_selected", "selected"):
        value = _finite_float_or_none(row.get(key))
        if value is not None:
            return int(value) == 1
    return False


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


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
            "sigma_error_sum": 0.0,
            "direction_correct": 0,
            "interval_1sigma_hits": 0,
            "interval_2sigma_hits": 0,
            "pit_sum": 0.0,
            "pit_sq_sum": 0.0,
            "gaussian_nll_sum": 0.0,
            "normalized_abs_error_sum": 0.0,
        }
        for horizon in config.horizons
    }
    state_component_dims: dict[str, int] = {}
    state_component_norm_sums: dict[str, float] = {}
    shape_posterior_sum: torch.Tensor | None = None
    shape_posterior_top_class_counts: dict[str, int] = {}
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
            forecast_mean = outputs.get("forecast_mu", outputs["mu"])[0]
            forecast_sigma = outputs.get("forecast_sigma", outputs["sigma"])[0]
            policy_mean = outputs.get("policy_mu", outputs["mu"])[0]
            policy_sigma = outputs.get("policy_sigma", outputs["sigma"])[0]
            gate = float(outputs["tradeability_gate"][0].item())
            entropy = float(outputs["shape_entropy"][0].item())
            smooth_policy = smooth_policy_distribution(
                mean=outputs.get("policy_mu", outputs["mu"]),
                sigma=outputs.get("policy_sigma", outputs["sigma"]),
                costs=batch["horizon_costs"],
                tradeability_gate=outputs["tradeability_gate"],
                previous_position=torch.tensor([previous_position], dtype=policy_mean.dtype),
                config=config,
            )
            decision = apply_selection_policy(
                example=example,
                mean=policy_mean.tolist(),
                sigma=policy_sigma.tolist(),
                config=config,
                previous_position=previous_position,
                tradeability_gate=gate,
                shape_probs=outputs["shape_posterior"][0].tolist(),
            )
            selected_row = dict(decision["selected_row"])
            selected_horizon = int(selected_row["horizon"])
            selected_index = config.horizons.index(selected_horizon)
            realized_return = float(example.returns_target[selected_index])
            trade_cost = float(selected_row["cost"]) * abs(float(decision["trade_delta"]))
            pnl = (float(decision["position"]) * realized_return) - trade_cost
            if not state_component_dims and "state_vector_component_dims" in outputs:
                state_component_dims = {
                    str(key): int(value)
                    for key, value in dict(outputs["state_vector_component_dims"]).items()
                }
            state_components = outputs.get("state_vector_components")
            if isinstance(state_components, dict):
                for name, tensor in state_components.items():
                    state_component_norm_sums[str(name)] = state_component_norm_sums.get(str(name), 0.0) + (
                        float(torch.linalg.vector_norm(tensor[0], ord=2).item())
                    )
            shape_posterior = outputs.get("shape_posterior", outputs.get("shape_probs"))
            if isinstance(shape_posterior, torch.Tensor):
                if shape_posterior_sum is None:
                    shape_posterior_sum = torch.zeros(
                        shape_posterior.size(-1),
                        dtype=torch.float64,
                    )
                shape_posterior_sum += shape_posterior[0].detach().to(dtype=torch.float64).cpu()
                top_class = str(int(torch.argmax(shape_posterior[0]).item()))
                shape_posterior_top_class_counts[top_class] = (
                    shape_posterior_top_class_counts.get(top_class, 0) + 1
                )
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
                forecast_mu = float(forecast_mean[horizon_index].item())
                forecast_sigma_value = float(forecast_sigma[horizon_index].item())
                sigma_safe = max(forecast_sigma_value, 1e-6)
                forecast_error = actual_return - forecast_mu
                z_score = forecast_error / sigma_safe
                pit_value = _gaussian_pit(z_score)
                inside_one_sigma = int(abs(forecast_error) <= sigma_safe)
                inside_two_sigma = int(abs(forecast_error) <= 2.0 * sigma_safe)
                horizon_stats[horizon_key]["count"] += 1
                horizon_stats[horizon_key]["selected"] += int(int(row["horizon"]) == selected_horizon)
                horizon_stats[horizon_key]["utility_sum"] += float(row["policy_utility"])
                horizon_stats[horizon_key]["position_sum"] += float(row["position"])
                horizon_stats[horizon_key]["abs_error_sum"] += abs(forecast_error)
                horizon_stats[horizon_key]["sigma_error_sum"] += abs(
                    abs(forecast_error) - forecast_sigma_value
                )
                horizon_stats[horizon_key]["direction_correct"] += int(
                    _sign_from_value(forecast_mu) == _sign_from_value(actual_return)
                )
                horizon_stats[horizon_key]["interval_1sigma_hits"] += inside_one_sigma
                horizon_stats[horizon_key]["interval_2sigma_hits"] += inside_two_sigma
                horizon_stats[horizon_key]["pit_sum"] += pit_value
                horizon_stats[horizon_key]["pit_sq_sum"] += pit_value * pit_value
                horizon_stats[horizon_key]["gaussian_nll_sum"] += (
                    0.5 * math.log(2.0 * math.pi * (sigma_safe**2))
                    + (0.5 * (z_score**2))
                )
                horizon_stats[horizon_key]["normalized_abs_error_sum"] += abs(z_score)
                validation_rows.append(
                    {
                        "sample_id": sample_id,
                        "timestamp": example.anchor_time.isoformat(),
                        "regime_id": example.regime_id,
                        "horizon": int(row["horizon"]),
                        "y_raw": actual_return,
                        "mu_t": forecast_mu,
                        "mu_raw": forecast_mu,
                        "sigma_t": forecast_sigma_value,
                        "sigma_raw": forecast_sigma_value,
                        "policy_mu_t": float(row["mean"]),
                        "policy_sigma_t": float(row["sigma"]),
                        "sigma_t_sq": float(row["sigma_sq"]),
                        "g_t": float(row["g_t"]),
                        "mu_t_tilde": float(row["mu_t_tilde"]),
                        "gated_mean": float(row["gated_mean"]),
                        "cost": float(row["cost"]),
                        "selected_policy_utility": float(row["policy_utility"]),
                        "policy_utility": float(row["policy_utility"]),
                        "q_t_candidate": float(row["position"]),
                        "position_if_chosen": float(row["position"]),
                        "tradeability_gate": gate,
                        "shape_entropy": entropy,
                        "selected_horizon": selected_horizon,
                        "executed_horizon": decision["executed_horizon"],
                        "smooth_policy_horizon": smooth_selected_horizon,
                        "smooth_position": smooth_selected_position,
                        "smooth_no_trade_band": int(smooth_selected_no_trade),
                        "policy_horizon_selected": int(int(row["horizon"]) == selected_horizon),
                        "selected": int(int(row["horizon"]) == selected_horizon),
                        "no_trade_band": int(bool(row["no_trade_band"])),
                        "abs_error": abs(forecast_error),
                        "sigma_abs_error": abs(abs(forecast_error) - forecast_sigma_value),
                        "normalized_abs_error": abs(z_score),
                        "pit": pit_value,
                        "inside_one_sigma": inside_one_sigma,
                        "inside_two_sigma": inside_two_sigma,
                    }
                )

            policy_summary.append(
                {
                    "sample_id": sample_id,
                    "timestamp": example.anchor_time.isoformat(),
                    "regime_id": example.regime_id,
                    "policy_horizon": decision["policy_horizon"],
                    "executed_horizon": decision["executed_horizon"],
                    "q_t_prev": float(decision["previous_position"]),
                    "previous_position": float(decision["previous_position"]),
                    "q_t": float(decision["position"]),
                    "position": float(decision["position"]),
                    "q_t_trade_delta": float(decision["trade_delta"]),
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
                    "selected_g_t": float(selected_row["g_t"]),
                    "selected_mu_t": float(forecast_mean[selected_index].item()),
                    "selected_sigma_t": float(forecast_sigma[selected_index].item()),
                    "selected_policy_mu_t": float(selected_row["mean"]),
                    "selected_policy_sigma_t": float(selected_row["sigma"]),
                    "selected_sigma_t_sq": float(selected_row["sigma_sq"]),
                    "selected_mu_t_tilde": float(selected_row["mu_t_tilde"]),
                    "selected_policy_utility": float(decision["selected_policy_utility"]),
                    "policy_score": float(decision["selected_policy_utility"]),
                    "selected_abs_error": abs(realized_return - float(forecast_mean[selected_index].item())),
                    "selected_sigma_abs_error": abs(
                        abs(realized_return - float(forecast_mean[selected_index].item()))
                        - float(forecast_sigma[selected_index].item())
                    ),
                    "selected_pit": _gaussian_pit(
                        (
                            realized_return - float(forecast_mean[selected_index].item())
                        )
                        / max(float(forecast_sigma[selected_index].item()), 1e-6)
                    ),
                    "selected_inside_one_sigma": int(
                        abs(realized_return - float(forecast_mean[selected_index].item()))
                        <= max(float(forecast_sigma[selected_index].item()), 1e-6)
                    ),
                    "selected_inside_two_sigma": int(
                        abs(realized_return - float(forecast_mean[selected_index].item()))
                        <= (2.0 * max(float(forecast_sigma[selected_index].item()), 1e-6))
                    ),
                    "tradeability_gate": gate,
                    "shape_entropy": entropy,
                    "realized_return": realized_return,
                    "realized_pnl": pnl,
                }
            )
            previous_position = float(decision["position"])
            previous_state = outputs["memory_state"].detach()
            previous_example = example
            sample_id += 1

    horizon_diag = []
    for horizon, stats in sorted(horizon_stats.items(), key=lambda item: int(item[0])):
        sample_count = max(int(stats["count"]), 1)
        pit_mean = float(stats["pit_sum"]) / sample_count
        pit_variance = max(
            (float(stats["pit_sq_sum"]) / sample_count) - (pit_mean**2),
            0.0,
        )
        interval_1sigma_coverage = float(stats["interval_1sigma_hits"]) / sample_count
        interval_2sigma_coverage = float(stats["interval_2sigma_hits"]) / sample_count
        horizon_diag.append(
            {
                "horizon": int(horizon),
                "sample_count": int(stats["count"]),
                "policy_horizon_share": int(stats["selected"]) / sample_count,
                "selection_rate": int(stats["selected"]) / sample_count,
                "mean_policy_utility": float(stats["utility_sum"]) / sample_count,
                "mean_position": float(stats["position_sum"]) / sample_count,
                "mu_calibration": float(stats["abs_error_sum"]) / sample_count,
                "sigma_calibration": float(stats["sigma_error_sum"]) / sample_count,
                "directional_accuracy": float(stats["direction_correct"]) / sample_count,
                "interval_1sigma_coverage": interval_1sigma_coverage,
                "interval_2sigma_coverage": interval_2sigma_coverage,
                "pit_mean": pit_mean,
                "pit_variance": pit_variance,
                "normalized_abs_error": float(stats["normalized_abs_error_sum"]) / sample_count,
                "gaussian_nll": float(stats["gaussian_nll_sum"]) / sample_count,
                "probabilistic_calibration_score": _probabilistic_calibration_score(
                    interval_1sigma_coverage=interval_1sigma_coverage,
                    interval_2sigma_coverage=interval_2sigma_coverage,
                    pit_mean=pit_mean,
                    pit_variance=pit_variance,
                ),
            }
        )
    state_vector_summary = _build_state_vector_summary(
        sample_count=sample_id,
        state_component_dims=state_component_dims,
        state_component_norm_sums=state_component_norm_sums,
        shape_posterior_sum=shape_posterior_sum,
        shape_posterior_top_class_counts=shape_posterior_top_class_counts,
    )
    return {
        "validation_rows": validation_rows,
        "policy_summary": policy_summary,
        "horizon_diag": horizon_diag,
        "state_vector_summary": state_vector_summary,
        "summary": summary,
    }


def build_validation_snapshots(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    diagnostics = build_validation_diagnostics(model, validation_examples, config)
    return diagnostics["policy_summary"]


def _build_shape_usage_summary(state_vector_summary: dict[str, object]) -> dict[str, object]:
    top_class_share = state_vector_summary.get("shape_posterior_top_class_share")
    resolved_top_class_share = (
        dict(top_class_share) if isinstance(top_class_share, dict) else {}
    )
    if not resolved_top_class_share:
        return {
            "dominant_class": None,
            "dominant_share": None,
            "collapsed": None,
        }
    dominant_class, dominant_share = max(
        resolved_top_class_share.items(),
        key=lambda item: float(item[1]),
    )
    dominant_share_value = float(dominant_share)
    return {
        "dominant_class": str(dominant_class),
        "dominant_share": dominant_share_value,
        "collapsed": dominant_share_value >= 0.70,
    }


def _build_policy_horizon_summary(validation_summary: dict[str, object]) -> dict[str, object]:
    distribution_payload = validation_summary.get("policy_horizon_distribution")
    distribution = (
        {
            str(key): float(value)
            for key, value in dict(distribution_payload).items()
        }
        if isinstance(distribution_payload, dict)
        else {}
    )
    if not distribution:
        return {
            "dominant_horizon": None,
            "dominant_share": None,
            "collapsed": None,
        }
    dominant_horizon, dominant_share = max(
        distribution.items(),
        key=lambda item: float(item[1]),
    )
    dominant_share_value = float(dominant_share)
    return {
        "distribution": distribution,
        "dominant_horizon": int(dominant_horizon),
        "dominant_share": dominant_share_value,
        "collapsed": dominant_share_value >= 0.75,
    }


def _build_blocked_walk_forward_evaluation(
    *,
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, object]:
    folds = _split_validation_examples_into_contiguous_folds(
        validation_examples,
        fold_count=config.walk_forward_folds,
    )
    if not folds:
        return {}

    state_reset_modes: dict[str, dict[str, object]] = {}
    best_mode_by_mean = None
    best_mode_mean = None
    for mode in config.diagnostic_state_reset_modes:
        fold_rows: list[dict[str, object]] = []
        for fold_index, fold_examples in enumerate(folds, start=1):
            metrics = evaluate_model(
                model=model,
                examples=fold_examples,
                config=config,
                state_reset_mode=mode,
            )
            fold_rows.append(
                {
                    "fold_index": fold_index,
                    "sample_count": len(fold_examples),
                    "start_timestamp": fold_examples[0].anchor_time.isoformat(),
                    "end_timestamp": fold_examples[-1].anchor_time.isoformat(),
                    "average_log_wealth": float(metrics["average_log_wealth"]),
                    "cvar_tail_loss": float(metrics["cvar_tail_loss"]),
                    "turnover": float(metrics["turnover"]),
                    "directional_accuracy": float(metrics["directional_accuracy"]),
                    "exact_smooth_position_mae": float(metrics["exact_smooth_position_mae"]),
                    "interval_1sigma_coverage": float(metrics["interval_1sigma_coverage"]),
                    "interval_2sigma_coverage": float(metrics["interval_2sigma_coverage"]),
                    "probabilistic_calibration_score": float(
                        metrics["probabilistic_calibration_score"]
                    ),
                }
            )
        wealth_values = [float(row["average_log_wealth"]) for row in fold_rows]
        turnover_values = [float(row["turnover"]) for row in fold_rows]
        direction_values = [float(row["directional_accuracy"]) for row in fold_rows]
        position_mae_values = [float(row["exact_smooth_position_mae"]) for row in fold_rows]
        interval_1sigma_values = [float(row["interval_1sigma_coverage"]) for row in fold_rows]
        interval_2sigma_values = [float(row["interval_2sigma_coverage"]) for row in fold_rows]
        probabilistic_score_values = [
            float(row["probabilistic_calibration_score"]) for row in fold_rows
        ]
        mode_summary = {
            "fold_count": len(fold_rows),
            "average_log_wealth_mean": sum(wealth_values) / max(len(wealth_values), 1),
            "average_log_wealth_min": min(wealth_values, default=0.0),
            "average_log_wealth_max": max(wealth_values, default=0.0),
            "turnover_mean": sum(turnover_values) / max(len(turnover_values), 1),
            "directional_accuracy_mean": sum(direction_values) / max(len(direction_values), 1),
            "exact_smooth_position_mae_mean": sum(position_mae_values) / max(len(position_mae_values), 1),
            "interval_1sigma_coverage_mean": sum(interval_1sigma_values)
            / max(len(interval_1sigma_values), 1),
            "interval_2sigma_coverage_mean": sum(interval_2sigma_values)
            / max(len(interval_2sigma_values), 1),
            "probabilistic_calibration_score_mean": sum(probabilistic_score_values)
            / max(len(probabilistic_score_values), 1),
            "folds": fold_rows,
        }
        state_reset_modes[mode] = mode_summary
        if best_mode_mean is None or float(mode_summary["average_log_wealth_mean"]) > float(best_mode_mean):
            best_mode_by_mean = mode
            best_mode_mean = float(mode_summary["average_log_wealth_mean"])

    return {
        "fold_count": len(folds),
        "fold_sample_counts": [len(fold) for fold in folds],
        "state_reset_modes": state_reset_modes,
        "best_state_reset_mode_by_mean_log_wealth": best_mode_by_mean,
    }


def _split_validation_examples_into_contiguous_folds(
    validation_examples: Sequence[TrainingExample],
    *,
    fold_count: int,
) -> list[list[TrainingExample]]:
    total_examples = len(validation_examples)
    if total_examples == 0:
        return []
    resolved_fold_count = max(1, min(int(fold_count), total_examples))
    base_fold_size = total_examples // resolved_fold_count
    remainder = total_examples % resolved_fold_count
    folds: list[list[TrainingExample]] = []
    start_index = 0
    for fold_index in range(resolved_fold_count):
        current_size = base_fold_size + (1 if fold_index < remainder else 0)
        end_index = start_index + current_size
        folds.append(list(validation_examples[start_index:end_index]))
        start_index = end_index
    return [fold for fold in folds if fold]


def _build_policy_calibration_sweep(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    sweep_rows: list[dict[str, object]] = []
    folds = _split_validation_examples_into_contiguous_folds(
        validation_examples,
        fold_count=config.walk_forward_folds,
    )
    cost_multipliers = _resolve_policy_sweep_axis(
        configured_values=config.policy_sweep_cost_multipliers,
        default_values=(0.5, 1.0, 2.0, 4.0),
        current_value=float(config.policy_cost_multiplier),
    )
    gamma_multipliers = _resolve_policy_sweep_axis(
        configured_values=config.policy_sweep_gamma_multipliers,
        default_values=(0.5, 1.0, 2.0),
        current_value=float(config.policy_gamma_multiplier),
    )
    q_max_values = tuple(float(value) for value in config.policy_sweep_q_max_values)
    if q_max_values == (1.0,) and float(config.q_max) != 1.0:
        q_max_values = (float(config.q_max),)
    cvar_weights = tuple(float(value) for value in config.policy_sweep_cvar_weights)
    if cvar_weights == (0.20,) and float(config.cvar_weight) != 0.20:
        cvar_weights = (float(config.cvar_weight),)
    for state_reset_mode in config.policy_sweep_state_reset_modes:
        for cvar_weight in cvar_weights:
            for q_max in q_max_values:
                for min_policy_sigma in config.policy_sweep_min_policy_sigmas:
                    for gamma_multiplier in gamma_multipliers:
                        for cost_multiplier in cost_multipliers:
                            sweep_config = replace(
                                config,
                                cvar_weight=float(cvar_weight),
                                q_max=float(q_max),
                                min_policy_sigma=float(min_policy_sigma),
                            )
                            metrics = evaluate_model(
                                model=model,
                                examples=validation_examples,
                                config=sweep_config,
                                state_reset_mode=state_reset_mode,
                                cost_multiplier=float(cost_multiplier),
                                gamma_multiplier=float(gamma_multiplier),
                            )
                            blocked_metrics = _evaluate_blocked_policy_sweep_row(
                                model=model,
                                folds=folds,
                                config=sweep_config,
                                state_reset_mode=state_reset_mode,
                                cost_multiplier=float(cost_multiplier),
                                gamma_multiplier=float(gamma_multiplier),
                            )
                            objective_value = _policy_sweep_objective_value(
                                average_log_wealth=float(metrics["average_log_wealth"]),
                                cvar_tail_loss=float(metrics["cvar_tail_loss"]),
                                cvar_weight=float(cvar_weight),
                            )
                            sweep_rows.append(
                                {
                                    "row_key": _policy_sweep_row_key(
                                        state_reset_mode=state_reset_mode,
                                        cost_multiplier=float(cost_multiplier),
                                        gamma_multiplier=float(gamma_multiplier),
                                        min_policy_sigma=float(min_policy_sigma),
                                        q_max=float(q_max),
                                        cvar_weight=float(cvar_weight),
                                    ),
                                    "state_reset_mode": state_reset_mode,
                                    "cost_multiplier": float(cost_multiplier),
                                    "gamma_multiplier": float(gamma_multiplier),
                                    "min_policy_sigma": float(min_policy_sigma),
                                    "q_max": float(q_max),
                                    "cvar_weight": float(cvar_weight),
                                    "average_log_wealth": float(metrics["average_log_wealth"]),
                                    "objective_log_wealth_minus_lambda_cvar": objective_value,
                                    "turnover": float(metrics["turnover"]),
                                    "cvar_tail_loss": float(metrics["cvar_tail_loss"]),
                                    "no_trade_band_hit_rate": float(metrics["no_trade_band_hit_rate"]),
                                    "exact_smooth_horizon_agreement": float(
                                        metrics["exact_smooth_horizon_agreement"]
                                    ),
                                    "exact_smooth_no_trade_agreement": float(
                                        metrics["exact_smooth_no_trade_agreement"]
                                    ),
                                    "exact_smooth_position_mae": float(
                                        metrics["exact_smooth_position_mae"]
                                    ),
                                    "exact_smooth_utility_regret": float(
                                        metrics["exact_smooth_utility_regret"]
                                    ),
                                    "blocked_walk_forward": blocked_metrics,
                                    "blocked_average_log_wealth_mean": float(
                                        blocked_metrics["average_log_wealth_mean"]
                                    ),
                                    "blocked_cvar_tail_loss_mean": float(
                                        blocked_metrics["cvar_tail_loss_mean"]
                                    ),
                                    "blocked_turnover_mean": float(
                                        blocked_metrics["turnover_mean"]
                                    ),
                                    "blocked_directional_accuracy_mean": float(
                                        blocked_metrics["directional_accuracy_mean"]
                                    ),
                                    "blocked_exact_smooth_position_mae_mean": float(
                                        blocked_metrics["exact_smooth_position_mae_mean"]
                                    ),
                                    "blocked_interval_1sigma_coverage_mean": float(
                                        blocked_metrics["interval_1sigma_coverage_mean"]
                                    ),
                                    "blocked_interval_2sigma_coverage_mean": float(
                                        blocked_metrics["interval_2sigma_coverage_mean"]
                                    ),
                                    "blocked_probabilistic_calibration_score_mean": float(
                                        blocked_metrics["probabilistic_calibration_score_mean"]
                                    ),
                                    "blocked_objective_log_wealth_minus_lambda_cvar_mean": float(
                                        blocked_metrics[
                                            "objective_log_wealth_minus_lambda_cvar_mean"
                                        ]
                                    ),
                                }
                            )
    return _annotate_policy_calibration_sweep(sweep_rows)


def _resolve_policy_sweep_axis(
    *,
    configured_values: Sequence[float],
    default_values: Sequence[float],
    current_value: float,
) -> tuple[float, ...]:
    resolved_values = tuple(float(value) for value in configured_values)
    if (
        tuple(resolved_values) == tuple(float(value) for value in default_values)
        and float(current_value) not in resolved_values
    ):
        resolved_values = tuple(sorted({*resolved_values, float(current_value)}))
    return resolved_values


def _evaluate_blocked_policy_sweep_row(
    *,
    model: SignalCascadeModel,
    folds: Sequence[Sequence[TrainingExample]],
    config: TrainingConfig,
    state_reset_mode: str,
    cost_multiplier: float,
    gamma_multiplier: float,
) -> dict[str, object]:
    fold_rows: list[dict[str, object]] = []
    for fold_index, fold_examples in enumerate(folds, start=1):
        metrics = evaluate_model(
            model=model,
            examples=fold_examples,
            config=config,
            state_reset_mode=state_reset_mode,
            cost_multiplier=cost_multiplier,
            gamma_multiplier=gamma_multiplier,
        )
        objective_value = _policy_sweep_objective_value(
            average_log_wealth=float(metrics["average_log_wealth"]),
            cvar_tail_loss=float(metrics["cvar_tail_loss"]),
            cvar_weight=float(config.cvar_weight),
        )
        fold_rows.append(
            {
                "fold_index": fold_index,
                "sample_count": len(fold_examples),
                "start_timestamp": fold_examples[0].anchor_time.isoformat(),
                "end_timestamp": fold_examples[-1].anchor_time.isoformat(),
                "average_log_wealth": float(metrics["average_log_wealth"]),
                "cvar_tail_loss": float(metrics["cvar_tail_loss"]),
                "turnover": float(metrics["turnover"]),
                "directional_accuracy": float(metrics["directional_accuracy"]),
                "exact_smooth_position_mae": float(metrics["exact_smooth_position_mae"]),
                "interval_1sigma_coverage": float(metrics["interval_1sigma_coverage"]),
                "interval_2sigma_coverage": float(metrics["interval_2sigma_coverage"]),
                "probabilistic_calibration_score": float(
                    metrics["probabilistic_calibration_score"]
                ),
                "objective_log_wealth_minus_lambda_cvar": objective_value,
            }
        )

    return {
        "fold_count": len(fold_rows),
        "fold_sample_counts": [int(row["sample_count"]) for row in fold_rows],
        "average_log_wealth_mean": _policy_sweep_fold_mean(fold_rows, "average_log_wealth"),
        "cvar_tail_loss_mean": _policy_sweep_fold_mean(fold_rows, "cvar_tail_loss"),
        "turnover_mean": _policy_sweep_fold_mean(fold_rows, "turnover"),
        "directional_accuracy_mean": _policy_sweep_fold_mean(fold_rows, "directional_accuracy"),
        "exact_smooth_position_mae_mean": _policy_sweep_fold_mean(
            fold_rows,
            "exact_smooth_position_mae",
        ),
        "interval_1sigma_coverage_mean": _policy_sweep_fold_mean(
            fold_rows,
            "interval_1sigma_coverage",
        ),
        "interval_2sigma_coverage_mean": _policy_sweep_fold_mean(
            fold_rows,
            "interval_2sigma_coverage",
        ),
        "probabilistic_calibration_score_mean": _policy_sweep_fold_mean(
            fold_rows,
            "probabilistic_calibration_score",
        ),
        "objective_log_wealth_minus_lambda_cvar_mean": _policy_sweep_fold_mean(
            fold_rows,
            "objective_log_wealth_minus_lambda_cvar",
        ),
        "folds": fold_rows,
    }


def _policy_sweep_fold_mean(
    fold_rows: Sequence[dict[str, object]],
    key: str,
) -> float:
    values = [float(row[key]) for row in fold_rows]
    return sum(values) / max(len(values), 1)


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
            -_policy_sweep_sort_objective(item),
            -_policy_sweep_sort_average_log_wealth(item),
            _policy_sweep_sort_cvar_tail_loss(item),
            _policy_sweep_sort_turnover(item),
            str(item["row_key"]),
        ),
    )


def _policy_sweep_row_dominates(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    comparisons = (
        _policy_sweep_sort_average_log_wealth(left) >= _policy_sweep_sort_average_log_wealth(right),
        _policy_sweep_sort_turnover(left) <= _policy_sweep_sort_turnover(right),
        _policy_sweep_sort_cvar_tail_loss(left) <= _policy_sweep_sort_cvar_tail_loss(right),
    )
    strictly_better = (
        _policy_sweep_sort_average_log_wealth(left) > _policy_sweep_sort_average_log_wealth(right)
        or _policy_sweep_sort_turnover(left) < _policy_sweep_sort_turnover(right)
        or _policy_sweep_sort_cvar_tail_loss(left) < _policy_sweep_sort_cvar_tail_loss(right)
    )
    return all(comparisons) and strictly_better


def _summarize_policy_calibration_sweep(
    sweep_rows: Sequence[dict[str, object]],
    *,
    config: TrainingConfig | None = None,
) -> dict[str, object]:
    pareto_rows = [dict(row) for row in sweep_rows if not bool(row.get("dominated", False))]
    best_row = pareto_rows[0] if pareto_rows else (dict(sweep_rows[0]) if sweep_rows else None)
    selected_row = _select_policy_calibration_row(pareto_rows, best_row)
    summary = {
        "row_count": len(sweep_rows),
        "pareto_optimal_count": len(pareto_rows),
        "dominated_count": max(len(sweep_rows) - len(pareto_rows), 0),
        "policy_calibration_rows_sha256": _policy_sweep_rows_sha256(sweep_rows),
        "selection_basis": POLICY_SELECTION_BASIS,
        "selection_rule_version": POLICY_SELECTION_RULE_VERSION,
        "selected_row": selected_row,
        "selected_row_key": None if selected_row is None else selected_row.get("row_key"),
        "selected_row_role": "diagnostic_recommendation_not_applied_runtime_config",
        "best_row": best_row,
    }
    if config is not None:
        applied_runtime_policy = _build_applied_runtime_policy(config)
        summary.update(
            {
                "applied_runtime_policy": applied_runtime_policy,
                "applied_runtime_row_key": applied_runtime_policy["row_key"],
                "applied_runtime_policy_role": "authoritative_runtime_config",
                "selected_row_matches_applied_runtime": (
                    False
                    if selected_row is None
                    else selected_row.get("row_key") == applied_runtime_policy["row_key"]
                ),
            }
        )
    return summary


def _select_policy_calibration_row(
    pareto_rows: Sequence[dict[str, object]],
    best_row: dict[str, object] | None,
) -> dict[str, object] | None:
    if best_row is None:
        return None
    best_objective = _policy_sweep_sort_objective(best_row)
    if best_objective > 0.0:
        objective_floor = best_objective * POLICY_SELECTION_OBJECTIVE_FLOOR_RATIO
    else:
        objective_floor = best_objective
    candidate_rows = [
        dict(row)
        for row in pareto_rows
        if _policy_sweep_sort_objective(row) >= objective_floor
    ]
    if not candidate_rows:
        candidate_rows = [dict(best_row)]
    candidate_rows.sort(
        key=lambda item: (
            _policy_sweep_sort_turnover(item),
            -_policy_sweep_sort_objective(item),
            -_policy_sweep_sort_average_log_wealth(item),
            _policy_sweep_sort_cvar_tail_loss(item),
            str(item["row_key"]),
        )
    )
    return candidate_rows[0]


def _policy_sweep_row_key(
    *,
    state_reset_mode: str,
    cost_multiplier: float,
    gamma_multiplier: float,
    min_policy_sigma: float,
    q_max: float,
    cvar_weight: float,
) -> str:
    return "|".join(
        (
            f"state_reset_mode={state_reset_mode}",
            f"cost_multiplier={cost_multiplier:.12g}",
            f"gamma_multiplier={gamma_multiplier:.12g}",
            f"min_policy_sigma={min_policy_sigma:.12g}",
            f"q_max={q_max:.12g}",
            f"cvar_weight={cvar_weight:.12g}",
        )
    )


def _build_applied_runtime_policy(config: TrainingConfig) -> dict[str, object]:
    return {
        "row_key": _policy_sweep_row_key(
            state_reset_mode=str(config.evaluation_state_reset_mode),
            cost_multiplier=float(config.policy_cost_multiplier),
            gamma_multiplier=float(config.policy_gamma_multiplier),
            min_policy_sigma=float(config.min_policy_sigma),
            q_max=float(config.q_max),
            cvar_weight=float(config.cvar_weight),
        ),
        "state_reset_mode": str(config.evaluation_state_reset_mode),
        "cost_multiplier": float(config.policy_cost_multiplier),
        "gamma_multiplier": float(config.policy_gamma_multiplier),
        "min_policy_sigma": float(config.min_policy_sigma),
        "q_max": float(config.q_max),
        "cvar_weight": float(config.cvar_weight),
    }


def _policy_sweep_sort_average_log_wealth(row: dict[str, object]) -> float:
    return float(row.get("blocked_average_log_wealth_mean", row["average_log_wealth"]))


def _policy_sweep_sort_cvar_tail_loss(row: dict[str, object]) -> float:
    return float(row.get("blocked_cvar_tail_loss_mean", row["cvar_tail_loss"]))


def _policy_sweep_sort_turnover(row: dict[str, object]) -> float:
    return float(row.get("blocked_turnover_mean", row["turnover"]))


def _policy_sweep_sort_objective(row: dict[str, object]) -> float:
    if "blocked_objective_log_wealth_minus_lambda_cvar_mean" in row:
        return float(row["blocked_objective_log_wealth_minus_lambda_cvar_mean"])
    if "objective_log_wealth_minus_lambda_cvar" in row:
        return float(row["objective_log_wealth_minus_lambda_cvar"])
    return _policy_sweep_objective_value(
        average_log_wealth=float(row["average_log_wealth"]),
        cvar_tail_loss=float(row["cvar_tail_loss"]),
        cvar_weight=float(row.get("cvar_weight", 0.0)),
    )


def _policy_sweep_objective_value(
    *,
    average_log_wealth: float,
    cvar_tail_loss: float,
    cvar_weight: float,
) -> float:
    return float(average_log_wealth) - (float(cvar_weight) * float(cvar_tail_loss))


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


def _build_state_vector_summary(
    *,
    sample_count: int,
    state_component_dims: dict[str, int],
    state_component_norm_sums: dict[str, float],
    shape_posterior_sum: torch.Tensor | None,
    shape_posterior_top_class_counts: dict[str, int],
) -> dict[str, object]:
    denominator = max(sample_count, 1)
    component_l2_mean = {
        str(name): float(total) / denominator
        for name, total in state_component_norm_sums.items()
    }
    shape_posterior_mean = (
        {
            str(index): float(value)
            for index, value in enumerate((shape_posterior_sum / denominator).tolist())
        }
        if shape_posterior_sum is not None
        else {}
    )
    return {
        "sample_count": int(sample_count),
        "component_dims": dict(state_component_dims),
        "component_l2_mean": component_l2_mean,
        "shape_posterior_mean": shape_posterior_mean,
        "shape_posterior_top_class_share": {
            str(index): float(count) / denominator
            for index, count in sorted(shape_posterior_top_class_counts.items(), key=lambda item: int(item[0]))
        },
    }


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
