from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .selection_rank_diagnostics import (
    backfill_forecast_quality_metrics_from_session,
    build_forecast_quality_ranking_diagnostics,
    build_selection_divergence_scorecard,
    build_selection_history_summary,
)
from ..infrastructure.persistence import load_json, save_json

JST = timezone(timedelta(hours=9))
UTC = timezone.utc
METRICS_SCHEMA_VERSION = 4
ANALYSIS_SCHEMA_VERSION = 14
REQUIRED_LIVE_DIAGNOSTICS_FILES = (
    "validation_summary.json",
    "policy_summary.csv",
    "horizon_diag.csv",
)


def _display_or_missing(value: object, digits: int = 6) -> str:
    if value is None:
        return "missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    return f"{numeric:.{digits}f}"


def _display_int_or_missing(value: object) -> str:
    if value is None:
        return "missing"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "missing"


def _display_divergence_cluster_counts(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "-"
    return ", ".join(
        f"{key}={int(value)}"
        for key, value in sorted(payload.items())
        if key is not None and value is not None
    ) or "-"


def _render_divergence_row(record: object) -> str:
    if not isinstance(record, dict):
        return "-"
    accepted_candidate = record.get("accepted_candidate") or "-"
    production_candidate = record.get("production_current_candidate") or "-"
    accepted_ranks = "/".join(
        _display_int_or_missing(record.get(key))
        for key in (
            "accepted_candidate_current_rank",
            "accepted_candidate_selected_horizon_rank",
            "accepted_candidate_all_horizon_rank",
        )
    )
    production_ranks = "/".join(
        _display_int_or_missing(record.get(key))
        for key in (
            "production_current_current_rank",
            "production_current_selected_horizon_rank",
            "production_current_all_horizon_rank",
        )
    )
    return (
        f"{record.get('session_id', '-')}"
        f" [{record.get('coverage_status', '-')}/{record.get('failure_mode_cluster', '-')}]"
        f" acc=`{accepted_candidate}`"
        f" ranks=`{accepted_ranks}`"
        f" turn=`{_display_or_missing(record.get('accepted_candidate_blocked_turnover_mean'))}`"
        f" mae=`{_display_or_missing(record.get('accepted_candidate_blocked_exact_smooth_position_mae_mean'))}`"
        f" mdd=`{_display_or_missing(record.get('accepted_candidate_max_drawdown'))}`"
        f" prod=`{production_candidate}`"
        f" ranks=`{production_ranks}`"
        f" turn=`{_display_or_missing(record.get('production_current_blocked_turnover_mean'))}`"
        f" mae=`{_display_or_missing(record.get('production_current_blocked_exact_smooth_position_mae_mean'))}`"
        f" mdd=`{_display_or_missing(record.get('production_current_max_drawdown'))}`"
    )


def load_required_diagnostics_summary(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    missing_files = [
        name for name in REQUIRED_LIVE_DIAGNOSTICS_FILES if not (output_dir / name).exists()
    ]
    artifact_label = "current artifact" if output_dir.name == "current" else "artifact"
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"diagnostics unpublished for {artifact_label}: missing {missing} under {output_dir}"
        )

    diagnostics_summary = load_json(output_dir / "validation_summary.json")
    generated_at_utc = diagnostics_summary.get("generated_at_utc")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.strip():
        raise ValueError(
            "diagnostics unpublished for "
            f"{artifact_label}: validation_summary.json is missing generated_at_utc under {output_dir}"
        )
    return diagnostics_summary


def generate_research_report(
    output_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    metrics = load_json(output_dir / "metrics.json")
    prediction = load_json(output_dir / "prediction.json")
    config = load_json(output_dir / "config.json")
    diagnostics_summary = load_required_diagnostics_summary(output_dir)
    source_payload = (
        load_json(output_dir / "source.json") if (output_dir / "source.json").exists() else {}
    )
    analysis = _build_analysis(
        output_dir,
        metrics,
        prediction,
        config,
        diagnostics_summary,
        source_payload,
    )
    save_json(output_dir / "analysis.json", analysis)

    markdown = _render_markdown_report(analysis)
    report_output_path = output_dir / "research_report.md"
    report_output_path.write_text(markdown, encoding="utf-8")
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown, encoding="utf-8")
    return analysis


def _build_analysis(
    output_dir: Path,
    metrics: dict[str, object],
    prediction: dict[str, object],
    config: dict[str, object],
    diagnostics_summary: dict[str, object],
    source_payload: dict[str, object],
) -> dict[str, object]:
    validation = dict(metrics.get("validation_metrics", {}))
    generated_at_utc = datetime.now(UTC)
    current_close = float(prediction.get("current_close", 0.0))
    current_close_raw = float(prediction.get("current_close_raw", current_close))
    current_close_display = float(prediction.get("current_close_display", current_close))
    price_scale = float(
        prediction.get("effective_price_scale", prediction.get("price_scale", 1.0)) or 1.0
    )
    policy_horizon = int(prediction.get("policy_horizon", prediction.get("proposed_horizon", 0)))
    expected_log_returns = dict(prediction.get("mu_t", prediction.get("expected_log_returns", {})))
    display_forecast = dict(prediction.get("display_forecast", {}))
    policy_driver = dict(prediction.get("policy_driver", {}))
    predicted_closes = dict(
        prediction.get(
            "median_predicted_close_by_horizon",
            prediction.get("median_predicted_closes", prediction.get("predicted_closes", {})),
        )
    )
    predicted_closes_display = dict(
        prediction.get(
            "median_predicted_close_display_by_horizon",
            prediction.get("median_predicted_closes_display", predicted_closes),
        )
    )
    uncertainties = dict(prediction.get("sigma_t", prediction.get("uncertainties", {})))
    uncertainties_sq = dict(prediction.get("sigma_t_sq", {}))
    shape_posterior = dict(
        prediction.get("shape_posterior", prediction.get("shape_probabilities", {}))
    )
    stateful_evaluation = dict(diagnostics_summary.get("stateful_evaluation", {}))
    blocked_walk_forward_evaluation = dict(
        diagnostics_summary.get("blocked_walk_forward_evaluation", {})
    )
    policy_calibration_sweep = list(diagnostics_summary.get("policy_calibration_sweep", []))
    policy_calibration_summary = _resolve_policy_calibration_summary(
        diagnostics_summary.get("policy_calibration_summary", {}),
        config,
    )
    forecast_quality_scorecards = (
        dict(diagnostics_summary.get("forecast_quality_scorecards", {}))
        if isinstance(diagnostics_summary.get("forecast_quality_scorecards"), dict)
        else {}
    )
    state_vector_summary = dict(diagnostics_summary.get("state_vector_summary", {}))
    current_alias_contract = dict(source_payload.get("current_alias_contract", {}))
    current_selection_governance = dict(source_payload.get("current_selection_governance", {}))
    forecast_quality_ranking_diagnostics = _load_forecast_quality_ranking_diagnostics(
        current_selection_governance
    )
    archive_root = _resolve_archive_root(output_dir, current_selection_governance)
    selection_divergence_scorecard = build_selection_divergence_scorecard(
        archive_root
    )
    selection_history_summary = build_selection_history_summary(
        archive_root,
        session_records=selection_divergence_scorecard.get("rows")
        if isinstance(selection_divergence_scorecard.get("rows"), list)
        else None,
    )
    checkpoint_audit = _resolve_checkpoint_audit(metrics, config)
    rows = []
    anchor_time = datetime.fromisoformat(str(prediction["anchor_time"]))
    for horizon_key in sorted(expected_log_returns, key=lambda value: int(value)):
        horizon = int(horizon_key)
        expected_log_return = float(expected_log_returns[horizon_key])
        predicted_close = float(predicted_closes[horizon_key])
        uncertainty = float(uncertainties[horizon_key])
        rows.append(
            {
                "horizon_4h": horizon,
                "forecast_time_utc": (anchor_time + timedelta(hours=4 * horizon)).isoformat(),
                "mu_t": expected_log_return,
                "expected_log_return": expected_log_return,
                "expected_return_pct": math.exp(expected_log_return) - 1.0,
                "predicted_close": predicted_close,
                "predicted_close_display": float(
                    predicted_closes_display.get(horizon_key, predicted_close)
                ),
                "sigma_t": uncertainty,
                "sigma_t_sq": float(
                    uncertainties_sq.get(horizon_key, uncertainty * uncertainty)
                ),
                "uncertainty": uncertainty,
            }
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc.isoformat(),
        "generated_at_jst": generated_at_utc.astimezone(JST).isoformat(),
        "artifact_dir": str(output_dir),
        "artifact_provenance": _summarize_artifact_provenance(source_payload),
        "artifact_versions": {
            "metrics": int(metrics.get("schema_version", 0)),
            "prediction": int(prediction.get("schema_version", 0)),
            "diagnostics": int(diagnostics_summary.get("schema_version", 0)),
        },
        "policy_mode": metrics.get("policy_mode", "shape_aware_profit_maximization"),
        "dataset": {
            "sample_count": int(metrics.get("sample_count", 0)),
            "effective_sample_count": int(metrics.get("effective_sample_count", 0)),
            "train_samples": int(metrics.get("train_samples", 0)),
            "validation_samples": int(metrics.get("validation_samples", 0)),
            "purged_samples": int(metrics.get("purged_samples", 0)),
            "source_rows_original": int(metrics.get("source_rows_original", 0)),
            "source_rows_used": int(metrics.get("source_rows_used", 0)),
        },
        "training": {
            "best_validation_loss": float(metrics.get("best_validation_loss", 0.0)),
            "best_epoch": float(metrics.get("best_epoch", 0.0)),
            "best_epoch_by_exact_log_wealth": checkpoint_audit.get(
                "best_epoch_by_exact_log_wealth"
            ),
            "best_epoch_by_exact_log_wealth_minus_lambda_cvar": checkpoint_audit.get(
                "best_epoch_by_exact_log_wealth_minus_lambda_cvar"
            ),
            "best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar": checkpoint_audit.get(
                "best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar"
            ),
            "epochs": int(config.get("epochs", 0)),
            "warmup_epochs": int(config.get("warmup_epochs", 0)),
            "oof_epochs": int(config.get("oof_epochs", 0)),
            "shape_classes": int(config.get("shape_classes", 0)),
            "state_dim": int(config.get("state_dim", 0)),
            "walk_forward_folds": int(config.get("walk_forward_folds", 0)),
            "checkpoint_audit": checkpoint_audit,
        },
        "validation_metrics": validation,
        "state_vector_summary": state_vector_summary,
        "stateful_evaluation": stateful_evaluation,
        "blocked_walk_forward_evaluation": blocked_walk_forward_evaluation,
        "policy_calibration_sweep": policy_calibration_sweep,
        "policy_calibration_summary": policy_calibration_summary,
        "forecast_quality_scorecards": forecast_quality_scorecards,
        "forecast_quality_ranking_diagnostics": forecast_quality_ranking_diagnostics,
        "selection_divergence_scorecard": selection_divergence_scorecard,
        "selection_history_summary": selection_history_summary,
        "artifact_contract": current_alias_contract,
        "governance": current_selection_governance,
        "claim_hardening": _build_claim_hardening(state_vector_summary),
        "forecast": {
            "anchor_time_utc": str(prediction["anchor_time"]),
            "anchor_time_jst": anchor_time.astimezone(JST).isoformat(),
            "anchor_close": current_close,
            "anchor_close_raw": current_close_raw,
            "anchor_close_display": current_close_display,
            "price_scale": price_scale,
            "policy_horizon": policy_horizon,
            "executed_horizon": prediction.get("executed_horizon"),
            "q_t_prev": float(prediction.get("q_t_prev", prediction.get("previous_position", 0.0))),
            "previous_position": float(prediction.get("previous_position", 0.0)),
            "q_t": float(prediction.get("q_t", prediction.get("position", 0.0))),
            "position": float(prediction.get("position", 0.0)),
            "q_t_trade_delta": float(prediction.get("q_t_trade_delta", prediction.get("trade_delta", 0.0))),
            "trade_delta": float(prediction.get("trade_delta", 0.0)),
            "no_trade_band_hit": bool(prediction.get("no_trade_band_hit", False)),
            "g_t": float(prediction.get("g_t", prediction.get("tradeability_gate", 0.0))),
            "tradeability_gate": float(prediction.get("tradeability_gate", 0.0)),
            "shape_entropy": float(prediction.get("shape_entropy", 0.0)),
            "selected_policy_utility": float(
                prediction.get("selected_policy_utility", prediction.get("policy_score", 0.0))
            ),
            "policy_score": float(prediction.get("policy_score", 0.0)),
            "mu_t": expected_log_returns,
            "sigma_t": uncertainties,
            "sigma_t_sq": {
                str(key): float(value)
                for key, value in (
                    uncertainties_sq
                    or {horizon: float(sigma) ** 2 for horizon, sigma in uncertainties.items()}
                ).items()
            },
            "shape_posterior": shape_posterior,
            "display_forecast": display_forecast,
            "policy_driver": policy_driver,
            "overlay_branch_contract": prediction.get("overlay_branch_contract"),
            "rows": rows,
        },
        "project_assessment": {
            "summary": _build_summary(validation, prediction),
        },
    }


def _load_forecast_quality_ranking_diagnostics(
    governance: dict[str, object],
) -> dict[str, object]:
    manifest_path = governance.get("selection_session_manifest_path")
    if isinstance(manifest_path, str) and manifest_path.strip():
        resolved_manifest_path = Path(manifest_path).expanduser().resolve()
        if resolved_manifest_path.exists():
            manifest_payload = load_json(resolved_manifest_path)
            diagnostics = manifest_payload.get("forecast_quality_ranking_diagnostics")
            if isinstance(diagnostics, dict):
                return diagnostics

    leaderboard_path = governance.get("selection_leaderboard_path")
    if isinstance(leaderboard_path, str) and leaderboard_path.strip():
        resolved_leaderboard_path = Path(leaderboard_path).expanduser().resolve()
        if resolved_leaderboard_path.exists():
            leaderboard_payload = load_json(resolved_leaderboard_path)
            diagnostics = leaderboard_payload.get("forecast_quality_ranking_diagnostics")
            if isinstance(diagnostics, dict):
                return diagnostics
            leaderboard_rows_payload = leaderboard_payload.get("results")
            if isinstance(leaderboard_rows_payload, list):
                accepted_candidate = governance.get("accepted_candidate")
                production_current = governance.get("production_current")
                return build_forecast_quality_ranking_diagnostics(
                    backfill_forecast_quality_metrics_from_session(
                        leaderboard_rows_payload,
                        resolved_leaderboard_path.parent,
                    ),
                    accepted_candidate=(
                        str(accepted_candidate.get("candidate"))
                        if isinstance(accepted_candidate, dict)
                        and accepted_candidate.get("candidate") is not None
                        else None
                    ),
                    production_current_candidate=(
                        str(production_current.get("candidate"))
                        if isinstance(production_current, dict)
                        and production_current.get("candidate") is not None
                        else None
                    ),
                )
    return {}


def _resolve_archive_root(output_dir: Path, governance: dict[str, object]) -> Path:
    manifest_path = governance.get("selection_session_manifest_path")
    if isinstance(manifest_path, str) and manifest_path.strip():
        resolved_manifest_path = Path(manifest_path).expanduser().resolve()
        if resolved_manifest_path.exists():
            return resolved_manifest_path.parent.parent
    return output_dir.parent / "archive"


def _build_summary(
    validation: dict[str, object],
    prediction: dict[str, object],
) -> str:
    g_t = float(prediction.get("g_t", prediction.get("tradeability_gate", 0.0)))
    selected_policy_utility = float(
        prediction.get("selected_policy_utility", prediction.get("policy_score", 0.0))
    )


def _resolve_policy_calibration_summary(
    policy_calibration_summary: object,
    config: dict[str, object],
) -> dict[str, object]:
    summary = dict(policy_calibration_summary) if isinstance(policy_calibration_summary, dict) else {}
    if "applied_runtime_policy" not in summary:
        applied_runtime_policy = {
            "row_key": (
                "state_reset_mode="
                f"{config.get('evaluation_state_reset_mode', '-')}"
                f"|cost_multiplier={float(config.get('policy_cost_multiplier', 0.0)):.12g}"
                f"|gamma_multiplier={float(config.get('policy_gamma_multiplier', 0.0)):.12g}"
                f"|min_policy_sigma={float(config.get('min_policy_sigma', 0.0)):.12g}"
                f"|q_max={float(config.get('q_max', 0.0)):.12g}"
                f"|cvar_weight={float(config.get('cvar_weight', 0.0)):.12g}"
            ),
            "state_reset_mode": config.get("evaluation_state_reset_mode"),
            "cost_multiplier": config.get("policy_cost_multiplier"),
            "gamma_multiplier": config.get("policy_gamma_multiplier"),
            "min_policy_sigma": config.get("min_policy_sigma"),
            "q_max": config.get("q_max"),
            "cvar_weight": config.get("cvar_weight"),
        }
        selected_row = summary.get("selected_row")
        summary.update(
            {
                "applied_runtime_policy": applied_runtime_policy,
                "applied_runtime_row_key": applied_runtime_policy["row_key"],
                "applied_runtime_policy_role": "authoritative_runtime_config",
                "selected_row_role": summary.get(
                    "selected_row_role",
                    "diagnostic_recommendation_not_applied_runtime_config",
                ),
                "selected_row_matches_applied_runtime": (
                    False
                    if not isinstance(selected_row, dict)
                    else selected_row.get("row_key") == applied_runtime_policy["row_key"]
                ),
            }
        )
    return summary
    return (
        "新 spec の主経路は threshold policy ではなく "
        "`shape -> return distribution -> q_t*` です。"
        f" validation では average_log_wealth={float(validation.get('average_log_wealth', 0.0)):.4f},"
        f" realized_pnl_per_anchor={float(validation.get('realized_pnl_per_anchor', 0.0)):.4f},"
        f" cvar_tail_loss={float(validation.get('cvar_tail_loss', 0.0)):.4f},"
        f" no_trade_band_hit_rate={float(validation.get('no_trade_band_hit_rate', 0.0)):.4f}。"
        f" latest policy_horizon={int(prediction.get('policy_horizon', 0))},"
        f" q_t={float(prediction.get('q_t', prediction.get('position', 0.0))):.4f},"
        f" g_t={g_t:.4f},"
        f" selected_policy_utility={selected_policy_utility:.4f}。"
        " 表示 forecast (`mu_t/sigma_t`) と policy driver "
        "(`policy_mu_t/policy_sigma_t`) は分離して読む必要があります。"
        " 現時点の evidence は continuous posterior weighting までで、"
        " shape-aware routing / regime-aware routing を再主張できる状態ではありません。"
    )


def _build_claim_hardening(state_vector_summary: dict[str, object]) -> dict[str, object]:
    top_class_share = state_vector_summary.get("shape_posterior_top_class_share")
    resolved_top_class_share = (
        dict(top_class_share) if isinstance(top_class_share, dict) else {}
    )
    dominant_class = None
    dominant_share = None
    if resolved_top_class_share:
        dominant_class, dominant_share = max(
            resolved_top_class_share.items(),
            key=lambda item: float(item[1]),
        )
    return {
        "supported_claims": [
            "continuous posterior weighting is present in the current artifact",
            "head coupling can move policy_horizon without restoring shape routing",
        ],
        "unsupported_claims": [
            "shape-aware routing",
            "regime-aware routing",
        ],
        "shape_top_class_collapse": {
            "dominant_class": dominant_class,
            "dominant_share": dominant_share,
        },
        "required_evidence_to_restore_claim": (
            "blocked folds must show materially lower shape top-class concentration and a "
            "paired variant must improve frontier metrics through richer shape usage"
        ),
    }


def _resolve_checkpoint_audit(
    metrics: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    existing = metrics.get("checkpoint_audit")
    if isinstance(existing, dict) and existing:
        return existing

    history_payload = metrics.get("history")
    if not isinstance(history_payload, list):
        return {}

    ranked_rows = [
        row
        for row in history_payload
        if isinstance(row, dict)
        and "epoch" in row
        and "validation_exact_log_wealth" in row
        and "validation_selection_score" in row
    ]
    if not ranked_rows:
        return {}

    cvar_weight = float(config.get("cvar_weight", 0.0) or 0.0)
    selection_rank = sorted(
        ranked_rows,
        key=lambda row: (
            float(row["validation_selection_score"]),
            float(row["epoch"]),
        ),
    )
    wealth_rank = sorted(
        ranked_rows,
        key=lambda row: (
            -float(row["validation_exact_log_wealth"]),
            float(row["epoch"]),
        ),
    )
    selected_row = selection_rank[0]
    best_wealth_row = wealth_rank[0]
    selected_epoch = int(float(selected_row["epoch"]))
    audit: dict[str, object] = {
        "selection_metric": metrics.get(
            "checkpoint_selection_metric",
            "exact_log_wealth_minus_lambda_cvar",
        ),
        "cvar_weight": cvar_weight,
        "selected_epoch": float(selected_epoch),
        "best_epoch_by_selection_score": float(selected_epoch),
        "best_epoch_by_exact_log_wealth": float(best_wealth_row["epoch"]),
        "selected_epoch_rank_by_exact_log_wealth": float(
            next(
                index
                for index, row in enumerate(wealth_rank, start=1)
                if int(float(row["epoch"])) == selected_epoch
            )
        ),
        "selected_epoch_exact_log_wealth": float(selected_row["validation_exact_log_wealth"]),
        "best_exact_log_wealth": float(best_wealth_row["validation_exact_log_wealth"]),
        "delta_to_best_exact_log_wealth": float(best_wealth_row["validation_exact_log_wealth"])
        - float(selected_row["validation_exact_log_wealth"]),
    }

    if all("validation_exact_cvar_tail_loss" in row for row in ranked_rows):
        wealth_minus_cvar_rank = sorted(
            ranked_rows,
            key=lambda row: (
                -(
                    float(row["validation_exact_log_wealth"])
                    - (cvar_weight * float(row["validation_exact_cvar_tail_loss"]))
                ),
                float(row["epoch"]),
            ),
        )
        best_wealth_minus_cvar_row = wealth_minus_cvar_rank[0]
        selected_wealth_minus_cvar = float(selected_row["validation_exact_log_wealth"]) - (
            cvar_weight * float(selected_row["validation_exact_cvar_tail_loss"])
        )
        best_wealth_minus_cvar = float(best_wealth_minus_cvar_row["validation_exact_log_wealth"]) - (
            cvar_weight * float(best_wealth_minus_cvar_row["validation_exact_cvar_tail_loss"])
        )
        audit.update(
            {
                "best_epoch_by_exact_log_wealth_minus_lambda_cvar": float(
                    best_wealth_minus_cvar_row["epoch"]
                ),
                "selected_epoch_rank_by_exact_log_wealth_minus_lambda_cvar": float(
                    next(
                        index
                        for index, row in enumerate(wealth_minus_cvar_rank, start=1)
                        if int(float(row["epoch"])) == selected_epoch
                    )
                ),
                "selected_epoch_exact_log_wealth_minus_lambda_cvar": selected_wealth_minus_cvar,
                "best_exact_log_wealth_minus_lambda_cvar": best_wealth_minus_cvar,
                "delta_to_best_exact_log_wealth_minus_lambda_cvar": (
                    best_wealth_minus_cvar - selected_wealth_minus_cvar
                ),
            }
        )

    if all(
        "validation_blocked_objective_log_wealth_minus_lambda_cvar_mean" in row
        for row in ranked_rows
    ):
        blocked_rank = sorted(
            ranked_rows,
            key=lambda row: (
                -float(row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"]),
                float(row["epoch"]),
            ),
        )
        best_blocked_row = blocked_rank[0]
        audit.update(
            {
                "best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    best_blocked_row["epoch"]
                ),
                "selected_epoch_rank_by_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    next(
                        index
                        for index, row in enumerate(blocked_rank, start=1)
                        if int(float(row["epoch"])) == selected_epoch
                    )
                ),
                "selected_epoch_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    selected_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"]
                ),
                "best_blocked_objective_log_wealth_minus_lambda_cvar": float(
                    best_blocked_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"]
                ),
                "delta_to_best_blocked_objective_log_wealth_minus_lambda_cvar": (
                    float(best_blocked_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"])
                    - float(selected_row["validation_blocked_objective_log_wealth_minus_lambda_cvar_mean"])
                ),
            }
        )

    return audit


def _render_markdown_report(analysis: dict[str, object]) -> str:
    dataset = analysis["dataset"]
    training = analysis["training"]
    validation = analysis["validation_metrics"]
    state_vector_summary = analysis.get("state_vector_summary", {})
    artifact_provenance = analysis.get("artifact_provenance", {})
    artifact_contract = analysis.get("artifact_contract", {})
    governance = analysis.get("governance", {})
    claim_hardening = analysis.get("claim_hardening", {})
    stateful_evaluation = analysis["stateful_evaluation"]
    blocked_walk_forward_evaluation = analysis.get("blocked_walk_forward_evaluation", {})
    policy_calibration_sweep = analysis["policy_calibration_sweep"]
    policy_calibration_summary = analysis["policy_calibration_summary"]
    forecast_quality_scorecards = analysis.get("forecast_quality_scorecards", {})
    forecast_quality_ranking_diagnostics = analysis.get(
        "forecast_quality_ranking_diagnostics",
        {},
    )
    selection_divergence_scorecard = analysis.get("selection_divergence_scorecard", {})
    selection_history_summary = analysis.get("selection_history_summary", {})
    forecast = analysis["forecast"]
    rows = forecast["rows"]
    blocked_modes = (
        dict(blocked_walk_forward_evaluation.get("state_reset_modes", {}))
        if isinstance(blocked_walk_forward_evaluation, dict)
        else {}
    )
    blocked_carry_on = dict(blocked_modes.get("carry_on", {}))
    blocked_reset_session = dict(blocked_modes.get("reset_each_session_or_window", {}))
    selected_sweep_row = (
        dict(policy_calibration_summary.get("selected_row", {}))
        if policy_calibration_summary.get("selected_row") is not None
        else None
    )
    selected_horizon_scorecard = (
        dict(forecast_quality_scorecards.get("selected_horizon", {}))
        if isinstance(forecast_quality_scorecards.get("selected_horizon"), dict)
        else {}
    )
    all_horizon_scorecard = (
        dict(forecast_quality_scorecards.get("all_horizon", {}))
        if isinstance(forecast_quality_scorecards.get("all_horizon"), dict)
        else {}
    )
    applied_runtime_policy = (
        dict(policy_calibration_summary.get("applied_runtime_policy", {}))
        if policy_calibration_summary.get("applied_runtime_policy") is not None
        else {}
    )
    authoritative_paths = (
        dict(artifact_contract.get("authoritative_paths", {}))
        if isinstance(artifact_contract.get("authoritative_paths"), dict)
        else {}
    )
    runtime_contract = (
        dict(artifact_contract.get("runtime_contract", {}))
        if isinstance(artifact_contract.get("runtime_contract"), dict)
        else {}
    )
    governance_best = (
        dict(governance.get("best_candidate", {}))
        if isinstance(governance.get("best_candidate"), dict)
        else {}
    )
    governance_current = (
        dict(governance.get("production_current", {}))
        if isinstance(governance.get("production_current"), dict)
        else {}
    )
    governance_accepted = (
        dict(governance.get("accepted_candidate", {}))
        if isinstance(governance.get("accepted_candidate"), dict)
        else {}
    )
    paired_frontier = (
        dict(governance.get("paired_frontier", {}))
        if isinstance(governance.get("paired_frontier"), dict)
        else {}
    )
    frontier_delta = (
        dict(paired_frontier.get("delta_production_minus_accepted", {}))
        if isinstance(paired_frontier.get("delta_production_minus_accepted"), dict)
        else {}
    )
    supported_claims = claim_hardening.get("supported_claims", [])
    unsupported_claims = claim_hardening.get("unsupported_claims", [])
    shape_collapse = (
        dict(claim_hardening.get("shape_top_class_collapse", {}))
        if isinstance(claim_hardening.get("shape_top_class_collapse"), dict)
        else {}
    )
    artifact_lines = [
        "## Artifact",
        f"- kind: `{artifact_provenance.get('artifact_kind', 'legacy')}`",
        f"- artifact id: `{artifact_provenance.get('artifact_id', '-')}`",
        f"- source kind / path: `{artifact_provenance.get('source_kind', '-')}` / `{artifact_provenance.get('source_path', '-')}`",
        f"- config origin: `{artifact_provenance.get('config_origin', '-')}`",
    ]
    if artifact_provenance.get("parent_artifact_dir") is not None:
        artifact_lines.append(
            f"- parent artifact dir: `{artifact_provenance['parent_artifact_dir']}`"
        )
    if artifact_provenance.get("parent_artifact_id") is not None:
        artifact_lines.append(
            f"- parent artifact id: `{artifact_provenance['parent_artifact_id']}`"
        )
    if artifact_provenance.get("git_head") is not None:
        artifact_lines.append(
            f"- git head / dirty: `{artifact_provenance['git_head']}` / `{artifact_provenance.get('git_dirty', False)}`"
        )
    if artifact_provenance.get("git_tree_sha") is not None:
        artifact_lines.append(
            f"- git tree sha: `{artifact_provenance['git_tree_sha']}`"
        )
    if artifact_provenance.get("data_snapshot_sha256") is not None:
        artifact_lines.append(
            f"- data snapshot sha256: `{artifact_provenance['data_snapshot_sha256'][:16]}`"
        )
    if artifact_provenance.get("sub_artifact_materialization") is not None:
        artifact_lines.append(
            f"- sub-artifact lineage: `{artifact_provenance['sub_artifact_materialization']}`"
        )
    row_lines = "\n".join(
        f"- h={int(row['horizon_4h'])}: mu_t={float(row['mu_t']):.4f}, "
        f"expected_return_pct={float(row['expected_return_pct']):.4f}, "
        f"predicted_close_display={float(row['predicted_close_display']):.4f}, "
        f"sigma_t={float(row['sigma_t']):.4f}"
        for row in rows
    )
    return "\n".join(
        [
            "# SignalCascade Research Report",
            "",
            f"- Generated (JST): `{analysis['generated_at_jst']}`",
            f"- Policy mode: `{analysis['policy_mode']}`",
            "",
            *artifact_lines,
            "",
            "## Dataset",
            f"- sample_count: `{dataset['sample_count']}`",
            f"- effective_sample_count / purged_samples: `{dataset['effective_sample_count']}` / `{dataset['purged_samples']}`",
            f"- train / validation: `{dataset['train_samples']}` / `{dataset['validation_samples']}`",
            f"- source_rows_original / used: `{dataset['source_rows_original']}` / `{dataset['source_rows_used']}`",
            "",
            "## Contract",
            f"- current alias role: `{artifact_contract.get('alias_role', '-')}`",
            f"- authoritative runtime config: `{authoritative_paths.get('runtime_config', '-')}`",
            f"- diagnostic recommendation pointer: "
            f"`{runtime_contract.get('diagnostic_recommendation_pointer', '-')}`",
            f"- top-level report role / path: "
            f"`{artifact_contract.get('top_level_report_role', '-')}` / "
            f"`{authoritative_paths.get('top_level_report', '-')}`",
            f"- current research report / prediction / forecast / source: "
            f"`{authoritative_paths.get('research_report', '-')}` / "
            f"`{authoritative_paths.get('prediction', '-')}` / "
            f"`{authoritative_paths.get('forecast_summary', '-')}` / "
            f"`{authoritative_paths.get('source', '-')}`",
            f"- source-of-truth summary: `{artifact_contract.get('source_of_truth_summary', '-')}`",
            "",
            "## Training",
            f"- best_validation_loss: `{float(training['best_validation_loss']):.6f}`",
            f"- best_epoch: `{float(training['best_epoch']):.0f}`",
            f"- best_epoch_by_exact_log_wealth: "
            f"`{_display_int_or_missing(training.get('best_epoch_by_exact_log_wealth'))}`",
            f"- best_epoch_by_exact_log_wealth_minus_lambda_cvar: "
            f"`{_display_int_or_missing(training.get('best_epoch_by_exact_log_wealth_minus_lambda_cvar'))}`",
            f"- best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar: "
            f"`{_display_int_or_missing(training.get('best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar'))}`",
            (
                f"- selected_epoch_rank_by_exact_log_wealth / delta_to_best_exact_log_wealth: "
                f"`{_display_int_or_missing(training.get('checkpoint_audit', {}).get('selected_epoch_rank_by_exact_log_wealth'))}` / "
                f"`{_display_or_missing(training.get('checkpoint_audit', {}).get('delta_to_best_exact_log_wealth'))}`"
            ),
            (
                f"- selected_epoch_rank_by_exact_log_wealth_minus_lambda_cvar / "
                f"delta_to_best_exact_log_wealth_minus_lambda_cvar: "
                f"`{_display_int_or_missing(training.get('checkpoint_audit', {}).get('selected_epoch_rank_by_exact_log_wealth_minus_lambda_cvar'))}` / "
                f"`{_display_or_missing(training.get('checkpoint_audit', {}).get('delta_to_best_exact_log_wealth_minus_lambda_cvar'))}`"
            ),
            (
                f"- selected_epoch_rank_by_blocked_objective_log_wealth_minus_lambda_cvar / "
                f"delta_to_best_blocked_objective_log_wealth_minus_lambda_cvar: "
                f"`{_display_int_or_missing(training.get('checkpoint_audit', {}).get('selected_epoch_rank_by_blocked_objective_log_wealth_minus_lambda_cvar'))}` / "
                f"`{_display_or_missing(training.get('checkpoint_audit', {}).get('delta_to_best_blocked_objective_log_wealth_minus_lambda_cvar'))}`"
            ),
            f"- epochs / warmup_epochs / oof_epochs: `{training['epochs']}` / `{training['warmup_epochs']}` / `{training.get('oof_epochs', 0)}`",
            f"- walk_forward_folds: `{training.get('walk_forward_folds', 0)}`",
            f"- shape_classes / state_dim: `{training['shape_classes']}` / `{training['state_dim']}`",
            "",
            "## Validation",
            f"- average_log_wealth: `{_display_or_missing(validation.get('average_log_wealth'))}`",
            f"- realized_pnl_per_anchor: `{_display_or_missing(validation.get('realized_pnl_per_anchor'))}`",
            f"- cvar_tail_loss: `{_display_or_missing(validation.get('cvar_tail_loss'))}`",
            f"- max_drawdown: `{_display_or_missing(validation.get('max_drawdown'))}`",
            f"- no_trade_band_hit_rate: `{_display_or_missing(validation.get('no_trade_band_hit_rate'))}`",
            f"- exact_smooth_horizon_agreement / no_trade_agreement: "
            f"`{_display_or_missing(validation.get('exact_smooth_horizon_agreement'))}` / "
            f"`{_display_or_missing(validation.get('exact_smooth_no_trade_agreement'))}`",
            f"- exact_smooth_position_mae / utility_regret: "
            f"`{_display_or_missing(validation.get('exact_smooth_position_mae'))}` / "
            f"`{_display_or_missing(validation.get('exact_smooth_utility_regret'))}`",
            f"- shape_gate_usage: `{_display_or_missing(validation.get('shape_gate_usage'))}`",
            f"- expert_entropy: `{_display_or_missing(validation.get('expert_entropy'))}`",
            f"- shape_posterior_top_class_share: "
            f"`{state_vector_summary.get('shape_posterior_top_class_share', {})}`",
            f"- mu_calibration / sigma_calibration: "
            f"`{_display_or_missing(validation.get('mu_calibration'))}` / "
            f"`{_display_or_missing(validation.get('sigma_calibration'))}`",
            f"- log_wealth_clamp_hit_rate / state_reset_mode: "
            f"`{_display_or_missing(validation.get('log_wealth_clamp_hit_rate'))}` / "
            f"`{validation.get('state_reset_mode', '-')}`",
            f"- project_value_score / utility_score: "
            f"`{_display_or_missing(validation.get('project_value_score'))}` / "
            f"`{_display_or_missing(validation.get('utility_score'))}`",
            "",
            "## Evaluation",
            f"- carry_on average_log_wealth: `{_display_or_missing(stateful_evaluation.get('carry_on', {}).get('average_log_wealth'))}`",
            f"- reset_each_example average_log_wealth: `{_display_or_missing(stateful_evaluation.get('reset_each_example', {}).get('average_log_wealth'))}`",
            f"- reset_each_session_or_window average_log_wealth: "
            f"`{_display_or_missing(stateful_evaluation.get('reset_each_session_or_window', {}).get('average_log_wealth'))}`",
            f"- blocked_walk_forward_folds / best_state_reset_mode_by_mean_log_wealth: "
            f"`{_display_int_or_missing(blocked_walk_forward_evaluation.get('fold_count'))}` / "
            f"`{blocked_walk_forward_evaluation.get('best_state_reset_mode_by_mean_log_wealth', '-')}`",
            f"- blocked carry_on mean/min/max average_log_wealth: "
            f"`{_display_or_missing(blocked_carry_on.get('average_log_wealth_mean'))}` / "
            f"`{_display_or_missing(blocked_carry_on.get('average_log_wealth_min'))}` / "
            f"`{_display_or_missing(blocked_carry_on.get('average_log_wealth_max'))}`",
            f"- blocked reset_each_session_or_window mean average_log_wealth / turnover_mean: "
            f"`{_display_or_missing(blocked_reset_session.get('average_log_wealth_mean'))}` / "
            f"`{_display_or_missing(blocked_reset_session.get('turnover_mean'))}`",
            f"- policy sweep rows / pareto_optimal: "
            f"`{_display_int_or_missing(policy_calibration_summary.get('row_count', len(policy_calibration_sweep) if policy_calibration_sweep else None))}` / "
            f"`{_display_int_or_missing(policy_calibration_summary.get('pareto_optimal_count'))}`",
            f"- policy sweep selection basis / version: "
            f"`{policy_calibration_summary.get('selection_basis', '-')}` / "
            f"`{policy_calibration_summary.get('selection_rule_version', '-')}`",
            (
                f"- selected policy sweep: reset=`{selected_sweep_row['state_reset_mode']}`, "
                f"cost x`{float(selected_sweep_row['cost_multiplier']):.2f}`, "
                f"gamma x`{float(selected_sweep_row['gamma_multiplier']):.2f}`, "
                f"min_sigma=`{float(selected_sweep_row['min_policy_sigma']):.6f}`, "
                f"q_max=`{_display_or_missing(selected_sweep_row.get('q_max'), digits=4)}`, "
                f"cvar_weight=`{_display_or_missing(selected_sweep_row.get('cvar_weight'), digits=4)}`, "
                f"blocked_objective_mean="
                f"`{_display_or_missing(selected_sweep_row.get('blocked_objective_log_wealth_minus_lambda_cvar_mean'))}`"
                if selected_sweep_row is not None
                else "- selected policy sweep: none"
            ),
            (
                f"- selected policy sweep blocked mean wealth / cvar / turnover: "
                f"`{_display_or_missing(selected_sweep_row.get('blocked_average_log_wealth_mean'))}` / "
                f"`{_display_or_missing(selected_sweep_row.get('blocked_cvar_tail_loss_mean'))}` / "
                f"`{_display_or_missing(selected_sweep_row.get('blocked_turnover_mean'))}`"
                if selected_sweep_row is not None
                else "- selected policy sweep blocked mean wealth / cvar / turnover: `-`"
            ),
            (
                f"- applied runtime policy: reset=`{applied_runtime_policy.get('state_reset_mode', '-')}`, "
                f"cost x`{_display_or_missing(applied_runtime_policy.get('cost_multiplier'), digits=4)}`, "
                f"gamma x`{_display_or_missing(applied_runtime_policy.get('gamma_multiplier'), digits=4)}`, "
                f"min_sigma=`{_display_or_missing(applied_runtime_policy.get('min_policy_sigma'), digits=6)}`, "
                f"q_max=`{_display_or_missing(applied_runtime_policy.get('q_max'), digits=4)}`, "
                f"cvar_weight=`{_display_or_missing(applied_runtime_policy.get('cvar_weight'), digits=4)}`"
                if applied_runtime_policy
                else "- applied runtime policy: `-`"
            ),
            f"- selected_row_matches_applied_runtime: "
            f"`{policy_calibration_summary.get('selected_row_matches_applied_runtime', '-')}`",
            f"- selected row key: `{policy_calibration_summary.get('selected_row_key', '-')}`",
            (
                f"- policy sweep rows sha256: "
                f"`{str(policy_calibration_summary.get('policy_calibration_rows_sha256'))[:16]}`"
                if policy_calibration_summary.get("policy_calibration_rows_sha256") is not None
                else "- policy sweep rows sha256: `-`"
            ),
            f"- forecast quality score (selected_horizon / all_horizon / gap): "
            f"`{_display_or_missing(selected_horizon_scorecard.get('quality_score'))}` / "
            f"`{_display_or_missing(all_horizon_scorecard.get('quality_score'))}` / "
            f"`{_display_or_missing(forecast_quality_scorecards.get('quality_score_gap_all_minus_selected'))}`",
            f"- forecast quality directional_accuracy (selected_horizon / all_horizon): "
            f"`{_display_or_missing(selected_horizon_scorecard.get('directional_accuracy'))}` / "
            f"`{_display_or_missing(all_horizon_scorecard.get('directional_accuracy'))}`",
            f"- forecast quality mu_calibration / probabilistic_score (selected_horizon / all_horizon): "
            f"`{_display_or_missing(selected_horizon_scorecard.get('mu_calibration'))}` / "
            f"`{_display_or_missing(selected_horizon_scorecard.get('probabilistic_calibration_score'))}` / "
            f"`{_display_or_missing(all_horizon_scorecard.get('mu_calibration'))}` / "
            f"`{_display_or_missing(all_horizon_scorecard.get('probabilistic_calibration_score'))}`",
            "",
            "## Forecast",
            f"- anchor_time_utc / jst: `{forecast['anchor_time_utc']}` / `{forecast['anchor_time_jst']}`",
            f"- anchor_close_display / raw / price_scale: `{float(forecast['anchor_close_display']):.4f}` / `{float(forecast['anchor_close_raw']):.4f}` / `{float(forecast['price_scale']):.4f}`",
            f"- policy_horizon / executed_horizon: `{forecast['policy_horizon']}` / `{forecast['executed_horizon']}`",
            f"- previous_position / position / trade_delta: "
            f"`{float(forecast['previous_position']):.4f}` / `{float(forecast['position']):.4f}` / `{float(forecast['trade_delta']):.4f}`",
            f"- no_trade_band_hit: `{forecast['no_trade_band_hit']}`",
            f"- g_t / shape_entropy / selected_policy_utility: "
            f"`{float(forecast['g_t']):.4f}` / `{float(forecast['shape_entropy']):.4f}` / `{float(forecast['selected_policy_utility']):.4f}`",
            (
                f"- display forecast label / policy driver label / head relationship: "
                f"`{forecast.get('display_forecast', {}).get('label', '-')}` / "
                f"`{forecast.get('policy_driver', {}).get('label', '-')}` / "
                f"`{forecast.get('policy_driver', {}).get('head_relationship', '-')}`"
            ),
            f"- overlay branch contract: `{forecast.get('overlay_branch_contract', '-')}`",
            row_lines or "- forecast rows: none",
            "",
            "## Governance",
            f"- selection mode / rule / version: "
            f"`{governance.get('selection_mode', '-')}` / "
            f"`{governance.get('selection_rule', '-')}` / "
            f"`{governance.get('selection_rule_version', '-')}`",
            f"- decision summary: `{governance.get('decision_summary', '-')}`",
            f"- best / accepted / production current: "
            f"`{governance_best.get('candidate', '-')}` / "
            f"`{governance_accepted.get('candidate', '-')}` / "
            f"`{governance_current.get('candidate', '-')}`",
            f"- production current user_value_score / chart_fidelity / sigma_band / execution_stability: "
            f"`{_display_or_missing(governance_current.get('user_value_score'))}` / "
            f"`{_display_or_missing(governance_current.get('user_value_chart_fidelity_score'))}` / "
            f"`{_display_or_missing(governance_current.get('user_value_sigma_band_score'))}` / "
            f"`{_display_or_missing(governance_current.get('user_value_execution_stability_score'))}`",
            f"- production current selected_horizon / all_horizon forecast quality: "
            f"`{_display_or_missing(governance_current.get('selected_horizon_forecast_quality_score'))}` / "
            f"`{_display_or_missing(governance_current.get('all_horizon_forecast_quality_score'))}`",
            f"- ranking split top candidate (current / selected / all): "
            f"`{forecast_quality_ranking_diagnostics.get('current_top_candidate', '-')}` / "
            f"`{forecast_quality_ranking_diagnostics.get('selected_horizon_top_candidate', '-')}` / "
            f"`{forecast_quality_ranking_diagnostics.get('all_horizon_top_candidate', '-')}`",
            f"- ranking split Spearman (selected/current / all/current / selected/all): "
            f"`{_display_or_missing(forecast_quality_ranking_diagnostics.get('selected_horizon_vs_current_spearman_rank_correlation'))}` / "
            f"`{_display_or_missing(forecast_quality_ranking_diagnostics.get('all_horizon_vs_current_spearman_rank_correlation'))}` / "
            f"`{_display_or_missing(forecast_quality_ranking_diagnostics.get('selected_horizon_vs_all_horizon_spearman_rank_correlation'))}`",
            f"- accepted candidate rank (current / selected / all): "
            f"`{_display_int_or_missing(forecast_quality_ranking_diagnostics.get('accepted_candidate_current_rank'))}` / "
            f"`{_display_int_or_missing(forecast_quality_ranking_diagnostics.get('accepted_candidate_selected_horizon_rank'))}` / "
            f"`{_display_int_or_missing(forecast_quality_ranking_diagnostics.get('accepted_candidate_all_horizon_rank'))}`",
            f"- top-{forecast_quality_ranking_diagnostics.get('top_k', 0)} overlap with current "
            f"(selected / all): "
            f"`{_display_int_or_missing(forecast_quality_ranking_diagnostics.get('selected_horizon_top_k_overlap_with_current_count'))}` / "
            f"`{_display_int_or_missing(forecast_quality_ranking_diagnostics.get('all_horizon_top_k_overlap_with_current_count'))}`",
            f"- history sessions / accepted / production / diverged: "
            f"`{_display_int_or_missing(selection_history_summary.get('session_count'))}` / "
            f"`{_display_int_or_missing(selection_history_summary.get('accepted_candidate_count'))}` / "
            f"`{_display_int_or_missing(selection_history_summary.get('production_current_candidate_count'))}` / "
            f"`{_display_int_or_missing(selection_history_summary.get('accepted_vs_production_divergence_count'))}`",
            f"- history accepted top-match rate (current / selected / all): "
            f"`{_display_or_missing(selection_history_summary.get('accepted_current_top_match_ratio'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('accepted_selected_horizon_top_match_ratio'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('accepted_all_horizon_top_match_ratio'))}`",
            f"- history accepted median rank (current / selected / all): "
            f"`{_display_or_missing(selection_history_summary.get('accepted_candidate_current_rank_median'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('accepted_candidate_selected_horizon_rank_median'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('accepted_candidate_all_horizon_rank_median'))}`",
            f"- history production top-match rate (current / selected / all): "
            f"`{_display_or_missing(selection_history_summary.get('production_current_top_match_ratio'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('production_selected_horizon_top_match_ratio'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('production_all_horizon_top_match_ratio'))}`",
            f"- history production median rank (current / selected / all): "
            f"`{_display_or_missing(selection_history_summary.get('production_current_current_rank_median'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('production_current_selected_horizon_rank_median'))}` / "
            f"`{_display_or_missing(selection_history_summary.get('production_current_all_horizon_rank_median'))}`",
            f"- divergence scorecard coverage (full / partial): "
            f"`{_display_int_or_missing(selection_divergence_scorecard.get('full_coverage_session_count'))}` / "
            f"`{_display_int_or_missing(selection_divergence_scorecard.get('partial_coverage_session_count'))}`",
            f"- divergence scorecard clusters (all sessions): "
            f"`{_display_divergence_cluster_counts(selection_divergence_scorecard.get('cluster_counts'))}`",
            f"- divergence scorecard clusters (full coverage): "
            f"`{_display_divergence_cluster_counts(selection_divergence_scorecard.get('full_coverage_cluster_counts'))}`",
            (
                "- recent history snapshots: "
                + " | ".join(
                    (
                        f"{session.get('session_id', '-')}:"
                        f"acc={session.get('accepted_candidate', '-')},"
                        f"prod={session.get('production_current_candidate', '-')},"
                        f"ranks={_display_int_or_missing(session.get('accepted_candidate_current_rank'))}/"
                        f"{_display_int_or_missing(session.get('accepted_candidate_selected_horizon_rank'))}/"
                        f"{_display_int_or_missing(session.get('accepted_candidate_all_horizon_rank'))}"
                    )
                    for session in selection_history_summary.get("recent_sessions", [])
                    if isinstance(session, dict)
                )
                if selection_history_summary.get("recent_sessions")
                else "- recent history snapshots: `-`"
            ),
            (
                "- divergence scorecard recent rows: "
                + " | ".join(
                    _render_divergence_row(session)
                    for session in selection_divergence_scorecard.get("recent_rows", [])
                    if isinstance(session, dict)
                )
                if selection_divergence_scorecard.get("recent_rows")
                else "- divergence scorecard recent rows: `-`"
            ),
            (
                f"- accepted snapshot user_value_score / chart_fidelity / sigma_band / execution_stability: "
                f"`{_display_or_missing(governance_accepted.get('user_value_score'))}` / "
                f"`{_display_or_missing(governance_accepted.get('user_value_chart_fidelity_score'))}` / "
                f"`{_display_or_missing(governance_accepted.get('user_value_sigma_band_score'))}` / "
                f"`{_display_or_missing(governance_accepted.get('user_value_execution_stability_score'))}`"
                if governance_accepted
                else "- accepted snapshot user_value_score / chart_fidelity / sigma_band / execution_stability: `-`"
            ),
            (
                f"- accepted snapshot selected_horizon / all_horizon forecast quality: "
                f"`{_display_or_missing(governance_accepted.get('selected_horizon_forecast_quality_score'))}` / "
                f"`{_display_or_missing(governance_accepted.get('all_horizon_forecast_quality_score'))}`"
                if governance_accepted
                else "- accepted snapshot selected_horizon / all_horizon forecast quality: `-`"
            ),
            f"- production current blocked objective / blocked turnover / max_drawdown / "
            f"exact_smooth_position_mae / trade_delta / policy_horizon: "
            f"`{_display_or_missing(governance_current.get('blocked_objective_log_wealth_minus_lambda_cvar_mean'))}` / "
            f"`{_display_or_missing(governance_current.get('blocked_turnover_mean'))}` / "
            f"`{_display_or_missing(governance_current.get('max_drawdown'))}` / "
            f"`{_display_or_missing(governance_current.get('exact_smooth_position_mae'))}` / "
            f"`{_display_or_missing(governance_current.get('trade_delta'))}` / "
            f"`{governance_current.get('policy_horizon', '-')}`",
            (
                f"- accepted snapshot blocked objective / blocked turnover / max_drawdown / "
                f"exact_smooth_position_mae / trade_delta / policy_horizon: "
                f"`{_display_or_missing(governance_accepted.get('blocked_objective_log_wealth_minus_lambda_cvar_mean'))}` / "
                f"`{_display_or_missing(governance_accepted.get('blocked_turnover_mean'))}` / "
                f"`{_display_or_missing(governance_accepted.get('max_drawdown'))}` / "
                f"`{_display_or_missing(governance_accepted.get('exact_smooth_position_mae'))}` / "
                f"`{_display_or_missing(governance_accepted.get('trade_delta'))}` / "
                f"`{governance_accepted.get('policy_horizon', '-')}`"
                if governance_accepted
                else "- accepted snapshot blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `-`"
            ),
            (
                f"- production minus accepted delta (avg_log_wealth / blocked_objective / "
                f"blocked_turnover / max_drawdown / exact_smooth_position_mae / "
                f"selected_horizon_quality / all_horizon_quality / trade_delta / policy_horizon): "
                f"`{_display_or_missing(frontier_delta.get('average_log_wealth'))}` / "
                f"`{_display_or_missing(frontier_delta.get('blocked_objective_log_wealth_minus_lambda_cvar_mean'))}` / "
                f"`{_display_or_missing(frontier_delta.get('blocked_turnover_mean'))}` / "
                f"`{_display_or_missing(frontier_delta.get('max_drawdown'))}` / "
                f"`{_display_or_missing(frontier_delta.get('exact_smooth_position_mae'))}` / "
                f"`{_display_or_missing(frontier_delta.get('selected_horizon_forecast_quality_score'))}` / "
                f"`{_display_or_missing(frontier_delta.get('all_horizon_forecast_quality_score'))}` / "
                f"`{_display_or_missing(frontier_delta.get('trade_delta'))}` / "
                f"`{_display_or_missing(frontier_delta.get('policy_horizon'))}`"
                if frontier_delta
                else "- production minus accepted delta (avg_log_wealth / blocked_objective / blocked_turnover / max_drawdown / exact_smooth_position_mae / selected_horizon_quality / all_horizon_quality / trade_delta / policy_horizon): `-`"
            ),
            f"- override priority metrics: `{governance.get('override_priority_metrics', [])}`",
            "",
            "## Claim Hardening",
            f"- supported claims: `{supported_claims}`",
            f"- unsupported claims: `{unsupported_claims}`",
            f"- dominant shape class / share: "
            f"`{shape_collapse.get('dominant_class', '-')}` / "
            f"`{_display_or_missing(shape_collapse.get('dominant_share'))}`",
            f"- restore-claim evidence gate: "
            f"`{claim_hardening.get('required_evidence_to_restore_claim', '-')}`",
            "",
            "## Assessment",
            f"- {analysis['project_assessment']['summary']}",
        ]
    )


def _summarize_artifact_provenance(source_payload: dict[str, object]) -> dict[str, object]:
    git_payload = source_payload.get("git")
    git = git_payload if isinstance(git_payload, dict) else {}
    sub_artifacts_payload = source_payload.get("sub_artifacts")
    sub_artifacts = sub_artifacts_payload if isinstance(sub_artifacts_payload, dict) else {}
    materialization = ", ".join(
        f"{name}:{details.get('materialization', '-')}"
        for name, details in sorted(sub_artifacts.items())
        if isinstance(details, dict)
    )
    return {
        "artifact_schema_version": source_payload.get("artifact_schema_version"),
        "artifact_kind": source_payload.get("artifact_kind"),
        "artifact_id": source_payload.get("artifact_id"),
        "artifact_dir": source_payload.get("artifact_dir"),
        "parent_artifact_dir": source_payload.get("parent_artifact_dir"),
        "parent_artifact_id": source_payload.get("parent_artifact_id"),
        "source_kind": source_payload.get("kind"),
        "source_path": source_payload.get("path"),
        "data_snapshot_sha256": source_payload.get("data_snapshot_sha256"),
        "config_sha256": source_payload.get("config_sha256"),
        "config_origin": source_payload.get("config_origin"),
        "state_reset_boundary_spec_version": source_payload.get(
            "state_reset_boundary_spec_version"
        ),
        "git_head": git.get("head"),
        "git_head": git.get("head", git.get("git_commit_sha", git.get("commit_sha"))),
        "git_commit_sha": git.get("git_commit_sha", git.get("commit_sha", git.get("head"))),
        "git_tree_sha": git.get("git_tree_sha", git.get("tree_sha")),
        "git_dirty": git.get("git_dirty", git.get("dirty")),
        "git_dirty_patch_sha256": git.get("dirty_patch_sha256"),
        "sub_artifact_materialization": materialization or None,
    }
