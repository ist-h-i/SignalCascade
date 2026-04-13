from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .diagnostics_service import _build_forecast_quality_scorecards
from ..infrastructure.persistence import load_json

_LEGACY_CURRENT_SELECTION_RULE = "optimization_gate_then_blocked_objective"
_LEGACY_CURRENT_SELECTION_RULE_VERSION = 1
_LEGACY_OVERRIDE_PRIORITY_METRICS = (
    "policy_horizon",
    "blocked_turnover_mean",
    "max_drawdown",
    "exact_smooth_position_mae",
    "trade_delta",
)


def build_current_alias_metadata(
    artifact_root: Path,
    source_artifact_dir: Path,
    *,
    selection_timestamp_utc: str | None = None,
) -> dict[str, object]:
    resolved_artifact_root = artifact_root.expanduser().resolve()
    resolved_source_dir = source_artifact_dir.expanduser().resolve()
    current_dir = resolved_artifact_root / "current"
    top_level_report_path = resolved_artifact_root.parent.parent / "report_signalcascade_xauusd.md"
    selected_row_pointer = "validation_summary.json.policy_calibration_summary.selected_row"

    session_context = _load_session_context(resolved_source_dir)
    production_current = _build_candidate_snapshot(
        resolved_source_dir,
        leaderboard_row=session_context["leaderboard_rows"].get(resolved_source_dir.name),
    )
    accepted_candidate = _resolve_session_candidate_snapshot(
        session_context,
        "accepted_candidate",
    )
    best_candidate = _resolve_session_candidate_snapshot(
        session_context,
        "best_candidate",
    )

    override_applied = bool(
        accepted_candidate
        and production_current.get("artifact_id") != accepted_candidate.get("artifact_id")
    )
    selection_time = selection_timestamp_utc or datetime.now(timezone.utc).isoformat()
    paired_frontier = _build_paired_frontier(
        accepted_candidate=accepted_candidate,
        production_current=production_current,
    )
    decision_summary = _build_decision_summary(
        accepted_candidate=accepted_candidate,
        production_current=production_current,
        override_applied=override_applied,
    )
    selection_metadata = _resolve_production_selection_metadata(
        session_context=session_context,
        production_current=production_current,
        accepted_candidate=accepted_candidate,
        override_applied=override_applied,
        fallback_decision_summary=decision_summary,
    )

    current_alias_contract: dict[str, object] = {
        "alias_role": "production_current",
        "current_alias_dir": str(current_dir),
        "top_level_report_role": "mirror_of_current_research_report",
        "authoritative_paths": {
            "runtime_config": str(current_dir / "config.json"),
            "prediction": str(current_dir / "prediction.json"),
            "forecast_summary": str(current_dir / "forecast_summary.json"),
            "source": str(current_dir / "source.json"),
            "research_report": str(current_dir / "research_report.md"),
            "manifest": str(current_dir / "manifest.json"),
            "top_level_report": str(top_level_report_path),
            "accepted_snapshot_report": accepted_candidate.get("report_path")
            if accepted_candidate
            else None,
        },
        "runtime_contract": {
            "authoritative_runtime_config": "config.json",
            "diagnostic_recommendation_pointer": selected_row_pointer,
            "diagnostic_recommendation_role": (
                "selected_row_is_diagnostic_recommendation_not_applied_runtime_config"
            ),
        },
        "source_of_truth_summary": (
            "current/config.json, current/prediction.json, current/forecast_summary.json, "
            "current/source.json, current/research_report.md are authoritative. "
            "PyTorch/report_signalcascade_xauusd.md is a synchronized mirror of current/research_report.md."
        ),
    }

    current_selection_governance: dict[str, object] = {
        "selection_mode": selection_metadata["selection_mode"],
        "selection_rule": selection_metadata["selection_rule"],
        "selection_rule_version": selection_metadata["selection_rule_version"],
        "selection_status": selection_metadata["selection_status"],
        "selection_timestamp_utc": selection_time,
        "selection_session_manifest_path": session_context.get("session_manifest_path"),
        "selection_leaderboard_path": session_context.get("leaderboard_path"),
        "best_candidate": best_candidate,
        "accepted_candidate": accepted_candidate,
        "production_current": production_current,
        "override_applied": override_applied,
        "override_reason": selection_metadata["override_reason"],
        "override_priority_metrics": selection_metadata["override_priority_metrics"],
        "decision_summary": selection_metadata["decision_summary"],
        "paired_frontier": paired_frontier,
    }
    return {
        "current_alias_contract": current_alias_contract,
        "current_selection_governance": current_selection_governance,
    }


def _load_session_context(source_artifact_dir: Path) -> dict[str, object]:
    session_dir = source_artifact_dir.parent
    manifest_path = session_dir / "manifest.json"
    leaderboard_path = session_dir / "leaderboard.json"
    manifest = (
        load_json(manifest_path)
        if session_dir.name.startswith("session_") and manifest_path.exists()
        else {}
    )
    leaderboard_rows: dict[str, dict[str, object]] = {}
    if session_dir.name.startswith("session_") and leaderboard_path.exists():
        leaderboard_payload = load_json(leaderboard_path)
        results = leaderboard_payload.get("results", [])
        if isinstance(results, list):
            leaderboard_rows = {
                str(row["candidate"]): dict(row)
                for row in results
                if isinstance(row, dict) and row.get("candidate") is not None
            }
    return {
        "session_dir": session_dir if manifest else None,
        "manifest": manifest if isinstance(manifest, dict) else {},
        "session_manifest_path": str(manifest_path) if manifest else None,
        "leaderboard_path": str(leaderboard_path) if leaderboard_rows else None,
        "leaderboard_rows": leaderboard_rows,
    }


def _resolve_session_candidate_snapshot(
    session_context: Mapping[str, object],
    manifest_key: str,
) -> dict[str, object] | None:
    manifest = session_context.get("manifest")
    if not isinstance(manifest, dict):
        return None
    session_dir = session_context.get("session_dir")
    if not isinstance(session_dir, Path):
        return None
    candidate_row = manifest.get(manifest_key)
    if not isinstance(candidate_row, dict):
        return None
    candidate_name = candidate_row.get("candidate")
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        return None
    candidate_dir = session_dir / candidate_name
    if not candidate_dir.exists():
        return None
    leaderboard_rows = session_context.get("leaderboard_rows")
    leaderboard_row = (
        leaderboard_rows.get(candidate_name)
        if isinstance(leaderboard_rows, dict)
        else None
    )
    return _build_candidate_snapshot(candidate_dir, leaderboard_row=leaderboard_row)


def _resolve_production_selection_metadata(
    *,
    session_context: Mapping[str, object],
    production_current: Mapping[str, object],
    accepted_candidate: Mapping[str, object] | None,
    override_applied: bool,
    fallback_decision_summary: str,
) -> dict[str, object]:
    manifest = session_context.get("manifest")
    manifest_production = (
        manifest.get("production_current_candidate")
        if isinstance(manifest, dict)
        else None
    )
    manifest_selection = (
        manifest.get("production_current_selection")
        if isinstance(manifest, dict)
        else None
    )
    manifest_production_name = (
        manifest_production.get("candidate")
        if isinstance(manifest_production, dict)
        else None
    )
    if (
        isinstance(manifest_selection, dict)
        and manifest_production_name == production_current.get("candidate")
    ):
        selection_status = manifest_selection.get("selection_status")
        if not isinstance(selection_status, str):
            selection_status = _resolve_selection_status(
                accepted_candidate=accepted_candidate,
                production_candidate=production_current,
                override_applied=override_applied,
            )
        return {
            "selection_mode": manifest_selection.get(
                "selection_mode",
                "accepted_candidate" if accepted_candidate else "manual_promote_without_session_context",
            ),
            "selection_rule": manifest_selection.get(
                "selection_rule",
                _LEGACY_CURRENT_SELECTION_RULE,
            ),
            "selection_rule_version": manifest_selection.get(
                "selection_rule_version",
                _LEGACY_CURRENT_SELECTION_RULE_VERSION,
            ),
            "selection_status": selection_status,
            "override_reason": manifest_selection.get("override_reason"),
            "override_priority_metrics": list(
                manifest_selection.get("override_priority_metrics", [])
            ),
            "decision_summary": manifest_selection.get(
                "decision_summary",
                fallback_decision_summary,
            ),
        }

    legacy_selection_mode = (
        "explicit_governance_override"
        if override_applied
        else (
            "accepted_candidate"
            if accepted_candidate
            else "manual_promote_without_session_context"
        )
    )
    return {
        "selection_mode": legacy_selection_mode,
        "selection_rule": _LEGACY_CURRENT_SELECTION_RULE,
        "selection_rule_version": _LEGACY_CURRENT_SELECTION_RULE_VERSION,
        "selection_status": _resolve_selection_status(
            accepted_candidate=accepted_candidate,
            production_candidate=production_current,
            override_applied=override_applied,
        ),
        "override_reason": (
            "production current prioritizes execution risk budget over blocked-objective winner"
            if override_applied
            else None
        ),
        "override_priority_metrics": (
            list(_LEGACY_OVERRIDE_PRIORITY_METRICS) if override_applied else []
        ),
        "decision_summary": fallback_decision_summary,
    }


def _build_candidate_snapshot(
    artifact_dir: Path,
    *,
    leaderboard_row: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source_payload = load_json(artifact_dir / "source.json") if (artifact_dir / "source.json").exists() else {}
    prediction = load_json(artifact_dir / "prediction.json") if (artifact_dir / "prediction.json").exists() else {}
    diagnostics = (
        load_json(artifact_dir / "validation_summary.json")
        if (artifact_dir / "validation_summary.json").exists()
        else {}
    )
    validation = diagnostics.get("validation")
    validation_metrics = validation if isinstance(validation, dict) else {}
    forecast_quality_scorecards = _resolve_forecast_quality_scorecards(
        artifact_dir,
        diagnostics if isinstance(diagnostics, dict) else {},
    )
    selected_horizon_scorecard = (
        dict(forecast_quality_scorecards.get("selected_horizon"))
        if isinstance(forecast_quality_scorecards.get("selected_horizon"), dict)
        else {}
    )
    all_horizon_scorecard = (
        dict(forecast_quality_scorecards.get("all_horizon"))
        if isinstance(forecast_quality_scorecards.get("all_horizon"), dict)
        else {}
    )
    row = dict(leaderboard_row or {})
    candidate_name = (
        artifact_dir.name
        if artifact_dir.name.startswith("candidate_")
        else row.get("candidate")
    )
    return {
        "candidate": candidate_name,
        "artifact_id": source_payload.get("artifact_id"),
        "artifact_dir": str(artifact_dir),
        "report_path": (
            str(artifact_dir / "research_report.md")
            if (artifact_dir / "research_report.md").exists()
            else None
        ),
        "policy_horizon": prediction.get("policy_horizon"),
        "trade_delta": prediction.get("trade_delta"),
        "tradeability_gate": prediction.get("g_t", prediction.get("tradeability_gate")),
        "average_log_wealth": validation_metrics.get("average_log_wealth"),
        "project_value_score": validation_metrics.get("project_value_score"),
        "utility_score": validation_metrics.get("utility_score"),
        "mu_calibration": validation_metrics.get("mu_calibration"),
        "sigma_calibration": validation_metrics.get("sigma_calibration"),
        "blocked_average_log_wealth_mean": row.get("blocked_average_log_wealth_mean"),
        "blocked_objective_log_wealth_minus_lambda_cvar_mean": row.get(
            "blocked_objective_log_wealth_minus_lambda_cvar_mean"
        ),
        "blocked_turnover_mean": row.get("blocked_turnover_mean"),
        "blocked_directional_accuracy_mean": row.get("blocked_directional_accuracy_mean"),
        "blocked_exact_smooth_position_mae_mean": row.get(
            "blocked_exact_smooth_position_mae_mean"
        ),
        "user_value_score": row.get("user_value_score"),
        "user_value_chart_fidelity_score": row.get(
            "user_value_chart_fidelity_score"
        ),
        "user_value_sigma_band_score": row.get("user_value_sigma_band_score"),
        "user_value_execution_stability_score": row.get(
            "user_value_execution_stability_score"
        ),
        "selected_horizon_forecast_quality_score": row.get(
            "selected_horizon_forecast_quality_score"
        )
        if row.get("selected_horizon_forecast_quality_score") is not None
        else selected_horizon_scorecard.get("quality_score"),
        "all_horizon_forecast_quality_score": row.get(
            "all_horizon_forecast_quality_score"
        )
        if row.get("all_horizon_forecast_quality_score") is not None
        else all_horizon_scorecard.get("quality_score"),
        "forecast_quality_score_gap_all_minus_selected": row.get(
            "forecast_quality_score_gap_all_minus_selected"
        )
        if row.get("forecast_quality_score_gap_all_minus_selected") is not None
        else _safe_delta(
            all_horizon_scorecard.get("quality_score"),
            selected_horizon_scorecard.get("quality_score"),
        ),
        "max_drawdown": validation_metrics.get("max_drawdown"),
        "turnover": validation_metrics.get("turnover"),
        "directional_accuracy": validation_metrics.get("directional_accuracy"),
        "exact_smooth_position_mae": validation_metrics.get("exact_smooth_position_mae"),
        "no_trade_band_hit_rate": validation_metrics.get("no_trade_band_hit_rate"),
        "optimization_gate_passed": row.get("optimization_gate_passed"),
    }


def _resolve_forecast_quality_scorecards(
    artifact_dir: Path,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    existing_scorecards = diagnostics.get("forecast_quality_scorecards")
    if isinstance(existing_scorecards, dict):
        return dict(existing_scorecards)

    dataset = diagnostics.get("dataset")
    dataset_payload = dict(dataset) if isinstance(dataset, dict) else {}
    validation = diagnostics.get("validation")
    validation_payload = dict(validation) if isinstance(validation, dict) else {}
    validation_rows = _load_validation_rows(artifact_dir / "validation_rows.csv")
    validation_sample_count = dataset_payload.get("validation_sample_count")
    return _build_forecast_quality_scorecards(
        validation=validation_payload,
        validation_rows=validation_rows,
        validation_sample_count=(
            int(validation_sample_count)
            if validation_sample_count is not None
            else None
        ),
    )


def _load_validation_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _build_paired_frontier(
    *,
    accepted_candidate: Mapping[str, object] | None,
    production_current: Mapping[str, object],
) -> dict[str, object] | None:
    if accepted_candidate is None:
        return None
    metrics = (
        "user_value_score",
        "average_log_wealth",
        "blocked_objective_log_wealth_minus_lambda_cvar_mean",
        "blocked_turnover_mean",
        "max_drawdown",
        "directional_accuracy",
        "exact_smooth_position_mae",
        "selected_horizon_forecast_quality_score",
        "all_horizon_forecast_quality_score",
        "forecast_quality_score_gap_all_minus_selected",
        "trade_delta",
        "policy_horizon",
    )
    delta = {
        key: _safe_delta(production_current.get(key), accepted_candidate.get(key))
        for key in metrics
    }
    return {
        "optimization_objective_metrics": [
            "average_log_wealth",
            "blocked_objective_log_wealth_minus_lambda_cvar_mean",
        ],
        "governance_risk_budget_metrics": list(_LEGACY_OVERRIDE_PRIORITY_METRICS),
        "user_value_metrics": [
            "user_value_score",
            "blocked_directional_accuracy_mean",
            "mu_calibration",
            "sigma_calibration",
            "blocked_exact_smooth_position_mae_mean",
            "selected_horizon_forecast_quality_score",
            "all_horizon_forecast_quality_score",
            "forecast_quality_score_gap_all_minus_selected",
            "max_drawdown",
            "blocked_turnover_mean",
        ],
        "accepted_candidate": dict(accepted_candidate),
        "production_current": dict(production_current),
        "delta_production_minus_accepted": delta,
    }


def _build_decision_summary(
    *,
    accepted_candidate: Mapping[str, object] | None,
    production_current: Mapping[str, object],
    override_applied: bool,
) -> str:
    if accepted_candidate is None:
        return "current alias was promoted without session manifest context."
    if not override_applied:
        return "production current matches the accepted candidate selected by optimization gate."
    return (
        "production current differs from the accepted candidate because execution risk budget "
        "took priority over blocked-objective rank."
    )


def _resolve_selection_status(
    accepted_candidate: Mapping[str, object] | None,
    production_candidate: Mapping[str, object] | None,
    override_applied: bool,
) -> str:
    accepted_candidate_name = (
        str(accepted_candidate["candidate"])
        if isinstance(accepted_candidate, Mapping)
        and isinstance(accepted_candidate.get("candidate"), str)
        else None
    )
    production_candidate_name = (
        str(production_candidate["candidate"])
        if isinstance(production_candidate, Mapping)
        and isinstance(production_candidate.get("candidate"), str)
        else None
    )

    if accepted_candidate_name is None and production_candidate_name is None:
        return "no_candidate_passed_gate"
    if accepted_candidate_name is not None and production_candidate_name is None:
        return "quick_mode_non_promotable"
    if accepted_candidate_name is not None and accepted_candidate_name == production_candidate_name:
        return "accepted_and_production_same" if not override_applied else "accepted_and_production_same"
    if accepted_candidate_name is None and production_candidate_name is not None:
        return "accepted_and_production_diverged"
    return "accepted_and_production_diverged"


def _safe_delta(left: object, right: object) -> float | None:
    try:
        if left is None or right is None:
            return None
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None
