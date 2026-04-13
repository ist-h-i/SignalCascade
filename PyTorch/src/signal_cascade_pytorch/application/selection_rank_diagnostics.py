from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

from .diagnostics_service import _build_forecast_quality_scorecards
from ..infrastructure.persistence import load_json

_SELECTED_HORIZON_SCORE_KEY = "selected_horizon_forecast_quality_score"
_ALL_HORIZON_SCORE_KEY = "all_horizon_forecast_quality_score"


def build_forecast_quality_ranking_diagnostics(
    leaderboard_rows: Sequence[Mapping[str, object]],
    *,
    accepted_candidate: str | None = None,
    production_current_candidate: str | None = None,
) -> dict[str, object]:
    current_ranked_candidates = [
        str(row["candidate"])
        for row in leaderboard_rows
        if isinstance(row, Mapping) and isinstance(row.get("candidate"), str)
    ]
    selected_ranked_candidates = _rank_candidates_by_metric(
        leaderboard_rows,
        metric_key=_SELECTED_HORIZON_SCORE_KEY,
    )
    all_ranked_candidates = _rank_candidates_by_metric(
        leaderboard_rows,
        metric_key=_ALL_HORIZON_SCORE_KEY,
    )

    top_k = min(3, len(current_ranked_candidates))
    selected_overlap_count = _top_k_overlap_count(
        current_ranked_candidates,
        selected_ranked_candidates,
        top_k=top_k,
    )
    all_overlap_count = _top_k_overlap_count(
        current_ranked_candidates,
        all_ranked_candidates,
        top_k=top_k,
    )

    return {
        "candidate_count": len(current_ranked_candidates),
        "top_k": top_k,
        "current_top_candidate": current_ranked_candidates[0] if current_ranked_candidates else None,
        "selected_horizon_top_candidate": (
            selected_ranked_candidates[0] if selected_ranked_candidates else None
        ),
        "all_horizon_top_candidate": all_ranked_candidates[0] if all_ranked_candidates else None,
        "selected_horizon_vs_current_spearman_rank_correlation": _spearman_rank_correlation(
            current_ranked_candidates,
            selected_ranked_candidates,
        ),
        "all_horizon_vs_current_spearman_rank_correlation": _spearman_rank_correlation(
            current_ranked_candidates,
            all_ranked_candidates,
        ),
        "selected_horizon_vs_all_horizon_spearman_rank_correlation": _spearman_rank_correlation(
            selected_ranked_candidates,
            all_ranked_candidates,
        ),
        "selected_horizon_top_k_overlap_with_current_count": selected_overlap_count,
        "selected_horizon_top_k_overlap_with_current_ratio": (
            None if top_k == 0 else selected_overlap_count / float(top_k)
        ),
        "all_horizon_top_k_overlap_with_current_count": all_overlap_count,
        "all_horizon_top_k_overlap_with_current_ratio": (
            None if top_k == 0 else all_overlap_count / float(top_k)
        ),
        "accepted_candidate": accepted_candidate,
        "accepted_candidate_current_rank": _candidate_rank(
            current_ranked_candidates,
            accepted_candidate,
        ),
        "accepted_candidate_selected_horizon_rank": _candidate_rank(
            selected_ranked_candidates,
            accepted_candidate,
        ),
        "accepted_candidate_all_horizon_rank": _candidate_rank(
            all_ranked_candidates,
            accepted_candidate,
        ),
        "production_current_candidate": production_current_candidate,
        "production_current_current_rank": _candidate_rank(
            current_ranked_candidates,
            production_current_candidate,
        ),
        "production_current_selected_horizon_rank": _candidate_rank(
            selected_ranked_candidates,
            production_current_candidate,
        ),
        "production_current_all_horizon_rank": _candidate_rank(
            all_ranked_candidates,
            production_current_candidate,
        ),
        "current_top_k_candidates": current_ranked_candidates[:top_k],
        "selected_horizon_top_k_candidates": selected_ranked_candidates[:top_k],
        "all_horizon_top_k_candidates": all_ranked_candidates[:top_k],
    }


def build_selection_history_summary(
    archive_root: Path,
    *,
    recent_limit: int = 3,
    session_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    resolved_records = _resolve_session_records(
        archive_root,
        session_records=session_records,
    )
    if not resolved_records:
        return {}

    accepted_records = [
        record for record in resolved_records if record.get("accepted_candidate") is not None
    ]
    production_records = [
        record
        for record in resolved_records
        if record.get("production_current_candidate") is not None
    ]
    divergence_count = sum(
        1
        for record in resolved_records
        if record.get("accepted_candidate") is not None
        and record.get("production_current_candidate") is not None
        and record.get("accepted_candidate") != record.get("production_current_candidate")
    )

    return {
        "session_count": len(resolved_records),
        "accepted_candidate_count": len(accepted_records),
        "production_current_candidate_count": len(production_records),
        "accepted_vs_production_divergence_count": divergence_count,
        "accepted_current_top_match_ratio": _rank_one_ratio(
            accepted_records,
            "accepted_candidate_current_rank",
        ),
        "accepted_selected_horizon_top_match_ratio": _rank_one_ratio(
            accepted_records,
            "accepted_candidate_selected_horizon_rank",
        ),
        "accepted_all_horizon_top_match_ratio": _rank_one_ratio(
            accepted_records,
            "accepted_candidate_all_horizon_rank",
        ),
        "accepted_candidate_current_rank_median": _median_rank(
            accepted_records,
            "accepted_candidate_current_rank",
        ),
        "accepted_candidate_selected_horizon_rank_median": _median_rank(
            accepted_records,
            "accepted_candidate_selected_horizon_rank",
        ),
        "accepted_candidate_all_horizon_rank_median": _median_rank(
            accepted_records,
            "accepted_candidate_all_horizon_rank",
        ),
        "production_current_top_match_ratio": _rank_one_ratio(
            production_records,
            "production_current_current_rank",
        ),
        "production_selected_horizon_top_match_ratio": _rank_one_ratio(
            production_records,
            "production_current_selected_horizon_rank",
        ),
        "production_all_horizon_top_match_ratio": _rank_one_ratio(
            production_records,
            "production_current_all_horizon_rank",
        ),
        "production_current_current_rank_median": _median_rank(
            production_records,
            "production_current_current_rank",
        ),
        "production_current_selected_horizon_rank_median": _median_rank(
            production_records,
            "production_current_selected_horizon_rank",
        ),
        "production_current_all_horizon_rank_median": _median_rank(
            production_records,
            "production_current_all_horizon_rank",
        ),
        "recent_sessions": list(resolved_records[-recent_limit:]),
    }


def build_selection_divergence_scorecard(
    archive_root: Path,
    *,
    recent_limit: int = 3,
    session_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    resolved_records = _resolve_session_records(
        archive_root,
        session_records=session_records,
    )
    if not resolved_records:
        return {}

    full_rows = [
        record for record in resolved_records if record.get("coverage_status") == "full"
    ]
    partial_rows = [
        record for record in resolved_records if record.get("coverage_status") != "full"
    ]
    cluster_counts = Counter(
        str(record.get("failure_mode_cluster"))
        for record in resolved_records
        if record.get("failure_mode_cluster") is not None
    )
    full_cluster_counts = Counter(
        str(record.get("failure_mode_cluster"))
        for record in full_rows
        if record.get("failure_mode_cluster") is not None
    )
    return {
        "session_count": len(resolved_records),
        "full_coverage_session_count": len(full_rows),
        "partial_coverage_session_count": len(partial_rows),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "full_coverage_cluster_counts": dict(sorted(full_cluster_counts.items())),
        "rows": list(resolved_records),
        "recent_rows": list(resolved_records[-recent_limit:]),
    }


def backfill_forecast_quality_metrics_from_session(
    leaderboard_rows: Sequence[Mapping[str, object]],
    session_dir: Path,
) -> list[dict[str, object]]:
    enriched_rows: list[dict[str, object]] = []
    for row in leaderboard_rows:
        enriched = dict(row)
        candidate = enriched.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            enriched_rows.append(enriched)
            continue
        if (
            enriched.get(_SELECTED_HORIZON_SCORE_KEY) is None
            or enriched.get(_ALL_HORIZON_SCORE_KEY) is None
            or enriched.get("forecast_quality_score_gap_all_minus_selected") is None
        ):
            enriched.update(_load_forecast_quality_metric_columns(session_dir / candidate))
        enriched_rows.append(enriched)
    return enriched_rows


def _load_forecast_quality_metric_columns(candidate_dir: Path) -> dict[str, object]:
    diagnostics_path = candidate_dir / "validation_summary.json"
    if not diagnostics_path.exists():
        return {}
    diagnostics = load_json(diagnostics_path)
    if not isinstance(diagnostics, dict):
        return {}
    scorecards_payload = diagnostics.get("forecast_quality_scorecards")
    if isinstance(scorecards_payload, dict):
        scorecards = dict(scorecards_payload)
    else:
        dataset = diagnostics.get("dataset")
        dataset_payload = dict(dataset) if isinstance(dataset, dict) else {}
        validation = diagnostics.get("validation")
        validation_payload = dict(validation) if isinstance(validation, dict) else {}
        validation_sample_count = dataset_payload.get("validation_sample_count")
        scorecards = _build_forecast_quality_scorecards(
            validation=validation_payload,
            validation_rows=_load_validation_rows(candidate_dir / "validation_rows.csv"),
            validation_sample_count=(
                int(validation_sample_count)
                if validation_sample_count is not None
                else None
            ),
        )

    selected = (
        dict(scorecards.get("selected_horizon"))
        if isinstance(scorecards.get("selected_horizon"), dict)
        else {}
    )
    all_horizon = (
        dict(scorecards.get("all_horizon"))
        if isinstance(scorecards.get("all_horizon"), dict)
        else {}
    )
    selected_quality_score = _finite_float_or_none(selected.get("quality_score"))
    all_quality_score = _finite_float_or_none(all_horizon.get("quality_score"))
    score_gap = _finite_float_or_none(scorecards.get("quality_score_gap_all_minus_selected"))
    if score_gap is None and selected_quality_score is not None and all_quality_score is not None:
        score_gap = all_quality_score - selected_quality_score
    return {
        _SELECTED_HORIZON_SCORE_KEY: selected_quality_score,
        _ALL_HORIZON_SCORE_KEY: all_quality_score,
        "forecast_quality_score_gap_all_minus_selected": score_gap,
    }


def _build_session_record(manifest_path: Path) -> dict[str, object]:
    manifest_payload = load_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        return {}
    accepted_candidate = _candidate_name(manifest_payload.get("accepted_candidate"))
    production_current_candidate = _candidate_name(
        manifest_payload.get("production_current_candidate")
    )
    leaderboard_rows, resolved_diagnostics = _resolve_session_leaderboard_context(
        manifest_payload,
        accepted_candidate=accepted_candidate,
        production_current_candidate=production_current_candidate,
    )
    record = {
        "session_id": manifest_payload.get("session_id", manifest_path.parent.name),
        "generated_at_utc": manifest_payload.get("generated_at_utc"),
        "selection_mode": (
            dict(manifest_payload.get("production_current_selection", {})).get("selection_mode")
            if isinstance(manifest_payload.get("production_current_selection"), dict)
            else None
        ),
        "selection_status": manifest_payload.get("selection_status"),
        "accepted_candidate": accepted_candidate,
        "production_current_candidate": production_current_candidate,
        "current_top_candidate": resolved_diagnostics.get("current_top_candidate"),
        "selected_horizon_top_candidate": resolved_diagnostics.get(
            "selected_horizon_top_candidate"
        ),
        "all_horizon_top_candidate": resolved_diagnostics.get("all_horizon_top_candidate"),
        "accepted_candidate_current_rank": resolved_diagnostics.get(
            "accepted_candidate_current_rank"
        ),
        "accepted_candidate_selected_horizon_rank": resolved_diagnostics.get(
            "accepted_candidate_selected_horizon_rank"
        ),
        "accepted_candidate_all_horizon_rank": resolved_diagnostics.get(
            "accepted_candidate_all_horizon_rank"
        ),
        "production_current_current_rank": resolved_diagnostics.get(
            "production_current_current_rank"
        ),
        "production_current_selected_horizon_rank": resolved_diagnostics.get(
            "production_current_selected_horizon_rank"
        ),
        "production_current_all_horizon_rank": resolved_diagnostics.get(
            "production_current_all_horizon_rank"
        ),
        "selected_horizon_vs_current_spearman_rank_correlation": resolved_diagnostics.get(
            "selected_horizon_vs_current_spearman_rank_correlation"
        ),
        "all_horizon_vs_current_spearman_rank_correlation": resolved_diagnostics.get(
            "all_horizon_vs_current_spearman_rank_correlation"
        ),
        "selected_horizon_vs_all_horizon_spearman_rank_correlation": resolved_diagnostics.get(
            "selected_horizon_vs_all_horizon_spearman_rank_correlation"
        ),
        "selected_horizon_top_k_overlap_with_current_count": resolved_diagnostics.get(
            "selected_horizon_top_k_overlap_with_current_count"
        ),
        "all_horizon_top_k_overlap_with_current_count": resolved_diagnostics.get(
            "all_horizon_top_k_overlap_with_current_count"
        ),
        "top_k": resolved_diagnostics.get("top_k"),
        "candidate_count": resolved_diagnostics.get("candidate_count"),
        "override_flag": (
            accepted_candidate is not None
            and production_current_candidate is not None
            and accepted_candidate != production_current_candidate
        ),
        **_candidate_snapshot_from_leaderboard_rows(
            leaderboard_rows,
            candidate=accepted_candidate,
            prefix="accepted_candidate",
        ),
        **_candidate_snapshot_from_leaderboard_rows(
            leaderboard_rows,
            candidate=production_current_candidate,
            prefix="production_current",
        ),
    }
    record["accepted_candidate_all_minus_current_rank"] = _rank_delta(
        record.get("accepted_candidate_current_rank"),
        record.get("accepted_candidate_all_horizon_rank"),
    )
    record["production_current_all_minus_current_rank"] = _rank_delta(
        record.get("production_current_current_rank"),
        record.get("production_current_all_horizon_rank"),
    )
    record["coverage_status"] = _coverage_status(record)
    (
        record["failure_mode_cluster"],
        record["failure_mode_reason"],
    ) = _classify_failure_mode(record)
    return record


def _resolve_session_records(
    archive_root: Path,
    *,
    session_records: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    if session_records is not None:
        return [dict(record) for record in session_records if isinstance(record, Mapping)]

    resolved_archive_root = archive_root.expanduser().resolve()
    if not resolved_archive_root.exists():
        return []
    manifest_paths = sorted(resolved_archive_root.glob("session_*/manifest.json"))
    if not manifest_paths:
        return []

    resolved_records: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        record = _build_session_record(manifest_path)
        if record:
            resolved_records.append(record)
    return resolved_records


def _resolve_session_leaderboard_context(
    manifest_payload: Mapping[str, object],
    *,
    accepted_candidate: str | None,
    production_current_candidate: str | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    diagnostics = manifest_payload.get("forecast_quality_ranking_diagnostics")
    resolved_diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
    leaderboard_rows: list[dict[str, object]] = []
    leaderboard_path_value = manifest_payload.get("leaderboard_path")
    if isinstance(leaderboard_path_value, str) and leaderboard_path_value.strip():
        leaderboard_path = Path(leaderboard_path_value).expanduser().resolve()
        if leaderboard_path.exists():
            leaderboard_payload = load_json(leaderboard_path)
            rows_payload = leaderboard_payload.get("results")
            if isinstance(rows_payload, list):
                leaderboard_rows = backfill_forecast_quality_metrics_from_session(
                    rows_payload,
                    leaderboard_path.parent,
                )
            if not resolved_diagnostics:
                leaderboard_diagnostics = leaderboard_payload.get(
                    "forecast_quality_ranking_diagnostics"
                )
                if isinstance(leaderboard_diagnostics, dict):
                    resolved_diagnostics = dict(leaderboard_diagnostics)
            if not resolved_diagnostics and leaderboard_rows:
                resolved_diagnostics = build_forecast_quality_ranking_diagnostics(
                    leaderboard_rows,
                    accepted_candidate=accepted_candidate,
                    production_current_candidate=production_current_candidate,
                )
    return leaderboard_rows, resolved_diagnostics


def _candidate_snapshot_from_leaderboard_rows(
    leaderboard_rows: Sequence[Mapping[str, object]],
    *,
    candidate: str | None,
    prefix: str,
) -> dict[str, object]:
    if candidate is None:
        return {}
    row = next(
        (
            leaderboard_row
            for leaderboard_row in leaderboard_rows
            if isinstance(leaderboard_row, Mapping)
            and leaderboard_row.get("candidate") == candidate
        ),
        None,
    )
    if not isinstance(row, Mapping):
        return {}
    return {
        f"{prefix}_blocked_objective_log_wealth_minus_lambda_cvar_mean": row.get(
            "blocked_objective_log_wealth_minus_lambda_cvar_mean"
        ),
        f"{prefix}_blocked_turnover_mean": row.get("blocked_turnover_mean"),
        f"{prefix}_blocked_directional_accuracy_mean": row.get(
            "blocked_directional_accuracy_mean"
        ),
        f"{prefix}_blocked_exact_smooth_position_mae_mean": row.get(
            "blocked_exact_smooth_position_mae_mean"
        ),
        f"{prefix}_max_drawdown": row.get("max_drawdown"),
        f"{prefix}_deployment_score": row.get("deployment_score"),
        f"{prefix}_user_value_score": row.get("user_value_score"),
        f"{prefix}_policy_cost_multiplier": row.get("policy_cost_multiplier"),
        f"{prefix}_evaluation_state_reset_mode": row.get("evaluation_state_reset_mode"),
        f"{prefix}_selected_horizon_forecast_quality_score": row.get(
            _SELECTED_HORIZON_SCORE_KEY
        ),
        f"{prefix}_all_horizon_forecast_quality_score": row.get(_ALL_HORIZON_SCORE_KEY),
    }


def _coverage_status(record: Mapping[str, object]) -> str:
    accepted_full = _has_rank_triplet(record, "accepted_candidate")
    production_candidate = record.get("production_current_candidate")
    production_full = (
        True
        if production_candidate is None
        else _has_rank_triplet(record, "production_current")
    )
    return "full" if accepted_full and production_full else "partial"


def _has_rank_triplet(record: Mapping[str, object], prefix: str) -> bool:
    return all(
        record.get(f"{prefix}_{suffix}") is not None
        for suffix in ("current_rank", "selected_horizon_rank", "all_horizon_rank")
    )


def _rank_delta(current_rank: object, comparison_rank: object) -> int | None:
    if current_rank is None or comparison_rank is None:
        return None
    return int(comparison_rank) - int(current_rank)


def _classify_failure_mode(record: Mapping[str, object]) -> tuple[str, str]:
    if record.get("coverage_status") != "full":
        return (
            "insufficient_coverage",
            "accepted/prod rank backfill is incomplete for this session",
        )

    if _is_stability_override(record):
        return (
            "stability_override",
            "production override sacrifices forecast-quality rank to gain execution stability",
        )

    accepted_rank_gap = _rank_delta(
        record.get("accepted_candidate_current_rank"),
        record.get("accepted_candidate_all_horizon_rank"),
    )
    if accepted_rank_gap is not None and accepted_rank_gap >= 3:
        if _has_stability_view(record, "accepted_candidate"):
            if _is_non_catastrophic_stability(record, "accepted_candidate"):
                return (
                    "objective_evaluation_mismatch",
                    "all-horizon rank deteriorates while accepted candidate stays operationally stable",
                )
            return (
                "rank_stability_coupled",
                "all-horizon rank deterioration co-occurs with stressed stability metrics",
            )
        return (
            "rank_only_pending_stability",
            "ranking split is visible but stability diagnostics are incomplete",
        )

    return (
        "broad_alignment",
        "current and all-horizon rankings remain broadly aligned",
    )


def _is_stability_override(record: Mapping[str, object]) -> bool:
    if not bool(record.get("override_flag")):
        return False
    if not _has_stability_view(record, "accepted_candidate"):
        return False
    if not _has_stability_view(record, "production_current"):
        return False

    accepted_all_rank = record.get("accepted_candidate_all_horizon_rank")
    production_all_rank = record.get("production_current_all_horizon_rank")
    accepted_selected_rank = record.get("accepted_candidate_selected_horizon_rank")
    production_selected_rank = record.get("production_current_selected_horizon_rank")
    if accepted_all_rank is None or production_all_rank is None:
        return False
    rank_penalty = int(production_all_rank) - int(accepted_all_rank) >= 3
    if (
        accepted_selected_rank is not None
        and production_selected_rank is not None
        and int(production_selected_rank) - int(accepted_selected_rank) >= 3
    ):
        rank_penalty = True
    if not rank_penalty:
        return False

    improvement_count = 0
    for metric_key in (
        "blocked_turnover_mean",
        "blocked_exact_smooth_position_mae_mean",
        "max_drawdown",
    ):
        accepted_value = _finite_float_or_none(
            record.get(f"accepted_candidate_{metric_key}")
        )
        production_value = _finite_float_or_none(
            record.get(f"production_current_{metric_key}")
        )
        if (
            accepted_value is not None
            and production_value is not None
            and production_value < accepted_value
        ):
            improvement_count += 1
    return improvement_count >= 2


def _has_stability_view(record: Mapping[str, object], prefix: str) -> bool:
    return any(
        _finite_float_or_none(record.get(f"{prefix}_{metric_key}")) is not None
        for metric_key in (
            "blocked_turnover_mean",
            "blocked_exact_smooth_position_mae_mean",
            "max_drawdown",
        )
    )


def _is_non_catastrophic_stability(record: Mapping[str, object], prefix: str) -> bool:
    max_drawdown = _finite_float_or_none(record.get(f"{prefix}_max_drawdown"))
    exact_smooth_position_mae = _finite_float_or_none(
        record.get(f"{prefix}_blocked_exact_smooth_position_mae_mean")
    )
    blocked_turnover = _finite_float_or_none(record.get(f"{prefix}_blocked_turnover_mean"))
    if max_drawdown is not None and max_drawdown > 0.05:
        return False
    if exact_smooth_position_mae is not None and exact_smooth_position_mae > 0.05:
        return False
    if blocked_turnover is not None and blocked_turnover > 1.25:
        return False
    return True


def _candidate_name(payload: object) -> str | None:
    if isinstance(payload, Mapping) and payload.get("candidate") is not None:
        return str(payload.get("candidate"))
    return None


def _load_validation_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rank_candidates_by_metric(
    leaderboard_rows: Sequence[Mapping[str, object]],
    *,
    metric_key: str,
) -> list[str]:
    ranked_rows: list[tuple[str, float]] = []
    for row in leaderboard_rows:
        if not isinstance(row, Mapping):
            continue
        candidate = row.get("candidate")
        metric_value = _finite_float_or_none(row.get(metric_key))
        if not isinstance(candidate, str) or metric_value is None:
            continue
        ranked_rows.append((candidate, metric_value))
    ranked_rows.sort(key=lambda item: (-item[1], item[0]))
    return [candidate for candidate, _ in ranked_rows]


def _candidate_rank(ranked_candidates: Sequence[str], candidate: str | None) -> int | None:
    if candidate is None:
        return None
    try:
        return ranked_candidates.index(candidate) + 1
    except ValueError:
        return None


def _median_rank(records: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [int(value) for value in (record.get(key) for record in records) if value is not None]
    if not values:
        return None
    return float(median(values))


def _rank_one_ratio(records: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [record.get(key) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(1 for value in values if int(value) == 1) / float(len(values))


def _top_k_overlap_count(
    left_ranked_candidates: Sequence[str],
    right_ranked_candidates: Sequence[str],
    *,
    top_k: int,
) -> int:
    if top_k <= 0:
        return 0
    return len(set(left_ranked_candidates[:top_k]) & set(right_ranked_candidates[:top_k]))


def _spearman_rank_correlation(
    left_ranked_candidates: Sequence[str],
    right_ranked_candidates: Sequence[str],
) -> float | None:
    left_rank_map = {candidate: rank for rank, candidate in enumerate(left_ranked_candidates, start=1)}
    right_rank_map = {candidate: rank for rank, candidate in enumerate(right_ranked_candidates, start=1)}
    common_candidates = sorted(set(left_rank_map) & set(right_rank_map))
    candidate_count = len(common_candidates)
    if candidate_count < 2:
        return None
    sum_of_squared_rank_deltas = sum(
        (left_rank_map[candidate] - right_rank_map[candidate]) ** 2
        for candidate in common_candidates
    )
    return 1.0 - (
        (6.0 * float(sum_of_squared_rank_deltas))
        / float(candidate_count * ((candidate_count**2) - 1))
    )


def _finite_float_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
