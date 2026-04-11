from __future__ import annotations

import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .artifact_provenance import (
    build_artifact_source_payload,
    build_subartifact_lineage,
    materialize_artifact_source,
)
from .artifact_manifest import build_artifact_entrypoints, build_artifact_manifest
from .config import TrainingConfig
from .current_alias import build_current_alias_metadata
from .dataset_service import (
    build_latest_inference_example_from_bars,
    build_training_examples_from_bars,
    limit_base_bars_to_lookback_days,
)
from .diagnostics_service import export_review_diagnostics
from .inference_service import (
    build_forecast_summary_payload,
    predict_latest,
    serialize_prediction_result,
)
from .price_scale import normalize_price_scale_payload, price_scale_manifest_fields, resolve_effective_price_scale
from .report_service import (
    METRICS_SCHEMA_VERSION,
    generate_research_report,
    load_required_diagnostics_summary,
)
from .training_service import train_model
from ..infrastructure.data.csv_source import CsvMarketDataSource
from ..infrastructure.persistence import ensure_directory, load_json, save_json

_RESERVED_ARTIFACT_DIRECTORIES = {"archive", "current", "live"}
_DEFAULT_QUICK_MODE_CANDIDATE_LIMIT = 8
_FALLBACK_PARAMETERS = {
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
_OPTIMIZATION_GATE_RULES = (
    {"metric": "average_log_wealth", "operator": "minimum", "threshold": 0.0},
    {"metric": "realized_pnl_per_anchor", "operator": "minimum", "threshold": 0.0},
    {
        "metric": "blocked_objective_log_wealth_minus_lambda_cvar_mean",
        "operator": "minimum",
        "threshold": -0.0010,
    },
    {"metric": "cvar_tail_loss", "operator": "maximum", "threshold": 0.08},
    {"metric": "max_drawdown", "operator": "maximum", "threshold": 0.15},
    {"metric": "directional_accuracy", "operator": "minimum", "threshold": 0.50},
    {
        "metric": "blocked_directional_accuracy_mean",
        "operator": "minimum",
        "threshold": 0.52,
    },
    {
        "metric": "blocked_exact_smooth_position_mae_mean",
        "operator": "maximum",
        "threshold": 0.05,
    },
    {"metric": "sigma_calibration", "operator": "maximum", "threshold": 0.12},
    {
        "metric": "runtime_policy_alignment_score",
        "operator": "minimum",
        "threshold": 0.60,
    },
    {"metric": "no_trade_band_hit_rate", "operator": "maximum", "threshold": 0.80},
)
_PRODUCTION_CURRENT_SELECTION_RULE = "optimization_gate_then_deployment_score"
_PRODUCTION_CURRENT_SELECTION_RULE_VERSION = 1
_PRODUCTION_CURRENT_PRIORITY_METRICS = (
    "deployment_score",
    "deployment_economic_core_score",
    "deployment_drawdown_score",
    "deployment_cvar_tail_score",
    "deployment_turnover_score",
    "deployment_sigma_score",
    "deployment_probabilistic_score",
    "deployment_runtime_alignment_score",
    "blocked_objective_log_wealth_minus_lambda_cvar_mean",
)


def tune_latest_dataset(
    csv_path: Path,
    artifact_root: Path,
    seed: int = 7,
    config_overrides: dict[str, object] | None = None,
    lookback_days: int | None = None,
    candidate_limit: int | None = None,
    quick_mode: bool = False,
    warm_start_from_current: bool = False,
) -> dict[str, object]:
    csv_path = csv_path.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    archive_root = ensure_directory(artifact_root / "archive")
    current_dir = artifact_root / "current"
    source = CsvMarketDataSource(csv_path)
    base_bars = source.load_bars()
    source_rows_original = len(base_bars)
    base_bars = limit_base_bars_to_lookback_days(base_bars, lookback_days)
    source_rows_used = len(base_bars)
    all_overrides = dict(config_overrides or {})
    tunable_overrides = _extract_tunable_overrides(all_overrides)
    static_overrides = {
        key: value for key, value in all_overrides.items() if key not in tunable_overrides
    }

    baseline_config = TrainingConfig(
        seed=seed,
        output_dir=str(current_dir),
        **static_overrides,
        **tunable_overrides,
    )
    source_payload = {"kind": "csv", "path": str(csv_path)}
    if lookback_days is not None:
        source_payload["lookback_days"] = int(lookback_days)
    source_payload = normalize_price_scale_payload(
        source_payload,
        requested_price_scale=static_overrides.get("requested_price_scale"),
    )
    price_scale = resolve_effective_price_scale(source_payload)
    examples = build_training_examples_from_bars(
        base_bars,
        baseline_config,
        price_scale=price_scale,
    )
    latest_inference_example = (
        build_latest_inference_example_from_bars(
            base_bars,
            baseline_config,
            price_scale=price_scale,
        )
        if len(base_bars) >= 512
        else None
    )
    inherited_parameters = _load_parameter_seed(artifact_root)
    inherited_parameters.update(tunable_overrides)
    candidate_parameters = _build_candidate_parameters(inherited_parameters)
    evaluated_candidate_parameters, resolved_candidate_limit = _resolve_session_candidates(
        candidate_parameters,
        candidate_limit=candidate_limit,
        quick_mode=quick_mode,
    )
    warm_start_checkpoint_path, warm_start_config = _resolve_current_warm_start_seed(
        artifact_root,
        warm_start_from_current=warm_start_from_current,
    )
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = ensure_directory(archive_root / f"session_{session_id}")
    leaderboard: list[dict[str, object]] = []

    for index, parameters in enumerate(evaluated_candidate_parameters, start=1):
        candidate_name = f"candidate_{index:02d}"
        candidate_dir = ensure_directory(session_dir / candidate_name)
        config = TrainingConfig(seed=seed, output_dir=str(candidate_dir), **static_overrides, **parameters)
        candidate_warm_start_checkpoint_path = (
            warm_start_checkpoint_path
            if _supports_candidate_warm_start(
                warm_start_config,
                config,
            )
            else None
        )
        model, summary = train_model(
            examples,
            config,
            candidate_dir,
            warm_start_checkpoint_path=candidate_warm_start_checkpoint_path,
        )
        prediction = predict_latest(
            model,
            examples,
            config,
            latest_example=latest_inference_example,
        )
        diagnostics_summary = _write_run_artifacts(
            output_dir=candidate_dir,
            model=model,
            examples=examples,
            config=config,
            source_payload=source_payload,
            base_bars=base_bars,
            summary=summary,
            prediction=prediction,
            sample_count=len(examples),
            source_rows_original=source_rows_original,
            source_rows_used=source_rows_used,
        )
        validation_metrics = dict(summary["validation_metrics"])
        result_row = {
            "candidate": candidate_name,
            "best_validation_loss": summary["best_validation_loss"],
            "project_value_score": validation_metrics["project_value_score"],
            "utility_score": validation_metrics["utility_score"],
            "average_log_wealth": validation_metrics["average_log_wealth"],
            "realized_pnl_per_anchor": validation_metrics["realized_pnl_per_anchor"],
            "cvar_tail_loss": validation_metrics["cvar_tail_loss"],
            "max_drawdown": validation_metrics["max_drawdown"],
            "no_trade_band_hit_rate": validation_metrics["no_trade_band_hit_rate"],
            "mu_calibration": validation_metrics["mu_calibration"],
            "sigma_calibration": validation_metrics["sigma_calibration"],
            "directional_accuracy": validation_metrics["directional_accuracy"],
            "policy_horizon": prediction.policy_horizon,
            "executed_horizon": prediction.executed_horizon,
            "position": prediction.position,
            "tradeability_gate": prediction.tradeability_gate,
            "anchor_time": prediction.anchor_time,
            **parameters,
        }
        result_row.update(_extract_forecast_profile_metrics(prediction))
        result_row.update(_extract_blocked_candidate_metrics(diagnostics_summary, config))
        result_row.update(_extract_runtime_policy_alignment_metrics(diagnostics_summary, config))
        result_row.update(_evaluate_optimization_gate(result_row))
        result_row.update(_build_user_value_metrics(result_row))
        result_row.update(_build_deployment_score_metrics(result_row))
        leaderboard.append(result_row)

    leaderboard.sort(
        key=lambda row: (
            not bool(row["optimization_gate_passed"]),
            _descending_metric(
                row,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean",
                "project_value_score",
            ),
            _descending_metric(
                row,
                "blocked_average_log_wealth_mean",
                "average_log_wealth",
            ),
            _ascending_metric(row, "blocked_turnover_mean"),
            -float(row["project_value_score"]),
            -float(row["average_log_wealth"]),
            -float(row["realized_pnl_per_anchor"]),
            float(row["cvar_tail_loss"]),
            float(row["max_drawdown"]),
            float(row["mu_calibration"]),
            float(row["sigma_calibration"]),
            float(row["no_trade_band_hit_rate"]),
            -float(row["directional_accuracy"]),
            float(row["best_validation_loss"]),
        )
    )
    save_json(
        session_dir / "leaderboard.json",
        {
            "results": leaderboard,
            "generated_candidate_count": len(candidate_parameters),
            "evaluated_candidate_count": len(evaluated_candidate_parameters),
            "candidate_limit": resolved_candidate_limit,
            "quick_mode": bool(quick_mode),
        },
    )

    best_result = leaderboard[0]
    accepted_result = next(
        (row for row in leaderboard if bool(row["optimization_gate_passed"])),
        None,
    )
    production_result = None if quick_mode else _select_production_current_candidate(leaderboard)
    production_current_selection = _build_production_current_selection_payload(
        accepted_result=accepted_result,
        production_result=production_result,
        quick_mode=quick_mode,
    )
    selection_status = production_current_selection.get("selection_status")
    if not isinstance(selection_status, str):
        if accepted_result is None:
            selection_status = "no_candidate_passed_gate"
        elif production_result is None:
            selection_status = "quick_mode_non_promotable" if quick_mode else "accepted_and_production_same"
        elif production_result.get("candidate") == accepted_result.get("candidate"):
            selection_status = "accepted_and_production_same"
        else:
            selection_status = "accepted_and_production_diverged"
    archived_current_dir = None
    archived_legacy_root_dirs: list[Path] = []

    if production_result is not None:
        accepted_candidate_dir = session_dir / str(accepted_result["candidate"])
        accepted_candidate_config = load_json(accepted_candidate_dir / "config.json")
        load_required_diagnostics_summary(accepted_candidate_dir)
        production_candidate_dir = session_dir / str(production_result["candidate"])
        load_required_diagnostics_summary(production_candidate_dir)
        archived_current_dir = _archive_existing_current(current_dir, session_dir)
        archived_legacy_root_dirs = _archive_legacy_root_runs(artifact_root, session_dir)

        if current_dir.exists():
            shutil.rmtree(current_dir)
        shutil.copytree(production_candidate_dir, current_dir)

    current_dir_payload = str(current_dir) if current_dir.exists() or production_result is not None else None

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "session_id": session_id,
        "generated_at": generated_at_utc,
        "generated_at_utc": generated_at_utc,
        "artifact_root": str(artifact_root),
        "current_dir": current_dir_payload,
        "archive_session_dir": str(session_dir),
        "leaderboard_path": str(session_dir / "leaderboard.json"),
        "manifest_path": str(session_dir / "manifest.json"),
        "source_path": str(csv_path),
        "lookback_days": lookback_days,
        "quick_mode": bool(quick_mode),
        "candidate_limit": resolved_candidate_limit,
        **price_scale_manifest_fields(source_payload),
        "sample_count": len(examples),
        "source_rows_original": source_rows_original,
        "source_rows_used": source_rows_used,
        "generated_candidate_count": len(candidate_parameters),
        "evaluated_candidate_count": len(evaluated_candidate_parameters),
        "best_candidate": best_result,
        "accepted_candidate": accepted_result,
        "production_current_candidate": production_result,
        "production_current_selection": production_current_selection,
        "current_updated": production_result is not None,
        "selection_status": selection_status,
        "interrupted_tuning": accepted_result is None,
        "optimization_gate": {
            "status": "passed" if accepted_result is not None else "failed",
            "candidate_count": len(leaderboard),
            "passed_candidate_count": sum(
                1 for row in leaderboard if bool(row["optimization_gate_passed"])
            ),
            "failed_candidate_count": sum(
                1 for row in leaderboard if not bool(row["optimization_gate_passed"])
            ),
            "thresholds": _optimization_gate_thresholds_payload(),
            "accepted_candidate": None if accepted_result is None else accepted_result["candidate"],
            "best_candidate": best_result["candidate"],
            "production_current_candidate": (
                None if production_result is None else production_result["candidate"]
            ),
        },
        "archived_previous_current_dir": str(archived_current_dir) if archived_current_dir else None,
        "archived_legacy_root_dirs": [str(path) for path in archived_legacy_root_dirs],
    }
    save_json(session_dir / "manifest.json", manifest)
    if production_result is not None:
        save_json(
            artifact_root / "best_params.json",
            {
                "updated_at": manifest["generated_at"],
                "source_path": str(csv_path),
                "parameters": {
                    key: accepted_candidate_config[key]
                    for key in (
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "hidden_dim",
                        "dropout",
                        "weight_decay",
                        "evaluation_state_reset_mode",
                        "min_policy_sigma",
                        "policy_cost_multiplier",
                        "policy_gamma_multiplier",
                        "q_max",
                        "cvar_weight",
                        "tie_policy_to_forecast_head",
                        "disable_overlay_branch",
                    )
                },
            },
        )
        current_source_payload = load_json(current_dir / "source.json")
        current_source_payload.update(
            build_current_alias_metadata(
                artifact_root,
                production_candidate_dir,
                selection_timestamp_utc=str(manifest["generated_at_utc"]),
            )
        )
        current_source_payload["sub_artifacts"] = _extend_current_sub_artifacts(
            current_source_payload,
            production_candidate_dir,
        )
        save_json(current_dir / "source.json", current_source_payload)
        generate_research_report(
            current_dir,
            report_path=artifact_root.parent.parent / "report_signalcascade_xauusd.md",
        )
        save_json(
            current_dir / "manifest.json",
            build_artifact_manifest(
                artifact_kind="training_run",
                artifact_id=str(current_source_payload["artifact_id"]),
                parent_artifact_id=(
                    None
                    if current_source_payload.get("parent_artifact_id") is None
                    else str(current_source_payload["parent_artifact_id"])
                ),
                generated_at_utc=str(current_source_payload["generated_at_utc"]),
                source_payload=current_source_payload,
                entrypoints=build_artifact_entrypoints(
                    current_source_payload,
                    include_model=True,
                ),
            ),
        )
    return manifest


def _resolve_session_candidates(
    candidate_parameters: list[dict[str, object]],
    *,
    candidate_limit: int | None,
    quick_mode: bool,
) -> tuple[list[dict[str, object]], int | None]:
    if candidate_limit is not None and int(candidate_limit) < 1:
        raise ValueError("candidate_limit must be >= 1")

    resolved_candidate_limit = (
        int(candidate_limit)
        if candidate_limit is not None
        else (
            min(len(candidate_parameters), _DEFAULT_QUICK_MODE_CANDIDATE_LIMIT)
            if quick_mode
            else None
        )
    )
    if resolved_candidate_limit is None:
        return list(candidate_parameters), None
    if not quick_mode:
        return list(candidate_parameters[:resolved_candidate_limit]), resolved_candidate_limit
    prioritized_candidates = _prioritize_quick_mode_candidates(candidate_parameters)
    return prioritized_candidates[:resolved_candidate_limit], resolved_candidate_limit


def _prioritize_quick_mode_candidates(
    candidate_parameters: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not candidate_parameters:
        return []

    prioritized: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    def add_candidate(candidate: dict[str, object]) -> None:
        key = tuple((key, candidate[key]) for key in sorted(candidate))
        if key in seen:
            return
        seen.add(key)
        prioritized.append(candidate)

    add_candidate(candidate_parameters[0])
    for target_tie, target_overlay_off in (
        (True, False),
        (False, True),
        (True, True),
    ):
        for candidate in candidate_parameters:
            if (
                bool(candidate.get("tie_policy_to_forecast_head")) == target_tie
                and bool(candidate.get("disable_overlay_branch")) == target_overlay_off
            ):
                add_candidate(candidate)
                break
    for candidate in candidate_parameters:
        add_candidate(candidate)
    return prioritized


def _write_run_artifacts(
    output_dir: Path,
    model,
    examples,
    config: TrainingConfig,
    source_payload: dict[str, object],
    base_bars,
    summary: dict[str, object],
    prediction,
    sample_count: int,
    source_rows_original: int,
    source_rows_used: int,
) -> dict[str, object]:
    artifact_source_payload = materialize_artifact_source(
        source_payload,
        output_dir,
        base_bars=base_bars,
    )
    metrics_payload = dict(summary)
    metrics_payload["sample_count"] = sample_count
    metrics_payload["effective_sample_count"] = int(summary.get("train_samples", 0)) + int(
        summary.get("validation_samples", 0)
    )
    metrics_payload["source"] = artifact_source_payload
    metrics_payload["schema_version"] = METRICS_SCHEMA_VERSION
    metrics_payload["source_rows_original"] = source_rows_original
    metrics_payload["source_rows_used"] = source_rows_used
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(output_dir / "prediction.json", serialize_prediction_result(prediction))
    save_json(
        output_dir / "forecast_summary.json",
        build_forecast_summary_payload(
            prediction=prediction,
            config=config,
            validation_metrics=summary.get("validation_metrics", {}),
            best_params={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
                "weight_decay": config.weight_decay,
                "evaluation_state_reset_mode": config.evaluation_state_reset_mode,
                "min_policy_sigma": config.min_policy_sigma,
                "policy_cost_multiplier": config.policy_cost_multiplier,
                "policy_gamma_multiplier": config.policy_gamma_multiplier,
                "q_max": config.q_max,
                "cvar_weight": config.cvar_weight,
                "tie_policy_to_forecast_head": config.tie_policy_to_forecast_head,
                "disable_overlay_branch": config.disable_overlay_branch,
            },
        ),
    )
    diagnostics_summary = export_review_diagnostics(
        output_dir=output_dir,
        model=model,
        examples=examples,
        config=config,
        checkpoint_audit=dict(summary.get("checkpoint_audit") or {}),
        source_payload=artifact_source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars,
    )
    save_json(
        output_dir / "source.json",
        build_artifact_source_payload(
            artifact_source_payload,
            output_dir,
            artifact_kind="training_run",
            sub_artifacts=build_subartifact_lineage(_training_sub_artifacts(include_report=False)),
        ),
    )
    return diagnostics_summary


def _training_sub_artifacts(*, include_report: bool) -> dict[str, str]:
    entries = {
        "config.json": "generated",
        "forecast_summary.json": "generated",
        "horizon_diag.csv": "generated",
        "metrics.json": "generated",
        "policy_summary.csv": "generated",
        "prediction.json": "generated",
        "source.json": "generated",
        "data_snapshot.csv": "generated",
        "validation_rows.csv": "generated",
        "validation_summary.json": "generated",
    }
    if include_report:
        entries["analysis.json"] = "regenerated"
        entries["manifest.json"] = "generated"
        entries["research_report.md"] = "regenerated"
    return entries


def _load_parameter_seed(artifact_root: Path) -> dict[str, object]:
    best_params_path = artifact_root / "best_params.json"
    if best_params_path.exists():
        payload = load_json(best_params_path)
        parameters = payload.get("parameters")
        if isinstance(parameters, dict):
            return _coerce_parameter_payload(parameters)

    current_config_path = artifact_root / "current" / "config.json"
    if current_config_path.exists():
        return _coerce_parameter_payload(load_json(current_config_path))

    candidate_configs = sorted(
        (
            path / "config.json"
            for path in artifact_root.iterdir()
            if path.is_dir()
            and path.name not in _RESERVED_ARTIFACT_DIRECTORIES
            and (path / "config.json").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for config_path in candidate_configs:
        if config_path.exists():
            return _coerce_parameter_payload(load_json(config_path))

    return dict(_FALLBACK_PARAMETERS)


def _coerce_parameter_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "epochs": int(payload.get("epochs", _FALLBACK_PARAMETERS["epochs"])),
        "batch_size": int(payload.get("batch_size", _FALLBACK_PARAMETERS["batch_size"])),
        "learning_rate": float(
            payload.get("learning_rate", _FALLBACK_PARAMETERS["learning_rate"])
        ),
        "hidden_dim": int(payload.get("hidden_dim", _FALLBACK_PARAMETERS["hidden_dim"])),
        "dropout": float(payload.get("dropout", _FALLBACK_PARAMETERS["dropout"])),
        "weight_decay": float(
            payload.get("weight_decay", _FALLBACK_PARAMETERS["weight_decay"])
        ),
        "evaluation_state_reset_mode": str(
            payload.get(
                "evaluation_state_reset_mode",
                _FALLBACK_PARAMETERS["evaluation_state_reset_mode"],
            )
        ),
        "min_policy_sigma": float(
            payload.get("min_policy_sigma", _FALLBACK_PARAMETERS["min_policy_sigma"])
        ),
        "policy_cost_multiplier": float(
            payload.get(
                "policy_cost_multiplier",
                _FALLBACK_PARAMETERS["policy_cost_multiplier"],
            )
        ),
        "policy_gamma_multiplier": float(
            payload.get(
                "policy_gamma_multiplier",
                _FALLBACK_PARAMETERS["policy_gamma_multiplier"],
            )
        ),
        "q_max": float(payload.get("q_max", _FALLBACK_PARAMETERS["q_max"])),
        "cvar_weight": float(payload.get("cvar_weight", _FALLBACK_PARAMETERS["cvar_weight"])),
        "tie_policy_to_forecast_head": bool(
            payload.get(
                "tie_policy_to_forecast_head",
                _FALLBACK_PARAMETERS["tie_policy_to_forecast_head"],
            )
        ),
        "disable_overlay_branch": bool(
            payload.get(
                "disable_overlay_branch",
                _FALLBACK_PARAMETERS["disable_overlay_branch"],
            )
        ),
    }


def _resolve_current_warm_start_seed(
    artifact_root: Path,
    *,
    warm_start_from_current: bool,
) -> tuple[Path | None, TrainingConfig | None]:
    if not warm_start_from_current:
        return None, None
    checkpoint_path = artifact_root / "current" / "model.pt"
    config_path = artifact_root / "current" / "config.json"
    if not checkpoint_path.exists() or not config_path.exists():
        return None, None
    try:
        return checkpoint_path, TrainingConfig.from_dict(load_json(config_path))
    except (KeyError, TypeError, ValueError):
        return None, None


def _supports_candidate_warm_start(
    source_config: TrainingConfig | None,
    target_config: TrainingConfig,
) -> bool:
    if source_config is None:
        return False
    comparable_fields = (
        "hidden_dim",
        "state_dim",
        "shape_classes",
        "branch_dilations",
        "tie_policy_to_forecast_head",
        "disable_overlay_branch",
        "horizons",
        "feature_contract_version",
        "timeframe_feature_names",
        "state_feature_names",
        "state_vector_component_names",
    )
    return all(
        getattr(source_config, field_name) == getattr(target_config, field_name)
        for field_name in comparable_fields
    )


def _extract_tunable_overrides(payload: dict[str, object]) -> dict[str, object]:
    tunable_keys = {
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
        "dropout",
        "weight_decay",
        "evaluation_state_reset_mode",
        "min_policy_sigma",
        "policy_cost_multiplier",
        "policy_gamma_multiplier",
        "q_max",
        "cvar_weight",
        "tie_policy_to_forecast_head",
        "disable_overlay_branch",
    }
    return {key: payload[key] for key in tunable_keys if key in payload}


def _build_candidate_parameters(previous: dict[str, object]) -> list[dict[str, object]]:
    base = _coerce_parameter_payload(previous)
    structural_candidates = [
        {
            **base,
            "tie_policy_to_forecast_head": False,
            "disable_overlay_branch": False,
        },
        {
            **base,
            "tie_policy_to_forecast_head": True,
            "disable_overlay_branch": False,
        },
        {
            **base,
            "tie_policy_to_forecast_head": False,
            "disable_overlay_branch": True,
        },
        {
            **base,
            "tie_policy_to_forecast_head": True,
            "disable_overlay_branch": True,
        },
    ]
    candidates = [
        *structural_candidates,
        {
            **base,
            "epochs": _clip_int(base["epochs"] + 2, minimum=6, maximum=24),
            "learning_rate": _round_float(base["learning_rate"] * 0.75),
        },
        {
            **base,
            "epochs": _clip_int(base["epochs"] + 4, minimum=6, maximum=24),
            "learning_rate": _round_float(base["learning_rate"] * 1.25),
        },
        {
            **base,
            "hidden_dim": _clip_hidden_dim(base["hidden_dim"] + 16),
            "learning_rate": _round_float(base["learning_rate"] * 0.85),
        },
        {
            **base,
            "hidden_dim": _clip_hidden_dim(base["hidden_dim"] - 16),
            "batch_size": _clip_batch_size(base["batch_size"] * 2),
        },
        {
            **base,
            "dropout": _clip_dropout(base["dropout"] + 0.05),
            "weight_decay": _round_float(base["weight_decay"] * 0.5),
        },
        {
            **base,
            "dropout": _clip_dropout(base["dropout"] - 0.05),
            "weight_decay": _round_float(base["weight_decay"] * 2.0),
        },
        {
            **base,
            "batch_size": _clip_batch_size(base["batch_size"] // 2),
            "epochs": _clip_int(base["epochs"] + 3, minimum=6, maximum=24),
        },
        {
            **base,
            "evaluation_state_reset_mode": _alternate_state_reset_mode(
                base["evaluation_state_reset_mode"]
            ),
        },
        {
            **base,
            "policy_cost_multiplier": _clip_policy_multiplier(
                base["policy_cost_multiplier"] * 0.5
            ),
        },
        {
            **base,
            "policy_cost_multiplier": _clip_policy_multiplier(
                base["policy_cost_multiplier"] * 2.0
            ),
        },
        {
            **base,
            "policy_gamma_multiplier": _clip_policy_multiplier(
                base["policy_gamma_multiplier"] * 0.5
            ),
        },
        {
            **base,
            "policy_gamma_multiplier": _clip_policy_multiplier(
                base["policy_gamma_multiplier"] * 2.0
            ),
        },
        {
            **base,
            "q_max": _clip_q_max(base["q_max"] + 0.25),
        },
        {
            **base,
            "q_max": _clip_q_max(base["q_max"] - 0.25),
        },
        {
            **base,
            "cvar_weight": _clip_cvar_weight(base["cvar_weight"] * 0.5),
        },
        {
            **base,
            "cvar_weight": _clip_cvar_weight(base["cvar_weight"] * 2.0),
        },
        {
            **base,
            "min_policy_sigma": _clip_min_policy_sigma(base["min_policy_sigma"] * 0.5),
        },
        {
            **base,
            "min_policy_sigma": _clip_min_policy_sigma(base["min_policy_sigma"] * 2.0),
        },
        dict(_FALLBACK_PARAMETERS),
    ]

    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        normalized = {
            "epochs": _clip_int(int(candidate["epochs"]), minimum=6, maximum=24),
            "batch_size": _clip_batch_size(int(candidate["batch_size"])),
            "learning_rate": _round_float(float(candidate["learning_rate"])),
            "hidden_dim": _clip_hidden_dim(int(candidate["hidden_dim"])),
            "dropout": _clip_dropout(float(candidate["dropout"])),
            "weight_decay": _round_float(float(candidate["weight_decay"])),
            "evaluation_state_reset_mode": str(candidate["evaluation_state_reset_mode"]),
            "min_policy_sigma": _clip_min_policy_sigma(float(candidate["min_policy_sigma"])),
            "policy_cost_multiplier": _clip_policy_multiplier(
                float(candidate["policy_cost_multiplier"])
            ),
            "policy_gamma_multiplier": _clip_policy_multiplier(
                float(candidate["policy_gamma_multiplier"])
            ),
            "q_max": _clip_q_max(float(candidate["q_max"])),
            "cvar_weight": _clip_cvar_weight(float(candidate["cvar_weight"])),
            "tie_policy_to_forecast_head": bool(candidate["tie_policy_to_forecast_head"]),
            "disable_overlay_branch": bool(candidate["disable_overlay_branch"]),
        }
        key = tuple(normalized[key] for key in sorted(normalized))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _extract_blocked_candidate_metrics(
    diagnostics_summary: dict[str, object],
    config: TrainingConfig,
) -> dict[str, object]:
    blocked_payload = diagnostics_summary.get("blocked_walk_forward_evaluation")
    if not isinstance(blocked_payload, dict):
        return {}

    state_reset_modes = blocked_payload.get("state_reset_modes")
    if not isinstance(state_reset_modes, dict):
        return {}

    mode_payload = state_reset_modes.get(config.evaluation_state_reset_mode)
    if not isinstance(mode_payload, dict):
        return {}

    metrics: dict[str, object] = {
        "blocked_state_reset_mode": config.evaluation_state_reset_mode,
        "blocked_best_state_reset_mode_by_mean_log_wealth": blocked_payload.get(
            "best_state_reset_mode_by_mean_log_wealth"
        ),
    }

    for source_key, target_key in (
        ("average_log_wealth_mean", "blocked_average_log_wealth_mean"),
        ("turnover_mean", "blocked_turnover_mean"),
        ("directional_accuracy_mean", "blocked_directional_accuracy_mean"),
        ("exact_smooth_position_mae_mean", "blocked_exact_smooth_position_mae_mean"),
        ("interval_1sigma_coverage_mean", "blocked_interval_1sigma_coverage_mean"),
        ("interval_2sigma_coverage_mean", "blocked_interval_2sigma_coverage_mean"),
        (
            "probabilistic_calibration_score_mean",
            "blocked_probabilistic_calibration_score_mean",
        ),
    ):
        value = _finite_float_or_none(mode_payload.get(source_key))
        if value is not None:
            metrics[target_key] = value

    folds = mode_payload.get("folds")
    if isinstance(folds, list):
        cvar_values = [
            float(fold["cvar_tail_loss"])
            for fold in folds
            if isinstance(fold, dict)
            and _finite_float_or_none(fold.get("cvar_tail_loss")) is not None
        ]
        if cvar_values:
            blocked_cvar_mean = sum(cvar_values) / len(cvar_values)
            metrics["blocked_cvar_tail_loss_mean"] = blocked_cvar_mean
            blocked_wealth = _finite_float_or_none(metrics.get("blocked_average_log_wealth_mean"))
            if blocked_wealth is not None:
                metrics["blocked_objective_log_wealth_minus_lambda_cvar_mean"] = (
                    blocked_wealth - (float(config.cvar_weight) * blocked_cvar_mean)
                )

    return metrics


def _extract_runtime_policy_alignment_metrics(
    diagnostics_summary: dict[str, object],
    config: TrainingConfig,
) -> dict[str, object]:
    policy_summary = diagnostics_summary.get("policy_calibration_summary")
    if not isinstance(policy_summary, dict):
        return {
            "runtime_policy_alignment_score": 1.0,
            "runtime_policy_selected_row_matches_applied": True,
        }

    selected_row = policy_summary.get("selected_row")
    selected_row_payload = dict(selected_row) if isinstance(selected_row, dict) else {}
    applied_runtime = policy_summary.get("applied_runtime_policy")
    applied_runtime_payload = (
        dict(applied_runtime) if isinstance(applied_runtime, dict) else _build_runtime_policy_payload(config)
    )
    selected_matches_runtime = bool(
        selected_row_payload
        and selected_row_payload.get("row_key") == applied_runtime_payload.get("row_key")
    )
    return {
        "runtime_policy_alignment_score": (
            1.0
            if selected_matches_runtime
            else _runtime_policy_alignment_score(
                selected_row_payload,
                applied_runtime_payload,
            )
        ),
        "runtime_policy_selected_row_matches_applied": selected_matches_runtime,
    }


def _extract_forecast_profile_metrics(prediction) -> dict[str, object]:
    current_close = _finite_float_or_none(getattr(prediction, "current_close", None))
    if current_close is None or current_close <= 0.0:
        return {}

    predicted_closes = getattr(prediction, "predicted_closes", None)
    if not isinstance(predicted_closes, dict):
        return {}

    horizon_returns: list[tuple[int, float]] = []
    for horizon, predicted_close in predicted_closes.items():
        predicted_close_value = _finite_float_or_none(predicted_close)
        if predicted_close_value is None:
            continue
        horizon_returns.append(
            (int(horizon), (predicted_close_value / current_close) - 1.0)
        )

    if not horizon_returns:
        return {}

    horizon_returns.sort(key=lambda item: item[0])
    max_abs_return = max(abs(expected_return) for _, expected_return in horizon_returns)
    jump_max = max(
        (
            abs(next_return - current_return)
            for (_, current_return), (_, next_return) in zip(
                horizon_returns,
                horizon_returns[1:],
            )
        ),
        default=0.0,
    )
    max_horizon = max(horizon for horizon, _ in horizon_returns)
    long_horizon_threshold = max(3, math.ceil(max_horizon * 0.4))
    long_horizon_abs_max = max(
        (
            abs(expected_return)
            for horizon, expected_return in horizon_returns
            if horizon >= long_horizon_threshold
        ),
        default=max_abs_return,
    )
    return {
        "forecast_return_pct_abs_max": max_abs_return,
        "forecast_return_pct_jump_max": jump_max,
        "forecast_long_horizon_return_pct_abs_max": long_horizon_abs_max,
    }


def _build_user_value_metrics(candidate: dict[str, object]) -> dict[str, object]:
    chart_fidelity_score = _build_chart_fidelity_score(candidate)
    sigma_band_score = _inverse_linear_score(
        candidate.get("sigma_calibration"),
        upper_bound=0.20,
        default=0.5,
    )
    execution_stability_score = _build_execution_stability_score(candidate)
    economic_score = _wealth_like_score(
        candidate.get("blocked_objective_log_wealth_minus_lambda_cvar_mean"),
        candidate.get("average_log_wealth"),
    )
    forecast_stability_score = _build_forecast_stability_score(candidate)
    base_user_value_score = (
        (0.55 * chart_fidelity_score)
        + (0.10 * sigma_band_score)
        + (0.20 * execution_stability_score)
        + (0.15 * economic_score)
    )
    user_value_score = (0.70 * base_user_value_score) + (0.30 * forecast_stability_score)
    return {
        "user_value_score": user_value_score,
        "user_value_base_score": base_user_value_score,
        "user_value_chart_fidelity_score": chart_fidelity_score,
        "user_value_sigma_band_score": sigma_band_score,
        "user_value_execution_stability_score": execution_stability_score,
        "user_value_economic_score": economic_score,
        "user_value_forecast_stability_score": forecast_stability_score,
    }


def _build_deployment_score_metrics(candidate: dict[str, object]) -> dict[str, object]:
    economic_core_score = _wealth_like_score(
        candidate.get("blocked_objective_log_wealth_minus_lambda_cvar_mean"),
        candidate.get("average_log_wealth"),
    )
    drawdown_score = _inverse_linear_score(
        candidate.get("max_drawdown"),
        upper_bound=0.15,
        default=0.5,
    )
    cvar_tail_score = _inverse_linear_score(
        candidate.get("blocked_cvar_tail_loss_mean"),
        upper_bound=0.10,
        default=0.5,
    )
    turnover_score = _inverse_linear_score(
        candidate.get("blocked_turnover_mean"),
        upper_bound=2.0,
        default=0.0,
    )
    sigma_score = _inverse_linear_score(
        candidate.get("sigma_calibration"),
        upper_bound=0.20,
        default=0.5,
    )
    probabilistic_score = _clamp01(
        _metric_with_fallback(
            candidate,
            primary_key="blocked_probabilistic_calibration_score_mean",
            fallback_key="probabilistic_calibration_score",
            default=0.5,
        )
    )
    runtime_alignment_score = _clamp01(
        _metric_with_fallback(
            candidate,
            primary_key="runtime_policy_alignment_score",
            fallback_key="runtime_policy_alignment_score",
            default=1.0,
        )
    )
    deployment_score = (
        (0.30 * economic_core_score)
        + (0.22 * drawdown_score)
        + (0.13 * cvar_tail_score)
        + (0.12 * turnover_score)
        + (0.10 * sigma_score)
        + (0.08 * probabilistic_score)
        + (0.05 * runtime_alignment_score)
    )
    return {
        "deployment_economic_core_score": economic_core_score,
        "deployment_drawdown_score": drawdown_score,
        "deployment_cvar_tail_score": cvar_tail_score,
        "deployment_turnover_score": turnover_score,
        "deployment_sigma_score": sigma_score,
        "deployment_probabilistic_score": probabilistic_score,
        "deployment_runtime_alignment_score": runtime_alignment_score,
        "deployment_score": deployment_score,
    }


def _build_chart_fidelity_score(candidate: dict[str, object]) -> float:
    directional_score = _clamp01(
        _metric_with_fallback(
            candidate,
            primary_key="blocked_directional_accuracy_mean",
            fallback_key="directional_accuracy",
            default=0.5,
        )
    )
    mu_fit_score = _inverse_linear_score(
        candidate.get("mu_calibration"),
        upper_bound=0.20,
        default=0.5,
    )
    position_consistency_score = _inverse_linear_score(
        _metric_with_fallback(
            candidate,
            primary_key="blocked_exact_smooth_position_mae_mean",
            fallback_key="exact_smooth_position_mae",
            default=0.25,
        ),
        upper_bound=0.25,
        default=0.5,
    )
    probabilistic_score = _clamp01(
        _metric_with_fallback(
            candidate,
            primary_key="blocked_probabilistic_calibration_score_mean",
            fallback_key="probabilistic_calibration_score",
            default=0.5,
        )
    )
    return (
        (0.45 * directional_score)
        + (0.15 * mu_fit_score)
        + (0.25 * position_consistency_score)
        + (0.15 * probabilistic_score)
    )


def _build_execution_stability_score(candidate: dict[str, object]) -> float:
    drawdown_score = _inverse_linear_score(
        candidate.get("max_drawdown"),
        upper_bound=0.15,
        default=0.0,
    )
    turnover_score = _inverse_linear_score(
        candidate.get("blocked_turnover_mean"),
        upper_bound=2.0,
        default=0.0,
    )
    return (0.60 * drawdown_score) + (0.40 * turnover_score)


def _build_forecast_stability_score(candidate: dict[str, object]) -> float:
    jump_score = _inverse_linear_score(
        candidate.get("forecast_return_pct_jump_max"),
        upper_bound=0.16,
        default=0.5,
    )
    long_horizon_score = _inverse_linear_score(
        candidate.get("forecast_long_horizon_return_pct_abs_max"),
        upper_bound=0.12,
        default=0.5,
    )
    return (0.60 * jump_score) + (0.40 * long_horizon_score)


def _select_production_current_candidate(
    leaderboard: list[dict[str, object]],
) -> dict[str, object] | None:
    passed_rows = [row for row in leaderboard if bool(row.get("optimization_gate_passed"))]
    if not passed_rows:
        return None
    return min(
        passed_rows,
        key=lambda row: (
            -_metric_with_fallback(
                row,
                primary_key="deployment_score",
                fallback_key="user_value_score",
                default=0.0,
            ),
            _ascending_metric(row, "blocked_exact_smooth_position_mae_mean"),
            _ascending_metric(row, "max_drawdown"),
            _descending_metric(
                row,
                "blocked_objective_log_wealth_minus_lambda_cvar_mean",
                "average_log_wealth",
            ),
            _ascending_metric(row, "blocked_turnover_mean"),
            _ascending_metric(row, "blocked_cvar_tail_loss_mean"),
            -_metric_with_fallback(
                row,
                primary_key="deployment_economic_core_score",
                fallback_key="user_value_economic_score",
                default=0.0,
            ),
            str(row.get("candidate", "")),
        ),
    )


def _build_production_current_selection_payload(
    *,
    accepted_result: dict[str, object] | None,
    production_result: dict[str, object] | None,
    quick_mode: bool,
) -> dict[str, object]:
    if production_result is None:
        if accepted_result is None:
            return {
                "selection_mode": "no_accepted_candidate",
                "selection_rule": _PRODUCTION_CURRENT_SELECTION_RULE,
                "selection_rule_version": _PRODUCTION_CURRENT_SELECTION_RULE_VERSION,
                "selection_status": "no_candidate_passed_gate",
                "decision_summary": (
                    "production current was not updated because no candidate passed the "
                    "optimization gate."
                ),
                "override_reason": None,
                "override_priority_metrics": [],
            }
        return {
            "selection_mode": "quick_mode_non_promotable",
            "selection_rule": _PRODUCTION_CURRENT_SELECTION_RULE,
            "selection_rule_version": _PRODUCTION_CURRENT_SELECTION_RULE_VERSION,
            "selection_status": "quick_mode_non_promotable",
            "decision_summary": (
                "production current was not updated because this run was quick-mode exploration."
            ),
            "override_reason": None,
            "override_priority_metrics": [],
        }
    if accepted_result is None or production_result.get("candidate") == accepted_result.get("candidate"):
        return {
            "selection_mode": "accepted_candidate",
            "selection_rule": _PRODUCTION_CURRENT_SELECTION_RULE,
            "selection_rule_version": _PRODUCTION_CURRENT_SELECTION_RULE_VERSION,
            "selection_status": (
                "accepted_and_production_same" if not quick_mode else "accepted_and_production_same"
            ),
            "decision_summary": (
                "production current matches the accepted candidate after deployment-score selection."
            ),
            "override_reason": None,
            "override_priority_metrics": [],
        }
    return {
        "selection_mode": "deployment_score_override",
        "selection_rule": _PRODUCTION_CURRENT_SELECTION_RULE,
        "selection_rule_version": _PRODUCTION_CURRENT_SELECTION_RULE_VERSION,
        "selection_status": "accepted_and_production_diverged",
        "decision_summary": (
            "production current differs from the accepted candidate because deployment score took "
            "priority over accepted rank."
        ),
        "override_reason": (
            "production current prioritizes deployment score over blocked-objective winner."
        ),
        "override_priority_metrics": list(_PRODUCTION_CURRENT_PRIORITY_METRICS),
    }


def _archive_existing_current(current_dir: Path, session_dir: Path) -> Path | None:
    if not current_dir.exists():
        return None
    if current_dir.is_dir() and not any(current_dir.iterdir()):
        shutil.rmtree(current_dir)
        return None

    archived_dir = session_dir / "previous_current"
    if archived_dir.exists():
        shutil.rmtree(archived_dir)
    shutil.move(str(current_dir), archived_dir)
    return archived_dir


def _extend_current_sub_artifacts(
    current_source_payload: dict[str, object],
    source_artifact_dir: Path,
) -> dict[str, dict[str, object]]:
    existing_payload = current_source_payload.get("sub_artifacts")
    existing = dict(existing_payload) if isinstance(existing_payload, dict) else {}
    source_artifact_id = current_source_payload.get("artifact_id")
    for name, materialization in (
        ("analysis.json", "regenerated"),
        ("manifest.json", "generated"),
        ("research_report.md", "regenerated"),
    ):
        entry = dict(existing.get(name, {})) if isinstance(existing.get(name), dict) else {}
        entry["materialization"] = materialization
        entry["source_artifact_dir"] = str(source_artifact_dir)
        if source_artifact_id is not None:
            entry["source_artifact_id"] = str(source_artifact_id)
        existing[name] = entry
    return existing


def _archive_legacy_root_runs(artifact_root: Path, session_dir: Path) -> list[Path]:
    archived_paths: list[Path] = []
    legacy_root = ensure_directory(session_dir / "legacy_root_runs")

    for path in sorted(artifact_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in _RESERVED_ARTIFACT_DIRECTORIES:
            continue
        destination = legacy_root / path.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(path), destination)
        archived_paths.append(destination)

    if not any(legacy_root.iterdir()):
        legacy_root.rmdir()
    return archived_paths


def _clip_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _clip_batch_size(value: int) -> int:
    candidates = (8, 16, 32, 64)
    return min(candidates, key=lambda candidate: abs(candidate - max(8, value)))


def _clip_hidden_dim(value: int) -> int:
    hidden = max(32, min(96, value))
    return int(round(hidden / 16) * 16)


def _clip_dropout(value: float) -> float:
    return round(max(0.05, min(0.25, value)), 2)


def _clip_min_policy_sigma(value: float) -> float:
    candidates = (5e-5, 1e-4, 2e-4, 4e-4)
    return min(candidates, key=lambda candidate: abs(candidate - max(5e-5, value)))


def _clip_policy_multiplier(value: float) -> float:
    candidates = (0.5, 1.0, 2.0, 4.0, 6.0)
    return min(candidates, key=lambda candidate: abs(candidate - max(0.5, value)))


def _clip_q_max(value: float) -> float:
    candidates = (0.5, 0.75, 1.0, 1.25, 1.5)
    return min(candidates, key=lambda candidate: abs(candidate - max(0.5, value)))


def _clip_cvar_weight(value: float) -> float:
    candidates = (0.05, 0.1, 0.2, 0.4, 0.8)
    return min(candidates, key=lambda candidate: abs(candidate - max(0.05, value)))


def _alternate_state_reset_mode(value: str) -> str:
    return (
        "reset_each_session_or_window"
        if str(value) == "carry_on"
        else "carry_on"
    )


def _round_float(value: float) -> float:
    return float(f"{value:.6g}")


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _build_runtime_policy_payload(config: TrainingConfig) -> dict[str, object]:
    return {
        "row_key": (
            "state_reset_mode="
            f"{config.evaluation_state_reset_mode}"
            f"|cost_multiplier={float(config.policy_cost_multiplier):.12g}"
            f"|gamma_multiplier={float(config.policy_gamma_multiplier):.12g}"
            f"|min_policy_sigma={float(config.min_policy_sigma):.12g}"
            f"|q_max={float(config.q_max):.12g}"
            f"|cvar_weight={float(config.cvar_weight):.12g}"
        ),
        "state_reset_mode": config.evaluation_state_reset_mode,
        "cost_multiplier": float(config.policy_cost_multiplier),
        "gamma_multiplier": float(config.policy_gamma_multiplier),
        "min_policy_sigma": float(config.min_policy_sigma),
        "q_max": float(config.q_max),
        "cvar_weight": float(config.cvar_weight),
    }


def _runtime_policy_alignment_score(
    selected_row: dict[str, object],
    applied_runtime_policy: dict[str, object],
) -> float:
    if not selected_row:
        return 0.0
    state_reset_score = float(
        str(selected_row.get("state_reset_mode"))
        == str(applied_runtime_policy.get("state_reset_mode"))
    )
    return _clamp01(
        (0.15 * state_reset_score)
        + (
            0.35
            * _ratio_alignment_score(
                selected_row.get("cost_multiplier"),
                applied_runtime_policy.get("cost_multiplier"),
                max_ratio=12.0,
            )
        )
        + (
            0.15
            * _ratio_alignment_score(
                selected_row.get("gamma_multiplier"),
                applied_runtime_policy.get("gamma_multiplier"),
                max_ratio=12.0,
            )
        )
        + (
            0.15
            * _ratio_alignment_score(
                selected_row.get("min_policy_sigma"),
                applied_runtime_policy.get("min_policy_sigma"),
                max_ratio=8.0,
            )
        )
        + (
            0.10
            * _ratio_alignment_score(
                selected_row.get("q_max"),
                applied_runtime_policy.get("q_max"),
                max_ratio=3.0,
            )
        )
        + (
            0.10
            * _ratio_alignment_score(
                selected_row.get("cvar_weight"),
                applied_runtime_policy.get("cvar_weight"),
                max_ratio=16.0,
            )
        )
    )


def _ratio_alignment_score(
    selected_value: object,
    applied_value: object,
    *,
    max_ratio: float,
) -> float:
    selected_numeric = _finite_float_or_none(selected_value)
    applied_numeric = _finite_float_or_none(applied_value)
    if (
        selected_numeric is None
        or applied_numeric is None
        or selected_numeric <= 0.0
        or applied_numeric <= 0.0
        or max_ratio <= 1.0
    ):
        return 0.0
    log_ratio = abs(math.log(selected_numeric / applied_numeric))
    return _clamp01(1.0 - (log_ratio / math.log(max_ratio)))


def _metric_with_fallback(
    row: dict[str, object],
    *,
    primary_key: str,
    fallback_key: str,
    default: float,
) -> float:
    primary_value = _finite_float_or_none(row.get(primary_key))
    if primary_value is not None:
        return primary_value
    fallback_value = _finite_float_or_none(row.get(fallback_key))
    if fallback_value is not None:
        return fallback_value
    return float(default)


def _inverse_linear_score(
    value: object,
    *,
    upper_bound: float,
    default: float,
) -> float:
    numeric_value = _finite_float_or_none(value)
    if numeric_value is None:
        return _clamp01(default)
    if upper_bound <= 0.0:
        return 0.0
    return _clamp01(1.0 - (numeric_value / upper_bound))


def _wealth_like_score(primary_value: object, fallback_value: object) -> float:
    value = _finite_float_or_none(primary_value)
    if value is None:
        value = _finite_float_or_none(fallback_value)
    if value is None:
        return 0.0
    return _clamp01(0.5 + (value * 10.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _descending_metric(
    row: dict[str, object],
    primary_key: str,
    fallback_key: str,
) -> float:
    primary_value = _finite_float_or_none(row.get(primary_key))
    if primary_value is not None:
        return -primary_value

    fallback_value = _finite_float_or_none(row.get(fallback_key))
    if fallback_value is not None:
        return -fallback_value

    return math.inf


def _ascending_metric(
    row: dict[str, object],
    primary_key: str,
) -> float:
    primary_value = _finite_float_or_none(row.get(primary_key))
    if primary_value is not None:
        return primary_value
    return math.inf


def _evaluate_optimization_gate(candidate: dict[str, object]) -> dict[str, object]:
    failed_rules: list[str] = []

    for rule in _OPTIMIZATION_GATE_RULES:
        metric = str(rule["metric"])
        operator = str(rule["operator"])
        threshold = float(rule["threshold"])
        numeric_value = _resolve_gate_metric_value(candidate, metric)

        if not math.isfinite(numeric_value):
            failed_rules.append(f"{metric}:missing")
            continue
        if operator == "minimum" and numeric_value < threshold:
            failed_rules.append(f"{metric}<{threshold:.6f}")
            continue
        if operator == "maximum" and numeric_value > threshold:
            failed_rules.append(f"{metric}>{threshold:.6f}")
            continue

    passed = not failed_rules
    return {
        "optimization_gate_status": "passed" if passed else "failed",
        "optimization_gate_passed": passed,
        "optimization_gate_failed_rules": failed_rules,
    }


def _resolve_gate_metric_value(candidate: dict[str, object], metric: str) -> float:
    direct_value = _finite_float_or_none(candidate.get(metric))
    if direct_value is not None:
        return direct_value

    if metric == "blocked_objective_log_wealth_minus_lambda_cvar_mean":
        average_log_wealth = _finite_float_or_none(candidate.get("average_log_wealth"))
        cvar_tail_loss = _finite_float_or_none(candidate.get("cvar_tail_loss"))
        cvar_weight = _finite_float_or_none(candidate.get("cvar_weight"))
        if (
            average_log_wealth is not None
            and cvar_tail_loss is not None
            and cvar_weight is not None
        ):
            return average_log_wealth - (cvar_weight * cvar_tail_loss)
    if metric == "blocked_directional_accuracy_mean":
        fallback = _finite_float_or_none(candidate.get("directional_accuracy"))
        if fallback is not None:
            return fallback
    if metric == "blocked_exact_smooth_position_mae_mean":
        fallback = _finite_float_or_none(candidate.get("exact_smooth_position_mae"))
        if fallback is not None:
            return fallback
        return 0.0
    if metric == "runtime_policy_alignment_score":
        return 1.0
    return math.nan


def _optimization_gate_thresholds_payload() -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for rule in _OPTIMIZATION_GATE_RULES:
        metric = str(rule["metric"])
        operator = str(rule["operator"])
        threshold = float(rule["threshold"])
        payload[metric] = {"minimum" if operator == "minimum" else "maximum": threshold}
    return payload
