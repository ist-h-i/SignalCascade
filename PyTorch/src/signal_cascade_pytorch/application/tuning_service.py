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
from .config import TrainingConfig
from .dataset_service import (
    build_latest_inference_example_from_bars,
    build_training_examples_from_bars,
    limit_base_bars_to_lookback_days,
    trim_base_bars_for_latest_inference,
)
from .inference_service import (
    build_forecast_summary_payload,
    predict_from_example,
    serialize_prediction_result,
)
from .report_service import METRICS_SCHEMA_VERSION, generate_research_report
from .training_service import train_model
from ..infrastructure.data.csv_source import CsvMarketDataSource
from ..infrastructure.persistence import ensure_directory, load_json, save_json

_RESERVED_ARTIFACT_DIRECTORIES = {"archive", "current", "live"}
_FALLBACK_PARAMETERS = {
    "epochs": 12,
    "batch_size": 16,
    "learning_rate": 5e-4,
    "hidden_dim": 48,
    "dropout": 0.1,
    "weight_decay": 5e-5,
}
_OPTIMIZATION_GATE_RULES = (
    {"metric": "average_log_wealth", "operator": "minimum", "threshold": 0.0},
    {"metric": "realized_pnl_per_anchor", "operator": "minimum", "threshold": 0.0},
    {"metric": "cvar_tail_loss", "operator": "maximum", "threshold": 0.08},
    {"metric": "max_drawdown", "operator": "maximum", "threshold": 0.15},
    {"metric": "directional_accuracy", "operator": "minimum", "threshold": 0.50},
    {"metric": "no_trade_band_hit_rate", "operator": "maximum", "threshold": 0.80},
)


def tune_latest_dataset(
    csv_path: Path,
    artifact_root: Path,
    seed: int = 7,
    config_overrides: dict[str, object] | None = None,
    lookback_days: int | None = None,
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
    examples = build_training_examples_from_bars(base_bars, baseline_config)
    inherited_parameters = _load_parameter_seed(artifact_root)
    inherited_parameters.update(tunable_overrides)
    candidate_parameters = _build_candidate_parameters(inherited_parameters)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = ensure_directory(archive_root / f"session_{session_id}")
    source_payload = {"kind": "csv", "path": str(csv_path)}
    if lookback_days is not None:
        source_payload["lookback_days"] = int(lookback_days)
    leaderboard: list[dict[str, object]] = []

    for index, parameters in enumerate(candidate_parameters, start=1):
        candidate_name = f"candidate_{index:02d}"
        candidate_dir = ensure_directory(session_dir / candidate_name)
        config = TrainingConfig(seed=seed, output_dir=str(candidate_dir), **static_overrides, **parameters)
        model, summary = train_model(examples, config, candidate_dir)
        latest_example = _build_latest_example(base_bars, config)
        prediction = predict_from_example(model, latest_example, config)
        _write_run_artifacts(
            output_dir=candidate_dir,
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
        result_row.update(_evaluate_optimization_gate(result_row))
        leaderboard.append(result_row)

    leaderboard.sort(
        key=lambda row: (
            not bool(row["optimization_gate_passed"]),
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
    save_json(session_dir / "leaderboard.json", {"results": leaderboard})

    best_result = leaderboard[0]
    accepted_result = next(
        (row for row in leaderboard if bool(row["optimization_gate_passed"])),
        None,
    )
    archived_current_dir = None
    archived_legacy_root_dirs: list[Path] = []

    if accepted_result is not None:
        accepted_candidate_dir = session_dir / str(accepted_result["candidate"])
        archived_current_dir = _archive_existing_current(current_dir, session_dir)
        archived_legacy_root_dirs = _archive_legacy_root_runs(artifact_root, session_dir)

        if current_dir.exists():
            shutil.rmtree(current_dir)
        shutil.copytree(accepted_candidate_dir, current_dir)

    current_dir_payload = str(current_dir) if current_dir.exists() or accepted_result is not None else None

    manifest = {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "current_dir": current_dir_payload,
        "archive_session_dir": str(session_dir),
        "leaderboard_path": str(session_dir / "leaderboard.json"),
        "manifest_path": str(session_dir / "manifest.json"),
        "source_path": str(csv_path),
        "lookback_days": lookback_days,
        "sample_count": len(examples),
        "source_rows_original": source_rows_original,
        "source_rows_used": source_rows_used,
        "best_candidate": best_result,
        "accepted_candidate": accepted_result,
        "current_updated": accepted_result is not None,
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
        },
        "archived_previous_current_dir": str(archived_current_dir) if archived_current_dir else None,
        "archived_legacy_root_dirs": [str(path) for path in archived_legacy_root_dirs],
    }
    save_json(session_dir / "manifest.json", manifest)
    if accepted_result is not None:
        save_json(current_dir / "manifest.json", manifest)
        save_json(
            artifact_root / "best_params.json",
            {
                "updated_at": manifest["generated_at"],
                "source_path": str(csv_path),
                "parameters": {
                    key: accepted_result[key]
                    for key in (
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "hidden_dim",
                        "dropout",
                        "weight_decay",
                    )
                },
            },
        )
        save_json(
            current_dir / "source.json",
            build_artifact_source_payload(
                materialize_artifact_source(source_payload, current_dir, base_bars=base_bars),
                current_dir,
                artifact_kind="training_run",
                generated_at_utc=manifest["generated_at"],
                sub_artifacts=build_subartifact_lineage(_training_sub_artifacts(include_report=True)),
            ),
        )
        generate_research_report(
            current_dir,
            report_path=artifact_root.parent.parent / "report_signalcascade_xauusd.md",
        )
        save_json(
            current_dir / "source.json",
            build_artifact_source_payload(
                materialize_artifact_source(source_payload, current_dir, base_bars=base_bars),
                current_dir,
                artifact_kind="training_run",
                generated_at_utc=manifest["generated_at"],
                sub_artifacts=build_subartifact_lineage(_training_sub_artifacts(include_report=True)),
            ),
        )
    return manifest


def _build_latest_example(base_bars, config: TrainingConfig):
    trimmed_bars = trim_base_bars_for_latest_inference(base_bars, config)
    return build_latest_inference_example_from_bars(trimmed_bars, config)


def _write_run_artifacts(
    output_dir: Path,
    config: TrainingConfig,
    source_payload: dict[str, object],
    base_bars,
    summary: dict[str, object],
    prediction,
    sample_count: int,
    source_rows_original: int,
    source_rows_used: int,
) -> None:
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
            },
        ),
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


def _training_sub_artifacts(*, include_report: bool) -> dict[str, str]:
    entries = {
        "config.json": "generated",
        "forecast_summary.json": "generated",
        "metrics.json": "generated",
        "prediction.json": "generated",
        "source.json": "generated",
        "data_snapshot.csv": "generated",
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
    }


def _extract_tunable_overrides(payload: dict[str, object]) -> dict[str, object]:
    tunable_keys = {
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
        "dropout",
        "weight_decay",
    }
    return {key: payload[key] for key in tunable_keys if key in payload}


def _build_candidate_parameters(previous: dict[str, object]) -> list[dict[str, object]]:
    base = _coerce_parameter_payload(previous)
    candidates = [
        base,
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
        }
        key = tuple(normalized.values())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


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


def _round_float(value: float) -> float:
    return float(f"{value:.6g}")


def _evaluate_optimization_gate(candidate: dict[str, object]) -> dict[str, object]:
    failed_rules: list[str] = []

    for rule in _OPTIMIZATION_GATE_RULES:
        metric = str(rule["metric"])
        operator = str(rule["operator"])
        threshold = float(rule["threshold"])
        value = candidate.get(metric)
        numeric_value = float(value) if value is not None else math.nan

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


def _optimization_gate_thresholds_payload() -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for rule in _OPTIMIZATION_GATE_RULES:
        metric = str(rule["metric"])
        operator = str(rule["operator"])
        threshold = float(rule["threshold"])
        payload[metric] = {"minimum" if operator == "minimum" else "maximum": threshold}
    return payload
