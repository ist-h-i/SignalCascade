from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from tempfile import mkdtemp
from uuid import uuid4

from .application.artifact_provenance import (
    build_artifact_source_payload,
    build_subartifact_lineage,
    materialize_artifact_source,
)
from .application.artifact_manifest import build_artifact_entrypoints, build_artifact_manifest
from .application.config import TrainingConfig
from .application.diagnostics_service import export_review_diagnostics
from .application.dataset_service import (
    build_latest_inference_example,
    build_latest_inference_example_from_bars,
    build_training_examples,
    build_training_examples_from_bars,
    limit_base_bars_to_lookback_days,
)
from .application.inference_service import (
    build_forecast_summary_payload,
    predict_latest,
    serialize_prediction_result,
)
from .application.price_scale import (
    normalize_price_scale_payload,
    price_scale_manifest_fields,
    resolve_effective_price_scale,
)
from .application.report_service import (
    METRICS_SCHEMA_VERSION,
    generate_research_report,
    load_required_diagnostics_summary,
)
from .application.training_service import train_model
from .application.tuning_service import tune_latest_dataset
from .infrastructure.data.csv_source import CsvMarketDataSource
from .infrastructure.data.synthetic_source import SyntheticMarketDataSource
from .infrastructure.ml.model import SignalCascadeModel
from .infrastructure.persistence import ensure_directory, load_checkpoint, load_json, save_json

def train_command(args) -> int:
    _emit_cli_compat_warnings(args)
    config = _build_config(args)
    output_dir = ensure_directory(Path(config.output_dir))
    source_payload = _build_source_payload(args, config)
    price_scale = _resolve_price_scale(source_payload)
    source = _create_data_source(source_payload)
    source_rows_original = None
    source_rows_used = None
    base_bars = None

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        source_rows_original = len(base_bars)
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        source_rows_used = len(base_bars)
        examples = build_training_examples_from_bars(base_bars, config, price_scale=price_scale)
        latest_inference_example = (
            build_latest_inference_example_from_bars(
                base_bars,
                config,
                price_scale=price_scale,
            )
            if len(base_bars) >= 512
            else None
        )
    else:
        examples = build_training_examples(source, config, price_scale=price_scale)
        latest_inference_example = build_latest_inference_example(
            source,
            config,
            price_scale=price_scale,
        )
    artifact_source_payload = materialize_artifact_source(
        source_payload,
        output_dir,
        base_bars=base_bars if source_payload["kind"] == "csv" else None,
    )
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    model, summary = train_model(examples, config, output_dir)
    prediction = predict_latest(
        model=model,
        examples=examples,
        config=config,
        previous_position=float(getattr(args, "previous_position", 0.0) or 0.0),
        latest_example=latest_inference_example,
    )

    summary["sample_count"] = len(examples)
    summary["effective_sample_count"] = summary["train_samples"] + summary["validation_samples"]
    summary["source"] = artifact_source_payload
    summary["schema_version"] = METRICS_SCHEMA_VERSION
    if source_rows_original is not None and source_rows_used is not None:
        summary["source_rows_original"] = source_rows_original
        summary["source_rows_used"] = source_rows_used
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "prediction.json", serialize_prediction_result(prediction))
    _save_forecast_summary(output_dir, prediction, config)
    export_review_diagnostics(
        output_dir=output_dir,
        model=model,
        examples=examples,
        config=config,
        source_payload=artifact_source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars if source_payload["kind"] == "csv" else None,
    )
    train_sub_artifacts = _training_sub_artifacts(artifact_source_payload)
    training_source_payload = build_artifact_source_payload(
        artifact_source_payload,
        output_dir,
        artifact_kind="training_run",
        generated_at_utc=generated_at_utc,
        sub_artifacts=build_subartifact_lineage(train_sub_artifacts),
    )
    save_json(output_dir / "source.json", training_source_payload)
    save_json(
        output_dir / "manifest.json",
        _build_artifact_manifest(
            artifact_kind="training_run",
            artifact_id=str(training_source_payload["artifact_id"]),
            parent_artifact_id=None,
            generated_at_utc=generated_at_utc,
            source_payload=artifact_source_payload,
            entrypoints=_build_artifact_entrypoints(
                artifact_source_payload,
                include_model=True,
            ),
        ),
    )
    generate_research_report(output_dir)
    save_json(
        output_dir / "source.json",
        build_artifact_source_payload(
            artifact_source_payload,
            output_dir,
            artifact_kind="training_run",
            generated_at_utc=generated_at_utc,
            sub_artifacts=build_subartifact_lineage(train_sub_artifacts),
        ),
    )

    print(f"trained samples: {summary['train_samples']}")
    print(f"validation samples: {summary['validation_samples']}")
    print(f"best validation loss: {summary['best_validation_loss']:.6f}")
    print(
        "validation average_log_wealth: "
        f"{summary['validation_metrics']['average_log_wealth']:.4f}"
    )
    print(
        "validation realized_pnl_per_anchor: "
        f"{summary['validation_metrics']['realized_pnl_per_anchor']:.4f}"
    )
    print(
        "validation cvar_tail_loss: "
        f"{summary['validation_metrics']['cvar_tail_loss']:.4f}"
    )
    print(f"latest policy horizon: {prediction.policy_horizon}")
    print(f"latest position: {prediction.position:.4f}")
    print(f"latest g_t: {prediction.tradeability_gate:.4f}")
    print(f"latest selected policy utility: {prediction.policy_score:.6f}")
    print(f"latest no-trade-band hit: {prediction.no_trade_band_hit}")
    return 0


def predict_command(args) -> int:
    _emit_cli_compat_warnings(args)
    output_dir = Path(args.output_dir)
    config = _load_config_with_overrides(output_dir, args)
    source_payload = _resolve_source_payload(args, output_dir)
    price_scale = _resolve_price_scale(source_payload)
    source = _create_data_source(source_payload)

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        examples = build_training_examples_from_bars(base_bars, config, price_scale=price_scale)
        latest_inference_example = (
            build_latest_inference_example_from_bars(
                base_bars,
                config,
                price_scale=price_scale,
            )
            if len(base_bars) >= 512
            else None
        )
    else:
        examples = build_training_examples(source, config, price_scale=price_scale)
        latest_inference_example = build_latest_inference_example(
            source,
            config,
            price_scale=price_scale,
        )

    model = _build_model_from_example(examples[0], config)
    load_checkpoint(output_dir / "model.pt", model)
    prediction = predict_latest(
        model=model,
        examples=examples,
        config=config,
        previous_position=float(getattr(args, "previous_position", 0.0) or 0.0),
        latest_example=latest_inference_example,
    )
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "prediction.json", serialize_prediction_result(prediction))
    _save_forecast_summary(output_dir, prediction, config)
    _merge_artifact_price_scale_metadata(output_dir, source_payload)

    print(f"policy horizon: {prediction.policy_horizon}")
    print(f"executed horizon: {prediction.executed_horizon}")
    print(f"position: {prediction.position:.4f}")
    print(f"trade delta: {prediction.trade_delta:.4f}")
    print(f"g_t: {prediction.tradeability_gate:.4f}")
    print(f"selected policy utility: {prediction.policy_score:.6f}")
    print(f"no-trade-band hit: {prediction.no_trade_band_hit}")
    return 0


def export_diagnostics_command(args) -> int:
    _emit_cli_compat_warnings(args)
    artifact_dir = Path(args.output_dir).expanduser().resolve()
    diagnostics_output_dir = _resolve_diagnostics_output_dir(
        artifact_dir,
        getattr(args, "diagnostics_output_dir", None),
    )
    config = _load_config_with_overrides(artifact_dir, args)
    source_payload = _resolve_source_payload(args, artifact_dir)
    price_scale = _resolve_price_scale(source_payload)
    source = _create_data_source(source_payload)
    source_rows_original = None
    source_rows_used = None
    base_bars = None

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        source_rows_original = len(base_bars)
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        source_rows_used = len(base_bars)
        examples = build_training_examples_from_bars(base_bars, config, price_scale=price_scale)
    else:
        examples = build_training_examples(source, config, price_scale=price_scale)
    artifact_source_payload = materialize_artifact_source(
        source_payload,
        diagnostics_output_dir,
        base_bars=base_bars if source_payload["kind"] == "csv" else None,
    )

    model = _build_model_from_example(examples[0], config)
    load_checkpoint(artifact_dir / "model.pt", model)
    summary = export_review_diagnostics(
        output_dir=diagnostics_output_dir,
        model=model,
        examples=examples,
        config=config,
        source_payload=artifact_source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars,
    )
    _materialize_replay_artifact(
        artifact_dir=artifact_dir,
        diagnostics_output_dir=diagnostics_output_dir,
        config=config,
        source_payload=artifact_source_payload,
        validation_summary=summary,
    )

    print(f"artifact dir: {artifact_dir}")
    print(f"diagnostics overlay dir: {diagnostics_output_dir}")
    print(f"validation rows: {diagnostics_output_dir / 'validation_rows.csv'}")
    print(f"policy summary: {diagnostics_output_dir / 'policy_summary.csv'}")
    print(f"horizon diagnostics: {diagnostics_output_dir / 'horizon_diag.csv'}")
    print(f"summary: {diagnostics_output_dir / 'validation_summary.json'}")
    print(
        "validation average_log_wealth: "
        f"{float(summary['validation']['average_log_wealth']):.6f}"
    )
    print(f"validation g_t mean: {float(summary['validation'].get('g_t_mean', 0.0)):.6f}")
    print(
        "validation policy utility mean: "
        f"{float(summary['validation'].get('policy_utility_mean', 0.0)):.6f}"
    )
    return 0


def tune_latest_command(args) -> int:
    _emit_cli_compat_warnings(args)
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv
        else artifact_root / "live" / "xauusd_m30_latest.csv"
    )
    manifest = tune_latest_dataset(
        csv_path=csv_path,
        artifact_root=artifact_root,
        seed=args.seed,
        config_overrides=_config_overrides_from_args(args),
        lookback_days=getattr(args, "csv_lookback_days", None),
    )
    best_candidate = manifest["best_candidate"]
    accepted_candidate = manifest.get("accepted_candidate")
    optimization_gate = manifest.get("optimization_gate", {})

    print(f"current run dir: {manifest['current_dir']}")
    print(f"archive session dir: {manifest['archive_session_dir']}")
    print(f"leaderboard: {manifest['leaderboard_path']}")
    print(f"optimization gate: {optimization_gate.get('status', 'unknown')}")
    print(
        "passed candidates: "
        f"{optimization_gate.get('passed_candidate_count', 0)}/"
        f"{optimization_gate.get('candidate_count', 0)}"
    )
    print(f"best validation loss: {best_candidate['best_validation_loss']:.6f}")
    print(f"best project value score: {best_candidate['project_value_score']:.6f}")
    print(f"best average_log_wealth: {best_candidate['average_log_wealth']:.6f}")
    print(f"best cvar_tail_loss: {best_candidate['cvar_tail_loss']:.6f}")
    print(f"best policy horizon: {best_candidate['policy_horizon']}")
    if accepted_candidate is None:
        print("accepted candidate: none")
        failed_rules = list(best_candidate.get("optimization_gate_failed_rules", []))
        if failed_rules:
            print(f"best candidate failed rules: {', '.join(str(rule) for rule in failed_rules)}")
        print("current updated: False")
        return 2
    print(f"accepted candidate: {accepted_candidate['candidate']}")
    print("current updated: True")
    return 0


def promote_current_command(args) -> int:
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    source_artifact_dir = Path(args.source_artifact_dir).expanduser().resolve()
    current_dir = _promote_training_run_to_current(artifact_root, source_artifact_dir)
    source_payload = load_json(current_dir / "source.json")
    print(f"current run dir: {current_dir}")
    print(f"artifact kind: {source_payload.get('artifact_kind')}")
    print(f"artifact id: {source_payload.get('artifact_id')}")
    print(f"git commit sha: {source_payload.get('git', {}).get('git_commit_sha')}")
    return 0


def _build_model_from_example(example, config: TrainingConfig) -> SignalCascadeModel:
    return SignalCascadeModel(
        feature_dim=len(example.main_sequences["4h"][0]),
        state_feature_dim=len(example.state_features),
        hidden_dim=config.hidden_dim,
        state_dim=config.state_dim,
        num_horizons=len(config.horizons),
        shape_classes=config.shape_classes,
        branch_dilations=config.branch_dilations,
        dropout=config.dropout,
    )


def _save_forecast_summary(output_dir: Path, prediction, config: TrainingConfig) -> None:
    save_json(
        output_dir / "forecast_summary.json",
        build_forecast_summary_payload(prediction, config),
    )


def _training_sub_artifacts(source_payload: dict[str, object]) -> dict[str, str]:
    entries = {
        "analysis.json": "regenerated",
        "config.json": "generated",
        "forecast_summary.json": "generated",
        "horizon_diag.csv": "generated",
        "manifest.json": "generated",
        "metrics.json": "generated",
        "prediction.json": "generated",
        "research_report.md": "regenerated",
        "source.json": "generated",
        "validation_rows.csv": "generated",
        "validation_summary.json": "generated",
    }
    if source_payload["kind"] == "csv":
        entries["data_snapshot.csv"] = "generated"
    return entries


def _overlay_sub_artifacts(source_payload: dict[str, object]) -> dict[str, str]:
    entries = {
        "analysis.json": "regenerated",
        "config.json": "regenerated",
        "forecast_summary.json": "copied",
        "horizon_diag.csv": "regenerated",
        "manifest.json": "generated",
        "metrics.json": "regenerated",
        "policy_summary.csv": "regenerated",
        "prediction.json": "copied",
        "source.json": "regenerated",
        "validation_rows.csv": "regenerated",
        "validation_summary.json": "regenerated",
    }
    if source_payload["kind"] == "csv":
        entries["data_snapshot.csv"] = "generated"
    return entries


def _materialize_replay_artifact(
    artifact_dir: Path,
    diagnostics_output_dir: Path,
    config: TrainingConfig,
    source_payload: dict[str, object],
    validation_summary: dict[str, object],
) -> None:
    generated_at_utc = str(validation_summary["generated_at_utc"])
    resolved_config = replace(config, output_dir=str(diagnostics_output_dir))
    save_json(diagnostics_output_dir / "config.json", resolved_config.to_dict())

    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        metrics_payload = load_json(metrics_path)
        metrics_payload["validation_metrics"] = dict(validation_summary["validation"])
        metrics_payload["schema_version"] = METRICS_SCHEMA_VERSION
        save_json(diagnostics_output_dir / "metrics.json", metrics_payload)

    for name in ("prediction.json", "forecast_summary.json"):
        source_path = artifact_dir / name
        if source_path.exists():
            save_json(diagnostics_output_dir / name, load_json(source_path))

    overlay_sub_artifacts = _overlay_sub_artifacts(source_payload)
    overlay_source_payload = build_artifact_source_payload(
        source_payload,
        diagnostics_output_dir,
        artifact_kind="diagnostic_replay_overlay",
        parent_artifact_dir=artifact_dir,
        generated_at_utc=generated_at_utc,
        sub_artifacts=build_subartifact_lineage(
            overlay_sub_artifacts,
            source_artifact_dir=artifact_dir,
        ),
    )
    save_json(
        diagnostics_output_dir / "manifest.json",
        _build_artifact_manifest(
            artifact_kind="diagnostic_replay_overlay",
            artifact_id=str(overlay_source_payload["artifact_id"]),
            parent_artifact_id=(
                None
                if overlay_source_payload.get("parent_artifact_id") is None
                else str(overlay_source_payload["parent_artifact_id"])
            ),
            generated_at_utc=generated_at_utc,
            source_payload=source_payload,
            entrypoints=_build_artifact_entrypoints(source_payload),
        ),
    )

    save_json(diagnostics_output_dir / "source.json", overlay_source_payload)

    if (diagnostics_output_dir / "metrics.json").exists() and (diagnostics_output_dir / "prediction.json").exists():
        generate_research_report(diagnostics_output_dir)
        save_json(
            diagnostics_output_dir / "source.json",
            build_artifact_source_payload(
                source_payload,
                diagnostics_output_dir,
                artifact_kind="diagnostic_replay_overlay",
                parent_artifact_dir=artifact_dir,
                generated_at_utc=generated_at_utc,
                sub_artifacts=build_subartifact_lineage(
                    overlay_sub_artifacts,
                    source_artifact_dir=artifact_dir,
                ),
            ),
        )


def _resolve_diagnostics_output_dir(
    artifact_dir: Path,
    diagnostics_output_dir: str | None,
) -> Path:
    resolved_artifact_dir = artifact_dir.expanduser().resolve()
    resolved_output_dir = (
        ensure_directory(Path(diagnostics_output_dir).expanduser().resolve())
        if diagnostics_output_dir is not None
        else ensure_directory(
            resolved_artifact_dir.parent / f"{resolved_artifact_dir.name}_diagnostic_replay_overlay"
        )
    )
    if resolved_output_dir == resolved_artifact_dir:
        raise ValueError(
            "diagnostic replay overlay must not overwrite the source artifact directory"
        )
    return resolved_output_dir


def _promote_training_run_to_current(
    artifact_root: Path,
    source_artifact_dir: Path,
) -> Path:
    resolved_artifact_root = artifact_root.expanduser().resolve()
    resolved_source_dir = source_artifact_dir.expanduser().resolve()
    current_dir = resolved_artifact_root / "current"

    if not resolved_artifact_root.exists():
        raise FileNotFoundError(f"artifact root was not found: {resolved_artifact_root}")
    if not resolved_source_dir.exists():
        raise FileNotFoundError(f"source artifact dir was not found: {resolved_source_dir}")
    if not resolved_source_dir.is_dir():
        raise ValueError(f"source artifact dir must be a directory: {resolved_source_dir}")
    if resolved_source_dir == current_dir:
        raise ValueError("current alias must be promoted from a distinct source artifact directory")

    source_payload_path = resolved_source_dir / "source.json"
    if not source_payload_path.exists():
        raise FileNotFoundError(f"source artifact is missing source.json: {source_payload_path}")
    source_payload = load_json(source_payload_path)
    if source_payload.get("artifact_kind") != "training_run":
        raise ValueError("current alias can only be promoted from a training_run artifact")
    load_required_diagnostics_summary(resolved_source_dir)

    stage_dir = resolved_artifact_root / f".current_promote_stage_{uuid4().hex}"
    backup_root = Path(mkdtemp(prefix="signalcascade-current-backup-")).resolve()
    backup_dir = backup_root / "current_legacy"

    try:
        shutil.copytree(resolved_source_dir, stage_dir)
        if current_dir.exists():
            shutil.move(str(current_dir), str(backup_dir))
        shutil.move(str(stage_dir), str(current_dir))
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        if backup_dir.exists() and not current_dir.exists():
            shutil.move(str(backup_dir), str(current_dir))
        shutil.rmtree(backup_root, ignore_errors=True)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    shutil.rmtree(backup_root, ignore_errors=True)
    if all((current_dir / name).exists() for name in ("config.json", "metrics.json", "prediction.json")):
        generate_research_report(current_dir)
    _write_current_artifact_manifest(current_dir)
    return current_dir


def _write_current_artifact_manifest(current_dir: Path) -> None:
    source_payload = load_json(current_dir / "source.json")
    if not isinstance(source_payload, dict):
        raise FileNotFoundError(f"current artifact is missing source.json: {current_dir / 'source.json'}")
    generated_at_utc = source_payload.get("generated_at_utc")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.strip():
        diagnostics_summary = load_required_diagnostics_summary(current_dir)
        generated_at_utc = str(diagnostics_summary["generated_at_utc"])
    artifact_id = source_payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError(f"current artifact is missing source artifact_id: {current_dir / 'source.json'}")
    artifact_kind = source_payload.get("artifact_kind")
    if not isinstance(artifact_kind, str) or not artifact_kind.strip():
        raise ValueError(f"current artifact is missing source artifact_kind: {current_dir / 'source.json'}")
    parent_artifact_id = source_payload.get("parent_artifact_id")
    resolved_parent_artifact_id = (
        str(parent_artifact_id) if parent_artifact_id is not None else None
    )
    save_json(
        current_dir / "manifest.json",
        build_artifact_manifest(
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            parent_artifact_id=resolved_parent_artifact_id,
            generated_at_utc=generated_at_utc,
            source_payload=source_payload,
            entrypoints=build_artifact_entrypoints(
                source_payload,
                include_model=(current_dir / "model.pt").exists(),
            ),
        ),
    )


_build_artifact_entrypoints = build_artifact_entrypoints
_build_artifact_manifest = build_artifact_manifest


def _build_config(args) -> TrainingConfig:
    return TrainingConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        **_config_overrides_from_args(args),
    )


def _load_config_with_overrides(output_dir: Path, args) -> TrainingConfig:
    payload = load_json(output_dir / "config.json")
    payload.update(_config_overrides_from_args(args))
    return TrainingConfig.from_dict(payload)


def _build_source_payload(args, config: TrainingConfig) -> dict[str, object]:
    if args.csv:
        payload: dict[str, object] = {"kind": "csv", "path": str(Path(args.csv).expanduser().resolve())}
        if getattr(args, "csv_lookback_days", None) is not None:
            payload["lookback_days"] = int(args.csv_lookback_days)
        return normalize_price_scale_payload(
            payload,
            requested_price_scale=config.requested_price_scale,
        )
    return normalize_price_scale_payload(
        {"kind": "synthetic", "bars": config.synthetic_bars, "seed": config.seed},
        requested_price_scale=config.requested_price_scale,
    )


def _resolve_source_payload(args, output_dir: Path) -> dict[str, object]:
    saved_payload = (
        load_json(output_dir / "source.json")
        if (output_dir / "source.json").exists()
        else {}
    )
    if getattr(args, "csv", None):
        payload = {
            **normalize_price_scale_payload(saved_payload),
            "kind": "csv",
            "path": str(Path(args.csv).expanduser().resolve()),
        }
        if getattr(args, "csv_lookback_days", None) is not None:
            payload["lookback_days"] = int(args.csv_lookback_days)
        elif "lookback_days" in saved_payload:
            payload["lookback_days"] = saved_payload["lookback_days"]
        return normalize_price_scale_payload(
            payload,
            requested_price_scale=getattr(args, "price_scale", None),
        )
    return normalize_price_scale_payload(
        saved_payload,
        requested_price_scale=getattr(args, "price_scale", None),
    )


def _resolve_price_scale(source_payload: dict[str, object]) -> float:
    return resolve_effective_price_scale(source_payload)


def _config_overrides_from_args(args) -> dict[str, object]:
    overrides: dict[str, object] = {}

    def maybe(name: str, value) -> None:
        if value is not None:
            overrides[name] = value

    maybe("synthetic_bars", getattr(args, "synthetic_bars", None))
    maybe("epochs", getattr(args, "epochs", None))
    maybe("warmup_epochs", getattr(args, "warmup_epochs", None))
    maybe("batch_size", getattr(args, "batch_size", None))
    maybe("learning_rate", getattr(args, "learning_rate", None))
    maybe("weight_decay", getattr(args, "weight_decay", None))
    maybe("hidden_dim", getattr(args, "hidden_dim", None))
    maybe("state_dim", getattr(args, "state_dim", None))
    maybe("shape_classes", getattr(args, "shape_classes", None))
    maybe("dropout", getattr(args, "dropout", None))
    maybe("walk_forward_folds", getattr(args, "walk_forward_folds", None))
    maybe("base_cost", getattr(args, "base_cost", None))
    maybe("delta_multiplier", getattr(args, "delta_multiplier", None))
    maybe("mae_multiplier", getattr(args, "mae_multiplier", None))
    maybe("training_state_reset_mode", getattr(args, "training_state_reset_mode", None))
    maybe("evaluation_state_reset_mode", getattr(args, "evaluation_state_reset_mode", None))
    diagnostic_modes = _parse_str_list(getattr(args, "diagnostic_state_reset_modes", None))
    if diagnostic_modes is not None:
        overrides["diagnostic_state_reset_modes"] = diagnostic_modes
    maybe("return_loss_weight", getattr(args, "return_loss_weight", None))
    maybe("shape_loss_weight", getattr(args, "shape_loss_weight", None))
    maybe("profit_loss_weight", getattr(args, "profit_loss_weight", None))
    maybe("cvar_weight", getattr(args, "cvar_weight", None))
    maybe("cvar_alpha", getattr(args, "cvar_alpha", None))
    maybe("risk_aversion_gamma", getattr(args, "risk_aversion_gamma", None))
    maybe("q_max", getattr(args, "q_max", None))
    maybe("allow_no_candidate", getattr(args, "allow_no_candidate", None))
    maybe("requested_price_scale", getattr(args, "price_scale", None))

    horizons = _parse_horizons(getattr(args, "horizons", None))
    if horizons is not None:
        overrides["horizons"] = horizons
    cost_multipliers = _parse_float_list(getattr(args, "policy_sweep_cost_multipliers", None))
    if cost_multipliers is not None:
        overrides["policy_sweep_cost_multipliers"] = cost_multipliers
    gamma_multipliers = _parse_float_list(getattr(args, "policy_sweep_gamma_multipliers", None))
    if gamma_multipliers is not None:
        overrides["policy_sweep_gamma_multipliers"] = gamma_multipliers
    min_policy_sigmas = _parse_float_list(getattr(args, "policy_sweep_min_policy_sigmas", None))
    if min_policy_sigmas is not None:
        overrides["policy_sweep_min_policy_sigmas"] = min_policy_sigmas
    sweep_state_reset_modes = _parse_str_list(getattr(args, "policy_sweep_state_reset_modes", None))
    if sweep_state_reset_modes is not None:
        overrides["policy_sweep_state_reset_modes"] = sweep_state_reset_modes

    return overrides


def _merge_artifact_price_scale_metadata(output_dir: Path, source_payload: dict[str, object]) -> None:
    normalized = normalize_price_scale_payload(source_payload)
    for name in ("source.json", "manifest.json"):
        path = output_dir / name
        if not path.exists():
            continue
        payload = load_json(path)
        payload.update(price_scale_manifest_fields(normalized))
        if name == "source.json":
            payload["effective_price_scale"] = normalized["effective_price_scale"]
            payload["price_scale"] = normalized["price_scale"]
            payload["price_scale_origin"] = normalized["price_scale_origin"]
            payload["provider_scale_confirmed"] = normalized["provider_scale_confirmed"]
            if "requested_price_scale" in normalized:
                payload["requested_price_scale"] = normalized["requested_price_scale"]
            else:
                payload.pop("requested_price_scale", None)
        save_json(path, payload)


def _emit_cli_compat_warnings(args) -> None:
    selection_score_source = getattr(args, "selection_score_source", None)
    if selection_score_source not in (None, "profit_utility"):
        print(
            "warning: --selection-score-source is deprecated and ignored; "
            "the active policy path always uses profit_utility."
        )

    selection_threshold_mode = getattr(args, "selection_threshold_mode", None)
    if selection_threshold_mode not in (None, "auto"):
        print(
            "warning: --selection-threshold-mode/--acceptance-threshold-mode is deprecated "
            "and ignored; diagnostics replay now uses the q_t policy path only."
        )


def _parse_horizons(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    values = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
    if not values:
        return None
    return tuple(int(value) for value in values)


def _parse_float_list(raw_value: str | None) -> tuple[float, ...] | None:
    if raw_value is None:
        return None
    values = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
    if not values:
        return None
    return tuple(float(value) for value in values)


def _parse_str_list(raw_value: str | None) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    values = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
    if not values:
        return None
    return tuple(values)


def _create_data_source(source_payload: dict[str, object]):
    if source_payload["kind"] == "csv":
        return CsvMarketDataSource(Path(str(source_payload["path"])))
    return SyntheticMarketDataSource(
        bar_count=int(source_payload["bars"]),
        seed=int(source_payload["seed"]),
    )
