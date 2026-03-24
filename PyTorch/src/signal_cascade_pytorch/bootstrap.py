from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .application.config import TrainingConfig
from .application.diagnostics_service import export_review_diagnostics
from .application.dataset_service import (
    build_latest_inference_example,
    build_latest_inference_example_from_bars,
    build_training_examples,
    build_training_examples_from_bars,
    limit_base_bars_to_lookback_days,
    trim_base_bars_for_latest_inference,
)
from .application.inference_service import predict_from_example, predict_latest
from .application.report_service import generate_research_report
from .application.training_service import train_model
from .application.tuning_service import tune_latest_dataset
from .infrastructure.data.csv_source import CsvMarketDataSource
from .infrastructure.data.synthetic_source import SyntheticMarketDataSource
from .infrastructure.ml.model import SignalCascadeModel
from .infrastructure.persistence import ensure_directory, load_checkpoint, load_json, save_json


def train_command(args) -> int:
    config = _build_config(args)
    output_dir = ensure_directory(Path(config.output_dir))
    source_payload = _build_source_payload(args, config)
    source = _create_data_source(source_payload)
    source_rows_original = None
    source_rows_used = None

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        source_rows_original = len(base_bars)
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        source_rows_used = len(base_bars)
        examples = build_training_examples_from_bars(base_bars, config)
    else:
        examples = build_training_examples(source, config)

    model, summary = train_model(examples, config, output_dir)
    selection_policy = dict(summary.pop("selection_policy"))
    prediction = predict_latest(model, examples, config, selection_policy)

    summary["sample_count"] = len(examples)
    summary["source"] = source_payload
    if source_rows_original is not None and source_rows_used is not None:
        summary["source_rows_original"] = source_rows_original
        summary["source_rows_used"] = source_rows_used
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "source.json", source_payload)
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "selection_policy.json", selection_policy)
    save_json(output_dir / "prediction.json", asdict(prediction))
    export_review_diagnostics(
        output_dir=output_dir,
        model=model,
        examples=examples,
        config=config,
        selection_policy=selection_policy,
        source_payload=source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars if source_payload["kind"] == "csv" else None,
    )
    generate_research_report(output_dir)

    print(f"trained samples: {summary['train_samples']}")
    print(f"validation samples: {summary['validation_samples']}")
    print(f"best validation loss: {summary['best_validation_loss']:.6f}")
    print(f"validation selection precision: {summary['validation_metrics']['selection_precision']:.4f}")
    print(f"validation coverage@precision: {summary['validation_metrics']['coverage_at_target_precision']:.4f}")
    print(f"validation directional accuracy: {summary['validation_metrics']['directional_accuracy']:.4f}")
    print(f"validation project value score: {summary['validation_metrics']['project_value_score']:.4f}")
    print(f"latest accepted signal: {prediction.accepted_signal}")
    print(f"latest selection probability: {prediction.selection_probability:.4f}")
    print(f"latest overlay action: {prediction.overlay_action}")
    return 0


def predict_command(args) -> int:
    output_dir = Path(args.output_dir)
    config = _load_config_with_overrides(output_dir, args)
    source_payload = _resolve_source_payload(args, output_dir)
    source = _create_data_source(source_payload)

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        trimmed_bars = trim_base_bars_for_latest_inference(base_bars, config)
        latest_example = build_latest_inference_example_from_bars(trimmed_bars, config)
    else:
        latest_example = build_latest_inference_example(source, config)

    feature_dim = len(latest_example.main_sequences["4h"][0])
    model = SignalCascadeModel(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_horizons=len(config.horizons),
        dropout=config.dropout,
    )
    load_checkpoint(output_dir / "model.pt", model)
    selection_policy_path = output_dir / "selection_policy.json"
    selection_policy = load_json(selection_policy_path) if selection_policy_path.exists() else None
    prediction = predict_from_example(model, latest_example, config, selection_policy)
    save_json(output_dir / "prediction.json", asdict(prediction))

    print(f"selected horizon: {prediction.selected_horizon}")
    print(f"accepted signal: {prediction.accepted_signal}")
    print(f"selection probability: {prediction.selection_probability:.4f}")
    print(f"position: {prediction.position:.4f}")
    print(f"overlay action: {prediction.overlay_action}")
    return 0


def export_diagnostics_command(args) -> int:
    output_dir = Path(args.output_dir)
    config = _load_config_with_overrides(output_dir, args)
    source_payload = _resolve_source_payload(args, output_dir)
    source = _create_data_source(source_payload)
    source_rows_original = None
    source_rows_used = None
    base_bars = None

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        source_rows_original = len(base_bars)
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        source_rows_used = len(base_bars)
        examples = build_training_examples_from_bars(base_bars, config)
    else:
        examples = build_training_examples(source, config)

    feature_dim = len(examples[0].main_sequences["4h"][0])
    model = SignalCascadeModel(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_horizons=len(config.horizons),
        dropout=config.dropout,
    )
    load_checkpoint(output_dir / "model.pt", model)
    selection_policy_path = output_dir / "selection_policy.json"
    selection_policy = load_json(selection_policy_path) if selection_policy_path.exists() else None
    summary = export_review_diagnostics(
        output_dir=output_dir,
        model=model,
        examples=examples,
        config=config,
        selection_policy=selection_policy,
        source_payload=source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars,
    )

    print(f"validation rows: {output_dir / 'validation_rows.csv'}")
    print(f"threshold scan: {output_dir / 'threshold_scan.csv'}")
    print(f"horizon diagnostics: {output_dir / 'horizon_diag.csv'}")
    print(f"summary: {output_dir / 'validation_summary.json'}")
    print(f"validation selected rows: {summary['validation']['selected_row_count']}")
    return 0


def tune_latest_command(args) -> int:
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

    print(f"current run dir: {manifest['current_dir']}")
    print(f"archive session dir: {manifest['archive_session_dir']}")
    print(f"best validation loss: {best_candidate['best_validation_loss']:.6f}")
    print(f"best project value score: {best_candidate['project_value_score']:.6f}")
    print(f"best selection precision: {best_candidate['selection_precision']:.6f}")
    print(f"best coverage@precision: {best_candidate['coverage_at_target_precision']:.6f}")
    print(f"selected horizon: {best_candidate['selected_horizon']}")
    return 0


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
        payload = {"kind": "csv", "path": str(Path(args.csv).expanduser().resolve())}
        if getattr(args, "csv_lookback_days", None) is not None:
            payload["lookback_days"] = int(args.csv_lookback_days)
        return payload
    return {"kind": "synthetic", "bars": config.synthetic_bars, "seed": config.seed}


def _resolve_source_payload(args, output_dir: Path) -> dict[str, object]:
    if getattr(args, "csv", None):
        payload = {"kind": "csv", "path": str(Path(args.csv).expanduser().resolve())}
        if getattr(args, "csv_lookback_days", None) is not None:
            payload["lookback_days"] = int(args.csv_lookback_days)
        return payload
    return load_json(output_dir / "source.json")


def _config_overrides_from_args(args) -> dict[str, object]:
    overrides: dict[str, object] = {}

    def maybe(name: str, value) -> None:
        if value is not None:
            overrides[name] = value

    maybe("synthetic_bars", getattr(args, "synthetic_bars", None))
    maybe("epochs", getattr(args, "epochs", None))
    maybe("batch_size", getattr(args, "batch_size", None))
    maybe("learning_rate", getattr(args, "learning_rate", None))
    maybe("weight_decay", getattr(args, "weight_decay", None))
    maybe("hidden_dim", getattr(args, "hidden_dim", None))
    maybe("dropout", getattr(args, "dropout", None))
    maybe("walk_forward_folds", getattr(args, "walk_forward_folds", None))
    maybe("precision_target", getattr(args, "precision_target", None))
    maybe("selection_min_support", getattr(args, "selection_min_support", None))
    maybe("precision_confidence_z", getattr(args, "precision_confidence_z", None))
    maybe("base_cost", getattr(args, "base_cost", None))
    maybe("delta_multiplier", getattr(args, "delta_multiplier", None))
    maybe("mae_multiplier", getattr(args, "mae_multiplier", None))
    maybe("allow_no_candidate", getattr(args, "allow_no_candidate", None))
    maybe("selection_score_source", getattr(args, "selection_score_source", None))
    maybe("shape_loss_weight", getattr(args, "shape_loss_weight", None))
    maybe("overlay_loss_weight", getattr(args, "overlay_loss_weight", None))
    maybe("direction_loss_weight", getattr(args, "direction_loss_weight", None))
    maybe("consistency_loss_weight", getattr(args, "consistency_loss_weight", None))

    horizons = _parse_horizons(getattr(args, "horizons", None))
    if horizons is not None:
        overrides["horizons"] = horizons

    return overrides


def _parse_horizons(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    values = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
    if not values:
        return None
    return tuple(int(value) for value in values)


def _create_data_source(source_payload: dict[str, object]):
    if source_payload["kind"] == "csv":
        return CsvMarketDataSource(Path(str(source_payload["path"])))
    return SyntheticMarketDataSource(
        bar_count=int(source_payload["bars"]),
        seed=int(source_payload["seed"]),
    )
