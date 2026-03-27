from __future__ import annotations

from datetime import datetime, timedelta
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
from .application.inference_service import (
    build_forecast_summary_payload,
    predict_from_example,
    predict_latest,
    serialize_prediction_result,
)
from .application.report_service import METRICS_SCHEMA_VERSION, generate_research_report
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
    base_bars = None

    if source_payload["kind"] == "csv":
        base_bars = source.load_bars()
        source_rows_original = len(base_bars)
        base_bars = limit_base_bars_to_lookback_days(base_bars, source_payload.get("lookback_days"))
        source_rows_used = len(base_bars)
        examples = build_training_examples_from_bars(base_bars, config)
    else:
        examples = build_training_examples(source, config)

    model, summary = train_model(examples, config, output_dir)
    prediction = predict_latest(
        model=model,
        examples=examples,
        config=config,
        previous_position=float(getattr(args, "previous_position", 0.0) or 0.0),
    )

    summary["sample_count"] = len(examples)
    summary["effective_sample_count"] = summary["train_samples"] + summary["validation_samples"]
    summary["source"] = source_payload
    summary["schema_version"] = METRICS_SCHEMA_VERSION
    if source_rows_original is not None and source_rows_used is not None:
        summary["source_rows_original"] = source_rows_original
        summary["source_rows_used"] = source_rows_used
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "source.json", source_payload)
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "prediction.json", serialize_prediction_result(prediction))
    _save_forecast_summary(output_dir, prediction, config)
    export_review_diagnostics(
        output_dir=output_dir,
        model=model,
        examples=examples,
        config=config,
        source_payload=source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars if source_payload["kind"] == "csv" else None,
    )
    generate_research_report(output_dir)

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
    print(f"latest no-trade-band hit: {prediction.no_trade_band_hit}")
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

    model = _build_model_from_example(latest_example, config)
    load_checkpoint(output_dir / "model.pt", model)
    prediction = predict_from_example(
        model=model,
        example=latest_example,
        config=config,
        previous_position=float(getattr(args, "previous_position", 0.0) or 0.0),
    )
    save_json(output_dir / "prediction.json", serialize_prediction_result(prediction))
    _save_forecast_summary(output_dir, prediction, config)

    print(f"policy horizon: {prediction.policy_horizon}")
    print(f"executed horizon: {prediction.executed_horizon}")
    print(f"position: {prediction.position:.4f}")
    print(f"trade delta: {prediction.trade_delta:.4f}")
    print(f"no-trade-band hit: {prediction.no_trade_band_hit}")
    return 0


def export_diagnostics_command(args) -> int:
    artifact_dir = Path(args.output_dir).expanduser().resolve()
    diagnostics_output_dir = (
        ensure_directory(Path(args.diagnostics_output_dir).expanduser().resolve())
        if getattr(args, "diagnostics_output_dir", None)
        else artifact_dir
    )
    config = _load_config_with_overrides(artifact_dir, args)
    source_payload = _resolve_source_payload(args, artifact_dir)
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

    model = _build_model_from_example(examples[0], config)
    load_checkpoint(artifact_dir / "model.pt", model)
    summary = export_review_diagnostics(
        output_dir=diagnostics_output_dir,
        model=model,
        examples=examples,
        config=config,
        source_payload=source_payload,
        source_rows_original=source_rows_original,
        source_rows_used=source_rows_used,
        base_bars=base_bars,
    )

    print(f"artifact dir: {artifact_dir}")
    print(f"diagnostics dir: {diagnostics_output_dir}")
    print(f"validation rows: {diagnostics_output_dir / 'validation_rows.csv'}")
    print(f"policy summary: {diagnostics_output_dir / 'policy_summary.csv'}")
    print(f"horizon diagnostics: {diagnostics_output_dir / 'horizon_diag.csv'}")
    print(f"summary: {diagnostics_output_dir / 'validation_summary.json'}")
    print(f"validation executed trades: {summary['validation']['executed_trade_count']}")
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
    print(f"best average_log_wealth: {best_candidate['average_log_wealth']:.6f}")
    print(f"best cvar_tail_loss: {best_candidate['cvar_tail_loss']:.6f}")
    print(f"best policy horizon: {best_candidate['policy_horizon']}")
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
