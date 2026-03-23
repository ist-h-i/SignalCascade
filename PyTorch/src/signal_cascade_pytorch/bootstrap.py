from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .application.config import TrainingConfig
from .application.dataset_service import build_training_examples
from .application.inference_service import predict_latest
from .application.training_service import train_model
from .infrastructure.data.csv_source import CsvMarketDataSource
from .infrastructure.data.synthetic_source import SyntheticMarketDataSource
from .infrastructure.ml.model import SignalCascadeModel
from .infrastructure.persistence import ensure_directory, load_checkpoint, load_json, save_json


def train_command(args) -> int:
    config = _build_config(args)
    output_dir = ensure_directory(Path(config.output_dir))
    source_payload = _build_source_payload(args, config)
    source = _create_data_source(source_payload)
    examples = build_training_examples(source, config)
    model, summary = train_model(examples, config, output_dir)
    prediction = predict_latest(model, examples, config)

    summary["sample_count"] = len(examples)
    summary["source"] = source_payload
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "source.json", source_payload)
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "prediction.json", asdict(prediction))

    print(f"trained samples: {summary['train_samples']}")
    print(f"validation samples: {summary['validation_samples']}")
    print(f"best validation loss: {summary['best_validation_loss']:.6f}")
    print(f"latest overlay action: {prediction.overlay_action}")
    return 0


def predict_command(args) -> int:
    output_dir = Path(args.output_dir)
    config = TrainingConfig.from_dict(load_json(output_dir / "config.json"))
    source_payload = _resolve_source_payload(args, output_dir)
    source = _create_data_source(source_payload)
    examples = build_training_examples(source, config)
    feature_dim = len(examples[0].main_sequences["4h"][0])
    model = SignalCascadeModel(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_horizons=len(config.horizons),
        dropout=config.dropout,
    )
    load_checkpoint(output_dir / "model.pt", model)
    prediction = predict_latest(model, examples, config)
    save_json(output_dir / "prediction.json", asdict(prediction))

    print(f"selected horizon: {prediction.selected_horizon}")
    print(f"position: {prediction.position:.4f}")
    print(f"overlay action: {prediction.overlay_action}")
    return 0


def _build_config(args) -> TrainingConfig:
    return TrainingConfig(
        seed=args.seed,
        synthetic_bars=args.synthetic_bars,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
    )


def _build_source_payload(args, config: TrainingConfig) -> dict[str, object]:
    if args.csv:
        return {"kind": "csv", "path": str(Path(args.csv).expanduser().resolve())}
    return {"kind": "synthetic", "bars": config.synthetic_bars, "seed": config.seed}


def _resolve_source_payload(args, output_dir: Path) -> dict[str, object]:
    if getattr(args, "csv", None):
        return {"kind": "csv", "path": str(Path(args.csv).expanduser().resolve())}
    return load_json(output_dir / "source.json")


def _create_data_source(source_payload: dict[str, object]):
    if source_payload["kind"] == "csv":
        return CsvMarketDataSource(Path(str(source_payload["path"])))
    return SyntheticMarketDataSource(
        bar_count=int(source_payload["bars"]),
        seed=int(source_payload["seed"]),
    )
