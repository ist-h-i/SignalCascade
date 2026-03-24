from __future__ import annotations

import argparse

from ..bootstrap import predict_command, train_command, tune_latest_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-cascade",
        description="PyTorch reference implementation for the SignalCascade project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model and emit a latest prediction.")
    train_parser.add_argument("--csv", default=None, help="Path to a 30m OHLCV CSV file.")
    train_parser.add_argument("--output-dir", default="artifacts/demo")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=32)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--synthetic-bars", type=int, default=10_080)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.set_defaults(handler=train_command)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Load a saved checkpoint and emit the latest prediction.",
    )
    predict_parser.add_argument("--output-dir", default="artifacts/demo")
    predict_parser.add_argument("--csv", default=None, help="Override the data source with a CSV file.")
    predict_parser.set_defaults(handler=predict_command)

    tune_parser = subparsers.add_parser(
        "tune-latest",
        help="Tune on the latest CSV, publish the best run to current, and archive the others.",
    )
    tune_parser.add_argument("--artifact-root", default="artifacts/gold_xauusd_m30")
    tune_parser.add_argument("--csv", default=None, help="Path to the latest 30m OHLCV CSV file.")
    tune_parser.add_argument("--seed", type=int, default=7)
    tune_parser.set_defaults(handler=tune_latest_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
