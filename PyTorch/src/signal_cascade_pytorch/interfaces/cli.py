from __future__ import annotations

import argparse

from ..bootstrap import export_diagnostics_command, predict_command, train_command, tune_latest_command


def build_parser() -> argparse.ArgumentParser:
    selection_score_sources = (
        "selector_probability",
        "correctness_probability",
        "actionable_edge",
        "edge_correctness_product",
    )
    parser = argparse.ArgumentParser(
        prog="signal-cascade",
        description="PyTorch reference implementation for the SignalCascade project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model and emit a latest prediction.")
    train_parser.add_argument("--csv", default=None, help="Path to a 30m OHLCV CSV file.")
    train_parser.add_argument("--csv-lookback-days", type=int, default=None)
    train_parser.add_argument("--output-dir", default="artifacts/demo")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=32)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--horizons", default=None, help="Comma-separated 4h horizons, e.g. 1,3,6")
    train_parser.add_argument("--walk-forward-folds", type=int, default=None)
    train_parser.add_argument("--precision-target", type=float, default=None)
    train_parser.add_argument("--selection-min-support", type=int, default=None)
    train_parser.add_argument("--precision-confidence-z", type=float, default=None)
    train_parser.add_argument("--base-cost", type=float, default=None)
    train_parser.add_argument("--delta-multiplier", type=float, default=None)
    train_parser.add_argument("--mae-multiplier", type=float, default=None)
    train_parser.add_argument(
        "--allow-no-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow policy evaluation to emit no selection when no horizon has positive actionable edge.",
    )
    train_parser.add_argument(
        "--selection-score-source",
        choices=selection_score_sources,
        default=None,
        help="Score source used for threshold calibration and acceptance.",
    )
    train_parser.add_argument("--shape-loss-weight", type=float, default=None)
    train_parser.add_argument("--overlay-loss-weight", type=float, default=None)
    train_parser.add_argument("--direction-loss-weight", type=float, default=None)
    train_parser.add_argument("--consistency-loss-weight", type=float, default=None)
    train_parser.add_argument("--synthetic-bars", type=int, default=10_080)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.set_defaults(handler=train_command)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Load a saved checkpoint and emit the latest prediction.",
    )
    predict_parser.add_argument("--output-dir", default="artifacts/demo")
    predict_parser.add_argument("--csv", default=None, help="Override the data source with a CSV file.")
    predict_parser.add_argument("--csv-lookback-days", type=int, default=None)
    predict_parser.add_argument(
        "--allow-no-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the saved config and allow no-selection outputs when no candidate survives.",
    )
    predict_parser.add_argument(
        "--selection-score-source",
        choices=selection_score_sources,
        default=None,
        help="Override the saved config threshold score source for diagnostics or replay.",
    )
    predict_parser.set_defaults(handler=predict_command)

    export_parser = subparsers.add_parser(
        "export-diagnostics",
        help="Export validation rows and threshold diagnostics from an existing artifact directory.",
    )
    export_parser.add_argument("--output-dir", default="artifacts/demo")
    export_parser.add_argument(
        "--diagnostics-output-dir",
        default=None,
        help="Optional directory to write replay diagnostics without overwriting the source artifact.",
    )
    export_parser.add_argument("--csv", default=None, help="Override the data source with a CSV file.")
    export_parser.add_argument("--csv-lookback-days", type=int, default=None)
    export_parser.add_argument(
        "--selection-threshold-mode",
        choices=("auto", "stored", "replay", "none"),
        default="auto",
        help="How export-diagnostics resolves selection thresholds. Defaults to auto replay on config mismatch.",
    )
    export_parser.add_argument(
        "--allow-no-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the saved config and allow no-selection diagnostics when no candidate survives.",
    )
    export_parser.add_argument(
        "--selection-score-source",
        choices=selection_score_sources,
        default=None,
        help="Override the saved config threshold score source for validation replay.",
    )
    export_parser.set_defaults(handler=export_diagnostics_command)

    tune_parser = subparsers.add_parser(
        "tune-latest",
        help="Tune on the latest CSV, publish the best run to current, and archive the others.",
    )
    tune_parser.add_argument("--artifact-root", default="artifacts/gold_xauusd_m30")
    tune_parser.add_argument("--csv", default=None, help="Path to the latest 30m OHLCV CSV file.")
    tune_parser.add_argument("--csv-lookback-days", type=int, default=None)
    tune_parser.add_argument("--epochs", type=int, default=None)
    tune_parser.add_argument("--batch-size", type=int, default=None)
    tune_parser.add_argument("--learning-rate", type=float, default=None)
    tune_parser.add_argument("--weight-decay", type=float, default=None)
    tune_parser.add_argument("--hidden-dim", type=int, default=None)
    tune_parser.add_argument("--dropout", type=float, default=None)
    tune_parser.add_argument("--horizons", default=None, help="Comma-separated 4h horizons, e.g. 1,3,6")
    tune_parser.add_argument("--walk-forward-folds", type=int, default=None)
    tune_parser.add_argument("--precision-target", type=float, default=None)
    tune_parser.add_argument("--selection-min-support", type=int, default=None)
    tune_parser.add_argument("--precision-confidence-z", type=float, default=None)
    tune_parser.add_argument("--base-cost", type=float, default=None)
    tune_parser.add_argument("--delta-multiplier", type=float, default=None)
    tune_parser.add_argument("--mae-multiplier", type=float, default=None)
    tune_parser.add_argument(
        "--allow-no-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow candidate-free anchors during tuning diagnostics and policy calibration.",
    )
    tune_parser.add_argument(
        "--selection-score-source",
        choices=selection_score_sources,
        default=None,
        help="Score source used for threshold calibration during tuning.",
    )
    tune_parser.add_argument("--shape-loss-weight", type=float, default=None)
    tune_parser.add_argument("--overlay-loss-weight", type=float, default=None)
    tune_parser.add_argument("--direction-loss-weight", type=float, default=None)
    tune_parser.add_argument("--consistency-loss-weight", type=float, default=None)
    tune_parser.add_argument("--seed", type=int, default=7)
    tune_parser.set_defaults(handler=tune_latest_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
