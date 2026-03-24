from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from ..infrastructure.persistence import load_json, save_json

JST = timezone(timedelta(hours=9))
UTC = timezone.utc


def generate_research_report(
    output_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    metrics = load_json(output_dir / "metrics.json")
    prediction = load_json(output_dir / "prediction.json")
    config = load_json(output_dir / "config.json")
    manifest = load_json(output_dir / "manifest.json") if (output_dir / "manifest.json").exists() else None
    forecast_summary = (
        load_json(output_dir / "forecast_summary.json")
        if (output_dir / "forecast_summary.json").exists()
        else {}
    )
    leaderboard = _load_leaderboard(manifest)
    previous_metrics = _load_previous_metrics(manifest)
    source_summary = _load_source_summary(metrics)

    analysis = _build_analysis(
        output_dir=output_dir,
        metrics=metrics,
        prediction=prediction,
        config=config,
        manifest=manifest,
        forecast_summary=forecast_summary,
        leaderboard=leaderboard,
        previous_metrics=previous_metrics,
        source_summary=source_summary,
    )
    save_json(output_dir / "analysis.json", analysis)

    markdown = _render_markdown_report(analysis)
    report_output_path = output_dir / "research_report.md"
    report_output_path.write_text(markdown, encoding="utf-8")

    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown, encoding="utf-8")

    return analysis


def _build_analysis(
    output_dir: Path,
    metrics: dict[str, object],
    prediction: dict[str, object],
    config: dict[str, object],
    manifest: dict[str, object] | None,
    forecast_summary: dict[str, object],
    leaderboard: Sequence[dict[str, object]],
    previous_metrics: dict[str, object] | None,
    source_summary: dict[str, object] | None,
) -> dict[str, object]:
    validation = dict(metrics.get("validation_metrics", {}))
    top_candidates = [dict(candidate) for candidate in leaderboard[:3]]
    forecast_rows = _build_forecast_rows(forecast_summary, prediction)
    selected_horizon = int(prediction["selected_horizon"])
    selected_forecast = next(
        (
            row
            for row in forecast_rows
            if int(row["horizon_4h"]) == selected_horizon
        ),
        forecast_rows[0] if forecast_rows else None,
    )
    comparison = _build_comparison(validation, previous_metrics)
    generated_at_utc = datetime.now(UTC)
    generated_at_jst = generated_at_utc.astimezone(JST)
    project_stage = _project_stage(validation)

    return {
        "instrument": _infer_instrument(metrics, output_dir),
        "generated_at_utc": generated_at_utc.isoformat(),
        "generated_at_jst": generated_at_jst.isoformat(),
        "artifact_dir": str(output_dir),
        "dataset": {
            "sample_count": int(metrics.get("sample_count", 0)),
            "train_samples": int(metrics.get("train_samples", 0)),
            "validation_samples": int(metrics.get("validation_samples", 0)),
            "source_rows_used": int(metrics.get("source_rows_used", 0)),
            "source_rows_original": int(metrics.get("source_rows_original", 0)),
            "source": source_summary,
        },
        "training": {
            "best_validation_loss": float(metrics.get("best_validation_loss", 0.0)),
            "best_epoch": float(metrics.get("best_epoch", 0.0)),
            "best_params": _extract_best_params(manifest, config),
            "candidate_count": len(leaderboard),
            "walk_forward_folds": int(config.get("walk_forward_folds", 0)),
        },
        "metric_definitions": {
            "utility_score": "シグナル品質の複合指標。precision, coverage, capture, overlay F1, directional accuracy, drawdown を混合する。",
            "project_value_score": "事業価値の複合指標。precision, coverage, capture, profit factor, sortino, drawdown, calibration を統合する。",
            "profit_factor": "利益総額 / 損失総額の絶対値。数値安定化のため 10.0 で上限 clip する。",
            "signal_sortino": "下方偏差で割ったリスク調整後価値。数値安定化のため 10.0 で上限 clip する。",
            "selection_brier_score": "採用確率の calibration 誤差。小さいほど良い。",
        },
        "validation_metrics": validation,
        "leaderboard_top": top_candidates,
        "comparison_to_previous": comparison,
        "forecast": {
            "anchor_time_utc": str(prediction["anchor_time"]),
            "anchor_time_jst": _to_jst_string(str(prediction["anchor_time"])),
            "anchor_close": float(prediction.get("current_close", 0.0)),
            "selected_horizon": selected_horizon,
            "selected_direction": int(prediction.get("selected_direction", 0)),
            "selected_direction_label": _direction_label(int(prediction.get("selected_direction", 0))),
            "accepted_signal": bool(prediction.get("accepted_signal", False)),
            "position": float(prediction.get("position", 0.0)),
            "selection_probability": float(prediction.get("selection_probability", 0.0)),
            "selection_threshold": (
                None
                if prediction.get("selection_threshold") is None
                else float(prediction.get("selection_threshold", 0.0))
            ),
            "correctness_probability": float(prediction.get("correctness_probability", 0.0)),
            "hold_probability": float(prediction.get("hold_probability", 0.0)),
            "hold_threshold": float(prediction.get("hold_threshold", 0.0)),
            "overlay_action": str(prediction.get("overlay_action", "hold")),
            "selected_forecast": selected_forecast,
            "expected_return_direction_label": _expected_return_direction_label(selected_forecast),
            "direction_alignment": _direction_alignment(
                int(prediction.get("selected_direction", 0)),
                selected_forecast,
            ),
            "rows": forecast_rows,
        },
        "project_assessment": {
            "stage": project_stage,
            "summary": _project_value_summary(validation, selected_forecast, project_stage),
            "limitations": _limitations(metrics, leaderboard),
        },
    }


def _load_source_summary(metrics: dict[str, object]) -> dict[str, object] | None:
    source = metrics.get("source")
    if not isinstance(source, dict):
        return None
    source_path = source.get("path")
    if not isinstance(source_path, str):
        return {"kind": str(source.get("kind", "unknown"))}

    path = Path(source_path)
    if not path.exists():
        return {"kind": str(source.get("kind", "unknown")), "path": source_path}

    first_timestamp = None
    last_timestamp = None
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = row.get("timestamp")
            if not timestamp:
                continue
            row_count += 1
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

    summary = {
        "kind": str(source.get("kind", "unknown")),
        "path": source_path,
        "row_count": row_count,
    }
    if first_timestamp:
        summary["start_utc"] = first_timestamp
        summary["start_jst"] = _to_jst_string(first_timestamp)
    if last_timestamp:
        summary["end_utc"] = last_timestamp
        summary["end_jst"] = _to_jst_string(last_timestamp)
    if first_timestamp and last_timestamp:
        start = _parse_iso_timestamp(first_timestamp)
        end = _parse_iso_timestamp(last_timestamp)
        summary["span_days"] = round((end - start).total_seconds() / 86400.0, 2)
    return summary


def _load_leaderboard(manifest: dict[str, object] | None) -> list[dict[str, object]]:
    if not manifest:
        return []
    leaderboard_path = manifest.get("leaderboard_path")
    if not isinstance(leaderboard_path, str):
        return []
    path = Path(leaderboard_path)
    if not path.exists():
        return []
    payload = load_json(path)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_previous_metrics(manifest: dict[str, object] | None) -> dict[str, object] | None:
    if not manifest:
        return None
    archived_previous_dir = manifest.get("archived_previous_current_dir")
    if not isinstance(archived_previous_dir, str):
        return None
    path = Path(archived_previous_dir) / "metrics.json"
    if not path.exists():
        return None
    return load_json(path)


def _build_forecast_rows(
    forecast_summary: dict[str, object],
    prediction: dict[str, object],
) -> list[dict[str, object]]:
    raw_rows = forecast_summary.get("forecast_rows")
    if isinstance(raw_rows, list) and raw_rows:
        return [dict(row) for row in raw_rows if isinstance(row, dict)]

    current_close = float(prediction.get("current_close", 0.0))
    predicted_closes = dict(prediction.get("predicted_closes", {}))
    uncertainties = dict(prediction.get("uncertainties", {}))
    expected_log_returns = dict(prediction.get("expected_log_returns", {}))
    anchor_time = _parse_iso_timestamp(str(prediction["anchor_time"]))
    rows: list[dict[str, object]] = []
    for horizon_key in sorted(predicted_closes, key=lambda key: int(key)):
        horizon = int(horizon_key)
        expected_log_return = float(expected_log_returns.get(horizon_key, 0.0))
        uncertainty = float(uncertainties.get(horizon_key, 0.0))
        predicted_close = float(predicted_closes[horizon_key])
        rows.append(
            {
                "horizon_4h": horizon,
                "forecast_time_utc": (anchor_time + timedelta(hours=4 * horizon)).isoformat(),
                "expected_log_return": expected_log_return,
                "expected_return_pct": (predicted_close / max(current_close, 1e-6)) - 1.0,
                "predicted_close": predicted_close,
                "uncertainty": uncertainty,
                "one_sigma_low_close": current_close * pow(2.718281828459045, expected_log_return - uncertainty),
                "one_sigma_high_close": current_close * pow(2.718281828459045, expected_log_return + uncertainty),
            }
        )
    return rows


def _build_comparison(
    validation: dict[str, object],
    previous_metrics: dict[str, object] | None,
) -> dict[str, object] | None:
    if not previous_metrics:
        return None

    previous_validation = previous_metrics.get("validation_metrics")
    if not isinstance(previous_validation, dict):
        return None

    metrics_to_compare = (
        "project_value_score",
        "utility_score",
        "selection_precision",
        "coverage_at_target_precision",
        "value_capture_ratio",
        "profit_factor",
        "signal_sortino",
        "max_drawdown",
    )
    deltas = []
    for metric_name in metrics_to_compare:
        current_value = validation.get(metric_name)
        previous_value = previous_validation.get(metric_name)
        if not isinstance(current_value, (int, float)) or not isinstance(previous_value, (int, float)):
            continue
        deltas.append(
            {
                "metric": metric_name,
                "current": float(current_value),
                "previous": float(previous_value),
                "delta": float(current_value) - float(previous_value),
            }
        )

    return {
        "previous_best_validation_loss": float(previous_metrics.get("best_validation_loss", 0.0)),
        "deltas": deltas,
    }


def _extract_best_params(
    manifest: dict[str, object] | None,
    config: dict[str, object],
) -> dict[str, object]:
    if manifest and isinstance(manifest.get("best_candidate"), dict):
        best_candidate = dict(manifest["best_candidate"])
        return {
            key: best_candidate[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "hidden_dim",
                "dropout",
                "weight_decay",
            )
            if key in best_candidate
        }
    return {
        key: config[key]
        for key in (
            "epochs",
            "batch_size",
            "learning_rate",
            "hidden_dim",
            "dropout",
            "weight_decay",
        )
        if key in config
    }


def _project_stage(validation: dict[str, object]) -> str:
    if not bool(validation.get("precision_feasible", False)):
        return "low-confidence"
    score = float(validation.get("project_value_score", 0.0))
    if score >= 0.70:
        return "pre-production candidate"
    if score >= 0.55:
        return "research-positive"
    if score >= 0.40:
        return "exploratory"
    return "low-confidence"


def _project_value_summary(
    validation: dict[str, object],
    selected_forecast: dict[str, object] | None,
    project_stage: str,
) -> str:
    project_value_score = float(validation.get("project_value_score", 0.0))
    utility_score = float(validation.get("utility_score", 0.0))
    selection_precision = float(validation.get("selection_precision", 0.0))
    precision_feasible = bool(validation.get("precision_feasible", False))
    capture_ratio = float(validation.get("value_capture_ratio", 0.0))
    sortino = float(validation.get("signal_sortino", 0.0))
    forecast_text = ""
    if selected_forecast is not None:
        forecast_text = (
            f" 選択 horizon は {int(selected_forecast['horizon_4h'])} 本先で、"
            f"期待収益率は {float(selected_forecast['expected_return_pct']) * 100:.2f}% と推定された。"
        )
    return (
        f"project_value_score={project_value_score:.4f}、utility_score={utility_score:.4f} により、"
        f"本 run の段階評価は `{project_stage}` である。"
        f" precision_feasible={precision_feasible}、selection_precision={selection_precision:.4f}、"
        f"value_capture_ratio={capture_ratio:.4f}、"
        f"signal_sortino={sortino:.4f} が価値面の中心指標となる。"
        f"{forecast_text}"
    )


def _limitations(metrics: dict[str, object], leaderboard: Sequence[dict[str, object]]) -> list[str]:
    validation_samples = int(metrics.get("validation_samples", 0))
    sample_count = int(metrics.get("sample_count", 0))
    return [
        f"検証サンプルは {validation_samples} 件で、学習全体 {sample_count} 件に対してまだ小さい。",
        f"ハイパーパラメータ探索は {len(leaderboard)} 候補の近傍探索であり、広域探索ではない。",
        "project_value_score は validation ベースの内部指標であり、実約定ベースの live PnL ではない。",
    ]


def _infer_instrument(metrics: dict[str, object], output_dir: Path) -> str:
    source = metrics.get("source")
    if isinstance(source, dict) and isinstance(source.get("path"), str):
        lowered = str(source["path"]).lower()
        if "xauusd" in lowered:
            return "XAUUSD"
    lowered_output = str(output_dir).lower()
    if "xauusd" in lowered_output:
        return "XAUUSD"
    return "SignalCascade"


def _render_markdown_report(analysis: dict[str, object]) -> str:
    dataset = dict(analysis["dataset"])
    training = dict(analysis["training"])
    validation = dict(analysis["validation_metrics"])
    forecast = dict(analysis["forecast"])
    selected_forecast = forecast.get("selected_forecast")
    comparison = analysis.get("comparison_to_previous")
    leaderboard_top = list(analysis.get("leaderboard_top", []))
    project_assessment = dict(analysis["project_assessment"])
    source = dataset.get("source") or {}

    lines = [
        f"# SignalCascade {analysis['instrument']} Research Report",
        "",
        "## Abstract",
        (
            f"本レポートは、{analysis['generated_at_jst']} 時点の最新成果物を用いて "
            f"`SignalCascade` の再学習結果を再評価したものである。"
            f" 採用モデルの `project_value_score` は {float(validation.get('project_value_score', 0.0)):.4f}、"
            f" `utility_score` は {float(validation.get('utility_score', 0.0)):.4f} であり、"
            f" 段階評価は `{project_assessment['stage']}` となった。"
            f" 選択 horizon は {forecast['selected_horizon']} 本先、"
            f" overlay 判定は `{forecast['overlay_action']}`、"
            f" ポジションは {float(forecast['position']):.4f} である。"
        ),
        "",
        "## 1. Experimental Setup",
        f"- 生成時刻: `{analysis['generated_at_jst']}`",
        f"- 成果物: `{analysis['artifact_dir']}`",
        f"- サンプル数: `train={dataset['train_samples']}`, `validation={dataset['validation_samples']}`, `total={dataset['sample_count']}`",
        f"- 候補数: `{training['candidate_count']}`",
        f"- best validation loss: `{float(training['best_validation_loss']):.6f}` at epoch `{float(training['best_epoch']):.0f}`",
    ]

    if source:
        lines.extend(
            [
                f"- 入力ソース: `{source.get('path', 'n/a')}`",
                f"- 期間: `{source.get('start_jst', 'n/a')}` から `{source.get('end_jst', 'n/a')}`",
                f"- 行数: `used={dataset['source_rows_used']}`, `original={dataset['source_rows_original']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 2. Metric System",
            "- `utility_score`: precision, coverage, capture, overlay F1, directional accuracy, drawdown, calibration の複合値。",
            "- `project_value_score`: utility に加えて profit factor, sortino, calibration を強めた事業価値指標。",
            "- `profit_factor`: 利益総額と損失総額の比率。数値安定化のため `10.0` で上限 clip する。",
            "- `signal_sortino`: 下方リスクで調整した価値指標。数値安定化のため `10.0` で上限 clip する。",
            "- `selection_brier_score`: 採用確率の calibration 誤差。0 に近いほど良い。",
            "",
            "## 3. Validation Results",
            _table(
                headers=("Metric", "Value"),
                rows=[
                    ("project_value_score", _fmt(validation.get("project_value_score"))),
                    ("utility_score", _fmt(validation.get("utility_score"))),
                    ("precision_feasible", _fmt(validation.get("precision_feasible"))),
                    ("threshold_calibration_feasible", _fmt(validation.get("threshold_calibration_feasible"))),
                    ("selection_precision", _fmt(validation.get("selection_precision"))),
                    ("selection_support", _fmt(validation.get("selection_support"))),
                    ("coverage_at_target_precision", _fmt(validation.get("coverage_at_target_precision"))),
                    ("value_capture_ratio", _fmt(validation.get("value_capture_ratio"))),
                    ("profit_factor", _fmt(validation.get("profit_factor"))),
                    ("signal_sortino", _fmt(validation.get("signal_sortino"))),
                    ("selection_brier_score", _fmt(validation.get("selection_brier_score"))),
                    ("max_drawdown", _fmt(validation.get("max_drawdown"))),
                ],
            ),
        ]
    )

    if comparison and comparison.get("deltas"):
        comparison_rows = [
            (
                str(row["metric"]),
                _signed_fmt(row["delta"]),
                _fmt(row["previous"]),
                _fmt(row["current"]),
            )
            for row in comparison["deltas"]
        ]
        lines.extend(
            [
                "",
                "## 4. Delta Vs Previous Current",
                _table(
                    headers=("Metric", "Delta", "Previous", "Current"),
                    rows=comparison_rows,
                ),
            ]
        )

    if leaderboard_top:
        leaderboard_rows = [
            (
                str(row.get("candidate", "n/a")),
                _fmt(row.get("project_value_score")),
                _fmt(row.get("utility_score")),
                _fmt(row.get("selection_precision")),
                _fmt(row.get("value_capture_ratio")),
                _fmt(row.get("profit_factor")),
            )
            for row in leaderboard_top
        ]
        lines.extend(
            [
                "",
                "## 5. Hyperparameter Optimization",
                _table(
                    headers=(
                        "Candidate",
                        "Project Value",
                        "Utility",
                        "Precision",
                        "Capture",
                        "Profit Factor",
                    ),
                    rows=leaderboard_rows,
                ),
                f"採用パラメータ: `{training['best_params']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 6. Forecast Estimation",
            f"- Anchor time JST: `{forecast['anchor_time_jst']}`",
            f"- Anchor close: `{float(forecast['anchor_close']):.4f}`",
            f"- Selected direction classifier: `{forecast['selected_direction_label']}`",
            f"- Expected return direction: `{forecast['expected_return_direction_label']}`",
            f"- Direction alignment: `{forecast['direction_alignment']}`",
            f"- Accepted signal: `{forecast['accepted_signal']}`",
            f"- Selection probability / threshold: `{float(forecast['selection_probability']):.4f}` / `{_fmt(forecast['selection_threshold'])}`",
            f"- Hold probability / threshold: `{float(forecast['hold_probability']):.4f}` / `{float(forecast['hold_threshold']):.4f}`",
        ]
    )

    if isinstance(selected_forecast, dict):
        lines.append(
            (
                f"- 選択 forecast: `{int(selected_forecast['horizon_4h'])}` 本先 "
                f"(`{_to_jst_string(str(selected_forecast['forecast_time_utc']))}`), "
                f"予測終値 `{float(selected_forecast['predicted_close']):.4f}`, "
                f"期待収益率 `{float(selected_forecast['expected_return_pct']) * 100:.2f}%`, "
                f"1σ帯 `["
                f"{float(selected_forecast['one_sigma_low_close']):.4f}, "
                f"{float(selected_forecast['one_sigma_high_close']):.4f}]`"
            )
        )

    forecast_rows = [
        (
            str(row["horizon_4h"]),
            _to_jst_string(str(row["forecast_time_utc"])),
            f"{float(row['expected_return_pct']) * 100:.2f}%",
            _fmt(row["predicted_close"]),
            f"{_fmt(row['one_sigma_low_close'])} .. {_fmt(row['one_sigma_high_close'])}",
        )
        for row in forecast["rows"]
    ]
    lines.extend(
        [
            _table(
                headers=("H(4h)", "Forecast Time JST", "Expected Return", "Pred Close", "1σ Band"),
                rows=forecast_rows,
            ),
            "",
            "## 7. Project Value Assessment",
            str(project_assessment["summary"]),
            "",
            "### Limitations",
        ]
    )
    lines.extend([f"- {item}" for item in project_assessment["limitations"]])
    lines.extend(
        [
            "",
            "## Conclusion",
            (
                f"現時点の `project_value_score={float(validation.get('project_value_score', 0.0)):.4f}` は、"
                f"本プロジェクトが `{project_assessment['stage']}` の段階にあることを示す。"
                " 価値指標の中心は `selection_precision`, `value_capture_ratio`, `profit_factor`, "
                "`signal_sortino`, `selection_brier_score` であり、今後は validation の拡張と"
                " forward simulation を通じて外部妥当性を詰めるのが次段階である。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    rendered = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rendered.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(rendered)


def _direction_label(direction: int) -> str:
    if direction > 0:
        return "long"
    if direction < 0:
        return "short"
    return "flat"


def _expected_return_direction_label(selected_forecast: dict[str, object] | None) -> str:
    if not isinstance(selected_forecast, dict):
        return "flat"
    expected_return = float(selected_forecast.get("expected_return_pct", 0.0))
    if expected_return > 0.0:
        return "long"
    if expected_return < 0.0:
        return "short"
    return "flat"


def _direction_alignment(
    selected_direction: int,
    selected_forecast: dict[str, object] | None,
) -> bool:
    expected_direction = _expected_return_direction_label(selected_forecast)
    classifier_direction = _direction_label(selected_direction)
    return classifier_direction == expected_direction


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _signed_fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+0.6f}"
    return str(value)


def _parse_iso_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _to_jst_string(value: str) -> str:
    parsed = _parse_iso_timestamp(value)
    return parsed.astimezone(JST).isoformat()
