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
    proposed_horizon = _optional_int(
        prediction.get("proposed_horizon", prediction.get("selected_horizon"))
    )
    accepted_horizon = _optional_int(prediction.get("accepted_horizon"))
    proposed_forecast = next(
        (
            row
            for row in forecast_rows
            if proposed_horizon is not None and int(row["horizon_4h"]) == proposed_horizon
        ),
        forecast_rows[0] if forecast_rows else None,
    )
    if proposed_horizon is None:
        proposed_forecast = None
    comparison = _build_comparison(validation, previous_metrics)
    generated_at_utc = datetime.now(UTC)
    generated_at_jst = generated_at_utc.astimezone(JST)
    project_stage = _project_stage(validation)
    learning_diagnostics = _build_learning_diagnostics(metrics)

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
            "selector_head_brier_score": "selector head 自体の calibration 誤差。小さいほど良い。",
            "best_selection_lcb": "infeasible run 同士を比較するための research 指標。候補 threshold 群で最大の Wilson LCB。",
            "alignment_rate": "horizon ごとに `sign(mu)` と actionable sign が一致した率。",
            "pre_threshold_capture": "threshold を課す前の仮想 capture。signal 自体に価値があるかを閾値分離で観察する。",
            "actionable_edge_rate": "cost 控除後に positive edge が残った horizon row の比率。",
            "acceptance_score_source": "実際の accept/reject に使った score source。",
        },
        "learning_diagnostics": learning_diagnostics,
        "learning_findings": _build_learning_findings(learning_diagnostics, validation),
        "validation_metrics": validation,
        "horizon_diagnostics": _build_horizon_diagnostics(validation),
        "leaderboard_top": top_candidates,
        "comparison_to_previous": comparison,
        "forecast": {
            "anchor_time_utc": str(prediction["anchor_time"]),
            "anchor_time_jst": _to_jst_string(str(prediction["anchor_time"])),
            "anchor_close": float(prediction.get("current_close", 0.0)),
            "proposed_horizon": proposed_horizon,
            "accepted_horizon": accepted_horizon,
            "selected_direction": int(prediction.get("selected_direction", 0)),
            "selected_direction_label": (
                "none"
                if proposed_horizon is None
                else _direction_label(int(prediction.get("selected_direction", 0)))
            ),
            "accepted_signal": bool(prediction.get("accepted_signal", False)),
            "position": float(prediction.get("position", 0.0)),
            "selection_probability": float(prediction.get("selection_probability", 0.0)),
            "selection_score": float(
                prediction.get("selection_score", prediction.get("selection_probability", 0.0))
            ),
            "acceptance_score_source": str(
                _first_value(
                    validation,
                    "acceptance_score_source",
                    "selection_score_source",
                )
                or config.get("selection_score_source", "selector_probability")
            ),
            "selection_threshold": (
                None
                if prediction.get("selection_threshold") is None
                else float(prediction.get("selection_threshold", 0.0))
            ),
            "threshold_status": str(prediction.get("threshold_status", "unknown")),
            "threshold_origin": str(prediction.get("threshold_origin", "unknown")),
            "correctness_probability": float(prediction.get("correctness_probability", 0.0)),
            "hold_probability": float(prediction.get("hold_probability", 0.0)),
            "hold_threshold": float(prediction.get("hold_threshold", 0.0)),
            "overlay_action": str(prediction.get("overlay_action", "hold")),
            "proposed_forecast": proposed_forecast,
            "expected_return_direction_label": (
                "none" if proposed_forecast is None else _expected_return_direction_label(proposed_forecast)
            ),
            "direction_alignment": (
                "n/a"
                if proposed_forecast is None
                else _direction_alignment(int(prediction.get("selected_direction", 0)), proposed_forecast)
            ),
            "rows": forecast_rows,
        },
        "project_assessment": {
            "stage": project_stage,
            "summary": _project_value_summary(validation, proposed_forecast, project_stage),
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
        ("project_value_score", ("project_value_score",)),
        ("utility_score", ("utility_score",)),
        ("acceptance_precision", ("acceptance_precision", "selection_precision")),
        ("acceptance_coverage", ("acceptance_coverage", "coverage_at_target_precision")),
        ("best_selection_lcb", ("best_selection_lcb",)),
        ("alignment_rate", ("alignment_rate",)),
        ("value_capture_ratio", ("value_capture_ratio",)),
        ("profit_factor", ("profit_factor",)),
        ("signal_sortino", ("signal_sortino",)),
        ("max_drawdown", ("max_drawdown",)),
    )
    deltas = []
    for metric_name, keys in metrics_to_compare:
        current_value = _first_numeric(validation, *keys)
        previous_value = _first_numeric(previous_validation, *keys)
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


def _build_learning_diagnostics(metrics: dict[str, object]) -> dict[str, object] | None:
    history_raw = metrics.get("history")
    if not isinstance(history_raw, list) or not history_raw:
        return None

    history = [dict(row) for row in history_raw if isinstance(row, dict)]
    if not history:
        return None

    epoch_count = len(history)
    best_epoch = max(1, int(round(float(metrics.get("best_epoch", 1.0)))))
    best_record = next(
        (
            row
            for row in history
            if int(round(float(row.get("epoch", 0.0)))) == best_epoch
        ),
        history[min(best_epoch - 1, epoch_count - 1)],
    )
    first_record = history[0]
    last_record = history[-1]
    best_validation_total = float(best_record.get("validation_total", 0.0))
    last_validation_total = float(last_record.get("validation_total", best_validation_total))
    best_train_total = float(best_record.get("train_total", 0.0))

    return {
        "epoch_count": epoch_count,
        "best_epoch": best_epoch,
        "best_epoch_ratio": best_epoch / max(epoch_count, 1),
        "first_validation_total": float(first_record.get("validation_total", 0.0)),
        "best_validation_total": best_validation_total,
        "last_validation_total": last_validation_total,
        "generalization_gap_at_best": best_validation_total - best_train_total,
        "validation_drift_from_best_to_last": last_validation_total - best_validation_total,
        "early_peak": best_epoch <= max(1, epoch_count // 4),
    }


def _build_learning_findings(
    learning_diagnostics: dict[str, object] | None,
    validation: dict[str, object],
) -> list[str]:
    findings: list[str] = []
    if learning_diagnostics:
        best_epoch = int(learning_diagnostics.get("best_epoch", 0))
        epoch_count = int(learning_diagnostics.get("epoch_count", 0))
        drift = float(learning_diagnostics.get("validation_drift_from_best_to_last", 0.0))
        if bool(learning_diagnostics.get("early_peak")):
            findings.append(
                f"best validation loss は epoch {best_epoch}/{epoch_count} で最小化し、その後の validation total は {drift:+.4f} 悪化した。早期ピーク型で、direction head の汎化がまだ不安定である。"
            )
        else:
            findings.append(
                f"best validation loss は epoch {best_epoch}/{epoch_count} で到達し、best 時点の generalization gap は {float(learning_diagnostics.get('generalization_gap_at_best', 0.0)):+.4f} であった。"
            )

    selection_brier_score = float(
        _first_numeric(validation, "selector_head_brier_score", "selection_brier_score", default=0.0)
    )
    actionable_edge_rate = float(validation.get("actionable_edge_rate", 0.0))
    findings.append(
        f"selector calibration は `selector_head_brier_score={selection_brier_score:.6f}` まで改善した一方、`actionable_edge_rate={actionable_edge_rate:.4f}` のため、cost 控除後に採用可能な edge は validation 上で未出現である。"
    )

    best_selection_lcb = float(validation.get("best_selection_lcb", 0.0))
    support_at_best_lcb = int(round(float(validation.get("support_at_best_lcb", 0.0))))
    precision_at_best_lcb = float(validation.get("precision_at_best_lcb", 0.0))
    findings.append(
        f"threshold search の研究進捗として `best_selection_lcb={best_selection_lcb:.4f}`、`support_at_best_lcb={support_at_best_lcb}`、`precision_at_best_lcb={precision_at_best_lcb:.4f}` を追跡する。feasible 未達でも threshold 面の改善度を比較できる。"
    )

    horizon_diagnostics = _build_horizon_diagnostics(validation)
    if horizon_diagnostics:
        densest = max(horizon_diagnostics, key=lambda row: float(row["nonflat_rate"]))
        aligned = max(horizon_diagnostics, key=lambda row: float(row["alignment_rate"]))
        findings.append(
            f"ラベル密度は horizon {densest['horizon']} で最大 (`nonflat_rate={float(densest['nonflat_rate']):.4f}`)、符号整合は horizon {aligned['horizon']} が最大 (`alignment_rate={float(aligned['alignment_rate']):.4f}`) で、長短 horizon の学習難度が分かれている。"
        )

    return findings


def _build_horizon_diagnostics(validation: dict[str, object]) -> list[dict[str, object]]:
    raw = validation.get("horizon_diagnostics")
    if not isinstance(raw, dict):
        return []

    rows: list[dict[str, object]] = []
    for horizon_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "horizon": int(horizon_key),
                "nonflat_rate": float(payload.get("nonflat_rate", 0.0)),
                "up_rate": float(payload.get("up_rate", 0.0)),
                "down_rate": float(payload.get("down_rate", 0.0)),
                "alignment_rate": float(payload.get("alignment_rate", 0.0)),
                "actionable_edge_rate": float(payload.get("actionable_edge_rate", 0.0)),
            }
        )
    rows.sort(key=lambda row: int(row["horizon"]))
    return rows


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
    proposed_forecast: dict[str, object] | None,
    project_stage: str,
) -> str:
    project_value_score = float(validation.get("project_value_score", 0.0))
    utility_score = float(validation.get("utility_score", 0.0))
    acceptance_precision = float(
        _first_numeric(validation, "acceptance_precision", "selection_precision", default=0.0)
    )
    precision_feasible = bool(validation.get("precision_feasible", False))
    capture_ratio = float(validation.get("value_capture_ratio", 0.0))
    sortino = float(validation.get("signal_sortino", 0.0))
    forecast_text = ""
    if proposed_forecast is not None:
        forecast_text = (
            f" proposal horizon は {int(proposed_forecast['horizon_4h'])} 本先で、"
            f"期待収益率は {float(proposed_forecast['expected_return_pct']) * 100:.2f}% と推定された。"
        )
    research_progress = ""
    if not precision_feasible:
        research_progress = (
            f" research progress として best_selection_lcb={float(validation.get('best_selection_lcb', 0.0)):.4f}、"
            f" support_at_best_lcb={float(validation.get('support_at_best_lcb', 0.0)):.0f}、"
            f" tau_at_best_lcb={_fmt(validation.get('tau_at_best_lcb'))}、"
            f" selector_head_brier_score={float(_first_numeric(validation, 'selector_head_brier_score', 'selection_brier_score', default=0.0)):.6f}、"
            f" actionable_edge_rate={float(validation.get('actionable_edge_rate', 0.0)):.4f} を併記する。"
        )
    return (
        f"project_value_score={project_value_score:.4f}、utility_score={utility_score:.4f} により、"
        f"本 run の段階評価は `{project_stage}` である。"
        f" precision_feasible={precision_feasible}、acceptance_precision={acceptance_precision:.4f}、"
        f"value_capture_ratio={capture_ratio:.4f}、"
        f"signal_sortino={sortino:.4f} が価値面の中心指標となる。"
        f"{research_progress}"
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
    proposed_forecast = forecast.get("proposed_forecast")
    comparison = analysis.get("comparison_to_previous")
    leaderboard_top = list(analysis.get("leaderboard_top", []))
    learning_diagnostics = analysis.get("learning_diagnostics") or {}
    learning_findings = list(analysis.get("learning_findings", []))
    horizon_diagnostics = list(analysis.get("horizon_diagnostics", []))
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
            f" `precision_feasible={validation.get('precision_feasible', False)}` で、"
            f" research 進捗として `best_selection_lcb={float(validation.get('best_selection_lcb', 0.0)):.4f}` を得た。"
            f" proposal / accepted horizon は {forecast['proposed_horizon'] if forecast['proposed_horizon'] is not None else 'none'} / "
            f"{forecast['accepted_horizon'] if forecast['accepted_horizon'] is not None else 'none'}、"
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
            "- `selector_head_brier_score`: selector head 自体の calibration 誤差。0 に近いほど良い。",
            "- `best_selection_lcb`: feasible 未達 run の比較用に、threshold 候補群で最大の Wilson LCB を追う研究指標。",
            "- `pre_threshold_capture`: selector threshold を適用する前の仮想 capture。シグナル原石の有無を確認する。",
            "- `alignment_rate` / `actionable_edge_rate`: return head と direction head の整合、およびコスト控除後の実行可能 edge の発生率を監視する。",
            "",
            "## 3. Learning Analysis",
        ]
    )

    if learning_diagnostics:
        lines.append(
            _table(
                headers=("Metric", "Value"),
                rows=[
                    ("best_epoch", _fmt(learning_diagnostics.get("best_epoch"))),
                    ("epoch_count", _fmt(learning_diagnostics.get("epoch_count"))),
                    ("best_epoch_ratio", _fmt(learning_diagnostics.get("best_epoch_ratio"))),
                    ("generalization_gap_at_best", _fmt(learning_diagnostics.get("generalization_gap_at_best"))),
                    (
                        "validation_drift_from_best_to_last",
                        _signed_fmt(learning_diagnostics.get("validation_drift_from_best_to_last")),
                    ),
                    ("early_peak", _fmt(learning_diagnostics.get("early_peak"))),
                ],
            )
        )
    lines.extend([f"- {item}" for item in learning_findings])
    lines.extend(
        [
            "",
            "## 4. Validation Results",
            _table(
                headers=("Metric", "Value"),
                rows=[
                    ("project_value_score", _fmt(validation.get("project_value_score"))),
                    ("utility_score", _fmt(validation.get("utility_score"))),
                    ("precision_feasible", _fmt(validation.get("precision_feasible"))),
                    ("threshold_calibration_feasible", _fmt(validation.get("threshold_calibration_feasible"))),
                    (
                        "acceptance_precision",
                        _fmt(_first_numeric(validation, "acceptance_precision", "selection_precision")),
                    ),
                    (
                        "acceptance_support",
                        _fmt(_first_numeric(validation, "acceptance_support", "selection_support")),
                    ),
                    ("proposal_count", _fmt(validation.get("proposal_count"))),
                    ("accepted_count", _fmt(validation.get("accepted_count"))),
                    ("proposal_coverage", _fmt(validation.get("proposal_coverage"))),
                    ("best_selection_lcb", _fmt(validation.get("best_selection_lcb"))),
                    ("support_at_best_lcb", _fmt(validation.get("support_at_best_lcb"))),
                    ("precision_at_best_lcb", _fmt(validation.get("precision_at_best_lcb"))),
                    ("tau_at_best_lcb", _fmt(validation.get("tau_at_best_lcb"))),
                    (
                        "acceptance_coverage",
                        _fmt(_first_numeric(validation, "acceptance_coverage", "coverage_at_target_precision")),
                    ),
                    ("value_per_anchor", _fmt(validation.get("value_per_anchor"))),
                    ("value_per_proposed", _fmt(validation.get("value_per_proposed"))),
                    ("value_per_accepted", _fmt(validation.get("value_per_accepted"))),
                    ("value_capture_ratio", _fmt(validation.get("value_capture_ratio"))),
                    ("pre_threshold_capture", _fmt(validation.get("pre_threshold_capture"))),
                    ("profit_factor", _fmt(validation.get("profit_factor"))),
                    ("signal_sortino", _fmt(validation.get("signal_sortino"))),
                    ("alignment_rate", _fmt(validation.get("alignment_rate"))),
                    ("actionable_edge_rate", _fmt(validation.get("actionable_edge_rate"))),
                    ("nonflat_rate", _fmt(validation.get("nonflat_rate"))),
                    (
                        "selector_head_brier_score",
                        _fmt(_first_numeric(validation, "selector_head_brier_score", "selection_brier_score")),
                    ),
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
                "## 5. Delta Vs Previous Current",
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
                _fmt(_first_numeric(row, "acceptance_precision", "selection_precision")),
                _fmt(row.get("best_selection_lcb")),
                _fmt(row.get("alignment_rate")),
                _fmt(row.get("value_capture_ratio")),
                _fmt(row.get("profit_factor")),
            )
            for row in leaderboard_top
        ]
        lines.extend(
            [
                "",
                "## 6. Hyperparameter Optimization",
                _table(
                    headers=(
                        "Candidate",
                        "Project Value",
                        "Utility",
                        "Precision",
                        "Best LCB",
                        "Align",
                        "Capture",
                        "Profit Factor",
                    ),
                    rows=leaderboard_rows,
                ),
                f"採用パラメータ: `{training['best_params']}`",
            ]
        )

    if horizon_diagnostics:
        horizon_rows = [
            (
                str(row["horizon"]),
                _fmt(row["nonflat_rate"]),
                _fmt(row["up_rate"]),
                _fmt(row["down_rate"]),
                _fmt(row["alignment_rate"]),
                _fmt(row["actionable_edge_rate"]),
            )
            for row in horizon_diagnostics
        ]
        lines.extend(
            [
                "",
                "## 7. Horizon Diagnostics",
                _table(
                    headers=("H(4h)", "Nonflat", "Up", "Down", "Align", "Actionable Edge"),
                    rows=horizon_rows,
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## 8. Forecast Estimation",
            f"- Anchor time JST: `{forecast['anchor_time_jst']}`",
            f"- Anchor close: `{float(forecast['anchor_close']):.4f}`",
            f"- Proposed / accepted horizon: `{forecast['proposed_horizon']}` / `{forecast['accepted_horizon']}`",
            f"- Proposed direction classifier: `{forecast['selected_direction_label']}`",
            f"- Expected return direction: `{forecast['expected_return_direction_label']}`",
            f"- Direction alignment: `{forecast['direction_alignment']}`",
            f"- Accepted signal: `{forecast['accepted_signal']}`",
            (
                f"- Acceptance score source / score / threshold: "
                f"`{forecast['acceptance_score_source']}` / "
                f"`{float(forecast['selection_score']):.4f}` / "
                f"`{_fmt(forecast['selection_threshold'])}`"
            ),
            f"- Selector probability: `{float(forecast['selection_probability']):.4f}`",
            f"- Threshold status / origin: `{forecast['threshold_status']}` / `{forecast['threshold_origin']}`",
            f"- Hold probability / threshold: `{float(forecast['hold_probability']):.4f}` / `{float(forecast['hold_threshold']):.4f}`",
        ]
    )

    if isinstance(proposed_forecast, dict):
        lines.append(
            (
                f"- Proposal forecast: `{int(proposed_forecast['horizon_4h'])}` 本先 "
                f"(`{_to_jst_string(str(proposed_forecast['forecast_time_utc']))}`), "
                f"予測終値 `{float(proposed_forecast['predicted_close']):.4f}`, "
                f"期待収益率 `{float(proposed_forecast['expected_return_pct']) * 100:.2f}%`, "
                f"1σ帯 `["
                f"{float(proposed_forecast['one_sigma_low_close']):.4f}, "
                f"{float(proposed_forecast['one_sigma_high_close']):.4f}]`"
            )
        )
    elif forecast.get("proposed_horizon") is None:
        lines.append("- Proposal forecast: `none`")

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
            "## 9. Project Value Assessment",
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
                " 価値指標の中心は `acceptance_precision`, `value_capture_ratio`, `profit_factor`, "
                "`signal_sortino`, `selector_head_brier_score` である。現状の主なボトルネックは "
                "`actionable_edge_rate` と `precision_feasible` であり、今後は validation の拡張と"
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


def _first_value(payload: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _first_numeric(payload: dict[str, object], *keys: str, default: float | None = None) -> float | None:
    value = _first_value(payload, *keys)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


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
