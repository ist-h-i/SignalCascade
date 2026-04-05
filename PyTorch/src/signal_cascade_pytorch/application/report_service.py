from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..infrastructure.persistence import load_json, save_json

JST = timezone(timedelta(hours=9))
UTC = timezone.utc
METRICS_SCHEMA_VERSION = 4
ANALYSIS_SCHEMA_VERSION = 5


def generate_research_report(
    output_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    metrics = load_json(output_dir / "metrics.json")
    prediction = load_json(output_dir / "prediction.json")
    config = load_json(output_dir / "config.json")
    diagnostics_summary = (
        load_json(output_dir / "validation_summary.json")
        if (output_dir / "validation_summary.json").exists()
        else {}
    )
    source_payload = (
        load_json(output_dir / "source.json") if (output_dir / "source.json").exists() else {}
    )
    analysis = _build_analysis(
        output_dir,
        metrics,
        prediction,
        config,
        diagnostics_summary,
        source_payload,
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
    diagnostics_summary: dict[str, object],
    source_payload: dict[str, object],
) -> dict[str, object]:
    validation = dict(metrics.get("validation_metrics", {}))
    generated_at_utc = datetime.now(UTC)
    current_close = float(prediction.get("current_close", 0.0))
    policy_horizon = int(prediction.get("policy_horizon", prediction.get("proposed_horizon", 0)))
    expected_log_returns = dict(prediction.get("expected_log_returns", {}))
    predicted_closes = dict(
        prediction.get("median_predicted_closes", prediction.get("predicted_closes", {}))
    )
    uncertainties = dict(prediction.get("uncertainties", {}))
    stateful_evaluation = dict(diagnostics_summary.get("stateful_evaluation", {}))
    policy_calibration_sweep = list(diagnostics_summary.get("policy_calibration_sweep", []))
    policy_calibration_summary = dict(diagnostics_summary.get("policy_calibration_summary", {}))
    rows = []
    anchor_time = datetime.fromisoformat(str(prediction["anchor_time"]))
    for horizon_key in sorted(expected_log_returns, key=lambda value: int(value)):
        horizon = int(horizon_key)
        expected_log_return = float(expected_log_returns[horizon_key])
        predicted_close = float(predicted_closes[horizon_key])
        uncertainty = float(uncertainties[horizon_key])
        rows.append(
            {
                "horizon_4h": horizon,
                "forecast_time_utc": (anchor_time + timedelta(hours=4 * horizon)).isoformat(),
                "expected_log_return": expected_log_return,
                "expected_return_pct": math.exp(expected_log_return) - 1.0,
                "predicted_close": predicted_close,
                "uncertainty": uncertainty,
            }
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc.isoformat(),
        "generated_at_jst": generated_at_utc.astimezone(JST).isoformat(),
        "artifact_dir": str(output_dir),
        "artifact_provenance": _summarize_artifact_provenance(source_payload),
        "artifact_versions": {
            "metrics": int(metrics.get("schema_version", 0)),
            "prediction": int(prediction.get("schema_version", 0)),
            "diagnostics": int(diagnostics_summary.get("schema_version", 0)),
        },
        "policy_mode": metrics.get("policy_mode", "shape_aware_profit_maximization"),
        "dataset": {
            "sample_count": int(metrics.get("sample_count", 0)),
            "effective_sample_count": int(metrics.get("effective_sample_count", 0)),
            "train_samples": int(metrics.get("train_samples", 0)),
            "validation_samples": int(metrics.get("validation_samples", 0)),
            "purged_samples": int(metrics.get("purged_samples", 0)),
            "source_rows_original": int(metrics.get("source_rows_original", 0)),
            "source_rows_used": int(metrics.get("source_rows_used", 0)),
        },
        "training": {
            "best_validation_loss": float(metrics.get("best_validation_loss", 0.0)),
            "best_epoch": float(metrics.get("best_epoch", 0.0)),
            "epochs": int(config.get("epochs", 0)),
            "warmup_epochs": int(config.get("warmup_epochs", 0)),
            "shape_classes": int(config.get("shape_classes", 0)),
            "state_dim": int(config.get("state_dim", 0)),
        },
        "validation_metrics": validation,
        "stateful_evaluation": stateful_evaluation,
        "policy_calibration_sweep": policy_calibration_sweep,
        "policy_calibration_summary": policy_calibration_summary,
        "forecast": {
            "anchor_time_utc": str(prediction["anchor_time"]),
            "anchor_time_jst": anchor_time.astimezone(JST).isoformat(),
            "anchor_close": current_close,
            "policy_horizon": policy_horizon,
            "executed_horizon": prediction.get("executed_horizon"),
            "previous_position": float(prediction.get("previous_position", 0.0)),
            "position": float(prediction.get("position", 0.0)),
            "trade_delta": float(prediction.get("trade_delta", 0.0)),
            "no_trade_band_hit": bool(prediction.get("no_trade_band_hit", False)),
            "tradeability_gate": float(prediction.get("tradeability_gate", 0.0)),
            "shape_entropy": float(prediction.get("shape_entropy", 0.0)),
            "policy_score": float(prediction.get("policy_score", 0.0)),
            "rows": rows,
        },
        "project_assessment": {
            "summary": _build_summary(validation, prediction),
        },
    }


def _build_summary(
    validation: dict[str, object],
    prediction: dict[str, object],
) -> str:
    return (
        "新 spec の主経路は threshold policy ではなく "
        "`shape -> return distribution -> q_t*` です。"
        f" validation では average_log_wealth={float(validation.get('average_log_wealth', 0.0)):.4f},"
        f" realized_pnl_per_anchor={float(validation.get('realized_pnl_per_anchor', 0.0)):.4f},"
        f" cvar_tail_loss={float(validation.get('cvar_tail_loss', 0.0)):.4f},"
        f" no_trade_band_hit_rate={float(validation.get('no_trade_band_hit_rate', 0.0)):.4f}。"
        f" latest policy_horizon={int(prediction.get('policy_horizon', 0))},"
        f" position={float(prediction.get('position', 0.0)):.4f},"
        f" tradeability_gate={float(prediction.get('tradeability_gate', 0.0)):.4f}。"
    )


def _render_markdown_report(analysis: dict[str, object]) -> str:
    dataset = analysis["dataset"]
    training = analysis["training"]
    validation = analysis["validation_metrics"]
    artifact_provenance = analysis.get("artifact_provenance", {})
    stateful_evaluation = analysis["stateful_evaluation"]
    policy_calibration_sweep = analysis["policy_calibration_sweep"]
    policy_calibration_summary = analysis["policy_calibration_summary"]
    forecast = analysis["forecast"]
    rows = forecast["rows"]
    selected_sweep_row = (
        dict(policy_calibration_summary.get("selected_row", {}))
        if policy_calibration_summary.get("selected_row") is not None
        else None
    )
    artifact_lines = [
        "## Artifact",
        f"- kind: `{artifact_provenance.get('artifact_kind', 'legacy')}`",
        f"- artifact id: `{artifact_provenance.get('artifact_id', '-')}`",
        f"- source kind / path: `{artifact_provenance.get('source_kind', '-')}` / `{artifact_provenance.get('source_path', '-')}`",
        f"- config origin: `{artifact_provenance.get('config_origin', '-')}`",
    ]
    if artifact_provenance.get("parent_artifact_dir") is not None:
        artifact_lines.append(
            f"- parent artifact dir: `{artifact_provenance['parent_artifact_dir']}`"
        )
    if artifact_provenance.get("parent_artifact_id") is not None:
        artifact_lines.append(
            f"- parent artifact id: `{artifact_provenance['parent_artifact_id']}`"
        )
    if artifact_provenance.get("git_head") is not None:
        artifact_lines.append(
            f"- git head / dirty: `{artifact_provenance['git_head']}` / `{artifact_provenance.get('git_dirty', False)}`"
        )
    if artifact_provenance.get("git_tree_sha") is not None:
        artifact_lines.append(
            f"- git tree sha: `{artifact_provenance['git_tree_sha']}`"
        )
    if artifact_provenance.get("data_snapshot_sha256") is not None:
        artifact_lines.append(
            f"- data snapshot sha256: `{artifact_provenance['data_snapshot_sha256'][:16]}`"
        )
    if artifact_provenance.get("sub_artifact_materialization") is not None:
        artifact_lines.append(
            f"- sub-artifact lineage: `{artifact_provenance['sub_artifact_materialization']}`"
        )
    row_lines = "\n".join(
        f"- h={int(row['horizon_4h'])}: expected_log_return={float(row['expected_log_return']):.4f}, "
        f"expected_return_pct={float(row['expected_return_pct']):.4f}, "
        f"predicted_close={float(row['predicted_close']):.4f}, "
        f"uncertainty={float(row['uncertainty']):.4f}"
        for row in rows
    )
    return "\n".join(
        [
            "# SignalCascade Research Report",
            "",
            f"- Generated (JST): `{analysis['generated_at_jst']}`",
            f"- Policy mode: `{analysis['policy_mode']}`",
            "",
            *artifact_lines,
            "",
            "## Dataset",
            f"- sample_count: `{dataset['sample_count']}`",
            f"- effective_sample_count / purged_samples: `{dataset['effective_sample_count']}` / `{dataset['purged_samples']}`",
            f"- train / validation: `{dataset['train_samples']}` / `{dataset['validation_samples']}`",
            f"- source_rows_original / used: `{dataset['source_rows_original']}` / `{dataset['source_rows_used']}`",
            "",
            "## Training",
            f"- best_validation_loss: `{float(training['best_validation_loss']):.6f}`",
            f"- best_epoch: `{float(training['best_epoch']):.0f}`",
            f"- epochs / warmup_epochs: `{training['epochs']}` / `{training['warmup_epochs']}`",
            f"- shape_classes / state_dim: `{training['shape_classes']}` / `{training['state_dim']}`",
            "",
            "## Validation",
            f"- average_log_wealth: `{float(validation.get('average_log_wealth', 0.0)):.6f}`",
            f"- realized_pnl_per_anchor: `{float(validation.get('realized_pnl_per_anchor', 0.0)):.6f}`",
            f"- cvar_tail_loss: `{float(validation.get('cvar_tail_loss', 0.0)):.6f}`",
            f"- max_drawdown: `{float(validation.get('max_drawdown', 0.0)):.6f}`",
            f"- no_trade_band_hit_rate: `{float(validation.get('no_trade_band_hit_rate', 0.0)):.6f}`",
            f"- exact_smooth_horizon_agreement / no_trade_agreement: "
            f"`{float(validation.get('exact_smooth_horizon_agreement', 0.0)):.6f}` / "
            f"`{float(validation.get('exact_smooth_no_trade_agreement', 0.0)):.6f}`",
            f"- exact_smooth_position_mae / utility_regret: "
            f"`{float(validation.get('exact_smooth_position_mae', 0.0)):.6f}` / "
            f"`{float(validation.get('exact_smooth_utility_regret', 0.0)):.6f}`",
            f"- shape_gate_usage: `{float(validation.get('shape_gate_usage', 0.0)):.6f}`",
            f"- expert_entropy: `{float(validation.get('expert_entropy', 0.0)):.6f}`",
            f"- mu_calibration / sigma_calibration: "
            f"`{float(validation.get('mu_calibration', 0.0)):.6f}` / "
            f"`{float(validation.get('sigma_calibration', 0.0)):.6f}`",
            f"- log_wealth_clamp_hit_rate / state_reset_mode: "
            f"`{float(validation.get('log_wealth_clamp_hit_rate', 0.0)):.6f}` / "
            f"`{validation.get('state_reset_mode', '-')}`",
            f"- project_value_score / utility_score: "
            f"`{float(validation.get('project_value_score', 0.0)):.6f}` / "
            f"`{float(validation.get('utility_score', 0.0)):.6f}`",
            "",
            "## Evaluation",
            f"- carry_on average_log_wealth: `{float(stateful_evaluation.get('carry_on', {}).get('average_log_wealth', 0.0)):.6f}`",
            f"- reset_each_example average_log_wealth: `{float(stateful_evaluation.get('reset_each_example', {}).get('average_log_wealth', 0.0)):.6f}`",
            f"- reset_each_session_or_window average_log_wealth: "
            f"`{float(stateful_evaluation.get('reset_each_session_or_window', {}).get('average_log_wealth', 0.0)):.6f}`",
            f"- policy sweep rows / pareto_optimal: "
            f"`{int(policy_calibration_summary.get('row_count', len(policy_calibration_sweep)))}` / "
            f"`{int(policy_calibration_summary.get('pareto_optimal_count', 0))}`",
            f"- policy sweep selection basis / version: "
            f"`{policy_calibration_summary.get('selection_basis', '-')}` / "
            f"`{policy_calibration_summary.get('selection_rule_version', '-')}`",
            (
                f"- selected policy sweep: reset=`{selected_sweep_row['state_reset_mode']}`, "
                f"cost x`{float(selected_sweep_row['cost_multiplier']):.2f}`, "
                f"gamma x`{float(selected_sweep_row['gamma_multiplier']):.2f}`, "
                f"min_sigma=`{float(selected_sweep_row['min_policy_sigma']):.6f}`, "
                f"log_wealth=`{float(selected_sweep_row['average_log_wealth']):.6f}`"
                if selected_sweep_row is not None
                else "- selected policy sweep: none"
            ),
            f"- selected row key: `{policy_calibration_summary.get('selected_row_key', '-')}`",
            (
                f"- policy sweep rows sha256: "
                f"`{str(policy_calibration_summary.get('policy_calibration_rows_sha256'))[:16]}`"
                if policy_calibration_summary.get("policy_calibration_rows_sha256") is not None
                else "- policy sweep rows sha256: `-`"
            ),
            "",
            "## Forecast",
            f"- anchor_time_utc / jst: `{forecast['anchor_time_utc']}` / `{forecast['anchor_time_jst']}`",
            f"- anchor_close: `{float(forecast['anchor_close']):.4f}`",
            f"- policy_horizon / executed_horizon: `{forecast['policy_horizon']}` / `{forecast['executed_horizon']}`",
            f"- previous_position / position / trade_delta: "
            f"`{float(forecast['previous_position']):.4f}` / `{float(forecast['position']):.4f}` / `{float(forecast['trade_delta']):.4f}`",
            f"- no_trade_band_hit: `{forecast['no_trade_band_hit']}`",
            f"- tradeability_gate / shape_entropy / policy_score: "
            f"`{float(forecast['tradeability_gate']):.4f}` / `{float(forecast['shape_entropy']):.4f}` / `{float(forecast['policy_score']):.4f}`",
            row_lines or "- forecast rows: none",
            "",
            "## Assessment",
            f"- {analysis['project_assessment']['summary']}",
        ]
    )


def _summarize_artifact_provenance(source_payload: dict[str, object]) -> dict[str, object]:
    git_payload = source_payload.get("git")
    git = git_payload if isinstance(git_payload, dict) else {}
    sub_artifacts_payload = source_payload.get("sub_artifacts")
    sub_artifacts = sub_artifacts_payload if isinstance(sub_artifacts_payload, dict) else {}
    materialization = ", ".join(
        f"{name}:{details.get('materialization', '-')}"
        for name, details in sorted(sub_artifacts.items())
        if isinstance(details, dict)
    )
    return {
        "artifact_schema_version": source_payload.get("artifact_schema_version"),
        "artifact_kind": source_payload.get("artifact_kind"),
        "artifact_id": source_payload.get("artifact_id"),
        "artifact_dir": source_payload.get("artifact_dir"),
        "parent_artifact_dir": source_payload.get("parent_artifact_dir"),
        "parent_artifact_id": source_payload.get("parent_artifact_id"),
        "source_kind": source_payload.get("kind"),
        "source_path": source_payload.get("path"),
        "data_snapshot_sha256": source_payload.get("data_snapshot_sha256"),
        "config_sha256": source_payload.get("config_sha256"),
        "config_origin": source_payload.get("config_origin"),
        "state_reset_boundary_spec_version": source_payload.get(
            "state_reset_boundary_spec_version"
        ),
        "git_head": git.get("head"),
        "git_head": git.get("head", git.get("git_commit_sha", git.get("commit_sha"))),
        "git_commit_sha": git.get("git_commit_sha", git.get("commit_sha", git.get("head"))),
        "git_tree_sha": git.get("git_tree_sha", git.get("tree_sha")),
        "git_dirty": git.get("git_dirty", git.get("dirty")),
        "git_dirty_patch_sha256": git.get("dirty_patch_sha256"),
        "sub_artifact_materialization": materialization or None,
    }
