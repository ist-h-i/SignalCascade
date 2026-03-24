from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from .config import TrainingConfig
from .dataset_service import _main_mae_threshold, _main_move_threshold
from .policy_service import apply_selection_policy
from .training_service import examples_to_batch, restore_return_units, split_examples
from ..domain.entities import OHLCVBar, TrainingExample
from ..infrastructure.ml.model import SignalCascadeModel
from ..infrastructure.persistence import save_json


def export_review_diagnostics(
    output_dir: Path,
    model: SignalCascadeModel,
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    selection_policy: dict[str, object] | None,
    source_payload: dict[str, object] | None = None,
    source_rows_original: int | None = None,
    source_rows_used: int | None = None,
    base_bars: Sequence[OHLCVBar] | None = None,
) -> dict[str, object]:
    _, validation_examples = split_examples(examples, config)
    diagnostics = build_validation_diagnostics(
        model=model,
        validation_examples=validation_examples,
        config=config,
        selection_policy=selection_policy,
    )

    validation_rows_path = output_dir / "validation_rows.csv"
    threshold_scan_path = output_dir / "threshold_scan.csv"
    horizon_diag_path = output_dir / "horizon_diag.csv"
    summary_path = output_dir / "validation_summary.json"

    _write_csv(validation_rows_path, diagnostics["validation_rows"])
    _write_csv(threshold_scan_path, diagnostics["threshold_scan"])
    _write_csv(horizon_diag_path, diagnostics["horizon_diag"])

    summary = {
        "dataset": _build_dataset_summary(
            examples=examples,
            validation_examples=validation_examples,
            source_payload=source_payload,
            source_rows_original=source_rows_original,
            source_rows_used=source_rows_used,
            base_bars=base_bars,
        ),
        "labels": {
            "all_examples": _build_label_summary(examples, config),
            "validation_examples": _build_label_summary(validation_examples, config),
        },
        "validation": {
            "selected_row_count": len(validation_examples),
            "accept_reject_reason_counts": diagnostics["accept_reject_reason_counts"],
            "reject_flag_counts": diagnostics["reject_flag_counts"],
            "selected_horizon_summary": diagnostics["selected_horizon_summary"],
            "threshold_scan_source": diagnostics["threshold_scan_source"],
        },
        "paths": {
            "validation_rows_csv": str(validation_rows_path),
            "threshold_scan_csv": str(threshold_scan_path),
            "horizon_diag_csv": str(horizon_diag_path),
        },
    }
    save_json(summary_path, summary)
    return summary


def build_validation_diagnostics(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
    selection_policy: dict[str, object] | None,
) -> dict[str, object]:
    model.eval()
    validation_rows: list[dict[str, object]] = []
    accept_reject_reason_counts = defaultdict(int)
    reject_flag_counts = defaultdict(int)
    selected_horizon_summary = {
        str(horizon): {
            "selected_count": 0,
            "accepted_count": 0,
            "clean_count": 0,
            "accepted_clean_count": 0,
        }
        for horizon in config.horizons
    }
    sample_id = 0

    with torch.no_grad():
        for chunk in _chunk_examples(validation_examples, config.batch_size):
            batch = examples_to_batch(chunk, config)
            outputs = model(batch["main"], batch["overlay"])
            mu_std = outputs["mu"]
            sigma_std = outputs["sigma"]
            mu_raw, sigma_raw = restore_return_units(
                mu_std,
                sigma_std,
                batch["return_scale"],
            )
            direction_probabilities = torch.softmax(outputs["direction_logits"], dim=-1)
            overlay_probabilities = torch.sigmoid(outputs["overlay_logits"].squeeze(-1))

            for example_index, example in enumerate(chunk):
                decision = apply_selection_policy(
                    example=example,
                    mean=mu_raw[example_index].tolist(),
                    sigma=sigma_raw[example_index].tolist(),
                    direction_probabilities=direction_probabilities[example_index].tolist(),
                    overlay_probability=float(overlay_probabilities[example_index].item()),
                    policy=selection_policy,
                    config=config,
                )
                selected_horizon = int(decision["selected_horizon"])
                selected_summary = selected_horizon_summary[str(selected_horizon)]
                selected_summary["selected_count"] += 1
                selected_summary["clean_count"] += int(decision["meta_label"])
                selected_summary["accepted_count"] += int(decision["accepted_signal"])
                selected_summary["accepted_clean_count"] += int(
                    decision["accepted_signal"] and decision["meta_label"]
                )

                accept_reject_reason_counts[str(decision["accept_reject_reason"])] += 1
                for flag_name, enabled in dict(decision["reject_flags"]).items():
                    reject_flag_counts[str(flag_name)] += int(bool(enabled))

                for horizon_index, row in enumerate(decision["horizon_rows"]):
                    horizon = int(row["horizon"])
                    row_selected = horizon == selected_horizon
                    row_accepted = row_selected and bool(decision["accepted_signal"])
                    row_alignment = (
                        int(row["predicted_sign"]) != 0
                        and int(row["predicted_sign"]) == _sign_of_value(float(row["mean"]))
                    )
                    row_meta_label = int(
                        int(row["predicted_sign"]) != 0
                        and int(row["predicted_sign"]) == int(row["true_direction"])
                    )
                    validation_rows.append(
                        {
                            "sample_id": sample_id,
                            "timestamp": example.anchor_time.isoformat(),
                            "regime_id": example.regime_id,
                            "horizon": horizon,
                            "y_raw": float(example.returns_target[horizon_index]),
                            "y_std": float(batch["returns"][example_index, horizon_index].item()),
                            "mu_std": float(mu_std[example_index, horizon_index].item()),
                            "sigma_std": float(sigma_std[example_index, horizon_index].item()),
                            "mu_raw": float(mu_raw[example_index, horizon_index].item()),
                            "sigma_raw": float(sigma_raw[example_index, horizon_index].item()),
                            "direction_label": int(example.direction_targets[horizon_index]),
                            "predicted_sign": int(row["predicted_sign"]),
                            "p_down": float(row["direction_probabilities"][0]),
                            "p_flat": float(row["direction_probabilities"][1]),
                            "p_up": float(row["direction_probabilities"][2]),
                            "edge": float(row["edge"]),
                            "prob_gap": float(row["prob_gap"]),
                            "sign_agreement": float(row["sign_agreement"]),
                            "direction_alignment": int(row_alignment),
                            "actionable_sign": int(row["actionable_sign"]),
                            "actionable_edge": float(row["actionable_edge"]),
                            "correctness_probability": float(row["q"]),
                            "selector_probability": float(row["selector_probability"]),
                            "selection_threshold": (
                                None
                                if decision["selection_threshold"] is None
                                else float(decision["selection_threshold"])
                            ),
                            "cost": float(example.horizon_costs[horizon_index]),
                            "direction_threshold": float(example.direction_thresholds[horizon_index]),
                            "direction_mae_threshold": float(
                                example.direction_mae_thresholds[horizon_index]
                            ),
                            "long_mae": float(example.long_mae[horizon_index]),
                            "short_mae": float(example.short_mae[horizon_index]),
                            "long_mfe": float(example.long_mfe[horizon_index]),
                            "short_mfe": float(example.short_mfe[horizon_index]),
                            "direction_weight": float(example.direction_weights[horizon_index]),
                            "realized_volatility": float(example.realized_volatility),
                            "return_scale": float(batch["return_scale"][example_index, horizon_index].item()),
                            "overlay_probability": float(decision["hold_probability"]),
                            "hold_threshold": float(decision["hold_threshold"]),
                            "overlay_action": str(decision["overlay_action"]),
                            "selected": int(row_selected),
                            "accepted": int(row_accepted),
                            "meta_label": row_meta_label,
                            "pre_threshold_eligible": int(
                                bool(decision["pre_threshold_eligible"]) if row_selected else False
                            ),
                            "accept_reject_reason": (
                                str(decision["accept_reject_reason"]) if row_selected else "not_selected"
                            ),
                        }
                    )
                sample_id += 1

    threshold_scan, threshold_scan_source = _resolve_threshold_scan(
        validation_rows=validation_rows,
        selection_policy=selection_policy,
        config=config,
    )
    return {
        "validation_rows": validation_rows,
        "threshold_scan": threshold_scan,
        "threshold_scan_source": threshold_scan_source,
        "horizon_diag": _build_horizon_diag(validation_rows, config),
        "accept_reject_reason_counts": dict(sorted(accept_reject_reason_counts.items())),
        "reject_flag_counts": dict(sorted(reject_flag_counts.items())),
        "selected_horizon_summary": selected_horizon_summary,
    }


def _build_dataset_summary(
    examples: Sequence[TrainingExample],
    validation_examples: Sequence[TrainingExample],
    source_payload: dict[str, object] | None,
    source_rows_original: int | None,
    source_rows_used: int | None,
    base_bars: Sequence[OHLCVBar] | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "anchor_samples_total": len(examples),
        "anchor_samples_validation": len(validation_examples),
    }
    if source_payload is not None:
        summary["source"] = dict(source_payload)
    if source_rows_original is not None:
        summary["source_rows_original"] = int(source_rows_original)
    if source_rows_used is not None:
        summary["source_rows_used"] = int(source_rows_used)
    if base_bars:
        ordered = sorted(base_bars, key=lambda bar: bar.timestamp)
        start = ordered[0].timestamp
        end = ordered[-1].timestamp
        summary.update(
            {
                "base_start_utc": start.isoformat(),
                "base_end_utc": end.isoformat(),
                "base_span_days": round((end - start).total_seconds() / 86400.0, 2),
            }
        )
    if validation_examples:
        validation_start = min(example.anchor_time for example in validation_examples)
        validation_end = max(example.anchor_time for example in validation_examples)
        summary.update(
            {
                "validation_anchor_start_utc": validation_start.isoformat(),
                "validation_anchor_end_utc": validation_end.isoformat(),
            }
        )
    return summary


def _build_label_summary(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> dict[str, object]:
    if not examples:
        return {"count": 0, "by_horizon": {}}

    by_horizon: dict[str, dict[str, object]] = {}
    for horizon_index, horizon in enumerate(config.horizons):
        regime_rows = [_regime_dict_from_id(example.regime_id) for example in examples]
        by_horizon[str(horizon)] = {
            "count": len(examples),
            "direction_counts": {
                "-1": sum(int(example.direction_targets[horizon_index] < 0) for example in examples),
                "0": sum(int(example.direction_targets[horizon_index] == 0) for example in examples),
                "+1": sum(int(example.direction_targets[horizon_index] > 0) for example in examples),
            },
            "target_return": _describe([example.returns_target[horizon_index] for example in examples]),
            "long_mae": _describe([example.long_mae[horizon_index] for example in examples]),
            "short_mae": _describe([example.short_mae[horizon_index] for example in examples]),
            "long_mfe": _describe([example.long_mfe[horizon_index] for example in examples]),
            "short_mfe": _describe([example.short_mfe[horizon_index] for example in examples]),
            "direction_weight": _describe([example.direction_weights[horizon_index] for example in examples]),
            "c_h": _describe([example.horizon_costs[horizon_index] for example in examples]),
            "delta_h": _describe([example.direction_thresholds[horizon_index] for example in examples]),
            "eta_h": _describe(
                [example.direction_mae_thresholds[horizon_index] for example in examples]
            ),
            "sigma_t": _describe([example.realized_volatility for example in examples]),
            "scale_h": _describe(
                [
                    _return_scale(example.realized_volatility, horizon, config)
                    for example in examples
                ]
            ),
            "delta_h_recomputed": _describe(
                [
                    _main_move_threshold(
                        config,
                        horizon,
                        example.realized_volatility,
                        regime_rows[index],
                    )
                    for index, example in enumerate(examples)
                ]
            ),
            "eta_h_recomputed": _describe(
                [
                    _main_mae_threshold(
                        config,
                        horizon,
                        example.realized_volatility,
                        regime_rows[index],
                    )
                    for index, example in enumerate(examples)
                ]
            ),
        }
    return {
        "count": len(examples),
        "by_horizon": by_horizon,
    }


def _build_horizon_diag(
    validation_rows: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    rows_by_horizon: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in validation_rows:
        rows_by_horizon[str(row["horizon"])].append(dict(row))

    diagnostics: list[dict[str, object]] = []
    for horizon in config.horizons:
        rows = rows_by_horizon.get(str(horizon), [])
        diagnostics.append(
            {
                "horizon": horizon,
                "nonflat_rate": _rate(rows, lambda row: int(row["direction_label"]) != 0),
                "up_rate": _rate(rows, lambda row: int(row["direction_label"]) > 0),
                "down_rate": _rate(rows, lambda row: int(row["direction_label"]) < 0),
                "align_rate": _rate(rows, lambda row: int(row["direction_alignment"]) == 1),
                "actionable_edge_rate": _rate(rows, lambda row: float(row["actionable_edge"]) > 0.0),
                "mean_mu": _mean([float(row["mu_raw"]) for row in rows]),
                "mean_sigma": _mean([float(row["sigma_raw"]) for row in rows]),
                "median_abs_mu": _median([abs(float(row["mu_raw"])) for row in rows]),
                "selected_rate": _rate(rows, lambda row: int(row["selected"]) == 1),
                "accepted_rate": _rate(rows, lambda row: int(row["accepted"]) == 1),
            }
        )
    return diagnostics


def _resolve_threshold_scan(
    validation_rows: Sequence[dict[str, object]],
    selection_policy: dict[str, object] | None,
    config: TrainingConfig,
) -> tuple[list[dict[str, object]], str]:
    if selection_policy is not None:
        scan = (
            selection_policy.get("selection_thresholds", {})
            .get("scan", {})
            .get("global", [])
        )
        if isinstance(scan, list) and scan:
            return [dict(row) for row in scan if isinstance(row, dict)], "policy_calibration"

    records = [
        {
            "score": float(row["selector_probability"]),
            "target": int(row["meta_label"]),
        }
        for row in validation_rows
        if int(row["selected"]) == 1 and int(row["pre_threshold_eligible"]) == 1
    ]
    return _build_threshold_scan(records, config), "validation_selected_rows"


def _build_threshold_scan(
    records: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    if not records:
        return []

    rows: list[dict[str, object]] = []
    candidates = sorted({float(record["score"]) for record in records})
    total = len(records)
    for tau in candidates:
        selected = [record for record in records if float(record["score"]) >= tau]
        selected_count = len(selected)
        success_count = sum(int(record["target"]) for record in selected)
        precision = success_count / selected_count if selected_count > 0 else 0.0
        lcb = (
            _precision_lower_bound(success_count, selected_count, config.precision_confidence_z)
            if selected_count > 0
            else 0.0
        )
        rows.append(
            {
                "tau": tau,
                "selected_count": selected_count,
                "success_count": success_count,
                "precision": precision,
                "lcb": lcb,
                "feasible": (
                    selected_count >= config.selection_min_support and lcb >= config.precision_target
                ),
                "coverage": selected_count / total if total > 0 else 0.0,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _chunk_examples(
    examples: Sequence[TrainingExample],
    batch_size: int,
) -> Sequence[list[TrainingExample]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _describe(values: Sequence[float]) -> dict[str, object]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"count": 0}
    ordered = sorted(numeric)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
        "mean": _mean(ordered),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = max(0.0, min(1.0, probability)) * (len(values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    weight = position - lower_index
    return (float(values[lower_index]) * (1.0 - weight)) + (float(values[upper_index]) * weight)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _rate(
    rows: Sequence[dict[str, object]],
    predicate,
) -> float:
    if not rows:
        return 0.0
    return sum(int(predicate(row)) for row in rows) / len(rows)


def _regime_dict_from_id(regime_id: str) -> dict[str, object]:
    session, volatility_bin, trend_bin = regime_id.split("|", maxsplit=2)
    return {
        "id": regime_id,
        "session": session,
        "volatility_bin": volatility_bin,
        "trend_bin": trend_bin,
    }


def _return_scale(
    realized_volatility: float,
    horizon: int,
    config: TrainingConfig,
) -> float:
    return max((float(realized_volatility) * (horizon**0.5)) + config.return_scale_epsilon, 1e-6)


def _sign_of_value(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _precision_lower_bound(success_count: int, total_count: int, z_score: float) -> float:
    if total_count <= 0:
        return 0.0
    proportion = success_count / total_count
    z2 = z_score**2
    denominator = 1.0 + (z2 / total_count)
    centre = proportion + (z2 / (2.0 * total_count))
    margin = z_score * (
        (
            ((proportion * (1.0 - proportion)) / total_count)
            + (z2 / (4.0 * (total_count**2)))
        )
        ** 0.5
    )
    return max(0.0, (centre - margin) / denominator)
