from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from .config import TrainingConfig
from .dataset_service import _main_mae_threshold, _main_move_threshold
from .policy_service import apply_selection_policy, build_prediction_snapshots
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
    threshold_resolution: dict[str, object] | None = None,
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
            "anchor_sample_count": len(validation_examples),
            "proposed_row_count": diagnostics["proposed_row_count"],
            "accepted_row_count": diagnostics["accepted_row_count"],
            "no_candidate_count": diagnostics["no_candidate_count"],
            "no_strict_candidate_count": diagnostics["no_strict_candidate_count"],
            "candidate_but_no_strict_count": diagnostics["candidate_but_no_strict_count"],
            "any_candidate_rate": diagnostics["any_candidate_rate"],
            "any_strict_candidate_rate": diagnostics["any_strict_candidate_rate"],
            "candidate_count_per_anchor": diagnostics["candidate_count_per_anchor"],
            "strict_candidate_count_per_anchor": diagnostics["strict_candidate_count_per_anchor"],
            "accept_reject_reason_counts": diagnostics["accept_reject_reason_counts"],
            "reject_flag_counts": diagnostics["reject_flag_counts"],
            "threshold_status": (
                "disabled"
                if threshold_resolution is not None
                and threshold_resolution.get("selection_threshold_mode_requested") == "none"
                else diagnostics["threshold_status"]
            ),
            "threshold_origin": diagnostics["threshold_origin"],
            "stored_threshold_compatibility": (
                diagnostics["stored_threshold_compatibility"]
                if threshold_resolution is None
                else threshold_resolution.get(
                    "stored_threshold_compatibility",
                    diagnostics["stored_threshold_compatibility"],
                )
            ),
            "threshold_score_source": diagnostics["threshold_score_source"],
            "selection_threshold_mode_requested": (
                None
                if threshold_resolution is None
                else threshold_resolution.get("selection_threshold_mode_requested")
            ),
            "selection_threshold_mode_resolved": (
                None
                if threshold_resolution is None
                else threshold_resolution.get("selection_threshold_mode_resolved")
            ),
            "threshold_calibration_anchor_count": len(validation_examples),
            "threshold_calibration_proposed_count": diagnostics["proposed_row_count"],
            "proposed_horizon_summary": diagnostics["proposed_horizon_summary"],
            "threshold_scan_source": diagnostics["threshold_scan_source"],
            "acceptance_score_source": str(config.selection_score_source),
            "allow_no_candidate": bool(config.allow_no_candidate),
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
    proposed_row_count = 0
    accepted_row_count = 0
    no_candidate_count = 0
    no_strict_candidate_count = 0
    candidate_but_no_strict_count = 0
    any_candidate_count = 0
    any_strict_candidate_count = 0
    total_candidate_count = 0
    total_strict_candidate_count = 0
    proposed_horizon_summary = {
        str(horizon): {
            "proposed_count": 0,
            "accepted_count": 0,
            "proposed_clean_count": 0,
            "accepted_clean_count": 0,
        }
        for horizon in config.horizons
    }
    sample_id = 0
    threshold_status = "missing"
    threshold_origin = "none"
    stored_threshold_compatibility = "not_applicable"
    threshold_score_source = str(config.selection_score_source)

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
                proposed_horizon_value = decision["proposed_horizon"]
                proposed_horizon = (
                    int(proposed_horizon_value) if proposed_horizon_value is not None else None
                )
                proposed_row_count += int(proposed_horizon is not None)
                accepted_row_count += int(bool(decision["accepted_signal"]))
                no_candidate_count += int(proposed_horizon is None)
                no_strict_candidate_count += int(not bool(decision["any_strict_candidate"]))
                candidate_but_no_strict_count += int(
                    bool(decision["any_candidate"]) and not bool(decision["any_strict_candidate"])
                )
                any_candidate_count += int(bool(decision["any_candidate"]))
                any_strict_candidate_count += int(bool(decision["any_strict_candidate"]))
                total_candidate_count += int(decision["candidate_count"])
                total_strict_candidate_count += int(decision["strict_candidate_count"])
                threshold_status = str(decision["threshold_status"])
                threshold_origin = str(decision["threshold_origin"])
                stored_threshold_compatibility = str(
                    decision["stored_threshold_compatibility"]
                )
                threshold_score_source = str(decision["threshold_score_source"])
                if proposed_horizon is not None:
                    proposed_summary = proposed_horizon_summary[str(proposed_horizon)]
                    proposed_summary["proposed_count"] += 1
                    proposed_summary["proposed_clean_count"] += int(decision["meta_label"])
                    proposed_summary["accepted_count"] += int(decision["accepted_signal"])
                    proposed_summary["accepted_clean_count"] += int(
                        decision["accepted_signal"] and decision["meta_label"]
                    )

                accept_reject_reason_counts[str(decision["accept_reject_reason"])] += 1
                for flag_name, enabled in dict(decision["reject_flags"]).items():
                    reject_flag_counts[str(flag_name)] += int(bool(enabled))

                for horizon_index, row in enumerate(decision["horizon_rows"]):
                    horizon = int(row["horizon"])
                    row_proposed = proposed_horizon is not None and horizon == proposed_horizon
                    row_accepted = row_proposed and bool(decision["accepted_signal"])
                    row_alignment = bool(row["direction_alignment"])
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
                            "candidate": int(bool(row["candidate"])),
                            "strict_candidate": int(bool(row["strict_candidate"])),
                            "chooser_score": float(row["score"]),
                            "actionable_sign": int(row["actionable_sign"]),
                            "actionable_edge": float(row["actionable_edge"]),
                            "correctness_probability": float(row["q"]),
                            "selector_probability": float(row["selector_probability"]),
                            "selection_score": float(row["selection_score"]),
                            "acceptance_score_source": str(config.selection_score_source),
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
                            "proposed_horizon": decision["proposed_horizon"],
                            "accepted_horizon": decision["accepted_horizon"],
                            "proposal_status": (
                                "proposed" if decision["proposed_horizon"] is not None else "no_candidate"
                            ),
                            "acceptance_status": _acceptance_status(decision),
                            "proposed": int(row_proposed),
                            "accepted": int(row_accepted),
                            "meta_label": row_meta_label,
                            "pre_threshold_eligible": int(
                                bool(decision["pre_threshold_eligible"]) if row_proposed else False
                            ),
                            "accept_reject_reason": (
                                str(decision["accept_reject_reason"])
                                if row_proposed or proposed_horizon is None
                                else "not_selected"
                            ),
                        }
                    )
                sample_id += 1

    threshold_scan, threshold_scan_source = _resolve_threshold_scan(
        validation_rows=validation_rows,
        selection_policy=selection_policy,
        config=config,
        anchor_count=len(validation_examples),
        proposal_count=proposed_row_count,
    )
    return {
        "validation_rows": validation_rows,
        "threshold_scan": threshold_scan,
        "threshold_scan_source": threshold_scan_source,
        "horizon_diag": _build_horizon_diag(validation_rows, config),
        "proposed_row_count": proposed_row_count,
        "accepted_row_count": accepted_row_count,
        "no_candidate_count": no_candidate_count,
        "no_strict_candidate_count": no_strict_candidate_count,
        "candidate_but_no_strict_count": candidate_but_no_strict_count,
        "any_candidate_rate": any_candidate_count / max(len(validation_examples), 1),
        "any_strict_candidate_rate": any_strict_candidate_count / max(len(validation_examples), 1),
        "candidate_count_per_anchor": total_candidate_count / max(len(validation_examples), 1),
        "strict_candidate_count_per_anchor": total_strict_candidate_count / max(
            len(validation_examples), 1
        ),
        "accept_reject_reason_counts": dict(sorted(accept_reject_reason_counts.items())),
        "reject_flag_counts": dict(sorted(reject_flag_counts.items())),
        "threshold_status": threshold_status,
        "threshold_origin": threshold_origin,
        "stored_threshold_compatibility": stored_threshold_compatibility,
        "threshold_score_source": threshold_score_source,
        "proposed_horizon_summary": proposed_horizon_summary,
    }


def build_validation_snapshots(
    model: SignalCascadeModel,
    validation_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    model.eval()
    snapshots: list[dict[str, object]] = []
    with torch.no_grad():
        for chunk in _chunk_examples(validation_examples, config.batch_size):
            batch = examples_to_batch(chunk, config)
            outputs = model(batch["main"], batch["overlay"])
            mu_raw, sigma_raw = restore_return_units(
                outputs["mu"],
                outputs["sigma"],
                batch["return_scale"],
            )
            direction_probabilities = torch.softmax(outputs["direction_logits"], dim=-1)
            overlay_probabilities = torch.sigmoid(outputs["overlay_logits"].squeeze(-1))
            snapshots.extend(
                build_prediction_snapshots(
                    chunk,
                    mu_raw,
                    sigma_raw,
                    direction_probabilities,
                    overlay_probabilities,
                    config,
                )
            )
    return snapshots


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
                "candidate_rate": _rate(rows, lambda row: int(row["candidate"]) == 1),
                "strict_candidate_rate": _rate(rows, lambda row: int(row["strict_candidate"]) == 1),
                "actionable_edge_rate": _rate(rows, lambda row: float(row["actionable_edge"]) > 0.0),
                "mean_mu": _mean([float(row["mu_raw"]) for row in rows]),
                "mean_sigma": _mean([float(row["sigma_raw"]) for row in rows]),
                "median_abs_mu": _median([abs(float(row["mu_raw"])) for row in rows]),
                "proposed_rate": _rate(rows, lambda row: int(row["proposed"]) == 1),
                "accepted_rate": _rate(rows, lambda row: int(row["accepted"]) == 1),
            }
        )
    return diagnostics


def _resolve_threshold_scan(
    validation_rows: Sequence[dict[str, object]],
    selection_policy: dict[str, object] | None,
    config: TrainingConfig,
    anchor_count: int,
    proposal_count: int,
) -> tuple[list[dict[str, object]], str]:
    if selection_policy is not None and _policy_thresholds_match_config(selection_policy, config):
        threshold_origin = str(
            selection_policy.get("selection_thresholds", {}).get("origin", "stored_policy")
        )
        scan = (
            selection_policy.get("selection_thresholds", {})
            .get("scan", {})
            .get("global", [])
        )
        if isinstance(scan, list) and scan:
            return [
                _normalize_threshold_scan_row(
                    dict(row),
                    proposal_count=proposal_count,
                    anchor_count=anchor_count,
                )
                for row in scan
                if isinstance(row, dict)
            ], f"policy_calibration:{threshold_origin}"

    records = [
        {
            "score": float(row["selection_score"]),
            "target": int(row["meta_label"]),
        }
        for row in validation_rows
        if int(row["proposed"]) == 1 and int(row["pre_threshold_eligible"]) == 1
    ]
    return (
        _build_threshold_scan(records, config, anchor_count=anchor_count),
        f"validation_proposed_rows:{config.selection_score_source}",
    )


def _build_threshold_scan(
    records: Sequence[dict[str, object]],
    config: TrainingConfig,
    anchor_count: int,
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
        proposal_coverage = selected_count / total if total > 0 else 0.0
        rows.append(
            {
                "tau": tau,
                "accepted_count_at_tau": selected_count,
                "success_count_at_tau": success_count,
                "precision_at_tau": precision,
                "lcb": lcb,
                "feasible": (
                    selected_count >= config.selection_min_support and lcb >= config.precision_target
                ),
                "proposal_coverage": proposal_coverage,
                "anchor_coverage": selected_count / max(anchor_count, 1),
            }
        )
    return rows


def _normalize_threshold_scan_row(
    row: dict[str, object],
    proposal_count: int,
    anchor_count: int,
) -> dict[str, object]:
    accepted_count = int(
        row.get(
            "accepted_count_at_tau",
            row.get("selected_count", 0),
        )
    )
    success_count = int(
        row.get(
            "success_count_at_tau",
            row.get("success_count", 0),
        )
    )
    precision = float(
        row.get(
            "precision_at_tau",
            row.get("precision", 0.0),
        )
    )
    proposal_coverage = float(
        row.get(
            "proposal_coverage",
            row.get(
                "coverage",
                accepted_count / max(proposal_count, 1),
            ),
        )
    )
    return {
        "tau": float(row.get("tau", 0.0)),
        "accepted_count_at_tau": accepted_count,
        "success_count_at_tau": success_count,
        "precision_at_tau": precision,
        "lcb": float(row.get("lcb", 0.0)),
        "feasible": bool(row.get("feasible", False)),
        "proposal_coverage": proposal_coverage,
        "anchor_coverage": float(
            row.get(
                "anchor_coverage",
                accepted_count / max(anchor_count, 1),
            )
        ),
    }


def _acceptance_status(decision: dict[str, object]) -> str:
    if decision["proposed_horizon"] is None:
        return "not_applicable"
    if bool(decision["accepted_signal"]):
        return "accepted"
    if decision["selection_threshold"] is None:
        return "threshold_unavailable"
    return "below_threshold"


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


def _policy_thresholds_match_config(
    selection_policy: dict[str, object],
    config: TrainingConfig,
) -> bool:
    thresholds = dict(selection_policy.get("selection_thresholds", {}))
    stored_score_source = str(
        thresholds.get(
            "score_source",
            selection_policy.get("selection_score_source", "selector_probability"),
        )
    )
    stored_allow_no_candidate = bool(
        thresholds.get(
            "allow_no_candidate",
            selection_policy.get("allow_no_candidate", False),
        )
    )
    return (
        stored_score_source == str(config.selection_score_source)
        and stored_allow_no_candidate == bool(config.allow_no_candidate)
    )


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
