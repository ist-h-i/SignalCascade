from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as functional

from .config import TrainingConfig
from ..domain.entities import TrainingExample

Q_FEATURE_NAMES = (
    "edge",
    "p_down",
    "p_flat",
    "p_up",
    "prob_gap",
    "sign_agreement",
    "session_asia",
    "session_london",
    "session_ny",
    "volatility_ratio",
    "trend_strength",
    "horizon_norm",
)
SELECTOR_FEATURE_NAMES = (
    "edge",
    "q",
    "p_down",
    "p_flat",
    "p_up",
    "prob_gap",
    "top_gap",
    "sign_agreement",
    "session_asia",
    "session_london",
    "session_ny",
    "volatility_ratio",
    "trend_strength",
    "horizon_norm",
)
SIGN_INDEX_TO_VALUE = (-1, 0, 1)


def build_default_policy(config: TrainingConfig) -> dict[str, object]:
    return {
        "precision_target": config.precision_target,
        "selection_alpha": config.selection_alpha,
        "correctness_model": _constant_model(0.5, Q_FEATURE_NAMES),
        "selector_model": _constant_model(0.5, SELECTOR_FEATURE_NAMES),
        "selection_thresholds": {
            "global": {str(horizon): 1.0 for horizon in config.horizons},
            "by_regime": {},
        },
        "overlay_thresholds": {
            "global": 0.5,
            "by_regime": {},
        },
        "metrics": {},
    }


def build_prediction_snapshots(
    examples: Sequence[TrainingExample],
    mean: torch.Tensor,
    sigma: torch.Tensor,
    direction_probabilities: torch.Tensor,
    overlay_probabilities: torch.Tensor,
    config: TrainingConfig,
) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for batch_index, example in enumerate(examples):
        horizon_rows: list[dict[str, object]] = []
        for horizon_index, horizon in enumerate(config.horizons):
            probability_vector = direction_probabilities[batch_index, horizon_index].tolist()
            predicted_index = int(torch.argmax(direction_probabilities[batch_index, horizon_index]).item())
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "mean": float(mean[batch_index, horizon_index].item()),
                    "sigma": float(sigma[batch_index, horizon_index].item()),
                    "cost": float(example.horizon_costs[horizon_index]),
                    "predicted_sign": SIGN_INDEX_TO_VALUE[predicted_index],
                    "true_return": float(example.returns_target[horizon_index]),
                    "true_direction": int(example.direction_targets[horizon_index]),
                    "direction_probabilities": [float(value) for value in probability_vector],
                }
            )

        snapshots.append(
            {
                "anchor_time": example.anchor_time.isoformat(),
                "regime_id": example.regime_id,
                "regime_features": [float(value) for value in example.regime_features],
                "realized_volatility": float(example.realized_volatility),
                "trend_strength": float(example.trend_strength),
                "overlay_target": int(example.overlay_target),
                "overlay_probability": float(overlay_probabilities[batch_index].item()),
                "horizons": horizon_rows,
            }
        )
    return snapshots


def build_selection_policy(
    snapshots: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> dict[str, object]:
    policy = build_default_policy(config)
    if not snapshots:
        return policy

    q_rows: list[list[float]] = []
    q_targets: list[int] = []
    for snapshot in snapshots:
        prepared_rows = _prepare_horizon_rows(snapshot, policy, config)
        for row in prepared_rows:
            q_rows.append(_q_feature_vector(row, snapshot, config))
            q_targets.append(
                int(
                    row["predicted_sign"] != 0
                    and row["predicted_sign"] == _sign_from_return(row["true_return"])
                )
            )
    policy["correctness_model"] = _fit_binary_model(
        feature_names=Q_FEATURE_NAMES,
        feature_rows=q_rows,
        targets=q_targets,
        brier_weight=0.05,
    )

    selector_rows: list[list[float]] = []
    selector_targets: list[int] = []
    for snapshot in snapshots:
        prepared_rows = _prepare_horizon_rows(snapshot, policy, config)
        for row in prepared_rows:
            selector_rows.append(_selector_feature_vector(row, snapshot, config))
            selector_targets.append(
                int(
                    row["predicted_sign"] != 0
                    and row["predicted_sign"] == row["true_direction"]
                )
            )
    policy["selector_model"] = _fit_binary_model(
        feature_names=SELECTOR_FEATURE_NAMES,
        feature_rows=selector_rows,
        targets=selector_targets,
        brier_weight=config.selector_brier_weight,
    )

    selection_records = _build_selection_threshold_records(snapshots, policy, config)
    policy["selection_thresholds"] = _build_selection_thresholds(selection_records, config)
    overlay_records = _build_overlay_threshold_records(snapshots, policy, config)
    policy["overlay_thresholds"] = _build_overlay_thresholds(overlay_records, config)
    policy["metrics"] = evaluate_policy_snapshots(policy, snapshots, config)
    return policy


def apply_selection_policy(
    example: TrainingExample,
    mean: Sequence[float],
    sigma: Sequence[float],
    direction_probabilities: Sequence[Sequence[float]],
    overlay_probability: float,
    policy: dict[str, object] | None,
    config: TrainingConfig,
) -> dict[str, object]:
    active_policy = policy or build_default_policy(config)
    snapshot = {
        "anchor_time": example.anchor_time.isoformat(),
        "regime_id": example.regime_id,
        "regime_features": [float(value) for value in example.regime_features],
        "realized_volatility": float(example.realized_volatility),
        "trend_strength": float(example.trend_strength),
        "overlay_target": int(example.overlay_target),
        "overlay_probability": float(overlay_probability),
        "horizons": [
            {
                "horizon": horizon,
                "mean": float(mean[horizon_index]),
                "sigma": float(sigma[horizon_index]),
                "cost": float(example.horizon_costs[horizon_index]),
                "predicted_sign": SIGN_INDEX_TO_VALUE[int(_argmax(direction_probabilities[horizon_index]))],
                "true_return": float(example.returns_target[horizon_index]),
                "true_direction": int(example.direction_targets[horizon_index]),
                "direction_probabilities": [float(value) for value in direction_probabilities[horizon_index]],
            }
            for horizon_index, horizon in enumerate(config.horizons)
        ],
    }
    return _augment_snapshot(snapshot, active_policy, config)


def evaluate_policy_snapshots(
    policy: dict[str, object],
    snapshots: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> dict[str, float]:
    accepted = 0
    accepted_clean = 0
    hold_predictions = 0
    hold_correct = 0
    overlay_correct = 0
    cumulative_value = 0.0
    cumulative_abs_value = 0.0
    turnover = 0.0
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    previous_position = 0.0

    for snapshot in snapshots:
        augmented = _augment_snapshot(snapshot, policy, config)
        if augmented["accepted_signal"]:
            accepted += 1
            accepted_clean += int(augmented["meta_label"])

        overlay_is_hold = augmented["overlay_action"] == "hold"
        overlay_target = int(snapshot["overlay_target"])
        overlay_correct += int(overlay_is_hold == bool(overlay_target))
        if overlay_is_hold:
            hold_predictions += 1
            hold_correct += overlay_target

        realized_return = float(augmented["selected_row"]["true_return"])
        realized_value = float(augmented["position"]) * realized_return
        cumulative_value += realized_value
        cumulative_abs_value += abs(realized_return)
        turnover += abs(float(augmented["position"]) - previous_position)
        previous_position = float(augmented["position"])
        equity += realized_value
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)

    total = max(len(snapshots), 1)
    coverage = accepted / total
    return {
        "selection_precision": accepted_clean / max(accepted, 1),
        "coverage_at_target_precision": coverage,
        "no_trade_rate": 1.0 - coverage,
        "overlay_accuracy": overlay_correct / total,
        "overlay_precision": hold_correct / max(hold_predictions, 1),
        "value_capture_ratio": cumulative_value / max(cumulative_abs_value, 1e-6),
        "turnover": turnover,
        "max_drawdown": max_drawdown,
    }


def _prepare_horizon_rows(
    snapshot: dict[str, object],
    policy: dict[str, object],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    rows = [dict(row) for row in snapshot["horizons"]]
    predicted_signs = [int(row["predicted_sign"]) for row in rows]
    scores: list[float] = []

    for row in rows:
        probabilities = list(row["direction_probabilities"])
        sorted_probabilities = sorted(probabilities, reverse=True)
        probability_gap = sorted_probabilities[0] - sorted_probabilities[1]
        agreement = _sign_agreement(predicted_signs, int(row["predicted_sign"]))
        row["edge"] = abs(float(row["mean"])) / max(float(row["sigma"]), 1e-6)
        row["prob_gap"] = probability_gap
        row["sign_agreement"] = agreement
        row["q"] = _predict_binary_model(policy["correctness_model"], _q_feature_vector(row, snapshot, config))
        raw_edge = max(abs(float(row["mean"])) - float(row["cost"]), 0.0)
        if int(row["predicted_sign"]) == 0 or raw_edge <= 0.0:
            score = 0.0
        else:
            score = (row["q"] ** config.selection_alpha) * (raw_edge ** (1.0 - config.selection_alpha))
        row["score"] = score
        scores.append(score)

    best_score = max(scores, default=0.0)
    second_score = _second_largest(scores)
    for row in rows:
        row["top_gap"] = float(row["score"]) - (second_score if float(row["score"]) >= best_score else best_score)
    return rows


def _augment_snapshot(
    snapshot: dict[str, object],
    policy: dict[str, object],
    config: TrainingConfig,
) -> dict[str, object]:
    rows = _prepare_horizon_rows(snapshot, policy, config)
    if not rows:
        raise ValueError("Prediction snapshot does not contain any horizon rows.")

    for row in rows:
        row["selector_probability"] = _predict_binary_model(
            policy["selector_model"],
            _selector_feature_vector(row, snapshot, config),
        )

    selected_row = max(rows, key=lambda row: (float(row["score"]), abs(float(row["mean"]))))
    selection_threshold = _lookup_selection_threshold(
        policy,
        str(snapshot["regime_id"]),
        int(selected_row["horizon"]),
    )
    hold_threshold = _lookup_overlay_threshold(policy, str(snapshot["regime_id"]))
    accepted_signal = (
        int(selected_row["predicted_sign"]) != 0
        and float(selected_row["score"]) > 0.0
        and float(selected_row["selector_probability"]) >= selection_threshold
    )
    raw_position = (
        math.tanh(config.position_scale * (float(selected_row["mean"]) / max(float(selected_row["sigma"]), 1e-6)))
        if accepted_signal
        else 0.0
    )
    overlay_probability = float(snapshot["overlay_probability"])
    overlay_action = "hold" if overlay_probability >= hold_threshold else "reduce"
    position = raw_position * overlay_probability

    return {
        "selected_row": selected_row,
        "selected_horizon": int(selected_row["horizon"]),
        "selected_direction": int(selected_row["predicted_sign"]),
        "position": position,
        "raw_position": raw_position,
        "accepted_signal": accepted_signal,
        "selection_probability": float(selected_row["selector_probability"]),
        "selection_threshold": selection_threshold,
        "correctness_probability": float(selected_row["q"]),
        "hold_probability": overlay_probability,
        "hold_threshold": hold_threshold,
        "overlay_action": overlay_action,
        "meta_label": int(
            int(selected_row["predicted_sign"]) != 0
            and int(selected_row["predicted_sign"]) == int(selected_row["true_direction"])
        ),
        "direction_correct": int(
            int(selected_row["predicted_sign"]) != 0
            and int(selected_row["predicted_sign"]) == _sign_from_return(float(selected_row["true_return"]))
        ),
        "horizon_rows": rows,
    }


def _q_feature_vector(
    row: dict[str, object],
    snapshot: dict[str, object] | None = None,
    config: TrainingConfig | None = None,
) -> list[float]:
    current_snapshot = snapshot or {"regime_features": [0.0, 0.0, 0.0, 0.0, 0.0]}
    max_horizon = max(config.horizons) if config is not None else max(int(row["horizon"]), 1)
    return [
        float(row["edge"]),
        float(row["direction_probabilities"][0]),
        float(row["direction_probabilities"][1]),
        float(row["direction_probabilities"][2]),
        float(row["prob_gap"]),
        float(row["sign_agreement"]),
        float(current_snapshot["regime_features"][0]),
        float(current_snapshot["regime_features"][1]),
        float(current_snapshot["regime_features"][2]),
        float(current_snapshot["regime_features"][3]),
        float(current_snapshot["regime_features"][4]),
        float(row["horizon"]) / max(max_horizon, 1),
    ]


def _selector_feature_vector(
    row: dict[str, object],
    snapshot: dict[str, object] | None = None,
    config: TrainingConfig | None = None,
) -> list[float]:
    current_snapshot = snapshot or {"regime_features": [0.0, 0.0, 0.0, 0.0, 0.0]}
    max_horizon = max(config.horizons) if config is not None else max(int(row["horizon"]), 1)
    return [
        float(row["edge"]),
        float(row["q"]),
        float(row["direction_probabilities"][0]),
        float(row["direction_probabilities"][1]),
        float(row["direction_probabilities"][2]),
        float(row["prob_gap"]),
        float(row["top_gap"]),
        float(row["sign_agreement"]),
        float(current_snapshot["regime_features"][0]),
        float(current_snapshot["regime_features"][1]),
        float(current_snapshot["regime_features"][2]),
        float(current_snapshot["regime_features"][3]),
        float(current_snapshot["regime_features"][4]),
        float(row["horizon"]) / max(max_horizon, 1),
    ]


def _build_selection_threshold_records(
    snapshots: Sequence[dict[str, object]],
    policy: dict[str, object],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for snapshot in snapshots:
        augmented = _augment_snapshot(snapshot, policy, config)
        records.append(
            {
                "regime_id": str(snapshot["regime_id"]),
                "horizon": int(augmented["selected_horizon"]),
                "score": float(augmented["selection_probability"]),
                "target": int(augmented["meta_label"]),
            }
        )
    return records


def _build_overlay_threshold_records(
    snapshots: Sequence[dict[str, object]],
    policy: dict[str, object],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    return [
        {
            "regime_id": str(snapshot["regime_id"]),
            "score": float(snapshot["overlay_probability"]),
            "target": int(snapshot["overlay_target"]),
        }
        for snapshot in snapshots
    ]


def _build_selection_thresholds(
    records: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> dict[str, object]:
    global_thresholds = {
        str(horizon): _calibrate_threshold(
            [
                record
                for record in records
                if int(record["horizon"]) == horizon
            ],
            config.precision_target,
        )
        for horizon in config.horizons
    }
    by_regime: dict[str, dict[str, float]] = {}
    regimes = sorted({str(record["regime_id"]) for record in records})
    for regime_id in regimes:
        thresholds: dict[str, float] = {}
        for horizon in config.horizons:
            horizon_records = [
                record
                for record in records
                if str(record["regime_id"]) == regime_id and int(record["horizon"]) == horizon
            ]
            if len(horizon_records) < 12:
                continue
            thresholds[str(horizon)] = _calibrate_threshold(horizon_records, config.precision_target)
        if thresholds:
            by_regime[regime_id] = thresholds
    return {
        "global": global_thresholds,
        "by_regime": by_regime,
    }


def _build_overlay_thresholds(
    records: Sequence[dict[str, object]],
    config: TrainingConfig,
) -> dict[str, object]:
    by_regime: dict[str, float] = {}
    regimes = sorted({str(record["regime_id"]) for record in records})
    for regime_id in regimes:
        regime_records = [record for record in records if str(record["regime_id"]) == regime_id]
        if len(regime_records) < 12:
            continue
        by_regime[regime_id] = _calibrate_threshold(regime_records, config.precision_target)
    return {
        "global": _calibrate_threshold(records, config.precision_target),
        "by_regime": by_regime,
    }


def _lookup_selection_threshold(
    policy: dict[str, object],
    regime_id: str,
    horizon: int,
) -> float:
    by_regime = dict(policy.get("selection_thresholds", {}).get("by_regime", {}))
    regime_thresholds = dict(by_regime.get(regime_id, {}))
    if str(horizon) in regime_thresholds:
        return float(regime_thresholds[str(horizon)])
    global_thresholds = dict(policy.get("selection_thresholds", {}).get("global", {}))
    return float(global_thresholds.get(str(horizon), 1.0))


def _lookup_overlay_threshold(
    policy: dict[str, object],
    regime_id: str,
) -> float:
    by_regime = dict(policy.get("overlay_thresholds", {}).get("by_regime", {}))
    if regime_id in by_regime:
        return float(by_regime[regime_id])
    return float(policy.get("overlay_thresholds", {}).get("global", 0.5))


def _fit_binary_model(
    feature_names: Sequence[str],
    feature_rows: Sequence[Sequence[float]],
    targets: Sequence[int],
    brier_weight: float,
) -> dict[str, object]:
    if not feature_rows:
        return _constant_model(0.5, feature_names)

    target_tensor = torch.tensor(list(targets), dtype=torch.float32)
    positive_rate = float(target_tensor.mean().item()) if len(targets) else 0.5
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return _constant_model(positive_rate, feature_names)

    inputs = torch.tensor(feature_rows, dtype=torch.float32)
    mean = inputs.mean(dim=0)
    std = inputs.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized_inputs = (inputs - mean) / std

    linear = torch.nn.Linear(normalized_inputs.size(1), 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()

    positive_count = float(target_tensor.sum().item())
    negative_count = float(len(targets) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.05, weight_decay=1e-3)

    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(normalized_inputs).squeeze(1)
        probabilities = torch.sigmoid(logits)
        loss = functional.binary_cross_entropy_with_logits(
            logits,
            target_tensor,
            pos_weight=pos_weight,
        )
        if brier_weight > 0.0:
            loss = loss + (brier_weight * torch.square(probabilities - target_tensor).mean())
        loss.backward()
        optimizer.step()

    return {
        "kind": "linear",
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": linear.weight.detach().reshape(-1).tolist(),
        "bias": float(linear.bias.detach().item()),
        "constant_probability": positive_rate,
    }


def _predict_binary_model(
    model: dict[str, object],
    feature_row: Sequence[float],
) -> float:
    if model.get("kind") != "linear":
        return float(model.get("constant_probability", 0.5))

    total = float(model["bias"])
    for value, mean, std, weight in zip(
        feature_row,
        model["mean"],
        model["std"],
        model["weights"],
    ):
        total += ((float(value) - float(mean)) / max(float(std), 1e-4)) * float(weight)
    return 1.0 / (1.0 + math.exp(-max(min(total, 20.0), -20.0)))


def _constant_model(
    probability: float,
    feature_names: Sequence[str],
) -> dict[str, object]:
    return {
        "kind": "constant",
        "feature_names": list(feature_names),
        "mean": [],
        "std": [],
        "weights": [],
        "bias": 0.0,
        "constant_probability": float(max(0.0, min(1.0, probability))),
    }


def _calibrate_threshold(
    records: Sequence[dict[str, object]],
    precision_target: float,
) -> float:
    if not records:
        return 1.0

    candidates = sorted({float(record["score"]) for record in records})
    best_threshold = 1.0
    best_coverage = 0.0

    for threshold in candidates:
        selected = [record for record in records if float(record["score"]) >= threshold]
        if not selected:
            continue
        precision = sum(int(record["target"]) for record in selected) / len(selected)
        coverage = len(selected) / len(records)
        if precision < precision_target:
            continue
        if coverage > best_coverage or (math.isclose(coverage, best_coverage) and threshold < best_threshold):
            best_threshold = threshold
            best_coverage = coverage

    if best_coverage == 0.0:
        return 1.0
    return float(max(0.05, min(0.99, best_threshold)))


def _sign_agreement(predicted_signs: Sequence[int], current_sign: int) -> float:
    actionable = [sign for sign in predicted_signs if sign != 0]
    if current_sign == 0 or not actionable:
        return 0.0
    agreeing = sum(1 for sign in actionable if sign == current_sign)
    return agreeing / len(actionable)


def _second_largest(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return ordered[1]


def _sign_from_return(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _argmax(values: Sequence[float]) -> int:
    best_index = 0
    best_value = float(values[0])
    for index, value in enumerate(values[1:], start=1):
        if float(value) > best_value:
            best_index = index
            best_value = float(value)
    return best_index
