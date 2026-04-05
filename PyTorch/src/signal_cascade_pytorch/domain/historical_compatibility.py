from __future__ import annotations

from typing import Protocol


class PredictionResultLike(Protocol):
    policy_horizon: int
    executed_horizon: int | None
    position: float
    trade_delta: float
    no_trade_band_hit: bool
    tradeability_gate: float
    policy_score: float
    expected_log_returns: dict[str, float]


def prediction_selected_direction(prediction: PredictionResultLike) -> int:
    mean = float(prediction.expected_log_returns.get(str(prediction.policy_horizon), 0.0))
    return sign_from_value(mean)


def build_prediction_legacy_compatibility(
    prediction: PredictionResultLike,
) -> dict[str, object]:
    return {
        "proposed_horizon": int(prediction.policy_horizon),
        "accepted_horizon": prediction.executed_horizon,
        "selected_horizon": int(prediction.policy_horizon),
        "selected_direction": prediction_selected_direction(prediction),
        "accepted_signal": prediction.executed_horizon is not None
        and (abs(prediction.position) > 1e-9 or abs(prediction.trade_delta) > 1e-9),
        "selection_probability": float(prediction.tradeability_gate),
        "selection_score": float(prediction.policy_score),
        "selection_threshold": None,
        "threshold_status": "retired",
        "threshold_origin": "profit_policy",
        "correctness_probability": float(prediction.tradeability_gate),
        "hold_probability": max(0.0, 1.0 - min(abs(prediction.position), 1.0)),
        "hold_threshold": 0.0,
        "overlay_action": "hold" if prediction.no_trade_band_hit else "reduce",
    }


def build_policy_decision_legacy_compatibility(
    *,
    policy_horizon: int,
    executed_horizon: int | None,
    position: float,
    trade_delta: float,
    no_trade_band_hit: bool,
    tradeability_gate: float,
    selected_policy_utility: float,
    selected_direction: int,
    meta_label: int,
    direction_correct: int,
    candidate_count: int,
) -> dict[str, object]:
    accepted_signal = executed_horizon is not None and (
        abs(position) > 1e-9 or abs(trade_delta) > 1e-9
    )
    return {
        "proposed_horizon": int(policy_horizon),
        "accepted_horizon": executed_horizon,
        "selected_horizon": int(policy_horizon),
        "selected_direction": int(selected_direction),
        "accepted_signal": accepted_signal,
        "selection_probability": float(tradeability_gate),
        "selection_score": float(selected_policy_utility),
        "selection_score_source": "profit_utility",
        "selection_threshold": None,
        "threshold_status": "retired",
        "threshold_origin": "profit_policy",
        "stored_threshold_compatibility": "retired",
        "threshold_score_source": "profit_utility",
        "precision_infeasible": False,
        "correctness_probability": float(tradeability_gate),
        "hold_probability": max(0.0, 1.0 - min(abs(position), 1.0)),
        "hold_threshold": 0.0,
        "overlay_action": "hold" if no_trade_band_hit else "reduce",
        "expected_direction": int(selected_direction),
        "direction_alignment": True,
        "accept_reject_reason": "no_trade_band" if no_trade_band_hit else "executed",
        "reject_flags": {
            "no_trade_band": bool(no_trade_band_hit),
            "retired_threshold_policy": True,
        },
        "meta_label": int(meta_label),
        "direction_correct": int(direction_correct),
        "candidate_count": int(candidate_count),
        "strict_candidate_count": int(candidate_count),
        "any_candidate": bool(candidate_count),
        "any_strict_candidate": bool(candidate_count),
    }


def sign_from_value(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0
