from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.close_anchor import TimeframeParameters
from ..domain.entities import (
    STATE_FEATURE_NAMES,
    STATE_VECTOR_COMPONENT_NAMES,
    TIMEFRAME_FEATURE_NAMES,
    TRAINING_EXAMPLE_CONTRACT_VERSION,
)
from ..domain.timeframes import HORIZONS


CONFIG_SCHEMA_VERSION = 7
LEGACY_CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SELECTION_METRICS = (
    "hybrid_exact",
    "exact_log_wealth",
    "exact_log_wealth_minus_lambda_cvar",
    "validation_total",
)


def _default_main_windows() -> dict[str, int]:
    return {"4h": 48, "1d": 21, "1w": 6}


def _default_overlay_windows() -> dict[str, int]:
    return {"1h": 48, "30m": 96}


def _default_timeframe_feature_names() -> tuple[str, ...]:
    return TIMEFRAME_FEATURE_NAMES


def _default_state_feature_names() -> tuple[str, ...]:
    return STATE_FEATURE_NAMES


def _default_state_vector_component_names() -> tuple[str, ...]:
    return STATE_VECTOR_COMPONENT_NAMES


def _default_timeframe_parameters() -> dict[str, TimeframeParameters]:
    return {
        "30m": TimeframeParameters(ema_window=32, gate_weights=(0.60, 0.30, -0.20)),
        "1h": TimeframeParameters(ema_window=32, gate_weights=(0.60, 0.30, -0.20)),
        "4h": TimeframeParameters(ema_window=24, gate_weights=(0.75, 0.40, -0.25)),
        "1d": TimeframeParameters(ema_window=20, gate_weights=(0.85, 0.45, -0.30)),
        "1w": TimeframeParameters(ema_window=13, gate_weights=(0.95, 0.50, -0.35)),
    }


def _default_diagnostic_state_reset_modes() -> tuple[str, ...]:
    return ("carry_on", "reset_each_session_or_window", "reset_each_example")


def _legacy_diagnostic_state_reset_modes() -> tuple[str, ...]:
    return ("carry_on", "reset_each_example", "reset_each_session_or_window")


def _default_policy_sweep_cost_multipliers() -> tuple[float, ...]:
    return (0.5, 1.0, 2.0, 4.0)


def _default_policy_sweep_gamma_multipliers() -> tuple[float, ...]:
    return (0.5, 1.0, 2.0)


def _default_policy_sweep_min_policy_sigmas() -> tuple[float, ...]:
    return (5e-5, 1e-4, 2e-4)


def _default_policy_sweep_q_max_values() -> tuple[float, ...]:
    return (1.0,)


def _default_policy_sweep_cvar_weights() -> tuple[float, ...]:
    return (0.20,)


def _default_policy_sweep_state_reset_modes() -> tuple[str, ...]:
    return ("carry_on", "reset_each_session_or_window")


def _legacy_policy_sweep_state_reset_modes() -> tuple[str, ...]:
    return ("carry_on",)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    synthetic_bars: int = 10_080
    epochs: int = 8
    warmup_epochs: int = 2
    oof_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 48
    state_dim: int = 24
    shape_classes: int = 6
    dropout: float = 0.1
    tie_policy_to_forecast_head: bool = False
    disable_overlay_branch: bool = False
    train_ratio: float = 0.8
    base_cost: float = 6e-4
    delta_multiplier: float = 1.35
    mae_multiplier: float = 0.95
    overlay_delta_multiplier: float = 0.75
    overlay_mae_multiplier: float = 0.7
    clean_weight_return_scale: float = 0.75
    clean_weight_bonus: float = 0.65
    clean_weight_ratio_scale: float = 0.35
    return_loss_weight: float = 0.15
    shape_loss_weight: float = 0.05
    profit_loss_weight: float = 1.0
    cvar_weight: float = 0.20
    cvar_alpha: float = 0.10
    risk_aversion_gamma: float = 3.0
    policy_cost_multiplier: float = 1.0
    policy_gamma_multiplier: float = 1.0
    q_max: float = 1.0
    policy_abs_epsilon: float = 1e-4
    policy_smoothing_beta: float = 15.0
    min_policy_sigma: float = 1e-4
    branch_dilations: tuple[int, ...] = (1, 2, 4)
    walk_forward_folds: int = 3
    position_scale: float = 1.0
    allow_no_candidate: bool = False
    selection_score_source: str = "profit_utility"
    training_state_reset_mode: str = "carry_on"
    evaluation_state_reset_mode: str = "carry_on"
    diagnostic_state_reset_modes: tuple[str, ...] = field(
        default_factory=_default_diagnostic_state_reset_modes
    )
    policy_sweep_cost_multipliers: tuple[float, ...] = field(
        default_factory=_default_policy_sweep_cost_multipliers
    )
    policy_sweep_gamma_multipliers: tuple[float, ...] = field(
        default_factory=_default_policy_sweep_gamma_multipliers
    )
    policy_sweep_min_policy_sigmas: tuple[float, ...] = field(
        default_factory=_default_policy_sweep_min_policy_sigmas
    )
    policy_sweep_q_max_values: tuple[float, ...] = field(
        default_factory=_default_policy_sweep_q_max_values
    )
    policy_sweep_cvar_weights: tuple[float, ...] = field(
        default_factory=_default_policy_sweep_cvar_weights
    )
    policy_sweep_state_reset_modes: tuple[str, ...] = field(
        default_factory=_default_policy_sweep_state_reset_modes
    )
    checkpoint_selection_metric: str = "exact_log_wealth_minus_lambda_cvar"
    checkpoint_selection_forecast_weight: float = 1.0
    checkpoint_selection_calibration_weight: float = 0.5
    checkpoint_selection_position_gap_weight: float = 0.25
    requested_price_scale: float | None = None
    output_dir: str = "artifacts/demo"
    horizons: tuple[int, ...] = HORIZONS
    main_windows: dict[str, int] = field(default_factory=_default_main_windows)
    overlay_windows: dict[str, int] = field(default_factory=_default_overlay_windows)
    feature_contract_version: int = TRAINING_EXAMPLE_CONTRACT_VERSION
    timeframe_feature_names: tuple[str, ...] = field(default_factory=_default_timeframe_feature_names)
    state_feature_names: tuple[str, ...] = field(default_factory=_default_state_feature_names)
    state_vector_component_names: tuple[str, ...] = field(
        default_factory=_default_state_vector_component_names
    )
    timeframe_parameters: dict[str, TimeframeParameters] = field(
        default_factory=_default_timeframe_parameters
    )

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def purge_examples(self) -> int:
        return max(self.max_horizon, 1)

    def cost_for_horizon(self, horizon: int) -> float:
        return self.base_cost * (horizon**0.5)

    def to_dict(self) -> dict[str, object]:
        return {
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "seed": self.seed,
            "synthetic_bars": self.synthetic_bars,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "oof_epochs": self.oof_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "state_dim": self.state_dim,
            "shape_classes": self.shape_classes,
            "dropout": self.dropout,
            "tie_policy_to_forecast_head": self.tie_policy_to_forecast_head,
            "disable_overlay_branch": self.disable_overlay_branch,
            "train_ratio": self.train_ratio,
            "base_cost": self.base_cost,
            "delta_multiplier": self.delta_multiplier,
            "mae_multiplier": self.mae_multiplier,
            "overlay_delta_multiplier": self.overlay_delta_multiplier,
            "overlay_mae_multiplier": self.overlay_mae_multiplier,
            "clean_weight_return_scale": self.clean_weight_return_scale,
            "clean_weight_bonus": self.clean_weight_bonus,
            "clean_weight_ratio_scale": self.clean_weight_ratio_scale,
            "return_loss_weight": self.return_loss_weight,
            "shape_loss_weight": self.shape_loss_weight,
            "profit_loss_weight": self.profit_loss_weight,
            "cvar_weight": self.cvar_weight,
            "cvar_alpha": self.cvar_alpha,
            "risk_aversion_gamma": self.risk_aversion_gamma,
            "policy_cost_multiplier": self.policy_cost_multiplier,
            "policy_gamma_multiplier": self.policy_gamma_multiplier,
            "q_max": self.q_max,
            "policy_abs_epsilon": self.policy_abs_epsilon,
            "policy_smoothing_beta": self.policy_smoothing_beta,
            "min_policy_sigma": self.min_policy_sigma,
            "branch_dilations": list(self.branch_dilations),
            "walk_forward_folds": self.walk_forward_folds,
            "position_scale": self.position_scale,
            "allow_no_candidate": self.allow_no_candidate,
            "selection_score_source": self.selection_score_source,
            "training_state_reset_mode": self.training_state_reset_mode,
            "evaluation_state_reset_mode": self.evaluation_state_reset_mode,
            "diagnostic_state_reset_modes": list(self.diagnostic_state_reset_modes),
            "policy_sweep_cost_multipliers": list(self.policy_sweep_cost_multipliers),
            "policy_sweep_gamma_multipliers": list(self.policy_sweep_gamma_multipliers),
            "policy_sweep_min_policy_sigmas": list(self.policy_sweep_min_policy_sigmas),
            "policy_sweep_q_max_values": list(self.policy_sweep_q_max_values),
            "policy_sweep_cvar_weights": list(self.policy_sweep_cvar_weights),
            "policy_sweep_state_reset_modes": list(self.policy_sweep_state_reset_modes),
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
            "checkpoint_selection_forecast_weight": self.checkpoint_selection_forecast_weight,
            "checkpoint_selection_calibration_weight": self.checkpoint_selection_calibration_weight,
            "checkpoint_selection_position_gap_weight": self.checkpoint_selection_position_gap_weight,
            "requested_price_scale": self.requested_price_scale,
            "output_dir": self.output_dir,
            "horizons": list(self.horizons),
            "main_windows": dict(self.main_windows),
            "overlay_windows": dict(self.overlay_windows),
            "feature_contract_version": self.feature_contract_version,
            "timeframe_feature_names": list(self.timeframe_feature_names),
            "state_feature_names": list(self.state_feature_names),
            "state_vector_component_names": list(self.state_vector_component_names),
            "timeframe_parameters": {
                key: value.to_dict() for key, value in self.timeframe_parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TrainingConfig":
        config_schema_version = int(
            payload.get("config_schema_version", LEGACY_CONFIG_SCHEMA_VERSION)
        )
        timeframe_parameters = {
            key: TimeframeParameters.from_dict(value)
            for key, value in dict(payload["timeframe_parameters"]).items()
        }
        if config_schema_version >= CONFIG_SCHEMA_VERSION:
            training_state_reset_mode = str(payload["training_state_reset_mode"])
            evaluation_state_reset_mode = str(payload["evaluation_state_reset_mode"])
            diagnostic_state_reset_modes = tuple(
                str(value) for value in payload["diagnostic_state_reset_modes"]
            )
            policy_sweep_cost_multipliers = tuple(
                float(value) for value in payload["policy_sweep_cost_multipliers"]
            )
            policy_sweep_gamma_multipliers = tuple(
                float(value) for value in payload["policy_sweep_gamma_multipliers"]
            )
            policy_sweep_min_policy_sigmas = tuple(
                float(value) for value in payload["policy_sweep_min_policy_sigmas"]
            )
            policy_sweep_q_max_values = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_q_max_values",
                    [payload.get("q_max", 1.0)],
                )
            )
            policy_sweep_cvar_weights = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_cvar_weights",
                    [payload.get("cvar_weight", 0.20)],
                )
            )
            policy_sweep_state_reset_modes = tuple(
                str(value) for value in payload["policy_sweep_state_reset_modes"]
            )
            checkpoint_selection_metric = str(
                payload.get("checkpoint_selection_metric", "exact_log_wealth_minus_lambda_cvar")
            )
            checkpoint_selection_forecast_weight = float(
                payload.get("checkpoint_selection_forecast_weight", 1.0)
            )
            checkpoint_selection_calibration_weight = float(
                payload.get("checkpoint_selection_calibration_weight", 0.5)
            )
            checkpoint_selection_position_gap_weight = float(
                payload.get("checkpoint_selection_position_gap_weight", 0.25)
            )
            requested_price_scale = payload.get("requested_price_scale")
            requested_price_scale = (
                None if requested_price_scale is None else float(requested_price_scale)
            )
        else:
            training_state_reset_mode = str(payload.get("training_state_reset_mode", "carry_on"))
            evaluation_state_reset_mode = str(payload.get("evaluation_state_reset_mode", "carry_on"))
            diagnostic_state_reset_modes = tuple(
                str(value)
                for value in payload.get(
                    "diagnostic_state_reset_modes",
                    list(_legacy_diagnostic_state_reset_modes()),
                )
            )
            policy_sweep_cost_multipliers = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_cost_multipliers",
                    list(_default_policy_sweep_cost_multipliers()),
                )
            )
            policy_sweep_gamma_multipliers = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_gamma_multipliers",
                    list(_default_policy_sweep_gamma_multipliers()),
                )
            )
            policy_sweep_min_policy_sigmas = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_min_policy_sigmas",
                    [payload.get("min_policy_sigma", 1e-4)],
                )
            )
            policy_sweep_q_max_values = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_q_max_values",
                    [payload.get("q_max", 1.0)],
                )
            )
            policy_sweep_cvar_weights = tuple(
                float(value)
                for value in payload.get(
                    "policy_sweep_cvar_weights",
                    [payload.get("cvar_weight", 0.20)],
                )
            )
            policy_sweep_state_reset_modes = tuple(
                str(value)
                for value in payload.get(
                    "policy_sweep_state_reset_modes",
                    list(_legacy_policy_sweep_state_reset_modes()),
                )
            )
            checkpoint_selection_metric = str(
                payload.get("checkpoint_selection_metric", "exact_log_wealth_minus_lambda_cvar")
            )
            checkpoint_selection_forecast_weight = float(
                payload.get("checkpoint_selection_forecast_weight", 1.0)
            )
            checkpoint_selection_calibration_weight = float(
                payload.get("checkpoint_selection_calibration_weight", 0.5)
            )
            checkpoint_selection_position_gap_weight = float(
                payload.get("checkpoint_selection_position_gap_weight", 0.25)
            )
            requested_price_scale = payload.get("requested_price_scale", payload.get("price_scale"))
            requested_price_scale = (
                None if requested_price_scale is None else float(requested_price_scale)
            )
        feature_contract_version = int(
            payload.get("feature_contract_version", TRAINING_EXAMPLE_CONTRACT_VERSION)
        )
        timeframe_feature_names = tuple(
            str(value)
            for value in payload.get(
                "timeframe_feature_names",
                list(_default_timeframe_feature_names()),
            )
        )
        state_feature_names = tuple(
            str(value)
            for value in payload.get(
                "state_feature_names",
                list(_default_state_feature_names()),
            )
        )
        state_vector_component_names = tuple(
            str(value)
            for value in payload.get(
                "state_vector_component_names",
                list(_default_state_vector_component_names()),
            )
        )
        return cls(
            seed=int(payload["seed"]),
            synthetic_bars=int(payload["synthetic_bars"]),
            epochs=int(payload["epochs"]),
            warmup_epochs=int(payload.get("warmup_epochs", 2)),
            oof_epochs=int(payload.get("oof_epochs", 3)),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            hidden_dim=int(payload["hidden_dim"]),
            state_dim=int(payload.get("state_dim", 24)),
            shape_classes=int(payload.get("shape_classes", 6)),
            dropout=float(payload["dropout"]),
            tie_policy_to_forecast_head=bool(payload.get("tie_policy_to_forecast_head", False)),
            disable_overlay_branch=bool(payload.get("disable_overlay_branch", False)),
            train_ratio=float(payload["train_ratio"]),
            base_cost=float(payload.get("base_cost", 6e-4)),
            delta_multiplier=float(payload.get("delta_multiplier", 1.35)),
            mae_multiplier=float(payload.get("mae_multiplier", 0.95)),
            overlay_delta_multiplier=float(payload.get("overlay_delta_multiplier", 0.75)),
            overlay_mae_multiplier=float(payload.get("overlay_mae_multiplier", 0.7)),
            clean_weight_return_scale=float(payload.get("clean_weight_return_scale", 0.75)),
            clean_weight_bonus=float(payload.get("clean_weight_bonus", 0.65)),
            clean_weight_ratio_scale=float(payload.get("clean_weight_ratio_scale", 0.35)),
            return_loss_weight=float(payload.get("return_loss_weight", 0.15)),
            shape_loss_weight=float(payload.get("shape_loss_weight", 0.05)),
            profit_loss_weight=float(payload.get("profit_loss_weight", 1.0)),
            cvar_weight=float(payload.get("cvar_weight", 0.20)),
            cvar_alpha=float(payload.get("cvar_alpha", 0.10)),
            risk_aversion_gamma=float(payload.get("risk_aversion_gamma", 3.0)),
            policy_cost_multiplier=float(payload.get("policy_cost_multiplier", 1.0)),
            policy_gamma_multiplier=float(payload.get("policy_gamma_multiplier", 1.0)),
            q_max=float(payload.get("q_max", 1.0)),
            policy_abs_epsilon=float(payload.get("policy_abs_epsilon", 1e-4)),
            policy_smoothing_beta=float(payload.get("policy_smoothing_beta", 15.0)),
            min_policy_sigma=float(payload.get("min_policy_sigma", 1e-4)),
            branch_dilations=tuple(int(value) for value in payload.get("branch_dilations", [1, 2, 4])),
            walk_forward_folds=int(payload.get("walk_forward_folds", 3)),
            position_scale=float(payload.get("position_scale", 1.0)),
            allow_no_candidate=bool(payload.get("allow_no_candidate", False)),
            selection_score_source=str(payload.get("selection_score_source", "profit_utility")),
            training_state_reset_mode=training_state_reset_mode,
            evaluation_state_reset_mode=evaluation_state_reset_mode,
            diagnostic_state_reset_modes=diagnostic_state_reset_modes,
            policy_sweep_cost_multipliers=policy_sweep_cost_multipliers,
            policy_sweep_gamma_multipliers=policy_sweep_gamma_multipliers,
            policy_sweep_min_policy_sigmas=policy_sweep_min_policy_sigmas,
            policy_sweep_q_max_values=policy_sweep_q_max_values,
            policy_sweep_cvar_weights=policy_sweep_cvar_weights,
            policy_sweep_state_reset_modes=policy_sweep_state_reset_modes,
            checkpoint_selection_metric=checkpoint_selection_metric,
            checkpoint_selection_forecast_weight=checkpoint_selection_forecast_weight,
            checkpoint_selection_calibration_weight=checkpoint_selection_calibration_weight,
            checkpoint_selection_position_gap_weight=checkpoint_selection_position_gap_weight,
            requested_price_scale=requested_price_scale,
            output_dir=str(payload["output_dir"]),
            horizons=tuple(int(value) for value in payload["horizons"]),
            main_windows={key: int(value) for key, value in dict(payload["main_windows"]).items()},
            overlay_windows={
                key: int(value) for key, value in dict(payload["overlay_windows"]).items()
            },
            feature_contract_version=feature_contract_version,
            timeframe_feature_names=timeframe_feature_names,
            state_feature_names=state_feature_names,
            state_vector_component_names=state_vector_component_names,
            timeframe_parameters=timeframe_parameters,
        )
