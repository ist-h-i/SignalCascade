from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from ...domain.entities import STATE_VECTOR_COMPONENT_NAMES
from ...domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES


class ResidualCausalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (3 - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = outputs[..., : inputs.size(-1)]
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return inputs + outputs


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branch_projections = nn.ModuleList(
            [nn.Linear(feature_dim, hidden_dim) for _ in dilations]
        )
        self.branch_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualCausalConvBlock(hidden_dim, dilation=dilation, dropout=dropout),
                        ResidualCausalConvBlock(hidden_dim, dilation=dilation, dropout=dropout),
                    ]
                )
                for dilation in dilations
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim * len(dilations))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        branch_outputs: list[torch.Tensor] = []
        for projection, blocks in zip(self.branch_projections, self.branch_blocks):
            outputs = projection(sequence).transpose(1, 2)
            for block in blocks:
                outputs = block(outputs)
            pooled = 0.5 * (outputs[:, :, -1] + outputs.mean(dim=-1))
            branch_outputs.append(pooled)
        return self.output_norm(torch.cat(branch_outputs, dim=1))


class SignalCascadeModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        state_feature_dim: int,
        hidden_dim: int,
        state_dim: int,
        num_horizons: int,
        shape_classes: int,
        branch_dilations: tuple[int, ...],
        dropout: float,
        tie_policy_to_forecast_head: bool = False,
        disable_overlay_branch: bool = False,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.shape_classes = shape_classes
        self.tie_policy_to_forecast_head = bool(tie_policy_to_forecast_head)
        self.disable_overlay_branch = bool(disable_overlay_branch)
        encoded_dim = hidden_dim * len(branch_dilations)
        self.main_encoders = nn.ModuleDict(
            {
                timeframe: MultiScaleTemporalEncoder(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    dilations=branch_dilations,
                    dropout=dropout,
                )
                for timeframe in MAIN_TIMEFRAMES
            }
        )
        self.overlay_encoders = nn.ModuleDict(
            {
                timeframe: MultiScaleTemporalEncoder(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    dilations=branch_dilations,
                    dropout=dropout,
                )
                for timeframe in OVERLAY_TIMEFRAMES
            }
        )
        self.main_shape_heads = nn.ModuleDict(
            {timeframe: nn.Linear(encoded_dim, 3) for timeframe in MAIN_TIMEFRAMES}
        )
        total_latent_dim = encoded_dim * (len(MAIN_TIMEFRAMES) + len(OVERLAY_TIMEFRAMES))
        self.latent_fusion = nn.Sequential(
            nn.Linear(total_latent_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
        )
        fused_dim = hidden_dim * 2
        self.shape_head = nn.Linear(fused_dim, shape_classes)
        self.state_feature_projection = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.memory_input = nn.Linear(fused_dim + shape_classes + hidden_dim, state_dim)
        self.memory_transition = nn.Linear(state_dim, state_dim, bias=False)
        self.state_vector_component_dims = {
            "h_t": fused_dim,
            "s_t": shape_classes,
            "z_t": hidden_dim,
            "m_t": state_dim,
        }
        state_vector_dim = fused_dim + shape_classes + hidden_dim + state_dim
        self.forecast_expert_mean_head = nn.Linear(state_vector_dim, num_horizons * shape_classes)
        self.forecast_expert_logvar_head = nn.Linear(state_vector_dim, num_horizons * shape_classes)
        self.policy_expert_mean_head = nn.Linear(state_vector_dim, num_horizons * shape_classes)
        self.policy_expert_logvar_head = nn.Linear(state_vector_dim, num_horizons * shape_classes)
        self.tradeability_logits = nn.Parameter(torch.zeros(shape_classes))
        self.state_vector_dim = state_vector_dim

    def forward(
        self,
        main_sequences: dict[str, torch.Tensor],
        overlay_sequences: dict[str, torch.Tensor],
        state_features: torch.Tensor,
        previous_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        device = state_features.device
        batch_size = state_features.size(0)
        if previous_state is None:
            previous_state = torch.zeros(
                batch_size,
                self.memory_transition.in_features,
                dtype=state_features.dtype,
                device=device,
            )

        main_latents = {
            timeframe: self.main_encoders[timeframe](main_sequences[timeframe])
            for timeframe in MAIN_TIMEFRAMES
        }
        overlay_latents = {
            timeframe: (
                torch.zeros_like(main_latents[MAIN_TIMEFRAMES[0]])
                if self.disable_overlay_branch
                else self.overlay_encoders[timeframe](overlay_sequences[timeframe])
            )
            for timeframe in OVERLAY_TIMEFRAMES
        }
        fused_latent = self.latent_fusion(
            torch.cat(
                [main_latents[timeframe] for timeframe in MAIN_TIMEFRAMES]
                + [overlay_latents[timeframe] for timeframe in OVERLAY_TIMEFRAMES],
                dim=1,
            )
        )
        shape_feature = fused_latent
        shape_posterior_logits = self.shape_head(shape_feature)
        shape_posterior = functional.softmax(shape_posterior_logits, dim=-1)
        state_projection = self.state_feature_projection(state_features)
        memory_input = torch.cat([shape_feature, shape_posterior, state_projection], dim=1)
        memory_state = torch.tanh(
            self.memory_input(memory_input) + self.memory_transition(previous_state)
        )
        state_vector_components = {
            "h_t": shape_feature,
            "s_t": shape_posterior,
            "z_t": state_projection,
            "m_t": memory_state,
        }
        state_vector = torch.cat(
            [state_vector_components[name] for name in STATE_VECTOR_COMPONENT_NAMES],
            dim=1,
        )

        forecast_mu_by_shape = self.forecast_expert_mean_head(state_vector).view(
            batch_size,
            self.num_horizons,
            self.shape_classes,
        )
        forecast_sigma_by_shape = functional.softplus(
            self.forecast_expert_logvar_head(state_vector).view(
                batch_size,
                self.num_horizons,
                self.shape_classes,
            )
        ) + 1e-4
        policy_mu_by_shape = self.policy_expert_mean_head(state_vector).view(
            batch_size,
            self.num_horizons,
            self.shape_classes,
        )
        policy_sigma_by_shape = functional.softplus(
            self.policy_expert_logvar_head(state_vector).view(
                batch_size,
                self.num_horizons,
                self.shape_classes,
            )
        ) + 1e-4
        mixture_weights = shape_posterior.unsqueeze(1)
        forecast_mean = torch.sum(forecast_mu_by_shape * mixture_weights, dim=-1)
        forecast_second_moment = torch.sum(
            mixture_weights * (forecast_sigma_by_shape.pow(2) + forecast_mu_by_shape.pow(2)),
            dim=-1,
        )
        forecast_variance = torch.clamp(
            forecast_second_moment - forecast_mean.pow(2),
            min=1e-6,
        )
        forecast_sigma = torch.sqrt(forecast_variance)
        if self.tie_policy_to_forecast_head:
            policy_mu_by_shape = forecast_mu_by_shape
            policy_sigma_by_shape = forecast_sigma_by_shape
            policy_mean = forecast_mean
            policy_sigma = forecast_sigma
        else:
            policy_mean = torch.sum(policy_mu_by_shape * mixture_weights, dim=-1)
            policy_second_moment = torch.sum(
                mixture_weights * (policy_sigma_by_shape.pow(2) + policy_mu_by_shape.pow(2)),
                dim=-1,
            )
            policy_variance = torch.clamp(policy_second_moment - policy_mean.pow(2), min=1e-6)
            policy_sigma = torch.sqrt(policy_variance)
        tradeability_weights = torch.sigmoid(self.tradeability_logits)
        tradeability_gate = torch.sum(
            shape_posterior * tradeability_weights.unsqueeze(0),
            dim=-1,
        )
        entropy = -torch.sum(
            shape_posterior * torch.log(shape_posterior.clamp_min(1e-6)),
            dim=-1,
        )
        normalized_entropy = entropy / math.log(max(self.shape_classes, 2))
        main_shape_predictions = {
            timeframe: self.main_shape_heads[timeframe](main_latents[timeframe])
            for timeframe in MAIN_TIMEFRAMES
        }

        return {
            "mu": forecast_mean,
            "sigma": forecast_sigma,
            "forecast_mu": forecast_mean,
            "forecast_sigma": forecast_sigma,
            "policy_mu": policy_mean,
            "policy_sigma": policy_sigma,
            "shape_feature": shape_feature,
            "shape_posterior_logits": shape_posterior_logits,
            "shape_posterior": shape_posterior,
            "shape_entropy": normalized_entropy,
            "tradeability_gate": tradeability_gate,
            "tradeability_weights": tradeability_weights,
            "policy_head_tied_to_forecast": torch.tensor(
                self.tie_policy_to_forecast_head,
                dtype=torch.bool,
                device=device,
            ),
            "overlay_branch_disabled": torch.tensor(
                self.disable_overlay_branch,
                dtype=torch.bool,
                device=device,
            ),
            "state_projection": state_projection,
            "state_vector": state_vector,
            "state_vector_components": state_vector_components,
            "state_vector_component_dims": dict(self.state_vector_component_dims),
            "memory_state": memory_state,
            "expert_mu_by_shape": forecast_mu_by_shape,
            "expert_sigma_by_shape": forecast_sigma_by_shape,
            "forecast_expert_mu_by_shape": forecast_mu_by_shape,
            "forecast_expert_sigma_by_shape": forecast_sigma_by_shape,
            "policy_expert_mu_by_shape": policy_mu_by_shape,
            "policy_expert_sigma_by_shape": policy_sigma_by_shape,
            "shape_conditioned_experts": {
                "mu_by_shape": forecast_mu_by_shape,
                "sigma_by_shape": forecast_sigma_by_shape,
                "mixture_weights": mixture_weights,
            },
            "forecast_shape_conditioned_experts": {
                "mu_by_shape": forecast_mu_by_shape,
                "sigma_by_shape": forecast_sigma_by_shape,
                "mixture_weights": mixture_weights,
            },
            "policy_shape_conditioned_experts": {
                "mu_by_shape": policy_mu_by_shape,
                "sigma_by_shape": policy_sigma_by_shape,
                "mixture_weights": mixture_weights,
            },
            "main_shape_predictions": main_shape_predictions,
            "shape_logits": shape_posterior_logits,
            "shape_probs": shape_posterior,
            "state_features_projected": state_projection,
            "internal_state": memory_state,
            "next_state": memory_state,
            "expert_mu": forecast_mu_by_shape,
            "expert_sigma": forecast_sigma_by_shape,
            "shape_predictions": main_shape_predictions,
        }
