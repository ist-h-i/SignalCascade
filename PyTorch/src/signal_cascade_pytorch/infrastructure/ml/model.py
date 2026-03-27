from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

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
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.shape_classes = shape_classes
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
        state_vector_dim = fused_dim + shape_classes + hidden_dim + state_dim
        self.expert_mean_head = nn.Linear(state_vector_dim, num_horizons * shape_classes)
        self.expert_logvar_head = nn.Linear(state_vector_dim, num_horizons * shape_classes)
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
            timeframe: self.overlay_encoders[timeframe](overlay_sequences[timeframe])
            for timeframe in OVERLAY_TIMEFRAMES
        }
        fused_latent = self.latent_fusion(
            torch.cat(
                [main_latents[timeframe] for timeframe in MAIN_TIMEFRAMES]
                + [overlay_latents[timeframe] for timeframe in OVERLAY_TIMEFRAMES],
                dim=1,
            )
        )
        shape_logits = self.shape_head(fused_latent)
        shape_probs = functional.softmax(shape_logits, dim=-1)
        projected_state_features = self.state_feature_projection(state_features)
        memory_input = torch.cat([fused_latent, shape_probs, projected_state_features], dim=1)
        next_state = torch.tanh(self.memory_input(memory_input) + self.memory_transition(previous_state))
        state_vector = torch.cat(
            [fused_latent, shape_probs, projected_state_features, next_state],
            dim=1,
        )

        expert_mean = self.expert_mean_head(state_vector).view(
            batch_size,
            self.num_horizons,
            self.shape_classes,
        )
        expert_sigma = functional.softplus(
            self.expert_logvar_head(state_vector).view(
                batch_size,
                self.num_horizons,
                self.shape_classes,
            )
        ) + 1e-4
        mixture_weights = shape_probs.unsqueeze(1)
        mean = torch.sum(expert_mean * mixture_weights, dim=-1)
        second_moment = torch.sum(
            mixture_weights * (expert_sigma.pow(2) + expert_mean.pow(2)),
            dim=-1,
        )
        variance = torch.clamp(second_moment - mean.pow(2), min=1e-6)
        sigma = torch.sqrt(variance)
        tradeability_weights = torch.sigmoid(self.tradeability_logits)
        tradeability_gate = torch.sum(shape_probs * tradeability_weights.unsqueeze(0), dim=-1)
        entropy = -torch.sum(shape_probs * torch.log(shape_probs.clamp_min(1e-6)), dim=-1)
        normalized_entropy = entropy / math.log(max(self.shape_classes, 2))

        return {
            "mu": mean,
            "sigma": sigma,
            "shape_logits": shape_logits,
            "shape_probs": shape_probs,
            "shape_entropy": normalized_entropy,
            "tradeability_gate": tradeability_gate,
            "tradeability_weights": tradeability_weights,
            "state_vector": state_vector,
            "state_features_projected": projected_state_features,
            "internal_state": next_state,
            "next_state": next_state,
            "expert_mu": expert_mean,
            "expert_sigma": expert_sigma,
            "shape_predictions": {
                timeframe: self.main_shape_heads[timeframe](main_latents[timeframe])
                for timeframe in MAIN_TIMEFRAMES
            },
        }
