from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from ...domain.timeframes import MAIN_TIMEFRAMES, OVERLAY_TIMEFRAMES


class ResidualTemporalBlock(nn.Module):
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
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = outputs[..., : inputs.size(-1)]
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return inputs + outputs


class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(hidden_dim, dilation=1, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dilation=2, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dilation=4, dropout=dropout),
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs = self.input_projection(sequence).transpose(1, 2)
        for block in self.blocks:
            outputs = block(outputs)
        return self.output_norm(outputs[:, :, -1])


class SignalCascadeModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_horizons: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.main_encoders = nn.ModuleDict(
            {
                timeframe: TemporalEncoder(feature_dim, hidden_dim, dropout)
                for timeframe in MAIN_TIMEFRAMES
            }
        )
        self.overlay_encoders = nn.ModuleDict(
            {
                timeframe: TemporalEncoder(feature_dim, hidden_dim, dropout)
                for timeframe in OVERLAY_TIMEFRAMES
            }
        )
        self.main_shape_heads = nn.ModuleDict(
            {timeframe: nn.Linear(hidden_dim, 3) for timeframe in MAIN_TIMEFRAMES}
        )
        self.main_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(MAIN_TIMEFRAMES), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.return_mean_head = nn.Linear(hidden_dim, num_horizons)
        self.return_scale_head = nn.Linear(hidden_dim, num_horizons)
        self.direction_head = nn.Linear(hidden_dim, num_horizons * 3)
        overlay_context_dim = (hidden_dim * 3) + num_horizons
        self.overlay_head = nn.Sequential(
            nn.Linear(overlay_context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        main_sequences: dict[str, torch.Tensor],
        overlay_sequences: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        main_latents = {
            timeframe: self.main_encoders[timeframe](main_sequences[timeframe])
            for timeframe in MAIN_TIMEFRAMES
        }
        fused = self.main_fusion(torch.cat([main_latents[timeframe] for timeframe in MAIN_TIMEFRAMES], dim=1))
        mean = self.return_mean_head(fused)
        sigma = functional.softplus(self.return_scale_head(fused)) + 1e-4
        direction_logits = self.direction_head(fused).view(fused.size(0), self.num_horizons, 3)
        overlay_latents = {
            timeframe: self.overlay_encoders[timeframe](overlay_sequences[timeframe])
            for timeframe in OVERLAY_TIMEFRAMES
        }
        edge = mean / sigma
        overlay_input = torch.cat(
            [fused, overlay_latents["1h"], overlay_latents["30m"], edge],
            dim=1,
        )
        return {
            "mu": mean,
            "sigma": sigma,
            "direction_logits": direction_logits,
            "shape_predictions": {
                timeframe: self.main_shape_heads[timeframe](main_latents[timeframe])
                for timeframe in MAIN_TIMEFRAMES
            },
            "overlay_logits": self.overlay_head(overlay_input),
        }
