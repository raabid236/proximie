"""Lightweight temporal classification model."""

from __future__ import annotations

import torch
from torch import nn


class TemporalGRU(nn.Module):
    """Small GRU over concatenated multi-camera feature embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        y, _ = self.gru(x)
        return self.head(y)


def fuse_sensor_features(
    features: torch.Tensor, quality: torch.Tensor
) -> torch.Tensor:
    """Quality-aware late fusion.

    Weighted mean retains the common semantic signal while reducing the impact
    of a heavily occluded camera. Quality is also appended as a small side
    channel so the temporal model knows when the fused representation is less
    reliable.
    """
    # features: [B, T, S, D], quality: [B, T, S]
    weights = quality.unsqueeze(-1).clamp_min(0.05)
    weighted = (features * weights).sum(dim=2) / weights.sum(dim=2)
    q_mean = quality.mean(dim=2, keepdim=True)
    q_min = quality.min(dim=2, keepdim=True).values
    return torch.cat([weighted, q_mean, q_min], dim=-1)
