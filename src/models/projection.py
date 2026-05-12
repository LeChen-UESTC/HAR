from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TokenProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        llm_dim: int,
        target_temporal_bins: int = 4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.llm_dim = llm_dim
        self.target_temporal_bins = target_temporal_bins
        self.proj = nn.Linear(in_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C_s, T', V]

        Returns:
            tokens: [B, target_temporal_bins * V, d_llm]
        """
        if feat.ndim != 4:
            raise ValueError(f"Expected feature shape [B,C,T,V], got {tuple(feat.shape)}")
        joints = feat.shape[-1]
        pooled = F.adaptive_avg_pool2d(feat, output_size=(self.target_temporal_bins, joints))
        tokens = pooled.permute(0, 2, 3, 1).contiguous().view(feat.shape[0], -1, feat.shape[1])
        return self.norm(self.proj(tokens))
