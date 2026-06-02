from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TextSpaceProjection(nn.Module):
    def __init__(self, source_dim: int) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.target_dim = int(source_dim)
        self.proj: nn.Module = nn.Identity()

    def configure(self, target_dim: int) -> None:
        target_dim = int(target_dim)
        if target_dim < 1:
            raise ValueError(f"text embedding dim must be >= 1, got {target_dim}")
        if target_dim == self.target_dim:
            return
        if target_dim == self.source_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.source_dim, target_dim),
                nn.LayerNorm(target_dim),
            )
        self.target_dim = target_dim

    def forward(self, z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        first_param = next(self.proj.parameters(), None)
        if first_param is not None and first_param.device != z.device:
            self.proj.to(z.device)
        z = self.proj(z.float())
        if normalize:
            z = F.normalize(z, dim=-1)
        return z
