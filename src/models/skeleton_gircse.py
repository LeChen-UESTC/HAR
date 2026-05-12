from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SkeletonGIRCSE(nn.Module):
    def __init__(
        self,
        shift_gcn: nn.Module,
        token_projector: nn.Module,
        soft_token_generator: nn.Module,
        prompt_builder: Any,
    ) -> None:
        super().__init__()
        self.shift_gcn = shift_gcn
        self.token_projector = token_projector
        self.soft_token_generator = soft_token_generator
        self.prompt_builder = prompt_builder

    def forward(self, skeleton: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        feat = self.shift_gcn.forward_features(skeleton)
        skeleton_tokens = self.token_projector(feat)
        prompt_tokens = self.prompt_builder.build(
            batch_size=skeleton_tokens.shape[0],
            device=skeleton_tokens.device,
        )
        input_embeds = torch.cat([prompt_tokens, skeleton_tokens], dim=1)
        return self.soft_token_generator(input_embeds)

    def warmup_embedding(self, skeleton: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        tokens = self.token_projector(feat)
        z = tokens.mean(dim=1)
        if normalize:
            z = torch.nn.functional.normalize(z.float(), dim=-1)
        return z
