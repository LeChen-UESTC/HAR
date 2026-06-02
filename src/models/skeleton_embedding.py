from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .text_space_projection import TextSpaceProjection


class SkeletonEmbeddingModel(nn.Module):
    def __init__(
        self,
        shift_gcn: nn.Module,
        token_projector: nn.Module,
        embedding_model: nn.Module,
        prompt_builder: Any,
    ) -> None:
        super().__init__()
        self.shift_gcn = shift_gcn
        self.token_projector = token_projector
        self.embedding_model = embedding_model
        self.prompt_builder = prompt_builder
        self.embedding_projection = TextSpaceProjection(int(self.embedding_model.config.hidden_size))

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        skeleton_tokens = self.token_projector(feat)
        input_device = self.input_device
        input_dtype = self.input_dtype
        skeleton_tokens = skeleton_tokens.to(device=input_device, dtype=input_dtype)
        prompt_tokens = self.prompt_builder.build(
            batch_size=skeleton_tokens.shape[0],
            device=input_device,
        )
        input_embeds = torch.cat([skeleton_tokens, prompt_tokens], dim=1)
        attention_mask = torch.ones(
            input_embeds.shape[:2],
            dtype=torch.long,
            device=input_device,
        )
        outputs = self.embedding_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        z = outputs.last_hidden_state[:, -1, :]
        return self.embedding_projection(z)

    def set_text_embedding_dim(self, target_dim: int) -> None:
        self.embedding_projection.configure(target_dim)

    @property
    def input_device(self) -> torch.device:
        return self.embedding_model.get_input_embeddings().weight.device

    @property
    def input_dtype(self) -> torch.dtype:
        return self.embedding_model.get_input_embeddings().weight.dtype


class DirectQFormerEmbedding(nn.Module):
    def __init__(self, shift_gcn: nn.Module, token_projector: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.shift_gcn = shift_gcn
        self.token_projector = token_projector
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.embedding_projection = TextSpaceProjection(hidden_dim)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        tokens = self.token_projector(feat)
        z = self.embedding_head(tokens.mean(dim=1))
        return self.embedding_projection(z)

    def set_text_embedding_dim(self, target_dim: int) -> None:
        self.embedding_projection.configure(target_dim)
