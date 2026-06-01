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
        skeleton_tokens = self.project_skeleton(skeleton)
        prompt_tokens = self.prompt_builder.build(
            batch_size=skeleton_tokens.shape[0],
            device=skeleton_tokens.device,
        )
        input_embeds = torch.cat([skeleton_tokens, prompt_tokens], dim=1)
        return self.soft_token_generator(input_embeds)

    def project_skeleton(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        skeleton_tokens = self.token_projector(feat)
        llm_input_device = getattr(self.soft_token_generator, "input_device", skeleton_tokens.device)
        llm_input_dtype = getattr(self.soft_token_generator, "input_dtype", skeleton_tokens.dtype)
        return skeleton_tokens.to(device=llm_input_device, dtype=llm_input_dtype)

    def warmup_embedding(self, skeleton: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        tokens = self.token_projector(feat)
        z = tokens.mean(dim=1)
        if normalize:
            z = torch.nn.functional.normalize(z.float(), dim=-1)
        return z


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

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        tokens = self.token_projector(feat)
        z = self.embedding_head(tokens.mean(dim=1))
        return torch.nn.functional.normalize(z.float(), dim=-1)


class AnchorHiddenStateEmbedding(nn.Module):
    def __init__(
        self,
        shift_gcn: nn.Module,
        token_projector: nn.Module,
        llm: nn.Module,
        prompt_builder: Any,
    ) -> None:
        super().__init__()
        self.shift_gcn = shift_gcn
        self.token_projector = token_projector
        self.llm = llm
        self.prompt_builder = prompt_builder

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
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        z = outputs.hidden_states[-1][:, -1, :]
        return torch.nn.functional.normalize(z.float(), dim=-1)

    @property
    def input_device(self) -> torch.device:
        return self.llm.get_input_embeddings().weight.device

    @property
    def input_dtype(self) -> torch.dtype:
        return self.llm.get_input_embeddings().weight.dtype
