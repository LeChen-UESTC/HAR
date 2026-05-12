from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SkeletonPromptBuilder:
    tokenizer: Any
    token_embedding: Any
    prompt_text: str

    def build(self, batch_size: int, device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(self.prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        embeds = self.token_embedding(input_ids)
        return embeds.expand(batch_size, -1, -1).contiguous()


class StaticPromptEmbeddings:
    def __init__(self, prompt_embeddings: torch.Tensor) -> None:
        if prompt_embeddings.ndim != 2:
            raise ValueError("prompt_embeddings must have shape [N_prompt, d_llm]")
        self.prompt_embeddings = prompt_embeddings

    def build(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.prompt_embeddings.to(device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
