from __future__ import annotations

import torch
from torch import nn

from .generative_pooling import cumulative_mean
from .gircse_adapter import gircse_iterative_soft_generation


class SoftTokenGenerator(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        token_embedding_table: torch.Tensor | nn.Parameter,
        K: int,
        normalize: bool = True,
        logit_temperature: float = 1.0,
        pooling_method: str = "generate_mean",
    ) -> None:
        super().__init__()
        self.llm = llm
        self.token_embedding_table = token_embedding_table
        self.K = K
        self.normalize = normalize
        self.logit_temperature = logit_temperature
        self.pooling_method = pooling_method

    def forward(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_embeds: [B, N_prompt + N_skel, d_llm]

        Returns:
            z_steps: z_steps[k-1] = Z_skel^(k), each [B, d_llm]
            z_final: [B, d_llm]
        """
        output = gircse_iterative_soft_generation(
            llm=self.llm,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            embedding_weight=self._embedding_weight(),
            max_new_tokens=self.K,
            logit_temperature=self.logit_temperature,
            pooling_method=self.pooling_method,
            use_cache=not self.training,
        )
        states = output.generated_hidden_states
        if self.pooling_method == "last":
            z_steps = [states[:, idx, :] for idx in range(states.shape[1])]
        else:
            z_steps = cumulative_mean(states)
        if self.normalize:
            z_steps = [torch.nn.functional.normalize(z.float(), dim=-1) for z in z_steps]
        z_final = z_steps[-1]
        return z_steps, z_final

    def _embedding_weight(self) -> torch.Tensor:
        weight: Any = self.token_embedding_table
        if isinstance(weight, nn.Embedding):
            return weight.weight
        return weight
