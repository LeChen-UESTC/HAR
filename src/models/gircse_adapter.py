from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class GIRCSEGenerationOutput:
    generated_hidden_states: torch.Tensor
    final_embedding: torch.Tensor


def gircse_soft_next_token_embedding(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    logit_temperature: float = 1.0,
) -> torch.Tensor:
    """Port of official GIRCSETrainer.get_next_token_embedding.

    Source: https://github.com/Roytsai27/GIRCSE/tree/main/embedding
    Original logic: token_weight = softmax(logits / temp); return token_weight @ embedding_weight.
    """
    token_weight = F.softmax(logits / logit_temperature, dim=-1)
    return token_weight.to(embedding_weight.dtype) @ embedding_weight


def apply_gircse_pooling(hidden_states: torch.Tensor, pooling_method: str = "generate_mean") -> torch.Tensor:
    """Port of official BaseReasoningTrainer.apply_pooling."""
    if pooling_method == "last":
        return hidden_states[:, -1, :]
    if pooling_method == "generate_mean":
        return hidden_states.mean(dim=1)
    raise ValueError(f"{pooling_method} pooling method not implemented")


def gircse_iterative_soft_generation(
    llm: Any,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    embedding_weight: torch.Tensor,
    max_new_tokens: int,
    logit_temperature: float = 1.0,
    pooling_method: str = "generate_mean",
    use_cache: bool | None = None,
) -> GIRCSEGenerationOutput:
    """GIRCSE iterative soft-token generation for pre-built input embeddings.

    This adapts the official `BaseReasoningTrainer.encode` /
    `_extend_sequence` loop to support skeleton prefix embeddings. It performs
    `max_new_tokens + 1` extensions, then drops the first collected hidden
    state exactly as the official implementation does.
    """
    if attention_mask is None:
        attention_mask = torch.ones(
            input_embeds.shape[:2],
            dtype=torch.long,
            device=input_embeds.device,
        )
    if use_cache is None:
        use_cache = not llm.training

    current_embeds = input_embeds
    current_mask = attention_mask
    past_key_values = None
    collected_hidden = []

    for _ in range(max_new_tokens + 1):
        model_inputs = {
            "inputs_embeds": (
                current_embeds if past_key_values is None else current_embeds[:, -1:, :]
            ),
            "attention_mask": current_mask,
            "output_hidden_states": True,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        outputs = llm(**model_inputs)
        logits = outputs.logits[:, -1, :]
        last_hidden = outputs.hidden_states[-1][:, -1:, :]
        next_token_embedding = gircse_soft_next_token_embedding(
            logits=logits,
            embedding_weight=embedding_weight,
            logit_temperature=logit_temperature,
        ).unsqueeze(1)

        current_embeds = torch.cat([current_embeds, next_token_embedding], dim=1)
        new_mask = torch.ones(
            (current_mask.size(0), 1),
            dtype=current_mask.dtype,
            device=current_mask.device,
        )
        current_mask = torch.cat([current_mask, new_mask], dim=-1)
        past_key_values = outputs.past_key_values if use_cache else None
        collected_hidden.append(last_hidden)

    generated_hidden_states = torch.cat(collected_hidden[1:], dim=1)
    final_embedding = apply_gircse_pooling(generated_hidden_states, pooling_method=pooling_method)
    return GIRCSEGenerationOutput(
        generated_hidden_states=generated_hidden_states,
        final_embedding=final_embedding,
    )
