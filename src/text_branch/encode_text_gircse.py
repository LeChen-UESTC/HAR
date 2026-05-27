from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.models.gircse_adapter import gircse_iterative_soft_generation
from src.models.gircse_loader import load_gircse_model_and_tokenizer
from src.utils.config_utils import text_mode_suffix_from_variant
from src.utils.torch_utils import resolve_torch_dtype


class TextGIRCSEEncoder:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str | None = None,
        k_text: int = 20,
        normalize: bool = True,
        device_map: str | dict[str, Any] | None = "auto",
        logit_temperature: float = 1.0,
        pooling_method: str = "generate_mean",
        torch_dtype: str = "bfloat16",
        fallback_to_float32_on_cpu: bool = True,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        import torch

        if int(k_text) < 1:
            raise ValueError(f"k_text must be >= 1, got {k_text}")
        self.torch = torch
        model_kwargs = {
            "torch_dtype": resolve_torch_dtype(torch_dtype, fallback_to_float32_on_cpu),
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model, self.tokenizer = load_gircse_model_and_tokenizer(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.k_text = int(k_text)
        self.normalize = normalize
        self.logit_temperature = logit_temperature
        self.pooling_method = pooling_method

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def encode(self, prompts: list[str]) -> Any:
        torch = self.torch
        tokenizer = self.tokenizer
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.device)
        with torch.no_grad():
            input_embeds = self.model.get_input_embeddings()(batch["input_ids"])
            z = self._soft_token_embedding(input_embeds, batch.get("attention_mask"))
        if self.normalize:
            z = torch.nn.functional.normalize(z.float(), dim=-1)
        return z.cpu()

    def _soft_token_embedding(self, input_embeds: Any, attention_mask: Any | None = None) -> Any:
        output = gircse_iterative_soft_generation(
            llm=self.model,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            embedding_weight=self.model.get_input_embeddings().weight,
            max_new_tokens=self.k_text,
            logit_temperature=self.logit_temperature,
            pooling_method=self.pooling_method,
            use_cache=True,
        )
        return output.final_embedding


def encode_text_bank(
    class_names: list[str],
    descriptions: dict[str, dict[str, str]],
    base_model_path: str,
    adapter_path: str | None,
    prompt_template: str,
    output_path: str | Path,
    variant: str,
    k_text: int = 20,
    normalize: bool = True,
    logit_temperature: float = 1.0,
    pooling_method: str = "generate_mean",
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from .description_templates import build_rich_description

    logger = logging.getLogger(__name__)
    rich = {
        label: build_rich_description(descriptions.get(label, label), variant=variant)
        for label in class_names
    }
    prompts = [prompt_template.format(rich_description=rich[label]) for label in class_names]
    logger.info("Loading GIRCSE text encoder: base=%s adapter=%s", base_model_path, adapter_path)
    encoder = TextGIRCSEEncoder(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        k_text=k_text,
        normalize=normalize,
        logit_temperature=logit_temperature,
        pooling_method=pooling_method,
        torch_dtype=str((runtime or {}).get("torch_dtype", "bfloat16")),
        fallback_to_float32_on_cpu=bool((runtime or {}).get("fallback_to_float32_on_cpu", True)),
        attn_implementation=(runtime or {}).get("attn_implementation"),
        trust_remote_code=bool((runtime or {}).get("trust_remote_code", True)),
        device_map=(runtime or {}).get("device_map_text", "auto"),
    )
    logger.info("Encoding %s text prompts with k_text=%s", len(prompts), k_text)
    z_text = encoder.encode(prompts)

    payload = {
        "class_names": class_names,
        "rich_descriptions": rich,
        "z_text": z_text,
        "metadata": {
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "prompt_template": prompt_template,
            "description_variant": variant,
            "text_mode": text_mode_suffix_from_variant(variant),
            "k_text": k_text,
            "normalize": normalize,
            "logit_temperature": logit_temperature,
            "pooling_method": pooling_method,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    logger.info("Saved text bank to %s", output_path)
    return payload
