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
    descriptions: dict[str, dict[str, Any]],
    base_model_path: str,
    adapter_path: str | None,
    prompt_template: str,
    output_path: str | Path,
    variant: str,
    k_text: int = 20,
    normalize: bool = True,
    logit_temperature: float = 1.0,
    pooling_method: str = "generate_mean",
    main_label_alpha: float = 0.7,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from .description_templates import build_text_views, normalize_description_record
    from .generate_rich_description import validate_description_record

    logger = logging.getLogger(__name__)
    if variant != "structured":
        raise ValueError(f"text_branch.description_variant must be structured, got {variant!r}")
    records = {}
    for label in class_names:
        if label not in descriptions:
            raise KeyError(
                f"Description cache is missing class {label!r}. "
                "Regenerate paths.description_cache with scripts/generate_rich_description.py."
            )
        record = descriptions[label]
        if not isinstance(record, dict):
            raise TypeError(
                f"Description for class {label!r} must be a structured object, got {type(record).__name__}. "
                "Regenerate paths.description_cache with the structured schema."
            )
        validate_description_record(record, label)
        records[label] = normalize_description_record(record)
    text_views = {
        label: build_text_views(records[label])
        for label in class_names
    }
    prompts_by_bank = {
        bank_name: [
            _format_text_prompt(prompt_template, text_views[label][bank_name])
            for label in class_names
        ]
        for bank_name in ("label", "motion", "phase")
    }
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
    encoded = {}
    for bank_name, prompts in prompts_by_bank.items():
        logger.info("Encoding %s %s text prompts with k_text=%s", len(prompts), bank_name, k_text)
        encoded[bank_name] = encoder.encode(prompts)
    alpha = float(main_label_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"text_branch.embedding.main_label_alpha must be in [0, 1], got {alpha}")
    z_text = F.normalize(alpha * encoded["label"] + (1.0 - alpha) * encoded["motion"], dim=-1)

    payload = {
        "class_names": class_names,
        "descriptions": records,
        "text_views": text_views,
        "z_text": z_text,
        "z_text_main": z_text,
        "z_text_label": encoded["label"],
        "z_text_motion": encoded["motion"],
        "z_text_phase": encoded["phase"],
        "class_ids": torch.arange(len(class_names), dtype=torch.long),
        "metadata": {
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "prompt_template": prompt_template,
            "description_variant": variant,
            "text_mode": text_mode_suffix_from_variant(variant),
            "text_banks": ["main", "label", "motion", "phase"],
            "main_fusion": {
                "type": "label_motion_residual",
                "label_alpha": alpha,
                "motion_alpha": 1.0 - alpha,
            },
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


def _format_text_prompt(prompt_template: str, text: str) -> str:
    if "{text}" in prompt_template:
        return prompt_template.format(text=text)
    return prompt_template.format(rich_description=text)
