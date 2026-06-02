from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.config_utils import text_mode_suffix_from_variant
from src.utils.torch_utils import resolve_torch_dtype


class TextEmbeddingEncoder:
    def __init__(
        self,
        model_path: str,
        normalize: bool = True,
        device_map: str | dict[str, Any] | None = "auto",
        pooling_method: str = "last",
        max_length: int = 512,
        padding_side: str = "left",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        fallback_to_float32_on_cpu: bool = True,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if pooling_method not in {"last", "mean"}:
            raise ValueError(f"text_branch.embedding.pooling must be last or mean, got {pooling_method!r}")
        if padding_side not in {"left", "right"}:
            raise ValueError(f"text_branch.embedding.padding_side must be left or right, got {padding_side!r}")
        self.normalize = normalize
        self.pooling_method = pooling_method
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = padding_side
        if isinstance(device_map, str) and device_map.lower() in {"", "null", "none"}:
            device_map = None
        model_kwargs = {
            "dtype": resolve_torch_dtype(torch_dtype, fallback_to_float32_on_cpu),
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        if device_map is None:
            import torch

            target = device
            if str(target).startswith("cuda") and not torch.cuda.is_available():
                target = "cpu"
            self.model.to(torch.device(target))
        self.model.eval()

    @property
    def device(self) -> Any:
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None:
                return embeddings.weight.device
        return next(self.model.parameters()).device

    def encode(self, prompts: list[str]) -> Any:
        import torch
        import torch.nn.functional as F

        batch = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch, return_dict=True)
            hidden = outputs.last_hidden_state
            attention_mask = batch["attention_mask"]
            if self.pooling_method == "last":
                z = _last_token_pool(hidden, attention_mask)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                z = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if self.normalize:
            z = F.normalize(z.float(), dim=-1)
        return z.cpu()


def encode_text_bank(
    class_names: list[str],
    descriptions: dict[str, dict[str, Any]],
    model_path: str,
    prompt_template: str,
    output_path: str | Path,
    variant: str,
    normalize: bool = True,
    pooling_method: str = "last",
    main_label_alpha: float = 0.7,
    max_length: int = 512,
    padding_side: str = "left",
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from .description_templates import build_text_views, normalize_description_record
    from .structured_descriptions import validate_description_record

    logger = logging.getLogger(__name__)
    if variant != "structured":
        raise ValueError(f"text_branch.description_variant must be structured, got {variant!r}")
    records = {}
    for label in class_names:
        if label not in descriptions:
            raise KeyError(
                f"Description cache is missing class {label!r}. "
                "Update paths.description_cache so it contains every configured class."
            )
        record = descriptions[label]
        if not isinstance(record, dict):
            raise TypeError(
                f"Description for class {label!r} must be a structured object, got {type(record).__name__}. "
                "Update paths.description_cache with the structured schema."
            )
        validate_description_record(record, label)
        records[label] = normalize_description_record(record)

    text_views = {label: build_text_views(records[label]) for label in class_names}
    prompts_by_bank = {
        bank_name: [
            _format_text_prompt(prompt_template, text_views[label][bank_name])
            for label in class_names
        ]
        for bank_name in ("label", "motion", "phase")
    }

    logger.info("Loading text embedding model: %s", model_path)
    encoder = TextEmbeddingEncoder(
        model_path=model_path,
        normalize=normalize,
        pooling_method=pooling_method,
        max_length=max_length,
        padding_side=padding_side,
        torch_dtype=str((runtime or {}).get("torch_dtype", "bfloat16")),
        device=str((runtime or {}).get("device", "cuda")),
        fallback_to_float32_on_cpu=bool((runtime or {}).get("fallback_to_float32_on_cpu", True)),
        attn_implementation=(runtime or {}).get("attn_implementation"),
        trust_remote_code=bool((runtime or {}).get("trust_remote_code", True)),
        device_map=(runtime or {}).get("device_map_text", "auto"),
    )
    encoded = {}
    for bank_name, prompts in prompts_by_bank.items():
        logger.info("Encoding %s %s text prompts", len(prompts), bank_name)
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
            "embedding_model_path": model_path,
            "prompt_template": prompt_template,
            "description_variant": variant,
            "text_mode": text_mode_suffix_from_variant(variant),
            "text_banks": ["main", "label", "motion", "phase"],
            "main_fusion": {
                "type": "label_motion_residual",
                "label_alpha": alpha,
                "motion_alpha": 1.0 - alpha,
            },
            "normalize": normalize,
            "pooling_method": pooling_method,
            "max_length": int(max_length),
            "padding_side": padding_side,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    logger.info("Saved text bank to %s", output_path)
    return payload


def _format_text_prompt(prompt_template: str, text: str) -> str:
    return prompt_template.format(text=text)


def _last_token_pool(hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, sequence_lengths]
