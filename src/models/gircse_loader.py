from __future__ import annotations

from typing import Any


def load_gircse_model_and_tokenizer(
    base_model_path: str,
    adapter_path: str | None,
    model_kwargs: dict[str, Any],
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load local Qwen base model plus optional local GIRCSE LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        add_eos_token=True,
        padding_side="left",
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = dict(model_kwargs)
    kwargs["local_files_only"] = True
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)

    if not adapter_path:
        return base_model, tokenizer

    from peft import PeftModel

    peft_kwargs: dict[str, Any] = {"local_files_only": True}
    if "device_map" in kwargs:
        peft_kwargs["device_map"] = kwargs["device_map"]
    model = PeftModel.from_pretrained(base_model, adapter_path, **peft_kwargs)
    return model, tokenizer
