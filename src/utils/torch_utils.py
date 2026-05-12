from __future__ import annotations

from typing import Any


def resolve_torch_dtype(dtype_name: str | None, fallback_to_float32_on_cpu: bool = True) -> Any:
    import torch

    if dtype_name is None or dtype_name == "auto":
        if fallback_to_float32_on_cpu and not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16
    normalized = str(dtype_name).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    if fallback_to_float32_on_cpu and not torch.cuda.is_available():
        return torch.float32
    return mapping[normalized]


def hf_model_kwargs(config: dict[str, Any], for_text: bool = False) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    kwargs = {
        "torch_dtype": resolve_torch_dtype(
            runtime.get("torch_dtype", "bfloat16"),
            bool(runtime.get("fallback_to_float32_on_cpu", True)),
        ),
        "trust_remote_code": bool(runtime.get("trust_remote_code", True)),
    }
    attn_implementation = runtime.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    device_key = "device_map_text" if for_text else "device_map_train"
    device_map = runtime.get(device_key)
    if device_map is not None:
        kwargs["device_map"] = device_map
    return kwargs
