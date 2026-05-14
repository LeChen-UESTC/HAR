from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .description_templates import generation_prompt, normalize_description_record


def load_class_names(path: str | Path, max_classes: int | None = None) -> list[str]:
    class_path = Path(path)
    class_names: list[str]
    if class_path.suffix.lower() == ".json":
        with class_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            class_names = [
                str(item.get("name", item.get("label", item)))
                if isinstance(item, dict)
                else str(item)
                for item in payload
            ]
        elif isinstance(payload, dict):
            for key in ("class_names", "classes", "action_names", "actions"):
                if isinstance(payload.get(key), list):
                    values = payload[key]
                    class_names = [
                        str(item.get("name", item.get("label", item)))
                        if isinstance(item, dict)
                        else str(item)
                        for item in values
                    ]
                    break
            else:
                def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
                    raw_key = str(item[0])
                    digits = "".join(ch for ch in raw_key if ch.isdigit())
                    if digits:
                        return int(digits), raw_key
                    return 10**9, raw_key

                class_names = []
                for _, value in sorted(payload.items(), key=sort_key):
                    if isinstance(value, dict):
                        class_names.append(
                            str(value.get("name", value.get("label", value)))
                        )
                    else:
                        class_names.append(str(value))
        else:
            raise ValueError(f"Unsupported class-name JSON payload: {class_path}")
    else:
        with class_path.open("r", encoding="utf-8") as handle:
            class_names = [line.strip() for line in handle if line.strip()]

    if max_classes is not None:
        return class_names[: int(max_classes)]
    return class_names


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def generate_descriptions(
    class_names: list[str],
    model_path: str,
    output_path: str | Path,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    dry_run: bool = False,
    runtime: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}

    if dry_run:
        for label in class_names:
            output[label] = normalize_description_record(label)
        save_descriptions(output, output_path)
        return output

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils.torch_utils import resolve_torch_dtype

    runtime = runtime or {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(runtime.get("trust_remote_code", True)),
    )
    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(
            runtime.get("torch_dtype", "bfloat16"),
            bool(runtime.get("fallback_to_float32_on_cpu", True)),
        ),
        "device_map": runtime.get("device_map_text", "auto"),
        "trust_remote_code": bool(runtime.get("trust_remote_code", True)),
    }
    if runtime.get("attn_implementation"):
        model_kwargs["attn_implementation"] = runtime["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    for label in class_names:
        prompt = generation_prompt(label)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        decoded = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        try:
            record = parse_json_object(decoded)
        except Exception:
            record = {"label": label}
        record["label"] = label
        output[label] = normalize_description_record(record)

    save_descriptions(output, output_path)
    return output


def save_descriptions(descriptions: dict[str, dict[str, str]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(descriptions, handle, indent=2, ensure_ascii=True, sort_keys=True)
