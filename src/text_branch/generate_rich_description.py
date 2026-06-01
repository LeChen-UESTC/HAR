from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .description_templates import (
    PHASE_KEYS,
    generation_prompt,
    heuristic_description as fallback_description,
    normalize_description_record,
)


REQUIRED_DESCRIPTION_FIELDS = (
    "label",
    "observable_motion",
    "key_body_parts",
    "temporal_phases",
)
ALLOWED_DESCRIPTION_FIELDS = set(REQUIRED_DESCRIPTION_FIELDS)
INVALID_FIELD_VALUES = {
    "",
    "unknown",
    "n/a",
    "na",
    "none",
    "none specified",
    "not specified",
    "not applicable",
    "null",
}


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
    decoder = json.JSONDecoder()
    parsed: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    if parsed:
        return parsed[-1]
    raise ValueError("No valid JSON object found in model output")


def generate_descriptions(
    class_names: list[str],
    model_path: str,
    output_path: str | Path,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    dry_run: bool = False,
    runtime: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> dict[str, dict[str, Any]]:
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)
    logger.info(
        "Generating structured descriptions: classes=%s model=%s output=%s",
        len(class_names),
        model_path,
        output_path,
    )
    output: dict[str, dict[str, Any]] = {}

    if dry_run:
        for label in _progress(class_names, desc="Generating descriptions"):
            output[label] = normalize_description_record(label)
            save_descriptions(output, output_path)
        save_descriptions(output, output_path)
        logger.info("Saved structured descriptions to %s", output_path)
        return output

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils.torch_utils import resolve_torch_dtype

    runtime = runtime or {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(runtime.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    failure_path = output_path.with_suffix(".failures.jsonl")
    for label in _progress(class_names, desc="Generating descriptions"):
        output[label] = generate_one_description(
            label=label,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
            failure_path=failure_path,
            logger=logger,
        )
        save_descriptions(output, output_path)

    save_descriptions(output, output_path)
    logger.info("Saved structured descriptions to %s", output_path)
    return output


def generate_one_description(
    label: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    failure_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    import torch

    feedback = None
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        prompt = generation_prompt(label, feedback=feedback)
        inputs = tokenize_generation_prompt(tokenizer, prompt, model)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        try:
            record = parse_json_object(decoded)
            validate_description_record(record, label)
            return normalize_description_record(record)
        except Exception as exc:
            feedback = str(exc)
            append_generation_failure(
                failure_path=failure_path,
                label=label,
                attempt=attempt,
                error=feedback,
                raw_output=decoded,
            )
            logger.warning(
                "Description generation failed label=%s attempt=%s/%s error=%s",
                label,
                attempt,
                attempts,
                feedback,
            )

    logger.warning(
        "Using heuristic fallback description after %s failed attempts for label=%s",
        attempts,
        label,
    )
    return heuristic_description(label)


def tokenize_generation_prompt(tokenizer: Any, prompt: str, model: Any) -> dict[str, Any]:
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return {"input_ids": input_ids.to(model.device)}
    return tokenizer(prompt, return_tensors="pt").to(model.device)


def validate_description_record(record: dict[str, Any], label: str) -> None:
    missing = [field for field in REQUIRED_DESCRIPTION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    extra = sorted(set(record) - ALLOWED_DESCRIPTION_FIELDS)
    if extra:
        raise ValueError(f"Unsupported fields in structured description: {extra}")
    for field in ("label", "observable_motion"):
        value = str(record.get(field, "")).strip()
        normalized = value.lower()
        if normalized in INVALID_FIELD_VALUES:
            raise ValueError(f"Invalid placeholder value for {field}: {value!r}")
    if record["label"].strip().lower() != label.strip().lower():
        raise ValueError(f"Label mismatch: expected {label!r}, got {record['label']!r}")
    if len(str(record["observable_motion"]).split()) < 6:
        raise ValueError("observable_motion is too short to describe skeleton-visible motion")
    key_body_parts = record.get("key_body_parts")
    if not isinstance(key_body_parts, list) or not key_body_parts:
        raise ValueError("key_body_parts must be a non-empty list")
    for index, part in enumerate(key_body_parts):
        value = str(part).strip()
        if value.lower() in INVALID_FIELD_VALUES:
            raise ValueError(f"Invalid key_body_parts[{index}]: {part!r}")
    temporal_phases = record.get("temporal_phases")
    if not isinstance(temporal_phases, dict):
        raise ValueError("temporal_phases must be an object")
    phase_keys = set(temporal_phases)
    expected_keys = set(PHASE_KEYS)
    if phase_keys != expected_keys:
        raise ValueError(
            f"temporal_phases keys must be exactly {sorted(expected_keys)}, got {sorted(phase_keys)}"
        )
    for key in PHASE_KEYS:
        value = str(temporal_phases.get(key, "")).strip()
        if value.lower() in INVALID_FIELD_VALUES:
            raise ValueError(f"Invalid temporal_phases.{key}: {value!r}")


def append_generation_failure(
    failure_path: Path,
    label: str,
    attempt: int,
    error: str,
    raw_output: str,
) -> None:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "attempt": attempt,
        "error": error,
        "raw_output": raw_output,
    }
    with failure_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def heuristic_description(label: str) -> dict[str, Any]:
    return fallback_description(label)


def save_descriptions(descriptions: dict[str, dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(descriptions, handle, indent=2, ensure_ascii=True, sort_keys=True)


def _progress(items: list[str], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc, unit="class")
    except Exception:
        return items
