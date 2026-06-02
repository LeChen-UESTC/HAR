from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .description_templates import PHASE_KEYS


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
                class_names = _class_names_from_mapping(payload)
        else:
            raise ValueError(f"Unsupported class-name JSON payload: {class_path}")
    else:
        with class_path.open("r", encoding="utf-8") as handle:
            class_names = [line.strip() for line in handle if line.strip()]

    if max_classes is not None:
        return class_names[: int(max_classes)]
    return class_names


def normalize_description_cache(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, list):
        records: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Description cache list item {index} must be an object, got {type(item).__name__}"
                )
            label = str(item.get("label", "")).strip()
            if not label:
                raise ValueError(f"Description cache list item {index} is missing a non-empty label")
            if label in records:
                raise ValueError(f"Duplicate description label in cache: {label!r}")
            records[label] = item
        return records

    if isinstance(payload, dict):
        if isinstance(payload.get("descriptions"), list):
            return normalize_description_cache(payload["descriptions"])
        records = {}
        for label, item in payload.items():
            if not isinstance(item, dict):
                raise TypeError(
                    f"Description cache entry {label!r} must be an object, got {type(item).__name__}"
                )
            record_label = str(item.get("label", label)).strip()
            if not record_label:
                raise ValueError(f"Description cache entry {label!r} is missing a non-empty label")
            if record_label in records:
                raise ValueError(f"Duplicate description label in cache: {record_label!r}")
            records[record_label] = item
        return records

    raise TypeError(
        "Description cache must be either a list of structured records or a dict keyed by label, "
        f"got {type(payload).__name__}"
    )


def validate_description_record(record: dict[str, Any], label: str) -> None:
    missing = [field for field in REQUIRED_DESCRIPTION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    extra = sorted(set(record) - ALLOWED_DESCRIPTION_FIELDS)
    if extra:
        raise ValueError(f"Unsupported fields in structured description: {extra}")
    for field in ("label", "observable_motion"):
        value = str(record.get(field, "")).strip()
        if value.lower() in INVALID_FIELD_VALUES:
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


def _class_names_from_mapping(payload: dict[str, Any]) -> list[str]:
    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        raw_key = str(item[0])
        digits = "".join(ch for ch in raw_key if ch.isdigit())
        if digits:
            return int(digits), raw_key
        return 10**9, raw_key

    class_names = []
    for _, value in sorted(payload.items(), key=sort_key):
        if isinstance(value, dict):
            class_names.append(str(value.get("name", value.get("label", value))))
        else:
            class_names.append(str(value))
    return class_names
