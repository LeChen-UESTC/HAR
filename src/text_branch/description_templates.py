from __future__ import annotations

from typing import Any


DESCRIPTION_VARIANTS = {
    "label_only",
    "label_local_motion",
    "label_local_motion_object",
    "full",
}


def normalize_description_record(record: dict[str, Any] | str) -> dict[str, str]:
    if isinstance(record, str):
        return {
            "label": record,
            "local_motion": "unknown",
            "used_object": "none",
            "target_object": "none",
            "environment": "unknown",
        }
    return {
        "label": str(record.get("label", "")).strip(),
        "local_motion": str(record.get("local_motion", "unknown")).strip(),
        "used_object": str(record.get("used_object", "none")).strip(),
        "target_object": str(record.get("target_object", "none")).strip(),
        "environment": str(record.get("environment", "unknown")).strip(),
    }


def build_rich_description(record: dict[str, Any] | str, variant: str = "full") -> str:
    if variant not in DESCRIPTION_VARIANTS:
        raise ValueError(f"Unknown description variant={variant}; expected one of {sorted(DESCRIPTION_VARIANTS)}")
    item = normalize_description_record(record)
    lines = [f"Action: {item['label']}."]
    if variant in {"label_local_motion", "label_local_motion_object", "full"}:
        lines.append(f"Local Motion: {item['local_motion']}.")
    if variant in {"label_local_motion_object", "full"}:
        lines.append(f"Used Object: {item['used_object']}.")
    if variant == "full":
        lines.append(f"Target Object: {item['target_object']}.")
        lines.append(f"Environment: {item['environment']}.")
    return "\n".join(lines)


def generation_prompt(label: str) -> str:
    return (
        "Generate a structured JSON description for zero-shot skeleton action recognition. "
        "Use concise physical motion language and plausible context. Return only JSON with "
        "fields: label, local_motion, used_object, target_object, environment.\n"
        f"Action label: {label}"
    )
