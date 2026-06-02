from __future__ import annotations
from typing import Any


DESCRIPTION_VARIANTS = {"structured"}

PHASE_KEYS = ("start", "middle", "end")

def normalize_description_record(record: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(record, str):
        return heuristic_description(record)

    label = str(record.get("label", "")).strip()
    observable_motion = str(record.get("observable_motion", "")).strip()
    key_body_parts = record.get("key_body_parts", [])
    if isinstance(key_body_parts, str):
        key_body_parts = [item.strip() for item in key_body_parts.split(",") if item.strip()]
    key_body_parts = [str(item).strip() for item in key_body_parts if str(item).strip()]

    raw_phases = record.get("temporal_phases", {})
    if not isinstance(raw_phases, dict):
        raw_phases = {}
    temporal_phases = {
        key: str(raw_phases.get(key, "")).strip()
        for key in PHASE_KEYS
    }

    if not label or not observable_motion or not key_body_parts or any(not temporal_phases[key] for key in PHASE_KEYS):
        fallback = heuristic_description(label or str(record.get("label", "action")).strip() or "action")
        return {
            "label": label or fallback["label"],
            "observable_motion": observable_motion or fallback["observable_motion"],
            "key_body_parts": key_body_parts or fallback["key_body_parts"],
            "temporal_phases": {
                key: temporal_phases[key] or fallback["temporal_phases"][key]
                for key in PHASE_KEYS
            },
        }

    return {
        "label": label,
        "observable_motion": observable_motion,
        "key_body_parts": key_body_parts,
        "temporal_phases": temporal_phases,
    }


def build_text_views(record: dict[str, Any] | str) -> dict[str, str]:
    item = normalize_description_record(record)
    label = item["label"]
    parts = ", ".join(item["key_body_parts"])
    phases = item["temporal_phases"]
    return {
        "label": f"Action: {label}.",
        "motion": (
            f"Action: {label}.\n"
            f"Observable skeleton motion: {item['observable_motion']}.\n"
            f"Key body parts: {parts}."
        ),
        "phase": (
            f"Action: {label}.\n"
            "Temporal pattern: "
            f"start: {phases['start']}; "
            f"middle: {phases['middle']}; "
            f"end: {phases['end']}."
        ),
    }


def build_rich_description(record: dict[str, Any] | str, variant: str = "structured") -> str:
    if variant not in DESCRIPTION_VARIANTS:
        raise ValueError(f"Unknown description variant={variant}; expected structured")
    views = build_text_views(record)
    return "\n".join([views["label"], views["motion"], views["phase"]])


def heuristic_description(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "observable_motion": (
            f"The person performs {label} with visible body posture changes "
            "and coordinated limb movements over time"
        ),
        "key_body_parts": ["body", "arms", "legs", "torso"],
        "temporal_phases": {
            "start": f"The body initiates the motion pattern for {label}",
            "middle": f"The main coordinated limb movement of {label} is performed",
            "end": f"The motion for {label} slows down or returns toward a neutral posture",
        },
    }
