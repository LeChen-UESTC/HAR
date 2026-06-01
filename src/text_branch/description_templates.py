from __future__ import annotations

import json
from typing import Any


DESCRIPTION_VARIANTS = {"structured"}

PHASE_KEYS = ("start", "middle", "end")

ICL_DESCRIPTION_EXAMPLES = [
    {
        "label": "typing on a keyboard",
        "observable_motion": (
            "Both hands stay in front of the torso and perform repeated small "
            "finger and wrist movements near a fixed surface."
        ),
        "key_body_parts": ["hands", "wrists", "arms", "torso"],
        "temporal_phases": {
            "start": "The hands move toward the front body area.",
            "middle": "The fingers and wrists perform repeated small movements.",
            "end": "The hand motion slows down or returns to a relaxed posture.",
        },
    },
    {
        "label": "writing",
        "observable_motion": (
            "One hand performs small repeated strokes in front of the upper body, "
            "while the torso may lean slightly forward."
        ),
        "key_body_parts": ["hand", "arm", "torso"],
        "temporal_phases": {
            "start": "The hand moves toward the front body area.",
            "middle": "The hand performs repeated small writing-like strokes.",
            "end": "The hand movement slows down or stops.",
        },
    },
    {
        "label": "put on glasses",
        "observable_motion": (
            "Both hands rise toward the face and adjust near the eyes or ears."
        ),
        "key_body_parts": ["hands", "arms", "head"],
        "temporal_phases": {
            "start": "The hands move upward toward the face.",
            "middle": "The hands align and adjust near the eyes or ears.",
            "end": "The hands move away from the face or settle.",
        },
    },
]


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


def generation_prompt(label: str, feedback: str | None = None) -> str:
    examples = "\n\n".join(
        "Action label: {label}\nOutput:\n{payload}".format(
            label=item["label"],
            payload=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
        )
        for item in ICL_DESCRIPTION_EXAMPLES
    )
    retry_feedback = ""
    if feedback:
        retry_feedback = (
            "\nThe previous output was invalid for this reason: "
            f"{feedback}\nRegenerate a corrected JSON object."
        )
    return f"""You are generating class-level skeleton-observable descriptions for zero-shot
skeleton action recognition.

Given one action label, output exactly one valid JSON object.

The JSON object must contain exactly these fields:
- label
- observable_motion
- key_body_parts
- temporal_phases

The temporal_phases object must contain exactly:
- start
- middle
- end

Rules:
- Return JSON only. Do not output markdown, code fences, bullet points, or explanations.
- observable_motion must describe visible body, joint, limb, posture, or temporal motion cues.
- key_body_parts must be a non-empty array of skeleton-observable body parts.
- temporal_phases must describe start, middle, and end motion patterns.
- Do not mention objects, scene context, environment, intent, or invisible information.
- Do not use "unknown", "N/A", "not specified", or empty strings.
- Keep every field concise and action-specific.

Examples:

{examples}
{retry_feedback}

Now generate for:
Action label: {label}
Output:
"""


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
