from __future__ import annotations

import json
from typing import Any


DESCRIPTION_VARIANTS = {
    "label_only",
    "label_local_motion",
    "label_local_motion_object",
    "full",
}


ICL_DESCRIPTION_EXAMPLES = [
    {
        "label": "typing on a keyboard",
        "local_motion": (
            "The person keeps both hands in front of the body and performs "
            "repeated small finger and wrist movements near a flat surface."
        ),
        "used_object": "keyboard",
        "target_object": "tablet or computer",
        "environment": "room or office",
    },
    {
        "label": "writing",
        "local_motion": (
            "The person moves one hand repeatedly with small strokes near a "
            "surface while the upper body may lean slightly forward."
        ),
        "used_object": "pen",
        "target_object": "paper",
        "environment": "office",
    },
    {
        "label": "put on glasses",
        "local_motion": (
            "The person raises both hands toward the face and adjusts an object "
            "around the eyes or ears."
        ),
        "used_object": "glasses",
        "target_object": "face or eyes",
        "environment": "home",
    },
    {
        "label": "put on hat",
        "local_motion": (
            "The person raises one or both hands above the head and places an "
            "object onto the head."
        ),
        "used_object": "hat",
        "target_object": "head",
        "environment": "home",
    },
]


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
    return f"""You are generating class-level semantic descriptions for zero-shot skeleton action recognition.

Given one action label, output exactly one valid JSON object.

The JSON object must contain exactly these fields:
- label
- local_motion
- used_object
- target_object
- environment

Rules:
- Return JSON only. Do not output markdown, code fences, bullet points, or explanations.
- local_motion must describe visible body, joint, limb, posture, or temporal motion cues.
- used_object should be a concrete object if one is commonly involved; otherwise use "none".
- target_object should be the object or body part being acted on; otherwise use "none".
- environment should be a short plausible scene.
- Do not use "unknown", "N/A", "not specified", or empty strings.
- Keep every field concise and action-specific.

Examples:

{examples}
{retry_feedback}

Now generate for:
Action label: {label}
Output:
"""
