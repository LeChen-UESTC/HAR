from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SamplingStrategy:
    fps: int | float | None = None
    max_frames: int = 64
    frame_interval: int = 1
    resolution: str | None = None
    augmentation_version: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SamplingStrategy":
        known = {
            "fps",
            "max_frames",
            "frame_interval",
            "resolution",
            "augmentation_version",
        }
        extra = {k: v for k, v in config.items() if k not in known}
        return cls(
            fps=config.get("fps"),
            max_frames=int(config.get("max_frames", 64)),
            frame_interval=int(config.get("frame_interval", 1)),
            resolution=config.get("resolution"),
            augmentation_version=str(config.get("augmentation_version", "none")),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extra"] = dict(sorted(payload.get("extra", {}).items()))
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self, length: int = 12) -> str:
        return hashlib.sha1(self.to_json().encode("utf-8")).hexdigest()[:length]


def sample_indices(num_frames: int, strategy: SamplingStrategy) -> list[int]:
    if num_frames <= 0:
        return []
    interval = max(1, strategy.frame_interval)
    candidates = list(range(0, num_frames, interval))
    if strategy.fps and strategy.fps > 0:
        # Dataset FPS is often unavailable for skeleton files; explicit fps is
        # still included in the hash and metadata. This function uses interval
        # sampling unless caller provides decoded frame timing.
        candidates = candidates
    if len(candidates) <= strategy.max_frames:
        return candidates
    step = len(candidates) / float(strategy.max_frames)
    return [candidates[min(int(i * step), len(candidates) - 1)] for i in range(strategy.max_frames)]
