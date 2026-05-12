from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .cache_manager import CacheManager


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if manifest_path.suffix == ".jsonl":
        records = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if manifest_path.suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if "samples" in payload:
            return payload["samples"]
        raise ValueError(f"JSON manifest must be a list or contain samples: {manifest_path}")
    if manifest_path.suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


class SkeletonDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        cache_manager: CacheManager | None = None,
        allow_raw_fallback: bool = True,
        skipped_log_path: str | Path = "logs/skipped_samples.log",
        logger: logging.Logger | None = None,
    ) -> None:
        self.samples = read_manifest(manifest_path)
        self.cache_manager = cache_manager
        self.allow_raw_fallback = allow_raw_fallback
        self.skipped_log_path = Path(skipped_log_path)
        self.skipped_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        sample = self.samples[index]
        sample_id = str(sample.get("sample_id") or sample.get("id") or index)
        try:
            skeleton = self._load_skeleton(sample, sample_id)
            label = int(sample["label"])
            return {
                "skeleton": skeleton,
                "label": label,
                "sample_id": sample_id,
                "index": index,
            }
        except Exception as exc:
            self._record_skipped(sample_id, index, exc)
            return None

    def _load_skeleton(self, sample: dict[str, Any], sample_id: str) -> Any:
        if self.cache_manager is not None:
            cached = self.cache_manager.load_tensor(sample_id)
            if cached is not None:
                return cached
            self.logger.warning(
                "WARNING: cache missing for key %s, falling back to raw data",
                sample_id,
                extra={"sample_index": sample.get("index", "-")},
            )

        if not self.allow_raw_fallback:
            raise RuntimeError(f"Cache missing and raw fallback disabled for key {sample_id}")
        raw_path = sample.get("skeleton_path") or sample.get("path")
        if raw_path is None:
            raise KeyError(f"Missing skeleton_path for sample {sample_id}")
        return load_skeleton_file(raw_path)

    def _record_skipped(self, sample_id: str, index: int, exc: Exception) -> None:
        message = f"{sample_id}\tindex={index}\t{type(exc).__name__}: {exc}\n"
        with self.skipped_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
        self.logger.warning(
            "Skipped bad sample sample_id=%s error=%s",
            sample_id,
            exc,
            extra={"sample_index": index},
        )


def load_skeleton_file(path: str | Path) -> Any:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Skeleton file not found: {data_path}")
    if data_path.suffix in {".npy", ".npz"}:
        import numpy as np

        loaded = np.load(data_path, allow_pickle=False)
        if data_path.suffix == ".npz":
            if "skeleton" in loaded:
                return loaded["skeleton"]
            first_key = loaded.files[0]
            return loaded[first_key]
        return loaded
    if data_path.suffix in {".pt", ".pth"}:
        import torch

        return torch.load(data_path, map_location="cpu")
    raise ValueError(f"Unsupported skeleton file format: {data_path.suffix}")


def safe_collate(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    import torch

    skeletons = [torch.as_tensor(item["skeleton"]).float() for item in valid]
    labels = torch.tensor([int(item["label"]) for item in valid], dtype=torch.long)
    return {
        "skeleton": torch.stack(skeletons, dim=0),
        "label": labels,
        "sample_id": [item["sample_id"] for item in valid],
        "index": torch.tensor([int(item["index"]) for item in valid], dtype=torch.long),
    }
