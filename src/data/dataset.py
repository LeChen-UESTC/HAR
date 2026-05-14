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


class NpzSkeletonDataset:
    def __init__(
        self,
        npz_path: str | Path,
        x_key: str = "x_data",
        y_key: str = "y_data",
        channels: int = 3,
        joints: int = 25,
        persons: int = 2,
        selected_classes: list[int] | None = None,
        skipped_log_path: str | Path = "logs/skipped_samples.log",
        logger: logging.Logger | None = None,
    ) -> None:
        import numpy as np

        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ dataset file not found: {self.npz_path}")
        self.payload = np.load(self.npz_path, allow_pickle=False)
        if x_key not in self.payload or y_key not in self.payload:
            raise KeyError(
                f"NPZ file {self.npz_path} must contain keys {x_key!r} and {y_key!r}; "
                f"found {self.payload.files}"
            )
        self.x_data = self.payload[x_key]
        self.y_data = self.payload[y_key]
        self.channels = int(channels)
        self.joints = int(joints)
        self.persons = int(persons)
        self.skipped_log_path = Path(skipped_log_path)
        self.skipped_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

        labels = self._labels_array()
        if selected_classes:
            selected = np.asarray(selected_classes, dtype=labels.dtype)
            self.indices = np.flatnonzero(np.isin(labels, selected))
        else:
            self.indices = np.arange(labels.shape[0])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        real_index = int(self.indices[index])
        sample_id = f"{self.npz_path.stem}:{real_index}"
        try:
            skeleton = self._convert_skeleton(self.x_data[real_index])
            label = int(self._label_at(real_index))
            return {
                "skeleton": skeleton,
                "label": label,
                "sample_id": sample_id,
                "index": index,
            }
        except Exception as exc:
            self._record_skipped(sample_id, index, exc)
            return None

    def _labels_array(self):
        import numpy as np

        labels = np.asarray(self.y_data)
        if labels.ndim == 2:
            labels = labels.argmax(axis=1)
        return labels.astype(np.int64)

    def _label_at(self, real_index: int) -> int:
        import numpy as np

        label = np.asarray(self.y_data[real_index])
        if label.ndim > 0 and label.size > 1:
            return int(label.argmax())
        return int(label)

    def _convert_skeleton(self, sample: Any) -> Any:
        import numpy as np

        array = np.asarray(sample, dtype=np.float32)
        if array.ndim == 2:
            expected_dim = self.persons * self.joints * self.channels
            if array.shape[1] != expected_dim:
                raise ValueError(
                    f"Expected flattened skeleton dim {expected_dim}, got {array.shape[1]}"
                )
            # Preprocessed NTU npz layout: [T, M * V * C].
            return array.reshape(
                array.shape[0],
                self.persons,
                self.joints,
                self.channels,
            ).transpose(3, 0, 2, 1)
        if array.ndim == 4:
            # Already in Shift-GCN layout: [C, T, V, M].
            if (
                array.shape[0] == self.channels
                and array.shape[2] == self.joints
                and array.shape[3] == self.persons
            ):
                return array
            # Common preprocessed layout: [T, V, M, C].
            if array.shape[-1] == self.channels:
                return array.transpose(3, 0, 1, 2)
        raise ValueError(f"Unsupported skeleton sample shape from npz: {array.shape}")

    def _record_skipped(self, sample_id: str, index: int, exc: Exception) -> None:
        message = f"{sample_id}\tindex={index}\t{type(exc).__name__}: {exc}\n"
        with self.skipped_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
        self.logger.warning(
            "Skipped bad npz sample sample_id=%s error=%s",
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
