from __future__ import annotations

import json
import logging
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .samplers import SamplingStrategy


@dataclass
class CacheStatus:
    exists: bool
    valid: bool
    cache_path: Path
    metadata_path: Path
    reason: str


class CacheManager:
    def __init__(
        self,
        dataset_name: str,
        cache_root: str | Path,
        sampling_strategy: SamplingStrategy,
        preprocess_version: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.cache_root = Path(cache_root)
        self.sampling_strategy = sampling_strategy
        self.preprocess_version = preprocess_version
        self.logger = logger or logging.getLogger(__name__)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_root / self.cache_name
        self.metadata_path = self.cache_path / "cache_metadata.json"

    @property
    def cache_name(self) -> str:
        return (
            f"{self.dataset_name}_{self.sampling_strategy.hash()}_"
            f"{self.preprocess_version}.lmdb"
        )

    def expected_metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "sampling_strategy": self.sampling_strategy.to_dict(),
            "sampling_strategy_hash": self.sampling_strategy.hash(),
            "preprocess_version": self.preprocess_version,
            "cache_name": self.cache_name,
        }

    def read_metadata(self) -> dict[str, Any] | None:
        if not self.metadata_path.exists():
            return None
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_metadata(self, extra: dict[str, Any] | None = None) -> None:
        self.cache_path.mkdir(parents=True, exist_ok=True)
        payload = self.expected_metadata()
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        if extra:
            payload.update(extra)
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def validate(self) -> CacheStatus:
        if not self.cache_path.exists():
            return CacheStatus(
                exists=False,
                valid=False,
                cache_path=self.cache_path,
                metadata_path=self.metadata_path,
                reason="cache path missing",
            )
        metadata = self.read_metadata()
        if metadata is None:
            return CacheStatus(
                exists=True,
                valid=False,
                cache_path=self.cache_path,
                metadata_path=self.metadata_path,
                reason="cache metadata missing",
            )

        expected = self.expected_metadata()
        mismatches = []
        for key in ["dataset_name", "sampling_strategy", "sampling_strategy_hash", "preprocess_version"]:
            if metadata.get(key) != expected.get(key):
                mismatches.append(key)
        if mismatches:
            return CacheStatus(
                exists=True,
                valid=False,
                cache_path=self.cache_path,
                metadata_path=self.metadata_path,
                reason=f"metadata mismatch: {', '.join(mismatches)}",
            )
        return CacheStatus(
            exists=True,
            valid=True,
            cache_path=self.cache_path,
            metadata_path=self.metadata_path,
            reason="ok",
        )

    def deprecate_existing(self) -> Path | None:
        if not self.cache_path.exists():
            return None
        suffix = datetime.now().strftime(".deprecated_%Y%m%d_%H%M%S")
        deprecated_path = self.cache_path.with_name(self.cache_path.name + suffix)
        shutil.move(str(self.cache_path), str(deprecated_path))
        self.logger.warning(
            "Cache metadata mismatch; moved old cache to %s", deprecated_path
        )
        return deprecated_path

    def ensure_valid_or_rebuild(
        self,
        rebuild_fn: Callable[["CacheManager"], None] | None = None,
    ) -> CacheStatus:
        status = self.validate()
        if status.valid:
            return status
        if status.exists:
            self.deprecate_existing()
        if rebuild_fn is not None:
            rebuild_fn(self)
            self.write_metadata()
            return self.validate()
        self.logger.warning(
            "Cache unavailable or invalid at %s (%s); raw fallback may be used",
            self.cache_path,
            status.reason,
        )
        return self.validate()

    def load_tensor(self, key: str) -> Any | None:
        if not self.cache_path.exists():
            return None
        try:
            import lmdb  # type: ignore
        except ImportError:
            self.logger.warning("lmdb is not installed; falling back to raw data")
            return None

        try:
            env = lmdb.open(
                str(self.cache_path),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=128,
            )
            with env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            env.close()
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as exc:
            self.logger.warning("Failed to read cache key=%s: %s", key, exc)
            return None

    def put_tensor(self, key: str, value: Any) -> None:
        try:
            import lmdb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("lmdb is required to write cache files") from exc

        self.cache_path.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(self.cache_path), map_size=1 << 40)
        with env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        env.close()
