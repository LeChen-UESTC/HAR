from pathlib import Path

from src.data.cache_manager import CacheManager
from src.data.samplers import SamplingStrategy


def test_cache_name_contains_sampling_hash_and_preprocess_version(tmp_path: Path) -> None:
    strategy = SamplingStrategy(fps=1, max_frames=32, frame_interval=2)
    manager = CacheManager("ntu60", tmp_path, strategy, "v1")

    assert strategy.hash() in manager.cache_name
    assert manager.cache_name.endswith("_v1.lmdb")


def test_cache_validation_detects_sampling_mismatch(tmp_path: Path) -> None:
    strategy = SamplingStrategy(fps=1, max_frames=32)
    manager = CacheManager("ntu60", tmp_path, strategy, "v1")
    manager.write_metadata()

    changed = CacheManager("ntu60", tmp_path, SamplingStrategy(fps=1, max_frames=64), "v1")
    changed.cache_path = manager.cache_path
    changed.metadata_path = manager.metadata_path

    status = changed.validate()

    assert status.exists is True
    assert status.valid is False
    assert "sampling_strategy" in status.reason


def test_invalid_cache_is_deprecated(tmp_path: Path) -> None:
    strategy = SamplingStrategy(fps=1, max_frames=32)
    manager = CacheManager("ntu60", tmp_path, strategy, "v1")
    manager.cache_path.mkdir(parents=True)

    status = manager.ensure_valid_or_rebuild()

    assert status.valid is False
    deprecated = list(tmp_path.glob("*.deprecated_*"))
    assert len(deprecated) == 1
