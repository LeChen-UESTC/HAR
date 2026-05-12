from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .distributed import get_local_rank, get_rank, is_main_process


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = get_rank()
        record.gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(get_local_rank()))
        if not hasattr(record, "sample_index"):
            record.sample_index = "-"
        return True


def setup_logger(
    name: str = "skeleton_gircse",
    log_root: str | Path = "logs",
    filename: str | None = None,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    Path(log_root).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = Path(log_root) / filename

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | rank=%(rank)s | gpu=%(gpu)s | sample=%(sample_index)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())
    logger.addHandler(file_handler)

    if is_main_process():
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(ContextFilter())
        logger.addHandler(stream_handler)

    latest_path = Path(log_root) / "experiment_latest.log"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(log_path.name)
    except OSError:
        pass

    return logger, log_path


def log_config_summary(logger: logging.Logger, config: dict[str, Any]) -> None:
    dataset = config.get("dataset", {}).get("name", "unknown")
    stage = config.get("train", {}).get("stage", "unknown")
    loss_type = config.get("loss", {}).get("type", "unknown")
    logger.info("Run summary: dataset=%s stage=%s loss=%s", dataset, stage, loss_type)
