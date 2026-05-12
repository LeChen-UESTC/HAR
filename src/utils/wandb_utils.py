from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config_utils import to_builtin
from .distributed import is_main_process


class DummyWandbRun:
    def __init__(self, reason: str = "wandb disabled") -> None:
        self.reason = reason
        self.name = "disabled"

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def config_update(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def init_wandb(
    config: dict[str, Any],
    exp_name: str,
    run_dir: str | Path,
    mode: str = "offline",
    logger: Any | None = None,
) -> Any:
    if not is_main_process():
        return DummyWandbRun("non-main process")

    mode = mode or config.get("experiment", {}).get("wandb_mode", "offline")
    if mode == "disabled":
        return DummyWandbRun("disabled by config")

    os.environ.setdefault("WANDB_MODE", mode)
    try:
        import wandb  # type: ignore

        run = wandb.init(
            project=config.get("experiment", {}).get("project_name", "skeleton-gircse"),
            name=exp_name,
            dir=str(run_dir),
            mode=mode,
            tags=config.get("experiment", {}).get("tags", []),
            config=to_builtin(config),
            reinit=True,
        )
        return run
    except Exception as exc:
        if logger:
            logger.warning("WandB init failed in mode=%s; falling back to disabled: %s", mode, exc)
        os.environ["WANDB_MODE"] = "disabled"
        return DummyWandbRun(str(exc))


def wandb_log(run: Any, payload: dict[str, Any], step: int | None = None) -> None:
    if hasattr(run, "log"):
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
