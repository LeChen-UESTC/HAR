from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .distributed import is_main_process, unwrap_model


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not is_main_process():
        return
    import torch

    payload: dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def update_run_registry(
    model_dir: str | Path,
    exp_name: str | Path,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    if not is_main_process():
        return

    model_path = Path(model_dir)
    registry_dir = model_path.parent / "all"
    registry_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_name": str(exp_name),
        "model_dir": str(model_path),
        "epoch": int(epoch),
        "metrics": metrics,
        "last_ckpt": str(model_path / "last.ckpt"),
        "epoch_ckpt": str(model_path / f"epoch_{epoch}.ckpt"),
    }
    with (registry_dir / "latest_run.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (registry_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
