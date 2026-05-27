from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config_utils import DESCRIPTION_VARIANT_SHORT_NAMES, PROJECTOR_TYPE_NAMES
from .distributed import is_main_process, unwrap_model


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
    include_prefixes: tuple[str, ...] | None = None,
    trainable_only: bool = False,
) -> None:
    if not is_main_process():
        return
    import torch

    unwrapped = unwrap_model(model)
    state_dict = unwrapped.state_dict()
    if include_prefixes:
        state_dict = {
            key: value.detach().cpu()
            for key, value in state_dict.items()
            if key.startswith(include_prefixes)
        }
    if trainable_only:
        trainable_names = {
            name
            for name, param in unwrapped.named_parameters()
            if param.requires_grad
        }
        trainable_module_prefixes = {
            name.rsplit(".", 1)[0]
            for name in trainable_names
            if "." in name
        }
        state_dict = {
            key: value.detach().cpu()
            for key, value in state_dict.items()
            if key in trainable_names
            or any(key.startswith(f"{prefix}.") for prefix in trainable_module_prefixes)
        }

    payload: dict[str, Any] = {
        "model": state_dict,
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
    include_prefixes: tuple[str, ...] | None = None,
    expected_projector_type: str | None = None,
    expected_text_mode: str | None = None,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location=map_location)
    validate_checkpoint_identity(
        payload,
        expected_text_mode=expected_text_mode,
        expected_projector_type=expected_projector_type,
        checkpoint_path=path,
    )
    state_dict = payload["model"]
    if include_prefixes:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(include_prefixes)
        }
    unwrap_model(model).load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def validate_checkpoint_text_mode(
    payload: Mapping[str, Any],
    expected_text_mode: str | None,
    checkpoint_path: str | Path,
) -> None:
    validate_checkpoint_identity(
        payload,
        expected_text_mode=expected_text_mode,
        expected_projector_type=None,
        checkpoint_path=checkpoint_path,
    )


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    expected_text_mode: str | None,
    expected_projector_type: str | None,
    checkpoint_path: str | Path,
) -> None:
    extra = payload.get("extra", {})
    actual_text_mode = extra.get("text_mode") if isinstance(extra, Mapping) else None
    if not actual_text_mode:
        actual_text_mode = infer_text_mode_from_path(checkpoint_path)
    if expected_text_mode and not actual_text_mode:
        raise ValueError(
            "Checkpoint is missing text_mode metadata and its path does not include a text_mode suffix: "
            f"{checkpoint_path}. Expected {expected_text_mode}."
        )
    if expected_text_mode and actual_text_mode != expected_text_mode:
        raise ValueError(
            "Checkpoint text_mode does not match current config. "
            f"checkpoint={checkpoint_path} actual={actual_text_mode} expected={expected_text_mode}"
        )

    actual_projector = extra.get("projector_type") if isinstance(extra, Mapping) else None
    if not actual_projector:
        actual_projector = infer_projector_type_from_path(checkpoint_path)
    if expected_projector_type and not actual_projector:
        raise ValueError(
            "Checkpoint is missing projector_type metadata and its path does not include a projector suffix: "
            f"{checkpoint_path}. Expected {expected_projector_type}."
        )
    if expected_projector_type and actual_projector != expected_projector_type:
        raise ValueError(
            "Checkpoint projector_type does not match current config. "
            f"checkpoint={checkpoint_path} actual={actual_projector} expected={expected_projector_type}"
        )


def infer_text_mode_from_path(path: str | Path) -> str | None:
    suffixes = sorted(
        {f"_{short}" for short in DESCRIPTION_VARIANT_SHORT_NAMES.values()},
        key=len,
        reverse=True,
    )
    for part in reversed(Path(path).parts):
        name = Path(part).stem
        for suffix in suffixes:
            if name.endswith(suffix) or f"{suffix}_" in name:
                return suffix
    return None


def infer_projector_type_from_path(path: str | Path) -> str | None:
    suffixes = sorted(
        {f"_{projector_type}" for projector_type in PROJECTOR_TYPE_NAMES},
        key=len,
        reverse=True,
    )
    for part in reversed(Path(path).parts):
        name = Path(part).stem
        for suffix in suffixes:
            if name.endswith(suffix) or f"{suffix}_" in name:
                return suffix[1:]
    return None
