#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import copy
import gc

import train_prealign
import train_skeleton_gircse

from src.train.common import finalize_run, initialize_run_for_kind, materialize_run_config, parse_common_args


def _k_train_values(config: dict) -> list[int]:
    raw = config.get("model", {}).get("soft_tokens", {}).get("k_train", 5)
    if not isinstance(raw, list):
        return []
    if not raw:
        raise ValueError("model.soft_tokens.k_train must not be an empty list")
    values = []
    for item in raw:
        value = int(item)
        if value < 1:
            raise ValueError(f"model.soft_tokens.k_train values must be >= 1, got {value}")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"model.soft_tokens.k_train contains duplicate values: {values}")
    return values


def _args_for_k(args, k: int):
    run_args = copy.copy(args)
    run_args.override = list(args.override) + [f"model.soft_tokens.k_train={k}"]
    if args.exp_name:
        run_args.exp_name = f"{args.exp_name}_K{k}"
    return run_args


def _run_once(args) -> None:
    ctx = initialize_run_for_kind(args, run_kind="train")
    stage = str(ctx["config"].get("train", {}).get("stage", "")).lower()
    try:
        if stage in {"prealign", "warmup"}:
            train_prealign.run(ctx, args)
        elif stage in {"skeleton_gircse", "gircse", "stage2"}:
            train_skeleton_gircse.run(ctx, args)
        else:
            raise ValueError(
                "Unsupported train.stage="
                f"{stage!r}. Expected prealign or skeleton_gircse."
            )
    except Exception as exc:
        finalize_run(ctx, status="failed", extra={"error": repr(exc)})
        raise
    finalize_run(ctx)


def _cleanup_after_run() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_common_args("Unified HAR training entrypoint.")
    preview = materialize_run_config(args, run_kind="train")
    stage = str(preview.get("train", {}).get("stage", "")).lower()
    k_values = _k_train_values(preview)
    if k_values and stage in {"skeleton_gircse", "gircse", "stage2"}:
        for k in k_values:
            _run_once(_args_for_k(args, k))
            _cleanup_after_run()
        return
    _run_once(args)


if __name__ == "__main__":
    main()
