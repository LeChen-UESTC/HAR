#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import gc

import train_embedding_baseline
import train_prealign

from src.train.common import finalize_run, initialize_run_for_kind, parse_common_args


def _run_once(args) -> None:
    ctx = initialize_run_for_kind(args, run_kind="train")
    stage = str(ctx["config"].get("train", {}).get("stage", "")).lower()
    try:
        if stage in {"prealign", "warmup"}:
            train_prealign.run(ctx, args)
        elif stage in {
            "skeleton_embedding",
            "direct_qformer_baseline",
        }:
            train_embedding_baseline.run(ctx, args)
        else:
            raise ValueError(
                "Unsupported train.stage="
                f"{stage!r}. Expected prealign, skeleton_embedding, "
                "or direct_qformer_baseline."
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
    _run_once(args)
    _cleanup_after_run()


if __name__ == "__main__":
    main()
