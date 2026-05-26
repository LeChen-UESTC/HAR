#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import train_prealign
import train_skeleton_gircse

from src.train.common import finalize_run, initialize_run_for_kind, parse_common_args


def main() -> None:
    args = parse_common_args("Unified HAR training entrypoint.")
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


if __name__ == "__main__":
    main()
