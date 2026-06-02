#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import eval_gzsl
import eval_zsl

from src.train.common import finalize_run, initialize_run_for_kind, parse_common_args


def main() -> None:
    args = parse_common_args("Unified HAR evaluation entrypoint.")
    ctx = initialize_run_for_kind(args, run_kind="eval")
    task = str(ctx["config"].get("eval", {}).get("task", "")).lower()
    try:
        if task == "zsl":
            eval_zsl.run(ctx, args)
        elif task == "gzsl":
            eval_gzsl.run(ctx, args)
        else:
            raise ValueError(
                "Unsupported eval.task="
                f"{task!r}. Expected zsl or gzsl."
            )
    except Exception as exc:
        finalize_run(ctx, status="failed", extra={"error": repr(exc)})
        raise
    finalize_run(ctx)


if __name__ == "__main__":
    main()
