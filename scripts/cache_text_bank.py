#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import logging

from src.text_branch.cache_text_bank import cache_text_bank_from_config
from src.train.common import apply_runtime_environment, materialize_run_config, parse_common_args


def main() -> None:
    args = parse_common_args("Cache Qwen3Embedding4B text banks.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = materialize_run_config(args, run_kind="text_bank")
    apply_runtime_environment(config)
    payload = cache_text_bank_from_config(config)
    logging.info(
        "Cached text bank: classes=%s shape=%s output=%s",
        len(payload["class_names"]),
        tuple(payload["z_text"].shape),
        config["paths"]["text_bank"],
    )


if __name__ == "__main__":
    main()
