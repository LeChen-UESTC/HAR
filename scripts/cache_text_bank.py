#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

from src.text_branch.cache_text_bank import cache_text_bank_from_config
from src.train.common import apply_runtime_environment, materialize_run_config, parse_common_args


def main() -> None:
    args = parse_common_args("Cache GIRCSE text embeddings.")
    config = materialize_run_config(args, run_kind="text_bank")
    apply_runtime_environment(config)
    cache_text_bank_from_config(config)


if __name__ == "__main__":
    main()
