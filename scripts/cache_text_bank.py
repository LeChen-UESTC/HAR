#!/usr/bin/env python
from __future__ import annotations

from src.text_branch.cache_text_bank import cache_text_bank_from_config
from src.train.common import parse_common_args
from src.utils.config_utils import apply_overrides, load_config


def main() -> None:
    args = parse_common_args("Cache GIRCSE text embeddings.")
    config = apply_overrides(load_config(args.config), args.override)
    cache_text_bank_from_config(config)


if __name__ == "__main__":
    main()
