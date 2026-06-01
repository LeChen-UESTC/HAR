#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import logging

from src.text_branch.generate_rich_description import generate_descriptions, load_class_names
from src.train.common import apply_runtime_environment, materialize_run_config, parse_common_args


def main() -> None:
    args = parse_common_args("Generate structured action descriptions with Qwen2.5.")
    config = materialize_run_config(args, run_kind="text_description")
    apply_runtime_environment(config)
    gen_cfg = config["text_branch"].get("generation", {})
    class_names = load_class_names(
        config["paths"]["class_names"],
        max_classes=gen_cfg.get("num_classes", config.get("dataset", {}).get("num_classes")),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    print(f"Class count: {len(class_names)}")
    print(f"Model path : {config['paths']['qwen_instruct_model']}")
    print(f"Output    : {config['paths']['description_cache']}")
    generate_descriptions(
        class_names=class_names,
        model_path=config["paths"]["qwen_instruct_model"],
        output_path=config["paths"]["description_cache"],
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
        temperature=float(gen_cfg.get("temperature", 0.2)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        dry_run=bool(gen_cfg.get("dry_run", False)),
        runtime=config.get("runtime", {}),
        max_retries=int(gen_cfg.get("max_retries", 3)),
    )


if __name__ == "__main__":
    main()
