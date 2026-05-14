#!/usr/bin/env python
from __future__ import annotations

from src.text_branch.generate_rich_description import generate_descriptions, load_class_names
from src.train.common import parse_common_args
from src.utils.config_utils import apply_overrides, load_config


def main() -> None:
    args = parse_common_args("Generate rich action descriptions with Qwen2.5.")
    config = apply_overrides(load_config(args.config), args.override)
    class_names = load_class_names(
        config["paths"]["class_names"],
        max_classes=config.get("dataset", {}).get("num_classes"),
    )
    gen_cfg = config["text_branch"].get("generation", {})
    generate_descriptions(
        class_names=class_names,
        model_path=config["paths"]["qwen_instruct_model"],
        output_path=config["paths"]["description_cache"],
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
        temperature=float(gen_cfg.get("temperature", 0.2)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        dry_run=bool(gen_cfg.get("dry_run", False)),
        runtime=config.get("runtime", {}),
    )


if __name__ == "__main__":
    main()
