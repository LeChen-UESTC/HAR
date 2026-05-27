from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .encode_text_gircse import encode_text_bank
from .generate_rich_description import load_class_names


def cache_text_bank_from_config(config: dict[str, Any]) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    paths = config["paths"]
    text_cfg = config["text_branch"]
    embedding_cfg = text_cfg["embedding"]

    class_names = load_class_names(
        paths["class_names"],
        max_classes=text_cfg.get("generation", {}).get(
            "num_classes",
            config.get("dataset", {}).get("num_classes"),
        ),
    )
    if not class_names:
        raise ValueError("No class names loaded for text bank caching")
    logger.info(
        "Caching text bank: classes=%s text_mode=%s description_cache=%s output=%s k_text=%s",
        len(class_names),
        config.get("_meta", {}).get("text_mode"),
        paths["description_cache"],
        paths["text_bank"],
        embedding_cfg.get("k_text", 20),
    )
    with Path(paths["description_cache"]).open("r", encoding="utf-8") as handle:
        descriptions = json.load(handle)

    return encode_text_bank(
        class_names=class_names,
        descriptions=descriptions,
        base_model_path=paths.get("gircse_base_model", paths["qwen_instruct_model"]),
        adapter_path=paths.get("gircse_adapter", paths.get("gircse_model")),
        prompt_template=embedding_cfg["prompt"],
        output_path=paths["text_bank"],
        variant=text_cfg.get("description_variant", "full"),
        k_text=int(embedding_cfg.get("k_text", 20)),
        normalize=bool(embedding_cfg.get("normalize", True)),
        logit_temperature=float(embedding_cfg.get("logit_temperature", 1.0)),
        pooling_method=str(embedding_cfg.get("pooling", "generate_mean")),
        runtime=config.get("runtime", {}),
    )
