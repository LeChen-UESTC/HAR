from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .encode_text_embedding import encode_text_bank
from .structured_descriptions import load_class_names, normalize_description_cache


def cache_text_bank_from_config(config: dict[str, Any]) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    paths = config["paths"]
    text_cfg = config["text_branch"]
    embedding_cfg = text_cfg["embedding"]

    class_names = load_class_names(
        paths["class_names"],
        max_classes=text_cfg.get("num_classes", config.get("dataset", {}).get("num_classes")),
    )
    if not class_names:
        raise ValueError("No class names loaded for text bank caching")
    logger.info(
        "Caching text bank: classes=%s text_mode=%s description_cache=%s output=%s embedding_model=%s",
        len(class_names),
        config.get("_meta", {}).get("text_mode"),
        paths["description_cache"],
        paths["text_bank"],
        paths["embedding_model"],
    )
    with Path(paths["description_cache"]).open("r", encoding="utf-8") as handle:
        descriptions = normalize_description_cache(json.load(handle))

    return encode_text_bank(
        class_names=class_names,
        descriptions=descriptions,
        model_path=paths["embedding_model"],
        prompt_template=embedding_cfg["prompt"],
        output_path=paths["text_bank"],
        variant=text_cfg.get("description_variant", "structured"),
        normalize=bool(embedding_cfg.get("normalize", True)),
        pooling_method=str(embedding_cfg.get("pooling", "last")),
        main_label_alpha=float(embedding_cfg.get("main_label_alpha", 0.7)),
        max_length=int(embedding_cfg.get("max_length", 512)),
        padding_side=str(embedding_cfg.get("padding_side", "left")),
        runtime=config.get("runtime", {}),
    )
