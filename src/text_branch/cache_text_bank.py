from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .encode_text_gircse import encode_text_bank
from .generate_rich_description import load_class_names


def cache_text_bank_from_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    text_cfg = config["text_branch"]
    embedding_cfg = text_cfg["embedding"]

    class_names = load_class_names(paths["class_names"])
    with Path(paths["description_cache"]).open("r", encoding="utf-8") as handle:
        descriptions = json.load(handle)

    return encode_text_bank(
        class_names=class_names,
        descriptions=descriptions,
        model_path=paths["gircse_model"],
        prompt_template=embedding_cfg["prompt"],
        output_path=paths["text_bank"],
        variant=text_cfg.get("description_variant", "full"),
        k_text=int(embedding_cfg.get("k_text", 20)),
        normalize=bool(embedding_cfg.get("normalize", True)),
        logit_temperature=float(embedding_cfg.get("logit_temperature", 1.0)),
        pooling_method=str(embedding_cfg.get("pooling", "generate_mean")),
        runtime=config.get("runtime", {}),
    )
