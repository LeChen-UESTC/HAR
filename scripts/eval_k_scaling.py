#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.evaluator import evaluate_embedding_model, select_text_classes
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    initialize_run,
    load_text_bank,
    parse_common_args,
    select_device,
)
from src.train.factory import build_skeleton_gircse_model
from src.utils.checkpoint import load_checkpoint


def main() -> None:
    args = parse_common_args("Evaluate test-time soft-token K scaling.")
    ctx = initialize_run(args)
    config = ctx["config"]
    logger = ctx["logger"]
    dirs = ctx["dirs"]
    device = select_device()

    cache_manager = build_cache_manager(config, logger)
    test_loader = build_dataloader(config, "manifest_test", cache_manager, logger, train=False)
    model = build_skeleton_gircse_model(config).to(device)
    checkpoint = args.checkpoint or config.get("paths", {}).get("checkpoint")
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=str(device), strict=False)
    else:
        logger.warning("No checkpoint provided; evaluating randomly initialized trainable modules.")

    z_text, class_ids = load_text_bank(config["paths"]["text_bank"], device)
    z_text, class_ids = select_text_classes(
        z_text,
        class_ids,
        config.get("dataset", {}).get("unseen_classes") or None,
    )

    results = []
    original_k = model.soft_token_generator.K
    for k in config["model"]["soft_tokens"].get("k_test", [1, 3, 5, 10, 20]):
        model.soft_token_generator.K = int(k)
        metrics = evaluate_embedding_model(
            model=model,
            dataloader=test_loader,
            z_text=z_text,
            device=device,
            class_ids=class_ids,
        )
        item = {"k_test": int(k), "top1": metrics["top1"], "num_samples": metrics["num_samples"]}
        results.append(item)
        logger.info("K_test=%s top1=%.4f", k, metrics["top1"])
    model.soft_token_generator.K = original_k

    output_dir = Path(dirs["eval_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "k_scaling_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    ctx["wandb_run"].finish()


if __name__ == "__main__":
    main()
