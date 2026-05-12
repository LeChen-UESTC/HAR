#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from src.evaluation.evaluator import evaluate_embedding_model, save_eval_outputs, select_text_classes
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
    args = parse_common_args("Evaluate Skeleton-GIRCSE under GZSL candidates.")
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
    candidate_classes = []
    candidate_classes.extend(config.get("dataset", {}).get("seen_classes") or [])
    candidate_classes.extend(config.get("dataset", {}).get("unseen_classes") or [])
    z_text, class_ids = select_text_classes(z_text, class_ids, candidate_classes or None)
    gamma = float(config.get("eval", {}).get("calibrated_stacking_gamma", 0.0))
    metrics = evaluate_embedding_model(
        model=model,
        dataloader=test_loader,
        z_text=z_text,
        device=device,
        class_ids=class_ids,
        seen_classes=config.get("dataset", {}).get("seen_classes") or None,
        gamma=gamma,
    )
    metrics["calibrated_stacking_gamma"] = gamma
    logger.info("GZSL top1=%.4f num_samples=%s gamma=%.4f", metrics["top1"], metrics["num_samples"], gamma)
    save_eval_outputs(metrics, Path(dirs["eval_dir"]))
    ctx["wandb_run"].finish()


if __name__ == "__main__":
    main()
