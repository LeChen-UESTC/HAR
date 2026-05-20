#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path

from src.evaluation.evaluator import (
    evaluate_embedding_model,
    resolve_class_scope,
    save_eval_outputs,
    select_text_classes,
)
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    initialize_run,
    load_text_bank,
    parse_common_args,
    select_device,
)
from src.train.factory import build_skeleton_gircse_model, place_skeleton_gircse_model
from src.utils.checkpoint import load_checkpoint


def main() -> None:
    args = parse_common_args("Evaluate Skeleton-GIRCSE under GZSL candidates.")
    ctx = initialize_run(args)
    config = ctx["config"]
    logger = ctx["logger"]
    dirs = ctx["dirs"]
    device = select_device()

    cache_manager = build_cache_manager(config, logger)
    eval_cfg = config.get("eval", {})
    split_key = str(eval_cfg.get("split_key", "manifest_test"))
    test_loader = build_dataloader(config, split_key, cache_manager, logger, train=False)
    model = place_skeleton_gircse_model(build_skeleton_gircse_model(config), config, device)
    eval_k = eval_cfg.get("k")
    if eval_k is not None:
        model.soft_token_generator.K = int(eval_k)
    checkpoint = args.checkpoint or config.get("paths", {}).get("checkpoint")
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=str(device), strict=False)
    else:
        logger.warning("No checkpoint provided; evaluating randomly initialized trainable modules.")

    z_text, class_ids = load_text_bank(config["paths"]["text_bank"], device)
    candidate_classes = resolve_class_scope(config, eval_cfg.get("candidate_scope", "all"))
    z_text, class_ids = select_text_classes(z_text, class_ids, candidate_classes or None)
    gamma = float(eval_cfg.get("calibrated_stacking_gamma", 0.0))
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
    logger.info(
        "GZSL top1=%.4f seen=%.4f unseen=%.4f H=%.4f num_samples=%s gamma=%.4f",
        metrics["top1"],
        metrics.get("seen_top1", 0.0),
        metrics.get("unseen_top1", 0.0),
        metrics.get("h_mean", 0.0),
        metrics["num_samples"],
        gamma,
    )
    save_eval_outputs(metrics, Path(dirs["eval_dir"]))
    ctx["wandb_run"].finish()


if __name__ == "__main__":
    main()
