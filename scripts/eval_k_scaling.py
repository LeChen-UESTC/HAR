#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

from src.evaluation.evaluator import evaluate_embedding_model, resolve_class_scope, select_text_classes
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    finalize_run,
    initialize_run_for_kind,
    load_text_bank,
    parse_common_args,
    select_device,
)
from src.train.factory import build_skeleton_gircse_model, place_skeleton_gircse_model
from src.utils.checkpoint import load_checkpoint


def run(ctx: dict, args) -> None:
    config = ctx["config"]
    logger = ctx["logger"]
    dirs = ctx["dirs"]
    device = select_device(config)

    cache_manager = build_cache_manager(config, logger)
    eval_cfg = config.get("eval", {})
    split_key = str(eval_cfg.get("split_key", "manifest_test"))
    test_loader = build_dataloader(config, split_key, cache_manager, logger, train=False)
    model = place_skeleton_gircse_model(build_skeleton_gircse_model(config), config, device)
    checkpoint = args.checkpoint or config.get("paths", {}).get("checkpoint")
    if checkpoint:
        load_checkpoint(
            checkpoint,
            model,
            map_location="cpu",
            strict=False,
            include_prefixes=("shift_gcn.", "token_projector."),
        )
    else:
        logger.warning("No checkpoint provided; evaluating randomly initialized trainable modules.")

    z_text, class_ids = load_text_bank(config["paths"]["text_bank"], device)
    candidate_classes = resolve_class_scope(config, eval_cfg.get("candidate_scope", "unseen"))
    z_text, class_ids = select_text_classes(
        z_text,
        class_ids,
        candidate_classes,
    )

    results = []
    original_k = model.soft_token_generator.K
    k_values = eval_cfg.get("k_values", config["model"]["soft_tokens"].get("k_test", [1, 3, 5, 10, 20]))
    for k in k_values:
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


def main() -> None:
    args = parse_common_args("Evaluate test-time soft-token K scaling.")
    ctx = initialize_run_for_kind(args, run_kind="eval")
    try:
        run(ctx, args)
    except Exception as exc:
        finalize_run(ctx, status="failed", extra={"error": repr(exc)})
        raise
    finalize_run(ctx)


if __name__ == "__main__":
    main()
