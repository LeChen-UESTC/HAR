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
    expected_text_bank_metadata,
    finalize_run,
    initialize_run_for_kind,
    load_text_bank,
    parse_common_args,
    resolve_text_bank_path,
    select_device,
)
from src.train.factory import (
    build_model_for_stage,
    checkpoint_include_prefixes_for_stage,
    configure_text_embedding_dim,
    place_model_for_stage,
)
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
    text_bank_path = resolve_text_bank_path(config, "eval")
    z_text, class_ids = load_text_bank(
        text_bank_path,
        device,
        expected_text_mode=config.get("_meta", {}).get("text_mode"),
        expected_metadata=expected_text_bank_metadata(config),
    )
    train_stage = str(config.get("train", {}).get("stage", "skeleton_embedding")).lower()
    model = place_model_for_stage(build_model_for_stage(config, train_stage), config, device, train_stage)
    configure_text_embedding_dim(model, int(z_text.shape[-1]), device)
    checkpoint = args.checkpoint or config.get("paths", {}).get("checkpoint")
    if checkpoint:
        load_checkpoint(
            checkpoint,
            model,
            map_location="cpu",
            strict=False,
            include_prefixes=checkpoint_include_prefixes_for_stage(train_stage),
            expected_projector_type=config.get("_meta", {}).get("projector_type"),
            expected_text_mode=config.get("_meta", {}).get("text_mode"),
            expected_train_stage=train_stage,
        )
    else:
        logger.warning("No checkpoint provided; evaluating randomly initialized trainable modules.")

    candidate_classes = resolve_class_scope(config, eval_cfg.get("candidate_scope", "unseen"))
    z_text, class_ids = select_text_classes(
        z_text,
        class_ids,
        candidate_classes,
    )
    metrics = evaluate_embedding_model(
        model=model,
        dataloader=test_loader,
        z_text=z_text,
        device=device,
        class_ids=class_ids,
    )
    metrics["text_mode"] = config.get("_meta", {}).get("text_mode")
    metrics["projector_type"] = config.get("_meta", {}).get("projector_type")
    metrics["train_stage"] = train_stage
    metrics["text_bank_path"] = text_bank_path
    logger.info("ZSL top1=%.4f num_samples=%s", metrics["top1"], metrics["num_samples"])
    save_eval_outputs(metrics, Path(dirs["eval_dir"]))


def main() -> None:
    args = parse_common_args("Evaluate skeleton embedding model under ZSL candidates.")
    ctx = initialize_run_for_kind(args, run_kind="eval")
    try:
        run(ctx, args)
    except Exception as exc:
        finalize_run(ctx, status="failed", extra={"error": repr(exc)})
        raise
    finalize_run(ctx)


if __name__ == "__main__":
    main()
