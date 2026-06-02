#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path

import torch

from src.evaluation.evaluator import select_text_classes
from src.losses.classwise_infonce import classwise_multibank_infonce
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    finalize_run,
    expected_text_bank_metadata,
    initialize_run_for_kind,
    load_text_bank_bundle,
    maybe_autocast,
    move_batch_to_device,
    parse_common_args,
    resolve_mixed_precision,
    resolve_text_bank_path,
    select_device,
)
from src.train.eval_during_train import (
    build_train_eval_loaders,
    build_train_eval_text_banks,
    evaluate_zsl_gzsl_during_train,
)
from src.train.factory import (
    build_embedding_model_for_stage,
    build_optimizer,
    checkpoint_include_prefixes,
    configure_text_embedding_dim,
    place_embedding_model,
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint, update_run_registry
from src.utils.distributed import is_main_process
from src.utils.metrics import append_jsonl
from src.utils.wandb_utils import wandb_log


def run(ctx: dict, args) -> None:
    config = ctx["config"]
    logger = ctx["logger"]
    dirs = ctx["dirs"]
    device = select_device(config)

    cache_manager = build_cache_manager(config, logger)
    train_loader = build_dataloader(config, "manifest_train", cache_manager, logger, train=True)
    train_eval_loaders = None
    if config["train"].get("eval_during_train", False):
        train_eval_loaders = build_train_eval_loaders(config, cache_manager, logger)

    model = place_embedding_model(build_embedding_model_for_stage(config), config, device)
    text_bank_path = resolve_text_bank_path(config, "train")
    logger.info("Using text bank: mode=%s path=%s", config.get("_meta", {}).get("text_mode"), text_bank_path)
    text_banks_all, class_ids_all = load_text_bank_bundle(
        text_bank_path,
        device,
        expected_text_mode=config.get("_meta", {}).get("text_mode"),
        expected_metadata=expected_text_bank_metadata(config),
    )
    configure_text_embedding_dim(model, int(text_banks_all["main"].shape[-1]), device)
    if args.checkpoint:
        load_checkpoint(
            args.checkpoint,
            model,
            map_location="cpu",
            strict=False,
            include_prefixes=checkpoint_include_prefixes(config),
            expected_projector_type=config.get("_meta", {}).get("projector_type"),
            expected_text_mode=config.get("_meta", {}).get("text_mode"),
        )
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    optimizer = build_optimizer(config, model)
    train_eval_text_banks = None
    if train_eval_loaders is not None:
        train_eval_text_banks = build_train_eval_text_banks(config, text_banks_all["main"], class_ids_all)

    seen_classes = config.get("dataset", {}).get("seen_classes") or None
    text_banks = {}
    class_ids = class_ids_all
    for bank_name in ("main", "motion", "phase"):
        text_banks[bank_name], class_ids = select_text_classes(
            text_banks_all[bank_name],
            class_ids_all,
            seen_classes,
        )

    temperature = float(config["loss"].get("temperature", 0.05))
    bank_weights = {
        "motion": float(config["loss"].get("lambda_motion", 0.0)),
        "phase": float(config["loss"].get("lambda_phase", 0.0)),
    }
    mixed_precision = resolve_mixed_precision(config["train"])
    use_amp = mixed_precision in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    grad_accum_steps = int(config["train"].get("gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {grad_accum_steps}")
    metrics_path = Path(dirs["model_dir"]) / "metrics.jsonl"
    best_top1 = -1.0
    global_step = 0
    checkpoint_extra = {
        "text_mode": config.get("_meta", {}).get("text_mode"),
        "text_variant": config.get("_meta", {}).get("text_variant"),
        "projector_type": config.get("_meta", {}).get("projector_type"),
        "projector_mode": config.get("_meta", {}).get("projector_mode"),
        "text_bank_path": text_bank_path,
        "text_embedding_dim": int(text_banks_all["main"].shape[-1]),
        "embedding_model_path": config.get("paths", {}).get("embedding_model"),
        "train_stage": config.get("train", {}).get("stage"),
    }

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        pending_backward_steps = 0
        valid_step = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            if batch is None:
                continue
            global_step += 1
            valid_step += 1
            with maybe_autocast(use_amp, mixed_precision):
                z = model(batch["skeleton"])
                loss, _logs = classwise_multibank_infonce(
                    z=z,
                    text_banks=text_banks,
                    labels=batch["label"],
                    temperature=temperature,
                    class_ids=class_ids,
                    bank_weights=bank_weights,
                )
                loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()
            pending_backward_steps += 1
            if valid_step % grad_accum_steps == 0:
                grad_clip = config["train"].get("grad_clip_norm")
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_backward_steps = 0

            batch_size = batch["label"].numel()
            total_loss += float(loss.detach()) * batch_size
            total += batch_size
            if step % int(config["train"].get("log_freq", 20)) == 0 and is_main_process():
                logger.info("epoch=%s step=%s loss=%.6f", epoch, step, float(loss.detach()))

        if pending_backward_steps:
            grad_clip = config["train"].get("grad_clip_norm")
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics = {"epoch": epoch, "train_loss": total_loss / max(total, 1)}
        if (
            train_eval_loaders is not None
            and train_eval_text_banks is not None
            and epoch % int(config["train"].get("eval_freq", 1)) == 0
        ):
            eval_metrics = evaluate_zsl_gzsl_during_train(
                model=model,
                loaders=train_eval_loaders,
                text_banks=train_eval_text_banks,
                config=config,
                device=device,
            )
            metrics.update(eval_metrics)
            if eval_metrics["zsl_top1"] > best_top1:
                best_top1 = eval_metrics["zsl_top1"]
                save_checkpoint(
                    Path(dirs["model_dir"]) / "best.ckpt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    extra=checkpoint_extra,
                    include_prefixes=checkpoint_include_prefixes(config),
                    trainable_only=True,
                )

        logger.info("epoch=%s train_loss=%.6f", epoch, metrics["train_loss"])
        append_jsonl(metrics_path, metrics)
        wandb_log(ctx["wandb_run"], metrics, step=epoch)
        model_dir = Path(dirs["model_dir"])
        save_freq = int(config["train"].get("save_freq", 1))
        if save_freq > 0 and epoch % save_freq == 0:
            save_checkpoint(
                model_dir / f"epoch_{epoch}.ckpt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                extra=checkpoint_extra,
                include_prefixes=checkpoint_include_prefixes(config),
                trainable_only=True,
            )
        save_checkpoint(
            model_dir / "last.ckpt",
            model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            extra=checkpoint_extra,
            include_prefixes=checkpoint_include_prefixes(config),
            trainable_only=True,
        )
        update_run_registry(model_dir, dirs["exp_name"], epoch, metrics)


def main() -> None:
    args = parse_common_args("Train skeleton embedding model or direct baseline.")
    ctx = initialize_run_for_kind(args, run_kind="train")
    try:
        run(ctx, args)
    except Exception as exc:
        finalize_run(ctx, status="failed", extra={"error": repr(exc)})
        raise
    finalize_run(ctx)


if __name__ == "__main__":
    main()
