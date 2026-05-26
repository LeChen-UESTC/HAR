#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path

import torch

from src.evaluation.evaluator import evaluate_embedding_model, select_text_classes
from src.losses.classwise_infonce import classwise_infonce
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    finalize_run,
    initialize_run_for_kind,
    load_text_bank,
    maybe_autocast,
    move_batch_to_device,
    parse_common_args,
    select_device,
)
from src.train.factory import build_optimizer, build_warmup_model
from src.utils.checkpoint import save_checkpoint
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
    val_loader = None
    if config["train"].get("eval_during_train", False):
        val_loader = build_dataloader(config, "manifest_val", cache_manager, logger, train=False)
    model = build_warmup_model(config).to(device)
    optimizer = build_optimizer(config, model)
    z_text, class_ids = load_text_bank(config["paths"]["text_bank"], device)
    z_text, class_ids = select_text_classes(
        z_text,
        class_ids,
        config.get("dataset", {}).get("seen_classes") or None,
    )
    projector_dim = int(config["model"]["projector"]["llm_dim"])
    text_dim = int(z_text.shape[-1])
    if projector_dim != text_dim:
        raise ValueError(
            f"Projector llm_dim={projector_dim} does not match text bank dim={text_dim}. "
            "Regenerate the text bank with the configured GIRCSE model or fix model.projector.llm_dim."
        )
    temperature = float(config["loss"].get("temperature", 0.05))
    use_amp = config["train"].get("mixed_precision", "none") in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    metrics_path = Path(dirs["model_dir"]) / "metrics.jsonl"
    best_top1 = -1.0
    global_step = 0
    eval_steps = config["train"].get("eval_steps")
    eval_steps = int(eval_steps) if eval_steps else None

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            if batch is None:
                continue
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(use_amp, config["train"].get("mixed_precision", "fp16")):
                z = model(batch["skeleton"])
                loss = classwise_infonce(z, z_text, batch["label"], temperature, class_ids=class_ids)
            scaler.scale(loss).backward()
            grad_clip = config["train"].get("grad_clip_norm")
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach()) * batch["label"].numel()
            total += batch["label"].numel()
            if step % int(config["train"].get("log_freq", 20)) == 0 and is_main_process():
                logger.info("epoch=%s step=%s loss=%.6f", epoch, step, float(loss.detach()))
            if val_loader is not None and eval_steps and global_step % eval_steps == 0:
                eval_metrics = evaluate_embedding_model(
                    model=model,
                    dataloader=val_loader,
                    z_text=z_text,
                    device=device,
                    class_ids=class_ids,
                )
                step_metrics = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_top1": eval_metrics["top1"],
                    "val_num_samples": eval_metrics["num_samples"],
                }
                append_jsonl(metrics_path, step_metrics)
                wandb_log(ctx["wandb_run"], step_metrics, step=global_step)
                logger.info(
                    "epoch=%s global_step=%s val_top1=%.4f",
                    epoch,
                    global_step,
                    eval_metrics["top1"],
                )
                model.train()

        metrics = {"epoch": epoch, "train_loss": total_loss / max(total, 1)}
        if val_loader is not None and epoch % int(config["train"].get("eval_freq", 1)) == 0:
            eval_metrics = evaluate_embedding_model(
                model=model,
                dataloader=val_loader,
                z_text=z_text,
                device=device,
                class_ids=class_ids,
            )
            metrics["val_top1"] = eval_metrics["top1"]
            if eval_metrics["top1"] > best_top1:
                best_top1 = eval_metrics["top1"]
                save_checkpoint(
                    Path(dirs["model_dir"]) / "best.ckpt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    include_prefixes=("shift_gcn.", "token_projector."),
                    trainable_only=True,
                )
        logger.info("epoch=%s train_loss=%.6f", epoch, metrics["train_loss"])
        append_jsonl(metrics_path, metrics)
        wandb_log(ctx["wandb_run"], metrics, step=global_step if eval_steps else epoch)
        model_dir = Path(dirs["model_dir"])
        save_freq = int(config["train"].get("save_freq", 1))
        if save_freq > 0 and epoch % save_freq == 0:
            save_checkpoint(
                model_dir / f"epoch_{epoch}.ckpt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                include_prefixes=("shift_gcn.", "token_projector."),
                trainable_only=True,
            )
        save_checkpoint(
            model_dir / "last.ckpt",
            model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            include_prefixes=("shift_gcn.", "token_projector."),
            trainable_only=True,
        )


def main() -> None:
    args = parse_common_args("Stage 1: skeleton-text coarse pre-alignment warmup.")
    ctx = initialize_run_for_kind(args, run_kind="train")
    try:
        run(ctx, args)
    except Exception as exc:
        finalize_run(ctx, status="failed", extra={"error": repr(exc)})
        raise
    finalize_run(ctx)


if __name__ == "__main__":
    main()
