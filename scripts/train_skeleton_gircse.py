#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path

import torch

from src.evaluation.evaluator import evaluate_embedding_model, select_text_classes
from src.losses.stepwise_infonce import stepwise_infonce
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    initialize_run,
    load_text_bank,
    maybe_autocast,
    move_batch_to_device,
    parse_common_args,
    select_device,
)
from src.train.factory import build_optimizer, build_skeleton_gircse_model, place_skeleton_gircse_model
from src.utils.checkpoint import load_checkpoint, save_checkpoint, update_run_registry
from src.utils.distributed import is_main_process
from src.utils.metrics import append_jsonl
from src.utils.wandb_utils import wandb_log


def main() -> None:
    args = parse_common_args("Stage 2: generative Skeleton-GIRCSE training.")
    ctx = initialize_run(args)
    config = ctx["config"]
    logger = ctx["logger"]
    dirs = ctx["dirs"]
    device = select_device()

    cache_manager = build_cache_manager(config, logger)
    train_loader = build_dataloader(config, "manifest_train", cache_manager, logger, train=True)
    val_loader = None
    if config["train"].get("eval_during_train", False):
        val_loader = build_dataloader(config, "manifest_val", cache_manager, logger, train=False)

    shift_cfg = config.get("model", {}).get("shift_gcn", {})
    if config["train"].get("freeze_shift_gcn", False) and not shift_cfg.get("pretrained_path"):
        raise ValueError(
            "train.freeze_shift_gcn=true requires model.shift_gcn.pretrained_path. "
            "Point pretrained_path to the official Shift-GCN checkpoint used as the skeleton encoder."
        )

    model = place_skeleton_gircse_model(build_skeleton_gircse_model(config), config, device)
    if config.get("runtime", {}).get("device_map_train") is not None:
        logger.info(
            "Using device_map_train=%s for GIRCSE LLM; trainable skeleton modules are on %s",
            config.get("runtime", {}).get("device_map_train"),
            device,
        )
    else:
        model.to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=str(device), strict=False)
        logger.info("Loaded warmup checkpoint: %s", args.checkpoint)
    optimizer = build_optimizer(config, model)
    z_text, class_ids = load_text_bank(config["paths"]["text_bank"], device)
    z_text, class_ids = select_text_classes(
        z_text,
        class_ids,
        config.get("dataset", {}).get("seen_classes") or None,
    )
    temperature = float(config["loss"].get("temperature", 0.05))
    lambda_irr = float(config["loss"].get("lambda_irr", 1.0))
    use_amp = config["train"].get("mixed_precision", "none") in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    grad_accum_steps = int(config["train"].get("gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {grad_accum_steps}")
    metrics_path = Path(dirs["model_dir"]) / "metrics.jsonl"
    best_top1 = -1.0

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        running_logs: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            if batch is None:
                continue
            with maybe_autocast(use_amp, config["train"].get("mixed_precision", "fp16")):
                z_steps, _z_final = model(batch["skeleton"])
                loss, logs = stepwise_infonce(
                    z_steps=z_steps,
                    z_text=z_text,
                    labels=batch["label"],
                    temperature=temperature,
                    lambda_irr=lambda_irr,
                    class_ids=class_ids,
                )
                loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()

            should_step = step % grad_accum_steps == 0
            if should_step:
                grad_clip = config["train"].get("grad_clip_norm")
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = batch["label"].numel()
            total_loss += float(loss.detach()) * batch_size
            total += batch_size
            for key, value in logs.items():
                running_logs[key] = running_logs.get(key, 0.0) + float(value) * batch_size
            if step % int(config["train"].get("log_freq", 20)) == 0 and is_main_process():
                logger.info("epoch=%s step=%s loss=%.6f", epoch, step, float(loss.detach()))

        if len(train_loader) % grad_accum_steps != 0:
            grad_clip = config["train"].get("grad_clip_norm")
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics = {"epoch": epoch, "train_loss": total_loss / max(total, 1)}
        metrics.update({key: value / max(total, 1) for key, value in running_logs.items()})

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
                save_checkpoint(Path(dirs["model_dir"]) / "best.ckpt", model, optimizer=optimizer, epoch=epoch, metrics=metrics)

        logger.info("epoch=%s train_loss=%.6f", epoch, metrics["train_loss"])
        append_jsonl(metrics_path, metrics)
        wandb_log(ctx["wandb_run"], metrics, step=epoch)
        model_dir = Path(dirs["model_dir"])
        save_freq = int(config["train"].get("save_freq", 1))
        if save_freq > 0 and epoch % save_freq == 0:
            save_checkpoint(model_dir / f"epoch_{epoch}.ckpt", model, optimizer=optimizer, epoch=epoch, metrics=metrics)
        save_checkpoint(model_dir / "last.ckpt", model, optimizer=optimizer, epoch=epoch, metrics=metrics)
        update_run_registry(model_dir, dirs["exp_name"], epoch, metrics)

    ctx["wandb_run"].finish()


if __name__ == "__main__":
    main()
