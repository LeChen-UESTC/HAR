#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.encoder import build_shift_gcn_from_config
from src.train.common import (
    build_cache_manager,
    build_dataloader,
    initialize_run,
    maybe_autocast,
    move_batch_to_device,
    parse_common_args,
    select_device,
)
from src.utils.checkpoint import save_checkpoint
from src.utils.distributed import is_main_process
from src.utils.metrics import append_jsonl
from src.utils.wandb_utils import wandb_log


def main() -> None:
    args = parse_common_args("Stage 0: supervised Shift-GCN pretraining on seen classes.")
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

    model = build_shift_gcn_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["train"].get("lr_shift_gcn", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    use_amp = config["train"].get("mixed_precision", "none") in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    metrics_path = Path(dirs["model_dir"]) / "metrics.jsonl"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            if batch is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(use_amp, config["train"].get("mixed_precision", "fp16")):
                logits = model(batch["skeleton"])
                loss = F.cross_entropy(logits, batch["label"])
            scaler.scale(loss).backward()
            grad_clip = config["train"].get("grad_clip_norm")
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()

            labels = batch["label"]
            total_loss += float(loss.detach()) * labels.numel()
            total += labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            if step % int(config["train"].get("log_freq", 20)) == 0 and is_main_process():
                logger.info("epoch=%s step=%s loss=%.6f", epoch, step, float(loss.detach()))

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_top1": correct / max(total, 1),
        }
        logger.info("epoch=%s train_loss=%.6f train_top1=%.4f", epoch, metrics["train_loss"], metrics["train_top1"])
        append_jsonl(metrics_path, metrics)
        wandb_log(ctx["wandb_run"], metrics, step=epoch)
        save_checkpoint(Path(dirs["model_dir"]) / "last.ckpt", model, optimizer=optimizer, epoch=epoch, metrics=metrics)
        if epoch % int(config["train"].get("save_freq", 1)) == 0:
            save_checkpoint(Path(dirs["model_dir"]) / f"epoch_{epoch}.ckpt", model, optimizer=optimizer, epoch=epoch, metrics=metrics)

    ctx["wandb_run"].finish()


if __name__ == "__main__":
    main()
