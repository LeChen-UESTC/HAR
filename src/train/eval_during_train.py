from __future__ import annotations

from typing import Any

import torch

from src.evaluation.evaluator import evaluate_embedding_model, select_text_classes
from src.train.common import build_dataloader, resolve_class_scope


def build_train_eval_loaders(
    config: dict[str, Any],
    cache_manager: Any,
    logger: Any,
) -> dict[str, Any]:
    split_key = str(config.get("eval", {}).get("split_key", "manifest_test"))
    return {
        "zsl": build_dataloader(
            config,
            split_key,
            cache_manager,
            logger,
            train=False,
            sample_scope="unseen",
        ),
        "gzsl": build_dataloader(
            config,
            split_key,
            cache_manager,
            logger,
            train=False,
            sample_scope="all",
        ),
    }


def build_train_eval_text_banks(
    config: dict[str, Any],
    z_text: torch.Tensor,
    class_ids: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    zsl_classes = resolve_class_scope(config, "unseen")
    gzsl_classes = resolve_class_scope(config, "all")
    if not zsl_classes:
        raise ValueError("Training-time ZSL eval requires non-empty dataset.unseen_classes")
    if not gzsl_classes:
        raise ValueError("Training-time GZSL eval requires non-empty seen+unseen classes")
    return {
        "zsl": select_text_classes(z_text, class_ids, zsl_classes),
        "gzsl": select_text_classes(z_text, class_ids, gzsl_classes),
    }


def evaluate_zsl_gzsl_during_train(
    model: Any,
    loaders: dict[str, Any],
    text_banks: dict[str, tuple[torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    zsl_text, zsl_class_ids = text_banks["zsl"]
    zsl_metrics = evaluate_embedding_model(
        model=model,
        dataloader=loaders["zsl"],
        z_text=zsl_text,
        device=device,
        class_ids=zsl_class_ids,
    )

    gzsl_text, gzsl_class_ids = text_banks["gzsl"]
    seen_classes = config.get("dataset", {}).get("seen_classes") or None
    gamma = float(config.get("eval", {}).get("calibrated_stacking_gamma", 0.0))
    gzsl_metrics = evaluate_embedding_model(
        model=model,
        dataloader=loaders["gzsl"],
        z_text=gzsl_text,
        device=device,
        class_ids=gzsl_class_ids,
        seen_classes=seen_classes,
        gamma=gamma,
    )

    return {
        "zsl_top1": float(zsl_metrics["top1"]),
        "zsl_num_samples": int(zsl_metrics["num_samples"]),
        "gzsl_top1": float(gzsl_metrics["top1"]),
        "gzsl_seen_top1": float(gzsl_metrics.get("seen_top1", 0.0)),
        "gzsl_unseen_top1": float(gzsl_metrics.get("unseen_top1", 0.0)),
        "gzsl_h_mean": float(gzsl_metrics.get("h_mean", 0.0)),
        "gzsl_num_samples": int(gzsl_metrics["num_samples"]),
    }
