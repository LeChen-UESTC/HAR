from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.train.common import move_batch_to_device


def compute_logits(
    z: torch.Tensor,
    z_text: torch.Tensor,
    class_ids: torch.Tensor | None = None,
    seen_classes: list[int] | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    z = F.normalize(z.float(), dim=-1)
    z_text = F.normalize(z_text.float(), dim=-1).to(z.device)
    logits = z @ z_text.t()
    if gamma and seen_classes:
        ids = class_ids if class_ids is not None else torch.arange(z_text.shape[0], device=z.device)
        seen = torch.tensor(seen_classes, dtype=ids.dtype, device=z.device)
        mask = (ids.view(-1, 1) == seen.view(1, -1)).any(dim=1)
        logits[:, mask] -= gamma
    return logits


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor, class_ids: torch.Tensor | None = None) -> float:
    pred_indices = logits.argmax(dim=1)
    if class_ids is None:
        preds = pred_indices
    else:
        preds = class_ids.to(logits.device)[pred_indices]
    return (preds.long() == labels.long()).float().mean().item()


def select_text_classes(
    z_text: torch.Tensor,
    class_ids: torch.Tensor,
    selected_classes: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not selected_classes:
        return z_text, class_ids
    selected = torch.tensor(selected_classes, dtype=class_ids.dtype, device=class_ids.device)
    mask = (class_ids.view(-1, 1) == selected.view(1, -1)).any(dim=1)
    return z_text[mask], class_ids[mask]


@torch.no_grad()
def evaluate_embedding_model(
    model: Any,
    dataloader: Any,
    z_text: torch.Tensor,
    device: torch.device,
    class_ids: torch.Tensor | None = None,
    seen_classes: list[int] | None = None,
    gamma: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    predictions = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if batch is None:
            continue
        if hasattr(model, "warmup_embedding"):
            _steps, z = model(batch["skeleton"])
        else:
            z = model(batch["skeleton"])
        logits = compute_logits(
            z=z,
            z_text=z_text,
            class_ids=class_ids,
            seen_classes=seen_classes,
            gamma=gamma,
        )
        pred_indices = logits.argmax(dim=1)
        pred_labels = pred_indices if class_ids is None else class_ids.to(device)[pred_indices]
        labels = batch["label"]
        total += labels.numel()
        correct += (pred_labels.long() == labels.long()).sum().item()
        predictions.extend(
            {
                "sample_id": sample_id,
                "label": int(label),
                "prediction": int(pred),
            }
            for sample_id, label, pred in zip(batch["sample_id"], labels.detach().cpu(), pred_labels.detach().cpu())
        )

    return {
        "top1": correct / max(total, 1),
        "num_samples": total,
        "predictions": predictions,
    }


def save_eval_outputs(metrics: dict[str, Any], output_dir: str | Path) -> None:
    import json
    import pickle

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = dict(metrics)
    predictions = metrics.pop("predictions", [])
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with (output_path / "predictions.pkl").open("wb") as handle:
        pickle.dump(predictions, handle)
