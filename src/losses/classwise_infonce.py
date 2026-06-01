from __future__ import annotations

import torch
import torch.nn.functional as F


def label_to_text_indices(labels: torch.Tensor, class_ids: torch.Tensor | None = None) -> torch.Tensor:
    if class_ids is None:
        return labels.long()
    class_ids = class_ids.to(labels.device)
    matches = labels.long().view(-1, 1) == class_ids.long().view(1, -1)
    if not torch.all(matches.any(dim=1)):
        missing = labels[~matches.any(dim=1)].detach().cpu().tolist()
        raise ValueError(f"Labels are not present in text bank class_ids: {missing}")
    return matches.float().argmax(dim=1).long()


def classwise_infonce(
    z: torch.Tensor,
    z_text: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05,
    class_ids: torch.Tensor | None = None,
    return_per_sample: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if z.shape[-1] != z_text.shape[-1]:
        raise ValueError(
            f"Embedding dim mismatch: z dim={z.shape[-1]} but z_text dim={z_text.shape[-1]}"
        )
    z = F.normalize(z.float(), dim=-1)
    z_text = F.normalize(z_text.float(), dim=-1).to(z.device)
    target = label_to_text_indices(labels.to(z.device), class_ids=class_ids)
    logits = z @ z_text.t()
    losses = F.cross_entropy(logits / temperature, target, reduction="none")
    loss = losses.mean()
    if return_per_sample:
        return loss, losses
    return loss


def classwise_multibank_infonce(
    z: torch.Tensor,
    text_banks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    temperature: float = 0.05,
    class_ids: torch.Tensor | None = None,
    bank_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "main" not in text_banks:
        raise ValueError("text_banks must contain a main bank")
    weights = bank_weights or {}
    total, _main_per_sample = classwise_infonce(
        z=z,
        z_text=text_banks["main"],
        labels=labels,
        temperature=temperature,
        class_ids=class_ids,
        return_per_sample=True,
    )
    logs = {"loss_main": total.detach()}
    for name in ("motion", "phase"):
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        if name not in text_banks:
            raise ValueError(f"Missing auxiliary text bank: {name}")
        loss = classwise_infonce(
            z=z,
            z_text=text_banks[name],
            labels=labels,
            temperature=temperature,
            class_ids=class_ids,
        )
        total = total + weight * loss
        logs[f"loss_{name}"] = loss.detach()
    logs["loss_total"] = total.detach()
    return total, logs
