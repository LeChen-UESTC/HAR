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
    z = F.normalize(z.float(), dim=-1)
    z_text = F.normalize(z_text.float(), dim=-1).to(z.device)
    target = label_to_text_indices(labels.to(z.device), class_ids=class_ids)
    logits = z @ z_text.t()
    losses = F.cross_entropy(logits / temperature, target, reduction="none")
    loss = losses.mean()
    if return_per_sample:
        return loss, losses
    return loss
