from __future__ import annotations

import torch

from .classwise_infonce import classwise_infonce
from .iterative_refinement_regularizer import iterative_refinement_regularizer


def stepwise_infonce(
    z_steps: list[torch.Tensor],
    z_text: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05,
    lambda_irr: float = 1.0,
    class_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not z_steps:
        raise ValueError("z_steps must not be empty")

    scalar_losses = []
    per_sample_losses = []
    for z in z_steps:
        loss, per_sample = classwise_infonce(
            z=z,
            z_text=z_text,
            labels=labels,
            temperature=temperature,
            class_ids=class_ids,
            return_per_sample=True,
        )
        scalar_losses.append(loss)
        per_sample_losses.append(per_sample)

    loss_step = torch.stack(scalar_losses).mean()
    loss_irr = iterative_refinement_regularizer(per_sample_losses)
    total = loss_step + lambda_irr * loss_irr
    logs = {
        "loss_step": loss_step.detach(),
        "loss_irr": loss_irr.detach(),
        "loss_total": total.detach(),
    }
    for idx, loss in enumerate(scalar_losses, start=1):
        logs[f"loss_step_{idx}"] = loss.detach()
    return total, logs
