from __future__ import annotations

import torch

from .classwise_infonce import classwise_infonce
from .iterative_refinement_regularizer import iterative_refinement_regularizer


def stepwise_infonce(
    z_steps: list[torch.Tensor],
    z_text: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05,
    lambda_irr: float = 0.1,
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


def stepwise_multibank_infonce(
    z_steps: list[torch.Tensor],
    text_banks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    temperature: float = 0.05,
    lambda_irr: float = 0.1,
    class_ids: torch.Tensor | None = None,
    bank_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not z_steps:
        raise ValueError("z_steps must not be empty")
    if "main" not in text_banks:
        raise ValueError("text_banks must contain a main bank")

    weights = bank_weights or {}
    main_losses = []
    main_per_sample = []
    aux_losses: dict[str, list[torch.Tensor]] = {"motion": [], "phase": []}
    for z in z_steps:
        main_loss, per_sample = classwise_infonce(
            z=z,
            z_text=text_banks["main"],
            labels=labels,
            temperature=temperature,
            class_ids=class_ids,
            return_per_sample=True,
        )
        main_losses.append(main_loss)
        main_per_sample.append(per_sample)
        for name in ("motion", "phase"):
            weight = float(weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            if name not in text_banks:
                raise ValueError(f"Missing auxiliary text bank: {name}")
            aux_losses[name].append(
                classwise_infonce(
                    z=z,
                    z_text=text_banks[name],
                    labels=labels,
                    temperature=temperature,
                    class_ids=class_ids,
                )
            )

    loss_main = torch.stack(main_losses).mean()
    loss_step = loss_main
    logs = {"loss_main": loss_main.detach()}
    for name, losses in aux_losses.items():
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss_step = loss_step + float(weights.get(name, 0.0)) * loss
        logs[f"loss_{name}"] = loss.detach()

    loss_irr = iterative_refinement_regularizer(main_per_sample)
    total = loss_step + lambda_irr * loss_irr
    logs.update(
        {
            "loss_step": loss_step.detach(),
            "loss_irr": loss_irr.detach(),
            "loss_total": total.detach(),
        }
    )
    for idx, loss in enumerate(main_losses, start=1):
        logs[f"loss_main_step_{idx}"] = loss.detach()
    return total, logs
