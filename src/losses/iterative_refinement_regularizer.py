from __future__ import annotations

import torch


def iterative_refinement_regularizer(
    step_losses: list[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    if len(step_losses) <= 1:
        return step_losses[0].new_tensor(0.0) if step_losses else torch.tensor(0.0)

    penalties = []
    for prev, nxt in zip(step_losses[:-1], step_losses[1:]):
        penalties.append(torch.relu(torch.log(nxt + eps) - torch.log(prev + eps)))
    return torch.stack(penalties, dim=0).mean()
