from __future__ import annotations

import torch


def cumulative_mean(states: torch.Tensor) -> list[torch.Tensor]:
    if states.ndim != 3:
        raise ValueError(f"Expected states [B,K,D], got {tuple(states.shape)}")
    z_steps = []
    running = torch.zeros_like(states[:, 0, :])
    for idx in range(states.shape[1]):
        running = running + states[:, idx, :]
        z_steps.append(running / float(idx + 1))
    return z_steps
