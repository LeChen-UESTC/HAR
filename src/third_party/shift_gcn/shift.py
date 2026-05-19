from __future__ import annotations

import torch
from torch import nn


class Shift(nn.Module):
    """Pure PyTorch fallback for the official Shift-GCN temporal shift op.

    The official implementation uses a custom CUDA extension with learnable
    per-channel offsets. This fallback keeps the same `xpos`/`ypos` parameter
    names so official checkpoints load, and applies the rounded temporal
    offsets with zero padding.
    """

    def __init__(self, channel: int, stride: int, init_scale: float = 3.0) -> None:
        super().__init__()
        self.stride = stride
        self.xpos = nn.Parameter(torch.empty(channel))
        self.ypos = nn.Parameter(torch.empty(channel))
        nn.init.uniform_(self.xpos, -1e-8, 1e-8)
        nn.init.uniform_(self.ypos, -init_scale, init_scale)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected [N,C,T,V], got {tuple(input_tensor.shape)}")
        offset = self.ypos
        if self.stride != 1:
            offset = offset + 0.5
        shifted = self._temporal_shift(input_tensor, offset)
        if self.stride != 1:
            shifted = shifted[:, :, :: self.stride, :]
        return shifted

    @staticmethod
    def _temporal_shift(input_tensor: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(input_tensor)
        max_shift = max(input_tensor.shape[2] - 1, 0)
        offsets = torch.round(offset.detach()).to(torch.long).clamp(-max_shift, max_shift)
        for value in torch.unique(offsets).tolist():
            channel_mask = offsets == int(value)
            if value == 0:
                output[:, channel_mask, :, :] = input_tensor[:, channel_mask, :, :]
            elif value > 0:
                output[:, channel_mask, value:, :] = input_tensor[:, channel_mask, :-value, :]
            else:
                lag = -value
                output[:, channel_mask, :-lag, :] = input_tensor[:, channel_mask, lag:, :]
        return output
