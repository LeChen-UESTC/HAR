from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .graph import NTUGraph
from .shift import Shift


def conv_init(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.modules.batchnorm._BatchNorm, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1) -> None:
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Shift_tcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1) -> None:
        super().__init__()
        del kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)
        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.temporal_linear.weight, mode="fan_out")
        if self.temporal_linear.bias is not None:
            nn.init.constant_(self.temporal_linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        coff_embedding: int = 4,
        num_subset: int = 3,
    ) -> None:
        super().__init__()
        del A, coff_embedding, num_subset
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels))
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))
        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.constant_(self.Linear_bias, 0)
        self.Feature_Mask = nn.Parameter(torch.ones(1, 25, in_channels))
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25 * out_channels)
        self.relu = nn.ReLU()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)

        index_array = np.empty(25 * in_channels, dtype=np.int64)
        for i in range(25):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (
                    i * in_channels + j + j * in_channels
                ) % (in_channels * 25)
        self.register_buffer("shift_in", torch.from_numpy(index_array), persistent=True)

        index_array = np.empty(25 * out_channels, dtype=np.int64)
        for i in range(25):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (
                    i * out_channels + j - j * out_channels
                ) % (out_channels * 25)
        self.register_buffer("shift_out", torch.from_numpy(index_array), persistent=True)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()

        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)

        x = torch.einsum("nwc,cd->nwd", (x, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias

        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    """Official Shift-GCN architecture adapted for skeleton embedding."""

    out_dim = 256

    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        graph: str | None = None,
        graph_args: dict[str, Any] | None = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        del graph
        graph_args = graph_args or {}
        self.graph = NTUGraph(**graph_args)
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.constant_(self.fc.bias, 0)
        bn_init(self.data_bn, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected skeleton shape [B,C,T,V,M], got {tuple(x.shape)}")
        n, c, t, v, m = x.size()
        if c != self.in_channels or v != self.num_point or m != self.num_person:
            raise ValueError(
                "Unexpected skeleton shape: "
                f"expected C={self.in_channels}, V={self.num_point}, M={self.num_person}; "
                f"got C={c}, V={v}, M={m}"
            )

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        _, c_new, t_new, v_new = x.size()
        x = x.view(n, m, c_new, t_new, v_new).mean(dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = feat.mean(dim=(-1, -2))
        return self.fc(pooled)
