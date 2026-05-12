from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ShiftGCNBackbone(nn.Module):
    """Shift-GCN compatible interface.

    This module provides the expected tensor contract for the rest of the
    Skeleton-GIRCSE pipeline. It can be replaced by a full Shift-GCN
    implementation without changing projector, loss, train, or eval code.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        out_dim: int = 512,
        num_classes: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(out_dim, num_classes) if num_classes else None

    def forward_features(self, skeleton: torch.Tensor) -> torch.Tensor:
        """Return feature map [B, C_s, T', V] from input [B, C, T, V, M]."""
        if skeleton.ndim != 5:
            raise ValueError(f"Expected skeleton shape [B,C,T,V,M], got {tuple(skeleton.shape)}")
        x = skeleton.mean(dim=-1)
        return self.features(x)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(skeleton)
        if self.classifier is None:
            return feat
        pooled = feat.mean(dim=(-1, -2))
        return self.classifier(pooled)


def build_shift_gcn_from_config(config: dict[str, Any]) -> ShiftGCNBackbone:
    model_cfg = config.get("model", {}).get("shift_gcn", {})
    dataset_cfg = config.get("dataset", {})
    model = ShiftGCNBackbone(
        in_channels=int(model_cfg.get("in_channels", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        out_dim=int(model_cfg.get("out_dim", 512)),
        num_classes=dataset_cfg.get("num_classes"),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    pretrained_path = model_cfg.get("pretrained_path")
    if pretrained_path:
        load_backbone_weights(model, pretrained_path, strict=False)
    if model_cfg.get("freeze", False):
        for param in model.parameters():
            param.requires_grad = False
    return model


def load_backbone_weights(model: nn.Module, path: str | Path, strict: bool = False) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
