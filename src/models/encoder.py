from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import torch
from torch import nn

from src.third_party.shift_gcn import Model as OfficialShiftGCN


LOGGER = logging.getLogger(__name__)


class ShiftGCNBackbone(nn.Module):
    """Shift-GCN compatible interface.

    This module provides the expected tensor contract for the rest of the
    skeleton embedding pipeline. It can be replaced by a full Shift-GCN
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


def build_shift_gcn_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {}).get("shift_gcn", {})
    dataset_cfg = config.get("dataset", {})
    model_type = str(model_cfg.get("type", "simple_conv")).lower()
    num_classes = int(dataset_cfg.get("num_classes", model_cfg.get("num_classes", 60)))

    if model_type in {"official_shift_gcn", "shift_gcn", "official"}:
        model = OfficialShiftGCN(
            num_class=num_classes,
            num_point=int(model_cfg.get("num_joints", 25)),
            num_person=int(model_cfg.get("num_persons", 2)),
            graph_args=dict(model_cfg.get("graph_args", {"labeling_mode": "spatial"})),
            in_channels=int(model_cfg.get("in_channels", 3)),
        )
    elif model_type in {"simple_conv", "lightweight", "conv"}:
        model = ShiftGCNBackbone(
            in_channels=int(model_cfg.get("in_channels", 3)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            out_dim=int(model_cfg.get("out_dim", 512)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported model.shift_gcn.type={model_type}")

    pretrained_path = model_cfg.get("pretrained_path")
    if pretrained_path:
        load_backbone_weights(model, pretrained_path, strict=False)
    if model_cfg.get("freeze", False):
        for param in model.parameters():
            param.requires_grad = False
    return model


def load_backbone_weights(model: nn.Module, path: str | Path, strict: bool = False) -> None:
    weight_path = Path(path).expanduser()
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Shift-GCN checkpoint not found: {weight_path}. "
            "Set model.shift_gcn.pretrained_path to an existing official Shift-GCN .pt file."
        )
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    if not strict:
        state = _filter_state_by_shape(model, state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        LOGGER.warning(
            "Loaded Shift-GCN weights with missing=%s unexpected=%s",
            len(missing),
            len(unexpected),
        )
    if strict and (missing or unexpected):
        raise RuntimeError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return checkpoint


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state):
        return state
    return {key.removeprefix(prefix): value for key, value in state.items()}


def _filter_state_by_shape(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key in current_state and tuple(current_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        LOGGER.warning("Skipped Shift-GCN checkpoint tensors with incompatible shapes: %s", skipped)
    return filtered
