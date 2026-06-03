from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.encoder import build_shift_gcn_from_config
from src.models.projection import TokenProjector
from src.models.qformer_projector import SkeletonQFormerProjector, qformer_config_from_dict
from src.models.skeleton_embedding import DirectQFormerEmbedding, SkeletonEmbeddingModel
from src.models.skeleton_prompt_builder import SkeletonPromptBuilder
from src.models.text_space_projection import TextSpaceProjection
from src.utils.torch_utils import hf_model_kwargs


class WarmupSkeletonTextModel(nn.Module):
    def __init__(self, shift_gcn: nn.Module, token_projector: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.shift_gcn = shift_gcn
        self.token_projector = token_projector
        self.embedding_projection = TextSpaceProjection(hidden_dim)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        tokens = self.token_projector(feat)
        return self.embedding_projection(tokens.mean(dim=1))

    def set_text_embedding_dim(self, target_dim: int) -> None:
        self.embedding_projection.configure(target_dim)


def build_projector(config: dict[str, Any], output_dim: int | None = None) -> nn.Module:
    cfg = dict(config["model"]["projector"])
    if output_dim is not None:
        cfg["llm_dim"] = int(output_dim)
    projector_type = str(cfg.get("type", "linear"))
    if projector_type in {"linear", "linear_layernorm"}:
        return TokenProjector(
            in_dim=int(cfg["in_dim"]),
            llm_dim=int(cfg["llm_dim"]),
            target_temporal_bins=int(cfg.get("target_temporal_bins", 4)),
        )
    if projector_type in {"qformer", "general_qformer", "part_aware_qformer"}:
        if projector_type == "general_qformer":
            cfg["type"] = "qformer"
            cfg["use_part_token_embeddings"] = False
        if projector_type == "part_aware_qformer":
            cfg["use_part_token_embeddings"] = bool(cfg.get("use_part_token_embeddings", True))
        return SkeletonQFormerProjector(qformer_config_from_dict(cfg))
    raise ValueError(f"Unsupported projector.type={projector_type}")


def build_warmup_model(config: dict[str, Any]) -> WarmupSkeletonTextModel:
    _validate_frozen_shift_gcn_checkpoint(config)
    hidden_dim = resolve_embedding_hidden_size(config)
    shift_gcn = build_shift_gcn_from_config(config)
    if config.get("train", {}).get("freeze_shift_gcn", False):
        for param in shift_gcn.parameters():
            param.requires_grad = False
    return WarmupSkeletonTextModel(
        shift_gcn=shift_gcn,
        token_projector=build_projector(config, output_dim=hidden_dim),
        hidden_dim=hidden_dim,
    )


def build_skeleton_embedding_model(config: dict[str, Any]) -> SkeletonEmbeddingModel:
    shift_gcn = build_shift_gcn_from_config(config)
    if config.get("train", {}).get("freeze_shift_gcn", False):
        for param in shift_gcn.parameters():
            param.requires_grad = False
    embedding_model, tokenizer = load_embedding_model_and_tokenizer(config)
    if config.get("train", {}).get("freeze_embedding_model", True):
        for param in embedding_model.parameters():
            param.requires_grad = False
    if config.get("train", {}).get("gradient_checkpointing", False):
        if hasattr(embedding_model, "gradient_checkpointing_enable"):
            embedding_model.gradient_checkpointing_enable()
        if hasattr(embedding_model, "config"):
            embedding_model.config.use_cache = False
    hidden_dim = int(embedding_model.config.hidden_size)
    prompt_builder = SkeletonPromptBuilder(
        tokenizer=tokenizer,
        token_embedding=embedding_model.get_input_embeddings(),
        prompt_text=config["model"]["prompt"]["text"],
    )
    return SkeletonEmbeddingModel(
        shift_gcn=shift_gcn,
        token_projector=build_projector(config, output_dim=hidden_dim),
        embedding_model=embedding_model,
        prompt_builder=prompt_builder,
    )


def build_direct_qformer_baseline(config: dict[str, Any]) -> DirectQFormerEmbedding:
    hidden_dim = resolve_embedding_hidden_size(config)
    shift_gcn = build_shift_gcn_from_config(config)
    if config.get("train", {}).get("freeze_shift_gcn", False):
        for param in shift_gcn.parameters():
            param.requires_grad = False
    return DirectQFormerEmbedding(
        shift_gcn=shift_gcn,
        token_projector=build_projector(config, output_dim=hidden_dim),
        hidden_dim=hidden_dim,
    )


def build_embedding_model_for_stage(config: dict[str, Any]) -> nn.Module:
    stage = str(config.get("train", {}).get("stage", "skeleton_embedding")).lower()
    if stage == "direct_qformer_baseline":
        return build_direct_qformer_baseline(config)
    if stage == "skeleton_embedding":
        return build_skeleton_embedding_model(config)
    raise ValueError(
        "Unsupported train.stage="
        f"{stage!r}. Expected prealign, skeleton_embedding, or direct_qformer_baseline."
    )


def build_model_for_stage(config: dict[str, Any], stage: str | None = None) -> nn.Module:
    resolved_stage = str(stage or config.get("train", {}).get("stage", "skeleton_embedding")).lower()
    if resolved_stage in {"prealign", "warmup"}:
        return build_warmup_model(config)
    if resolved_stage in {"skeleton_embedding", "direct_qformer_baseline"}:
        stage_config = dict(config)
        stage_config["train"] = dict(config.get("train", {}))
        stage_config["train"]["stage"] = resolved_stage
        return build_embedding_model_for_stage(stage_config)
    raise ValueError(
        f"Unsupported train.stage={resolved_stage!r}. "
        "Expected prealign, skeleton_embedding, or direct_qformer_baseline."
    )


def place_model_for_stage(
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    stage: str | None = None,
) -> nn.Module:
    resolved_stage = str(stage or config.get("train", {}).get("stage", "skeleton_embedding")).lower()
    if resolved_stage in {"prealign", "warmup"}:
        return model.to(device)
    return place_embedding_model(model, config, device)


def load_embedding_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    model_path = config["paths"]["embedding_model"]
    trust_remote_code = bool(config.get("runtime", {}).get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side=str(config.get("text_branch", {}).get("embedding", {}).get("padding_side", "left")),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = hf_model_kwargs(config, for_text=False)
    kwargs["local_files_only"] = True
    model = AutoModel.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def resolve_embedding_hidden_size(config: dict[str, Any]) -> int:
    from transformers import AutoConfig

    model_path = config["paths"]["embedding_model"]
    model_cfg = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("runtime", {}).get("trust_remote_code", True)),
        local_files_only=True,
    )
    hidden_size = getattr(model_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Embedding model config is missing hidden_size: {model_path}")
    return int(hidden_size)


def place_embedding_model(
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if config.get("runtime", {}).get("device_map_train") is not None:
        model.shift_gcn.to(device)
        model.token_projector.to(device)
        if hasattr(model, "embedding_projection"):
            model.embedding_projection.to(device)
        return model
    return model.to(device)


def _validate_frozen_shift_gcn_checkpoint(config: dict[str, Any]) -> None:
    train_cfg = config.get("train", {})
    shift_cfg = config.get("model", {}).get("shift_gcn", {})
    if train_cfg.get("freeze_shift_gcn", False) and not shift_cfg.get("pretrained_path"):
        raise ValueError(
            "train.freeze_shift_gcn=true requires model.shift_gcn.pretrained_path. "
            "Point pretrained_path to the official Shift-GCN checkpoint used as the skeleton encoder."
        )


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    param_groups = []
    projector_params = [
        p for name, p in model.named_parameters()
        if _is_projector_side_parameter(name)
        if p.requires_grad
    ]
    shift_params = [
        p for name, p in model.named_parameters()
        if "shift_gcn" in name
        if p.requires_grad
    ]
    other_params = [
        p for name, p in model.named_parameters()
        if not _is_projector_side_parameter(name) and "shift_gcn" not in name
        if p.requires_grad
    ]
    if projector_params:
        param_groups.append({"params": projector_params, "lr": float(train_cfg.get("lr_projector", 1e-4))})
    if shift_params:
        param_groups.append({"params": shift_params, "lr": float(train_cfg.get("lr_shift_gcn", 1e-5))})
    if other_params:
        param_groups.append({"params": other_params, "lr": float(train_cfg.get("lr", 1e-4))})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(
        param_groups,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )


def _is_projector_side_parameter(name: str) -> bool:
    return name.startswith(
        (
            "token_projector.",
            "embedding_projection.",
            "embedding_head.",
        )
    )


def checkpoint_include_prefixes(config: dict[str, Any]) -> tuple[str, ...]:
    stage = str(config.get("train", {}).get("stage", "")).lower()
    return checkpoint_include_prefixes_for_stage(stage)


def checkpoint_include_prefixes_for_stage(stage: str) -> tuple[str, ...]:
    stage = str(stage).lower()
    if stage == "direct_qformer_baseline":
        return ("shift_gcn.", "token_projector.", "embedding_head.", "embedding_projection.")
    return ("shift_gcn.", "token_projector.", "embedding_projection.")


def configure_text_embedding_dim(model: nn.Module, text_dim: int, device: torch.device) -> nn.Module:
    if hasattr(model, "set_text_embedding_dim"):
        model.set_text_embedding_dim(int(text_dim))
        if hasattr(model, "embedding_projection"):
            model.embedding_projection.to(device)
    else:
        raise TypeError(f"Model {type(model).__name__} does not support text embedding projection")
    return model
