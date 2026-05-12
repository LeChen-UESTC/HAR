from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.encoder import build_shift_gcn_from_config
from src.models.projection import TokenProjector
from src.models.qformer_projector import SkeletonQFormerProjector, qformer_config_from_dict
from src.models.skeleton_gircse import SkeletonGIRCSE
from src.models.skeleton_prompt_builder import SkeletonPromptBuilder
from src.models.soft_token_generator import SoftTokenGenerator
from src.utils.torch_utils import hf_model_kwargs


class WarmupSkeletonTextModel(nn.Module):
    def __init__(self, shift_gcn: nn.Module, token_projector: nn.Module) -> None:
        super().__init__()
        self.shift_gcn = shift_gcn
        self.token_projector = token_projector

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        feat = self.shift_gcn.forward_features(skeleton)
        tokens = self.token_projector(feat)
        return torch.nn.functional.normalize(tokens.mean(dim=1).float(), dim=-1)


def build_projector(config: dict[str, Any]) -> nn.Module:
    cfg = config["model"]["projector"]
    projector_type = str(cfg.get("type", "linear"))
    if projector_type in {"linear", "linear_layernorm"}:
        return TokenProjector(
            in_dim=int(cfg["in_dim"]),
            llm_dim=int(cfg["llm_dim"]),
            target_temporal_bins=int(cfg.get("target_temporal_bins", 4)),
        )
    if projector_type in {"qformer", "general_qformer", "part_aware_qformer"}:
        qformer_cfg = dict(cfg)
        if projector_type == "general_qformer":
            qformer_cfg["type"] = "qformer"
            qformer_cfg["use_part_token_embeddings"] = False
        if projector_type == "part_aware_qformer":
            qformer_cfg["use_part_token_embeddings"] = bool(
                qformer_cfg.get("use_part_token_embeddings", True)
            )
        return SkeletonQFormerProjector(qformer_config_from_dict(qformer_cfg))
    raise ValueError(f"Unsupported projector.type={projector_type}")


def build_warmup_model(config: dict[str, Any]) -> WarmupSkeletonTextModel:
    shift_gcn = build_shift_gcn_from_config(config)
    if config.get("train", {}).get("freeze_shift_gcn", False):
        for param in shift_gcn.parameters():
            param.requires_grad = False
    return WarmupSkeletonTextModel(shift_gcn, build_projector(config))


def build_skeleton_gircse_model(config: dict[str, Any]) -> SkeletonGIRCSE:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shift_gcn = build_shift_gcn_from_config(config)
    projector = build_projector(config)
    model_path = config["paths"]["gircse_model"]
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        **hf_model_kwargs(config, for_text=False),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("runtime", {}).get("trust_remote_code", True)),
    )

    if config.get("train", {}).get("freeze_llm", True):
        for param in llm.parameters():
            param.requires_grad = False
    if config.get("train", {}).get("freeze_lm_head", True) and hasattr(llm, "lm_head"):
        for param in llm.lm_head.parameters():
            param.requires_grad = False

    expected_dim = int(config["model"]["projector"]["llm_dim"])
    actual_dim = int(llm.config.hidden_size)
    if expected_dim != actual_dim:
        raise ValueError(
            f"Projector llm_dim={expected_dim} does not match GIRCSE hidden_size={actual_dim}"
        )

    prompt_builder = SkeletonPromptBuilder(
        tokenizer=tokenizer,
        token_embedding=llm.get_input_embeddings(),
        prompt_text=config["model"]["prompt"]["text"],
    )
    generator = SoftTokenGenerator(
        llm=llm,
        token_embedding_table=llm.get_input_embeddings(),
        K=int(config["model"]["soft_tokens"].get("k_train", 5)),
        normalize=True,
        logit_temperature=float(config["model"]["soft_tokens"].get("logit_temperature", 1.0)),
        pooling_method=str(config["model"]["soft_tokens"].get("pooling", "generate_mean")),
    )
    return SkeletonGIRCSE(
        shift_gcn=shift_gcn,
        token_projector=projector,
        soft_token_generator=generator,
        prompt_builder=prompt_builder,
    )


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    param_groups = []
    projector_params = [
        p for name, p in model.named_parameters()
        if "token_projector" in name or "projector" in name
        if p.requires_grad
    ]
    shift_params = [
        p for name, p in model.named_parameters()
        if "shift_gcn" in name
        if p.requires_grad
    ]
    other_params = [
        p for name, p in model.named_parameters()
        if "token_projector" not in name and "projector" not in name and "shift_gcn" not in name
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
