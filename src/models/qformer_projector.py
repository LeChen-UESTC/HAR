from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


DEFAULT_PART_AWARE_QUERY_ROLES = (
    "head",
    "left_arm",
    "right_arm",
    "torso",
    "left_leg",
    "right_leg",
    "global",
)


@dataclass(frozen=True)
class QFormerProjectorConfig:
    projector_type: str
    in_dim: int
    llm_dim: int
    num_query_tokens: int = 7
    query_roles: tuple[str, ...] = DEFAULT_PART_AWARE_QUERY_ROLES
    qformer_hidden_dim: int = 768
    qformer_num_layers: int = 6
    qformer_num_heads: int = 12
    qformer_intermediate_dim: int = 3072
    cross_attention_freq: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    init_from_pretrained: bool = False
    pretrained_name_or_path: str = "bert-base-uncased"
    use_part_token_embeddings: bool = False
    joint_part_roles: tuple[str, ...] = ()


class SkeletonQFormerProjector(nn.Module):
    """BLIP-2 Q-Former based skeleton tokenizer.

    Uses vendored Salesforce LAVIS Q-Former code instead of a hand-written
    Transformer. Input Shift-GCN feature maps are flattened into joint-time
    tokens and attended by learnable query tokens.
    """

    def __init__(self, cfg: QFormerProjectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.query_roles = list(cfg.query_roles)
        if cfg.num_query_tokens != len(self.query_roles):
            raise ValueError(
                f"num_query_tokens={cfg.num_query_tokens} must match query_roles length={len(self.query_roles)}"
            )

        self.input_norm = nn.LayerNorm(cfg.in_dim)
        self.part_role_to_id = self._build_part_role_ids(cfg)
        if cfg.use_part_token_embeddings:
            self.part_embeddings = nn.Embedding(len(self.part_role_to_id), cfg.in_dim)
        else:
            self.part_embeddings = None

        self.qformer = self._build_qformer(cfg)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, cfg.num_query_tokens, cfg.qformer_hidden_dim)
        )
        self.query_tokens.data.normal_(mean=0.0, std=cfg.initializer_range)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.qformer_hidden_dim, cfg.llm_dim),
            nn.LayerNorm(cfg.llm_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C_s, T', V]

        Returns:
            tokens: [B, M, d_llm]
        """
        if feat.ndim != 4:
            raise ValueError(f"Expected feature shape [B,C,T,V], got {tuple(feat.shape)}")
        batch_size, channels, time_steps, joints = feat.shape
        if channels != self.cfg.in_dim:
            raise ValueError(f"Projector in_dim={self.cfg.in_dim} does not match feature C={channels}")

        skeleton_tokens = feat.permute(0, 2, 3, 1).contiguous().view(batch_size, time_steps * joints, channels)
        skeleton_tokens = self.input_norm(skeleton_tokens)
        skeleton_tokens = self._add_part_embeddings(skeleton_tokens, time_steps, joints)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        encoder_attention_mask = torch.ones(
            skeleton_tokens.shape[:2],
            dtype=torch.long,
            device=skeleton_tokens.device,
        )
        outputs = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=skeleton_tokens,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        qformer_tokens = outputs.last_hidden_state[:, : self.cfg.num_query_tokens, :]
        return self.output_proj(qformer_tokens)

    def _add_part_embeddings(
        self,
        skeleton_tokens: torch.Tensor,
        time_steps: int,
        joints: int,
    ) -> torch.Tensor:
        if self.part_embeddings is None:
            return skeleton_tokens
        if len(self.cfg.joint_part_roles) != joints:
            raise ValueError(
                f"joint_part_roles length={len(self.cfg.joint_part_roles)} must match feature joints={joints}"
            )
        joint_part_ids = torch.tensor(
            [self.part_role_to_id[role] for role in self.cfg.joint_part_roles],
            dtype=torch.long,
            device=skeleton_tokens.device,
        )
        part_ids = joint_part_ids.repeat(time_steps)
        part_embed = self.part_embeddings(part_ids).unsqueeze(0)
        return skeleton_tokens + part_embed.to(skeleton_tokens.dtype)

    @staticmethod
    def _build_part_role_ids(cfg: QFormerProjectorConfig) -> dict[str, int]:
        roles = list(dict.fromkeys([role for role in cfg.query_roles if role != "global"]))
        for role in cfg.joint_part_roles:
            if role not in roles:
                roles.append(role)
        if "global" not in roles:
            roles.append("global")
        return {role: idx for idx, role in enumerate(roles)}

    @staticmethod
    def _build_qformer(cfg: QFormerProjectorConfig) -> nn.Module:
        from src.third_party.lavis_blip2_qformer.Qformer import BertConfig, BertLMHeadModel

        encoder_config = BertConfig()
        encoder_config.vocab_size = 30522
        encoder_config.hidden_size = cfg.qformer_hidden_dim
        encoder_config.num_hidden_layers = cfg.qformer_num_layers
        encoder_config.num_attention_heads = cfg.qformer_num_heads
        encoder_config.intermediate_size = cfg.qformer_intermediate_dim
        encoder_config.hidden_dropout_prob = cfg.hidden_dropout_prob
        encoder_config.attention_probs_dropout_prob = cfg.attention_probs_dropout_prob
        encoder_config.initializer_range = cfg.initializer_range
        encoder_config.encoder_width = cfg.in_dim
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cfg.cross_attention_freq
        encoder_config.query_length = cfg.num_query_tokens

        if cfg.init_from_pretrained:
            return BertLMHeadModel.from_pretrained(
                cfg.pretrained_name_or_path,
                config=encoder_config,
            )
        return BertLMHeadModel(encoder_config)


def qformer_config_from_dict(projector_cfg: dict[str, Any]) -> QFormerProjectorConfig:
    projector_type = str(projector_cfg.get("type", "qformer"))
    num_query_tokens = int(projector_cfg.get("num_query_tokens", 7))
    query_roles = tuple(projector_cfg.get("query_roles") or [])
    if not query_roles:
        if projector_type == "part_aware_qformer":
            query_roles = DEFAULT_PART_AWARE_QUERY_ROLES
        else:
            query_roles = tuple(f"query_{idx}" for idx in range(num_query_tokens))
    return QFormerProjectorConfig(
        projector_type=projector_type,
        in_dim=int(projector_cfg["in_dim"]),
        llm_dim=int(projector_cfg["llm_dim"]),
        num_query_tokens=num_query_tokens,
        query_roles=query_roles,
        qformer_hidden_dim=int(projector_cfg.get("qformer_hidden_dim", 768)),
        qformer_num_layers=int(projector_cfg.get("qformer_num_layers", 6)),
        qformer_num_heads=int(projector_cfg.get("qformer_num_heads", 12)),
        qformer_intermediate_dim=int(projector_cfg.get("qformer_intermediate_dim", 3072)),
        cross_attention_freq=int(projector_cfg.get("cross_attention_freq", 2)),
        hidden_dropout_prob=float(projector_cfg.get("hidden_dropout_prob", 0.1)),
        attention_probs_dropout_prob=float(projector_cfg.get("attention_probs_dropout_prob", 0.1)),
        initializer_range=float(projector_cfg.get("initializer_range", 0.02)),
        init_from_pretrained=bool(projector_cfg.get("init_from_pretrained", False)),
        pretrained_name_or_path=str(projector_cfg.get("pretrained_name_or_path", "bert-base-uncased")),
        use_part_token_embeddings=bool(projector_cfg.get("use_part_token_embeddings", False)),
        joint_part_roles=tuple(projector_cfg.get("joint_part_roles") or ()),
    )
