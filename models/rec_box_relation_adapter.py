"""Bounded text-conditioned box-relation adapter over a frozen V99 anchor."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.rec_hierarchical_reranker import (
    HierarchicalQueryVariantReranker,
    VARIANT_COUNT,
)


V129_HIDDEN_DIM = 128
V129_DROPOUT = 0.1
V129_HEAD_COUNT = 4
V129_RESIDUAL_SCALE = 0.25
V129_QUERY_PROJ_START = 0
V129_QUERY_PROJ_END = 64
V129_TARGET_TEXT_START = 64
V129_TARGET_TEXT_END = 128
V129_CENTER_START = 128
V129_CENTER_END = 131
V129_SIZE_START = 131
V129_SIZE_END = 134
V129_TARGET_COSINE_INDEX = 151
V129_EDGE_DIM = 19


def _zero_linear(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


def build_directed_box_relation_features(query_features, query_valid):
    """Build general directed semantic/box relations for every query pair."""
    if query_features.dim() != 3 or query_features.shape[-1] != 152:
        raise ValueError("query_features must have shape [B,Q,152]")
    if (query_valid.dtype != torch.bool
            or query_valid.shape != query_features.shape[:2]):
        raise ValueError("query_valid must be bool with shape [B,Q]")
    valid = query_valid.unsqueeze(-1)
    safe = torch.where(valid, query_features, torch.zeros_like(query_features))
    query_proj = F.normalize(
        safe[:, :, V129_QUERY_PROJ_START:V129_QUERY_PROJ_END],
        p=2,
        dim=-1,
        eps=1e-6,
    )
    centers = safe[:, :, V129_CENTER_START:V129_CENTER_END]
    sizes = safe[:, :, V129_SIZE_START:V129_SIZE_END].clamp_min(1e-4)
    target_cosine = safe[:, :, V129_TARGET_COSINE_INDEX]

    center_i = centers.unsqueeze(2)
    center_j = centers.unsqueeze(1)
    size_i = sizes.unsqueeze(2)
    size_j = sizes.unsqueeze(1)
    delta = center_j - center_i
    abs_delta = delta.abs()
    mean_size = (0.5 * (size_i + size_j)).clamp_min(1e-4)
    scale_delta = delta / mean_size
    log_size_ratio = torch.log(size_j / size_i)
    distance_3d = delta.square().sum(dim=-1, keepdim=True).sqrt()
    distance_2d = delta[..., :2].square().sum(dim=-1, keepdim=True).sqrt()
    semantic_cosine = (
        query_proj.unsqueeze(2) * query_proj.unsqueeze(1)
    ).sum(dim=-1, keepdim=True)
    target_i = target_cosine.unsqueeze(2).unsqueeze(-1).expand(
        -1, -1, query_features.shape[1], -1
    )
    target_j = target_cosine.unsqueeze(1).unsqueeze(-1).expand(
        -1, query_features.shape[1], -1, -1
    )

    minimum_i = center_i - 0.5 * size_i
    maximum_i = center_i + 0.5 * size_i
    minimum_j = center_j - 0.5 * size_j
    maximum_j = center_j + 0.5 * size_j
    intersection_size = (
        torch.minimum(maximum_i, maximum_j)
        - torch.maximum(minimum_i, minimum_j)
    ).clamp_min(0.0)
    intersection = intersection_size.prod(dim=-1, keepdim=True)
    volume_i = size_i.prod(dim=-1, keepdim=True)
    volume_j = size_j.prod(dim=-1, keepdim=True)
    box_iou = intersection / (volume_i + volume_j - intersection).clamp_min(
        1e-6
    )
    relations = torch.cat((
        delta,
        abs_delta,
        scale_delta,
        log_size_ratio,
        distance_3d,
        distance_2d,
        semantic_cosine,
        target_i,
        target_j,
        target_j - target_i,
        box_iou,
    ), dim=-1)
    if relations.shape[-1] != V129_EDGE_DIM:
        raise RuntimeError("V129 directed relation dimension changed")
    pair_valid = query_valid.unsqueeze(2) & query_valid.unsqueeze(1)
    relations = torch.where(
        pair_valid.unsqueeze(-1), relations, torch.zeros_like(relations)
    )
    if not bool(torch.isfinite(relations).all().item()):
        raise ValueError("V129 directed relation features must be finite")
    return relations, pair_valid


class AnchoredTextConditionedBoxRelationAdapter(nn.Module):
    """Refine frozen contextual queries with directed semantic/box messages."""

    def __init__(
            self, anchor, hidden_dim=V129_HIDDEN_DIM,
            dropout=V129_DROPOUT, residual_scale=V129_RESIDUAL_SCALE):
        super().__init__()
        if (not isinstance(anchor, HierarchicalQueryVariantReranker)
                or not isinstance(
                    getattr(anchor, "query_context", None),
                    nn.TransformerEncoder,
                )
                or len(anchor.query_context.layers) != 1):
            raise TypeError(
                "V129 anchor must be the one-layer contextual hierarchy "
                "returned by the V99 factory"
            )
        if hidden_dim != V129_HIDDEN_DIM or anchor.hidden_dim != hidden_dim:
            raise ValueError("V129 requires hidden_dim=128")
        if float(residual_scale) != V129_RESIDUAL_SCALE:
            raise ValueError("V129 residual scale is frozen at 0.25")
        if hidden_dim % V129_HEAD_COUNT:
            raise ValueError("V129 hidden dimension must divide four heads")
        self.hidden_dim = hidden_dim
        self.head_count = V129_HEAD_COUNT
        self.head_dim = hidden_dim // self.head_count
        self.dropout = float(dropout)
        self.residual_scale = float(residual_scale)
        self.anchor = anchor.eval().requires_grad_(False)

        self.text_condition = nn.Sequential(
            nn.Linear(64, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(V129_EDGE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_score = nn.Linear(hidden_dim, self.head_count, bias=False)
        self.edge_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.reliability_gate = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        _zero_linear(self.reliability_gate[-1])
        nn.init.constant_(self.reliability_gate[-1].bias, -2.0)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.query_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.variant_delta_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2),
        )
        _zero_linear(self.query_delta_head[-1])
        _zero_linear(self.variant_delta_head[-1])

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def _relation_context(
            self, anchor_context, query_features, query_valid, text):
        batch_size, query_count, _ = anchor_context.shape
        relations, pair_valid = build_directed_box_relation_features(
            query_features, query_valid
        )
        identity = torch.eye(
            query_count, dtype=torch.bool, device=query_valid.device
        ).unsqueeze(0)
        relation_valid = pair_valid & ~identity
        encoded_edge = self.edge_encoder(relations)
        gamma, beta = self.text_condition(text).chunk(2, dim=-1)
        encoded_edge = (
            encoded_edge * (1.0 + 0.5 * gamma.tanh().unsqueeze(1).unsqueeze(1))
            + 0.5 * beta.tanh().unsqueeze(1).unsqueeze(1)
        )

        def split_heads(value):
            return value.view(
                batch_size, query_count, self.head_count, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.query_projection(anchor_context))
        key = split_heads(self.key_projection(anchor_context))
        value = split_heads(self.value_projection(anchor_context))
        content_score = torch.einsum("bhid,bhjd->bhij", query, key)
        content_score = content_score / math.sqrt(float(self.head_dim))
        relation_score = self.edge_score(encoded_edge).permute(0, 3, 1, 2)
        score = (content_score + relation_score).masked_fill(
            ~relation_valid.unsqueeze(1), -1e4
        )
        attention = score.softmax(dim=-1)
        attention = attention * relation_valid.unsqueeze(1).to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        edge_value = self.edge_value(encoded_edge).view(
            batch_size, query_count, query_count,
            self.head_count, self.head_dim,
        ).permute(0, 3, 1, 2, 4)
        message = value.unsqueeze(2) + edge_value
        context = (attention.unsqueeze(-1) * message).sum(dim=3)
        context = context.transpose(1, 2).reshape(
            batch_size, query_count, self.hidden_dim
        )
        context = self.relation_output(context)
        context = torch.where(
            query_valid.unsqueeze(-1), context, torch.zeros_like(context)
        )
        return context, attention

    def forward(
            self, query_features, variant_features,
            query_aux_continuous, query_aux_binary,
            variant_aux_continuous, variant_aux_binary,
            query_valid, variant_valid):
        self.anchor._validate_inputs(
            query_features,
            variant_features,
            query_aux_continuous,
            query_aux_binary,
            variant_aux_continuous,
            variant_aux_binary,
            query_valid,
            variant_valid,
        )
        anchor_inputs = {
            "query_features": query_features,
            "variant_features": variant_features,
            "query_aux_continuous": query_aux_continuous,
            "query_aux_binary": query_aux_binary,
            "variant_aux_continuous": variant_aux_continuous,
            "variant_aux_binary": variant_aux_binary,
            "query_valid": query_valid,
            "variant_valid": variant_valid,
        }
        with torch.no_grad():
            anchor_output = self.anchor(**anchor_inputs)
        query_mask = query_valid.unsqueeze(-1)
        variant_mask = variant_valid.unsqueeze(-1)
        safe_query_features = torch.where(
            query_mask, query_features, torch.zeros_like(query_features)
        )
        valid_count = query_valid.sum(dim=1, keepdim=True).clamp_min(1).to(
            query_features.dtype
        )
        target_text = safe_query_features[
            :, :, V129_TARGET_TEXT_START:V129_TARGET_TEXT_END
        ].sum(dim=1) / valid_count
        anchor_context = anchor_output["query_embedding"].detach()
        relation_context, attention = self._relation_context(
            anchor_context,
            safe_query_features,
            query_valid,
            target_text,
        )
        text_context = self.text_condition(target_text)[..., :self.hidden_dim]
        expanded_text = text_context.unsqueeze(1).expand_as(anchor_context)
        gate_input = torch.cat((
            anchor_context,
            relation_context,
            (relation_context - anchor_context).abs(),
            expanded_text,
        ), dim=-1)
        gate = torch.sigmoid(self.reliability_gate(gate_input))
        fused_context = self.fusion_norm(
            anchor_context + gate * relation_context
        )
        fused_context = torch.where(
            query_mask, fused_context, torch.zeros_like(fused_context)
        )
        query_delta = self.query_delta_head(fused_context).tanh()
        query_logits = anchor_output["query_logits"] + (
            self.residual_scale * query_delta
        )
        query_logits = torch.where(
            query_mask, query_logits, torch.zeros_like(query_logits)
        )

        safe_variant_aux = torch.where(
            variant_mask,
            variant_aux_continuous,
            torch.zeros_like(variant_aux_continuous),
        )
        expanded_context = fused_context.unsqueeze(2).expand(
            -1, -1, VARIANT_COUNT, -1
        )
        variant_delta_input = torch.cat((
            expanded_context,
            anchor_output["variant_embedding"].detach(),
            safe_variant_aux,
            variant_aux_binary.to(query_features.dtype),
        ), dim=-1)
        variant_delta = self.variant_delta_head(variant_delta_input).tanh()
        variant_logits = anchor_output["variant_logits"] + (
            self.residual_scale * variant_delta
        )
        variant_logits = torch.where(
            variant_mask, variant_logits, torch.zeros_like(variant_logits)
        )
        return {
            "query_logits": query_logits,
            "variant_logits": variant_logits,
            "query_embedding": fused_context,
            "variant_embedding": anchor_output["variant_embedding"],
            "relation_gate": gate,
            "relation_attention": attention,
        }
