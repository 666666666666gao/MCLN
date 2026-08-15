"""Bounded hyperspherical query-text adapter over a frozen V99 anchor."""

import torch
from torch import nn
import torch.nn.functional as F

from models.rec_hierarchical_reranker import (
    HierarchicalQueryVariantReranker,
    VARIANT_COUNT,
)


V130_HIDDEN_DIM = 128
V130_DROPOUT = 0.1
V130_HEAD_COUNT = 4
V130_RESIDUAL_SCALE = 0.25
V130_QUERY_FEATURE_DIM = 152
V130_QUERY_PROJ_START = 0
V130_QUERY_PROJ_END = 64
V130_TARGET_TEXT_START = 64
V130_TARGET_TEXT_END = 128
V130_SEMANTIC_EVIDENCE_START = 134
V130_SEMANTIC_EVIDENCE_END = 143
V130_SEMANTIC_INPUT_DIM = 266


def _zero_linear(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


def _normalization_tensors(statistics):
    if not isinstance(statistics, dict):
        raise TypeError("V130 normalization statistics must be a mapping")
    group = statistics.get("groups", {}).get("query_features", {})
    mean = group.get("mean")
    std = group.get("std")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or mean.dtype != torch.float32
            or std.dtype != torch.float32
            or mean.shape != (V130_QUERY_FEATURE_DIM,)
            or std.shape != (V130_QUERY_FEATURE_DIM,)
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(std).all().item())
            or not bool(std.gt(0.0).all().item())):
        raise ValueError("V130 query-feature normalization is invalid")
    return mean.detach().clone(), std.detach().clone()


def recover_hyperspherical_semantic_features(
        normalized_query_features, query_valid, mean, std):
    """Invert fold normalization and expose unit-sphere matching features."""
    if (normalized_query_features.dim() != 3
            or normalized_query_features.shape[-1]
            != V130_QUERY_FEATURE_DIM):
        raise ValueError("normalized query features must have shape [B,Q,152]")
    if (query_valid.dtype != torch.bool
            or query_valid.shape != normalized_query_features.shape[:2]):
        raise ValueError("query_valid must be bool with shape [B,Q]")
    if (mean.shape != (V130_QUERY_FEATURE_DIM,)
            or std.shape != (V130_QUERY_FEATURE_DIM,)):
        raise ValueError("V130 mean/std must have shape [152]")
    raw = normalized_query_features * std + mean
    raw = torch.where(
        query_valid.unsqueeze(-1), raw, torch.zeros_like(raw)
    )
    query_projection = F.normalize(
        raw[:, :, V130_QUERY_PROJ_START:V130_QUERY_PROJ_END],
        p=2,
        dim=-1,
        eps=1e-6,
    )
    target_text = F.normalize(
        raw[:, :, V130_TARGET_TEXT_START:V130_TARGET_TEXT_END],
        p=2,
        dim=-1,
        eps=1e-6,
    )
    product = query_projection * target_text
    absolute_difference = (query_projection - target_text).abs()
    cosine = product.sum(dim=-1, keepdim=True)
    semantic_evidence = raw[
        :, :, V130_SEMANTIC_EVIDENCE_START:V130_SEMANTIC_EVIDENCE_END
    ]
    features = torch.cat((
        query_projection,
        target_text,
        product,
        absolute_difference,
        semantic_evidence,
        cosine,
    ), dim=-1)
    if features.shape[-1] != V130_SEMANTIC_INPUT_DIM:
        raise RuntimeError("V130 semantic interaction dimension changed")
    features = torch.where(
        query_valid.unsqueeze(-1), features, torch.zeros_like(features)
    )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("V130 semantic interaction features must be finite")
    return {
        "features": features,
        "query_projection": query_projection,
        "target_text": target_text,
        "cosine": cosine,
        "raw_query_features": raw,
    }


class AnchoredHypersphericalSemanticResidualAdapter(nn.Module):
    """Use explicit unit-sphere query-text interactions for bounded repair."""

    def __init__(
            self, anchor, normalization_statistics,
            hidden_dim=V130_HIDDEN_DIM, dropout=V130_DROPOUT,
            residual_scale=V130_RESIDUAL_SCALE):
        super().__init__()
        if (not isinstance(anchor, HierarchicalQueryVariantReranker)
                or not isinstance(
                    getattr(anchor, "query_context", None),
                    nn.TransformerEncoder,
                )
                or len(anchor.query_context.layers) != 1):
            raise TypeError(
                "V130 anchor must be the one-layer contextual hierarchy "
                "returned by the V99 factory"
            )
        if hidden_dim != V130_HIDDEN_DIM or anchor.hidden_dim != hidden_dim:
            raise ValueError("V130 requires hidden_dim=128")
        if float(residual_scale) != V130_RESIDUAL_SCALE:
            raise ValueError("V130 residual scale is frozen at 0.25")
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.residual_scale = float(residual_scale)
        self.anchor = anchor.eval().requires_grad_(False)
        mean, std = _normalization_tensors(normalization_statistics)
        self.register_buffer("query_feature_mean", mean, persistent=True)
        self.register_buffer("query_feature_std", std, persistent=True)

        self.semantic_encoder = nn.Sequential(
            nn.Linear(V130_SEMANTIC_INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=V130_HEAD_COUNT,
            dim_feedforward=2 * hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.semantic_context = nn.TransformerEncoder(layer, num_layers=1)
        self.text_condition = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
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
        recovered = recover_hyperspherical_semantic_features(
            query_features,
            query_valid,
            self.query_feature_mean.to(query_features),
            self.query_feature_std.to(query_features),
        )
        semantic_embedding = self.semantic_encoder(recovered["features"])
        semantic_embedding = torch.where(
            query_mask, semantic_embedding, torch.zeros_like(semantic_embedding)
        )
        semantic_context = self.semantic_context(
            semantic_embedding,
            src_key_padding_mask=~query_valid,
        )
        semantic_context = torch.where(
            query_mask, semantic_context, torch.zeros_like(semantic_context)
        )
        valid_count = query_valid.sum(dim=1, keepdim=True).clamp_min(1).to(
            query_features.dtype
        )
        pooled_text = recovered["target_text"].sum(dim=1) / valid_count
        text_context = self.text_condition(pooled_text)
        expanded_text = text_context.unsqueeze(1).expand_as(semantic_context)
        anchor_context = anchor_output["query_embedding"].detach()
        gate_input = torch.cat((
            anchor_context,
            semantic_context,
            (semantic_context - anchor_context).abs(),
            expanded_text,
        ), dim=-1)
        gate = torch.sigmoid(self.reliability_gate(gate_input))
        fused_context = self.fusion_norm(
            anchor_context + gate * semantic_context
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
            "semantic_gate": gate,
            "semantic_context": semantic_context,
            "hyperspherical_cosine": recovered["cosine"],
        }
