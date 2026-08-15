"""Bounded dual semantic-spatial adapter over a frozen V99 anchor."""

import torch
from torch import nn

from models.encoder_decoder_layers import (
    MultiHeadAttentionSpatial,
    calc_pairwise_locs,
)
from models.rec_hierarchical_reranker import (
    HierarchicalQueryVariantReranker,
    VARIANT_COUNT,
)
from models.rec_hyperspherical_semantic_adapter import (
    V130_QUERY_FEATURE_DIM,
    V130_SEMANTIC_INPUT_DIM,
    recover_hyperspherical_semantic_features,
)


V131_HIDDEN_DIM = 128
V131_DROPOUT = 0.1
V131_HEAD_COUNT = 4
V131_RESIDUAL_SCALE = 0.25
V131_TARGET_TEXT_START = 64
V131_TARGET_TEXT_END = 128
V131_CENTER_START = 128
V131_CENTER_END = 131


def _zero_linear(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


def _normalization_tensors(statistics):
    if not isinstance(statistics, dict):
        raise TypeError("V131 normalization statistics must be a mapping")
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
        raise ValueError("V131 query-feature normalization is invalid")
    return mean.detach().clone(), std.detach().clone()


class AnchoredDualSemanticSpatialResidualAdapter(nn.Module):
    """Fuse V115 center relations and V130 unit-sphere semantics."""

    def __init__(
            self, anchor, normalization_statistics,
            hidden_dim=V131_HIDDEN_DIM, dropout=V131_DROPOUT,
            residual_scale=V131_RESIDUAL_SCALE):
        super().__init__()
        if (not isinstance(anchor, HierarchicalQueryVariantReranker)
                or not isinstance(
                    getattr(anchor, "query_context", None),
                    nn.TransformerEncoder,
                )
                or len(anchor.query_context.layers) != 1):
            raise TypeError(
                "V131 anchor must be the one-layer contextual hierarchy "
                "returned by the V99 factory"
            )
        if hidden_dim != V131_HIDDEN_DIM or anchor.hidden_dim != hidden_dim:
            raise ValueError("V131 requires hidden_dim=128")
        if float(residual_scale) != V131_RESIDUAL_SCALE:
            raise ValueError("V131 residual scale is frozen at 0.25")
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.residual_scale = float(residual_scale)
        self.anchor = anchor.eval().requires_grad_(False)
        mean, std = _normalization_tensors(normalization_statistics)
        self.register_buffer("query_feature_mean", mean, persistent=True)
        self.register_buffer("query_feature_std", std, persistent=True)

        self.spatial_text_condition = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_context = MultiHeadAttentionSpatial(
            d_model=hidden_dim,
            n_head=V131_HEAD_COUNT,
            dropout=float(dropout),
            spatial_multihead=True,
            spatial_dim=5,
            spatial_attn_fusion="cond",
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(V130_SEMANTIC_INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        semantic_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=V131_HEAD_COUNT,
            dim_feedforward=2 * hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.semantic_context = nn.TransformerEncoder(
            semantic_layer, num_layers=1
        )
        self.semantic_text_condition = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dual_reliability_gate = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2),
        )
        _zero_linear(self.dual_reliability_gate[-1])
        nn.init.constant_(self.dual_reliability_gate[-1].bias, -2.0)
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
        safe_query_features = torch.where(
            query_mask, query_features, torch.zeros_like(query_features)
        )
        valid_count = query_valid.sum(dim=1, keepdim=True).clamp_min(1).to(
            query_features.dtype
        )
        anchor_context = anchor_output["query_embedding"].detach()

        normalized_text = safe_query_features[
            :, :, V131_TARGET_TEXT_START:V131_TARGET_TEXT_END
        ].sum(dim=1) / valid_count
        spatial_text = self.spatial_text_condition(normalized_text)
        centers = safe_query_features[
            :, :, V131_CENTER_START:V131_CENTER_END
        ]
        pairwise_locs = calc_pairwise_locs(centers)
        spatial_context, _ = self.spatial_context(
            anchor_context,
            anchor_context,
            anchor_context,
            pairwise_locs,
            key_padding_mask=~query_valid,
            txt_embeds=spatial_text,
        )
        spatial_context = torch.where(
            query_mask, spatial_context, torch.zeros_like(spatial_context)
        )

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
        raw_text = recovered["target_text"].sum(dim=1) / valid_count
        semantic_text = self.semantic_text_condition(raw_text)

        spatial_delta = spatial_context - anchor_context
        semantic_delta = semantic_context - anchor_context
        gate_input = torch.cat((
            anchor_context,
            spatial_delta,
            semantic_delta,
            spatial_text.unsqueeze(1).expand_as(anchor_context),
            semantic_text.unsqueeze(1).expand_as(anchor_context),
        ), dim=-1)
        gates = torch.sigmoid(self.dual_reliability_gate(gate_input))
        fused_context = self.fusion_norm(
            anchor_context
            + gates[..., 0:1] * spatial_delta
            + gates[..., 1:2] * semantic_delta
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
            "dual_gates": gates,
            "spatial_context": spatial_context,
            "semantic_context": semantic_context,
        }
