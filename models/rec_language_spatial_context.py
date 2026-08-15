"""Language-conditioned spatial context for hierarchical REC reranking."""

import torch
from torch import nn

from models.encoder_decoder_layers import (
    MultiHeadAttentionSpatial,
    calc_pairwise_locs,
)
from models.rec_hierarchical_reranker import (
    QUERY_FEATURE_DIM,
    VARIANT_COUNT,
    HierarchicalQueryVariantReranker,
)


V114_HIDDEN_DIM = 128
V114_DROPOUT = 0.1
V114_HEAD_COUNT = 4
V114_TARGET_TEXT_START = 64
V114_TARGET_TEXT_END = 128
V114_CENTER_START = 128
V114_CENTER_END = 131


class LanguageConditionedSpatialHierarchicalReranker(
        HierarchicalQueryVariantReranker):
    """Use target-language-conditioned 3D relations between candidates."""

    def __init__(self, hidden_dim=V114_HIDDEN_DIM, dropout=V114_DROPOUT):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout)
        if hidden_dim != V114_HIDDEN_DIM:
            raise ValueError("V114 requires hidden_dim=128")
        if (V114_TARGET_TEXT_END - V114_TARGET_TEXT_START != 64
                or V114_CENTER_END - V114_CENTER_START != 3
                or V114_CENTER_END > QUERY_FEATURE_DIM):
            raise RuntimeError("V114 frozen feature coordinates changed")
        self.text_condition = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_context = MultiHeadAttentionSpatial(
            d_model=hidden_dim,
            n_head=V114_HEAD_COUNT,
            dropout=float(dropout),
            spatial_multihead=True,
            spatial_dim=5,
            spatial_attn_fusion="cond",
        )
        self.spatial_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.spatial_ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
            self, query_features, variant_features,
            query_aux_continuous, query_aux_binary,
            variant_aux_continuous, variant_aux_binary,
            query_valid, variant_valid):
        self._validate_inputs(
            query_features,
            variant_features,
            query_aux_continuous,
            query_aux_binary,
            variant_aux_continuous,
            variant_aux_binary,
            query_valid,
            variant_valid,
        )
        query_mask = query_valid.unsqueeze(-1)
        variant_mask = variant_valid.unsqueeze(-1)
        safe_variant_features = torch.where(
            variant_mask, variant_features, torch.zeros_like(variant_features)
        )
        variant_embedding = self.variant_encoder(safe_variant_features)
        variant_embedding = torch.where(
            variant_mask,
            variant_embedding,
            torch.zeros_like(variant_embedding),
        )
        variant_count = variant_valid.sum(dim=2, keepdim=True).clamp_min(1)
        variant_mean = variant_embedding.sum(dim=2) / variant_count.to(
            variant_embedding.dtype
        )
        variant_max = variant_embedding.masked_fill(
            ~variant_mask, -float("inf")
        ).max(dim=2).values
        variant_max = torch.where(
            query_mask, variant_max, torch.zeros_like(variant_max)
        )

        safe_query_features = torch.where(
            query_mask, query_features, torch.zeros_like(query_features)
        )
        safe_query_aux = torch.where(
            query_mask,
            query_aux_continuous,
            torch.zeros_like(query_aux_continuous),
        )
        query_input = torch.cat((
            safe_query_features,
            safe_query_aux,
            query_aux_binary.to(query_features.dtype),
            variant_mean,
            variant_max,
        ), dim=-1)
        query_embedding = self.query_encoder(query_input)
        query_embedding = torch.where(
            query_mask, query_embedding, torch.zeros_like(query_embedding)
        )

        valid_count = query_valid.sum(dim=1, keepdim=True).clamp_min(1).to(
            query_features.dtype
        )
        target_text = safe_query_features[
            :, :, V114_TARGET_TEXT_START:V114_TARGET_TEXT_END
        ].sum(dim=1) / valid_count
        text_condition = self.text_condition(target_text)
        centers = safe_query_features[
            :, :, V114_CENTER_START:V114_CENTER_END
        ]
        pairwise_locs = calc_pairwise_locs(centers)
        contextual, _ = self.spatial_context(
            query_embedding,
            query_embedding,
            query_embedding,
            pairwise_locs,
            key_padding_mask=~query_valid,
            txt_embeds=text_condition,
        )
        contextual = self.spatial_ffn_norm(
            contextual + self.spatial_ffn(contextual)
        )
        contextual = torch.where(
            query_mask, contextual, torch.zeros_like(contextual)
        )
        query_logits = self.query_head(contextual)
        query_logits = torch.where(
            query_mask, query_logits, torch.zeros_like(query_logits)
        )

        safe_variant_aux = torch.where(
            variant_mask,
            variant_aux_continuous,
            torch.zeros_like(variant_aux_continuous),
        )
        expanded_context = contextual.unsqueeze(2).expand(
            -1, -1, VARIANT_COUNT, -1
        )
        variant_input = torch.cat((
            expanded_context,
            variant_embedding,
            safe_variant_aux,
            variant_aux_binary.to(query_features.dtype),
        ), dim=-1)
        variant_logits = self.variant_head(variant_input)
        variant_logits = torch.where(
            variant_mask, variant_logits, torch.zeros_like(variant_logits)
        )
        return {
            "query_logits": query_logits,
            "variant_logits": variant_logits,
            "query_embedding": contextual,
            "variant_embedding": variant_embedding,
        }
