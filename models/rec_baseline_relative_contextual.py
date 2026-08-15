"""Baseline-conditioned contextual relative-effect REC reranker."""

import torch
from torch import nn

from models.rec_hierarchical_reranker import (
    QUERY_COUNT,
    VARIANT_COUNT,
    HierarchicalQueryVariantReranker,
)


V100_HIDDEN_DIM = 128
V100_DROPOUT = 0.1
V100_CLASS_COUNT = 3


class BaselineRelativeContextualReranker(HierarchicalQueryVariantReranker):
    """Predict break/neutral/fix for every candidate versus the baseline."""

    def __init__(self, hidden_dim=V100_HIDDEN_DIM, dropout=V100_DROPOUT):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout)
        if hidden_dim != V100_HIDDEN_DIM:
            raise ValueError("V100 requires hidden_dim=128")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.query_context = nn.TransformerEncoder(layer, num_layers=1)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.relative_head = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2 * V100_CLASS_COUNT),
        )

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
            variant_mask, variant_embedding,
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
            query_mask, query_aux_continuous,
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
        contextual = self.query_context(
            query_embedding, src_key_padding_mask=~query_valid
        )
        contextual = torch.where(
            query_mask, contextual, torch.zeros_like(contextual)
        )

        safe_variant_aux = torch.where(
            variant_mask, variant_aux_continuous,
            torch.zeros_like(variant_aux_continuous),
        )
        expanded_context = contextual.unsqueeze(2).expand(
            -1, -1, VARIANT_COUNT, -1
        )
        candidate_input = torch.cat((
            expanded_context,
            variant_embedding,
            safe_variant_aux,
            variant_aux_binary.to(query_features.dtype),
        ), dim=-1)
        candidate_embedding = self.candidate_encoder(candidate_input)
        candidate_embedding = torch.where(
            variant_mask, candidate_embedding,
            torch.zeros_like(candidate_embedding),
        )

        baseline_mask = variant_aux_binary[..., 1] & variant_valid
        flat_baseline_mask = baseline_mask.reshape(baseline_mask.shape[0], -1)
        if not bool(flat_baseline_mask.sum(dim=1).eq(1).all().item()):
            raise ValueError("V100 requires exactly one baseline candidate")
        baseline_indices = flat_baseline_mask.to(torch.long).argmax(dim=1)
        rows = torch.arange(query_features.shape[0], device=query_features.device)
        flat_candidate = candidate_embedding.reshape(
            query_features.shape[0], QUERY_COUNT * VARIANT_COUNT,
            self.hidden_dim,
        )
        baseline_embedding = flat_candidate[
            rows, baseline_indices
        ].view(-1, 1, 1, self.hidden_dim).expand_as(candidate_embedding)
        candidate_queries = torch.arange(
            QUERY_COUNT, dtype=torch.long, device=query_features.device
        ).view(1, QUERY_COUNT, 1)
        baseline_queries = baseline_indices.div(
            VARIANT_COUNT, rounding_mode="floor"
        ).view(-1, 1, 1)
        same_query = candidate_queries.eq(baseline_queries).expand(
            -1, -1, VARIANT_COUNT
        ).unsqueeze(-1).to(query_features.dtype)
        relative_input = torch.cat((
            candidate_embedding,
            baseline_embedding,
            candidate_embedding - baseline_embedding,
            same_query,
        ), dim=-1)
        relative_logits = self.relative_head(relative_input).reshape(
            query_features.shape[0], QUERY_COUNT, VARIANT_COUNT,
            2, V100_CLASS_COUNT,
        )
        relative_logits = torch.where(
            variant_mask.unsqueeze(-1), relative_logits,
            torch.zeros_like(relative_logits),
        )
        return {
            "relative_logits": relative_logits,
            "query_embedding": contextual,
            "variant_embedding": variant_embedding,
            "candidate_embedding": candidate_embedding,
            "baseline_indices": baseline_indices,
        }


def signed_effects(relative_logits):
    """Return P(fix)-P(break) for both REC thresholds."""
    if (not isinstance(relative_logits, torch.Tensor)
            or relative_logits.dtype != torch.float32
            or relative_logits.dim() != 5
            or tuple(relative_logits.shape[-2:]) != (2, V100_CLASS_COUNT)
            or not bool(torch.isfinite(relative_logits).all().item())):
        raise ValueError("V100 relative logits are malformed")
    probability = relative_logits.softmax(dim=-1)
    return probability[..., 2] - probability[..., 0]


def apply_baseline_relative_policy(
        relative_logits, variant_valid, baseline_indices):
    """Select the best doubly-positive relative effect or retain baseline."""
    if (not isinstance(variant_valid, torch.Tensor)
            or variant_valid.dtype != torch.bool
            or variant_valid.dim() != 3):
        raise ValueError("V100 variant_valid is malformed")
    batch_size = variant_valid.shape[0]
    if tuple(variant_valid.shape[1:]) != (QUERY_COUNT, VARIANT_COUNT):
        raise ValueError("V100 candidate axes changed")
    if (not isinstance(baseline_indices, torch.Tensor)
            or baseline_indices.dtype != torch.long
            or tuple(baseline_indices.shape) != (batch_size,)
            or baseline_indices.device != variant_valid.device):
        raise ValueError("V100 baseline indices are malformed")
    if (bool((baseline_indices < 0).any().item())
            or bool((baseline_indices >= QUERY_COUNT * VARIANT_COUNT).any().item())):
        raise ValueError("V100 baseline indices are out of range")
    effects = signed_effects(relative_logits).reshape(batch_size, -1, 2)
    flat_valid = variant_valid.reshape(batch_size, -1)
    rows = torch.arange(batch_size, device=variant_valid.device)
    if not bool(flat_valid[rows, baseline_indices].all().item()):
        raise ValueError("V100 baseline must be valid")
    effects = effects.clone()
    effects[rows, baseline_indices] = 0.0
    aggregate = 2.0 * effects[..., 0] + effects[..., 1]
    eligible = (
        flat_valid & effects[..., 0].gt(0.0) & effects[..., 1].gt(0.0)
    )
    eligible[rows, baseline_indices] = True
    utility = aggregate.masked_fill(~eligible, -float("inf"))
    utility[rows, baseline_indices] = 0.0
    selected = utility.argmax(dim=1)
    switch_mask = selected.ne(baseline_indices) & utility[
        rows, selected
    ].gt(0.0)
    selected = torch.where(switch_mask, selected, baseline_indices)
    return {
        "selected_indices": selected,
        "baseline_indices": baseline_indices,
        "switch_mask": switch_mask,
        "signed_effects": effects,
        "aggregate_effect": aggregate,
        "eligible": eligible,
        "utility": utility,
    }
