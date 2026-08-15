"""Contextual hierarchy and deterministic Pareto deployment policy."""

import math

import torch
from torch import nn

from models.rec_hierarchical_reranker import (
    VARIANT_COUNT,
    HierarchicalQueryVariantReranker,
    apply_hierarchical_policy,
    monotone_hit_probabilities,
    select_hierarchical_proposal,
)


V99_ARTIFACT_SCHEMA = "rec-pareto-contextual-hierarchical-v1"
V99_ARTIFACT_VERSION = 1
V99_HIDDEN_DIM = 128
V99_DROPOUT = 0.1
V99_AGGREGATE_MARGIN = 0.13312220573425293
V113_ARTIFACT_SCHEMA = (
    "rec-pareto-contextual-meshsp-asymmetric-risk-committee-"
    "full-train-artifact-v1"
)
V113_ARTIFACT_VERSION = 1
V113_MEMBER_COUNT = 3


class ParetoContextualHierarchicalReranker(
        HierarchicalQueryVariantReranker):
    """Compare all query candidates with one masked attention layer."""

    def __init__(self, hidden_dim=V99_HIDDEN_DIM, dropout=V99_DROPOUT):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout)
        if hidden_dim != V99_HIDDEN_DIM:
            raise ValueError("V99 requires hidden_dim=128")
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
        contextual = self.query_context(
            query_embedding,
            src_key_padding_mask=~query_valid,
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


class AsymmetricRiskContextualHierarchyCommittee(nn.Module):
    """Three independently seeded contextual hierarchy members."""

    def __init__(
            self, hidden_dim=V99_HIDDEN_DIM, dropout=V99_DROPOUT,
            member_count=V113_MEMBER_COUNT):
        super().__init__()
        if (isinstance(member_count, bool)
                or not isinstance(member_count, int)
                or member_count != V113_MEMBER_COUNT):
            raise ValueError("V113 requires exactly three committee members")
        self.members = nn.ModuleList([
            ParetoContextualHierarchicalReranker(
                hidden_dim=hidden_dim, dropout=dropout
            )
            for _ in range(member_count)
        ])

    def forward(
            self, query_features, variant_features,
            query_aux_continuous, query_aux_binary,
            variant_aux_continuous, variant_aux_binary,
            query_valid, variant_valid):
        inputs = {
            "query_features": query_features,
            "variant_features": variant_features,
            "query_aux_continuous": query_aux_continuous,
            "query_aux_binary": query_aux_binary,
            "variant_aux_continuous": variant_aux_continuous,
            "variant_aux_binary": variant_aux_binary,
            "query_valid": query_valid,
            "variant_valid": variant_valid,
        }
        outputs = [member(**inputs) for member in self.members]
        return {
            "member_query_logits": torch.stack([
                output["query_logits"] for output in outputs
            ], dim=0),
            "member_variant_logits": torch.stack([
                output["variant_logits"] for output in outputs
            ], dim=0),
        }


def apply_pareto_contextual_policy(
        base_scores, query_logits, variant_logits,
        query_valid, variant_valid, aggregate_margin,
        min_head_gain025=0.0, min_head_gain050=0.0):
    """Promote only proposals predicted to improve both REC thresholds."""
    if (isinstance(aggregate_margin, bool)
            or not isinstance(aggregate_margin, (int, float))):
        raise TypeError("aggregate_margin must be numeric")
    aggregate_margin = float(aggregate_margin)
    if not math.isfinite(aggregate_margin) or aggregate_margin <= 0.0:
        raise ValueError("aggregate_margin must be finite and positive")
    head_gain_floors = {
        "min_head_gain025": min_head_gain025,
        "min_head_gain050": min_head_gain050,
    }
    for name, value in head_gain_floors.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("{} must be numeric".format(name))
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("{} must be finite and nonnegative".format(name))
        head_gain_floors[name] = value
    selected = select_hierarchical_proposal(
        query_logits, variant_logits, query_valid, variant_valid
    )
    probabilities = monotone_hit_probabilities(variant_logits).reshape(
        base_scores.shape[0], -1, 2
    )
    rows = torch.arange(base_scores.shape[0], device=base_scores.device)
    baselines = base_scores.argmax(dim=1)
    proposals = selected["flat_indices"]
    head_gain = (
        probabilities[rows, proposals]
        - probabilities[rows, baselines]
    )
    aggregate_gain = 2.0 * head_gain[:, 0] + head_gain[:, 1]
    pareto_pass = (
        head_gain[:, 0].gt(head_gain_floors["min_head_gain025"])
        & head_gain[:, 1].gt(head_gain_floors["min_head_gain050"])
    )
    gated_gain = torch.where(
        pareto_pass, aggregate_gain, torch.full_like(aggregate_gain, -1.0)
    )
    policy = apply_hierarchical_policy(
        base_scores,
        proposals,
        gated_gain.float(),
        variant_valid,
        aggregate_margin,
    )
    policy.update({
        "proposal_indices": proposals,
        "head_gain": head_gain,
        "aggregate_gain": aggregate_gain,
        "pareto_pass": pareto_pass,
    })
    return policy


def _validated_v113_policy_value(name, value, *, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be numeric".format(name))
    value = float(value)
    if (not math.isfinite(value)
            or value < 0.0
            or (strictly_positive and value <= 0.0)):
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError("{} must be finite and {}".format(name, qualifier))
    return value


def apply_asymmetric_risk_contextual_policy(
        base_scores, member_query_logits, member_variant_logits,
        query_valid, variant_valid, aggregate_lcb_margin,
        min_head_lcb025=0.0, min_head_lcb050=0.0,
        risk_lambda025=0.0, risk_lambda050=0.0):
    """Gate the seed-0 proposal with asymmetric committee-risk LCBs."""
    policy_values = {
        "aggregate_lcb_margin": _validated_v113_policy_value(
            "aggregate_lcb_margin", aggregate_lcb_margin,
            strictly_positive=True,
        ),
        "min_head_lcb025": _validated_v113_policy_value(
            "min_head_lcb025", min_head_lcb025
        ),
        "min_head_lcb050": _validated_v113_policy_value(
            "min_head_lcb050", min_head_lcb050
        ),
        "risk_lambda025": _validated_v113_policy_value(
            "risk_lambda025", risk_lambda025
        ),
        "risk_lambda050": _validated_v113_policy_value(
            "risk_lambda050", risk_lambda050
        ),
    }
    if (not isinstance(base_scores, torch.Tensor)
            or base_scores.dtype != torch.float32
            or base_scores.dim() != 2
            or not isinstance(member_query_logits, torch.Tensor)
            or member_query_logits.dtype != torch.float32
            or member_query_logits.dim() != 4
            or member_query_logits.shape[0] != V113_MEMBER_COUNT
            or not isinstance(member_variant_logits, torch.Tensor)
            or member_variant_logits.dtype != torch.float32
            or member_variant_logits.dim() != 5
            or member_variant_logits.shape[0] != V113_MEMBER_COUNT
            or tuple(member_query_logits.shape[1:3])
            != tuple(member_variant_logits.shape[1:3])
            or member_query_logits.shape[-1] != 2
            or member_variant_logits.shape[-2:] != (VARIANT_COUNT, 2)
            or member_query_logits.shape[1] != base_scores.shape[0]
            or member_variant_logits.shape[1] != base_scores.shape[0]
            or query_valid.shape != member_query_logits.shape[1:3]
            or variant_valid.shape != member_variant_logits.shape[1:4]
            or query_valid.dtype != torch.bool
            or variant_valid.dtype != torch.bool):
        raise ValueError("V113 committee policy tensors are malformed")
    tensors = (
        base_scores, member_query_logits, member_variant_logits,
        query_valid, variant_valid,
    )
    if any(value.device != base_scores.device for value in tensors[1:]):
        raise ValueError("V113 committee policy tensors changed device")
    if (not bool(torch.isfinite(
            member_query_logits[query_valid.unsqueeze(0).expand_as(
                member_query_logits[..., 0]
            )]
        ).all().item())
            or not bool(torch.isfinite(
                member_variant_logits[variant_valid.unsqueeze(0).unsqueeze(-1)
                    .expand_as(member_variant_logits)]
            ).all().item())):
        raise ValueError("V113 valid committee logits must be finite")

    member_proposals = torch.stack([
        select_hierarchical_proposal(
            member_query_logits[index], member_variant_logits[index],
            query_valid, variant_valid,
        )["flat_indices"]
        for index in range(V113_MEMBER_COUNT)
    ], dim=0)
    proposals = member_proposals[0]
    baselines = base_scores.argmax(dim=1)
    probabilities = monotone_hit_probabilities(
        member_variant_logits
    ).reshape(V113_MEMBER_COUNT, base_scores.shape[0], -1, 2)
    seed_rows = torch.arange(
        V113_MEMBER_COUNT, device=base_scores.device
    ).view(-1, 1)
    batch_rows = torch.arange(
        base_scores.shape[0], device=base_scores.device
    ).view(1, -1)
    proposal_probability = probabilities[
        seed_rows, batch_rows, proposals.view(1, -1)
    ]
    baseline_probability = probabilities[
        seed_rows, batch_rows, baselines.view(1, -1)
    ]
    member_head_gain = proposal_probability - baseline_probability
    anchor_head_gain = member_head_gain[0]
    head_risk = (
        (member_head_gain - anchor_head_gain.unsqueeze(0)).square()
        .mean(dim=0).sqrt()
    )
    lambdas = torch.tensor([
        policy_values["risk_lambda025"],
        policy_values["risk_lambda050"],
    ], dtype=torch.float32, device=base_scores.device)
    head_lcb = anchor_head_gain - lambdas * head_risk
    aggregate_lcb = 2.0 * head_lcb[:, 0] + head_lcb[:, 1]
    pareto_pass = (
        head_lcb[:, 0].gt(policy_values["min_head_lcb025"])
        & head_lcb[:, 1].gt(policy_values["min_head_lcb050"])
    )
    gated_lcb = torch.where(
        pareto_pass, aggregate_lcb,
        torch.full_like(aggregate_lcb, -1.0),
    )
    policy = apply_hierarchical_policy(
        base_scores,
        proposals,
        gated_lcb.float(),
        variant_valid,
        policy_values["aggregate_lcb_margin"],
    )
    policy.update({
        "proposal_indices": proposals,
        "member_proposal_indices": member_proposals,
        "anchor_agreement": member_proposals.eq(
            proposals.unsqueeze(0)
        ).float().mean(dim=0),
        "member_head_gain": member_head_gain,
        "anchor_head_gain": anchor_head_gain,
        "head_risk": head_risk,
        "head_lcb": head_lcb,
        "aggregate_lcb": aggregate_lcb,
        "pareto_pass": pareto_pass,
    })
    return policy
