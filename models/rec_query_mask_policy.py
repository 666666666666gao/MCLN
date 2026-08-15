"""Dataset-agnostic, mask-only policy over a frozen REC parent query."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.rec_joint_box_mask import (
    LEGACY_MASK_POLICY_INDEX,
    MASK_POLICY_COUNT,
)


MASK_POLICY_FEATURE_DIM = 52
GEOMETRY_FEATURE_DIM = 179
QUERY_COUNT = 16
VARIANT_COUNT = 7


def monotone_mask_hit_probabilities(logits):
    """Map two logits to nested P(IoU>.25), P(IoU>.50)."""
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.shape[-1] != 2
            or not bool(torch.isfinite(logits).all().item())):
        raise ValueError("mask hit logits must be finite with final dimension 2")
    hit025 = logits[..., 0].sigmoid()
    hit050 = hit025 * logits[..., 1].sigmoid()
    return torch.stack((hit025, hit050), dim=-1)


class QueryMaskPolicyPostprocessor(nn.Module):
    """Predict 15 source/threshold outcomes without changing REC identity."""

    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("V102 requires hidden_dim=128")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.variant_encoder = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + MASK_POLICY_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
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
        self.iou_head = nn.Linear(hidden_dim, MASK_POLICY_COUNT)
        self.hit_head = nn.Linear(hidden_dim, MASK_POLICY_COUNT * 2)

    def forward(self, geometry_features, mask_features, variant_valid):
        if (not isinstance(geometry_features, torch.Tensor)
                or geometry_features.dtype != torch.float32
                or geometry_features.dim() != 4
                or tuple(geometry_features.shape[1:]) != (
                    QUERY_COUNT, VARIANT_COUNT, GEOMETRY_FEATURE_DIM)):
            raise ValueError("geometry_features must have shape [B,16,7,179]")
        batch_size = geometry_features.shape[0]
        if (not isinstance(mask_features, torch.Tensor)
                or mask_features.dtype != torch.float32
                or tuple(mask_features.shape) != (
                    batch_size, QUERY_COUNT, MASK_POLICY_FEATURE_DIM)):
            raise ValueError("mask_features must have shape [B,16,52]")
        if (not isinstance(variant_valid, torch.Tensor)
                or variant_valid.dtype != torch.bool
                or tuple(variant_valid.shape) != (
                    batch_size, QUERY_COUNT, VARIANT_COUNT)):
            raise ValueError("variant_valid must have shape [B,16,7]")
        if ({geometry_features.device, mask_features.device,
             variant_valid.device} != {geometry_features.device}):
            raise ValueError("V102 inputs must share one device")
        if (not bool(variant_valid.reshape(batch_size, -1).any(dim=1).all())
                or not bool(torch.isfinite(geometry_features).all())
                or not bool(torch.isfinite(mask_features).all())):
            raise ValueError("V102 inputs must be finite with valid candidates")
        query_valid = variant_valid.any(dim=2)
        variant_mask = variant_valid.unsqueeze(-1)
        local = self.variant_encoder(torch.where(
            variant_mask, geometry_features, torch.zeros_like(geometry_features)
        ))
        local = torch.where(variant_mask, local, torch.zeros_like(local))
        count = variant_valid.sum(dim=2, keepdim=True).clamp_min(1)
        mean = local.sum(dim=2) / count.to(local.dtype)
        maximum = local.masked_fill(~variant_mask, -float("inf")).max(2).values
        maximum = torch.where(
            query_valid.unsqueeze(-1), maximum, torch.zeros_like(maximum)
        )
        query_input = torch.cat((mean, maximum, mask_features), dim=-1)
        encoded = self.query_encoder(query_input)
        encoded = torch.where(
            query_valid.unsqueeze(-1), encoded, torch.zeros_like(encoded)
        )
        contextual = self.query_context(
            encoded, src_key_padding_mask=~query_valid
        )
        contextual = torch.where(
            query_valid.unsqueeze(-1), contextual, torch.zeros_like(contextual)
        )
        iou = self.iou_head(contextual).sigmoid()
        hit_logits = self.hit_head(contextual).reshape(
            batch_size, QUERY_COUNT, MASK_POLICY_COUNT, 2
        )
        hits = monotone_mask_hit_probabilities(hit_logits)
        mask = query_valid.unsqueeze(-1)
        return {
            "iou": torch.where(mask, iou, torch.zeros_like(iou)),
            "hit_logits": torch.where(
                mask.unsqueeze(-1), hit_logits, torch.zeros_like(hit_logits)
            ),
            "hit_probabilities": torch.where(
                mask.unsqueeze(-1), hits, torch.zeros_like(hits)
            ),
            "query_valid": query_valid,
        }


def select_mask_only_policy(
        outputs, selected_parent_positions, aggregate_margin=0.02):
    """Select a mask policy while preserving the supplied REC parent query."""
    if not isinstance(outputs, dict):
        raise ValueError("V102 outputs must be an object")
    iou = outputs.get("iou")
    hits = outputs.get("hit_probabilities")
    valid = outputs.get("query_valid")
    if (not isinstance(iou, torch.Tensor) or iou.dim() != 3
            or iou.shape[-1] != MASK_POLICY_COUNT
            or not isinstance(hits, torch.Tensor)
            or hits.shape != iou.shape + (2,)
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool or valid.shape != iou.shape[:2]
            or iou.device != hits.device or valid.device != iou.device
            or not bool(torch.isfinite(iou).all())
            or not bool(torch.isfinite(hits).all())):
        raise ValueError("V102 prediction tensors are malformed")
    selected_parent_positions = torch.as_tensor(
        selected_parent_positions, device=iou.device
    )
    if (selected_parent_positions.dtype != torch.long
            or selected_parent_positions.shape != (iou.shape[0],)
            or bool(((selected_parent_positions < 0)
                     | (selected_parent_positions >= iou.shape[1])).any())):
        raise ValueError("selected parent positions are malformed")
    if (isinstance(aggregate_margin, bool)
            or not isinstance(aggregate_margin, (int, float))
            or not math.isfinite(float(aggregate_margin))
            or float(aggregate_margin) <= 0.0):
        raise ValueError("aggregate_margin must be finite and positive")
    rows = torch.arange(iou.shape[0], device=iou.device)
    if not bool(valid[rows, selected_parent_positions].all()):
        raise ValueError("selected REC parent query is invalid")
    selected_iou = iou[rows, selected_parent_positions]
    selected_hits = hits[rows, selected_parent_positions]
    utility = selected_iou + selected_hits[..., 0] + 2.0 * selected_hits[..., 1]
    maximum = utility.max(dim=-1, keepdim=True).values
    ties = utility.eq(maximum)
    indices = torch.arange(MASK_POLICY_COUNT, device=iou.device).expand_as(utility)
    proposal = indices.masked_fill(~ties, MASK_POLICY_COUNT).min(dim=-1).values
    baseline = torch.full_like(proposal, LEGACY_MASK_POLICY_INDEX)
    proposal_iou = selected_iou[rows, proposal]
    baseline_iou = selected_iou[rows, baseline]
    proposal_hits = selected_hits[rows, proposal]
    baseline_hits = selected_hits[rows, baseline]
    delta_iou = proposal_iou - baseline_iou
    delta_hits = proposal_hits - baseline_hits
    aggregate_gain = delta_iou + delta_hits[:, 0] + 2.0 * delta_hits[:, 1]
    accepted = (
        delta_iou.gt(0.0)
        & delta_hits[:, 0].gt(0.0)
        & delta_hits[:, 1].gt(0.0)
        & aggregate_gain.ge(float(aggregate_margin))
    )
    selected = torch.where(accepted, proposal, baseline)
    return {
        "selected_policy_indices": selected,
        "proposal_policy_indices": proposal,
        "baseline_policy_indices": baseline,
        "accepted": accepted,
        "aggregate_gain": aggregate_gain,
        "delta_iou": delta_iou,
        "delta_hits": delta_hits,
        "selected_parent_positions": selected_parent_positions.clone(),
    }


def compute_mask_policy_loss(outputs, policy_ious, query_valid,
                             target_temperature=0.25):
    """Dense threshold-aligned V102 loss over every valid query/policy."""
    iou = outputs["iou"]
    hit_logits = outputs["hit_logits"]
    hit_probabilities = outputs["hit_probabilities"]
    policy_ious = torch.as_tensor(
        policy_ious, device=iou.device, dtype=iou.dtype
    )
    if policy_ious.shape != iou.shape:
        raise ValueError("policy_ious must match [B,16,15]")
    if (not isinstance(query_valid, torch.Tensor)
            or query_valid.dtype != torch.bool
            or query_valid.shape != iou.shape[:2]
            or query_valid.device != iou.device):
        raise ValueError("query_valid must match [B,16]")
    if not 0.0 < float(target_temperature):
        raise ValueError("target_temperature must be positive")
    targets = torch.stack((
        policy_ious.gt(0.25), policy_ious.gt(0.50)
    ), dim=-1).to(iou.dtype)
    expanded_valid = query_valid.unsqueeze(-1).expand_as(iou)
    iou_loss = F.smooth_l1_loss(iou, policy_ious, reduction="none")
    hit_loss = F.binary_cross_entropy_with_logits(
        hit_logits[..., 0], targets[..., 0], reduction="none"
    ) + F.binary_cross_entropy(
        hit_probabilities[..., 1].clamp(1e-6, 1.0 - 1e-6),
        targets[..., 1], reduction="none",
    )
    target_utility = policy_ious + targets[..., 0] + 2.0 * targets[..., 1]
    predicted_utility = iou + hit_probabilities[..., 0] + 2.0 * hit_probabilities[..., 1]
    target_distribution = (target_utility / float(target_temperature)).softmax(-1)
    listwise = -(target_distribution * predicted_utility.log_softmax(-1)).sum(-1)
    predicted_distribution = predicted_utility.softmax(-1)
    expected_iou = (predicted_distribution * policy_ious).sum(-1)
    regret = (policy_ious.max(-1).values - expected_iou).clamp_min(0.0)
    denominator = expanded_valid.sum().clamp_min(1).to(iou.dtype)
    query_denominator = query_valid.sum().clamp_min(1).to(iou.dtype)
    components = {
        "iou": (iou_loss * expanded_valid).sum() / denominator,
        "hit": (hit_loss * expanded_valid).sum() / denominator,
        "listwise": (listwise * query_valid).sum() / query_denominator,
        "regret": (regret * query_valid).sum() / query_denominator,
    }
    total = (
        components["iou"] + components["hit"]
        + 0.5 * components["listwise"] + 0.5 * components["regret"]
    )
    return total, components
