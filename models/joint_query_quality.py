"""Query-consistent box and mask quality reranking for MCLN."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mask_fusion import QUERY_MASK_SOURCE_EVIDENCE_DIM


JOINT_QUERY_GATE_EVIDENCE_NAMES = (
    "candidate_eligible",
    "default_query",
    "selected_query",
    "action_anchor_query",
    "candidate_score_rank",
    "candidate_score_probability",
    "expected_utility_probability",
    "direct_utility_probability",
    "action_margin_probability",
    "box_025_break_probability",
    "box_025_neutral_probability",
    "box_025_fix_probability",
    "box_050_break_probability",
    "box_050_neutral_probability",
    "box_050_fix_probability",
    "mask_025_break_probability",
    "mask_025_neutral_probability",
    "mask_025_fix_probability",
    "mask_050_break_probability",
    "mask_050_neutral_probability",
    "mask_050_fix_probability",
    "decision_fallback_probability",
    "decision_neutral_probability",
    "decision_override_probability",
)
JOINT_QUERY_GATE_EVIDENCE_DIM = len(JOINT_QUERY_GATE_EVIDENCE_NAMES)
SOURCE_DISTRIBUTION_RELIABILITY_NAMES = (
    "standardized_score",
    "query_probability",
    "normalized_entropy",
    "top1_margin",
    "shared_top1_disagreement",
    "shared_js_divergence",
)
SOURCE_DISTRIBUTION_RELIABILITY_DIM = len(
    SOURCE_DISTRIBUTION_RELIABILITY_NAMES
)


def _finite_non_negative(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) < 0.0):
        raise ValueError("{} must be finite and non-negative".format(name))
    return float(value)


def _rank_normalize(scores, valid_mask):
    masked = scores.masked_fill(~valid_mask, float("-inf"))
    order = masked.argsort(dim=1, descending=True)
    ranks = torch.zeros_like(scores)
    values = torch.arange(
        scores.shape[1], device=scores.device, dtype=scores.dtype
    ).unsqueeze(0).expand_as(scores)
    ranks.scatter_(1, order, values)
    denominator = valid_mask.sum(dim=1, keepdim=True).sub(1).clamp(min=1)
    normalized = 1.0 - ranks / denominator.to(scores.dtype)
    return normalized.masked_fill(~valid_mask, 0.0)


def _straight_through_rank_normalize(scores, valid_mask, eps=1e-6):
    """Keep exact query ranks in forward while exposing a smooth gradient."""
    exact = _rank_normalize(scores, valid_mask)
    valid = valid_mask.to(dtype=scores.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    finite = torch.where(valid_mask, scores, torch.zeros_like(scores))
    mean = finite.sum(dim=1, keepdim=True) / count
    centered = torch.where(
        valid_mask, finite - mean, torch.zeros_like(scores)
    )
    variance = centered.square().sum(dim=1, keepdim=True) / count
    scale = variance.clamp(min=float(eps) ** 2).sqrt()
    proxy = torch.sigmoid(centered / scale)
    normalized = proxy + (exact - proxy).detach()
    return normalized.masked_fill(~valid_mask, 0.0)


def _score_standardize(scores, valid_mask):
    """Preserve parent-score confidence without depending on its scale."""
    weights = valid_mask.to(scores.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    safe_scores = scores.masked_fill(~valid_mask, 0.0)
    mean = safe_scores.sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill(~valid_mask, 0.0)
    variance = centered.square().sum(dim=1, keepdim=True) / count
    standardized = centered / variance.clamp(min=1e-6).sqrt()
    return standardized.masked_fill(~valid_mask, 0.0)


def build_joint_query_gate_evidence(gate_outputs, valid_mask):
    """Build target-free query evidence from an existing fallback gate."""
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool or valid_mask.dim() != 2
            or not bool(valid_mask.any(dim=1).all().item())):
        raise ValueError("valid_mask must be bool [B,Q] with a valid query")
    if not isinstance(gate_outputs, dict):
        raise ValueError("gate_outputs must be a dictionary")
    batch_size, query_count = valid_mask.shape
    device = valid_mask.device
    required = (
        "moe_gate_candidate_mask",
        "moe_gate_default_query",
        "moe_gate_selected_query",
        "moe_candidate_scores",
        "moe_gate_expected_utility",
        "moe_gate_direct_utility",
        "moe_gate_action_margin",
        "moe_gate_box_logits",
        "moe_gate_mask_logits",
        "moe_gate_decision_logits",
    )
    missing = [name for name in required if name not in gate_outputs]
    if missing:
        raise ValueError(
            "gate evidence outputs are missing: {}".format(
                ", ".join(missing)
            )
        )

    candidate_mask = gate_outputs["moe_gate_candidate_mask"]
    if (not isinstance(candidate_mask, torch.Tensor)
            or candidate_mask.dtype != torch.bool
            or candidate_mask.shape != valid_mask.shape
            or candidate_mask.device != device
            or bool((candidate_mask & ~valid_mask).any().item())):
        raise ValueError("gate candidate mask must be valid bool [B,Q]")

    def index_indicator(name, fallback=None):
        indices = gate_outputs.get(name, fallback)
        if (not isinstance(indices, torch.Tensor)
                or indices.dtype != torch.long
                or indices.shape != (batch_size,)
                or indices.device != device
                or bool(((indices < 0) | (indices >= query_count)).any().item())):
            raise ValueError("{} must be valid int64 [B]".format(name))
        row = torch.arange(batch_size, device=device)
        if not bool(valid_mask[row, indices].all().item()):
            raise ValueError("{} must identify valid queries".format(name))
        indicator = torch.zeros(
            batch_size, query_count, device=device, dtype=torch.float32
        )
        indicator.scatter_(1, indices.unsqueeze(1), 1.0)
        return indicator

    default_indices = gate_outputs["moe_gate_default_query"]
    default_indicator = index_indicator("moe_gate_default_query")
    selected_indicator = index_indicator("moe_gate_selected_query")
    anchor_indicator = index_indicator(
        "moe_gate_action_anchor_query", fallback=default_indices
    )

    def scalar_query(name):
        value = gate_outputs[name]
        if (not isinstance(value, torch.Tensor)
                or value.shape != valid_mask.shape
                or not value.is_floating_point()
                or value.device != device
                or not bool(torch.isfinite(value).all().item())):
            raise ValueError("{} must be finite floating [B,Q]".format(name))
        return value.float()

    candidate_scores = scalar_query("moe_candidate_scores")
    expected_utility = scalar_query("moe_gate_expected_utility")
    direct_utility = scalar_query("moe_gate_direct_utility")
    action_margin = scalar_query("moe_gate_action_margin")

    def transition_probabilities(name):
        value = gate_outputs[name]
        expected_shape = (batch_size, query_count, 2, 3)
        if (not isinstance(value, torch.Tensor)
                or value.shape != expected_shape
                or not value.is_floating_point()
                or value.device != device
                or not bool(torch.isfinite(value).all().item())):
            raise ValueError(
                "{} must be finite floating [B,Q,2,3]".format(name)
            )
        return value.float().softmax(dim=-1).flatten(start_dim=2)

    decision_logits = gate_outputs["moe_gate_decision_logits"]
    if (not isinstance(decision_logits, torch.Tensor)
            or decision_logits.shape != (batch_size, query_count, 3)
            or not decision_logits.is_floating_point()
            or decision_logits.device != device
            or not bool(torch.isfinite(decision_logits).all().item())):
        raise ValueError(
            "moe_gate_decision_logits must be finite floating [B,Q,3]"
        )

    scalar_evidence = torch.stack((
        candidate_mask.float(),
        default_indicator,
        selected_indicator,
        anchor_indicator,
        _rank_normalize(candidate_scores, valid_mask),
        _score_standardize(candidate_scores, valid_mask).sigmoid(),
        expected_utility.sigmoid(),
        direct_utility.sigmoid(),
        action_margin.sigmoid(),
    ), dim=-1)
    evidence = torch.cat((
        scalar_evidence,
        transition_probabilities("moe_gate_box_logits"),
        transition_probabilities("moe_gate_mask_logits"),
        decision_logits.float().softmax(dim=-1),
    ), dim=-1).masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    if (evidence.shape != (
            batch_size, query_count, JOINT_QUERY_GATE_EVIDENCE_DIM)
            or not bool(torch.isfinite(evidence).all().item())
            or bool(((evidence < 0.0) | (evidence > 1.0)).any().item())):
        raise RuntimeError("joint query gate evidence contract is invalid")
    return evidence


def ordinal_threshold_logits(raw_logits):
    """Return nested threshold logits with P(>0.50) <= P(>0.25)."""
    if (not isinstance(raw_logits, torch.Tensor)
            or raw_logits.dim() != 3 or raw_logits.shape[-1] != 2
            or not raw_logits.is_floating_point()
            or not bool(torch.isfinite(raw_logits).all().item())):
        raise ValueError("raw threshold logits must be finite floating [B,Q,2]")
    probability_025 = raw_logits[..., 0].sigmoid()
    probability_050 = probability_025 * raw_logits[..., 1].sigmoid()
    probabilities = torch.stack((probability_025, probability_050), dim=-1)
    epsilon = torch.finfo(probabilities.dtype).eps
    return torch.logit(probabilities.clamp(min=epsilon, max=1.0 - epsilon))


def joint_query_target_quality(box_ious, mask_ious, mask_weight=0.25):
    """Build a lexicographic box-tier-first quality target.

    The stride of four is larger than the full within-tier box and weighted
    mask range, so mask quality can never promote a lower box tier.
    """
    mask_weight = _finite_non_negative("mask_weight", mask_weight)
    if mask_weight >= 0.8:
        raise ValueError(
            "mask_weight must be below 0.8 to preserve box-tier priority"
        )
    if (not isinstance(box_ious, torch.Tensor)
            or not isinstance(mask_ious, torch.Tensor)
            or box_ious.shape != mask_ious.shape
            or box_ious.dim() != 2):
        raise ValueError("box_ious and mask_ious must align as [B,Q]")
    if (not box_ious.is_floating_point()
            or not mask_ious.is_floating_point()
            or not bool(torch.isfinite(box_ious).all().item())
            or not bool(torch.isfinite(mask_ious).all().item())
            or bool(((box_ious < 0.0) | (box_ious > 1.0)).any().item())
            or bool(((mask_ious < 0.0) | (mask_ious > 1.0)).any().item())):
        raise ValueError("IoU targets must be finite values in [0,1]")
    box_tier = (
        (box_ious > 0.25).to(box_ious.dtype)
        + (box_ious > 0.50).to(box_ious.dtype)
    )
    mask_tier = (
        (mask_ious > 0.25).to(mask_ious.dtype)
        + (mask_ious > 0.50).to(mask_ious.dtype)
    )
    return (
        4.0 * box_tier + box_ious
        + mask_weight * (2.0 * mask_tier + mask_ious)
    )



def smooth_metric_aligned_query_utility(
        box_ious, mask_ious, temperature=0.05, mask_weight=0.25):
    """Build a smooth utility aligned to REC/RES 0.25 and 0.50 metrics."""
    mask_weight = _finite_non_negative("mask_weight", mask_weight)
    if (not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("temperature must be finite and positive")
    if (not isinstance(box_ious, torch.Tensor)
            or not isinstance(mask_ious, torch.Tensor)
            or box_ious.shape != mask_ious.shape
            or box_ious.dim() != 2):
        raise ValueError("box_ious and mask_ious must align as [B,Q]")
    if (not box_ious.is_floating_point()
            or not mask_ious.is_floating_point()
            or not bool(torch.isfinite(box_ious).all().item())
            or not bool(torch.isfinite(mask_ious).all().item())
            or bool(((box_ious < 0.0) | (box_ious > 1.0)).any().item())
            or bool(((mask_ious < 0.0) | (mask_ious > 1.0)).any().item())):
        raise ValueError("IoU targets must be finite values in [0,1]")
    tau = float(temperature)
    box_025 = torch.sigmoid((box_ious - 0.25) / tau)
    box_050 = torch.sigmoid((box_ious - 0.50) / tau)
    mask_025 = torch.sigmoid((mask_ious - 0.25) / tau)
    mask_050 = torch.sigmoid((mask_ious - 0.50) / tau)
    box_utility = box_025 + 2.0 * box_050 + 0.5 * box_ious
    mask_utility = 0.5 * mask_025 + mask_050 + 0.5 * mask_ious
    return box_utility + mask_weight * mask_utility

def predicted_joint_query_quality(
        box_logits, box_iou, mask_logits, mask_iou, mask_weight=0.25,
        metric_aligned=False):
    """Map multitask predictions to a monotonic diagnostic quality score."""
    mask_weight = _finite_non_negative("mask_weight", mask_weight)
    if (not isinstance(box_logits, torch.Tensor) or box_logits.dim() != 3
            or box_logits.shape[-1] != 2
            or not isinstance(mask_logits, torch.Tensor)
            or mask_logits.shape != box_logits.shape
            or not isinstance(box_iou, torch.Tensor)
            or box_iou.shape != box_logits.shape[:2]
            or not isinstance(mask_iou, torch.Tensor)
            or mask_iou.shape != box_iou.shape):
        raise ValueError("quality predictions must align as [B,Q,2]/[B,Q]")
    if not all(bool(torch.isfinite(value).all().item()) for value in (
            box_logits, box_iou, mask_logits, mask_iou)):
        raise ValueError("quality predictions must be finite")
    if not isinstance(metric_aligned, bool):
        raise ValueError("metric_aligned must be boolean")
    if metric_aligned:
        box_quality = (
            box_logits.sigmoid() * box_logits.new_tensor((1.0, 2.0))
        ).sum(-1) + 0.5 * box_iou
        mask_quality = (
            mask_logits.sigmoid() * mask_logits.new_tensor((0.5, 1.0))
        ).sum(-1) + 0.5 * mask_iou
        return (box_quality + mask_weight * mask_quality) / (
            3.5 + 2.0 * mask_weight
        )
    box_quality = (
        box_logits.sigmoid() * box_logits.new_tensor((1.0, 2.0))
    ).sum(-1) + box_iou
    mask_quality = (
        mask_logits.sigmoid() * mask_logits.new_tensor((1.0, 2.0))
    ).sum(-1) + mask_iou
    return (box_quality + mask_weight * mask_quality) / (
        4.0 * (1.0 + mask_weight)
    )

def summarize_joint_query_residual(residual, valid_mask):
    """Return scalar residual magnitude and within-row query variation."""
    if (not isinstance(residual, torch.Tensor) or residual.dim() != 2
            or not residual.is_floating_point()
            or not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != residual.shape
            or valid_mask.device != residual.device
            or not bool(valid_mask.any(dim=1).all().item())):
        raise ValueError(
            "residual and valid_mask must align as floating/bool [B,Q]"
        )
    if not bool(torch.isfinite(residual[valid_mask]).all().item()):
        raise ValueError("valid residual values must be finite")
    with torch.no_grad():
        valid_weights = valid_mask.to(residual.dtype)
        valid_count = valid_weights.sum().clamp(min=1.0)
        abs_values = residual.abs() * valid_weights
        row_count = valid_weights.sum(dim=1).clamp(min=1.0)
        row_mean = (
            (residual * valid_weights).sum(dim=1) / row_count
        )
        centered = residual - row_mean.unsqueeze(1)
        row_variance = (
            centered.square() * valid_weights
        ).sum(dim=1) / row_count
        return {
            "residual_abs_mean": abs_values.sum() / valid_count,
            "residual_abs_max": residual.abs().masked_fill(
                ~valid_mask, 0.0
            ).max(),
            "residual_query_std": row_variance.sqrt().mean(),
        }


class QuerySuperpointMaskRefiner(nn.Module):
    """Low-rank, zero-residual spatial mask correction."""

    def __init__(self, d_model=288, hidden_dim=32, max_delta=2.0,
                 detach_inputs=True):
        super().__init__()
        for name, value in (("d_model", d_model), ("hidden_dim", hidden_dim)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value <= 0):
                raise ValueError("{} must be a positive integer".format(name))
        if (not isinstance(max_delta, (int, float))
                or isinstance(max_delta, bool)
                or not math.isfinite(float(max_delta))
                or float(max_delta) <= 0.0):
            raise ValueError("max_delta must be finite and positive")
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be boolean")
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.detach_inputs = detach_inputs
        self.query_norm = nn.LayerNorm(self.d_model)
        self.superpoint_norm = nn.LayerNorm(self.d_model)
        self.query_projection = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.superpoint_projection = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.query_projection[-1].weight)
        nn.init.zeros_(self.query_projection[-1].bias)

    def forward(self, query_features, superpoint_features, valid_mask=None):
        if (not isinstance(query_features, torch.Tensor)
                or query_features.dim() != 3
                or query_features.shape[-1] != self.d_model
                or not query_features.is_floating_point()
                or not isinstance(superpoint_features, (list, tuple))
                or len(superpoint_features) != query_features.shape[0]):
            raise ValueError(
                "query features and superpoint feature rows must align"
            )
        batch_size, query_count, _ = query_features.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, query_count, dtype=torch.bool,
                device=query_features.device,
            )
        if (not isinstance(valid_mask, torch.Tensor)
                or valid_mask.dtype != torch.bool
                or valid_mask.shape != (batch_size, query_count)
                or valid_mask.device != query_features.device
                or not bool(valid_mask.any(dim=1).all().item())):
            raise ValueError("valid_mask must be bool [B,Q] with a valid query")
        query = query_features.float()
        if self.detach_inputs:
            query = query.detach()
        if not bool(torch.isfinite(query).all().item()):
            raise ValueError("query features must be finite")
        query_embedding = self.query_projection(self.query_norm(query))
        query_embedding = query_embedding.masked_fill(
            ~valid_mask.unsqueeze(-1), 0.0
        )
        scale = math.sqrt(float(self.hidden_dim))
        residuals = []
        for batch_idx, superpoint_row in enumerate(superpoint_features):
            if (not isinstance(superpoint_row, torch.Tensor)
                    or superpoint_row.dim() != 2
                    or superpoint_row.shape[0] != self.d_model
                    or superpoint_row.shape[1] <= 0
                    or not superpoint_row.is_floating_point()
                    or superpoint_row.device != query_features.device):
                raise ValueError(
                    "each superpoint row must be floating [D,S] on query device"
                )
            superpoints = superpoint_row.transpose(0, 1).float()
            if self.detach_inputs:
                superpoints = superpoints.detach()
            if not bool(torch.isfinite(superpoints).all().item()):
                raise ValueError("superpoint features must be finite")
            superpoint_embedding = self.superpoint_projection(
                self.superpoint_norm(superpoints)
            )
            raw = torch.matmul(
                query_embedding[batch_idx],
                superpoint_embedding.transpose(0, 1),
            ) / scale
            residual = self.max_delta * torch.tanh(raw)
            residuals.append(residual.masked_fill(
                ~valid_mask[batch_idx].unsqueeze(1), 0.0
            ))
        return residuals


def build_source_distribution_reliability_features(
        source_scores, source_validity, shared_index):
    """Build scale-stable RAPF-style evidence for anonymous sources."""
    if (not isinstance(source_scores, torch.Tensor)
            or source_scores.dim() != 3
            or not source_scores.is_floating_point()
            or not bool(torch.isfinite(source_scores).all().item())):
        raise ValueError("source_scores must be finite floating [B,Q,S]")
    if (not isinstance(source_validity, torch.Tensor)
            or source_validity.dtype != torch.bool
            or source_validity.shape != source_scores.shape
            or source_validity.device != source_scores.device):
        raise ValueError("source_validity must be bool [B,Q,S]")
    source_count = source_scores.shape[-1]
    if (not isinstance(shared_index, int) or isinstance(shared_index, bool)
            or shared_index < 0 or shared_index >= source_count):
        raise ValueError("shared_index is invalid")
    if not bool(source_validity[..., shared_index].any(dim=1).all().item()):
        raise ValueError("each sample needs a valid shared source query")

    validity = source_validity.to(source_scores.dtype)
    counts = validity.sum(dim=1, keepdim=True)
    safe_counts = counts.clamp(min=1.0)
    safe_scores = source_scores.masked_fill(~source_validity, 0.0)
    means = safe_scores.sum(dim=1, keepdim=True) / safe_counts
    centered = (source_scores - means).masked_fill(~source_validity, 0.0)
    variances = centered.square().sum(dim=1, keepdim=True) / safe_counts
    standardized = centered / variances.clamp(min=1e-6).sqrt()
    standardized = standardized.masked_fill(~source_validity, 0.0)

    probabilities = F.softmax(
        standardized.masked_fill(~source_validity, -1e4), dim=1
    ) * validity
    probabilities = probabilities / probabilities.sum(
        dim=1, keepdim=True
    ).clamp(min=1e-8)
    entropy = -(
        probabilities * probabilities.clamp(min=1e-8).log()
    ).sum(dim=1)
    entropy = entropy / counts.squeeze(1).clamp(min=2.0).log()
    entropy = entropy.masked_fill(counts.squeeze(1) <= 0, 0.0)

    top_count = min(2, source_scores.shape[1])
    top_values = torch.topk(
        standardized.masked_fill(~source_validity, -1e4),
        k=top_count,
        dim=1,
    ).values
    if top_count == 1:
        top1_margin = torch.zeros_like(top_values[:, 0])
    else:
        top1_margin = torch.tanh(
            (top_values[:, 0] - top_values[:, 1]).clamp(min=0.0)
        )
        top1_margin = top1_margin.masked_fill(
            counts.squeeze(1) < 2, 0.0
        )
    top1_indices = standardized.masked_fill(
        ~source_validity, -1e4
    ).argmax(dim=1)
    top1_disagreement = (
        top1_indices != top1_indices[:, shared_index].unsqueeze(1)
    ).to(source_scores.dtype)

    shared_probabilities = probabilities[..., shared_index].unsqueeze(-1)
    mixture = 0.5 * (probabilities + shared_probabilities)
    epsilon = 1e-8
    source_kl = (
        probabilities
        * (
            probabilities.clamp(min=epsilon).log()
            - mixture.clamp(min=epsilon).log()
        )
    ).sum(dim=1)
    shared_kl = (
        shared_probabilities
        * (
            shared_probabilities.clamp(min=epsilon).log()
            - mixture.clamp(min=epsilon).log()
        )
    ).sum(dim=1)
    js_divergence = 0.5 * (source_kl + shared_kl) / math.log(2.0)
    js_divergence = js_divergence.clamp(min=0.0, max=1.0)

    batch_size, query_count, _ = source_scores.shape
    row_features = torch.stack((
        entropy,
        top1_margin,
        top1_disagreement,
        js_divergence,
    ), dim=-1).unsqueeze(1).expand(-1, query_count, -1, -1)
    features = torch.cat((
        standardized.unsqueeze(-1),
        probabilities.unsqueeze(-1),
        row_features,
    ), dim=-1)
    return features.masked_fill(~source_validity.unsqueeze(-1), 0.0)


class JointQualityAdaptiveSourceMixer(nn.Module):
    """Use joint box/mask quality to mix sources for every query.

    Source identity is represented only by the shared-source flag.  The same
    encoder is applied to every source, so routed sources can be reordered or
    replaced without changing the learned policy.  The final strength head is
    zero initialized to preserve the parent scores exactly at step zero.
    """

    def __init__(self, hidden_dim, source_count, shared_index,
                 max_delta=1.0, temperature=0.5,
                 use_distribution_reliability=False):
        super().__init__()
        for name, value in (
                ("hidden_dim", hidden_dim),
                ("source_count", source_count)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError("{} must be a positive integer".format(name))
        if source_count < 2:
            raise ValueError("adaptive source mixing needs at least two sources")
        if (not isinstance(shared_index, int) or isinstance(shared_index, bool)
                or shared_index < 0 or shared_index >= source_count):
            raise ValueError("shared_index is invalid")
        self.max_delta = _finite_non_negative("max_delta", max_delta)
        if self.max_delta <= 0.0:
            raise ValueError("max_delta must be positive")
        if (not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or not math.isfinite(float(temperature))
                or float(temperature) <= 0.0):
            raise ValueError("temperature must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.source_count = int(source_count)
        self.shared_index = int(shared_index)
        self.temperature = float(temperature)
        if not isinstance(use_distribution_reliability, bool):
            raise ValueError("use_distribution_reliability must be boolean")
        self.use_distribution_reliability = use_distribution_reliability
        # hidden + six absolute-quality values + rank/gap/shared flag
        source_input_dim = self.hidden_dim + 9
        if self.use_distribution_reliability:
            source_input_dim += SOURCE_DISTRIBUTION_RELIABILITY_DIM
        self.source_encoder = nn.Sequential(
            nn.Linear(source_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.source_router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        # hidden + six quality values + parent/mixed/shared/disagreement
        self.strength_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 10, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.source_router[-1].weight)
        nn.init.zeros_(self.strength_head[-1].weight)
        nn.init.zeros_(self.strength_head[-1].bias)

    def _source_ranks(self, source_scores, source_validity, valid_mask):
        ranks = []
        for source_index in range(self.source_count):
            source_valid = (
                source_validity[..., source_index] & valid_mask
            )
            normalization_mask = source_valid.clone()
            empty_rows = ~normalization_mask.any(dim=1)
            normalization_mask[empty_rows, 0] = True
            ranks.append(_straight_through_rank_normalize(
                source_scores[..., source_index], normalization_mask
            ).masked_fill(~source_valid, 0.0))
        return torch.stack(ranks, dim=-1)

    def forward(self, hidden, quality_evidence, parent_scores,
                source_scores, source_validity, valid_mask):
        if (not isinstance(hidden, torch.Tensor) or hidden.dim() != 3
                or hidden.shape[-1] != self.hidden_dim
                or not hidden.is_floating_point()
                or not bool(torch.isfinite(hidden).all().item())):
            raise ValueError("source mixer hidden must be finite [B,Q,H]")
        batch_size, query_count, _ = hidden.shape
        if (not isinstance(quality_evidence, torch.Tensor)
                or quality_evidence.shape != (batch_size, query_count, 6)
                or quality_evidence.device != hidden.device
                or not quality_evidence.is_floating_point()
                or not bool(torch.isfinite(quality_evidence).all().item())
                or bool(((quality_evidence < 0.0)
                         | (quality_evidence > 1.0)).any().item())):
            raise ValueError(
                "quality_evidence must be finite [B,Q,6] in [0,1]"
            )
        if (not isinstance(parent_scores, torch.Tensor)
                or parent_scores.shape != (batch_size, query_count)
                or parent_scores.device != hidden.device
                or not parent_scores.is_floating_point()
                or not bool(torch.isfinite(
                    parent_scores.masked_fill(~valid_mask, 0.0)
                ).all().item())):
            raise ValueError("parent_scores must be finite floating [B,Q]")
        expected_source_shape = (
            batch_size, query_count, self.source_count
        )
        if (not isinstance(source_scores, torch.Tensor)
                or source_scores.shape != expected_source_shape
                or source_scores.device != hidden.device
                or not source_scores.is_floating_point()
                or not bool(torch.isfinite(source_scores).all().item())):
            raise ValueError("source_scores must be finite floating [B,Q,S]")
        if (not isinstance(source_validity, torch.Tensor)
                or source_validity.dtype != torch.bool
                or source_validity.shape != expected_source_shape
                or source_validity.device != hidden.device):
            raise ValueError("source_validity must be bool [B,Q,S]")
        if (not isinstance(valid_mask, torch.Tensor)
                or valid_mask.dtype != torch.bool
                or valid_mask.shape != (batch_size, query_count)
                or valid_mask.device != hidden.device
                or not bool(valid_mask.any(dim=1).all().item())):
            raise ValueError("valid_mask must be bool [B,Q] with a valid query")
        source_validity = source_validity & valid_mask.unsqueeze(-1)
        shared_validity = source_validity[..., self.shared_index]
        if not bool(shared_validity.any(dim=1).all().item()):
            raise ValueError("each sample needs a valid shared source query")
        if not bool(source_validity.any(dim=2)[valid_mask].all().item()):
            raise ValueError("each valid query needs at least one valid source")

        source_ranks = self._source_ranks(
            source_scores, source_validity, valid_mask
        )
        shared_rank = source_ranks[..., self.shared_index]
        shared_flag = source_ranks.new_zeros(self.source_count)
        shared_flag[self.shared_index] = 1.0
        source_feature_parts = [
            hidden.unsqueeze(2).expand(-1, -1, self.source_count, -1),
            quality_evidence.unsqueeze(2).expand(
                -1, -1, self.source_count, -1
            ),
            source_ranks.unsqueeze(-1),
            (source_ranks - shared_rank.unsqueeze(-1)).unsqueeze(-1),
            shared_flag.view(1, 1, self.source_count, 1).expand(
                batch_size, query_count, -1, -1
            ),
        ]
        distribution_reliability = None
        if self.use_distribution_reliability:
            distribution_reliability = (
                build_source_distribution_reliability_features(
                    source_scores, source_validity, self.shared_index
                )
            )
            source_feature_parts.append(distribution_reliability)
        source_features = torch.cat(source_feature_parts, dim=-1)
        source_hidden = self.source_encoder(source_features)
        router_residual = self.source_router(source_hidden).squeeze(-1)
        router_logits = (
            source_ranks / self.temperature + router_residual
        ).masked_fill(~source_validity, -1e4)
        source_weights = F.softmax(router_logits, dim=-1)
        source_weights = (
            source_weights * source_validity.to(source_weights.dtype)
        )
        source_weights = source_weights / source_weights.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        mixed_rank = (source_weights * source_ranks).sum(dim=-1)
        parent_rank = _rank_normalize(parent_scores, valid_mask)
        source_count = source_validity.sum(dim=-1).clamp(min=1)
        disagreement = (
            (source_ranks - shared_rank.unsqueeze(-1)).abs()
            * source_validity.to(source_ranks.dtype)
        ).sum(dim=-1) / source_count.to(source_ranks.dtype)
        raw_strength = self.strength_head(torch.cat((
            hidden,
            quality_evidence,
            parent_rank.unsqueeze(-1),
            mixed_rank.unsqueeze(-1),
            shared_rank.unsqueeze(-1),
            disagreement.unsqueeze(-1),
        ), dim=-1)).squeeze(-1)
        strength = torch.tanh(raw_strength).masked_fill(~valid_mask, 0.0)
        proposal_gap = (mixed_rank - parent_rank).masked_fill(
            ~valid_mask, 0.0
        )
        residual_logit = (
            self.max_delta * strength * proposal_gap
        ).masked_fill(~valid_mask, 0.0)
        entropy = -(
            source_weights.clamp(min=1e-8).log() * source_weights
        ).sum(dim=-1).masked_fill(~valid_mask, 0.0)
        effective_source_count = entropy.exp().masked_fill(~valid_mask, 0.0)
        result = {
            "source_mix_residual_logit": residual_logit,
            "source_mix_strength": strength,
            "source_mix_weights": source_weights,
            "source_mix_router_logits": router_logits,
            "source_mix_router_residual": router_residual.masked_fill(
                ~source_validity, 0.0
            ),
            "source_mix_ranks": source_ranks,
            "source_mix_parent_rank": parent_rank,
            "source_mix_mixed_rank": mixed_rank,
            "source_mix_proposal_gap": proposal_gap,
            "source_mix_disagreement": disagreement,
            "source_mix_effective_source_count": effective_source_count,
            "source_mix_validity": source_validity,
        }
        if distribution_reliability is not None:
            result["source_mix_distribution_reliability"] = (
                distribution_reliability
            )
        return result


class JointQueryQualityReranker(nn.Module):
    """Zero-residual set ranker with shared box and mask quality evidence."""

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=1,
                 dropout=0.1, max_delta=1.25, mask_weight=0.25,
                 quality_score_weight=1.0, direct_residual_scale=1.0,
                 use_metric_aligned_utility=False, detach_inputs=True,
                 use_mask_calibration=False, max_mask_alpha_delta=1.0,
                 max_mask_logit_bias=2.0,
                 use_source_mask_evidence=False, use_gate_evidence=False,
                 use_spatial_mask_refiner=False,
                 spatial_mask_d_model=288,
                 spatial_mask_hidden_dim=32,
                 max_spatial_mask_delta=2.0,
                 use_adaptive_source_mixing=False,
                 source_count=None, shared_source_index=None,
                 max_source_mix_delta=1.0,
                 source_mix_temperature=0.5,
                 use_source_distribution_reliability=False,
                 preserve_parent_score=False,
                 candidate_promotion_margin=0.0,
                 use_parent_transition_advantage=False,
                 use_decomposed_transition_advantage=False,
                 use_setwise_tier_advantage=False,
                 use_decoupled_setwise_heads=False,
                 use_factorized_setwise_safety=False,
                 use_factorized_setwise_risk_bound=False,
                 use_setwise_safety_veto_gate=False,
                 use_cost_calibrated_setwise_risk_bound=False,
                 use_setwise_safety_slack_quantile_bound=False,
                 use_setwise_safety_slack_pairwise_order=False,
                 use_proposal_conditioned_safety=False,
                 use_parent_referenced_safety=False,
                 use_coupled_safe_repair_witness=False,
                 use_bidirectional_coupled_boundary=False,
                 use_centered_coupled_separation=False,
                 use_hazard_conditioned_coupled_separation=False,
                 use_monotonic_box_safety_folding=False,
                 use_same_candidate_branchwise_witness=False,
                 use_parent_non_degradation_certificate=False,
                 use_criterion_responsible_hazard_attribution=False,
                 use_independent_joint_hazard_certificate=False,
                 use_frozen_raw_joint_hazard_features=False,
                 use_factorized_hit_advantage=False,
                 use_factorized_nested_dominance=False,
                 factorized_hit_break_cost=4.0,
                 parent_transition_break_cost=4.0,
                 parent_transition_candidate_top_k=0):
        super().__init__()
        for name, value in (
                ("input_dim", input_dim), ("hidden_dim", hidden_dim),
                ("num_heads", num_heads), ("num_layers", num_layers)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError("{} must be a positive integer".format(name))
        if hidden_dim % num_heads != 0:
            raise ValueError("num_heads must divide hidden_dim")
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or not 0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout must lie in [0,1)")
        if (not isinstance(max_delta, (int, float))
                or isinstance(max_delta, bool)
                or not math.isfinite(float(max_delta))
                or float(max_delta) <= 0.0):
            raise ValueError("max_delta must be finite and positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.mask_weight = _finite_non_negative("mask_weight", mask_weight)
        if self.mask_weight >= 0.8:
            raise ValueError(
                "mask_weight must be below 0.8 to preserve box-tier priority"
            )
        self.quality_score_weight = _finite_non_negative(
            "quality_score_weight", quality_score_weight
        )
        self.direct_residual_scale = _finite_non_negative(
            "direct_residual_scale", direct_residual_scale
        )
        if self.direct_residual_scale > 1.0:
            raise ValueError("direct_residual_scale must be at most one")
        if not isinstance(use_metric_aligned_utility, bool):
            raise ValueError("use_metric_aligned_utility must be boolean")
        self.use_metric_aligned_utility = use_metric_aligned_utility
        if not isinstance(preserve_parent_score, bool):
            raise ValueError("preserve_parent_score must be boolean")
        self.preserve_parent_score = preserve_parent_score
        self.candidate_promotion_margin = _finite_non_negative(
            "candidate_promotion_margin", candidate_promotion_margin
        )
        if (self.candidate_promotion_margin > 0.0
                and not self.preserve_parent_score):
            raise ValueError(
                "candidate promotion margin requires preserved parent score"
            )
        if self.candidate_promotion_margin >= self.max_delta:
            raise ValueError(
                "candidate promotion margin must be below max_delta"
            )
        if not isinstance(use_parent_transition_advantage, bool):
            raise ValueError(
                "use_parent_transition_advantage must be boolean"
            )
        self.use_parent_transition_advantage = (
            use_parent_transition_advantage
        )
        if not isinstance(use_decomposed_transition_advantage, bool):
            raise ValueError(
                "use_decomposed_transition_advantage must be boolean"
            )
        self.use_decomposed_transition_advantage = (
            use_decomposed_transition_advantage
        )
        if not isinstance(use_setwise_tier_advantage, bool):
            raise ValueError(
                "use_setwise_tier_advantage must be boolean"
            )
        self.use_setwise_tier_advantage = use_setwise_tier_advantage
        if not isinstance(use_decoupled_setwise_heads, bool):
            raise ValueError("use_decoupled_setwise_heads must be boolean")
        self.use_decoupled_setwise_heads = use_decoupled_setwise_heads
        if (self.use_decoupled_setwise_heads
                and not self.use_setwise_tier_advantage):
            raise ValueError(
                "decoupled setwise heads require setwise tier advantage"
            )
        if not isinstance(use_factorized_setwise_safety, bool):
            raise ValueError(
                "use_factorized_setwise_safety must be boolean"
            )
        self.use_factorized_setwise_safety = (
            use_factorized_setwise_safety
        )
        if (self.use_factorized_setwise_safety
                and not self.use_decoupled_setwise_heads):
            raise ValueError(
                "factorized setwise safety requires decoupled setwise heads"
            )
        if not isinstance(use_factorized_setwise_risk_bound, bool):
            raise ValueError(
                "use_factorized_setwise_risk_bound must be boolean"
            )
        self.use_factorized_setwise_risk_bound = (
            use_factorized_setwise_risk_bound
        )
        if (self.use_factorized_setwise_risk_bound
                and not self.use_factorized_setwise_safety):
            raise ValueError(
                "factorized setwise risk bound requires factorized safety"
            )
        if not isinstance(use_setwise_safety_veto_gate, bool):
            raise ValueError("use_setwise_safety_veto_gate must be boolean")
        self.use_setwise_safety_veto_gate = (
            use_setwise_safety_veto_gate
        )
        if (self.use_setwise_safety_veto_gate
                and not self.use_decoupled_setwise_heads):
            raise ValueError(
                "setwise safety veto gate requires decoupled setwise heads"
            )
        if not isinstance(
                use_cost_calibrated_setwise_risk_bound, bool):
            raise ValueError(
                "use_cost_calibrated_setwise_risk_bound must be boolean"
            )
        self.use_cost_calibrated_setwise_risk_bound = (
            use_cost_calibrated_setwise_risk_bound
        )
        if (self.use_cost_calibrated_setwise_risk_bound
                and not self.use_factorized_setwise_risk_bound):
            raise ValueError(
                "cost-calibrated risk bound requires factorized risk bound"
            )
        if not isinstance(
                use_setwise_safety_slack_quantile_bound, bool):
            raise ValueError(
                "use_setwise_safety_slack_quantile_bound must be boolean"
            )
        self.use_setwise_safety_slack_quantile_bound = (
            use_setwise_safety_slack_quantile_bound
        )
        if (self.use_setwise_safety_slack_quantile_bound
                and not self.use_factorized_setwise_risk_bound):
            raise ValueError(
                "safety-slack quantile bound requires factorized risk bound"
            )
        if (self.use_setwise_safety_slack_quantile_bound
                and self.use_cost_calibrated_setwise_risk_bound):
            raise ValueError(
                "safety-slack quantile bound and cost-calibrated risk bound "
                "are mutually exclusive"
            )
        if not isinstance(
                use_setwise_safety_slack_pairwise_order, bool):
            raise ValueError(
                "use_setwise_safety_slack_pairwise_order must be boolean"
            )
        self.use_setwise_safety_slack_pairwise_order = (
            use_setwise_safety_slack_pairwise_order
        )
        if (self.use_setwise_safety_slack_pairwise_order
                and not self.use_setwise_safety_slack_quantile_bound):
            raise ValueError(
                "safety-slack pairwise order requires slack quantile bound"
            )
        if not isinstance(use_proposal_conditioned_safety, bool):
            raise ValueError(
                "use_proposal_conditioned_safety must be boolean"
            )
        self.use_proposal_conditioned_safety = (
            use_proposal_conditioned_safety
        )
        if (self.use_proposal_conditioned_safety
                and not self.use_setwise_safety_slack_pairwise_order):
            raise ValueError(
                "proposal-conditioned safety requires safety-slack pairwise "
                "order"
            )
        if (self.use_proposal_conditioned_safety
                and not self.use_setwise_safety_veto_gate):
            raise ValueError(
                "proposal-conditioned safety requires the safety veto gate"
            )
        if not isinstance(use_parent_referenced_safety, bool):
            raise ValueError(
                "use_parent_referenced_safety must be boolean"
            )
        self.use_parent_referenced_safety = (
            use_parent_referenced_safety
        )
        if (self.use_parent_referenced_safety
                and not self.use_setwise_safety_slack_pairwise_order):
            raise ValueError(
                "parent-referenced safety requires safety-slack pairwise "
                "order"
            )
        if (self.use_parent_referenced_safety
                and self.use_proposal_conditioned_safety):
            raise ValueError(
                "parent-referenced and proposal-conditioned safety are "
                "mutually exclusive"
            )
        if not isinstance(use_coupled_safe_repair_witness, bool):
            raise ValueError(
                "use_coupled_safe_repair_witness must be boolean"
            )
        self.use_coupled_safe_repair_witness = (
            use_coupled_safe_repair_witness
        )
        if (self.use_coupled_safe_repair_witness
                and not self.use_parent_referenced_safety):
            raise ValueError(
                "coupled safe-repair witness requires parent-referenced "
                "safety"
            )
        if not isinstance(use_bidirectional_coupled_boundary, bool):
            raise ValueError(
                "use_bidirectional_coupled_boundary must be boolean"
            )
        self.use_bidirectional_coupled_boundary = (
            use_bidirectional_coupled_boundary
        )
        if (self.use_bidirectional_coupled_boundary
                and not self.use_coupled_safe_repair_witness):
            raise ValueError(
                "bidirectional coupled boundary requires coupled "
                "safe-repair witness"
            )
        if not isinstance(use_centered_coupled_separation, bool):
            raise ValueError(
                "use_centered_coupled_separation must be boolean"
            )
        self.use_centered_coupled_separation = (
            use_centered_coupled_separation
        )
        if (self.use_centered_coupled_separation
                and not self.use_bidirectional_coupled_boundary):
            raise ValueError(
                "centered coupled separation requires bidirectional "
                "coupled boundary"
            )
        if not isinstance(
                use_hazard_conditioned_coupled_separation, bool):
            raise ValueError(
                "use_hazard_conditioned_coupled_separation must be boolean"
            )
        self.use_hazard_conditioned_coupled_separation = (
            use_hazard_conditioned_coupled_separation
        )
        if (self.use_hazard_conditioned_coupled_separation
                and not self.use_centered_coupled_separation):
            raise ValueError(
                "hazard-conditioned coupled separation requires centered "
                "coupled separation"
            )
        if not isinstance(use_monotonic_box_safety_folding, bool):
            raise ValueError(
                "use_monotonic_box_safety_folding must be boolean"
            )
        self.use_monotonic_box_safety_folding = (
            use_monotonic_box_safety_folding
        )
        if (self.use_monotonic_box_safety_folding
                and not self.use_hazard_conditioned_coupled_separation):
            raise ValueError(
                "monotonic box-safety folding requires hazard-conditioned "
                "coupled separation"
            )
        if not isinstance(use_same_candidate_branchwise_witness, bool):
            raise ValueError(
                "use_same_candidate_branchwise_witness must be boolean"
            )
        self.use_same_candidate_branchwise_witness = (
            use_same_candidate_branchwise_witness
        )
        if (self.use_same_candidate_branchwise_witness
                and not self.use_monotonic_box_safety_folding):
            raise ValueError(
                "same-candidate branchwise witness requires monotonic "
                "box-safety folding"
            )
        if not isinstance(use_parent_non_degradation_certificate, bool):
            raise ValueError(
                "use_parent_non_degradation_certificate must be boolean"
            )
        self.use_parent_non_degradation_certificate = (
            use_parent_non_degradation_certificate
        )
        if (self.use_parent_non_degradation_certificate
                and not self.use_same_candidate_branchwise_witness):
            raise ValueError(
                "parent non-degradation certificate requires same-candidate "
                "branchwise witness"
            )
        if not isinstance(use_criterion_responsible_hazard_attribution, bool):
            raise ValueError(
                "use_criterion_responsible_hazard_attribution must be boolean"
            )
        self.use_criterion_responsible_hazard_attribution = (
            use_criterion_responsible_hazard_attribution
        )
        if (self.use_criterion_responsible_hazard_attribution
                and not self.use_parent_non_degradation_certificate):
            raise ValueError(
                "criterion-responsible hazard attribution requires parent "
                "non-degradation certificate"
            )
        if not isinstance(use_independent_joint_hazard_certificate, bool):
            raise ValueError(
                "use_independent_joint_hazard_certificate must be boolean"
            )
        self.use_independent_joint_hazard_certificate = (
            use_independent_joint_hazard_certificate
        )
        if (self.use_independent_joint_hazard_certificate
                and not self.use_parent_non_degradation_certificate):
            raise ValueError(
                "independent joint-hazard certificate requires parent "
                "non-degradation certificate"
            )
        if (self.use_independent_joint_hazard_certificate
                and self.use_criterion_responsible_hazard_attribution):
            raise ValueError(
                "independent joint-hazard certificate and criterion-"
                "responsible hazard attribution are mutually exclusive"
            )
        if not isinstance(use_frozen_raw_joint_hazard_features, bool):
            raise ValueError(
                "use_frozen_raw_joint_hazard_features must be boolean"
            )
        self.use_frozen_raw_joint_hazard_features = (
            use_frozen_raw_joint_hazard_features
        )
        if (self.use_frozen_raw_joint_hazard_features
                and not self.use_independent_joint_hazard_certificate):
            raise ValueError(
                "frozen raw joint-hazard features require independent "
                "joint-hazard certificate"
            )
        if not isinstance(use_factorized_hit_advantage, bool):
            raise ValueError(
                "use_factorized_hit_advantage must be boolean"
            )
        self.use_factorized_hit_advantage = use_factorized_hit_advantage
        if not isinstance(use_factorized_nested_dominance, bool):
            raise ValueError(
                "use_factorized_nested_dominance must be boolean"
            )
        self.use_factorized_nested_dominance = (
            use_factorized_nested_dominance
        )
        if (self.use_factorized_nested_dominance
                and not self.use_factorized_hit_advantage):
            raise ValueError(
                "factorized nested dominance requires factorized hit "
                "advantage"
            )
        self.factorized_hit_break_cost = _finite_non_negative(
            "factorized_hit_break_cost", factorized_hit_break_cost
        )
        if self.factorized_hit_break_cost <= 0.0:
            raise ValueError("factorized hit break cost must be positive")
        transition_mode_count = sum((
            self.use_parent_transition_advantage,
            self.use_decomposed_transition_advantage,
            self.use_setwise_tier_advantage,
            self.use_factorized_hit_advantage,
        ))
        if transition_mode_count > 1:
            raise ValueError(
                "direct, decomposed, setwise-tier, and factorized "
                "transition advantages are mutually exclusive"
            )
        self.parent_transition_break_cost = _finite_non_negative(
            "parent_transition_break_cost", parent_transition_break_cost
        )
        if self.parent_transition_break_cost <= 0.0:
            raise ValueError("parent transition break cost must be positive")
        if (self.use_parent_transition_advantage
                and not self.preserve_parent_score):
            raise ValueError(
                "parent transition advantage requires preserved parent score"
            )
        if (self.use_decomposed_transition_advantage
                and not self.preserve_parent_score):
            raise ValueError(
                "decomposed transition advantage requires preserved parent "
                "score"
            )
        if (self.use_setwise_tier_advantage
                and not self.preserve_parent_score):
            raise ValueError(
                "setwise tier advantage requires preserved parent score"
            )
        if (self.use_factorized_hit_advantage
                and not self.preserve_parent_score):
            raise ValueError(
                "factorized hit advantage requires preserved parent score"
            )
        if (not isinstance(parent_transition_candidate_top_k, int)
                or isinstance(parent_transition_candidate_top_k, bool)
                or parent_transition_candidate_top_k < 0):
            raise ValueError(
                "parent transition candidate top k must be non-negative int"
            )
        self.parent_transition_candidate_top_k = int(
            parent_transition_candidate_top_k
        )
        if (self.parent_transition_candidate_top_k > 0
                and not (self.use_parent_transition_advantage
                         or self.use_decomposed_transition_advantage
                         or self.use_setwise_tier_advantage
                         or self.use_factorized_hit_advantage)):
            raise ValueError(
                "parent transition candidate restriction requires "
                "a transition advantage"
            )
        self.detach_inputs = bool(detach_inputs)
        if not isinstance(use_mask_calibration, bool):
            raise ValueError("use_mask_calibration must be boolean")
        self.use_mask_calibration = use_mask_calibration
        if not isinstance(use_source_mask_evidence, bool):
            raise ValueError("use_source_mask_evidence must be boolean")
        if use_source_mask_evidence and not self.use_mask_calibration:
            raise ValueError(
                "source mask evidence requires enabled mask calibration"
            )
        self.use_source_mask_evidence = use_source_mask_evidence
        self.source_mask_evidence_dim = (
            QUERY_MASK_SOURCE_EVIDENCE_DIM
            if self.use_source_mask_evidence else 0
        )
        if not isinstance(use_gate_evidence, bool):
            raise ValueError("use_gate_evidence must be boolean")
        self.use_gate_evidence = use_gate_evidence
        self.gate_evidence_dim = (
            JOINT_QUERY_GATE_EVIDENCE_DIM if self.use_gate_evidence else 0
        )
        if not isinstance(use_spatial_mask_refiner, bool):
            raise ValueError("use_spatial_mask_refiner must be boolean")
        if use_spatial_mask_refiner and not self.use_mask_calibration:
            raise ValueError(
                "spatial mask refinement requires mask calibration"
            )
        self.use_spatial_mask_refiner = use_spatial_mask_refiner
        if not isinstance(use_adaptive_source_mixing, bool):
            raise ValueError("use_adaptive_source_mixing must be boolean")
        self.use_adaptive_source_mixing = use_adaptive_source_mixing
        if not isinstance(use_source_distribution_reliability, bool):
            raise ValueError(
                "use_source_distribution_reliability must be boolean"
            )
        if (use_source_distribution_reliability
                and not self.use_adaptive_source_mixing):
            raise ValueError(
                "source distribution reliability requires adaptive source "
                "mixing"
            )
        self.use_source_distribution_reliability = (
            use_source_distribution_reliability
        )
        self.max_mask_alpha_delta = _finite_non_negative(
            "max_mask_alpha_delta", max_mask_alpha_delta
        )
        self.max_mask_logit_bias = _finite_non_negative(
            "max_mask_logit_bias", max_mask_logit_bias
        )
        if self.use_mask_calibration and (
                self.max_mask_alpha_delta <= 0.0
                or self.max_mask_alpha_delta > 1.0
                or self.max_mask_logit_bias <= 0.0):
            raise ValueError(
                "enabled mask calibration requires alpha delta in (0,1] "
                "and positive logit bias"
            )
        input_extra_dim = (
            (3 if self.use_mask_calibration else 2)
            + self.source_mask_evidence_dim
            + self.gate_evidence_dim
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim + input_extra_dim),
            nn.Linear(self.input_dim + input_extra_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(
                self.hidden_dim, num_heads, dropout=float(dropout),
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)
        ])
        self.quality_head = nn.Linear(self.hidden_dim, 6)
        self.residual_head = nn.Linear(self.hidden_dim, 1)
        self.parent_transition_head = (
            nn.Sequential(
                nn.Linear(4 * self.hidden_dim + 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 6),
            )
            if self.use_parent_transition_advantage else None
        )
        self.decomposed_transition_head = (
            nn.Sequential(
                nn.Linear(4 * self.hidden_dim + 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 4),
            )
            if self.use_decomposed_transition_advantage else None
        )
        self.setwise_tier_head = (
            nn.Sequential(
                nn.Linear(4 * self.hidden_dim + 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 2, bias=False),
            )
            if (self.use_setwise_tier_advantage
                and not self.use_decoupled_setwise_heads) else None
        )
        # V69: promotion and safety use independent nonlinear heads.  The
        # shared pair representation transfers context, while neither output
        # head can satisfy the other head's objective through shared logits.
        self.setwise_promotion_head = (
            nn.Sequential(
                nn.Linear(4 * self.hidden_dim + 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1, bias=False),
            )
            if self.use_decoupled_setwise_heads else None
        )
        self.setwise_safety_head = (
            nn.Sequential(
                nn.Linear(4 * self.hidden_dim + 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(
                    self.hidden_dim,
                    (6 if self.use_factorized_setwise_risk_bound else
                     3 if self.use_factorized_setwise_safety else 1),
                    bias=False,
                ),
            )
            if self.use_decoupled_setwise_heads else None
        )
        # V88: one separately parameterized lower-bound certificate sees the
        # same parent/candidate evidence but cannot send conservative hazard
        # gradients into the shared representation or V86 branches.
        self.independent_joint_hazard_head = (
            nn.Sequential(
                nn.Linear(
                    4 * (
                        self.input_dim
                        if self.use_frozen_raw_joint_hazard_features
                        else self.hidden_dim
                    ) + 2,
                    self.hidden_dim,
                ),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1, bias=False),
            )
            if self.use_independent_joint_hazard_certificate else None
        )
        self.factorized_hit_head = (
            nn.Linear(self.hidden_dim, 2)
            if self.use_factorized_hit_advantage else None
        )
        self.mask_calibration_head = (
            nn.Linear(self.hidden_dim, 2)
            if self.use_mask_calibration else None
        )
        self.spatial_mask_refiner = (
            QuerySuperpointMaskRefiner(
                d_model=spatial_mask_d_model,
                hidden_dim=spatial_mask_hidden_dim,
                max_delta=max_spatial_mask_delta,
                detach_inputs=self.detach_inputs,
            )
            if self.use_spatial_mask_refiner else None
        )
        self.adaptive_source_mixer = (
            JointQualityAdaptiveSourceMixer(
                hidden_dim=self.hidden_dim,
                source_count=source_count,
                shared_index=shared_source_index,
                max_delta=max_source_mix_delta,
                temperature=source_mix_temperature,
                use_distribution_reliability=(
                    self.use_source_distribution_reliability
                ),
            )
            if self.use_adaptive_source_mixing else None
        )
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.parent_transition_head is not None:
            nn.init.zeros_(self.parent_transition_head[-1].weight)
            nn.init.zeros_(self.parent_transition_head[-1].bias)
        if self.decomposed_transition_head is not None:
            nn.init.zeros_(self.decomposed_transition_head[-1].weight)
            nn.init.zeros_(self.decomposed_transition_head[-1].bias)
            direction_prior = math.log(self.parent_transition_break_cost)
            with torch.no_grad():
                self.decomposed_transition_head[-1].bias[1::2].fill_(
                    direction_prior
                )
        if self.setwise_tier_head is not None:
            nn.init.zeros_(self.setwise_tier_head[-1].weight)
        if self.setwise_promotion_head is not None:
            nn.init.zeros_(self.setwise_promotion_head[-1].weight)
            nn.init.zeros_(self.setwise_safety_head[-1].weight)
        if self.independent_joint_hazard_head is not None:
            nn.init.zeros_(self.independent_joint_hazard_head[-1].weight)
        if self.factorized_hit_head is not None:
            nn.init.zeros_(self.factorized_hit_head.weight)
            nn.init.zeros_(self.factorized_hit_head.bias)
        if self.mask_calibration_head is not None:
            nn.init.zeros_(self.mask_calibration_head.weight)
            nn.init.zeros_(self.mask_calibration_head.bias)

    def forward(self, features, baseline_scores, valid_mask=None,
                base_mask_weights=None, source_mask_evidence=None,
                gate_evidence=None, spatial_query_features=None,
                spatial_superpoint_features=None,
                source_score_stack=None, source_validity=None):
        if (not isinstance(features, torch.Tensor) or features.dim() != 3
                or features.shape[-1] != self.input_dim
                or not features.is_floating_point()):
            raise ValueError("features must be floating [B,Q,input_dim]")
        batch_size, query_count, _ = features.shape
        if (not isinstance(baseline_scores, torch.Tensor)
                or baseline_scores.shape != (batch_size, query_count)
                or not baseline_scores.is_floating_point()):
            raise ValueError("baseline_scores must be floating [B,Q]")
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, query_count, dtype=torch.bool,
                device=features.device,
            )
        if (not isinstance(valid_mask, torch.Tensor)
                or valid_mask.dtype != torch.bool
                or valid_mask.shape != (batch_size, query_count)
                or valid_mask.device != features.device
                or not bool(valid_mask.any(dim=1).all().item())):
            raise ValueError("valid_mask must be bool [B,Q] with a valid query")
        if baseline_scores.device != features.device:
            raise ValueError("features and baseline_scores must share a device")
        if (not bool(torch.isfinite(features).all().item())
                or not bool(torch.isfinite(
                    baseline_scores.masked_fill(~valid_mask, 0.0)
                ).all().item())):
            raise ValueError("valid reranker inputs must be finite")

        mask_weights = None
        if self.use_mask_calibration:
            if (not isinstance(base_mask_weights, torch.Tensor)
                    or base_mask_weights.shape != (batch_size, query_count)
                    or not base_mask_weights.is_floating_point()
                    or base_mask_weights.device != features.device
                    or not bool(torch.isfinite(base_mask_weights).all().item())
                    or bool(((base_mask_weights < 0.0)
                             | (base_mask_weights > 1.0)).any().item())):
                raise ValueError(
                    "base_mask_weights must be finite floating [B,Q] in [0,1]"
                )
            mask_weights = base_mask_weights.float()

        mask_evidence = None
        if self.use_source_mask_evidence:
            expected_shape = (
                batch_size, query_count, self.source_mask_evidence_dim
            )
            if (not isinstance(source_mask_evidence, torch.Tensor)
                    or source_mask_evidence.shape != expected_shape
                    or not source_mask_evidence.is_floating_point()
                    or source_mask_evidence.device != features.device
                    or not bool(torch.isfinite(
                        source_mask_evidence
                    ).all().item())
                    or bool(((source_mask_evidence < 0.0)
                             | (source_mask_evidence > 1.0)).any().item())):
                raise ValueError(
                    "source_mask_evidence must be finite floating "
                    "[B,Q,{}] in [0,1]".format(
                        self.source_mask_evidence_dim
                    )
                )
            mask_evidence = source_mask_evidence.float()
        elif source_mask_evidence is not None:
            raise ValueError(
                "source_mask_evidence requires use_source_mask_evidence"
            )

        normalized_gate_evidence = None
        if self.use_gate_evidence:
            expected_shape = (
                batch_size, query_count, self.gate_evidence_dim
            )
            if (not isinstance(gate_evidence, torch.Tensor)
                    or gate_evidence.shape != expected_shape
                    or not gate_evidence.is_floating_point()
                    or gate_evidence.device != features.device
                    or not bool(torch.isfinite(gate_evidence).all().item())
                    or bool(((gate_evidence < 0.0)
                             | (gate_evidence > 1.0)).any().item())):
                raise ValueError(
                    "gate_evidence must be finite floating [B,Q,{}] in [0,1]"
                    .format(self.gate_evidence_dim)
                )
            normalized_gate_evidence = gate_evidence.float()
        elif gate_evidence is not None:
            raise ValueError("gate_evidence requires use_gate_evidence")

        input_features = features.float()
        baseline = baseline_scores.float()
        if self.detach_inputs:
            input_features = input_features.detach()
            baseline = baseline.detach()
            if mask_weights is not None:
                mask_weights = mask_weights.detach()
            if mask_evidence is not None:
                mask_evidence = mask_evidence.detach()
            if normalized_gate_evidence is not None:
                normalized_gate_evidence = normalized_gate_evidence.detach()
        baseline_rank = _rank_normalize(baseline, valid_mask)
        baseline_indices = baseline_rank.argmax(dim=1)
        transition_candidate_mask = valid_mask
        if self.parent_transition_candidate_top_k > 0:
            top_k = min(
                self.parent_transition_candidate_top_k, query_count
            )
            top_indices = baseline.masked_fill(
                ~valid_mask, float("-inf")
            ).topk(top_k, dim=1).indices
            transition_candidate_mask = torch.zeros_like(valid_mask)
            transition_candidate_mask.scatter_(
                1, top_indices,
                valid_mask.gather(1, top_indices),
            )
            row = torch.arange(batch_size, device=features.device)
            transition_candidate_mask[row, baseline_indices] = True
        baseline_standardized = _score_standardize(baseline, valid_mask)
        input_parts = [
            input_features,
            baseline_rank.unsqueeze(-1),
            baseline_standardized.unsqueeze(-1),
        ]
        if mask_weights is not None:
            input_parts.append(mask_weights.unsqueeze(-1))
        if mask_evidence is not None:
            input_parts.append(mask_evidence)
        if normalized_gate_evidence is not None:
            input_parts.append(normalized_gate_evidence)
        hidden = self.input_projection(torch.cat(input_parts, dim=-1))
        hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        for attention, attention_norm, ffn, ffn_norm in zip(
                self.attention, self.attention_norm,
                self.ffn, self.ffn_norm):
            attended, _ = attention(
                hidden, hidden, hidden,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            hidden = attention_norm(hidden + attended)
            hidden = ffn_norm(hidden + ffn(hidden))
            hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        raw_quality = self.quality_head(hidden)
        box_logits = ordinal_threshold_logits(raw_quality[..., :2])
        box_iou = raw_quality[..., 2].sigmoid()
        mask_logits = ordinal_threshold_logits(raw_quality[..., 3:5])
        mask_iou = raw_quality[..., 5].sigmoid()
        quality = predicted_joint_query_quality(
            box_logits, box_iou, mask_logits, mask_iou,
            mask_weight=self.mask_weight,
            metric_aligned=self.use_metric_aligned_utility,
        )
        quality_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        quality_mean = quality.masked_fill(~valid_mask, 0.0).sum(
            dim=1, keepdim=True
        ) / quality_count.to(quality.dtype)
        centered_quality = (quality - quality_mean).masked_fill(
            ~valid_mask, 0.0
        )
        quality_evidence = torch.cat((
            box_logits.sigmoid(),
            box_iou.unsqueeze(-1),
            mask_logits.sigmoid(),
            mask_iou.unsqueeze(-1),
        ), dim=-1)
        source_mix_out = None
        source_mix_residual_logit = baseline.new_zeros(baseline.shape)
        if self.adaptive_source_mixer is not None:
            if self.detach_inputs and source_score_stack is not None:
                source_score_stack = source_score_stack.detach()
            source_mix_out = self.adaptive_source_mixer(
                hidden=hidden,
                quality_evidence=quality_evidence,
                parent_scores=baseline,
                source_scores=source_score_stack,
                source_validity=source_validity,
                valid_mask=valid_mask,
            )
            source_mix_residual_logit = source_mix_out[
                "source_mix_residual_logit"
            ]
        elif source_score_stack is not None or source_validity is not None:
            raise ValueError(
                "source mixing inputs require adaptive source mixing"
            )
        raw_direct_residual_logit = self.residual_head(hidden).squeeze(-1)
        direct_residual_logit = (
            self.direct_residual_scale * raw_direct_residual_logit
        )
        parent_transition_logits = None
        decomposed_transition_logits = None
        setwise_tier_advantage = None
        setwise_tier_branch_scores = None
        setwise_safety_criterion_scores = None
        setwise_safety_bound_scores = None
        setwise_independent_joint_hazard_scores = None
        setwise_tier_reachable_mask = None
        setwise_proposal_indices = None
        setwise_proposal_mask = None
        setwise_proposal_promotable_mask = None
        decomposed_fix_break_utility = None
        decomposed_counterfactual_costs = None
        decomposed_counterfactual_selected = None
        parent_transition_advantage = None
        factorized_hit_logits = None
        factorized_hit_probabilities = None
        factorized_fix_break_utility = None
        if (self.parent_transition_head is not None
                or self.decomposed_transition_head is not None
                or self.setwise_tier_head is not None
                or self.setwise_promotion_head is not None
                or self.independent_joint_hazard_head is not None):
            row = torch.arange(batch_size, device=features.device)
            parent_hidden = hidden[row, baseline_indices].unsqueeze(1).expand(
                -1, query_count, -1
            )
            parent_rank = baseline_rank[row, baseline_indices].unsqueeze(1)
            parent_standardized = baseline_standardized[
                row, baseline_indices
            ].unsqueeze(1)
            pair_features = torch.cat((
                hidden,
                parent_hidden,
                hidden - parent_hidden,
                hidden * parent_hidden,
                (baseline_rank - parent_rank).unsqueeze(-1),
                (baseline_standardized - parent_standardized).unsqueeze(-1),
            ), dim=-1)
            if self.use_frozen_raw_joint_hazard_features:
                raw_parent_features = input_features[
                    row, baseline_indices
                ].unsqueeze(1).expand(-1, query_count, -1)
                independent_pair_features = torch.cat((
                    input_features,
                    raw_parent_features,
                    input_features - raw_parent_features,
                    input_features * raw_parent_features,
                    (baseline_rank - parent_rank).unsqueeze(-1),
                    (
                        baseline_standardized - parent_standardized
                    ).unsqueeze(-1),
                ), dim=-1)
            else:
                independent_pair_features = pair_features
            raw_independent_joint_hazard = (
                self.independent_joint_hazard_head(
                    independent_pair_features.detach()
                ).squeeze(-1)
                if self.independent_joint_hazard_head is not None else None
            )
            if self.parent_transition_head is not None:
                parent_transition_logits = self.parent_transition_head(
                    pair_features
                ).view(batch_size, query_count, 2, 3)
                break_logits = parent_transition_logits[..., 0]
                neutral_logits = parent_transition_logits[..., 1]
                fix_logits = parent_transition_logits[..., 2]
                log_cost = math.log(self.parent_transition_break_cost)
                safe_logit = (
                    fix_logits
                    - torch.logsumexp(torch.stack((
                        neutral_logits, break_logits + log_cost
                    ), dim=-1), dim=-1)
                    + math.log1p(self.parent_transition_break_cost)
                )
                threshold_weights = safe_logit.new_tensor((1.0, 2.0))
                parent_transition_advantage = (
                    safe_logit * threshold_weights
                ).sum(dim=-1) / threshold_weights.sum()
                residual_logit = parent_transition_advantage / 2.0
            elif self.decomposed_transition_head is not None:
                decomposed_transition_logits = (
                    self.decomposed_transition_head(pair_features).view(
                        batch_size, query_count, 2, 2
                    )
                )
                change_probability = (
                    decomposed_transition_logits[..., 0].sigmoid()
                )
                direction_log_odds = decomposed_transition_logits[..., 1]
                # Normalized expected transition utility.  Its sign matches
                # p(fix|change) - cost*p(break|change), but the bounded form is
                # stable.  Initial direction logits equal log(cost), making
                # step-zero utility exactly zero without changing the threshold.
                decomposed_fix_break_utility = (
                    change_probability * torch.tanh(
                        0.5 * (
                            direction_log_odds
                            - math.log(self.parent_transition_break_cost)
                        )
                    )
                )
                parent_transition_advantage = (
                    decomposed_fix_break_utility.min(dim=-1).values
                )
                residual_logit = 4.0 * parent_transition_advantage
                decomposed_counterfactual_costs = baseline.new_tensor((
                    1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                ))
                with torch.no_grad():
                    cf_utility = change_probability.detach().unsqueeze(-1) * (
                        torch.tanh(0.5 * (
                            direction_log_odds.detach().unsqueeze(-1)
                            - decomposed_counterfactual_costs.log().view(
                                1, 1, 1, -1
                            )
                        ))
                    )
                    cf_advantage = cf_utility.min(dim=2).values
                    cf_residual = self.max_delta * torch.tanh(
                        4.0 * cf_advantage
                    )
                    parent_mask = torch.zeros_like(valid_mask)
                    parent_mask[row, baseline_indices] = True
                    cf_candidate_mask = (
                        transition_candidate_mask & ~parent_mask
                    )
                    cf_residual = torch.where(
                        cf_candidate_mask.unsqueeze(-1),
                        cf_residual - self.candidate_promotion_margin,
                        torch.zeros_like(cf_residual),
                    ).masked_fill(~valid_mask.unsqueeze(-1), 0.0)
                    cf_scores = (
                        baseline.detach().unsqueeze(-1) + cf_residual
                    ).masked_fill(~valid_mask.unsqueeze(-1), -1e4)
                    decomposed_counterfactual_selected = cf_scores.argmax(
                        dim=1
                    )
            else:
                # A single pairwise advantage is trained by a row-level
                # repair-or-stay objective over the exact deployment scores.
                # Zero final-layer initialization preserves the parent
                # selector exactly before training.
                if self.use_decoupled_setwise_heads:
                    raw_setwise_safety = self.setwise_safety_head(
                        pair_features
                    )
                    if self.use_factorized_setwise_safety:
                        if self.use_factorized_setwise_risk_bound:
                            raw_setwise_safety_bounds = (
                                raw_setwise_safety.view(
                                    batch_size, query_count, 3, 2
                                )
                            )
                            if self.use_parent_referenced_safety:
                                # V79: make the safety zero point identifiable
                                # within each parent-conditioned row.  A shared
                                # row bias is subtracted exactly, while
                                # candidate-specific evidence and all pairwise
                                # gaps are preserved.  The immutable parent is
                                # the architectural reference, not a tuned
                                # deployment threshold.
                                parent_safety_bounds = (
                                    raw_setwise_safety_bounds[
                                        row, baseline_indices
                                    ].unsqueeze(1)
                                )
                                raw_setwise_safety_bounds = (
                                    raw_setwise_safety_bounds
                                    - parent_safety_bounds
                                )
                            if self.use_cost_calibrated_setwise_risk_bound:
                                raw_setwise_safety_bounds = torch.stack((
                                    raw_setwise_safety_bounds[..., 0],
                                    raw_setwise_safety_bounds[..., 1]
                                    + math.log(
                                        self.parent_transition_break_cost
                                    ),
                                ), dim=-1)
                            raw_setwise_safety_criteria = (
                                raw_setwise_safety_bounds.min(
                                    dim=-1
                                ).values
                            )
                        else:
                            raw_setwise_safety_bounds = None
                            raw_setwise_safety_criteria = raw_setwise_safety
                        aggregate_setwise_safety = (
                            raw_setwise_safety_criteria[..., 2:3]
                            if self.use_monotonic_box_safety_folding
                            else raw_setwise_safety_criteria.min(
                                dim=-1, keepdim=True
                            ).values
                        )
                    else:
                        aggregate_setwise_safety = raw_setwise_safety
                    raw_setwise_tier_advantage = torch.cat((
                        self.setwise_promotion_head(pair_features),
                        aggregate_setwise_safety,
                    ), dim=-1)
                else:
                    raw_setwise_tier_advantage = self.setwise_tier_head(
                        pair_features
                    )
                parent_mask = torch.zeros_like(valid_mask)
                parent_mask[row, baseline_indices] = True
                centered_candidate_mask = (
                    transition_candidate_mask & ~parent_mask
                )
                centered_weights = centered_candidate_mask.unsqueeze(
                    -1
                ).to(raw_setwise_tier_advantage.dtype)
                centered_count = centered_weights.sum(
                    dim=1, keepdim=True
                ).clamp(min=1.0)
                candidate_mean = (
                    raw_setwise_tier_advantage * centered_weights
                ).sum(dim=1, keepdim=True) / centered_count
                centered_advantage = torch.where(
                    centered_candidate_mask.unsqueeze(-1),
                    raw_setwise_tier_advantage - candidate_mean,
                    torch.zeros_like(raw_setwise_tier_advantage),
                )
                if raw_independent_joint_hazard is not None:
                    independent_joint_hazard = torch.where(
                        centered_candidate_mask,
                        raw_independent_joint_hazard,
                        torch.zeros_like(raw_independent_joint_hazard),
                    )
                else:
                    independent_joint_hazard = None
                if self.use_decoupled_setwise_heads:
                    # Promotion is candidate-centered, so it can only learn
                    # relative repair order. Safety remains absolute, allowing
                    # every hazardous candidate in a row to be vetoed.
                    if self.use_factorized_setwise_safety:
                        if self.use_factorized_setwise_risk_bound:
                            absolute_safety_bounds = torch.where(
                                centered_candidate_mask[
                                    ..., None, None
                                ],
                                raw_setwise_safety_bounds,
                                torch.zeros_like(raw_setwise_safety_bounds),
                            )
                            absolute_safety_criteria = (
                                absolute_safety_bounds.min(dim=-1).values
                            )
                        else:
                            absolute_safety_bounds = None
                            absolute_safety_criteria = torch.where(
                                centered_candidate_mask.unsqueeze(-1),
                                raw_setwise_safety,
                                torch.zeros_like(raw_setwise_safety),
                            )
                        if self.use_parent_non_degradation_certificate:
                            # V86: a promoted candidate must carry an explicit
                            # parent-relative non-degradation certificate for
                            # both Box thresholds. Use the calibrated point
                            # slack for Box@.25/.50, avoiding the conservative
                            # quantile guard that caused Box@.50 false rejects
                            # in V83, while Mask@.25 retains point+guard.
                            box_point_certificate = (
                                absolute_safety_bounds[..., :2, 0]
                                .min(dim=-1).values
                            )
                            mask_certificate = (
                                absolute_safety_criteria[..., 2]
                            )
                            absolute_safety = torch.minimum(
                                box_point_certificate, mask_certificate
                            )
                            if independent_joint_hazard is not None:
                                absolute_safety = torch.minimum(
                                    absolute_safety,
                                    independent_joint_hazard,
                                )
                        else:
                            absolute_safety = (
                                absolute_safety_criteria[..., 2]
                                if self.use_monotonic_box_safety_folding
                                else absolute_safety_criteria.min(
                                    dim=-1
                                ).values
                            )
                    else:
                        absolute_safety = torch.where(
                            centered_candidate_mask,
                            raw_setwise_tier_advantage[..., 1],
                            torch.zeros_like(
                                raw_setwise_tier_advantage[..., 1]
                            ),
                        )
                    setwise_tier_advantage = torch.stack((
                        centered_advantage[..., 0], absolute_safety,
                    ), dim=-1)
                else:
                    setwise_tier_advantage = centered_advantage
                parent_score = baseline[
                    row, baseline_indices
                ].unsqueeze(1)
                required_residual = (
                    parent_score - baseline
                    + self.candidate_promotion_margin
                ).clamp(min=0.0)
                setwise_tier_reachable_mask = (
                    transition_candidate_mask
                    & (required_residual < self.max_delta)
                )
                setwise_tier_reachable_mask[row, baseline_indices] = True
                # Convert the exact bounded deployment requirement
                #   base_q + max_delta*tanh(4*a_q) - margin > base_parent
                # into an unbounded signed training margin. Positive values
                # are equivalent to a branch being able to beat the parent.
                ratio = (
                    required_residual / self.max_delta
                ).clamp(max=1.0 - 1e-6)
                required_advantage = torch.atanh(ratio) / 4.0
                if self.use_setwise_safety_veto_gate:
                    # V74: promotion alone controls residual magnitude.
                    # Safety is an absolute veto and therefore no longer has
                    # to reproduce the candidate-to-parent score gap.  The
                    # straight-through gate has an exact hard forward while
                    # retaining a local safety gradient for positive
                    # promotion margins.
                    promotion_advantage = setwise_tier_advantage[..., 0]
                    safety_veto_margin = setwise_tier_advantage[..., 1]
                    if self.use_proposal_conditioned_safety:
                        # V78: strict two-stage deployment. Promotion proposes
                        # exactly one non-parent candidate using its exact
                        # boundary margin. Safety then verifies only that
                        # proposal; rejection falls back to the immutable
                        # parent instead of silently choosing another query.
                        proposal_margin = (
                            promotion_advantage - required_advantage
                        )
                        proposal_valid = (
                            centered_candidate_mask
                            & setwise_tier_reachable_mask
                        )
                        setwise_proposal_indices = proposal_margin.masked_fill(
                            ~proposal_valid, -1e4
                        ).argmax(dim=1)
                        has_proposal = proposal_valid.any(dim=1)
                        setwise_proposal_mask = torch.zeros_like(valid_mask)
                        setwise_proposal_mask.scatter_(
                            1,
                            setwise_proposal_indices.unsqueeze(1),
                            has_proposal.unsqueeze(1),
                        )
                        proposal_is_promotable = (
                            proposal_margin[
                                row, setwise_proposal_indices
                            ] > 0.0
                        ) & has_proposal
                        setwise_proposal_promotable_mask = (
                            setwise_proposal_mask
                            & proposal_is_promotable.unsqueeze(1)
                        )
                        hard_safe_gate = (
                            safety_veto_margin > 0.0
                        ).to(promotion_advantage.dtype)
                        soft_safe_gate = torch.sigmoid(
                            safety_veto_margin / 0.05
                        )
                        safe_gate = (
                            soft_safe_gate
                            + (hard_safe_gate - soft_safe_gate).detach()
                        )
                        gated_promotion = (
                            promotion_advantage
                            - (1.0 - safe_gate)
                            * F.relu(promotion_advantage)
                        )
                        parent_transition_advantage = torch.where(
                            setwise_proposal_promotable_mask,
                            gated_promotion,
                            torch.zeros_like(gated_promotion),
                        )
                    else:
                        hard_safe_gate = (
                            safety_veto_margin > 0.0
                        ).to(promotion_advantage.dtype)
                        soft_safe_gate = torch.sigmoid(
                            safety_veto_margin / 0.05
                        )
                        safe_gate = (
                            soft_safe_gate
                            + (hard_safe_gate - soft_safe_gate).detach()
                        )
                        parent_transition_advantage = (
                            promotion_advantage
                            - (1.0 - safe_gate)
                            * F.relu(promotion_advantage)
                        )
                else:
                    # Legacy V63--V73 use a conservative two-branch minimum:
                    # promotion and safety both determine residual magnitude.
                    hard_minimum = setwise_tier_advantage.min(dim=-1).values
                    smooth_temperature = 0.25
                    smooth_minimum = (
                        -smooth_temperature * torch.logsumexp(
                            -setwise_tier_advantage / smooth_temperature,
                            dim=-1,
                        )
                        + smooth_temperature * math.log(2.0)
                    )
                    parent_transition_advantage = (
                        smooth_minimum
                        + (hard_minimum - smooth_minimum).detach()
                    )
                residual_logit = 4.0 * parent_transition_advantage
                if self.use_setwise_safety_veto_gate:
                    setwise_tier_branch_scores = torch.stack((
                        setwise_tier_advantage[..., 0]
                        - required_advantage,
                        setwise_tier_advantage[..., 1],
                    ), dim=-1)
                else:
                    setwise_tier_branch_scores = (
                        setwise_tier_advantage
                        - required_advantage.unsqueeze(-1)
                    )
                setwise_tier_branch_scores = setwise_tier_branch_scores.masked_fill(
                    ~setwise_tier_reachable_mask.unsqueeze(-1), -1e4
                )
                setwise_tier_branch_scores = torch.where(
                    parent_mask.unsqueeze(-1),
                    torch.zeros_like(setwise_tier_branch_scores),
                    setwise_tier_branch_scores,
                )
                if independent_joint_hazard is not None:
                    setwise_independent_joint_hazard_scores = (
                        independent_joint_hazard.masked_fill(
                            ~setwise_tier_reachable_mask, -1e4
                        )
                    )
                    setwise_independent_joint_hazard_scores = torch.where(
                        parent_mask,
                        torch.zeros_like(
                            setwise_independent_joint_hazard_scores
                        ),
                        setwise_independent_joint_hazard_scores,
                    )
                if self.use_factorized_setwise_safety:
                    setwise_safety_criterion_scores = (
                        absolute_safety_criteria
                        if self.use_setwise_safety_veto_gate else
                        absolute_safety_criteria
                        - required_advantage.unsqueeze(-1)
                    ).masked_fill(
                        ~setwise_tier_reachable_mask.unsqueeze(-1), -1e4
                    )
                    setwise_safety_criterion_scores = torch.where(
                        parent_mask.unsqueeze(-1),
                        torch.zeros_like(setwise_safety_criterion_scores),
                        setwise_safety_criterion_scores,
                    )
                    if self.use_factorized_setwise_risk_bound:
                        setwise_safety_bound_scores = (
                            absolute_safety_bounds
                            if self.use_setwise_safety_veto_gate else
                            absolute_safety_bounds
                            - required_advantage[..., None, None]
                        ).masked_fill(
                            ~setwise_tier_reachable_mask[
                                ..., None, None
                            ], -1e4
                        )
                        setwise_safety_bound_scores = torch.where(
                            parent_mask[..., None, None],
                            torch.zeros_like(setwise_safety_bound_scores),
                            setwise_safety_bound_scores,
                        )
        elif self.factorized_hit_head is not None:
            # Dense absolute-hit supervision is substantially less sparse than
            # direct fix/break labels. At deployment, candidate and immutable
            # parent probabilities are factorized into transition risks.
            row = torch.arange(batch_size, device=features.device)
            factorized_hit_logits = ordinal_threshold_logits(
                self.factorized_hit_head(hidden)
            )
            factorized_hit_probabilities = factorized_hit_logits.sigmoid()
            parent_probability = factorized_hit_probabilities[
                row, baseline_indices
            ].unsqueeze(1)
            candidate_probability = factorized_hit_probabilities
            fix_probability = (
                (1.0 - parent_probability) * candidate_probability
            )
            break_probability = (
                parent_probability * (1.0 - candidate_probability)
            )
            factorized_fix_break_utility = (
                fix_probability
                - self.factorized_hit_break_cost * break_probability
            )
            if self.use_factorized_nested_dominance:
                # The metrics are nested: do not promote a candidate by
                # trading away either threshold. The minimum is positive iff
                # both predicted utilities are positive, and its magnitude is
                # the weakest predicted improvement.
                parent_transition_advantage = (
                    factorized_fix_break_utility.min(dim=-1).values
                )
            else:
                threshold_weights = factorized_fix_break_utility.new_tensor(
                    (1.0, 2.0)
                )
                parent_transition_advantage = (
                    factorized_fix_break_utility * threshold_weights
                ).sum(dim=-1) / threshold_weights.sum()
            residual_logit = 4.0 * parent_transition_advantage
        else:
            residual_logit = (
                direct_residual_logit
                + self.quality_score_weight * centered_quality
                + source_mix_residual_logit
            )
        learned_residual = self.max_delta * torch.tanh(
            residual_logit
        )
        learned_residual = learned_residual.masked_fill(~valid_mask, 0.0)
        residual = learned_residual
        if self.preserve_parent_score:
            row = torch.arange(batch_size, device=features.device)
            parent_mask = torch.zeros_like(valid_mask)
            parent_mask[row, baseline_indices] = True
            candidate_mask = (
                transition_candidate_mask & ~parent_mask
            )
            residual = torch.where(
                candidate_mask,
                learned_residual - self.candidate_promotion_margin,
                torch.zeros_like(learned_residual),
            ).masked_fill(~valid_mask, 0.0)
        scores = (baseline + residual).masked_fill(~valid_mask, -1e4)
        result = {
            "scores": scores,
            "residual": residual,
            "learned_residual": learned_residual,
            "residual_logit": residual_logit,
            "direct_residual_logit": direct_residual_logit,
            "raw_direct_residual_logit": raw_direct_residual_logit,
            "centered_quality": centered_quality,
            "baseline_rank": baseline_rank,
            "baseline_standardized": baseline_standardized,
            "baseline_indices": baseline_indices,
            "selected_indices": scores.argmax(dim=1),
            "box_logits": box_logits,
            "box_iou": box_iou,
            "mask_logits": mask_logits,
            "mask_iou": mask_iou,
            "quality": quality,
            "quality_evidence": quality_evidence,
            "valid_mask": valid_mask,
        }
        if parent_transition_logits is not None:
            result.update({
                "parent_transition_logits": parent_transition_logits,
                "parent_transition_advantage": (
                    parent_transition_advantage.masked_fill(~valid_mask, 0.0)
                ),
                "parent_transition_candidate_mask": (
                    transition_candidate_mask
                ),
            })
        if decomposed_transition_logits is not None:
            result.update({
                "decomposed_transition_logits": decomposed_transition_logits,
                "decomposed_fix_break_utility": (
                    decomposed_fix_break_utility.masked_fill(
                        ~valid_mask.unsqueeze(-1), 0.0
                    )
                ),
                "parent_transition_advantage": (
                    parent_transition_advantage.masked_fill(~valid_mask, 0.0)
                ),
                "decomposed_counterfactual_costs": (
                    decomposed_counterfactual_costs
                ),
                "decomposed_counterfactual_selected_indices": (
                    decomposed_counterfactual_selected
                ),
                "parent_transition_candidate_mask": transition_candidate_mask,
            })
        if setwise_tier_advantage is not None:
            result.update({
                "setwise_tier_advantage": (
                    setwise_tier_advantage.masked_fill(
                        ~valid_mask.unsqueeze(-1), 0.0
                    )
                ),
                "setwise_tier_branch_scores": setwise_tier_branch_scores,
                "setwise_tier_reachable_mask": (
                    setwise_tier_reachable_mask
                ),
                "setwise_decoupled_promotion_safety": scores.new_tensor(
                    float(self.use_decoupled_setwise_heads)
                ),
                "setwise_factorized_safety": scores.new_tensor(
                    float(self.use_factorized_setwise_safety)
                ),
                "setwise_factorized_risk_bound": scores.new_tensor(
                    float(self.use_factorized_setwise_risk_bound)
                ),
                "setwise_safety_veto_gate": scores.new_tensor(
                    float(self.use_setwise_safety_veto_gate)
                ),
                "setwise_cost_calibrated_risk_bound": scores.new_tensor(
                    float(self.use_cost_calibrated_setwise_risk_bound)
                ),
                "setwise_safety_slack_quantile_bound": scores.new_tensor(
                    float(self.use_setwise_safety_slack_quantile_bound)
                ),
                "setwise_safety_slack_pairwise_order": scores.new_tensor(
                    float(self.use_setwise_safety_slack_pairwise_order)
                ),
                "setwise_proposal_conditioned_safety": scores.new_tensor(
                    float(self.use_proposal_conditioned_safety)
                ),
                "setwise_parent_referenced_safety": scores.new_tensor(
                    float(self.use_parent_referenced_safety)
                ),
                "setwise_coupled_safe_repair_witness": scores.new_tensor(
                    float(self.use_coupled_safe_repair_witness)
                ),
                "setwise_bidirectional_coupled_boundary": scores.new_tensor(
                    float(self.use_bidirectional_coupled_boundary)
                ),
                "setwise_centered_coupled_separation": scores.new_tensor(
                    float(self.use_centered_coupled_separation)
                ),
                "setwise_hazard_conditioned_coupled_separation": (
                    scores.new_tensor(float(
                        self.use_hazard_conditioned_coupled_separation
                    ))
                ),
                "setwise_monotonic_box_safety_folding": scores.new_tensor(
                    float(self.use_monotonic_box_safety_folding)
                ),
                "setwise_same_candidate_branchwise_witness": (
                    scores.new_tensor(float(
                        self.use_same_candidate_branchwise_witness
                    ))
                ),
                "setwise_parent_non_degradation_certificate": (
                    scores.new_tensor(float(
                        self.use_parent_non_degradation_certificate
                    ))
                ),
                "setwise_criterion_responsible_hazard_attribution": (
                    scores.new_tensor(float(
                        self.use_criterion_responsible_hazard_attribution
                    ))
                ),
                "setwise_independent_joint_hazard_certificate": (
                    scores.new_tensor(float(
                        self.use_independent_joint_hazard_certificate
                    ))
                ),
                "setwise_frozen_raw_joint_hazard_features": (
                    scores.new_tensor(float(
                        self.use_frozen_raw_joint_hazard_features
                    ))
                ),
                "parent_transition_advantage": (
                    parent_transition_advantage.masked_fill(~valid_mask, 0.0)
                ),
                "parent_transition_candidate_mask": transition_candidate_mask,
            })
            if setwise_safety_criterion_scores is not None:
                result["setwise_safety_criterion_scores"] = (
                    setwise_safety_criterion_scores
                )
            if setwise_safety_bound_scores is not None:
                result["setwise_safety_bound_scores"] = (
                    setwise_safety_bound_scores
                )
            if setwise_independent_joint_hazard_scores is not None:
                result["setwise_independent_joint_hazard_scores"] = (
                    setwise_independent_joint_hazard_scores
                )
            if setwise_proposal_indices is not None:
                result.update({
                    "setwise_proposal_indices": setwise_proposal_indices,
                    "setwise_proposal_mask": setwise_proposal_mask,
                    "setwise_proposal_promotable_mask": (
                        setwise_proposal_promotable_mask
                    ),
                })
        if factorized_hit_logits is not None:
            # Compare deployment risk costs on the exact same learned weights
            # and batch. These selections are detached diagnostics only: they
            # neither affect gradients nor read validation-only information.
            counterfactual_costs = factorized_hit_probabilities.new_tensor((
                1.0, 1.25, 1.5, 2.0, 3.0, 4.0,
            ))
            with torch.no_grad():
                counterfactual_utility = (
                    fix_probability.detach().unsqueeze(-1)
                    - break_probability.detach().unsqueeze(-1)
                    * counterfactual_costs.view(1, 1, 1, -1)
                )
                counterfactual_advantage = (
                    counterfactual_utility.min(dim=2).values
                    if self.use_factorized_nested_dominance
                    else (
                        counterfactual_utility
                        * counterfactual_utility.new_tensor(
                            (1.0, 2.0)
                        ).view(1, 1, 2, 1)
                    ).sum(dim=2) / 3.0
                )
                counterfactual_residual = self.max_delta * torch.tanh(
                    4.0 * counterfactual_advantage
                )
                counterfactual_residual = torch.where(
                    candidate_mask.unsqueeze(-1),
                    counterfactual_residual
                    - self.candidate_promotion_margin,
                    torch.zeros_like(counterfactual_residual),
                ).masked_fill(~valid_mask.unsqueeze(-1), 0.0)
                counterfactual_scores = (
                    baseline.detach().unsqueeze(-1)
                    + counterfactual_residual
                ).masked_fill(~valid_mask.unsqueeze(-1), -1e4)
                counterfactual_selected = counterfactual_scores.argmax(dim=1)
            result.update({
                "factorized_hit_logits": factorized_hit_logits,
                "factorized_hit_probabilities": (
                    factorized_hit_probabilities.masked_fill(
                        ~valid_mask.unsqueeze(-1), 0.0
                    )
                ),
                "factorized_fix_break_utility": (
                    factorized_fix_break_utility.masked_fill(
                        ~valid_mask.unsqueeze(-1), 0.0
                    )
                ),
                "factorized_counterfactual_costs": counterfactual_costs,
                "factorized_counterfactual_selected_indices": (
                    counterfactual_selected
                ),
                "parent_transition_advantage": (
                    parent_transition_advantage.masked_fill(~valid_mask, 0.0)
                ),
                "parent_transition_candidate_mask": transition_candidate_mask,
            })
        if source_mix_out is not None:
            result.update(source_mix_out)
        if self.mask_calibration_head is not None:
            raw_mask_calibration = self.mask_calibration_head(hidden)
            mask_alpha_residual = self.max_mask_alpha_delta * torch.tanh(
                raw_mask_calibration[..., 0]
            )
            mask_logit_bias = self.max_mask_logit_bias * torch.tanh(
                raw_mask_calibration[..., 1]
            )
            mask_alpha_residual = mask_alpha_residual.masked_fill(
                ~valid_mask, 0.0
            )
            mask_logit_bias = mask_logit_bias.masked_fill(~valid_mask, 0.0)
            result.update({
                "mask_fusion_weights": (
                    mask_weights + mask_alpha_residual
                ).clamp(0.0, 1.0),
                "mask_alpha_residual": mask_alpha_residual,
                "mask_logit_bias": mask_logit_bias,
            })
        if self.spatial_mask_refiner is not None:
            result["mask_spatial_residuals"] = self.spatial_mask_refiner(
                spatial_query_features,
                spatial_superpoint_features,
                valid_mask=valid_mask,
            )
        elif (spatial_query_features is not None
              or spatial_superpoint_features is not None):
            raise ValueError(
                "spatial mask inputs require enabled spatial mask refiner"
            )
        return result


def _masked_mean(values, active):
    weights = active.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp(
        min=1.0
    )


def compute_joint_query_source_mix_alignment_loss(
        outputs, target_quality, sample_mask=None, temperature=0.25,
        query_relevance=None, query_focus_weight=0.0):
    """Align per-query source weights with target-free source reliability.

    Each source ranks the same query set.  During training only, the Box/Mask
    joint-quality rank supplies an oracle query rank.  A source is reliable for
    a query when its normalized rank agrees with that oracle rank.  The target
    distribution therefore depends on agreement rather than source identity,
    preserving routed-source permutation equivariance and dataset portability.
    """
    if (not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "source mix alignment temperature must be finite and positive"
        )
    query_focus_weight = _finite_non_negative(
        "query_focus_weight", query_focus_weight
    )
    if query_focus_weight > 1.0:
        raise ValueError("query_focus_weight must be at most one")
    required = (
        "source_mix_weights", "source_mix_ranks",
        "source_mix_validity", "valid_mask",
    )
    if not isinstance(outputs, dict) or any(
            name not in outputs for name in required):
        raise ValueError("adaptive source mix outputs are incomplete")
    weights = outputs["source_mix_weights"]
    ranks = outputs["source_mix_ranks"]
    source_validity = outputs["source_mix_validity"]
    valid = outputs["valid_mask"]
    if (not isinstance(weights, torch.Tensor) or weights.dim() != 3
            or not weights.is_floating_point()
            or not isinstance(ranks, torch.Tensor)
            or ranks.shape != weights.shape
            or not ranks.is_floating_point()
            or not isinstance(source_validity, torch.Tensor)
            or source_validity.dtype != torch.bool
            or source_validity.shape != weights.shape
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != weights.shape[:2]):
        raise ValueError(
            "source weights/ranks/validity must align as [B,Q,S]"
        )
    if any(value.device != weights.device for value in (
            ranks, source_validity, valid)):
        raise ValueError("adaptive source mix outputs must share a device")
    if (not isinstance(target_quality, torch.Tensor)
            or target_quality.shape != valid.shape
            or target_quality.device != weights.device
            or not target_quality.is_floating_point()
            or not bool(torch.isfinite(target_quality).all().item())):
        raise ValueError("target_quality must be finite floating [B,Q]")
    if (not bool(torch.isfinite(weights).all().item())
            or not bool(torch.isfinite(ranks).all().item())
            or bool(((weights < 0.0) | (weights > 1.0)).any().item())
            or bool(((ranks < 0.0) | (ranks > 1.0)).any().item())):
        raise ValueError("source weights and ranks must be finite in [0,1]")
    source_validity = source_validity & valid.unsqueeze(-1)
    if not bool(source_validity.any(dim=-1)[valid].all().item()):
        raise ValueError("each valid query needs a valid source")
    valid_sums = (
        weights * source_validity.to(weights.dtype)
    ).sum(dim=-1)[valid]
    if not bool(torch.allclose(
            valid_sums, torch.ones_like(valid_sums), atol=1e-5, rtol=1e-5)):
        raise ValueError("valid source weights must sum to one")
    if sample_mask is None:
        sample_mask = torch.ones(
            valid.shape[0], dtype=torch.bool, device=valid.device
        )
    if (not isinstance(sample_mask, torch.Tensor)
            or sample_mask.dtype != torch.bool
            or sample_mask.shape != (valid.shape[0],)
            or sample_mask.device != valid.device):
        raise ValueError("sample_mask must be bool [B]")
    active = valid & sample_mask.unsqueeze(1)
    if not bool(active.any().item()):
        zero = weights.sum() * 0.0
        return {
            "loss": zero,
            "target_top1_acc": zero.detach(),
            "target_effective_count_mean": zero.detach(),
        }

    target_rank = _rank_normalize(target_quality, valid).detach()
    target_logits = (
        -(ranks.detach() - target_rank.unsqueeze(-1)).abs()
        / float(temperature)
    ).masked_fill(~source_validity, -1e4)
    target_weights = F.softmax(target_logits, dim=-1)
    target_weights = (
        target_weights * source_validity.to(target_weights.dtype)
    )
    target_weights = target_weights / target_weights.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-6)
    cross_entropy = -(
        target_weights * weights.clamp(min=1e-8).log()
    ).sum(dim=-1)
    if query_focus_weight > 0.0:
        if (not isinstance(query_relevance, torch.Tensor)
                or query_relevance.shape != valid.shape
                or query_relevance.device != valid.device
                or not query_relevance.is_floating_point()
                or not bool(torch.isfinite(query_relevance).all().item())
                or bool((query_relevance < 0.0).any().item())):
            raise ValueError(
                "query_relevance must be finite non-negative floating [B,Q]"
            )
        focused = query_relevance * active.to(query_relevance.dtype)
        focused = focused / focused.sum(dim=1, keepdim=True).clamp(min=1e-8)
        focused = focused / sample_mask.float().sum().clamp(min=1.0)
        uniform = active.to(query_relevance.dtype)
        uniform = uniform / uniform.sum().clamp(min=1.0)
        query_weights = (
            (1.0 - query_focus_weight) * uniform
            + query_focus_weight * focused
        )
        loss = (cross_entropy * query_weights).sum()
    else:
        # Preserve the exact V49 objective when focus is disabled.
        loss = cross_entropy[active].mean()
    predicted_source = weights.masked_fill(
        ~source_validity, -1.0
    ).argmax(dim=-1)
    target_source = target_weights.masked_fill(
        ~source_validity, -1.0
    ).argmax(dim=-1)
    target_entropy = -(
        target_weights.clamp(min=1e-8).log() * target_weights
    ).sum(dim=-1)
    return {
        "loss": loss,
        "target_top1_acc": (
            predicted_source[active] == target_source[active]
        ).float().mean().detach(),
        "target_effective_count_mean": (
            target_entropy[active].exp().mean().detach()
        ),
    }


def compute_joint_query_quality_loss(
        outputs, box_ious, mask_ious, sample_mask=None, temperature=0.25,
        mask_weight=0.25, quality_loss_weight=1.0,
        anchor_loss_weight=0.5, anchor_margin=0.05,
        use_metric_aligned_utility=False,
        metric_utility_temperature=0.05,
        bidirectional_anchor=False, anchor_margin_050=0.10,
        pairwise_loss_weight=0.0,
        deploy_candidate_top_k=0, source_candidate_top_k=0,
        oracle_candidate_top_k=0,
        source_mix_loss_weight=0.0,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.0,
        listwise_loss_weight=1.0,
        transition_loss_weight=0.0,
        setwise_repair_boundary_loss_weight=0.0,
        setwise_negative_tail_loss_weight=0.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        factorized_hit_loss_weight=0.0,
        factorized_pair_loss_weight=0.0,
        transition_break_cost=4.0,
        transition_neutral_weight=0.25):
    """Train metric-aligned Top-1 ranking plus absolute quality evidence."""
    for name, value in (
            ("mask_weight", mask_weight),
            ("quality_loss_weight", quality_loss_weight),
            ("anchor_loss_weight", anchor_loss_weight),
            ("anchor_margin", anchor_margin),
            ("anchor_margin_050", anchor_margin_050),
            ("pairwise_loss_weight", pairwise_loss_weight),
            ("source_mix_loss_weight", source_mix_loss_weight),
            ("source_mix_query_focus_weight",
             source_mix_query_focus_weight),
            ("listwise_loss_weight", listwise_loss_weight),
            ("transition_loss_weight", transition_loss_weight),
            ("setwise_repair_boundary_loss_weight",
             setwise_repair_boundary_loss_weight),
            ("setwise_negative_tail_loss_weight",
             setwise_negative_tail_loss_weight),
            ("setwise_rank_loss_weight", setwise_rank_loss_weight),
            ("setwise_dense_safety_loss_weight",
             setwise_dense_safety_loss_weight),
            ("setwise_balanced_safety_loss_weight",
             setwise_balanced_safety_loss_weight),
            ("setwise_factorized_safety_loss_weight",
             setwise_factorized_safety_loss_weight),
            ("setwise_factorized_risk_bound_loss_weight",
             setwise_factorized_risk_bound_loss_weight),
            ("factorized_hit_loss_weight", factorized_hit_loss_weight),
            ("factorized_pair_loss_weight", factorized_pair_loss_weight),
            ("transition_break_cost", transition_break_cost),
            ("transition_neutral_weight", transition_neutral_weight)):
        _finite_non_negative(name, value)
    if float(transition_break_cost) <= 0.0:
        raise ValueError("transition_break_cost must be positive")
    if not isinstance(use_metric_aligned_utility, bool):
        raise ValueError("use_metric_aligned_utility must be boolean")
    if not isinstance(bidirectional_anchor, bool):
        raise ValueError("bidirectional_anchor must be boolean")
    if (not isinstance(metric_utility_temperature, (int, float))
            or isinstance(metric_utility_temperature, bool)
            or not math.isfinite(float(metric_utility_temperature))
            or float(metric_utility_temperature) <= 0.0):
        raise ValueError(
            "metric_utility_temperature must be finite and positive"
        )
    for name, value in (
            ("deploy_candidate_top_k", deploy_candidate_top_k),
            ("source_candidate_top_k", source_candidate_top_k),
            ("oracle_candidate_top_k", oracle_candidate_top_k)):
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 0):
            raise ValueError("{} must be a non-negative integer".format(name))
    if float(source_mix_query_focus_weight) > 1.0:
        raise ValueError(
            "source_mix_query_focus_weight must be at most one"
        )
    if (not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("temperature must be finite and positive")
    required = {
        "scores", "baseline_indices", "selected_indices", "box_logits",
        "box_iou", "mask_logits", "mask_iou", "valid_mask",
    }
    if not isinstance(outputs, dict) or not required.issubset(outputs):
        raise ValueError("reranker outputs are incomplete")
    if (not isinstance(source_mix_alignment_temperature, (int, float))
            or isinstance(source_mix_alignment_temperature, bool)
            or not math.isfinite(float(source_mix_alignment_temperature))
            or float(source_mix_alignment_temperature) <= 0.0):
        raise ValueError(
            "source_mix_alignment_temperature must be finite and positive"
        )
    scores = outputs["scores"]
    valid = outputs["valid_mask"]
    if (not isinstance(scores, torch.Tensor) or scores.dim() != 2
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != scores.shape):
        raise ValueError("scores and valid_mask must align as [B,Q]")
    if (not isinstance(box_ious, torch.Tensor) or box_ious.shape != scores.shape
            or not isinstance(mask_ious, torch.Tensor)
            or mask_ious.shape != scores.shape
            or box_ious.device != scores.device
            or mask_ious.device != scores.device):
        raise ValueError("box and mask IoUs must align with scores")
    anchors = outputs["baseline_indices"]
    if (not isinstance(anchors, torch.Tensor) or anchors.dtype != torch.long
            or anchors.shape != (scores.shape[0],)
            or anchors.device != scores.device):
        raise ValueError("baseline_indices must be int64 [B]")
    row = torch.arange(scores.shape[0], device=scores.device)
    if not bool(valid[row, anchors].all().item()):
        raise ValueError("baseline_indices must select valid queries")

    target_quality = (
        smooth_metric_aligned_query_utility(
            box_ious.float(), mask_ious.float(),
            temperature=float(metric_utility_temperature),
            mask_weight=mask_weight,
        )
        if use_metric_aligned_utility else joint_query_target_quality(
            box_ious.float(), mask_ious.float(), mask_weight=mask_weight
        )
    )
    if sample_mask is None:
        sample_mask = torch.ones(
            scores.shape[0], dtype=torch.bool, device=scores.device
        )
    if (not isinstance(sample_mask, torch.Tensor)
            or sample_mask.dtype != torch.bool
            or sample_mask.shape != (scores.shape[0],)
            or sample_mask.device != scores.device):
        raise ValueError("sample_mask must be bool [B]")

    deployment_candidate_mask = outputs.get(
        "parent_transition_candidate_mask"
    )
    if deployment_candidate_mask is not None:
        if (not isinstance(deployment_candidate_mask, torch.Tensor)
                or deployment_candidate_mask.dtype != torch.bool
                or deployment_candidate_mask.shape != valid.shape
                or deployment_candidate_mask.device != valid.device
                or bool((
                    deployment_candidate_mask & ~valid
                ).any().item())
                or not bool(deployment_candidate_mask[
                    row, anchors
                ].all().item())):
            raise ValueError(
                "parent transition candidate mask must be valid bool [B,Q]"
            )
        candidate_valid = deployment_candidate_mask
    else:
        candidate_valid = valid
    if any(value > 0 for value in (
            deploy_candidate_top_k, source_candidate_top_k,
            oracle_candidate_top_k)):
        candidate_valid = torch.zeros_like(valid)

        def include_topk(values, value_valid, top_k):
            if top_k <= 0:
                return
            count = min(int(top_k), values.shape[1])
            selected = values.detach().masked_fill(
                ~value_valid, float("-inf")
            ).topk(count, dim=1).indices
            selected_valid = value_valid.gather(1, selected)
            selected_mask = torch.zeros_like(candidate_valid)
            selected_mask.scatter_(1, selected, selected_valid)
            candidate_valid.logical_or_(selected_mask)

        include_topk(scores, valid, deploy_candidate_top_k)
        if source_candidate_top_k > 0:
            if ("source_mix_ranks" not in outputs
                    or "source_mix_validity" not in outputs):
                raise ValueError(
                    "source candidate selection requires source mix outputs"
                )
            source_ranks = outputs["source_mix_ranks"]
            source_validity = outputs["source_mix_validity"]
            if (not isinstance(source_ranks, torch.Tensor)
                    or not isinstance(source_validity, torch.Tensor)
                    or source_ranks.dim() != 3
                    or source_ranks.shape != source_validity.shape
                    or source_ranks.shape[:2] != scores.shape
                    or source_validity.dtype != torch.bool):
                raise ValueError(
                    "source mix candidates must align as [B,Q,S]"
                )
            for source_index in range(source_ranks.shape[-1]):
                include_topk(
                    source_ranks[..., source_index],
                    valid & source_validity[..., source_index],
                    source_candidate_top_k,
                )
        include_topk(target_quality, valid, oracle_candidate_top_k)
        candidate_valid[row, anchors] = True
        candidate_valid &= valid
        if deployment_candidate_mask is not None:
            candidate_valid &= deployment_candidate_mask

    active = candidate_valid & sample_mask.unsqueeze(1)
    if not bool(active.any().item()):
        zero = scores.sum() * 0.0
        return {
            "loss": zero, "listwise_loss": zero, "pairwise_loss": zero,
            "quality_loss": zero, "anchor_loss": zero,
            "transition_loss": zero, "factorized_hit_loss": zero,
            "factorized_pair_loss": zero,
            "protect_anchor_loss": zero, "repair_anchor_loss": zero,
            "source_mix_alignment_loss": zero, "stats": {},
        }

    scaled_target = (target_quality / float(temperature)).masked_fill(
        ~candidate_valid, -1e4
    )
    target_distribution = F.softmax(scaled_target, dim=1)
    listwise_rows = -(
        target_distribution
        * F.log_softmax(
            scores.masked_fill(~candidate_valid, -1e4), dim=1
        )
    ).sum(dim=1)
    listwise_loss = listwise_rows[sample_mask].mean()

    pairwise_rows = []
    if float(pairwise_loss_weight) > 0.0:
        for batch_index in sample_mask.nonzero(as_tuple=False).flatten():
            row_valid = candidate_valid[batch_index]
            row_scores = scores[batch_index, row_valid]
            row_target = target_quality[batch_index, row_valid]
            target_gap = row_target[:, None] - row_target[None, :]
            upper = torch.triu(
                torch.ones_like(target_gap, dtype=torch.bool), diagonal=1
            )
            comparable = upper & (target_gap.abs() > 1e-6)
            if bool(comparable.any().item()):
                score_gap = row_scores[:, None] - row_scores[None, :]
                gain = target_gap.abs()[comparable]
                violations = F.softplus(
                    -target_gap.sign()[comparable] * score_gap[comparable]
                )
                pairwise_rows.append(
                    (gain * violations).sum() / gain.sum().clamp(min=1e-6)
                )
    pairwise_loss = (
        torch.stack(pairwise_rows).mean()
        if pairwise_rows else scores.sum() * 0.0
    )

    transition_loss = scores.sum() * 0.0
    transition_stats = {}
    if float(transition_loss_weight) > 0.0:
        setwise_advantage = outputs.get("setwise_tier_advantage")
        if setwise_advantage is not None:
            branch_scores = outputs.get("setwise_tier_branch_scores")
            reachable_mask = outputs.get("setwise_tier_reachable_mask")
            if (not isinstance(setwise_advantage, torch.Tensor)
                    or setwise_advantage.shape != (
                        scores.shape[0], scores.shape[1], 2)
                    or setwise_advantage.device != scores.device
                    or not setwise_advantage.is_floating_point()
                    or not bool(torch.isfinite(
                        setwise_advantage
                    ).all().item())
                    or not isinstance(branch_scores, torch.Tensor)
                    or branch_scores.shape != setwise_advantage.shape
                    or branch_scores.device != scores.device
                    or not branch_scores.is_floating_point()
                    or not bool(torch.isfinite(
                        branch_scores.masked_fill(
                            ~valid.unsqueeze(-1), 0.0
                        )
                    ).all().item())
                    or not isinstance(reachable_mask, torch.Tensor)
                    or reachable_mask.dtype != torch.bool
                    or reachable_mask.shape != valid.shape
                    or reachable_mask.device != valid.device
                    or bool((reachable_mask & ~candidate_valid).any().item())
                    or not bool(reachable_mask[
                        row, anchors
                    ].all().item())):
                raise ValueError(
                    "positive setwise transition loss requires finite "
                    "advantages and branch scores [B,Q,2]"
                )
            transition_active = reachable_mask.clone()
            transition_active[row, anchors] = False
            decoupled_marker = outputs.get(
                "setwise_decoupled_promotion_safety"
            )
            decoupled_setwise = (
                isinstance(decoupled_marker, torch.Tensor)
                and decoupled_marker.numel() == 1
                and float(decoupled_marker.detach().item()) == 1.0
            )
            veto_gate_marker = outputs.get("setwise_safety_veto_gate")
            setwise_safety_veto_gate = (
                isinstance(veto_gate_marker, torch.Tensor)
                and veto_gate_marker.numel() == 1
                and float(veto_gate_marker.detach().item()) == 1.0
            )
            if setwise_safety_veto_gate and not decoupled_setwise:
                raise ValueError(
                    "setwise safety veto gate requires decoupled heads"
                )
            cost_calibration_marker = outputs.get(
                "setwise_cost_calibrated_risk_bound"
            )
            cost_calibrated_risk_bound = (
                isinstance(cost_calibration_marker, torch.Tensor)
                and cost_calibration_marker.numel() == 1
                and float(
                    cost_calibration_marker.detach().item()
                ) == 1.0
            )
            slack_quantile_marker = outputs.get(
                "setwise_safety_slack_quantile_bound"
            )
            safety_slack_quantile_bound = (
                isinstance(slack_quantile_marker, torch.Tensor)
                and slack_quantile_marker.numel() == 1
                and float(
                    slack_quantile_marker.detach().item()
                ) == 1.0
            )
            slack_pairwise_marker = outputs.get(
                "setwise_safety_slack_pairwise_order"
            )
            safety_slack_pairwise_order = (
                isinstance(slack_pairwise_marker, torch.Tensor)
                and slack_pairwise_marker.numel() == 1
                and float(
                    slack_pairwise_marker.detach().item()
                ) == 1.0
            )
            proposal_conditioned_marker = outputs.get(
                "setwise_proposal_conditioned_safety"
            )
            proposal_conditioned_safety = (
                isinstance(proposal_conditioned_marker, torch.Tensor)
                and proposal_conditioned_marker.numel() == 1
                and float(
                    proposal_conditioned_marker.detach().item()
                ) == 1.0
            )
            parent_referenced_marker = outputs.get(
                "setwise_parent_referenced_safety"
            )
            parent_referenced_safety = (
                isinstance(parent_referenced_marker, torch.Tensor)
                and parent_referenced_marker.numel() == 1
                and float(parent_referenced_marker.detach().item()) == 1.0
            )
            coupled_witness_marker = outputs.get(
                "setwise_coupled_safe_repair_witness"
            )
            coupled_safe_repair_witness = (
                isinstance(coupled_witness_marker, torch.Tensor)
                and coupled_witness_marker.numel() == 1
                and float(coupled_witness_marker.detach().item()) == 1.0
            )
            bidirectional_boundary_marker = outputs.get(
                "setwise_bidirectional_coupled_boundary"
            )
            bidirectional_coupled_boundary = (
                isinstance(bidirectional_boundary_marker, torch.Tensor)
                and bidirectional_boundary_marker.numel() == 1
                and float(
                    bidirectional_boundary_marker.detach().item()
                ) == 1.0
            )
            centered_separation_marker = outputs.get(
                "setwise_centered_coupled_separation"
            )
            centered_coupled_separation = (
                isinstance(centered_separation_marker, torch.Tensor)
                and centered_separation_marker.numel() == 1
                and float(centered_separation_marker.detach().item()) == 1.0
            )
            hazard_conditioned_marker = outputs.get(
                "setwise_hazard_conditioned_coupled_separation"
            )
            hazard_conditioned_coupled_separation = (
                isinstance(hazard_conditioned_marker, torch.Tensor)
                and hazard_conditioned_marker.numel() == 1
                and float(
                    hazard_conditioned_marker.detach().item()
                ) == 1.0
            )
            monotonic_box_folding_marker = outputs.get(
                "setwise_monotonic_box_safety_folding"
            )
            monotonic_box_safety_folding = (
                isinstance(monotonic_box_folding_marker, torch.Tensor)
                and monotonic_box_folding_marker.numel() == 1
                and float(
                    monotonic_box_folding_marker.detach().item()
                ) == 1.0
            )
            branchwise_witness_marker = outputs.get(
                "setwise_same_candidate_branchwise_witness"
            )
            same_candidate_branchwise_witness = (
                isinstance(branchwise_witness_marker, torch.Tensor)
                and branchwise_witness_marker.numel() == 1
                and float(
                    branchwise_witness_marker.detach().item()
                ) == 1.0
            )
            parent_certificate_marker = outputs.get(
                "setwise_parent_non_degradation_certificate"
            )
            parent_non_degradation_certificate = (
                isinstance(parent_certificate_marker, torch.Tensor)
                and parent_certificate_marker.numel() == 1
                and float(
                    parent_certificate_marker.detach().item()
                ) == 1.0
            )
            responsible_hazard_marker = outputs.get(
                "setwise_criterion_responsible_hazard_attribution"
            )
            criterion_responsible_hazard_attribution = (
                isinstance(responsible_hazard_marker, torch.Tensor)
                and responsible_hazard_marker.numel() == 1
                and float(
                    responsible_hazard_marker.detach().item()
                ) == 1.0
            )
            independent_hazard_marker = outputs.get(
                "setwise_independent_joint_hazard_certificate"
            )
            independent_joint_hazard_certificate = (
                isinstance(independent_hazard_marker, torch.Tensor)
                and independent_hazard_marker.numel() == 1
                and float(
                    independent_hazard_marker.detach().item()
                ) == 1.0
            )
            independent_joint_hazard_scores = outputs.get(
                "setwise_independent_joint_hazard_scores"
            )
            frozen_raw_marker = outputs.get(
                "setwise_frozen_raw_joint_hazard_features"
            )
            frozen_raw_joint_hazard_features = (
                isinstance(frozen_raw_marker, torch.Tensor)
                and frozen_raw_marker.numel() == 1
                and float(frozen_raw_marker.detach().item()) == 1.0
            )
            proposal_indices = outputs.get("setwise_proposal_indices")
            proposal_mask = outputs.get("setwise_proposal_mask")
            proposal_promotable_mask = outputs.get(
                "setwise_proposal_promotable_mask"
            )
            factorized_safety_marker = outputs.get(
                "setwise_factorized_safety"
            )
            factorized_setwise_safety = (
                isinstance(factorized_safety_marker, torch.Tensor)
                and factorized_safety_marker.numel() == 1
                and float(
                    factorized_safety_marker.detach().item()
                ) == 1.0
            )
            risk_bound_marker = outputs.get(
                "setwise_factorized_risk_bound"
            )
            factorized_setwise_risk_bound = (
                isinstance(risk_bound_marker, torch.Tensor)
                and risk_bound_marker.numel() == 1
                and float(risk_bound_marker.detach().item()) == 1.0
            )
            safety_criterion_scores = outputs.get(
                "setwise_safety_criterion_scores"
            )
            safety_bound_scores = outputs.get(
                "setwise_safety_bound_scores"
            )
            if factorized_setwise_safety:
                if (not isinstance(safety_criterion_scores, torch.Tensor)
                        or safety_criterion_scores.shape != (
                            scores.shape[0], scores.shape[1], 3
                        )
                        or safety_criterion_scores.device != scores.device
                        or not safety_criterion_scores.is_floating_point()
                        or not bool(torch.isfinite(
                            safety_criterion_scores.masked_fill(
                                ~valid.unsqueeze(-1), 0.0
                            )
                        ).all().item())):
                    raise ValueError(
                        "factorized setwise safety requires finite criterion "
                        "scores [B,Q,3]"
                    )
            if factorized_setwise_risk_bound:
                if (not factorized_setwise_safety
                        or not isinstance(safety_bound_scores, torch.Tensor)
                        or safety_bound_scores.shape != (
                            scores.shape[0], scores.shape[1], 3, 2
                        )
                        or safety_bound_scores.device != scores.device
                        or not safety_bound_scores.is_floating_point()
                        or not bool(torch.isfinite(
                            safety_bound_scores.masked_fill(
                                ~valid[..., None, None], 0.0
                            )
                        ).all().item())
                        or not torch.equal(
                            safety_criterion_scores,
                            safety_bound_scores.min(dim=-1).values,
                        )):
                    raise ValueError(
                        "factorized risk bound requires finite bound scores "
                        "[B,Q,3,2] whose minimum is the deployed criterion"
                    )
            if (cost_calibrated_risk_bound
                    and not factorized_setwise_risk_bound):
                raise ValueError(
                    "cost-calibrated risk bound requires factorized "
                    "risk-bound safety heads"
                )
            if (safety_slack_quantile_bound
                    and not factorized_setwise_risk_bound):
                raise ValueError(
                    "safety-slack quantile bound requires factorized "
                    "risk-bound safety heads"
                )
            if (safety_slack_quantile_bound
                    and cost_calibrated_risk_bound):
                raise ValueError(
                    "safety-slack quantile bound and cost calibration are "
                    "mutually exclusive"
                )
            if (safety_slack_pairwise_order
                    and not safety_slack_quantile_bound):
                raise ValueError(
                    "safety-slack pairwise order requires slack quantile "
                    "bound"
                )
            if (proposal_conditioned_safety
                    and not (safety_slack_pairwise_order
                             and setwise_safety_veto_gate)):
                raise ValueError(
                    "proposal-conditioned safety requires pairwise slack "
                    "order and the safety veto gate"
                )
            if (parent_referenced_safety
                    and not safety_slack_pairwise_order):
                raise ValueError(
                    "parent-referenced safety requires pairwise slack order"
                )
            if parent_referenced_safety and proposal_conditioned_safety:
                raise ValueError(
                    "parent-referenced and proposal-conditioned safety are "
                    "mutually exclusive"
                )
            if (coupled_safe_repair_witness
                    and not parent_referenced_safety):
                raise ValueError(
                    "coupled safe-repair witness requires parent-referenced "
                    "safety"
                )
            if (bidirectional_coupled_boundary
                    and not coupled_safe_repair_witness):
                raise ValueError(
                    "bidirectional coupled boundary requires coupled "
                    "safe-repair witness"
                )
            if (centered_coupled_separation
                    and not bidirectional_coupled_boundary):
                raise ValueError(
                    "centered coupled separation requires bidirectional "
                    "coupled boundary"
                )
            if (hazard_conditioned_coupled_separation
                    and not centered_coupled_separation):
                raise ValueError(
                    "hazard-conditioned coupled separation requires "
                    "centered coupled separation"
                )
            if (monotonic_box_safety_folding
                    and not hazard_conditioned_coupled_separation):
                raise ValueError(
                    "monotonic box-safety folding requires "
                    "hazard-conditioned coupled separation"
                )
            if (same_candidate_branchwise_witness
                    and not monotonic_box_safety_folding):
                raise ValueError(
                    "same-candidate branchwise witness requires monotonic "
                    "box-safety folding"
                )
            if (parent_non_degradation_certificate
                    and not same_candidate_branchwise_witness):
                raise ValueError(
                    "parent non-degradation certificate requires "
                    "same-candidate branchwise witness"
                )
            if (criterion_responsible_hazard_attribution
                    and not parent_non_degradation_certificate):
                raise ValueError(
                    "criterion-responsible hazard attribution requires "
                    "parent non-degradation certificate"
                )
            if (independent_joint_hazard_certificate
                    and not parent_non_degradation_certificate):
                raise ValueError(
                    "independent joint-hazard certificate requires parent "
                    "non-degradation certificate"
                )
            if (independent_joint_hazard_certificate
                    and criterion_responsible_hazard_attribution):
                raise ValueError(
                    "independent joint-hazard certificate and criterion-"
                    "responsible attribution are mutually exclusive"
                )
            if independent_joint_hazard_certificate:
                if (not isinstance(
                        independent_joint_hazard_scores, torch.Tensor)
                        or independent_joint_hazard_scores.shape != valid.shape
                        or independent_joint_hazard_scores.device
                        != scores.device
                        or not independent_joint_hazard_scores
                        .is_floating_point()
                        or not bool(torch.isfinite(
                            independent_joint_hazard_scores.masked_fill(
                                ~valid, 0.0
                            )
                        ).all().item())):
                    raise ValueError(
                        "independent joint-hazard scores must be finite [B,Q]"
                    )
            if (frozen_raw_joint_hazard_features
                    and not independent_joint_hazard_certificate):
                raise ValueError(
                    "frozen raw joint-hazard features require independent "
                    "joint-hazard certificate"
                )
            if proposal_conditioned_safety:
                if (not isinstance(proposal_indices, torch.Tensor)
                        or proposal_indices.dtype != torch.long
                        or proposal_indices.shape != (scores.shape[0],)
                        or proposal_indices.device != scores.device
                        or not isinstance(proposal_mask, torch.Tensor)
                        or proposal_mask.dtype != torch.bool
                        or proposal_mask.shape != valid.shape
                        or proposal_mask.device != scores.device
                        or not isinstance(
                            proposal_promotable_mask, torch.Tensor)
                        or proposal_promotable_mask.dtype != torch.bool
                        or proposal_promotable_mask.shape != valid.shape
                        or proposal_promotable_mask.device != scores.device
                        or bool((proposal_mask & ~transition_active).any().item())
                        or bool((
                            proposal_promotable_mask & ~proposal_mask
                        ).any().item())
                        or bool((
                            proposal_mask.sum(dim=1) > 1
                        ).any().item())
                        or not torch.equal(
                            proposal_mask.any(dim=1),
                            proposal_mask[row, proposal_indices],
                        )):
                    raise ValueError(
                        "proposal-conditioned safety requires one valid "
                        "non-parent proposal and aligned promotable mask"
                    )
            box_tier = (
                (box_ious > 0.25).long()
                + (box_ious > 0.50).long()
            )
            parent_tier = box_tier[row, anchors]
            # V69 safety hazards are exact evaluation regressions.  Mask@0.25
            # is explicitly protected; the low baseline Mask@0.50 remains free
            # to improve without vetoing useful box repairs.
            parent_box = box_ious[row, anchors].unsqueeze(1)
            parent_mask_iou = mask_ious[row, anchors].unsqueeze(1)
            criterion_hazards = torch.stack((
                transition_active
                & (parent_box > 0.25) & (box_ious <= 0.25),
                transition_active
                & (parent_box > 0.50) & (box_ious <= 0.50),
                transition_active
                & (parent_mask_iou > 0.25) & (mask_ious <= 0.25),
            ), dim=-1)
            hazardous = criterion_hazards.any(dim=-1)
            independent_joint_hazard_loss = scores.sum() * 0.0
            if independent_joint_hazard_certificate:
                # V88: learn a continuous lower quantile of the exact joint
                # parent-relative slack. A candidate is safe only when all
                # three criterion slacks are positive, so their hard minimum
                # is the deployment-aligned target with no OR-class shortcut.
                joint_slack_target = torch.stack((
                    torch.where(
                        parent_box > 0.25,
                        box_ious - 0.25,
                        0.25 - parent_box,
                    ) / 0.25,
                    torch.where(
                        parent_box > 0.50,
                        box_ious - 0.50,
                        0.50 - parent_box,
                    ) / 0.50,
                    torch.where(
                        parent_mask_iou > 0.25,
                        mask_ious - 0.25,
                        0.25 - parent_mask_iou,
                    ) / 0.25,
                ), dim=-1).min(dim=-1).values
                independent_active = transition_active
                independent_float = independent_active.float()
                independent_count = independent_float.sum(dim=1).clamp(
                    min=1.0
                )
                independent_rows = (
                    sample_mask & independent_active.any(dim=1)
                )
                independent_error = (
                    joint_slack_target - independent_joint_hazard_scores
                )
                independent_quantile = (
                    1.0 / (1.0 + transition_break_cost)
                )
                independent_values = (
                    (1.0 + transition_break_cost)
                    * torch.maximum(
                        independent_quantile * independent_error,
                        (independent_quantile - 1.0) * independent_error,
                    )
                    * independent_float
                )
                independent_row_loss = independent_values.sum(dim=1) / (
                    independent_count
                )
                if bool(independent_rows.any().item()):
                    independent_joint_hazard_loss = (
                        independent_row_loss[independent_rows].mean()
                    )
                independent_denominator = independent_float.sum().clamp(
                    min=1.0
                )
                transition_stats[
                    "setwise_independent_joint_hazard_loss"
                ] = independent_joint_hazard_loss.detach()
                transition_stats[
                    "setwise_independent_joint_hazard_target_negative_ratio"
                ] = (
                    ((joint_slack_target <= 0.0) & independent_active)
                    .float().sum() / independent_denominator
                ).detach()
                transition_stats[
                    "setwise_independent_joint_hazard_mae"
                ] = (
                    (
                        (independent_joint_hazard_scores - joint_slack_target)
                        .abs() * independent_float
                    ).sum() / independent_denominator
                ).detach()
                transition_stats[
                    "setwise_independent_joint_hazard_quantile_coverage"
                ] = (
                    ((joint_slack_target <= independent_joint_hazard_scores)
                     & independent_active).float().sum()
                    / independent_denominator
                ).detach()
                safe_active = independent_active & (joint_slack_target > 0.0)
                hazard_active = (
                    independent_active & (joint_slack_target <= 0.0)
                )
                transition_stats[
                    "setwise_independent_joint_hazard_safe_reject_ratio"
                ] = (
                    ((independent_joint_hazard_scores <= 0.0) & safe_active)
                    .float().sum() / safe_active.float().sum().clamp(min=1.0)
                ).detach()
                transition_stats[
                    "setwise_independent_joint_hazard_unsafe_accept_ratio"
                ] = (
                    ((independent_joint_hazard_scores > 0.0) & hazard_active)
                    .float().sum()
                    / hazard_active.float().sum().clamp(min=1.0)
                ).detach()
            improving = (
                transition_active
                & (box_tier > parent_tier.unsqueeze(1))
            )
            if decoupled_setwise:
                improving = improving & ~hazardous
            has_repair = improving.any(dim=1)
            best_tier = box_tier.masked_fill(~improving, -1).max(
                dim=1
            ).values
            best_repair = (
                improving & (box_tier == best_tier.unsqueeze(1))
            )
            setwise_terms = []
            setwise_repair_boundary_terms = []
            setwise_repair_boundary_eligible_ratios = []
            setwise_negative_tail_terms = []
            setwise_negative_tail_violation_ratios = []
            setwise_rank_terms = []
            setwise_dense_safety_terms = []
            setwise_dense_safety_violation_ratios = []
            setwise_balanced_safety_terms = []
            setwise_balanced_safety_safe_violation_ratios = []
            setwise_balanced_safety_hazard_violation_ratios = []
            setwise_factorized_safety_terms = []
            setwise_factorized_safety_safe_violation_ratios = []
            setwise_factorized_safety_hazard_violation_ratios = []
            setwise_factorized_risk_bound_point_terms = []
            setwise_factorized_risk_bound_guard_terms = []
            setwise_coupled_witness_terms = []
            setwise_bidirectional_boundary_terms = []
            setwise_centered_separation_terms = []
            setwise_branchwise_witness_terms = []
            repair_rows = sample_mask & has_repair
            stay_rows = sample_mask & ~has_repair
            setwise_margin = 0.02
            if coupled_safe_repair_witness:
                # V80: the promotion and safety objectives must be satisfied
                # by the same repair candidate. Independent row maxima can be
                # optimized by different queries and leave the deployed
                # intersection empty. The hard minimum is the exact joint
                # deployment margin; a smooth-min straight-through surrogate
                # keeps gradients on promotion and all safety criteria.
                promotion_boundary = branch_scores[..., 0]
                # V84: box tier improvement is monotonic in Box@.25/.50, so
                # safe-repair candidates cannot break either box threshold.
                # Their box hazards are folded into the promotion negative
                # set; the independent hard veto remains only for the
                # orthogonal Mask@.25 criterion. All three safety heads remain
                # trained and observable.
                if parent_non_degradation_certificate:
                    box025_certificate = safety_bound_scores[..., 0, 0]
                    box050_certificate = safety_bound_scores[..., 1, 0]
                    mask025_certificate = safety_criterion_scores[..., 2]
                    joint_components = torch.stack((
                        promotion_boundary,
                        box025_certificate,
                        box050_certificate,
                        mask025_certificate,
                        *(
                            (independent_joint_hazard_scores,)
                            if independent_joint_hazard_certificate else ()
                        ),
                    ), dim=-1)
                    safety_boundary = joint_components[..., 1:].min(
                        dim=-1
                    ).values
                else:
                    safety_boundary = (
                        safety_criterion_scores[..., 2]
                        if monotonic_box_safety_folding
                        else safety_criterion_scores.min(dim=-1).values
                    )
                    joint_components = torch.stack((
                        promotion_boundary, safety_boundary
                    ), dim=-1)
                hard_joint_margin = joint_components.min(dim=-1).values
                smooth_temperature = 0.05
                smooth_joint_margin = (
                    -smooth_temperature * torch.logsumexp(
                        -joint_components / smooth_temperature,
                        dim=-1,
                    )
                    + smooth_temperature * math.log(
                        float(joint_components.shape[-1])
                    )
                )
                joint_margin = (
                    smooth_joint_margin
                    + (hard_joint_margin - smooth_joint_margin).detach()
                )
                best_joint_repair = joint_margin.masked_fill(
                    ~best_repair, -1e4
                ).max(dim=1).values
                best_joint_repair_index = joint_margin.masked_fill(
                    ~best_repair, -1e4
                ).argmax(dim=1)
                chosen_repair_promotion = promotion_boundary[
                    row, best_joint_repair_index
                ]
                chosen_repair_safety = safety_boundary[
                    row, best_joint_repair_index
                ]
                if parent_non_degradation_certificate:
                    chosen_repair_box025 = box025_certificate[
                        row, best_joint_repair_index
                    ]
                    chosen_repair_box050 = box050_certificate[
                        row, best_joint_repair_index
                    ]
                    chosen_repair_mask025 = mask025_certificate[
                        row, best_joint_repair_index
                    ]
                    if independent_joint_hazard_certificate:
                        chosen_repair_independent_hazard = (
                            independent_joint_hazard_scores[
                                row, best_joint_repair_index
                            ]
                        )
                if (bool(repair_rows.any().item())
                        and not centered_coupled_separation):
                    setwise_coupled_witness_terms.append(F.softplus(
                        setwise_margin - best_joint_repair[repair_rows]
                    ).mean())
                if bidirectional_coupled_boundary:
                    # V81: V80 supplied only a positive existential witness.
                    # Reject the hardest candidate that is not a best safe
                    # repair on the exact same deployed joint boundary.
                    joint_negative_candidates = (
                        hazardous
                        if hazard_conditioned_coupled_separation
                        else transition_active & ~best_repair
                    )
                    joint_negative_rows = (
                        sample_mask & joint_negative_candidates.any(dim=1)
                    )
                    hardest_joint_negative = joint_margin.masked_fill(
                        ~joint_negative_candidates, -1e4
                    ).max(dim=1).values
                    if (bool(joint_negative_rows.any().item())
                            and not centered_coupled_separation):
                        setwise_bidirectional_boundary_terms.append(
                            F.softplus(
                                setwise_margin
                                + hardest_joint_negative[joint_negative_rows]
                            ).mean()
                        )
                    transition_stats[
                        "setwise_bidirectional_coupled_negative_margin"
                    ] = (
                        hardest_joint_negative[joint_negative_rows].mean()
                        if bool(joint_negative_rows.any().item())
                        else scores.new_zeros(())
                    ).detach()
                    transition_stats[
                        "setwise_bidirectional_coupled_negative_violation_ratio"
                    ] = (
                        (
                            hardest_joint_negative[joint_negative_rows] > 0.0
                        ).float().mean()
                        if bool(joint_negative_rows.any().item())
                        else scores.new_zeros(())
                    ).detach()
                    separation_rows = repair_rows & joint_negative_rows
                    separation = (
                        best_joint_repair - hardest_joint_negative
                    )
                    unpaired_repair_rows = (
                        repair_rows & ~joint_negative_rows
                        if hazard_conditioned_coupled_separation
                        else torch.zeros_like(repair_rows)
                    )
                    if (centered_coupled_separation
                            and not same_candidate_branchwise_witness):
                        # V82: replace the two independent absolute boundary
                        # terms on repair rows with a shift-invariant pair gap,
                        # then anchor its midpoint to the parent-reference zero.
                        # Stay rows have no positive witness, so their hardest
                        # candidate retains the one-sided negative boundary.
                        if bool(separation_rows.any().item()):
                            midpoint = 0.5 * (
                                best_joint_repair
                                + hardest_joint_negative
                            )
                            pair_loss = F.softplus(
                                2.0 * setwise_margin
                                - separation[separation_rows]
                            )
                            center_loss = midpoint[separation_rows].abs()
                            setwise_centered_separation_terms.append(
                                (pair_loss + center_loss).mean()
                            )
                            transition_stats[
                                "setwise_centered_coupled_midpoint_abs"
                            ] = midpoint[separation_rows].abs().mean().detach()
                        else:
                            transition_stats[
                                "setwise_centered_coupled_midpoint_abs"
                            ] = scores.new_zeros(())
                        # V83: paired centering is defined only when a real
                        # Box@.25/Box@.50/Mask@.25 regression exists. Repair
                        # rows without a hazard still need an existential
                        # positive witness; safe neutral candidates are left
                        # unconstrained instead of being mislabeled negative.
                        if bool(unpaired_repair_rows.any().item()):
                            setwise_centered_separation_terms.append(
                                F.softplus(
                                    setwise_margin
                                    - best_joint_repair[
                                        unpaired_repair_rows
                                    ]
                                ).mean()
                            )
                        stay_negative_rows = stay_rows & joint_negative_rows
                        if bool(stay_negative_rows.any().item()):
                            setwise_centered_separation_terms.append(
                                F.softplus(
                                    setwise_margin
                                    + hardest_joint_negative[
                                        stay_negative_rows
                                    ]
                                ).mean()
                            )
                    if same_candidate_branchwise_witness:
                        # V85: first select one oracle-safe repair by the exact
                        # deployed joint score, then put both of that same
                        # candidate's active deployment branches across zero.
                        # This preserves V80's same-candidate contract while
                        # preventing a single hard-min loss from alternating
                        # between promotion and Mask@.25 bottlenecks. Exact
                        # hazards retain a one-sided joint negative boundary.
                        if bool(repair_rows.any().item()):
                            positive_components = (
                                (
                                    chosen_repair_promotion,
                                    chosen_repair_box025,
                                    chosen_repair_box050,
                                    chosen_repair_mask025,
                                    *(
                                        (
                                            chosen_repair_independent_hazard,
                                        )
                                        if independent_joint_hazard_certificate
                                        else ()
                                    ),
                                )
                                if parent_non_degradation_certificate else
                                (
                                    chosen_repair_promotion,
                                    chosen_repair_safety,
                                )
                            )
                            positive_branch_loss = torch.stack(tuple(
                                F.softplus(
                                    setwise_margin - value[repair_rows]
                                )
                                for value in positive_components
                            ), dim=-1).mean(dim=-1)
                            setwise_branchwise_witness_terms.append(
                                positive_branch_loss.mean()
                            )
                        if (criterion_responsible_hazard_attribution
                                or independent_joint_hazard_certificate):
                            responsible_terms = []
                            stay_promotion_rows = (
                                stay_rows & transition_active.any(dim=1)
                            )
                            hardest_stay_promotion = (
                                promotion_boundary.masked_fill(
                                    ~transition_active, -1e4
                                ).max(dim=1).values
                            )
                            if bool(stay_promotion_rows.any().item()):
                                responsible_terms.append(F.softplus(
                                    setwise_margin
                                    + hardest_stay_promotion[
                                        stay_promotion_rows
                                    ]
                                ).mean())
                            transition_stats[
                                "setwise_criterion_responsible_stay_promotion_"
                                "margin"
                            ] = (
                                hardest_stay_promotion[
                                    stay_promotion_rows
                                ].mean()
                                if bool(stay_promotion_rows.any().item())
                                else scores.new_zeros(())
                            ).detach()
                            if criterion_responsible_hazard_attribution:
                                responsible_boundaries = (
                                    box025_certificate,
                                    box050_certificate,
                                    mask025_certificate,
                                )
                                for criterion_index, criterion_name in enumerate((
                                        "box025", "box050", "mask025")):
                                    criterion_hazard = criterion_hazards[
                                        ..., criterion_index
                                    ]
                                    criterion_rows = (
                                        sample_mask & criterion_hazard.any(dim=1)
                                    )
                                    criterion_hardest = responsible_boundaries[
                                        criterion_index
                                    ].masked_fill(
                                        ~criterion_hazard, -1e4
                                    ).max(dim=1).values
                                    if bool(criterion_rows.any().item()):
                                        responsible_terms.append(F.softplus(
                                            setwise_margin
                                            + criterion_hardest[criterion_rows]
                                        ).mean())
                                    transition_stats[
                                        "setwise_criterion_responsible_{}_margin"
                                        .format(criterion_name)
                                    ] = (
                                        criterion_hardest[criterion_rows].mean()
                                        if bool(criterion_rows.any().item())
                                        else scores.new_zeros(())
                                    ).detach()
                                    transition_stats[
                                        "setwise_criterion_responsible_{}_violation_"
                                        "ratio".format(criterion_name)
                                    ] = (
                                        (
                                            criterion_hardest[
                                                criterion_rows
                                            ] > 0
                                        ).float().mean()
                                        if bool(criterion_rows.any().item())
                                        else scores.new_zeros(())
                                    ).detach()
                            if responsible_terms:
                                setwise_branchwise_witness_terms.append(
                                    torch.stack(responsible_terms).mean()
                                )
                        elif bool(joint_negative_rows.any().item()):
                            setwise_branchwise_witness_terms.append(
                                F.softplus(
                                    setwise_margin
                                    + hardest_joint_negative[
                                        joint_negative_rows
                                    ]
                                ).mean()
                            )
                        transition_stats[
                            "setwise_same_candidate_branchwise_promotion_"
                            "margin"
                        ] = (
                            chosen_repair_promotion[repair_rows].mean()
                            if bool(repair_rows.any().item())
                            else scores.new_zeros(())
                        ).detach()
                        transition_stats[
                            "setwise_same_candidate_branchwise_mask_safety_"
                            "margin"
                        ] = (
                            chosen_repair_safety[repair_rows].mean()
                            if bool(repair_rows.any().item())
                            else scores.new_zeros(())
                        ).detach()
                        transition_stats[
                            "setwise_same_candidate_branchwise_recall"
                        ] = (
                            (
                                (chosen_repair_promotion[repair_rows] > 0.0)
                                & (chosen_repair_safety[repair_rows] > 0.0)
                            ).float().mean()
                            if bool(repair_rows.any().item())
                            else scores.new_zeros(())
                        ).detach()
                        if parent_non_degradation_certificate:
                            transition_stats[
                                "setwise_parent_non_degradation_box025_margin"
                            ] = chosen_repair_box025[
                                repair_rows
                            ].mean().detach() if bool(
                                repair_rows.any().item()
                            ) else scores.new_zeros(())
                            transition_stats[
                                "setwise_parent_non_degradation_box050_margin"
                            ] = chosen_repair_box050[
                                repair_rows
                            ].mean().detach() if bool(
                                repair_rows.any().item()
                            ) else scores.new_zeros(())
                            transition_stats[
                                "setwise_parent_non_degradation_mask025_margin"
                            ] = chosen_repair_mask025[
                                repair_rows
                            ].mean().detach() if bool(
                                repair_rows.any().item()
                            ) else scores.new_zeros(())
                            transition_stats[
                                "setwise_parent_non_degradation_recall"
                            ] = (
                                (
                                    (chosen_repair_promotion[repair_rows] > 0)
                                    & (chosen_repair_box025[repair_rows] > 0)
                                    & (chosen_repair_box050[repair_rows] > 0)
                                    & (chosen_repair_mask025[repair_rows] > 0)
                                ).float().mean()
                                if bool(repair_rows.any().item())
                                else scores.new_zeros(())
                            ).detach()
                    transition_stats[
                        "setwise_bidirectional_coupled_separation_margin"
                    ] = (
                        separation[separation_rows].mean()
                        if bool(separation_rows.any().item())
                        else scores.new_zeros(())
                    ).detach()
                    transition_stats[
                        "setwise_bidirectional_coupled_separation_recall"
                    ] = (
                        (
                            (best_joint_repair[separation_rows] > 0.0)
                            & (
                                hardest_joint_negative[separation_rows] < 0.0
                            )
                        ).float().mean()
                        if bool(separation_rows.any().item())
                        else scores.new_zeros(())
                    ).detach()
                    if centered_coupled_separation:
                        transition_stats[
                            "setwise_centered_coupled_margin_recall"
                        ] = (
                            (
                                separation[separation_rows]
                                > 2.0 * setwise_margin
                            ).float().mean()
                            if bool(separation_rows.any().item())
                            else scores.new_zeros(())
                        ).detach()
                    if hazard_conditioned_coupled_separation:
                        transition_stats[
                            "setwise_hazard_conditioned_coupled_pair_row_ratio"
                        ] = (
                            separation_rows.float().sum()
                            / repair_rows.float().sum().clamp(min=1.0)
                        ).detach()
                        transition_stats[
                            "setwise_hazard_conditioned_coupled_unpaired_"
                            "positive_row_ratio"
                        ] = (
                            unpaired_repair_rows.float().sum()
                            / repair_rows.float().sum().clamp(min=1.0)
                        ).detach()
                        transition_stats[
                            "setwise_hazard_conditioned_coupled_negative_ratio"
                        ] = (
                            joint_negative_candidates.float().sum()
                            / transition_active.float().sum().clamp(min=1.0)
                        ).detach()
                transition_stats[
                    "setwise_coupled_safe_repair_witness_margin"
                ] = (
                    best_joint_repair[repair_rows].mean()
                    if bool(repair_rows.any().item())
                    else scores.new_zeros(())
                ).detach()
                transition_stats[
                    "setwise_coupled_safe_repair_witness_recall"
                ] = (
                    (best_joint_repair[repair_rows] > 0.0).float().mean()
                    if bool(repair_rows.any().item())
                    else scores.new_zeros(())
                ).detach()
            for branch_index, suffix in enumerate(("025", "050")):
                branch_margin = branch_scores[..., branch_index]
                positive_score = branch_margin.masked_fill(
                    ~best_repair, -1e4
                ).max(dim=1).values
                if decoupled_setwise and branch_index == 1:
                    negative_candidates = hazardous
                else:
                    negative_candidates = (
                        transition_active & ~best_repair
                    )
                negative_score = branch_margin.masked_fill(
                    ~negative_candidates, -1e4
                ).max(dim=1).values
                branch_terms = []
                if bool(repair_rows.any().item()):
                    # At least one best-tier repair must cross the exact
                    # deployment boundary by a positive margin.
                    repair_boundary_term = F.softplus(
                        setwise_margin - positive_score[repair_rows]
                    ).mean()
                    branch_terms.append(repair_boundary_term)
                negative_rows = (
                    sample_mask & negative_candidates.any(dim=1)
                )
                has_negative = negative_candidates.any(dim=1)
                rank_rows = repair_rows & has_negative
                if decoupled_setwise and branch_index == 1:
                    rank_rows = torch.zeros_like(rank_rows)
                # V65c: extra boundary force is self-paced. A repair row is
                # eligible only after its best repair already outranks the
                # most dangerous non-repair. Eligibility is detached so this
                # path cannot game its own admission criterion.
                repair_boundary_eligible = repair_rows & ~has_negative
                if bool(rank_rows.any().item()):
                    # V64: rank inside the candidate set, excluding the
                    # parent. Unlike an absolute repair/stay boundary alone,
                    # this cannot be optimized by rejecting every candidate.
                    rank_gap = (
                        positive_score[rank_rows]
                        - negative_score[rank_rows]
                    )
                    setwise_rank_terms.append(F.softplus(
                        setwise_margin - rank_gap
                    ).mean())
                    rank_eligible = torch.zeros_like(repair_rows)
                    rank_eligible[rank_rows] = rank_gap.detach() > 0.0
                    repair_boundary_eligible = (
                        repair_boundary_eligible | rank_eligible
                    )
                    transition_stats[
                        "setwise_tier_branch_{}_rank_margin".format(suffix)
                    ] = rank_gap.mean().detach()
                    transition_stats[
                        "setwise_tier_branch_{}_rank_recall".format(suffix)
                    ] = (rank_gap > 0.0).float().mean().detach()
                else:
                    transition_stats[
                        "setwise_tier_branch_{}_rank_margin".format(suffix)
                    ] = scores.new_zeros(())
                    transition_stats[
                        "setwise_tier_branch_{}_rank_recall".format(suffix)
                    ] = scores.new_zeros(())
                eligible_denominator = repair_rows.float().sum().clamp(
                    min=1.0
                )
                eligible_ratio = (
                    repair_boundary_eligible.float().sum()
                    / eligible_denominator
                )
                setwise_repair_boundary_eligible_ratios.append(
                    eligible_ratio.detach()
                )
                transition_stats[
                    "setwise_tier_branch_{}_repair_boundary_eligible_ratio"
                    .format(suffix)
                ] = eligible_ratio.detach()
                if bool(repair_boundary_eligible.any().item()):
                    setwise_repair_boundary_terms.append(F.softplus(
                        setwise_margin
                        - positive_score[repair_boundary_eligible]
                    ).mean())
                penalize_negative_boundary = (
                    not decoupled_setwise
                    or (
                        branch_index == 1
                        and not criterion_responsible_hazard_attribution
                        and not independent_joint_hazard_certificate
                    )
                )
                if (bool(negative_rows.any().item())
                        and penalize_negative_boundary):
                    # Legacy branches reject every non-repair.  V69's safety
                    # head rejects only exact Box/Mask hazards, leaving the
                    # promotion head free to learn repair order.
                    branch_terms.append(F.softplus(
                        setwise_margin + negative_score[negative_rows]
                    ).mean())
                    if decoupled_setwise and branch_index == 1:
                        # V70: every exact hazard receives a gradient.  Average
                        # within each row first so scenes with many hazards do
                        # not dominate the batch, while the hardest-hazard term
                        # above preserves boundary pressure.
                        dense_values = (
                            F.softplus(setwise_margin + branch_margin)
                            * negative_candidates.float()
                        )
                        dense_row_loss = dense_values.sum(dim=1) / (
                            negative_candidates.float().sum(dim=1)
                            .clamp(min=1.0)
                        )
                        setwise_dense_safety_terms.append(
                            dense_row_loss[negative_rows].mean()
                        )
                        setwise_dense_safety_violation_ratios.append((
                            (branch_margin > 0.0) & negative_candidates
                        ).float().sum() / negative_candidates.float().sum(
                        ).clamp(min=1.0))
                    # V66: protect the whole dangerous tail, not just the
                    # current argmax. Otherwise shared updates can push the
                    # second/third non-repair over zero without receiving a
                    # gradient until it becomes the new maximum.
                    tail_k = min(4, branch_margin.shape[1])
                    tail_values, tail_indices = branch_margin.masked_fill(
                        ~negative_candidates, -1e4
                    ).topk(tail_k, dim=1)
                    tail_valid = negative_candidates.gather(
                        1, tail_indices
                    )
                    tail_row_loss = (
                        F.softplus(setwise_margin + tail_values)
                        * tail_valid.float()
                    ).sum(dim=1) / tail_valid.float().sum(
                        dim=1
                    ).clamp(min=1.0)
                    setwise_negative_tail_terms.append(
                        tail_row_loss[negative_rows].mean()
                    )
                    setwise_negative_tail_violation_ratios.append(
                        (
                            (tail_values > 0.0) & tail_valid
                        ).float().sum()
                        / tail_valid.float().sum().clamp(min=1.0)
                    )
                if decoupled_setwise and branch_index == 1:
                    # V71: a safety classifier needs evidence on both sides
                    # of the veto boundary. V70 supplied dense hazard
                    # negatives only, which admits an all-negative safety
                    # head. Average candidates within each row, rows within
                    # each class, then the safe/hazard classes equally.
                    safe_candidates = transition_active & ~hazardous
                    safe_rows = sample_mask & safe_candidates.any(dim=1)
                    balanced_class_terms = []
                    if bool(safe_rows.any().item()):
                        safe_values = (
                            F.softplus(setwise_margin - branch_margin)
                            * safe_candidates.float()
                        )
                        safe_row_loss = safe_values.sum(dim=1) / (
                            safe_candidates.float().sum(dim=1)
                            .clamp(min=1.0)
                        )
                        balanced_class_terms.append(
                            safe_row_loss[safe_rows].mean()
                        )
                        setwise_balanced_safety_safe_violation_ratios.append((
                            (branch_margin <= 0.0) & safe_candidates
                        ).float().sum() / safe_candidates.float().sum(
                        ).clamp(min=1.0))
                    if bool(negative_rows.any().item()):
                        hazard_values = (
                            F.softplus(setwise_margin + branch_margin)
                            * negative_candidates.float()
                        )
                        hazard_row_loss = hazard_values.sum(dim=1) / (
                            negative_candidates.float().sum(dim=1)
                            .clamp(min=1.0)
                        )
                        balanced_class_terms.append(
                            hazard_row_loss[negative_rows].mean()
                        )
                        setwise_balanced_safety_hazard_violation_ratios.append((
                            (branch_margin > 0.0) & negative_candidates
                        ).float().sum() / negative_candidates.float().sum(
                        ).clamp(min=1.0))
                    if balanced_class_terms:
                        setwise_balanced_safety_terms.append(
                            torch.stack(balanced_class_terms).mean()
                        )
                if branch_terms:
                    setwise_terms.append(torch.stack(branch_terms).mean())

                predicted = branch_margin.masked_fill(
                    ~reachable_mask, -1e4
                ).argmax(dim=1)
                repair_correct = best_repair[row, predicted]
                transition_stats[
                    "setwise_tier_branch_{}_repair_recall".format(suffix)
                ] = (
                    repair_correct[repair_rows].float().mean()
                    if bool(repair_rows.any().item())
                    else scores.new_zeros(())
                ).detach()
                transition_stats[
                    "setwise_tier_branch_{}_stay_recall".format(suffix)
                ] = (
                    (branch_margin[stay_rows].masked_fill(
                        ~transition_active[stay_rows], -1e4
                    ).max(dim=1).values < 0.0).float().mean()
                    if bool(stay_rows.any().item())
                    else scores.new_zeros(())
                ).detach()
                transition_stats[
                    "setwise_tier_branch_{}_positive_margin".format(suffix)
                ] = (
                    positive_score[repair_rows].mean()
                    if bool(repair_rows.any().item())
                    else scores.new_zeros(())
                ).detach()
                transition_stats[
                    "setwise_tier_branch_{}_negative_margin".format(suffix)
                ] = (
                    negative_score[negative_rows].mean()
                    if bool(negative_rows.any().item())
                    else scores.new_zeros(())
                ).detach()
            if factorized_setwise_safety:
                # V72: learn one safety boundary per protected evaluation
                # criterion. A candidate may be safe for one criterion and
                # hazardous for another; the deployment minimum combines the
                # three independent vetoes without collapsing their labels.
                for criterion_index, criterion_name in enumerate((
                        "box025", "box050", "mask025")):
                    criterion_margin = safety_criterion_scores[
                        ..., criterion_index
                    ]
                    criterion_hazard = criterion_hazards[
                        ..., criterion_index
                    ]
                    criterion_safe = transition_active & ~criterion_hazard
                    criterion_class_terms = []
                    safe_rows = sample_mask & criterion_safe.any(dim=1)
                    if bool(safe_rows.any().item()):
                        safe_values = (
                            F.softplus(setwise_margin - criterion_margin)
                            * criterion_safe.float()
                        )
                        safe_row_loss = safe_values.sum(dim=1) / (
                            criterion_safe.float().sum(dim=1).clamp(min=1.0)
                        )
                        criterion_class_terms.append(
                            safe_row_loss[safe_rows].mean()
                        )
                        safe_violation = (
                            ((criterion_margin <= 0.0) & criterion_safe)
                            .float().sum()
                            / criterion_safe.float().sum().clamp(min=1.0)
                        )
                    else:
                        safe_violation = scores.new_zeros(())
                    hazard_rows = sample_mask & criterion_hazard.any(dim=1)
                    if bool(hazard_rows.any().item()):
                        hazard_values = (
                            F.softplus(setwise_margin + criterion_margin)
                            * criterion_hazard.float()
                        )
                        hazard_row_loss = hazard_values.sum(dim=1) / (
                            criterion_hazard.float().sum(dim=1)
                            .clamp(min=1.0)
                        )
                        criterion_class_terms.append(
                            hazard_row_loss[hazard_rows].mean()
                        )
                        hazard_violation = (
                            ((criterion_margin > 0.0) & criterion_hazard)
                            .float().sum()
                            / criterion_hazard.float().sum().clamp(min=1.0)
                        )
                    else:
                        hazard_violation = scores.new_zeros(())
                    if criterion_class_terms:
                        setwise_factorized_safety_terms.append(
                            torch.stack(criterion_class_terms).mean()
                        )
                    setwise_factorized_safety_safe_violation_ratios.append(
                        safe_violation
                    )
                    setwise_factorized_safety_hazard_violation_ratios.append(
                        hazard_violation
                    )
                    transition_stats[
                        "setwise_factorized_safety_{}_safe_violation_ratio"
                        .format(criterion_name)
                    ] = safe_violation.detach()
                    transition_stats[
                        "setwise_factorized_safety_{}_hazard_violation_ratio"
                        .format(criterion_name)
                    ] = hazard_violation.detach()
                    if factorized_setwise_risk_bound:
                        point_margin = safety_bound_scores[
                            ..., criterion_index, 0
                        ]
                        guard_margin = safety_bound_scores[
                            ..., criterion_index, 1
                        ]
                        if safety_slack_quantile_bound:
                            # V76: supervise the signed distance to the exact
                            # parent-relative break boundary instead of a
                            # dataset-level safe/hazard class prior.  When the
                            # parent is currently correct, the candidate's
                            # distance to the metric threshold is the safety
                            # slack.  When the parent is currently incorrect,
                            # no candidate can cause a break, so the parent's
                            # distance below the threshold is a positive slack
                            # shared by the row.  Dividing by the threshold
                            # makes the three criteria dimensionless while
                            # preserving the deployment zero boundary.
                            if criterion_index == 0:
                                candidate_metric = box_ious
                                parent_metric = parent_box
                                criterion_threshold = 0.25
                            elif criterion_index == 1:
                                candidate_metric = box_ious
                                parent_metric = parent_box
                                criterion_threshold = 0.50
                            else:
                                candidate_metric = mask_ious
                                parent_metric = parent_mask_iou
                                criterion_threshold = 0.25
                            safety_slack_target = torch.where(
                                parent_metric > criterion_threshold,
                                candidate_metric - criterion_threshold,
                                criterion_threshold - parent_metric,
                            ) / criterion_threshold
                            calibration_active = (
                                transition_active & proposal_mask
                                if proposal_conditioned_safety else
                                transition_active
                            )
                            active_float = calibration_active.float()
                            active_rows = (
                                sample_mask & calibration_active.any(dim=1)
                            )
                            active_count = active_float.sum(dim=1).clamp(
                                min=1.0
                            )
                            point_row_loss = (
                                (point_margin - safety_slack_target).abs()
                                * active_float
                            ).sum(dim=1) / active_count
                            # tau=1/(1+cost) is the Bayes boundary for a unit
                            # safe cost and `cost` break penalty. Multiplying
                            # the pinball loss by (1+cost) gives exact 1:cost
                            # under/over-estimation slopes without a tuned
                            # deployment threshold.
                            quantile = 1.0 / (1.0 + transition_break_cost)
                            quantile_error = (
                                safety_slack_target - guard_margin
                            )
                            quantile_values = (
                                (1.0 + transition_break_cost)
                                * torch.maximum(
                                    quantile * quantile_error,
                                    (quantile - 1.0) * quantile_error,
                                )
                                * active_float
                            )
                            quantile_row_loss = quantile_values.sum(
                                dim=1
                            ) / active_count
                            point_term = (
                                point_row_loss[active_rows].mean()
                                if bool(active_rows.any().item())
                                else scores.sum() * 0.0
                            )
                            guard_term = (
                                quantile_row_loss[active_rows].mean()
                                if bool(active_rows.any().item())
                                else scores.sum() * 0.0
                            )
                            if safety_slack_pairwise_order:
                                # V77: absolute regression/quantile objectives
                                # admit a row-wise constant shortcut.  Exact
                                # safe-vs-hazard slack differences cancel every
                                # shared row bias and force candidate-specific
                                # ordering.  Regressing the observed continuous
                                # gap supplies its scale without a tuned margin.
                                pair_mask = (
                                    criterion_safe.unsqueeze(2)
                                    & criterion_hazard.unsqueeze(1)
                                )
                                pair_float = pair_mask.float()
                                pair_count = pair_float.sum(
                                    dim=(1, 2)
                                ).clamp(min=1.0)
                                pair_rows = (
                                    sample_mask
                                    & pair_mask.flatten(1).any(dim=1)
                                )
                                target_pair_gap = (
                                    safety_slack_target.unsqueeze(2)
                                    - safety_slack_target.unsqueeze(1)
                                )
                                point_pair_gap = (
                                    point_margin.unsqueeze(2)
                                    - point_margin.unsqueeze(1)
                                )
                                guard_pair_gap = (
                                    guard_margin.unsqueeze(2)
                                    - guard_margin.unsqueeze(1)
                                )
                                point_pair_row_loss = (
                                    (point_pair_gap - target_pair_gap).abs()
                                    * pair_float
                                ).sum(dim=(1, 2)) / pair_count
                                guard_pair_row_loss = (
                                    (guard_pair_gap - target_pair_gap).abs()
                                    * pair_float
                                ).sum(dim=(1, 2)) / pair_count
                                if bool(pair_rows.any().item()):
                                    point_pair_loss = point_pair_row_loss[
                                        pair_rows
                                    ].mean()
                                    guard_pair_loss = guard_pair_row_loss[
                                        pair_rows
                                    ].mean()
                                    point_term = point_term + point_pair_loss
                                    guard_term = guard_term + guard_pair_loss
                                else:
                                    point_pair_loss = scores.sum() * 0.0
                                    guard_pair_loss = scores.sum() * 0.0
                                transition_stats[
                                    "setwise_safety_slack_{}_point_pair_mae"
                                    .format(criterion_name)
                                ] = point_pair_loss.detach()
                                transition_stats[
                                    "setwise_safety_slack_{}_guard_pair_mae"
                                    .format(criterion_name)
                                ] = guard_pair_loss.detach()
                                transition_stats[
                                    "setwise_safety_slack_{}_point_pair_order_"
                                    "accuracy".format(criterion_name)
                                ] = (
                                    ((point_pair_gap > 0.0) & pair_mask)
                                    .float().sum()
                                    / pair_float.sum().clamp(min=1.0)
                                ).detach()
                                transition_stats[
                                    "setwise_safety_slack_{}_guard_pair_order_"
                                    "accuracy".format(criterion_name)
                                ] = (
                                    ((guard_pair_gap > 0.0) & pair_mask)
                                    .float().sum()
                                    / pair_float.sum().clamp(min=1.0)
                                ).detach()
                            if bool(active_rows.any().item()):
                                setwise_factorized_risk_bound_point_terms.append(
                                    point_term
                                )
                                setwise_factorized_risk_bound_guard_terms.append(
                                    guard_term
                                )
                            transition_stats[
                                "setwise_safety_slack_{}_target_negative_ratio"
                                .format(criterion_name)
                            ] = (
                                ((safety_slack_target <= 0.0)
                                 & calibration_active).float().sum()
                                / active_float.sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_safety_slack_{}_point_mae".format(
                                    criterion_name
                                )
                            ] = (
                                ((point_margin - safety_slack_target).abs()
                                 * active_float).sum()
                                / active_float.sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_safety_slack_{}_quantile_coverage"
                                .format(criterion_name)
                            ] = (
                                ((safety_slack_target <= guard_margin)
                                 & calibration_active).float().sum()
                                / active_float.sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_point_safe_"
                                "violation_ratio".format(criterion_name)
                            ] = (
                                ((point_margin <= 0.0) & criterion_safe)
                                .float().sum()
                                / criterion_safe.float().sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_guard_safe_"
                                "violation_ratio".format(criterion_name)
                            ] = (
                                ((guard_margin <= 0.0) & criterion_safe)
                                .float().sum()
                                / criterion_safe.float().sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_point_"
                                "hazard_violation_ratio".format(
                                    criterion_name
                                )
                            ] = (
                                ((point_margin > 0.0) & criterion_hazard)
                                .float().sum()
                                / criterion_hazard.float().sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_guard_"
                                "hazard_violation_ratio".format(
                                    criterion_name
                                )
                            ] = (
                                ((guard_margin > 0.0) & criterion_hazard)
                                .float().sum()
                                / criterion_hazard.float().sum().clamp(min=1.0)
                            ).detach()
                            continue
                        guard_training_margin = (
                            guard_margin - math.log(transition_break_cost)
                            if cost_calibrated_risk_bound else guard_margin
                        )
                        point_class_terms = []
                        guard_weighted_terms = []
                        guard_weights = []
                        if bool(safe_rows.any().item()):
                            point_safe_values = (
                                F.softplus(setwise_margin - point_margin)
                                * criterion_safe.float()
                            )
                            point_safe_loss = (
                                point_safe_values.sum(dim=1)
                                / criterion_safe.float().sum(dim=1)
                                .clamp(min=1.0)
                            )[safe_rows].mean()
                            guard_safe_values = (
                                F.softplus(
                                    setwise_margin - guard_training_margin
                                )
                                * criterion_safe.float()
                            )
                            guard_safe_loss = (
                                guard_safe_values.sum(dim=1)
                                / criterion_safe.float().sum(dim=1)
                                .clamp(min=1.0)
                            )[safe_rows].mean()
                            point_class_terms.append(point_safe_loss)
                            guard_weighted_terms.append(guard_safe_loss)
                            guard_weights.append(1.0)
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_point_safe_"
                                "violation_ratio".format(criterion_name)
                            ] = (
                                ((point_margin <= 0.0) & criterion_safe)
                                .float().sum()
                                / criterion_safe.float().sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_guard_safe_"
                                "violation_ratio".format(criterion_name)
                            ] = (
                                ((guard_margin <= 0.0) & criterion_safe)
                                .float().sum()
                                / criterion_safe.float().sum().clamp(min=1.0)
                            ).detach()
                        if bool(hazard_rows.any().item()):
                            point_hazard_values = (
                                F.softplus(setwise_margin + point_margin)
                                * criterion_hazard.float()
                            )
                            point_hazard_loss = (
                                point_hazard_values.sum(dim=1)
                                / criterion_hazard.float().sum(dim=1)
                                .clamp(min=1.0)
                            )[hazard_rows].mean()
                            guard_hazard_values = (
                                F.softplus(
                                    setwise_margin + guard_training_margin
                                )
                                * criterion_hazard.float()
                            )
                            guard_hazard_loss = (
                                guard_hazard_values.sum(dim=1)
                                / criterion_hazard.float().sum(dim=1)
                                .clamp(min=1.0)
                            )[hazard_rows].mean()
                            point_class_terms.append(point_hazard_loss)
                            guard_weighted_terms.append(guard_hazard_loss)
                            guard_weights.append(float(transition_break_cost))
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_point_"
                                "hazard_violation_ratio".format(
                                    criterion_name
                                )
                            ] = (
                                ((point_margin > 0.0) & criterion_hazard)
                                .float().sum()
                                / criterion_hazard.float().sum().clamp(min=1.0)
                            ).detach()
                            transition_stats[
                                "setwise_factorized_risk_bound_{}_guard_"
                                "hazard_violation_ratio".format(
                                    criterion_name
                                )
                            ] = (
                                ((guard_margin > 0.0) & criterion_hazard)
                                .float().sum()
                                / criterion_hazard.float().sum().clamp(min=1.0)
                            ).detach()
                        if point_class_terms:
                            setwise_factorized_risk_bound_point_terms.append(
                                torch.stack(point_class_terms).mean()
                            )
                        if guard_weighted_terms:
                            guard_weight_tensor = scores.new_tensor(
                                guard_weights
                            )
                            setwise_factorized_risk_bound_guard_terms.append(
                                (
                                    torch.stack(guard_weighted_terms)
                                    * guard_weight_tensor
                                ).sum() / guard_weight_tensor.sum()
                            )
            if setwise_terms:
                transition_loss = torch.stack(setwise_terms).mean()
            setwise_repair_boundary_loss = (
                torch.stack(setwise_repair_boundary_terms).mean()
                if setwise_repair_boundary_terms
                else scores.sum() * 0.0
            )
            setwise_negative_tail_loss = (
                torch.stack(setwise_negative_tail_terms).mean()
                if setwise_negative_tail_terms
                else scores.sum() * 0.0
            )
            setwise_rank_loss = (
                torch.stack(setwise_rank_terms).mean()
                if setwise_rank_terms else scores.sum() * 0.0
            )
            setwise_dense_safety_loss = (
                torch.stack(setwise_dense_safety_terms).mean()
                if setwise_dense_safety_terms else scores.sum() * 0.0
            )
            setwise_balanced_safety_loss = (
                torch.stack(setwise_balanced_safety_terms).mean()
                if setwise_balanced_safety_terms else scores.sum() * 0.0
            )
            setwise_factorized_safety_loss = (
                torch.stack(setwise_factorized_safety_terms).mean()
                if setwise_factorized_safety_terms else scores.sum() * 0.0
            )
            setwise_factorized_risk_bound_point_loss = (
                torch.stack(
                    setwise_factorized_risk_bound_point_terms
                ).mean()
                if setwise_factorized_risk_bound_point_terms
                else scores.sum() * 0.0
            )
            setwise_factorized_risk_bound_guard_loss = (
                torch.stack(
                    setwise_factorized_risk_bound_guard_terms
                ).mean()
                if setwise_factorized_risk_bound_guard_terms
                else scores.sum() * 0.0
            )
            setwise_factorized_risk_bound_loss = 0.5 * (
                setwise_factorized_risk_bound_point_loss
                + setwise_factorized_risk_bound_guard_loss
            )
            setwise_coupled_witness_loss = (
                torch.stack(setwise_coupled_witness_terms).mean()
                if setwise_coupled_witness_terms
                else scores.sum() * 0.0
            )
            setwise_bidirectional_boundary_loss = (
                torch.stack(setwise_bidirectional_boundary_terms).mean()
                if setwise_bidirectional_boundary_terms
                else scores.sum() * 0.0
            )
            setwise_centered_separation_loss = (
                torch.stack(setwise_centered_separation_terms).mean()
                if setwise_centered_separation_terms
                else scores.sum() * 0.0
            )
            setwise_branchwise_witness_loss = (
                torch.stack(setwise_branchwise_witness_terms).mean()
                if setwise_branchwise_witness_terms
                else scores.sum() * 0.0
            )
            if ((float(setwise_dense_safety_loss_weight) > 0.0
                 or float(setwise_balanced_safety_loss_weight) > 0.0
                 or float(setwise_factorized_safety_loss_weight) > 0.0
                 or float(
                     setwise_factorized_risk_bound_loss_weight
                 ) > 0.0)
                    and not decoupled_setwise):
                raise ValueError(
                    "safety loss requires decoupled setwise heads"
                )
            if (sum((
                    float(setwise_dense_safety_loss_weight) > 0.0,
                    float(setwise_balanced_safety_loss_weight) > 0.0,
                    float(setwise_factorized_safety_loss_weight) > 0.0,
                    float(
                        setwise_factorized_risk_bound_loss_weight
                    ) > 0.0,
                    )) > 1):
                raise ValueError(
                    "dense, balanced, factorized, and factorized risk-bound "
                    "safety losses are mutually exclusive"
                )
            if (float(setwise_factorized_safety_loss_weight) > 0.0
                    and not factorized_setwise_safety):
                raise ValueError(
                    "factorized safety loss requires factorized setwise "
                    "safety heads"
                )
            if (float(setwise_factorized_risk_bound_loss_weight) > 0.0
                    and not factorized_setwise_risk_bound):
                raise ValueError(
                    "factorized risk-bound loss requires factorized "
                    "risk-bound safety heads"
                )
            transition_loss = (
                transition_loss
                + float(setwise_repair_boundary_loss_weight)
                * setwise_repair_boundary_loss
                + float(setwise_negative_tail_loss_weight)
                * setwise_negative_tail_loss
                + float(setwise_rank_loss_weight) * setwise_rank_loss
                + float(setwise_dense_safety_loss_weight)
                * setwise_dense_safety_loss
                + float(setwise_balanced_safety_loss_weight)
                * setwise_balanced_safety_loss
                + float(setwise_factorized_safety_loss_weight)
                * setwise_factorized_safety_loss
                + float(setwise_factorized_risk_bound_loss_weight)
                * setwise_factorized_risk_bound_loss
                + setwise_coupled_witness_loss
                + setwise_bidirectional_boundary_loss
                + setwise_centered_separation_loss
                + setwise_branchwise_witness_loss
                + independent_joint_hazard_loss
            )
            transition_stats["setwise_tier_repair_boundary_loss"] = (
                setwise_repair_boundary_loss.detach()
            )
            transition_stats[
                "setwise_tier_repair_boundary_eligible_ratio"
            ] = (
                torch.stack(setwise_repair_boundary_eligible_ratios).mean()
                if setwise_repair_boundary_eligible_ratios
                else scores.new_zeros(())
            )
            transition_stats["setwise_tier_negative_tail_loss"] = (
                setwise_negative_tail_loss.detach()
            )
            transition_stats[
                "setwise_tier_negative_tail_violation_ratio"
            ] = (
                torch.stack(setwise_negative_tail_violation_ratios).mean()
                if setwise_negative_tail_violation_ratios
                else scores.new_zeros(())
            ).detach()
            transition_stats["setwise_tier_rank_loss"] = (
                setwise_rank_loss.detach()
            )
            transition_stats["setwise_dense_safety_loss"] = (
                setwise_dense_safety_loss.detach()
            )
            transition_stats["setwise_dense_safety_violation_ratio"] = (
                torch.stack(setwise_dense_safety_violation_ratios).mean()
                if setwise_dense_safety_violation_ratios
                else scores.new_zeros(())
            ).detach()
            transition_stats["setwise_balanced_safety_loss"] = (
                setwise_balanced_safety_loss.detach()
            )
            transition_stats["setwise_factorized_safety_loss"] = (
                setwise_factorized_safety_loss.detach()
            )
            transition_stats[
                "setwise_factorized_risk_bound_point_loss"
            ] = setwise_factorized_risk_bound_point_loss.detach()
            transition_stats[
                "setwise_factorized_risk_bound_guard_loss"
            ] = setwise_factorized_risk_bound_guard_loss.detach()
            transition_stats[
                "setwise_factorized_risk_bound_loss"
            ] = setwise_factorized_risk_bound_loss.detach()
            transition_stats[
                "setwise_coupled_safe_repair_witness_loss"
            ] = setwise_coupled_witness_loss.detach()
            transition_stats[
                "setwise_bidirectional_coupled_boundary_loss"
            ] = setwise_bidirectional_boundary_loss.detach()
            transition_stats[
                "setwise_centered_coupled_separation_loss"
            ] = setwise_centered_separation_loss.detach()
            transition_stats[
                "setwise_same_candidate_branchwise_witness_loss"
            ] = setwise_branchwise_witness_loss.detach()
            transition_stats[
                "setwise_balanced_safety_safe_violation_ratio"
            ] = (
                torch.stack(
                    setwise_balanced_safety_safe_violation_ratios
                ).mean()
                if setwise_balanced_safety_safe_violation_ratios
                else scores.new_zeros(())
            ).detach()
            transition_stats[
                "setwise_balanced_safety_hazard_violation_ratio"
            ] = (
                torch.stack(
                    setwise_balanced_safety_hazard_violation_ratios
                ).mean()
                if setwise_balanced_safety_hazard_violation_ratios
                else scores.new_zeros(())
            ).detach()
            transition_stats[
                "setwise_factorized_safety_safe_violation_ratio"
            ] = (
                torch.stack(
                    setwise_factorized_safety_safe_violation_ratios
                ).mean()
                if setwise_factorized_safety_safe_violation_ratios
                else scores.new_zeros(())
            ).detach()
            transition_stats[
                "setwise_factorized_safety_hazard_violation_ratio"
            ] = (
                torch.stack(
                    setwise_factorized_safety_hazard_violation_ratios
                ).mean()
                if setwise_factorized_safety_hazard_violation_ratios
                else scores.new_zeros(())
            ).detach()
            denominator = sample_mask.float().sum().clamp(min=1.0)
            transition_stats["setwise_tier_reachable_query_ratio"] = (
                reachable_mask.float().sum()
                / candidate_valid.float().sum().clamp(min=1.0)
            ).detach()
            transition_stats["setwise_decoupled_promotion_safety"] = (
                scores.new_tensor(float(decoupled_setwise))
            )
            transition_stats["setwise_factorized_safety"] = (
                scores.new_tensor(float(factorized_setwise_safety))
            )
            transition_stats["setwise_factorized_risk_bound"] = (
                scores.new_tensor(float(factorized_setwise_risk_bound))
            )
            transition_stats["setwise_safety_veto_gate"] = (
                scores.new_tensor(float(setwise_safety_veto_gate))
            )
            transition_stats["setwise_cost_calibrated_risk_bound"] = (
                scores.new_tensor(float(cost_calibrated_risk_bound))
            )
            transition_stats["setwise_safety_slack_quantile_bound"] = (
                scores.new_tensor(float(safety_slack_quantile_bound))
            )
            transition_stats["setwise_safety_slack_pairwise_order"] = (
                scores.new_tensor(float(safety_slack_pairwise_order))
            )
            transition_stats["setwise_proposal_conditioned_safety"] = (
                scores.new_tensor(float(proposal_conditioned_safety))
            )
            transition_stats["setwise_parent_referenced_safety"] = (
                scores.new_tensor(float(parent_referenced_safety))
            )
            transition_stats["setwise_coupled_safe_repair_witness"] = (
                scores.new_tensor(float(coupled_safe_repair_witness))
            )
            transition_stats["setwise_bidirectional_coupled_boundary"] = (
                scores.new_tensor(float(bidirectional_coupled_boundary))
            )
            transition_stats["setwise_centered_coupled_separation"] = (
                scores.new_tensor(float(centered_coupled_separation))
            )
            transition_stats[
                "setwise_hazard_conditioned_coupled_separation"
            ] = scores.new_tensor(float(
                hazard_conditioned_coupled_separation
            ))
            transition_stats["setwise_monotonic_box_safety_folding"] = (
                scores.new_tensor(float(monotonic_box_safety_folding))
            )
            transition_stats[
                "setwise_same_candidate_branchwise_witness"
            ] = scores.new_tensor(float(same_candidate_branchwise_witness))
            transition_stats[
                "setwise_parent_non_degradation_certificate"
            ] = scores.new_tensor(float(parent_non_degradation_certificate))
            transition_stats[
                "setwise_criterion_responsible_hazard_attribution"
            ] = scores.new_tensor(float(
                criterion_responsible_hazard_attribution
            ))
            transition_stats[
                "setwise_independent_joint_hazard_certificate"
            ] = scores.new_tensor(float(
                independent_joint_hazard_certificate
            ))
            transition_stats[
                "setwise_frozen_raw_joint_hazard_features"
            ] = scores.new_tensor(float(
                frozen_raw_joint_hazard_features
            ))
            transition_stats["setwise_safety_veto_accept_ratio"] = (
                (
                    (branch_scores[..., 1] > 0.0) & transition_active
                ).float().sum()
                / transition_active.float().sum().clamp(min=1.0)
                if setwise_safety_veto_gate else scores.new_zeros(())
            ).detach()
            transition_stats["setwise_safety_hazard_query_ratio"] = (
                hazardous.float().sum()
                / transition_active.float().sum().clamp(min=1.0)
            ).detach()
            if proposal_conditioned_safety:
                proposal_count = proposal_mask.float().sum().clamp(min=1.0)
                transition_stats["setwise_proposal_hazard_ratio"] = (
                    (proposal_mask & hazardous).float().sum()
                    / proposal_count
                ).detach()
                transition_stats["setwise_proposal_promotable_ratio"] = (
                    proposal_promotable_mask.float().sum()
                    / proposal_count
                ).detach()
                transition_stats["setwise_proposal_safety_accept_ratio"] = (
                    (
                        proposal_mask & (branch_scores[..., 1] > 0.0)
                    ).float().sum() / proposal_count
                ).detach()
            transition_stats["setwise_tier_repair_row_ratio"] = (
                repair_rows.float().sum() / denominator
            ).detach()
            transition_stats["setwise_tier_stay_row_ratio"] = (
                stay_rows.float().sum() / denominator
            ).detach()
            selected_setwise = outputs["selected_indices"]
            transition_stats["setwise_tier_repair_recall"] = (
                best_repair[row, selected_setwise][repair_rows].float().mean()
                if bool(repair_rows.any().item())
                else scores.new_zeros(())
            ).detach()
            transition_stats["setwise_tier_stay_recall"] = (
                (selected_setwise[stay_rows] == anchors[stay_rows])
                .float().mean()
                if bool(stay_rows.any().item())
                else scores.new_zeros(())
            ).detach()
        else:
            decomposed_logits = outputs.get("decomposed_transition_logits")
            if decomposed_logits is not None:
                if (not isinstance(decomposed_logits, torch.Tensor)
                        or decomposed_logits.shape != (
                            scores.shape[0], scores.shape[1], 2, 2)
                        or decomposed_logits.device != scores.device
                        or not decomposed_logits.is_floating_point()
                        or not bool(torch.isfinite(
                            decomposed_logits
                        ).all().item())):
                    raise ValueError(
                        "positive transition loss requires finite decomposed "
                        "transition logits [B,Q,2,2]"
                    )
                transition_active = active.clone()
                transition_active[row, anchors] = False
                grouped_terms = []
                grouped_weights = []
                for threshold_index, (threshold, suffix) in enumerate((
                        (0.25, "025"), (0.50, "050"))):
                    parent_hit = box_ious[row, anchors] > threshold
                    candidate_hit = box_ious > threshold
                    fix_target = ~parent_hit.unsqueeze(1) & candidate_hit
                    break_target = parent_hit.unsqueeze(1) & ~candidate_hit
                    change_target = fix_target | break_target
                    neutral_target = ~change_target
                    change_logit = decomposed_logits[..., threshold_index, 0]
                    direction_logit = decomposed_logits[..., threshold_index, 1]
                    change_loss = F.binary_cross_entropy_with_logits(
                        change_logit, change_target.to(scores.dtype),
                        reduction="none",
                    )
                    direction_loss = F.binary_cross_entropy_with_logits(
                        direction_logit, fix_target.to(scores.dtype),
                        reduction="none",
                    )
                    for group_mask in (change_target, neutral_target):
                        group_mask = transition_active & group_mask
                        if bool(group_mask.any().item()):
                            grouped_terms.append(change_loss[group_mask].mean())
                            grouped_weights.append(1.0)
                    # Estimate the conditional transition direction without
                    # baking deployment risk preference into the probability head.
                    # Fix and break groups are equally normalized here; the break
                    # cost is applied exactly once by the deployment utility.
                    for group_mask in (break_target, fix_target):
                        group_mask = transition_active & group_mask
                        if bool(group_mask.any().item()):
                            grouped_terms.append(direction_loss[group_mask].mean())
                            grouped_weights.append(1.0)

                    predicted_change = change_logit > 0.0
                    predicted_fix = direction_logit > 0.0
                    active_count = transition_active.float().sum().clamp(min=1.0)
                    for name, target, prediction in (
                            ("change", change_target, predicted_change),
                            ("neutral", neutral_target, ~predicted_change),
                            ("break", break_target,
                             predicted_change & ~predicted_fix),
                            ("fix", fix_target,
                             predicted_change & predicted_fix)):
                        target_active = transition_active & target
                        target_count = target_active.float().sum()
                        transition_stats[
                            "decomposed_{}_{}_target_ratio".format(name, suffix)
                        ] = (target_count / active_count).detach()
                        transition_stats[
                            "decomposed_{}_{}_recall".format(name, suffix)
                        ] = (
                            (prediction & target_active).float().sum()
                            / target_count.clamp(min=1.0)
                        ).detach()
                if grouped_terms:
                    term_weights = scores.new_tensor(grouped_weights)
                    transition_loss = (
                        torch.stack(grouped_terms) * term_weights
                    ).sum() / term_weights.sum().clamp(min=1e-6)
            else:
                transition_logits = outputs.get("parent_transition_logits")
                if (not isinstance(transition_logits, torch.Tensor)
                        or transition_logits.shape != (
                            scores.shape[0], scores.shape[1], 2, 3)
                        or transition_logits.device != scores.device
                        or not transition_logits.is_floating_point()
                        or not bool(torch.isfinite(transition_logits).all().item())):
                    raise ValueError(
                        "positive transition loss requires finite "
                        "parent_transition_logits [B,Q,2,3]"
                    )
                transition_active = active.clone()
                transition_active[row, anchors] = False
                transition_rows = []
                target_counts = scores.new_zeros(3)
                correct_counts = scores.new_zeros(3)
                for threshold_index, threshold in enumerate((0.25, 0.50)):
                    parent_hit = box_ious[row, anchors] > threshold
                    candidate_hit = box_ious > threshold
                    targets = torch.ones_like(candidate_hit, dtype=torch.long)
                    targets = torch.where(
                        parent_hit.unsqueeze(1) & ~candidate_hit,
                        torch.zeros_like(targets),
                        targets,
                    )
                    targets = torch.where(
                        ~parent_hit.unsqueeze(1) & candidate_hit,
                        torch.full_like(targets, 2),
                        targets,
                    )
                    predictions = transition_logits[
                        ..., threshold_index, :
                    ].argmax(dim=-1)
                    per_query_loss = F.cross_entropy(
                        transition_logits[..., threshold_index, :].reshape(-1, 3),
                        targets.reshape(-1),
                        reduction="none",
                    ).view_as(scores)
                    class_weights = scores.new_tensor((
                        float(transition_break_cost),
                        float(transition_neutral_weight),
                        1.0,
                    ))
                    target_one_hot = F.one_hot(
                        targets, num_classes=3
                    ).to(scores.dtype)
                    active_one_hot = (
                        target_one_hot
                        * transition_active.unsqueeze(-1).to(scores.dtype)
                    )
                    class_counts = active_one_hot.sum(dim=1)
                    query_class_count = class_counts.gather(1, targets).clamp(min=1.0)
                    query_weight = class_weights[targets] / query_class_count
                    query_weight = (
                        query_weight
                        * transition_active.to(query_weight.dtype)
                    )
                    active_class_weight = (
                        (class_counts > 0).to(scores.dtype) * class_weights
                    ).sum(dim=1).clamp(min=1e-6)
                    row_loss = (
                        (per_query_loss * query_weight).sum(dim=1)
                        / active_class_weight
                    )
                    active_rows = sample_mask & transition_active.any(dim=1)
                    if bool(active_rows.any().item()):
                        transition_rows.append(row_loss[active_rows])
                    for class_index in range(3):
                        class_active = transition_active & (targets == class_index)
                        count = class_active.float().sum()
                        target_counts[class_index] += count
                        correct_counts[class_index] += (
                            (predictions == class_index) & class_active
                        ).float().sum()
                if transition_rows:
                    transition_loss = torch.cat(transition_rows).mean()
                names = ("break", "neutral", "fix")
                for class_index, name in enumerate(names):
                    transition_stats[
                        "transition_{}_target_ratio".format(name)
                    ] = (
                        target_counts[class_index]
                        / target_counts.sum().clamp(min=1.0)
                    ).detach()
                    transition_stats[
                        "transition_{}_recall".format(name)
                    ] = (
                        correct_counts[class_index]
                        / target_counts[class_index].clamp(min=1.0)
                    ).detach()
    factorized_hit_loss = scores.sum() * 0.0
    if float(factorized_hit_loss_weight) > 0.0:
        hit_logits = outputs.get("factorized_hit_logits")
        if (not isinstance(hit_logits, torch.Tensor)
                or hit_logits.shape != (scores.shape[0], scores.shape[1], 2)
                or hit_logits.device != scores.device
                or not hit_logits.is_floating_point()
                or not bool(torch.isfinite(hit_logits).all().item())):
            raise ValueError(
                "positive factorized hit loss requires finite "
                "factorized_hit_logits [B,Q,2]"
            )
        hit_targets = torch.stack((
            box_ious > 0.25, box_ious > 0.50
        ), dim=-1).to(hit_logits.dtype)
        per_hit_loss = F.binary_cross_entropy_with_logits(
            hit_logits, hit_targets, reduction="none"
        )
        factorized_rows = []
        for threshold_index, suffix in enumerate(("025", "050")):
            threshold_target = hit_targets[..., threshold_index]
            threshold_active = active
            positives = threshold_active & threshold_target.bool()
            negatives = threshold_active & ~threshold_target.bool()
            positive_count = positives.sum(dim=1).clamp(min=1)
            negative_count = negatives.sum(dim=1).clamp(min=1)
            query_weight = (
                positives.to(scores.dtype)
                / positive_count.unsqueeze(1).to(scores.dtype)
                + negatives.to(scores.dtype)
                / negative_count.unsqueeze(1).to(scores.dtype)
            )
            present_count = (
                positives.any(dim=1).to(scores.dtype)
                + negatives.any(dim=1).to(scores.dtype)
            ).clamp(min=1.0)
            row_loss = (
                per_hit_loss[..., threshold_index] * query_weight
            ).sum(dim=1) / present_count
            active_rows = sample_mask & threshold_active.any(dim=1)
            if bool(active_rows.any().item()):
                factorized_rows.append(row_loss[active_rows])
            predicted_hit = hit_logits[..., threshold_index] > 0.0
            transition_stats["factorized_hit_{}_target_ratio".format(
                suffix
            )] = (
                positives.float().sum()
                / threshold_active.float().sum().clamp(min=1.0)
            ).detach()
            transition_stats["factorized_hit_{}_positive_recall".format(
                suffix
            )] = (
                (predicted_hit & positives).float().sum()
                / positives.float().sum().clamp(min=1.0)
            ).detach()
            transition_stats["factorized_hit_{}_negative_recall".format(
                suffix
            )] = (
                (~predicted_hit & negatives).float().sum()
                / negatives.float().sum().clamp(min=1.0)
            ).detach()
        if factorized_rows:
            factorized_hit_loss = torch.cat(factorized_rows).mean()

    factorized_pair_loss = scores.sum() * 0.0
    if float(factorized_pair_loss_weight) > 0.0:
        hit_logits = outputs.get("factorized_hit_logits")
        if (not isinstance(hit_logits, torch.Tensor)
                or hit_logits.shape != (scores.shape[0], scores.shape[1], 2)
                or hit_logits.device != scores.device
                or not hit_logits.is_floating_point()
                or not bool(torch.isfinite(hit_logits).all().item())):
            raise ValueError(
                "positive factorized pair loss requires finite "
                "factorized_hit_logits [B,Q,2]"
            )
        pair_active = active.clone()
        pair_active[row, anchors] = False
        protect_rows = []
        repair_rows = []
        for threshold_index, (threshold, suffix, margin) in enumerate((
                (0.25, "025", float(anchor_margin)),
                (0.50, "050", float(anchor_margin_050)))):
            parent_hit = box_ious[row, anchors] > threshold
            candidate_hit = box_ious > threshold
            miss_candidates = pair_active & ~candidate_hit
            hit_candidates = pair_active & candidate_hit
            parent_logit = hit_logits[row, anchors, threshold_index]
            threshold_logits = hit_logits[..., threshold_index]

            protect = (
                sample_mask & parent_hit & miss_candidates.any(dim=1)
            )
            protect_correct = scores.new_zeros(())
            if bool(protect.any().item()):
                hardest_miss = threshold_logits.masked_fill(
                    ~miss_candidates, -1e4
                ).max(dim=1).values
                protect_gap = parent_logit - hardest_miss
                protect_rows.append(F.softplus(
                    margin - protect_gap[protect]
                ))
                protect_correct = (
                    (protect_gap[protect] > 0.0).float().mean()
                )

            repair = (
                sample_mask & ~parent_hit & hit_candidates.any(dim=1)
            )
            repair_correct = scores.new_zeros(())
            if bool(repair.any().item()):
                best_hit = threshold_logits.masked_fill(
                    ~hit_candidates, -1e4
                ).max(dim=1).values
                repair_gap = best_hit - parent_logit
                repair_rows.append(F.softplus(
                    margin - repair_gap[repair]
                ))
                repair_correct = (
                    (repair_gap[repair] > 0.0).float().mean()
                )

            denominator = sample_mask.float().sum().clamp(min=1.0)
            transition_stats[
                "factorized_hard_anchor_{}_protect_ratio".format(suffix)
            ] = (protect.float().sum() / denominator).detach()
            transition_stats[
                "factorized_hard_anchor_{}_repair_ratio".format(suffix)
            ] = (repair.float().sum() / denominator).detach()
            transition_stats[
                "factorized_hard_anchor_{}_protect_recall".format(suffix)
            ] = protect_correct.detach()
            transition_stats[
                "factorized_hard_anchor_{}_repair_recall".format(suffix)
            ] = repair_correct.detach()

        hard_anchor_terms = []
        if protect_rows:
            hard_anchor_terms.append(torch.cat(protect_rows).mean())
        if repair_rows:
            hard_anchor_terms.append(torch.cat(repair_rows).mean())
        if hard_anchor_terms:
            # Protect and repair are equally important regardless of their
            # sample counts, preventing the abundant protect pairs from
            # drowning the sparse fix signal.
            factorized_pair_loss = torch.stack(hard_anchor_terms).mean()

    box_targets = torch.stack((
        box_ious > 0.25, box_ious > 0.50
    ), dim=-1).to(outputs["box_logits"].dtype)
    mask_targets = torch.stack((
        mask_ious > 0.25, mask_ious > 0.50
    ), dim=-1).to(outputs["mask_logits"].dtype)
    box_bce = F.binary_cross_entropy_with_logits(
        outputs["box_logits"], box_targets, reduction="none"
    )
    mask_bce = F.binary_cross_entropy_with_logits(
        outputs["mask_logits"], mask_targets, reduction="none"
    )
    box_regression = F.smooth_l1_loss(
        outputs["box_iou"], box_ious, reduction="none"
    )
    mask_regression = F.smooth_l1_loss(
        outputs["mask_iou"], mask_ious, reduction="none"
    )
    quality_loss = (
        _masked_mean(box_bce, active)
        + _masked_mean(box_regression, active)
        + float(mask_weight) * (
            _masked_mean(mask_bce, active)
            + _masked_mean(mask_regression, active)
        )
    )

    anchor_score = scores[row, anchors]
    protect_anchor_losses = []
    repair_anchor_losses = []
    anchor_stats = {}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        margin = (
            float(anchor_margin_050)
            if bidirectional_anchor and threshold == 0.50
            else float(anchor_margin)
        )
        correct = (box_ious > threshold) & candidate_valid
        anchor_correct = correct[row, anchors]
        wrong = candidate_valid & ~correct
        protect = sample_mask & anchor_correct & wrong.any(dim=1)
        anchor_stats["anchor_protect_{}_ratio".format(suffix)] = (
            protect.float().sum()
            / sample_mask.float().sum().clamp(min=1.0)
        ).detach()
        if bool(protect.any().item()):
            hardest_wrong = scores.masked_fill(~wrong, -1e4).max(dim=1).values
            protect_anchor_losses.append(F.relu(
                hardest_wrong[protect] + margin - anchor_score[protect]
            ))
        repair = (
            sample_mask & ~anchor_correct & correct.any(dim=1)
            & wrong.any(dim=1)
        ) if bidirectional_anchor else torch.zeros_like(sample_mask)
        anchor_stats["anchor_repair_{}_ratio".format(suffix)] = (
            repair.float().sum()
            / sample_mask.float().sum().clamp(min=1.0)
        ).detach()
        if bool(repair.any().item()):
            best_hit = scores.masked_fill(~correct, -1e4).max(dim=1).values
            hardest_wrong = scores.masked_fill(~wrong, -1e4).max(dim=1).values
            repair_anchor_losses.append(F.relu(
                hardest_wrong[repair] + margin - best_hit[repair]
            ))
    protect_anchor_loss = (
        torch.cat(protect_anchor_losses).mean()
        if protect_anchor_losses else scores.sum() * 0.0
    )
    repair_anchor_loss = (
        torch.cat(repair_anchor_losses).mean()
        if repair_anchor_losses else scores.sum() * 0.0
    )
    anchor_terms = protect_anchor_losses + repair_anchor_losses
    anchor_loss = (
        torch.cat(anchor_terms).mean()
        if anchor_terms else scores.sum() * 0.0
    )

    source_mix_keys = (
        "source_mix_weights", "source_mix_ranks",
        "source_mix_validity",
    )
    has_source_mix = any(key in outputs for key in source_mix_keys)
    if has_source_mix and not all(key in outputs for key in source_mix_keys):
        raise ValueError("adaptive source mix outputs are incomplete")
    if float(source_mix_loss_weight) > 0.0 and not has_source_mix:
        raise ValueError(
            "positive source mix loss requires adaptive source mixing"
        )
    source_mix_alignment = (
        compute_joint_query_source_mix_alignment_loss(
            outputs, target_quality, sample_mask=sample_mask,
            temperature=float(source_mix_alignment_temperature),
            query_relevance=target_distribution.detach(),
            query_focus_weight=float(source_mix_query_focus_weight),
        )
        if has_source_mix else {
            "loss": scores.sum() * 0.0,
            "target_top1_acc": scores.new_zeros(()),
            "target_effective_count_mean": scores.new_zeros(()),
        }
    )
    total = (
        float(listwise_loss_weight) * listwise_loss
        + float(pairwise_loss_weight) * pairwise_loss
        + float(quality_loss_weight) * quality_loss
        + float(anchor_loss_weight) * anchor_loss
        + float(source_mix_loss_weight) * source_mix_alignment["loss"]
        + float(transition_loss_weight) * transition_loss
        + float(factorized_hit_loss_weight) * factorized_hit_loss
        + float(factorized_pair_loss_weight) * factorized_pair_loss
    )

    selected = outputs["selected_indices"]
    baseline = outputs["baseline_indices"]
    selected_box = box_ious[row, selected]
    baseline_box = box_ious[row, baseline]
    selected_mask = mask_ious[row, selected]
    baseline_mask = mask_ious[row, baseline]
    stats = dict(anchor_stats, **transition_stats)
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        baseline_hit = baseline_box > threshold
        selected_hit = selected_box > threshold
        stats["fix{}".format(suffix)] = (
            (~baseline_hit & selected_hit & sample_mask).float().sum()
            / sample_mask.float().sum().clamp(min=1.0)
        ).detach()
        stats["break{}".format(suffix)] = (
            (baseline_hit & ~selected_hit & sample_mask).float().sum()
            / sample_mask.float().sum().clamp(min=1.0)
        ).detach()
    stats["switch_ratio"] = (
        ((selected != baseline) & sample_mask).float().sum()
        / sample_mask.float().sum().clamp(min=1.0)
    ).detach()
    stats["mask_delta_mean"] = (
        ((selected_mask - baseline_mask) * sample_mask.to(selected_mask.dtype))
        .sum() / sample_mask.float().sum().clamp(min=1.0)
    ).detach()
    stats["candidate_query_ratio"] = (
        candidate_valid.float().sum() / valid.float().sum().clamp(min=1.0)
    ).detach()
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        candidate_oracle_hit = (
            (box_ious > threshold) & candidate_valid
        ).any(dim=1)
        stats["candidate_oracle_{}".format(suffix)] = (
            (candidate_oracle_hit & sample_mask).float().sum()
            / sample_mask.float().sum().clamp(min=1.0)
        ).detach()
    stats["source_mix_alignment_target_top1_acc"] = (
        source_mix_alignment["target_top1_acc"]
    )
    stats["source_mix_alignment_target_effective_count_mean"] = (
        source_mix_alignment["target_effective_count_mean"]
    )
    if "parent_transition_advantage" in outputs:
        advantage = outputs["parent_transition_advantage"]
        stats["transition_advantage_positive_ratio"] = (
            ((advantage > 0.0) & valid).float().sum()
            / valid.float().sum().clamp(min=1.0)
        ).detach()
    if "factorized_fix_break_utility" in outputs:
        utility = outputs["factorized_fix_break_utility"]
        denominator = valid.float().sum().clamp(min=1.0)
        stats["factorized_utility_025_positive_ratio"] = (
            ((utility[..., 0] > 0.0) & valid).float().sum() / denominator
        ).detach()
        stats["factorized_utility_050_positive_ratio"] = (
            ((utility[..., 1] > 0.0) & valid).float().sum() / denominator
        ).detach()
        stats["factorized_nested_positive_ratio"] = (
            ((utility > 0.0).all(dim=-1) & valid).float().sum()
            / denominator
        ).detach()
    counterfactual_family = "factorized"
    counterfactual_costs = outputs.get("factorized_counterfactual_costs")
    counterfactual_selected = outputs.get(
        "factorized_counterfactual_selected_indices"
    )
    if counterfactual_costs is None and counterfactual_selected is None:
        counterfactual_family = "decomposed"
        counterfactual_costs = outputs.get("decomposed_counterfactual_costs")
        counterfactual_selected = outputs.get(
            "decomposed_counterfactual_selected_indices"
        )
    if ((counterfactual_costs is None)
            != (counterfactual_selected is None)):
        raise ValueError("factorized counterfactual outputs are incomplete")
    if counterfactual_costs is not None:
        if (not isinstance(counterfactual_costs, torch.Tensor)
                or counterfactual_costs.dim() != 1
                or not isinstance(counterfactual_selected, torch.Tensor)
                or counterfactual_selected.dtype != torch.long
                or counterfactual_selected.shape != (
                    scores.shape[0], counterfactual_costs.numel()
                )
                or counterfactual_selected.device != scores.device
                or counterfactual_costs.device != scores.device
                or not bool(torch.isfinite(
                    counterfactual_costs
                ).all().item())):
            raise ValueError("factorized counterfactual outputs are invalid")
        sample_denominator = sample_mask.float().sum().clamp(min=1.0)
        for cost_index, cost in enumerate(counterfactual_costs.tolist()):
            cost_suffix = ("{:.2f}".format(cost)).replace(".", "p")
            selected_cf = counterfactual_selected[:, cost_index]
            selected_cf_box = box_ious[row, selected_cf]
            prefix = "{}_cf_cost{}_".format(
                counterfactual_family, cost_suffix
            )
            stats[prefix + "switch_ratio"] = (
                ((selected_cf != baseline) & sample_mask).float().sum()
                / sample_denominator
            ).detach()
            for threshold, suffix in ((0.25, "025"), (0.50, "050")):
                baseline_hit = baseline_box > threshold
                selected_cf_hit = selected_cf_box > threshold
                stats[prefix + "fix{}".format(suffix)] = (
                    (~baseline_hit & selected_cf_hit & sample_mask)
                    .float().sum() / sample_denominator
                ).detach()
                stats[prefix + "break{}".format(suffix)] = (
                    (baseline_hit & ~selected_cf_hit & sample_mask)
                    .float().sum() / sample_denominator
                ).detach()
    return {
        "loss": total,
        "listwise_loss": listwise_loss,
        "pairwise_loss": pairwise_loss,
        "quality_loss": quality_loss,
        "anchor_loss": anchor_loss,
        "transition_loss": transition_loss,
        "factorized_hit_loss": factorized_hit_loss,
        "factorized_pair_loss": factorized_pair_loss,
        "protect_anchor_loss": protect_anchor_loss,
        "repair_anchor_loss": repair_anchor_loss,
        "source_mix_alignment_loss": source_mix_alignment["loss"],
        "target_quality": target_quality.detach(),
        "stats": stats,
    }
