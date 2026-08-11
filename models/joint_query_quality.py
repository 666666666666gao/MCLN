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
    proxy = torch.sigmoid(centered / (variance.sqrt() + float(eps)))
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


def predicted_joint_query_quality(box_logits, box_iou, mask_logits, mask_iou,
                                  mask_weight=0.25):
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
                 quality_score_weight=1.0, detach_inputs=True,
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
                 use_source_distribution_reliability=False):
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
        direct_residual_logit = self.residual_head(hidden).squeeze(-1)
        residual_logit = (
            direct_residual_logit
            + self.quality_score_weight * centered_quality
            + source_mix_residual_logit
        )
        residual = self.max_delta * torch.tanh(
            residual_logit
        )
        residual = residual.masked_fill(~valid_mask, 0.0)
        scores = (baseline + residual).masked_fill(~valid_mask, -1e4)
        result = {
            "scores": scores,
            "residual": residual,
            "residual_logit": residual_logit,
            "direct_residual_logit": direct_residual_logit,
            "centered_quality": centered_quality,
            "baseline_rank": baseline_rank,
            "baseline_standardized": baseline_standardized,
            "baseline_indices": baseline_rank.argmax(dim=1),
            "selected_indices": scores.argmax(dim=1),
            "box_logits": box_logits,
            "box_iou": box_iou,
            "mask_logits": mask_logits,
            "mask_iou": mask_iou,
            "quality": quality,
            "quality_evidence": quality_evidence,
            "valid_mask": valid_mask,
        }
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
        source_mix_loss_weight=0.0,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.0):
    """Train direct Top-1 ranking plus shared absolute quality evidence."""
    for name, value in (
            ("mask_weight", mask_weight),
            ("quality_loss_weight", quality_loss_weight),
            ("anchor_loss_weight", anchor_loss_weight),
            ("anchor_margin", anchor_margin),
            ("source_mix_loss_weight", source_mix_loss_weight),
            ("source_mix_query_focus_weight",
             source_mix_query_focus_weight)):
        _finite_non_negative(name, value)
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
    target_quality = joint_query_target_quality(
        box_ious.float(), mask_ious.float(), mask_weight=mask_weight
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
    active = valid & sample_mask.unsqueeze(1)
    if not bool(active.any().item()):
        zero = scores.sum() * 0.0
        return {"loss": zero, "listwise_loss": zero,
                "quality_loss": zero, "anchor_loss": zero,
                "source_mix_alignment_loss": zero, "stats": {}}

    scaled_target = (target_quality / float(temperature)).masked_fill(
        ~valid, -1e4
    )
    target_distribution = F.softmax(scaled_target, dim=1)
    listwise_rows = -(
        target_distribution
        * F.log_softmax(scores.masked_fill(~valid, -1e4), dim=1)
    ).sum(dim=1)
    listwise_loss = listwise_rows[sample_mask].mean()

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

    row = torch.arange(scores.shape[0], device=scores.device)
    anchors = outputs["baseline_indices"]
    if (anchors.dtype != torch.long or anchors.shape != (scores.shape[0],)):
        raise ValueError("baseline_indices must be int64 [B]")
    anchor_score = scores[row, anchors]
    anchor_losses = []
    for threshold in (0.25, 0.50):
        correct = (box_ious > threshold) & valid
        anchor_correct = correct[row, anchors]
        wrong = valid & ~correct
        protect = sample_mask & anchor_correct & wrong.any(dim=1)
        if bool(protect.any().item()):
            hardest_wrong = scores.masked_fill(~wrong, -1e4).max(dim=1).values
            anchor_losses.append(F.relu(
                hardest_wrong[protect] + float(anchor_margin)
                - anchor_score[protect]
            ))
    anchor_loss = (
        torch.cat(anchor_losses).mean()
        if anchor_losses else scores.sum() * 0.0
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
        listwise_loss
        + float(quality_loss_weight) * quality_loss
        + float(anchor_loss_weight) * anchor_loss
        + float(source_mix_loss_weight) * source_mix_alignment["loss"]
    )

    selected = outputs["selected_indices"]
    baseline = outputs["baseline_indices"]
    selected_box = box_ious[row, selected]
    baseline_box = box_ious[row, baseline]
    selected_mask = mask_ious[row, selected]
    baseline_mask = mask_ious[row, baseline]
    stats = {}
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
    stats["source_mix_alignment_target_top1_acc"] = (
        source_mix_alignment["target_top1_acc"]
    )
    stats["source_mix_alignment_target_effective_count_mean"] = (
        source_mix_alignment["target_effective_count_mean"]
    )
    return {
        "loss": total,
        "listwise_loss": listwise_loss,
        "quality_loss": quality_loss,
        "anchor_loss": anchor_loss,
        "source_mix_alignment_loss": source_mix_alignment["loss"],
        "target_quality": target_quality.detach(),
        "stats": stats,
    }
