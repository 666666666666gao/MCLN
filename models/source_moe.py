"""Trainable source fusion and in-graph query reranking for MCLN.

The shared source preserves the original ranking at initialization. Routed
sources and a lightweight self-attention reranker then learn residual changes
from an explicit query-quality objective in :mod:`models.losses`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_fusion import as_query_mask_logits, fuse_query_mask_logits
from utils.scatter_util import scatter_mean


def rank_normalize(scores):
    """Map scores to [0, 1] by descending rank along the query axis."""
    if not isinstance(scores, torch.Tensor) or scores.dim() != 2:
        raise ValueError("scores must have shape [B,Q]")
    order = scores.argsort(dim=1, descending=True)
    rank = torch.zeros_like(order, dtype=scores.dtype)
    values = torch.arange(
        scores.shape[1], device=scores.device, dtype=scores.dtype
    ).unsqueeze(0).expand_as(rank)
    rank.scatter_(1, order, values)
    denom = max(scores.shape[1] - 1, 1)
    return 1.0 - rank / float(denom)


def standardize_source_scores(scores, valid_mask, eps=1e-5):
    """Standardize one source across valid queries in each scene."""
    if not isinstance(scores, torch.Tensor) or scores.dim() != 2:
        raise ValueError("scores must have shape [B,Q]")
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != scores.shape
            or valid_mask.device != scores.device):
        raise ValueError("valid_mask must be bool with shape [B,Q]")
    if (not isinstance(eps, (float, int)) or isinstance(eps, bool)
            or not math.isfinite(float(eps)) or float(eps) <= 0.0):
        raise ValueError("eps must be finite and positive")
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("every sample needs at least one valid query")

    scores = torch.nan_to_num(
        scores.float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    valid = valid_mask.to(dtype=scores.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (scores * valid).sum(dim=1, keepdim=True) / count
    centered = (scores - mean) * valid
    variance = centered.square().sum(dim=1, keepdim=True) / count
    standardized = centered / (variance.sqrt() + float(eps))
    return standardized.masked_fill(~valid_mask, 0.0)


def straight_through_rank_normalize(scores, valid_mask=None, eps=1e-6):
    """Use exact ranks in the forward pass and a smooth proxy in backward.

    Raw source scales are incompatible, so inference uses rank normalization.
    A plain ``argsort`` rank has no gradient, however. The straight-through
    proxy keeps the exact scale-invariant forward values while allowing the
    ranking loss to reach the source-producing network during joint training.
    """
    if not isinstance(scores, torch.Tensor) or scores.dim() != 2:
        raise ValueError("scores must have shape [B,Q]")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != scores.shape):
        raise ValueError("valid_mask must be bool with shape [B,Q]")
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("every sample needs at least one valid query")

    valid = valid_mask.to(dtype=scores.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    finite_scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    mean = finite_scores.sum(dim=1, keepdim=True) / count
    centered = torch.where(
        valid_mask, finite_scores - mean, torch.zeros_like(scores)
    )
    variance = centered.square().sum(dim=1, keepdim=True) / count
    proxy = torch.sigmoid(centered / (variance.sqrt() + float(eps)))

    ranked_input = scores.masked_fill(~valid_mask, -1e4)
    exact = rank_normalize(ranked_input)
    normalized = proxy + (exact - proxy).detach()
    return normalized.masked_fill(~valid_mask, 0.0)


def compute_load_balance_loss(router_probs, expert_mask, valid_mask=None):
    """Top-k load balance loss normalized to a minimum value of one."""
    if (router_probs.shape != expert_mask.shape
            or router_probs.dim() != 3):
        raise ValueError("router_probs and expert_mask must match [B,Q,S]")
    num_experts = router_probs.shape[-1]
    if num_experts == 0:
        return router_probs.sum() * 0.0
    if valid_mask is None:
        valid_mask = torch.ones(
            router_probs.shape[:2], dtype=torch.bool,
            device=router_probs.device,
        )
    if (valid_mask.dtype != torch.bool
            or valid_mask.shape != router_probs.shape[:2]):
        raise ValueError("valid_mask must be bool with shape [B,Q]")
    if not bool(valid_mask.any().item()):
        return router_probs.sum() * 0.0

    valid_probs = router_probs.float()[valid_mask]
    valid_dispatch = expert_mask.float()[valid_mask]
    assignment_count = valid_dispatch.sum().clamp(min=1.0)
    fraction = valid_dispatch.sum(dim=0) / assignment_count
    probability = valid_probs.mean(dim=0)
    return float(num_experts) * torch.sum(fraction * probability)


def _box_cxcyczwhd_to_xyzxyz(boxes):
    center = boxes[..., :3]
    size = boxes[..., 3:6].clamp(min=1e-6)
    return torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1)


def compute_query_box_ious(candidate_boxes, gt_boxes, gt_valid):
    """Return each query's maximum IoU against valid GT boxes, shape [B,Q]."""
    if (candidate_boxes.dim() != 3 or candidate_boxes.shape[-1] != 6
            or gt_boxes.dim() != 3 or gt_boxes.shape[-1] != 6):
        raise ValueError("candidate_boxes and gt_boxes must have shape [B,N,6]")
    if candidate_boxes.shape[0] != gt_boxes.shape[0]:
        raise ValueError("candidate and GT batch sizes must match")
    if (gt_valid.dtype != torch.bool
            or gt_valid.shape != gt_boxes.shape[:2]):
        raise ValueError("gt_valid must be bool with shape [B,G]")
    if not bool(gt_valid.any(dim=1).all().item()):
        raise ValueError("every sample needs at least one valid GT box")

    candidate = _box_cxcyczwhd_to_xyzxyz(candidate_boxes.float())
    target = _box_cxcyczwhd_to_xyzxyz(gt_boxes.float())
    mins = torch.maximum(candidate[:, :, None, :3], target[:, None, :, :3])
    maxs = torch.minimum(candidate[:, :, None, 3:], target[:, None, :, 3:])
    intersection_size = (maxs - mins).clamp(min=0.0)
    intersection = intersection_size.prod(dim=-1)
    candidate_volume = (candidate[..., 3:] - candidate[..., :3]).prod(
        dim=-1
    )
    target_volume = (target[..., 3:] - target[..., :3]).prod(dim=-1)
    union = (
        candidate_volume[:, :, None]
        + target_volume[:, None, :]
        - intersection
    ).clamp(min=1e-6)
    ious = intersection / union
    ious = ious.masked_fill(~gt_valid[:, None, :], -1.0)
    return ious.max(dim=-1).values.clamp(min=0.0, max=1.0)


def _as_query_mask(mask_tensor):
    return as_query_mask_logits(mask_tensor, "query mask logits")


def build_fused_query_mask_logits(text_mask_logits, query_mask_logits,
                                  adaptive_weights):
    """Build the exact mask logits used by the standard MCLN evaluator."""
    if not (len(text_mask_logits) == len(query_mask_logits)
            == len(adaptive_weights)):
        raise ValueError("mask and adaptive-weight batch sizes must match")
    rows = []
    for text_row, query_row, weight in zip(
            text_mask_logits, query_mask_logits, adaptive_weights):
        text_row = _as_query_mask(text_row).float()
        query_row = _as_query_mask(query_row).float()
        if text_row.shape != query_row.shape:
            raise ValueError("text and query mask logits must align")
        rows.append(fuse_query_mask_logits(text_row, query_row, weight))
    return rows


def compute_query_mask_ious(query_mask_logits, gt_point_masks, superpoints,
                            gt_valid):
    """Return soft mask IoU to the best valid GT mask for every query."""
    if not isinstance(query_mask_logits, (list, tuple)):
        raise ValueError("query_mask_logits must be a per-sample list")
    batch_size = gt_point_masks.shape[0]
    if len(query_mask_logits) != batch_size:
        raise ValueError("mask-logit and GT batch sizes must match")
    if (superpoints.dim() != 2
            or superpoints.shape[0] != batch_size
            or superpoints.shape[1] != gt_point_masks.shape[-1]):
        raise ValueError("superpoints must align with GT point masks")
    if (gt_valid.dtype != torch.bool
            or gt_valid.shape != gt_point_masks.shape[:2]):
        raise ValueError("gt_valid must align with GT point masks")

    rows = []
    for batch_idx, logits in enumerate(query_mask_logits):
        logits = _as_query_mask(logits).float()
        valid_targets = gt_point_masks[batch_idx, gt_valid[batch_idx]].float()
        if valid_targets.numel() == 0:
            raise ValueError("every sample needs at least one valid GT mask")
        if logits.shape[1] <= int(superpoints[batch_idx].max().item()):
            raise ValueError("superpoint index exceeds query-mask width")
        target_superpoints = scatter_mean(
            valid_targets,
            superpoints[batch_idx],
            dim=-1,
            dim_size=logits.shape[1],
        )
        target_superpoints = (target_superpoints > 0.5).float()
        probabilities = logits.sigmoid()
        intersection = torch.matmul(
            probabilities, target_superpoints.transpose(0, 1)
        )
        union = (
            probabilities.sum(dim=1, keepdim=True)
            + target_superpoints.sum(dim=1).unsqueeze(0)
            - intersection
        ).clamp(min=1e-6)
        rows.append((intersection / union).max(dim=1).values)
    return torch.stack(rows, dim=0).clamp(min=0.0, max=1.0)


def threshold_aware_quality(ious):
    """Prioritize the two reported accuracy thresholds while retaining IoU."""
    return (
        ious
        + 0.25 * torch.sigmoid((ious - 0.25) / 0.05)
        + 0.50 * torch.sigmoid((ious - 0.50) / 0.05)
    )


def box_tier_constrained_mask_quality(box_ious, mask_ious):
    """Rank masks only after preserving the query's REC threshold tier."""
    if box_ious.shape != mask_ious.shape:
        raise ValueError("box and mask IoUs must have matching shape")
    box_tier = (
        (box_ious > 0.25).to(dtype=mask_ious.dtype)
        + (box_ious > 0.50).to(dtype=mask_ious.dtype)
    )
    # threshold_aware_quality is bounded below 1.75 for IoU in [0,1].
    # A 2.0 tier stride therefore makes the ordering lexicographic: box tier
    # first, mask quality second.
    return 2.0 * box_tier + threshold_aware_quality(mask_ious)


def listwise_quality_loss(scores, quality, valid_mask=None,
                          sample_mask=None, temperature=0.1):
    """KL listwise distillation from detached query-quality targets."""
    if scores.shape != quality.shape or scores.dim() != 2:
        raise ValueError("scores and quality must match [B,Q]")
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("temperature must be finite and positive")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if valid_mask.dtype != torch.bool or valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must be bool with shape [B,Q]")
    if sample_mask is None:
        sample_mask = torch.ones(
            scores.shape[0], dtype=torch.bool, device=scores.device
        )
    if (sample_mask.dtype != torch.bool
            or sample_mask.shape != (scores.shape[0],)):
        raise ValueError("sample_mask must be bool with shape [B]")

    informative = (
        quality.masked_fill(~valid_mask, -1.0).max(dim=1).values
        - quality.masked_fill(~valid_mask, 2.0).min(dim=1).values
    ) > 1e-6
    active = sample_mask & informative & valid_mask.any(dim=1)
    if not bool(active.any().item()):
        return scores.sum() * 0.0

    scaled_scores = (scores / float(temperature)).masked_fill(
        ~valid_mask, -1e4
    )
    scaled_quality = (
        quality.detach() / float(temperature)
    ).masked_fill(~valid_mask, -1e4)
    target = F.softmax(scaled_quality, dim=1)
    log_probs = F.log_softmax(scaled_scores, dim=1)
    per_sample = F.kl_div(log_probs, target, reduction="none").sum(dim=1)
    return per_sample[active].mean()


def threshold_anchor_ranking_loss(
        scores, box_ious, anchor_indices, valid_mask=None, sample_mask=None,
        thresholds=(0.25, 0.50), margin=0.05):
    """Protect correct shared-source queries while learning explicit fixes."""
    if scores.shape != box_ious.shape or scores.dim() != 2:
        raise ValueError("scores and box_ious must match [B,Q]")
    if (not isinstance(anchor_indices, torch.Tensor)
            or anchor_indices.dtype != torch.long
            or anchor_indices.shape != (scores.shape[0],)):
        raise ValueError("anchor_indices must be int64 with shape [B]")
    if (bool((anchor_indices < 0).any().item())
            or bool((anchor_indices >= scores.shape[1]).any().item())):
        raise ValueError("anchor index is out of range")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if valid_mask.dtype != torch.bool or valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must be bool with shape [B,Q]")
    if sample_mask is None:
        sample_mask = torch.ones(
            scores.shape[0], dtype=torch.bool, device=scores.device
        )
    if (sample_mask.dtype != torch.bool
            or sample_mask.shape != (scores.shape[0],)):
        raise ValueError("sample_mask must be bool with shape [B]")
    if (not isinstance(margin, (float, int)) or isinstance(margin, bool)
            or not math.isfinite(float(margin)) or float(margin) < 0.0):
        raise ValueError("anchor margin must be finite and non-negative")
    if (not isinstance(thresholds, (list, tuple)) or not thresholds
            or any(not isinstance(value, (float, int))
                   or isinstance(value, bool)
                   or not math.isfinite(float(value))
                   for value in thresholds)):
        raise ValueError("thresholds must be a non-empty finite sequence")

    row_index = torch.arange(scores.shape[0], device=scores.device)
    if not bool(valid_mask[row_index, anchor_indices].all().item()):
        raise ValueError("every anchor query must be valid")
    anchor_scores = scores[row_index, anchor_indices]
    detached_ious = box_ious.detach()
    losses = []
    for threshold in thresholds:
        correct = (detached_ious > float(threshold)) & valid_mask
        anchor_correct = correct[row_index, anchor_indices]

        incorrect = valid_mask & ~correct
        protect_active = (
            sample_mask & anchor_correct & incorrect.any(dim=1)
        )
        if bool(protect_active.any().item()):
            hardest_incorrect = scores.masked_fill(
                ~incorrect, -1e4
            ).max(dim=1).values
            losses.append(F.relu(
                hardest_incorrect[protect_active]
                + float(margin)
                - anchor_scores[protect_active]
            ))

        fix_active = sample_mask & ~anchor_correct & correct.any(dim=1)
        if bool(fix_active.any().item()):
            best_correct_idx = detached_ious.masked_fill(
                ~correct, -1.0
            ).argmax(dim=1)
            best_correct_scores = scores[row_index, best_correct_idx]
            hardest_incorrect = scores.masked_fill(
                ~incorrect, -1e4
            ).max(dim=1).values
            losses.append(F.relu(
                hardest_incorrect[fix_active]
                + float(margin)
                - best_correct_scores[fix_active]
            ))

    if not losses:
        return scores.sum() * 0.0
    return torch.cat(losses).mean()


def compute_source_moe_ranking_loss(
        scores, candidate_boxes, gt_boxes, gt_valid,
        valid_mask=None, sample_mask=None, query_mask_logits=None,
        gt_point_masks=None, superpoints=None, temperature=0.1,
        mask_loss_weight=0.25, anchor_indices=None,
        anchor_loss_weight=0.0, anchor_margin=0.05):
    """Compute box-primary, mask-aware training loss for query reranking."""
    if (not isinstance(mask_loss_weight, (float, int))
            or isinstance(mask_loss_weight, bool)
            or not math.isfinite(float(mask_loss_weight))
            or float(mask_loss_weight) < 0.0):
        raise ValueError("mask_loss_weight must be finite and non-negative")
    if (not isinstance(anchor_loss_weight, (float, int))
            or isinstance(anchor_loss_weight, bool)
            or not math.isfinite(float(anchor_loss_weight))
            or float(anchor_loss_weight) < 0.0):
        raise ValueError("anchor_loss_weight must be finite and non-negative")
    with torch.no_grad():
        box_ious = compute_query_box_ious(
            candidate_boxes.detach(), gt_boxes.detach(), gt_valid
        )
        box_quality = threshold_aware_quality(box_ious)
    box_loss = listwise_quality_loss(
        scores, box_quality, valid_mask=valid_mask,
        sample_mask=sample_mask, temperature=temperature,
    )

    mask_loss = scores.sum() * 0.0
    mask_ious = None
    mask_inputs = (query_mask_logits, gt_point_masks, superpoints)
    if all(value is not None for value in mask_inputs):
        with torch.no_grad():
            mask_ious = compute_query_mask_ious(
                query_mask_logits, gt_point_masks.detach(),
                superpoints, gt_valid,
            )
            mask_quality = box_tier_constrained_mask_quality(
                box_ious, mask_ious
            )
        mask_loss = listwise_quality_loss(
            scores, mask_quality, valid_mask=valid_mask,
            sample_mask=sample_mask, temperature=temperature,
        )
    anchor_loss = scores.sum() * 0.0
    if anchor_indices is not None and float(anchor_loss_weight) > 0.0:
        anchor_loss = threshold_anchor_ranking_loss(
            scores,
            box_ious,
            anchor_indices,
            valid_mask=valid_mask,
            sample_mask=sample_mask,
            margin=anchor_margin,
        )
    total = (
        box_loss
        + float(mask_loss_weight) * mask_loss
        + float(anchor_loss_weight) * anchor_loss
    )
    return {
        "loss": total,
        "box_loss": box_loss,
        "mask_loss": mask_loss,
        "anchor_loss": anchor_loss,
        "box_ious": box_ious,
        "mask_ious": mask_ious,
    }


def threshold_transition_targets(quality, default_indices,
                                 thresholds=(0.25, 0.50)):
    """Label every query as break, neutral, or fix versus the fallback.

    Class ids are ordered as ``break=0, neutral=1, fix=2``.  The labels are
    defined only by threshold transitions, so they transfer across datasets
    without a dataset-specific IoU-gap heuristic.
    """
    if not isinstance(quality, torch.Tensor) or quality.dim() != 2:
        raise ValueError("quality must have shape [B,Q]")
    if (not isinstance(default_indices, torch.Tensor)
            or default_indices.dtype != torch.long
            or default_indices.shape != (quality.shape[0],)
            or default_indices.device != quality.device):
        raise ValueError("default_indices must be int64 with shape [B]")
    if (bool((default_indices < 0).any().item())
            or bool((default_indices >= quality.shape[1]).any().item())):
        raise ValueError("default index is out of range")
    if (not isinstance(thresholds, (tuple, list)) or not thresholds
            or any(not isinstance(value, (float, int))
                   or isinstance(value, bool)
                   or not math.isfinite(float(value))
                   for value in thresholds)):
        raise ValueError("thresholds must be a non-empty finite sequence")

    row_index = torch.arange(quality.shape[0], device=quality.device)
    default_quality = quality[row_index, default_indices]
    labels = []
    for threshold in thresholds:
        default_ok = default_quality > float(threshold)
        query_ok = quality > float(threshold)
        target = torch.ones_like(quality, dtype=torch.long)
        target = torch.where(
            default_ok.unsqueeze(1) & ~query_ok,
            torch.zeros_like(target),
            target,
        )
        target = torch.where(
            ~default_ok.unsqueeze(1) & query_ok,
            torch.full_like(target, 2),
            target,
        )
        labels.append(target)
    return torch.stack(labels, dim=2)


def _class_balanced_focal_loss(logits, targets, active_mask,
                               gamma=2.0, false_override_weight=2.0):
    if (logits.dim() != 3 or logits.shape[-1] != 3
            or targets.shape != logits.shape[:2]
            or active_mask.shape != logits.shape[:2]
            or active_mask.dtype != torch.bool):
        raise ValueError("focal inputs must align as [B,Q,3]")
    if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool)
            or not math.isfinite(float(gamma)) or float(gamma) < 0.0):
        raise ValueError("focal gamma must be finite and non-negative")
    if (not isinstance(false_override_weight, (float, int))
            or isinstance(false_override_weight, bool)
            or not math.isfinite(float(false_override_weight))
            or float(false_override_weight) < 1.0):
        raise ValueError(
            "false_override_weight must be finite and at least one"
        )
    if not bool(active_mask.any().item()):
        return logits.sum() * 0.0

    active_logits = logits[active_mask]
    active_targets = targets[active_mask]
    counts = torch.bincount(active_targets, minlength=3).to(
        dtype=active_logits.dtype
    )
    inverse_frequency = (
        float(active_targets.numel())
        / (3.0 * counts.clamp(min=1.0))
    )
    # A break is precisely a false override of a correct fallback query.
    # Raising only that class cost makes the learned switch conservative,
    # while inverse-frequency balancing adapts to each dataset automatically.
    inverse_frequency[0] = (
        inverse_frequency[0] * float(false_override_weight)
    )
    sample_weight = inverse_frequency[active_targets]
    log_probs = F.log_softmax(active_logits, dim=-1)
    probs = log_probs.exp()
    row_index = torch.arange(
        active_targets.numel(), device=active_targets.device
    )
    target_log_prob = log_probs[row_index, active_targets]
    target_prob = probs[row_index, active_targets]
    focal = (1.0 - target_prob).clamp(min=0.0).pow(float(gamma))
    weighted = -sample_weight * focal * target_log_prob
    return weighted.sum() / sample_weight.sum().clamp(min=1e-6)


def _class_balanced_binary_focal_loss(
        logits, targets, active_mask, gamma=2.0,
        false_positive_weight=2.0):
    """Binary focal loss that balances safety positives and hard negatives."""
    if (logits.dim() != 2 or targets.shape != logits.shape
            or targets.dtype != torch.bool
            or active_mask.shape != logits.shape
            or active_mask.dtype != torch.bool):
        raise ValueError("binary focal inputs must align as [B,Q]")
    if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool)
            or not math.isfinite(float(gamma)) or float(gamma) < 0.0):
        raise ValueError("binary focal gamma must be finite and non-negative")
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "false_positive_weight must be finite and at least one"
        )
    if not bool(active_mask.any().item()):
        return logits.sum() * 0.0

    active_logits = logits[active_mask].float()
    active_targets = targets[active_mask]
    counts = torch.bincount(
        active_targets.long(), minlength=2
    ).to(dtype=active_logits.dtype)
    class_weight = (
        float(active_targets.numel()) / (2.0 * counts.clamp(min=1.0))
    )
    class_weight[0] = (
        class_weight[0] * float(false_positive_weight)
    )
    target_values = active_targets.to(dtype=active_logits.dtype)
    binary_ce = F.binary_cross_entropy_with_logits(
        active_logits, target_values, reduction="none"
    )
    target_probability = torch.where(
        active_targets,
        active_logits.sigmoid(),
        (-active_logits).sigmoid(),
    )
    focal = (1.0 - target_probability).pow(float(gamma))
    weights = class_weight[active_targets.long()]
    return (weights * focal * binary_ce).sum() / weights.sum().clamp(min=1e-6)


def _cost_sensitive_binary_focal_loss(
        logits, targets, active_mask, gamma=2.0,
        false_positive_weight=2.0):
    """Binary focal risk without removing the empirical class prior."""
    if (logits.dim() != 2 or targets.shape != logits.shape
            or targets.dtype != torch.bool
            or active_mask.shape != logits.shape
            or active_mask.dtype != torch.bool):
        raise ValueError("binary focal inputs must align as [B,Q]")
    if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool)
            or not math.isfinite(float(gamma)) or float(gamma) < 0.0):
        raise ValueError("binary focal gamma must be finite and non-negative")
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "false_positive_weight must be finite and at least one"
        )
    if not bool(active_mask.any().item()):
        return logits.sum() * 0.0

    active_logits = logits[active_mask].float()
    active_targets = targets[active_mask]
    target_values = active_targets.to(dtype=active_logits.dtype)
    binary_ce = F.binary_cross_entropy_with_logits(
        active_logits, target_values, reduction="none"
    )
    target_probability = torch.where(
        active_targets,
        active_logits.sigmoid(),
        (-active_logits).sigmoid(),
    )
    focal = (1.0 - target_probability).pow(float(gamma))
    weights = torch.where(
        active_targets,
        active_logits.new_ones(()),
        active_logits.new_full((), float(false_positive_weight)),
    )
    return (weights * focal * binary_ce).sum() / weights.sum().clamp(min=1e-6)


def _prior_restored_balanced_benefit_loss(
        logits, targets, active_mask, false_positive_weight):
    """Balance rare benefits while restoring the empirical raw boundary.

    Class-balanced BCE learns a likelihood ratio under a reweighted class
    prior.  Adding ``log((1-p)/p)`` to the training logit cancels that prior
    shift, leaving the unshifted output calibrated for the observed prior and
    false-positive cost.  The deployed margin therefore keeps the fixed zero
    boundary without a validation threshold.
    """
    if (logits.dim() != 2 or targets.shape != logits.shape
            or targets.dtype != torch.bool
            or active_mask.shape != logits.shape
            or active_mask.dtype != torch.bool):
        raise ValueError(
            "prior-restored benefit inputs must align as [B,Q]"
        )
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "prior-restored false-positive weight must be at least one"
        )
    if not bool(active_mask.any().item()):
        zero = logits.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    active_logits = logits[active_mask].float()
    active_targets = targets[active_mask]
    positive_count = active_targets.float().sum()
    negative_count = active_targets.numel() - positive_count
    # Beta(1,1) smoothing keeps a minibatch with no useful candidate finite.
    prior_shift = torch.log(
        (negative_count + 1.0) / (positive_count + 1.0)
    ).detach()
    total = positive_count + negative_count + 2.0
    positive_weight = total / (2.0 * (positive_count + 1.0))
    negative_weight = (
        total / (2.0 * (negative_count + 1.0))
        * float(false_positive_weight)
    )
    adjusted_logits = active_logits + prior_shift
    targets_float = active_targets.to(dtype=active_logits.dtype)
    # gamma=0 is intentional: focal reweighting would destroy the calibrated
    # likelihood-ratio interpretation of the raw zero deployment boundary.
    per_item = F.binary_cross_entropy_with_logits(
        adjusted_logits, targets_float, reduction="none"
    )
    weights = torch.where(
        active_targets,
        positive_weight.to(dtype=active_logits.dtype),
        negative_weight.to(dtype=active_logits.dtype),
    )
    loss = (weights * per_item).sum() / weights.sum().clamp(min=1e-6)
    positive_prior = (
        (positive_count + 1.0) / (positive_count + negative_count + 2.0)
    )
    return loss, prior_shift, positive_prior.detach()


def _rowwise_boundary_calibration_loss(
        candidate_margin, decision_utility, active_mask, sample_mask,
        temperature, false_positive_weight, boundary_gap=0.05):
    """Train the row-level max margin against the fixed deployment boundary.

    Candidate-level class balancing changes the effective prior.  This
    auxiliary loss instead supervises the actual row action: a row with at
    least one beneficial candidate must have a positive smooth-max margin,
    while a row with no beneficial candidate must keep its smooth-max below
    zero.  Log-mean-exp makes an all-zero, variable-size candidate set score
    exactly zero, preserving V19 identity at initialization.
    """
    if (not isinstance(candidate_margin, torch.Tensor)
            or candidate_margin.dim() != 2
            or decision_utility.shape != candidate_margin.shape
            or active_mask.shape != candidate_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "row boundary inputs must align as [B,Q] and [B]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "row boundary temperature must be finite and positive"
        )
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "row boundary false-positive weight must be at least one"
        )
    if (not isinstance(boundary_gap, (float, int))
            or isinstance(boundary_gap, bool)
            or not math.isfinite(float(boundary_gap))
            or float(boundary_gap) < 0.0):
        raise ValueError("row boundary gap must be finite and non-negative")

    active = active_mask & sample_mask.unsqueeze(1)
    row_valid = active.any(dim=1)
    if not bool(row_valid.any().item()):
        zero = candidate_margin.sum() * 0.0
        return {
            "loss": zero,
            "positive_loss": zero.detach(),
            "fallback_loss": zero.detach(),
            "row_margin": candidate_margin.new_zeros(
                candidate_margin.shape[0]
            ).detach(),
            "positive_ratio": zero.detach(),
        }

    work_margin = candidate_margin.float().masked_fill(~active, -1e4)
    valid_count = active.sum(dim=1).clamp(min=1).to(dtype=work_margin.dtype)
    log_mean_exp = (
        torch.logsumexp(work_margin / float(temperature), dim=1)
        - valid_count.log()
    ) * float(temperature)
    utility = decision_utility.float().masked_fill(~active, -1e4)
    row_target = utility.max(dim=1).values > 0.0
    signed_margin = torch.where(
        row_target,
        log_mean_exp - float(boundary_gap),
        -log_mean_exp - float(boundary_gap),
    )
    per_row = F.softplus(-signed_margin)
    valid_targets = row_target[row_valid]
    count = torch.bincount(
        valid_targets.long(), minlength=2
    ).to(dtype=per_row.dtype)
    total = valid_targets.numel()
    class_weight = per_row.new_tensor((
        float(total) / (2.0 * float(count[0].clamp(min=1.0).item()))
        * float(false_positive_weight),
        float(total) / (2.0 * float(count[1].clamp(min=1.0).item())),
    ))
    weights = class_weight[row_target.long()]
    weighted = per_row * weights
    loss = weighted[row_valid].sum() / weights[row_valid].sum().clamp(min=1e-6)
    positive_rows = row_valid & row_target
    fallback_rows = row_valid & ~row_target
    positive_loss = (
        per_row[positive_rows].mean().detach()
        if bool(positive_rows.any().item()) else loss.detach() * 0.0
    )
    fallback_loss = (
        per_row[fallback_rows].mean().detach()
        if bool(fallback_rows.any().item()) else loss.detach() * 0.0
    )
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "fallback_loss": fallback_loss,
        "row_margin": log_mean_exp.detach(),
        "positive_ratio": row_target[row_valid].float().mean().detach(),
    }


def _prior_corrected_setwise_action_loss(
        candidate_margin, decision_utility, active_mask, sample_mask,
        temperature, gamma=2.0, false_positive_weight=2.0):
    """Train one fallback-or-candidate action with a calibrated raw margin.

    Positive and fallback rows are normalized separately so sparse useful
    rows cannot disappear behind the number of candidate negatives.  The
    empirical row prior and false-positive cost are then restored as a
    detached log-odds offset.  Consequently the unadjusted margin keeps zero
    as the deployment boundary without a validation-set threshold.
    """
    if (candidate_margin.dim() != 2
            or decision_utility.shape != candidate_margin.shape
            or active_mask.shape != candidate_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "joint action inputs must align as candidate [B,Q] and row [B]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "joint action temperature must be finite and positive"
        )
    if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool)
            or not math.isfinite(float(gamma)) or float(gamma) < 0.0):
        raise ValueError(
            "joint action focal gamma must be finite and non-negative"
        )
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "joint action false-positive weight must be at least one"
        )
    if not bool(sample_mask.any().item()):
        zero = candidate_margin.sum() * 0.0
        target = build_risk_separated_action_target(
            decision_utility, active_mask, temperature=float(temperature)
        )
        return zero, target, zero.detach(), zero.detach()

    target = build_risk_separated_action_target(
        decision_utility, active_mask, temperature=float(temperature)
    )
    positive_rows = target[:, 1:].sum(dim=1) > 0.0
    active_positive = (positive_rows & sample_mask).float().sum()
    active_count = sample_mask.float().sum()
    # Beta(1,1) smoothing keeps all-positive/all-fallback debug batches finite.
    positive_prior = (active_positive + 1.0) / (active_count + 2.0)
    prior_log_odds = torch.log(
        positive_prior / (1.0 - positive_prior).clamp(min=1e-6)
    ) - math.log(float(false_positive_weight))

    calibrated_candidate_logits = (
        candidate_margin.float() - prior_log_odds.detach()
    ).masked_fill(~active_mask, -1e4)
    logits = torch.cat((
        torch.zeros(
            candidate_margin.shape[0], 1,
            dtype=calibrated_candidate_logits.dtype,
            device=calibrated_candidate_logits.device,
        ),
        calibrated_candidate_logits,
    ), dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    cross_entropy = -(target * log_probabilities).sum(dim=1)
    positive_support = target[:, 1:] > 0.0
    positive_mass = (
        probabilities[:, 1:] * positive_support.to(probabilities.dtype)
    ).sum(dim=1)
    target_probability = torch.where(
        positive_rows, positive_mass, probabilities[:, 0]
    )
    focal = (1.0 - target_probability).clamp(min=0.0).pow(float(gamma))
    row_loss = focal * cross_entropy

    terms = []
    fallback_rows = sample_mask & ~positive_rows
    useful_rows = sample_mask & positive_rows
    if bool(fallback_rows.any().item()):
        terms.append(row_loss[fallback_rows].mean())
    if bool(useful_rows.any().item()):
        terms.append(row_loss[useful_rows].mean())
    loss = sum(terms) / float(len(terms))
    return (
        loss,
        target,
        positive_prior.detach(),
        prior_log_odds.detach(),
    )


def _empirical_setwise_action_risk_loss(
        candidate_margin, decision_utility, active_mask, sample_mask,
        temperature, gamma=2.0, false_positive_weight=2.0):
    """Score fallback versus candidates under the empirical row prior.

    The fallback is an explicit class with fixed logit zero.  Unlike the
    separately balanced objectives, this proper multiclass risk averages over
    the observed rows and applies the false-switch cost only to fallback rows.
    The learned raw margin can therefore use zero directly at deployment.
    """
    if (candidate_margin.dim() != 2
            or decision_utility.shape != candidate_margin.shape
            or active_mask.shape != candidate_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "empirical set-risk inputs must align as candidate [B,Q] and "
            "row [B]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "empirical set-risk temperature must be finite and positive"
        )
    if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool)
            or not math.isfinite(float(gamma)) or float(gamma) < 0.0):
        raise ValueError(
            "empirical set-risk focal gamma must be finite and non-negative"
        )
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "empirical set-risk false-positive weight must be at least one"
        )

    target = build_risk_separated_action_target(
        decision_utility, active_mask, temperature=float(temperature)
    )
    positive_rows = target[:, 1:].sum(dim=1) > 0.0
    active_count = sample_mask.float().sum()
    active_positive = (positive_rows & sample_mask).float().sum()
    positive_prior = (active_positive + 1.0) / (active_count + 2.0)
    prior_log_odds = torch.log(
        positive_prior / (1.0 - positive_prior).clamp(min=1e-6)
    ) - math.log(float(false_positive_weight))
    if not bool(sample_mask.any().item()):
        zero = candidate_margin.sum() * 0.0
        return zero, target, positive_prior.detach(), prior_log_odds.detach()

    candidate_logits = candidate_margin.float().masked_fill(
        ~active_mask, -1e4
    )
    logits = torch.cat((
        torch.zeros(
            candidate_margin.shape[0], 1,
            dtype=candidate_logits.dtype,
            device=candidate_logits.device,
        ),
        candidate_logits,
    ), dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    cross_entropy = -(target * log_probabilities).sum(dim=1)
    positive_support = target[:, 1:] > 0.0
    positive_mass = (
        probabilities[:, 1:] * positive_support.to(probabilities.dtype)
    ).sum(dim=1)
    target_probability = torch.where(
        positive_rows, positive_mass, probabilities[:, 0]
    )
    focal = (1.0 - target_probability).clamp(min=0.0).pow(float(gamma))
    row_weight = torch.where(
        positive_rows,
        cross_entropy.new_ones(()),
        cross_entropy.new_full((), float(false_positive_weight)),
    )
    weighted = row_weight * focal * cross_entropy
    loss = weighted[sample_mask].mean()
    return loss, target, positive_prior.detach(), prior_log_odds.detach()


def build_relative_risk_action_target(decision_utility, active_mask,
                                      temperature, fallback_cost=2.0):
    """Build a fallback-preserving target from candidate utility.

    The fallback is always an explicit action.  Positive candidate rows retain
    non-zero fallback mass, while rows without a positive candidate remain
    fallback-only.  ``fallback_cost`` is a deployment-independent risk prior,
    not a validation threshold, so the target transfers across datasets.
    """
    if (not isinstance(decision_utility, torch.Tensor)
            or decision_utility.dim() != 2
            or active_mask.shape != decision_utility.shape
            or active_mask.dtype != torch.bool
            or active_mask.device != decision_utility.device):
        raise ValueError(
            "relative-risk utility and active mask must align [B,Q]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "relative-risk temperature must be finite and positive"
        )
    if (not isinstance(fallback_cost, (float, int))
            or isinstance(fallback_cost, bool)
            or not math.isfinite(float(fallback_cost))
            or float(fallback_cost) < 1.0):
        raise ValueError(
            "relative-risk fallback_cost must be finite and at least one"
        )

    positive = active_mask & (decision_utility > 0.0)
    candidate_logits = (
        decision_utility.detach().float() / float(temperature)
    ).masked_fill(~positive, -1e4)
    fallback_logit = decision_utility.new_full(
        (decision_utility.shape[0], 1),
        math.log(float(fallback_cost)),
        dtype=torch.float32,
    )
    target = F.softmax(
        torch.cat((fallback_logit, candidate_logits), dim=1), dim=1
    )
    has_positive = positive.any(dim=1)
    target[:, 0] = torch.where(
        has_positive,
        target[:, 0],
        torch.ones_like(target[:, 0]),
    )
    target[:, 1:] = torch.where(
        has_positive.unsqueeze(1),
        target[:, 1:],
        torch.zeros_like(target[:, 1:]),
    )
    return target


def _relative_setwise_action_risk_loss(
        candidate_margin, decision_utility, active_mask, sample_mask,
        temperature, gamma=2.0, false_positive_weight=2.0):
    """Calibrate a candidate-vs-fallback margin with an explicit fallback.

    Unlike V23's risk-separated target, useful rows are not forced to switch:
    the target distribution keeps fallback mass and the deployment boundary
    remains the raw zero margin.
    """
    if (candidate_margin.dim() != 2
            or decision_utility.shape != candidate_margin.shape
            or active_mask.shape != candidate_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "relative set-risk inputs must align as candidate [B,Q] and row [B]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "relative set-risk temperature must be finite and positive"
        )
    if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool)
            or not math.isfinite(float(gamma)) or float(gamma) < 0.0):
        raise ValueError(
            "relative set-risk focal gamma must be finite and non-negative"
        )
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError(
            "relative set-risk false-positive weight must be at least one"
        )

    target = build_relative_risk_action_target(
        decision_utility,
        active_mask,
        temperature=float(temperature),
        fallback_cost=float(false_positive_weight),
    )
    if not bool(sample_mask.any().item()):
        zero = candidate_margin.sum() * 0.0
        return zero, target, zero.detach(), zero.detach()

    candidate_logits = candidate_margin.float().masked_fill(
        ~active_mask, -1e4
    )
    logits = torch.cat((
        torch.zeros(
            candidate_margin.shape[0], 1,
            dtype=candidate_logits.dtype,
            device=candidate_logits.device,
        ),
        candidate_logits,
    ), dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    target_probability = (target.to(dtype=probabilities.dtype)
                          * probabilities).sum(dim=1)
    cross_entropy = -(target.to(dtype=log_probabilities.dtype)
                      * log_probabilities).sum(dim=1)
    focal = (1.0 - target_probability).clamp(min=0.0).pow(float(gamma))
    # Preserve the empirical row prior while charging extra for a false
    # candidate override.  This is a training cost only; inference still uses
    # the fixed zero boundary on the raw margin.
    target_fallback = target[:, 0]
    row_weight = torch.where(
        target_fallback > 0.5,
        cross_entropy.new_full((), float(false_positive_weight)),
        cross_entropy.new_ones(()),
    )
    weighted = row_weight * focal * cross_entropy
    loss = weighted[sample_mask].mean()
    positive_mass = 1.0 - target_fallback
    positive_prior = positive_mass[sample_mask].mean()
    prior_log_odds = torch.log(
        positive_prior / (1.0 - positive_prior).clamp(min=1e-6)
    )
    return loss, target, positive_prior.detach(), prior_log_odds.detach()


def _balanced_deployment_boundary_loss(
        candidate_margin, decision_utility, active_mask, sample_mask,
        temperature):
    """Align raw candidate margins with the fixed deployment boundary zero."""
    if (candidate_margin.dim() != 2
            or decision_utility.shape != candidate_margin.shape
            or active_mask.shape != candidate_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "boundary inputs must align as candidate [B,Q] and row [B]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("boundary temperature must be finite and positive")

    def masked_smooth_max(values, mask):
        scaled = (values.float() / float(temperature)).masked_fill(
            ~mask, -1e4
        )
        count = mask.sum(dim=1).clamp(min=1).to(dtype=scaled.dtype)
        smooth = float(temperature) * (
            torch.logsumexp(scaled, dim=1) - count.log()
        )
        return torch.where(
            mask.any(dim=1), smooth, values.float().sum(dim=1) * 0.0
        )

    positive_candidates = active_mask & (decision_utility > 0.0)
    has_active = active_mask.any(dim=1)
    positive_rows = sample_mask & positive_candidates.any(dim=1)
    fallback_rows = sample_mask & has_active & ~positive_candidates.any(dim=1)
    positive_boundary = masked_smooth_max(
        candidate_margin, positive_candidates
    )
    fallback_boundary = masked_smooth_max(candidate_margin, active_mask)
    zero = candidate_margin.sum() * 0.0
    positive_loss = (
        F.softplus(-positive_boundary[positive_rows] / float(temperature)).mean()
        if bool(positive_rows.any().item()) else zero
    )
    fallback_loss = (
        F.softplus(fallback_boundary[fallback_rows] / float(temperature)).mean()
        if bool(fallback_rows.any().item()) else zero
    )
    terms = []
    if bool(positive_rows.any().item()):
        terms.append(positive_loss)
    if bool(fallback_rows.any().item()):
        terms.append(fallback_loss)
    loss = sum(terms) / float(len(terms)) if terms else zero
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "fallback_loss": fallback_loss,
        "positive_rows": positive_rows,
        "fallback_rows": fallback_rows,
    }


def _cost_sensitive_cross_entropy(logits, targets, active_mask,
                                  false_override_weight):
    """Three-way CE with a fixed break cost and no empirical class prior."""
    if (logits.dim() != 3 or logits.shape[-1] != 3
            or targets.shape != logits.shape[:2]
            or active_mask.shape != logits.shape[:2]
            or active_mask.dtype != torch.bool):
        raise ValueError("cross-entropy inputs must align as [B,Q,3]")
    if not bool(active_mask.any().item()):
        return logits.sum() * 0.0
    class_weight = logits.new_tensor((
        float(false_override_weight), 1.0, 1.0
    ))
    return F.cross_entropy(
        logits[active_mask], targets[active_mask], weight=class_weight
    )


def _absolute_quality_prediction_loss(
        box_threshold_logits, box_iou_estimate, box_ious, active_mask,
        thresholds, mask_threshold_logits=None, mask_iou_estimate=None,
        mask_ious=None, mask_loss_weight=0.25):
    """Train dense absolute query quality without changing class priors."""
    expected_threshold_shape = box_ious.shape + (len(thresholds),)
    if (box_threshold_logits.shape != expected_threshold_shape
            or box_iou_estimate.shape != box_ious.shape
            or active_mask.shape != box_ious.shape
            or active_mask.dtype != torch.bool):
        raise ValueError(
            "absolute box quality tensors must align as [B,Q,T]"
        )
    if not bool(torch.isfinite(box_threshold_logits).all().item()):
        raise ValueError("absolute box threshold logits must be finite")
    if (not bool(torch.isfinite(box_iou_estimate).all().item())
            or bool((box_iou_estimate < 0.0).any().item())
            or bool((box_iou_estimate > 1.0).any().item())):
        raise ValueError("absolute box IoU estimates must lie in [0,1]")
    if not bool(active_mask.any().item()):
        zero = box_threshold_logits.sum() * 0.0
        return {
            "loss": zero,
            "box_threshold_loss": zero,
            "box_iou_loss": zero,
            "mask_threshold_loss": zero,
            "mask_iou_loss": zero,
        }

    box_threshold_targets = torch.stack([
        box_ious.detach() > float(threshold)
        for threshold in thresholds
    ], dim=-1).to(dtype=box_threshold_logits.dtype)
    box_threshold_loss = F.binary_cross_entropy_with_logits(
        box_threshold_logits[active_mask],
        box_threshold_targets[active_mask],
    )
    box_iou_loss = F.smooth_l1_loss(
        box_iou_estimate[active_mask].float(),
        box_ious.detach()[active_mask].float(),
    )

    mask_threshold_loss = box_threshold_logits.sum() * 0.0
    mask_iou_loss = box_threshold_logits.sum() * 0.0
    has_mask_targets = mask_ious is not None
    if has_mask_targets:
        if (mask_threshold_logits is None or mask_iou_estimate is None
                or mask_threshold_logits.shape != expected_threshold_shape
                or mask_iou_estimate.shape != box_ious.shape
                or mask_ious.shape != box_ious.shape):
            raise ValueError(
                "absolute mask quality tensors must align with box quality"
            )
        if (not bool(torch.isfinite(mask_threshold_logits).all().item())
                or not bool(torch.isfinite(mask_iou_estimate).all().item())
                or bool((mask_iou_estimate < 0.0).any().item())
                or bool((mask_iou_estimate > 1.0).any().item())):
            raise ValueError("absolute mask quality predictions are invalid")
        mask_threshold_targets = torch.stack([
            mask_ious.detach() > float(threshold)
            for threshold in thresholds
        ], dim=-1).to(dtype=mask_threshold_logits.dtype)
        mask_threshold_loss = F.binary_cross_entropy_with_logits(
            mask_threshold_logits[active_mask],
            mask_threshold_targets[active_mask],
        )
        mask_iou_loss = F.smooth_l1_loss(
            mask_iou_estimate[active_mask].float(),
            mask_ious.detach()[active_mask].float(),
        )

    box_loss = 0.5 * (box_threshold_loss + box_iou_loss)
    if has_mask_targets:
        mask_loss = 0.5 * (mask_threshold_loss + mask_iou_loss)
        total = (
            box_loss + float(mask_loss_weight) * mask_loss
        ) / (1.0 + float(mask_loss_weight))
    else:
        total = box_loss
    return {
        "loss": total,
        "box_threshold_loss": box_threshold_loss,
        "box_iou_loss": box_iou_loss,
        "mask_threshold_loss": mask_threshold_loss,
        "mask_iou_loss": mask_iou_loss,
    }


def dense_quality_expected_score(
        box_threshold_logits, box_iou, mask_threshold_logits=None,
        mask_iou=None, mask_utility_weight=0.25):
    """Map absolute quality predictions to a coordinate-wise monotonic score."""
    if (not isinstance(box_threshold_logits, torch.Tensor)
            or box_threshold_logits.dim() < 1
            or box_threshold_logits.shape[-1] < 1
            or not isinstance(box_iou, torch.Tensor)
            or box_iou.shape != box_threshold_logits.shape[:-1]):
        raise ValueError("dense box quality predictions are invalid")
    if (not isinstance(mask_utility_weight, (float, int))
            or isinstance(mask_utility_weight, bool)
            or not math.isfinite(float(mask_utility_weight))
            or float(mask_utility_weight) < 0.0):
        raise ValueError("mask_utility_weight must be finite and non-negative")
    if (not bool(torch.isfinite(box_threshold_logits).all().item())
            or not bool(torch.isfinite(box_iou).all().item())
            or bool((box_iou < 0.0).any().item())
            or bool((box_iou > 1.0).any().item())):
        raise ValueError("dense box quality predictions must be finite")
    tier_weights = torch.arange(
        1, box_threshold_logits.shape[-1] + 1,
        dtype=box_threshold_logits.dtype,
        device=box_threshold_logits.device,
    )
    denominator = tier_weights.sum() + 1.0
    box_quality = (
        (box_threshold_logits.sigmoid() * tier_weights).sum(dim=-1)
        + box_iou
    ) / denominator
    if mask_threshold_logits is None and mask_iou is None:
        return box_quality
    if (not isinstance(mask_threshold_logits, torch.Tensor)
            or mask_threshold_logits.shape != box_threshold_logits.shape
            or not isinstance(mask_iou, torch.Tensor)
            or mask_iou.shape != box_iou.shape
            or not bool(torch.isfinite(mask_threshold_logits).all().item())
            or not bool(torch.isfinite(mask_iou).all().item())
            or bool((mask_iou < 0.0).any().item())
            or bool((mask_iou > 1.0).any().item())):
        raise ValueError("dense mask quality predictions are invalid")
    mask_quality = (
        (mask_threshold_logits.sigmoid() * tier_weights).sum(dim=-1)
        + mask_iou
    ) / denominator
    return (
        box_quality + float(mask_utility_weight) * mask_quality
    ) / (1.0 + float(mask_utility_weight))


def dense_quality_prediction_uncertainty(
        box_threshold_logits, box_iou, mask_threshold_logits=None,
        mask_iou=None, mask_utility_weight=0.25):
    """Estimate query uncertainty from the dense Bernoulli quality heads.

    The quality heads already predict threshold probabilities and a bounded
    IoU estimate.  Their Bernoulli variance is a deterministic, calibrated
    confidence signal: confident candidates approach zero variance while the
    zero-initialized head gives every token the same uncertainty.  This keeps
    the V19 migration exactly identity-preserving without adding a separate
    dataset-specific confidence head.
    """
    if (not isinstance(box_threshold_logits, torch.Tensor)
            or box_threshold_logits.dim() < 1
            or box_threshold_logits.shape[-1] < 1
            or not isinstance(box_iou, torch.Tensor)
            or box_iou.shape != box_threshold_logits.shape[:-1]):
        raise ValueError("dense box uncertainty inputs are invalid")
    if (not isinstance(mask_utility_weight, (float, int))
            or isinstance(mask_utility_weight, bool)
            or not math.isfinite(float(mask_utility_weight))
            or float(mask_utility_weight) < 0.0):
        raise ValueError("mask utility weight must be finite and non-negative")
    if (not bool(torch.isfinite(box_threshold_logits).all().item())
            or not bool(torch.isfinite(box_iou).all().item())
            or bool((box_iou < 0.0).any().item())
            or bool((box_iou > 1.0).any().item())):
        raise ValueError("dense box uncertainty values are invalid")

    box_probability = box_threshold_logits.sigmoid()
    box_variance = box_probability * (1.0 - box_probability)
    box_iou_variance = box_iou * (1.0 - box_iou)
    box_uncertainty = torch.cat((
        box_variance,
        box_iou_variance.unsqueeze(-1),
    ), dim=-1).mean(dim=-1)
    if mask_threshold_logits is None and mask_iou is None:
        return box_uncertainty.clamp(min=0.0).sqrt()
    if (not isinstance(mask_threshold_logits, torch.Tensor)
            or mask_threshold_logits.shape != box_threshold_logits.shape
            or not isinstance(mask_iou, torch.Tensor)
            or mask_iou.shape != box_iou.shape
            or not bool(torch.isfinite(mask_threshold_logits).all().item())
            or not bool(torch.isfinite(mask_iou).all().item())
            or bool((mask_iou < 0.0).any().item())
            or bool((mask_iou > 1.0).any().item())):
        raise ValueError("dense mask uncertainty values are invalid")
    mask_probability = mask_threshold_logits.sigmoid()
    mask_variance = mask_probability * (1.0 - mask_probability)
    mask_iou_variance = mask_iou * (1.0 - mask_iou)
    mask_uncertainty = torch.cat((
        mask_variance,
        mask_iou_variance.unsqueeze(-1),
    ), dim=-1).mean(dim=-1)
    combined = (
        box_uncertainty + float(mask_utility_weight) * mask_uncertainty
    ) / (1.0 + float(mask_utility_weight))
    return combined.clamp(min=0.0).sqrt()


def _calibrated_utility_regression_loss(
        override_margin, decision_utility, active_mask,
        false_override_weight):
    """Regress the deployed margin while charging extra for overestimation."""
    if (override_margin.shape != decision_utility.shape
            or active_mask.shape != override_margin.shape
            or active_mask.dtype != torch.bool):
        raise ValueError("utility regression inputs must align as [B,Q]")
    if not bool(active_mask.any().item()):
        return override_margin.sum() * 0.0
    margin = override_margin.float()
    target = decision_utility.detach().float()
    regression = F.smooth_l1_loss(margin, target, reduction="none")
    overestimate_weight = torch.where(
        margin > target,
        regression.new_full((), float(false_override_weight)),
        regression.new_ones(()),
    )
    return (regression * overestimate_weight)[active_mask].mean()


def _risk_aware_dense_quality_action_loss(
        candidate_margin, deployment_utility, box_ious, mask_ious,
        fallback_indices,
        active_mask, sample_mask, mask_utility_weight,
        false_override_weight, temperature):
    """Train the deployed uncertainty-adjusted margin with dense targets.

    The fixed zero boundary is supervised by the cost-aware threshold utility,
    while continuous box/mask quality only trains candidate ordering.  A query
    can therefore rank above another candidate without being incorrectly
    promoted above fallback when it has no deployment benefit.
    """
    if (not isinstance(candidate_margin, torch.Tensor)
            or candidate_margin.dim() != 2
            or not isinstance(deployment_utility, torch.Tensor)
            or deployment_utility.shape != candidate_margin.shape
            or box_ious.shape != candidate_margin.shape
            or active_mask.shape != candidate_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "risk-aware quality inputs must align as [B,Q] and [B]"
        )
    if (not isinstance(fallback_indices, torch.Tensor)
            or fallback_indices.dtype != torch.long
            or fallback_indices.shape != (candidate_margin.shape[0],)
            or fallback_indices.device != candidate_margin.device):
        raise ValueError("risk-aware fallback indices must be int64 [B]")
    if (bool((fallback_indices < 0).any().item())
            or bool((fallback_indices >= candidate_margin.shape[1]).any().item())):
        raise ValueError("risk-aware fallback index is out of range")
    if (not bool(torch.isfinite(deployment_utility).all().item())
            or not bool(torch.isfinite(box_ious).all().item())):
        raise ValueError("risk-aware quality targets must be finite")
    for name, value, lower_bound in (
            ("mask_utility_weight", mask_utility_weight, 0.0),
            ("false_override_weight", false_override_weight, 1.0)):
        if (not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < lower_bound):
            raise ValueError("risk-aware {} is invalid".format(name))
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("risk-aware temperature is invalid")
    active = active_mask & sample_mask.unsqueeze(1)
    if not bool(active.any().item()):
        zero = candidate_margin.sum() * 0.0
        return {
            "loss": zero,
            "regression_loss": zero,
            "rank_loss": zero,
            "target_positive_ratio": zero.detach(),
            "false_positive_ratio": zero.detach(),
        }

    # threshold_aware_quality is bounded by 1.75; the tier-constrained mask
    # target is bounded by 5.75.  Normalizing both keeps the deployed margin
    # and its regression target in a shared [-1, 1] scale.
    target_quality = threshold_aware_quality(box_ious.detach()) / 1.75
    if mask_ious is not None and float(mask_utility_weight) > 0.0:
        if mask_ious.shape != box_ious.shape:
            raise ValueError("risk-aware mask IoUs must align with boxes")
        target_mask_quality = box_tier_constrained_mask_quality(
            box_ious.detach(), mask_ious.detach()
        ) / 5.75
        target_quality = (
            target_quality
            + float(mask_utility_weight) * target_mask_quality
        ) / (1.0 + float(mask_utility_weight))
    row_index = torch.arange(
        candidate_margin.shape[0], device=candidate_margin.device
    )
    if not bool(torch.allclose(
            deployment_utility[row_index, fallback_indices],
            deployment_utility.new_zeros(deployment_utility.shape[0]),
            rtol=0.0, atol=1e-6)):
        raise ValueError("risk-aware fallback utility must be zero")
    target_delta = (
        deployment_utility.detach().float()
        / (1.0 + float(mask_utility_weight))
    ).clamp(min=-1.0, max=1.0)
    margin = candidate_margin.float()
    regression = F.smooth_l1_loss(
        margin, target_delta.float(), reduction="none"
    )
    false_override = (target_delta <= 0.0) & (margin > target_delta)
    regression_weight = torch.where(
        false_override,
        regression.new_full((), float(false_override_weight)),
        regression.new_ones(()),
    )
    regression_loss = (
        regression[active] * regression_weight[active]
    ).sum() / regression_weight[active].sum().clamp(min=1e-6)
    rank_loss = listwise_quality_loss(
        scores=candidate_margin,
        quality=target_quality,
        valid_mask=active,
        sample_mask=sample_mask,
        temperature=float(temperature),
    )
    denominator = active.float().sum().clamp(min=1.0)
    target_positive = active & (target_delta > 0.0)
    predicted_positive = active & (candidate_margin > 0.0)
    false_positive = predicted_positive & ~target_positive
    return {
        "loss": regression_loss + rank_loss,
        "regression_loss": regression_loss,
        "rank_loss": rank_loss,
        "target_positive_ratio": (
            target_positive.float().sum() / denominator
        ).detach(),
        "false_positive_ratio": (
            false_positive.float().sum() / denominator
        ).detach(),
    }


def _weighted_calibrated_row_regression_loss(
        override_margin, decision_utility, active_mask, row_weight,
        false_override_weight):
    """Regress row opportunity with the same balanced risk weights as BCE."""
    if (override_margin.dim() != 1
            or decision_utility.shape != override_margin.shape
            or active_mask.shape != override_margin.shape
            or active_mask.dtype != torch.bool
            or row_weight.shape != override_margin.shape):
        raise ValueError(
            "weighted row regression inputs must align as [B]"
        )
    if not bool(active_mask.any().item()):
        return override_margin.sum() * 0.0
    margin = override_margin.float()
    target = decision_utility.detach().float()
    regression = F.smooth_l1_loss(margin, target, reduction="none")
    overestimate_weight = torch.where(
        margin > target,
        regression.new_full((), float(false_override_weight)),
        regression.new_ones(()),
    )
    weights = row_weight.float() * overestimate_weight
    return (
        regression[active_mask] * weights[active_mask]
    ).sum() / weights[active_mask].sum().clamp(min=1e-6)


def _threshold_transition_utility(targets, threshold_weights, break_cost):
    """Convert break/neutral/fix labels into one weighted metric utility."""
    weights = torch.as_tensor(
        threshold_weights, dtype=torch.float32, device=targets.device
    )
    weights = weights / weights.sum()
    utility = torch.zeros_like(targets, dtype=torch.float32)
    utility = torch.where(
        targets == 2, torch.ones_like(utility), utility
    )
    utility = torch.where(
        targets == 0,
        torch.full_like(utility, -float(break_cost)),
        utility,
    )
    return (utility * weights.view(1, 1, -1)).sum(dim=-1)


def transition_logits_expected_utility(logits, threshold_weights,
                                       break_cost):
    """Convert break/neutral/fix logits into expected metric utility."""
    if (not isinstance(logits, torch.Tensor) or logits.dim() != 4
            or logits.shape[-1] != 3):
        raise ValueError("transition logits must have shape [B,Q,T,3]")
    if (not isinstance(threshold_weights, torch.Tensor)
            or threshold_weights.dim() != 1
            or threshold_weights.shape[0] != logits.shape[2]
            or threshold_weights.device != logits.device
            or not bool(torch.isfinite(threshold_weights).all().item())
            or not bool((threshold_weights > 0).all().item())):
        raise ValueError(
            "threshold weights must be positive and align with logits"
        )
    if (not isinstance(break_cost, (float, int))
            or isinstance(break_cost, bool)
            or not math.isfinite(float(break_cost))
            or float(break_cost) < 1.0):
        raise ValueError("break_cost must be finite and at least one")

    probabilities = F.softmax(logits.float(), dim=-1)
    transition_values = probabilities.new_tensor((
        -float(break_cost), 0.0, 1.0,
    ))
    per_threshold = (probabilities * transition_values).sum(dim=-1)
    weights = threshold_weights.float()
    weights = weights / weights.sum()
    return (per_threshold * weights.view(1, 1, -1)).sum(dim=-1)


def build_setwise_action_target(decision_utility, active_mask,
                                temperature):
    """Build a fallback-plus-candidates target without arbitrary tie labels."""
    if (not isinstance(decision_utility, torch.Tensor)
            or decision_utility.dim() != 2
            or active_mask.shape != decision_utility.shape
            or active_mask.dtype != torch.bool
            or active_mask.device != decision_utility.device):
        raise ValueError(
            "setwise action utility and active mask must align [B,Q]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "setwise action temperature must be finite and positive"
        )

    positive = active_mask & (decision_utility > 0.0)
    candidate_logits = (
        decision_utility.detach().float() / float(temperature)
    ).masked_fill(~positive, -1e4)
    fallback_logits = torch.zeros(
        decision_utility.shape[0], 1,
        dtype=candidate_logits.dtype,
        device=candidate_logits.device,
    )
    return F.softmax(torch.cat((fallback_logits, candidate_logits), dim=1),
                     dim=1)


def build_risk_separated_action_target(decision_utility, active_mask,
                                       temperature):
    """Give positive rows no fallback mass and negative rows exact fallback."""
    if (not isinstance(decision_utility, torch.Tensor)
            or decision_utility.dim() != 2
            or active_mask.shape != decision_utility.shape
            or active_mask.dtype != torch.bool
            or active_mask.device != decision_utility.device):
        raise ValueError(
            "risk-separated utility and active mask must align [B,Q]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError(
            "risk-separated temperature must be finite and positive"
        )
    positive = active_mask & (decision_utility > 0.0)
    has_positive = positive.any(dim=1)
    candidate_logits = (
        decision_utility.detach().float() / float(temperature)
    ).masked_fill(~positive, -1e4)
    candidate_distribution = F.softmax(candidate_logits, dim=1)
    target = torch.zeros(
        decision_utility.shape[0], decision_utility.shape[1] + 1,
        dtype=candidate_distribution.dtype,
        device=candidate_distribution.device,
    )
    target[:, 0] = (~has_positive).to(dtype=target.dtype)
    target[:, 1:] = (
        candidate_distribution
        * has_positive.unsqueeze(1).to(candidate_distribution.dtype)
    )
    return target


def _positive_candidate_mass_loss(
        selection_margin, decision_utility, active_mask, sample_mask,
        temperature):
    """Train selection probability mass to cover any useful candidate."""
    if (selection_margin.dim() != 2
            or decision_utility.shape != selection_margin.shape
            or active_mask.shape != selection_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (selection_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "positive candidate mass inputs must align as [B,Q] and [B]"
        )
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("positive candidate mass temperature must be positive")
    positive = active_mask & (decision_utility > 0.0)
    positive_rows = sample_mask & positive.any(dim=1)
    masked_scores = (selection_margin / float(temperature)).masked_fill(
        ~active_mask, -1e4
    )
    probabilities = F.softmax(masked_scores, dim=1)
    positive_mass = (probabilities * positive.to(probabilities.dtype)).sum(dim=1)
    zero = selection_margin.sum() * 0.0
    loss = (
        -torch.log(positive_mass[positive_rows].clamp(min=1e-6)).mean()
        if bool(positive_rows.any().item()) else zero
    )
    return loss, positive_rows, positive_mass.detach()


def _positive_candidate_top1_margin_loss(
        selection_margin, decision_utility, active_mask, sample_mask,
        margin):
    """Make the deployed hard top-1 candidate positive when one exists."""
    if (selection_margin.dim() != 2
            or decision_utility.shape != selection_margin.shape
            or active_mask.shape != selection_margin.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (selection_margin.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError(
            "positive candidate top1 inputs must align as [B,Q] and [B]"
        )
    if (not isinstance(margin, (float, int))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
            or float(margin) < 0.0):
        raise ValueError("positive candidate top1 margin must be non-negative")
    positive = active_mask & (decision_utility > 0.0)
    negative = active_mask & ~positive
    rows = sample_mask & positive.any(dim=1) & negative.any(dim=1)
    masked_positive = selection_margin.masked_fill(~positive, -1e4)
    masked_negative = selection_margin.masked_fill(~negative, -1e4)
    best_positive = masked_positive.max(dim=1).values
    best_negative = masked_negative.max(dim=1).values
    violation = best_negative - best_positive + float(margin)
    zero = selection_margin.sum() * 0.0
    loss = (
        F.relu(violation[rows]).mean()
        if bool(rows.any().item()) else zero
    )
    return loss, rows, violation.detach()


def _counterfactual_selected_risk_loss(
        candidate_risk, selection_margin, decision_utility, active_mask,
        sample_mask, false_positive_weight, temperature):
    """Supervise candidate-conditioned risk with row-balanced counterfactuals."""
    if (candidate_risk.dim() != 2
            or selection_margin.shape != candidate_risk.shape
            or decision_utility.shape != candidate_risk.shape
            or active_mask.shape != candidate_risk.shape
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (candidate_risk.shape[0],)
            or sample_mask.dtype != torch.bool):
        raise ValueError("counterfactual risk inputs must align as [B,Q] and [B]")
    if (not isinstance(false_positive_weight, (float, int))
            or isinstance(false_positive_weight, bool)
            or not math.isfinite(float(false_positive_weight))
            or float(false_positive_weight) < 1.0):
        raise ValueError("counterfactual false-positive weight must be >= 1")
    if (not isinstance(temperature, (float, int))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0):
        raise ValueError("counterfactual risk temperature must be positive")
    positive = active_mask & (decision_utility > 0.0)
    negative = active_mask & ~positive
    positive_rows = sample_mask & positive.any(dim=1)
    active_rows = sample_mask & active_mask.any(dim=1)
    negative_rows = active_rows & ~positive_rows
    row_index = torch.arange(candidate_risk.shape[0], device=candidate_risk.device)
    positive_count = positive.sum(dim=1).clamp(min=1).to(candidate_risk.dtype)
    has_negative_candidate = negative.any(dim=1)
    negative_indices = selection_margin.masked_fill(~negative, -1e4).max(dim=1).indices
    negative_values = candidate_risk[row_index, negative_indices]
    negative_examples = negative_rows | (positive_rows & has_negative_candidate)
    positive_prior = (
        (positive_rows.float().sum() + 1.0)
        / (active_rows.float().sum() + 2.0)
    ).detach()
    prior_shift = torch.log(
        (1.0 - positive_prior).clamp(min=1e-6)
        / positive_prior.clamp(min=1e-6)
    ).detach()
    positive_logits = candidate_risk + prior_shift
    negative_logits = negative_values[negative_examples] + prior_shift
    zero = candidate_risk.sum() * 0.0
    if bool(positive_rows.any().item()):
        positive_candidate_loss = F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
            reduction="none",
        )
        positive_row_loss = (
            positive_candidate_loss * positive.to(positive_candidate_loss.dtype)
        ).sum(dim=1) / positive_count
        positive_loss = positive_row_loss[positive_rows].mean()
    else:
        positive_loss = zero
    negative_loss = (
        F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        ) * float(false_positive_weight)
        if bool(negative_examples.any().item()) else zero
    )
    prior_terms = []
    if bool(positive_rows.any().item()):
        prior_terms.append(positive_loss)
    if bool(negative_examples.any().item()):
        prior_terms.append(negative_loss)
    prior_classification_loss = (
        sum(prior_terms) / float(len(prior_terms))
        if prior_terms else zero
    )

    # The deployment gate compares raw selected risk against zero. Symmetric
    # margins around that boundary give useful and unsafe candidates equal
    # pressure away from an ambiguous zero without shifting the threshold.
    # Prior-shifted losses remain diagnostics only.
    if bool(positive_rows.any().item()):
        deployment_positive_candidate_loss = (
            F.binary_cross_entropy_with_logits(
                candidate_risk - float(temperature),
                torch.ones_like(candidate_risk),
                reduction="none",
            )
        )
        deployment_positive_row_loss = (
            deployment_positive_candidate_loss
            * positive.to(deployment_positive_candidate_loss.dtype)
        ).sum(dim=1) / positive_count
        deployment_positive_loss = (
            deployment_positive_row_loss[positive_rows].mean()
        )
    else:
        deployment_positive_loss = zero
    deployment_negative_loss = (
        F.binary_cross_entropy_with_logits(
            negative_values[negative_examples] + float(temperature),
            torch.zeros_like(negative_values[negative_examples]),
        )
        if bool(negative_examples.any().item()) else zero
    )
    deployment_terms = []
    if bool(positive_rows.any().item()):
        deployment_terms.append(deployment_positive_loss)
    if bool(negative_examples.any().item()):
        deployment_terms.append(deployment_negative_loss)
    deployment_boundary_loss = (
        sum(deployment_terms) / float(len(deployment_terms))
        if deployment_terms else zero
    )
    classification_loss = deployment_boundary_loss
    regression_terms = []
    if bool(positive_rows.any().item()):
        positive_targets = decision_utility.detach().float()
        positive_regression = F.smooth_l1_loss(
            candidate_risk.float(), positive_targets, reduction="none"
        )
        positive_regression_by_row = (
            positive_regression * positive.to(positive_regression.dtype)
        ).sum(dim=1) / positive_count
        regression_terms.append(
            positive_regression_by_row[positive_rows].mean()
        )
    if bool(negative_examples.any().item()):
        negative_regression_values = negative_values[negative_examples]
        negative_regression_targets = decision_utility[
            row_index, negative_indices
        ][negative_examples].detach().float()
        negative_regression = F.smooth_l1_loss(
            negative_regression_values.float(),
            negative_regression_targets,
            reduction="none",
        )
        regression_terms.append(negative_regression.mean())
    if regression_terms:
        regression_loss = sum(regression_terms) / float(len(regression_terms))
    else:
        regression_loss = zero
    return {
        "loss": classification_loss + regression_loss,
        "classification_loss": classification_loss,
        "regression_loss": regression_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "deployment_boundary_loss": deployment_boundary_loss,
        "prior_shift": prior_shift,
        "positive_prior": positive_prior,
        "positive_rows": positive_rows,
        "negative_examples": negative_examples,
    }


def _counterfactual_benefit_hazard_loss(
        candidate_benefit, candidate_hazard, selection_margin,
        decision_utility, active_mask, sample_mask, focal_gamma):
    """Calibrate candidate gain and break risk as separate evidence paths."""
    expected = decision_utility.shape
    for name, value in (
            ("candidate_benefit", candidate_benefit),
            ("candidate_hazard", candidate_hazard),
            ("selection_margin", selection_margin)):
        if (not isinstance(value, torch.Tensor)
                or value.shape != expected
                or value.device != decision_utility.device
                or not bool(torch.isfinite(value).all().item())):
            raise ValueError(
                "{} must be finite and align with utility".format(name)
            )
    if (decision_utility.dim() != 2
            or active_mask.shape != expected
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (expected[0],)
            or sample_mask.dtype != torch.bool
            or active_mask.device != decision_utility.device
            or sample_mask.device != decision_utility.device):
        raise ValueError(
            "decomposed risk masks must align as [B,Q] and [B]"
        )
    if (not isinstance(focal_gamma, (float, int))
            or isinstance(focal_gamma, bool)
            or not math.isfinite(float(focal_gamma))
            or float(focal_gamma) < 0.0):
        raise ValueError("decomposed risk focal_gamma must be non-negative")

    positive = active_mask & (decision_utility > 0.0)
    negative = active_mask & ~positive
    positive_rows = sample_mask & positive.any(dim=1)
    active_rows = sample_mask & active_mask.any(dim=1)
    negative_rows = active_rows & ~positive_rows
    row_index = torch.arange(expected[0], device=decision_utility.device)
    positive_count = positive.sum(dim=1).clamp(min=1).to(
        dtype=candidate_benefit.dtype
    )
    negative_indices = selection_margin.masked_fill(
        ~negative, -1e4
    ).max(dim=1).indices
    has_negative_candidate = negative.any(dim=1)
    negative_examples = negative_rows | (
        positive_rows & has_negative_candidate
    )

    candidate_risk = (
        F.relu(candidate_benefit) - F.relu(candidate_hazard)
    )
    negative_risk = candidate_risk[row_index, negative_indices]
    gamma = float(focal_gamma)
    zero = candidate_risk.sum() * 0.0

    if bool(positive_rows.any().item()):
        positive_boundary = F.softplus(-candidate_risk)
        positive_modulator = torch.sigmoid(-candidate_risk).pow(gamma)
        positive_candidate_loss = positive_boundary * positive_modulator
        positive_row_loss = (
            positive_candidate_loss
            * positive.to(dtype=positive_candidate_loss.dtype)
        ).sum(dim=1) / positive_count
        positive_loss = positive_row_loss[positive_rows].mean()
    else:
        positive_loss = zero
    if bool(negative_examples.any().item()):
        selected_negative_risk = negative_risk[negative_examples]
        negative_loss = (
            F.softplus(selected_negative_risk)
            * torch.sigmoid(selected_negative_risk).pow(gamma)
        ).mean()
    else:
        negative_loss = zero
    classification_terms = []
    if bool(positive_rows.any().item()):
        classification_terms.append(positive_loss)
    if bool(negative_examples.any().item()):
        classification_terms.append(negative_loss)
    classification_loss = (
        sum(classification_terms) / float(len(classification_terms))
        if classification_terms else zero
    )

    regression_terms = []
    benefit_regression_terms = []
    hazard_regression_terms = []
    if bool(positive_rows.any().item()):
        positive_benefit_target = decision_utility.detach().float().clamp(
            min=0.0
        )
        positive_hazard_target = torch.zeros_like(positive_benefit_target)
        benefit_error = F.smooth_l1_loss(
            candidate_benefit.float(), positive_benefit_target,
            reduction="none",
        )
        hazard_error = F.smooth_l1_loss(
            candidate_hazard.float(), positive_hazard_target,
            reduction="none",
        )
        positive_benefit_regression = (
            benefit_error * positive.to(dtype=benefit_error.dtype)
        ).sum(dim=1) / positive_count
        positive_hazard_regression = (
            hazard_error * positive.to(dtype=hazard_error.dtype)
        ).sum(dim=1) / positive_count
        benefit_regression_terms.append(
            positive_benefit_regression[positive_rows].mean()
        )
        hazard_regression_terms.append(
            positive_hazard_regression[positive_rows].mean()
        )
        regression_terms.append(
            0.5 * (
                positive_benefit_regression[positive_rows].mean()
                + positive_hazard_regression[positive_rows].mean()
            )
        )
    if bool(negative_examples.any().item()):
        selected_benefit = candidate_benefit[
            row_index, negative_indices
        ][negative_examples]
        selected_hazard = candidate_hazard[
            row_index, negative_indices
        ][negative_examples]
        selected_utility = decision_utility[
            row_index, negative_indices
        ][negative_examples].detach().float()
        negative_benefit_target = torch.zeros_like(selected_utility)
        negative_hazard_target = (-selected_utility).clamp(min=0.0)
        negative_benefit_regression = F.smooth_l1_loss(
            selected_benefit.float(), negative_benefit_target
        )
        negative_hazard_regression = F.smooth_l1_loss(
            selected_hazard.float(), negative_hazard_target
        )
        benefit_regression_terms.append(negative_benefit_regression)
        hazard_regression_terms.append(negative_hazard_regression)
        regression_terms.append(
            0.5 * (
                negative_benefit_regression + negative_hazard_regression
            )
        )
    regression_loss = (
        sum(regression_terms) / float(len(regression_terms))
        if regression_terms else zero
    )
    benefit_regression_loss = (
        sum(benefit_regression_terms) / float(len(benefit_regression_terms))
        if benefit_regression_terms else zero
    )
    hazard_regression_loss = (
        sum(hazard_regression_terms) / float(len(hazard_regression_terms))
        if hazard_regression_terms else zero
    )
    return {
        "loss": classification_loss + regression_loss,
        "classification_loss": classification_loss,
        "regression_loss": regression_loss,
        "benefit_regression_loss": benefit_regression_loss,
        "hazard_regression_loss": hazard_regression_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "positive_rows": positive_rows,
        "negative_examples": negative_examples,
    }


def _counterfactual_complementary_logodds_loss(
        candidate_benefit, candidate_hazard, selection_margin,
        decision_utility, active_mask, sample_mask, focal_gamma):
    """Learn complementary gain/hazard log-odds at one zero boundary."""
    expected = decision_utility.shape
    for name, value in (
            ("candidate_benefit", candidate_benefit),
            ("candidate_hazard", candidate_hazard),
            ("selection_margin", selection_margin)):
        if (not isinstance(value, torch.Tensor)
                or value.shape != expected
                or value.device != decision_utility.device
                or not bool(torch.isfinite(value).all().item())):
            raise ValueError(
                "{} must be finite and align with utility".format(name)
            )
    if (decision_utility.dim() != 2
            or active_mask.shape != expected
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (expected[0],)
            or sample_mask.dtype != torch.bool
            or active_mask.device != decision_utility.device
            or sample_mask.device != decision_utility.device):
        raise ValueError(
            "complementary log-odds masks must align as [B,Q] and [B]"
        )
    if (not isinstance(focal_gamma, (float, int))
            or isinstance(focal_gamma, bool)
            or not math.isfinite(float(focal_gamma))
            or float(focal_gamma) < 0.0):
        raise ValueError(
            "complementary log-odds focal_gamma must be non-negative"
        )

    positive = active_mask & (decision_utility > 0.0)
    negative = active_mask & ~positive
    positive_rows = sample_mask & positive.any(dim=1)
    active_rows = sample_mask & active_mask.any(dim=1)
    negative_rows = active_rows & ~positive_rows
    row_index = torch.arange(expected[0], device=decision_utility.device)
    positive_count = positive.sum(dim=1).clamp(min=1).to(
        dtype=candidate_benefit.dtype
    )
    negative_indices = selection_margin.masked_fill(
        ~negative, -1e4
    ).max(dim=1).indices
    has_negative_candidate = negative.any(dim=1)
    negative_examples = negative_rows | (
        positive_rows & has_negative_candidate
    )
    selected_benefit = candidate_benefit[
        row_index, negative_indices
    ]
    selected_hazard = candidate_hazard[
        row_index, negative_indices
    ]
    gamma = float(focal_gamma)
    zero = (candidate_benefit + candidate_hazard).sum() * 0.0

    if bool(positive_rows.any().item()):
        positive_benefit_candidate = (
            F.softplus(-candidate_benefit)
            * torch.sigmoid(-candidate_benefit).pow(gamma)
        )
        positive_hazard_candidate = (
            F.softplus(candidate_hazard)
            * torch.sigmoid(candidate_hazard).pow(gamma)
        )
        positive_benefit_row = (
            positive_benefit_candidate
            * positive.to(dtype=positive_benefit_candidate.dtype)
        ).sum(dim=1) / positive_count
        positive_hazard_row = (
            positive_hazard_candidate
            * positive.to(dtype=positive_hazard_candidate.dtype)
        ).sum(dim=1) / positive_count
        positive_benefit_loss = positive_benefit_row[positive_rows].mean()
        positive_hazard_loss = positive_hazard_row[positive_rows].mean()
        positive_loss = 0.5 * (
            positive_benefit_loss + positive_hazard_loss
        )
    else:
        positive_benefit_loss = zero
        positive_hazard_loss = zero
        positive_loss = zero
    if bool(negative_examples.any().item()):
        negative_benefit_values = selected_benefit[negative_examples]
        negative_hazard_values = selected_hazard[negative_examples]
        negative_benefit_loss = (
            F.softplus(negative_benefit_values)
            * torch.sigmoid(negative_benefit_values).pow(gamma)
        ).mean()
        negative_hazard_loss = (
            F.softplus(-negative_hazard_values)
            * torch.sigmoid(-negative_hazard_values).pow(gamma)
        ).mean()
        negative_loss = 0.5 * (
            negative_benefit_loss + negative_hazard_loss
        )
    else:
        negative_benefit_loss = zero
        negative_hazard_loss = zero
        negative_loss = zero
    classification_terms = []
    benefit_terms = []
    hazard_terms = []
    if bool(positive_rows.any().item()):
        classification_terms.append(positive_loss)
        benefit_terms.append(positive_benefit_loss)
        hazard_terms.append(positive_hazard_loss)
    if bool(negative_examples.any().item()):
        classification_terms.append(negative_loss)
        benefit_terms.append(negative_benefit_loss)
        hazard_terms.append(negative_hazard_loss)
    classification_loss = (
        sum(classification_terms) / float(len(classification_terms))
        if classification_terms else zero
    )
    benefit_classification_loss = (
        sum(benefit_terms) / float(len(benefit_terms))
        if benefit_terms else zero
    )
    hazard_classification_loss = (
        sum(hazard_terms) / float(len(hazard_terms))
        if hazard_terms else zero
    )

    candidate_risk = candidate_benefit - candidate_hazard
    regression_terms = []
    if bool(positive_rows.any().item()):
        positive_regression = F.smooth_l1_loss(
            candidate_risk.float(),
            torch.ones_like(candidate_risk, dtype=torch.float32),
            reduction="none",
        )
        positive_regression_row = (
            positive_regression
            * positive.to(dtype=positive_regression.dtype)
        ).sum(dim=1) / positive_count
        regression_terms.append(
            positive_regression_row[positive_rows].mean()
        )
    if bool(negative_examples.any().item()):
        selected_risk = candidate_risk[
            row_index, negative_indices
        ][negative_examples]
        regression_terms.append(F.smooth_l1_loss(
            selected_risk.float(), -torch.ones_like(selected_risk).float()
        ))
    regression_loss = (
        sum(regression_terms) / float(len(regression_terms))
        if regression_terms else zero
    )
    return {
        "loss": classification_loss + regression_loss,
        "classification_loss": classification_loss,
        "regression_loss": regression_loss,
        "benefit_classification_loss": benefit_classification_loss,
        "hazard_classification_loss": hazard_classification_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "positive_rows": positive_rows,
        "negative_examples": negative_examples,
    }


def _counterfactual_hazard_residual_loss(
        candidate_gain, candidate_hazard, selection_margin,
        decision_utility, active_mask, sample_mask, focal_gamma):
    """Learn an unconstrained gain with a non-negative hazard veto."""
    expected = decision_utility.shape
    for name, value in (
            ("candidate_gain", candidate_gain),
            ("candidate_hazard", candidate_hazard),
            ("selection_margin", selection_margin)):
        if (not isinstance(value, torch.Tensor)
                or value.shape != expected
                or value.device != decision_utility.device
                or not bool(torch.isfinite(value).all().item())):
            raise ValueError(
                "{} must be finite and align with utility".format(name)
            )
    if (decision_utility.dim() != 2
            or active_mask.shape != expected
            or active_mask.dtype != torch.bool
            or sample_mask.shape != (expected[0],)
            or sample_mask.dtype != torch.bool
            or active_mask.device != decision_utility.device
            or sample_mask.device != decision_utility.device):
        raise ValueError("hazard residual masks must align as [B,Q] and [B]")
    if (not isinstance(focal_gamma, (float, int))
            or isinstance(focal_gamma, bool)
            or not math.isfinite(float(focal_gamma))
            or float(focal_gamma) < 0.0):
        raise ValueError("hazard residual focal_gamma must be non-negative")

    positive = active_mask & (decision_utility > 0.0)
    negative = active_mask & ~positive
    positive_rows = sample_mask & positive.any(dim=1)
    active_rows = sample_mask & active_mask.any(dim=1)
    negative_rows = active_rows & ~positive_rows
    row_index = torch.arange(expected[0], device=decision_utility.device)
    positive_count = positive.sum(dim=1).clamp(min=1).to(
        dtype=candidate_gain.dtype
    )
    negative_indices = selection_margin.masked_fill(
        ~negative, -1e4
    ).max(dim=1).indices
    has_negative_candidate = negative.any(dim=1)
    negative_examples = negative_rows | (
        positive_rows & has_negative_candidate
    )
    selected_gain = candidate_gain[row_index, negative_indices]
    selected_hazard = candidate_hazard[row_index, negative_indices]
    zero = (candidate_gain + candidate_hazard).sum() * 0.0

    gain_classification_terms = []
    gain_regression_terms = []
    hazard_classification_terms = []
    gamma = float(focal_gamma)
    if bool(positive_rows.any().item()):
        positive_gain_classification = F.softplus(-candidate_gain)
        positive_gain_row = (
            positive_gain_classification
            * positive.to(dtype=positive_gain_classification.dtype)
        ).sum(dim=1) / positive_count
        gain_classification_terms.append(
            positive_gain_row[positive_rows].mean()
        )
        positive_gain_regression = F.smooth_l1_loss(
            candidate_gain.float(), decision_utility.detach().float(),
            reduction="none",
        )
        positive_gain_regression_row = (
            positive_gain_regression
            * positive.to(dtype=positive_gain_regression.dtype)
        ).sum(dim=1) / positive_count
        gain_regression_terms.append(
            positive_gain_regression_row[positive_rows].mean()
        )
        positive_hazard_classification = (
            F.softplus(candidate_hazard)
            * torch.sigmoid(candidate_hazard).pow(gamma)
        )
        positive_hazard_row = (
            positive_hazard_classification
            * positive.to(dtype=positive_hazard_classification.dtype)
        ).sum(dim=1) / positive_count
        hazard_classification_terms.append(
            positive_hazard_row[positive_rows].mean()
        )
    if bool(negative_examples.any().item()):
        negative_gain_values = selected_gain[negative_examples]
        gain_classification_terms.append(
            F.softplus(negative_gain_values).mean()
        )
        selected_utility = decision_utility[
            row_index, negative_indices
        ][negative_examples].detach().float()
        gain_regression_terms.append(F.smooth_l1_loss(
            negative_gain_values.float(), selected_utility
        ))
        negative_hazard_values = selected_hazard[negative_examples]
        hazard_classification_terms.append((
            F.softplus(-negative_hazard_values)
            * torch.sigmoid(-negative_hazard_values).pow(gamma)
        ).mean())
    gain_classification_loss = (
        sum(gain_classification_terms)
        / float(len(gain_classification_terms))
        if gain_classification_terms else zero
    )
    gain_regression_loss = (
        sum(gain_regression_terms) / float(len(gain_regression_terms))
        if gain_regression_terms else zero
    )
    hazard_classification_loss = (
        sum(hazard_classification_terms)
        / float(len(hazard_classification_terms))
        if hazard_classification_terms else zero
    )
    classification_loss = 0.5 * (
        gain_classification_loss + hazard_classification_loss
    )
    return {
        "loss": classification_loss + gain_regression_loss,
        "classification_loss": classification_loss,
        "regression_loss": gain_regression_loss,
        "benefit_classification_loss": gain_classification_loss,
        "hazard_classification_loss": hazard_classification_loss,
        "positive_rows": positive_rows,
        "negative_examples": negative_examples,
    }


def compute_source_moe_fallback_gate_loss(
        box_logits, decision_logits, box_ious, default_indices, candidate_mask,
        sample_mask=None, mask_logits=None, mask_ious=None,
        thresholds=(0.25, 0.50), threshold_weights=(2.0, 1.0),
        mask_loss_weight=0.25, focal_gamma=2.0,
        false_override_weight=2.0, break_cost=2.0,
        mask_utility_weight=0.25, objective="balanced_focal",
        setwise_temperature=0.0, boundary_loss_weight=0.0,
        action_margin=None,
        row_switch_margin=None, row_benefit_margin=None,
        row_safety_margin=None, joint_action_margin=None,
        pairwise_utility_margin=None,
        candidate_selection_margin=None,
        selected_abstention_margin=None,
        counterfactual_risk_margin=None,
        counterfactual_benefit_margin=None,
        counterfactual_hazard_margin=None,
        absolute_box_logits=None,
        absolute_box_iou=None, absolute_mask_logits=None,
        absolute_mask_iou=None):
    """Supervise candidate quality and the final fallback/override action."""
    if (not isinstance(box_logits, torch.Tensor)
            or box_logits.dim() != 4 or box_logits.shape[-1] != 3):
        raise ValueError("box_logits must have shape [B,Q,T,3]")
    if (not isinstance(decision_logits, torch.Tensor)
            or decision_logits.shape != box_logits.shape[:2] + (3,)):
        raise ValueError("decision_logits must have shape [B,Q,3]")
    if box_ious.shape != box_logits.shape[:2]:
        raise ValueError("box_ious must align with gate logits")
    if (candidate_mask.dtype != torch.bool
            or candidate_mask.shape != box_logits.shape[:2]
            or candidate_mask.device != box_logits.device):
        raise ValueError("candidate_mask must be bool with shape [B,Q]")
    if len(thresholds) != box_logits.shape[2]:
        raise ValueError("gate threshold axis does not match thresholds")
    if (not isinstance(threshold_weights, (tuple, list))
            or len(threshold_weights) != len(thresholds)
            or any(not isinstance(value, (float, int))
                   or isinstance(value, bool)
                   or not math.isfinite(float(value))
                   or float(value) <= 0.0
                   for value in threshold_weights)):
        raise ValueError("threshold_weights must be finite and positive")
    if (not isinstance(mask_loss_weight, (float, int))
            or isinstance(mask_loss_weight, bool)
            or not math.isfinite(float(mask_loss_weight))
            or float(mask_loss_weight) < 0.0):
        raise ValueError("mask_loss_weight must be finite and non-negative")
    if (not isinstance(break_cost, (float, int))
            or isinstance(break_cost, bool)
            or not math.isfinite(float(break_cost))
            or float(break_cost) < 1.0):
        raise ValueError("break_cost must be finite and at least one")
    if (not isinstance(mask_utility_weight, (float, int))
            or isinstance(mask_utility_weight, bool)
            or not math.isfinite(float(mask_utility_weight))
            or float(mask_utility_weight) < 0.0):
        raise ValueError(
            "mask_utility_weight must be finite and non-negative"
        )
    if objective not in (
            "balanced_focal", "calibrated_utility",
            "balanced_calibrated_utility",
            "hierarchical_risk_calibrated",
            "pairwise_risk_calibrated", "topn_risk_calibrated",
            "topn_dual_risk_calibrated",
            "topn_absolute_quality_calibrated",
            "cascade_absolute_quality_calibrated",
            "cascade_opportunity_balanced_calibrated",
            "cascade_opportunity_verified_calibrated",
            "cascade_joint_risk_calibrated",
            "cascade_v19_fallback_set_risk_calibrated",
            "cascade_v19_rich_set_empirical_risk",
            "cascade_v23_dense_quality_risk",
            "cascade_v24_relative_risk",
            "cascade_v25_pairwise_calibrated_risk",
            "cascade_v26_prior_restored_pairwise_risk",
            "cascade_v27_uncertainty_quality_risk",
            "cascade_v28_selected_abstention_risk",
            "cascade_v29_counterfactual_selected_risk",
            "cascade_v37_counterfactual_benefit_hazard_risk",
            "cascade_v38_complementary_logodds_risk",
            "cascade_v39_hazard_residual_risk"):
        raise ValueError(
            "objective must be balanced_focal, calibrated_utility, or "
            "balanced_calibrated_utility, hierarchical_risk_calibrated, "
            "pairwise_risk_calibrated, topn_risk_calibrated, or "
            "topn_dual_risk_calibrated, topn_absolute_quality_calibrated, "
            "cascade_absolute_quality_calibrated, or "
            "cascade_opportunity_balanced_calibrated, or "
            "cascade_opportunity_verified_calibrated, or "
            "cascade_joint_risk_calibrated, or "
            "cascade_v19_fallback_set_risk_calibrated, or "
            "cascade_v19_rich_set_empirical_risk, or "
            "cascade_v23_dense_quality_risk, cascade_v24_relative_risk, or "
            "cascade_v25_pairwise_calibrated_risk, or "
            "cascade_v26_prior_restored_pairwise_risk, or "
            "cascade_v27_uncertainty_quality_risk, or "
            "cascade_v28_selected_abstention_risk, or "
            "cascade_v29_counterfactual_selected_risk, or "
            "cascade_v37_counterfactual_benefit_hazard_risk, or "
            "cascade_v38_complementary_logodds_risk, or "
            "cascade_v39_hazard_residual_risk"
        )
    if (not isinstance(setwise_temperature, (float, int))
            or isinstance(setwise_temperature, bool)
            or not math.isfinite(float(setwise_temperature))
            or float(setwise_temperature) < 0.0):
        raise ValueError(
            "setwise_temperature must be finite and non-negative"
        )
    if (not isinstance(boundary_loss_weight, (float, int))
            or isinstance(boundary_loss_weight, bool)
            or not math.isfinite(float(boundary_loss_weight))
            or float(boundary_loss_weight) < 0.0):
        raise ValueError(
            "boundary_loss_weight must be finite and non-negative"
        )
    if action_margin is not None and (
            not isinstance(action_margin, torch.Tensor)
            or action_margin.shape != box_ious.shape
            or action_margin.device != box_logits.device
            or not bool(torch.isfinite(action_margin).all().item())):
        raise ValueError("action_margin must be finite and align with boxes")
    if row_switch_margin is not None and (
            not isinstance(row_switch_margin, torch.Tensor)
            or row_switch_margin.shape not in (
                (box_logits.shape[0],), box_ious.shape
            )
            or row_switch_margin.device != box_logits.device
            or not bool(torch.isfinite(row_switch_margin).all().item())):
        raise ValueError(
            "row_switch_margin must be finite with shape [B] or [B,Q]"
        )
    if row_switch_margin is not None and action_margin is None:
        raise ValueError(
            "row_switch_margin requires a separate candidate action_margin"
        )
    for name, margin in (
            ("row_benefit_margin", row_benefit_margin),
            ("row_safety_margin", row_safety_margin),
            ("joint_action_margin", joint_action_margin),
            ("pairwise_utility_margin", pairwise_utility_margin),
            ("candidate_selection_margin", candidate_selection_margin),
            ("counterfactual_risk_margin", counterfactual_risk_margin),
            ("counterfactual_benefit_margin", counterfactual_benefit_margin),
            ("counterfactual_hazard_margin", counterfactual_hazard_margin)):
        if margin is not None and (
                not isinstance(margin, torch.Tensor)
                or margin.shape != box_ious.shape
                or margin.device != box_logits.device
                or not bool(torch.isfinite(margin).all().item())):
            raise ValueError(
                "{} must be finite with shape [B,Q]".format(name)
            )
    if selected_abstention_margin is not None and (
            not isinstance(selected_abstention_margin, torch.Tensor)
            or selected_abstention_margin.shape != (box_logits.shape[0],)
            or selected_abstention_margin.device != box_logits.device
            or not bool(torch.isfinite(
                selected_abstention_margin
            ).all().item())):
        raise ValueError(
            "selected_abstention_margin must be finite with shape [B]"
        )
    if sample_mask is None:
        sample_mask = torch.ones(
            box_logits.shape[0], dtype=torch.bool, device=box_logits.device
        )
    if (sample_mask.dtype != torch.bool
            or sample_mask.shape != (box_logits.shape[0],)
            or sample_mask.device != box_logits.device):
        raise ValueError("sample_mask must be bool with shape [B]")

    absolute_quality_objective = objective in (
        "topn_absolute_quality_calibrated",
        "cascade_absolute_quality_calibrated",
        "cascade_opportunity_balanced_calibrated",
        "cascade_opportunity_verified_calibrated",
        "cascade_joint_risk_calibrated",
        "cascade_v23_dense_quality_risk",
        "cascade_v24_relative_risk",
        "cascade_v25_pairwise_calibrated_risk",
        "cascade_v26_prior_restored_pairwise_risk",
        "cascade_v27_uncertainty_quality_risk",
        "cascade_v28_selected_abstention_risk",
        "cascade_v29_counterfactual_selected_risk",
        "cascade_v37_counterfactual_benefit_hazard_risk",
        "cascade_v38_complementary_logodds_risk",
        "cascade_v39_hazard_residual_risk",
    )
    cascade_quality_objective = objective in (
        "cascade_absolute_quality_calibrated",
        "cascade_opportunity_balanced_calibrated",
        "cascade_opportunity_verified_calibrated",
        "cascade_joint_risk_calibrated",
    )
    opportunity_cascade_objective = objective in (
        "cascade_opportunity_balanced_calibrated",
        "cascade_opportunity_verified_calibrated",
        "cascade_joint_risk_calibrated",
    )
    verified_opportunity_objective = objective in (
        "cascade_opportunity_verified_calibrated",
        "cascade_joint_risk_calibrated",
    )
    fallback_set_objective = (
        objective == "cascade_v19_fallback_set_risk_calibrated"
    )
    rich_set_objective = (
        objective == "cascade_v19_rich_set_empirical_risk"
    )
    dense_quality_objective = objective in (
        "cascade_v23_dense_quality_risk",
        "cascade_v24_relative_risk",
        "cascade_v25_pairwise_calibrated_risk",
        "cascade_v26_prior_restored_pairwise_risk",
        "cascade_v27_uncertainty_quality_risk",
        "cascade_v28_selected_abstention_risk",
        "cascade_v29_counterfactual_selected_risk",
        "cascade_v37_counterfactual_benefit_hazard_risk",
        "cascade_v38_complementary_logodds_risk",
        "cascade_v39_hazard_residual_risk",
    )
    relative_risk_objective = objective == "cascade_v24_relative_risk"
    pairwise_calibrated_objective = (
        objective == "cascade_v25_pairwise_calibrated_risk"
    )
    prior_restored_pairwise_objective = (
        objective == "cascade_v26_prior_restored_pairwise_risk"
    )
    quality_risk_objective = (
        objective == "cascade_v27_uncertainty_quality_risk"
    )
    selected_abstention_objective = (
        objective in (
            "cascade_v28_selected_abstention_risk",
            "cascade_v29_counterfactual_selected_risk",
            "cascade_v37_counterfactual_benefit_hazard_risk",
            "cascade_v38_complementary_logodds_risk",
            "cascade_v39_hazard_residual_risk",
        )
    )
    counterfactual_selected_objective = (
        objective in (
            "cascade_v29_counterfactual_selected_risk",
            "cascade_v37_counterfactual_benefit_hazard_risk",
            "cascade_v38_complementary_logodds_risk",
            "cascade_v39_hazard_residual_risk",
        )
    )
    decomposed_counterfactual_objective = objective in (
        "cascade_v37_counterfactual_benefit_hazard_risk",
        "cascade_v38_complementary_logodds_risk",
        "cascade_v39_hazard_residual_risk",
    )
    complementary_logodds_objective = (
        objective == "cascade_v38_complementary_logodds_risk"
    )
    hazard_residual_objective = (
        objective == "cascade_v39_hazard_residual_risk"
    )
    pairwise_utility_objective = (
        pairwise_calibrated_objective
        or prior_restored_pairwise_objective
        or selected_abstention_objective
    )
    v19_set_objective = (
        fallback_set_objective
        or rich_set_objective
        or dense_quality_objective
    )
    joint_action_objective = objective in (
        "cascade_joint_risk_calibrated",
        "cascade_v19_fallback_set_risk_calibrated",
        "cascade_v19_rich_set_empirical_risk",
        "cascade_v23_dense_quality_risk",
        "cascade_v24_relative_risk",
        "cascade_v25_pairwise_calibrated_risk",
        "cascade_v26_prior_restored_pairwise_risk",
        "cascade_v27_uncertainty_quality_risk",
        "cascade_v28_selected_abstention_risk",
        "cascade_v29_counterfactual_selected_risk",
        "cascade_v37_counterfactual_benefit_hazard_risk",
        "cascade_v38_complementary_logodds_risk",
        "cascade_v39_hazard_residual_risk",
    )
    if absolute_quality_objective and (
            action_margin is None or float(setwise_temperature) <= 0.0
            or absolute_box_logits is None or absolute_box_iou is None):
        raise ValueError(
            "absolute-quality objective requires absolute quality "
            "predictions, action_margin, and positive setwise_temperature"
        )

    active = candidate_mask & sample_mask.unsqueeze(1)
    row_index = torch.arange(box_logits.shape[0], device=box_logits.device)
    active = active.clone()
    active[row_index, default_indices] = False
    absolute_quality_active = active.clone()
    absolute_quality_active[row_index, default_indices] = sample_mask
    box_targets = threshold_transition_targets(
        box_ious.detach(), default_indices, thresholds=thresholds
    )
    total_threshold_weight = float(sum(threshold_weights))
    box_losses = []
    stats = {}
    for threshold_index, (threshold, threshold_weight) in enumerate(zip(
            thresholds, threshold_weights)):
        current_targets = box_targets[:, :, threshold_index]
        current_loss = _class_balanced_focal_loss(
            box_logits[:, :, threshold_index],
            current_targets,
            active,
            gamma=focal_gamma,
            false_override_weight=false_override_weight,
        )
        box_losses.append(float(threshold_weight) * current_loss)
        suffix = "025" if float(threshold) == 0.25 else (
            "050" if float(threshold) == 0.50
            else str(threshold).replace(".", "")
        )
        denominator = active.float().sum().clamp(min=1.0)
        stats["source_moe_gate_target_break{}_ratio".format(suffix)] = (
            ((current_targets == 0) & active).float().sum() / denominator
        ).detach()
        stats["source_moe_gate_target_fix{}_ratio".format(suffix)] = (
            ((current_targets == 2) & active).float().sum() / denominator
        ).detach()
    box_loss = sum(box_losses) / total_threshold_weight

    mask_loss = box_logits.sum() * 0.0
    mask_targets = None
    if mask_logits is not None or mask_ious is not None:
        if (not isinstance(mask_logits, torch.Tensor)
                or mask_logits.shape != box_logits.shape
                or mask_ious is None
                or mask_ious.shape != box_ious.shape):
            raise ValueError("mask gate supervision must align with box inputs")
        mask_targets = threshold_transition_targets(
            mask_ious.detach(), default_indices, thresholds=thresholds
        )
        mask_losses = []
        for threshold_index, threshold_weight in enumerate(threshold_weights):
            mask_losses.append(
                float(threshold_weight) * _class_balanced_focal_loss(
                    mask_logits[:, :, threshold_index],
                    mask_targets[:, :, threshold_index],
                    active,
                    gamma=focal_gamma,
                    false_override_weight=false_override_weight,
                )
            )
        mask_loss = sum(mask_losses) / total_threshold_weight

    absolute_quality_losses = {
        "loss": box_logits.sum() * 0.0,
        "box_threshold_loss": box_logits.sum() * 0.0,
        "box_iou_loss": box_logits.sum() * 0.0,
        "mask_threshold_loss": box_logits.sum() * 0.0,
        "mask_iou_loss": box_logits.sum() * 0.0,
    }
    if absolute_quality_objective:
        absolute_quality_losses = _absolute_quality_prediction_loss(
            box_threshold_logits=absolute_box_logits,
            box_iou_estimate=absolute_box_iou,
            box_ious=box_ious,
            active_mask=absolute_quality_active,
            thresholds=thresholds,
            mask_threshold_logits=absolute_mask_logits,
            mask_iou_estimate=absolute_mask_iou,
            mask_ious=mask_ious,
            mask_loss_weight=mask_loss_weight,
        )
    absolute_quality_loss = absolute_quality_losses["loss"]
    dense_box_rank_loss = box_logits.sum() * 0.0
    dense_mask_rank_loss = box_logits.sum() * 0.0
    dense_quality_rank_loss = box_logits.sum() * 0.0
    if dense_quality_objective:
        predicted_box_quality = dense_quality_expected_score(
            absolute_box_logits, absolute_box_iou,
            mask_utility_weight=0.0,
        )
        dense_box_rank_loss = listwise_quality_loss(
            scores=predicted_box_quality,
            quality=threshold_aware_quality(box_ious),
            valid_mask=absolute_quality_active,
            sample_mask=sample_mask,
            temperature=float(setwise_temperature),
        )
        if mask_ious is not None:
            predicted_mask_quality = dense_quality_expected_score(
                absolute_mask_logits, absolute_mask_iou,
                mask_utility_weight=0.0,
            )
            dense_mask_rank_loss = listwise_quality_loss(
                scores=predicted_mask_quality,
                quality=box_tier_constrained_mask_quality(
                    box_ious, mask_ious
                ),
                valid_mask=absolute_quality_active,
                sample_mask=sample_mask,
                temperature=float(setwise_temperature),
            )
            dense_quality_rank_loss = (
                dense_box_rank_loss
                + float(mask_loss_weight) * dense_mask_rank_loss
            ) / (1.0 + float(mask_loss_weight))
        else:
            dense_quality_rank_loss = dense_box_rank_loss

    decision_utility = _threshold_transition_utility(
        box_targets, threshold_weights, break_cost
    )
    if mask_targets is not None and float(mask_utility_weight) > 0.0:
        decision_utility = decision_utility + float(mask_utility_weight) * (
            _threshold_transition_utility(
                mask_targets, threshold_weights, break_cost
            )
        )
    decision_targets = torch.ones_like(decision_utility, dtype=torch.long)
    decision_targets = torch.where(
        decision_utility < 0.0,
        torch.zeros_like(decision_targets),
        decision_targets,
    )
    decision_targets = torch.where(
        decision_utility > 0.0,
        torch.full_like(decision_targets, 2),
        decision_targets,
    )
    balanced_objective = objective in (
        "balanced_focal", "balanced_calibrated_utility",
        "hierarchical_risk_calibrated", "pairwise_risk_calibrated",
        "topn_risk_calibrated", "topn_dual_risk_calibrated",
    )
    calibrated_objective = objective in (
        "calibrated_utility", "balanced_calibrated_utility",
        "hierarchical_risk_calibrated", "pairwise_risk_calibrated",
        "topn_risk_calibrated", "topn_dual_risk_calibrated",
        "topn_absolute_quality_calibrated",
        "cascade_absolute_quality_calibrated",
        "cascade_opportunity_balanced_calibrated",
        "cascade_opportunity_verified_calibrated",
        "cascade_joint_risk_calibrated",
        "cascade_v19_fallback_set_risk_calibrated",
        "cascade_v19_rich_set_empirical_risk",
        "cascade_v23_dense_quality_risk",
        "cascade_v24_relative_risk",
        "cascade_v25_pairwise_calibrated_risk",
        "cascade_v26_prior_restored_pairwise_risk",
    )
    if balanced_objective:
        decision_class_loss = _class_balanced_focal_loss(
            decision_logits,
            decision_targets,
            active,
            gamma=focal_gamma,
            false_override_weight=false_override_weight,
        )
    else:
        decision_class_loss = _cost_sensitive_cross_entropy(
            decision_logits,
            decision_targets,
            active,
            false_override_weight=false_override_weight,
        )

    # The deployed action is one row-wise choice: fallback, or one candidate.
    # This loss uses exactly that action space, so a positive decision margin
    # has a learned meaning instead of being reconstructed from reweighted
    # per-threshold probabilities.
    override_margin = (
        decision_logits[..., 2]
        - torch.maximum(decision_logits[..., 0], decision_logits[..., 1])
    )
    selection_margin = (
        override_margin if action_margin is None else action_margin
    )
    masked_margin = selection_margin.masked_fill(~active, -1e4)
    best_predicted_margin, best_predicted_query = masked_margin.max(dim=1)
    masked_utility = decision_utility.masked_fill(~active, -1e4)
    best_utility, best_query = masked_utility.max(dim=1)
    has_candidate = active.any(dim=1)
    oracle_switch = has_candidate & (best_utility > 0.0) & sample_mask
    selected_utility = decision_utility[row_index, best_predicted_query]
    pairwise_objective = objective in (
        "pairwise_risk_calibrated", "topn_risk_calibrated",
        "topn_dual_risk_calibrated",
        "cascade_absolute_quality_calibrated",
        "cascade_opportunity_balanced_calibrated",
        "cascade_opportunity_verified_calibrated",
        "cascade_joint_risk_calibrated",
    )
    pairwise_candidate_action = (
        row_switch_margin is not None and row_switch_margin.dim() == 2
    )
    dual_verifier_objective = objective == "topn_dual_risk_calibrated"
    if objective in (
            "topn_risk_calibrated", "topn_dual_risk_calibrated",
            "cascade_absolute_quality_calibrated") and (
            not pairwise_candidate_action
            or float(setwise_temperature) <= 0.0):
        raise ValueError(
            "{} requires [B,Q] row_switch_margin and positive "
            "setwise_temperature".format(objective)
        )
    if opportunity_cascade_objective and (
            row_switch_margin is None or row_switch_margin.dim() != 1
            or float(setwise_temperature) <= 0.0):
        raise ValueError(
            "opportunity cascade objective requires [B] row_switch_margin "
            "and positive setwise_temperature"
        )
    if verified_opportunity_objective and row_safety_margin is None:
        raise ValueError(
            "verified/joint cascade objective requires [B,Q] "
            "row_safety_margin"
        )
    if joint_action_objective and joint_action_margin is None:
        raise ValueError(
            "joint fallback-set objective requires [B,Q] "
            "joint_action_margin"
        )
    if pairwise_utility_objective and (
            row_benefit_margin is None
            or (prior_restored_pairwise_objective
                and pairwise_utility_margin is None)):
        raise ValueError(
            "pairwise calibrated risk requires benefit margins and, for V26, "
            "a separate utility margin"
        )
    if selected_abstention_objective and (
            pairwise_utility_margin is None
            or candidate_selection_margin is None
            or selected_abstention_margin is None):
        raise ValueError(
            "selected abstention risk requires candidate selection, "
            "pairwise utility, and row abstention margins"
        )
    if (counterfactual_selected_objective
            and not decomposed_counterfactual_objective
            and counterfactual_risk_margin is None):
        raise ValueError(
            "counterfactual selected risk requires candidate risk margins"
        )
    if decomposed_counterfactual_objective and (
            counterfactual_benefit_margin is None
            or counterfactual_hazard_margin is None):
        raise ValueError(
            "decomposed counterfactual risk requires benefit and hazard "
            "margins"
        )
    if dual_verifier_objective and (
            row_benefit_margin is None or row_safety_margin is None):
        raise ValueError(
            "topn_dual_risk_calibrated requires [B,Q] benefit and safety "
            "margins"
        )
    if opportunity_cascade_objective:
        row_utility_target = torch.where(
            has_candidate,
            best_utility,
            torch.zeros_like(best_utility),
        )
        row_switch_target = oracle_switch
    elif pairwise_candidate_action:
        row_utility_target = torch.where(
            active,
            decision_utility,
            torch.zeros_like(decision_utility),
        )
        row_switch_target = (
            active & (row_utility_target > 0.0) & sample_mask.unsqueeze(1)
        )
    else:
        row_utility_target = torch.where(
            has_candidate,
            selected_utility if pairwise_objective else best_utility,
            torch.zeros_like(best_utility),
        )
        row_switch_target = (
            has_candidate & (row_utility_target > 0.0) & sample_mask
        )
    row_target_switch_rows = (
        row_switch_target.any(dim=1)
        if pairwise_candidate_action else row_switch_target
    )
    selection_targets = torch.where(
        oracle_switch, best_query + 1, torch.zeros_like(best_query)
    )
    row_classes = row_switch_target.long()
    if pairwise_candidate_action:
        balanced_row_objective = objective in (
            "balanced_focal", "balanced_calibrated_utility",
            "cascade_opportunity_balanced_calibrated",
            "cascade_opportunity_verified_calibrated",
            "cascade_joint_risk_calibrated",
        )
        if bool(sample_mask.any().item()):
            if balanced_row_objective:
                active_classes = row_target_switch_rows[sample_mask].long()
                counts = torch.bincount(active_classes, minlength=2).to(
                    dtype=decision_logits.dtype
                )
                class_weight = (
                    float(active_classes.numel())
                    / (2.0 * counts.clamp(min=1.0))
                )
                class_weight[0] = (
                    class_weight[0] * float(false_override_weight)
                )
            else:
                class_weight = decision_logits.new_tensor((
                    float(false_override_weight), 1.0
                ))
            row_weight = class_weight[row_target_switch_rows.long()]
        else:
            row_weight = decision_logits.new_zeros(
                row_target_switch_rows.shape
            )
    elif bool(sample_mask.any().item()):
        balanced_row_objective = objective in (
            "balanced_focal", "balanced_calibrated_utility",
            "cascade_opportunity_balanced_calibrated",
            "cascade_opportunity_verified_calibrated",
            "cascade_joint_risk_calibrated",
        )
        if balanced_row_objective:
            active_classes = row_classes[sample_mask]
            counts = torch.bincount(active_classes, minlength=2).to(
                dtype=decision_logits.dtype
            )
            class_weight = (
                float(active_classes.numel())
                / (2.0 * counts.clamp(min=1.0))
            )
            class_weight[0] = (
                class_weight[0] * float(false_override_weight)
            )
        else:
            class_weight = decision_logits.new_tensor((
                float(false_override_weight), 1.0
            ))
        row_weight = class_weight[row_classes]
    else:
        row_weight = decision_logits.new_zeros(row_classes.shape)

    hierarchical_action = row_switch_margin is not None
    if (objective in (
            "hierarchical_risk_calibrated", "pairwise_risk_calibrated",
            "topn_risk_calibrated", "topn_dual_risk_calibrated",
            "cascade_absolute_quality_calibrated",
            "cascade_opportunity_balanced_calibrated",
            "cascade_opportunity_verified_calibrated",
            "cascade_joint_risk_calibrated")
            and not hierarchical_action):
        raise ValueError(
            "{} requires row_switch_margin".format(objective)
        )
    row_switch_loss = decision_logits.sum() * 0.0
    candidate_safety_loss = decision_logits.sum() * 0.0
    safety_utility_regression_loss = decision_logits.sum() * 0.0
    joint_action_loss = decision_logits.sum() * 0.0
    deployment_boundary_loss = decision_logits.sum() * 0.0
    deployment_boundary_positive_loss = decision_logits.sum() * 0.0
    deployment_boundary_fallback_loss = decision_logits.sum() * 0.0
    joint_target_distribution = None
    joint_positive_prior = decision_logits.sum().detach() * 0.0
    joint_prior_log_odds = decision_logits.sum().detach() * 0.0
    candidate_safety_target = None
    verifier_target_distribution = None
    if hierarchical_action:
        if opportunity_cascade_objective:
            opportunity_target = build_risk_separated_action_target(
                decision_utility,
                active,
                temperature=float(setwise_temperature),
            )
            selection_target_distribution = opportunity_target[:, 1:]
        elif pairwise_objective and float(setwise_temperature) > 0.0:
            candidate_target_logits = (
                decision_utility.detach().float()
                / float(setwise_temperature)
            ).masked_fill(~active, -1e4)
            selection_target_distribution = F.softmax(
                candidate_target_logits, dim=1
            ) * has_candidate.unsqueeze(1).to(candidate_target_logits.dtype)
        elif pairwise_objective:
            selection_target_distribution = F.one_hot(
                best_query, num_classes=masked_margin.shape[1]
            ).to(dtype=masked_margin.dtype)
            selection_target_distribution = (
                selection_target_distribution
                * has_candidate.unsqueeze(1).to(masked_margin.dtype)
            )
        elif float(setwise_temperature) > 0.0:
            setwise_target = build_setwise_action_target(
                decision_utility,
                active,
                temperature=float(setwise_temperature),
            )
            selection_target_distribution = setwise_target[:, 1:]
            candidate_mass = selection_target_distribution.sum(
                dim=1, keepdim=True
            )
            selection_target_distribution = torch.where(
                candidate_mass > 0.0,
                selection_target_distribution
                / candidate_mass.clamp(min=1e-8),
                torch.zeros_like(selection_target_distribution),
            )
        else:
            selection_target_distribution = F.one_hot(
                best_query, num_classes=masked_margin.shape[1]
            ).to(dtype=masked_margin.dtype)
            selection_target_distribution = (
                selection_target_distribution
                * oracle_switch.unsqueeze(1).to(masked_margin.dtype)
            )
        candidate_row_loss = -(
            selection_target_distribution
            * F.log_softmax(masked_margin, dim=1)
        ).sum(dim=1)
        if opportunity_cascade_objective:
            candidate_rows = oracle_switch & sample_mask
        else:
            candidate_rows = (
                has_candidate & sample_mask
                if pairwise_objective else oracle_switch & sample_mask
            )
        if bool(candidate_rows.any().item()):
            selection_loss = candidate_row_loss[candidate_rows].mean()
        else:
            selection_loss = action_margin.sum() * 0.0

        if pairwise_candidate_action:
            verifier_training_margin = (
                row_benefit_margin
                if dual_verifier_objective else row_switch_margin
            )
            verifier_logits = torch.cat((
                torch.zeros(
                    decision_logits.shape[0], 1,
                    dtype=verifier_training_margin.dtype,
                    device=verifier_training_margin.device,
                ),
                verifier_training_margin.masked_fill(~active, -1e4),
            ), dim=1)
            if float(setwise_temperature) > 0.0:
                target_builder = (
                    build_risk_separated_action_target
                    if objective in (
                        "topn_risk_calibrated",
                        "topn_dual_risk_calibrated",
                        "cascade_absolute_quality_calibrated",
                    )
                    else build_setwise_action_target
                )
                verifier_target_distribution = target_builder(
                    decision_utility, active,
                    temperature=float(setwise_temperature),
                )
                verifier_row_loss = -(
                    verifier_target_distribution
                    * F.log_softmax(verifier_logits, dim=1)
                ).sum(dim=1)
            else:
                verifier_target_distribution = F.one_hot(
                    selection_targets, num_classes=verifier_logits.shape[1]
                ).to(dtype=verifier_logits.dtype)
                verifier_row_loss = F.cross_entropy(
                    verifier_logits, selection_targets, reduction="none"
                )
            if bool(sample_mask.any().item()):
                row_switch_loss = (
                    verifier_row_loss[sample_mask] * row_weight[sample_mask]
                ).sum() / row_weight[sample_mask].sum().clamp(min=1e-6)
            if dual_verifier_objective:
                safety_target = active & (decision_utility > 0.0)
                safety_loss = _class_balanced_binary_focal_loss(
                    row_safety_margin,
                    safety_target,
                    active,
                    gamma=focal_gamma,
                    false_positive_weight=false_override_weight,
                )
                row_switch_loss = 0.5 * (row_switch_loss + safety_loss)
        else:
            switch_row_loss = F.binary_cross_entropy_with_logits(
                row_switch_margin.float(),
                row_classes.to(dtype=torch.float32),
                reduction="none",
            )
            if bool(sample_mask.any().item()):
                row_switch_loss = (
                    switch_row_loss[sample_mask] * row_weight[sample_mask]
                ).sum() / row_weight[sample_mask].sum().clamp(min=1e-6)
        if verified_opportunity_objective:
            candidate_safety_target = active & (decision_utility > 0.0)
            candidate_safety_class_loss = (
                _cost_sensitive_binary_focal_loss(
                    row_safety_margin,
                    candidate_safety_target,
                    active,
                    gamma=focal_gamma,
                    false_positive_weight=false_override_weight,
                )
            )
            safety_gap = float(setwise_temperature)
            candidate_safety_utility_target = torch.where(
                decision_utility > 0.0,
                decision_utility,
                torch.minimum(
                    decision_utility,
                    decision_utility.new_full((), -safety_gap),
                ),
            )
            safety_utility_regression_loss = (
                _calibrated_utility_regression_loss(
                    row_safety_margin,
                    candidate_safety_utility_target,
                    active,
                    false_override_weight=false_override_weight,
                )
            )
            candidate_safety_loss = 0.5 * (
                candidate_safety_class_loss
                + safety_utility_regression_loss
            )
    else:
        selection_logits = torch.cat((
            torch.zeros(
                decision_logits.shape[0], 1,
                dtype=decision_logits.dtype,
                device=decision_logits.device,
            ),
            masked_margin,
        ), dim=1)
        if float(setwise_temperature) > 0.0:
            selection_target_builder = (
                build_risk_separated_action_target
                if absolute_quality_objective and not relative_risk_objective
                else build_setwise_action_target
            )
            selection_target_distribution = selection_target_builder(
                decision_utility,
                active,
                temperature=float(setwise_temperature),
            )
            row_loss = -(
                selection_target_distribution
                * F.log_softmax(selection_logits, dim=1)
            ).sum(dim=1)
        else:
            selection_target_distribution = F.one_hot(
                selection_targets, num_classes=selection_logits.shape[1]
            ).to(dtype=selection_logits.dtype)
            row_loss = F.cross_entropy(
                selection_logits, selection_targets, reduction="none"
            )
        if bool(sample_mask.any().item()):
            selection_loss = (
                row_loss[sample_mask] * row_weight[sample_mask]
            ).sum() / row_weight[sample_mask].sum().clamp(min=1e-6)
        else:
            selection_loss = decision_logits.sum() * 0.0

    if joint_action_objective:
        if relative_risk_objective:
            joint_loss_builder = _relative_setwise_action_risk_loss
        else:
            joint_loss_builder = (
                _empirical_setwise_action_risk_loss
                if rich_set_objective or dense_quality_objective
                else _prior_corrected_setwise_action_loss
            )
        (
            joint_action_loss,
            joint_target_distribution,
            joint_positive_prior,
            joint_prior_log_odds,
        ) = joint_loss_builder(
            candidate_margin=joint_action_margin,
            decision_utility=decision_utility,
            active_mask=active,
            sample_mask=sample_mask,
            temperature=float(setwise_temperature),
            gamma=focal_gamma,
            false_positive_weight=false_override_weight,
        )
    if fallback_set_objective:
        boundary = _balanced_deployment_boundary_loss(
            candidate_margin=joint_action_margin,
            decision_utility=decision_utility,
            active_mask=active,
            sample_mask=sample_mask,
            temperature=float(setwise_temperature),
        )
        deployment_boundary_loss = boundary["loss"]
        deployment_boundary_positive_loss = boundary["positive_loss"]
        deployment_boundary_fallback_loss = boundary["fallback_loss"]

    verifier_utility_target = row_utility_target
    if (pairwise_candidate_action
            and objective in (
                "topn_risk_calibrated",
                "cascade_absolute_quality_calibrated",
            )):
        safety_gap = float(setwise_temperature)
        verifier_utility_target = torch.where(
            row_utility_target > 0.0,
            row_utility_target,
            torch.minimum(
                row_utility_target,
                row_utility_target.new_full((), -safety_gap),
            ),
        )
    elif opportunity_cascade_objective:
        safety_gap = float(setwise_temperature)
        verifier_utility_target = torch.where(
            row_utility_target > 0.0,
            row_utility_target,
            torch.minimum(
                row_utility_target,
                row_utility_target.new_full((), -safety_gap),
            ),
        )
    utility_regression_loss = decision_logits.sum() * 0.0
    benefit_loss = decision_logits.sum() * 0.0
    benefit_prior_shift = decision_logits.sum().detach() * 0.0
    benefit_positive_prior = decision_logits.sum().detach() * 0.0
    pairwise_rank_loss = decision_logits.sum() * 0.0
    boundary_calibration_loss = decision_logits.sum() * 0.0
    boundary_positive_loss = decision_logits.sum().detach() * 0.0
    boundary_fallback_loss = decision_logits.sum().detach() * 0.0
    boundary_positive_ratio = decision_logits.sum().detach() * 0.0
    quality_action_loss = decision_logits.sum() * 0.0
    quality_action_regression_loss = decision_logits.sum() * 0.0
    quality_action_rank_loss = decision_logits.sum() * 0.0
    quality_action_target_positive_ratio = decision_logits.sum().detach() * 0.0
    quality_action_false_positive_ratio = decision_logits.sum().detach() * 0.0
    abstention_regression_loss = decision_logits.sum() * 0.0
    abstention_benefit_loss = decision_logits.sum() * 0.0
    abstention_selection_regression_loss = decision_logits.sum() * 0.0
    abstention_selection_rank_loss = decision_logits.sum() * 0.0
    abstention_prior_shift = decision_logits.sum().detach() * 0.0
    abstention_positive_prior = decision_logits.sum().detach() * 0.0
    abstention_target_positive_ratio = decision_logits.sum().detach() * 0.0
    counterfactual_risk_loss = decision_logits.sum() * 0.0
    counterfactual_risk_classification_loss = decision_logits.sum() * 0.0
    counterfactual_risk_regression_loss = decision_logits.sum() * 0.0
    counterfactual_benefit_regression_loss = decision_logits.sum() * 0.0
    counterfactual_hazard_regression_loss = decision_logits.sum() * 0.0
    counterfactual_benefit_classification_loss = (
        decision_logits.sum() * 0.0
    )
    counterfactual_hazard_classification_loss = (
        decision_logits.sum() * 0.0
    )
    counterfactual_risk_deployment_boundary_loss = (
        decision_logits.sum() * 0.0
    )
    positive_mass_loss = decision_logits.sum() * 0.0
    positive_top1_loss = decision_logits.sum() * 0.0
    counterfactual_risk_prior_shift = decision_logits.sum().detach() * 0.0
    counterfactual_risk_positive_prior = decision_logits.sum().detach() * 0.0
    benefit_targets = None
    if selected_abstention_objective:
        abstention_selection_regression_loss = (
            _calibrated_utility_regression_loss(
                pairwise_utility_margin,
                decision_utility,
                active,
                false_override_weight=false_override_weight,
            )
        )
        abstention_selection_rank_loss = listwise_quality_loss(
            scores=candidate_selection_margin,
            quality=decision_utility,
            valid_mask=active,
            sample_mask=sample_mask,
            temperature=float(setwise_temperature),
        )
        selected_candidate_margin = candidate_selection_margin.masked_fill(
            ~active, -1e4
        )
        _, abstention_selected_query = selected_candidate_margin.max(dim=1)
        abstention_target_utility = decision_utility[
            row_index, abstention_selected_query
        ]
        abstention_row_active = (
            active.any(dim=1) & sample_mask
        ).unsqueeze(1)
        abstention_margin_matrix = selected_abstention_margin.unsqueeze(1)
        abstention_target_matrix = abstention_target_utility.unsqueeze(1)
        abstention_regression_loss = _calibrated_utility_regression_loss(
            abstention_margin_matrix,
            abstention_target_matrix,
            abstention_row_active,
            false_override_weight=false_override_weight,
        )
        abstention_targets = (
            abstention_row_active & (abstention_target_matrix > 0.0)
        )
        (
            abstention_benefit_loss,
            abstention_prior_shift,
            abstention_positive_prior,
        ) = _prior_restored_balanced_benefit_loss(
            abstention_margin_matrix,
            abstention_targets,
            abstention_row_active,
            false_positive_weight=false_override_weight,
        )
        abstention_target_positive_ratio = (
            abstention_targets.float().sum()
            / abstention_row_active.float().sum().clamp(min=1.0)
        ).detach()
        benefit_targets = active & (decision_utility > 0.0)
        (
            benefit_loss,
            benefit_prior_shift,
            benefit_positive_prior,
        ) = _prior_restored_balanced_benefit_loss(
            row_benefit_margin,
            benefit_targets,
            active,
            false_positive_weight=false_override_weight,
        )
        if counterfactual_selected_objective:
            (
                positive_mass_loss,
                _positive_mass_rows,
                _positive_mass,
            ) = _positive_candidate_mass_loss(
                candidate_selection_margin,
                decision_utility,
                active,
                sample_mask,
                temperature=float(setwise_temperature),
            )
            positive_top1_loss, _positive_top1_rows, _top1_violation = (
                _positive_candidate_top1_margin_loss(
                    candidate_selection_margin,
                    decision_utility,
                    active,
                    sample_mask,
                    margin=max(0.05, 0.25 * float(setwise_temperature)),
                )
            )
            if hazard_residual_objective:
                counterfactual = _counterfactual_hazard_residual_loss(
                    counterfactual_benefit_margin,
                    counterfactual_hazard_margin,
                    candidate_selection_margin,
                    decision_utility,
                    active,
                    sample_mask,
                    focal_gamma=float(focal_gamma),
                )
            elif complementary_logodds_objective:
                counterfactual = (
                    _counterfactual_complementary_logodds_loss(
                        counterfactual_benefit_margin,
                        counterfactual_hazard_margin,
                        candidate_selection_margin,
                        decision_utility,
                        active,
                        sample_mask,
                        focal_gamma=float(focal_gamma),
                    )
                )
            elif decomposed_counterfactual_objective:
                counterfactual = _counterfactual_benefit_hazard_loss(
                    counterfactual_benefit_margin,
                    counterfactual_hazard_margin,
                    candidate_selection_margin,
                    decision_utility,
                    active,
                    sample_mask,
                    focal_gamma=float(focal_gamma),
                )
            else:
                counterfactual = _counterfactual_selected_risk_loss(
                    counterfactual_risk_margin,
                    candidate_selection_margin,
                    decision_utility,
                    active,
                    sample_mask,
                    false_positive_weight=float(false_override_weight),
                    temperature=float(setwise_temperature),
                )
            counterfactual_risk_loss = counterfactual["loss"]
            counterfactual_risk_classification_loss = (
                counterfactual["classification_loss"]
            )
            counterfactual_risk_regression_loss = (
                counterfactual["regression_loss"]
            )
            if decomposed_counterfactual_objective:
                if (complementary_logodds_objective
                        or hazard_residual_objective):
                    counterfactual_benefit_classification_loss = (
                        counterfactual["benefit_classification_loss"]
                    )
                    counterfactual_hazard_classification_loss = (
                        counterfactual["hazard_classification_loss"]
                    )
                else:
                    counterfactual_benefit_regression_loss = counterfactual[
                        "benefit_regression_loss"
                    ]
                    counterfactual_hazard_regression_loss = counterfactual[
                        "hazard_regression_loss"
                    ]
                counterfactual_risk_deployment_boundary_loss = (
                    counterfactual_risk_classification_loss
                )
            else:
                counterfactual_risk_deployment_boundary_loss = counterfactual[
                    "deployment_boundary_loss"
                ]
                counterfactual_risk_prior_shift = counterfactual[
                    "prior_shift"
                ]
                counterfactual_risk_positive_prior = counterfactual[
                    "positive_prior"
                ]
        decision_loss = (
            abstention_regression_loss
            # V29 counterfactual risk already classifies every candidate at
            # the raw deployment boundary.  Reusing V28's selected-row
            # prior-shifted BCE on that same risk margin would move the
            # effective boundary back to -prior_shift and cause undercut.
            + (
                decision_logits.sum() * 0.0
                if counterfactual_selected_objective
                else abstention_benefit_loss
            )
            + abstention_selection_regression_loss
            + abstention_selection_rank_loss
            + benefit_loss
            + positive_mass_loss
            + positive_top1_loss
            + counterfactual_risk_loss
            + absolute_quality_loss
            + dense_quality_rank_loss
        )
    elif prior_restored_pairwise_objective:
        utility_regression_loss = _calibrated_utility_regression_loss(
            pairwise_utility_margin,
            decision_utility,
            active,
            false_override_weight=false_override_weight,
        )
        benefit_targets = active & (decision_utility > 0.0)
        (
            benefit_loss,
            benefit_prior_shift,
            benefit_positive_prior,
        ) = _prior_restored_balanced_benefit_loss(
            row_benefit_margin,
            benefit_targets,
            active,
            false_positive_weight=false_override_weight,
        )
        pairwise_rank_loss = listwise_quality_loss(
            scores=joint_action_margin,
            quality=decision_utility,
            valid_mask=active,
            sample_mask=sample_mask,
            temperature=float(setwise_temperature),
        )
        if float(boundary_loss_weight) > 0.0:
            boundary = _rowwise_boundary_calibration_loss(
                row_benefit_margin,
                decision_utility,
                active,
                sample_mask,
                temperature=max(float(setwise_temperature), 0.05),
                # This auxiliary term calibrates the row decision boundary,
                # so class balancing must not add the deployment false-
                # positive cost a second time.  That cost remains explicit
                # in the candidate-level benefit likelihood above.
                false_positive_weight=1.0,
            )
            boundary_calibration_loss = boundary["loss"]
            boundary_positive_loss = boundary["positive_loss"]
            boundary_fallback_loss = boundary["fallback_loss"]
            boundary_positive_ratio = boundary["positive_ratio"]
        # Setwise fallback risk is retained as a diagnostic target, but is
        # intentionally excluded from the deployed-margin loss: its fallback
        # prior would double-count the calibrated binary boundary.
        decision_loss = (
            benefit_loss
            + utility_regression_loss
            + pairwise_rank_loss
            + float(boundary_loss_weight) * boundary_calibration_loss
            + absolute_quality_loss
            + dense_quality_rank_loss
        )
    elif pairwise_calibrated_objective:
        utility_regression_loss = _calibrated_utility_regression_loss(
            joint_action_margin,
            decision_utility,
            active,
            false_override_weight=false_override_weight,
        )
        benefit_targets = active & (decision_utility > 0.0)
        benefit_loss = _cost_sensitive_binary_focal_loss(
            row_benefit_margin,
            benefit_targets,
            active,
            gamma=focal_gamma,
            false_positive_weight=false_override_weight,
        )
        decision_loss = (
            joint_action_loss
            + utility_regression_loss
            + benefit_loss
            + absolute_quality_loss
            + dense_quality_rank_loss
        )
    elif quality_risk_objective:
        quality_action = _risk_aware_dense_quality_action_loss(
            candidate_margin=joint_action_margin,
            deployment_utility=decision_utility,
            box_ious=box_ious,
            mask_ious=mask_ious,
            fallback_indices=default_indices,
            active_mask=active,
            sample_mask=sample_mask,
            mask_utility_weight=mask_utility_weight,
            false_override_weight=false_override_weight,
            temperature=float(setwise_temperature),
        )
        quality_action_loss = quality_action["loss"]
        quality_action_regression_loss = quality_action["regression_loss"]
        quality_action_rank_loss = quality_action["rank_loss"]
        quality_action_target_positive_ratio = quality_action[
            "target_positive_ratio"
        ]
        quality_action_false_positive_ratio = quality_action[
            "false_positive_ratio"
        ]
        decision_loss = (
            quality_action_loss
            + absolute_quality_loss
            + dense_quality_rank_loss
        )
    elif fallback_set_objective:
        decision_loss = joint_action_loss + deployment_boundary_loss
    elif rich_set_objective:
        decision_loss = joint_action_loss
    elif dense_quality_objective:
        decision_loss = (
            joint_action_loss
            + absolute_quality_loss
            + dense_quality_rank_loss
        )
    elif calibrated_objective:
        if pairwise_candidate_action:
            utility_margin = (
                row_benefit_margin
                if dual_verifier_objective else row_switch_margin
            )
            utility_regression_loss = _calibrated_utility_regression_loss(
                utility_margin,
                verifier_utility_target,
                active,
                false_override_weight=false_override_weight,
            )
            decision_terms = (
                decision_class_loss,
                selection_loss,
                row_switch_loss,
                utility_regression_loss,
            )
            if cascade_quality_objective:
                decision_terms = decision_terms + (absolute_quality_loss,)
            decision_loss = sum(decision_terms) / float(len(decision_terms))
        elif hierarchical_action:
            row_active = has_candidate & sample_mask
            if opportunity_cascade_objective:
                utility_regression_loss = (
                    _weighted_calibrated_row_regression_loss(
                        row_switch_margin,
                        verifier_utility_target,
                        row_active,
                        row_weight,
                        false_override_weight=false_override_weight,
                    )
                )
                decision_terms = (
                    selection_loss,
                    row_switch_loss,
                    utility_regression_loss,
                    absolute_quality_loss,
                )
                if verified_opportunity_objective:
                    decision_terms = decision_terms + (
                        candidate_safety_loss,
                    )
                if joint_action_objective:
                    decision_terms = decision_terms + (joint_action_loss,)
                decision_loss = sum(decision_terms) / float(
                    len(decision_terms)
                )
            else:
                utility_regression_loss = (
                    _calibrated_utility_regression_loss(
                        row_switch_margin.unsqueeze(1),
                        row_utility_target.unsqueeze(1),
                        row_active.unsqueeze(1),
                        false_override_weight=false_override_weight,
                    )
                )
                decision_loss = (
                    decision_class_loss
                    + selection_loss
                    + row_switch_loss
                    + utility_regression_loss
                ) / 4.0
        else:
            utility_regression_loss = _calibrated_utility_regression_loss(
                selection_margin,
                decision_utility,
                active,
                false_override_weight=false_override_weight,
            )
            if absolute_quality_objective:
                decision_loss = (
                    decision_class_loss
                    + selection_loss
                    + utility_regression_loss
                    + absolute_quality_loss
                ) / 4.0
            else:
                decision_loss = (
                    decision_class_loss
                    + selection_loss
                    + utility_regression_loss
                ) / 3.0
    elif hierarchical_action:
        decision_loss = (
            decision_class_loss + selection_loss + row_switch_loss
        ) / 3.0
    else:
        decision_loss = 0.5 * (decision_class_loss + selection_loss)

    decision_denominator = active.float().sum().clamp(min=1.0)
    stats["source_moe_gate_target_decision_break_ratio"] = (
        ((decision_targets == 0) & active).float().sum()
        / decision_denominator
    ).detach()
    stats["source_moe_gate_target_decision_neutral_ratio"] = (
        ((decision_targets == 1) & active).float().sum()
        / decision_denominator
    ).detach()
    stats["source_moe_gate_target_decision_fix_ratio"] = (
        ((decision_targets == 2) & active).float().sum()
        / decision_denominator
    ).detach()
    if pairwise_candidate_action:
        deployed_candidate_margin = row_switch_margin.masked_fill(
            ~active, -1e4
        )
        deployed_switch_margin, deployed_query = (
            deployed_candidate_margin.max(dim=1)
        )
        selected_utility = decision_utility[row_index, deployed_query]
    elif joint_action_objective:
        deployed_candidate_margin = joint_action_margin.masked_fill(
            ~active, -1e4
        )
        deployed_switch_margin, deployed_query = (
            deployed_candidate_margin.max(dim=1)
        )
        selected_utility = decision_utility[row_index, deployed_query]
    else:
        deployed_query = best_predicted_query
        if verified_opportunity_objective:
            selected_safety_margin = row_safety_margin[
                row_index, deployed_query
            ]
            deployed_switch_margin = torch.minimum(
                row_switch_margin, selected_safety_margin
            )
        else:
            deployed_switch_margin = (
                row_switch_margin
                if hierarchical_action else best_predicted_margin
            )
    predicted_switch = (
        has_candidate & (deployed_switch_margin > 0.0) & sample_mask
    )
    beneficial_switch = predicted_switch & (selected_utility > 0.0)
    harmful_switch = predicted_switch & (selected_utility <= 0.0)
    oracle_query_match = (
        torch.isclose(selected_utility, best_utility) & oracle_switch
    )
    stats["source_moe_gate_supervised_sample_count"] = (
        sample_mask.float().sum().detach()
    )
    stats["source_moe_gate_oracle_switch_count"] = (
        oracle_switch.float().sum().detach()
    )
    stats["source_moe_gate_row_target_switch_count"] = (
        row_target_switch_rows.float().sum().detach()
    )
    stats["source_moe_gate_predicted_switch_count"] = (
        predicted_switch.float().sum().detach()
    )
    stats["source_moe_gate_beneficial_switch_count"] = (
        beneficial_switch.float().sum().detach()
    )
    stats["source_moe_gate_harmful_switch_count"] = (
        harmful_switch.float().sum().detach()
    )
    stats["source_moe_gate_oracle_query_match_count"] = (
        oracle_query_match.float().sum().detach()
    )
    if bool(sample_mask.any().item()):
        stats["source_moe_gate_oracle_switch_ratio"] = (
            oracle_switch[sample_mask].float().mean().detach()
        )
        stats["source_moe_gate_row_target_switch_ratio"] = (
            row_target_switch_rows[sample_mask].float().mean().detach()
        )
        oracle_switch_count = oracle_switch.float().sum().clamp(min=1.0)
        predicted_switch_count = predicted_switch.float().sum().clamp(min=1.0)
        stats["source_moe_gate_oracle_switch_recall_ratio"] = (
            beneficial_switch.float().sum() / oracle_switch_count
        ).detach()
        stats["source_moe_gate_predicted_switch_precision_ratio"] = (
            beneficial_switch.float().sum() / predicted_switch_count
        ).detach()
        stats["source_moe_gate_false_switch_ratio"] = (
            harmful_switch.float().sum() / predicted_switch_count
        ).detach()
        stats["source_moe_gate_oracle_query_match_ratio"] = (
            oracle_query_match.float().sum() / oracle_switch_count
        ).detach()
        finite_best_utility = torch.where(
            has_candidate, best_utility, torch.zeros_like(best_utility)
        )
        stats["source_moe_gate_oracle_best_utility_mean"] = (
            finite_best_utility[sample_mask].mean().detach()
        )
    if calibrated_objective:
        stats["source_moe_gate_utility_regression_loss"] = (
            utility_regression_loss.detach()
        )
        if pairwise_calibrated_objective:
            stats["source_moe_gate_utility_overestimate_ratio"] = (
                ((joint_action_margin > decision_utility) & active).float().sum()
                / decision_denominator
            ).detach()
        elif prior_restored_pairwise_objective:
            stats["source_moe_gate_utility_overestimate_ratio"] = (
                ((pairwise_utility_margin > decision_utility) & active)
                .float().sum() / decision_denominator
            ).detach()
        elif pairwise_candidate_action:
            utility_margin = (
                row_benefit_margin
                if dual_verifier_objective else row_switch_margin
            )
            stats["source_moe_gate_utility_overestimate_ratio"] = (
                ((utility_margin > verifier_utility_target)
                 & active).float().sum()
                / decision_denominator
            ).detach()
        elif hierarchical_action:
            row_denominator = (
                has_candidate & sample_mask
            ).float().sum().clamp(min=1.0)
            row_regression_target = (
                verifier_utility_target
                if opportunity_cascade_objective else row_utility_target
            )
            stats["source_moe_gate_utility_overestimate_ratio"] = (
                ((row_switch_margin > row_regression_target)
                 & has_candidate & sample_mask).float().sum()
                / row_denominator
            ).detach()
        else:
            stats["source_moe_gate_utility_overestimate_ratio"] = (
                ((selection_margin > decision_utility) & active).float().sum()
                / decision_denominator
            ).detach()
    if hierarchical_action:
        stats["source_moe_gate_row_switch_loss"] = row_switch_loss.detach()
        row_margin_values = (
            deployed_switch_margin[has_candidate & sample_mask]
            if pairwise_candidate_action
            else (
                deployed_switch_margin[sample_mask]
                if verified_opportunity_objective
                else row_switch_margin[sample_mask]
            )
        )
        stats["source_moe_gate_row_margin_mean"] = (
            row_margin_values.mean().detach()
            if bool(row_margin_values.numel() > 0)
            else row_switch_margin.sum().detach() * 0.0
        )
        if opportunity_cascade_objective:
            opportunity_denominator = sample_mask.float().sum().clamp(min=1.0)
            stats["source_moe_gate_opportunity_positive_ratio"] = (
                ((row_switch_margin > 0.0) & sample_mask).float().sum()
                / opportunity_denominator
            ).detach()
    if dual_verifier_objective:
        safety_target = active & (decision_utility > 0.0)
        stats["source_moe_gate_safety_loss"] = safety_loss.detach()
        stats["source_moe_gate_safety_positive_ratio"] = (
            ((row_safety_margin > 0.0) & active).float().sum()
            / decision_denominator
        ).detach()
        stats["source_moe_gate_safety_false_positive_ratio"] = (
            ((row_safety_margin > 0.0) & ~safety_target & active).float().sum()
            / decision_denominator
        ).detach()
    elif verified_opportunity_objective:
        stats["source_moe_gate_safety_loss"] = (
            candidate_safety_loss.detach()
        )
        stats["source_moe_gate_safety_utility_regression_loss"] = (
            safety_utility_regression_loss.detach()
        )
        stats["source_moe_gate_safety_positive_ratio"] = (
            ((row_safety_margin > 0.0) & active).float().sum()
            / decision_denominator
        ).detach()
        stats["source_moe_gate_safety_false_positive_ratio"] = (
            ((row_safety_margin > 0.0)
             & ~candidate_safety_target & active).float().sum()
            / decision_denominator
        ).detach()
    if joint_action_objective:
        joint_target_action = joint_target_distribution.argmax(dim=1)
        joint_predicted_action = torch.where(
            predicted_switch,
            deployed_query + 1,
            torch.zeros_like(deployed_query),
        )
        joint_target_match = (
            (joint_predicted_action == joint_target_action) & sample_mask
        )
        supervised_denominator = sample_mask.float().sum().clamp(min=1.0)
        joint_margin_values = joint_action_margin[active]
        stats["source_moe_gate_joint_action_loss"] = (
            joint_action_loss.detach()
        )
        stats["source_moe_gate_joint_action_positive_prior"] = (
            joint_positive_prior
        )
        stats["source_moe_gate_joint_action_prior_log_odds"] = (
            joint_prior_log_odds
        )
        stats["source_moe_gate_joint_action_positive_ratio"] = (
            ((joint_action_margin > 0.0) & active).float().sum()
            / decision_denominator
        ).detach()
        stats["source_moe_gate_joint_action_margin_mean"] = (
            joint_margin_values.mean().detach()
            if bool(joint_margin_values.numel() > 0)
            else joint_action_margin.sum().detach() * 0.0
        )
        stats["source_moe_gate_joint_action_target_match_count"] = (
            joint_target_match.float().sum().detach()
        )
        stats["source_moe_gate_joint_action_target_match_ratio"] = (
            joint_target_match.float().sum() / supervised_denominator
        ).detach()
        if fallback_set_objective:
            stats["source_moe_gate_deployment_boundary_loss"] = (
                deployment_boundary_loss.detach()
            )
            stats["source_moe_gate_deployment_boundary_positive_loss"] = (
                deployment_boundary_positive_loss.detach()
            )
            stats["source_moe_gate_deployment_boundary_fallback_loss"] = (
                deployment_boundary_fallback_loss.detach()
            )
    if pairwise_utility_objective:
        stats["source_moe_gate_benefit_loss"] = benefit_loss.detach()
        stats["source_moe_gate_benefit_target_positive_ratio"] = (
            (benefit_targets & active).float().sum() / decision_denominator
        ).detach()
        stats["source_moe_gate_benefit_predicted_positive_ratio"] = (
            ((row_benefit_margin > 0.0) & active).float().sum()
            / decision_denominator
        ).detach()
        stats["source_moe_gate_benefit_false_positive_ratio"] = (
            ((row_benefit_margin > 0.0) & ~benefit_targets & active)
            .float().sum() / decision_denominator
        ).detach()
    if prior_restored_pairwise_objective:
        stats["source_moe_gate_benefit_prior_shift"] = (
            benefit_prior_shift.detach()
        )
        stats["source_moe_gate_benefit_positive_prior"] = (
            benefit_positive_prior.detach()
        )
        stats["source_moe_gate_pairwise_rank_loss"] = (
            pairwise_rank_loss.detach()
        )
        stats["source_moe_gate_boundary_calibration_loss"] = (
            boundary_calibration_loss.detach()
        )
        stats["source_moe_gate_boundary_positive_loss"] = (
            boundary_positive_loss
        )
        stats["source_moe_gate_boundary_fallback_loss"] = (
            boundary_fallback_loss
        )
        stats["source_moe_gate_boundary_positive_ratio"] = (
            boundary_positive_ratio
        )
    if selected_abstention_objective:
        selected_positive_rows = (
            abstention_targets.squeeze(1) & sample_mask
        )
        selected_positive_count = selected_positive_rows.float().sum()
        oracle_positive_count = oracle_switch.float().sum().clamp(min=1.0)
        selected_positive_denominator = selected_positive_count.clamp(
            min=1.0
        )
        stats["source_moe_gate_abstention_regression_loss"] = (
            abstention_regression_loss.detach()
        )
        stats["source_moe_gate_abstention_benefit_loss"] = (
            abstention_benefit_loss.detach()
        )
        stats["source_moe_gate_abstention_selection_regression_loss"] = (
            abstention_selection_regression_loss.detach()
        )
        stats["source_moe_gate_abstention_selection_rank_loss"] = (
            abstention_selection_rank_loss.detach()
        )
        stats["source_moe_gate_abstention_prior_shift"] = (
            abstention_prior_shift.detach()
        )
        stats["source_moe_gate_abstention_positive_prior"] = (
            abstention_positive_prior.detach()
        )
        stats["source_moe_gate_abstention_target_positive_ratio"] = (
            abstention_target_positive_ratio
        )
        stats["source_moe_gate_abstention_predicted_positive_ratio"] = (
            ((selected_abstention_margin > 0.0) & sample_mask).float().sum()
            / sample_mask.float().sum().clamp(min=1.0)
        ).detach()
        stats["source_moe_gate_policy_selected_positive_count"] = (
            selected_positive_count.detach()
        )
        stats["source_moe_gate_policy_opportunity_capture_ratio"] = (
            selected_positive_count / oracle_positive_count
        ).detach()
        stats["source_moe_gate_abstention_conditional_recall_ratio"] = (
            beneficial_switch.float().sum()
            / selected_positive_denominator
        ).detach()
        if counterfactual_selected_objective:
            stats["source_moe_gate_counterfactual_risk_loss"] = (
                counterfactual_risk_loss.detach()
            )
            stats["source_moe_gate_counterfactual_risk_classification_loss"] = (
                counterfactual_risk_classification_loss.detach()
            )
            stats["source_moe_gate_counterfactual_risk_regression_loss"] = (
                counterfactual_risk_regression_loss.detach()
            )
            if decomposed_counterfactual_objective:
                if (complementary_logodds_objective
                        or hazard_residual_objective):
                    stats[
                        "source_moe_gate_counterfactual_benefit_classification_loss"
                    ] = counterfactual_benefit_classification_loss.detach()
                    stats[
                        "source_moe_gate_counterfactual_hazard_classification_loss"
                    ] = counterfactual_hazard_classification_loss.detach()
                else:
                    stats[
                        "source_moe_gate_counterfactual_benefit_regression_loss"
                    ] = counterfactual_benefit_regression_loss.detach()
                    stats[
                        "source_moe_gate_counterfactual_hazard_regression_loss"
                    ] = counterfactual_hazard_regression_loss.detach()
            stats[
                "source_moe_gate_counterfactual_risk_deployment_boundary_loss"
            ] = counterfactual_risk_deployment_boundary_loss.detach()
            stats["source_moe_gate_positive_candidate_mass_loss"] = (
                positive_mass_loss.detach()
            )
            stats["source_moe_gate_positive_candidate_top1_loss"] = (
                positive_top1_loss.detach()
            )
            stats["source_moe_gate_counterfactual_risk_prior_shift"] = (
                counterfactual_risk_prior_shift.detach()
            )
            stats["source_moe_gate_counterfactual_risk_positive_prior"] = (
                counterfactual_risk_positive_prior.detach()
            )
    if absolute_quality_objective:
        stats["source_moe_gate_absolute_quality_loss"] = (
            absolute_quality_loss.detach()
        )
        stats["source_moe_gate_absolute_box_threshold_loss"] = (
            absolute_quality_losses["box_threshold_loss"].detach()
        )
        stats["source_moe_gate_absolute_box_iou_loss"] = (
            absolute_quality_losses["box_iou_loss"].detach()
        )
        stats["source_moe_gate_absolute_mask_threshold_loss"] = (
            absolute_quality_losses["mask_threshold_loss"].detach()
        )
        stats["source_moe_gate_absolute_mask_iou_loss"] = (
            absolute_quality_losses["mask_iou_loss"].detach()
        )
    if dense_quality_objective:
        stats["source_moe_gate_dense_box_rank_loss"] = (
            dense_box_rank_loss.detach()
        )
        stats["source_moe_gate_dense_mask_rank_loss"] = (
            dense_mask_rank_loss.detach()
        )
        stats["source_moe_gate_dense_quality_rank_loss"] = (
            dense_quality_rank_loss.detach()
        )
    if quality_risk_objective:
        stats["source_moe_gate_quality_action_loss"] = (
            quality_action_loss.detach()
        )
        stats["source_moe_gate_quality_action_regression_loss"] = (
            quality_action_regression_loss.detach()
        )
        stats["source_moe_gate_quality_action_rank_loss"] = (
            quality_action_rank_loss.detach()
        )
        stats["source_moe_gate_quality_action_target_positive_ratio"] = (
            quality_action_target_positive_ratio
        )
        stats["source_moe_gate_quality_action_false_positive_ratio"] = (
            quality_action_false_positive_ratio
        )

    total_loss = (
        decision_loss
        if v19_set_objective else (
            box_loss + float(mask_loss_weight) * mask_loss + decision_loss
        )
    )
    return {
        "loss": total_loss,
        "box_loss": box_loss,
        "mask_loss": mask_loss,
        "decision_loss": decision_loss,
        "decision_class_loss": decision_class_loss,
        "selection_loss": selection_loss,
        "row_switch_loss": row_switch_loss,
        "safety_loss": (
            safety_loss if dual_verifier_objective else (
                candidate_safety_loss if verified_opportunity_objective
                else decision_logits.sum() * 0.0
            )
        ),
        "safety_utility_regression_loss": safety_utility_regression_loss,
        "joint_action_loss": joint_action_loss,
        "deployment_boundary_loss": deployment_boundary_loss,
        "absolute_quality_loss": absolute_quality_loss,
        "dense_quality_rank_loss": dense_quality_rank_loss,
        "utility_regression_loss": utility_regression_loss,
        "benefit_loss": benefit_loss,
        "benefit_prior_shift": benefit_prior_shift,
        "benefit_positive_prior": benefit_positive_prior,
        "pairwise_rank_loss": pairwise_rank_loss,
        "boundary_calibration_loss": boundary_calibration_loss,
        "quality_action_loss": quality_action_loss,
        "abstention_regression_loss": abstention_regression_loss,
        "abstention_benefit_loss": abstention_benefit_loss,
        "abstention_selection_regression_loss": (
            abstention_selection_regression_loss
        ),
        "abstention_selection_rank_loss": abstention_selection_rank_loss,
        "counterfactual_risk_loss": counterfactual_risk_loss,
        "counterfactual_risk_classification_loss": (
            counterfactual_risk_classification_loss
        ),
        "counterfactual_risk_regression_loss": (
            counterfactual_risk_regression_loss
        ),
        "counterfactual_benefit_regression_loss": (
            counterfactual_benefit_regression_loss
        ),
        "counterfactual_hazard_regression_loss": (
            counterfactual_hazard_regression_loss
        ),
        "counterfactual_benefit_classification_loss": (
            counterfactual_benefit_classification_loss
        ),
        "counterfactual_hazard_classification_loss": (
            counterfactual_hazard_classification_loss
        ),
        "counterfactual_risk_deployment_boundary_loss": (
            counterfactual_risk_deployment_boundary_loss
        ),
        "positive_mass_loss": positive_mass_loss,
        "positive_top1_loss": positive_top1_loss,
        "box_targets": box_targets,
        "mask_targets": mask_targets,
        "decision_targets": decision_targets,
        "decision_utility": decision_utility,
        "row_switch_targets": row_classes,
        "row_utility_target": row_utility_target,
        "verifier_utility_target": verifier_utility_target,
        "selection_targets": selection_targets,
        "selection_target_distribution": selection_target_distribution,
        "verifier_target_distribution": verifier_target_distribution,
        "joint_target_distribution": joint_target_distribution,
        "active_mask": active,
        "stats": stats,
    }


class QueryContextReranker(nn.Module):
    """One or more self-attention blocks producing per-query score residuals."""

    def __init__(self, input_dim, hidden_dim=128, num_heads=4,
                 num_layers=1, dropout=0.1, max_delta=0.25):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.max_delta = float(max_delta)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, features, valid_mask):
        hidden = self.input_projection(features)
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
        delta = self.max_delta * torch.tanh(self.score(hidden).squeeze(-1))
        return delta.masked_fill(~valid_mask, 0.0)


class CandidateSetContextEncoder(nn.Module):
    """Contextualize only the fallback gate's small candidate action set."""

    def __init__(self, hidden_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        if (not isinstance(num_layers, int) or isinstance(num_layers, bool)
                or num_layers < 1):
            raise ValueError("candidate context num_layers must be positive")
        if (not isinstance(num_heads, int) or isinstance(num_heads, bool)
                or num_heads < 1 or hidden_dim % num_heads != 0):
            raise ValueError(
                "candidate context heads must divide the hidden dimension"
            )
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0 or float(dropout) >= 1.0):
            raise ValueError(
                "candidate context dropout must be in [0,1)"
            )
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=float(dropout),
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, context_indices, context_valid):
        if (not isinstance(hidden, torch.Tensor) or hidden.dim() != 3
                or not isinstance(context_indices, torch.Tensor)
                or context_indices.dtype != torch.long
                or context_indices.dim() != 2
                or context_indices.shape[0] != hidden.shape[0]
                or context_indices.device != hidden.device
                or not isinstance(context_valid, torch.Tensor)
                or context_valid.dtype != torch.bool
                or context_valid.shape != context_indices.shape
                or context_valid.device != hidden.device):
            raise ValueError("candidate context inputs are invalid")
        if (bool((context_indices < 0).any().item())
                or bool((context_indices >= hidden.shape[1]).any().item())
                or not bool(context_valid[:, 0].all().item())
                or not bool(context_valid.any(dim=1).all().item())):
            raise ValueError("candidate context indices are invalid")

        gather_index = context_indices.unsqueeze(-1).expand(
            -1, -1, hidden.shape[-1]
        )
        base = torch.gather(hidden, 1, gather_index)
        contextual = base.masked_fill(~context_valid.unsqueeze(-1), 0.0)
        for attention, attention_norm, ffn, ffn_norm in zip(
                self.attention, self.attention_norm,
                self.ffn, self.ffn_norm):
            attended, _ = attention(
                contextual, contextual, contextual,
                key_padding_mask=~context_valid,
                need_weights=False,
            )
            contextual = attention_norm(contextual + attended)
            contextual = ffn_norm(contextual + ffn(contextual))

        alternative_indices = context_indices[:, 1:]
        alternative_valid = context_valid[:, 1:]
        alternative_delta = (
            contextual[:, 1:] - base[:, 1:]
        ) * alternative_valid.unsqueeze(-1).to(dtype=hidden.dtype)
        full_delta = torch.zeros_like(hidden)
        full_delta.scatter_add_(
            1,
            alternative_indices.unsqueeze(-1).expand_as(alternative_delta),
            alternative_delta,
        )
        scale = torch.tanh(self.residual_scale)
        return hidden + scale * full_delta, scale


class FallbackTokenSetActionHead(nn.Module):
    """Score a compact permutation-equivariant fallback-plus-candidate set."""

    def __init__(self, input_dim, hidden_dim, max_candidates, num_heads=4,
                 dropout=0.1):
        super().__init__()
        if (not isinstance(input_dim, int) or isinstance(input_dim, bool)
                or input_dim < 1):
            raise ValueError("fallback-set input_dim must be positive")
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool)
                or hidden_dim < 1):
            raise ValueError("fallback-set hidden_dim must be positive")
        if (not isinstance(max_candidates, int)
                or isinstance(max_candidates, bool) or max_candidates < 1):
            raise ValueError("fallback-set max_candidates must be positive")
        if (not isinstance(num_heads, int) or isinstance(num_heads, bool)
                or num_heads < 1 or hidden_dim % num_heads != 0):
            raise ValueError(
                "fallback-set heads must divide the hidden dimension"
            )
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0 or float(dropout) >= 1.0):
            raise ValueError("fallback-set dropout must be in [0,1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=float(dropout), batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, action_features, fallback_indices, candidate_mask):
        if (not isinstance(action_features, torch.Tensor)
                or action_features.dim() != 3
                or action_features.shape[-1] != self.input_dim
                or not bool(torch.isfinite(action_features).all().item())):
            raise ValueError(
                "fallback-set action_features must be finite [B,Q,D]"
            )
        batch_size, num_queries, _ = action_features.shape
        if (not isinstance(fallback_indices, torch.Tensor)
                or fallback_indices.dtype != torch.long
                or fallback_indices.shape != (batch_size,)
                or fallback_indices.device != action_features.device):
            raise ValueError(
                "fallback-set fallback_indices must be int64 [B]"
            )
        if (not isinstance(candidate_mask, torch.Tensor)
                or candidate_mask.dtype != torch.bool
                or candidate_mask.shape != (batch_size, num_queries)
                or candidate_mask.device != action_features.device):
            raise ValueError("fallback-set candidate_mask must be bool [B,Q]")
        row_index = torch.arange(batch_size, device=action_features.device)
        if (bool((fallback_indices < 0).any().item())
                or bool((fallback_indices >= num_queries).any().item())
                or bool(candidate_mask[row_index, fallback_indices].any().item())):
            raise ValueError(
                "fallback-set fallback must be valid and excluded from candidates"
            )

        slot_count = min(self.max_candidates, num_queries)
        candidate_count = candidate_mask.sum(dim=1)
        if bool((candidate_count > slot_count).any().item()):
            raise ValueError("fallback-set candidate count exceeds its contract")
        query_indices = torch.arange(
            num_queries, device=action_features.device
        ).unsqueeze(0).expand(batch_size, -1)
        packed_candidates = query_indices.masked_fill(
            ~candidate_mask, -1
        ).topk(slot_count, dim=1).values
        candidate_valid = packed_candidates >= 0
        packed_candidates = packed_candidates.clamp(min=0)
        packed_indices = torch.cat((
            fallback_indices.unsqueeze(1), packed_candidates,
        ), dim=1)
        packed_valid = torch.cat((
            torch.ones(
                batch_size, 1, dtype=torch.bool,
                device=action_features.device,
            ),
            candidate_valid,
        ), dim=1)
        gather_index = packed_indices.unsqueeze(-1).expand(
            -1, -1, action_features.shape[-1]
        )
        packed_features = torch.gather(
            action_features, 1, gather_index
        ).masked_fill(~packed_valid.unsqueeze(-1), 0.0)
        hidden = self.input_projection(packed_features)
        attended, _ = self.attention(
            hidden, hidden, hidden,
            key_padding_mask=~packed_valid,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        action_logits = self.score(hidden).squeeze(-1)
        compact_margin = (
            action_logits[:, 1:] - action_logits[:, :1]
        ).masked_fill(~candidate_valid, 0.0)
        margin = action_features.new_zeros(batch_size, num_queries)
        margin.scatter_add_(1, packed_candidates, compact_margin)
        return {
            "margin": margin,
            "fallback_logit": action_logits[:, 0],
        }


class RichFallbackTokenSetActionHead(nn.Module):
    """Fuse deployable rich evidence into a fallback-token set scorer."""

    def __init__(self, action_dim, rich_dim, hidden_dim, max_candidates,
                 num_heads=4, dropout=0.1):
        super().__init__()
        if (not isinstance(rich_dim, int) or isinstance(rich_dim, bool)
                or rich_dim < 1):
            raise ValueError("rich fallback-set feature dimension must be positive")
        self.action_dim = int(action_dim)
        self.rich_dim = int(rich_dim)
        self.rich_norm = nn.LayerNorm(self.rich_dim)
        self.set_head = FallbackTokenSetActionHead(
            input_dim=self.action_dim + self.rich_dim,
            hidden_dim=hidden_dim,
            max_candidates=max_candidates,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, action_features, rich_features, fallback_indices,
                candidate_mask):
        if (not isinstance(action_features, torch.Tensor)
                or action_features.dim() != 3
                or action_features.shape[-1] != self.action_dim):
            raise ValueError("rich fallback-set action features are invalid")
        if (not isinstance(rich_features, torch.Tensor)
                or rich_features.shape != action_features.shape[:2] + (
                    self.rich_dim,
                )
                or rich_features.device != action_features.device
                or not bool(torch.isfinite(rich_features).all().item())):
            raise ValueError(
                "rich fallback-set evidence must be finite [B,Q,D]"
            )
        fused = torch.cat((
            action_features,
            self.rich_norm(rich_features.to(dtype=action_features.dtype)),
        ), dim=-1)
        return self.set_head(fused, fallback_indices, candidate_mask)


class AdaptiveSourceMixer(nn.Module):
    """Learn query-wise source routing as a zero-residual V19 extension."""

    def __init__(self, context_dim, rich_dim, hidden_dim, source_count,
                 shared_index):
        super().__init__()
        for name, value in (
                ("context_dim", context_dim), ("rich_dim", rich_dim),
                ("hidden_dim", hidden_dim), ("source_count", source_count)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError("adaptive mixer {} must be positive".format(name))
        if (not isinstance(shared_index, int) or isinstance(shared_index, bool)
                or shared_index < 0 or shared_index >= source_count):
            raise ValueError("adaptive mixer shared_index is invalid")
        self.context_dim = int(context_dim)
        self.rich_dim = int(rich_dim)
        self.hidden_dim = int(hidden_dim)
        self.source_count = int(source_count)
        self.shared_index = int(shared_index)
        self.routed_indices = tuple(
            index for index in range(self.source_count)
            if index != self.shared_index
        )
        self.feature_dim = 4

        self.context_norm = nn.LayerNorm(self.context_dim)
        self.rich_norm = nn.LayerNorm(self.rich_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.context_dim + self.rich_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.source_router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.mix_residual = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.source_router[-1].weight)
        nn.init.zeros_(self.mix_residual[-1].weight)
        nn.init.zeros_(self.mix_residual[-1].bias)

    def forward(self, context_features, rich_features, source_ranks,
                source_validity, base_router_logits, base_scale, top_k):
        if (not isinstance(context_features, torch.Tensor)
                or context_features.dim() != 3
                or context_features.shape[-1] != self.context_dim
                or not bool(torch.isfinite(context_features).all().item())):
            raise ValueError(
                "adaptive mixer context must be finite [B,Q,D]"
            )
        batch_size, num_queries, _ = context_features.shape
        if (not isinstance(rich_features, torch.Tensor)
                or rich_features.shape != (
                    batch_size, num_queries, self.rich_dim
                )
                or rich_features.device != context_features.device
                or not bool(torch.isfinite(rich_features).all().item())):
            raise ValueError("adaptive mixer rich features are invalid")
        expected_source_shape = (
            batch_size, num_queries, self.source_count
        )
        if (not isinstance(source_ranks, torch.Tensor)
                or source_ranks.shape != expected_source_shape
                or source_ranks.device != context_features.device
                or not bool(torch.isfinite(source_ranks).all().item())):
            raise ValueError("adaptive mixer source ranks are invalid")
        if (not isinstance(source_validity, torch.Tensor)
                or source_validity.dtype != torch.bool
                or source_validity.shape != expected_source_shape
                or source_validity.device != context_features.device):
            raise ValueError(
                "adaptive mixer source_validity must be bool [B,Q,S]"
            )
        routed_count = len(self.routed_indices)
        if (not isinstance(base_router_logits, torch.Tensor)
                or base_router_logits.shape != (
                    batch_size, num_queries, routed_count
                )
                or base_router_logits.device != context_features.device
                or not bool(torch.isfinite(base_router_logits).all().item())):
            raise ValueError("adaptive mixer base router logits are invalid")
        if (not isinstance(base_scale, torch.Tensor)
                or base_scale.numel() != 1
                or base_scale.device != context_features.device
                or not bool(torch.isfinite(base_scale).all().item())):
            raise ValueError("adaptive mixer base scale is invalid")
        if (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
            raise ValueError("adaptive mixer top_k must be positive")

        query_hidden = self.query_encoder(torch.cat((
            self.context_norm(context_features),
            self.rich_norm(rich_features),
        ), dim=-1))
        shared_flag = source_ranks.new_zeros(self.source_count)
        shared_flag[self.shared_index] = 1.0
        source_features = torch.cat((
            query_hidden.unsqueeze(2).expand(-1, -1, self.source_count, -1),
            source_ranks.unsqueeze(-1),
            shared_flag.view(1, 1, self.source_count, 1).expand(
                batch_size, num_queries, -1, -1
            ),
        ), dim=-1)
        source_hidden = self.source_encoder(source_features)
        source_residual = self.source_router(source_hidden).squeeze(-1)
        routed_index = torch.tensor(
            self.routed_indices, dtype=torch.long,
            device=context_features.device,
        )
        routed_residual = source_residual.index_select(2, routed_index)
        routed_validity = source_validity.index_select(2, routed_index)
        routed_logits = base_router_logits.detach() + routed_residual
        routed_probabilities = F.softmax(
            routed_logits.masked_fill(~routed_validity, -1e4), dim=-1
        )
        routed_probabilities = (
            routed_probabilities
            * routed_validity.to(dtype=routed_probabilities.dtype)
        )
        routed_probabilities = routed_probabilities / routed_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        active_top_k = min(int(top_k), max(routed_count, 1))
        top_indices = routed_probabilities.topk(
            active_top_k, dim=-1
        ).indices
        hard_mask = torch.zeros_like(routed_probabilities)
        hard_mask.scatter_(-1, top_indices, 1.0)
        hard_mask = hard_mask * routed_validity.to(dtype=hard_mask.dtype)
        sparse_gate = routed_probabilities * hard_mask
        sparse_gate = sparse_gate / sparse_gate.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        gate = (
            sparse_gate.detach()
            + routed_probabilities - routed_probabilities.detach()
        )

        routed_ranks = source_ranks.index_select(2, routed_index)
        routed_score = (gate * routed_ranks).sum(dim=-1)
        shared_score = source_ranks[..., self.shared_index]
        has_routed = routed_validity.any(dim=-1)
        routed_score = torch.where(has_routed, routed_score, shared_score)

        routed_source_hidden = source_hidden.index_select(2, routed_index)
        routed_weight = routed_validity.to(
            dtype=routed_source_hidden.dtype
        ).unsqueeze(-1)
        routed_pool = (
            (routed_source_hidden * routed_weight).sum(dim=2)
            / routed_weight.sum(dim=2).clamp(min=1.0)
        )
        mix_delta = self.mix_residual(torch.cat((
            query_hidden, routed_pool,
        ), dim=-1)).squeeze(-1)
        frozen_scale = base_scale.detach().reshape(()).to(
            dtype=mix_delta.dtype
        )
        scale_room = torch.where(
            mix_delta >= 0.0,
            1.0 - frozen_scale,
            1.0 + frozen_scale,
        ).clamp(min=0.0)
        adaptive_scale = frozen_scale + scale_room * torch.tanh(mix_delta)
        adaptive_scale = torch.where(
            has_routed, adaptive_scale, torch.zeros_like(adaptive_scale)
        )
        fused_score = shared_score + adaptive_scale * (
            routed_score - shared_score
        )
        entropy = -(
            routed_probabilities.clamp(min=1e-8).log()
            * routed_probabilities
        ).sum(dim=-1)
        valid_fraction = routed_validity.to(dtype=fused_score.dtype).mean(
            dim=-1
        )
        evidence = torch.stack((
            adaptive_scale,
            fused_score - shared_score,
            entropy,
            valid_fraction,
        ), dim=-1)
        return {
            "fused_score": fused_score,
            "router_probs": routed_probabilities,
            "expert_mask": hard_mask,
            "mix_scale": adaptive_scale,
            "features": evidence,
            "has_routed": has_routed,
        }


class DenseQualityFallbackSetActionHead(nn.Module):
    """Predict dense token quality and derive a monotonic fallback margin."""

    def __init__(self, action_dim, rich_dim, adaptive_dim, hidden_dim,
                 max_candidates, threshold_count=2, mask_utility_weight=0.25,
                 num_heads=4, dropout=0.1):
        super().__init__()
        for name, value in (
                ("action_dim", action_dim), ("rich_dim", rich_dim),
                ("adaptive_dim", adaptive_dim), ("hidden_dim", hidden_dim),
                ("max_candidates", max_candidates),
                ("threshold_count", threshold_count)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(
                    "dense-quality {} must be positive".format(name)
                )
        if hidden_dim % num_heads != 0:
            raise ValueError("dense-quality heads must divide hidden_dim")
        if (not isinstance(mask_utility_weight, (float, int))
                or isinstance(mask_utility_weight, bool)
                or not math.isfinite(float(mask_utility_weight))
                or float(mask_utility_weight) < 0.0):
            raise ValueError(
                "dense-quality mask utility weight must be non-negative"
            )
        self.action_dim = int(action_dim)
        self.rich_dim = int(rich_dim)
        self.adaptive_dim = int(adaptive_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.threshold_count = int(threshold_count)
        self.mask_utility_weight = float(mask_utility_weight)
        self.rich_norm = nn.LayerNorm(self.rich_dim)
        self.adaptive_norm = nn.LayerNorm(self.adaptive_dim)
        input_dim = self.action_dim + self.rich_dim + self.adaptive_dim
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=float(dropout),
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.quality_head = nn.Linear(
            self.hidden_dim, 2 * self.threshold_count + 2
        )
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
    def forward(self, action_features, rich_features, adaptive_features,
                fallback_indices, candidate_mask):
        if (not isinstance(action_features, torch.Tensor)
                or action_features.dim() != 3
                or action_features.shape[-1] != self.action_dim
                or not bool(torch.isfinite(action_features).all().item())):
            raise ValueError("dense-quality action features are invalid")
        batch_size, num_queries, _ = action_features.shape
        if (not isinstance(rich_features, torch.Tensor)
                or rich_features.shape != (
                    batch_size, num_queries, self.rich_dim
                )
                or rich_features.device != action_features.device
                or not bool(torch.isfinite(rich_features).all().item())):
            raise ValueError("dense-quality rich features are invalid")
        if (not isinstance(adaptive_features, torch.Tensor)
                or adaptive_features.shape != (
                    batch_size, num_queries, self.adaptive_dim
                )
                or adaptive_features.device != action_features.device
                or not bool(torch.isfinite(adaptive_features).all().item())):
            raise ValueError("dense-quality adaptive features are invalid")
        if (not isinstance(fallback_indices, torch.Tensor)
                or fallback_indices.dtype != torch.long
                or fallback_indices.shape != (batch_size,)
                or fallback_indices.device != action_features.device):
            raise ValueError("dense-quality fallback indices are invalid")
        if (not isinstance(candidate_mask, torch.Tensor)
                or candidate_mask.dtype != torch.bool
                or candidate_mask.shape != (batch_size, num_queries)
                or candidate_mask.device != action_features.device):
            raise ValueError("dense-quality candidate mask is invalid")
        row_index = torch.arange(batch_size, device=action_features.device)
        if (bool((fallback_indices < 0).any().item())
                or bool((fallback_indices >= num_queries).any().item())
                or bool(candidate_mask[
                    row_index, fallback_indices
                ].any().item())):
            raise ValueError("dense-quality fallback must be excluded")

        slot_count = min(self.max_candidates, num_queries)
        if bool((candidate_mask.sum(dim=1) > slot_count).any().item()):
            raise ValueError("dense-quality candidate count exceeds contract")
        query_indices = torch.arange(
            num_queries, device=action_features.device
        ).unsqueeze(0).expand(batch_size, -1)
        packed_candidates = query_indices.masked_fill(
            ~candidate_mask, -1
        ).topk(slot_count, dim=1).values
        candidate_valid = packed_candidates >= 0
        packed_candidates = packed_candidates.clamp(min=0)
        packed_indices = torch.cat((
            fallback_indices.unsqueeze(1), packed_candidates,
        ), dim=1)
        packed_valid = torch.cat((
            torch.ones(
                batch_size, 1, dtype=torch.bool,
                device=action_features.device,
            ),
            candidate_valid,
        ), dim=1)
        fused = torch.cat((
            action_features,
            self.rich_norm(rich_features.to(dtype=action_features.dtype)),
            self.adaptive_norm(
                adaptive_features.to(dtype=action_features.dtype)
            ),
        ), dim=-1)
        gather_index = packed_indices.unsqueeze(-1).expand(
            -1, -1, fused.shape[-1]
        )
        packed_features = torch.gather(
            fused, 1, gather_index
        ).masked_fill(~packed_valid.unsqueeze(-1), 0.0)
        hidden = self.input_projection(packed_features)
        attended, _ = self.attention(
            hidden, hidden, hidden,
            key_padding_mask=~packed_valid,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        packed_prediction = self.quality_head(hidden).masked_fill(
            ~packed_valid.unsqueeze(-1), 0.0
        )

        threshold_count = self.threshold_count
        box_logits = packed_prediction[..., :threshold_count]
        box_iou = packed_prediction[..., threshold_count].sigmoid()
        mask_start = threshold_count + 1
        mask_logits = packed_prediction[
            ..., mask_start:mask_start + threshold_count
        ]
        mask_iou = packed_prediction[..., -1].sigmoid()
        token_quality = dense_quality_expected_score(
            box_threshold_logits=box_logits,
            box_iou=box_iou,
            mask_threshold_logits=mask_logits,
            mask_iou=mask_iou,
            mask_utility_weight=self.mask_utility_weight,
        )
        token_uncertainty = dense_quality_prediction_uncertainty(
            box_threshold_logits=box_logits,
            box_iou=box_iou,
            mask_threshold_logits=mask_logits,
            mask_iou=mask_iou,
            mask_utility_weight=self.mask_utility_weight,
        )
        compact_margin = (
            token_quality[:, 1:] - token_quality[:, :1]
        ).masked_fill(~candidate_valid, 0.0)

        margin = action_features.new_zeros(batch_size, num_queries)
        margin.scatter_add_(1, packed_candidates, compact_margin)
        full_prediction = action_features.new_zeros(
            batch_size, num_queries, packed_prediction.shape[-1]
        )
        full_prediction.scatter_add_(
            1,
            packed_indices.unsqueeze(-1).expand_as(packed_prediction),
            packed_prediction,
        )
        full_box_logits = full_prediction[..., :threshold_count]
        full_box_iou = full_prediction[..., threshold_count].sigmoid()
        full_mask_logits = full_prediction[
            ..., mask_start:mask_start + threshold_count
        ]
        full_mask_iou = full_prediction[..., -1].sigmoid()
        full_evidence = torch.cat((
            full_box_logits.sigmoid(),
            full_box_iou.unsqueeze(-1),
            full_mask_logits.sigmoid(),
            full_mask_iou.unsqueeze(-1),
        ), dim=-1)
        full_quality = action_features.new_zeros(batch_size, num_queries)
        full_quality.scatter_add_(
            1, packed_indices,
            token_quality * packed_valid.to(dtype=token_quality.dtype),
        )
        full_uncertainty = action_features.new_zeros(batch_size, num_queries)
        full_uncertainty.scatter_add_(
            1, packed_indices,
            token_uncertainty * packed_valid.to(
                dtype=token_uncertainty.dtype
            ),
        )
        quality_mask = candidate_mask.clone()
        quality_mask[row_index, fallback_indices] = True
        return {
            "margin": margin,
            "fallback_logit": margin.new_zeros(batch_size),
            "box_threshold_logits": full_box_logits,
            "box_iou": full_box_iou,
            "mask_threshold_logits": full_mask_logits,
            "mask_iou": full_mask_iou,
            "evidence": full_evidence,
            "quality": full_quality,
            "uncertainty": full_uncertainty,
            "quality_mask": quality_mask,
        }


class RelativeRiskFallbackSetActionHead(nn.Module):
    """Predict a permutation-equivariant candidate-vs-fallback risk margin.

    The fallback token is packed into the same attention set as candidates and
    receives a fixed reference logit.  The final risk projection is zero
    initialized, so a migrated V19 checkpoint keeps its exact deployment
    behavior until the V24 head learns a relative correction.
    """

    def __init__(self, action_dim, rich_dim, adaptive_dim, evidence_dim,
                 hidden_dim, max_candidates, num_heads=4, dropout=0.1):
        super().__init__()
        for name, value in (
                ("action_dim", action_dim), ("rich_dim", rich_dim),
                ("adaptive_dim", adaptive_dim), ("evidence_dim", evidence_dim),
                ("hidden_dim", hidden_dim), ("max_candidates", max_candidates)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(
                    "relative-risk {} must be positive".format(name)
                )
        if hidden_dim % num_heads != 0:
            raise ValueError("relative-risk heads must divide hidden_dim")
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0 or float(dropout) >= 1.0):
            raise ValueError("relative-risk dropout must be in [0,1)")
        self.action_dim = int(action_dim)
        self.rich_dim = int(rich_dim)
        self.adaptive_dim = int(adaptive_dim)
        self.evidence_dim = int(evidence_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.rich_norm = nn.LayerNorm(self.rich_dim)
        self.adaptive_norm = nn.LayerNorm(self.adaptive_dim)
        self.evidence_norm = nn.LayerNorm(self.evidence_dim)
        input_dim = (
            self.action_dim + self.rich_dim + self.adaptive_dim
            + self.evidence_dim
        )
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=float(dropout),
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.risk_head = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.risk_head.weight)

    def forward(self, action_features, rich_features, adaptive_features,
                dense_quality_evidence, fallback_indices, candidate_mask):
        if (not isinstance(action_features, torch.Tensor)
                or action_features.dim() != 3
                or action_features.shape[-1] != self.action_dim
                or not bool(torch.isfinite(action_features).all().item())):
            raise ValueError("relative-risk action features are invalid")
        batch_size, num_queries, _ = action_features.shape
        expected = (batch_size, num_queries)
        for name, values, width in (
                ("rich_features", rich_features, self.rich_dim),
                ("adaptive_features", adaptive_features, self.adaptive_dim),
                ("dense_quality_evidence", dense_quality_evidence,
                 self.evidence_dim)):
            if (not isinstance(values, torch.Tensor)
                    or values.shape != expected + (width,)
                    or values.device != action_features.device
                    or not bool(torch.isfinite(values).all().item())):
                raise ValueError(
                    "relative-risk {} are invalid".format(name)
                )
        if (not isinstance(fallback_indices, torch.Tensor)
                or fallback_indices.dtype != torch.long
                or fallback_indices.shape != (batch_size,)
                or fallback_indices.device != action_features.device):
            raise ValueError("relative-risk fallback indices are invalid")
        if (not isinstance(candidate_mask, torch.Tensor)
                or candidate_mask.dtype != torch.bool
                or candidate_mask.shape != expected
                or candidate_mask.device != action_features.device):
            raise ValueError("relative-risk candidate mask is invalid")
        row_index = torch.arange(batch_size, device=action_features.device)
        if (bool((fallback_indices < 0).any().item())
                or bool((fallback_indices >= num_queries).any().item())
                or bool(candidate_mask[row_index, fallback_indices].any().item())):
            raise ValueError("relative-risk fallback must be excluded")

        slot_count = min(self.max_candidates, num_queries)
        if bool((candidate_mask.sum(dim=1) > slot_count).any().item()):
            raise ValueError("relative-risk candidate count exceeds contract")
        query_indices = torch.arange(
            num_queries, device=action_features.device
        ).unsqueeze(0).expand(batch_size, -1)
        packed_candidates = query_indices.masked_fill(
            ~candidate_mask, -1
        ).topk(slot_count, dim=1).values
        candidate_valid = packed_candidates >= 0
        packed_candidates = packed_candidates.clamp(min=0)
        packed_indices = torch.cat((
            fallback_indices.unsqueeze(1), packed_candidates,
        ), dim=1)
        packed_valid = torch.cat((
            torch.ones(
                batch_size, 1, dtype=torch.bool,
                device=action_features.device,
            ),
            candidate_valid,
        ), dim=1)
        fused = torch.cat((
            action_features,
            self.rich_norm(rich_features.to(dtype=action_features.dtype)),
            self.adaptive_norm(
                adaptive_features.to(dtype=action_features.dtype)
            ),
            self.evidence_norm(
                dense_quality_evidence.to(dtype=action_features.dtype)
            ),
        ), dim=-1)
        gather_index = packed_indices.unsqueeze(-1).expand(
            -1, -1, fused.shape[-1]
        )
        packed_features = torch.gather(
            fused, 1, gather_index
        ).masked_fill(~packed_valid.unsqueeze(-1), 0.0)
        hidden = self.input_projection(packed_features)
        attended, _ = self.attention(
            hidden, hidden, hidden,
            key_padding_mask=~packed_valid,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        packed_logits = self.risk_head(hidden).squeeze(-1).masked_fill(
            ~packed_valid, 0.0
        )
        compact_margin = (
            packed_logits[:, 1:] - packed_logits[:, :1]
        ).masked_fill(~candidate_valid, 0.0)
        margin = action_features.new_zeros(batch_size, num_queries)
        margin.scatter_add_(1, packed_candidates, compact_margin)
        full_logits = action_features.new_zeros(batch_size, num_queries)
        full_logits.scatter_add_(1, packed_indices, packed_logits)
        return {
            "margin": margin,
            "fallback_logit": packed_logits[:, 0],
            "logits": full_logits,
        }


class CalibratedPairwiseRiskSetActionHead(nn.Module):
    """Regress explicit candidate-vs-fallback utility for set actions.

    A permutation-equivariant set encoder first contextualizes the deployed
    V19 fallback and all candidate tokens.  Each candidate is then compared
    to the fallback with role-aware pair features and dense-quality evidence.
    The utility and auxiliary benefit projections are zero initialized, while
    retaining learnable biases that can represent the empirical switch prior.
    """

    def __init__(self, action_dim, rich_dim, adaptive_dim, evidence_dim,
                 hidden_dim, max_candidates, num_heads=4, dropout=0.1):
        super().__init__()
        for name, value in (
                ("action_dim", action_dim), ("rich_dim", rich_dim),
                ("adaptive_dim", adaptive_dim), ("evidence_dim", evidence_dim),
                ("hidden_dim", hidden_dim), ("max_candidates", max_candidates)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(
                    "pairwise calibrated {} must be positive".format(name)
                )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "pairwise calibrated heads must divide hidden_dim"
            )
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0 or float(dropout) >= 1.0):
            raise ValueError(
                "pairwise calibrated dropout must be in [0,1)"
            )
        self.action_dim = int(action_dim)
        self.rich_dim = int(rich_dim)
        self.adaptive_dim = int(adaptive_dim)
        self.evidence_dim = int(evidence_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.rich_norm = nn.LayerNorm(self.rich_dim)
        self.adaptive_norm = nn.LayerNorm(self.adaptive_dim)
        self.evidence_norm = nn.LayerNorm(self.evidence_dim)
        input_dim = (
            self.action_dim + self.rich_dim + self.adaptive_dim
            + self.evidence_dim
        )
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=float(dropout),
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.evidence_delta_norm = nn.LayerNorm(self.evidence_dim)
        pair_dim = 4 * self.hidden_dim + self.evidence_dim
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.utility_head = nn.Linear(self.hidden_dim, 1, bias=True)
        self.benefit_head = nn.Linear(self.hidden_dim, 1, bias=True)
        for head in (self.utility_head, self.benefit_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, action_features, rich_features, adaptive_features,
                dense_quality_evidence, fallback_indices, candidate_mask):
        if (not isinstance(action_features, torch.Tensor)
                or action_features.dim() != 3
                or action_features.shape[-1] != self.action_dim
                or not bool(torch.isfinite(action_features).all().item())):
            raise ValueError(
                "pairwise calibrated action features are invalid"
            )
        batch_size, num_queries, _ = action_features.shape
        expected = (batch_size, num_queries)
        for name, values, width in (
                ("rich_features", rich_features, self.rich_dim),
                ("adaptive_features", adaptive_features, self.adaptive_dim),
                ("dense_quality_evidence", dense_quality_evidence,
                 self.evidence_dim)):
            if (not isinstance(values, torch.Tensor)
                    or values.shape != expected + (width,)
                    or values.device != action_features.device
                    or not bool(torch.isfinite(values).all().item())):
                raise ValueError(
                    "pairwise calibrated {} are invalid".format(name)
                )
        if (not isinstance(fallback_indices, torch.Tensor)
                or fallback_indices.dtype != torch.long
                or fallback_indices.shape != (batch_size,)
                or fallback_indices.device != action_features.device):
            raise ValueError(
                "pairwise calibrated fallback indices are invalid"
            )
        if (not isinstance(candidate_mask, torch.Tensor)
                or candidate_mask.dtype != torch.bool
                or candidate_mask.shape != expected
                or candidate_mask.device != action_features.device):
            raise ValueError(
                "pairwise calibrated candidate mask is invalid"
            )
        row_index = torch.arange(batch_size, device=action_features.device)
        if (bool((fallback_indices < 0).any().item())
                or bool((fallback_indices >= num_queries).any().item())
                or bool(candidate_mask[
                    row_index, fallback_indices
                ].any().item())):
            raise ValueError(
                "pairwise calibrated fallback must be excluded"
            )

        slot_count = min(self.max_candidates, num_queries)
        if bool((candidate_mask.sum(dim=1) > slot_count).any().item()):
            raise ValueError(
                "pairwise calibrated candidate count exceeds contract"
            )
        query_indices = torch.arange(
            num_queries, device=action_features.device
        ).unsqueeze(0).expand(batch_size, -1)
        packed_candidates = query_indices.masked_fill(
            ~candidate_mask, -1
        ).topk(slot_count, dim=1).values
        candidate_valid = packed_candidates >= 0
        packed_candidates = packed_candidates.clamp(min=0)
        packed_indices = torch.cat((
            fallback_indices.unsqueeze(1), packed_candidates,
        ), dim=1)
        packed_valid = torch.cat((
            torch.ones(
                batch_size, 1, dtype=torch.bool,
                device=action_features.device,
            ),
            candidate_valid,
        ), dim=1)
        normalized_evidence = self.evidence_norm(
            dense_quality_evidence.to(dtype=action_features.dtype)
        )
        fused = torch.cat((
            action_features,
            self.rich_norm(rich_features.to(dtype=action_features.dtype)),
            self.adaptive_norm(
                adaptive_features.to(dtype=action_features.dtype)
            ),
            normalized_evidence,
        ), dim=-1)
        gather_index = packed_indices.unsqueeze(-1).expand(
            -1, -1, fused.shape[-1]
        )
        packed_features = torch.gather(
            fused, 1, gather_index
        ).masked_fill(~packed_valid.unsqueeze(-1), 0.0)
        packed_evidence = torch.gather(
            normalized_evidence,
            1,
            packed_indices.unsqueeze(-1).expand(
                -1, -1, self.evidence_dim
            ),
        ).masked_fill(~packed_valid.unsqueeze(-1), 0.0)

        hidden = self.input_projection(packed_features)
        attended, _ = self.attention(
            hidden, hidden, hidden,
            key_padding_mask=~packed_valid,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        fallback_hidden = hidden[:, :1].expand(
            -1, slot_count, -1
        )
        candidate_hidden = hidden[:, 1:]
        fallback_evidence = packed_evidence[:, :1].expand(
            -1, slot_count, -1
        )
        evidence_delta = self.evidence_delta_norm(
            packed_evidence[:, 1:] - fallback_evidence
        )
        pair_hidden = self.pair_projection(torch.cat((
            candidate_hidden,
            fallback_hidden,
            candidate_hidden - fallback_hidden,
            candidate_hidden * fallback_hidden,
            evidence_delta,
        ), dim=-1))
        compact_utility = self.utility_head(
            pair_hidden
        ).squeeze(-1).masked_fill(~candidate_valid, 0.0)
        compact_benefit = self.benefit_head(
            pair_hidden
        ).squeeze(-1).masked_fill(~candidate_valid, 0.0)

        utility_margin = action_features.new_zeros(batch_size, num_queries)
        utility_margin.scatter_add_(
            1, packed_candidates, compact_utility
        )
        benefit_margin = action_features.new_zeros(batch_size, num_queries)
        benefit_margin.scatter_add_(
            1, packed_candidates, compact_benefit
        )
        full_pair_features = action_features.new_zeros(
            batch_size, num_queries, self.hidden_dim
        )
        full_pair_features.scatter_add_(
            1,
            packed_candidates.unsqueeze(-1).expand(
                -1, -1, self.hidden_dim
            ),
            pair_hidden * candidate_valid.unsqueeze(-1).to(
                dtype=pair_hidden.dtype
            ),
        )
        return {
            "margin": utility_margin,
            "benefit_margin": benefit_margin,
            "fallback_logit": utility_margin.new_zeros(batch_size),
            "pair_features": full_pair_features,
        }


class SelectedCandidateAbstentionHead(nn.Module):
    """Calibrate one row action after candidate ranking.

    Candidate ranking and fallback abstention are deliberately separate.  The
    returned candidate margins retain the ranking order, while their row-wise
    maximum is exactly the learned abstention risk.  A zero-initialized output
    therefore preserves the fallback at migration time without a validation
    threshold.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool)
                or hidden_dim < 1):
            raise ValueError("selected abstention hidden_dim must be positive")
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0 or float(dropout) >= 1.0):
            raise ValueError("selected abstention dropout must be in [0,1)")
        self.hidden_dim = int(hidden_dim)
        self.row_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.row_head[-1].weight)
        nn.init.zeros_(self.row_head[-1].bias)

    def forward(self, pair_features, selection_margin, candidate_mask):
        if (not isinstance(pair_features, torch.Tensor)
                or pair_features.dim() != 3
                or pair_features.shape[-1] != self.hidden_dim
                or not bool(torch.isfinite(pair_features).all().item())):
            raise ValueError(
                "selected abstention pair features must be finite [B,Q,D]"
            )
        batch_size, num_queries, _ = pair_features.shape
        expected = (batch_size, num_queries)
        if (not isinstance(selection_margin, torch.Tensor)
                or selection_margin.shape != expected
                or selection_margin.device != pair_features.device
                or not bool(torch.isfinite(selection_margin).all().item())):
            raise ValueError(
                "selected abstention selection margin is invalid"
            )
        if (not isinstance(candidate_mask, torch.Tensor)
                or candidate_mask.dtype != torch.bool
                or candidate_mask.shape != expected
                or candidate_mask.device != pair_features.device):
            raise ValueError(
                "selected abstention candidate mask must be bool [B,Q]"
            )

        has_candidate = candidate_mask.any(dim=1)
        masked_selection = selection_margin.masked_fill(
            ~candidate_mask, -1e4
        )
        best_selection, selected_indices = masked_selection.max(dim=1)
        row_index = torch.arange(batch_size, device=pair_features.device)
        selected_features = pair_features[row_index, selected_indices]
        candidate_weight = candidate_mask.unsqueeze(-1).to(
            dtype=pair_features.dtype
        )
        candidate_count = candidate_weight.sum(dim=1).clamp(min=1.0)
        mean_features = (
            (pair_features * candidate_weight).sum(dim=1) / candidate_count
        )
        negative_sentinel = torch.finfo(pair_features.dtype).min
        max_features = pair_features.masked_fill(
            ~candidate_mask.unsqueeze(-1), negative_sentinel
        ).max(dim=1).values
        max_features = torch.where(
            has_candidate.unsqueeze(1),
            max_features,
            torch.zeros_like(max_features),
        )
        masked_runner_up = masked_selection.clone()
        masked_runner_up[row_index, selected_indices] = -1e4
        runner_up = masked_runner_up.max(dim=1).values
        runner_up = torch.where(
            candidate_mask.sum(dim=1) > 1,
            runner_up,
            best_selection,
        )
        scalar_features = torch.stack((
            best_selection,
            best_selection - runner_up,
            candidate_mask.to(dtype=pair_features.dtype).mean(dim=1),
            selection_margin.masked_fill(~candidate_mask, 0.0).abs().sum(dim=1)
            / candidate_mask.sum(dim=1).clamp(min=1).to(
                dtype=pair_features.dtype
            ),
        ), dim=1)
        row_features = torch.cat((
            selected_features, mean_features, max_features, scalar_features,
        ), dim=1)
        row_margin = self.row_head(row_features).squeeze(-1)
        row_margin = torch.where(
            has_candidate, row_margin, torch.zeros_like(row_margin)
        )
        # Preserve the selected ordering, but make max(candidate_margin)
        # exactly equal to the independently calibrated row decision.
        candidate_margin = (
            selection_margin - best_selection.unsqueeze(1)
            + row_margin.unsqueeze(1)
        ).masked_fill(~candidate_mask, 0.0)
        return {
            "margin": candidate_margin,
            "row_margin": row_margin,
            "selection_margin": selection_margin,
            "selected_indices": selected_indices,
        }


class CounterfactualSelectedRiskHead(nn.Module):
    """Score every candidate counterfactually before hard top-1 deployment.

    V28 only trained the abstention head on the currently selected query.  A
    wrong early selection therefore produced a negative training target for a
    row that contained a useful candidate.  This head shares one risk
    projection across all valid candidates, allowing the loss to supervise
    each candidate while inference still gathers the policy-selected one.
    """

    def __init__(self, hidden_dim, dropout=0.1, decomposed=False,
                 complementary_log_odds=False, hazard_residual=False):
        super().__init__()
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool)
                or hidden_dim < 1):
            raise ValueError(
                "counterfactual risk hidden_dim must be positive"
            )
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0 or float(dropout) >= 1.0):
            raise ValueError(
                "counterfactual risk dropout must be in [0,1)"
            )
        if not isinstance(decomposed, bool):
            raise ValueError("counterfactual decomposed flag must be boolean")
        if not isinstance(complementary_log_odds, bool):
            raise ValueError(
                "counterfactual complementary flag must be boolean"
            )
        if not isinstance(hazard_residual, bool):
            raise ValueError(
                "counterfactual hazard residual flag must be boolean"
            )
        if complementary_log_odds and not decomposed:
            raise ValueError(
                "counterfactual complementary mode requires decomposition"
            )
        if hazard_residual and not decomposed:
            raise ValueError(
                "counterfactual hazard residual mode requires decomposition"
            )
        if complementary_log_odds and hazard_residual:
            raise ValueError(
                "counterfactual decomposition modes are mutually exclusive"
            )
        self.hidden_dim = int(hidden_dim)
        self.decomposed = decomposed
        self.complementary_log_odds = complementary_log_odds
        self.hazard_residual = hazard_residual
        self.risk_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2 if self.decomposed else 1),
        )
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].bias)

    def forward(self, pair_features, selection_margin, candidate_mask):
        if (not isinstance(pair_features, torch.Tensor)
                or pair_features.dim() != 3
                or pair_features.shape[-1] != self.hidden_dim
                or not bool(torch.isfinite(pair_features).all().item())):
            raise ValueError(
                "counterfactual risk pair features must be finite [B,Q,D]"
            )
        batch_size, num_queries, _ = pair_features.shape
        expected = (batch_size, num_queries)
        if (not isinstance(selection_margin, torch.Tensor)
                or selection_margin.shape != expected
                or selection_margin.device != pair_features.device
                or not bool(torch.isfinite(selection_margin).all().item())):
            raise ValueError(
                "counterfactual risk selection margin is invalid"
            )
        if (not isinstance(candidate_mask, torch.Tensor)
                or candidate_mask.dtype != torch.bool
                or candidate_mask.shape != expected
                or candidate_mask.device != pair_features.device):
            raise ValueError(
                "counterfactual risk candidate mask must be bool [B,Q]"
            )

        has_candidate = candidate_mask.any(dim=1)
        masked_selection = selection_margin.masked_fill(
            ~candidate_mask, -1e4
        )
        best_selection, selected_indices = masked_selection.max(dim=1)
        candidate_weight = candidate_mask.unsqueeze(-1).to(
            dtype=pair_features.dtype
        )
        candidate_count = candidate_weight.sum(dim=1).clamp(min=1.0)
        mean_features = (
            (pair_features * candidate_weight).sum(dim=1)
            / candidate_count
        )
        negative_sentinel = torch.finfo(pair_features.dtype).min
        max_features = pair_features.masked_fill(
            ~candidate_mask.unsqueeze(-1), negative_sentinel
        ).max(dim=1).values
        max_features = torch.where(
            has_candidate.unsqueeze(1),
            max_features,
            torch.zeros_like(max_features),
        )
        mean_abs_selection = (
            selection_margin.masked_fill(~candidate_mask, 0.0).abs().sum(dim=1)
            / candidate_count.squeeze(-1)
        )
        scalar_features = torch.stack((
            selection_margin,
            selection_margin - best_selection.unsqueeze(1),
            candidate_mask.to(dtype=pair_features.dtype).mean(dim=1)
            .unsqueeze(1).expand(-1, num_queries),
            mean_abs_selection.unsqueeze(1).expand(-1, num_queries),
        ), dim=-1)
        row_context = torch.cat((
            pair_features,
            mean_features.unsqueeze(1).expand(-1, num_queries, -1),
            max_features.unsqueeze(1).expand(-1, num_queries, -1),
            scalar_features,
        ), dim=-1)
        risk_output = self.risk_head(row_context)
        if self.decomposed:
            candidate_benefit = risk_output[..., 0]
            candidate_hazard = risk_output[..., 1]
            if self.complementary_log_odds:
                candidate_risk = candidate_benefit - candidate_hazard
            elif self.hazard_residual:
                candidate_risk = (
                    candidate_benefit - F.relu(candidate_hazard)
                )
            else:
                candidate_risk = (
                    F.relu(candidate_benefit) - F.relu(candidate_hazard)
                )
        else:
            candidate_benefit = None
            candidate_hazard = None
            candidate_risk = risk_output.squeeze(-1)
        candidate_risk = candidate_risk.masked_fill(~candidate_mask, 0.0)
        if self.decomposed:
            candidate_benefit = candidate_benefit.masked_fill(
                ~candidate_mask, 0.0
            )
            candidate_hazard = candidate_hazard.masked_fill(
                ~candidate_mask, 0.0
            )
        row_index = torch.arange(batch_size, device=pair_features.device)
        selected_risk = candidate_risk[row_index, selected_indices]
        selected_risk = torch.where(
            has_candidate, selected_risk, torch.zeros_like(selected_risk)
        )
        candidate_margin = (
            selection_margin - best_selection.unsqueeze(1)
            + selected_risk.unsqueeze(1)
        ).masked_fill(~candidate_mask, 0.0)
        result = {
            "margin": candidate_margin,
            "row_margin": selected_risk,
            "selection_margin": selection_margin,
            "selected_indices": selected_indices,
            "candidate_risk": candidate_risk,
        }
        if self.decomposed:
            result["candidate_benefit"] = candidate_benefit
            result["candidate_hazard"] = candidate_hazard
        return result


class QueryFallbackGate(nn.Module):
    """Select a safe query from the MoE top-k with an exact shared fallback."""

    def __init__(self, query_dim, hidden_dim=128, candidate_top_k=8,
                 thresholds=(0.25, 0.50), threshold_weights=(2.0, 1.0),
                 break_cost=2.0, decision_margin=0.0,
                 mask_utility_weight=0.25, context_layers=0,
                 context_heads=4, context_dropout=0.1,
                 action_mode="decision", rich_feature_dim=None,
                 adaptive_source_feature_dim=None,
                 uncertainty_weight=0.0):
        super().__init__()
        if (not isinstance(candidate_top_k, int)
                or isinstance(candidate_top_k, bool)
                or candidate_top_k < 1):
            raise ValueError("candidate_top_k must be a positive integer")
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool)
                or hidden_dim < 1):
            raise ValueError("gate hidden_dim must be a positive integer")
        if (not isinstance(context_layers, int)
                or isinstance(context_layers, bool)
                or context_layers < 0):
            raise ValueError(
                "gate context_layers must be a non-negative integer"
            )
        if (not isinstance(context_heads, int)
                or isinstance(context_heads, bool)
                or context_heads < 1):
            raise ValueError(
                "gate context_heads must be a positive integer"
            )
        if (not isinstance(context_dropout, (float, int))
                or isinstance(context_dropout, bool)
                or not math.isfinite(float(context_dropout))
                or float(context_dropout) < 0.0
                or float(context_dropout) >= 1.0):
            raise ValueError("gate context_dropout must be in [0,1)")
        if context_layers > 0 and hidden_dim % context_heads != 0:
            raise ValueError(
                "gate context_heads must divide the hidden dimension"
            )
        if (not isinstance(thresholds, (tuple, list)) or not thresholds
                or len(thresholds) != len(threshold_weights)):
            raise ValueError("gate thresholds and weights must align")
        for name, value, lower_bound in (
                ("break_cost", break_cost, 1.0),
                ("decision_margin", decision_margin, 0.0),
                ("mask_utility_weight", mask_utility_weight, 0.0)):
            if (not isinstance(value, (float, int))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < lower_bound):
                raise ValueError(
                    "{} must be finite and at least {}".format(
                        name, lower_bound
                    )
                )
        if (not isinstance(uncertainty_weight, (float, int))
                or isinstance(uncertainty_weight, bool)
                or not math.isfinite(float(uncertainty_weight))
                or float(uncertainty_weight) < 0.0):
            raise ValueError(
                "uncertainty_weight must be finite and non-negative"
            )
        if any(not isinstance(value, (float, int))
               or isinstance(value, bool)
               or not math.isfinite(float(value))
               or float(value) <= 0.0
               for value in threshold_weights):
            raise ValueError("gate threshold weights must be finite and positive")
        if action_mode not in (
                "decision", "expected_utility", "direct_utility",
                "hierarchical_utility", "pairwise_verifier",
                "topn_pairwise_verifier",
                "topn_dual_evidence_verifier",
                "topn_absolute_quality_delta",
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            raise ValueError(
                "gate action_mode must be decision, expected_utility, or "
                "direct_utility, hierarchical_utility, pairwise_verifier, "
                "topn_pairwise_verifier, topn_dual_evidence_verifier, or "
                "topn_absolute_quality_delta, or "
                "cascade_absolute_quality_correction, or "
                "cascade_opportunity_quality_correction, or "
                "cascade_opportunity_verified_correction, or "
                "cascade_joint_risk_correction, or "
                "cascade_v19_fallback_set_correction, or "
                "cascade_v19_rich_set_correction, or "
                "cascade_v23_dense_quality_correction, "
                "cascade_v24_relative_risk_correction, or "
                "cascade_v25_pairwise_calibrated_correction, or "
                "cascade_v26_prior_restored_pairwise_correction, or "
                "cascade_v28_selected_abstention_correction, or "
                "cascade_v29_counterfactual_selected_correction, or "
                "cascade_v37_counterfactual_benefit_hazard_correction, or "
                "cascade_v38_complementary_logodds_correction, or "
                "cascade_v39_hazard_residual_correction"
            )
        rich_action = action_mode in (
            "cascade_v19_rich_set_correction",
            "cascade_v23_dense_quality_correction",
            "cascade_v24_relative_risk_correction",
            "cascade_v25_pairwise_calibrated_correction",
            "cascade_v26_prior_restored_pairwise_correction",
            "cascade_v28_selected_abstention_correction",
            "cascade_v29_counterfactual_selected_correction",
            "cascade_v37_counterfactual_benefit_hazard_correction",
            "cascade_v38_complementary_logodds_correction",
            "cascade_v39_hazard_residual_correction",
        )
        if rich_action and (
                not isinstance(rich_feature_dim, int)
                or isinstance(rich_feature_dim, bool)
                or rich_feature_dim < 1):
            raise ValueError(
                "rich_feature_dim must be positive for the rich set action"
            )
        dense_quality_action = (
            action_mode in (
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction",
            )
        )
        if dense_quality_action and (
                not isinstance(adaptive_source_feature_dim, int)
                or isinstance(adaptive_source_feature_dim, bool)
                or adaptive_source_feature_dim < 1):
            raise ValueError(
                "adaptive_source_feature_dim must be positive for dense quality"
            )

        self.query_dim = int(query_dim)
        self.candidate_top_k = int(candidate_top_k)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.break_cost = float(break_cost)
        self.decision_margin = float(decision_margin)
        self.mask_utility_weight = float(mask_utility_weight)
        self.context_layers = int(context_layers)
        self.context_heads = int(context_heads)
        self.context_dropout = float(context_dropout)
        self.action_mode = str(action_mode)
        self.uncertainty_weight = float(uncertainty_weight)
        self.rich_feature_dim = (
            int(rich_feature_dim) if rich_action else None
        )
        self.adaptive_source_feature_dim = (
            int(adaptive_source_feature_dim) if dense_quality_action else None
        )
        self.register_buffer(
            "threshold_weights",
            torch.tensor(tuple(float(value) for value in threshold_weights)),
        )
        pair_dim = 3 * self.query_dim + 6
        self.encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_encoder = None
        if self.context_layers > 0:
            self.context_encoder = CandidateSetContextEncoder(
                hidden_dim=hidden_dim,
                num_heads=context_heads,
                num_layers=self.context_layers,
                dropout=context_dropout,
            )
        output_dim = len(self.thresholds) * 3
        self.box_head = nn.Linear(hidden_dim, output_dim)
        self.mask_head = nn.Linear(hidden_dim, output_dim)
        self.decision_head = nn.Linear(hidden_dim, 3)
        self.utility_head = nn.Linear(hidden_dim, 1)
        for head in (
                self.box_head, self.mask_head, self.decision_head,
                self.utility_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.row_switch_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.row_switch_head[-1].weight)
        nn.init.zeros_(self.row_switch_head[-1].bias)
        self.pairwise_switch_head = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.pairwise_switch_head[-1].weight)
        nn.init.zeros_(self.pairwise_switch_head[-1].bias)
        risk_evidence_dim = 2 * output_dim + 5
        self.safety_switch_head = nn.Sequential(
            nn.Linear(hidden_dim + risk_evidence_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.safety_switch_head[-1].weight)
        nn.init.zeros_(self.safety_switch_head[-1].bias)
        self.absolute_quality_head = None
        if self.action_mode == "topn_absolute_quality_delta":
            absolute_output_dim = 2 * len(self.thresholds) + 2
            self.absolute_quality_head = nn.Linear(
                hidden_dim, absolute_output_dim
            )
            nn.init.zeros_(self.absolute_quality_head.weight)
            nn.init.zeros_(self.absolute_quality_head.bias)
        self.cascade_quality_adapter = None
        self.cascade_correction_head = None
        self.cascade_opportunity_head = None
        self.cascade_candidate_safety_head = None
        self.cascade_joint_action_head = None
        self.cascade_fallback_set_action_head = None
        self.cascade_rich_fallback_set_action_head = None
        self.cascade_dense_quality_set_head = None
        self.cascade_relative_risk_set_head = None
        self.cascade_pairwise_calibrated_set_head = None
        self.cascade_selected_abstention_head = None
        self.cascade_counterfactual_selected_risk_head = None
        self.cascade_counterfactual_benefit_hazard_head = None
        self.cascade_counterfactual_logodds_head = None
        self.cascade_counterfactual_hazard_residual_head = None
        if self.action_mode in (
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            # These modules are absent from V12.  Restoring the RNG state keeps
            # the data-augmentation stream identical when comparing a migrated
            # cascade against the original checkpoint in the same process seed.
            with torch.random.fork_rng(devices=[]):
                absolute_output_dim = 2 * len(self.thresholds) + 2
                self.absolute_quality_head = nn.Linear(
                    hidden_dim, absolute_output_dim
                )
                nn.init.zeros_(self.absolute_quality_head.weight)
                nn.init.zeros_(self.absolute_quality_head.bias)
                self.cascade_quality_adapter = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                correction_input_dim = (
                    4 * hidden_dim + absolute_output_dim + 3
                )
                self.cascade_correction_head = nn.Sequential(
                    nn.Linear(correction_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.cascade_correction_head[-1].weight)
                nn.init.zeros_(self.cascade_correction_head[-1].bias)
                if self.action_mode in (
                        "cascade_opportunity_quality_correction",
                        "cascade_opportunity_verified_correction",
                        "cascade_joint_risk_correction",
                        "cascade_v19_fallback_set_correction",
                        "cascade_v19_rich_set_correction",
                        "cascade_v23_dense_quality_correction",
                        "cascade_v24_relative_risk_correction",
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                    self.cascade_opportunity_head = nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(
                        self.cascade_opportunity_head[-1].weight
                    )
                    nn.init.zeros_(
                        self.cascade_opportunity_head[-1].bias
                    )
                if self.action_mode in (
                        "cascade_opportunity_verified_correction",
                        "cascade_joint_risk_correction",
                        "cascade_v19_fallback_set_correction",
                        "cascade_v19_rich_set_correction",
                        "cascade_v23_dense_quality_correction",
                        "cascade_v24_relative_risk_correction",
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                    self.cascade_candidate_safety_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(
                        self.cascade_candidate_safety_head[-1].weight
                    )
                    nn.init.zeros_(
                        self.cascade_candidate_safety_head[-1].bias
                    )
                if self.action_mode == "cascade_joint_risk_correction":
                    self.cascade_joint_action_head = nn.Sequential(
                        nn.Linear(hidden_dim + 3, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(
                        self.cascade_joint_action_head[-1].weight
                    )
                    nn.init.zeros_(
                        self.cascade_joint_action_head[-1].bias
                    )
                if self.action_mode == "cascade_v19_fallback_set_correction":
                    self.cascade_fallback_set_action_head = (
                        FallbackTokenSetActionHead(
                            input_dim=hidden_dim + 3,
                            hidden_dim=hidden_dim,
                            max_candidates=self.candidate_top_k,
                            num_heads=context_heads,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode == "cascade_v19_rich_set_correction":
                    self.cascade_rich_fallback_set_action_head = (
                        RichFallbackTokenSetActionHead(
                            action_dim=hidden_dim + 3,
                            rich_dim=self.rich_feature_dim,
                            hidden_dim=hidden_dim,
                            max_candidates=self.candidate_top_k,
                            num_heads=context_heads,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode in (
                        "cascade_v23_dense_quality_correction",
                        "cascade_v24_relative_risk_correction",
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                    self.cascade_dense_quality_set_head = (
                        DenseQualityFallbackSetActionHead(
                            action_dim=hidden_dim + 3,
                            rich_dim=self.rich_feature_dim,
                            adaptive_dim=self.adaptive_source_feature_dim,
                            hidden_dim=hidden_dim,
                            max_candidates=self.candidate_top_k,
                            threshold_count=len(self.thresholds),
                            mask_utility_weight=self.mask_utility_weight,
                            num_heads=context_heads,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode == "cascade_v24_relative_risk_correction":
                    self.cascade_relative_risk_set_head = (
                        RelativeRiskFallbackSetActionHead(
                            action_dim=hidden_dim + 3,
                            rich_dim=self.rich_feature_dim,
                            adaptive_dim=self.adaptive_source_feature_dim,
                            evidence_dim=2 * len(self.thresholds) + 2,
                            hidden_dim=hidden_dim,
                            max_candidates=self.candidate_top_k,
                            num_heads=context_heads,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode in (
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                    self.cascade_pairwise_calibrated_set_head = (
                        CalibratedPairwiseRiskSetActionHead(
                            action_dim=hidden_dim + 3,
                            rich_dim=self.rich_feature_dim,
                            adaptive_dim=self.adaptive_source_feature_dim,
                            evidence_dim=2 * len(self.thresholds) + 2,
                            hidden_dim=hidden_dim,
                            max_candidates=self.candidate_top_k,
                            num_heads=context_heads,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode == (
                        "cascade_v28_selected_abstention_correction"):
                    self.cascade_selected_abstention_head = (
                        SelectedCandidateAbstentionHead(
                            hidden_dim=hidden_dim,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode == (
                        "cascade_v29_counterfactual_selected_correction"):
                    self.cascade_counterfactual_selected_risk_head = (
                        CounterfactualSelectedRiskHead(
                            hidden_dim=hidden_dim,
                            dropout=context_dropout,
                        )
                    )
                if self.action_mode == (
                        "cascade_v37_counterfactual_benefit_hazard_correction"):
                    self.cascade_counterfactual_benefit_hazard_head = (
                        CounterfactualSelectedRiskHead(
                            hidden_dim=hidden_dim,
                            dropout=context_dropout,
                            decomposed=True,
                        )
                    )
                if self.action_mode == (
                        "cascade_v38_complementary_logodds_correction"):
                    self.cascade_counterfactual_logodds_head = (
                        CounterfactualSelectedRiskHead(
                            hidden_dim=hidden_dim,
                            dropout=context_dropout,
                            decomposed=True,
                            complementary_log_odds=True,
                        )
                    )
                if self.action_mode == (
                        "cascade_v39_hazard_residual_correction"):
                    self.cascade_counterfactual_hazard_residual_head = (
                        CounterfactualSelectedRiskHead(
                            hidden_dim=hidden_dim,
                            dropout=context_dropout,
                            decomposed=True,
                            hazard_residual=True,
                        )
                    )

    @staticmethod
    def _gather_query(values, indices):
        gather_index = indices.view(-1, 1, 1).expand(
            -1, 1, values.shape[-1]
        )
        return torch.gather(values, 1, gather_index).squeeze(1)

    def forward(self, query_features, candidate_scores, shared_scores,
                valid_mask, default_indices=None,
                rich_candidate_features=None,
                adaptive_source_features=None):
        if (not isinstance(query_features, torch.Tensor)
                or query_features.dim() != 3
                or query_features.shape[-1] != self.query_dim):
            raise ValueError("query_features must have shape [B,Q,D]")
        batch_size, num_queries, _ = query_features.shape
        expected_shape = (batch_size, num_queries)
        if (candidate_scores.shape != expected_shape
                or shared_scores.shape != expected_shape):
            raise ValueError("gate scores must align with query features")
        if (valid_mask.dtype != torch.bool
                or valid_mask.shape != expected_shape
                or valid_mask.device != query_features.device):
            raise ValueError("gate valid_mask must be bool with shape [B,Q]")
        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("every gate sample needs a valid query")
        if self.action_mode in (
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            expected_rich_shape = (
                batch_size, num_queries, self.rich_feature_dim
            )
            if (not isinstance(rich_candidate_features, torch.Tensor)
                    or rich_candidate_features.shape != expected_rich_shape
                    or rich_candidate_features.device
                    != query_features.device
                    or not bool(torch.isfinite(
                        rich_candidate_features
                    ).all().item())):
                raise ValueError(
                    "rich_candidate_features must be finite with shape {}"
                    .format(expected_rich_shape)
                )
        if self.action_mode in (
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            expected_adaptive_shape = (
                batch_size, num_queries, self.adaptive_source_feature_dim
            )
            if (not isinstance(adaptive_source_features, torch.Tensor)
                    or adaptive_source_features.shape
                    != expected_adaptive_shape
                    or adaptive_source_features.device
                    != query_features.device
                    or not bool(torch.isfinite(
                        adaptive_source_features
                    ).all().item())):
                raise ValueError(
                    "adaptive_source_features must be finite with shape {}"
                    .format(expected_adaptive_shape)
                )

        shared_valid = shared_scores.masked_fill(~valid_mask, -1e4)
        candidate_valid = candidate_scores.masked_fill(~valid_mask, -1e4)
        if default_indices is None:
            default_indices = shared_valid.argmax(dim=1)
        elif (not isinstance(default_indices, torch.Tensor)
              or default_indices.dtype != torch.long
              or default_indices.shape != (batch_size,)
              or default_indices.device != query_features.device):
            raise ValueError(
                "default_indices must be int64 with shape [B]"
            )
        row_index = torch.arange(batch_size, device=query_features.device)
        if (bool((default_indices < 0).any().item())
                or bool((default_indices >= num_queries).any().item())
                or not bool(valid_mask[row_index, default_indices].all().item())):
            raise ValueError("default_indices must identify valid queries")
        default_features = self._gather_query(
            query_features, default_indices
        ).unsqueeze(1).expand(-1, num_queries, -1)
        default_candidate_scores = candidate_valid[
            row_index, default_indices
        ].unsqueeze(1).expand(-1, num_queries)
        default_shared_scores = shared_valid[
            row_index, default_indices
        ].unsqueeze(1).expand(-1, num_queries)
        score_features = torch.stack((
            candidate_valid,
            default_candidate_scores,
            candidate_valid - default_candidate_scores,
            shared_valid,
            default_shared_scores,
            shared_valid - default_shared_scores,
        ), dim=-1)
        pair_features = torch.cat((
            query_features,
            default_features,
            query_features - default_features,
            score_features,
        ), dim=-1)

        if self.context_encoder is None:
            # Preserve the legacy action space exactly when the new module is
            # disabled: top-k is selected first and the fallback is removed.
            top_k = min(self.candidate_top_k, num_queries)
            top_indices = candidate_valid.topk(top_k, dim=1).indices
        else:
            # The contextual gate treats fallback as a separate action, so K
            # denotes K genuine alternatives instead of including default.
            top_k = min(self.candidate_top_k, max(num_queries - 1, 0))
            alternative_scores = candidate_valid.clone()
            alternative_scores[row_index, default_indices] = -1e4
            top_indices = alternative_scores.topk(top_k, dim=1).indices
        candidate_mask = torch.zeros_like(valid_mask)
        candidate_mask.scatter_(1, top_indices, True)
        candidate_mask &= valid_mask
        candidate_mask[row_index, default_indices] = False

        hidden = self.encoder(pair_features)
        context_scale = None
        if self.context_encoder is not None:
            context_indices = torch.cat((
                default_indices.unsqueeze(1), top_indices,
            ), dim=1)
            context_valid = torch.cat((
                torch.ones(
                    batch_size, 1, dtype=torch.bool,
                    device=query_features.device,
                ),
                torch.gather(candidate_mask, 1, top_indices),
            ), dim=1)
            hidden, context_scale = self.context_encoder(
                hidden, context_indices, context_valid
            )
        output_shape = (
            batch_size, num_queries, len(self.thresholds), 3
        )
        box_logits = self.box_head(hidden).view(output_shape)
        mask_logits = self.mask_head(hidden).view(output_shape)
        decision_logits = self.decision_head(hidden)
        override_margin = (
            decision_logits[..., 2]
            - torch.maximum(
                decision_logits[..., 0], decision_logits[..., 1]
            )
        )

        expected_utility = transition_logits_expected_utility(
            box_logits,
            self.threshold_weights,
            self.break_cost,
        )
        if self.mask_utility_weight > 0.0:
            expected_utility = expected_utility + self.mask_utility_weight * (
                transition_logits_expected_utility(
                    mask_logits,
                    self.threshold_weights,
                    self.break_cost,
                )
            )
        direct_utility = self.utility_head(hidden).squeeze(-1)
        absolute_box_logits = None
        absolute_box_iou = None
        absolute_mask_logits = None
        absolute_mask_iou = None
        absolute_quality = None
        absolute_quality_margin = None
        absolute_quality_evidence = None
        quality_hidden = hidden
        if self.absolute_quality_head is not None:
            if self.cascade_quality_adapter is not None:
                quality_hidden = self.cascade_quality_adapter(hidden.detach())
            threshold_count = len(self.thresholds)
            absolute_prediction = self.absolute_quality_head(quality_hidden)
            absolute_box_logits = absolute_prediction[
                ..., :threshold_count
            ]
            absolute_box_iou = absolute_prediction[
                ..., threshold_count
            ].sigmoid()
            mask_start = threshold_count + 1
            absolute_mask_logits = absolute_prediction[
                ..., mask_start:mask_start + threshold_count
            ]
            absolute_mask_iou = absolute_prediction[..., -1].sigmoid()
            absolute_quality_evidence = torch.cat((
                absolute_box_logits.sigmoid(),
                absolute_box_iou.unsqueeze(-1),
                absolute_mask_logits.sigmoid(),
                absolute_mask_iou.unsqueeze(-1),
            ), dim=-1)

            tier_weights = torch.arange(
                1, threshold_count + 1,
                dtype=absolute_prediction.dtype,
                device=absolute_prediction.device,
            )
            quality_denominator = tier_weights.sum() + 1.0
            absolute_box_quality = (
                (absolute_box_logits.sigmoid() * tier_weights).sum(dim=-1)
                + absolute_box_iou
            ) / quality_denominator
            absolute_mask_quality = (
                (absolute_mask_logits.sigmoid() * tier_weights).sum(dim=-1)
                + absolute_mask_iou
            ) / quality_denominator
            absolute_quality = (
                absolute_box_quality
                + self.mask_utility_weight * absolute_mask_quality
            )
            default_absolute_quality = absolute_quality[
                row_index, default_indices
            ].unsqueeze(1)
            absolute_quality_margin = (
                absolute_quality - default_absolute_quality
            )
        if self.action_mode == "expected_utility":
            action_margin = expected_utility
        elif self.action_mode in (
                "direct_utility", "hierarchical_utility",
                "pairwise_verifier", "topn_pairwise_verifier",
                "topn_dual_evidence_verifier"):
            action_margin = direct_utility
        elif self.action_mode == "topn_absolute_quality_delta":
            action_margin = absolute_quality_margin
        elif self.action_mode in (
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            # Stage one must remain byte-for-byte compatible with V12's
            # pairwise verifier. The new correction action is constructed
            # below after that frozen verifier chooses its dynamic anchor.
            action_margin = direct_utility
        else:
            action_margin = override_margin

        eligible_margin = action_margin.masked_fill(~candidate_mask, -1e4)
        best_margin, best_indices = eligible_margin.max(dim=1)
        has_candidate = candidate_mask.any(dim=1)
        row_switch_margin = None
        deployed_margin = best_margin
        action_fallback_indices = default_indices
        cascade_base_indices = None
        cascade_base_switch = None
        cascade_base_margin = None
        joint_action_margin = None
        v19_fallback_indices = None
        v19_correction_switch = None
        v19_deployed_margin = None
        dense_quality_mask = None
        quality_uncertainty = None
        risk_quality_margin = None
        relative_risk_margin = None
        pairwise_calibrated_margin = None
        pairwise_utility_margin = None
        pairwise_benefit_margin = None
        selected_abstention_margin = None
        candidate_selection_margin = None
        counterfactual_risk_margin = None
        counterfactual_benefit_margin = None
        counterfactual_hazard_margin = None
        if self.action_mode == "hierarchical_utility":
            default_hidden = self._gather_query(hidden, default_indices)
            candidate_weight = candidate_mask.unsqueeze(-1).to(hidden.dtype)
            candidate_mean = (
                (hidden * candidate_weight).sum(dim=1)
                / candidate_weight.sum(dim=1).clamp(min=1.0)
            )
            row_features = torch.cat((
                default_hidden,
                candidate_mean,
                candidate_mean - default_hidden,
            ), dim=-1)
            row_switch_margin = self.row_switch_head(
                row_features
            ).squeeze(-1)
            deployed_margin = row_switch_margin
        elif self.action_mode in (
                "pairwise_verifier",
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            default_hidden = self._gather_query(hidden, default_indices)
            selected_hidden = self._gather_query(hidden, best_indices)
            pairwise_features = torch.cat((
                default_hidden,
                selected_hidden,
                selected_hidden - default_hidden,
                selected_hidden * default_hidden,
                best_margin.unsqueeze(1),
            ), dim=-1)
            row_switch_margin = self.pairwise_switch_head(
                pairwise_features
            ).squeeze(-1)
            if self.action_mode == "pairwise_verifier":
                deployed_margin = row_switch_margin
            else:
                cascade_base_margin = row_switch_margin
                cascade_base_switch = has_candidate & (
                    cascade_base_margin > self.decision_margin
                )
                cascade_base_indices = torch.where(
                    cascade_base_switch, best_indices, default_indices
                )
                action_fallback_indices = cascade_base_indices

                correction_mask = candidate_mask.clone()
                correction_mask[row_index, default_indices] = valid_mask[
                    row_index, default_indices
                ]
                correction_mask[row_index, cascade_base_indices] = False

                anchor_quality = absolute_quality[
                    row_index, cascade_base_indices
                ].unsqueeze(1)
                absolute_quality_margin = absolute_quality - anchor_quality
                action_margin = absolute_quality_margin

                anchor_quality_hidden = self._gather_query(
                    quality_hidden, cascade_base_indices
                ).unsqueeze(1).expand(-1, num_queries, -1)
                anchor_quality_evidence = self._gather_query(
                    absolute_quality_evidence, cascade_base_indices
                ).unsqueeze(1)
                evidence_delta = (
                    absolute_quality_evidence - anchor_quality_evidence
                )
                anchor_direct_utility = direct_utility[
                    row_index, cascade_base_indices
                ].detach().unsqueeze(1)
                frozen_direct_utility = direct_utility.detach()
                cascade_features = torch.cat((
                    anchor_quality_hidden,
                    quality_hidden,
                    quality_hidden - anchor_quality_hidden,
                    quality_hidden * anchor_quality_hidden,
                    evidence_delta,
                    absolute_quality_margin.unsqueeze(-1),
                    (
                        frozen_direct_utility - anchor_direct_utility
                    ).unsqueeze(-1),
                    cascade_base_margin.detach().view(-1, 1, 1).expand(
                        -1, num_queries, 1
                    ),
                ), dim=-1)
                candidate_mask = correction_mask
                has_candidate = candidate_mask.any(dim=1)
                candidate_safety_margin = None
                if self.action_mode in (
                        "cascade_opportunity_quality_correction",
                        "cascade_opportunity_verified_correction",
                        "cascade_joint_risk_correction",
                        "cascade_v19_fallback_set_correction",
                        "cascade_v19_rich_set_correction",
                        "cascade_v23_dense_quality_correction",
                        "cascade_v24_relative_risk_correction",
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                    correction_hidden = cascade_features
                    for correction_module in tuple(
                            self.cascade_correction_head.children())[:-1]:
                        correction_hidden = correction_module(
                            correction_hidden
                        )
                    candidate_rank_margin = self.cascade_correction_head[-1](
                        correction_hidden
                    ).squeeze(-1)
                    candidate_weight = candidate_mask.unsqueeze(-1).to(
                        dtype=correction_hidden.dtype
                    )
                    candidate_count = candidate_weight.sum(dim=1).clamp(
                        min=1.0
                    )
                    candidate_mean = (
                        (correction_hidden * candidate_weight).sum(dim=1)
                        / candidate_count
                    )
                    negative_sentinel = torch.finfo(
                        correction_hidden.dtype
                    ).min
                    candidate_max = correction_hidden.masked_fill(
                        ~candidate_mask.unsqueeze(-1), negative_sentinel
                    ).max(dim=1).values
                    candidate_max = torch.where(
                        has_candidate.unsqueeze(1),
                        candidate_max,
                        torch.zeros_like(candidate_max),
                    )
                    opportunity_features = torch.cat((
                        candidate_mean, candidate_max,
                    ), dim=-1)
                    row_switch_margin = self.cascade_opportunity_head(
                        opportunity_features
                    ).squeeze(-1)
                    action_margin = candidate_rank_margin
                    if self.action_mode in (
                            "cascade_opportunity_verified_correction",
                            "cascade_joint_risk_correction",
                            "cascade_v19_fallback_set_correction",
                            "cascade_v19_rich_set_correction",
                            "cascade_v23_dense_quality_correction",
                            "cascade_v24_relative_risk_correction",
                            "cascade_v25_pairwise_calibrated_correction",
                            "cascade_v26_prior_restored_pairwise_correction",
                            "cascade_v28_selected_abstention_correction",
                            "cascade_v29_counterfactual_selected_correction",
                            "cascade_v37_counterfactual_benefit_hazard_correction",
                            "cascade_v38_complementary_logodds_correction",
                            "cascade_v39_hazard_residual_correction"):
                        candidate_safety_margin = (
                            self.cascade_candidate_safety_head(
                                correction_hidden
                            ).squeeze(-1)
                        )
                    if self.action_mode == "cascade_joint_risk_correction":
                        joint_features = torch.cat((
                            correction_hidden,
                            candidate_rank_margin.unsqueeze(-1),
                            candidate_safety_margin.unsqueeze(-1),
                            row_switch_margin.view(-1, 1, 1).expand(
                                -1, num_queries, 1
                            ),
                        ), dim=-1)
                        joint_action_margin = self.cascade_joint_action_head(
                            joint_features
                        ).squeeze(-1)
                    elif self.action_mode in (
                            "cascade_v19_fallback_set_correction",
                            "cascade_v19_rich_set_correction",
                            "cascade_v23_dense_quality_correction",
                            "cascade_v24_relative_risk_correction",
                            "cascade_v25_pairwise_calibrated_correction",
                            "cascade_v26_prior_restored_pairwise_correction",
                            "cascade_v28_selected_abstention_correction",
                            "cascade_v29_counterfactual_selected_correction",
                            "cascade_v37_counterfactual_benefit_hazard_correction",
                            "cascade_v38_complementary_logodds_correction",
                            "cascade_v39_hazard_residual_correction"):
                        v19_rank_margin = candidate_rank_margin.masked_fill(
                            ~candidate_mask, -1e4
                        )
                        _, v19_best_indices = v19_rank_margin.max(dim=1)
                        v19_selected_safety = candidate_safety_margin[
                            row_index, v19_best_indices
                        ]
                        v19_deployed_margin = torch.minimum(
                            row_switch_margin, v19_selected_safety
                        )
                        v19_correction_switch = has_candidate & (
                            v19_deployed_margin > self.decision_margin
                        )
                        v19_fallback_indices = torch.where(
                            v19_correction_switch,
                            v19_best_indices,
                            cascade_base_indices,
                        )

                        fallback_set_mask = candidate_mask.clone()
                        fallback_set_mask[
                            row_index, v19_fallback_indices
                        ] = False
                        anchor_is_candidate = fallback_set_mask[
                            row_index, cascade_base_indices
                        ]
                        fallback_set_mask[
                            row_index, cascade_base_indices
                        ] = anchor_is_candidate | v19_correction_switch
                        fallback_set_mask &= valid_mask
                        fallback_set_features = torch.cat((
                            correction_hidden,
                            candidate_rank_margin.unsqueeze(-1),
                            candidate_safety_margin.unsqueeze(-1),
                            row_switch_margin.view(-1, 1, 1).expand(
                                -1, num_queries, 1
                            ),
                        ), dim=-1).detach()
                        if self.action_mode == (
                                "cascade_v19_fallback_set_correction"):
                            fallback_set_output = (
                                self.cascade_fallback_set_action_head(
                                    fallback_set_features,
                                    v19_fallback_indices,
                                    fallback_set_mask,
                                )
                            )
                        elif self.action_mode == (
                                "cascade_v19_rich_set_correction"):
                            fallback_set_output = (
                                self.cascade_rich_fallback_set_action_head(
                                    fallback_set_features,
                                    rich_candidate_features.detach(),
                                    v19_fallback_indices,
                                    fallback_set_mask,
                                )
                            )
                        elif self.action_mode in (
                                "cascade_v23_dense_quality_correction",
                                "cascade_v24_relative_risk_correction",
                                "cascade_v25_pairwise_calibrated_correction",
                            "cascade_v26_prior_restored_pairwise_correction",
                            "cascade_v28_selected_abstention_correction",
                            "cascade_v29_counterfactual_selected_correction",
                            "cascade_v37_counterfactual_benefit_hazard_correction",
                            "cascade_v38_complementary_logodds_correction",
                            "cascade_v39_hazard_residual_correction"):
                            fallback_set_output = (
                                self.cascade_dense_quality_set_head(
                                    fallback_set_features,
                                    rich_candidate_features.detach(),
                                    adaptive_source_features,
                                    v19_fallback_indices,
                                    fallback_set_mask,
                                )
                            )
                            absolute_box_logits = fallback_set_output[
                                "box_threshold_logits"
                            ]
                            absolute_box_iou = fallback_set_output["box_iou"]
                            absolute_mask_logits = fallback_set_output[
                                "mask_threshold_logits"
                            ]
                            absolute_mask_iou = fallback_set_output["mask_iou"]
                            absolute_quality = fallback_set_output["quality"]
                            quality_uncertainty = fallback_set_output[
                                "uncertainty"
                            ]
                            absolute_quality_margin = fallback_set_output[
                                "margin"
                            ]
                            if self.uncertainty_weight > 0.0:
                                fallback_risk_quality = (
                                    absolute_quality
                                    - self.uncertainty_weight
                                    * quality_uncertainty
                                )
                                fallback_risk_anchor = fallback_risk_quality[
                                    row_index, v19_fallback_indices
                                ].unsqueeze(1)
                                risk_quality_margin = (
                                    fallback_risk_quality
                                    - fallback_risk_anchor
                                ).masked_fill(
                                    ~fallback_set_output["quality_mask"], 0.0
                                )
                            dense_quality_mask = fallback_set_output[
                                "quality_mask"
                            ]
                            if self.action_mode == (
                                    "cascade_v24_relative_risk_correction"):
                                relative_output = (
                                    self.cascade_relative_risk_set_head(
                                        fallback_set_features,
                                        rich_candidate_features.detach(),
                                        adaptive_source_features,
                                        fallback_set_output[
                                            "evidence"
                                        ].detach(),
                                        v19_fallback_indices,
                                        fallback_set_mask,
                                    )
                                )
                                relative_risk_margin = relative_output[
                                    "margin"
                                ]
                                joint_action_margin = relative_risk_margin
                            elif self.action_mode in (
                                    "cascade_v25_pairwise_calibrated_correction",
                                    "cascade_v26_prior_restored_pairwise_correction",
                                    "cascade_v28_selected_abstention_correction",
                                    "cascade_v29_counterfactual_selected_correction",
                                    "cascade_v37_counterfactual_benefit_hazard_correction",
                                    "cascade_v38_complementary_logodds_correction",
                                    "cascade_v39_hazard_residual_correction"):
                                pairwise_output = (
                                    self.cascade_pairwise_calibrated_set_head(
                                        fallback_set_features,
                                        rich_candidate_features.detach(),
                                        adaptive_source_features,
                                        fallback_set_output["evidence"],
                                        v19_fallback_indices,
                                        fallback_set_mask,
                                    )
                                )
                                pairwise_utility_margin = pairwise_output[
                                    "margin"
                                ]
                                pairwise_benefit_margin = pairwise_output[
                                    "benefit_margin"
                                ]
                                if self.action_mode == (
                                        "cascade_v28_selected_abstention_correction"):
                                    abstention_output = (
                                        self.cascade_selected_abstention_head(
                                            pairwise_output["pair_features"],
                                            pairwise_utility_margin,
                                            fallback_set_mask,
                                        )
                                    )
                                    candidate_selection_margin = (
                                        abstention_output["selection_margin"]
                                    )
                                    selected_abstention_margin = (
                                        abstention_output["row_margin"]
                                    )
                                    joint_action_margin = abstention_output[
                                        "margin"
                                    ]
                                elif self.action_mode == (
                                        "cascade_v29_counterfactual_selected_correction"):
                                    counterfactual_output = (
                                        self.cascade_counterfactual_selected_risk_head(
                                            pairwise_output["pair_features"],
                                            pairwise_utility_margin,
                                            fallback_set_mask,
                                        )
                                    )
                                    candidate_selection_margin = (
                                        counterfactual_output["selection_margin"]
                                    )
                                    selected_abstention_margin = (
                                        counterfactual_output["row_margin"]
                                    )
                                    counterfactual_risk_margin = (
                                        counterfactual_output["candidate_risk"]
                                    )
                                    joint_action_margin = counterfactual_output[
                                        "margin"
                                    ]
                                elif self.action_mode == (
                                        "cascade_v37_counterfactual_benefit_hazard_correction"):
                                    counterfactual_output = (
                                        self.cascade_counterfactual_benefit_hazard_head(
                                            pairwise_output["pair_features"],
                                            pairwise_utility_margin,
                                            fallback_set_mask,
                                        )
                                    )
                                    candidate_selection_margin = (
                                        counterfactual_output["selection_margin"]
                                    )
                                    selected_abstention_margin = (
                                        counterfactual_output["row_margin"]
                                    )
                                    counterfactual_risk_margin = (
                                        counterfactual_output["candidate_risk"]
                                    )
                                    counterfactual_benefit_margin = (
                                        counterfactual_output[
                                            "candidate_benefit"
                                        ]
                                    )
                                    counterfactual_hazard_margin = (
                                        counterfactual_output[
                                            "candidate_hazard"
                                        ]
                                    )
                                    joint_action_margin = counterfactual_output[
                                        "margin"
                                    ]
                                elif self.action_mode == (
                                        "cascade_v38_complementary_logodds_correction"):
                                    counterfactual_output = (
                                        self.cascade_counterfactual_logodds_head(
                                            pairwise_output["pair_features"],
                                            pairwise_utility_margin,
                                            fallback_set_mask,
                                        )
                                    )
                                    candidate_selection_margin = (
                                        counterfactual_output["selection_margin"]
                                    )
                                    selected_abstention_margin = (
                                        counterfactual_output["row_margin"]
                                    )
                                    counterfactual_risk_margin = (
                                        counterfactual_output["candidate_risk"]
                                    )
                                    counterfactual_benefit_margin = (
                                        counterfactual_output[
                                            "candidate_benefit"
                                        ]
                                    )
                                    counterfactual_hazard_margin = (
                                        counterfactual_output[
                                            "candidate_hazard"
                                        ]
                                    )
                                    joint_action_margin = counterfactual_output[
                                        "margin"
                                    ]
                                elif self.action_mode == (
                                        "cascade_v39_hazard_residual_correction"):
                                    counterfactual_output = (
                                        self.cascade_counterfactual_hazard_residual_head(
                                            pairwise_output["pair_features"],
                                            pairwise_utility_margin,
                                            fallback_set_mask,
                                        )
                                    )
                                    candidate_selection_margin = (
                                        counterfactual_output["selection_margin"]
                                    )
                                    selected_abstention_margin = (
                                        counterfactual_output["row_margin"]
                                    )
                                    counterfactual_risk_margin = (
                                        counterfactual_output["candidate_risk"]
                                    )
                                    counterfactual_benefit_margin = (
                                        counterfactual_output[
                                            "candidate_benefit"
                                        ]
                                    )
                                    counterfactual_hazard_margin = (
                                        counterfactual_output[
                                            "candidate_hazard"
                                        ]
                                    )
                                    joint_action_margin = counterfactual_output[
                                        "margin"
                                    ]
                                else:
                                    joint_action_margin = (
                                        pairwise_benefit_margin
                                        if self.action_mode == (
                                            "cascade_v26_prior_restored_pairwise_correction"
                                        )
                                        else pairwise_utility_margin
                                    )
                                if self.action_mode == (
                                        "cascade_v25_pairwise_calibrated_correction"):
                                    pairwise_calibrated_margin = (
                                        pairwise_utility_margin
                                    )
                            else:
                                joint_action_margin = fallback_set_output[
                                    "margin"
                                ]
                            if (self.uncertainty_weight > 0.0
                                    and risk_quality_margin is not None):
                                joint_action_margin = risk_quality_margin
                        if self.action_mode not in (
                                "cascade_v23_dense_quality_correction",
                                "cascade_v24_relative_risk_correction",
                                "cascade_v25_pairwise_calibrated_correction",
                                "cascade_v26_prior_restored_pairwise_correction",
                                "cascade_v28_selected_abstention_correction",
                                "cascade_v29_counterfactual_selected_correction",
                                "cascade_v37_counterfactual_benefit_hazard_correction",
                                "cascade_v38_complementary_logodds_correction",
                                "cascade_v39_hazard_residual_correction"):
                            joint_action_margin = fallback_set_output["margin"]
                        candidate_mask = fallback_set_mask
                        has_candidate = candidate_mask.any(dim=1)
                        action_fallback_indices = v19_fallback_indices
                        action_margin = joint_action_margin
                        best_margin, best_indices = (
                            joint_action_margin.masked_fill(
                                ~candidate_mask, -1e4
                            ).max(dim=1)
                        )
                        deployed_margin = best_margin
                else:
                    row_switch_margin = self.cascade_correction_head(
                        cascade_features
                    ).squeeze(-1)
                eligible_correction_margin = row_switch_margin.masked_fill(
                    ~candidate_mask, -1e4
                ) if row_switch_margin.dim() == 2 else None
                if self.action_mode in (
                        "cascade_v19_fallback_set_correction",
                        "cascade_v19_rich_set_correction",
                        "cascade_v23_dense_quality_correction",
                        "cascade_v24_relative_risk_correction",
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                    pass
                elif self.action_mode in (
                        "cascade_opportunity_quality_correction",
                        "cascade_opportunity_verified_correction",
                        "cascade_joint_risk_correction"):
                    if self.action_mode == "cascade_joint_risk_correction":
                        best_margin, best_indices = (
                            joint_action_margin.masked_fill(
                                ~candidate_mask, -1e4
                            ).max(dim=1)
                        )
                        deployed_margin = best_margin
                    else:
                        best_margin, best_indices = action_margin.masked_fill(
                            ~candidate_mask, -1e4
                        ).max(dim=1)
                    if self.action_mode == (
                            "cascade_opportunity_verified_correction"):
                        selected_safety_margin = candidate_safety_margin[
                            row_index, best_indices
                        ]
                        deployed_margin = torch.minimum(
                            row_switch_margin, selected_safety_margin
                        )
                    else:
                        if self.action_mode != "cascade_joint_risk_correction":
                            deployed_margin = row_switch_margin
                else:
                    best_margin, best_indices = (
                        eligible_correction_margin.max(dim=1)
                    )
                    deployed_margin = best_margin
        elif self.action_mode in (
                "topn_pairwise_verifier",
                "topn_dual_evidence_verifier"):
            default_hidden = self._gather_query(
                hidden, default_indices
            ).unsqueeze(1).expand(-1, num_queries, -1)
            pairwise_features = torch.cat((
                default_hidden,
                hidden,
                hidden - default_hidden,
                hidden * default_hidden,
                action_margin.unsqueeze(-1),
            ), dim=-1)
            row_benefit_margin = self.pairwise_switch_head(
                pairwise_features
            ).squeeze(-1)
            if self.action_mode == "topn_dual_evidence_verifier":
                risk_evidence = torch.cat((
                    F.softmax(box_logits, dim=-1).flatten(start_dim=2),
                    F.softmax(mask_logits, dim=-1).flatten(start_dim=2),
                    F.softmax(decision_logits, dim=-1),
                    expected_utility.unsqueeze(-1),
                    direct_utility.unsqueeze(-1),
                ), dim=-1)
                safety_features = torch.cat((hidden, risk_evidence), dim=-1)
                row_safety_margin = self.safety_switch_head(
                    safety_features
                ).squeeze(-1)
                row_switch_margin = torch.minimum(
                    row_benefit_margin, row_safety_margin
                )
            else:
                row_safety_margin = None
                row_switch_margin = row_benefit_margin
            eligible_verifier_margin = row_switch_margin.masked_fill(
                ~candidate_mask, -1e4
            )
            best_margin, best_indices = eligible_verifier_margin.max(dim=1)
            deployed_margin = best_margin
        deployment_boundary = (
            0.0
            if self.action_mode in (
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction",
            )
            else self.decision_margin
        )
        switch = has_candidate & (deployed_margin > deployment_boundary)
        selected_indices = torch.where(
            switch, best_indices, action_fallback_indices
        )
        final_switch = selected_indices != default_indices

        promoted = candidate_valid.clone()
        promoted_score = promoted.max(dim=1).values + 1.0
        promoted.scatter_(1, selected_indices.unsqueeze(1),
                          promoted_score.unsqueeze(1))
        selected_scores = torch.where(
            final_switch.unsqueeze(1), promoted, shared_valid
        ).masked_fill(~valid_mask, -1e4)

        finite_deployed_margin = torch.where(
            has_candidate, deployed_margin, torch.zeros_like(deployed_margin)
        )
        deployed_candidate_action_margin = (
            joint_action_margin
            if self.action_mode in (
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction",
            )
            else action_margin
        )
        outputs = {
            "selected_source_scores": selected_scores,
            "moe_gate_box_logits": box_logits,
            "moe_gate_mask_logits": mask_logits,
            "moe_gate_decision_logits": decision_logits,
            "moe_gate_candidate_mask": candidate_mask,
            "moe_gate_default_query": default_indices,
            "moe_gate_selected_query": selected_indices,
            "moe_gate_decision_margin": override_margin,
            "moe_gate_expected_utility": expected_utility,
            "moe_gate_direct_utility": direct_utility,
            "moe_gate_action_margin": action_margin,
            "moe_gate_switch": final_switch,
            "moe_gate_switch_ratio": final_switch.float().mean().detach(),
            "moe_gate_max_margin_mean": (
                finite_deployed_margin.mean().detach()
            ),
            "moe_gate_positive_candidate_ratio": (
                (deployed_candidate_action_margin > deployment_boundary)
                & candidate_mask
            ).float().sum().div(
                candidate_mask.float().sum().clamp(min=1.0)
            ).detach(),
        }
        if row_switch_margin is not None:
            outputs["moe_gate_row_switch_margin"] = row_switch_margin
        if self.action_mode == "topn_dual_evidence_verifier":
            outputs["moe_gate_row_benefit_margin"] = row_benefit_margin
            outputs["moe_gate_row_safety_margin"] = row_safety_margin
        if self.action_mode in (
                "topn_absolute_quality_delta",
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            outputs["moe_gate_absolute_box_logits"] = absolute_box_logits
            outputs["moe_gate_absolute_box_iou"] = absolute_box_iou
            outputs["moe_gate_absolute_mask_logits"] = absolute_mask_logits
            outputs["moe_gate_absolute_mask_iou"] = absolute_mask_iou
            outputs["moe_gate_absolute_quality"] = absolute_quality
            outputs["moe_gate_absolute_quality_margin"] = (
                absolute_quality_margin
            )
            if quality_uncertainty is not None:
                outputs["moe_gate_quality_uncertainty"] = (
                    quality_uncertainty
                )
                uncertainty_mask = dense_quality_mask.to(
                    dtype=quality_uncertainty.dtype
                )
                outputs["moe_gate_quality_uncertainty_mean"] = (
                    (quality_uncertainty * uncertainty_mask).sum()
                    / uncertainty_mask.sum().clamp(min=1.0)
                ).detach()
            if risk_quality_margin is not None:
                outputs["moe_gate_risk_quality_margin"] = (
                    risk_quality_margin
                )
        if self.action_mode in (
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            outputs["moe_gate_action_anchor_query"] = cascade_base_indices
            outputs["moe_gate_cascade_base_margin"] = cascade_base_margin
            outputs["moe_gate_cascade_base_switch"] = cascade_base_switch
            outputs["moe_gate_correction_switch"] = switch
            outputs["moe_gate_correction_switch_ratio"] = (
                switch.float().mean().detach()
            )
            if self.action_mode in (
                    "cascade_opportunity_quality_correction",
                    "cascade_opportunity_verified_correction",
                    "cascade_joint_risk_correction"):
                outputs["moe_gate_opportunity_margin"] = row_switch_margin
            if self.action_mode in (
                    "cascade_opportunity_verified_correction",
                    "cascade_joint_risk_correction"):
                outputs["moe_gate_row_safety_margin"] = (
                    candidate_safety_margin
                )
            if self.action_mode in (
                    "cascade_joint_risk_correction",
                    "cascade_v19_fallback_set_correction",
                    "cascade_v19_rich_set_correction",
                    "cascade_v23_dense_quality_correction",
                    "cascade_v24_relative_risk_correction",
                    "cascade_v25_pairwise_calibrated_correction",
                    "cascade_v26_prior_restored_pairwise_correction",
                    "cascade_v28_selected_abstention_correction",
                    "cascade_v29_counterfactual_selected_correction",
                    "cascade_v37_counterfactual_benefit_hazard_correction",
                    "cascade_v38_complementary_logodds_correction",
                    "cascade_v39_hazard_residual_correction"):
                outputs["moe_gate_joint_action_margin"] = (
                    joint_action_margin
                )
            if self.action_mode in (
                    "cascade_v19_fallback_set_correction",
                    "cascade_v19_rich_set_correction",
                    "cascade_v23_dense_quality_correction",
                    "cascade_v24_relative_risk_correction",
                    "cascade_v25_pairwise_calibrated_correction",
                    "cascade_v26_prior_restored_pairwise_correction",
                    "cascade_v28_selected_abstention_correction",
                    "cascade_v29_counterfactual_selected_correction",
                    "cascade_v37_counterfactual_benefit_hazard_correction",
                    "cascade_v38_complementary_logodds_correction",
                    "cascade_v39_hazard_residual_correction"):
                outputs["moe_gate_supervision_fallback_query"] = (
                    v19_fallback_indices
                )
                outputs["moe_gate_v19_fallback_query"] = (
                    v19_fallback_indices
                )
                outputs["moe_gate_v19_correction_switch"] = (
                    v19_correction_switch
                )
                outputs["moe_gate_v19_correction_switch_ratio"] = (
                    v19_correction_switch.float().mean().detach()
                )
                outputs["moe_gate_v19_deployed_margin"] = (
                    v19_deployed_margin
                )
                outputs["moe_gate_v19_opportunity_margin"] = (
                    row_switch_margin
                )
                outputs["moe_gate_v19_candidate_safety_margin"] = (
                    candidate_safety_margin
                )
            if self.action_mode in (
                    "cascade_v23_dense_quality_correction",
                    "cascade_v24_relative_risk_correction",
                    "cascade_v25_pairwise_calibrated_correction",
                    "cascade_v26_prior_restored_pairwise_correction",
                    "cascade_v28_selected_abstention_correction",
                    "cascade_v29_counterfactual_selected_correction",
                    "cascade_v37_counterfactual_benefit_hazard_correction",
                    "cascade_v38_complementary_logodds_correction",
                    "cascade_v39_hazard_residual_correction"):
                outputs["moe_gate_dense_quality_mask"] = dense_quality_mask
            if self.action_mode == "cascade_v24_relative_risk_correction":
                outputs["moe_gate_relative_risk_margin"] = relative_risk_margin
            if self.action_mode in (
                        "cascade_v25_pairwise_calibrated_correction",
                        "cascade_v26_prior_restored_pairwise_correction",
                        "cascade_v28_selected_abstention_correction",
                        "cascade_v29_counterfactual_selected_correction",
                        "cascade_v37_counterfactual_benefit_hazard_correction",
                        "cascade_v38_complementary_logodds_correction",
                        "cascade_v39_hazard_residual_correction"):
                outputs["moe_gate_row_benefit_margin"] = (
                    pairwise_benefit_margin
                )
            if self.action_mode == (
                    "cascade_v25_pairwise_calibrated_correction"):
                outputs["moe_gate_pairwise_calibrated_margin"] = (
                    pairwise_calibrated_margin
                )
            if self.action_mode == (
                    "cascade_v26_prior_restored_pairwise_correction"):
                outputs["moe_gate_pairwise_utility_margin"] = (
                    pairwise_utility_margin
                )
            if self.action_mode == (
                    "cascade_v28_selected_abstention_correction"):
                outputs["moe_gate_pairwise_utility_margin"] = (
                    pairwise_utility_margin
                )
                outputs["moe_gate_candidate_selection_margin"] = (
                    candidate_selection_margin
                )
                outputs["moe_gate_selected_abstention_margin"] = (
                    selected_abstention_margin
                )
            if self.action_mode == (
                    "cascade_v29_counterfactual_selected_correction"):
                outputs["moe_gate_pairwise_utility_margin"] = (
                    pairwise_utility_margin
                )
                outputs["moe_gate_candidate_selection_margin"] = (
                    candidate_selection_margin
                )
                outputs["moe_gate_selected_abstention_margin"] = (
                    selected_abstention_margin
                )
                outputs["moe_gate_counterfactual_risk_margin"] = (
                    counterfactual_risk_margin
                )
            if self.action_mode == (
                    "cascade_v37_counterfactual_benefit_hazard_correction"):
                outputs["moe_gate_pairwise_utility_margin"] = (
                    pairwise_utility_margin
                )
                outputs["moe_gate_candidate_selection_margin"] = (
                    candidate_selection_margin
                )
                outputs["moe_gate_selected_abstention_margin"] = (
                    selected_abstention_margin
                )
                outputs["moe_gate_counterfactual_risk_margin"] = (
                    counterfactual_risk_margin
                )
                outputs["moe_gate_counterfactual_benefit_margin"] = (
                    counterfactual_benefit_margin
                )
                outputs["moe_gate_counterfactual_hazard_margin"] = (
                    counterfactual_hazard_margin
                )
            if self.action_mode == (
                    "cascade_v38_complementary_logodds_correction"):
                outputs["moe_gate_pairwise_utility_margin"] = (
                    pairwise_utility_margin
                )
                outputs["moe_gate_candidate_selection_margin"] = (
                    candidate_selection_margin
                )
                outputs["moe_gate_selected_abstention_margin"] = (
                    selected_abstention_margin
                )
                outputs["moe_gate_counterfactual_risk_margin"] = (
                    counterfactual_risk_margin
                )
                outputs["moe_gate_counterfactual_benefit_margin"] = (
                    counterfactual_benefit_margin
                )
                outputs["moe_gate_counterfactual_hazard_margin"] = (
                    counterfactual_hazard_margin
                )
            if self.action_mode == (
                    "cascade_v39_hazard_residual_correction"):
                outputs["moe_gate_pairwise_utility_margin"] = (
                    pairwise_utility_margin
                )
                outputs["moe_gate_candidate_selection_margin"] = (
                    candidate_selection_margin
                )
                outputs["moe_gate_selected_abstention_margin"] = (
                    selected_abstention_margin
                )
                outputs["moe_gate_counterfactual_risk_margin"] = (
                    counterfactual_risk_margin
                )
                outputs["moe_gate_counterfactual_benefit_margin"] = (
                    counterfactual_benefit_margin
                )
                outputs["moe_gate_counterfactual_hazard_margin"] = (
                    counterfactual_hazard_margin
                )
        if context_scale is not None:
            outputs["moe_gate_context_scale"] = context_scale.detach()
        return outputs


class SourceMoE(nn.Module):
    """Shared-source sparse fusion followed by contextual query reranking."""

    def __init__(self, source_names, shared_source="default", d_model=64,
                 hidden_dim=128, text_dim=None, top_k=2,
                 balance_loss_weight=0.01, query_layers=1,
                 query_heads=4, query_dropout=0.1,
                 query_max_delta=0.25, use_fallback_gate=False,
                 gate_hidden_dim=128, gate_candidate_top_k=8,
                 gate_break_cost=2.0, gate_decision_margin=0.0,
                 gate_mask_utility_weight=0.25,
                 gate_uncertainty_weight=0.0,
                 gate_use_evidence_features=False,
                 gate_evidence_dim=None, gate_context_layers=0,
                 gate_context_heads=4, gate_context_dropout=0.1,
                 gate_action_mode="decision"):
        super().__init__()
        source_names = tuple(source_names)
        if not source_names:
            raise ValueError("source_names must not be empty")
        if shared_source not in source_names:
            raise ValueError("shared_source must be one of source_names")
        if len(source_names) != len(set(source_names)):
            raise ValueError("source_names must be unique")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.source_names = source_names
        self.shared_source = shared_source
        self.routed_source_names = tuple(
            name for name in source_names if name != shared_source
        )
        self.num_routed = len(self.routed_source_names)
        self.top_k = min(top_k, max(self.num_routed, 1))
        self.balance_loss_weight = float(balance_loss_weight)
        self.d_model = int(d_model)
        if (not isinstance(gate_uncertainty_weight, (float, int))
                or isinstance(gate_uncertainty_weight, bool)
                or not math.isfinite(float(gate_uncertainty_weight))
                or float(gate_uncertainty_weight) < 0.0):
            raise ValueError(
                "gate_uncertainty_weight must be finite and non-negative"
            )
        self.uncertainty_weight = float(gate_uncertainty_weight)
        if gate_action_mode is None:
            gate_action_mode = "decision"
        if gate_action_mode not in (
                "decision", "expected_utility", "direct_utility",
                "hierarchical_utility", "pairwise_verifier",
                "topn_pairwise_verifier",
                "topn_dual_evidence_verifier",
                "topn_absolute_quality_delta",
                "cascade_absolute_quality_correction",
                "cascade_opportunity_quality_correction",
                "cascade_opportunity_verified_correction",
                "cascade_joint_risk_correction",
                "cascade_v19_fallback_set_correction",
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            raise ValueError(
                "gate_action_mode must be decision, expected_utility, or "
                "direct_utility, hierarchical_utility, pairwise_verifier, "
                "topn_pairwise_verifier, topn_dual_evidence_verifier, or "
                "topn_absolute_quality_delta, or "
                "cascade_absolute_quality_correction, or "
                "cascade_opportunity_quality_correction, or "
                "cascade_opportunity_verified_correction, or "
                "cascade_joint_risk_correction, or "
                "cascade_v19_fallback_set_correction, or "
                "cascade_v19_rich_set_correction, or "
                "cascade_v23_dense_quality_correction, "
                "cascade_v24_relative_risk_correction, or "
                "cascade_v25_pairwise_calibrated_correction, or "
                "cascade_v26_prior_restored_pairwise_correction, or "
                "cascade_v28_selected_abstention_correction, or "
                "cascade_v29_counterfactual_selected_correction, or "
                "cascade_v37_counterfactual_benefit_hazard_correction, or "
                "cascade_v38_complementary_logodds_correction, or "
                "cascade_v39_hazard_residual_correction"
            )
        if gate_action_mode != "decision" and not use_fallback_gate:
            raise ValueError(
                "non-default gate_action_mode requires the fallback gate"
            )
        self.gate_action_mode = str(gate_action_mode)
        self.use_dense_quality_adaptive_mixer = (
            self.gate_action_mode in (
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction",
            )
        )
        self.gate_rich_feature_dim = (
            2 * self.d_model + 24
            if self.gate_action_mode in (
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction",
            )
            else None
        )
        if self.use_dense_quality_adaptive_mixer and self.num_routed < 1:
            raise ValueError(
                "dense-quality adaptive mixing requires a routed source"
            )

        self.text_dim = int(text_dim or d_model)
        self.text_projector = nn.Linear(self.text_dim, self.d_model)
        router_input_dim = (
            self.d_model + self.d_model + 6 + len(source_names)
        )
        self.router_input_dim = router_input_dim
        if self.num_routed > 0:
            self.router = nn.Sequential(
                nn.Linear(router_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_routed),
            )
            # The residual scale below is exactly zero at initialization, so
            # a small random gate does not change baseline predictions.  It
            # does avoid a deterministic top-k tie that dispatches every
            # query to the same expert on the first optimization step.
            nn.init.normal_(self.router[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.router[-1].bias)
            self.routed_scale = nn.Parameter(torch.zeros(1))
        else:
            self.router = None
            self.routed_scale = None
        self.adaptive_source_mixer = None
        if self.use_dense_quality_adaptive_mixer:
            with torch.random.fork_rng(devices=[]):
                self.adaptive_source_mixer = AdaptiveSourceMixer(
                    context_dim=2 * self.d_model + 6,
                    rich_dim=self.gate_rich_feature_dim,
                    hidden_dim=hidden_dim,
                    source_count=len(self.source_names),
                    shared_index=self.source_names.index(self.shared_source),
                )
        self.query_reranker = QueryContextReranker(
            input_dim=router_input_dim,
            hidden_dim=hidden_dim,
            num_heads=query_heads,
            num_layers=query_layers,
            dropout=query_dropout,
            max_delta=query_max_delta,
        )
        self.use_fallback_gate = bool(use_fallback_gate)
        if (not isinstance(gate_context_layers, int)
                or isinstance(gate_context_layers, bool)
                or gate_context_layers < 0):
            raise ValueError(
                "gate_context_layers must be a non-negative integer"
            )
        if gate_context_layers > 0 and not self.use_fallback_gate:
            raise ValueError(
                "gate context layers require the fallback gate"
            )
        if not isinstance(gate_use_evidence_features, bool):
            raise ValueError("gate_use_evidence_features must be boolean")
        self.gate_use_evidence_features = gate_use_evidence_features
        if self.gate_use_evidence_features and not self.use_fallback_gate:
            raise ValueError(
                "gate evidence features require the fallback gate"
            )
        if self.gate_use_evidence_features:
            if (not isinstance(gate_evidence_dim, int)
                    or isinstance(gate_evidence_dim, bool)
                    or gate_evidence_dim < 1):
                raise ValueError(
                    "gate_evidence_dim must be a positive integer when gate "
                    "evidence features are enabled"
                )
            self.gate_evidence_dim = int(gate_evidence_dim)
        else:
            self.gate_evidence_dim = None
        self.fallback_gate = None
        if self.use_fallback_gate:
            gate_query_dim = router_input_dim
            if self.gate_use_evidence_features:
                gate_query_dim += self.gate_evidence_dim + len(source_names)
            self.fallback_gate = QueryFallbackGate(
                query_dim=gate_query_dim,
                hidden_dim=gate_hidden_dim,
                candidate_top_k=gate_candidate_top_k,
                break_cost=gate_break_cost,
                decision_margin=gate_decision_margin,
                mask_utility_weight=gate_mask_utility_weight,
                uncertainty_weight=gate_uncertainty_weight,
                context_layers=gate_context_layers,
                context_heads=gate_context_heads,
                context_dropout=gate_context_dropout,
                action_mode=self.gate_action_mode,
                rich_feature_dim=self.gate_rich_feature_dim,
                adaptive_source_feature_dim=(
                    self.adaptive_source_mixer.feature_dim
                    if self.adaptive_source_mixer is not None else None
                ),
            )

    @staticmethod
    def _pool_text(text_feats, text_mask):
        if text_feats is None:
            return None
        if text_mask is None:
            return text_feats.mean(dim=1)
        valid = (~text_mask.bool()).to(dtype=text_feats.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (text_feats * valid).sum(dim=1) / denom

    @staticmethod
    def _normalize_boxes(candidate_boxes, valid_mask):
        centers = candidate_boxes[..., :3]
        log_sizes = candidate_boxes[..., 3:6].clamp(min=1e-6).log()
        values = torch.cat((centers, log_sizes), dim=-1)
        valid = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (values * valid).sum(dim=1, keepdim=True) / count
        centered = (values - mean) * valid
        variance = centered.square().sum(dim=1, keepdim=True) / count
        return centered / (variance.sqrt() + 1e-5)

    def _normalized_sources(self, source_scores, source_validity, valid_mask,
                            batch_size, num_queries, device, dtype):
        ranks = {}
        for source_index, name in enumerate(self.source_names):
            source_valid = (
                source_validity[..., source_index] & valid_mask
            )
            if name not in source_scores:
                if bool(source_valid.any().item()):
                    raise KeyError("missing valid source score: {}".format(name))
                scores = torch.zeros(
                    batch_size, num_queries, device=device, dtype=dtype
                )
            else:
                scores = source_scores[name]
            if scores.shape != (batch_size, num_queries):
                raise ValueError(
                    "{} scores must have shape {}".format(
                        name, (batch_size, num_queries)
                    )
                )
            scores = torch.nan_to_num(
                scores.to(device=device, dtype=dtype),
                nan=0.0, posinf=1e4, neginf=-1e4,
            )
            normalization_mask = source_valid.clone()
            empty_rows = ~normalization_mask.any(dim=1)
            normalization_mask[empty_rows, 0] = True
            normalized = straight_through_rank_normalize(
                scores, valid_mask=normalization_mask
            )
            ranks[name] = normalized.masked_fill(~source_valid, 0.0)
        return ranks

    def forward(self, candidate_feats, candidate_boxes, source_scores,
                valid_mask=None, text_feats=None, text_mask=None,
                gate_candidate_feats=None, gate_rich_candidate_feats=None,
                source_validity=None):
        if candidate_feats.dim() != 3:
            raise ValueError("candidate_feats must have shape [B,Q,D]")
        batch_size, num_queries, feat_dim = candidate_feats.shape
        if feat_dim != self.d_model:
            raise ValueError("candidate feature dimension does not match d_model")
        if candidate_boxes.shape != (batch_size, num_queries, 6):
            raise ValueError("candidate_boxes must have shape [B,Q,6]")
        device = candidate_feats.device
        dtype = torch.float32
        candidate_feats = candidate_feats.to(dtype)
        candidate_boxes = candidate_boxes.to(dtype)
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, num_queries, dtype=torch.bool, device=device
            )
        elif (valid_mask.dtype != torch.bool
              or valid_mask.shape != (batch_size, num_queries)):
            raise ValueError("valid_mask must be bool with shape [B,Q]")
        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("every sample needs at least one valid query")
        expected_source_validity_shape = (
            batch_size, num_queries, len(self.source_names)
        )
        if source_validity is None:
            if self.use_dense_quality_adaptive_mixer:
                raise ValueError(
                    "V23 requires explicit source_validity [B,Q,S]"
                )
            source_validity = valid_mask.unsqueeze(-1).expand(
                expected_source_validity_shape
            )
        elif (not isinstance(source_validity, torch.Tensor)
              or source_validity.dtype != torch.bool
              or source_validity.shape != expected_source_validity_shape
              or source_validity.device != device):
            raise ValueError("source_validity must be bool with shape [B,Q,S]")
        source_validity = source_validity & valid_mask.unsqueeze(-1)
        shared_source_index = self.source_names.index(self.shared_source)
        shared_validity = source_validity[..., shared_source_index]
        if not bool(shared_validity.any(dim=1).all().item()):
            raise ValueError("every sample needs a valid shared-source query")

        ranks = self._normalized_sources(
            source_scores, source_validity, valid_mask,
            batch_size, num_queries, device, dtype
        )
        shared = ranks[self.shared_source]
        shared_query = shared.masked_fill(~shared_validity, -1e4).argmax(dim=1)
        text_context = self._pool_text(text_feats, text_mask)
        if text_context is None:
            text_context = candidate_feats.new_zeros(batch_size, feat_dim)
        text_context = self.text_projector(text_context.to(dtype))
        text_context = text_context.unsqueeze(1).expand(
            batch_size, num_queries, text_context.shape[-1]
        )
        rank_stack = torch.stack(
            [ranks[name] for name in self.source_names], dim=-1
        )
        normalized_boxes = self._normalize_boxes(candidate_boxes, valid_mask)
        adaptive_context = torch.cat((
            candidate_feats,
            text_context,
            normalized_boxes,
        ), dim=-1)
        router_input = torch.cat((
            adaptive_context,
            rank_stack,
        ), dim=-1)
        gate_query_features = router_input
        if self.gate_use_evidence_features:
            expected_evidence_shape = (
                batch_size, num_queries, self.gate_evidence_dim
            )
            if (not isinstance(gate_candidate_feats, torch.Tensor)
                    or gate_candidate_feats.shape != expected_evidence_shape):
                raise ValueError(
                    "gate_candidate_feats must have shape {}".format(
                        expected_evidence_shape
                    )
                )
            if gate_candidate_feats.device != device:
                raise ValueError(
                    "gate_candidate_feats must share the candidate device"
                )
            gate_candidate_feats = gate_candidate_feats.to(dtype)
            if not bool(torch.isfinite(gate_candidate_feats).all().item()):
                raise ValueError("gate_candidate_feats must be finite")
            raw_source_rows = []
            for source_index, name in enumerate(self.source_names):
                source_valid = source_validity[..., source_index]
                source_value = source_scores.get(name)
                if source_value is None:
                    source_value = torch.zeros(
                        batch_size, num_queries, device=device, dtype=dtype
                    )
                else:
                    source_value = torch.as_tensor(
                        source_value, device=device, dtype=dtype
                    )
                normalization_mask = source_valid.clone()
                empty_rows = ~normalization_mask.any(dim=1)
                normalization_mask[empty_rows, 0] = True
                standardized = standardize_source_scores(
                    source_value, normalization_mask
                ).masked_fill(~source_valid, 0.0)
                raw_source_rows.append(standardized)
            raw_source_features = torch.stack(raw_source_rows, dim=-1)
            gate_query_features = torch.cat((
                router_input,
                gate_candidate_feats,
                raw_source_features,
            ), dim=-1)
        if self.gate_rich_feature_dim is not None:
            expected_rich_shape = (
                batch_size, num_queries, self.gate_rich_feature_dim
            )
            if (not isinstance(gate_rich_candidate_feats, torch.Tensor)
                    or gate_rich_candidate_feats.shape != expected_rich_shape):
                raise ValueError(
                    "gate_rich_candidate_feats must have shape {}".format(
                        expected_rich_shape
                    )
                )
            if gate_rich_candidate_feats.device != device:
                raise ValueError(
                    "gate_rich_candidate_feats must share the candidate device"
                )
            gate_rich_candidate_feats = gate_rich_candidate_feats.to(dtype)
            if not bool(torch.isfinite(
                    gate_rich_candidate_feats
            ).all().item()):
                raise ValueError("gate_rich_candidate_feats must be finite")

        outputs = {
            "moe_source_names": list(self.source_names),
            "moe_shared_source": self.shared_source,
            "moe_shared_query": shared_query,
            "moe_routed_source_names": list(self.routed_source_names),
            "moe_valid_mask": valid_mask,
            "moe_source_validity": source_validity,
        }
        fused = shared
        adaptive_source_features = None
        if self.num_routed > 0:
            gate_logits = self.router(router_input)
            routed_scale = torch.tanh(self.routed_scale)
            if self.adaptive_source_mixer is not None:
                adaptive = self.adaptive_source_mixer(
                    context_features=adaptive_context.detach(),
                    rich_features=gate_rich_candidate_feats,
                    source_ranks=rank_stack.detach(),
                    source_validity=source_validity,
                    base_router_logits=gate_logits,
                    base_scale=routed_scale,
                    top_k=self.top_k,
                )
                fused = adaptive["fused_score"]
                router_probs = adaptive["router_probs"]
                hard_mask = adaptive["expert_mask"]
                adaptive_source_features = adaptive["features"]
                balance_valid = valid_mask & adaptive["has_routed"]
                outputs["moe_query_routed_scale"] = adaptive[
                    "mix_scale"
                ]
                outputs["moe_adaptive_mix_mean"] = adaptive[
                    "mix_scale"
                ][balance_valid].mean().detach() if bool(
                    balance_valid.any().item()
                ) else fused.sum().detach() * 0.0
            else:
                router_probs = F.softmax(gate_logits, dim=-1)
                top_k = min(self.top_k, self.num_routed)
                topk_idx = router_probs.topk(top_k, dim=-1).indices
                hard_mask = torch.zeros_like(router_probs)
                hard_mask.scatter_(-1, topk_idx, 1.0)
                sparse_gate = router_probs * hard_mask
                sparse_gate = sparse_gate / sparse_gate.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-6)
                gate = (
                    sparse_gate.detach()
                    + router_probs - router_probs.detach()
                )
                routed_ranks = torch.stack(
                    [ranks[name] for name in self.routed_source_names], dim=-1
                )
                routed = (gate * routed_ranks).sum(dim=-1)
                fused = shared + routed_scale * (routed - shared)
                balance_valid = valid_mask
            outputs["moe_router_probs"] = router_probs
            outputs["moe_expert_mask"] = hard_mask
            outputs["moe_balance_loss"] = compute_load_balance_loss(
                router_probs, hard_mask, valid_mask=balance_valid
            )
            outputs["moe_routed_scale"] = routed_scale.detach()
            with torch.no_grad():
                entropy = -(
                    router_probs.clamp(min=1e-8).log() * router_probs
                ).sum(dim=-1)
                outputs["moe_router_entropy"] = entropy[valid_mask].mean()
                active = hard_mask[valid_mask].sum(dim=0)
                total = active.sum().clamp(min=1.0)
                for idx, name in enumerate(self.routed_source_names):
                    outputs["moe_expert_usage_{}".format(name)] = (
                        active[idx] / total
                    )
        else:
            outputs["moe_balance_loss"] = fused.sum() * 0.0

        rerank_delta = self.query_reranker(router_input, valid_mask)
        fused = (fused + rerank_delta).masked_fill(~valid_mask, -1e4)
        outputs["moe_candidate_scores"] = fused
        if self.fallback_gate is not None:
            freeze_v19_evidence = self.use_dense_quality_adaptive_mixer
            outputs.update(self.fallback_gate(
                query_features=(
                    gate_query_features.detach()
                    if freeze_v19_evidence else gate_query_features
                ),
                candidate_scores=(
                    fused.detach() if freeze_v19_evidence else fused
                ),
                shared_scores=(
                    shared.detach() if freeze_v19_evidence else shared
                ),
                valid_mask=valid_mask,
                default_indices=shared_query,
                rich_candidate_features=gate_rich_candidate_feats,
                adaptive_source_features=adaptive_source_features,
            ))
        else:
            outputs["selected_source_scores"] = fused
        outputs["moe_rerank_delta"] = rerank_delta
        with torch.no_grad():
            outputs["moe_rerank_abs_mean"] = (
                rerank_delta[valid_mask].abs().mean()
            )
            outputs["moe_rerank_abs_max"] = (
                rerank_delta[valid_mask].abs().max()
            )
        return outputs
