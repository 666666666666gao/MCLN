"""Utilities for selecting and auditing REC query candidates."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .source_choice_selector import _pairwise_iou3d


def _validate_score_inputs(default_scores, contrastive_scores,
                           topk_per_source, max_candidates):
    if default_scores.dim() != 2 or contrastive_scores.dim() != 2:
        raise ValueError("candidate scores must have shape [B,Q]")
    if default_scores.shape != contrastive_scores.shape:
        raise ValueError("default and contrastive scores must have equal shape")
    if default_scores.shape[0] == 0 or default_scores.shape[1] == 0:
        raise ValueError("candidate scores cannot be empty")
    if not isinstance(topk_per_source, int) or topk_per_source <= 0:
        raise ValueError("topk_per_source must be a positive integer")
    if not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if not torch.isfinite(default_scores).all():
        raise ValueError("default scores must be finite")
    if not torch.isfinite(contrastive_scores).all():
        raise ValueError("contrastive scores must be finite")


def _stable_descending_order(scores):
    values = scores.detach().cpu().tolist()
    return sorted(range(len(values)), key=lambda idx: (-values[idx], idx))


def select_candidate_indices(default_scores, contrastive_scores,
                             topk_per_source=8, max_candidates=16):
    """Return a deterministic union of deployable query rankings."""
    _validate_score_inputs(
        default_scores,
        contrastive_scores,
        topk_per_source,
        max_candidates,
    )
    batch_size, _ = default_scores.shape
    device = default_scores.device
    indices = torch.zeros(
        batch_size, max_candidates, dtype=torch.long, device=device
    )
    valid = torch.zeros(
        batch_size, max_candidates, dtype=torch.bool, device=device
    )

    for batch_idx in range(batch_size):
        default_order = _stable_descending_order(default_scores[batch_idx])
        contrastive_order = _stable_descending_order(
            contrastive_scores[batch_idx]
        )
        selected = []
        selected_set = set()
        for query_idx in (
                default_order[:topk_per_source]
                + contrastive_order[:topk_per_source]
                + default_order):
            if query_idx in selected_set:
                continue
            selected.append(query_idx)
            selected_set.add(query_idx)
            if len(selected) == max_candidates:
                break
        if selected:
            count = len(selected)
            indices[batch_idx, :count] = torch.tensor(
                selected, dtype=torch.long, device=device
            )
            valid[batch_idx, :count] = True
    return indices, valid


def _validate_box_inputs(candidate_boxes, gt_boxes, gt_mask):
    if candidate_boxes.dim() != 3 or candidate_boxes.shape[-1] != 6:
        raise ValueError("candidate_boxes must have shape [B,K,6]")
    if gt_boxes.dim() != 3 or gt_boxes.shape[-1] != 6:
        raise ValueError("gt_boxes must have shape [B,G,6]")
    if gt_mask.dim() != 2:
        raise ValueError("gt_mask must have shape [B,G]")
    if candidate_boxes.shape[0] != gt_boxes.shape[0]:
        raise ValueError("candidate and GT batch sizes must match")
    if gt_mask.shape != gt_boxes.shape[:2]:
        raise ValueError("gt_mask shape must match gt_boxes[:2]")


def compute_query_ious(candidate_boxes, gt_boxes, gt_mask):
    """Return each candidate's maximum IoU over valid target boxes."""
    _validate_box_inputs(candidate_boxes, gt_boxes, gt_mask)
    batch_size, num_candidates, _ = candidate_boxes.shape
    output = candidate_boxes.new_zeros(batch_size, num_candidates)
    for batch_idx in range(batch_size):
        valid_gt = gt_mask[batch_idx].bool()
        if not valid_gt.any():
            continue
        output[batch_idx] = _pairwise_iou3d(
            candidate_boxes[batch_idx],
            gt_boxes[batch_idx, valid_gt],
        ).max(dim=1).values
    return output


def compute_candidate_oracle(candidate_ious, valid_mask):
    """Compute strict-threshold Top-1 oracle accuracy for a candidate set."""
    if candidate_ious.dim() != 2 or valid_mask.dim() != 2:
        raise ValueError("candidate_ious and valid_mask must have shape [B,K]")
    if candidate_ious.shape != valid_mask.shape:
        raise ValueError("candidate_ious and valid_mask shapes must match")
    if candidate_ious.shape[0] == 0 or candidate_ious.shape[1] == 0:
        raise ValueError("candidate batches cannot be empty")

    valid = valid_mask.bool()
    masked_ious = candidate_ious.masked_fill(~valid, float("-inf"))
    best_ious = masked_ious.max(dim=1).values
    best_ious = torch.where(
        valid.any(dim=1), best_ious, torch.zeros_like(best_ious)
    )
    return {
        "acc025": (best_ious > 0.25).float().mean().item(),
        "acc050": (best_ious > 0.50).float().mean().item(),
    }


def _masked_rank_normalize(scores, valid_mask):
    valid = valid_mask.bool()
    ordered = scores.masked_fill(~valid, -float("inf")).argsort(
        dim=1, descending=True
    )
    ranks = torch.zeros_like(ordered, dtype=scores.dtype)
    values = torch.arange(
        scores.shape[1], dtype=scores.dtype, device=scores.device
    ).unsqueeze(0).expand_as(ranks)
    ranks.scatter_(1, ordered, values)
    denominator = (valid.sum(dim=1) - 1).clamp(min=1).to(scores.dtype)
    normalized = 1.0 - ranks / denominator.unsqueeze(1)
    return normalized.masked_fill(~valid, -1e4)


def blend_candidate_scores(default_scores, ranking_logits, valid_mask,
                           reranker_weight):
    """Blend scale-free default and learned candidate ranks."""
    if (default_scores.dim() != 2
            or ranking_logits.shape != default_scores.shape
            or valid_mask.shape != default_scores.shape):
        raise ValueError("candidate score tensors must share shape [B,K]")
    weight = float(reranker_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("reranker weight must lie in [0, 1]")
    valid = valid_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every sample needs at least one valid candidate")
    if (not torch.isfinite(default_scores[valid]).all()
            or not torch.isfinite(ranking_logits[valid]).all()):
        raise ValueError("valid candidate scores must be finite")
    default_rank = _masked_rank_normalize(default_scores, valid)
    reranker_rank = _masked_rank_normalize(ranking_logits, valid)
    blended = (1.0 - weight) * default_rank + weight * reranker_rank
    return blended.masked_fill(~valid, -1e4)


def _validate_feature_inputs(features, valid_mask):
    if features.dim() != 3:
        raise ValueError("features must have shape [B,K,D]")
    if valid_mask.dim() != 2 or valid_mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must match features[:2]")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("feature batches cannot be empty")


class QueryReranker(nn.Module):
    """Score compact REC candidates using query and sample context."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        context_dim = input_dim * 3
        output_dim = max(hidden_dim // 2, 1)
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.ranking_head = nn.Linear(output_dim, 1)
        self.threshold_head = nn.Linear(output_dim, 2)
        self.iou_head = nn.Linear(output_dim, 1)

    def forward(self, features, valid_mask):
        _validate_feature_inputs(features, valid_mask)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                "expected feature dimension {}, got {}".format(
                    self.input_dim, features.shape[-1]
                )
            )
        valid = valid_mask.bool()
        weights = valid.unsqueeze(-1).to(features.dtype)
        count = weights.sum(dim=1).clamp(min=1.0)
        feature_mean = (features * weights).sum(dim=1) / count

        masked_features = features.masked_fill(~valid.unsqueeze(-1), -float("inf"))
        feature_max = masked_features.max(dim=1).values
        feature_max = torch.where(
            valid.any(dim=1, keepdim=True),
            feature_max,
            torch.zeros_like(feature_max),
        )
        context = torch.cat([
            features,
            feature_mean.unsqueeze(1).expand_as(features),
            feature_max.unsqueeze(1).expand_as(features),
        ], dim=-1)
        encoded = self.encoder(context)
        ranking_logits = self.ranking_head(encoded).squeeze(-1)
        threshold_logits = self.threshold_head(encoded)
        iou_estimate = torch.sigmoid(self.iou_head(encoded).squeeze(-1))

        ranking_logits = ranking_logits.masked_fill(~valid, -1e4)
        threshold_logits = threshold_logits.masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )
        iou_estimate = iou_estimate.masked_fill(~valid, 0.0)
        return {
            "ranking_logits": ranking_logits,
            "threshold_logits": threshold_logits,
            "iou_estimate": iou_estimate,
        }


def select_listwise_targets(candidate_ious, valid_mask):
    """Select a target query using the two REC thresholds, then IoU."""
    if candidate_ious.dim() != 2 or valid_mask.dim() != 2:
        raise ValueError("candidate_ious and valid_mask must have shape [B,K]")
    if candidate_ious.shape != valid_mask.shape:
        raise ValueError("candidate_ious and valid_mask shapes must match")
    valid = valid_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every sample needs at least one valid candidate")
    quality = (
        candidate_ious
        + (candidate_ious > 0.25).to(candidate_ious.dtype)
        + 2.0 * (candidate_ious > 0.50).to(candidate_ious.dtype)
    )
    return quality.masked_fill(~valid, -float("inf")).argmax(dim=1)


def _best_tier_pairwise_loss(ranking_logits, candidate_ious, valid,
                             return_coverage=False):
    tiers = (
        (candidate_ious > 0.25).to(dtype=torch.long)
        + (candidate_ious > 0.50).to(dtype=torch.long)
    )
    best_tiers = tiers.masked_fill(~valid, -1).max(dim=1).values
    positives = valid & tiers.eq(best_tiers.unsqueeze(1))
    negatives = valid & tiers.lt(best_tiers.unsqueeze(1))
    pair_mask = positives.unsqueeze(2) & negatives.unsqueeze(1)

    safe_logits = ranking_logits.masked_fill(~valid, 0.0)
    pair_losses = F.softplus(
        safe_logits.unsqueeze(1) - safe_logits.unsqueeze(2)
    )
    pair_counts = pair_mask.sum(dim=(1, 2))
    row_losses = (
        pair_losses * pair_mask.to(dtype=pair_losses.dtype)
    ).sum(dim=(1, 2)) / pair_counts.clamp(min=1).to(pair_losses.dtype)
    informative = pair_counts > 0
    if bool(informative.any().item()):
        loss = row_losses[informative].mean()
    else:
        loss = safe_logits.sum() * 0.0
    if not return_coverage:
        return loss
    return loss, {
        "tier_pairwise_informative_rows": informative.sum(),
        "tier_pairwise_pair_count": pair_counts.sum(),
        "tier_pairwise_positive_count": positives.sum(),
        "tier_pairwise_negative_count": negatives.sum(),
    }


def compute_rec_reranker_loss(outputs, candidate_ious, valid_mask,
                              listwise_weight=1.0,
                              threshold_weight=1.0,
                              iou_weight=0.5,
                              tier_pairwise_alpha=0.0):
    """Compute ranking, threshold classification, and IoU regression loss."""
    if (not isinstance(tier_pairwise_alpha, (int, float))
            or isinstance(tier_pairwise_alpha, bool)
            or not math.isfinite(float(tier_pairwise_alpha))
            or not 0.0 <= float(tier_pairwise_alpha) <= 1.0):
        raise ValueError("tier_pairwise_alpha must lie in [0, 1]")
    tier_pairwise_alpha = float(tier_pairwise_alpha)
    valid = valid_mask.bool()
    targets = select_listwise_targets(candidate_ious, valid)
    ranking_logits = outputs["ranking_logits"]
    threshold_logits = outputs["threshold_logits"]
    iou_estimate = outputs["iou_estimate"]
    if ranking_logits.shape != candidate_ious.shape:
        raise ValueError("ranking_logits must match candidate_ious")
    if threshold_logits.shape != candidate_ious.shape + (2,):
        raise ValueError("threshold_logits must have shape [B,K,2]")
    if iou_estimate.shape != candidate_ious.shape:
        raise ValueError("iou_estimate must match candidate_ious")

    loss_listwise = F.cross_entropy(
        ranking_logits.masked_fill(~valid, -1e4), targets
    )
    loss_best_tier_pairwise, tier_pairwise_coverage = (
        _best_tier_pairwise_loss(
            ranking_logits, candidate_ious, valid, return_coverage=True
        )
    )
    loss_ranking = (
        (1.0 - tier_pairwise_alpha) * loss_listwise
        + tier_pairwise_alpha * loss_best_tier_pairwise
    )
    threshold_targets = torch.stack([
        candidate_ious > 0.25,
        candidate_ious > 0.50,
    ], dim=-1).to(threshold_logits.dtype)
    threshold_valid = valid.unsqueeze(-1).expand_as(threshold_targets)
    threshold_loss_values = F.binary_cross_entropy_with_logits(
        threshold_logits,
        threshold_targets,
        reduction="none",
    )
    loss_threshold = (
        threshold_loss_values * threshold_valid.to(threshold_loss_values.dtype)
    ).sum() / threshold_valid.sum().clamp(min=1)
    loss_iou = F.smooth_l1_loss(
        iou_estimate[valid],
        candidate_ious.to(iou_estimate.dtype)[valid],
    )
    loss_total = (
        float(listwise_weight) * loss_ranking
        + float(threshold_weight) * loss_threshold
        + float(iou_weight) * loss_iou
    )
    stats = {
        "loss_listwise": loss_listwise.detach(),
        "loss_best_tier_pairwise": loss_best_tier_pairwise.detach(),
        "loss_ranking": loss_ranking.detach(),
        "loss_threshold": loss_threshold.detach(),
        "loss_iou": loss_iou.detach(),
        "loss_total": loss_total.detach(),
    }
    stats.update({
        name: value.detach()
        for name, value in tier_pairwise_coverage.items()
    })
    return loss_total, stats
