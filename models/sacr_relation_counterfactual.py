"""Relation-specific counterfactual mining and score-only deployment."""

import math

import torch
import torch.nn.functional as F


RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES = (
    "structured_slot_builder.rel_attn.",
    "structured_slot_builder.anchor_attn.",
    "sacr_head.anchor_mlp.",
    "sacr_head.relation_mlp.",
    "sacr_head.geo_encoder.",
)


def _first_map(inputs, key, batch_size, token_dim, device):
    value = inputs.get(key)
    if value is None:
        return torch.zeros(batch_size, token_dim, device=device)
    value = value.float().to(device)
    if value.dim() == 3:
        value = value[:, 0]
    elif value.dim() != 2:
        raise ValueError("{} must have shape [B,T] or [B,G,T]".format(key))
    aligned = torch.zeros(batch_size, token_dim, device=device)
    copy_dim = min(token_dim, value.shape[-1])
    aligned[:, :copy_dim] = value[:, :copy_dim]
    return aligned


def compute_relation_text_affinities(end_points, inputs):
    """Return dataset-agnostic target-class and attribute affinities."""
    semantic_logits = end_points["last_sem_cls_scores"].float()
    batch_size, _query_count, token_dim = semantic_logits.shape
    device = semantic_logits.device
    target_map = _first_map(
        inputs, "positive_map", batch_size, token_dim, device
    )
    attribute_map = _first_map(
        inputs, "modify_positive_map", batch_size, token_dim, device
    )
    probabilities = semantic_logits.softmax(dim=-1)
    target_affinity = torch.matmul(
        probabilities, (target_map > 0).float().unsqueeze(-1)
    ).squeeze(-1)
    attribute_affinity = torch.matmul(
        probabilities, (attribute_map > 0).float().unsqueeze(-1)
    ).squeeze(-1)
    return {
        "target_affinity": target_affinity,
        "attribute_affinity": attribute_affinity,
        "attribute_present": attribute_map.gt(0).any(dim=1),
    }


def _validate_inputs(
        relation_scores, geometry_signatures, relation_candidate_mask,
        target_affinity, attribute_affinity, attribute_present,
        parent_scores, candidate_valid, structured_valid_mask):
    if not isinstance(relation_scores, torch.Tensor) or relation_scores.dim() != 2:
        raise ValueError("relation scores must have shape [B,Q]")
    batch_size, query_count = relation_scores.shape
    matrix_values = (
        target_affinity, attribute_affinity, parent_scores,
    )
    matrix_masks = (relation_candidate_mask, candidate_valid)
    if any(
            not isinstance(value, torch.Tensor)
            or value.shape != (batch_size, query_count)
            for value in matrix_values):
        raise ValueError("counterfactual query tensors must align as [B,Q]")
    if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.shape != (batch_size, query_count)
            for value in matrix_masks):
        raise ValueError("counterfactual query masks must align as [B,Q]")
    if (
            not isinstance(geometry_signatures, torch.Tensor)
            or geometry_signatures.shape != (batch_size, query_count, 11)):
        raise ValueError("relation geometry signatures must have shape [B,Q,11]")
    for value, label in (
            (attribute_present, "attribute presence"),
            (structured_valid_mask, "structured validity")):
        if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.bool
                or value.shape != (batch_size,)):
            raise ValueError("{} must have shape [B]".format(label))
    tensors = matrix_values + matrix_masks + (
        geometry_signatures, attribute_present, structured_valid_mask,
    )
    if any(value.device != relation_scores.device for value in tensors):
        raise ValueError("counterfactual tensors must share a device")


def _top_k_mask(scores, valid_mask, top_k):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("counterfactual parent_top_k must be positive")
    count = min(top_k, scores.shape[1])
    indices = torch.topk(
        scores.float().masked_fill(~valid_mask, torch.finfo(torch.float32).min),
        count,
        dim=1,
    ).indices
    result = torch.zeros_like(valid_mask)
    result.scatter_(1, indices, True)
    return result & valid_mask


def build_relation_counterfactual_context(
        relation_scores, geometry_signatures, relation_candidate_mask,
        target_affinity, attribute_affinity, attribute_present,
        parent_scores, candidate_valid, structured_valid_mask,
        max_delta=0.25, promotion_margin=0.01, parent_top_k=16,
        target_tolerance=0.05, attribute_tolerance=0.05):
    """Build inference-time candidates without dataset subgroup labels."""
    _validate_inputs(
        relation_scores, geometry_signatures, relation_candidate_mask,
        target_affinity, attribute_affinity, attribute_present,
        parent_scores, candidate_valid, structured_valid_mask,
    )
    for name, value, lower in (
            ("max_delta", max_delta, 0.0),
            ("promotion_margin", promotion_margin, 0.0),
            ("target_tolerance", target_tolerance, 0.0),
            ("attribute_tolerance", attribute_tolerance, 0.0)):
        if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < lower
                or (name == "max_delta" and (
                    float(value) == 0.0 or float(value) > 0.25
                ))
                or (name == "promotion_margin" and (
                    float(value) >= float(max_delta)
                ))):
            raise ValueError("invalid relation counterfactual {}".format(name))

    masked_parent = parent_scores.float().masked_fill(
        ~candidate_valid, torch.finfo(torch.float32).min
    )
    parent_indices = masked_parent.argmax(dim=1)
    row = torch.arange(parent_scores.shape[0], device=parent_scores.device)
    parent_score = parent_scores.float()[row, parent_indices]
    parent_relation = relation_scores.float().tanh()[row, parent_indices]
    parent_target = target_affinity.float()[row, parent_indices]
    parent_attribute = attribute_affinity.float()[row, parent_indices]
    bounded_relation = relation_scores.float().tanh()
    relation_advantage = bounded_relation - parent_relation.unsqueeze(1)
    promotion_budget = parent_score.unsqueeze(1) - parent_scores.float()
    feasible = promotion_budget + float(promotion_margin) < float(max_delta)
    same_target = (
        target_affinity.float() >= parent_target.unsqueeze(1) - float(target_tolerance)
    ) & (
        target_affinity.float() >= 0.5 * parent_target.unsqueeze(1)
    )
    attribute_close = (
        (attribute_affinity.float() - parent_attribute.unsqueeze(1)).abs()
        <= float(attribute_tolerance)
    ) | ~attribute_present.unsqueeze(1)
    parent_high = _top_k_mask(parent_scores, candidate_valid, parent_top_k)
    parent_mask = torch.zeros_like(candidate_valid)
    parent_mask[row, parent_indices] = True
    active_rows = (
        structured_valid_mask
        & candidate_valid.any(dim=1)
        & relation_candidate_mask[row, parent_indices]
    )
    candidate_mask = (
        active_rows.unsqueeze(1)
        & candidate_valid
        & relation_candidate_mask
        & ~parent_mask
        & feasible
        & parent_high
        & same_target
        & attribute_close
    )
    return {
        "active_rows": active_rows,
        "parent_indices": parent_indices,
        "parent_mask": parent_mask,
        "bounded_relation_scores": bounded_relation,
        "relation_advantage": relation_advantage,
        "promotion_budget": promotion_budget,
        "feasible_candidate_mask": feasible & ~parent_mask,
        "same_target_mask": same_target,
        "attribute_close_mask": attribute_close,
        "parent_high_mask": parent_high,
        "counterfactual_candidate_mask": candidate_mask,
    }


def apply_relation_counterfactual_refinement(
        relation_scores, geometry_signatures, relation_candidate_mask,
        target_affinity, attribute_affinity, attribute_present,
        parent_scores, candidate_valid, structured_valid_mask,
        parse_confidence, anchor_top1_mass, max_delta=0.25,
        promotion_margin=0.01, parent_top_k=16, target_tolerance=0.05,
        attribute_tolerance=0.05, relation_scale=4.0,
        deployment_threshold=0.05):
    """Promote only relation-supported, parent-feasible counterfactuals."""
    context = build_relation_counterfactual_context(
        relation_scores=relation_scores,
        geometry_signatures=geometry_signatures,
        relation_candidate_mask=relation_candidate_mask,
        target_affinity=target_affinity,
        attribute_affinity=attribute_affinity,
        attribute_present=attribute_present,
        parent_scores=parent_scores,
        candidate_valid=candidate_valid,
        structured_valid_mask=structured_valid_mask,
        max_delta=max_delta,
        promotion_margin=promotion_margin,
        parent_top_k=parent_top_k,
        target_tolerance=target_tolerance,
        attribute_tolerance=attribute_tolerance,
    )
    for name, value in (
            ("relation_scale", relation_scale),
            ("deployment_threshold", deployment_threshold)):
        if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or (name == "relation_scale" and float(value) == 0.0)):
            raise ValueError("invalid relation counterfactual {}".format(name))
    batch_size = relation_scores.shape[0]
    row_values = (parse_confidence, anchor_top1_mass)
    if any(
            not isinstance(value, torch.Tensor)
            or value.shape != (batch_size,)
            or value.device != relation_scores.device
            for value in row_values):
        raise ValueError("counterfactual reliability inputs must align as [B]")
    # Candidate filters carry the safety burden.  A fixed confidence floor
    # avoids recreating V134's learned gate collapse while still reducing
    # weakly parsed rows.
    reliability = 0.75 + 0.25 * torch.sqrt(
        parse_confidence.float().clamp(0.0, 1.0)
        * anchor_top1_mass.float().clamp(0.0, 1.0)
    )
    positive_advantage = (
        context["relation_advantage"] - float(deployment_threshold)
    ).clamp(min=0.0)
    residual = (
        float(max_delta)
        * reliability.unsqueeze(1)
        * torch.tanh(float(relation_scale) * positive_advantage)
    )
    proposal_mask = context["counterfactual_candidate_mask"]
    promotion_mask = proposal_mask & (
        residual
        > context["promotion_budget"] + float(promotion_margin)
    )
    residual = residual.masked_fill(~promotion_mask, 0.0)
    scores = torch.where(
        promotion_mask,
        parent_scores.float() + residual,
        parent_scores.float(),
    )
    return {
        "scores": scores,
        "residual": residual,
        "promotion_mask": promotion_mask,
        "proposal_mask": proposal_mask,
        "reliability": reliability,
        **context,
    }


def compute_relation_counterfactual_loss(
        relation_scores, geometry_signatures, relation_candidate_mask,
        target_affinity, attribute_affinity, attribute_present,
        parent_scores, candidate_valid, structured_valid_mask, box_ious,
        sample_mask=None, mask_ious=None, mask_supervision_mask=None,
        parent_top_k=16, target_tolerance=0.05,
        attribute_tolerance=0.05, geometry_threshold=0.08,
        iou_gap=0.10, correct_iou_threshold=0.25, pair_margin=0.25,
        max_negatives=4, mask_tolerance=0.02):
    """Train correct target/anchor geometry over hard target swaps only."""
    _validate_inputs(
        relation_scores, geometry_signatures, relation_candidate_mask,
        target_affinity, attribute_affinity, attribute_present,
        parent_scores, candidate_valid, structured_valid_mask,
    )
    batch_size, query_count = relation_scores.shape
    if box_ious.shape != (batch_size, query_count):
        raise ValueError("counterfactual Box IoUs must align as [B,Q]")
    if sample_mask is None:
        sample_mask = torch.ones(
            batch_size, dtype=torch.bool, device=relation_scores.device
        )
    if mask_supervision_mask is None:
        mask_supervision_mask = torch.zeros_like(sample_mask)
    if (
            sample_mask.shape != (batch_size,)
            or sample_mask.dtype != torch.bool
            or mask_supervision_mask.shape != (batch_size,)
            or mask_supervision_mask.dtype != torch.bool):
        raise ValueError("counterfactual row masks must align as [B]")
    scalars = (
        ("target_tolerance", target_tolerance),
        ("attribute_tolerance", attribute_tolerance),
        ("geometry_threshold", geometry_threshold),
        ("iou_gap", iou_gap),
        ("correct_iou_threshold", correct_iou_threshold),
        ("pair_margin", pair_margin),
        ("mask_tolerance", mask_tolerance),
    )
    if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for _name, value in scalars):
        raise ValueError("counterfactual loss scalars must be non-negative")
    if (
            not isinstance(max_negatives, int)
            or isinstance(max_negatives, bool)
            or max_negatives < 1):
        raise ValueError("counterfactual max_negatives must be positive")

    valid_relation = candidate_valid & relation_candidate_mask
    masked_ious = box_ious.float().masked_fill(~valid_relation, -1.0)
    correct_ious, correct_indices = masked_ious.max(dim=1)
    row = torch.arange(batch_size, device=relation_scores.device)
    correct_target = target_affinity.float()[row, correct_indices]
    correct_attribute = attribute_affinity.float()[row, correct_indices]
    correct_geometry = geometry_signatures.float()[row, correct_indices]
    correct_mask = torch.zeros_like(candidate_valid)
    correct_mask[row, correct_indices] = True
    same_target = (
        target_affinity.float()
        >= correct_target.unsqueeze(1) - float(target_tolerance)
    ) & (
        target_affinity.float() >= 0.5 * correct_target.unsqueeze(1)
    )
    attribute_close = (
        (attribute_affinity.float() - correct_attribute.unsqueeze(1)).abs()
        <= float(attribute_tolerance)
    ) | ~attribute_present.unsqueeze(1)
    geometry_distance = (
        geometry_signatures.float() - correct_geometry.unsqueeze(1)
    ).abs().mean(dim=-1)
    geometry_inconsistent = geometry_distance >= float(geometry_threshold)
    parent_high = _top_k_mask(parent_scores, candidate_valid, parent_top_k)
    wrong_box = (
        box_ious.float() <= correct_ious.unsqueeze(1) - float(iou_gap)
    )
    active_rows = (
        structured_valid_mask
        & sample_mask
        & (correct_ious >= float(correct_iou_threshold))
        & valid_relation[row, correct_indices]
    )
    if mask_ious is not None:
        if mask_ious.shape != (batch_size, query_count):
            raise ValueError("counterfactual Mask IoUs must align as [B,Q]")
        masked_parent = parent_scores.float().masked_fill(
            ~candidate_valid, torch.finfo(torch.float32).min
        )
        parent_indices = masked_parent.argmax(dim=1)
        mask_safe = (
            mask_ious.float()[row, correct_indices] + float(mask_tolerance)
            >= mask_ious.float()[row, parent_indices]
        )
        active_rows = active_rows & (~mask_supervision_mask | mask_safe)
    hard_negative_mask = (
        active_rows.unsqueeze(1)
        & valid_relation
        & ~correct_mask
        & same_target
        & attribute_close
        & parent_high
        & geometry_inconsistent
        & wrong_box
    )
    bounded_relation = relation_scores.float().tanh()
    hardness = (
        parent_scores.float() + bounded_relation
    ).masked_fill(~hard_negative_mask, torch.finfo(torch.float32).min)
    negative_count = min(max_negatives, query_count)
    negative_indices = torch.topk(hardness, negative_count, dim=1).indices
    selected_mask = torch.gather(
        hard_negative_mask, 1, negative_indices
    )
    negative_scores = torch.gather(
        bounded_relation, 1, negative_indices
    )
    positive_scores = bounded_relation[row, correct_indices].unsqueeze(1)
    pair_losses = F.relu(
        float(pair_margin) - positive_scores + negative_scores
    )
    selected_float = selected_mask.float()
    loss = torch.where(
        selected_mask, pair_losses, torch.zeros_like(pair_losses)
    ).sum() / selected_float.sum().clamp(min=1.0)
    rows_with_negative = hard_negative_mask.any(dim=1)
    parent_indices = parent_scores.float().masked_fill(
        ~candidate_valid, torch.finfo(torch.float32).min
    ).argmax(dim=1)
    parent_is_negative = hard_negative_mask[row, parent_indices]
    zero = loss.detach() * 0.0
    return {
        "loss": loss,
        "active_row_ratio": active_rows.float().mean(),
        "hard_negative_row_ratio": rows_with_negative.float().mean(),
        "hard_negative_count_mean": hard_negative_mask.float().sum(1).mean(),
        "selected_negative_count_mean": selected_float.sum(1).mean(),
        "parent_hard_negative_ratio": parent_is_negative.float().mean(),
        "geometry_inconsistent_ratio": torch.where(
            valid_relation,
            geometry_inconsistent.float(),
            torch.zeros_like(geometry_distance),
        ).sum() / valid_relation.float().sum().clamp(min=1.0),
        "correct_iou_mean": torch.where(
            active_rows, correct_ious, torch.zeros_like(correct_ious)
        ).sum() / active_rows.float().sum().clamp(min=1.0),
        "pair_margin_violation_mean": torch.where(
            selected_mask, pair_losses, torch.zeros_like(pair_losses)
        ).sum() / selected_float.sum().clamp(min=1.0),
        "unused_zero": zero,
    }
