"""Training-only tier supervision for deployed grounding query scores.

The auxiliary objective does not add parameters or change inference.  It mines
query pairs from detached box IoU and text affinity, then sends gradients only
through the already-deployed score tensor.
"""

import math

import torch
import torch.nn.functional as F


def _finite_in_range(name, value, lower, upper):
    if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper):
        raise ValueError(
            "{} must be a finite value in [{}, {}]".format(
                name, lower, upper
            )
        )


def _positive_int(name, value, upper=4096):
    if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= upper):
        raise ValueError("{} must be in [1, {}]".format(name, upper))


def _validate_inputs(
        deployed_scores, box_ious, target_affinity, candidate_valid,
        sample_mask):
    if not isinstance(deployed_scores, torch.Tensor) or deployed_scores.dim() != 2:
        raise ValueError("deployed_scores must have shape [B,Q]")
    batch_size, query_count = deployed_scores.shape
    for name, value in (
            ("box_ious", box_ious),
            ("target_affinity", target_affinity)):
        if (
                not isinstance(value, torch.Tensor)
                or value.shape != (batch_size, query_count)):
            raise ValueError("{} must have shape [B,Q]".format(name))
    if (
            not isinstance(candidate_valid, torch.Tensor)
            or candidate_valid.dtype != torch.bool
            or candidate_valid.shape != (batch_size, query_count)):
        raise ValueError("candidate_valid must be a boolean [B,Q] tensor")
    if (
            not isinstance(sample_mask, torch.Tensor)
            or sample_mask.dtype != torch.bool
            or sample_mask.shape != (batch_size,)):
        raise ValueError("sample_mask must be a boolean [B] tensor")
    values = (
        box_ious, target_affinity, candidate_valid, sample_mask,
    )
    if any(value.device != deployed_scores.device for value in values):
        raise ValueError("tier hard-query tensors must share one device")
    if not bool(torch.isfinite(deployed_scores).all().item()):
        raise ValueError("deployed_scores must be finite")
    if (
            not bool(torch.isfinite(box_ious[candidate_valid]).all().item())
            or not bool(torch.isfinite(
                target_affinity[candidate_valid]
            ).all().item())):
        raise ValueError(
            "active tier hard-query mining inputs must be finite"
        )


def _masked_row_mean(values, row_mask):
    row_float = row_mask.float()
    return (
        (values * row_float).sum() / row_float.sum().clamp(min=1.0)
    )


def compute_tier_hard_query_auxiliary_loss(
        deployed_scores, box_ious, target_affinity, candidate_valid,
        sample_mask, candidate_top_k=128, max_negatives=8,
        target_tolerance=0.15, target_confidence_floor=0.01,
        pair_margin=0.05, preserve_weight=0.25,
        acc025_pair_weight=2.0):
    """Supervise threshold tiers without changing the deployed score path.

    A detector-valid teacher is the highest deployed-score query in the best
    IoU tier among candidates whose text affinity is close to the row maximum.
    A row is a repair when this teacher is in a higher tier than deployed Top-1.
    Rows already selecting the teacher (or a strictly higher-tier query) are
    preservation rows.  Different candidates in the same tier are intentionally
    ignored because swapping them cannot improve Acc@0.25/0.50.

    Mining is detached.  Gradients flow only through selected positive and
    lower-tier negative entries of ``deployed_scores``.
    """
    _validate_inputs(
        deployed_scores, box_ious, target_affinity, candidate_valid,
        sample_mask,
    )
    _positive_int("candidate_top_k", candidate_top_k)
    _positive_int("max_negatives", max_negatives)
    for name, value, lower, upper in (
            ("target_tolerance", target_tolerance, 0.0, 1.0),
            ("target_confidence_floor", target_confidence_floor, 0.0, 1.0),
            ("pair_margin", pair_margin, 0.0, 1.0),
            ("preserve_weight", preserve_weight, 0.0, 1.0),
            ("acc025_pair_weight", acc025_pair_weight, 1.0, 10.0)):
        _finite_in_range(name, value, lower, upper)

    batch_size, query_count = deployed_scores.shape
    row = torch.arange(batch_size, device=deployed_scores.device)
    with torch.no_grad():
        detached_scores = deployed_scores.detach().float()
        detached_ious = box_ious.detach().float().clamp(0.0, 1.0)
        detached_affinity = target_affinity.detach().float().clamp(0.0, 1.0)
        valid = candidate_valid.detach()
        sample_rows = sample_mask.detach()
        active = sample_rows & valid.any(dim=1)

        row_affinity_max = detached_affinity.masked_fill(
            ~valid, -1.0
        ).max(dim=1).values
        detector_valid = (
            valid
            & active.unsqueeze(1)
            & (
                detached_affinity
                >= row_affinity_max.unsqueeze(1) - float(target_tolerance)
            )
            & (detached_affinity >= float(target_confidence_floor))
        )
        teacher_available = detector_valid.any(dim=1)
        tiers = (
            detached_ious.gt(0.25).long()
            + detached_ious.gt(0.50).long()
        )
        best_teacher_tier = tiers.masked_fill(
            ~detector_valid, -1
        ).max(dim=1).values
        best_tier_candidates = (
            detector_valid
            & tiers.eq(best_teacher_tier.unsqueeze(1))
        )
        teacher_indices = detached_scores.masked_fill(
            ~best_tier_candidates,
            torch.finfo(torch.float32).min,
        ).argmax(dim=1)

        parent_indices = detached_scores.masked_fill(
            ~valid, torch.finfo(torch.float32).min
        ).argmax(dim=1)
        teacher_tiers = tiers[row, teacher_indices]
        parent_tiers = tiers[row, parent_indices]
        teacher_is_parent = teacher_indices.eq(parent_indices)

        repair_rows = (
            active & teacher_available & teacher_tiers.gt(parent_tiers)
        )
        preserve_rows = (
            active
            & teacher_available
            & (
                teacher_is_parent
                | parent_tiers.gt(teacher_tiers)
            )
            & parent_tiers.gt(0)
        )
        same_tier_rows = (
            active
            & teacher_available
            & teacher_tiers.eq(parent_tiers)
            & ~preserve_rows
        )
        empty_rows = sample_rows & ~teacher_available

        positive_indices = torch.where(
            repair_rows, teacher_indices, parent_indices
        )
        positive_tiers = tiers[row, positive_indices]
        supervised_rows = repair_rows | preserve_rows

        top_k_count = min(candidate_top_k, query_count)
        top_k_indices = torch.topk(
            detached_scores.masked_fill(
                ~valid, torch.finfo(torch.float32).min
            ),
            top_k_count,
            dim=1,
        ).indices
        top_k_mask = torch.zeros_like(valid)
        top_k_mask.scatter_(1, top_k_indices, True)
        top_k_mask &= valid

        negative_mask = (
            supervised_rows.unsqueeze(1)
            & top_k_mask
            & tiers.lt(positive_tiers.unsqueeze(1))
        )
        negative_count = min(max_negatives, query_count)
        negative_indices = torch.topk(
            detached_scores.masked_fill(
                ~negative_mask, torch.finfo(torch.float32).min
            ),
            negative_count,
            dim=1,
        ).indices
        selected_mask = torch.gather(
            negative_mask, 1, negative_indices
        )
        selected_ious = torch.gather(
            detached_ious, 1, negative_indices
        )
        coarse_break = (
            positive_tiers.unsqueeze(1).gt(0)
            & selected_ious.le(0.25)
        )

    positive_scores = deployed_scores.float()[
        row, positive_indices
    ].unsqueeze(1)
    negative_scores = torch.gather(
        deployed_scores.float(), 1, negative_indices
    )
    pair_losses = F.relu(
        float(pair_margin) - positive_scores + negative_scores
    )
    pair_weights = torch.where(
        coarse_break,
        pair_losses.new_full((), float(acc025_pair_weight)),
        pair_losses.new_ones(()),
    )
    weighted_mask = selected_mask.float() * pair_weights
    row_denominator = selected_mask.float().sum(dim=1).clamp(min=1.0)
    row_losses = (
        pair_losses * weighted_mask
    ).sum(dim=1) / row_denominator

    repair_loss = _masked_row_mean(row_losses, repair_rows)
    preserve_loss = _masked_row_mean(row_losses, preserve_rows)
    loss = repair_loss + float(preserve_weight) * preserve_loss

    selected_float = selected_mask.float()
    selected_count = selected_float.sum().clamp(min=1.0)
    violating = pair_losses.detach().gt(0).float() * selected_float
    repair_count = repair_rows.float().sum().clamp(min=1.0)
    preserve_count = preserve_rows.float().sum().clamp(min=1.0)
    group_row_scale = (
        repair_rows.float() / repair_count
        + float(preserve_weight) * preserve_rows.float() / preserve_count
    )
    pair_objective_scale = (
        selected_float
        * pair_weights.detach()
        / row_denominator.detach().unsqueeze(1)
        * group_row_scale.unsqueeze(1)
    )
    selected_score_gradient_l1 = (
        2.0 * violating * pair_objective_scale
    ).sum()
    sample_count = sample_rows.float().sum().clamp(min=1.0)
    teacher_count = teacher_available.float().sum().clamp(min=1.0)
    stats = {
        "loss": loss.reshape(()),
        "repair_loss": repair_loss.detach(),
        "preserve_loss": preserve_loss.detach(),
        "sample_row_ratio": sample_rows.float().mean(),
        "active_row_ratio": active.float().sum() / sample_count,
        "detector_valid_row_ratio": (
            teacher_available.float().sum() / sample_count
        ),
        "repair_row_ratio": repair_rows.float().sum() / sample_count,
        "preserve_row_ratio": preserve_rows.float().sum() / sample_count,
        "same_tier_row_ratio": same_tier_rows.float().sum() / sample_count,
        "empty_row_ratio": empty_rows.float().sum() / sample_count,
        "selected_negative_count_mean": (
            selected_float.sum(dim=1) * supervised_rows.float()
        ).sum() / supervised_rows.float().sum().clamp(min=1.0),
        "pair_violation_ratio": violating.sum() / selected_count,
        "coarse_break_selected_ratio": (
            coarse_break.float() * selected_float
        ).sum() / selected_count,
        "selected_score_gradient_l1": selected_score_gradient_l1,
        "parent_acc025": (
            parent_tiers.gt(0).float() * active.float()
        ).sum() / sample_count,
        "parent_acc050": (
            parent_tiers.gt(1).float() * active.float()
        ).sum() / sample_count,
        "teacher_oracle_acc025": (
            teacher_tiers.gt(0).float() * teacher_available.float()
        ).sum() / sample_count,
        "teacher_oracle_acc050": (
            teacher_tiers.gt(1).float() * teacher_available.float()
        ).sum() / sample_count,
        "teacher_oracle_conditional_acc025": (
            teacher_tiers.gt(0).float() * teacher_available.float()
        ).sum() / teacher_count,
        "teacher_oracle_conditional_acc050": (
            teacher_tiers.gt(1).float() * teacher_available.float()
        ).sum() / teacher_count,
    }
    return {
        key: value if key == "loss" else value.detach()
        for key, value in stats.items()
    }
