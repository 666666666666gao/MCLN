"""Train-only relation counterfactual hard-negative supervision.

Ground-truth target/anchor geometry is used only to mine reliable training
pairs. The loss acts on the already deployed query-score tensor and therefore
adds no inference branch, parameters, or dataset-specific subgroup labels.
"""

import math

import torch
import torch.nn.functional as F


_DIRECTIONAL_RELATION_AXES = {
    "on the left of": 0,
    "on the right of": 0,
    "behind": 1,
    "in front of": 1,
    "above": 2,
    "below": 2,
}
_DIRECTIONAL_RELATION_SIGNS = {
    "on the left of": -1.0,
    "on the right of": 1.0,
    "behind": 1.0,
    "in front of": -1.0,
    "above": 1.0,
    "below": -1.0,
}
_VIEW_DEPENDENT_HORIZONTAL_RELATIONS = frozenset((
    "on the left of",
    "on the right of",
    "behind",
    "in front of",
))
_SUPPORTED_RELATIONS = frozenset(
    tuple(_DIRECTIONAL_RELATION_AXES)
    + ("near", "far from", "on")
)


def _finite_in_range(name, value, lower, upper):
    if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper):
        raise ValueError(
            "{} must be finite in [{}, {}]".format(name, lower, upper)
        )


def _top_k_mask(scores, valid_mask, top_k):
    if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 4096):
        raise ValueError("parent_top_k must be in [1, 4096]")
    count = min(top_k, scores.shape[1])
    masked = scores.float().masked_fill(
        ~valid_mask, torch.finfo(torch.float32).min
    )
    indices = torch.topk(masked, count, dim=1).indices
    result = torch.zeros_like(valid_mask)
    result.scatter_(1, indices, True)
    return result & valid_mask


def _as_anchor_boxes(anchor_boxes, batch_size):
    if not isinstance(anchor_boxes, torch.Tensor):
        raise ValueError("anchor_boxes must be a tensor")
    if anchor_boxes.dim() == 2:
        if anchor_boxes.shape != (batch_size, 6):
            raise ValueError("anchor_boxes must have shape [B,A,6] or [B,6]")
        anchor_boxes = anchor_boxes.unsqueeze(1)
    if (
            anchor_boxes.dim() != 3
            or anchor_boxes.shape[0] != batch_size
            or anchor_boxes.shape[1] < 1
            or anchor_boxes.shape[2] != 6):
        raise ValueError("anchor_boxes must have shape [B,A,6] or [B,6]")
    return anchor_boxes


def _as_anchor_valid(anchor_valid, anchor_boxes):
    expected = anchor_boxes.shape[:2]
    if anchor_valid is None:
        return torch.ones(
            expected, dtype=torch.bool, device=anchor_boxes.device
        )
    if (
            not isinstance(anchor_valid, torch.Tensor)
            or anchor_valid.dtype != torch.bool
            or anchor_valid.shape != expected):
        raise ValueError("anchor_valid must be bool with shape [B,A]")
    return anchor_valid


def _as_conservative_rows(conservative_rows, reference):
    batch_size = reference.shape[0]
    if conservative_rows is None:
        return torch.zeros(
            batch_size, dtype=torch.bool, device=reference.device
        )
    if (
            not isinstance(conservative_rows, torch.Tensor)
            or conservative_rows.dtype != torch.bool
            or conservative_rows.shape != (batch_size,)):
        raise ValueError("conservative_rows must be bool with shape [B]")
    if conservative_rows.device != reference.device:
        raise ValueError("conservative_rows must share the reference device")
    return conservative_rows


def resolve_train_only_relation_anchors(
        pseudo_anchor_boxes, gt_boxes, gt_valid, scene_boxes,
        scene_class_ids, scene_valid, target_ids, sample_datasets,
        match_tolerance=1e-4, conservative_anchor_set=False):
    """Resolve exact Sr3D anchors and pseudo-anchor classes elsewhere.

    Sr3D exposes the annotated anchor in GT slot one. Nr3D and ScanRefer do not
    expose an anchor identity, so their nearest-class pseudo anchor is trusted
    only when exactly one non-target scene object has the matched class by
    default. The opt-in conservative mode returns every non-target scene object
    of that parsed anchor class; downstream predicates keep a negative only
    when it violates the relation for every plausible anchor instance.
    """
    if not isinstance(gt_boxes, torch.Tensor) or gt_boxes.dim() != 3:
        raise ValueError("gt_boxes must have shape [B,G,6]")
    batch_size = gt_boxes.shape[0]
    pseudo_anchor_boxes = _as_anchor_boxes(
        pseudo_anchor_boxes, batch_size
    )
    if pseudo_anchor_boxes.shape[1] != 1:
        raise ValueError(
            "pseudo_anchor_boxes must contain exactly one parsed anchor"
        )
    if gt_boxes.shape[-1] != 6:
        raise ValueError("gt_boxes must have shape [B,G,6]")
    if (
            not isinstance(gt_valid, torch.Tensor)
            or gt_valid.dtype != torch.bool
            or gt_valid.shape != gt_boxes.shape[:2]):
        raise ValueError("gt_valid must be bool with shape [B,G]")
    if (
            not isinstance(scene_boxes, torch.Tensor)
            or scene_boxes.dim() != 3
            or scene_boxes.shape[0] != batch_size
            or scene_boxes.shape[-1] != 6):
        raise ValueError("scene_boxes must have shape [B,N,6]")
    scene_shape = scene_boxes.shape[:2]
    if (
            not isinstance(scene_class_ids, torch.Tensor)
            or scene_class_ids.shape != scene_shape
            or not isinstance(scene_valid, torch.Tensor)
            or scene_valid.dtype != torch.bool
            or scene_valid.shape != scene_shape):
        raise ValueError("scene class ids and validity must align as [B,N]")
    if not isinstance(target_ids, torch.Tensor):
        raise ValueError("target_ids must be a tensor")
    target_ids = target_ids.reshape(batch_size, -1)
    if target_ids.shape[1] < 1:
        raise ValueError("target_ids must contain one id per row")
    if isinstance(sample_datasets, str):
        sample_datasets = [sample_datasets] * batch_size
    if (
            not isinstance(sample_datasets, (list, tuple))
            or len(sample_datasets) != batch_size
            or any(not isinstance(name, str) for name in sample_datasets)):
        raise ValueError("sample_datasets must align with the batch")
    tensors = (
        pseudo_anchor_boxes, gt_boxes, gt_valid, scene_boxes,
        scene_class_ids, scene_valid, target_ids,
    )
    if any(value.device != gt_boxes.device for value in tensors):
        raise ValueError("anchor-resolution tensors must share one device")
    _finite_in_range(
        "anchor match_tolerance", match_tolerance, 0.0, 0.01
    )
    if not isinstance(conservative_anchor_set, bool):
        raise ValueError("conservative_anchor_set must be bool")

    if conservative_anchor_set:
        resolved = torch.zeros_like(scene_boxes.detach().float())
        anchor_valid = torch.zeros_like(scene_valid)
    else:
        resolved = pseudo_anchor_boxes.detach().float().clone()
        anchor_valid = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=gt_boxes.device
        )
    reliable = torch.zeros(
        batch_size, dtype=torch.bool, device=gt_boxes.device
    )
    exact_gt = torch.zeros_like(reliable)
    unique_pseudo = torch.zeros_like(reliable)
    conservative_set = torch.zeros_like(reliable)
    conservative_rows = torch.zeros_like(reliable)
    with torch.no_grad():
        for batch_index, dataset_name in enumerate(sample_datasets):
            dataset_name = dataset_name.strip().lower()
            if (
                    dataset_name.startswith("sr3d")
                    and gt_boxes.shape[1] > 1
                    and bool(gt_valid[batch_index, 1].item())):
                resolved[batch_index, 0] = gt_boxes[
                    batch_index, 1
                ].detach().float()
                anchor_valid[batch_index, 0] = True
                reliable[batch_index] = True
                exact_gt[batch_index] = True
                continue
            if dataset_name not in ("nr3d", "scanrefer"):
                continue
            if conservative_anchor_set:
                conservative_rows[batch_index] = True

            pseudo = pseudo_anchor_boxes[batch_index, 0].detach().float()
            if (
                    not bool(torch.isfinite(pseudo).all().item())
                    or not bool(pseudo[3:].abs().gt(1e-4).all().item())):
                continue
            distances = (
                scene_boxes[batch_index].detach().float() - pseudo
            ).abs().amax(dim=-1)
            distances = distances.masked_fill(
                ~scene_valid[batch_index], float("inf")
            )
            matched_index = int(distances.argmin().item())
            if float(distances[matched_index].item()) > float(match_tolerance):
                continue
            matched_class = scene_class_ids[
                batch_index, matched_index
            ]
            same_class = (
                scene_valid[batch_index]
                & scene_class_ids[batch_index].eq(matched_class)
            )
            target_index = int(target_ids[batch_index, 0].item())
            if 0 <= target_index < same_class.shape[0]:
                same_class[target_index] = False
            if (
                    int(same_class.sum().item()) == 1
                    and bool(same_class[matched_index].item())):
                unique_pseudo[batch_index] = True
            if conservative_anchor_set:
                resolved[batch_index, same_class] = scene_boxes[
                    batch_index, same_class
                ].detach().float()
                anchor_valid[batch_index] = same_class
                reliable[batch_index] = bool(same_class.any().item())
                conservative_set[batch_index] = (
                    int(same_class.sum().item()) > 1
                )
            elif bool(unique_pseudo[batch_index].item()):
                anchor_valid[batch_index, 0] = True
                reliable[batch_index] = True

        finite_size = (
            torch.isfinite(resolved).all(dim=-1)
            & resolved[..., 3:].abs().gt(1e-4).all(dim=-1)
        )
        anchor_valid &= finite_size
        reliable &= anchor_valid.any(dim=1)
        exact_gt &= reliable
        unique_pseudo &= reliable
        conservative_set &= reliable
        anchor_count = anchor_valid.float().sum(dim=1)
    return {
        "anchor_boxes": resolved,
        "anchor_valid_mask": anchor_valid,
        "reliable_mask": reliable,
        "exact_gt_ratio": exact_gt.float().mean(),
        "unique_pseudo_ratio": unique_pseudo.float().mean(),
        "conservative_anchor_set_ratio": conservative_set.float().mean(),
        "anchor_candidate_count_mean": anchor_count.mean(),
        "conservative_row_mask": conservative_rows,
    }


def _pairwise_axis_aligned_iou(candidate_boxes, anchor_boxes):
    candidate_size = candidate_boxes[..., 3:].abs().clamp(min=1e-6)
    anchor_size = anchor_boxes[..., 3:].abs().clamp(min=1e-6)
    candidate_min = candidate_boxes[..., :3] - 0.5 * candidate_size
    candidate_max = candidate_boxes[..., :3] + 0.5 * candidate_size
    anchor_min = anchor_boxes[..., :3] - 0.5 * anchor_size
    anchor_max = anchor_boxes[..., :3] + 0.5 * anchor_size
    overlap_min = torch.maximum(
        candidate_min[:, :, None], anchor_min[:, None]
    )
    overlap_max = torch.minimum(
        candidate_max[:, :, None], anchor_max[:, None]
    )
    intersection = (overlap_max - overlap_min).clamp(min=0.0).prod(dim=-1)
    candidate_volume = candidate_size.prod(dim=-1)
    anchor_volume = anchor_size.prod(dim=-1)
    union = (
        candidate_volume[:, :, None]
        + anchor_volume[:, None]
        - intersection
    ).clamp(min=1e-6)
    return (intersection / union).clamp(min=0.0, max=1.0)


def build_relation_predicate_masks(
        candidate_boxes, target_boxes, anchor_boxes, relation_labels,
        predicate_margin=0.08, anchor_valid=None, conservative_rows=None):
    """Return conservative relation-specific candidate inconsistency masks.

    A row may contain multiple same-class anchor hypotheses. An inferred hard
    negative must violate the target relation for every reference-valid anchor.
    In multi-anchor-set mode, a candidate overlapping an anchor at IoU > 0.5
    is not compared against itself; this fixed identity guard is part of
    mining, not a tuned threshold. Single-anchor legacy behavior is unchanged.
    """
    if (
            not isinstance(candidate_boxes, torch.Tensor)
            or candidate_boxes.dim() != 3
            or candidate_boxes.shape[-1] != 6):
        raise ValueError("candidate_boxes must have shape [B,Q,6]")
    batch_size, query_count = candidate_boxes.shape[:2]
    if (
            not isinstance(target_boxes, torch.Tensor)
            or target_boxes.shape != (batch_size, 6)):
        raise ValueError("target_boxes must have shape [B,6]")
    anchor_boxes = _as_anchor_boxes(anchor_boxes, batch_size)
    anchor_valid = _as_anchor_valid(anchor_valid, anchor_boxes)
    conservative_rows = _as_conservative_rows(
        conservative_rows, candidate_boxes
    )
    if isinstance(relation_labels, str):
        relation_labels = [relation_labels] * batch_size
    if (
            not isinstance(relation_labels, (list, tuple))
            or len(relation_labels) != batch_size
            or any(not isinstance(value, str) for value in relation_labels)):
        raise ValueError("relation_labels must align with the batch")
    if (
            target_boxes.device != candidate_boxes.device
            or anchor_boxes.device != candidate_boxes.device
            or anchor_valid.device != candidate_boxes.device
            or conservative_rows.device != candidate_boxes.device):
        raise ValueError("relation predicate boxes must share one device")
    _finite_in_range(
        "predicate_margin", predicate_margin, 0.0, 2.0
    )

    candidates = candidate_boxes.detach().float()
    targets = target_boxes.detach().float()
    anchors = anchor_boxes.detach().float()
    valid_anchors = anchor_valid.detach()
    anchor_size = anchors[..., 3:].abs().clamp(min=1e-6)
    anchor_diagonal = anchor_size.square().sum(dim=-1).sqrt().clamp(min=1e-6)
    candidate_delta = (
        candidates[:, :, None, :3] - anchors[:, None, :, :3]
    ) / anchor_diagonal[:, None, :, None]
    target_delta = (
        targets[:, None, :3] - anchors[:, :, :3]
    ) / anchor_diagonal[:, :, None]
    candidate_distance = candidate_delta.square().sum(dim=-1).sqrt()
    target_distance = target_delta.square().sum(dim=-1).sqrt()

    candidate_bottom = candidates[..., 2] - 0.5 * candidates[..., 5].abs()
    anchor_top = anchors[..., 2] + 0.5 * anchors[..., 5].abs()
    target_bottom = targets[:, 2] - 0.5 * targets[:, 5].abs()
    candidate_support_gap = (
        candidate_bottom[:, :, None] - anchor_top[:, None, :]
    ).abs() / anchor_diagonal[:, None, :]
    target_support_gap = (
        target_bottom[:, None] - anchor_top
    ).abs() / anchor_diagonal
    candidate_anchor_iou = _pairwise_axis_aligned_iou(candidates, anchors)

    inconsistent = torch.zeros(
        batch_size, query_count, dtype=torch.bool,
        device=candidate_boxes.device,
    )
    reference_valid = torch.zeros(
        batch_size, dtype=torch.bool, device=candidate_boxes.device
    )
    disagreement = torch.zeros(
        batch_size, query_count, dtype=torch.float32,
        device=candidate_boxes.device,
    )
    reference_anchor_count = torch.zeros(
        batch_size, dtype=torch.float32, device=candidate_boxes.device
    )
    ambiguous_reference = torch.zeros_like(reference_valid)
    self_excluded_count = candidate_anchor_iou.new_zeros(())
    reference_comparison_count = candidate_anchor_iou.new_zeros(())
    margin = float(predicate_margin)
    for batch_index, relation_name in enumerate(relation_labels):
        relation_name = relation_name.strip().lower()
        if relation_name not in _SUPPORTED_RELATIONS:
            continue
        conservative_row = bool(conservative_rows[batch_index].item())
        if (
                conservative_row
                and relation_name not in _DIRECTIONAL_RELATION_AXES):
            # Near/far/on need dataset-dependent metric/contact thresholds to
            # certify the GT relation. Fail closed instead of mining from an
            # anchor that merely has the parsed class. The legacy single-
            # anchor path below remains byte-for-byte behavior compatible.
            continue
        relation_inconsistent = torch.zeros(
            query_count, anchors.shape[1], dtype=torch.bool,
            device=candidate_boxes.device,
        )
        relation_disagreement = torch.zeros_like(
            relation_inconsistent, dtype=torch.float32
        )
        valid_reference = valid_anchors[batch_index].clone()
        if relation_name in _DIRECTIONAL_RELATION_AXES:
            if (
                    conservative_row
                    and relation_name
                    in _VIEW_DEPENDENT_HORIZONTAL_RELATIONS):
                # Nr3D viewpoints are not aligned with world x/y. Treat the
                # GT target direction as a rotation-invariant demonstration
                # of the parsed horizontal relation for every plausible
                # anchor. A negative must disagree with every demonstration.
                reference_horizontal = target_delta[batch_index, :, :2]
                reference_norm = reference_horizontal.square().sum(
                    dim=-1
                ).sqrt()
                valid_reference &= reference_norm > margin
                reference_direction = reference_horizontal / (
                    reference_norm.unsqueeze(-1).clamp(min=1e-6)
                )
                signed_candidate = (
                    candidate_delta[batch_index, :, :, :2]
                    * reference_direction.unsqueeze(0)
                ).sum(dim=-1)
                relation_inconsistent = signed_candidate <= margin
                relation_disagreement = (
                    margin - signed_candidate
                ).clamp(min=0.0)
            else:
                axis = _DIRECTIONAL_RELATION_AXES[relation_name]
                reference = target_delta[batch_index, :, axis]
                if conservative_row:
                    expected_sign = _DIRECTIONAL_RELATION_SIGNS[relation_name]
                    valid_reference &= reference * expected_sign > margin
                    reference_sign = reference.new_full(
                        reference.shape, expected_sign
                    )
                else:
                    valid_reference &= reference.abs() > margin
                    reference_sign = torch.sign(reference)
                signed_candidate = (
                    candidate_delta[batch_index, :, :, axis]
                    * reference_sign.unsqueeze(0)
                )
                relation_inconsistent = signed_candidate <= margin
                relation_disagreement = (
                    margin - signed_candidate
                ).clamp(min=0.0)
        elif relation_name == "near":
            boundary = target_distance[batch_index].unsqueeze(0) + margin
            relation_inconsistent = (
                candidate_distance[batch_index] >= boundary
            )
            relation_disagreement = (
                candidate_distance[batch_index] - boundary
            ).clamp(min=0.0)
        elif relation_name == "far from":
            valid_reference &= target_distance[batch_index] > margin
            boundary = (
                target_distance[batch_index] - margin
            ).clamp(min=0.0).unsqueeze(0)
            relation_inconsistent = (
                candidate_distance[batch_index] <= boundary
            )
            relation_disagreement = (
                boundary - candidate_distance[batch_index]
            ).clamp(min=0.0)
        elif relation_name == "on":
            valid_reference &= target_delta[batch_index, :, 2] > margin
            gap_boundary = (
                target_support_gap[batch_index] + margin
            ).unsqueeze(0)
            wrong_side = candidate_delta[batch_index, :, :, 2] <= margin
            wrong_support = (
                candidate_support_gap[batch_index] >= gap_boundary
            )
            relation_inconsistent = wrong_side | wrong_support
            side_strength = (
                margin - candidate_delta[batch_index, :, :, 2]
            ).clamp(min=0.0)
            support_strength = (
                candidate_support_gap[batch_index] - gap_boundary
            ).clamp(min=0.0)
            relation_disagreement = torch.maximum(
                side_strength, support_strength
            )
        if not bool(valid_reference.any().item()):
            continue
        if bool(conservative_rows[batch_index].item()):
            usable_reference = (
                valid_reference.unsqueeze(0)
                & candidate_anchor_iou[batch_index].le(0.5)
            )
        else:
            usable_reference = valid_reference.unsqueeze(0).expand(
                query_count, -1
            )
        has_usable_reference = usable_reference.any(dim=1)
        robust_inconsistent = (
            relation_inconsistent | ~usable_reference
        ).all(dim=1) & has_usable_reference
        strength = relation_disagreement.masked_fill(
            ~usable_reference, float("inf")
        ).min(dim=1).values
        strength = torch.where(
            robust_inconsistent, strength, torch.zeros_like(strength)
        )
        inconsistent[batch_index] = robust_inconsistent
        disagreement[batch_index] = strength
        reference_valid[batch_index] = True
        reference_anchor_count[batch_index] = valid_reference.float().sum()
        ambiguous_reference[batch_index] = valid_reference.sum() > 1
        if bool(conservative_rows[batch_index].item()):
            self_excluded_count += (
                valid_reference.unsqueeze(0)
                & candidate_anchor_iou[batch_index].gt(0.5)
            ).float().sum()
        reference_comparison_count += (
            valid_reference.float().sum() * float(query_count)
        )
    return {
        "inconsistent_mask": inconsistent,
        "reference_valid_mask": reference_valid,
        "disagreement_strength": disagreement,
        "reference_anchor_count": reference_anchor_count,
        "ambiguous_reference_mask": ambiguous_reference,
        "anchor_self_exclusion_ratio": (
            self_excluded_count / reference_comparison_count.clamp(min=1.0)
        ),
    }


def _validate_inputs(
        deployed_scores, candidate_boxes, candidate_valid, box_ious,
        target_boxes, anchor_boxes, anchor_reliable, relation_labels,
        target_affinity, attribute_affinity, attribute_present,
        anchor_text_present, relation_text_present, sample_mask,
        anchor_valid, conservative_rows):
    if not isinstance(deployed_scores, torch.Tensor) or deployed_scores.dim() != 2:
        raise ValueError("deployed_scores must have shape [B,Q]")
    batch_size, query_count = deployed_scores.shape
    matrices = (target_affinity, attribute_affinity, box_ious)
    if any(
            not isinstance(value, torch.Tensor)
            or value.shape != (batch_size, query_count)
            for value in matrices):
        raise ValueError("query matrices must align with deployed_scores")
    if (
            not isinstance(candidate_valid, torch.Tensor)
            or candidate_valid.dtype != torch.bool
            or candidate_valid.shape != (batch_size, query_count)):
        raise ValueError("candidate_valid must be bool with shape [B,Q]")
    if (
            not isinstance(candidate_boxes, torch.Tensor)
            or candidate_boxes.shape != (batch_size, query_count, 6)):
        raise ValueError("candidate_boxes must have shape [B,Q,6]")
    if (
            not isinstance(target_boxes, torch.Tensor)
            or target_boxes.shape != (batch_size, 6)):
        raise ValueError("target_boxes must have shape [B,6]")
    anchor_boxes = _as_anchor_boxes(anchor_boxes, batch_size)
    anchor_valid = _as_anchor_valid(anchor_valid, anchor_boxes)
    row_masks = (
        anchor_reliable, attribute_present, anchor_text_present,
        relation_text_present, sample_mask, conservative_rows,
    )
    if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.shape != (batch_size,)
            for value in row_masks):
        raise ValueError("row masks must be bool with shape [B]")
    if isinstance(relation_labels, str):
        relation_labels = [relation_labels] * batch_size
    if (
            not isinstance(relation_labels, (list, tuple))
            or len(relation_labels) != batch_size):
        raise ValueError("relation_labels must align with the batch")
    tensors = matrices + row_masks + (
        candidate_boxes, candidate_valid, target_boxes, anchor_boxes,
        anchor_valid,
    )
    if any(value.device != deployed_scores.device for value in tensors):
        raise ValueError("relation auxiliary tensors must share one device")
    if not bool(candidate_valid.any(dim=1).all().item()):
        raise ValueError("every row must contain a valid candidate")


def compute_relation_counterfactual_auxiliary_loss(
        deployed_scores, candidate_boxes, candidate_valid, box_ious,
        target_boxes, anchor_boxes, anchor_reliable, relation_labels,
        target_affinity, attribute_affinity, attribute_present,
        anchor_text_present, relation_text_present, sample_mask,
        parent_top_k=32, target_tolerance=0.10,
        attribute_tolerance=0.10, geometry_threshold=0.08,
        correct_iou_threshold=0.25, pair_margin=0.05,
        max_negatives=8, target_confidence_floor=0.05,
        attribute_confidence_floor=0.02, acc025_pair_weight=2.0,
        anchor_valid=None, conservative_rows=None):
    """Rank the correct target above relation-inconsistent hard negatives.

    All mining decisions are detached. Gradients flow only through selected
    positive and negative entries of deployed_scores.
    """
    conservative_rows = _as_conservative_rows(
        conservative_rows, deployed_scores
    )
    _validate_inputs(
        deployed_scores=deployed_scores,
        candidate_boxes=candidate_boxes,
        candidate_valid=candidate_valid,
        box_ious=box_ious,
        target_boxes=target_boxes,
        anchor_boxes=anchor_boxes,
        anchor_reliable=anchor_reliable,
        relation_labels=relation_labels,
        target_affinity=target_affinity,
        attribute_affinity=attribute_affinity,
        attribute_present=attribute_present,
        anchor_text_present=anchor_text_present,
        relation_text_present=relation_text_present,
        sample_mask=sample_mask,
        anchor_valid=anchor_valid,
        conservative_rows=conservative_rows,
    )
    for name, value in (
            ("target_tolerance", target_tolerance),
            ("attribute_tolerance", attribute_tolerance),
            ("correct_iou_threshold", correct_iou_threshold),
            ("pair_margin", pair_margin),
            ("target_confidence_floor", target_confidence_floor),
            ("attribute_confidence_floor", attribute_confidence_floor)):
        _finite_in_range(name, value, 0.0, 1.0)
    _finite_in_range(
        "geometry_threshold", geometry_threshold, 0.0, 2.0
    )
    _finite_in_range(
        "acc025_pair_weight", acc025_pair_weight, 1.0, 10.0
    )
    for name, value in (
            ("parent_top_k", parent_top_k),
            ("max_negatives", max_negatives)):
        if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 4096):
            raise ValueError("{} must be in [1, 4096]".format(name))

    batch_size, query_count = deployed_scores.shape
    row = torch.arange(batch_size, device=deployed_scores.device)
    with torch.no_grad():
        detached_scores = deployed_scores.detach().float()
        detached_ious = box_ious.detach().float().clamp(0.0, 1.0)
        detached_target = target_affinity.detach().float()
        detached_attribute = attribute_affinity.detach().float()
        valid = candidate_valid.detach()
        masked_ious = detached_ious.masked_fill(~valid, -1.0)
        correct_ious, correct_indices = masked_ious.max(dim=1)
        correct_mask = torch.zeros_like(valid)
        correct_mask[row, correct_indices] = True

        correct_target = detached_target[row, correct_indices]
        correct_attribute = detached_attribute[row, correct_indices]
        target_confident = (
            correct_target >= float(target_confidence_floor)
        )
        attribute_confident = (
            ~attribute_present
            | (
                correct_attribute
                >= float(attribute_confidence_floor)
            )
        )
        same_target = (
            detached_target
            >= correct_target.unsqueeze(1) - float(target_tolerance)
        ) & (
            detached_target >= float(target_confidence_floor)
        )
        attribute_close = (
            ~attribute_present.unsqueeze(1)
            | (
                (
                    detached_attribute
                    - correct_attribute.unsqueeze(1)
                ).abs() <= float(attribute_tolerance)
            )
            & (
                detached_attribute
                >= float(attribute_confidence_floor)
            )
        )

        predicates = build_relation_predicate_masks(
            candidate_boxes=candidate_boxes,
            target_boxes=target_boxes,
            anchor_boxes=anchor_boxes,
            relation_labels=relation_labels,
            predicate_margin=float(geometry_threshold),
            anchor_valid=anchor_valid,
            conservative_rows=conservative_rows,
        )
        geometry_inconsistent = predicates["inconsistent_mask"]
        reference_valid = predicates["reference_valid_mask"]
        disagreement = predicates["disagreement_strength"]
        active_rows = (
            sample_mask
            & anchor_text_present
            & relation_text_present
            & anchor_reliable
            & reference_valid
            & target_confident
            & attribute_confident
            & (correct_ious > float(correct_iou_threshold))
        )

        tiers = (
            (detached_ious > 0.25).long()
            + (detached_ious > 0.50).long()
        )
        correct_tiers = tiers[row, correct_indices]
        lower_tier = tiers < correct_tiers.unsqueeze(1)
        parent_high = _top_k_mask(detached_scores, valid, parent_top_k)
        hard_negative_mask = (
            active_rows.unsqueeze(1)
            & valid
            & ~correct_mask
            & same_target
            & attribute_close
            & geometry_inconsistent
            & lower_tier
            & parent_high
        )
        negative_count = min(max_negatives, query_count)
        hardness = detached_scores.masked_fill(
            ~hard_negative_mask, torch.finfo(torch.float32).min
        )
        negative_indices = torch.topk(
            hardness, negative_count, dim=1
        ).indices
        selected_mask = torch.gather(
            hard_negative_mask, 1, negative_indices
        )
        selected_ious = torch.gather(detached_ious, 1, negative_indices)
        selected_tiers = torch.gather(tiers, 1, negative_indices)
        selected_disagreement = torch.gather(
            disagreement, 1, negative_indices
        )
        positive_025 = correct_ious.unsqueeze(1) > 0.25
        positive_050 = correct_ious.unsqueeze(1) > 0.50
        coarse_break = positive_025 & (selected_ious <= 0.25)
        strict_break = positive_050 & (selected_ious <= 0.50)

        parent_indices = detached_scores.masked_fill(
            ~valid, torch.finfo(torch.float32).min
        ).argmax(dim=1)
        parent_is_hard_negative = hard_negative_mask[row, parent_indices]
        correct_is_parent = parent_indices.eq(correct_indices)

    positive_scores = deployed_scores.float()[row, correct_indices].unsqueeze(1)
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
    selected_float = selected_mask.float()
    weighted_mask = selected_float * pair_weights
    loss = (
        pair_losses * weighted_mask
    ).sum() / weighted_mask.sum().clamp(min=1.0)

    selected_count = selected_float.sum().clamp(min=1.0)
    active_count = active_rows.float().sum().clamp(min=1.0)
    selected_margin = positive_scores.detach() - negative_scores.detach()
    violating_selected = (
        pair_losses.detach().gt(0).float() * selected_float
    )
    selected_score_gradient_l1 = (
        2.0 * violating_selected * pair_weights.detach()
    ).sum() / weighted_mask.detach().sum().clamp(min=1.0)
    stats = {
        "loss": loss.reshape(()),
        # These three audit statistics distinguish a merely connected loss
        # from one that supplies a non-zero score gradient on this batch.
        "nonzero_loss_batch_ratio": loss.detach().gt(0).float(),
        "violating_selected_count_mean": (
            violating_selected.sum(dim=1).mean()
        ),
        "selected_score_gradient_l1": selected_score_gradient_l1,
        "anchor_reliable_ratio": anchor_reliable.float().mean(),
        "relation_reference_valid_ratio": reference_valid.float().mean(),
        "relation_reference_anchor_count_mean": predicates[
            "reference_anchor_count"
        ].mean(),
        "relation_ambiguous_reference_ratio": predicates[
            "ambiguous_reference_mask"
        ].float().mean(),
        "anchor_self_exclusion_ratio": predicates[
            "anchor_self_exclusion_ratio"
        ],
        "conservative_row_ratio": conservative_rows.float().mean(),
        "target_confident_ratio": target_confident.float().mean(),
        "attribute_confident_ratio": attribute_confident.float().mean(),
        "active_row_ratio": active_rows.float().mean(),
        "hard_negative_row_ratio": hard_negative_mask.any(dim=1).float().mean(),
        "hard_negative_count_mean": hard_negative_mask.float().sum(1).mean(),
        "selected_negative_count_mean": selected_float.sum(1).mean(),
        "parent_hard_negative_ratio": parent_is_hard_negative.float().mean(),
        "correct_is_parent_ratio": correct_is_parent.float().mean(),
        "coarse_break_selected_ratio": (
            coarse_break.float() * selected_float
        ).sum() / selected_count,
        "strict_break_selected_ratio": (
            strict_break.float() * selected_float
        ).sum() / selected_count,
        "pair_violation_ratio": (
            pair_losses.detach().gt(0).float() * selected_float
        ).sum() / selected_count,
        "correct_iou_mean": (
            correct_ious * active_rows.float()
        ).sum() / active_count,
        "selected_negative_iou_mean": (
            selected_ious * selected_float
        ).sum() / selected_count,
        "selected_negative_tier_mean": (
            selected_tiers.float() * selected_float
        ).sum() / selected_count,
        "selected_relation_disagreement_mean": (
            selected_disagreement * selected_float
        ).sum() / selected_count,
        "deployed_pair_margin_mean": (
            selected_margin * selected_float
        ).sum() / selected_count,
    }
    return {
        key: value if key == "loss" else value.detach()
        for key, value in stats.items()
    }
