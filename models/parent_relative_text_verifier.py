"""Conservative candidate-level text verification relative to a parent.

The verifier never rescales an entire query row.  It compares a compact,
deployable Top-K candidate union against the immutable parent query, exposes an
explicit fallback action, and can promote at most one feasible candidate.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_counterfactual_parent_score_axis(
        parent_scores, module_training, counterfactual_training):
    """Create the train-only differentiable score axis without unfreezing V99."""
    if not isinstance(parent_scores, torch.Tensor):
        raise ValueError("parent scores must be a tensor")
    if not isinstance(module_training, bool):
        raise ValueError("module_training must be boolean")
    if not isinstance(counterfactual_training, bool):
        raise ValueError("counterfactual_training must be boolean")
    if module_training and counterfactual_training:
        return parent_scores.detach().requires_grad_(True)
    return parent_scores

STRUCTURED_EVIDENCE_DIM = 15
PAIR_GEOMETRY_DIM = 12
ROW_RELIABILITY_DIM = 7
COUNTERFACTUAL_PARENT_VIEW_LIMIT = 2


def _require_finite_scalar(name, value, lower=None, upper=None):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError("{} must be a finite scalar".format(name))
    value = float(value)
    if lower is not None and value < float(lower):
        raise ValueError("{} must be >= {}".format(name, lower))
    if upper is not None and value > float(upper):
        raise ValueError("{} must be <= {}".format(name, upper))
    return value


def _finite_rows(value):
    if not isinstance(value, torch.Tensor) or value.dim() < 1:
        raise ValueError("row-finite inputs must be tensors with a batch axis")
    return torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)


def _finite_or_zero(value):
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _select_exact_deployable_union(
        parent_scores, contrastive_scores, candidate_valid,
        topk_per_source, max_candidates):
    """Select the exact Parent/Text Top-K union on the evaluator-valid axis."""
    batch_size, query_count = parent_scores.shape
    if (candidate_valid.shape != parent_scores.shape
            or candidate_valid.dtype != torch.bool
            or candidate_valid.device != parent_scores.device):
        raise ValueError(
            "candidate_valid must be a boolean [B,Q] tensor on the score device"
        )
    query_indices = torch.zeros(
        batch_size, max_candidates, dtype=torch.long,
        device=parent_scores.device,
    )
    valid = torch.zeros(
        batch_size, max_candidates, dtype=torch.bool,
        device=parent_scores.device,
    )
    parent_indices = torch.zeros(
        batch_size, dtype=torch.long, device=parent_scores.device
    )
    deployable_rows = candidate_valid.any(dim=1)
    top_count = min(int(topk_per_source), query_count)
    for batch_idx in range(batch_size):
        parent_values = parent_scores[batch_idx].detach().cpu().tolist()
        text_values = contrastive_scores[batch_idx].detach().cpu().tolist()
        deployable = candidate_valid[batch_idx].nonzero(
            as_tuple=False
        ).reshape(-1).detach().cpu().tolist()
        if not deployable:
            # Structural fallback keeps attention numerically defined.  The
            # row is marked non-deployable below and can never train or switch.
            fallback = sorted(
                range(query_count),
                key=lambda idx: (-parent_values[idx], idx),
            )[0]
            parent_indices[batch_idx] = int(fallback)
            query_indices[batch_idx, 0] = int(fallback)
            valid[batch_idx, 0] = True
            continue
        parent_order = sorted(
            deployable,
            key=lambda idx: (-parent_values[idx], idx),
        )
        text_order = sorted(
            deployable,
            key=lambda idx: (-text_values[idx], idx),
        )
        parent_indices[batch_idx] = int(parent_order[0])
        selected = []
        selected_set = set()
        for query_idx in (
                parent_order[:top_count] + text_order[:top_count]):
            if query_idx not in selected_set:
                selected.append(query_idx)
                selected_set.add(query_idx)
        if len(selected) > max_candidates:
            raise ValueError(
                "max_candidates truncated the deployable Parent/Text Top-K union"
            )
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=query_indices.device
        )
        query_indices[batch_idx, :len(selected)] = selected_tensor
        valid[batch_idx, :len(selected)] = True
    return query_indices, valid, parent_indices, deployable_rows


def build_parent_relative_detector_valid(candidate_boxes, inputs):
    """Apply the exact formal ``butd_cls`` detector-overlap candidate filter."""
    if not isinstance(inputs, dict):
        raise TypeError("detector-valid inputs must be a mapping")
    detected_boxes = inputs.get("det_boxes")
    detected_valid = inputs.get("det_bbox_label_mask")
    if (not isinstance(detected_boxes, torch.Tensor)
            or not isinstance(detected_valid, torch.Tensor)):
        raise ValueError(
            "parent-relative verification requires formal detector boxes"
        )
    # Lazy import avoids the losses -> verifier import cycle while reusing the
    # same implementation as the formal grounding evaluator.
    from .rec_evaluator_filter import build_detector_overlap_valid
    with torch.no_grad():
        candidate_finite = torch.isfinite(candidate_boxes).all(dim=-1)
        candidate_positive_size = (candidate_boxes[..., 3:] > 0.0).all(dim=-1)
        candidate_active = candidate_finite & candidate_positive_size
        safe_candidate_boxes = torch.nan_to_num(
            candidate_boxes.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )
        safe_candidate_boxes[..., 3:] = safe_candidate_boxes[
            ..., 3:
        ].clamp(min=1e-6)
        return build_detector_overlap_valid(
            safe_candidate_boxes,
            candidate_active,
            detected_boxes.detach(),
            detected_valid.detach().bool(),
            iou_threshold=0.25,
        )


def _gather_query_values(values, query_indices):
    if values.dim() < 2 or query_indices.dim() != 2:
        raise ValueError("query tensors must have [B,Q,...] and [B,K] shapes")
    if values.shape[0] != query_indices.shape[0]:
        raise ValueError("query tensor batch sizes must match")
    gather_index = query_indices
    for _ in values.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(
        query_indices.shape + values.shape[2:]
    )
    return torch.gather(values, 1, gather_index)


def _gather_parent(values, parent_positions):
    if values.shape[:2] != parent_positions.shape:
        raise ValueError("parent positions must align with compact candidates")
    parent_index = parent_positions.long().argmax(dim=1)
    gather_index = parent_index.view(-1, 1)
    for _ in values.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(
        (values.shape[0], 1) + values.shape[2:]
    )
    return torch.gather(values, 1, gather_index).squeeze(1)


def _masked_slot_mean(values, valid):
    if values.dim() != 3 or valid.shape != values.shape[:2]:
        raise ValueError("slot values and masks must align as [B,S,D]/[B,S]")
    weights = valid.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _aligned_box_iou(first, second):
    first_min = first[..., :3] - 0.5 * first[..., 3:].clamp(min=1e-6)
    first_max = first[..., :3] + 0.5 * first[..., 3:].clamp(min=1e-6)
    second_min = second[..., :3] - 0.5 * second[..., 3:].clamp(min=1e-6)
    second_max = second[..., :3] + 0.5 * second[..., 3:].clamp(min=1e-6)
    inter_size = (
        torch.minimum(first_max, second_max)
        - torch.maximum(first_min, second_min)
    ).clamp(min=0.0)
    intersection = inter_size.prod(dim=-1)
    first_volume = (first_max - first_min).clamp(min=0.0).prod(dim=-1)
    second_volume = (second_max - second_min).clamp(min=0.0).prod(dim=-1)
    union = (first_volume + second_volume - intersection).clamp(min=1e-6)
    return intersection / union


def _parent_candidate_geometry(candidate_boxes, parent_boxes):
    if candidate_boxes.dim() != 3 or candidate_boxes.shape[-1] != 6:
        raise ValueError("candidate boxes must have shape [B,K,6]")
    if parent_boxes.shape != (candidate_boxes.shape[0], 6):
        raise ValueError("parent boxes must have shape [B,6]")
    parent = parent_boxes.unsqueeze(1).expand_as(candidate_boxes)
    delta = candidate_boxes[..., :3] - parent[..., :3]
    distance = delta.square().sum(dim=-1).add(1e-6).sqrt()
    direction = delta / distance.unsqueeze(-1).clamp(min=1e-6)
    candidate_size = candidate_boxes[..., 3:].clamp(min=1e-6)
    parent_size = parent[..., 3:].clamp(min=1e-6)
    log_size_ratio = (candidate_size / parent_size).log().clamp(-8.0, 8.0)
    log_volume_ratio = (
        candidate_size.prod(dim=-1) / parent_size.prod(dim=-1)
    ).log().clamp(-8.0, 8.0)
    iou = _aligned_box_iou(candidate_boxes, parent)
    return torch.cat((
        delta,
        distance.unsqueeze(-1),
        direction,
        log_size_ratio,
        log_volume_ratio.unsqueeze(-1),
        iou.unsqueeze(-1),
    ), dim=-1)


def _structured_evidence(sacr_outputs, batch_size, query_count, device,
                         dtype):
    if sacr_outputs is None:
        return (
            torch.zeros(
                batch_size, query_count, STRUCTURED_EVIDENCE_DIM,
                device=device, dtype=dtype,
            ),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=dtype),
        )
    scalar_names = (
        "structured_scores",
        "target_attr_scores",
        "relation_anchor_scores",
    )
    scalar_values = []
    for name in scalar_names:
        value = sacr_outputs.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (
                batch_size, query_count):
            raise ValueError("{} must have shape [B,Q]".format(name))
        scalar_values.append(value.to(device=device, dtype=dtype).unsqueeze(-1))
    geometry = sacr_outputs.get("relation_geometry_signatures")
    if (not isinstance(geometry, torch.Tensor)
            or geometry.shape != (batch_size, query_count, 11)):
        raise ValueError(
            "relation_geometry_signatures must have shape [B,Q,11]"
        )
    relation_mask = sacr_outputs.get("relation_candidate_mask")
    if (not isinstance(relation_mask, torch.Tensor)
            or relation_mask.shape != (batch_size, query_count)):
        raise ValueError("relation_candidate_mask must have shape [B,Q]")
    evidence = torch.cat(
        tuple(scalar_values) + (
            geometry.to(device=device, dtype=dtype),
            relation_mask.to(device=device, dtype=dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    if evidence.shape[-1] != STRUCTURED_EVIDENCE_DIM:
        raise RuntimeError("structured evidence width is inconsistent")

    structured_valid = sacr_outputs.get("structured_valid_mask")
    anchor_mass = sacr_outputs.get("anchor_top1_mass")
    anchor_entropy = sacr_outputs.get("anchor_entropy")
    relation_active = sacr_outputs.get("relation_active_ratio_per_sample")
    row_values = (anchor_mass, anchor_entropy, relation_active)
    if (not isinstance(structured_valid, torch.Tensor)
            or structured_valid.shape != (batch_size,)
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != (batch_size,)
                for value in row_values
            )):
        raise ValueError("SACR row reliability must align as [B]")
    return (
        evidence,
        structured_valid.to(device=device).bool(),
        anchor_mass.to(device=device, dtype=dtype),
        anchor_entropy.to(device=device, dtype=dtype),
        relation_active.to(device=device, dtype=dtype),
    )


def build_parent_relative_text_verifier_batch(
        full_state, query_features, parent_scores, candidate_valid, slot_dict,
        sacr_outputs=None, topk_per_source=5, max_candidates=10):
    """Build a GT-free compact verifier batch from deployable tensors."""
    required_state = (
        "features", "boxes", "default_scores", "contrastive_scores",
        "num_queries",
    )
    if any(name not in full_state for name in required_state):
        raise KeyError("full REC state is incomplete")
    if not isinstance(query_features, torch.Tensor) or query_features.dim() != 3:
        raise ValueError("query_features must have shape [B,Q,D]")
    batch_size, query_count, _ = query_features.shape
    if (not isinstance(parent_scores, torch.Tensor)
            or parent_scores.shape != (batch_size, query_count)):
        raise ValueError("parent_scores must have shape [B,Q]")
    if full_state["features"].shape[:2] != (batch_size, query_count):
        raise ValueError("full REC features must align with query_features")
    if full_state["boxes"].shape != (batch_size, query_count, 6):
        raise ValueError("full REC boxes must have shape [B,Q,6]")
    if full_state["contrastive_scores"].shape != parent_scores.shape:
        raise ValueError("contrastive scores must align with parent_scores")
    if (not isinstance(candidate_valid, torch.Tensor)
            or candidate_valid.shape != parent_scores.shape
            or candidate_valid.dtype != torch.bool
            or candidate_valid.device != parent_scores.device):
        raise ValueError(
            "candidate_valid must be a boolean [B,Q] tensor on the score device"
        )

    required_capacity = min(query_count, 2 * int(topk_per_source))
    if max_candidates < required_capacity:
        raise ValueError(
            "max_candidates must hold the complete Parent/Text Top-K union"
        )
    input_valid_rows = (
        _finite_rows(parent_scores)
        & _finite_rows(full_state["contrastive_scores"])
        & _finite_rows(full_state["features"])
        & _finite_rows(full_state["boxes"])
        & _finite_rows(query_features)
        & candidate_valid.any(dim=1)
    )
    safe_parent_scores = torch.nan_to_num(
        parent_scores, nan=-1e4, posinf=1e4, neginf=-1e4
    )
    safe_contrastive_scores = torch.nan_to_num(
        full_state["contrastive_scores"],
        nan=-1e4, posinf=1e4, neginf=-1e4,
    )
    query_indices, valid, parent_query_index, deployable_rows = (
        _select_exact_deployable_union(
        safe_parent_scores,
        safe_contrastive_scores,
        candidate_valid,
        topk_per_source,
        max_candidates,
        )
    )
    compact = {
        "schema_version": full_state["schema_version"],
        "feature_names": full_state["feature_names"],
        "features": _gather_query_values(
            _finite_or_zero(full_state["features"]), query_indices
        ),
        "boxes": _gather_query_values(
            _finite_or_zero(full_state["boxes"]), query_indices
        ),
        "query_indices": query_indices,
        "valid_mask": valid,
        "default_scores": _gather_query_values(
            safe_parent_scores.unsqueeze(-1), query_indices
        ).squeeze(-1),
        "contrastive_scores": _gather_query_values(
            safe_contrastive_scores.unsqueeze(-1), query_indices
        ).squeeze(-1),
        "default_top1_query_index": parent_query_index,
        "num_queries": query_count,
    }
    compact["model_inputs"] = {
        "features": compact["features"],
        "valid_mask": valid,
    }
    parent_position = (
        query_indices == parent_query_index.unsqueeze(1)
    ) & valid
    if not bool((parent_position.sum(dim=1) == 1).all().item()):
        raise RuntimeError("the compact candidate set must contain one parent")

    slot_names = ("global_slot", "target_slot", "attr_slot")
    if any(name not in slot_dict for name in slot_names):
        raise KeyError("structured target slots are incomplete")
    raw_slots = tuple(slot_dict[name].to(query_features) for name in slot_names)
    for value in raw_slots:
        input_valid_rows &= _finite_rows(value)
    global_slot, target_slot, attr_slot = tuple(
        _finite_or_zero(value) for value in raw_slots
    )
    if any(
            value.shape != (batch_size, query_features.shape[-1])
            for value in (global_slot, target_slot, attr_slot)):
        raise ValueError("global/target/attribute slots must have shape [B,D]")
    rel_slots = slot_dict.get("rel_slots")
    anchor_slots = slot_dict.get("anchor_slots")
    slot_mask = slot_dict.get("slot_mask")
    if (not isinstance(rel_slots, torch.Tensor)
            or not isinstance(anchor_slots, torch.Tensor)
            or rel_slots.shape != anchor_slots.shape
            or rel_slots.dim() != 3
            or rel_slots.shape[0] != batch_size
            or rel_slots.shape[-1] != query_features.shape[-1]
            or not isinstance(slot_mask, torch.Tensor)
            or slot_mask.shape != rel_slots.shape[:2]):
        raise ValueError("relation/anchor slots have an invalid contract")
    slot_mask = slot_mask.to(device=query_features.device).bool()
    rel_slots = rel_slots.to(query_features)
    anchor_slots = anchor_slots.to(query_features)
    input_valid_rows &= _finite_rows(rel_slots) & _finite_rows(anchor_slots)
    relation_slot = _masked_slot_mean(
        _finite_or_zero(rel_slots), slot_mask
    )
    anchor_slot = _masked_slot_mean(
        _finite_or_zero(anchor_slots), slot_mask
    )
    language_context = torch.stack((
        global_slot, target_slot, attr_slot, relation_slot, anchor_slot,
    ), dim=1)

    coverage = slot_dict.get("coverage_stats", {})
    has_target = coverage.get("has_target")
    if not isinstance(has_target, torch.Tensor) or has_target.shape != (
            batch_size,):
        raise ValueError("structured coverage has_target must have shape [B]")
    parse_confidence = slot_dict.get("parse_confidence")
    if not isinstance(parse_confidence, torch.Tensor) or parse_confidence.shape != (
            batch_size,):
        raise ValueError("parse_confidence must have shape [B]")

    input_valid_rows &= _finite_rows(parse_confidence)
    evidence, structured_valid, anchor_mass, anchor_entropy, relation_active = (
        _structured_evidence(
            sacr_outputs,
            batch_size,
            query_count,
            query_features.device,
            query_features.dtype,
        )
    )
    input_valid_rows &= (
        _finite_rows(evidence)
        & _finite_rows(anchor_mass)
        & _finite_rows(anchor_entropy)
        & _finite_rows(relation_active)
    )
    evidence = _finite_or_zero(evidence)
    anchor_mass = _finite_or_zero(anchor_mass)
    anchor_entropy = _finite_or_zero(anchor_entropy)
    relation_active = _finite_or_zero(relation_active)
    result = dict(compact)
    result.update({
        "query_features": _gather_query_values(
            _finite_or_zero(query_features), query_indices
        ),
        "structured_evidence": _gather_query_values(
            evidence, query_indices
        ),
        "language_context": language_context,
        "parent_query_index": parent_query_index,
        "parent_position": parent_position,
        "parent_full_scores": parent_scores,
        "parse_confidence": _finite_or_zero(
            parse_confidence.to(query_features)
        ).clamp(0.0, 1.0),
        "has_target": has_target.to(device=query_features.device).bool(),
        "relation_required": slot_mask.any(dim=1),
        "structured_valid": structured_valid,
        "anchor_top1_mass": anchor_mass,
        "anchor_entropy": anchor_entropy,
        "relation_active_ratio": relation_active,
        "input_valid_rows": input_valid_rows,
        "deployable_rows": deployable_rows,
        "detector_valid_mask": _gather_query_values(
            candidate_valid.unsqueeze(-1), query_indices
        ).squeeze(-1) & valid,
    })
    return result


def build_counterfactual_parent_views(candidate_batch, verifier_outputs):
    """Build fixed training-only Parent views without GT-dependent selection.

    The first view uses the deployable text-score Top-1 when it differs from
    the actual Parent.  The second removes the actual Parent and uses the next
    deployable Parent-score Top-1.  Both views keep the same compact candidate
    tensors and recompute all Parent-relative evidence in ``forward``.  The
    helper is never used by deployment.
    """
    required = (
        "features", "query_features", "boxes", "valid_mask",
        "parent_position", "default_scores", "contrastive_scores",
        "query_indices", "parent_query_index", "input_valid_rows",
    )
    if any(name not in candidate_batch for name in required):
        raise KeyError("counterfactual Parent batch is incomplete")
    valid = candidate_batch["valid_mask"]
    parent_position = candidate_batch["parent_position"]
    default_scores = candidate_batch["default_scores"]
    contrastive_scores = candidate_batch["contrastive_scores"]
    if (not isinstance(valid, torch.Tensor) or valid.dim() != 2
            or valid.dtype != torch.bool
            or parent_position.shape != valid.shape
            or default_scores.shape != valid.shape
            or contrastive_scores.shape != valid.shape):
        raise ValueError("counterfactual Parent tensors must align as [B,K]")
    if not bool((parent_position.sum(dim=1) == 1).all().item()):
        raise ValueError("counterfactual source rows need one actual Parent")
    feasible = verifier_outputs.get("feasible_mask")
    if (not isinstance(feasible, torch.Tensor)
            or feasible.shape != valid.shape
            or feasible.dtype != torch.bool
            or feasible.device != valid.device):
        raise ValueError(
            "counterfactual Parent selection needs the actual feasible mask"
        )

    source_rows = []
    view_parent_positions = []
    view_valid_masks = []
    view_score_axes = []
    view_kinds = []
    batch_size, candidate_count = valid.shape
    for batch_idx in range(batch_size):
        row_valid = valid[batch_idx]
        valid_positions = row_valid.nonzero(
            as_tuple=False
        ).reshape(-1).detach().cpu().tolist()
        feasible_positions = feasible[batch_idx].nonzero(
            as_tuple=False
        ).reshape(-1).detach().cpu().tolist()
        actual_parent = int(
            parent_position[batch_idx].long().argmax().item()
        )
        used_parents = {actual_parent}

        if valid_positions:
            text_values = contrastive_scores[
                batch_idx
            ].detach().cpu().tolist()
            text_parent = sorted(
                valid_positions,
                key=lambda idx: (-text_values[idx], idx),
            )[0]
            if (text_parent not in used_parents
                    and text_parent in feasible_positions):
                source_rows.append(batch_idx)
                text_parent_position = torch.zeros_like(row_valid)
                text_parent_position[text_parent] = True
                view_parent_positions.append(text_parent_position)
                view_valid_masks.append(row_valid.clone())
                view_score_axes.append(contrastive_scores[batch_idx])
                view_kinds.append(0)
                used_parents.add(text_parent)

        leave_one_out_valid = row_valid.clone()
        leave_one_out_valid[actual_parent] = False
        leave_one_out_positions = leave_one_out_valid.nonzero(
            as_tuple=False
        ).reshape(-1).detach().cpu().tolist()
        if leave_one_out_positions:
            parent_values = default_scores[
                batch_idx
            ].detach().cpu().tolist()
            leave_one_out_parent = sorted(
                leave_one_out_positions,
                key=lambda idx: (-parent_values[idx], idx),
            )[0]
            if (leave_one_out_parent not in used_parents
                    and leave_one_out_parent in feasible_positions):
                source_rows.append(batch_idx)
                leave_one_out_parent_position = torch.zeros_like(row_valid)
                leave_one_out_parent_position[leave_one_out_parent] = True
                # This is an intervention on the Parent state, not merely a
                # different parent_position label.  Promote the selected LOO
                # Parent to the original row maximum so that the score axis is
                # self-consistent and the actual Parent remains a feasible
                # zero-gap repair candidate.  Candidate selection above still
                # uses the untouched deployable scores and no GT information.
                leave_one_out_scores = default_scores[batch_idx].clone()
                leave_one_out_scores[leave_one_out_parent] = default_scores[
                    batch_idx, actual_parent
                ]
                view_parent_positions.append(leave_one_out_parent_position)
                view_valid_masks.append(row_valid.clone())
                view_score_axes.append(leave_one_out_scores)
                view_kinds.append(1)
                used_parents.add(leave_one_out_parent)

        if len(used_parents) - 1 > COUNTERFACTUAL_PARENT_VIEW_LIMIT:
            raise RuntimeError("counterfactual Parent view limit was exceeded")

    if not source_rows:
        return None
    source_rows = torch.tensor(
        source_rows, dtype=torch.long, device=valid.device
    )
    result = {}
    for name, value in candidate_batch.items():
        if name == "model_inputs":
            continue
        if (isinstance(value, torch.Tensor) and value.dim() >= 1
                and value.shape[0] == batch_size):
            result[name] = value.index_select(0, source_rows)
        else:
            result[name] = value
    result["valid_mask"] = torch.stack(view_valid_masks, dim=0)
    result["parent_position"] = torch.stack(
        view_parent_positions, dim=0
    )
    result["default_scores"] = torch.stack(
        view_score_axes, dim=0
    ).detach().requires_grad_(True)
    result["parent_query_index"] = torch.gather(
        result["query_indices"].long(),
        1,
        result["parent_position"].long().argmax(dim=1, keepdim=True),
    ).squeeze(1)
    if "detector_valid_mask" in result:
        result["detector_valid_mask"] = (
            result["detector_valid_mask"].bool() & result["valid_mask"]
        )
    result["counterfactual_source_rows"] = source_rows
    result["counterfactual_view_kind"] = torch.tensor(
        view_kinds, dtype=torch.long, device=valid.device
    )
    result["model_inputs"] = {
        "features": result["features"],
        "valid_mask": result["valid_mask"],
    }
    if not bool((result["parent_position"].sum(dim=1) == 1).all().item()):
        raise RuntimeError("counterfactual views need exactly one Parent")
    if not bool(
            (result["valid_mask"].sum(dim=1) >= 2).all().item()):
        raise RuntimeError("counterfactual views need a Parent and candidate")
    return result


def apply_discrete_parent_relative_selection(
        parent_scores, selected_query_indices, parent_query_indices,
        switch_mask, promotion_epsilon=1e-4):
    """Promote one verified query while leaving the parent score unchanged."""
    promotion_epsilon = _require_finite_scalar(
        "promotion_epsilon", promotion_epsilon, lower=1e-8, upper=1e-2
    )
    if parent_scores.dim() != 2:
        raise ValueError("parent_scores must have shape [B,Q]")
    batch_size, query_count = parent_scores.shape
    row_values = (
        selected_query_indices, parent_query_indices, switch_mask,
    )
    if any(
            not isinstance(value, torch.Tensor)
            or value.shape != (batch_size,)
            for value in row_values):
        raise ValueError("selection tensors must have shape [B]")
    selected = selected_query_indices.long()
    parent = parent_query_indices.long()
    if bool(((selected < 0) | (selected >= query_count)).any().item()) or bool(
            ((parent < 0) | (parent >= query_count)).any().item()):
        raise ValueError("selection indices are out of range")
    switch = switch_mask.bool()
    if bool((switch & (selected == parent)).any().item()):
        raise ValueError("a switch must select a non-parent query")
    batch_indices = torch.arange(batch_size, device=parent_scores.device)
    parent_values = parent_scores[batch_indices, parent]
    refined = parent_scores.clone()
    if switch.any():
        switch_rows = batch_indices[switch]
        switch_queries = selected[switch]
        refined[switch_rows, switch_queries] = (
            parent_values[switch] + promotion_epsilon
        )
    refined_parent = refined[batch_indices, parent]
    parent_unchanged = (
        (refined_parent == parent_values)
        | (torch.isnan(refined_parent) & torch.isnan(parent_values))
    )
    if not bool(parent_unchanged.all().item()):
        raise RuntimeError("discrete verifier changed the parent score")
    return refined


class ParentRelativeTextVerifier(nn.Module):
    """Candidate-vs-parent verifier with an explicit fixed fallback action."""

    def __init__(self, query_dim=288, base_feature_dim=152, slot_dim=288,
                 hidden_dim=256, num_heads=4, dropout=0.1,
                 max_parent_score_gap=0.25, promotion_margin=1e-4,
                 min_parse_confidence=0.5, min_anchor_mass=0.5,
                 detach_inputs=True, counterfactual_training=False):
        super().__init__()
        for name, value in (
                ("query_dim", query_dim),
                ("base_feature_dim", base_feature_dim),
                ("slot_dim", slot_dim),
                ("hidden_dim", hidden_dim),
                ("num_heads", num_heads)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError("{} must be a positive integer".format(name))
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be boolean")
        if not isinstance(counterfactual_training, bool):
            raise ValueError("counterfactual_training must be boolean")
        self.query_dim = int(query_dim)
        self.base_feature_dim = int(base_feature_dim)
        self.slot_dim = int(slot_dim)
        self.max_parent_score_gap = _require_finite_scalar(
            "max_parent_score_gap", max_parent_score_gap,
            lower=1e-4, upper=1.0,
        )
        self.promotion_margin = _require_finite_scalar(
            "promotion_margin", promotion_margin,
            lower=0.0, upper=min(self.max_parent_score_gap, 0.05),
        )
        self.min_parse_confidence = _require_finite_scalar(
            "min_parse_confidence", min_parse_confidence,
            lower=0.0, upper=1.0,
        )
        self.min_anchor_mass = _require_finite_scalar(
            "min_anchor_mass", min_anchor_mass, lower=0.0, upper=1.0
        )
        self.detach_inputs = detach_inputs
        self.counterfactual_training = counterfactual_training

        evidence_dim = (
            3 * STRUCTURED_EVIDENCE_DIM
            + PAIR_GEOMETRY_DIM
            + 1
            + ROW_RELIABILITY_DIM
        )
        self.query_pair_projector = nn.Sequential(
            nn.Linear(4 * query_dim, hidden_dim), nn.ReLU()
        )
        self.base_pair_projector = nn.Sequential(
            nn.Linear(3 * base_feature_dim, hidden_dim), nn.ReLU()
        )
        self.language_projector = nn.Sequential(
            nn.Linear(5 * slot_dim, hidden_dim), nn.ReLU()
        )
        self.evidence_projector = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.set_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout
        )
        self.set_norm = nn.LayerNorm(hidden_dim)
        self.set_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 1)
        self.repair_head = nn.Linear(hidden_dim, 2)
        self.break_head = nn.Linear(hidden_dim, 2)
        self.transition_utility_head = (
            nn.Linear(hidden_dim, 2) if counterfactual_training else None
        )
        self.iou_advantage_head = nn.Linear(hidden_dim, 1)

        # Exact fallback at initialization, without a dead-gradient gate.
        nn.init.zeros_(self.action_head.weight)
        nn.init.constant_(self.action_head.bias, -4.0)
        if self.transition_utility_head is not None:
            nn.init.zeros_(self.transition_utility_head.weight)
            nn.init.zeros_(self.transition_utility_head.bias)

    def _maybe_detach(self, value):
        return value.detach() if self.detach_inputs else value

    def forward(self, candidate_batch):
        required = (
            "features", "query_features", "boxes", "valid_mask",
            "parent_position", "default_scores", "structured_evidence",
            "language_context", "parse_confidence", "has_target",
            "relation_required", "structured_valid", "anchor_top1_mass",
            "anchor_entropy", "relation_active_ratio", "query_indices",
            "parent_query_index", "input_valid_rows",
        )
        if any(name not in candidate_batch for name in required):
            raise KeyError("parent-relative verifier batch is incomplete")
        raw_base = candidate_batch["features"].float()
        raw_queries = candidate_batch["query_features"].float()
        raw_boxes = candidate_batch["boxes"].float()
        raw_structured = candidate_batch["structured_evidence"].float()
        raw_language = candidate_batch["language_context"].float()
        base = self._maybe_detach(_finite_or_zero(raw_base))
        queries = self._maybe_detach(_finite_or_zero(raw_queries))
        boxes = self._maybe_detach(_finite_or_zero(raw_boxes))
        structured = self._maybe_detach(_finite_or_zero(raw_structured))
        language = self._maybe_detach(_finite_or_zero(raw_language))
        valid = candidate_batch["valid_mask"].bool()
        parent_position = candidate_batch["parent_position"].bool()
        if (base.dim() != 3 or queries.dim() != 3
                or base.shape[:2] != queries.shape[:2]
                or boxes.shape != base.shape[:2] + (6,)
                or valid.shape != base.shape[:2]
                or parent_position.shape != valid.shape
                or structured.shape != base.shape[:2] + (
                    STRUCTURED_EVIDENCE_DIM,
                )):
            raise ValueError("candidate-level verifier tensors do not align")
        batch_size, candidate_count, _ = base.shape
        input_valid_rows = candidate_batch["input_valid_rows"]
        if (not isinstance(input_valid_rows, torch.Tensor)
                or input_valid_rows.shape != (batch_size,)):
            raise ValueError("input_valid_rows must have shape [B]")
        input_valid_rows = input_valid_rows.to(device=base.device).bool()
        input_valid_rows &= (
            _finite_rows(raw_base)
            & _finite_rows(raw_queries)
            & _finite_rows(raw_boxes)
            & _finite_rows(raw_structured)
            & _finite_rows(raw_language)
        )
        if base.shape[-1] != self.base_feature_dim:
            raise ValueError("unexpected REC base feature width")
        if queries.shape[-1] != self.query_dim:
            raise ValueError("unexpected decoder query width")
        if language.shape != (batch_size, 5, self.slot_dim):
            raise ValueError("language context must have shape [B,5,D]")
        if not bool(valid.any(dim=1).all().item()) or not bool(
                (parent_position.sum(dim=1) == 1).all().item()):
            raise ValueError("each verifier row needs exactly one parent")

        parent_query = _gather_parent(queries, parent_position)
        parent_base = _gather_parent(base, parent_position)
        parent_box = _gather_parent(boxes, parent_position)
        parent_structured = _gather_parent(structured, parent_position)
        query_pair = torch.cat((
            queries,
            parent_query.unsqueeze(1).expand_as(queries),
            queries - parent_query.unsqueeze(1),
            queries * parent_query.unsqueeze(1),
        ), dim=-1)
        base_pair = torch.cat((
            base,
            parent_base.unsqueeze(1).expand_as(base),
            base - parent_base.unsqueeze(1),
        ), dim=-1)
        structured_pair = torch.cat((
            structured,
            parent_structured.unsqueeze(1).expand_as(structured),
            structured - parent_structured.unsqueeze(1),
        ), dim=-1)
        geometry = _parent_candidate_geometry(boxes, parent_box)

        raw_compact_parent_scores = candidate_batch["default_scores"].float()
        compact_parent_scores = _finite_or_zero(raw_compact_parent_scores)
        if compact_parent_scores.shape != valid.shape:
            raise ValueError("compact parent scores must have shape [B,K]")
        compact_parent_scores = self._maybe_detach(compact_parent_scores)
        parent_score = _gather_parent(
            compact_parent_scores.unsqueeze(-1), parent_position
        ).squeeze(-1)
        score_gap = parent_score.unsqueeze(1) - compact_parent_scores

        row_names = (
            "parse_confidence", "has_target", "relation_required",
            "anchor_top1_mass", "anchor_entropy", "relation_active_ratio",
            "structured_valid",
        )
        row_values = []
        for name in row_names:
            value = candidate_batch[name]
            if not isinstance(value, torch.Tensor) or value.shape != (
                    batch_size,):
                raise ValueError("{} must have shape [B]".format(name))
            value = value.to(device=base.device).float()
            input_valid_rows &= _finite_rows(value)
            row_values.append(self._maybe_detach(_finite_or_zero(value)))
        input_valid_rows &= _finite_rows(raw_compact_parent_scores)
        row_reliability = torch.stack(row_values, dim=-1)
        reliability_features = row_reliability.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        evidence = torch.cat((
            structured_pair,
            geometry,
            score_gap.unsqueeze(-1),
            reliability_features,
        ), dim=-1)

        query_hidden = self.query_pair_projector(query_pair)
        base_hidden = self.base_pair_projector(base_pair)
        language_hidden = self.language_projector(
            language.reshape(batch_size, -1)
        ).unsqueeze(1).expand(-1, candidate_count, -1)
        evidence_hidden = self.evidence_projector(evidence)
        hidden = self.fusion(torch.cat((
            query_hidden, base_hidden, language_hidden, evidence_hidden,
        ), dim=-1))
        sequence = hidden.transpose(0, 1)
        attended, _ = self.set_attention(
            sequence, sequence, sequence, key_padding_mask=~valid
        )
        hidden = self.set_norm(hidden + attended.transpose(0, 1))
        hidden = self.output_norm(hidden + self.set_ffn(hidden))

        action_logits = self.action_head(hidden).squeeze(-1)
        repair_logits = self.repair_head(hidden)
        break_logits = self.break_head(hidden)
        transition_utility = (
            self.transition_utility_head(hidden)
            if self.transition_utility_head is not None else None
        )
        iou_advantage = torch.tanh(
            self.iou_advantage_head(hidden).squeeze(-1)
        )
        non_parent = valid & ~parent_position
        feasible = (
            non_parent
            & (score_gap >= -1e-6)
            & (
                score_gap + self.promotion_margin
                <= self.max_parent_score_gap
            )
        )
        parse_confidence = row_values[0]
        relation_required = candidate_batch["relation_required"].bool()
        deterministic_reliable = (
            input_valid_rows
            & candidate_batch["has_target"].bool()
            & candidate_batch["structured_valid"].bool()
            & (parse_confidence >= self.min_parse_confidence)
            & (
                (~relation_required)
                | (
                    candidate_batch["anchor_top1_mass"].float()
                    >= self.min_anchor_mass
                )
            )
        )
        repair_probability = torch.sigmoid(repair_logits)
        break_probability = torch.sigmoid(break_logits)
        # Repair probability remains an auxiliary diagnostic.  Deployment
        # does not require a second rare-event repair/reliability certificate:
        # the explicit action-vs-fallback competition is trained on exactly
        # the safe-repair target.  Keeping those duplicate 0.5 vetoes made an
        # empirical-prior model correctly learn the roughly two-percent prior
        # and then abstain on every row.  Absolute break risk remains an
        # independent safety veto.
        predicted_repair = repair_probability.max(dim=-1).values > 0.5
        predicted_no_break = break_probability.max(dim=-1).values < 0.5
        eligible = (
            feasible
            & deterministic_reliable.unsqueeze(1)
            & predicted_no_break
        )
        deploy_values = action_logits.masked_fill(~eligible, -1e4)
        best_value, best_position = deploy_values.max(dim=1)
        switch_mask = best_value > 0.0
        parent_compact_position = parent_position.long().argmax(dim=1)
        selected_position = torch.where(
            switch_mask, best_position, parent_compact_position
        )
        selected_query_indices = torch.gather(
            candidate_batch["query_indices"].long(),
            1,
            selected_position.unsqueeze(1),
        ).squeeze(1)
        result = {
            "action_logits": action_logits,
            "repair_logits": repair_logits,
            "break_logits": break_logits,
            "iou_advantage": iou_advantage,
            "score_gap": score_gap,
            "feasible_mask": feasible,
            "non_parent_mask": non_parent,
            "deterministic_reliable_rows": deterministic_reliable,
            "input_valid_rows": input_valid_rows,
            "predicted_repair_mask": predicted_repair,
            "predicted_no_break_mask": predicted_no_break,
            "eligible_mask": eligible,
            "switch_mask": switch_mask,
            "selected_position": selected_position,
            "selected_query_indices": selected_query_indices,
            "parent_query_indices": candidate_batch[
                "parent_query_index"
            ].long(),
            "parent_position": parent_position,
            "valid_mask": valid,
        }
        if transition_utility is not None:
            result["transition_utility"] = transition_utility
        return result


def _masked_mean(values, mask):
    mask = mask.bool()
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def _empirical_binary_loss(logits, targets, valid):
    """Binary log loss that preserves the observed class prior.

    Heads thresholded as probabilities at deployment cannot use a
    class-balanced objective without a separate calibration step: balancing
    deliberately changes their prior.  This helper keeps the fixed 0.5
    boundary aligned with equal fix and break costs.
    """
    valid = valid.bool()
    targets = targets.bool()
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    return _masked_mean(losses, valid)


def compute_parent_relative_text_verifier_loss(
        verifier_outputs, candidate_ious, positive_margin=0.25,
        neutral_margin=0.25, sample_mask=None,
        counterfactual_training=False):
    """Train explicit repair, break-risk, advantage, and fallback actions."""
    positive_margin = _require_finite_scalar(
        "positive_margin", positive_margin, lower=0.0, upper=1.0
    )
    neutral_margin = _require_finite_scalar(
        "neutral_margin", neutral_margin, lower=0.0, upper=1.0
    )
    if not isinstance(counterfactual_training, bool):
        raise ValueError("counterfactual_training must be boolean")
    valid = verifier_outputs["valid_mask"].bool()
    parent_position = verifier_outputs["parent_position"].bool()
    if (not isinstance(candidate_ious, torch.Tensor)
            or candidate_ious.shape != valid.shape):
        raise ValueError("candidate_ious must align with compact candidates")
    candidate_ious = candidate_ious.detach().float()
    if sample_mask is None:
        sample_mask = torch.ones(
            valid.shape[0], dtype=torch.bool, device=valid.device
        )
    if (not isinstance(sample_mask, torch.Tensor)
            or sample_mask.shape != (valid.shape[0],)):
        raise ValueError("sample_mask must have shape [B]")
    sample_mask = sample_mask.to(device=valid.device).bool()
    finite_iou_rows = _finite_rows(candidate_ious)
    audit_sample_mask = sample_mask & finite_iou_rows
    candidate_ious = _finite_or_zero(candidate_ious).clamp(
        min=0.0, max=1.0
    )
    input_valid_rows = verifier_outputs.get("input_valid_rows")
    if (not isinstance(input_valid_rows, torch.Tensor)
            or input_valid_rows.shape != (valid.shape[0],)):
        raise ValueError("verifier input validity must have shape [B]")
    sample_mask = (
        audit_sample_mask
        & input_valid_rows.to(device=valid.device).bool()
    )
    parent_iou = _gather_parent(
        candidate_ious.unsqueeze(-1), parent_position
    ).squeeze(-1)
    parent_hit = torch.stack((
        parent_iou > 0.25,
        parent_iou > 0.50,
    ), dim=-1)
    candidate_hit = torch.stack((
        candidate_ious > 0.25,
        candidate_ious > 0.50,
    ), dim=-1)
    repair = (~parent_hit.unsqueeze(1)) & candidate_hit
    breaks = parent_hit.unsqueeze(1) & (~candidate_hit)
    valid_non_parent = valid & ~parent_position
    training_candidates = valid_non_parent & sample_mask.unsqueeze(1)
    supervised = (
        training_candidates
        & verifier_outputs["feasible_mask"].bool()
        & verifier_outputs["deterministic_reliable_rows"].bool().unsqueeze(1)
    )
    safe_repair = (
        supervised
        & repair.any(dim=-1)
        & (~breaks.any(dim=-1))
    )

    quality = (
        2.0 * candidate_hit[..., 0].float()
        + candidate_hit[..., 1].float()
        + 0.5 * candidate_ious
    )
    positive_quality = quality.masked_fill(~safe_repair, -1e4)
    best_positive = positive_quality.argmax(dim=1)
    has_positive = safe_repair.any(dim=1)
    target_action = torch.where(
        has_positive, best_positive + 1, torch.zeros_like(best_positive)
    )
    candidate_action = verifier_outputs["action_logits"].masked_fill(
        ~supervised, -1e4
    )
    fallback = candidate_action.new_zeros(candidate_action.shape[0], 1)
    action_loss = _masked_mean(
        F.cross_entropy(
            torch.cat((fallback, candidate_action), dim=1),
            target_action,
            reduction="none",
        ),
        sample_mask,
    )

    auxiliary_mask = supervised if counterfactual_training else (
        training_candidates
    )
    break_loss = _empirical_binary_loss(
        verifier_outputs["break_logits"],
        breaks,
        auxiliary_mask.unsqueeze(-1).expand_as(breaks),
    )
    repair_loss = _empirical_binary_loss(
        verifier_outputs["repair_logits"],
        repair,
        auxiliary_mask.unsqueeze(-1).expand_as(repair),
    )
    transition_utility_target = (
        candidate_hit.float() - parent_hit.unsqueeze(1).float()
    )
    if counterfactual_training:
        transition_utility = verifier_outputs.get("transition_utility")
        if (not isinstance(transition_utility, torch.Tensor)
                or transition_utility.shape
                != transition_utility_target.shape):
            raise ValueError(
                "counterfactual training requires [B,K,2] transition utility"
            )
        transition_utility_loss = _masked_mean(
            F.smooth_l1_loss(
                transition_utility,
                transition_utility_target,
                reduction="none",
            ),
            supervised.unsqueeze(-1).expand_as(transition_utility_target),
        )
    else:
        transition_utility_loss = (
            verifier_outputs["action_logits"].sum() * 0.0
        )
    iou_advantage_target = (
        candidate_ious - parent_iou.unsqueeze(1)
    ).clamp(min=-1.0, max=1.0)
    iou_loss = _masked_mean(
        F.smooth_l1_loss(
            verifier_outputs["iou_advantage"],
            iou_advantage_target,
            reduction="none",
        ),
        auxiliary_mask,
    )
    positive_action_loss = _masked_mean(
        F.relu(positive_margin - verifier_outputs["action_logits"]),
        safe_repair,
    )
    preserve_mask = auxiliary_mask & ~safe_repair
    preserve_loss = _masked_mean(
        F.relu(neutral_margin + verifier_outputs["action_logits"]),
        preserve_mask,
    )
    loss = (
        action_loss
        + 0.5 * repair_loss
        + break_loss
        + 0.25 * iou_loss
        + 0.5 * positive_action_loss
        + 0.5 * preserve_loss
    )
    if counterfactual_training:
        loss = loss + transition_utility_loss

    selected_iou = torch.gather(
        candidate_ious,
        1,
        verifier_outputs["selected_position"].long().unsqueeze(1),
    ).squeeze(1)
    selected_hit025 = selected_iou > 0.25
    selected_hit050 = selected_iou > 0.50
    selected_hit = torch.stack((selected_hit025, selected_hit050), dim=-1)
    transition_masks = {
        "fix": (~parent_hit) & selected_hit,
        "break": parent_hit & (~selected_hit),
        "kept_correct": parent_hit & selected_hit,
        "kept_wrong": (~parent_hit) & (~selected_hit),
    }
    stats = {
        "action_loss": action_loss.detach(),
        "repair_loss": repair_loss.detach(),
        "break_loss": break_loss.detach(),
        "iou_loss": iou_loss.detach(),
        "positive_action_loss": positive_action_loss.detach(),
        "preserve_loss": preserve_loss.detach(),
        "candidate_positive_ratio": _masked_mean(
            safe_repair.float(), training_candidates
        ).detach(),
        "positive_row_ratio": _masked_mean(
            has_positive.float(), sample_mask
        ).detach(),
        "feasible_candidate_ratio": _masked_mean(
            verifier_outputs["feasible_mask"].float(), training_candidates
        ).detach(),
        "switch_ratio": _masked_mean(
            verifier_outputs["switch_mask"].float(), sample_mask
        ).detach(),
        "fallback_ratio": _masked_mean(
            (~verifier_outputs["switch_mask"].bool()).float(), sample_mask
        ).detach(),
        "predicted_repair_candidate_ratio": _masked_mean(
            verifier_outputs["predicted_repair_mask"].float(),
            training_candidates,
        ).detach(),
        "predicted_no_break_candidate_ratio": _masked_mean(
            verifier_outputs["predicted_no_break_mask"].float(),
            training_candidates,
        ).detach(),
        "eligible_candidate_ratio": _masked_mean(
            verifier_outputs["eligible_mask"].float(), training_candidates
        ).detach(),
        "fix025_ratio": (
            _masked_mean(
                ((~parent_hit[:, 0]) & selected_hit025).float(), sample_mask
            ).detach()
        ),
        "break025_ratio": (
            _masked_mean(
                (parent_hit[:, 0] & (~selected_hit025)).float(), sample_mask
            ).detach()
        ),
        "fix050_ratio": (
            _masked_mean(
                ((~parent_hit[:, 1]) & selected_hit050).float(), sample_mask
            ).detach()
        ),
        "break050_ratio": (
            _masked_mean(
                (parent_hit[:, 1] & (~selected_hit050)).float(), sample_mask
            ).detach()
        ),
        "parent_acc025": _masked_mean(
            parent_hit[:, 0].float(), sample_mask
        ).detach(),
        "parent_acc050": _masked_mean(
            parent_hit[:, 1].float(), sample_mask
        ).detach(),
        "selected_acc025": _masked_mean(
            selected_hit025.float(), sample_mask
        ).detach(),
        "selected_acc050": _masked_mean(
            selected_hit050.float(), sample_mask
        ).detach(),
        "audit_sample_count": audit_sample_mask.float().sum().detach(),
        "audit_switch_count": (
            verifier_outputs["switch_mask"].bool()
            & audit_sample_mask
        ).float().sum().detach(),
    }
    for name, transition in transition_masks.items():
        for index, suffix in ((0, "025"), (1, "050")):
            stats["audit_{}{}_count".format(name, suffix)] = (
                transition[:, index] & audit_sample_mask
            ).float().sum().detach()
    if counterfactual_training:
        pair_mask = supervised.unsqueeze(-1).expand_as(repair)
        stats.update({
            "transition_utility_loss": transition_utility_loss.detach(),
            "risk_loss": break_loss.detach(),
            "sample_count": sample_mask.float().sum().detach(),
            "supervised_candidate_count": supervised.float().sum().detach(),
            "positive_candidate_count": safe_repair.float().sum().detach(),
            "positive_row_count": (
                has_positive & sample_mask
            ).float().sum().detach(),
            "fix_pair_count": (pair_mask & repair).float().sum().detach(),
            "break_pair_count": (pair_mask & breaks).float().sum().detach(),
            "neutral_pair_count": (
                pair_mask & (~repair) & (~breaks)
            ).float().sum().detach(),
            "nonfinite_count": sum(
                (~torch.isfinite(value)).float().sum()
                for value in (
                    verifier_outputs["action_logits"],
                    verifier_outputs["repair_logits"],
                    verifier_outputs["break_logits"],
                    verifier_outputs["transition_utility"],
                    verifier_outputs["iou_advantage"],
                )
            ).detach(),
            "utility_positive_pair_ratio": _masked_mean(
                (transition_utility_target > 0).float(), pair_mask
            ).detach(),
            "utility_negative_pair_ratio": _masked_mean(
                (transition_utility_target < 0).float(), pair_mask
            ).detach(),
            "utility_neutral_pair_ratio": _masked_mean(
                (transition_utility_target == 0).float(), pair_mask
            ).detach(),
        })
    return {"loss": loss, "stats": stats}
