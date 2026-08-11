"""Pure query-then-variant reranking helpers for ScanRefer REC."""

import math
import numbers

import torch
from torch import nn

from .rec_selective_residual import (
    build_residual_scene_folds,
    canonical_scene_fold_sha256,
    scene_clustered_hit_delta_bootstrap,
)


QUERY_COUNT = 16
VARIANT_COUNT = 7
QUERY_FEATURE_DIM = 152
VARIANT_FEATURE_DIM = 25
QUERY_AUX_CONTINUOUS_DIM = 4
QUERY_AUX_BINARY_DIM = 2
VARIANT_AUX_CONTINUOUS_DIM = 2
VARIANT_AUX_BINARY_DIM = 2
HIERARCHICAL_THRESHOLDS = (0.25, 0.50)
HIERARCHICAL_THRESHOLD_WEIGHTS = (2.0, 1.0)
HIERARCHICAL_HIDDEN_DIMS = (64, 128)
HIERARCHICAL_WEIGHT_DECAYS = (1e-4, 1e-3)
HIERARCHICAL_FALSE_POSITIVE_COSTS = (2.0, 4.0)
HIERARCHICAL_MARGIN_PERCENTILES = (
    50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0,
)
HIERARCHICAL_FOLD_COUNT = 5
HIERARCHICAL_SEED = 0
HIERARCHICAL_BOOTSTRAP_REPLICATES = 10000


def _require_float32_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(name))
    if value.dtype != torch.float32:
        raise TypeError("{} must have float32 dtype".format(name))


def _require_bool_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(name))
    if value.dtype != torch.bool:
        raise TypeError("{} must have bool dtype".format(name))


def _require_int64_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(name))
    if value.dtype != torch.long:
        raise TypeError("{} must have int64 dtype".format(name))


def _require_shape(value, expected, name):
    if tuple(value.shape) != tuple(expected):
        raise ValueError(
            "{} must have shape {}, got {}".format(
                name, tuple(expected), tuple(value.shape)
            )
        )


def _require_finite_at_mask(value, mask, name):
    expanded_mask = mask
    while expanded_mask.dim() < value.dim():
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(value)
    if not bool(torch.isfinite(value[expanded_mask]).all().item()):
        raise ValueError("{} must be finite at valid positions".format(name))


def monotone_hit_probabilities(logits):
    """Map two logits to ordered probabilities for IoU 0.25 and 0.50."""
    _require_float32_tensor(logits, "logits")
    if logits.dim() < 1 or not all(logits.shape) or logits.shape[-1] != 2:
        raise ValueError("logits must have nonempty shape [...,2]")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    probability25 = logits[..., 0].sigmoid()
    probability50 = probability25 * logits[..., 1].sigmoid()
    return torch.stack((probability25, probability50), dim=-1)


def _validate_proposal_inputs(
        query_logits, variant_logits, query_valid, variant_valid):
    _require_float32_tensor(query_logits, "query_logits")
    _require_float32_tensor(variant_logits, "variant_logits")
    _require_bool_tensor(query_valid, "query_valid")
    _require_bool_tensor(variant_valid, "variant_valid")
    if query_logits.dim() != 3 or query_logits.shape[0] <= 0:
        raise ValueError("query_logits must have nonempty shape [B,16,2]")
    batch_size = query_logits.shape[0]
    shapes = (
        (query_logits, (batch_size, QUERY_COUNT, 2), "query_logits"),
        (
            variant_logits,
            (batch_size, QUERY_COUNT, VARIANT_COUNT, 2),
            "variant_logits",
        ),
        (query_valid, (batch_size, QUERY_COUNT), "query_valid"),
        (
            variant_valid,
            (batch_size, QUERY_COUNT, VARIANT_COUNT),
            "variant_valid",
        ),
    )
    for value, expected, name in shapes:
        _require_shape(value, expected, name)
    values = (variant_logits, query_valid, variant_valid)
    if any(value.device != query_logits.device for value in values):
        raise ValueError("hierarchical proposal inputs must share a device")
    if not bool(torch.isfinite(query_logits).all().item()):
        raise ValueError("query_logits must be finite")
    if not bool(torch.isfinite(variant_logits).all().item()):
        raise ValueError("variant_logits must be finite")
    if bool((variant_valid & ~query_valid.unsqueeze(2)).any().item()):
        raise ValueError(
            "variant_valid cannot be true under an invalid query"
        )
    if not torch.equal(query_valid, variant_valid.any(dim=2)):
        raise ValueError(
            "query_valid must exactly identify queries with variant_valid"
        )
    if not bool(query_valid.any(dim=1).all().item()):
        raise ValueError("every row must contain a valid query")


def select_hierarchical_proposal(
        query_logits, variant_logits, query_valid, variant_valid):
    """Select a query and then one of its variants deterministically."""
    _validate_proposal_inputs(
        query_logits, variant_logits, query_valid, variant_valid
    )
    query_probability = monotone_hit_probabilities(query_logits)
    variant_probability = monotone_hit_probabilities(variant_logits)
    query_utility = (
        HIERARCHICAL_THRESHOLD_WEIGHTS[0] * query_probability[..., 0]
        + HIERARCHICAL_THRESHOLD_WEIGHTS[1] * query_probability[..., 1]
    )
    variant_utility = (
        HIERARCHICAL_THRESHOLD_WEIGHTS[0] * variant_probability[..., 0]
        + HIERARCHICAL_THRESHOLD_WEIGHTS[1] * variant_probability[..., 1]
    )
    selected_query = query_utility.masked_fill(
        ~query_valid, -float("inf")
    ).argmax(dim=1)
    rows = torch.arange(
        query_logits.shape[0], device=query_logits.device
    )
    selected_variant = variant_utility[rows, selected_query].masked_fill(
        ~variant_valid[rows, selected_query], -float("inf")
    ).argmax(dim=1)
    selected_flat = selected_query * VARIANT_COUNT + selected_variant
    return {
        "query_indices": selected_query,
        "variant_indices": selected_variant,
        "flat_indices": selected_flat,
        "query_utility": query_utility,
        "variant_utility": variant_utility,
    }


def build_hierarchical_targets(candidate_ious, variant_valid):
    """Build detached strict hit targets for both hierarchy levels."""
    _require_float32_tensor(candidate_ious, "candidate_ious")
    _require_bool_tensor(variant_valid, "variant_valid")
    if candidate_ious.dim() != 3 or candidate_ious.shape[0] <= 0:
        raise ValueError(
            "candidate_ious must have nonempty shape [B,16,7]"
        )
    batch_size = candidate_ious.shape[0]
    expected = (batch_size, QUERY_COUNT, VARIANT_COUNT)
    _require_shape(candidate_ious, expected, "candidate_ious")
    _require_shape(variant_valid, expected, "variant_valid")
    if candidate_ious.device != variant_valid.device:
        raise ValueError(
            "candidate_ious and variant_valid must share a device"
        )
    if not bool(variant_valid.reshape(batch_size, -1).any(dim=1).all().item()):
        raise ValueError("every row must contain a valid variant")
    _require_finite_at_mask(candidate_ious, variant_valid, "candidate_ious")
    valid_ious = candidate_ious[variant_valid]
    if (bool((valid_ious < 0.0).any().item())
            or bool((valid_ious > 1.0).any().item())):
        raise ValueError("valid candidate_ious must lie in [0,1]")
    variant_targets = torch.stack(
        tuple(
            candidate_ious.gt(threshold)
            for threshold in HIERARCHICAL_THRESHOLDS
        ),
        dim=-1,
    ) & variant_valid.unsqueeze(-1)
    query_targets = variant_targets.any(dim=2)
    query_valid = variant_valid.any(dim=2)
    return {
        "query_targets": query_targets.detach(),
        "variant_targets": variant_targets.detach(),
        "query_valid": query_valid.detach(),
    }


def _require_fixed_grid_value(value, allowed, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("{} must be numeric".format(name))
    if value not in allowed:
        raise ValueError("{} is outside the fixed grid".format(name))
    return float(value)


def _false_positive_weighted_bce(probability, target, cost):
    epsilon = torch.finfo(probability.dtype).eps
    probability = probability.clamp(min=epsilon, max=1.0 - epsilon)
    target_value = target.to(probability.dtype)
    return -(
        target_value * probability.log()
        + cost * (1.0 - target_value) * torch.log1p(-probability)
    )


def compute_hierarchical_loss(
        query_logits, variant_logits, query_targets, variant_targets,
        query_valid, variant_valid, false_positive_cost):
    """Compute query and variant BCEs with equal weight per input row."""
    _validate_proposal_inputs(
        query_logits, variant_logits, query_valid, variant_valid
    )
    _require_bool_tensor(query_targets, "query_targets")
    _require_bool_tensor(variant_targets, "variant_targets")
    batch_size = query_logits.shape[0]
    _require_shape(
        query_targets,
        (batch_size, QUERY_COUNT, 2),
        "query_targets",
    )
    _require_shape(
        variant_targets,
        (batch_size, QUERY_COUNT, VARIANT_COUNT, 2),
        "variant_targets",
    )
    tensors = (
        variant_logits,
        query_targets,
        variant_targets,
        query_valid,
        variant_valid,
    )
    if any(value.device != query_logits.device for value in tensors):
        raise ValueError("hierarchical loss tensors must share a device")
    if bool(query_targets[~query_valid].any().item()):
        raise ValueError("query_targets must be false for invalid queries")
    if bool(variant_targets[~variant_valid].any().item()):
        raise ValueError("variant_targets must be false for invalid variants")
    if not torch.equal(query_targets, variant_targets.any(dim=2)):
        raise ValueError(
            "query_targets must equal any valid variant target"
        )
    false_positive_cost = _require_fixed_grid_value(
        false_positive_cost,
        HIERARCHICAL_FALSE_POSITIVE_COSTS,
        "false_positive_cost",
    )

    query_probability = monotone_hit_probabilities(query_logits)
    variant_probability = monotone_hit_probabilities(variant_logits)
    query_mask = query_valid.to(query_logits.dtype)
    variant_mask = variant_valid.to(variant_logits.dtype)
    query_denominator = query_mask.sum(dim=1)
    variant_denominator = variant_mask.sum(dim=2).clamp_min(1.0)
    query_head_losses = []
    variant_head_losses = []
    for threshold_index in range(2):
        query_values = _false_positive_weighted_bce(
            query_probability[..., threshold_index],
            query_targets[..., threshold_index],
            false_positive_cost,
        )
        query_row_loss = (
            query_values * query_mask
        ).sum(dim=1) / query_denominator
        query_head_losses.append(query_row_loss.mean())

        variant_values = _false_positive_weighted_bce(
            variant_probability[..., threshold_index],
            variant_targets[..., threshold_index],
            false_positive_cost,
        )
        variant_query_loss = (
            variant_values * variant_mask
        ).sum(dim=2) / variant_denominator
        variant_row_loss = (
            variant_query_loss * query_mask
        ).sum(dim=1) / query_denominator
        variant_head_losses.append(variant_row_loss.mean())
    weight_total = float(sum(HIERARCHICAL_THRESHOLD_WEIGHTS))
    query_loss = sum(
        weight * loss for weight, loss in zip(
            HIERARCHICAL_THRESHOLD_WEIGHTS, query_head_losses
        )
    ) / weight_total
    variant_loss = sum(
        weight * loss for weight, loss in zip(
            HIERARCHICAL_THRESHOLD_WEIGHTS, variant_head_losses
        )
    ) / weight_total
    loss = query_loss + variant_loss
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("hierarchical loss must be finite")

    stats = {}
    for prefix, targets, valid in (
            ("query", query_targets, query_valid),
            ("variant", variant_targets, variant_valid)):
        for threshold_index, suffix in enumerate(("025", "050")):
            positive = valid & targets[..., threshold_index]
            negative = valid & ~targets[..., threshold_index]
            stats["{}_positive{}".format(prefix, suffix)] = \
                positive.sum().detach()
            stats["{}_negative{}".format(prefix, suffix)] = \
                negative.sum().detach()
    return loss, stats


def apply_hierarchical_policy(
        base_scores, proposed_flat_indices, predicted_gain,
        variant_valid, margin):
    """Promote a hierarchical proposal only when its gain clears the gate."""
    _require_float32_tensor(base_scores, "base_scores")
    _require_int64_tensor(
        proposed_flat_indices, "proposed_flat_indices"
    )
    _require_float32_tensor(predicted_gain, "predicted_gain")
    _require_bool_tensor(variant_valid, "variant_valid")
    if base_scores.dim() != 2 or base_scores.shape[0] <= 0:
        raise ValueError("base_scores must have nonempty shape [B,112]")
    batch_size = base_scores.shape[0]
    candidate_count = QUERY_COUNT * VARIANT_COUNT
    _require_shape(
        base_scores, (batch_size, candidate_count), "base_scores"
    )
    _require_shape(
        proposed_flat_indices,
        (batch_size,),
        "proposed_flat_indices",
    )
    _require_shape(predicted_gain, (batch_size,), "predicted_gain")
    _require_shape(
        variant_valid,
        (batch_size, QUERY_COUNT, VARIANT_COUNT),
        "variant_valid",
    )
    values = (proposed_flat_indices, predicted_gain, variant_valid)
    if any(value.device != base_scores.device for value in values):
        raise ValueError("hierarchical policy tensors must share a device")
    if (bool(torch.isnan(base_scores).any().item())
            or bool(torch.isposinf(base_scores).any().item())):
        raise ValueError("base_scores may contain only finite values and -inf")
    flat_valid = variant_valid.reshape(batch_size, candidate_count)
    finite_scores = torch.isfinite(base_scores)
    if not torch.equal(flat_valid, finite_scores):
        raise ValueError(
            "variant_valid must exactly identify finite base_scores"
        )
    if not bool(flat_valid.any(dim=1).all().item()):
        raise ValueError("every base score row needs a valid candidate")
    if (bool((proposed_flat_indices < 0).any().item())
            or bool((proposed_flat_indices >= candidate_count).any().item())):
        raise ValueError("proposed_flat_indices are out of range")
    if not bool(torch.isfinite(predicted_gain).all().item()):
        raise ValueError("predicted_gain must be finite")
    if isinstance(margin, bool) or not isinstance(margin, numbers.Real):
        raise TypeError("margin must be numeric")
    margin = float(margin)
    if math.isnan(margin) or margin < 0.0:
        raise ValueError("margin must be nonnegative")

    baseline_indices = base_scores.argmax(dim=1)
    rows = torch.arange(batch_size, device=base_scores.device)
    proposal_valid = flat_valid[rows, proposed_flat_indices]
    switch_mask = (
        proposal_valid
        & proposed_flat_indices.ne(baseline_indices)
        & predicted_gain.gt(0.0)
        & predicted_gain.ge(margin)
    )
    selected_indices = torch.where(
        switch_mask, proposed_flat_indices, baseline_indices
    )
    scores = base_scores.clone()
    positive_infinity = torch.full_like(predicted_gain, float("inf"))
    promoted = torch.nextafter(
        base_scores.max(dim=1).values, positive_infinity
    )
    if not bool(torch.isfinite(promoted).all().item()):
        raise ValueError("base score maximum cannot be promoted finitely")
    switched_rows = switch_mask.nonzero(as_tuple=False).reshape(-1)
    scores[switched_rows, selected_indices[switched_rows]] = promoted[
        switched_rows
    ]
    return {
        "scores": scores,
        "selected_indices": selected_indices,
        "switch_mask": switch_mask,
        "baseline_indices": baseline_indices,
    }


def build_hierarchical_scene_folds(
        scan_ids, fold_count=HIERARCHICAL_FOLD_COUNT,
        seed=HIERARCHICAL_SEED):
    """Assign whole scenes to the fixed hierarchical OOF folds."""
    return build_residual_scene_folds(
        scan_ids, fold_count=fold_count, seed=seed
    )


def canonical_hierarchical_scene_fold_sha256(mapping):
    """Hash the fixed scene-to-fold assignment canonically."""
    return canonical_scene_fold_sha256(mapping)


def hierarchical_scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline_hits, proposed_hits):
    """Bootstrap paired hierarchical hit deltas by whole scene."""
    return scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline_hits, proposed_hits
    )


def _coerce_hierarchical_scan_ids(scan_ids):
    if (not isinstance(scan_ids, (tuple, list))
            or isinstance(scan_ids, (str, bytes))):
        raise TypeError("scan_ids must be a sequence")
    values = tuple(scan_ids)
    if not values:
        raise ValueError("scan_ids must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("scan_ids must contain nonempty strings")
    if len(set(values)) < HIERARCHICAL_FOLD_COUNT:
        raise ValueError("scan_ids must contain at least five scenes")
    return values


def _coerce_hierarchical_hit_bits(values, name, expected_length):
    if isinstance(values, torch.Tensor):
        if values.dim() != 1:
            raise ValueError("{} must be one-dimensional".format(name))
        values = values.detach().cpu().tolist()
    if (not isinstance(values, (tuple, list))
            or isinstance(values, (str, bytes))):
        raise TypeError("{} must be a sequence".format(name))
    if len(values) != expected_length:
        raise ValueError("{} must align with scan_ids".format(name))
    result = []
    for value in values:
        if not isinstance(value, numbers.Integral):
            raise TypeError("{} must contain integer hit bits".format(name))
        value = int(value)
        if value not in (0, 1):
            raise ValueError("{} must contain only 0 and 1".format(name))
        result.append(value)
    return tuple(result)


_HIERARCHICAL_SELECTION_REQUIRED_FIELDS = (
    "hidden_dim",
    "weight_decay",
    "false_positive_cost",
    "margin_percentile",
    "margin",
    "scan_ids",
    "baseline_hits025",
    "proposed_hits025",
    "baseline_hits050",
    "proposed_hits050",
    "switch_bits",
)


def _validate_hierarchical_selection_candidate(candidate):
    if not isinstance(candidate, dict):
        raise TypeError("each selection candidate must be a dictionary")
    missing = [
        field for field in _HIERARCHICAL_SELECTION_REQUIRED_FIELDS
        if field not in candidate
    ]
    if missing:
        raise ValueError(
            "selection candidate is missing {}".format(
                ", ".join(missing)
            )
        )
    hidden_dim = int(_require_fixed_grid_value(
        candidate["hidden_dim"], HIERARCHICAL_HIDDEN_DIMS, "hidden_dim"
    ))
    weight_decay = _require_fixed_grid_value(
        candidate["weight_decay"],
        HIERARCHICAL_WEIGHT_DECAYS,
        "weight_decay",
    )
    false_positive_cost = _require_fixed_grid_value(
        candidate["false_positive_cost"],
        HIERARCHICAL_FALSE_POSITIVE_COSTS,
        "false_positive_cost",
    )
    percentile = candidate["margin_percentile"]
    margin = candidate["margin"]
    if isinstance(margin, bool) or not isinstance(margin, numbers.Real):
        raise TypeError("margin must be numeric")
    margin = float(margin)
    if math.isnan(margin) or margin < 0.0:
        raise ValueError("margin must be nonnegative")
    sentinel = percentile is None
    if sentinel:
        if not math.isinf(margin) or margin < 0.0:
            raise ValueError("no-switch sentinel margin must be +inf")
    else:
        percentile = _require_fixed_grid_value(
            percentile,
            HIERARCHICAL_MARGIN_PERCENTILES,
            "margin_percentile",
        )
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("selected margin must be positive and finite")

    scan_ids = _coerce_hierarchical_scan_ids(candidate["scan_ids"])
    hit_fields = {}
    for name in (
            "baseline_hits025", "proposed_hits025",
            "baseline_hits050", "proposed_hits050"):
        hit_fields[name] = _coerce_hierarchical_hit_bits(
            candidate[name], name, len(scan_ids)
        )
    switch_bits = _coerce_hierarchical_hit_bits(
        candidate["switch_bits"], "switch_bits", len(scan_ids)
    )
    for hits025, hits050, name in (
            (hit_fields["baseline_hits025"],
             hit_fields["baseline_hits050"], "baseline"),
            (hit_fields["proposed_hits025"],
             hit_fields["proposed_hits050"], "proposed")):
        if any(hit050 > hit025 for hit025, hit050 in zip(
                hits025, hits050)):
            raise ValueError(
                "{} 0.50 hits must be a subset of 0.25 hits".format(name)
            )
    for row_index in range(len(scan_ids)):
        changed = any(
            hit_fields[proposed][row_index]
            != hit_fields[baseline][row_index]
            for baseline, proposed in (
                ("baseline_hits025", "proposed_hits025"),
                ("baseline_hits050", "proposed_hits050"),
            )
        )
        if changed and not switch_bits[row_index]:
            raise ValueError("hit changes require a policy switch")
    if sentinel and (
            any(switch_bits)
            or hit_fields["baseline_hits025"]
            != hit_fields["proposed_hits025"]
            or hit_fields["baseline_hits050"]
            != hit_fields["proposed_hits050"]):
        raise ValueError("no-switch sentinel must preserve the baseline")
    transition_diagnostics = candidate.get("transition_diagnostics")
    if transition_diagnostics is not None:
        transition_fields = (
            "selected_query_changes",
            "same_query_variant_changes",
            "wrong_query_recoveries025",
            "wrong_query_recoveries050",
            "wrong_variant_recoveries025",
            "wrong_variant_recoveries050",
        )
        if (not isinstance(transition_diagnostics, dict)
                or set(transition_diagnostics) != set(transition_fields)):
            raise ValueError("transition diagnostics fields changed")
        for name in transition_fields:
            value = transition_diagnostics[name]
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ValueError(
                    "transition diagnostic {} is invalid".format(name)
                )
        switches = int(sum(switch_bits))
        if (transition_diagnostics["selected_query_changes"]
                + transition_diagnostics["same_query_variant_changes"]
                != switches):
            raise ValueError("transition switches do not reconcile")
        for suffix, baseline_field, proposed_field in (
                ("025", "baseline_hits025", "proposed_hits025"),
                ("050", "baseline_hits050", "proposed_hits050")):
            fixes = sum(
                int(before == 0 and after == 1)
                for before, after in zip(
                    hit_fields[baseline_field], hit_fields[proposed_field]
                )
            )
            recoveries = (
                transition_diagnostics[
                    "wrong_query_recoveries{}".format(suffix)
                ]
                + transition_diagnostics[
                    "wrong_variant_recoveries{}".format(suffix)
                ]
            )
            if recoveries != fixes:
                raise ValueError(
                    "transition recoveries do not reconcile at {}".format(
                        suffix
                    )
                )
    return {
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "false_positive_cost": false_positive_cost,
        "margin_percentile": percentile,
        "margin": margin,
        "scan_ids": scan_ids,
        "baseline_hits025": hit_fields["baseline_hits025"],
        "proposed_hits025": hit_fields["proposed_hits025"],
        "baseline_hits050": hit_fields["baseline_hits050"],
        "proposed_hits050": hit_fields["proposed_hits050"],
        "switch_bits": switch_bits,
        "sentinel": sentinel,
        "transition_diagnostics": (
            None if transition_diagnostics is None
            else dict(transition_diagnostics)
        ),
    }


def _hierarchical_candidate_diagnostics(candidate):
    scene_folds = build_hierarchical_scene_folds(candidate["scan_ids"])
    fold_deltas = {
        str(fold): {"hits025": 0, "hits050": 0}
        for fold in range(HIERARCHICAL_FOLD_COUNT)
    }
    for row_index, scene_id in enumerate(candidate["scan_ids"]):
        record = fold_deltas[str(scene_folds[scene_id])]
        record["hits025"] += (
            candidate["proposed_hits025"][row_index]
            - candidate["baseline_hits025"][row_index]
        )
        record["hits050"] += (
            candidate["proposed_hits050"][row_index]
            - candidate["baseline_hits050"][row_index]
        )
    bootstrap025 = hierarchical_scene_clustered_hit_delta_bootstrap(
        candidate["scan_ids"],
        candidate["baseline_hits025"],
        candidate["proposed_hits025"],
    )
    bootstrap050 = hierarchical_scene_clustered_hit_delta_bootstrap(
        candidate["scan_ids"],
        candidate["baseline_hits050"],
        candidate["proposed_hits050"],
    )
    predicates = {
        "not_no_switch": not candidate["sentinel"],
        "all_folds_nonnegative025": all(
            record["hits025"] >= 0 for record in fold_deltas.values()
        ),
        "all_folds_nonnegative050": all(
            record["hits050"] >= 0 for record in fold_deltas.values()
        ),
        "pooled_delta025_positive": bootstrap025["delta_hits"] > 0,
        "bootstrap025_lower_bound_nonnegative": (
            bootstrap025["lower_bound_95"] >= 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            bootstrap050["lower_bound_95"] >= 0
        ),
    }
    sample_count = len(candidate["scan_ids"])
    switches = int(sum(candidate["switch_bits"]))
    baseline = {}
    proposed = {}
    effects = {}
    for threshold, baseline_field, proposed_field in (
            ("0.25", "baseline_hits025", "proposed_hits025"),
            ("0.50", "baseline_hits050", "proposed_hits050")):
        baseline_hits = candidate[baseline_field]
        proposed_hits = candidate[proposed_field]
        threshold_effects = {
            "fixes": 0,
            "breaks": 0,
            "neutral_switches": 0,
            "kept_correct": 0,
            "kept_wrong": 0,
        }
        for baseline_hit, proposed_hit, switched in zip(
                baseline_hits, proposed_hits, candidate["switch_bits"]):
            if switched:
                if baseline_hit == 0 and proposed_hit == 1:
                    threshold_effects["fixes"] += 1
                elif baseline_hit == 1 and proposed_hit == 0:
                    threshold_effects["breaks"] += 1
                else:
                    threshold_effects["neutral_switches"] += 1
            elif baseline_hit:
                threshold_effects["kept_correct"] += 1
            else:
                threshold_effects["kept_wrong"] += 1
        baseline[threshold] = {"hits": int(sum(baseline_hits))}
        proposed[threshold] = {"hits": int(sum(proposed_hits))}
        effects[threshold] = threshold_effects
    eligible = all(predicates.values())
    result = {
        "hidden_dim": candidate["hidden_dim"],
        "weight_decay": candidate["weight_decay"],
        "false_positive_cost": candidate["false_positive_cost"],
        "margin_percentile": candidate["margin_percentile"],
        "margin": None if candidate["sentinel"] else candidate["margin"],
        "no_switch": candidate["sentinel"],
        "sample_count": sample_count,
        "switches": switches,
        "abstentions": sample_count - switches,
        "switch_rate": switches / float(sample_count),
        "baseline": baseline,
        "proposed": proposed,
        "effects": effects,
        "delta_hits025": bootstrap025["delta_hits"],
        "delta_hits050": bootstrap050["delta_hits"],
        "fold_deltas": fold_deltas,
        "bootstrap025": bootstrap025,
        "bootstrap050": bootstrap050,
        "eligibility_predicates": predicates,
        "failed_predicates": sorted(
            name for name, passed in predicates.items() if not passed
        ),
        "eligible": bool(eligible),
        "selected": "hierarchical" if eligible else "baseline",
    }
    if candidate["transition_diagnostics"] is not None:
        result["transition_diagnostics"] = dict(
            candidate["transition_diagnostics"]
        )
    return result


def choose_hierarchical_configuration(candidates):
    """Choose one policy using only scene-disjoint OOF hit records."""
    if not isinstance(candidates, (tuple, list)) or not candidates:
        raise ValueError("candidates must be a nonempty sequence")
    validated = [
        _validate_hierarchical_selection_candidate(candidate)
        for candidate in candidates
    ]
    reference = validated[0]
    for candidate in validated[1:]:
        if (candidate["scan_ids"] != reference["scan_ids"]
                or candidate["baseline_hits025"]
                != reference["baseline_hits025"]
                or candidate["baseline_hits050"]
                != reference["baseline_hits050"]):
            raise ValueError(
                "all candidates must use the same OOF baseline rows"
            )
    diagnostics = [
        _hierarchical_candidate_diagnostics(candidate)
        for candidate in validated
    ]
    eligible = [record for record in diagnostics if record["eligible"]]
    summary = {
        "candidate_count": len(diagnostics),
        "eligible_candidate_count": len(eligible),
        "candidate_diagnostics": diagnostics,
    }
    if not eligible:
        summary.update({
            "eligible": False,
            "reason": "no-eligible-configuration",
            "selected": "baseline",
        })
        return summary
    winner = max(
        eligible,
        key=lambda record: (
            2 * record["delta_hits025"] + record["delta_hits050"],
            record["margin"],
            -record["switches"],
            -record["hidden_dim"],
            record["weight_decay"],
            record["false_positive_cost"],
        ),
    )
    selection = dict(winner)
    selection.update(summary)
    return selection


class HierarchicalQueryVariantReranker(nn.Module):
    """Select a compact query first and a geometry variant second."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool)
                or hidden_dim not in HIERARCHICAL_HIDDEN_DIMS):
            raise ValueError(
                "hidden_dim must be one of {}".format(
                    HIERARCHICAL_HIDDEN_DIMS
                )
            )
        if (not isinstance(dropout, (float, int))
                or isinstance(dropout, bool)
                or not math.isfinite(float(dropout))
                or not 0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout must be finite and in [0,1)")
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.variant_encoder = nn.Sequential(
            nn.Linear(VARIANT_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(
                QUERY_FEATURE_DIM
                + QUERY_AUX_CONTINUOUS_DIM
                + QUERY_AUX_BINARY_DIM
                + 2 * hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.query_head = nn.Linear(hidden_dim, 2)
        self.variant_head = nn.Sequential(
            nn.Linear(
                2 * hidden_dim
                + VARIANT_AUX_CONTINUOUS_DIM
                + VARIANT_AUX_BINARY_DIM,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _validate_inputs(
            query_features, variant_features, query_aux_continuous,
            query_aux_binary, variant_aux_continuous,
            variant_aux_binary, query_valid, variant_valid):
        float_values = (
            (query_features, "query_features"),
            (variant_features, "variant_features"),
            (query_aux_continuous, "query_aux_continuous"),
            (variant_aux_continuous, "variant_aux_continuous"),
        )
        bool_values = (
            (query_aux_binary, "query_aux_binary"),
            (variant_aux_binary, "variant_aux_binary"),
            (query_valid, "query_valid"),
            (variant_valid, "variant_valid"),
        )
        for value, name in float_values:
            _require_float32_tensor(value, name)
        for value, name in bool_values:
            _require_bool_tensor(value, name)
        if query_features.dim() != 3 or query_features.shape[0] <= 0:
            raise ValueError(
                "query_features must have nonempty shape [B,16,152]"
            )
        batch_size = query_features.shape[0]
        shapes = (
            (query_features, (batch_size, QUERY_COUNT, QUERY_FEATURE_DIM),
             "query_features"),
            (variant_features,
             (batch_size, QUERY_COUNT, VARIANT_COUNT, VARIANT_FEATURE_DIM),
             "variant_features"),
            (query_aux_continuous,
             (batch_size, QUERY_COUNT, QUERY_AUX_CONTINUOUS_DIM),
             "query_aux_continuous"),
            (query_aux_binary,
             (batch_size, QUERY_COUNT, QUERY_AUX_BINARY_DIM),
             "query_aux_binary"),
            (variant_aux_continuous,
             (batch_size, QUERY_COUNT, VARIANT_COUNT,
              VARIANT_AUX_CONTINUOUS_DIM),
             "variant_aux_continuous"),
            (variant_aux_binary,
             (batch_size, QUERY_COUNT, VARIANT_COUNT,
              VARIANT_AUX_BINARY_DIM),
             "variant_aux_binary"),
            (query_valid, (batch_size, QUERY_COUNT), "query_valid"),
            (variant_valid,
             (batch_size, QUERY_COUNT, VARIANT_COUNT), "variant_valid"),
        )
        for value, expected, name in shapes:
            _require_shape(value, expected, name)
        devices = tuple(value.device for value, _ in float_values + bool_values)
        if any(device != devices[0] for device in devices[1:]):
            raise ValueError("hierarchical input tensors must share a device")
        derived_query_valid = variant_valid.any(dim=2)
        if bool((variant_valid & ~query_valid.unsqueeze(2)).any().item()):
            raise ValueError(
                "variant_valid cannot be true under an invalid query"
            )
        if not torch.equal(query_valid, derived_query_valid):
            raise ValueError(
                "query_valid must exactly identify queries with valid variants"
            )
        if not bool(query_valid.any(dim=1).all().item()):
            raise ValueError("every row must contain a valid query")
        _require_finite_at_mask(
            query_features, query_valid, "query_features"
        )
        _require_finite_at_mask(
            query_aux_continuous, query_valid, "query_aux_continuous"
        )
        _require_finite_at_mask(
            variant_features, variant_valid, "variant_features"
        )
        _require_finite_at_mask(
            variant_aux_continuous,
            variant_valid,
            "variant_aux_continuous",
        )

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
        query_input = torch.cat(
            (
                safe_query_features,
                safe_query_aux,
                query_aux_binary.to(query_features.dtype),
                variant_mean,
                variant_max,
            ),
            dim=-1,
        )
        query_embedding = self.query_encoder(query_input)
        query_embedding = torch.where(
            query_mask, query_embedding, torch.zeros_like(query_embedding)
        )
        query_logits = self.query_head(query_embedding)
        query_logits = torch.where(
            query_mask, query_logits, torch.zeros_like(query_logits)
        )

        safe_variant_aux = torch.where(
            variant_mask,
            variant_aux_continuous,
            torch.zeros_like(variant_aux_continuous),
        )
        expanded_query_embedding = query_embedding.unsqueeze(2).expand(
            -1, -1, VARIANT_COUNT, -1
        )
        variant_input = torch.cat(
            (
                expanded_query_embedding,
                variant_embedding,
                safe_variant_aux,
                variant_aux_binary.to(query_features.dtype),
            ),
            dim=-1,
        )
        variant_logits = self.variant_head(variant_input)
        variant_logits = torch.where(
            variant_mask, variant_logits, torch.zeros_like(variant_logits)
        )
        return {
            "query_logits": query_logits,
            "variant_logits": variant_logits,
            "query_embedding": query_embedding,
            "variant_embedding": variant_embedding,
        }
