"""Pure train/runtime contracts for selective residual REC reranking."""

import functools
import hashlib
import json
import math
import numbers
import random

import numpy as np
import torch
import torch.nn.functional as functional


PAIR_FEATURE_DIM = 185
RESIDUAL_THRESHOLDS = (0.25, 0.50)
RESIDUAL_HEAD_WEIGHTS = (2.0, 1.0)
RESIDUAL_CLASS_NAMES = ("break", "neutral", "fix")
RESIDUAL_HIDDEN_DIMS = (0, 64)
RESIDUAL_WEIGHT_DECAYS = (1e-4, 1e-3)
RESIDUAL_BREAK_COSTS = (2.0, 4.0, 8.0)
RESIDUAL_MARGIN_PERCENTILES = (
    50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0,
)
RESIDUAL_FOLD_COUNT = 5
RESIDUAL_SEED = 0
RESIDUAL_BOOTSTRAP_REPLICATES = 10000


def _require_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(name))


def _require_float32(value, name):
    _require_tensor(value, name)
    if value.dtype != torch.float32:
        raise TypeError("{} must have float32 dtype".format(name))


def _require_bool(value, name):
    _require_tensor(value, name)
    if value.dtype != torch.bool:
        raise TypeError("{} must have bool dtype".format(name))


def _require_int64(value, name):
    _require_tensor(value, name)
    if value.dtype != torch.long:
        raise TypeError("{} must have int64 dtype".format(name))


def _require_same_device(named_tensors):
    devices = {value.device for _name, value in named_tensors}
    if len(devices) != 1:
        names = ", ".join(name for name, _value in named_tensors)
        raise ValueError("{} must be on the same device".format(names))


def _validate_candidate_axis(valid_mask, baseline_indices):
    if valid_mask.dim() != 2 or not all(valid_mask.shape):
        raise ValueError("valid_mask must have nonempty shape [B,C]")
    batch_size, candidate_count = valid_mask.shape
    if baseline_indices.shape != (batch_size,):
        raise ValueError("baseline_indices must have shape [B]")
    if (bool((baseline_indices < 0).any().item())
            or bool((baseline_indices >= candidate_count).any().item())):
        raise ValueError("baseline index is out of range")
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("every row needs at least one valid candidate")
    rows = torch.arange(batch_size, device=valid_mask.device)
    if not bool(valid_mask[rows, baseline_indices].all().item()):
        raise ValueError("every baseline index must identify a valid candidate")
    return batch_size, candidate_count, rows


def _require_finite_valid(value, valid_mask, name):
    if not bool(torch.isfinite(value[valid_mask]).all().item()):
        raise ValueError("{} must be finite for valid candidates".format(name))


def build_selective_pair_features(
        normalized_features, valid_mask, baseline_indices,
        parent_rank, geometry_rank, threshold_logits, iou_estimate,
        query_positions):
    """Build deployable alternative-minus-baseline features."""
    _require_float32(normalized_features, "normalized_features")
    _require_bool(valid_mask, "valid_mask")
    _require_int64(baseline_indices, "baseline_indices")
    _require_float32(parent_rank, "parent_rank")
    _require_float32(geometry_rank, "geometry_rank")
    _require_float32(threshold_logits, "threshold_logits")
    _require_float32(iou_estimate, "iou_estimate")
    _require_int64(query_positions, "query_positions")
    named_tensors = (
        ("normalized_features", normalized_features),
        ("valid_mask", valid_mask),
        ("baseline_indices", baseline_indices),
        ("parent_rank", parent_rank),
        ("geometry_rank", geometry_rank),
        ("threshold_logits", threshold_logits),
        ("iou_estimate", iou_estimate),
        ("query_positions", query_positions),
    )
    _require_same_device(named_tensors)
    batch_size, candidate_count, rows = _validate_candidate_axis(
        valid_mask, baseline_indices
    )
    if normalized_features.shape != (
            batch_size, candidate_count, 179):
        raise ValueError(
            "normalized_features must have shape [B,C,179]"
        )
    matrix_shape = (batch_size, candidate_count)
    for name, value in (
            ("parent_rank", parent_rank),
            ("geometry_rank", geometry_rank),
            ("iou_estimate", iou_estimate),
            ("query_positions", query_positions)):
        if value.shape != matrix_shape:
            raise ValueError("{} must have shape [B,C]".format(name))
    if threshold_logits.shape != (
            batch_size, candidate_count, 2):
        raise ValueError("threshold_logits must have shape [B,C,2]")

    for name, value in (
            ("normalized_features", normalized_features),
            ("parent_rank", parent_rank),
            ("geometry_rank", geometry_rank),
            ("threshold_logits", threshold_logits),
            ("iou_estimate", iou_estimate)):
        _require_finite_valid(value, valid_mask, name)

    baseline_features = normalized_features[rows, baseline_indices]
    feature_delta = normalized_features - baseline_features.unsqueeze(1)
    parent_delta = (
        parent_rank
        - parent_rank[rows, baseline_indices].unsqueeze(1)
    )
    geometry_delta = (
        geometry_rank
        - geometry_rank[rows, baseline_indices].unsqueeze(1)
    )
    threshold_probability = threshold_logits.sigmoid()
    threshold_delta = (
        threshold_probability
        - threshold_probability[rows, baseline_indices].unsqueeze(1)
    )
    iou_delta = (
        iou_estimate
        - iou_estimate[rows, baseline_indices].unsqueeze(1)
    )
    same_query = query_positions.eq(
        query_positions[rows, baseline_indices].unsqueeze(1)
    ).to(normalized_features.dtype)
    pair_features = torch.cat([
        feature_delta,
        parent_delta.unsqueeze(-1),
        geometry_delta.unsqueeze(-1),
        threshold_delta,
        iou_delta.unsqueeze(-1),
        same_query.unsqueeze(-1),
    ], dim=-1)
    positions = torch.arange(candidate_count, device=valid_mask.device)
    pair_valid = valid_mask & positions.unsqueeze(0).ne(
        baseline_indices.unsqueeze(1)
    )
    pair_features = torch.where(
        pair_valid.unsqueeze(-1),
        pair_features,
        torch.zeros_like(pair_features),
    )
    if pair_features.shape[-1] != PAIR_FEATURE_DIM:
        raise AssertionError("selective pair feature dimension changed")
    if not bool(torch.isfinite(pair_features).all().item()):
        raise ValueError("built pair features must be finite")
    return {
        "features": pair_features,
        "valid_mask": pair_valid,
        "baseline_indices": baseline_indices,
    }


def _validate_thresholds(thresholds):
    if not isinstance(thresholds, (tuple, list)) or len(thresholds) != 2:
        raise TypeError("thresholds must be a two-item sequence")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in thresholds):
        raise TypeError("thresholds must contain real numbers")
    values = tuple(float(value) for value in thresholds)
    if values != RESIDUAL_THRESHOLDS:
        raise ValueError("thresholds must match the fixed residual thresholds")
    return values


def build_selective_pair_targets(
        candidate_ious, valid_mask, baseline_indices,
        thresholds=RESIDUAL_THRESHOLDS):
    """Build detached break/neutral/fix labels at both strict IoU tiers."""
    _require_float32(candidate_ious, "candidate_ious")
    _require_bool(valid_mask, "valid_mask")
    _require_int64(baseline_indices, "baseline_indices")
    _require_same_device((
        ("candidate_ious", candidate_ious),
        ("valid_mask", valid_mask),
        ("baseline_indices", baseline_indices),
    ))
    batch_size, candidate_count, rows = _validate_candidate_axis(
        valid_mask, baseline_indices
    )
    if candidate_ious.shape != (batch_size, candidate_count):
        raise ValueError("candidate_ious must have shape [B,C]")
    if not bool(torch.isfinite(candidate_ious).all().item()):
        raise ValueError("candidate_ious must be finite")
    if (bool((candidate_ious < 0.0).any().item())
            or bool((candidate_ious > 1.0).any().item())):
        raise ValueError("candidate_ious must lie in [0,1]")
    threshold_values = _validate_thresholds(thresholds)

    targets = []
    for threshold in threshold_values:
        baseline_hit = (
            candidate_ious[rows, baseline_indices] > threshold
        )
        alternative_hit = candidate_ious > threshold
        target = torch.ones_like(candidate_ious, dtype=torch.long)
        target[baseline_hit.unsqueeze(1) & ~alternative_hit] = 0
        target[~baseline_hit.unsqueeze(1) & alternative_hit] = 2
        target.masked_fill_(~valid_mask, 1)
        target[rows, baseline_indices] = 1
        targets.append(target)
    return torch.stack(targets, dim=-1).detach()


class SelectiveResidualModel(torch.nn.Module):
    """Small two-threshold classifier initialized to baseline parity."""

    def __init__(self, input_dim=PAIR_FEATURE_DIM, hidden_dim=64,
                 dropout=0.1):
        super().__init__()
        if input_dim != PAIR_FEATURE_DIM:
            raise ValueError("input_dim must be {}".format(PAIR_FEATURE_DIM))
        if hidden_dim not in (0, 64):
            raise ValueError("hidden_dim must be 0 or 64")
        if (isinstance(dropout, bool)
                or not isinstance(dropout, (int, float))
                or not math.isfinite(float(dropout))
                or float(dropout) < 0.0
                or float(dropout) >= 1.0):
            raise ValueError("dropout must lie in [0,1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        if hidden_dim == 0:
            self.encoder = torch.nn.Identity()
            width = input_dim
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            width = hidden_dim
        self.head = torch.nn.Linear(width, 6)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, pair_features, pair_valid):
        _require_float32(pair_features, "pair_features")
        _require_bool(pair_valid, "pair_valid")
        _require_same_device((
            ("pair_features", pair_features),
            ("pair_valid", pair_valid),
        ))
        if (pair_features.dim() != 3 or not all(pair_features.shape)
                or pair_features.shape[-1] != self.input_dim):
            raise ValueError("pair_features must have shape [B,C,185]")
        if pair_valid.shape != pair_features.shape[:2]:
            raise ValueError("pair_valid must have shape [B,C]")
        if not bool(torch.isfinite(pair_features).all().item()):
            raise ValueError("pair_features must be finite")
        encoded = self.encoder(pair_features)
        logits = self.head(encoded).reshape(
            pair_features.shape[0], pair_features.shape[1], 2, 3
        )
        return logits.masked_fill(~pair_valid[:, :, None, None], 0.0)


def _validate_positive_number(value, name, allow_infinity=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a real number".format(name))
    value = float(value)
    if math.isnan(value) or value <= 0.0:
        raise ValueError("{} must be positive".format(name))
    if not allow_infinity and not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def compute_selective_residual_loss(
        logits, targets, pair_valid, break_cost,
        threshold_weights=RESIDUAL_HEAD_WEIGHTS):
    """Compute class-weighted loss after balancing alternatives per row."""
    _require_float32(logits, "logits")
    _require_int64(targets, "targets")
    _require_bool(pair_valid, "pair_valid")
    _require_same_device((
        ("logits", logits),
        ("targets", targets),
        ("pair_valid", pair_valid),
    ))
    if logits.dim() != 4 or logits.shape[-2:] != (2, 3):
        raise ValueError("logits must have shape [B,C,2,3]")
    if not all(logits.shape[:2]):
        raise ValueError("logits must have a nonempty candidate axis")
    if targets.shape != logits.shape[:3]:
        raise ValueError("targets must have shape [B,C,2]")
    if pair_valid.shape != logits.shape[:2]:
        raise ValueError("pair_valid must have shape [B,C]")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    if (bool((targets < 0).any().item())
            or bool((targets > 2).any().item())):
        raise ValueError("targets must use classes 0, 1, and 2")
    break_cost = _validate_positive_number(break_cost, "break_cost")
    if (not isinstance(threshold_weights, (tuple, list))
            or len(threshold_weights) != 2):
        raise ValueError("threshold_weights must contain two values")
    weights = tuple(
        _validate_positive_number(value, "threshold weight")
        for value in threshold_weights
    )

    class_weights = logits.new_tensor([break_cost, 1.0, 1.0])
    informative = pair_valid.any(dim=1)
    if not bool(informative.any().item()):
        zero = logits.sum() * 0.0
        count = pair_valid.sum().detach()
        return zero, {
            "informative_rows": count,
            "break025": count,
            "neutral025": count,
            "fix025": count,
            "break050": count,
            "neutral050": count,
            "fix050": count,
        }

    head_losses = []
    denominator = pair_valid.sum(dim=1).clamp(min=1).to(logits.dtype)
    for head_index in range(2):
        values = functional.cross_entropy(
            logits[:, :, head_index, :].reshape(-1, 3),
            targets[:, :, head_index].reshape(-1),
            weight=class_weights,
            reduction="none",
        ).reshape_as(pair_valid)
        row_loss = (
            values * pair_valid.to(values.dtype)
        ).sum(dim=1) / denominator
        head_losses.append(row_loss[informative].mean())
    loss = (
        weights[0] * head_losses[0] + weights[1] * head_losses[1]
    ) / sum(weights)
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("selective residual loss must be finite")
    stats = {"informative_rows": informative.sum().detach()}
    for head_index, suffix in enumerate(("025", "050")):
        for class_index, name in enumerate(RESIDUAL_CLASS_NAMES):
            stats[name + suffix] = (
                pair_valid & targets[:, :, head_index].eq(class_index)
            ).sum().detach()
    return loss, stats


def expected_selective_gain(logits):
    """Return the fixed 2:1 weighted expected signed hit change."""
    _require_float32(logits, "logits")
    if (logits.dim() != 4 or logits.shape[-2:] != (2, 3)
            or not all(logits.shape[:2])):
        raise ValueError("logits must have shape [B,C,2,3]")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    probabilities = logits.softmax(dim=-1)
    signed = probabilities[:, :, :, 2] - probabilities[:, :, :, 0]
    return (
        RESIDUAL_HEAD_WEIGHTS[0] * signed[:, :, 0]
        + RESIDUAL_HEAD_WEIGHTS[1] * signed[:, :, 1]
    )


def apply_selective_policy(base_scores, pair_gain, pair_valid, margin):
    """Promote one alternative only when its expected gain clears the gate."""
    _require_float32(base_scores, "base_scores")
    _require_float32(pair_gain, "pair_gain")
    _require_bool(pair_valid, "pair_valid")
    _require_same_device((
        ("base_scores", base_scores),
        ("pair_gain", pair_gain),
        ("pair_valid", pair_valid),
    ))
    if base_scores.dim() != 2 or not all(base_scores.shape):
        raise ValueError("base_scores must have nonempty shape [B,C]")
    if pair_gain.shape != base_scores.shape:
        raise ValueError("pair_gain must have shape [B,C]")
    if pair_valid.shape != base_scores.shape:
        raise ValueError("pair_valid must have shape [B,C]")
    if (bool(torch.isnan(base_scores).any().item())
            or bool(torch.isposinf(base_scores).any().item())):
        raise ValueError("base_scores may contain only finite values and -inf")
    valid = torch.isfinite(base_scores)
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("every base score row needs a finite candidate")
    if bool((pair_valid & ~valid).any().item()):
        raise ValueError("pair_valid must be a subset of finite base scores")
    if not bool(torch.isfinite(pair_gain[pair_valid]).all().item()):
        raise ValueError("pair_gain must be finite for valid pairs")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise TypeError("margin must be a real number")
    margin = float(margin)
    if math.isnan(margin) or margin < 0.0:
        raise ValueError("margin must be nonnegative")

    baseline_indices = base_scores.argmax(dim=1)
    rows = torch.arange(base_scores.shape[0], device=base_scores.device)
    if bool(pair_valid[rows, baseline_indices].any().item()):
        raise ValueError("the baseline candidate cannot be a valid pair")
    candidate_gain = pair_gain.masked_fill(~pair_valid, -float("inf"))
    best_gain, selected_indices = candidate_gain.max(dim=1)
    switch_mask = pair_valid.any(dim=1) & (best_gain > 0.0) & (
        best_gain >= margin
    )
    selected_indices = torch.where(
        switch_mask, selected_indices, baseline_indices
    )
    scores = base_scores.clone()
    positive_infinity = torch.full_like(scores[:, 0], float("inf"))
    promoted = torch.nextafter(
        base_scores.masked_fill(~valid, -float("inf")).max(dim=1).values,
        positive_infinity,
    )
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


def _coerce_scan_ids(scan_ids, minimum_scene_count=1):
    if (not isinstance(scan_ids, (tuple, list))
            or isinstance(scan_ids, (str, bytes))):
        raise TypeError("scan_ids must be a sequence")
    values = tuple(scan_ids)
    if not values:
        raise ValueError("scan_ids must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("scan_ids must contain nonempty strings")
    if len(set(values)) < minimum_scene_count:
        raise ValueError(
            "scan_ids must contain at least {} scenes".format(
                minimum_scene_count
            )
        )
    return values


def build_residual_scene_folds(
        scan_ids, fold_count=RESIDUAL_FOLD_COUNT, seed=RESIDUAL_SEED):
    """Assign complete scenes to the fixed seed-0 five-fold protocol."""
    if (not isinstance(fold_count, int) or isinstance(fold_count, bool)
            or fold_count != RESIDUAL_FOLD_COUNT):
        raise ValueError(
            "fold_count must be {}".format(RESIDUAL_FOLD_COUNT)
        )
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or seed != RESIDUAL_SEED):
        raise ValueError("seed must be {}".format(RESIDUAL_SEED))
    values = _coerce_scan_ids(scan_ids, minimum_scene_count=fold_count)
    scenes = sorted(set(values))
    random.Random(seed).shuffle(scenes)
    return {
        scene_id: index % fold_count
        for index, scene_id in enumerate(scenes)
    }


def canonical_scene_fold_sha256(mapping):
    """Hash the canonical sorted scene-to-fold assignment."""
    if not isinstance(mapping, dict) or not mapping:
        raise TypeError("mapping must be a nonempty dictionary")
    if any(not isinstance(scene_id, str) or not scene_id
           for scene_id in mapping):
        raise ValueError("scene fold keys must be nonempty strings")
    folds = tuple(mapping.values())
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in folds):
        raise TypeError("scene folds must be integers")
    if any(value < 0 or value >= RESIDUAL_FOLD_COUNT for value in folds):
        raise ValueError("scene fold is out of range")
    if set(folds) != set(range(RESIDUAL_FOLD_COUNT)):
        raise ValueError("scene fold mapping must cover all five folds")
    payload = json.dumps(
        [[scene_id, mapping[scene_id]] for scene_id in sorted(mapping)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _coerce_hit_bits(values, name, expected_length):
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


@functools.lru_cache(maxsize=8)
def _scene_bootstrap_indices(scene_count):
    random_state = random.Random(RESIDUAL_SEED)
    sample_count = RESIDUAL_BOOTSTRAP_REPLICATES * scene_count
    flat = np.fromiter(
        (random_state.randrange(scene_count) for _index in range(sample_count)),
        dtype=np.int64,
        count=sample_count,
    )
    indices = flat.reshape(RESIDUAL_BOOTSTRAP_REPLICATES, scene_count)
    indices.setflags(write=False)
    return indices


def scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline_hits, proposed_hits):
    """Bootstrap paired hit deltas by resampling whole scenes."""
    scan_ids = _coerce_scan_ids(scan_ids, minimum_scene_count=2)
    baseline_hits = _coerce_hit_bits(
        baseline_hits, "baseline_hits", len(scan_ids)
    )
    proposed_hits = _coerce_hit_bits(
        proposed_hits, "proposed_hits", len(scan_ids)
    )
    scene_deltas = {scene_id: 0 for scene_id in sorted(set(scan_ids))}
    for scene_id, baseline_hit, proposed_hit in zip(
            scan_ids, baseline_hits, proposed_hits):
        scene_deltas[scene_id] += proposed_hit - baseline_hit
    ordered_deltas = np.asarray(
        [scene_deltas[scene_id] for scene_id in sorted(scene_deltas)],
        dtype=np.int64,
    )
    sampled = ordered_deltas[
        _scene_bootstrap_indices(len(ordered_deltas))
    ].sum(axis=1)
    sampled.sort()
    lower_index = int(math.ceil(
        0.05 * RESIDUAL_BOOTSTRAP_REPLICATES
    )) - 1
    return {
        "confidence": 0.95,
        "delta_hits": int(ordered_deltas.sum()),
        "lower_bound_95": int(sampled[lower_index]),
        "replicates": RESIDUAL_BOOTSTRAP_REPLICATES,
        "scene_count": int(len(ordered_deltas)),
        "seed": RESIDUAL_SEED,
    }


_SELECTION_REQUIRED_FIELDS = (
    "hidden_dim",
    "weight_decay",
    "break_cost",
    "margin_percentile",
    "margin",
    "scan_ids",
    "baseline_hits025",
    "proposed_hits025",
    "baseline_hits050",
    "proposed_hits050",
    "switch_bits",
)


def _require_grid_value(value, allowed, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("{} must be numeric".format(name))
    if value not in allowed:
        raise ValueError("{} is outside the fixed grid".format(name))
    return value


def _validate_selection_candidate(candidate):
    if not isinstance(candidate, dict):
        raise TypeError("each selection candidate must be a dictionary")
    missing = [
        field for field in _SELECTION_REQUIRED_FIELDS
        if field not in candidate
    ]
    if missing:
        raise ValueError(
            "selection candidate is missing {}".format(", ".join(missing))
        )
    hidden_dim = int(_require_grid_value(
        candidate["hidden_dim"], RESIDUAL_HIDDEN_DIMS, "hidden_dim"
    ))
    weight_decay = float(_require_grid_value(
        candidate["weight_decay"],
        RESIDUAL_WEIGHT_DECAYS,
        "weight_decay",
    ))
    break_cost = float(_require_grid_value(
        candidate["break_cost"], RESIDUAL_BREAK_COSTS, "break_cost"
    ))
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
        percentile = float(_require_grid_value(
            percentile,
            RESIDUAL_MARGIN_PERCENTILES,
            "margin_percentile",
        ))
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("selected margin must be positive and finite")

    scan_ids = _coerce_scan_ids(
        candidate["scan_ids"], minimum_scene_count=RESIDUAL_FOLD_COUNT
    )
    hit_fields = {}
    for name in (
            "baseline_hits025", "proposed_hits025",
            "baseline_hits050", "proposed_hits050"):
        hit_fields[name] = _coerce_hit_bits(
            candidate[name], name, len(scan_ids)
        )
    switch_bits = _coerce_hit_bits(
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
    if sentinel:
        if any(switch_bits) or any(
                hit_fields[baseline] != hit_fields[proposed]
                for baseline, proposed in (
                    ("baseline_hits025", "proposed_hits025"),
                    ("baseline_hits050", "proposed_hits050"),
                )):
            raise ValueError("no-switch sentinel must preserve the baseline")
    return {
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "break_cost": break_cost,
        "margin_percentile": percentile,
        "margin": margin,
        "scan_ids": scan_ids,
        "baseline_hits025": hit_fields["baseline_hits025"],
        "proposed_hits025": hit_fields["proposed_hits025"],
        "baseline_hits050": hit_fields["baseline_hits050"],
        "proposed_hits050": hit_fields["proposed_hits050"],
        "switch_bits": switch_bits,
        "sentinel": sentinel,
    }


def _candidate_diagnostics(candidate):
    scene_folds = build_residual_scene_folds(candidate["scan_ids"])
    fold_deltas = {
        str(fold): {"hits025": 0, "hits050": 0}
        for fold in range(RESIDUAL_FOLD_COUNT)
    }
    for row_index, scene_id in enumerate(candidate["scan_ids"]):
        fold_record = fold_deltas[str(scene_folds[scene_id])]
        fold_record["hits025"] += (
            candidate["proposed_hits025"][row_index]
            - candidate["baseline_hits025"][row_index]
        )
        fold_record["hits050"] += (
            candidate["proposed_hits050"][row_index]
            - candidate["baseline_hits050"][row_index]
        )
    bootstrap025 = scene_clustered_hit_delta_bootstrap(
        candidate["scan_ids"],
        candidate["baseline_hits025"],
        candidate["proposed_hits025"],
    )
    bootstrap050 = scene_clustered_hit_delta_bootstrap(
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
        "bootstrap025_lower_bound_positive": (
            bootstrap025["lower_bound_95"] > 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            bootstrap050["lower_bound_95"] >= 0
        ),
    }
    eligible = all(predicates.values())
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
    return {
        "hidden_dim": candidate["hidden_dim"],
        "weight_decay": candidate["weight_decay"],
        "break_cost": candidate["break_cost"],
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
        "selected": "residual" if eligible else "baseline",
    }


def choose_selective_configuration(candidates):
    """Choose one policy using only pooled scene-cross-fitted hit records."""
    if not isinstance(candidates, (tuple, list)) or not candidates:
        raise ValueError("candidates must be a nonempty sequence")
    validated = [
        _validate_selection_candidate(candidate) for candidate in candidates
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
        _candidate_diagnostics(candidate) for candidate in validated
    ]
    eligible = [record for record in diagnostics if record["eligible"]]
    selection_summary = {
        "candidate_count": len(diagnostics),
        "eligible_candidate_count": len(eligible),
        "candidate_diagnostics": diagnostics,
    }
    if not eligible:
        selection_summary.update({
            "eligible": False,
            "reason": "no-eligible-configuration",
            "selected": "baseline",
        })
        return selection_summary
    winner = max(
        eligible,
        key=lambda record: (
            2 * record["delta_hits025"] + record["delta_hits050"],
            record["margin"],
            -record["switches"],
            int(record["hidden_dim"] == 0),
            record["weight_decay"],
            record["break_cost"],
        ),
    )
    selection = dict(winner)
    selection.update(selection_summary)
    return selection


__all__ = [
    "PAIR_FEATURE_DIM",
    "RESIDUAL_BREAK_COSTS",
    "RESIDUAL_THRESHOLDS",
    "RESIDUAL_HEAD_WEIGHTS",
    "RESIDUAL_CLASS_NAMES",
    "RESIDUAL_HIDDEN_DIMS",
    "RESIDUAL_MARGIN_PERCENTILES",
    "RESIDUAL_WEIGHT_DECAYS",
    "SelectiveResidualModel",
    "apply_selective_policy",
    "build_residual_scene_folds",
    "build_selective_pair_features",
    "build_selective_pair_targets",
    "canonical_scene_fold_sha256",
    "choose_selective_configuration",
    "compute_selective_residual_loss",
    "expected_selective_gain",
    "scene_clustered_hit_delta_bootstrap",
]
