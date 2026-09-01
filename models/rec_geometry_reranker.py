"""Pure feature and score helpers for flat REC geometry reranking."""

import math

import torch

from .rec_candidate_adapter import scatter_candidate_scores


REC_GEOMETRY_MODEL_SCHEMA_VERSION = "rec-geometry-flat-v1"
FLAT_PARENT_PRIOR_VERSION = \
    "score-desc-query-index-asc-regressed-first-v2"


def _require_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(name))


def _require_float_tensor(value, name):
    _require_tensor(value, name)
    if not torch.is_floating_point(value):
        raise TypeError("{} must be floating point".format(name))


def _require_bool_tensor(value, name):
    _require_tensor(value, name)
    if value.dtype != torch.bool:
        raise TypeError("{} must have bool dtype".format(name))


def _all_finite(value):
    return bool(torch.isfinite(value).all().item())


def _validate_feature_names(names, expected_size, name):
    if not isinstance(names, (list, tuple)):
        raise TypeError("{} must be a list or tuple".format(name))
    if len(names) != expected_size:
        raise ValueError("{} must match its feature dimension".format(name))
    if any(not isinstance(item, str) or not item for item in names):
        raise ValueError("{} must contain nonempty strings".format(name))
    return tuple(names)


def stable_query_descending_order(query_scores):
    """Return query indices ordered by ``(-score, query_index)``."""
    _require_float_tensor(query_scores, "query_scores")
    if query_scores.dim() != 2 or not all(query_scores.shape):
        raise ValueError("query_scores must have nonempty shape [B,Q]")
    if (bool(torch.isnan(query_scores).any().item())
            or bool(torch.isposinf(query_scores).any().item())):
        raise ValueError("query_scores may contain only finite values and -inf")
    if not bool(torch.isfinite(query_scores).any(dim=1).all().item()):
        raise ValueError("every query score row needs a finite value")

    cpu_scores = query_scores.detach().cpu()
    orders = []
    for batch_idx in range(query_scores.shape[0]):
        values = cpu_scores[batch_idx].tolist()
        orders.append(sorted(
            range(query_scores.shape[1]),
            key=lambda query_index: (-values[query_index], query_index),
        ))
    return torch.tensor(
        orders, dtype=torch.long, device=query_scores.device
    )


def build_deployed_parent_state(compact_scores, query_indices,
                                candidate_valid, num_queries):
    """Reconstruct the exact deployed parent ranking on the full Q axis."""
    _require_float_tensor(compact_scores, "compact_scores")
    _require_tensor(query_indices, "query_indices")
    _require_bool_tensor(candidate_valid, "candidate_valid")
    if compact_scores.dim() != 2 or not all(compact_scores.shape):
        raise ValueError("compact_scores must have nonempty shape [B,K]")
    if query_indices.shape != compact_scores.shape:
        raise ValueError("query_indices must match compact_scores")
    if candidate_valid.shape != compact_scores.shape:
        raise ValueError("candidate_valid must match compact_scores")
    if query_indices.dtype != torch.long:
        raise TypeError("query_indices must have int64 dtype")
    if (compact_scores.device != query_indices.device
            or compact_scores.device != candidate_valid.device):
        raise ValueError("parent tensors must be on the same device")
    if (not isinstance(num_queries, int) or isinstance(num_queries, bool)
            or num_queries <= 0):
        raise ValueError("num_queries must be a positive integer")
    if not bool(candidate_valid.any(dim=1).all().item()):
        raise ValueError("every sample needs at least one valid candidate")
    valid_indices = query_indices[candidate_valid]
    if (bool((valid_indices < 0).any().item())
            or bool((valid_indices >= num_queries).any().item())):
        raise ValueError("valid query index is out of range")
    for batch_idx in range(query_indices.shape[0]):
        row_indices = query_indices[batch_idx, candidate_valid[batch_idx]]
        if int(torch.unique(row_indices).numel()) != int(row_indices.numel()):
            raise ValueError("valid query indices must be unique per sample")
    if not _all_finite(compact_scores[candidate_valid]):
        raise ValueError("valid compact scores must be finite")

    query_scores = scatter_candidate_scores(
        compact_scores,
        query_indices,
        candidate_valid,
        num_queries,
        fill_value=-float("inf"),
    )
    query_order = stable_query_descending_order(query_scores)
    top1_query_index = query_order[:, 0]
    parent_top1_mask = (
        query_indices == top1_query_index.unsqueeze(1)
    ) & candidate_valid
    return {
        "compact_scores": compact_scores,
        "query_scores": query_scores,
        "query_indices": query_indices,
        "candidate_valid": candidate_valid,
        "query_order": query_order,
        "top1_query_index": top1_query_index,
        "parent_top1_mask": parent_top1_mask,
    }


def build_rec_geometry_model_inputs(
        base_features, geometry_features, parent_scores,
        parent_top1_mask, geometry_valid,
        base_feature_names, geometry_feature_names):
    """Build query-major, variant-minor flat geometry model inputs."""
    _require_float_tensor(base_features, "base_features")
    _require_float_tensor(geometry_features, "geometry_features")
    _require_float_tensor(parent_scores, "parent_scores")
    _require_bool_tensor(parent_top1_mask, "parent_top1_mask")
    _require_bool_tensor(geometry_valid, "geometry_valid")
    if base_features.dim() != 3 or not all(base_features.shape):
        raise ValueError("base_features must have nonempty shape [B,K,Dq]")
    if geometry_features.dim() != 4 or not all(geometry_features.shape):
        raise ValueError(
            "geometry_features must have nonempty shape [B,K,G,Dg]"
        )
    batch_size, candidates, base_dim = base_features.shape
    if geometry_features.shape[:2] != (batch_size, candidates):
        raise ValueError("base and geometry candidate axes must match")
    variants = geometry_features.shape[2]
    if parent_scores.shape != (batch_size, candidates):
        raise ValueError("parent_scores must have shape [B,K]")
    if parent_top1_mask.shape != (batch_size, candidates):
        raise ValueError("parent_top1_mask must have shape [B,K]")
    if geometry_valid.shape != (batch_size, candidates, variants):
        raise ValueError("geometry_valid must have shape [B,K,G]")
    tensors = (
        geometry_features, parent_scores, parent_top1_mask, geometry_valid
    )
    if any(value.device != base_features.device for value in tensors):
        raise ValueError("geometry model input tensors must share a device")
    if (geometry_features.dtype != base_features.dtype
            or parent_scores.dtype != base_features.dtype):
        raise TypeError("floating geometry inputs must share a dtype")

    base_names = _validate_feature_names(
        base_feature_names, base_dim, "base_feature_names"
    )
    geometry_names = _validate_feature_names(
        geometry_feature_names,
        geometry_features.shape[-1],
        "geometry_feature_names",
    )
    feature_names = (
        base_names + geometry_names
        + ("parent_score", "parent_is_deployed_top1")
    )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("geometry model feature names must be unique")

    valid = geometry_valid
    if not bool(valid.reshape(batch_size, -1).any(dim=1).all().item()):
        raise ValueError("every sample needs at least one valid geometry")
    query_has_valid_geometry = valid.any(dim=2)
    if not _all_finite(base_features[query_has_valid_geometry]):
        raise ValueError("base features used by valid geometry must be finite")
    if not _all_finite(geometry_features[valid]):
        raise ValueError("valid geometry features must be finite")
    if not _all_finite(parent_scores[query_has_valid_geometry]):
        raise ValueError("parent scores used by valid geometry must be finite")

    base = base_features.unsqueeze(2).expand(-1, -1, variants, -1)
    score = parent_scores.unsqueeze(2).unsqueeze(3).expand(
        -1, -1, variants, 1
    )
    top1 = parent_top1_mask.unsqueeze(2).unsqueeze(3).expand(
        -1, -1, variants, 1
    ).to(base_features.dtype)
    features = torch.cat(
        [base, geometry_features, score, top1], dim=-1
    ).reshape(batch_size, candidates * variants, -1)
    flat_valid = valid.reshape(batch_size, candidates * variants)
    features = torch.where(
        flat_valid.unsqueeze(-1), features, torch.zeros_like(features)
    )
    query_positions = torch.arange(
        candidates, dtype=torch.long, device=base_features.device
    ).view(1, candidates, 1).expand(
        batch_size, candidates, variants
    ).reshape(batch_size, -1)
    variant_indices = torch.arange(
        variants, dtype=torch.long, device=base_features.device
    ).view(1, 1, variants).expand(
        batch_size, candidates, variants
    ).reshape(batch_size, -1)
    return {
        "schema_version": REC_GEOMETRY_MODEL_SCHEMA_VERSION,
        "feature_names": feature_names,
        "features": features,
        "valid_mask": flat_valid,
        "query_positions": query_positions,
        "variant_indices": variant_indices,
    }


def _validate_parent_prior_inputs(parent_state, geometry_valid,
                                  regressed_variant_index):
    if not isinstance(parent_state, dict):
        raise TypeError("parent_state must be a dictionary")
    required = (
        "compact_scores", "query_scores", "query_order", "query_indices",
        "candidate_valid",
    )
    if any(key not in parent_state for key in required):
        raise ValueError("parent_state is missing required deployed state")
    compact_scores = parent_state["compact_scores"]
    query_scores = parent_state["query_scores"]
    query_order = parent_state["query_order"]
    query_indices = parent_state["query_indices"]
    candidate_valid = parent_state["candidate_valid"]
    _require_float_tensor(compact_scores, "parent compact_scores")
    _require_float_tensor(query_scores, "parent query_scores")
    _require_tensor(query_order, "parent query_order")
    _require_tensor(query_indices, "parent query_indices")
    _require_bool_tensor(candidate_valid, "parent candidate_valid")
    _require_bool_tensor(geometry_valid, "geometry_valid")
    if query_scores.dim() != 2 or not all(query_scores.shape):
        raise ValueError("parent query_scores must have nonempty shape [B,Q]")
    if query_order.shape != query_scores.shape or query_order.dtype != torch.long:
        raise ValueError("parent query_order must be int64 with shape [B,Q]")
    if query_indices.dim() != 2 or query_indices.dtype != torch.long:
        raise ValueError("parent query_indices must be int64 with shape [B,K]")
    if compact_scores.shape != query_indices.shape:
        raise ValueError("parent compact_scores must match query_indices")
    if compact_scores.dtype != query_scores.dtype:
        raise TypeError("parent compact_scores and query_scores must share a dtype")
    if query_indices.shape[0] != query_scores.shape[0]:
        raise ValueError("parent query and score batch axes must match")
    if candidate_valid.shape != query_indices.shape:
        raise ValueError("parent candidate_valid must match query_indices")
    if geometry_valid.dim() != 3 or not all(geometry_valid.shape):
        raise ValueError("geometry_valid must have nonempty shape [B,K,G]")
    if geometry_valid.shape[:2] != query_indices.shape:
        raise ValueError("geometry_valid must match parent [B,K]")
    values = (
        compact_scores, query_order, query_indices, candidate_valid,
        geometry_valid,
    )
    if any(value.device != query_scores.device for value in values):
        raise ValueError("parent prior tensors must share a device")
    variants = geometry_valid.shape[2]
    if (not isinstance(regressed_variant_index, int)
            or isinstance(regressed_variant_index, bool)
            or not 0 <= regressed_variant_index < variants):
        raise ValueError("regressed_variant_index is out of range")
    if not bool(candidate_valid.any(dim=1).all().item()):
        raise ValueError("every sample needs a valid parent candidate")
    if not _all_finite(compact_scores[candidate_valid]):
        raise ValueError("valid parent compact_scores must be finite")
    if bool((geometry_valid & ~candidate_valid.unsqueeze(2)).any().item()):
        raise ValueError("padded parent candidates cannot have valid geometry")
    if not torch.equal(geometry_valid.any(dim=2), candidate_valid):
        raise ValueError(
            "geometry query validity must exactly match candidate_valid"
        )
    num_queries = query_scores.shape[1]
    expected_indices = torch.arange(
        num_queries, dtype=torch.long, device=query_scores.device
    ).unsqueeze(0).expand_as(query_order)
    if not torch.equal(query_order.sort(dim=1).values, expected_indices):
        raise ValueError("parent query_order must be a query-axis permutation")
    ordered_scores = torch.gather(query_scores, 1, query_order)
    if bool((ordered_scores[:, 1:] > ordered_scores[:, :-1]).any().item()):
        raise ValueError("parent query_order scores must be non-increasing")
    canonical_order = stable_query_descending_order(query_scores)
    canonical_scores = torch.gather(query_scores, 1, canonical_order)
    finite_order = query_order[torch.isfinite(ordered_scores)]
    canonical_finite_order = canonical_order[torch.isfinite(canonical_scores)]
    if not torch.equal(finite_order, canonical_finite_order):
        raise ValueError(
            "parent finite query_order must follow (-score, query_index)"
        )

    selected_query_mask = torch.zeros_like(query_scores, dtype=torch.bool)
    for batch_idx in range(query_indices.shape[0]):
        row_indices = query_indices[batch_idx, candidate_valid[batch_idx]]
        if (bool((row_indices < 0).any().item())
                or bool((row_indices >= num_queries).any().item())):
            raise ValueError("valid parent query index is out of range")
        if int(torch.unique(row_indices).numel()) != int(row_indices.numel()):
            raise ValueError("valid parent query indices must be unique")
        selected_query_mask[batch_idx, row_indices] = True
        selected_scores = query_scores[batch_idx, row_indices]
        if not _all_finite(selected_scores):
            raise ValueError("selected parent query scores must be finite")
    if not torch.equal(torch.isfinite(query_scores), selected_query_mask):
        raise ValueError(
            "finite parent query scores must exactly match valid parent query indices"
        )
    reconstructed_scores = scatter_candidate_scores(
        compact_scores,
        query_indices,
        candidate_valid,
        num_queries,
        fill_value=-float("inf"),
    )
    if not torch.equal(query_scores, reconstructed_scores):
        raise ValueError(
            "parent query_scores do not match compact score reconstruction"
        )
    return query_scores, query_order, query_indices, candidate_valid


def build_flat_parent_prior(parent_state, geometry_valid,
                            regressed_variant_index):
    """Expand deployed Q ranks into a unique flat geometry prior."""
    query_scores, query_order, query_indices, candidate_valid = (
        _validate_parent_prior_inputs(
            parent_state, geometry_valid, regressed_variant_index
        )
    )
    query_ranks = torch.empty_like(query_order)
    rank_values = torch.arange(
        query_scores.shape[1], dtype=torch.long, device=query_scores.device
    ).unsqueeze(0).expand_as(query_order)
    query_ranks.scatter_(1, query_order, rank_values)
    safe_query_indices = query_indices.masked_fill(~candidate_valid, 0)
    compact_ranks = torch.gather(query_ranks, 1, safe_query_indices)

    variants = geometry_valid.shape[2]
    variant_priority = [regressed_variant_index] + [
        index for index in range(variants)
        if index != regressed_variant_index
    ]
    priority_by_index = torch.empty(
        variants, dtype=torch.long, device=query_scores.device
    )
    priority_by_index[torch.tensor(
        variant_priority, dtype=torch.long, device=query_scores.device
    )] = torch.arange(
        variants, dtype=torch.long, device=query_scores.device
    )
    order_code = (
        compact_ranks.unsqueeze(2) * variants
        + priority_by_index.view(1, 1, variants)
    )
    prior = -order_code.reshape(order_code.shape[0], -1).to(
        dtype=torch.float32
    )
    return prior.masked_fill(
        ~geometry_valid.reshape(geometry_valid.shape[0], -1),
        -float("inf"),
    )


def stable_flat_descending_indices(scores, valid):
    """Return valid flat indices ordered by ``(-score, flat_index)``."""
    _require_float_tensor(scores, "scores")
    _require_bool_tensor(valid, "valid")
    if scores.dim() != 2 or not all(scores.shape):
        raise ValueError("scores must have nonempty shape [B,C]")
    if valid.shape != scores.shape:
        raise ValueError("valid must match scores")
    if valid.device != scores.device:
        raise ValueError("scores and valid must share a device")
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("every row needs at least one valid flat candidate")
    if not _all_finite(scores[valid]):
        raise ValueError("valid flat candidate scores must be finite")

    cpu_scores = scores.detach().cpu()
    cpu_valid = valid.detach().cpu()
    result = []
    for batch_idx in range(scores.shape[0]):
        values = cpu_scores[batch_idx].tolist()
        indices = cpu_valid[batch_idx].nonzero(
            as_tuple=False
        ).reshape(-1).tolist()
        result.append(tuple(sorted(
            indices, key=lambda index: (-values[index], index)
        )))
    return tuple(result)


def _stable_masked_rank_normalize(scores, valid):
    orders = stable_flat_descending_indices(scores, valid)
    normalized = torch.full(
        scores.shape,
        -float("inf"),
        dtype=torch.float32,
        device=scores.device,
    )
    for batch_idx, order in enumerate(orders):
        indices = torch.tensor(
            order, dtype=torch.long, device=scores.device
        )
        denominator = float(max(len(order) - 1, 1))
        ranks = torch.arange(
            len(order), dtype=torch.float32, device=scores.device
        )
        normalized[batch_idx, indices] = 1.0 - ranks / denominator
    return normalized


def blend_rec_geometry_scores(parent_state, learned_logits,
                              geometry_valid, geometry_weight,
                              regressed_variant_index):
    """Blend parent and learned flat ranks, or bypass exactly at zero."""
    try:
        weight = float(geometry_weight)
    except (TypeError, ValueError, OverflowError):
        raise TypeError("geometry_weight must be numeric")
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("geometry_weight must lie in [0, 1]")
    if weight == 0.0:
        if (not isinstance(parent_state, dict)
                or "query_scores" not in parent_state):
            raise ValueError("parent_state must contain query_scores")
        return {
            "use_parent_query_axis": True,
            "query_scores": parent_state["query_scores"],
        }

    prior = build_flat_parent_prior(
        parent_state, geometry_valid, regressed_variant_index
    )
    _require_float_tensor(learned_logits, "learned_logits")
    if learned_logits.dim() != 2 or learned_logits.shape != prior.shape:
        raise ValueError("learned_logits must have shape [B,K*G]")
    if learned_logits.device != prior.device:
        raise ValueError("learned_logits and parent prior must share a device")
    flat_valid = geometry_valid.reshape(geometry_valid.shape[0], -1)
    if not _all_finite(learned_logits[flat_valid]):
        raise ValueError("valid learned logits must be finite")

    parent_rank = _stable_masked_rank_normalize(prior, flat_valid)
    learned_rank = _stable_masked_rank_normalize(
        learned_logits, flat_valid
    )
    flat_scores = (
        (1.0 - weight) * parent_rank + weight * learned_rank
    ).masked_fill(~flat_valid, -float("inf"))
    if not _all_finite(flat_scores[flat_valid]):
        raise RuntimeError("valid blended geometry scores must be finite")
    return {
        "use_parent_query_axis": False,
        "flat_scores": flat_scores,
        "flat_valid_mask": flat_valid,
    }
