"""Mask-derived geometry helpers for REC candidates.

Candidate construction consumes only inference-time MCLN outputs and input
point geometry. Training targets are attached separately and explicitly.
"""

import math

import torch

from .mask_fusion import gather_query_fusion_weight
from .rec_reranker import compute_query_ious


MASK_GEOMETRY_SCHEMA_VERSION = "rec-mask-geometry-v1"


DEFAULT_REC_MASK_GEOMETRY_VARIANTS = (
    {
        "name": "regressed",
        "source": "regressed",
        "logit_threshold": 0.0,
        "quantile": 0.0,
        "regressed_weight": 1.0,
    },
    {
        "name": "fused_t0_exact",
        "source": "fused",
        "logit_threshold": 0.0,
        "quantile": 0.0,
        "regressed_weight": 0.0,
    },
    {
        "name": "fused_t0_q0.005",
        "source": "fused",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.0,
    },
    {
        "name": "query_t0_q0.005",
        "source": "query",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.0,
    },
    {
        "name": "fused_t0.5_q0.005",
        "source": "fused",
        "logit_threshold": 0.5,
        "quantile": 0.005,
        "regressed_weight": 0.0,
    },
    {
        "name": "blend_regressed_fused_q0.005",
        "source": "fused",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.5,
    },
    {
        "name": "blend_regressed_query_q0.005",
        "source": "query",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.5,
    },
)


REC_MASK_GEOMETRY_FEATURE_NAMES = (
    "valid",
    "source_is_regressed",
    "source_is_text",
    "source_is_query",
    "source_is_fused",
    "logit_threshold",
    "quantile",
    "regressed_weight",
    "adaptive_alpha",
    "selected_point_count",
    "selected_point_fraction",
    "selected_superpoint_count",
    "selected_superpoint_fraction",
    "foreground_logit_mean",
    "foreground_logit_std",
    "foreground_logit_min",
    "foreground_logit_max",
    "source_mask_to_regressed_volume_ratio",
    "center_delta_x_scene_norm",
    "center_delta_y_scene_norm",
    "center_delta_z_scene_norm",
    "size_delta_x_scene_norm",
    "size_delta_y_scene_norm",
    "size_delta_z_scene_norm",
    "source_mask_vs_regressed_iou",
)


def attach_rec_mask_geometry_targets(geometry_batch, end_points,
                                     root_only=True):
    """Return a copy of a geometry batch with training-only IoU targets."""
    boxes = geometry_batch["boxes"]
    valid = geometry_batch["valid_mask"].bool()
    if boxes.dim() != 4 or boxes.shape[-1] != 6:
        raise ValueError("geometry boxes must have shape [B,K,G,6]")
    if valid.shape != boxes.shape[:3]:
        raise ValueError("geometry valid_mask must match boxes")

    gt_boxes = torch.cat([
        end_points["center_label"][..., :3].float(),
        end_points["size_gts"].float(),
    ], dim=-1)
    gt_mask = end_points["box_label_mask"]
    if root_only:
        gt_boxes = gt_boxes[:, :1]
        gt_mask = gt_mask[:, :1]
    ious = compute_query_ious(
        boxes.reshape(boxes.shape[0], -1, 6), gt_boxes, gt_mask
    ).reshape(valid.shape)
    ious = ious.masked_fill(~valid, 0.0)

    result = dict(geometry_batch)
    result["geometry_ious"] = ious
    result["threshold_labels"] = torch.stack([
        ious > 0.25,
        ious > 0.50,
    ], dim=-1)
    return result


def project_variant_rejection_codes(geometry_batch):
    """Project per-group mask diagnostics onto the geometry variant axis."""
    boxes = geometry_batch["boxes"]
    valid = geometry_batch["valid_mask"]
    if not isinstance(boxes, torch.Tensor) or (
            boxes.dim() != 4 or boxes.shape[-1] != 6):
        raise ValueError("geometry boxes must have shape [B,K,G,6]")
    if not isinstance(valid, torch.Tensor) or valid.shape != boxes.shape[:3]:
        raise ValueError("geometry valid_mask must match boxes [B,K,G]")
    batch_size, num_candidates, num_variants = valid.shape

    variant_configs = geometry_batch["variant_configs"]
    if not isinstance(variant_configs, (list, tuple)) or (
            len(variant_configs) != num_variants):
        raise ValueError(
            "variant configs must match the geometry variant axis"
        )
    regressed_indices = [
        idx for idx, config in enumerate(variant_configs)
        if isinstance(config, dict) and config.get("source") == "regressed"
    ]
    if len(regressed_indices) != 1:
        raise ValueError("geometry needs exactly one regressed variant")

    mask_diagnostics = geometry_batch["mask_diagnostics"]
    if not isinstance(mask_diagnostics, (list, tuple)) or (
            len(mask_diagnostics) != batch_size):
        raise ValueError("mask diagnostics must match the geometry batch axis")
    result = torch.zeros(
        batch_size, num_candidates, num_variants,
        dtype=torch.int16, device=boxes.device
    )
    integer_dtypes = (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    )
    int16_max = torch.iinfo(torch.int16).max

    for batch_idx, sample_diagnostics in enumerate(mask_diagnostics):
        if not isinstance(sample_diagnostics, dict):
            raise ValueError(
                "each sample's mask diagnostics must be a mapping"
            )
        for variant_idx, config in enumerate(variant_configs):
            if not isinstance(config, dict):
                raise ValueError("each variant config must be a mapping")
            source = config.get("source")
            if source == "regressed":
                continue
            try:
                threshold = float(config["logit_threshold"])
                target_quantile = float(config["quantile"])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ValueError(
                    "variant source, logit_threshold, and quantile are "
                    "required"
                )
            if not math.isfinite(threshold) or not math.isfinite(
                    target_quantile):
                raise ValueError(
                    "variant thresholds and quantiles must be finite"
                )
            group_name = "{}_t{:g}".format(source, threshold)
            diagnostics = sample_diagnostics.get(group_name)
            if diagnostics is None:
                raise ValueError(
                    "mask diagnostics are missing {}".format(group_name)
                )
            if not isinstance(diagnostics, dict):
                raise ValueError("mask diagnostic groups must be mappings")
            if "quantiles" not in diagnostics:
                raise ValueError("mask diagnostics are missing quantiles")
            if "rejection_codes" not in diagnostics:
                raise ValueError(
                    "mask diagnostics are missing rejection_codes"
                )

            quantiles = torch.as_tensor(diagnostics["quantiles"])
            if quantiles.dim() != 1:
                raise ValueError(
                    "mask diagnostic quantiles must have shape [Q]"
                )
            quantiles = quantiles.double()
            if not bool(torch.isfinite(quantiles).all().item()):
                raise ValueError("mask diagnostic quantiles must be finite")
            target = quantiles.new_tensor(target_quantile)
            matches = torch.isclose(
                quantiles, target, atol=1e-8, rtol=0.0
            ).nonzero(as_tuple=False).reshape(-1)
            if matches.numel() != 1:
                raise ValueError(
                    "mask diagnostics need exactly one matching quantile"
                )

            rejection_codes = torch.as_tensor(
                diagnostics["rejection_codes"]
            )
            if rejection_codes.dim() != 2:
                raise ValueError(
                    "mask diagnostic rejection_codes must have shape [K,Q]"
                )
            if rejection_codes.shape[0] != num_candidates:
                raise ValueError(
                    "mask diagnostic rejection_codes candidate axis mismatch"
                )
            if rejection_codes.shape[1] != quantiles.numel():
                raise ValueError(
                    "mask diagnostic rejection_codes quantile axis mismatch"
                )
            if rejection_codes.dtype not in integer_dtypes:
                raise ValueError(
                    "mask diagnostic rejection_codes must be integers"
                )
            if bool((rejection_codes < 0).any().item()) or bool(
                    (rejection_codes > int16_max).any().item()):
                raise ValueError(
                    "mask diagnostic rejection_codes must fit "
                    "nonnegative int16"
                )
            quantile_idx = int(matches[0].item())
            result[batch_idx, :, variant_idx] = rejection_codes[
                :, quantile_idx
            ].to(device=boxes.device, dtype=torch.int16)
    return result


def _mask_batch_item(value, batch_idx, name):
    if isinstance(value, (list, tuple)):
        if batch_idx < 0 or batch_idx >= len(value):
            raise IndexError("{} batch index is out of range".format(name))
        item = value[batch_idx]
    elif isinstance(value, torch.Tensor):
        if value.dim() == 2:
            if batch_idx != 0:
                raise IndexError("{} has no batch axis".format(name))
            item = value
        elif value.dim() >= 3:
            if batch_idx < 0 or batch_idx >= value.shape[0]:
                raise IndexError("{} batch index is out of range".format(name))
            item = value[batch_idx]
        else:
            raise ValueError("{} must contain [Q,S] mask logits".format(name))
    else:
        raise TypeError("{} must be a tensor or tensor sequence".format(name))

    if not isinstance(item, torch.Tensor):
        item = torch.as_tensor(item)
    while item.dim() > 2 and item.shape[0] == 1:
        item = item.squeeze(0)
    if item.dim() != 2:
        raise ValueError(
            "{} entries must have shape [Q,S] or singleton-prefixed [Q,S]"
            .format(name)
        )
    return item


def _adaptive_alpha(end_points, batch_idx, query_indices, query_count,
                    reference):
    if "adaptive_weights" not in end_points:
        raise KeyError("adaptive_weights is required to fuse mask logits")
    value = end_points["adaptive_weights"]
    if isinstance(value, (list, tuple)):
        if batch_idx < 0 or batch_idx >= len(value):
            raise IndexError("adaptive_weights batch index is out of range")
        value = value[batch_idx]
    elif isinstance(value, torch.Tensor):
        if value.dim() == 0:
            if batch_idx != 0:
                raise IndexError("adaptive_weights has no batch axis")
        else:
            if batch_idx < 0 or batch_idx >= value.shape[0]:
                raise IndexError("adaptive_weights batch index is out of range")
            value = value[batch_idx]
    return gather_query_fusion_weight(
        value, query_indices, query_count, reference
    )


def normalize_mcln_mask_logits(end_points, batch_idx, query_indices):
    """Gather one sample's text/query masks on original detector indices.

    Returns ``(text_logits, query_logits, fused_logits, alpha)``. All three
    logit tensors have shape ``[K,S]`` and fusion happens before sigmoid.
    """
    if "last_pred_masks" not in end_points:
        raise KeyError("last_pred_masks is required")
    if "sp_last_pred_masks" not in end_points:
        raise KeyError("sp_last_pred_masks is required")
    text_logits = _mask_batch_item(
        end_points["last_pred_masks"], batch_idx, "last_pred_masks"
    )
    query_logits = _mask_batch_item(
        end_points["sp_last_pred_masks"], batch_idx, "sp_last_pred_masks"
    )
    if not torch.is_floating_point(text_logits):
        text_logits = text_logits.float()
    if not torch.is_floating_point(query_logits):
        query_logits = query_logits.float()
    query_logits = query_logits.to(
        device=text_logits.device, dtype=text_logits.dtype
    )
    if text_logits.shape != query_logits.shape:
        raise ValueError("text and query mask logits must have the same shape")

    indices = torch.as_tensor(
        query_indices, dtype=torch.long, device=text_logits.device
    ).reshape(-1)
    if indices.numel() and bool(
            ((indices < 0) | (indices >= text_logits.shape[0])).any().item()):
        raise IndexError("query index is out of mask-logit range")
    query_count = text_logits.shape[0]
    text_logits = text_logits.index_select(0, indices)
    query_logits = query_logits.index_select(0, indices)
    alpha = _adaptive_alpha(
        end_points, batch_idx, indices, query_count, text_logits
    )
    fused_logits = alpha * text_logits + (1.0 - alpha) * query_logits
    return text_logits, query_logits, fused_logits, alpha


def _validate_superpoint_ids(superpoint_ids, num_superpoints):
    if torch.is_floating_point(superpoint_ids):
        if not bool(torch.isfinite(superpoint_ids).all().item()):
            raise ValueError("superpoint_ids must be finite integers")
        if not torch.equal(superpoint_ids, superpoint_ids.round()):
            raise ValueError("superpoint_ids must contain integer values")
    ids = superpoint_ids.long()
    if ids.numel() and bool((ids < 0).any().item()):
        raise ValueError("superpoint_ids must be nonnegative")
    if ids.numel() and bool((ids >= num_superpoints).any().item()):
        raise ValueError(
            "raw superpoint ID is outside the mask-logit column range"
        )
    return ids


def mask_logits_to_point_aabbs(
        coords, superpoint_ids, mask_logits, logit_threshold,
        quantiles=(0.0, 0.005), min_points=5, max_point_fraction=0.5):
    """Convert superpoint logits into audited point-level AABBs.

    Raw superpoint IDs index mask columns directly; they are never compacted.
    Output boxes use ``[center_x, center_y, center_z, size_x, size_y, size_z]``.
    Invalid boxes are zero-filled and marked false in the returned validity
    tensor. Diagnostics retain occupancy, finite-state, bounds, and a bitmask
    rejection code for every query/quantile pair.
    """
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords)
    if not isinstance(superpoint_ids, torch.Tensor):
        superpoint_ids = torch.as_tensor(superpoint_ids)
    if not isinstance(mask_logits, torch.Tensor):
        mask_logits = torch.as_tensor(mask_logits)
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N,3]")
    if superpoint_ids.dim() != 1 or superpoint_ids.shape[0] != coords.shape[0]:
        raise ValueError("superpoint_ids must have shape [N]")
    if mask_logits.dim() != 2:
        raise ValueError("mask_logits must have shape [K,S]")
    if coords.shape[0] == 0:
        raise ValueError("coords must contain at least one point")
    if mask_logits.shape[1] == 0:
        raise ValueError("mask_logits must contain at least one superpoint")
    if isinstance(min_points, bool) or int(min_points) != min_points:
        raise ValueError("min_points must be a positive integer")
    min_points = int(min_points)
    if min_points <= 0:
        raise ValueError("min_points must be a positive integer")
    max_point_fraction = float(max_point_fraction)
    if not math.isfinite(max_point_fraction) or not (
            0.0 < max_point_fraction <= 1.0):
        raise ValueError("max_point_fraction must be in (0,1]")
    logit_threshold = float(logit_threshold)
    if not math.isfinite(logit_threshold):
        raise ValueError("logit_threshold must be finite")

    quantiles = tuple(float(value) for value in quantiles)
    if not quantiles:
        raise ValueError("quantiles must not be empty")
    if any(not math.isfinite(value) or value < 0.0 or value >= 0.5
           for value in quantiles):
        raise ValueError("quantiles must be finite values in [0,0.5)")

    device = coords.device
    coords = coords.float()
    superpoint_ids = _validate_superpoint_ids(
        superpoint_ids.to(device=device), mask_logits.shape[1]
    )
    mask_logits = mask_logits.to(device=device)
    if not torch.is_floating_point(mask_logits):
        mask_logits = mask_logits.float()
    else:
        mask_logits = mask_logits.float()

    num_queries = mask_logits.shape[0]
    num_points = coords.shape[0]
    num_variants = len(quantiles)
    point_logits = mask_logits.index_select(1, superpoint_ids)
    finite_logits = torch.isfinite(point_logits).all(dim=1)
    selected = point_logits > logit_threshold
    point_counts = selected.sum(dim=1)
    point_fractions = point_counts.to(coords.dtype) / float(num_points)

    present_ids = torch.unique(superpoint_ids)
    present_logits = mask_logits.index_select(1, present_ids)
    superpoint_counts = (present_logits > logit_threshold).sum(dim=1)
    superpoint_fractions = (
        superpoint_counts.to(coords.dtype) / float(present_ids.numel())
    )

    finite_coord_rows = torch.isfinite(coords).all(dim=1)
    finite_coordinates = (
        (~selected) | finite_coord_rows.unsqueeze(0)
    ).all(dim=1)
    nonempty = point_counts > 0
    enough_points = point_counts >= min_points
    full = point_counts == num_points
    over_max_fraction = point_fractions > max_point_fraction
    within_point_fraction = ~over_max_fraction
    base_valid = (
        nonempty
        & enough_points
        & ~full
        & within_point_fraction
        & finite_logits
        & finite_coordinates
    )

    foreground_mean = coords.new_zeros(num_queries)
    foreground_std = coords.new_zeros(num_queries)
    foreground_min = coords.new_zeros(num_queries)
    foreground_max = coords.new_zeros(num_queries)
    for query_idx in range(num_queries):
        values = point_logits[query_idx][
            selected[query_idx] & torch.isfinite(point_logits[query_idx])
        ]
        if values.numel():
            foreground_mean[query_idx] = values.mean()
            foreground_std[query_idx] = values.std(unbiased=False)
            foreground_min[query_idx] = values.min()
            foreground_max[query_idx] = values.max()

    # Compute every query in one set of reductions. The previous per-query
    # quantile loop launched many small kernels and synchronized the GPU for
    # every validity check, which dominated full-cache extraction.
    expanded_coords = coords.unsqueeze(0).expand(num_queries, -1, -1)
    selected_coords = selected.unsqueeze(-1)
    can_compute = nonempty & finite_coordinates
    lower_bounds = coords.new_zeros(num_queries, num_variants, 3)
    upper_bounds = coords.new_zeros(num_queries, num_variants, 3)

    exact_lower = torch.where(
        selected_coords,
        expanded_coords,
        torch.full_like(expanded_coords, float("inf")),
    ).amin(dim=1)
    exact_upper = torch.where(
        selected_coords,
        expanded_coords,
        torch.full_like(expanded_coords, float("-inf")),
    ).amax(dim=1)

    nonzero_quantiles = []
    nonzero_indices = []
    for quantile_idx, quantile in enumerate(quantiles):
        if quantile == 0.0:
            lower_bounds[:, quantile_idx] = exact_lower
            upper_bounds[:, quantile_idx] = exact_upper
        else:
            nonzero_quantiles.extend((quantile, 1.0 - quantile))
            nonzero_indices.append(quantile_idx)

    if nonzero_indices:
        masked_coords = expanded_coords.masked_fill(
            ~selected_coords, float("nan")
        )
        bounds = torch.nanquantile(
            masked_coords,
            coords.new_tensor(nonzero_quantiles),
            dim=1,
        )
        for offset, quantile_idx in enumerate(nonzero_indices):
            lower_bounds[:, quantile_idx] = bounds[2 * offset]
            upper_bounds[:, quantile_idx] = bounds[2 * offset + 1]

    box_computed = can_compute.unsqueeze(1).expand(
        num_queries, num_variants
    )
    lower_bounds = torch.where(
        box_computed.unsqueeze(-1),
        lower_bounds,
        torch.zeros_like(lower_bounds),
    )
    upper_bounds = torch.where(
        box_computed.unsqueeze(-1),
        upper_bounds,
        torch.zeros_like(upper_bounds),
    )
    sizes = upper_bounds - lower_bounds
    candidate_boxes = torch.cat(
        [(lower_bounds + upper_bounds) * 0.5, sizes], dim=-1
    )
    finite_boxes = (
        box_computed & torch.isfinite(candidate_boxes).all(dim=-1)
    )
    nondegenerate = box_computed & (sizes > 0.0).all(dim=-1)
    valid = (
        base_valid.unsqueeze(1)
        & finite_boxes
        & nondegenerate
    )
    boxes = torch.where(
        valid.unsqueeze(-1), candidate_boxes,
        torch.zeros_like(candidate_boxes)
    )

    # Bit assignments: empty, too few, full, over fraction, nonfinite logits,
    # nonfinite coordinates, nonfinite box, and degenerate box.
    rejection_codes = torch.zeros(
        num_queries, num_variants, dtype=torch.long, device=device
    )
    rejection_codes += (~nonempty).long().unsqueeze(1) * 1
    rejection_codes += (~enough_points).long().unsqueeze(1) * 2
    rejection_codes += full.long().unsqueeze(1) * 4
    rejection_codes += over_max_fraction.long().unsqueeze(1) * 8
    rejection_codes += (~finite_logits).long().unsqueeze(1) * 16
    rejection_codes += (~finite_coordinates).long().unsqueeze(1) * 32
    rejection_codes += (box_computed & ~finite_boxes).long() * 64
    rejection_codes += (box_computed & ~nondegenerate).long() * 128

    diagnostics = {
        "selected_point_counts": point_counts,
        "selected_point_fractions": point_fractions,
        "selected_superpoint_counts": superpoint_counts,
        "selected_superpoint_fractions": superpoint_fractions,
        "foreground_logit_mean": foreground_mean,
        "foreground_logit_std": foreground_std,
        "foreground_logit_min": foreground_min,
        "foreground_logit_max": foreground_max,
        "finite_logits": finite_logits,
        "finite_coordinates": finite_coordinates,
        "nonempty": nonempty,
        "enough_points": enough_points,
        "full": full,
        "over_max_point_fraction": over_max_fraction,
        "within_point_fraction": within_point_fraction,
        "base_valid": base_valid,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "finite_boxes": finite_boxes,
        "box_computed": box_computed,
        "nondegenerate": nondegenerate,
        "valid": valid,
        "rejection_codes": rejection_codes,
        "quantiles": coords.new_tensor(quantiles),
        "logit_threshold": coords.new_tensor(logit_threshold),
        "total_point_count": torch.tensor(
            num_points, dtype=torch.long, device=device
        ),
        "present_superpoint_count": torch.tensor(
            present_ids.numel(), dtype=torch.long, device=device
        ),
    }
    return boxes, valid, diagnostics


def _point_batch_item(value, batch_idx, unbatched_dims, name):
    if isinstance(value, (list, tuple)):
        if batch_idx < 0 or batch_idx >= len(value):
            raise IndexError("{} batch index is out of range".format(name))
        item = value[batch_idx]
    elif isinstance(value, torch.Tensor):
        if value.dim() == unbatched_dims:
            if batch_idx != 0:
                raise IndexError("{} has no batch axis".format(name))
            item = value
        elif value.dim() > unbatched_dims:
            if batch_idx < 0 or batch_idx >= value.shape[0]:
                raise IndexError("{} batch index is out of range".format(name))
            item = value[batch_idx]
        else:
            raise ValueError("{} has too few dimensions".format(name))
    else:
        item = torch.as_tensor(value)
        return _point_batch_item(item, batch_idx, unbatched_dims, name)
    while item.dim() > unbatched_dims and item.shape[0] == 1:
        item = item.squeeze(0)
    if item.dim() != unbatched_dims:
        raise ValueError("{} has an unsupported shape".format(name))
    return item


def _find_superpoint_map(end_points, inputs):
    for container, key in (
            (inputs, "superpoint"),
            (inputs, "superpoints"),
            (end_points, "superpoints"),
            (end_points, "superpoint")):
        if key in container:
            return container[key], key
    raise KeyError(
        "raw superpoint IDs are required as superpoint or superpoints"
    )


def _normalized_variant_config(variant_config):
    min_points = 5
    max_point_fraction = 0.5
    variants = DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    if variant_config is not None:
        if isinstance(variant_config, dict):
            min_points = variant_config.get("min_points", min_points)
            max_point_fraction = variant_config.get(
                "max_point_fraction", max_point_fraction
            )
            variants = variant_config.get("variants", variants)
        elif isinstance(variant_config, (list, tuple)):
            variants = variant_config
        else:
            raise TypeError("variant_config must be a mapping or sequence")
    try:
        canonical_min_points = int(min_points)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("min_points must be a positive integer")
    if (isinstance(min_points, bool)
            or canonical_min_points != min_points
            or canonical_min_points <= 0):
        raise ValueError("min_points must be a positive integer")
    min_points = canonical_min_points
    try:
        max_point_fraction = float(max_point_fraction)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_point_fraction must be in (0,1]")
    if not math.isfinite(max_point_fraction) or not (
            0.0 < max_point_fraction <= 1.0):
        raise ValueError("max_point_fraction must be in (0,1]")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError("variant_config must contain at least one variant")

    defaults_by_name = {
        value["name"]: value for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    }
    normalized = []
    names = set()
    for raw_variant in variants:
        if isinstance(raw_variant, str):
            if raw_variant not in defaults_by_name:
                raise KeyError("unknown geometry variant: {}".format(raw_variant))
            raw_variant = defaults_by_name[raw_variant]
        if not isinstance(raw_variant, dict):
            raise TypeError("each geometry variant must be a mapping")
        source = raw_variant.get("source")
        if source not in ("regressed", "text", "query", "fused"):
            raise ValueError("unsupported mask geometry source: {}".format(source))
        threshold = float(raw_variant.get(
            "logit_threshold", raw_variant.get("threshold", 0.0)
        ))
        quantile = float(raw_variant.get("quantile", 0.0))
        default_weight = 1.0 if source == "regressed" else 0.0
        regressed_weight = float(raw_variant.get(
            "regressed_weight", default_weight
        ))
        if not math.isfinite(threshold):
            raise ValueError("variant logit thresholds must be finite")
        if not math.isfinite(quantile) or quantile < 0.0 or quantile >= 0.5:
            raise ValueError("variant quantiles must be in [0,0.5)")
        if not math.isfinite(regressed_weight) or not (
                0.0 <= regressed_weight <= 1.0):
            raise ValueError("regressed_weight must be in [0,1]")
        name = raw_variant.get("name")
        if not isinstance(name, str) or not name:
            name = "{}_t{:g}_q{:g}_r{:g}".format(
                source, threshold, quantile, regressed_weight
            )
        if name in names:
            raise ValueError("geometry variant names must be unique")
        names.add(name)
        normalized.append({
            "name": name,
            "source": source,
            "logit_threshold": threshold,
            "quantile": quantile,
            "regressed_weight": regressed_weight,
        })
    return tuple(normalized), min_points, max_point_fraction


def _aligned_aabb_iou(first, second):
    first_size = first[..., 3:].clamp(min=0.0)
    second_size = second[..., 3:].clamp(min=0.0)
    first_min = first[..., :3] - first_size * 0.5
    first_max = first[..., :3] + first_size * 0.5
    second_min = second[..., :3] - second_size * 0.5
    second_max = second[..., :3] + second_size * 0.5
    intersection_size = (
        torch.minimum(first_max, second_max)
        - torch.maximum(first_min, second_min)
    ).clamp(min=0.0)
    intersection = intersection_size.prod(dim=-1)
    first_volume = first_size.prod(dim=-1)
    second_volume = second_size.prod(dim=-1)
    union = first_volume + second_volume - intersection
    iou = torch.where(union > 0.0, intersection / union.clamp(min=1e-12),
                      torch.zeros_like(union))
    return torch.where(torch.isfinite(iou), iou, torch.zeros_like(iou))


def _scene_extent(coords):
    finite_rows = torch.isfinite(coords).all(dim=1)
    if bool(finite_rows.any().item()):
        finite_coords = coords[finite_rows]
        extent = finite_coords.max(dim=0).values - finite_coords.min(dim=0).values
        return extent.clamp(min=1e-6)
    return coords.new_ones(3)


def build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch, variant_config=None):
    """Build deployable mask/regressed geometry variants for REC candidates.

    The returned dictionary contains ``boxes [B,K,G,6]``, ``valid_mask
    [B,K,G]``, finite named geometry features, and stable variant names. The
    regressed candidate is retained whenever its input candidate is valid;
    every mask or blended variant requires its source mask box to be valid.
    """
    if "boxes" not in candidate_batch:
        raise KeyError("candidate_batch boxes are required")
    if "valid_mask" not in candidate_batch:
        raise KeyError("candidate_batch valid_mask is required")
    if "query_indices" not in candidate_batch:
        raise KeyError("candidate_batch query_indices are required")
    if "point_clouds" not in inputs:
        raise KeyError("inputs point_clouds are required")

    regressed_boxes = candidate_batch["boxes"]
    if not isinstance(regressed_boxes, torch.Tensor):
        regressed_boxes = torch.as_tensor(regressed_boxes)
    regressed_boxes = regressed_boxes.float()
    if regressed_boxes.dim() != 3 or regressed_boxes.shape[-1] != 6:
        raise ValueError("candidate boxes must have shape [B,K,6]")
    device = regressed_boxes.device
    batch_size, num_candidates, _ = regressed_boxes.shape
    candidate_valid = torch.as_tensor(
        candidate_batch["valid_mask"], device=device
    ).bool()
    query_indices = torch.as_tensor(
        candidate_batch["query_indices"], device=device, dtype=torch.long
    )
    if candidate_valid.shape != (batch_size, num_candidates):
        raise ValueError("candidate valid_mask must have shape [B,K]")
    if query_indices.shape != (batch_size, num_candidates):
        raise ValueError("candidate query_indices must have shape [B,K]")

    variants, min_points, max_point_fraction = _normalized_variant_config(
        variant_config
    )
    variant_names = tuple(value["name"] for value in variants)
    num_variants = len(variants)
    feature_names = REC_MASK_GEOMETRY_FEATURE_NAMES
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    boxes = regressed_boxes.new_zeros(
        batch_size, num_candidates, num_variants, 6
    )
    valid_mask = torch.zeros(
        batch_size, num_candidates, num_variants,
        dtype=torch.bool, device=device
    )
    geometry_features = regressed_boxes.new_zeros(
        batch_size, num_candidates, num_variants, len(feature_names)
    )
    alpha_values = regressed_boxes.new_zeros(batch_size)
    superpoint_map, superpoint_key = _find_superpoint_map(end_points, inputs)
    batch_diagnostics = []

    source_flag_names = {
        "regressed": "source_is_regressed",
        "text": "source_is_text",
        "query": "source_is_query",
        "fused": "source_is_fused",
    }
    for batch_idx in range(batch_size):
        coords = _point_batch_item(
            inputs["point_clouds"], batch_idx, 2, "point_clouds"
        )[..., :3].to(device=device).float()
        raw_ids = _point_batch_item(
            superpoint_map, batch_idx, 1, superpoint_key
        ).to(device=device)
        if raw_ids.shape[0] != coords.shape[0]:
            raise ValueError("point clouds and superpoint IDs must align")
        text_logits, query_logits, fused_logits, alpha = (
            normalize_mcln_mask_logits(
                end_points, batch_idx, query_indices[batch_idx]
            )
        )
        text_logits = text_logits.to(device=device)
        query_logits = query_logits.to(device=device)
        fused_logits = fused_logits.to(device=device)
        alpha = alpha.to(device=device)
        alpha_values[batch_idx] = alpha
        source_logits = {
            "text": text_logits,
            "query": query_logits,
            "fused": fused_logits,
        }

        grouped_quantiles = {}
        for variant in variants:
            if variant["source"] == "regressed":
                continue
            key = (variant["source"], variant["logit_threshold"])
            grouped_quantiles.setdefault(key, [])
            if variant["quantile"] not in grouped_quantiles[key]:
                grouped_quantiles[key].append(variant["quantile"])

        group_results = {}
        sample_diagnostics = {}
        for key, group_quantiles in grouped_quantiles.items():
            source, threshold = key
            group_boxes, group_valid, diagnostics = (
                mask_logits_to_point_aabbs(
                    coords,
                    raw_ids,
                    source_logits[source],
                    logit_threshold=threshold,
                    quantiles=tuple(group_quantiles),
                    min_points=min_points,
                    max_point_fraction=max_point_fraction,
                )
            )
            group_results[key] = (
                group_boxes,
                group_valid,
                diagnostics,
                {value: idx for idx, value in enumerate(group_quantiles)},
            )
            sample_diagnostics[
                "{}_t{:g}".format(source, threshold)
            ] = diagnostics
        batch_diagnostics.append(sample_diagnostics)

        extent = _scene_extent(coords)
        regressed = regressed_boxes[batch_idx]
        regressed_volume = regressed[..., 3:].clamp(min=0.0).prod(dim=-1)
        for variant_idx, variant in enumerate(variants):
            source = variant["source"]
            features = geometry_features[batch_idx, :, variant_idx]
            features[:, feature_index[source_flag_names[source]]] = 1.0
            features[:, feature_index["logit_threshold"]] = (
                variant["logit_threshold"]
            )
            features[:, feature_index["quantile"]] = variant["quantile"]
            features[:, feature_index["regressed_weight"]] = (
                variant["regressed_weight"]
            )
            features[:, feature_index["adaptive_alpha"]] = alpha

            if source == "regressed":
                boxes[batch_idx, :, variant_idx] = regressed
                variant_valid = candidate_valid[batch_idx]
                valid_mask[batch_idx, :, variant_idx] = variant_valid
                features[:, feature_index["valid"]] = variant_valid.float()
                features[:, feature_index[
                    "source_mask_to_regressed_volume_ratio"
                ]] = 1.0
                features[:, feature_index[
                    "source_mask_vs_regressed_iou"
                ]] = 1.0
                continue

            key = (source, variant["logit_threshold"])
            group_boxes, group_valid, diagnostics, quantile_indices = (
                group_results[key]
            )
            quantile_idx = quantile_indices[variant["quantile"]]
            source_boxes = group_boxes[:, quantile_idx]
            source_valid = group_valid[:, quantile_idx]
            regressed_weight = variant["regressed_weight"]
            final_boxes = (
                regressed_weight * regressed
                + (1.0 - regressed_weight) * source_boxes
            )
            final_finite = torch.isfinite(final_boxes).all(dim=-1)
            final_nondegenerate = (final_boxes[..., 3:] > 0.0).all(dim=-1)
            variant_valid = (
                candidate_valid[batch_idx]
                & source_valid
                & final_finite
                & final_nondegenerate
            )
            valid_mask[batch_idx, :, variant_idx] = variant_valid
            boxes[batch_idx, :, variant_idx] = torch.where(
                variant_valid.unsqueeze(-1),
                final_boxes,
                torch.zeros_like(final_boxes),
            )
            features[:, feature_index["valid"]] = variant_valid.float()
            for diagnostic_name, feature_name in (
                    ("selected_point_counts", "selected_point_count"),
                    ("selected_point_fractions", "selected_point_fraction"),
                    ("selected_superpoint_counts", "selected_superpoint_count"),
                    ("selected_superpoint_fractions", "selected_superpoint_fraction"),
                    ("foreground_logit_mean", "foreground_logit_mean"),
                    ("foreground_logit_std", "foreground_logit_std"),
                    ("foreground_logit_min", "foreground_logit_min"),
                    ("foreground_logit_max", "foreground_logit_max")):
                features[:, feature_index[feature_name]] = diagnostics[
                    diagnostic_name
                ].to(device=device, dtype=features.dtype)

            source_volume = source_boxes[..., 3:].clamp(min=0.0).prod(dim=-1)
            volume_ratio = torch.where(
                source_valid & (regressed_volume > 0.0),
                source_volume / regressed_volume.clamp(min=1e-12),
                torch.zeros_like(source_volume),
            )
            source_iou = _aligned_aabb_iou(source_boxes, regressed)
            source_iou = torch.where(
                source_valid, source_iou, torch.zeros_like(source_iou)
            )
            center_delta = (final_boxes[..., :3] - regressed[..., :3]) / extent
            size_delta = (final_boxes[..., 3:] - regressed[..., 3:]) / extent
            center_delta = torch.where(
                variant_valid.unsqueeze(-1), center_delta,
                torch.zeros_like(center_delta)
            )
            size_delta = torch.where(
                variant_valid.unsqueeze(-1), size_delta,
                torch.zeros_like(size_delta)
            )
            features[:, feature_index[
                "source_mask_to_regressed_volume_ratio"
            ]] = volume_ratio
            features[:, feature_index[
                "center_delta_x_scene_norm"
            ]:feature_index["center_delta_z_scene_norm"] + 1] = center_delta
            features[:, feature_index[
                "size_delta_x_scene_norm"
            ]:feature_index["size_delta_z_scene_norm"] + 1] = size_delta
            features[:, feature_index[
                "source_mask_vs_regressed_iou"
            ]] = source_iou

    geometry_features = torch.where(
        torch.isfinite(geometry_features),
        geometry_features,
        torch.zeros_like(geometry_features),
    )
    return {
        "schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "boxes": boxes,
        "valid_mask": valid_mask,
        "geometry_features": geometry_features,
        "geometry_feature_names": feature_names,
        "variant_names": variant_names,
        "variant_configs": variants,
        "variant_config": variants,
        "min_points": min_points,
        "max_point_fraction": max_point_fraction,
        "adaptive_alpha": alpha_values,
        "mask_diagnostics": tuple(batch_diagnostics),
    }
