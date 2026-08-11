"""Build deployable per-query features for ScanRefer REC reranking."""

import torch
import torch.nn.functional as F

from .mask_fusion import (
    as_query_mask_logits,
    fuse_query_mask_logits,
)
from .rec_reranker import compute_query_ious, select_candidate_indices


FEATURE_SCHEMA_VERSION = "rec-query-v1"


_COMPONENT_KEYS = (
    ("main", "positive_map"),
    ("modifier", "modify_positive_map"),
    ("pronoun", "pron_positive_map"),
    ("relation", "rel_positive_map"),
    ("other", "other_entity_map"),
)


def _first_map_row(inputs, key, batch_size, token_dim, device, dtype):
    value = inputs.get(key)
    if value is None:
        return torch.zeros(batch_size, token_dim, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    if value.dim() == 3:
        value = value[:, 0]
    elif value.dim() != 2:
        raise ValueError("{} must have shape [B,T] or [B,G,T]".format(key))
    if value.shape[0] != batch_size:
        raise ValueError("{} batch size does not match predictions".format(key))
    aligned = torch.zeros(batch_size, token_dim, device=device, dtype=dtype)
    copy_dim = min(token_dim, value.shape[-1])
    aligned[:, :copy_dim] = value[:, :copy_dim]
    return aligned


def _rank_normalize(scores):
    order = scores.argsort(dim=1, descending=True)
    rank = torch.zeros_like(order, dtype=scores.dtype)
    values = torch.arange(
        scores.shape[1], dtype=scores.dtype, device=scores.device
    ).unsqueeze(0).expand_as(rank)
    rank.scatter_(1, order, values)
    return 1.0 - rank / float(max(scores.shape[1] - 1, 1))


def _top_and_margin(scores):
    top = torch.topk(scores, k=min(2, scores.shape[1]), dim=1).values
    top_score = top[:, :1]
    if top.shape[1] == 1:
        margin = torch.zeros_like(top_score)
    else:
        margin = top[:, :1] - top[:, 1:2]
    return top_score, margin


def _as_query_mask(value):
    return as_query_mask_logits(value)


def _mask_statistics(end_points, batch_size, num_queries, device):
    rows = []
    for batch_idx in range(batch_size):
        text_logits = _as_query_mask(
            end_points["last_pred_masks"][batch_idx]
        ).float().to(device)
        query_logits = _as_query_mask(
            end_points["sp_last_pred_masks"][batch_idx]
        ).float().to(device)
        if text_logits.shape != query_logits.shape:
            raise ValueError("text and query mask logits must align")
        if text_logits.shape[0] != num_queries:
            raise ValueError("mask query count does not match box queries")
        text_prob = text_logits.sigmoid()
        query_prob = query_logits.sigmoid()
        fused_prob = fuse_query_mask_logits(
            text_logits,
            query_logits,
            end_points["adaptive_weights"][batch_idx],
        ).sigmoid()
        confidence = (2.0 * (fused_prob - 0.5).abs()).mean(dim=1)
        foreground = (fused_prob > 0.5).float().mean(dim=1)
        dice = 2.0 * (text_prob * query_prob).sum(dim=1) / (
            text_prob.sum(dim=1) + query_prob.sum(dim=1)
        ).clamp(min=1e-6)
        rows.append(torch.stack([confidence, foreground, dice], dim=-1))
    return torch.stack(rows, dim=0)


def _gather_query_values(values, query_indices):
    expand_shape = query_indices.shape + values.shape[2:]
    gather_index = query_indices
    for _ in values.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(expand_shape)
    return torch.gather(values, 1, gather_index)


def _feature_names(proj_dim):
    names = ["query_proj_{}".format(idx) for idx in range(proj_dim)]
    names += ["target_text_proj_{}".format(idx) for idx in range(proj_dim)]
    names += [
        "center_x_norm", "center_y_norm", "center_z_norm",
        "size_x_norm", "size_y_norm", "size_z_norm",
        "score_main", "score_modifier", "score_pronoun",
        "score_relation", "score_other", "score_default",
        "score_contrastive", "rank_default", "rank_contrastive",
        "default_top_score", "default_top_margin",
        "contrastive_top_score", "contrastive_top_margin",
        "query_objectness", "mask_confidence", "mask_foreground_ratio",
        "mask_text_query_dice", "target_text_cosine",
    ]
    return names


def build_full_rec_query_state(end_points, inputs):
    """Build deployable features and scores for every detector query."""
    centers = end_points["last_center"].float()
    sizes = end_points["last_pred_size"].float().clamp(min=1e-6)
    sem_logits = end_points["last_sem_cls_scores"].float()
    proj_queries = F.normalize(end_points["last_proj_queries"].float(), p=2, dim=-1)
    proj_tokens = F.normalize(end_points["proj_tokens"].float(), p=2, dim=-1)
    batch_size, num_queries, _ = centers.shape
    if sizes.shape != centers.shape:
        raise ValueError("last_center and last_pred_size must align")
    if sem_logits.shape[:2] != (batch_size, num_queries):
        raise ValueError("semantic logits must align with box queries")
    if proj_queries.shape[:2] != (batch_size, num_queries):
        raise ValueError("projected queries must align with box queries")

    sem_maps = {
        name: _first_map_row(
            inputs, key, batch_size, sem_logits.shape[-1],
            sem_logits.device, sem_logits.dtype
        )
        for name, key in _COMPONENT_KEYS
    }
    sem_prob = sem_logits.softmax(dim=-1)
    component_scores = {
        name: torch.matmul(sem_prob, value.unsqueeze(-1)).squeeze(-1)
        for name, value in sem_maps.items()
    }
    default_scores = (
        component_scores["main"]
        + component_scores["modifier"]
        + component_scores["pronoun"]
        + component_scores["relation"]
        - component_scores["other"]
    )

    proj_maps = {
        name: _first_map_row(
            inputs, key, batch_size, proj_tokens.shape[1],
            proj_tokens.device, proj_tokens.dtype
        )
        for name, key in _COMPONENT_KEYS
    }
    signed_text_map = (
        proj_maps["main"] + proj_maps["modifier"]
        + proj_maps["pronoun"] + proj_maps["relation"]
        - proj_maps["other"]
    )
    token_similarity = torch.matmul(
        proj_queries, proj_tokens.transpose(-1, -2)
    )
    contrastive_scores = (
        token_similarity * signed_text_map.unsqueeze(1)
    ).sum(dim=-1)

    positive_text_map = (
        proj_maps["main"] + proj_maps["modifier"]
        + proj_maps["pronoun"] + proj_maps["relation"]
    ).clamp(min=0.0)
    positive_text_map = positive_text_map / positive_text_map.sum(
        dim=1, keepdim=True
    ).clamp(min=1e-6)
    target_text = torch.matmul(
        positive_text_map.unsqueeze(1), proj_tokens
    ).squeeze(1)
    target_text = F.normalize(target_text, p=2, dim=-1)
    target_text_full = target_text.unsqueeze(1).expand(
        -1, num_queries, -1
    )
    target_cosine = (proj_queries * target_text_full).sum(dim=-1, keepdim=True)

    coords = inputs["point_clouds"][..., :3].float().to(centers.device)
    scene_min = coords.min(dim=1).values
    scene_max = coords.max(dim=1).values
    scene_extent = (scene_max - scene_min).clamp(min=1e-6)
    normalized_boxes = torch.cat([
        (centers - scene_min.unsqueeze(1)) / scene_extent.unsqueeze(1),
        sizes / scene_extent.unsqueeze(1),
    ], dim=-1)

    default_rank = _rank_normalize(default_scores)
    contrastive_rank = _rank_normalize(contrastive_scores)
    default_top, default_margin = _top_and_margin(default_scores)
    contrastive_top, contrastive_margin = _top_and_margin(contrastive_scores)

    objectness = centers.new_zeros(batch_size, num_queries)
    if ("seeds_obj_cls_logits" in end_points
            and "query_points_sample_inds" in end_points):
        seed_logits = end_points["seeds_obj_cls_logits"].float().squeeze(1)
        sample_indices = end_points["query_points_sample_inds"].long()
        if sample_indices.shape != (batch_size, num_queries):
            raise ValueError("query sample indices must align with queries")
        objectness = torch.gather(seed_logits, 1, sample_indices).sigmoid()

    mask_stats = _mask_statistics(
        end_points, batch_size, num_queries, centers.device
    )
    repeated_globals = torch.cat([
        default_top, default_margin, contrastive_top, contrastive_margin
    ], dim=-1).unsqueeze(1).expand(-1, num_queries, -1)
    full_features = torch.cat([
        proj_queries,
        target_text_full,
        normalized_boxes,
        component_scores["main"].unsqueeze(-1),
        component_scores["modifier"].unsqueeze(-1),
        component_scores["pronoun"].unsqueeze(-1),
        component_scores["relation"].unsqueeze(-1),
        component_scores["other"].unsqueeze(-1),
        default_scores.unsqueeze(-1),
        contrastive_scores.unsqueeze(-1),
        default_rank.unsqueeze(-1),
        contrastive_rank.unsqueeze(-1),
        repeated_globals,
        objectness.unsqueeze(-1),
        mask_stats,
        target_cosine,
    ], dim=-1)
    full_features = torch.nan_to_num(full_features)

    full_boxes = torch.cat([centers, sizes], dim=-1)
    feature_names = _feature_names(proj_queries.shape[-1])
    if full_features.shape[-1] != len(feature_names):
        raise RuntimeError("REC feature schema does not match feature tensor")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "features": full_features,
        "boxes": full_boxes,
        "default_scores": default_scores,
        "contrastive_scores": contrastive_scores,
        "num_queries": num_queries,
    }


def compact_rec_query_state(full_state, topk_per_source=8,
                            max_candidates=16):
    """Select and gather the deployable compact REC candidate state."""
    default_scores = full_state["default_scores"]
    contrastive_scores = full_state["contrastive_scores"]
    num_queries = full_state["num_queries"]
    if (not isinstance(num_queries, int) or isinstance(num_queries, bool)
            or num_queries <= 0):
        raise ValueError("num_queries must be a positive integer")
    if default_scores.dim() != 2:
        raise ValueError("default_scores must have shape [B,Q]")
    if num_queries != default_scores.shape[1]:
        raise ValueError("num_queries must match default_scores")
    if contrastive_scores.shape != default_scores.shape:
        raise ValueError("contrastive_scores must match default_scores")
    for key in ("features", "boxes"):
        if full_state[key].shape[:2] != default_scores.shape:
            raise ValueError("{} must align with default_scores".format(key))
    query_indices, valid_mask = select_candidate_indices(
        default_scores,
        contrastive_scores,
        topk_per_source=topk_per_source,
        max_candidates=max_candidates,
    )
    features = _gather_query_values(full_state["features"], query_indices)
    boxes = _gather_query_values(full_state["boxes"], query_indices)
    compact_default = _gather_query_values(
        default_scores.unsqueeze(-1), query_indices
    ).squeeze(-1)
    compact_contrastive = _gather_query_values(
        contrastive_scores.unsqueeze(-1), query_indices
    ).squeeze(-1)
    model_inputs = {"features": features, "valid_mask": valid_mask}
    return {
        "schema_version": full_state["schema_version"],
        "feature_names": full_state["feature_names"],
        "features": features,
        "boxes": boxes,
        "query_indices": query_indices,
        "valid_mask": valid_mask,
        "default_scores": compact_default,
        "contrastive_scores": compact_contrastive,
        "default_top1_query_index": default_scores.argmax(dim=1),
        "num_queries": num_queries,
        "model_inputs": model_inputs,
    }


def build_rec_candidate_batch(end_points, inputs, topk_per_source=8,
                              max_candidates=16):
    """Build compact candidate features without reading ground truth."""
    full_state = build_full_rec_query_state(end_points, inputs)
    return compact_rec_query_state(
        full_state,
        topk_per_source=topk_per_source,
        max_candidates=max_candidates,
    )


def attach_candidate_targets(candidate_batch, end_points, root_only=False):
    """Attach training-only IoU targets without changing model inputs.

    ``root_only`` matches the ScanRefer REC evaluator, which scores the first
    target box even when detector-intermediate supervision also adds anchors.
    """
    gt_boxes = torch.cat([
        end_points["center_label"][..., :3].float(),
        end_points["size_gts"].float(),
    ], dim=-1)
    gt_mask = end_points["box_label_mask"]
    if root_only:
        gt_boxes = gt_boxes[:, :1]
        gt_mask = gt_mask[:, :1]
    candidate_ious = compute_query_ious(
        candidate_batch["boxes"],
        gt_boxes,
        gt_mask,
    )
    candidate_ious = candidate_ious.masked_fill(
        ~candidate_batch["valid_mask"].bool(), 0.0
    )
    result = dict(candidate_batch)
    result["candidate_ious"] = candidate_ious
    result["threshold_labels"] = torch.stack([
        candidate_ious > 0.25,
        candidate_ious > 0.50,
    ], dim=-1)
    return result


def scatter_candidate_scores(candidate_scores, query_indices, valid_mask,
                             num_queries, fill_value=float("-inf")):
    """Map compact candidate scores to the original query axis."""
    if candidate_scores.dim() != 2:
        raise ValueError("candidate_scores must have shape [B,K]")
    if query_indices.shape != candidate_scores.shape:
        raise ValueError("query_indices must match candidate_scores")
    if valid_mask.shape != candidate_scores.shape:
        raise ValueError("valid_mask must match candidate_scores")
    if not isinstance(num_queries, int) or num_queries <= 0:
        raise ValueError("num_queries must be a positive integer")
    valid = valid_mask.bool()
    if valid.any():
        valid_indices = query_indices[valid]
        if (valid_indices < 0).any() or (valid_indices >= num_queries).any():
            raise ValueError("valid query index is out of range")
    output = candidate_scores.new_full(
        (candidate_scores.shape[0], num_queries), float(fill_value)
    )
    for batch_idx in range(candidate_scores.shape[0]):
        row_valid = valid[batch_idx]
        output[batch_idx, query_indices[batch_idx, row_valid]] = (
            candidate_scores[batch_idx, row_valid]
        )
    return output
