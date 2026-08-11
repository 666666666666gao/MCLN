"""MCLN adapter for the generic source-choice selector."""

import torch
import torch.nn.functional as F

from .mask_fusion import as_query_mask_logits, query_fusion_weight
from .rec_candidate_adapter import build_full_rec_query_state


def _align_token_scores(token_scores, target_map):
    if token_scores.shape[-1] == target_map.shape[-1]:
        return token_scores
    aligned = token_scores.new_zeros(
        token_scores.shape[0], token_scores.shape[1], target_map.shape[-1]
    )
    copy_dim = min(token_scores.shape[-1], target_map.shape[-1])
    aligned[:, :, :copy_dim] = token_scores[:, :, :copy_dim]
    return aligned


def _first_row(inputs, key, batch_size, token_dim, device):
    value = inputs.get(key)
    if value is None:
        return torch.zeros(batch_size, token_dim, device=device)
    value = value.float().to(device)
    if value.dim() == 3:
        return value[:, 0]
    if value.dim() == 2:
        return value
    raise ValueError("{} must have shape [B,T] or [B,G,T]".format(key))


def compute_default_source_scores(end_points, inputs):
    sem_scores = end_points["last_sem_cls_scores"].float().softmax(-1)
    batch_size, _, token_dim = sem_scores.shape
    device = sem_scores.device
    positive = _first_row(inputs, "positive_map", batch_size, token_dim, device)
    positive = torch.where(
        positive > 0, torch.ones_like(positive), positive
    )
    modify = _first_row(inputs, "modify_positive_map", batch_size, token_dim, device)
    pron = _first_row(inputs, "pron_positive_map", batch_size, token_dim, device)
    other = _first_row(inputs, "other_entity_map", batch_size, token_dim, device)
    rel = _first_row(inputs, "rel_positive_map", batch_size, token_dim, device)

    target_map = positive + modify + pron + rel - other
    sem_scores = _align_token_scores(sem_scores, target_map)
    return (sem_scores * target_map.unsqueeze(1)).sum(-1)


def _target_token_map(inputs, batch_size, token_dim, device):
    positive = _first_row(inputs, "positive_map", batch_size, token_dim, device)
    modify = _first_row(inputs, "modify_positive_map", batch_size, token_dim, device)
    pron = _first_row(inputs, "pron_positive_map", batch_size, token_dim, device)
    other = _first_row(inputs, "other_entity_map", batch_size, token_dim, device)
    rel = _first_row(inputs, "rel_positive_map", batch_size, token_dim, device)
    return positive + modify + pron + rel - other


def compute_contrastive_text_source_scores(end_points, inputs, temperature=1.0):
    if "last_proj_queries" not in end_points or "proj_tokens" not in end_points:
        raise KeyError(
            "contrastive_text requires last_proj_queries and proj_tokens"
        )
    proj_queries = F.normalize(end_points["last_proj_queries"].float(), p=2, dim=-1)
    proj_tokens = F.normalize(end_points["proj_tokens"].float(), p=2, dim=-1)
    token_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
    if temperature != 1.0:
        token_scores = token_scores / float(temperature)
    batch_size, _, token_dim = token_scores.shape
    target_map = _target_token_map(
        inputs, batch_size, token_dim, token_scores.device
    )
    token_scores = _align_token_scores(token_scores, target_map)
    return (token_scores * target_map.unsqueeze(1)).sum(-1)


def _rank_normalize(scores):
    order = scores.argsort(dim=1, descending=True)
    rank = torch.zeros_like(order, dtype=scores.dtype)
    values = torch.arange(
        scores.shape[1], device=scores.device, dtype=scores.dtype
    ).unsqueeze(0).expand_as(rank)
    rank.scatter_(1, order, values)
    denom = max(scores.shape[1] - 1, 1)
    return 1.0 - rank / float(denom)


def compute_default_rank_blend_contrastive_source_scores(
        end_points, inputs, contrastive_weight=0.1):
    default_scores = compute_default_source_scores(end_points, inputs)
    contrastive_scores = compute_contrastive_text_source_scores(end_points, inputs)
    default_rank = _rank_normalize(default_scores)
    contrastive_rank = _rank_normalize(contrastive_scores)
    contrastive_weight = float(contrastive_weight)
    return (
        (1.0 - contrastive_weight) * default_rank
        + contrastive_weight * contrastive_rank
    )


def _as_query_mask(mask_tensor):
    return as_query_mask_logits(mask_tensor, "mask tensors")


def compute_mask_text_source_scores(end_points):
    rows = []
    for batch_idx, text_mask_logits in enumerate(end_points["last_pred_masks"]):
        text_mask_logits = _as_query_mask(text_mask_logits).float()
        query_mask_logits = _as_query_mask(
            end_points["sp_last_pred_masks"][batch_idx]
        ).float()
        if text_mask_logits.shape != query_mask_logits.shape:
            raise ValueError("text and query mask logits must align")
        text_prob = text_mask_logits.sigmoid()
        query_prob = query_mask_logits.sigmoid()
        intersection = (text_prob * query_prob).sum(dim=1)
        denom = (text_prob.sum(dim=1) + query_prob.sum(dim=1)).clamp(min=1e-6)
        dice = 2.0 * intersection / denom
        adaptive_weight = query_fusion_weight(
            end_points["adaptive_weights"][batch_idx],
            dice.shape[0],
            dice,
        )
        if adaptive_weight.dim() == 2:
            adaptive_weight = adaptive_weight.squeeze(-1)
        query_conf = query_prob.mean(dim=1)
        rows.append(adaptive_weight * dice + (1.0 - adaptive_weight) * query_conf)
    return torch.stack(rows, dim=0)


def compute_sacr_structured_source_scores(end_points, inputs):
    """Apply a learned, confidence-scaled SACR residual to the shared source."""
    if "sacr_structured_residual" not in end_points:
        raise KeyError("SACR residual is unavailable")
    default_scores = compute_default_source_scores(end_points, inputs)
    residual = end_points["sacr_structured_residual"].float()
    if residual.shape != default_scores.shape:
        raise ValueError("SACR residual must align with default scores")
    return default_scores + residual


def build_mcln_source_choice_batch(
        end_points, inputs, source_names=None,
        include_rich_candidate_feats=False):
    if not isinstance(include_rich_candidate_feats, bool):
        raise ValueError("include_rich_candidate_feats must be boolean")
    source_names = tuple(source_names or ("default", "mask_text"))
    candidate_boxes = torch.cat(
        [end_points["last_center"], end_points["last_pred_size"].clamp(min=1e-6)],
        dim=-1,
    )
    if "source_choice_candidate_feats" in end_points:
        candidate_feats = end_points["source_choice_candidate_feats"]
    elif "last_proj_queries" in end_points:
        candidate_feats = end_points["last_proj_queries"]
    else:
        candidate_feats = end_points["query_points_feature"].transpose(1, 2)
    valid_mask = torch.ones(
        candidate_boxes.shape[:2], dtype=torch.bool, device=candidate_boxes.device
    )

    score_builders = {
        "default": lambda: compute_default_source_scores(end_points, inputs),
        "mask_text": lambda: compute_mask_text_source_scores(end_points),
        "contrastive_text": lambda: compute_contrastive_text_source_scores(
            end_points, inputs
        ),
        "sacr_structured": lambda: compute_sacr_structured_source_scores(
            end_points, inputs
        ),
        "default_rank_blend_contrastive005": lambda:
            compute_default_rank_blend_contrastive_source_scores(
                end_points, inputs, contrastive_weight=0.05
            ),
        "default_rank_blend_contrastive010": lambda:
            compute_default_rank_blend_contrastive_source_scores(
                end_points, inputs, contrastive_weight=0.10
            ),
    }
    source_requirements = {
        "default": ("last_sem_cls_scores",),
        "mask_text": (
            "last_pred_masks", "sp_last_pred_masks", "adaptive_weights",
        ),
        "contrastive_text": ("last_proj_queries", "proj_tokens"),
        "sacr_structured": (
            "last_sem_cls_scores",
            "sacr_structured_residual",
            "sacr_structured_valid_mask",
        ),
        "default_rank_blend_contrastive005": (
            "last_sem_cls_scores", "last_proj_queries", "proj_tokens",
        ),
        "default_rank_blend_contrastive010": (
            "last_sem_cls_scores", "last_proj_queries", "proj_tokens",
        ),
    }
    source_scores = {}
    source_validity_columns = []
    for name in source_names:
        if name not in score_builders:
            raise KeyError("unsupported source score: {}".format(name))
        available = all(
            key in end_points for key in source_requirements[name]
        )
        if available:
            source_scores[name] = score_builders[name]()
            if name == "sacr_structured":
                row_valid = end_points[
                    "sacr_structured_valid_mask"
                ].to(device=valid_mask.device).bool()
                if row_valid.shape != (candidate_boxes.shape[0],):
                    raise ValueError(
                        "SACR validity must have shape [B]"
                    )
                source_validity_columns.append(
                    valid_mask & row_valid.unsqueeze(1)
                )
            else:
                source_validity_columns.append(valid_mask)
        else:
            source_scores[name] = candidate_boxes.new_zeros(
                candidate_boxes.shape[:2]
            )
            source_validity_columns.append(torch.zeros_like(valid_mask))
    source_validity = torch.stack(source_validity_columns, dim=-1)
    rich_candidate_feats = None
    if include_rich_candidate_feats:
        rich_candidate_feats = build_full_rec_query_state(
            end_points, inputs
        )["features"]
    return {
        "candidate_boxes": candidate_boxes,
        "candidate_feats": candidate_feats,
        "gate_candidate_feats": end_points.get(
            "source_moe_gate_candidate_feats"
        ),
        "rich_candidate_feats": rich_candidate_feats,
        "source_scores": source_scores,
        "source_validity": source_validity,
        "valid_mask": valid_mask,
        "text_feats": end_points.get("text_feats"),
        "text_mask": end_points.get("text_attention_mask"),
        "meta": {
            "source_names": list(source_names),
            "source_available": source_validity.flatten(0, 1).any(
                dim=0
            ).tolist(),
        },
    }
