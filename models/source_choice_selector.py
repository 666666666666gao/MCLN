"""Generic source-choice selector for MCLN candidate score arbitration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _box_cxcyczwhd_to_xyzxyz(boxes):
    center = boxes[..., :3]
    size = boxes[..., 3:6].clamp(min=1e-6)
    return torch.cat([center - 0.5 * size, center + 0.5 * size], dim=-1)


def _pairwise_iou3d(boxes1, boxes2):
    boxes1 = _box_cxcyczwhd_to_xyzxyz(boxes1)
    boxes2 = _box_cxcyczwhd_to_xyzxyz(boxes2)
    mins = torch.max(boxes1[:, None, :3], boxes2[None, :, :3])
    maxs = torch.min(boxes1[:, None, 3:], boxes2[None, :, 3:])
    inter_size = (maxs - mins).clamp(min=0)
    inter = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]
    vol1_size = (boxes1[:, 3:] - boxes1[:, :3]).clamp(min=0)
    vol2_size = (boxes2[:, 3:] - boxes2[:, :3]).clamp(min=0)
    vol1 = vol1_size[:, 0] * vol1_size[:, 1] * vol1_size[:, 2]
    vol2 = vol2_size[:, 0] * vol2_size[:, 1] * vol2_size[:, 2]
    union = (vol1[:, None] + vol2[None, :] - inter).clamp(min=1e-6)
    return inter / union


def compute_source_top1_ious(candidate_boxes, source_scores, source_names,
                             gt_boxes, gt_mask):
    """Return top-1 GT IoU for each deployable source, shape [B, S]."""
    batch_size, num_queries, _ = candidate_boxes.shape
    rows = []
    for source_name in source_names:
        scores = source_scores[source_name].detach().float()
        if scores.shape != (batch_size, num_queries):
            raise ValueError(
                "{} scores must have shape {}".format(
                    source_name, (batch_size, num_queries)
                )
            )
        top_idx = scores.argmax(dim=1)
        top_boxes = candidate_boxes[
            torch.arange(batch_size, device=candidate_boxes.device), top_idx
        ]
        ious = []
        for batch_idx in range(batch_size):
            valid_gt = gt_mask[batch_idx].bool()
            if valid_gt.any():
                gt = gt_boxes[batch_idx, valid_gt]
            else:
                gt = gt_boxes[batch_idx, :1]
            iou = _pairwise_iou3d(
                top_boxes[batch_idx:batch_idx + 1], gt
            ).max()
            ious.append(iou)
        rows.append(torch.stack(ious, dim=0))
    return torch.stack(rows, dim=1)


def _threshold_bucket(values, thresholds):
    bucket = torch.zeros_like(values, dtype=torch.long)
    for threshold in thresholds:
        bucket = bucket + (values >= float(threshold)).long()
    return bucket


def compute_precision_gain_source_targets(
        source_ious, source_names, default_source="default",
        min_iou_gap=0.05, thresholds=(0.25, 0.5)):
    """Precision-first source targets.

    The default source is kept unless a non-default source both crosses a
    higher IoU threshold bucket and beats default by the requested IoU gap.
    """
    if default_source not in source_names:
        raise ValueError("default_source must be in source_names")
    default_idx = list(source_names).index(default_source)
    targets = torch.full(
        (source_ious.shape[0],), default_idx,
        dtype=torch.long, device=source_ious.device
    )
    default_iou = source_ious[:, default_idx]
    default_bucket = _threshold_bucket(default_iou, thresholds)

    best_bucket = default_bucket
    best_iou = default_iou
    for source_idx, source_name in enumerate(source_names):
        if source_name == default_source:
            continue
        source_iou = source_ious[:, source_idx]
        source_bucket = _threshold_bucket(source_iou, thresholds)
        improves_bucket = source_bucket > best_bucket
        clears_gap = (source_iou - default_iou) >= float(min_iou_gap)
        tie_break = (source_bucket == best_bucket) & (source_iou > best_iou)
        choose = clears_gap & (improves_bucket | tie_break)
        targets = torch.where(
            choose,
            torch.full_like(targets, source_idx),
            targets,
        )
        best_bucket = torch.where(choose, source_bucket, best_bucket)
        best_iou = torch.where(choose, source_iou, best_iou)
    return targets


def _sourcewise_focal_bce(choice_scores, targets, gamma=2.0, alpha=0.25):
    one_hot = F.one_hot(
        targets, num_classes=choice_scores.shape[1]
    ).to(dtype=choice_scores.dtype, device=choice_scores.device)
    bce = F.binary_cross_entropy_with_logits(
        choice_scores, one_hot, reduction="none"
    )
    prob = torch.sigmoid(choice_scores)
    pt = one_hot * prob + (1.0 - one_hot) * (1.0 - prob)
    alpha_t = one_hot * alpha + (1.0 - one_hot) * (1.0 - alpha)
    loss = alpha_t * ((1.0 - pt).clamp(min=0) ** gamma) * bce
    return loss.mean()


def compute_source_choice_loss(
        choice_scores, source_ious, source_names, default_source="default",
        target_mode="precision_gain_default_sourcewise_focal_bce",
        min_iou_gap=0.05, thresholds=(0.25, 0.5)):
    if target_mode not in (
            "precision_gain_default_sourcewise_focal_bce",
            "precision_gain_default_ce",
    ):
        raise ValueError(
            "unsupported source-choice target mode: {}".format(target_mode)
        )
    targets = compute_precision_gain_source_targets(
        source_ious=source_ious,
        source_names=source_names,
        default_source=default_source,
        min_iou_gap=min_iou_gap,
        thresholds=thresholds,
    )
    if target_mode.endswith("sourcewise_focal_bce"):
        loss = _sourcewise_focal_bce(choice_scores, targets)
    else:
        loss = F.cross_entropy(choice_scores, targets)

    default_idx = list(source_names).index(default_source)
    selected = choice_scores.argmax(dim=1)
    false_override = (targets == default_idx) & (selected != default_idx)
    row_idx = torch.arange(source_ious.shape[0], device=source_ious.device)
    default_iou = source_ious[:, default_idx]
    selected_iou = source_ious[row_idx, selected]
    oracle_iou = source_ious.max(dim=1).values
    stats = {
        "source_choice_target_non_default_ratio":
            (targets != default_idx).float().mean().detach(),
        "source_choice_selected_non_default_ratio":
            (selected != default_idx).float().mean().detach(),
        "source_choice_false_override_ratio":
            false_override.float().mean().detach(),
        "source_choice_target_acc":
            (selected == targets).float().mean().detach(),
    }
    threshold_suffixes = [(0.25, "025"), (0.50, "050")]
    for threshold, suffix in threshold_suffixes:
        default_ok = default_iou > threshold
        selected_ok = selected_iou > threshold
        oracle_ok = oracle_iou > threshold
        stats["source_choice_default_acc{}".format(suffix)] = (
            default_ok.float().mean().detach()
        )
        stats["source_choice_selected_acc{}".format(suffix)] = (
            selected_ok.float().mean().detach()
        )
        stats["source_choice_oracle_acc{}".format(suffix)] = (
            oracle_ok.float().mean().detach()
        )
        stats["source_choice_selector_fix{}".format(suffix)] = (
            ((~default_ok) & selected_ok).float().mean().detach()
        )
        stats["source_choice_selector_break{}".format(suffix)] = (
            (default_ok & (~selected_ok)).float().mean().detach()
        )
        stats["source_choice_oracle_headroom{}".format(suffix)] = (
            ((~default_ok) & oracle_ok).float().mean().detach()
        )
    return loss, stats


class SourceChoiceSelector(nn.Module):
    """Choose one deployable score source for each sample."""

    def __init__(self, d_model=288, hidden_dim=288, source_names=None,
                 text_dim=None, source_embed_dim=16):
        super().__init__()
        self.source_names = tuple(source_names or ("default", "mask_text"))
        self.source_embedding = nn.Embedding(
            len(self.source_names), source_embed_dim
        )
        self.text_dim = int(text_dim or d_model)
        self.text_projector = nn.Linear(self.text_dim, d_model)
        input_dim = d_model + d_model + 6 + 2 + source_embed_dim
        self.choice_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    @staticmethod
    def _pool_text(text_feats, text_mask):
        if text_feats is None:
            return None
        if text_mask is None:
            return text_feats.mean(dim=1)
        valid = (~text_mask.bool()).to(dtype=text_feats.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (text_feats * valid).sum(dim=1) / denom

    def forward(self, candidate_feats, candidate_boxes, source_scores,
                valid_mask=None, text_feats=None, text_mask=None):
        batch_size, num_queries, feat_dim = candidate_feats.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, num_queries, dtype=torch.bool,
                device=candidate_feats.device,
            )
        text_context = self._pool_text(text_feats, text_mask)
        if text_context is None:
            text_context = candidate_feats.new_zeros(batch_size, feat_dim)
        text_context = self.text_projector(text_context.float())

        choice_rows = []
        source_score_rows = []
        top_indices = []
        for source_idx, source_name in enumerate(self.source_names):
            if source_name not in source_scores:
                raise KeyError("missing source score: {}".format(source_name))
            scores = torch.nan_to_num(
                source_scores[source_name].float(),
                nan=0.0, posinf=1e4, neginf=-1e4,
            )
            masked_scores = scores.masked_fill(~valid_mask.bool(), -1e4)
            top2 = torch.topk(
                masked_scores, k=min(2, num_queries), dim=1
            ).values
            top_score = top2[:, :1]
            if top2.shape[1] > 1:
                margin = top2[:, :1] - top2[:, 1:2]
            else:
                margin = torch.zeros_like(top_score)
            top_idx = masked_scores.argmax(dim=1)
            top_indices.append(top_idx)
            source_score_rows.append(scores)

            batch_idx = torch.arange(batch_size, device=candidate_feats.device)
            feat = candidate_feats[batch_idx, top_idx].float()
            box = candidate_boxes[batch_idx, top_idx].float()
            source_embed = self.source_embedding(
                torch.full(
                    (batch_size,), source_idx, dtype=torch.long,
                    device=candidate_feats.device,
                )
            )
            source_input = torch.cat(
                [feat, text_context, box, top_score, margin, source_embed],
                dim=1,
            )
            choice_rows.append(self.choice_mlp(source_input).squeeze(1))

        choice_scores = torch.stack(choice_rows, dim=1)
        selected_source_id = choice_scores.argmax(dim=1)
        selected_scores = []
        for batch_idx, source_idx in enumerate(selected_source_id.tolist()):
            selected_scores.append(source_score_rows[source_idx][batch_idx])
        selected_source_scores = torch.stack(selected_scores, dim=0)
        return {
            "selector_choice_scores": choice_scores,
            "selector_choice_source_names": list(self.source_names),
            "selected_source_scores": selected_source_scores,
            "selected_source_id": selected_source_id,
            "selector_source_top_indices": torch.stack(top_indices, dim=1),
        }
