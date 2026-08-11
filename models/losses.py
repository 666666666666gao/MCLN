# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.scatter_util import scatter_mean
import math
import numpy as np
from .source_choice_selector import (
    compute_source_choice_loss,
    compute_source_top1_ious,
)
from .mask_fusion import (
    as_query_mask_logits,
    fuse_query_mask_logits,
    gather_query_fusion_weight,
)
from .source_moe import (
    build_fused_query_mask_logits,
    compute_load_balance_loss,
    compute_query_box_ious,
    compute_query_mask_ious,
    compute_source_moe_fallback_gate_loss,
    compute_source_moe_ranking_loss,
)
from .joint_query_quality import compute_joint_query_quality_loss


def build_source_moe_grounding_sample_mask(end_points, batch_size, device):
    """Select samples with a single referring target for MoE supervision."""
    sample_datasets = end_points.get("sample_dataset")
    if sample_datasets is None:
        raise ValueError(
            "SourceMoE supervision requires per-sample sample_dataset metadata"
        )
    if isinstance(sample_datasets, str):
        sample_datasets = [sample_datasets] * batch_size
    if len(sample_datasets) != batch_size:
        raise ValueError("sample_dataset must align with MoE scores")
    if any(not isinstance(name, str) or not name for name in sample_datasets):
        raise ValueError("sample_dataset entries must be non-empty strings")
    return torch.tensor(
        [name != "scannet" for name in sample_datasets],
        dtype=torch.bool,
        device=device,
    )

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union

# BRIEF 3DIoU loss
def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.

    This class is taken from Group-Free code.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter for hard and easy examples.
            alpha: Weighting parameter for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
        max(x, 0) - x * z + log(1 + exp(-abs(x))) in

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0) - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()      # B, K
    seed_xyz = end_points['seed_xyz']               # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']   # B, 1, K
    gt_center = end_points['center_label'][:, :, :3]            # B, G=132, 3
    gt_size = end_points['size_gts'][:, :, :3]                  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]   # number if points from p++ output  1024
    G = gt_center.shape[1]  # number of gt boxes (with padding) 132

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']           # B, num_points=5000
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)   # B, K=1024
    obj_assignment[obj_assignment < 0] = G - 1                          # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)       # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()

    # Find the points that lie closest to each gt centroid
    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0 

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss


class HungarianMatcher(nn.Module):
    """
    Assign targets to predictions.

    This class is taken from MDETR and is modified for our purposes.

    For efficiency reasons, the [targets don't include the no_object].
    Because of this, in general, there are [more predictions than targets].
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        """
        Initialize matcher.

        Args:
            cost_class: relative weight of the classification error
            cost_bbox: relative weight of the L1 bounding box regression error
            cost_giou: relative weight of the giou loss of the bounding box
            soft_token: whether to use soft-token prediction
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_masks = 0.0002  # mask weight
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits" (tensor): [batch_size, num_queries, num_classes]
                "pred_boxes" (tensor): [batch_size, num_queries, 6], cxcyczwhd
            targets: list (len(targets) = batch_size) of dict:
                "labels" (tensor): [num_target_boxes]
                    (where num_target_boxes is the no. of ground-truth objects)
                "boxes" (tensor): [num_target_boxes, 6], cxcyczwhd
                "positive_map" (tensor): [num_target_boxes, 256]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Notation: {B: batch_size, Q: num_queries, C: num_classes}
        bs, num_queries = outputs["pred_logits"].shape[:2]  # Q: num_queries = 256

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C=256]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 6]

        cost_masks = 0.0
        if "pred_masks" in outputs:
            cost_masks = []
            out_masks =  None
            tgt_masks = torch.cat([v["masks"] for v in targets]) # (B, 50000)
            for idx in range(len(outputs["pred_masks"])):
                out_mask = outputs["pred_masks"][idx].squeeze(0)  # [Q, super_num]
                out_mask = (out_mask > 0).float()  # [Q, super_num]
                superpoint = outputs["superpoints"][idx].unsqueeze(0).expand(out_mask.shape[0], -1)  # (Q, 50000)
                out_mask = torch.gather(out_mask, 1, superpoint)  # (Q, 50000)
                if out_masks == None:
                    out_masks = out_mask
                else:
                    out_masks = torch.cat([out_masks, out_mask], dim=0)  # (B*Q, 50000)
                
            cost_masks = torch.cdist(out_masks, tgt_masks.float(), p=1)    # ([B*Q, 2]) 110 - 2092 * 0.0002

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets])  # (B, 256)
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (B)
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # (B, 6)

        if self.soft_token:
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))  # (256, 1)
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)    # ([B*Q, 2])  0.08 - 15.3 * 1

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(     # ([B*Q, 2])  -0.8 - 0.98 * 2
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox          # 0 * 
            + self.cost_class * cost_class      # 1 * ([B*Q, 2])
            + self.cost_giou * cost_giou        # 2 * ([B*Q, 2])
            + self.cost_masks * cost_masks
        ).view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred boxes
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt boxes
            )
            for i, j in indices
        ]

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def lovasz_gradient(sorted_targets):
    """Return the Lovasz extension gradient for sorted binary targets."""
    if (not isinstance(sorted_targets, torch.Tensor)
            or sorted_targets.dim() != 1
            or not sorted_targets.is_floating_point()):
        raise ValueError("sorted Lovasz targets must be floating and one-dimensional")
    if sorted_targets.numel() == 0:
        return sorted_targets
    target_sum = sorted_targets.sum()
    intersection = target_sum - sorted_targets.cumsum(0)
    union = target_sum + (1.0 - sorted_targets).cumsum(0)
    gradient = 1.0 - intersection / union.clamp(min=1.0)
    if sorted_targets.numel() > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


def lovasz_hinge_loss(inputs, targets, num_masks):
    """Binary Lovasz hinge loss from the Lovasz-Softmax formulation."""
    if (not isinstance(inputs, torch.Tensor) or inputs.dim() != 2
            or not inputs.is_floating_point()
            or not isinstance(targets, torch.Tensor)
            or targets.shape != inputs.shape
            or not targets.is_floating_point()
            or targets.device != inputs.device
            or bool(((targets < 0.0) | (targets > 1.0)).any().item())):
        raise ValueError("Lovasz inputs and binary targets must align as [N,S]")
    if (not isinstance(num_masks, int) or isinstance(num_masks, bool)
            or num_masks <= 0 or num_masks != inputs.shape[0]):
        raise ValueError("Lovasz num_masks must match the mask batch")
    losses = []
    for logits, labels in zip(inputs, targets):
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, permutation = torch.sort(errors, descending=True)
        sorted_labels = labels.index_select(0, permutation.detach())
        losses.append(torch.dot(
            F.relu(errors_sorted), lovasz_gradient(sorted_labels)
        ))
    return torch.stack(losses).sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_boxes, weight=1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    ce_loss=ce_loss*weight
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def build_joint_query_mask_candidate_mask(
        scores, box_ious, valid_mask, sample_mask, top_k):
    """Select deployable and box-oracle candidates for dense mask supervision."""
    if (not isinstance(scores, torch.Tensor) or scores.dim() != 2
            or not scores.is_floating_point()
            or not isinstance(box_ious, torch.Tensor)
            or box_ious.shape != scores.shape
            or not box_ious.is_floating_point()
            or box_ious.device != scores.device
            or not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != scores.shape
            or valid_mask.device != scores.device
            or not isinstance(sample_mask, torch.Tensor)
            or sample_mask.dtype != torch.bool
            or sample_mask.shape != (scores.shape[0],)
            or sample_mask.device != scores.device):
        raise ValueError(
            "candidate scores, IoUs, and masks must align as [B,Q]/[B]"
        )
    if (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError("candidate mask top_k must be a positive integer")
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("every row must contain a valid query")
    if (not bool(torch.isfinite(scores.masked_fill(~valid_mask, 0.0)).all().item())
            or not bool(torch.isfinite(box_ious).all().item())
            or bool(((box_ious < 0.0) | (box_ious > 1.0)).any().item())):
        raise ValueError("valid candidate scores and box IoUs must be finite")

    count = min(top_k, scores.shape[1])
    selected = torch.zeros_like(valid_mask)
    with torch.no_grad():
        for ranking_values in (scores, box_ious):
            top_values, top_indices = ranking_values.detach().masked_fill(
                ~valid_mask, float("-inf")
            ).topk(count, dim=1)
            top_valid = torch.isfinite(top_values)
            selected.scatter_(1, top_indices, top_valid)
    return selected & valid_mask & sample_mask.unsqueeze(1)


def compute_joint_query_mask_candidate_loss(
        text_mask_logits, query_mask_logits, adaptive_weights,
        gt_point_masks, superpoints, candidate_mask, compute_lovasz=False):
    """Apply focal/dice supervision to selected query-wise fused masks."""
    if (not isinstance(text_mask_logits, (list, tuple))
            or not isinstance(query_mask_logits, (list, tuple))
            or not isinstance(adaptive_weights, (list, tuple))
            or not (len(text_mask_logits) == len(query_mask_logits)
                    == len(adaptive_weights))):
        raise ValueError("candidate mask logits and weights must be batch lists")
    batch_size = len(text_mask_logits)
    if (not isinstance(gt_point_masks, torch.Tensor)
            or gt_point_masks.dim() != 3
            or gt_point_masks.shape[0] != batch_size
            or gt_point_masks.shape[1] < 1
            or not isinstance(superpoints, torch.Tensor)
            or superpoints.dim() != 2
            or superpoints.shape[0] != batch_size
            or superpoints.shape[1] != gt_point_masks.shape[-1]
            or not isinstance(candidate_mask, torch.Tensor)
            or candidate_mask.dtype != torch.bool
            or candidate_mask.dim() != 2
            or candidate_mask.shape[0] != batch_size):
        raise ValueError("candidate mask targets must align by batch and point")
    if not isinstance(compute_lovasz, bool):
        raise ValueError("compute_lovasz must be boolean")

    selected_fused = []
    selected_targets = []
    differentiable_zero = None
    for batch_idx, (text_row, query_row, weight) in enumerate(zip(
            text_mask_logits, query_mask_logits, adaptive_weights)):
        text_row = as_query_mask_logits(text_row, "text mask logits")
        query_row = as_query_mask_logits(query_row, "query mask logits")
        if (text_row.shape != query_row.shape
                or query_row.device != text_row.device
                or text_row.shape[0] != candidate_mask.shape[1]
                or candidate_mask.device != text_row.device
                or gt_point_masks.device != text_row.device
                or superpoints.device != text_row.device):
            raise ValueError("candidate queries, masks, and targets must align")
        row_zero = (text_row.sum() + query_row.sum()) * 0.0
        differentiable_zero = (
            row_zero if differentiable_zero is None
            else differentiable_zero + row_zero
        )
        query_indices = candidate_mask[batch_idx].nonzero(
            as_tuple=False
        ).flatten()
        if query_indices.numel() == 0:
            continue
        selected_text = text_row.index_select(0, query_indices)
        selected_query = query_row.index_select(0, query_indices)
        selected_weight = gather_query_fusion_weight(
            weight, query_indices, text_row.shape[0], selected_text
        )
        fused_row = fuse_query_mask_logits(
            selected_text, selected_query, selected_weight
        )
        target_superpoints = scatter_mean(
            gt_point_masks[batch_idx, 0].float(),
            superpoints[batch_idx],
            dim=-1,
            dim_size=fused_row.shape[1],
        )
        target_superpoints = (target_superpoints > 0.5).to(fused_row.dtype)
        selected_fused.append(fused_row)
        selected_targets.append(target_superpoints.unsqueeze(0).expand_as(
            fused_row
        ))

    if not selected_fused:
        if differentiable_zero is None:
            raise ValueError("candidate mask batch must not be empty")
        return {
            "mask_loss": differentiable_zero,
            "dice_loss": differentiable_zero,
            "lovasz_loss": differentiable_zero,
        }
    fused = torch.cat(selected_fused, dim=0)
    targets = torch.cat(selected_targets, dim=0)
    selected_count = fused.shape[0]
    return {
        "mask_loss": sigmoid_focal_loss(fused, targets, selected_count),
        "dice_loss": dice_loss(fused, targets, selected_count),
        "lovasz_loss": (
            lovasz_hinge_loss(fused, targets, selected_count)
            if compute_lovasz else fused.sum() * 0.0
        ),
    }

# BRIEF Compute loss
class SetCriterion(nn.Module):
    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef    # 0.1
        self.losses = losses
        self.temperature = temperature
    
    #####################################
    # BRIEF dense position-aligned loss #
    #####################################
    def loss_pos_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)
        
        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])                  # main object
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])    # attribute(modify)
        pron_positive_map = torch.cat([t["pron_positive_map"] for t in targets])        # pron
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])          # other(auxi)
        rel_positive_map = torch.cat([t["rel_positive_map"] for t in targets])          # relation

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]
        mod_pos = modify_positive_map[tgt_idx]
        pron_pos = pron_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]
        rel_pos = rel_positive_map[tgt_idx]
        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos * 0.6 + mod_pos * 0.2 + pron_pos * 0.2 + rel_pos*0.1
        # TODO SR3D (5:1:1:1)/8 = 0.625: 0.125: 0.125: 0.125
        if outputs["language_dataset"][0] == "sr3d":
            tgt_weight_pos = tgt_pos * 0.625 + mod_pos * 0.125 + pron_pos * 0.125 + rel_pos * 0.125

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + pron_pos + rel_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    # BRIEF object detection loss.
    def loss_boxes(self, outputs, targets, indices, num_boxes, auxi_indices):
        """Compute bbox losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        
        loss_bbox = (
            F.l1_loss(
                src_boxes[..., :3], target_boxes[..., :3],
                reduction='none'
            )
            + 0.2 * F.l1_loss(
                src_boxes[..., 3:], target_boxes[..., 3:],
                reduction='none'
            )
        )
        losses = {}
        
        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # BRIEF object detection loss.
    def loss_masks(self, outputs, targets, indices, num_boxes, auxi_indices):
        """Compute mask losses."""
        losses = {}
        focal = 0.0
        dice = 0.0
        sp_focal=0.0
        sp_dice=0.0
        adaptive_weight_focal = 0.0
        adaptive_weight_dice = 0.0
        corresponding_focal=0.0
        corresponding_dice=0.0

        if 'pred_masks' in outputs:
            for bs in range(len(outputs['pred_masks'])):
                idx0 = indices[bs][0]  #预测的mask的idx
                superpoint = outputs['superpoints'][bs]
                idx1 = indices[bs][1]  #gt mask的idx
                target = targets[bs]['masks'][idx1].float()  # [len(indices), 50000] [bs,50000] 
                target_masks = scatter_mean(target, superpoint, dim=-1)  # [len(indices), super_num] [bs,super_num]
                target_masks = (target_masks > 0.5).float()

                all_text_masks = as_query_mask_logits(
                    outputs['pred_masks'][bs], 'pred_masks'
                )
                all_query_masks = as_query_mask_logits(
                    outputs['sp_pred_masks'][bs], 'sp_pred_masks'
                )
                mask_query_indices = idx0.to(
                    device=all_text_masks.device, dtype=torch.long
                )
                src_masks = all_text_masks.index_select(
                    0, mask_query_indices
                )
                sp_src_masks = all_query_masks.index_select(
                    0, mask_query_indices
                )
                
                focal += sigmoid_focal_loss(src_masks, target_masks, num_boxes)
                dice += dice_loss(src_masks, target_masks, num_boxes)
                
                sp_focal += sigmoid_focal_loss(sp_src_masks,target_masks,  num_boxes)
                sp_dice += dice_loss(sp_src_masks,target_masks,  num_boxes)
                
                sp_src_masks_2=(sp_src_masks > 0.5).float()
                bs_super_xyz=outputs['super_xyz_list'][bs]
                if torch.sum(sp_src_masks_2)>0:
                    selected_index=torch.nonzero(sp_src_masks_2)[:,1]
                    selected_xyz=torch.index_select(bs_super_xyz,dim=1,index=selected_index)
                    num_points = selected_xyz.size(1)
                else:
                    num_points=0

                if num_points > 1:
                    selected_xyz = selected_xyz.unsqueeze(2)

                    distances = torch.norm(selected_xyz - selected_xyz.permute(0, 2, 1, 3), dim=3)
                    distance_sum = distances.sum(dim=(1, 2))
                    distance_mean=distance_sum/((num_points-1)*num_points )
                else:
                    distance_mean = sp_src_masks.new_zeros(())
                
                dice_weight = 1.0 / (1+distance_mean)
                corresponding_dice+=dice_weight*dice_loss(src_masks ,sp_src_masks_2,  num_boxes)
                # Keep the focal weighting on-device; the legacy NumPy path
                # forced a GPU synchronization for every sample.
                u1 = 0.5
                sigma1 = 0.1
                left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
                right = torch.exp(
                    -(sp_src_masks.sigmoid().detach() - u1).square()
                    / (2 * sigma1)
                )
                weight = 2 - left * right
                corresponding_focal+=sigmoid_focal_loss(src_masks ,sp_src_masks_2,  num_boxes,weight)

                adaptive_weight = gather_query_fusion_weight(
                    outputs['adaptive_weights'][bs],
                    mask_query_indices,
                    all_query_masks.shape[0],
                    sp_src_masks,
                )
                adaptive_weight_mask = fuse_query_mask_logits(
                    src_masks, sp_src_masks, adaptive_weight
                )
                
                adaptive_weight_focal+=sigmoid_focal_loss(adaptive_weight_mask, target_masks, num_boxes)
                adaptive_weight_dice+=dice_loss(adaptive_weight_mask, target_masks, num_boxes)

                



        losses = {
            "loss_mask": focal,
            "loss_dice": dice,
            "sp_loss_mask":sp_focal,
            "sp_loss_dice":sp_dice,
            "corresponding_loss_mask":corresponding_focal,
            "corresponding_loss_dice":corresponding_dice,
            "adaptive_weight_loss_mask":adaptive_weight_focal,
            "adaptive_weight_loss_dice":adaptive_weight_dice,

        }

        return losses

    def loss_query_mask_fusion(self, outputs, targets, indices, num_boxes):
        """Compute only losses that update the query-wise fusion head."""
        adaptive_weight_focal = 0.0
        adaptive_weight_dice = 0.0
        for batch_idx in range(len(outputs["pred_masks"])):
            query_indices = indices[batch_idx][0]
            target_indices = indices[batch_idx][1]
            superpoint = outputs["superpoints"][batch_idx]
            target = targets[batch_idx]["masks"][target_indices].float()
            target_masks = scatter_mean(target, superpoint, dim=-1)
            target_masks = (target_masks > 0.5).float()

            all_text_masks = as_query_mask_logits(
                outputs["pred_masks"][batch_idx], "pred_masks"
            )
            all_query_masks = as_query_mask_logits(
                outputs["sp_pred_masks"][batch_idx], "sp_pred_masks"
            )
            query_indices = query_indices.to(
                device=all_text_masks.device, dtype=torch.long
            )
            text_masks = all_text_masks.index_select(0, query_indices)
            query_masks = all_query_masks.index_select(0, query_indices)
            fusion_weight = gather_query_fusion_weight(
                outputs["adaptive_weights"][batch_idx],
                query_indices,
                all_query_masks.shape[0],
                query_masks,
            )
            fused_masks = fuse_query_mask_logits(
                text_masks, query_masks, fusion_weight
            )
            adaptive_weight_focal += sigmoid_focal_loss(
                fused_masks, target_masks, num_boxes
            )
            adaptive_weight_dice += dice_loss(
                fused_masks, target_masks, num_boxes
            )
        return {
            "adaptive_weight_loss_mask": adaptive_weight_focal,
            "adaptive_weight_loss_dice": adaptive_weight_dice,
        }

    ############################
    # BRIEF semantic alignment #
    ############################
    def loss_sem_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1  # attention_mask
        positive_map[torch.arange(len(inds)), :, inds] = 0.5  
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5  # 将token seq
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0  # 从gt中得到真正的pmap

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        pron_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)
        rel_positive_map = torch.zeros(logits.shape, device=logits.device)
        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]   
        pmap_pron = torch.cat([
            t['pron_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_rel = torch.cat([
            t['rel_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        pron_positive_map[idx] = pmap_pron
        other_positive_map[idx] = pmap_other
        rel_positive_map[idx] = pmap_rel

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )   # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map>0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map>0), 0)
        pron_positive_logits = -logits.masked_fill(~(pron_positive_map>0), 0)
        rel_positive_logits = -logits.masked_fill(~(rel_positive_map>0), 0)

        pos_modi_term = modi_positive_logits.sum(2)
        pos_pron_term = pron_positive_logits.sum(2)
        pos_rel_term = rel_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map>0).sum(2) + 1e-6
        nb_pron_pos_token = (pron_positive_map>0).sum(2) + 1e-6
        nb_rel_pos_token = (rel_positive_map>0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits+other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token+1e-6) / nb_pos_token
        box_to_token_loss_ = (
            pos_term/nb_pos_token \
            + 0.2*pos_modi_term/nb_modi_pos_token \
            + 0.2*pos_pron_term/nb_pron_pos_token \
            + 0.1*pos_rel_term/nb_rel_pos_token \
            + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map + (modi_positive_map>0) + (pron_positive_map>0) + (rel_positive_map>0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map>0).any(1)] = 0.2
        tmask[(pron_positive_map>0).any(1)] = 0.2
        tmask[(rel_positive_map>0).any(1)] = 0.1
        tmask[torch.arange(len(inds)), inds-1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        pos_pron_term = pron_positive_logits.sum(1)
        pos_rel_term = rel_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term + pos_pron_term + pos_rel_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + pron_positive_map.sum(1) \
             + rel_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj+1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()   

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    # BRIEF get loss.
    def get_loss(self, loss, outputs, targets, indices, num_boxes, auxi_indices, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,      # box loss
            'masks': self.loss_masks,      # mask loss
            'labels': self.loss_pos_align, # position alignment
            'contrastive_align': self.loss_sem_align   # semantic alignment
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # STEP Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)

        # auxi object
        auxi_target = [
            {
                "labels": targets[b]["labels"],
                "boxes": targets[b]["auxi_box"],
                "positive_map": targets[b]["auxi_entity_positive_map"]
            }
            for b in range(outputs["pred_boxes"].shape[0])
        ]
        # auxi_indices = self.matcher(outputs, auxi_target)
        auxi_indices = None  # avoid bugs

        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes, auxi_indices
            ))

        return losses, indices

    def forward_query_mask_fusion(self, outputs, targets):
        """Match once and compute only query-fusion-head gradients."""
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(target_indices) for _, target_indices in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=outputs["pred_boxes"].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        losses = self.loss_query_mask_fusion(
            outputs, targets, indices, num_boxes
        )
        return losses, indices

# BRIEF loss
def resolve_source_moe_gate_loss_fallback_query(
        end_points, gate_default_query):
    """Resolve the query against which gate action utility is supervised."""
    return end_points.get(
        "moe_gate_supervision_fallback_query",
        end_points.get("moe_gate_action_anchor_query", gate_default_query),
    )


def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5,
                           source_choice_selector_loss_weight=0.0,
                           source_choice_selector_default_source="default",
                           source_choice_selector_choice_target="precision_gain_default_sourcewise_focal_bce",
                           source_choice_selector_min_iou_gap=0.05,
                           mask_loss_scale=1.0,
                           consistency_loss_scale=1.0,
                           source_moe_balance_loss_weight=0.0,
                           source_moe_rank_loss_weight=0.0,
                           source_moe_mask_rank_loss_weight=0.25,
                           source_moe_rank_temperature=0.1,
                           source_moe_anchor_loss_weight=0.0,
                           source_moe_anchor_margin=0.05,
                           source_moe_gate_loss_weight=0.0,
                           source_moe_gate_mask_loss_weight=0.25,
                           source_moe_gate_focal_gamma=2.0,
                           source_moe_gate_false_override_weight=2.0,
                           source_moe_gate_break_cost=2.0,
                           source_moe_gate_mask_utility_weight=0.25,
                           source_moe_gate_objective="balanced_focal",
                           source_moe_gate_setwise_temperature=0.0,
                           source_moe_gate_boundary_loss_weight=0.0,
                           joint_query_quality_loss_weight=0.0,
                           joint_query_quality_mask_weight=0.25,
                           joint_query_quality_temperature=0.25,
                           joint_query_quality_aux_loss_weight=1.0,
                           joint_query_quality_anchor_loss_weight=0.5,
                           joint_query_quality_anchor_margin=0.05,
                           joint_query_quality_source_mix_loss_weight=0.0,
                           joint_query_quality_source_mix_alignment_temperature=0.25,
                           joint_query_quality_source_mix_query_focus_weight=0.0,
                           joint_query_quality_candidate_mask_loss_weight=0.0,
                           joint_query_quality_candidate_lovasz_loss_weight=0.0,
                           joint_query_quality_candidate_mask_top_k=16,
                           query_mask_fusion_train_only=False,
                           joint_query_quality_train_only=False):
    """Compute Hungarian matching loss containing CE, bbox and giou."""
    for scale_name, scale in (
            ("mask_loss_scale", mask_loss_scale),
            ("source_moe_balance_loss_weight", source_moe_balance_loss_weight),
            ("source_moe_rank_loss_weight", source_moe_rank_loss_weight),
            ("source_moe_mask_rank_loss_weight",
             source_moe_mask_rank_loss_weight),
            ("source_moe_anchor_loss_weight",
             source_moe_anchor_loss_weight),
            ("source_moe_anchor_margin", source_moe_anchor_margin),
            ("source_moe_gate_loss_weight", source_moe_gate_loss_weight),
            ("source_moe_gate_mask_loss_weight",
             source_moe_gate_mask_loss_weight),
            ("source_moe_gate_focal_gamma",
             source_moe_gate_focal_gamma),
            ("source_moe_gate_mask_utility_weight",
             source_moe_gate_mask_utility_weight),
            ("source_moe_gate_setwise_temperature",
             source_moe_gate_setwise_temperature),
            ("source_moe_gate_boundary_loss_weight",
             source_moe_gate_boundary_loss_weight),
            ("joint_query_quality_loss_weight",
             joint_query_quality_loss_weight),
            ("joint_query_quality_mask_weight",
             joint_query_quality_mask_weight),
            ("joint_query_quality_temperature",
             joint_query_quality_temperature),
            ("joint_query_quality_aux_loss_weight",
             joint_query_quality_aux_loss_weight),
            ("joint_query_quality_anchor_loss_weight",
             joint_query_quality_anchor_loss_weight),
            ("joint_query_quality_anchor_margin",
             joint_query_quality_anchor_margin),
            ("joint_query_quality_source_mix_loss_weight",
             joint_query_quality_source_mix_loss_weight),
            ("joint_query_quality_candidate_mask_loss_weight",
             joint_query_quality_candidate_mask_loss_weight),
            ("joint_query_quality_candidate_lovasz_loss_weight",
             joint_query_quality_candidate_lovasz_loss_weight),
            ("consistency_loss_scale", consistency_loss_scale)):
        try:
            is_valid = math.isfinite(scale) and scale >= 0
        except (TypeError, ValueError, OverflowError):
            is_valid = False
        if not is_valid:
            raise ValueError(
                f"{scale_name} must be a finite non-negative number"
            )
    if (not isinstance(joint_query_quality_candidate_mask_top_k, int)
            or isinstance(joint_query_quality_candidate_mask_top_k, bool)
            or joint_query_quality_candidate_mask_top_k <= 0):
        raise ValueError(
            "joint_query_quality_candidate_mask_top_k must be positive"
        )
    candidate_mask_objective_enabled = (
        joint_query_quality_candidate_mask_loss_weight > 0
        or joint_query_quality_candidate_lovasz_loss_weight > 0
    )
    if (candidate_mask_objective_enabled
            and joint_query_quality_loss_weight <= 0):
        raise ValueError(
            "candidate mask supervision requires joint query quality loss"
        )
    try:
        valid_rank_temperature = (
            math.isfinite(source_moe_rank_temperature)
            and source_moe_rank_temperature > 0
        )
    except (TypeError, ValueError, OverflowError):
        valid_rank_temperature = False
    if not valid_rank_temperature:
        raise ValueError(
            "source_moe_rank_temperature must be a finite positive number"
        )
    try:
        valid_false_override_weight = (
            math.isfinite(source_moe_gate_false_override_weight)
            and source_moe_gate_false_override_weight >= 1.0
        )
    except (TypeError, ValueError, OverflowError):
        valid_false_override_weight = False
    if not valid_false_override_weight:
        raise ValueError(
            "source_moe_gate_false_override_weight must be finite and at "
            "least one"
        )
    try:
        valid_break_cost = (
            math.isfinite(source_moe_gate_break_cost)
            and source_moe_gate_break_cost >= 1.0
        )
    except (TypeError, ValueError, OverflowError):
        valid_break_cost = False
    if not valid_break_cost:
        raise ValueError(
            "source_moe_gate_break_cost must be finite and at least one"
        )
    if source_moe_gate_objective not in (
            "balanced_focal", "calibrated_utility",
            "balanced_calibrated_utility",
            "hierarchical_risk_calibrated",
            "pairwise_risk_calibrated",
            "topn_risk_calibrated", "topn_dual_risk_calibrated",
            "topn_absolute_quality_calibrated",
            "cascade_absolute_quality_calibrated",
            "cascade_opportunity_balanced_calibrated",
            "cascade_opportunity_verified_calibrated",
            "cascade_joint_risk_calibrated",
            "cascade_v19_fallback_set_risk_calibrated",
            "cascade_v19_rich_set_empirical_risk",
            "cascade_v23_dense_quality_risk",
            "cascade_v24_relative_risk",
            "cascade_v25_pairwise_calibrated_risk",
            "cascade_v26_prior_restored_pairwise_risk",
            "cascade_v27_uncertainty_quality_risk",
            "cascade_v28_selected_abstention_risk",
            "cascade_v29_counterfactual_selected_risk",
            "cascade_v37_counterfactual_benefit_hazard_risk",
            "cascade_v38_complementary_logodds_risk",
            "cascade_v39_hazard_residual_risk"):
        raise ValueError(
            "source_moe_gate_objective must be balanced_focal, "
            "calibrated_utility, balanced_calibrated_utility, or "
            "hierarchical_risk_calibrated, pairwise_risk_calibrated, or "
            "topn_risk_calibrated, topn_dual_risk_calibrated, "
            "topn_absolute_quality_calibrated, or "
            "cascade_absolute_quality_calibrated, or "
            "cascade_opportunity_balanced_calibrated, or "
            "cascade_opportunity_verified_calibrated, or "
            "cascade_joint_risk_calibrated, or "
            "cascade_v19_fallback_set_risk_calibrated, or "
            "cascade_v19_rich_set_empirical_risk, or "
            "cascade_v23_dense_quality_risk, cascade_v24_relative_risk, or "
            "cascade_v25_pairwise_calibrated_risk, or "
            "cascade_v26_prior_restored_pairwise_risk, or "
            "cascade_v27_uncertainty_quality_risk, or "
            "cascade_v28_selected_abstention_risk, or "
            "cascade_v29_counterfactual_selected_risk, or "
            "cascade_v37_counterfactual_benefit_hazard_risk, or "
            "cascade_v38_complementary_logodds_risk, or "
            "cascade_v39_hazard_residual_risk"
        )

    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes     # 6+1: 'proposal_'  'last_' '0head_'  '1head_'  '2head_'  '3head_'  '4head_'
    is_multi_mask = "proposal_pred_masks" in end_points

    # STEP target GT box
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_labels = end_points['sem_cls_label']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)
    gt_masks = end_points['gt_masks']
    # text
    positive_map = end_points['positive_map']               # main obj.
    modify_positive_map = end_points['modify_positive_map'] # attribute(modify)
    pron_positive_map = end_points['pron_positive_map']     # pron
    other_entity_map = end_points['other_entity_map']       # other(auxi)
    rel_positive_map = end_points['rel_positive_map']       # relation
    box_label_mask = end_points['box_label_mask']           # (132,) target object mask
    auxi_entity_positive_map = end_points['auxi_entity_positive_map']
    auxi_box = end_points['auxi_box']
    source_choice_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_rank_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_box_rank_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_mask_rank_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_anchor_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_gate_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_gate_box_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_gate_mask_loss = torch.tensor(0.0, device=gt_bbox.device)
    source_moe_gate_decision_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_loss = torch.tensor(0.0, device=gt_bbox.device)
    joint_query_quality_listwise_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_aux_loss = torch.tensor(0.0, device=gt_bbox.device)
    joint_query_quality_anchor_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_source_mix_alignment_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_candidate_mask_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_candidate_dice_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_candidate_lovasz_loss = torch.tensor(
        0.0, device=gt_bbox.device
    )
    joint_query_quality_candidate_mask_query_ratio = torch.tensor(
        0.0, device=gt_bbox.device
    )

    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],
            "masks": gt_masks[b, box_label_mask[b].bool()],
            "positive_map": positive_map[b, box_label_mask[b].bool()],
            "modify_positive_map": modify_positive_map[b, box_label_mask[b].bool()],
            "pron_positive_map": pron_positive_map[b, box_label_mask[b].bool()],
            "other_entity_map": other_entity_map[b, box_label_mask[b].bool()],
            "rel_positive_map": rel_positive_map[b, box_label_mask[b].bool()],
            "auxi_entity_positive_map": auxi_entity_positive_map[b, 0].unsqueeze(0),
            "auxi_box": auxi_box[b]
        }
        for b in range(gt_labels.shape[0])
    ]

    if query_mask_fusion_train_only and joint_query_quality_train_only:
        raise ValueError(
            "query-mask-fusion-only and joint-query-quality-only modes are "
            "mutually exclusive"
        )

    if joint_query_quality_train_only:
        if joint_query_quality_loss_weight <= 0:
            raise ValueError(
                "joint-query-quality-only mode requires a positive loss "
                "weight"
            )
        required = (
            "last_center", "last_pred_size", "last_pred_masks",
            "sp_last_pred_masks", "adaptive_weights", "superpoints",
            "selected_source_scores", "moe_shared_source",
        )
        missing = [key for key in required if key not in end_points]
        if missing:
            raise ValueError(
                "joint-query-quality-only inputs are missing: "
                + ", ".join(missing)
            )
        joint_keys = (
            "scores", "baseline_indices", "selected_indices",
            "box_logits", "box_iou", "mask_logits", "mask_iou",
            "valid_mask",
        )
        missing_joint = [
            key for key in joint_keys
            if "joint_query_quality_{}".format(key) not in end_points
        ]
        if missing_joint:
            raise ValueError(
                "joint query quality loss is enabled but outputs are "
                "missing: " + ", ".join(missing_joint)
            )
        joint_outputs = {
            key: end_points["joint_query_quality_{}".format(key)]
            for key in joint_keys
        }
        source_mix_keys = (
            "source_mix_weights", "source_mix_ranks",
            "source_mix_validity",
        )
        for key in source_mix_keys:
            endpoint_key = "joint_query_quality_{}".format(key)
            if endpoint_key in end_points:
                joint_outputs[key] = end_points[endpoint_key]
        scores = joint_outputs["scores"]
        valid_mask = joint_outputs["valid_mask"]
        if (not isinstance(scores, torch.Tensor) or scores.dim() != 2
                or not isinstance(valid_mask, torch.Tensor)
                or valid_mask.dtype != torch.bool
                or valid_mask.shape != scores.shape):
            raise ValueError(
                "joint query quality scores and valid mask must align as "
                "[B,Q]"
            )
        moe_valid_mask = end_points.get("moe_valid_mask")
        if moe_valid_mask is not None and not torch.equal(
                moe_valid_mask, valid_mask):
            raise ValueError(
                "joint query quality and source-arbiter valid masks disagree"
            )
        candidate_boxes = torch.cat((
            end_points["last_center"],
            end_points["last_pred_size"].clamp(min=1e-6),
        ), dim=-1)
        if candidate_boxes.shape[:2] != scores.shape:
            raise ValueError(
                "joint query quality scores must align with last-layer boxes"
            )
        grounding_gt_valid = torch.zeros_like(
            box_label_mask, dtype=torch.bool
        )
        grounding_gt_valid[:, 0] = box_label_mask[:, 0].bool()
        if not bool(grounding_gt_valid[:, 0].all().item()):
            raise ValueError(
                "joint query quality requires the root GT in slot zero"
            )
        sample_mask = build_source_moe_grounding_sample_mask(
            end_points, scores.shape[0], scores.device
        )
        end_points["source_moe_supervised_sample_ratio"] = (
            sample_mask.float().mean().detach()
        )
        with torch.no_grad():
            box_ious = compute_query_box_ious(
                candidate_boxes.detach(), gt_bbox.detach(),
                grounding_gt_valid,
            )
            fused_mask_logits = build_fused_query_mask_logits(
                end_points["last_pred_masks"],
                end_points["sp_last_pred_masks"],
                end_points["adaptive_weights"],
            )
            mask_ious = compute_query_mask_ious(
                fused_mask_logits, gt_masks.detach(),
                end_points["superpoints"], grounding_gt_valid,
            )
        joint_supervision = compute_joint_query_quality_loss(
            joint_outputs,
            box_ious,
            mask_ious,
            sample_mask=sample_mask,
            temperature=float(joint_query_quality_temperature),
            mask_weight=float(joint_query_quality_mask_weight),
            quality_loss_weight=float(
                joint_query_quality_aux_loss_weight
            ),
            anchor_loss_weight=float(
                joint_query_quality_anchor_loss_weight
            ),
            anchor_margin=float(joint_query_quality_anchor_margin),
            source_mix_loss_weight=float(
                joint_query_quality_source_mix_loss_weight
            ),
            source_mix_alignment_temperature=float(
                joint_query_quality_source_mix_alignment_temperature
            ),
            source_mix_query_focus_weight=float(
                joint_query_quality_source_mix_query_focus_weight
            ),
        )
        joint_query_quality_source_mix_alignment_loss = joint_supervision[
            "source_mix_alignment_loss"
        ]
        mask_calibration_enabled = (
            "joint_query_quality_mask_fusion_weights" in end_points
            and "joint_query_quality_mask_logit_bias" in end_points
        )
        calibration_mask_loss = joint_supervision["loss"] * 0.0
        calibration_dice_loss = joint_supervision["loss"] * 0.0
        if mask_calibration_enabled:
            if set_criterion is None:
                raise ValueError(
                    "joint query mask calibration requires SetCriterion"
                )
            calibration_output = {
                "pred_logits": end_points["last_sem_cls_scores"],
                "pred_boxes": candidate_boxes,
                "pred_masks": end_points["last_pred_masks"],
                "sp_pred_masks": end_points["sp_last_pred_masks"],
                "adaptive_weights": end_points["adaptive_weights"],
                "superpoints": end_points["superpoints"],
                "language_dataset": end_points["language_dataset"],
            }
            calibration_losses, _ = (
                set_criterion.forward_query_mask_fusion(
                    calibration_output, target
                )
            )
            calibration_mask_loss = calibration_losses[
                "adaptive_weight_loss_mask"
            ]
            calibration_dice_loss = calibration_losses[
                "adaptive_weight_loss_dice"
            ]
        if candidate_mask_objective_enabled:
            if not mask_calibration_enabled:
                raise ValueError(
                    "candidate mask supervision requires mask calibration"
                )
            candidate_mask = build_joint_query_mask_candidate_mask(
                scores,
                box_ious,
                valid_mask,
                sample_mask,
                joint_query_quality_candidate_mask_top_k,
            )
            candidate_losses = compute_joint_query_mask_candidate_loss(
                end_points["last_pred_masks"],
                end_points["sp_last_pred_masks"],
                end_points["adaptive_weights"],
                gt_masks,
                end_points["superpoints"],
                candidate_mask,
                compute_lovasz=(
                    joint_query_quality_candidate_lovasz_loss_weight > 0
                ),
            )
            joint_query_quality_candidate_mask_loss = candidate_losses[
                "mask_loss"
            ]
            joint_query_quality_candidate_dice_loss = candidate_losses[
                "dice_loss"
            ]
            joint_query_quality_candidate_lovasz_loss = candidate_losses[
                "lovasz_loss"
            ]
            supervised_queries = (
                valid_mask & sample_mask.unsqueeze(1)
            ).float().sum().clamp(min=1.0)
            joint_query_quality_candidate_mask_query_ratio = (
                candidate_mask.float().sum() / supervised_queries
            ).detach()
        loss = (
            float(joint_query_quality_loss_weight)
            * joint_supervision["loss"]
            + float(mask_loss_scale) * (
                10.0 * calibration_mask_loss
                + 2.0 * calibration_dice_loss
                + float(joint_query_quality_candidate_mask_loss_weight) * (
                    10.0 * joint_query_quality_candidate_mask_loss
                    + 2.0 * joint_query_quality_candidate_dice_loss
                )
                + float(joint_query_quality_candidate_lovasz_loss_weight)
                * joint_query_quality_candidate_lovasz_loss
            )
        ).reshape(())
        zero = loss * 0.0
        zero_keys = (
            "loss_ce", "loss_bbox", "loss_giou",
            "query_points_generation_loss", "loss_sem_align",
            "loss_mask", "loss_dice", "sp_loss_mask", "sp_loss_dice",
            "corresponding_loss_mask", "corresponding_loss_dice",
            "source_choice_loss", "moe_balance_loss",
            "source_moe_rank_loss", "source_moe_box_rank_loss",
            "source_moe_mask_rank_loss", "source_moe_anchor_loss",
            "source_moe_gate_loss", "source_moe_gate_box_loss",
            "source_moe_gate_mask_loss", "source_moe_gate_decision_loss",
        )
        for key in zero_keys:
            end_points[key] = zero
        end_points["adaptive_weight_loss_mask"] = calibration_mask_loss
        end_points["adaptive_weight_loss_dice"] = calibration_dice_loss
        end_points["joint_query_quality_loss"] = joint_supervision["loss"]
        end_points["joint_query_quality_listwise_loss"] = (
            joint_supervision["listwise_loss"]
        )
        end_points["joint_query_quality_aux_loss"] = (
            joint_supervision["quality_loss"]
        )
        end_points["joint_query_quality_anchor_loss"] = (
            joint_supervision["anchor_loss"]
        )
        end_points["joint_query_quality_source_mix_alignment_loss"] = (
            joint_query_quality_source_mix_alignment_loss
        )
        end_points["joint_query_quality_candidate_mask_loss"] = (
            joint_query_quality_candidate_mask_loss
        )
        end_points["joint_query_quality_candidate_dice_loss"] = (
            joint_query_quality_candidate_dice_loss
        )
        end_points["joint_query_quality_candidate_lovasz_loss"] = (
            joint_query_quality_candidate_lovasz_loss
        )
        end_points["joint_query_quality_candidate_mask_query_ratio"] = (
            joint_query_quality_candidate_mask_query_ratio
        )
        for key, value in joint_supervision["stats"].items():
            end_points["joint_query_quality_{}".format(key)] = value
        end_points["loss"] = loss
        return loss, end_points

    if query_mask_fusion_train_only:
        output = {
            "pred_logits": end_points["last_sem_cls_scores"],
            "pred_boxes": torch.cat((
                end_points["last_center"],
                end_points["last_pred_size"],
            ), dim=-1),
            "pred_masks": end_points["last_pred_masks"],
            "sp_pred_masks": end_points["sp_last_pred_masks"],
            "adaptive_weights": end_points["adaptive_weights"],
            "superpoints": end_points["superpoints"],
            "language_dataset": end_points["language_dataset"],
        }
        fusion_losses, _ = set_criterion.forward_query_mask_fusion(
            output, target
        )
        adaptive_weight_loss_mask = fusion_losses[
            "adaptive_weight_loss_mask"
        ]
        adaptive_weight_loss_dice = fusion_losses[
            "adaptive_weight_loss_dice"
        ]
        zero = adaptive_weight_loss_mask * 0.0
        loss = mask_loss_scale * (
            10 * adaptive_weight_loss_mask
            + 2 * adaptive_weight_loss_dice
        )
        if loss.numel() == 1:
            loss = loss.reshape(())
        zero_keys = (
            "loss_ce", "loss_bbox", "loss_giou",
            "query_points_generation_loss", "loss_sem_align",
            "loss_mask", "loss_dice", "sp_loss_mask", "sp_loss_dice",
            "corresponding_loss_mask", "corresponding_loss_dice",
            "source_choice_loss", "moe_balance_loss",
            "source_moe_rank_loss", "source_moe_box_rank_loss",
            "source_moe_mask_rank_loss", "source_moe_anchor_loss",
            "source_moe_gate_loss", "source_moe_gate_box_loss",
            "source_moe_gate_mask_loss", "source_moe_gate_decision_loss",
        )
        for key in zero_keys:
            end_points[key] = zero
        end_points["adaptive_weight_loss_mask"] = (
            adaptive_weight_loss_mask
        )
        end_points["adaptive_weight_loss_dice"] = (
            adaptive_weight_loss_dice
        )
        end_points["loss"] = loss
        return loss, end_points

    loss_ce, loss_bbox, loss_giou, loss_sem_align, loss_mask, loss_dice, sp_loss_mask,sp_loss_dice,corresponding_loss_mask,adaptive_weight_loss_mask, adaptive_weight_loss_dice,corresponding_loss_dice= 0, 0, 0, 0, 0, 0 ,0 ,0,0,0,0,0
    #loss_ce, loss_bbox, loss_giou, loss_sem_align, loss_mask, loss_dice = 0, 0, 0, 0, 0, 0 
    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']           
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']

        # STEP Get predicted boxes and labels
        pred_center = end_points[f'{prefix}center']     # B, K, 3
        pred_size = end_points[f'{prefix}pred_size']    # (B,K,3) (l,w,h)
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']     # (B, Q, n_class)
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox
        output["superpoints"] = end_points["superpoints"]
        output["language_dataset"] = end_points["language_dataset"] # dataset
        if is_multi_mask:
            output["pred_masks"] = end_points[f"{prefix}pred_masks"]
        else:
            if prefix == 'last_':
                output["pred_masks"] = end_points["last_pred_masks"]
                output["sp_pred_masks"]=end_points["sp_last_pred_masks"]
                output["adaptive_weights"]=end_points['adaptive_weights']
                output['super_xyz_list']=end_points['super_xyz_list']

        # NOTE Compute all the requested losses, forward
        losses, _ = set_criterion(output, target)
        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        loss_mask += losses.get('loss_mask')
        loss_dice += losses.get('loss_dice')
        sp_loss_mask+=losses.get('sp_loss_mask')
        sp_loss_dice+=losses.get('sp_loss_dice')
        corresponding_loss_mask+=losses.get('corresponding_loss_mask')
        corresponding_loss_dice+=losses.get('corresponding_loss_dice')
        adaptive_weight_loss_mask+= losses.get('adaptive_weight_loss_mask')
        adaptive_weight_loss_dice+=losses.get('adaptive_weight_loss_dice')
        
        if 'proj_tokens' in end_points:
            loss_sem_align += losses['loss_sem_align']

    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    if (
            source_choice_selector_loss_weight > 0
            and "selector_choice_scores" in end_points
            and "source_choice_source_scores" in end_points):
        source_names = end_points.get(
            "selector_choice_source_names",
            list(end_points["source_choice_source_scores"].keys()),
        )
        pred_bbox = torch.cat(
            [end_points["last_center"], end_points["last_pred_size"].clamp(min=1e-6)],
            dim=-1,
        )
        source_ious = compute_source_top1_ious(
            candidate_boxes=pred_bbox.detach(),
            source_scores=end_points["source_choice_source_scores"],
            source_names=source_names,
            gt_boxes=gt_bbox.detach(),
            gt_mask=box_label_mask.detach(),
        )
        source_choice_loss, source_choice_stats = compute_source_choice_loss(
            choice_scores=end_points["selector_choice_scores"],
            source_ious=source_ious,
            source_names=source_names,
            default_source=source_choice_selector_default_source,
            target_mode=source_choice_selector_choice_target,
            min_iou_gap=source_choice_selector_min_iou_gap,
        )
        for key, value in source_choice_stats.items():
            end_points[key] = value

    if (
            (source_moe_rank_loss_weight > 0
             or source_moe_gate_loss_weight > 0
             or joint_query_quality_loss_weight > 0)
            and "selected_source_scores" in end_points
            and "moe_shared_source" in end_points):
        selected_moe_scores = end_points["selected_source_scores"]
        moe_scores = end_points.get(
            "moe_candidate_scores", selected_moe_scores
        )
        if (not isinstance(moe_scores, torch.Tensor)
                or moe_scores.shape != end_points["last_center"].shape[:2]):
            raise ValueError("MoE query scores must align with last-layer boxes")
        if (not isinstance(selected_moe_scores, torch.Tensor)
                or selected_moe_scores.shape != moe_scores.shape):
            raise ValueError("selected MoE scores must align with candidates")
        moe_candidate_boxes = torch.cat((
            end_points["last_center"],
            end_points["last_pred_size"].clamp(min=1e-6),
        ), dim=-1)
        grounding_gt_valid = torch.zeros_like(box_label_mask, dtype=torch.bool)
        grounding_gt_valid[:, 0] = box_label_mask[:, 0].bool()
        if not bool(grounding_gt_valid[:, 0].all().item()):
            raise ValueError("MoE ranking requires the root GT in slot zero")

        sample_mask = build_source_moe_grounding_sample_mask(
            end_points, moe_scores.shape[0], moe_scores.device
        )
        end_points["source_moe_supervised_sample_ratio"] = (
            sample_mask.float().mean().detach()
        )

        query_mask_logits = None
        if all(key in end_points for key in (
                "last_pred_masks", "sp_last_pred_masks",
                "adaptive_weights", "superpoints")):
            query_mask_logits = build_fused_query_mask_logits(
                end_points["last_pred_masks"],
                end_points["sp_last_pred_masks"],
                end_points["adaptive_weights"],
            )
        shared_name = end_points["moe_shared_source"]
        source_scores = end_points.get("source_choice_source_scores", {})
        if shared_name not in source_scores:
            raise ValueError("MoE shared source scores are unavailable")
        shared_query = end_points.get("moe_shared_query")
        if shared_query is None:
            shared_query = source_scores[shared_name].argmax(dim=1)
        elif (not isinstance(shared_query, torch.Tensor)
              or shared_query.dtype != torch.long
              or shared_query.shape != (moe_scores.shape[0],)
              or shared_query.device != moe_scores.device):
            raise ValueError(
                "MoE shared query must be int64 with shape [B]"
            )
        row_index = torch.arange(
            moe_scores.shape[0], device=moe_scores.device
        )
        moe_valid_mask = end_points.get("moe_valid_mask")
        if moe_valid_mask is None:
            moe_valid_mask = torch.ones_like(moe_scores, dtype=torch.bool)
        elif (not isinstance(moe_valid_mask, torch.Tensor)
              or moe_valid_mask.dtype != torch.bool
              or moe_valid_mask.shape != moe_scores.shape
              or moe_valid_mask.device != moe_scores.device):
            raise ValueError("MoE valid mask must be bool with shape [B,Q]")
        if all(key in end_points for key in (
                "moe_router_probs", "moe_expert_mask")):
            end_points["moe_balance_loss"] = compute_load_balance_loss(
                end_points["moe_router_probs"],
                end_points["moe_expert_mask"],
                valid_mask=moe_valid_mask & sample_mask.unsqueeze(1),
            )
        if (bool((shared_query < 0).any().item())
                or bool((shared_query >= moe_scores.shape[1]).any().item())
                or not bool(moe_valid_mask[
                    row_index, shared_query
                ].all().item())):
            raise ValueError("MoE shared query must identify valid candidates")
        ranking = compute_source_moe_ranking_loss(
            scores=moe_scores,
            candidate_boxes=moe_candidate_boxes,
            gt_boxes=gt_bbox,
            gt_valid=grounding_gt_valid,
            valid_mask=moe_valid_mask,
            sample_mask=sample_mask,
            query_mask_logits=query_mask_logits,
            gt_point_masks=gt_masks if query_mask_logits is not None else None,
            superpoints=(
                end_points.get("superpoints")
                if query_mask_logits is not None else None
            ),
            temperature=float(source_moe_rank_temperature),
            mask_loss_weight=float(source_moe_mask_rank_loss_weight),
            anchor_indices=shared_query,
            anchor_loss_weight=float(source_moe_anchor_loss_weight),
            anchor_margin=float(source_moe_anchor_margin),
        )
        source_moe_rank_loss = ranking["loss"]
        source_moe_box_rank_loss = ranking["box_loss"]
        source_moe_mask_rank_loss = ranking["mask_loss"]
        source_moe_anchor_loss = ranking["anchor_loss"]

        if joint_query_quality_loss_weight > 0:
            joint_keys = (
                "scores", "baseline_indices", "selected_indices",
                "box_logits", "box_iou", "mask_logits", "mask_iou",
                "valid_mask",
            )
            missing_joint_keys = [
                key for key in joint_keys
                if "joint_query_quality_{}".format(key) not in end_points
            ]
            if missing_joint_keys:
                raise ValueError(
                    "joint query quality loss is enabled but outputs are "
                    "missing: " + ", ".join(missing_joint_keys)
                )
            if ranking["mask_ious"] is None:
                raise ValueError(
                    "joint query quality training requires query mask IoUs"
                )
            joint_outputs = {
                key: end_points[
                    "joint_query_quality_{}".format(key)
                ]
                for key in joint_keys
            }
            source_mix_keys = (
                "source_mix_weights", "source_mix_ranks",
                "source_mix_validity",
            )
            for key in source_mix_keys:
                endpoint_key = "joint_query_quality_{}".format(key)
                if endpoint_key in end_points:
                    joint_outputs[key] = end_points[endpoint_key]
            joint_supervision = compute_joint_query_quality_loss(
                joint_outputs,
                ranking["box_ious"],
                ranking["mask_ious"],
                sample_mask=sample_mask,
                temperature=float(joint_query_quality_temperature),
                mask_weight=float(joint_query_quality_mask_weight),
                quality_loss_weight=float(
                    joint_query_quality_aux_loss_weight
                ),
                anchor_loss_weight=float(
                    joint_query_quality_anchor_loss_weight
                ),
                anchor_margin=float(joint_query_quality_anchor_margin),
                source_mix_loss_weight=float(
                    joint_query_quality_source_mix_loss_weight
                ),
                source_mix_alignment_temperature=float(
                    joint_query_quality_source_mix_alignment_temperature
                ),
                source_mix_query_focus_weight=float(
                    joint_query_quality_source_mix_query_focus_weight
                ),
            )
            joint_query_quality_loss = joint_supervision["loss"]
            joint_query_quality_listwise_loss = joint_supervision[
                "listwise_loss"
            ]
            joint_query_quality_aux_loss = joint_supervision[
                "quality_loss"
            ]
            joint_query_quality_anchor_loss = joint_supervision[
                "anchor_loss"
            ]
            joint_query_quality_source_mix_alignment_loss = (
                joint_supervision["source_mix_alignment_loss"]
            )
            for key, value in joint_supervision["stats"].items():
                end_points[
                    "joint_query_quality_{}".format(key)
                ] = value
            if candidate_mask_objective_enabled:
                mask_calibration_enabled = (
                    "joint_query_quality_mask_fusion_weights" in end_points
                    and "joint_query_quality_mask_logit_bias" in end_points
                )
                if not mask_calibration_enabled:
                    raise ValueError(
                        "candidate mask supervision requires mask calibration"
                    )
                candidate_mask = build_joint_query_mask_candidate_mask(
                    joint_outputs["scores"],
                    ranking["box_ious"],
                    moe_valid_mask,
                    sample_mask,
                    joint_query_quality_candidate_mask_top_k,
                )
                candidate_losses = compute_joint_query_mask_candidate_loss(
                    end_points["last_pred_masks"],
                    end_points["sp_last_pred_masks"],
                    end_points["adaptive_weights"],
                    gt_masks,
                    end_points["superpoints"],
                    candidate_mask,
                    compute_lovasz=(
                        joint_query_quality_candidate_lovasz_loss_weight > 0
                    ),
                )
                joint_query_quality_candidate_mask_loss = candidate_losses[
                    "mask_loss"
                ]
                joint_query_quality_candidate_dice_loss = candidate_losses[
                    "dice_loss"
                ]
                joint_query_quality_candidate_lovasz_loss = candidate_losses[
                    "lovasz_loss"
                ]
                supervised_queries = (
                    moe_valid_mask & sample_mask.unsqueeze(1)
                ).float().sum().clamp(min=1.0)
                joint_query_quality_candidate_mask_query_ratio = (
                    candidate_mask.float().sum() / supervised_queries
                ).detach()

        if source_moe_gate_loss_weight > 0:
            gate_keys = (
                "moe_gate_box_logits",
                "moe_gate_mask_logits",
                "moe_gate_decision_logits",
                "moe_gate_candidate_mask",
                "moe_gate_default_query",
            )
            missing_gate_keys = [
                key for key in gate_keys if key not in end_points
            ]
            if missing_gate_keys:
                raise ValueError(
                    "fallback gate loss is enabled but outputs are missing: "
                    + ", ".join(missing_gate_keys)
                )
            gate_default_query = end_points["moe_gate_default_query"]
            if not torch.equal(gate_default_query, shared_query):
                raise ValueError(
                    "fallback gate and shared source disagree on default query"
                )
            gate_action_anchor = resolve_source_moe_gate_loss_fallback_query(
                end_points, gate_default_query
            )
            if (not isinstance(gate_action_anchor, torch.Tensor)
                    or gate_action_anchor.dtype != torch.long
                    or gate_action_anchor.shape != gate_default_query.shape
                    or gate_action_anchor.device != gate_default_query.device
                    or bool((gate_action_anchor < 0).any().item())
                    or bool((gate_action_anchor >= moe_scores.shape[1]).any().item())
                    or not bool(moe_valid_mask[
                        row_index, gate_action_anchor
                    ].all().item())):
                raise ValueError(
                    "fallback gate loss fallback must identify valid queries"
                )
            gate_supervision = compute_source_moe_fallback_gate_loss(
                box_logits=end_points["moe_gate_box_logits"],
                decision_logits=end_points["moe_gate_decision_logits"],
                box_ious=ranking["box_ious"],
                default_indices=gate_action_anchor,
                candidate_mask=end_points["moe_gate_candidate_mask"],
                sample_mask=sample_mask,
                mask_logits=(
                    end_points["moe_gate_mask_logits"]
                    if ranking["mask_ious"] is not None else None
                ),
                mask_ious=ranking["mask_ious"],
                mask_loss_weight=float(source_moe_gate_mask_loss_weight),
                focal_gamma=float(source_moe_gate_focal_gamma),
                false_override_weight=float(
                    source_moe_gate_false_override_weight
                ),
                break_cost=float(source_moe_gate_break_cost),
                mask_utility_weight=float(
                    source_moe_gate_mask_utility_weight
                ),
                objective=source_moe_gate_objective,
                setwise_temperature=source_moe_gate_setwise_temperature,
                boundary_loss_weight=(
                    source_moe_gate_boundary_loss_weight
                ),
                action_margin=end_points.get("moe_gate_action_margin"),
                row_switch_margin=end_points.get(
                    "moe_gate_row_switch_margin"
                ),
                row_benefit_margin=end_points.get(
                    "moe_gate_row_benefit_margin"
                ),
                row_safety_margin=end_points.get(
                    "moe_gate_row_safety_margin"
                ),
                joint_action_margin=end_points.get(
                    "moe_gate_joint_action_margin"
                ),
                pairwise_utility_margin=end_points.get(
                    "moe_gate_pairwise_utility_margin"
                ),
                candidate_selection_margin=end_points.get(
                    "moe_gate_candidate_selection_margin"
                ),
                selected_abstention_margin=end_points.get(
                    "moe_gate_selected_abstention_margin"
                ),
                counterfactual_risk_margin=end_points.get(
                    "moe_gate_counterfactual_risk_margin"
                ),
                counterfactual_benefit_margin=end_points.get(
                    "moe_gate_counterfactual_benefit_margin"
                ),
                counterfactual_hazard_margin=end_points.get(
                    "moe_gate_counterfactual_hazard_margin"
                ),
                absolute_box_logits=end_points.get(
                    "moe_gate_absolute_box_logits"
                ),
                absolute_box_iou=end_points.get(
                    "moe_gate_absolute_box_iou"
                ),
                absolute_mask_logits=end_points.get(
                    "moe_gate_absolute_mask_logits"
                ),
                absolute_mask_iou=end_points.get(
                    "moe_gate_absolute_mask_iou"
                ),
            )
            source_moe_gate_loss = gate_supervision["loss"]
            source_moe_gate_box_loss = gate_supervision["box_loss"]
            source_moe_gate_mask_loss = gate_supervision["mask_loss"]
            source_moe_gate_decision_loss = gate_supervision[
                "decision_loss"
            ]
            end_points.update(gate_supervision["stats"])

        with torch.no_grad():
            active = sample_mask
            selected_query = selected_moe_scores.argmax(dim=1)
            candidate_query = moe_scores.argmax(dim=1)
            selected_box_iou = ranking["box_ious"][row_index, selected_query]
            candidate_box_iou = ranking["box_ious"][row_index, candidate_query]
            shared_box_iou = ranking["box_ious"][row_index, shared_query]
            for threshold, suffix in ((0.25, "025"), (0.50, "050")):
                end_points["source_moe_selected_acc{}".format(suffix)] = (
                    (selected_box_iou[active] > threshold).float().mean()
                    if bool(active.any().item())
                    else selected_box_iou.sum() * 0.0
                )
                shared_ok = shared_box_iou > threshold
                selected_ok = selected_box_iou > threshold
                end_points["source_moe_fix{}_ratio".format(suffix)] = (
                    ((~shared_ok[active]) & selected_ok[active]).float().mean()
                    if bool(active.any().item())
                    else selected_box_iou.sum() * 0.0
                )
                end_points["source_moe_break{}_ratio".format(suffix)] = (
                    (shared_ok[active] & (~selected_ok[active])).float().mean()
                    if bool(active.any().item())
                    else selected_box_iou.sum() * 0.0
                )
                candidate_ok = candidate_box_iou > threshold
                end_points[
                    "source_moe_candidate_acc{}".format(suffix)
                ] = (
                    candidate_ok[active].float().mean()
                    if bool(active.any().item())
                    else candidate_box_iou.sum() * 0.0
                )
                end_points[
                    "source_moe_candidate_fix{}_ratio".format(suffix)
                ] = (
                    ((~shared_ok[active]) & candidate_ok[active]).float().mean()
                    if bool(active.any().item())
                    else candidate_box_iou.sum() * 0.0
                )
                end_points[
                    "source_moe_candidate_break{}_ratio".format(suffix)
                ] = (
                    (shared_ok[active] & (~candidate_ok[active])).float().mean()
                    if bool(active.any().item())
                    else candidate_box_iou.sum() * 0.0
                )
            if ranking["mask_ious"] is not None:
                selected_mask_iou = ranking["mask_ious"][
                    row_index, selected_query
                ]
                for threshold, suffix in ((0.25, "025"), (0.50, "050")):
                    end_points[
                        "source_moe_selected_mask_acc{}".format(suffix)
                    ] = (
                        (selected_mask_iou[active] > threshold).float().mean()
                        if bool(active.any().item())
                        else selected_mask_iou.sum() * 0.0
                    )

    # total loss
    weight = 1
    if end_points["language_dataset"][0] == "scanrefer":
        weight = 0.5
    supervised_mask_loss = (
        10 * loss_mask
        + 2 * loss_dice
        + 5 * sp_loss_mask
        + sp_loss_dice
        + 10 * adaptive_weight_loss_mask
        + 2 * adaptive_weight_loss_dice
    )
    corresponding_consistency_loss = (
        10 * corresponding_loss_mask
        + 2 * corresponding_loss_dice
    )
    moe_balance_loss = end_points.get("moe_balance_loss")
    if not isinstance(moe_balance_loss, torch.Tensor):
        moe_balance_loss = torch.tensor(0.0, device=gt_bbox.device)
    loss = (
        8 * query_points_generation_loss
        + 1.0 / (num_decoder_layers + 1) * (
            weight * loss_ce
            + 5 * loss_bbox
            + loss_giou
            + weight * loss_sem_align
        )
        + mask_loss_scale * supervised_mask_loss
        + consistency_loss_scale * corresponding_consistency_loss
        + source_choice_selector_loss_weight * source_choice_loss
        + source_moe_balance_loss_weight * moe_balance_loss
        + source_moe_rank_loss_weight * source_moe_rank_loss
        + source_moe_gate_loss_weight * source_moe_gate_loss
        + joint_query_quality_loss_weight * joint_query_quality_loss
        + mask_loss_scale
        * joint_query_quality_candidate_mask_loss_weight
        * (
            10.0 * joint_query_quality_candidate_mask_loss
            + 2.0 * joint_query_quality_candidate_dice_loss
        )
        + mask_loss_scale
        * joint_query_quality_candidate_lovasz_loss_weight
        * joint_query_quality_candidate_lovasz_loss
    )
    if isinstance(loss, torch.Tensor) and loss.numel() == 1:
        loss = loss.reshape(())
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_sem_align'] = loss_sem_align
    end_points['loss'] = loss
    end_points['loss_mask'] = loss_mask
    end_points['loss_dice'] = loss_dice
    end_points['sp_loss_mask'] = sp_loss_mask
    end_points['sp_loss_dice'] = sp_loss_dice
    end_points['corresponding_loss_mask']=corresponding_loss_mask
    end_points['corresponding_loss_dice']=corresponding_loss_dice
    end_points['adaptive_weight_loss_mask']=adaptive_weight_loss_mask
    end_points['adaptive_weight_loss_dice']=adaptive_weight_loss_dice
    end_points['source_choice_loss'] = source_choice_loss
    end_points['moe_balance_loss'] = moe_balance_loss
    end_points['source_moe_rank_loss'] = source_moe_rank_loss
    end_points['source_moe_box_rank_loss'] = source_moe_box_rank_loss
    end_points['source_moe_mask_rank_loss'] = source_moe_mask_rank_loss
    end_points['source_moe_anchor_loss'] = source_moe_anchor_loss
    end_points['source_moe_gate_loss'] = source_moe_gate_loss
    end_points['source_moe_gate_box_loss'] = source_moe_gate_box_loss
    end_points['source_moe_gate_mask_loss'] = source_moe_gate_mask_loss
    end_points['source_moe_gate_decision_loss'] = (
        source_moe_gate_decision_loss
    )
    end_points['joint_query_quality_loss'] = joint_query_quality_loss
    end_points['joint_query_quality_listwise_loss'] = (
        joint_query_quality_listwise_loss
    )
    end_points['joint_query_quality_aux_loss'] = joint_query_quality_aux_loss
    end_points['joint_query_quality_anchor_loss'] = (
        joint_query_quality_anchor_loss
    )
    end_points['joint_query_quality_source_mix_alignment_loss'] = (
        joint_query_quality_source_mix_alignment_loss
    )
    end_points['joint_query_quality_candidate_mask_loss'] = (
        joint_query_quality_candidate_mask_loss
    )
    end_points['joint_query_quality_candidate_dice_loss'] = (
        joint_query_quality_candidate_dice_loss
    )
    end_points['joint_query_quality_candidate_lovasz_loss'] = (
        joint_query_quality_candidate_lovasz_loss
    )
    end_points['joint_query_quality_candidate_mask_query_ratio'] = (
        joint_query_quality_candidate_mask_query_ratio
    )
    
    return loss, end_points
