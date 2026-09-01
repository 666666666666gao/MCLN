"""Training-only density-aware regression for the referred target box."""

import torch
import torch.nn.functional as F


TARGET_POINT_THRESHOLD = 256
TARGET_SIZE_LOSS_WEIGHT = 0.2


def _zero_stats(pred_boxes):
    zero = pred_boxes.sum() * 0.0
    return {
        "density_aware_target_box_loss": zero,
        "density_aware_target_box_active_row_ratio": zero.detach(),
        "density_aware_target_box_active_row_count": zero.detach(),
        "density_aware_target_box_referring_row_count": zero.detach(),
        "density_aware_target_box_target_point_count_mean": zero.detach(),
        "density_aware_target_box_sparsity_weight_mean": zero.detach(),
        "density_aware_target_box_center_l1": zero.detach(),
        "density_aware_target_box_size_l1": zero.detach(),
    }


def _validated_sample_datasets(sample_datasets, batch_size):
    if isinstance(sample_datasets, str):
        sample_datasets = [sample_datasets] * batch_size
    if not isinstance(sample_datasets, (list, tuple)):
        raise ValueError("sample_dataset must be a string or a batch sequence")
    if len(sample_datasets) != batch_size:
        raise ValueError("sample_dataset must align with predicted boxes")
    if any(not isinstance(name, str) or not name.strip()
           for name in sample_datasets):
        raise ValueError("sample_dataset entries must be non-empty strings")
    return [name.strip().lower() for name in sample_datasets]


def compute_density_aware_target_box_loss(
        pred_boxes, targets, match_indices, point_instance_label,
        sample_datasets):
    """Return the frozen sparse-target box loss and detached audit statistics.

    `targets[b]["boxes"][0]` is the referred target, and `match_indices[b]`
    is the existing final-layer Hungarian assignment. Detection-only ScanNet
    rows are excluded by construction.
    """
    if (not torch.is_tensor(pred_boxes) or pred_boxes.dim() != 3
            or pred_boxes.shape[-1] != 6
            or not pred_boxes.is_floating_point()):
        raise ValueError("pred_boxes must be floating [B,Q,6]")
    batch_size = pred_boxes.shape[0]
    if (not torch.is_tensor(point_instance_label)
            or point_instance_label.dim() != 2
            or point_instance_label.shape[0] != batch_size):
        raise ValueError("point_instance_label must be [B,P]")
    if not isinstance(targets, (list, tuple)) or len(targets) != batch_size:
        raise ValueError("targets must align with predicted boxes")
    if (not isinstance(match_indices, (list, tuple))
            or len(match_indices) != batch_size):
        raise ValueError("match_indices must align with predicted boxes")

    datasets = _validated_sample_datasets(sample_datasets, batch_size)
    stats = _zero_stats(pred_boxes)
    referring_rows = [index for index, name in enumerate(datasets)
                      if name != "scannet"]
    if not referring_rows:
        return stats

    referred_query_indices = {}
    for batch_index in referring_rows:
        target = targets[batch_index]
        if not isinstance(target, dict) or "boxes" not in target:
            raise ValueError("every target row must contain boxes")
        target_boxes = target["boxes"]
        if (not torch.is_tensor(target_boxes) or target_boxes.dim() != 2
                or target_boxes.shape[-1] != 6
                or target_boxes.shape[0] < 1):
            raise ValueError("target boxes must be non-empty [G,6]")
        source_index, target_index = match_indices[batch_index]
        if (not torch.is_tensor(source_index)
                or not torch.is_tensor(target_index)
                or source_index.dim() != 1 or target_index.dim() != 1
                or source_index.numel() != target_index.numel()):
            raise ValueError("each Hungarian assignment must be paired 1-D tensors")
        referred_matches = (target_index.to(dtype=torch.long) == 0).nonzero(
            as_tuple=False
        ).reshape(-1)
        if referred_matches.numel() != 1:
            raise ValueError(
                "each referring row must match GT target 0 exactly once"
            )
        match_position = int(referred_matches[0].item())
        query_index = int(source_index[match_position].item())
        if query_index < 0 or query_index >= pred_boxes.shape[1]:
            raise ValueError("Hungarian source index is outside the query axis")
        referred_query_indices[batch_index] = query_index

    point_counts = (point_instance_label.detach() == 0).sum(dim=1)
    active_rows = [
        index for index in referring_rows
        if 0 < int(point_counts[index].item()) < TARGET_POINT_THRESHOLD
    ]
    if not active_rows:
        stats["density_aware_target_box_referring_row_count"] = torch.tensor(
            float(len(referring_rows)), device=pred_boxes.device
        )
        return stats

    row_losses = []
    center_losses = []
    size_losses = []
    density_weights = []
    active_point_counts = []
    for batch_index in active_rows:
        target_boxes = targets[batch_index]["boxes"]
        query_index = referred_query_indices[batch_index]

        predicted = pred_boxes[batch_index, query_index]
        ground_truth = target_boxes[0].to(
            device=predicted.device, dtype=predicted.dtype
        ).detach()
        center_loss = F.l1_loss(
            predicted[:3], ground_truth[:3], reduction="sum"
        )
        size_loss = F.l1_loss(
            predicted[3:], ground_truth[3:], reduction="sum"
        )
        point_count = point_counts[batch_index].to(
            device=predicted.device, dtype=predicted.dtype
        )
        density_weight = (
            1.0 - point_count / float(TARGET_POINT_THRESHOLD)
        ).detach()
        row_losses.append(
            center_loss + TARGET_SIZE_LOSS_WEIGHT * size_loss
        )
        center_losses.append(center_loss)
        size_losses.append(size_loss)
        density_weights.append(density_weight)
        active_point_counts.append(point_count.detach())

    row_losses = torch.stack(row_losses)
    center_losses = torch.stack(center_losses)
    size_losses = torch.stack(size_losses)
    density_weights = torch.stack(density_weights)
    active_point_counts = torch.stack(active_point_counts)
    denominator = density_weights.sum()
    loss = (row_losses * density_weights).sum() / denominator

    stats["density_aware_target_box_loss"] = loss
    stats["density_aware_target_box_active_row_ratio"] = torch.tensor(
        float(len(active_rows)) / float(len(referring_rows)),
        device=pred_boxes.device,
    )
    stats["density_aware_target_box_active_row_count"] = torch.tensor(
        float(len(active_rows)), device=pred_boxes.device
    )
    stats["density_aware_target_box_referring_row_count"] = torch.tensor(
        float(len(referring_rows)), device=pred_boxes.device
    )
    stats["density_aware_target_box_target_point_count_mean"] = (
        active_point_counts.mean().detach()
    )
    stats["density_aware_target_box_sparsity_weight_mean"] = (
        density_weights.mean().detach()
    )
    stats["density_aware_target_box_center_l1"] = (
        (center_losses * density_weights).sum() / denominator
    ).detach()
    stats["density_aware_target_box_size_l1"] = (
        (size_losses * density_weights).sum() / denominator
    ).detach()
    return stats
