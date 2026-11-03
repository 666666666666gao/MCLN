"""Matched-object supervision for the existing three raw Span axis gates."""

import torch
from torch.nn import functional as F


def span_gate_supervision(raw_gate, native_center, native_size,
                          reference_center, reference_size, reference_valid,
                          targets, match_indices):
    """Project each matched GT (center, size) onto its detached box segment.

    Equal center/size squared-error weights define a geometry proxy, not an
    IoU-optimal gate. Supervise raw gates so clamp saturation cannot block
    the auxiliary gradient. The prediction path itself is unchanged.
    """
    loss_sum = raw_gate.sum() * 0.0
    valid_axes = raw_gate.new_zeros(())
    target_sum = raw_gate.new_zeros(())
    for batch, (queries, objects) in enumerate(match_indices):
        gt = targets[batch]["boxes"][objects].detach()
        center = native_center[batch, queries].detach()
        size = native_size[batch, queries].detach().clamp_min(1e-6)
        center_direction = reference_center[batch, queries].detach() - center
        size_direction = reference_size[batch, queries].detach() - size
        length_squared = center_direction.square() + size_direction.square()
        projection = (
            (gt[:, :3] - center) * center_direction
            + (gt[:, 3:] - size) * size_direction
        ) / length_squared.clamp_min(1e-8)
        gate_target = projection.clamp(0.0, 1.0)
        valid = (
            reference_valid[batch, queries, None]
            & (length_squared > 1e-8)
            & (gt[:, 3:] > 0)
        )
        errors = F.smooth_l1_loss(
            raw_gate[batch, queries], gate_target, reduction="none",
        )
        loss_sum = loss_sum + errors[valid].sum()
        valid_axes = valid_axes + valid.sum()
        target_sum = target_sum + gate_target[valid].sum()
    denominator = valid_axes.clamp_min(1)
    return {
        "loss": loss_sum / denominator,
        "valid_axes": valid_axes.detach(),
        "target_mean": (target_sum / denominator).detach(),
    }
