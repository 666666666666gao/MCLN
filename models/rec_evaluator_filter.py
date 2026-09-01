"""Shared detector-overlap validity used by cache, runtime, and evaluator."""

import math

import torch

from .losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz


def build_detector_overlap_valid(
        candidate_boxes, candidate_valid, detected_boxes, detected_valid,
        iou_threshold=0.25):
    """Keep valid candidates whose box overlaps any active detector box."""
    tensors = {
        "candidate_boxes": candidate_boxes,
        "candidate_valid": candidate_valid,
        "detected_boxes": detected_boxes,
        "detected_valid": detected_valid,
    }
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("detector-overlap inputs must be tensors")
    if (not torch.is_floating_point(candidate_boxes)
            or not torch.is_floating_point(detected_boxes)):
        raise TypeError("detector-overlap boxes must be floating point")
    if candidate_valid.dtype != torch.bool or detected_valid.dtype != torch.bool:
        raise TypeError("detector-overlap validity masks must be boolean")
    if (candidate_boxes.dim() < 3 or candidate_boxes.shape[-1] != 6
            or candidate_valid.shape != candidate_boxes.shape[:-1]):
        raise ValueError(
            "candidate boxes and validity must have shape [B,...,6]/[B,...]"
        )
    if (detected_boxes.dim() != 3 or detected_boxes.shape[-1] != 6
            or detected_valid.shape != detected_boxes.shape[:2]
            or detected_boxes.shape[0] != candidate_boxes.shape[0]):
        raise ValueError(
            "detector boxes and validity must have shape [B,D,6]/[B,D]"
        )
    devices = {value.device for value in tensors.values()}
    if len(devices) != 1:
        raise ValueError("detector-overlap inputs must share one device")
    if (not isinstance(iou_threshold, (int, float))
            or isinstance(iou_threshold, bool)
            or not math.isfinite(float(iou_threshold))
            or not 0.0 <= float(iou_threshold) <= 1.0):
        raise ValueError("detector-overlap IoU threshold must lie in [0,1]")
    if (not bool(torch.isfinite(
            candidate_boxes[candidate_valid]
    ).all().item())
            or not bool(torch.isfinite(
                detected_boxes[detected_valid]
            ).all().item())):
        raise ValueError("active detector-overlap boxes must be finite")
    if bool((candidate_boxes[..., 3:][candidate_valid] <= 0.0).any().item()):
        raise ValueError(
            "active candidate detector-overlap boxes need positive sizes"
        )
    # The formal evaluator clamps active detector sizes to 1e-6 before IoU.

    batch_size = candidate_boxes.shape[0]
    flat_boxes = candidate_boxes.reshape(batch_size, -1, 6)
    flat_valid = candidate_valid.reshape(batch_size, -1)
    surviving = torch.zeros_like(flat_valid)
    for batch_index in range(batch_size):
        candidate_indices = flat_valid[batch_index].nonzero(
            as_tuple=False
        ).reshape(-1)
        detector_indices = detected_valid[batch_index].nonzero(
            as_tuple=False
        ).reshape(-1)
        if not candidate_indices.numel() or not detector_indices.numel():
            continue
        detector_ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(
                detected_boxes[batch_index, detector_indices]
            ),
            box_cxcyczwhd_to_xyzxyz(
                flat_boxes[batch_index, candidate_indices]
            ),
        )
        surviving[batch_index, candidate_indices] = (
            detector_ious.max(dim=0).values > float(iou_threshold)
        )
    return surviving.reshape(candidate_valid.shape)
