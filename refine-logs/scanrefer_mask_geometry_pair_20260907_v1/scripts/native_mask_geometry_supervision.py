"""Experimental train-only root geometry objective; no deployment replacement.

The current native Hungarian root assignment selects the instance. The Mask
probability support supplies a differentiable box, supervised by the root GT.
This is distinct from teacher-box regression and candidate Mask focal/Dice.
"""
import torch
from torch.nn import functional as F

from models.losses import box_cxcyczwhd_to_xyzxyz, generalized_box_iou3d
from models.rec_mask_geometry import normalize_mcln_mask_logits, _find_superpoint_map
from scripts.native_teacher_box_transfer import root_query_indices
from scripts.prototype_probability_geometry import probability_quantile_boxes


def native_mask_geometry_loss(end_points, inputs, root_boxes, matches):
    assert root_boxes.ndim == 2 and root_boxes.shape[1] == 6
    query_indices = root_query_indices(matches, root_boxes.device)
    assert len(query_indices) == len(root_boxes)
    superpoints, _ = _find_superpoint_map(end_points, inputs)
    boxes = []
    for index in range(len(root_boxes)):
        _, _, fused, _ = normalize_mcln_mask_logits(end_points, index, query_indices[index:index + 1])
        geometry, _ = probability_quantile_boxes(
            inputs['point_clouds'][index, :, :3].detach(),
            superpoints[index].long().reshape(-1), fused, .005)
        boxes.append(geometry[0])
    boxes = torch.stack(boxes)
    targets = root_boxes.detach()
    l1 = (F.l1_loss(boxes[:, :3], targets[:, :3], reduction='none')
          + .2 * F.l1_loss(boxes[:, 3:], targets[:, 3:], reduction='none')).sum(-1)
    giou = 1. - torch.diag(generalized_box_iou3d(
        box_cxcyczwhd_to_xyzxyz(boxes), box_cxcyczwhd_to_xyzxyz(targets)))
    loss = (5. * l1 + giou).mean()
    return loss, {'query_indices': query_indices.detach(), 'soft_boxes': boxes.detach(),
                  'l1_per_row': l1.detach(), 'giou_per_row': giou.detach()}
