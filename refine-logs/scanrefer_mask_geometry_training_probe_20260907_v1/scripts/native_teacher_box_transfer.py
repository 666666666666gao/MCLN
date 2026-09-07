"""Train-only geometry transfer into the existing native box regression heads.

GT remains the primary objective. A frozen system's selected box is an auxiliary
target only when it overlaps the referring root and is better than the current
GT-matched student box. The current Hungarian assignment binds the target to the
student; archived teacher Query IDs never enter this loss.
"""

import torch
from torch.nn import functional as F

from models.losses import box_cxcyczwhd_to_xyzxyz, generalized_box_iou3d
from models.rec_reranker import compute_query_ious


BOX_PARAMETER_PREFIXES = (
    'prediction_heads.5.center_residual_head.',
    'prediction_heads.5.size_pred_head.',
)


def root_query_indices(matches, device):
    result = []
    for student_indices, target_indices in matches:
        root = student_indices[target_indices == 0]
        assert root.numel() == 1
        result.append(int(root.item()))
    return torch.tensor(result, dtype=torch.long, device=device)


def native_teacher_box_loss(student_boxes, teacher_boxes, root_boxes, matches):
    assert student_boxes.ndim == 3 and student_boxes.shape[-1] == 6
    assert teacher_boxes.shape == root_boxes.shape == (len(student_boxes), 6)
    query_indices = root_query_indices(matches, student_boxes.device)
    student = student_boxes[torch.arange(len(student_boxes), device=student_boxes.device), query_indices]
    teacher = teacher_boxes.detach()
    roots = root_boxes.detach()
    valid = torch.ones((len(student), 1), dtype=torch.bool, device=student.device)
    with torch.no_grad():
        teacher_iou = compute_query_ious(teacher[:, None], roots[:, None], valid)[:, 0]
        student_iou = compute_query_ious(student.detach()[:, None], roots[:, None], valid)[:, 0]
        gain = (teacher_iou - student_iou).clamp(min=0.)
        gain = gain * (teacher_iou > .25).to(gain)
        eligible = gain > 0
    # Match the existing MCLN center/size and GIoU conventions, including its
    # 1e-6 evaluation extent floor. This does not alter native outputs.
    l1 = (F.l1_loss(student[:, :3], teacher[:, :3], reduction='none')
          + .2 * F.l1_loss(student[:, 3:], teacher[:, 3:], reduction='none')).sum(dim=-1)
    giou = 1. - torch.diag(generalized_box_iou3d(
        box_cxcyczwhd_to_xyzxyz(student), box_cxcyczwhd_to_xyzxyz(teacher)))
    per_row = 5. * l1 + giou
    if bool(eligible.any()):
        loss = (per_row * gain).sum() / gain.sum()
    else:
        loss = per_row.sum() * 0.
    return loss, {
        'student_query_indices': query_indices.detach(),
        'student_root_ious': student_iou,
        'teacher_root_ious': teacher_iou,
        'eligible': eligible,
        'gain_weights': gain,
        'l1_per_row': l1.detach(),
        'giou_per_row': giou.detach(),
    }
