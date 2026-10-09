"""Existing last-layer box terms, on the same frozen-parent actual-GT assignment."""
import torch


def matched_span_loss(predictions, indices, targets, criterion):
    count = sum(len(target['boxes']) for target in targets)
    assert count > 0
    boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
    native = criterion.loss_boxes(dict(pred_boxes=boxes), targets, indices, float(count), None)
    # Other native terms and layers are frozen. Preserve the original geometry
    # coefficient and seven-output averaging, not a new quality/score loss.
    loss = (10 * native['loss_bbox'] + 2 * native['loss_giou']) / 7
    assert torch.isfinite(loss)
    return loss, dict(loss_bbox=float(native['loss_bbox']), loss_giou=float(native['loss_giou']),
                       actual_valid_gt=count, correspondence='frozen_parent_actual_gt',
                       original_box_coefficients=[10, 2], original_output_average=7,
                       additional_root_geometry_queries=0, new_score_loss=False)
