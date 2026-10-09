"""The native trainable Query/fused Mask terms on frozen-parent assignments."""
import torch
from models.losses import dice_loss, sigmoid_focal_loss, scatter_mean


def native_mask_targets(batch):
    valid = batch['box_label_mask'].bool()
    boxes = torch.cat([batch['center_label'][..., :3], batch['size_gts']], -1)
    fields = dict(labels='sem_cls_label', masks='gt_masks', positive_map='positive_map',
                  modify_positive_map='modify_positive_map', pron_positive_map='pron_positive_map',
                  other_entity_map='other_entity_map', rel_positive_map='rel_positive_map')
    targets = []
    for bid in range(len(valid)):
        target = {name: batch[key][bid][valid[bid]] for name, key in fields.items()}
        target['boxes'] = boxes[bid][valid[bid]]
        targets.append(target)
    return targets


def frozen_parent_assignments(predictions, batch, matcher):
    """Use uncorrected frozen-parent final predictions, not corrected Box quality."""
    targets = native_mask_targets(batch)
    output = dict(pred_logits=predictions['last_sem_cls_scores'].detach(),
                  pred_boxes=torch.cat([predictions['last_center'],
                                        predictions['last_pred_size']], -1).detach(),
                  pred_masks=predictions['last_pred_masks'],
                  superpoints=predictions['superpoints'])
    with torch.no_grad():
        indices = matcher(output, targets)
    return indices, targets


def matched_mask_loss(predictions, indices, targets):
    """Reuse native majority-SP GT and its four active Mask loss coefficients."""
    num_boxes = sum(len(target['boxes']) for target in targets)
    assert num_boxes > 0
    sp_focal, sp_dice, fused_focal, fused_dice = 0, 0, 0, 0
    matched = 0
    for bid, (queries, target_ids) in enumerate(indices):
        truth = targets[bid]['masks'][target_ids].float()
        target_mask = (scatter_mean(truth, predictions['superpoints'][bid], dim=-1) > .5).float()
        own = predictions['sp_last_pred_masks'][bid][queries]
        text = predictions['last_pred_masks'][bid][0][:len(queries)]
        alpha = predictions['adaptive_weights'][bid]
        fused = alpha * text + (1 - alpha) * own
        sp_focal = sp_focal + sigmoid_focal_loss(own, target_mask, num_boxes)
        sp_dice = sp_dice + dice_loss(own, target_mask, num_boxes)
        fused_focal = fused_focal + sigmoid_focal_loss(fused, target_mask, num_boxes)
        fused_dice = fused_dice + dice_loss(fused, target_mask, num_boxes)
        matched += len(queries)
    loss = 5 * sp_focal + sp_dice + 10 * fused_focal + 2 * fused_dice
    assert torch.isfinite(loss)
    return loss, dict(matched_queries=matched, valid_gt=num_boxes,
        query_focal=float(sp_focal), query_dice=float(sp_dice),
        fused_focal=float(fused_focal), fused_dice=float(fused_dice),
        loss=float(loss), correspondence='frozen_parent_actual_gt',
        native_coefficients=[5, 1, 10, 2], expanded_positive_queries=0)
