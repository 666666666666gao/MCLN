"""Train the deployed but unmatched Query Mask on the expression root GT."""
import torch
from models.losses import dice_loss, sigmoid_focal_loss, scatter_mean
from native_root_bbs import native_root_bbs


def selected_query_mask_loss(predictions, batch, indices, targets):
    scores = native_root_bbs(predictions['last_sem_cls_scores'].detach(), batch)
    selected = scores.argsort(dim=-1, descending=True)[:, 0]
    total = predictions['sp_last_pred_masks'][0].sum() * 0
    records = []
    for bid, (matched_queries, matched_targets) in enumerate(indices):
        assert batch['language_dataset'][bid] == 'scanrefer'
        valid = batch['box_label_mask'][bid].bool().nonzero().flatten()
        assert int(valid[0]) == 0
        assert int((matched_targets == 0).sum()) == 1
        query = int(selected[bid])
        if bool((matched_queries == query).any()):
            target_index = int(matched_targets[matched_queries == query].item())
            actual_gt_slot = int(valid[target_index])
            records.append(dict(query=query, supervised=False, target_gt_slot=actual_gt_slot,
                role='matched_root' if actual_gt_slot == 0 else 'matched_other',
                reason='native_matched_query_preserved'))
            continue
        truth = targets[bid]['masks'][0:1].float()
        target = (scatter_mean(truth, predictions['superpoints'][bid], dim=-1) > .5).float()
        own = predictions['sp_last_pred_masks'][bid][query:query+1]
        text = predictions['last_pred_masks'][bid][0][query:query+1]
        alpha = predictions['adaptive_weights'][bid]
        fused = alpha * text + (1-alpha) * own
        loss = (5*sigmoid_focal_loss(own,target,1) + dice_loss(own,target,1)
                + 10*sigmoid_focal_loss(fused,target,1) + 2*dice_loss(fused,target,1))
        total = total + loss
        records.append(dict(query=query, supervised=True, role='unmatched', target_gt_slot=0, loss=float(loss)))
    total = total / len(indices)
    assert torch.isfinite(total)
    return total, dict(extra_rows=sum(row['supervised'] for row in records), rows=records,
        selected_roles={role:sum(row['role']==role for row in records)
                        for role in ('matched_root','matched_other','unmatched')},
        budget='one selected unmatched Query per expression; empty expressions contribute zero',
        score_path_frozen=False, selection_gradient_stopped=True, original_matched_instances_preserved=True,
        native_coefficients=[5,1,10,2], loss=float(total))

