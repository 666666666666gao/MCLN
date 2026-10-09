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
        score_path_frozen=True, original_matched_instances_preserved=True,
        native_coefficients=[5,1,10,2], loss=float(total))


def selected_output_gradient_witness(extra, predictions, batch, indices, targets, record):
    """Actual extra-loss gradients must reach only the selected unmatched output."""
    outputs = predictions['sp_last_pred_masks']
    with torch.no_grad():
        terms = outputs[0].new_zeros(4)
        for bid,evidence in enumerate(record['rows']):
            if not evidence['supervised']:
                continue
            root_truth = batch['gt_masks'][bid,0:1].float()
            assert torch.equal(root_truth,targets[bid]['masks'][0:1].float())
            target = (scatter_mean(root_truth,predictions['superpoints'][bid],dim=-1)>.5).float()
            query=evidence['query']
            own=outputs[bid][query:query+1].detach()
            text=predictions['last_pred_masks'][bid][0][query:query+1].detach()
            alpha=predictions['adaptive_weights'][bid].detach()
            fused=alpha*text+(1-alpha)*own
            terms += torch.stack((sigmoid_focal_loss(own,target,1),dice_loss(own,target,1),
                                  sigmoid_focal_loss(fused,target,1),dice_loss(fused,target,1)))
        expected = (terms*terms.new_tensor([5,1,10,2])).sum()/len(indices)
        assert torch.allclose(extra.detach(),expected,rtol=1e-6,atol=1e-6)
    gradients = torch.autograd.grad(extra, outputs, retain_graph=True, allow_unused=True)
    rows = []
    for bid, (gradient, output) in enumerate(zip(gradients, outputs)):
        evidence = record['rows'][bid]
        query = evidence['query']
        if gradient is None:
            assert not evidence['supervised']
            norms = output.new_zeros(output.shape[0])
        else:
            assert torch.isfinite(gradient).all()
            norms = gradient.abs().sum(dim=-1)
        permitted = norms.new_zeros(norms.shape, dtype=torch.bool)
        if evidence['supervised']:
            permitted[query] = True
            assert float(norms[query]) > 0
        assert int(torch.count_nonzero(norms[~permitted])) == 0
        assert int(torch.count_nonzero(norms[indices[bid][0]])) == 0
        rows.append(dict(query=query, role=evidence['role'], supervised=evidence['supervised'],
            selected_direct_gradient_l1=float(norms[query]),
            matched_direct_gradient_l1=float(norms[indices[bid][0]].sum()),
            all_unselected_direct_gradients_zero=True))
    assert record['extra_rows'] > 0
    return dict(status='pass', actual_eligible_rows=record['extra_rows'], rows=rows,
        actual_root_gt_reconciled=True, actual_batch_size=len(indices),
        extra_scalar_reconciled=True, recomputed_extra_loss=float(expected),
        native_mask_coefficients=[5,1,10,2], selected_roles=record['selected_roles'],
        original_matched_outputs_excluded=True, all_unselected_outputs_excluded=True,
        eligible_selected_outputs_nonzero=True, scope='extra loss direct output gradient only')
