"""One deployed, supported, unmatched Query receives a geometry target in training.

Scores are the unchanged native bbs scores. GT qualification is training-only.
This changes no matching, semantic target, inference score or candidate count.
"""
import torch


def selected_support_geometry_loss(predictions, batch, indices, targets,
                                   criterion, native_scores):
    batch_size = len(indices)
    assert native_scores.shape == (batch_size, 256)
    selected = native_scores.detach().argsort(dim=-1, descending=True)[:, 0]
    boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
    assert boxes.shape == (batch_size, 256, 6) and len(targets) == batch_size
    auxiliary_indices = []
    records = []
    for bid, (matched_queries, matched_targets) in enumerate(indices):
        assert batch['language_dataset'][bid] == 'scanrefer'
        valid = batch['box_label_mask'][bid].bool().nonzero().flatten()
        assert int(valid[0]) == 0 and len(valid) == len(targets[bid]['boxes'])
        assert int((matched_targets == 0).sum()) == 1
        query = int(selected[bid])
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        current = (empty, empty)
        if bool((matched_queries == query).any()):
            target_index = int(matched_targets[matched_queries == query].item())
            records.append(dict(query=query, extra_geometry=False,
                role='matched_root' if int(valid[target_index]) == 0 else 'matched_other',
                target_gt_slot=int(valid[target_index]), reason='original_matching_preserved'))
        else:
            with torch.no_grad():
                mapping = predictions['superpoints'][bid].long()
                own = predictions['sp_last_pred_masks'][bid][query]
                text = predictions['last_pred_masks'][bid][0][query]
                alpha = predictions['adaptive_weights'][bid]
                truth = targets[bid]['masks'][0].bool()
                assert mapping.shape == truth.shape
                own_foreground = own[mapping] > 0
                fused_foreground = (alpha * text[mapping] + (1 - alpha) * own[mapping]) > 0
                own_intersection = (own_foreground & truth).sum()
                own_union = (own_foreground | truth).sum()
                fused_intersection = (fused_foreground & truth).sum()
                fused_union = (fused_foreground | truth).sum()
                # Strict IoU > .5, expressed with integer counts. Empty support
                # is not a positive. No GT or qualification enters inference.
                qualified = bool((2 * own_intersection > own_union)
                                 & (2 * fused_intersection > fused_union))
            if qualified:
                current = (torch.tensor([query], device=boxes.device, dtype=torch.long),
                           torch.tensor([0], device=boxes.device, dtype=torch.long))
            records.append(dict(query=query, extra_geometry=qualified, role='unmatched',
                target_gt_slot=0 if qualified else None,
                own_intersection=int(own_intersection), own_union=int(own_union),
                fused_intersection=int(fused_intersection), fused_union=int(fused_union),
                reason='own_and_fused_support' if qualified else 'support_not_qualified'))
        auxiliary_indices.append(current)

    extra_rows = sum(record['extra_geometry'] for record in records)
    total = boxes.sum() * 0
    if extra_rows:
        losses = criterion.loss_boxes(dict(pred_boxes=boxes), targets,
                                      auxiliary_indices, float(batch_size), None)
        total = (10 * losses['loss_bbox'] + 2 * losses['loss_giou']) / 7
    assert torch.isfinite(total)
    return total, dict(extra_rows=extra_rows, rows=records,
        budget='at most one selected Query per expression; denominator is actual batch size',
        selected_roles={role: sum(record['role'] == role for record in records)
                        for role in ('matched_root', 'matched_other', 'unmatched')},
        original_matching_preserved=True, existing_loss_denominators_unchanged=True,
        selection_and_qualification_gradient_stopped=True,
        eligibility_uses_box_iou=False, new_semantic_target=False,
        native_geometry_coefficients=[10, 2], native_output_average=7,
        loss=float(total.detach()))
