"""Training-only geometry retention relative to the same candidate's reference."""
import torch

from query_supported_geometry import mask_root_iou


def box_iou(boxes, truth):
    lower=torch.maximum(boxes[..., :3]-boxes[..., 3:]/2,truth[..., :3]-truth[..., 3:]/2)
    upper=torch.minimum(boxes[..., :3]+boxes[..., 3:]/2,truth[..., :3]+truth[..., 3:]/2)
    overlap=(upper-lower).clamp(min=0).prod(-1)
    return overlap/(boxes[..., 3:].prod(-1)+truth[..., 3:].prod(-1)-overlap)


def reference_keep_loss(predictions,batch,indices):
    """Matched queries retain their own GT; support-confirmed extras use root.

    Match and support eligibility are stopped-gradient training decisions.
    Extras do not include any matched query and have no Box-IoU cutoff, so
    crossing .5 does not remove the reference-preservation responsibility.
    Within each expression, average once over its union; then average B.
    """
    boxes=torch.cat((predictions['last_center'],predictions['last_pred_size']),-1)
    references=torch.cat((predictions['geometry_reference_center'].detach(),
        predictions['geometry_reference_size'].detach().clamp(min=1e-6)),-1)
    total=boxes.sum()*0
    selected=[]
    counts=dict(matched=0,additional=0,reference_good25=0,reference_good50=0,
        degraded=0,improved=0,kept_per_row=[])
    for bid,(matched_queries,matched_targets) in enumerate(indices):
        matched_queries=matched_queries.to(device=boxes.device)
        matched_targets=matched_targets.to(device=boxes.device)
        assert batch['language_dataset'][bid]=='scanrefer'
        valid=batch['box_label_mask'][bid].bool().nonzero().flatten()
        assert int(valid[0])==0
        all_truth=torch.cat((batch['center_label'][bid,valid,:3],batch['size_gts'][bid,valid]),-1)
        with torch.no_grad():
            own=predictions['sp_last_pred_masks'][bid]
            text=predictions['last_pred_masks'][bid][0]
            alpha=predictions['adaptive_weights'][bid]
            members=predictions['superpoints'][bid]
            root_mask=batch['gt_masks'][bid,0].bool()
            eligible=(mask_root_iou(own,members,root_mask)>.5)&(
                mask_root_iou(alpha*text+(1-alpha)*own,members,root_mask)>.5)
            eligible[matched_queries]=False
            extra=eligible.nonzero().flatten()
            queries=torch.cat((matched_queries,extra))
            targets=torch.cat((matched_targets,torch.zeros_like(extra)))
            assert queries.unique().numel()==queries.numel()
            truth=all_truth[targets]
            ref_iou=box_iou(references[bid,queries],truth)
        selected.append(queries)
        counts['matched']+=int(matched_queries.numel())
        counts['additional']+=int(extra.numel())
        counts['kept_per_row'].append(int(queries.numel()))
        if queries.numel()==0:
            continue
        final_iou=box_iou(boxes[bid,queries],truth)
        total=total+(ref_iou-final_iou).clamp(min=0).square().mean()
        with torch.no_grad():
            counts['reference_good25']+=int((ref_iou>.25).sum())
            counts['reference_good50']+=int((ref_iou>.5).sum())
            counts['degraded']+=int((final_iou<ref_iou).sum())
            counts['improved']+=int((final_iou>ref_iou).sum())
    return total/boxes.shape[0],counts,selected
