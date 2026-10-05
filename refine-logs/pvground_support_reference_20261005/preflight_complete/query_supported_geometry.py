"""Training-only geometry targets for unmatched, Query-supported root candidates."""
import torch
from pvground_boundary_box_refiner import distribution_loss


def mask_root_iou(logits, point_superpoints, truth):
    members=torch.bincount(point_superpoints,minlength=logits.shape[-1]).float()
    root_members=torch.bincount(point_superpoints[truth],minlength=logits.shape[-1]).float()
    support=logits.detach().sigmoid()>.5
    intersection=support.float()@root_members
    union=support.float()@members+truth.sum()-intersection
    return intersection/union


def query_supported_geometry_loss(predictions,batch,indices,set_criterion,auxiliary_roots):
    """Leave native matching/losses intact; separately budget additional geometry.

    Eligibility is a training-GT support proxy, never an inference input. Query
    AND fused Mask IoU must exceed .5; final Box IoU must not. Every native
    matched candidate is excluded. The observed fixed64 includes empty rows,
    which contribute zero to the same batch denominator.
    """
    boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
    total=boxes.sum()*0
    counts=dict(extra_candidates=0,extra_rows=0,extra_boundary_faces=0,extra_boundary_outside=0,extra_row_counts=[])
    qualified=[]
    for bid,(matched_queries,matched_targets) in enumerate(indices):
        assert batch['language_dataset'][bid]=='scanrefer'
        valid=batch['box_label_mask'][bid].bool().nonzero().flatten()
        assert int(valid[0])==0
        truth_mask=batch['gt_masks'][bid,0].bool()
        assert truth_mask.any()
        superpoints=predictions['superpoints'][bid]
        text=predictions['last_pred_masks'][bid][0]
        query=predictions['sp_last_pred_masks'][bid]
        alpha=predictions['adaptive_weights'][bid]
        with torch.no_grad():
            query_iou=mask_root_iou(query,superpoints,truth_mask)
            fused_iou=mask_root_iou(alpha*text+(1-alpha)*query,superpoints,truth_mask)
            root=auxiliary_roots[bid]
            low=torch.maximum(boxes[bid,:,:3]-boxes[bid,:,3:]/2,root[:3]-root[3:]/2)
            high=torch.minimum(boxes[bid,:,:3]+boxes[bid,:,3:]/2,root[:3]+root[3:]/2)
            intersection=(high-low).clamp(min=0).prod(-1)
            iou=intersection/(boxes[bid,:,3:].prod(-1)+root[3:].prod()-intersection)
            eligible=(query_iou>.5)&(fused_iou>.5)&(iou<=.5)
            eligible[matched_queries]=False
            queries=eligible.nonzero().flatten()
        qualified.append(queries)
        counts['extra_row_counts'].append(int(queries.numel()))
        if queries.numel()==0:
            continue
        targets=torch.zeros_like(queries)
        single_indices=[(queries,targets)]
        losses=set_criterion.loss_boxes({'pred_boxes':boxes[bid:bid+1]},
            [{'boxes':root[None]}],single_indices,float(queries.numel()),None)
        edge,edge_counts=distribution_loss(
            {name:predictions[name][bid:bid+1] for name in ('boundary_logits','p3_coarse_center','p3_coarse_size')},
            dict(box_label_mask=torch.ones((1,1),device=root.device),
                 center_label=root[:3].view(1,1,3),size_gts=root[3:].view(1,1,3)),single_indices)
        total=total+(10*losses['loss_bbox']+2*losses['loss_giou']+edge)/7
        counts['extra_candidates']+=int(queries.numel())
        counts['extra_rows']+=1
        counts['extra_boundary_faces']+=edge_counts['boundary_faces']
        counts['extra_boundary_outside']+=edge_counts['boundary_target_outside']
    return total/boxes.shape[0],counts,qualified
