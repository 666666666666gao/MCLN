"""Native ReferIt text weights; auxiliary root targets never affect detection rows."""
import torch

from pvground_boundary_box_refiner import distribution_loss
from query_supported_geometry import mask_root_iou


def token_weights(dataset):
    assert dataset in ('nr3d','sr3d')
    return (.625,.125,.125,.125) if dataset=='sr3d' else (.6,.2,.2,.1)


def semantic_assignment_correction(predictions,batch,indices,eos_coef,dataset):
    assert eos_coef==.1 and all(name==dataset for name in batch['language_dataset'])
    logp=predictions['last_sem_cls_scores'].log_softmax(-1)
    selected=torch.zeros(logp.shape[:2],dtype=torch.bool,device=logp.device)
    boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1).detach()
    target=sum(weight*batch[name][:,0] for weight,name in zip(token_weights(dataset),
        ('positive_map','modify_positive_map','pron_positive_map','rel_positive_map'))).detach()
    for bid,(queries,targets) in enumerate(indices):
        assert batch['sample_dataset'][bid] in (dataset,'scannet')
        if batch['sample_dataset'][bid]=='scannet':
            continue
        assert batch['box_label_mask'][bid,0] and int((targets==0).sum())==1
        root=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]]).detach()
        size=boxes[bid,:,3:].clamp(min=1e-6)
        root_size=root[3:].clamp(min=1e-6)
        lower=torch.maximum(boxes[bid,:,:3]-size/2,root[:3]-root_size/2)
        upper=torch.minimum(boxes[bid,:,:3]+size/2,root[:3]+root_size/2)
        intersection=(upper-lower).clamp(min=0).prod(-1)
        iou=intersection/(size.prod(-1)+root_size.prod()-intersection).clamp(min=1e-6)
        selected[bid]=iou>.5
        selected[bid,queries]=False
    denominator=sum(len(queries) for queries,_ in indices)
    replacement=(target*torch.log(target+1e-6)).sum(-1)[:,None]-(logp*target[:,None]).sum(-1)
    original=torch.log(torch.ones_like(logp[...,-1])+1e-6)-logp[...,-1]
    correction=((replacement-original)[selected].sum()*eos_coef/denominator)/7
    return correction,selected


def verify_native_replacement(predictions,batch,indices,eos_coef,dataset,correction,selected):
    logits=predictions['last_sem_cls_scores']
    logp=logits.log_softmax(-1)
    labels=torch.zeros_like(logp);labels[...,-1]=1
    weights=torch.full_like(logp[...,-1],eos_coef)
    maps=('positive_map','modify_positive_map','pron_positive_map','rel_positive_map')
    for bid,(queries,targets) in enumerate(indices):
        valid=batch['box_label_mask'][bid].bool()
        labels[bid,queries]=sum(weight*batch[name][bid,valid][targets]
            for weight,name in zip(token_weights(dataset),maps))
        weights[bid,queries]=1
    denominator=sum(len(queries) for queries,_ in indices)
    def ce(target):
        return ((target*torch.log(target+1e-6)-target*logp).sum(-1)*weights).sum()/denominator
    old=ce(labels)
    actual=predictions['last__loss_ce']
    assert torch.allclose(old,actual,rtol=1e-6,atol=1e-6)
    target=sum(weight*batch[name][:,0] for weight,name in zip(token_weights(dataset),maps)).detach()
    changed=labels.clone();changed[selected]=target[:,None].expand_as(labels)[selected]
    expected=ce(changed)/7;corrected=actual/7+correction
    assert torch.allclose(expected,corrected,rtol=1e-6,atol=1e-6)
    grad=torch.autograd.grad(corrected,logits,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,logits,retain_graph=True)[0]
    difference=torch.autograd.grad(correction,logits,retain_graph=True)[0]
    assert torch.allclose(grad,expected_grad,rtol=1e-5,atol=1e-7)
    assert (difference[~selected]==0).all()
    detection=torch.as_tensor([name=='scannet' for name in batch['sample_dataset']],device=logits.device)
    assert not selected[detection].any() and (difference[detection]==0).all()
    return dict(native_ce_error=float((old-actual).abs()),replacement_error=float((expected-corrected).abs()),
        gradient_error=float((grad-expected_grad).abs().max()),detection_semantic_correction_zero=True)


def extra_geometry_loss(predictions,batch,indices,set_criterion,dataset):
    boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
    total=boxes.sum()*0
    qualified=[]
    counts=dict(extra_candidates=0,extra_rows=0,detection_rows_excluded=0,extra_boundary_outside=0)
    for bid,(matched_queries,matched_targets) in enumerate(indices):
        assert batch['sample_dataset'][bid] in (dataset,'scannet')
        if batch['sample_dataset'][bid]=='scannet':
            qualified.append(matched_queries.new_empty(0));counts['detection_rows_excluded']+=1
            continue
        assert batch['box_label_mask'][bid,0] and int((matched_targets==0).sum())==1
        truth=batch['gt_masks'][bid,0].bool()
        assert truth.any()
        root=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])
        superpoints=predictions['superpoints'][bid]
        query=predictions['sp_last_pred_masks'][bid]
        text=predictions['last_pred_masks'][bid][0];alpha=predictions['adaptive_weights'][bid]
        with torch.no_grad():
            own_iou=mask_root_iou(query,superpoints,truth)
            fused_iou=mask_root_iou(alpha*text+(1-alpha)*query,superpoints,truth)
            low=torch.maximum(boxes[bid,:,:3]-boxes[bid,:,3:]/2,root[:3]-root[3:]/2)
            high=torch.minimum(boxes[bid,:,:3]+boxes[bid,:,3:]/2,root[:3]+root[3:]/2)
            intersection=(high-low).clamp(min=0).prod(-1)
            iou=intersection/(boxes[bid,:,3:].prod(-1)+root[3:].prod()-intersection)
            eligible=(own_iou>.5)&(fused_iou>.5)&(iou<=.5)
            eligible[matched_queries]=False
            selected=eligible.nonzero().flatten()
        qualified.append(selected)
        if selected.numel()==0:
            continue
        target_indices=[(selected,torch.zeros_like(selected))]
        loss=set_criterion.loss_boxes({'pred_boxes':boxes[bid:bid+1]},[{'boxes':root[None]}],
            target_indices,float(selected.numel()),None)
        edge,edge_counts=distribution_loss(
            {name:predictions[name][bid:bid+1] for name in ('boundary_logits','p3_coarse_center','p3_coarse_size')},
            dict(box_label_mask=torch.ones((1,1),device=root.device),center_label=root[:3].view(1,1,3),
                size_gts=root[3:].view(1,1,3)),target_indices)
        total=total+(10*loss['loss_bbox']+2*loss['loss_giou']+edge)/7
        counts['extra_candidates']+=int(selected.numel());counts['extra_rows']+=1
        counts['extra_boundary_outside']+=edge_counts['boundary_target_outside']
    return total/boxes.shape[0],counts,qualified
