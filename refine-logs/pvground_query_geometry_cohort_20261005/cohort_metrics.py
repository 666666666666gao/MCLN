import numpy as np

ARMS=('parent','control','query_supported')


def cpu_iou(boxes,truth):
    boxes=np.asarray(boxes,dtype=np.float64)
    truth=np.asarray(truth,dtype=np.float64)[:,None,:]
    assert boxes.shape[1:]==(256,6) and truth.shape[1:]==(1,6)
    assert np.isfinite(boxes).all() and np.isfinite(truth).all()
    assert (boxes[...,3:]>0).all() and (truth[...,3:]>0).all()
    low=np.maximum(boxes[...,:3]-boxes[...,3:]/2,truth[...,:3]-truth[...,3:]/2)
    high=np.minimum(boxes[...,:3]+boxes[...,3:]/2,truth[...,:3]+truth[...,3:]/2)
    intersection=np.maximum(high-low,0).prod(-1)
    union=boxes[...,3:].prod(-1)+truth[...,3:].prod(-1)-intersection
    return intersection/union


def faces(boxes):
    boxes=np.asarray(boxes,dtype=np.float64)
    return np.concatenate([boxes[...,:3]-boxes[...,3:]/2,boxes[...,:3]+boxes[...,3:]/2],axis=-1)


def paired(before,after,group):
    assert group.dtype==np.bool_ and before.shape==after.shape==group.shape
    count=int(group.sum())
    assert count>0
    values=dict(candidates=count,mean_delta_iou=float((after-before)[group].mean()))
    for threshold in (.25,.5):
        repair=int(((before<=threshold)&(after>threshold)&group).sum())
        damage=int(((before>threshold)&(after<=threshold)&group).sum())
        values[str(threshold)]=dict(before_hits=int(((before>threshold)&group).sum()),
            after_hits=int(((after>threshold)&group).sum()),repairs=repair,damages=damage,net=repair-damage)
    return values


def summarize(arrays):
    truth=arrays['root_box']
    own_good=2*arrays['query_intersection']>arrays['query_union']
    fused_good=2*arrays['fused_intersection']>arrays['fused_union']
    matched=arrays['matched_slot']>=0
    gpu_parent=arrays['parent_iou']
    qualified=own_good&fused_good&~matched&(gpu_parent<=.5)
    outside=arrays['outside'].any(-1)
    groups=dict(all_candidates=np.ones_like(qualified),parent_qualified=qualified,
        qualified_inside=qualified&~outside,qualified_outside=qualified&outside,
        parent_box_good=gpu_parent>.5)
    row_index=np.arange(len(truth))
    selected=arrays['bbs'].argmax(-1)
    selected_group=np.zeros_like(qualified)
    selected_group[row_index,selected]=True
    groups['selected_query']=selected_group
    groups['selected_qualified']=selected_group&qualified
    ious={arm:cpu_iou(arrays[arm+'_boxes'],truth) for arm in ARMS}
    summaries={}
    changes={}
    for arm in ARMS:
        gpu=arrays[arm+'_iou']
        assert np.max(np.abs(ious[arm]-gpu))<1e-5
        error=np.abs(faces(arrays[arm+'_boxes'])-faces(truth)[:,None]).max(-1)
        move=np.abs(faces(arrays[arm+'_boxes'])-faces(arrays['coarse_box'])).max(-1)
        assert np.max(np.abs(error-arrays[arm+'_max_face_error']))<1e-5
        assert np.max(np.abs(move-arrays[arm+'_max_face_move']))<1e-5
        summaries[arm]=dict(threshold_changes_vs_saved_GPU={
            str(t):int(((ious[arm]>t)!=(gpu>t)).sum()) for t in (.25,.5)},groups={})
        for name,group in groups.items():
            count=int(group.sum())
            item=dict(candidates=count)
            if count:
                item.update(mean_iou=float(ious[arm][group].mean()),median_iou=float(np.median(ious[arm][group])),
                    hits25=int(((ious[arm]>.25)&group).sum()),hits50=int(((ious[arm]>.5)&group).sum()),
                    median_max_face_error_m=float(np.median(error[group])),
                    median_max_face_move_m=float(np.median(move[group])),
                    GPU_DFL_mean=float(arrays[arm+'_dfl'][group].mean()))
            summaries[arm]['groups'][name]=item
    for before,after in (('parent','control'),('parent','query_supported'),('control','query_supported')):
        changes[before+'_to_'+after]={name:paired(ious[before],ious[after],group) for name,group in groups.items() if group.any()}
    return dict(rows=len(truth),candidates_per_row=256,parent_qualified_candidates=int(qualified.sum()),
        parent_qualified_rows=int(qualified.any(-1).sum()),qualified_outside_candidates=int((qualified&outside).sum()),
        qualified_outside_faces=int((qualified[...,None]&arrays['outside']).sum()),
        CPU_parent_qualification_changes=int((qualified!=(own_good&fused_good&~matched&(ious['parent']<=.5))).sum()),
        qualification_uses_saved_parent_GPU_threshold=True,arms=summaries,paired=changes,
        accuracy_result=False,limitation='Fixed augmented training64, correlated candidates. GPU DFL scalar is not independently CPU-recomputed. CPU floating IoU threshold differences are reported, not hidden.')
