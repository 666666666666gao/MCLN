"""Independent NumPy recount for the pinned Nr3D/Sr3D native REC outputs."""
import numpy as np


def pairwise_iou(boxes, objects):
    boxes=np.asarray(boxes,dtype=np.float64)
    objects=np.asarray(objects,dtype=np.float64)
    intersection=np.maximum(
        np.minimum(boxes[:,None,:3]+boxes[:,None,3:]/2,objects[None,:,:3]+objects[None,:,3:]/2)
        -np.maximum(boxes[:,None,:3]-boxes[:,None,3:]/2,objects[None,:,:3]-objects[None,:,3:]/2),0).prod(-1)
    return intersection/(boxes[:,3:].prod(-1)[:,None]+objects[:,3:].prod(-1)[None,:]-intersection)


def audit_packet(raw_boxes,raw_scores,scores,overlap_valid,objects,object_mask,queries,root_box):
    """Recompute one row's filter, selected maximum and root IoU from exported values.

    Inputs are CPU arrays. Object slots retain their original mask; root_box is
    used only after candidate selection has been checked. Ties are accepted only
    when the recorded Query has the actual maximum score, not re-broken on CPU.
    """
    raw_boxes=np.asarray(raw_boxes)
    raw_scores=np.asarray(raw_scores)
    scores=np.asarray(scores)
    overlap_valid=np.asarray(overlap_valid)
    objects=np.asarray(objects)
    object_mask=np.asarray(object_mask)
    queries=np.asarray(queries)
    root_box=np.asarray(root_box,dtype=np.float64)
    assert raw_boxes.shape==(256,6) and raw_scores.shape==scores.shape==(2,256)
    assert overlap_valid.shape==(256,) and overlap_valid.dtype==np.bool_
    assert objects.ndim==2 and objects.shape[1]==6 and object_mask.shape==(len(objects),)
    assert object_mask.dtype==np.bool_ and object_mask.any()
    assert queries.shape==(2,) and np.issubdtype(queries.dtype,np.integer) and ((queries>=0)&(queries<256)).all()
    assert root_box.shape==(6,) and (root_box[3:]>0).all()
    for array in [raw_boxes,raw_scores,scores,objects,root_box]:assert np.isfinite(array).all()
    # Match the model's float32 size clamp before using float64 for independent geometry.
    boxes=raw_boxes.copy();boxes[:,3:]=np.maximum(boxes[:,3:],np.float32(1e-6))
    valid_objects=objects[object_mask]
    assert (valid_objects[:,3:]>0).all()
    independent_overlap=pairwise_iou(boxes,valid_objects).max(-1)
    expected_valid=independent_overlap>.25
    assert np.array_equal(expected_valid,overlap_valid),'Object overlap mask differs from independent geometry'
    assert np.array_equal(scores,raw_scores*expected_valid[None,:]),'Filtered scores do not match zero multiplication'
    ious=pairwise_iou(boxes,root_box[None])[:,0]
    modes={}
    for index,mode in enumerate(['bbs','bbf']):
        query=int(queries[index])
        assert scores[index,query]==scores[index].max(),'Selected Query is not a score maximum'
        modes[mode]=dict(query=query,iou=float(ious[query]),hit25=bool(ious[query]>.25),
            hit50=bool(ious[query]>.5),selected_overlap_valid=bool(expected_valid[query]),
            tied_maxima=int((scores[index]==scores[index,query]).sum()))
    return dict(modes=modes,valid_candidates=int(expected_valid.sum()),objects=int(object_mask.sum()),
        min_overlap_distance_to_threshold=float(np.abs(independent_overlap-.25).min()))


def formal_rec_check(dataset,rows,hits25,hits50):
    """Apply the user's REC floors only to complete, audited dataset counts."""
    counts={'nr3d':7899,'sr3d':17726}
    floors={'nr3d':(4726,4059),'sr3d':(12130,10157)}
    assert dataset in counts and rows==counts[dataset]
    assert isinstance(hits25,int) and isinstance(hits50,int) and 0<=hits50<=hits25<=rows
    lower,strict=floors[dataset]
    return dict(dataset=dataset,rows=rows,rec25=100.*hits25/rows,rec50=100.*hits50/rows,
        rec25_pass=hits25>=lower,rec50_pass=hits50>=strict,rec_baseline_pass=hits25>=lower and hits50>=strict,
        mask_gate=False)
