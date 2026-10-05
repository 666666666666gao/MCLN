"""CPU row recount only, against stored GT; no neural-model execution."""
import math
import statistics

def cpu_iou(box,truth):
    assert len(box)==len(truth)==6 and all(math.isfinite(v) for v in box+truth)
    assert all(v>0 for v in box[3:]+truth[3:])
    intersection=math.prod(max(0.,min(box[a]+box[a+3]/2,truth[a]+truth[a+3]/2)
        -max(box[a]-box[a+3]/2,truth[a]-truth[a+3]/2)) for a in range(3))
    return intersection/(math.prod(box[3:])+math.prod(truth[3:])-intersection)

def paired(before,after,threshold):
    assert len(before)==len(after)
    repairs=damages=changes=0
    for old,new in zip(before,after):
        assert all(old[key]==new[key] for key in ('row_id','scan_id','target_id','root_box','point_sha256'))
        a,b=old['bbs']['iou']>threshold,new['bbs']['iou']>threshold
        repairs+=not a and b
        damages+=a and not b
        changes+=old['bbs']['query']!=new['bbs']['query']
    return dict(repairs=repairs,damages=damages,net=repairs-damages,selected_query_changes=changes)

def summarize(rows,metric):
    threshold_changes=0
    maximum_iou_delta=0.
    displacements=[]
    for row in rows:
        prediction=row['bbs']
        assert 0<=prediction['mask_iou']<=1
        for box_key,iou_key in (('box','iou'),('coarse_box','coarse_iou')):
            value=cpu_iou(prediction[box_key],row['root_box'])
            maximum_iou_delta=max(maximum_iou_delta,abs(value-prediction[iou_key]))
            threshold_changes+=sum((value>t)!=(prediction[iou_key]>t) for t in (.25,.5))
        assert all(len(prediction[key])==4 and prediction[key]==sorted(prediction[key])
            and set(prediction[key]).issubset({0,1}) for key in ('oracle25','oracle50'))
        before,after=prediction['coarse_box'],prediction['box']
        displacements.append(max(abs((before[a]+sign*before[a+3]/2)
            -(after[a]+sign*after[a+3]/2)) for a in range(3) for sign in (-1,1)))
    assert threshold_changes==0
    hits=[sum(row['bbs']['iou']>t for row in rows) for t in (.25,.5)]
    mask_hits=[sum(row['bbs']['mask_iou']>t for row in rows) for t in (.25,.5)]
    assert hits==[metric['rec_hits25'],metric['rec_hits50']]
    assert mask_hits==[metric['mask_hits25'],metric['mask_hits50']]
    mask_miou=100*statistics.mean(row['bbs']['mask_iou'] for row in rows)
    assert abs(mask_miou-metric['mask_miou'])<1e-6
    refinement,coverage={},{ }
    for threshold,key in ((.25,'oracle25'),(.5,'oracle50')):
        repairs=sum(row['bbs']['coarse_iou']<=threshold<row['bbs']['iou'] for row in rows)
        damages=sum(row['bbs']['iou']<=threshold<row['bbs']['coarse_iou'] for row in rows)
        refinement[str(threshold)]=dict(coarse_hits=sum(row['bbs']['coarse_iou']>threshold for row in rows),
            final_hits=sum(row['bbs']['iou']>threshold for row in rows),repairs=repairs,damages=damages,net=repairs-damages)
        buckets=dict(first_good_2_16=0,first_good_17_32=0,first_good_33_64=0,first_good_65_256=0,none_full256=0)
        for row in rows:
            if row['bbs']['iou']>threshold:
                assert row['bbs'][key][0]
                continue
            for index,label in enumerate(list(buckets)[:4]):
                if row['bbs'][key][index]:
                    buckets[label]+=1
                    break
            else:
                buckets['none_full256']+=1
        assert sum(buckets.values())==len(rows)-sum(row['bbs']['iou']>threshold for row in rows)
        coverage[str(threshold)]=dict(topk=[16,32,64,256],
            oracle_hits=[sum(row['bbs'][key][k] for row in rows) for k in range(4)],
            errors_with_good_full256=sum(list(buckets.values())[:4]),errors_without_good_full256=buckets['none_full256'],
            error_first_good_rank=buckets)
    box_mask=dict(both_good=0,box_only_good=0,mask_only_good=0,both_bad=0)
    for row in rows:
        a,b=row['bbs']['iou']>.5,row['bbs']['mask_iou']>.5
        key='both_good' if a and b else 'box_only_good' if a else 'mask_only_good' if b else 'both_bad'
        box_mask[key]+=1
    return dict(rows=len(rows),rec_hits25=hits[0],rec_hits50=hits[1],rec_acc25=100*hits[0]/len(rows),rec_acc50=100*hits[1]/len(rows),
        mask_hits25=mask_hits[0],mask_hits50=mask_hits[1],mask_miou=mask_miou,
        cpu_box_threshold_changes=threshold_changes,max_cpu_vs_gpu_iou_absolute_difference=maximum_iou_delta,
        same_query_refinement=refinement,candidate_availability=coverage,selected_box_mask_at50=box_mask,
        median_selected_max_face_displacement_m=statistics.median(displacements),max_selected_face_displacement_m=max(displacements))
