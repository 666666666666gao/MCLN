"""Independently recount every retained box and error candidate on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--directory',type=Path,required=True)
args=parser.parse_args();root=args.directory
receipt=json.loads((root/'receipt.json').read_bytes())
raw=(root/'rows.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()==receipt['rows_sha256']
rows=[json.loads(line) for line in raw.splitlines()]
assert len(rows)==receipt['rows'] and receipt['all_candidates_retained']==256
all_rows=[];partitions={};qualification_counts={}
hits={'25':0,'50':0};max_iou_error=0.;comparisons=[]
formal=Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\complete_tail_fused_retry\arm\formal\rows.jsonl')
assert formal.is_file()
reference=[json.loads(line) for line in formal.read_bytes().splitlines()]
assert len(reference)==9508
for chunk in sorted(root.glob('candidates_*.npz')):
    data=np.load(chunk,allow_pickle=False)
    assert data['boxes'].shape[1:]==(256,6) and data['coarse_boxes'].shape==data['boxes'].shape
    for offset,identity in enumerate(data['row_id']):
        index=len(all_rows);row=rows[index];assert int(identity)==index==row['row_id']
        boxes=data['boxes'][offset].astype(np.float64);gt=np.asarray(row['root_box'],dtype=np.float64)
        assert np.isfinite(boxes).all() and (boxes[:,3:]>0).all()
        lo=np.maximum(boxes[:,:3]-boxes[:,3:]/2,gt[:3]-gt[3:]/2)
        hi=np.minimum(boxes[:,:3]+boxes[:,3:]/2,gt[:3]+gt[3:]/2)
        inter=np.maximum(hi-lo,0).prod(-1)
        iou=inter/(boxes[:,3:].prod(-1)+gt[3:].prod()-inter)
        saved=data['root_iou'][offset];max_iou_error=max(max_iou_error,float(np.abs(iou-saved).max()))
        for threshold in (.25,.5):assert np.array_equal(iou>threshold,saved>threshold)
        scores=data['bbs_scores'][offset];rank=scores.argsort()[::-1]
        selected=row['selected_query']
        # Torch and NumPy may order tied slots differently. The captured native
        # top-1 must be a maximum; first-qualified ranks are independently checked
        # by counting scores, with ties explicitly bounded rather than invented.
        assert scores[selected]==scores.max()
        assert float(saved[selected])==row['selected_iou']
        assert float(data['root_mask_iou'][offset,selected])==row['selected_mask_iou']
        matched=data['matched_GT_slot'][offset]
        assert int((matched==0).sum())==1
        proxy=data['root_joint_best_scene_overlap'][offset]
        limits={}
        for threshold,suffix in ((.25,'25'),(.5,'50')):
            good=iou>threshold
            counts={'matched_root':int((good&(matched==0)).sum()),
                'matched_other':int((good&(matched>0)).sum()),'unmatched':int((good&(matched<0)).sum())}
            expected=row['geometric_qualification'][suffix]
            assert all(counts[key]==expected[key] for key in counts)
            assert int((good&proxy).sum())==expected['qualified_root_overlap_proxy']
            candidates=np.flatnonzero(good)
            if len(candidates):
                top_score=scores[candidates].max()
                lower=1+int((scores>top_score).sum())
                upper=int((scores>=top_score).sum())
                assert lower<=row['first_qualified_rank'][suffix]<=upper
                limits[suffix]=dict(lower=lower,upper=upper)
            else:
                assert row['first_qualified_rank'][suffix] is None
                limits[suffix]=None
            hits[suffix]+=int(iou[selected]>threshold)
            group=partitions.setdefault(suffix,{'errors':0,'has_alternative':0,'has_unmatched_qualified':0,
                'has_unmatched_qualified_root_overlap_proxy':0,'has_matched_other_qualified':0,
                'no_qualified':0,'selected_loose_only':0})
            total=qualification_counts.setdefault(suffix,dict(matched_root=0,matched_other=0,unmatched=0))
            for key,value in counts.items():total[key]+=value
            if iou[selected]<=threshold:
                group['errors']+=1;group['has_alternative']+=int(good.any())
                group['has_unmatched_qualified']+=int((good&(matched<0)).any())
                group['has_unmatched_qualified_root_overlap_proxy']+=int((good&(matched<0)&proxy).any())
                group['has_matched_other_qualified']+=int((good&(matched>0)).any())
                group['no_qualified']+=int(not good.any())
                group['selected_loose_only']+=int(.25<iou[selected]<=.5)
        prior=reference[index]
        for key in ('row_id','scan_id','target_id','root_box','point_sha256'):assert row[key]==prior[key],(index,key)
        comparisons.append(dict(row_id=index,selected_query_equal=selected==prior['bbs']['query'],
            hit25_equal=(row['selected_iou']>.25)==(prior['bbs']['iou']>.25),
            hit50_equal=(row['selected_iou']>.5)==(prior['bbs']['iou']>.5)))
        all_rows.append(dict(row_id=index,first_rank_tie_bounds=limits))
assert len(all_rows)==receipt['rows'] and hits==receipt['rec_hits']
for group in partitions.values():assert group['errors']==group['has_alternative']+group['no_qualified']
result=dict(status='pass',rows=len(all_rows),rec_hits=hits,all_candidates_retained=256,
    CPU_all_box_thresholds_exact=True,max_CPU_vs_GPU_iou_error=max_iou_error,
    error_partition=partitions,qualified_candidate_counts=qualification_counts,
    versus_archived_formal=dict(selected_query_changes=sum(not r['selected_query_equal'] for r in comparisons),
        hit25_changes=sum(not r['hit25_equal'] for r in comparisons),hit50_changes=sum(not r['hit50_equal'] for r in comparisons)),
    evidence_limits=receipt['evidence_limits'],
    unmatched_rule='Same native last-layer Hungarian matcher on evaluation inputs. G already replaces CE for unmatched IoU>.5; lower IoU is not an automatic semantic-negative or positive identity proof.',
    rank_rule='Exact native ranks are captured; CPU validates first-qualified score-tie bounds, not arbitrary tie reordering.')
(root/'CPU_RECOUNT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
