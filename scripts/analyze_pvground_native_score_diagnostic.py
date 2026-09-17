"""Recount archived fixed-logit results; no model execution or new labels."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'refine-logs/pvground_native_score_diagnostic_20260917_v2'
result=json.loads((root/'diagnostic.json').read_bytes())
assert result['status']=='complete' and (root/'controller.exit').read_text().strip()=='0'
for filename,key in [('rows.json','rows_sha256'),('input_selection.json','input_selection_sha256'),('candidate_values.npz','arrays_sha256')]:
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==result[key]
rows=json.loads((root/'rows.json').read_bytes())
selection=json.loads((root/'input_selection.json').read_bytes())['rows']
assert len(rows)==128 and [r['row_id'] for r in rows]==[r['row_id'] for r in selection]
assert len({r['scan_id'].split('_')[0] for r in rows})==128
arrays=np.load(root/'candidate_values.npz')
assert len(arrays.files)==256
summary={}
for r in rows:
    values=arrays[str(r['row_id'])+'_values'];targets=arrays[str(r['row_id'])+'_matched_target']
    assert values.shape==(256,4) and targets.shape==(256,)
    assert np.isfinite(values).all()
    score,formula,iou,velocity=values.T
    assert score[r['selected']]==score.max()
    assert formula[r['formula_selection']]==formula.max()
    assert iou[r['best_iou_query']]==iou.max()
    assert int(np.flatnonzero(targets==0)[0])==r['matched_root']
    for name,index_key in [('selected','selected'),('matched','matched_root'),('best','best_iou_query')]:
        i=r[index_key]
        assert abs(r[name+'_iou']-iou[i])<1e-7
        assert abs(r[name+'_score']-score[i])<1e-7
        assert abs(r[name+'_velocity']-velocity[i])<1e-7
    for label,index in [('matched',r['matched_root']),('best',r['best_iou_query'])]:
        assert r[label+'_score_rank']==int((score>score[index]).sum())+1
        assert abs(r[label+'_margin_velocity']-(velocity[index]-velocity[r['selected']]))<1e-7
for threshold in [.25,.5]:
    covered=[r for r in rows if r['selected_iou']<=threshold<r['best_iou']]
    computed=dict(selected_hits=sum(r['selected_iou']>threshold for r in rows),
        matched_root_hits=sum(r['matched_iou']>threshold for r in rows),
        raw_oracle_hits=sum(r['best_iou']>threshold for r in rows),errors_with_good_candidate=len(covered),
        covered_error_best_margin_increasing=sum(r['best_margin_velocity']>0 for r in covered),
        covered_error_matched_margin_increasing=sum(r['matched_margin_velocity']>0 for r in covered),
        covered_error_matched_root_good=sum(r['matched_iou']>threshold for r in covered))
    groups={name:dict(candidates=0,score_decreasing=0) for name in ['root','other_target','unmatched']}
    for r in rows:
        a=arrays[str(r['row_id'])+'_values'];m=arrays[str(r['row_id'])+'_matched_target'];good=a[:,2]>threshold
        masks=dict(root=m==0,other_target=m>0,unmatched=m<0)
        for name,mask in masks.items():
            groups[name]['candidates']+=int((good&mask).sum())
            groups[name]['score_decreasing']+=int((good&mask&(a[:,3]<0)).sum())
    computed.update(good_unmatched_candidates=groups['unmatched']['candidates'],good_unmatched_score_decreasing=groups['unmatched']['score_decreasing'])
    assert computed==result['summary'][str(threshold)]
    summary[str(threshold)]=dict(recount=computed,good_candidate_groups=groups,
        covered_error_row_ids=[r['row_id'] for r in covered],
        matched_bad_covered_error_row_ids=[r['row_id'] for r in covered if r['matched_iou']<=threshold])
output=dict(status='pass_export_recount',rows=128,candidates=32768,summary=summary,
    scope='Independent arithmetic from exported score/IoU/velocity/assignment arrays; does not recompute gradients, IoU from raw GT/boxes, or actual training updates.',
    diagnostic_sha256=hashlib.sha256((root/'diagnostic.json').read_bytes()).hexdigest())
(root/'recount.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output))
