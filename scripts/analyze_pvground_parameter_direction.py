"""Recount exported parameter-direction evidence without running the network."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'refine-logs/pvground_parameter_direction_20260917_v1'
result=json.loads((root/'diagnostic.json').read_bytes())
assert result['status']=='complete' and (root/'controller.exit').read_text().strip()=='0'
for filename,key in [('rows.json','rows_sha256'),('batches.json','batches_sha256'),('input_selection.json','input_selection_sha256'),('candidate_values.npz','arrays_sha256')]:
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==result[key]
rows=json.loads((root/'rows.json').read_bytes());batches=json.loads((root/'batches.json').read_bytes())
selection=json.loads((root/'input_selection.json').read_bytes())['rows']
assert len(rows)==128 and [r['row_id'] for r in rows]==[r['row_id'] for r in selection]
assert len({r['scan_id'].split('_')[0] for r in rows})==128 and len(batches)==16
assert [i for b in batches for i in b['rows']]==[r['row_id'] for r in rows]
arrays=np.load(root/'candidate_values.npz')
assert len(arrays.files)==128
for r in rows:
    a=arrays[str(r['row_id'])+'_values'];assert a.shape==(256,2) and np.isfinite(a).all()
    assert a[r['selected'],0]==a[:,0].max() and a[r['best_iou_query'],1]==a[:,1].max()
    for label,index in [('selected',r['selected']),('matched',r['matched_root'])]:
        assert abs(r[label+'_score']-a[index,0])<1e-7 and abs(r[label+'_iou']-a[index,1])<1e-7
    assert np.isclose(sum(r['full_velocity_by_group'].values()),r['parameter_full_velocity'],rtol=1e-12,atol=1e-12)
    assert np.isclose(sum(r['last_ce_velocity_by_group'].values()),r['parameter_last_ce_velocity'],rtol=1e-12,atol=1e-12)
    assert np.isclose(r['parameter_full_velocity']-r['parameter_last_ce_velocity'],r['parameter_remainder_velocity'],rtol=1e-12,atol=1e-12)
    if r['selected']==r['matched_root']:
        assert all(r[k]==0 for k in ['parameter_full_velocity','parameter_last_ce_velocity','parameter_remainder_velocity','logit_last_ce_velocity'])
summary={}
for t in [.25,.5]:
    errors=[r for r in rows if r['selected_iou']<=t<r['matched_iou']]
    recomputed=dict(selected_hits=sum(r['selected_iou']>t for r in rows),root_matched_hits=sum(r['matched_iou']>t for r in rows),
        root_covered_errors=len(errors),logit_ce_positive=sum(r['logit_last_ce_velocity']>0 for r in errors),
        parameter_ce_positive=sum(r['parameter_last_ce_velocity']>0 for r in errors),parameter_full_positive=sum(r['parameter_full_velocity']>0 for r in errors),
        parameter_full_negative=sum(r['parameter_full_velocity']<0 for r in errors),
        ce_positive_full_negative=sum(r['parameter_last_ce_velocity']>0 and r['parameter_full_velocity']<0 for r in errors),
        full_negative_row_ids=[r['row_id'] for r in errors if r['parameter_full_velocity']<0])
    assert recomputed==result['summary'][str(t)]
    negative=[r for r in errors if r['parameter_full_velocity']<0]
    summary[str(t)]=dict(recount=recomputed,
        logit_positive_parameter_ce_negative=[r['row_id'] for r in errors if r['logit_last_ce_velocity']>0 and r['parameter_last_ce_velocity']<0],
        negative_case_full_group_sums={g:sum(r['full_velocity_by_group'][g] for r in negative) for g in rows[0]['full_velocity_by_group']},
        negative_case_last_ce_group_sums={g:sum(r['last_ce_velocity_by_group'][g] for r in negative) for g in rows[0]['last_ce_velocity_by_group']})
output=dict(status='pass_export_recount',rows=128,candidates=32768,summary=summary,
    scope='Exported number/index/group-sum recount only; no independent raw-parameter gradient or GT IoU recomputation',
    diagnostic_sha256=hashlib.sha256((root/'diagnostic.json').read_bytes()).hexdigest())
(root/'recount.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))
