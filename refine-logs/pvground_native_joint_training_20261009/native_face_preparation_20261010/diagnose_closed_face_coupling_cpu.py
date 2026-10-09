"""GT-only offline diagnosis of opposite-face coupling, never a deployment rule."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

root=Path(__file__).resolve().parent
source=Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1')
path=source/'complete_fit/formal/rows.jsonl'
recount=json.loads((source/'postrun_results/CPU_RECOUNT.json').read_bytes())
assert recount['status']=='CLOSED_SPAN_CPU_RECOUNT_COMPLETE'
assert hashlib.sha256(path.read_bytes()).hexdigest()==recount['formal_rows_sha256']
rows=[json.loads(line) for line in path.read_text().splitlines()]
assert len(rows)==9508 and len({row['row_id'] for row in rows})==9508
output=root/'face_coupling_diagnostic_20261010'
assert not output.exists()
output.mkdir()
names=('native','mask','fixed_half','whole_support','extremal_support')
boxes={name:np.asarray([row['arms' if name.endswith('_support') else 'controls'][name]['box']
    for row in rows],dtype=np.float32) for name in names}
truth=np.asarray([row['root_box'] for row in rows],dtype=np.float32)
assert all((value[:,3:]>0).all() and np.isfinite(value).all() for value in list(boxes.values())+[truth])


def iou(box, gt):
    intersection=np.maximum(0,np.minimum(box[:,:3]+box[:,3:]/2,gt[:,:3]+gt[:,3:]/2)
        -np.maximum(box[:,:3]-box[:,3:]/2,gt[:,:3]-gt[:,3:]/2)).prod(-1)
    return intersection/(box[:,3:].prod(-1)+gt[:,3:].prod(-1)-intersection)


values={name:iou(box,truth) for name,box in boxes.items()}
for name in names:
    assert [int((values[name]>threshold).sum()) for threshold in (.25,.5)]==recount['table'][name]['hits']
precise={name:box.astype(np.float64) for name,box in boxes.items()}
gt=truth.astype(np.float64)
gt_low,gt_high=gt[:,:3]-gt[:,3:]/2,gt[:,:3]+gt[:,3:]/2
native,mask=precise['native'],precise['mask']
n_low,n_high=native[:,:3]-native[:,3:]/2,native[:,:3]+native[:,3:]/2
m_low,m_high=mask[:,:3]-mask[:,3:]/2,mask[:,:3]+mask[:,3:]/2
lo_error_delta=np.abs(m_low-gt_low)-np.abs(n_low-gt_low)
hi_error_delta=np.abs(m_high-gt_high)-np.abs(n_high-gt_high)
opposed=(lo_error_delta*hi_error_delta)<0
mask_qualified=np.asarray([row['mask_iou']>.5 for row in rows])
selected_wrong=values['extremal_support']<=.5
lower_slopes,upper_slopes=n_low-m_low,n_high-m_high
# Identical endpoint faces occur in these stored predictions, so constant
# slopes have no crossing point to add; g=0 is already an endpoint candidate.
candidates=[np.zeros_like(lower_slopes),np.ones_like(lower_slopes)]
for base,slope in ((m_low,lower_slopes),(m_high,upper_slopes)):
    for target in (gt_low,gt_high):
        cross=np.divide(target-base,slope,out=np.zeros_like(slope),where=slope!=0)
        candidates.append(np.clip(cross,0,1))
one_d_best=np.zeros_like(lower_slopes)
for gate in candidates:
    low=m_low+gate*lower_slopes
    high=m_high+gate*upper_slopes
    intersection=np.maximum(0,np.minimum(high,gt_high)-np.maximum(low,gt_low))
    candidate=intersection/(high-low+gt_high-gt_low-intersection)
    one_d_best=np.maximum(one_d_best,candidate)
upper_bound=one_d_best.min(1)
# IoU3D <= each projected IoU1D; the minimum of independently maximized
# 1D IoUs is therefore an upper bound for every coupled three-axis gate.
observed_max_violation=max(float((iou(box,gt)-upper_bound).max()) for box in precise.values())
assert observed_max_violation<1e-6
low=np.clip(gt_low,np.minimum(m_low,n_low),np.maximum(m_low,n_low))
high=np.clip(gt_high,np.minimum(m_high,n_high),np.maximum(m_high,n_high))
assert (high>low).all()
independent=np.concatenate([(low+high)/2,high-low],-1)
independent_iou=iou(independent,gt)
bound_limited=upper_bound<=.5-1e-6
independent_good=independent_iou>.5+1e-6
evidence=bound_limited & independent_good
table={}
for group,select in [('all',np.ones(9508,dtype=bool)),('selected_strict_errors',selected_wrong),
    ('own_selected_fused_mask_qualified_strict_errors',mask_qualified & selected_wrong)]:
    table[group]=dict(rows=int(select.sum()),
        opposite_faces_prefer_different_endpoints=int((select & opposed.any(1)).sum()),
        opposite_preferences_both_margins_over_2cm=int((select & (opposed & (np.minimum(np.abs(lo_error_delta),np.abs(hi_error_delta))>.02)).any(1)).sum()),
        coupled_projection_upper_bound_below_half=int((select & bound_limited).sum()),
        independent_GT_face_projection_above_half=int((select & independent_good).sum()),
        independent_projection_qualifies_but_all_coupled_axis_gates_excluded_by_bound=int((select & evidence).sum()))
with (output/'rows.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.writer(stream)
    writer.writerow(['row_id','scan_id','target_id','query','fused_mask_iou','native_iou','mask_iou','fixed_half_iou',
        'extremal_iou','coupled_3d_upper_bound','GT_independent_face_projection_iou','opposed_axes',
        'bound_below_half_and_independent_above_half'])
    for index,row in enumerate(rows):
        writer.writerow([row['row_id'],row['scan_id'],row['target_id'],row['query'],row['mask_iou'],
            values['native'][index],values['mask'][index],values['fixed_half'][index],values['extremal_support'][index],
            upper_bound[index],independent_iou[index],int(opposed[index].sum()),int(evidence[index])])
result=dict(status='CLOSED_SELECTED_QUERY_FACE_COUPLING_OFFLINE_DIAGNOSTIC',
    time_cst=datetime.datetime.now().astimezone().isoformat(),rows=9508,
    source_scope='Closed frozen Span pair selected Query boxes, not ongoing normal training',
    source_formal_rows_sha256=recount['formal_rows_sha256'],
    source_CPU_recount_sha256=hashlib.sha256((source/'postrun_results/CPU_RECOUNT.json').read_bytes()).hexdigest(),
    counts={name:[int((values[name]>t).sum()) for t in (.25,.5)] for name in names},table=table,
    upper_bound_method='Exact piecewise-affine 1D IoU maxima at endpoints and GT-face crossings, then min across axes',
    independent_projection_method='GT-clipped target faces within each native/Mask endpoint interval; one feasible six-face choice, not maximized IoU',
    opposite_face_preference_method='Lower absolute GT face error; source preference margins over2cm is diagnostic subgroup only',
    numerical_scope='float64 bound/projection with1e-6 exclusion and qualification margins; deployed count recheck float32',
    maximum_observed_bound_violation=observed_max_violation,
    constant_lower_slopes=int((lower_slopes==0).sum()),constant_upper_slopes=int((upper_slopes==0).sum()),
    diagnostic_uses_GT=True,deployable_oracle=False,new_training_rule=False,
    fresh_diagnostic_review_pending=True,network_source_changed=False,
    SSH_calls=0,remote_status_queries=0,GPU_calls=0,neural_calls=0,new_formal_accuracy=None,
    full_goal_complete=False)
(output/'FACE_COUPLING_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
