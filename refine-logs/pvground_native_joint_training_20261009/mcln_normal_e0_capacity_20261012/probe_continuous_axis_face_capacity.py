"""Offline GT probes for tied-axis and independent-face interval mixing."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

root=Path(__file__).resolve().parent
raw=(root/'full_paired_rows.jsonl').read_bytes()
binding=json.loads((root/'full_ROWS_TRANSFER.json').read_bytes())
assert len(raw)==binding['bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256']
rows=[json.loads(s) for s in raw.decode('utf-8').splitlines()]
assert len(rows)==9508
native=np.asarray([r['native']['selected_box'] for r in rows],dtype=np.float64)
reference=np.asarray([r['mask_reference']['selected_box'] for r in rows],dtype=np.float64)
gt=np.asarray([r['native']['root_gt_box'] for r in rows],dtype=np.float64)


def bounds(boxes):
    sizes=np.maximum(boxes[...,3:],1e-6)
    return boxes[...,:3]-sizes*.5,boxes[...,:3]+sizes*.5


def overlap_iou(low,high,gt_low,gt_high):
    extent=high-low
    target=gt_high-gt_low
    intersection=np.maximum(np.minimum(high,gt_high)-np.maximum(low,gt_low),0)
    shared=intersection.prod(-1)
    union=extent.prod(-1)+target.prod(-1)-shared
    return shared/union


nlow,nhigh=bounds(native)
rlow,rhigh=bounds(reference)
glow,ghigh=bounds(gt)
assert (nhigh>nlow).all() and (rhigh>rlow).all()
native_f64=overlap_iou(nlow,nhigh,glow,ghigh)
reference_f64=overlap_iou(rlow,rhigh,glow,ghigh)
native_saved=np.asarray([r['native']['selected_iou'] for r in rows])
reference_saved=np.asarray([r['mask_reference']['selected_iou'] for r in rows])

# For fixed other axes, IoU is fractional-linear between the points where
# predicted lower/upper faces cross either GT face. Its maximum on each piece
# lies at an endpoint (or the whole piece is flat). The global Cartesian set
# of these per-axis knots therefore contains a maximizer of this continuous
# three-axis family in real arithmetic. Calculation here uses float64.
knots=[np.zeros_like(rlow),np.ones_like(rlow)]
for target in (glow,ghigh):
    for first,last in ((rlow,nlow),(rhigh,nhigh)):
        delta=last-first
        value=np.divide(target-first,delta,out=np.zeros_like(delta),where=delta!=0)
        knots.append(np.clip(value,0,1))
knots=np.stack(knots,1)
assert knots.shape==(9508,6,3)
combinations=np.asarray(list(itertools.product(range(6),repeat=3)),dtype=np.int64)
axis_iou=[];axis_gate=[]
for start in range(0,len(rows),128):
    k=knots[start:start+128]
    gates=np.stack([k[:,combinations[:,a],a] for a in range(3)],-1)
    low=(1-gates)*rlow[start:start+128,None]+gates*nlow[start:start+128,None]
    high=(1-gates)*rhigh[start:start+128,None]+gates*nhigh[start:start+128,None]
    values=overlap_iou(low,high,glow[start:start+128,None],ghigh[start:start+128,None])
    chosen=values.argmax(-1)
    axis_iou.extend(values[np.arange(len(k)),chosen].tolist())
    axis_gate.extend(gates[np.arange(len(k)),chosen].tolist())
axis_iou=np.asarray(axis_iou)
axis_gate=np.asarray(axis_gate)
assert np.all(axis_iou+1e-12>=np.maximum(native_f64,reference_f64))

# Each GT face projects to its permitted endpoint interval. This is a legal
# independent-face mixture; no learned gate is being estimated here. Both
# endpoint boxes are ordered, and these GT projections remain ordered.
face_low=np.clip(glow,np.minimum(rlow,nlow),np.maximum(rlow,nlow))
face_high=np.clip(ghigh,np.minimum(rhigh,nhigh),np.maximum(rhigh,nhigh))
assert (face_high>face_low).all()
face_iou=overlap_iou(face_low,face_high,glow,ghigh)
assert np.all(face_iou+1e-12>=axis_iou)
coarse=json.loads((root/'AXIS_MIXTURE_CAPACITY_PROBE.json').read_bytes())
table=[]
for threshold in (.25,.5):
    axis_good=axis_iou>threshold
    face_good=face_iou>threshold
    union=(native_f64>threshold)|(reference_f64>threshold)
    table.append(dict(threshold=threshold, native_hits_under_float64_geometry=int((native_f64>threshold).sum()),
        reference_hits_under_float64_geometry=int((reference_f64>threshold).sum()),
        endpoint_native_threshold_disagreements_with_saved_float32=int(((native_f64>threshold)!=(native_saved>threshold)).sum()),
        endpoint_reference_threshold_disagreements_with_saved_float32=int(((reference_f64>threshold)!=(reference_saved>threshold)).sum()),
        offline_GT_whole_box_choice_hits=int(union.sum()), offline_GT_continuous_axis_knot_hits=int(axis_good.sum()),
        offline_GT_independent_face_projection_hits=int(face_good.sum()),
        independent_face_extra_over_continuous_axis=int((face_good&~axis_good).sum())))
examples=[]
affected=(face_iou>.5)&(axis_iou<=.5)
ids=np.flatnonzero(affected)
ids=ids[np.argsort(-(face_iou[ids]-axis_iou[ids]))][:12]
for index in ids:
    row=rows[int(index)]
    examples.append(dict(index=int(index),scan_id=row['scan_id'],selected_query=row['native']['selected_query'],
        native_iou_saved=float(native_saved[index]),reference_iou_saved=float(reference_saved[index]),
        GT_axis_knot_iou=float(axis_iou[index]),GT_axis_gate=axis_gate[index].tolist(),
        GT_independent_face_iou=float(face_iou[index]),GT_face_lower=face_low[index].tolist(),
        GT_face_upper=face_high[index].tolist(),root_gt_box=gt[index].tolist(),
        native_box=native[index].tolist(),reference_box=reference[index].tolist()))
result=dict(status='OFFLINE_SAVED_SELECTED_QUERY_GEOMETRIC_PARAMETERIZATION_PROBE',rows=9508,
    arithmetic='float64',candidate_axis_knot_combinations=216,table=table,examples=examples,
    paired_rows_sha256=binding['sha256'],GT_used_for_all_gate_choices=True,
    method_accuracy_claim=False,model_modified=False,new_neural_forwards=0,new_optimizer_updates=0,
    full_goal_complete=False,limitations=[
    'These GT choices cannot enter inference and do not establish learnable or achievable model accuracy.',
    'Native/uncorrected Mask endpoints are saved; the exact A-corrected pre-B reference is unavailable.',
    'Tied-axis knot sufficiency is for the stated continuous geometry family in real arithmetic; float64 results are diagnostics.',
    'Independent-face values are GT interval projections; an arbitrary learned six-face gate also needs valid-box parameterization.',
    'This does not validate a new module or replace ongoing normal-training E0/epoch metrics.'])
(root/'CONTINUOUS_AXIS_FACE_CAPACITY_PROBE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],table=table,method_accuracy_claim=False)))
