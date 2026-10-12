"""GT-only coordinate projections of retained boxes; no neural or remote calls."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

study = Path(__file__).resolve().parent
source = study.parent / 'source/extremal_span_mixer.py'
stage = json.loads((study / 'SOURCE_STAGE_R4_RECEIPT.json').read_bytes())
assert hashlib.sha256(source.read_bytes()).hexdigest() == stage['source_files']['extremal_span_mixer.py']['sha256']
assert b'gate = raw_gate.clamp(0, 1)' in source.read_bytes()
input_path = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/formal/rows.jsonl')
input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
assert input_sha == '7b29b1646e52e8d3d253b3167b6895df9d789e7325022b0d9440814ec143c7b9'
rows = [json.loads(line) for line in input_path.read_text().splitlines()]
assert len(rows) == 9508 and [row['row_id'] for row in rows] == list(range(9508))
native = np.array([row['controls']['native']['box'] for row in rows],dtype=np.float64)
mask = np.array([row['controls']['mask']['box'] for row in rows],dtype=np.float64)
final = np.array([row['arms']['extremal_support']['box'] for row in rows],dtype=np.float64)
truth = np.array([row['root_box'] for row in rows],dtype=np.float64)
gate = np.array([row['arms']['extremal_support']['axis_gate'] for row in rows],dtype=np.float64)
for boxes in (native,mask,final,truth):
    assert boxes.shape == (9508,6) and np.isfinite(boxes).all() and (boxes[:,3:] > 0).all()
assert gate.shape == (9508,3) and ((gate >= 0) & (gate <= 1)).all()
expected = (1-gate.repeat(2,axis=1).reshape(-1,3,2)[:,:,0])
# Center and size use the same per-axis coefficient; keep the actual layout.
reconstructed = np.concatenate([(1-gate)*mask[:,:3]+gate*native[:,:3],
                               (1-gate)*mask[:,3:]+gate*native[:,3:]],axis=1)
reconstruction_error = float(np.abs(reconstructed-final).max())
assert reconstruction_error < 1e-5
def faces(boxes):
    return boxes[:,:3]-boxes[:,3:]/2, boxes[:,:3]+boxes[:,3:]/2
nl,nu = faces(native); ml,mu = faces(mask); tl,tu = faces(truth)
a,c,b,d = nl-ml,nu-mu,tl-ml,tu-mu
# For each axis the convex, piecewise-linear minimax error can attain its
# minimum at an endpoint, a zero, or a crossing of the two signed errors.
# Identical endpoints occur in this archive. A constant/parallel equation has
# no isolated breakpoint; omitting it is exact, not a new box fallback.
candidates = [np.zeros_like(a),np.ones_like(a)]
for numerator,denominator in ((b,a),(d,c),(b-d,a-c),(b+d,a+c)):
    value = np.zeros_like(a)
    np.divide(numerator,denominator,out=value,where=denominator != 0)
    candidates.append(value.clip(0,1))
candidate_gates = np.stack(candidates,axis=-1)
errors = np.maximum(np.abs(a[:,:,None]*candidate_gates-b[:,:,None]),
                    np.abs(c[:,:,None]*candidate_gates-d[:,:,None]))
best_index = errors.argmin(-1)
shared_gate = np.take_along_axis(candidate_gates,best_index[:,:,None],axis=2)[:,:,0]
shared_axis_error = errors.min(-1)
shared_lower = ml+a*shared_gate
shared_upper = mu+c*shared_gate
assert (shared_upper > shared_lower).all()
# Six independent face coefficients can attain these nearest face positions.
# The two endpoint boxes and GT are ordered, so these projected faces remain ordered.
independent_lower = np.maximum(np.minimum(tl,np.maximum(ml,nl)),np.minimum(ml,nl))
independent_upper = np.maximum(np.minimum(tu,np.maximum(mu,nu)),np.minimum(mu,nu))
assert (independent_upper > independent_lower).all()
independent_axis_error = np.maximum(np.abs(independent_lower-tl),np.abs(independent_upper-tu))
assert (independent_axis_error <= shared_axis_error+1e-10).all()
shared_error = shared_axis_error.max(-1)
independent_error = independent_axis_error.max(-1)
coupling_gap = shared_error-independent_error
assert (coupling_gap >= -1e-10).all()
fused_mask_iou = np.array([row['mask_iou'] for row in rows])
final_iou = np.array([row['arms']['extremal_support']['iou'] for row in rows])
mask_box_iou = np.array([row['controls']['mask']['iou'] for row in rows])
assert int((final_iou > .25).sum()) == 5677 and int((final_iou > .5).sum()) == 4920
def summary(selection):
    count = int(selection.sum())
    assert count > 0
    return dict(expressions=count,shared_minimum_max_face_error_median_m=float(np.median(shared_error[selection])),
        independent_minimum_max_face_error_median_m=float(np.median(independent_error[selection])),
        coupling_gap_median_m=float(np.median(coupling_gap[selection])),
        coupling_gap_above_1cm=int((coupling_gap[selection] > .01).sum()),
        coupling_gap_above_2cm=int((coupling_gap[selection] > .02).sum()),
        shared_error_above_2cm_but_independent_at_most_2cm=int(((shared_error > .02)&(independent_error <= .02)&selection).sum()),
        selected_axes_with_gate_zero=int((gate[selection] == 0).sum()),
        selected_axes_with_gate_one=int((gate[selection] == 1).sum()))
result = dict(scope='RETAINED_SELECTED_QUERY_FIXED_ENDPOINT_GT_COORDINATE_PROJECTION_DIAGNOSTIC',
    input_path=str(input_path.resolve()),input_sha256=input_sha,rows=9508,
    source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    actual_formula='center=(1-g)*Mask+g*native; size=(1-g)*Mask+g*native; g=clamp(raw,0,1)',
    GT_used_only_in_offline_diagnostic=True,new_model_forwards=0,remote_progress_queries=0,
    current_training_changed=False,method_accuracy=None,
    coordinate_projection_is_not_iou_optimization=True,whole_trainable_network_capacity_not_bounded=True,
    source_mask_iou_provenance_not_reaudited=True,gate_zero_not_proof_of_negative_raw_gate=True,
    max_same_gate_reconstruction_error_m=reconstruction_error,
    identical_lower_and_upper_endpoint_axes=int(((a==0)&(c==0)).sum()),
    groups={
      'all':summary(np.ones(9508,dtype=bool)),
      'selected_strict_errors':summary(final_iou <= .5),
      'saved_mask_iou_above_half_selected_box_error':summary((fused_mask_iou > .5)&(final_iou <= .5)),
      'mask_endpoint_correct_final_wrong':summary((mask_box_iou > .5)&(final_iou <= .5))},
    limitations=[
      'Fixed saved endpoints and selected Query only; core and support training can move endpoints.',
      'Coordinate minimax loss is not IoU; counts are not attainable REC gains or deployment scores.',
      'Mask IoU is the saved historical row field; it is not revalidated own-and-fused C eligibility.',
      'Independent face coordinate freedom is diagnostic, not a trained model or a new positive result.',
      'A previous independently parameterized face residual model had negative normal-training results.',
      'Zero clamped gates alone cannot distinguish raw<0 from raw==0; no chosen-C clamp-gradient incidence is claimed.'])
target = study / 'fixed_endpoint_axis_coupling_20261013'
assert not target.exists()
target.mkdir()
with (target / 'per_expression.csv').open('w',newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['row_id','scan_id','target_id','selected_query','final_iou','saved_mask_iou',
        'shared_minimax_max_face_error_m','independent_minimax_max_face_error_m','coupling_gap_m',
        'gate_x','gate_y','gate_z'])
    for index,row in enumerate(rows):
        writer.writerow([index,row['scan_id'],row['target_id'],row['query'],final_iou[index],fused_mask_iou[index],
            shared_error[index],independent_error[index],coupling_gap[index],*gate[index]])
result['per_expression_sha256'] = hashlib.sha256((target/'per_expression.csv').read_bytes()).hexdigest()
(target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
