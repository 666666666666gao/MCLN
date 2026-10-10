"""Real closed predictions: initialization identity and ordered endpoint geometry.

CPU NumPy algebra only; this is not a torch model or training preflight.
"""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

root=Path(__file__).resolve().parent
source=Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1')
recount=json.loads((source/'postrun_results/CPU_RECOUNT.json').read_bytes())
path=source/'complete_fit/formal/rows.jsonl'
assert hashlib.sha256(path.read_bytes()).hexdigest()==recount['formal_rows_sha256']
rows=[json.loads(line) for line in path.read_text().splitlines()]
native=np.asarray([row['controls']['native']['box'] for row in rows],dtype=np.float32)
mask=np.asarray([row['controls']['mask']['box'] for row in rows],dtype=np.float32)
prior=np.asarray([row['arms']['extremal_support']['box'] for row in rows],dtype=np.float32)
gate=np.asarray([row['arms']['extremal_support']['axis_gate'] for row in rows],dtype=np.float32)
assert len(rows)==9508 and gate.shape==(9508,3) and (prior[:,3:]>0).all()
native_faces=np.stack([native[:,:3]-native[:,3:]/2,native[:,:3]+native[:,3:]/2],-1)
mask_faces=np.stack([mask[:,:3]-mask[:,3:]/2,mask[:,:3]+mask[:,3:]/2],-1)
prior_gate=np.repeat(gate[:,:,None],2,axis=2)
movement=(np.clip(prior_gate+np.zeros_like(prior_gate),0,1)-prior_gate)*(native_faces-mask_faces)
center=prior[:,:3]+(movement[:,:,0]+movement[:,:,1])/2
size=np.maximum(np.abs(prior[:,3:]+movement[:,:,1]-movement[:,:,0]),np.float32(1e-6))
assert np.array_equal(center,prior[:,:3]) and np.array_equal(size,prior[:,3:])
inverted_axes=set()
inverted_rows=set()
negative_combinations=0
zero_or_below_floor_combinations=0
for endpoints in itertools.product((0.,1.),repeat=6):
    face_gate=np.asarray(endpoints,dtype=np.float32).reshape(1,3,2)
    movement=(face_gate-prior_gate)*(native_faces-mask_faces)
    current_center=prior[:,:3]+(movement[:,:,0]+movement[:,:,1])/2
    raw_size=prior[:,3:]+movement[:,:,1]-movement[:,:,0]
    current_size=np.maximum(np.abs(raw_size),np.float32(1e-6))
    assert np.isfinite(current_center).all() and np.isfinite(current_size).all() and (current_size>0).all()
    indices=np.argwhere(raw_size<0)
    inverted_axes.update(tuple(index) for index in indices)
    inverted_rows.update(int(index[0]) for index in indices)
    negative_combinations+=int((raw_size<0).sum())
    zero_or_below_floor_combinations+=int((np.abs(raw_size)<1e-6).sum())
result=dict(status='CLOSED_PREDICTION_FACE_RESIDUAL_ALGEBRA_WITNESS',
    time_cst=datetime.datetime.now().astimezone().isoformat(),rows=9508,
    source_formal_rows_sha256=recount['formal_rows_sha256'],float32_zero_residual_center_and_size_bitwise_identity=True,
    all64_endpoint_gate_combinations_checked=True,possible_inverted_row_axes=len(inverted_axes),
    possible_inverted_expressions=len(inverted_rows),negative_axis_combinations=negative_combinations,
    size_floor_cases_across_combinations=zero_or_below_floor_combinations,
    ordered_finite_positive_outputs_all_combinations=True,
    new_face_parameters_analytical=332*64+64+64*32+32+32+1,
    prior_parameters=29793,new_mixer_parameters_analytical=29793+332*64+64+64*32+32+32+1,
    new_module_state_tensors_analytical=6,expected_new_model_state_tensors=1301,
    numerical_algebra_only=True,actual_torch_construction=False,actual_native_integration=False,
    gradient_or_optimizer_or_recovery_proven=False,engineering_review_pending=True,
    neural_calls=0,GPU_calls=0,SSH_calls=0,current_training_source_changed=False,
    new_formal_accuracy=None,full_goal_complete=False)
assert not (root/'CLOSED_ALGEBRA_WITNESS.json').exists()
(root/'CLOSED_ALGEBRA_WITNESS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
