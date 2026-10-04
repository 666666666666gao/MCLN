"""Seal a read-only dependency finding while the fixed experiment runs."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
support = local.parent / 'pvground_fused_support_20261002'
port_file = support / 'complete_tail_fused_retry/source/source_port.json'
port = json.loads(port_file.read_bytes())
assert hashlib.sha256(port_file.read_bytes()).hexdigest() == json.loads((local/'face_fit_spec.json').read_bytes())['source_port_sha256']
paths = dict(model=support/'complete_tail_fused_retry/source/imported/models.pv_ground.py',
    head=support/'source_preview/PV-Ground/models/modules.py',
    evaluator=support/'complete_tail_fused_retry/source/imported/evaluator.py',
    assignment=local/'pvground_semantic_assignment.py',
    face=local/'pvground_face_conditioned_box_refiner.py')
assert hashlib.sha256(paths['model'].read_bytes()).hexdigest() == port['files']['models/pv_ground.py']
assert hashlib.sha256(paths['head'].read_bytes()).hexdigest() == port['files']['models/modules.py']
imports = json.loads((local/'complete_preflight/face_conditioned/imports.json').read_bytes())
assert hashlib.sha256(paths['evaluator'].read_bytes()).hexdigest() == imports['sha256']['evaluator']
models = ast.parse(paths['model'].read_text(encoding='utf-8'))
head = ast.parse(paths['head'].read_text(encoding='utf-8'))
face = ast.parse(paths['face'].read_text(encoding='utf-8'))
assert any(isinstance(node, ast.ClassDef) and node.name == 'ClsAgnosticPredictHead' for node in ast.walk(head))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='READ_ONLY_SOURCE_DEPENDENCY_WITNESS', remote_queries=0,
    native_forwards=0, optimizer_updates=0, new_semantic_model_implemented=False,
    active_fit_source_changed=False,
    files={name:dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name,path in paths.items()},
    model_prediction_before_mask=True, tail_box_replacement_after_mask=True,
    semantic_head_contains_batchnorm_and_dropout=True,
    geometry_branch_uses_distinct_geometry_features=True,
    whole_mask_and_face_refiner_do_not_require_final_semantic_logits=True,
    existing_G_target_mass_is_not_normalized=True,
    proof_scope='actual pinned source and prior actual import receipt; no new runtime call-order experiment')
(local/'READBACK_INTERFACE_WITNESS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
text = '''# Final geometry to native semantic score: actual interface finding

This is a read-only dependency finding, not an implemented or trained S3. The
current face fit is unchanged. Its eventual metrics must decide the geometry
provider before any next source/factory is deployed.

Pinned source follows the actual preflight model/import receipts. The tail port
changes only models/pv_ground.py; unchanged models/modules.py is SHA-bound by
the same port. READBACK_INTERFACE_WITNESS lists exact local bytes and paths.

## Actual native order

1. Final decoder returns distinct semantic query and geometry query.
2. Contrast projection uses the semantic query before the prediction head.
3. ClsAgnosticPredictHead uses semantic features for sem_cls_scores_head;
   center/size use geometry_features. The semantic branch is ThreeLayerMLP,
   containing two BatchNorm1d and two Dropout(0.3) operations.
4. Query Mask uses the final geometry query; Text Mask uses original text
   features and its separate text prediction head. Native adaptive fusion and
   both final Mask lists are produced before candidate_box_refiner.
5. CandidateBoxRefiner replaces only last_center/last_pred_size. The current
   face module also stores boundary_logits and Bx256x6x64 face states. Those
   states are not read by the semantic branch or native evaluator.
6. The evaluator computes token-softmax and expression-weighted token evidence.
   One scalar broadcast over all token logits cancels in that softmax. No
   additional quality score is currently combined with bbs.

## Minimal future insertion, conditional on terminal evidence

A future readback needs the original semantic query, text features/padding and
the actual final geometry evidence. None is currently a new GT inference input.
Final semantic scoring is not required to form the two Mask paths or the
current support/face refinement; the final score can therefore be delayed in
an isolated source port. Preserve all other prediction responsibilities and
the existing parameter names. Skip the final semantic subhead inside the last
box head and invoke that same semantic subhead once after readback. Do not call
the entire box prediction head again, or replay a scoring head with Dropout/BN
twice. Actual zero-readback/native call count/state/score equality still needs
a new real-model preflight; this source graph alone does not prove it.

For a fixed-geometry source experiment, preserve the selected geometry provider
and compare visibility of its final evidence under a common same-start/same
budget semantic adaptation. This is a control choice, not a claim that semantic
gradients have been proved to damage geometry. Count the additional adaptation
history explicitly; a new stage is not a free continuation from official PV.

## Supervision and scope limits

Current G preserves every Hungarian-matched query, including other instances,
and replaces only unmatched final-IoU>.5 root CE targets. Its native ScanRefer
root target is .6positive+.2modify+.2pron+.1rel with unnormalized target mass;
its .5/7 coefficient is not a generic normalized binary probability target.
Do not silently substitute BCE or renormalize this target when adding geometry
quality. Any proposed within-root quality ranking would have to use actual
final boxes, stop-gradient training qualifications and the actual bbs score,
protect matched others and uncertain candidates, and be tested separately from
structural readback. It has not been implemented or approved as a result here.

All256 candidates remain. Existing score maps and two-stage object inputs are
native protocol dependencies and must remain disclosed. No teacher, GT gate,
second deployed ranking, candidate truncation or new loss was added by this
inspection. Same checkpoint ScanRefer5615/4754 and later independent Nr/Sr
remain unmet. The current metric best remains5616/4506 pending face terminal.
'''
(local/'READBACK_INTERFACE.md').write_text(text,encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' read-only pinned final-score dependency witness prepared while sole face fit runs: last semantic head precedes Mask/tail, has BN/Dropout; final geometry not read by current score. Future minimal delayed single semantic-subhead call identified, not implemented or GPU-tested; current source/job unchanged. No additional SSH query.\n')
print(json.dumps(dict(status=record['status'],source_files=len(paths),active_fit_changed=False,new_runtime_calls=0)))
