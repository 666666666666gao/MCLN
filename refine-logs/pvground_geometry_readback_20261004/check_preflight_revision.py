"""Bind the revised source draft; AST checks are not a native runtime witness."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

local = Path(__file__).resolve().parent
assert not (local / 'READBACK_PREFLIGHT_SOURCE_CHECK.json').exists()
old_review = json.loads((local / 'READBACK_SOURCE_REVIEW.json').read_bytes())
previous = json.loads((local / 'review_publication.json').read_bytes())
prefix = 'refine-logs/pvground_geometry_readback_20261004/'
names = ['pvground_boundary_evidence_readback.py', 'install_boundary_evidence_readback.py',
         'native_root_bbs.py', 'readback_preflight_checks.py', 'READBACK_PREFLIGHT_SOURCE_SCOPE.md']
files = {}
for name in names:
    raw = (local / name).read_bytes()
    if name.endswith('.py'):
        ast.parse(raw.decode('utf-8'), filename=name, feature_version=(3, 7))
    files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
trace = local / '.aris/traces/experiment-bridge/2026-10-04_readback_source_run01/reviewed_inputs_from_git'
trace.mkdir()
restored_old_inputs = {}
for name in names[:3]:
    raw = subprocess.check_output(['git', '-C', 'C:/Users/gb/.codex_mcln_g0_20260905',
        'show', previous['github_main'] + ':' + prefix + name])
    expected = next(item for item in old_review['reviewed_files'] if Path(item['path']).name == name)
    assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
    (trace / name).write_bytes(raw)
    restored_old_inputs[name] = dict(bytes=len(raw), sha256=expected['sha256'])
face = local.parent / 'pvground_face_conditioned_20261004'
active_review = json.loads((face / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
for item in active_review['reviewed_files']:
    raw = Path(item['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
record = dict(status='REVISED_PREFLIGHT_SOURCE_AST_ONLY', time_cst=datetime.datetime.now().astimezone().isoformat(),
    files=files, python37_ast_files=4, previous_source_review_applies_to_old_bytes_only=True,
    reviewed_inputs_restored_from_exact_committed_git=previous['github_main'], restored_old_inputs=restored_old_inputs,
    old_source_review_reexecuted=False, active_face_local_source_bindings_verified=len(active_review['reviewed_files']),
    native_factory_constructed=False, selected_geometry_provider=False, real_gpu_forwards=0,
    optimizer_updates=0, runtime_pass=False, accuracy_result=False,
    manual_parameters=96672, manual_parameter_tensors=23, measured_parameters=False,
    full_factory_runner_source_gate_pending=True)
(local / 'READBACK_PREFLIGHT_SOURCE_CHECK.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = face / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(readback_current_revision=str(local / 'READBACK_PREFLIGHT_SOURCE_CHECK.json'),
    readback_current_revision_review='FULL_FACTORY_RUNNER_SOURCE_GATE_PENDING',
    readback_runtime_checked=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (face / 'NEXT_CONTINUATION.md').open('a', encoding='utf-8') as stream:
    stream.write('\nIsolated readback draft revised: exact per-map bbs reduction order, same-capacity44-D geometry visibility control, unexecuted actual-model preflight helpers. Check READBACK_PREFLIGHT_SOURCE_CHECK.json; old SOURCE_ONLY review binds old bytes preserved from main75eaf6b, not this revision. Current factory/provider/GPU still pending; active face sources verified unchanged.\n')
print(json.dumps(dict(status=record['status'], python37_ast_files=4,
    active_face_bindings_verified=record['active_face_local_source_bindings_verified'],
    full_source_gate_pending=True, runtime_pass=False, accuracy_result=False)), flush=True)
