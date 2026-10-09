"""R2 AST/hash verification only. No Torch import, model build, or diagnostic replay."""
import ast
import datetime
import difflib
import hashlib
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, content):
    (HERE / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


r1_manifest = json.loads((HERE / 'R1_ARTIFACT_HASHES.json').read_text(encoding='utf-8-sig'))
for item in r1_manifest:
    assert sha(Path(item['path'])) == item['sha256']
r1_hashes = json.loads((HERE / 'INPUT_HASHES.json').read_bytes())
changed_names = {'span_refinement_model.py', 'METHOD_PROPOSAL.md'}
r2_hashes = {}
for name, record in r1_hashes.items():
    path = Path(name)
    actual = sha(path)
    if not (path.parent == ROOT and path.name in changed_names):
        assert actual == record['sha256'], str(path)
    r2_hashes[str(path)] = dict(sha256=actual, bytes=path.stat().st_size)
request = ROOT / 'SOURCE_R2_REQUEST.txt'
r2_hashes[str(request)] = dict(sha256=sha(request), bytes=request.stat().st_size)
assert sha(request) == sha(HERE / 'SOURCE_R2_REQUEST.exact.txt')
assert sha(ROOT / 'span_refinement_model.py') == '1139094df8f282715a795469cedc0dd6c4c95fea3f6b68ce13ab56b4a6ffb68b'
snapshot_dir = HERE / 'r2_source_snapshots'
snapshot_dir.mkdir(exist_ok=True)
snapshots = []
for name in ('span_refinement_model.py', 'METHOD_PROPOSAL.md', 'extremal_span_mixer.py',
             'matched_span_objective.py', 'SOURCE_R2_REQUEST.txt'):
    target = snapshot_dir / name
    target.write_bytes((ROOT / name).read_bytes())
    snapshots.append(dict(source=str(ROOT / name), snapshot=str(target), sha256=sha(target), bytes=target.stat().st_size))
write('R2_INPUT_HASHES.json', r2_hashes)
write('R2_SOURCE_SNAPSHOTS.json', snapshots)

old = (HERE / 'r1_source_snapshots/span_refinement_model.py').read_text(encoding='utf-8-sig')
new = (snapshot_dir / 'span_refinement_model.py').read_text(encoding='utf-8-sig')
(HERE / 'R2_WRAPPER.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
    fromfile='r1_source_snapshots/span_refinement_model.py', tofile='r2_source_snapshots/span_refinement_model.py')), encoding='utf-8')
old_proposal = (HERE / 'r1_source_snapshots/METHOD_PROPOSAL.md').read_text(encoding='utf-8-sig')
new_proposal = (snapshot_dir / 'METHOD_PROPOSAL.md').read_text(encoding='utf-8-sig')
(HERE / 'R2_PROPOSAL.diff').write_text(''.join(difflib.unified_diff(old_proposal.splitlines(True), new_proposal.splitlines(True),
    fromfile='r1_source_snapshots/METHOD_PROPOSAL.md', tofile='r2_source_snapshots/METHOD_PROPOSAL.md')), encoding='utf-8')
a, b = ast.parse(old), ast.parse(new)
old_top = {n.name: n for n in a.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new_top = {n.name: n for n in b.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
assert set(new_top) - set(old_top) == {'span_checkpoint_payload', 'restore_span_checkpoint'}
assert ast.dump(old_top['apply_span_mixer']) == ast.dump(new_top['apply_span_mixer'])
old_methods = {n.name: n for n in old_top['SpanRefinementModel'].body if isinstance(n, ast.FunctionDef)}
new_methods = {n.name: n for n in new_top['SpanRefinementModel'].body if isinstance(n, ast.FunctionDef)}
assert old_methods.keys() == new_methods.keys() == {'__init__', 'forward'}
assert ast.dump(old_methods['__init__']) == ast.dump(new_methods['__init__'])
forward = new_methods['forward']
assert [arg.arg for arg in forward.args.args] == ['self', 'inputs']
assert forward.args.vararg is None and forward.args.kwarg is None and forward.args.kwonlyargs == []
parent_calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self' and n.func.attr == 'parent']
assert len(parent_calls) == 1
assert len(parent_calls[0].args) == 1 and isinstance(parent_calls[0].args[0], ast.Name)
assert parent_calls[0].args[0].id == 'inputs' and parent_calls[0].keywords == []
bound = inspect.Signature([inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
                          inspect.Parameter('inputs', inspect.Parameter.POSITIONAL_OR_KEYWORD)]).bind(object(), {'points': 'symbolic'})
assert bound.arguments['inputs'] == {'points': 'symbolic'}

payload = new_top['span_checkpoint_payload']
ret = next(n for n in payload.body if isinstance(n, ast.Return))
assert isinstance(ret.value, ast.Call) and isinstance(ret.value.func, ast.Name) and ret.value.func.id == 'dict'
assert [k.arg for k in ret.value.keywords] == ['source_mode', 'mixer_state', 'optimizer', 'step', 'parent_identity']
restore = new_top['restore_span_checkpoint']
active = [n for n in restore.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
assert len(active) == 5 and all(isinstance(n, ast.Assert) for n in active[:2])
assert ast.unparse(active[0].test) == "payload['source_mode'] == model.mixer.source_mode"
assert ast.unparse(active[1].test) == "payload['parent_identity'] == parent_identity"
assert ast.unparse(active[2]) == "model.mixer.load_state_dict(payload['mixer_state'], strict=True)"
assert ast.unparse(active[3]) == "optimizer.load_state_dict(payload['optimizer'])"
assert ast.unparse(active[4]) == "return payload['step']"
assert not any(isinstance(n, (ast.Try, ast.If)) for function in (payload, restore, forward) for n in ast.walk(function))
for function in (payload, restore):
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'parent' for n in ast.walk(function))
assert 'torch' not in sys.modules
for item in r1_manifest:
    assert sha(Path(item['path'])) == item['sha256']
result = dict(status='PASS_STATIC_ONLY_R1_B1_B2_RESOLVED',
    generated_at=datetime.datetime.now().astimezone().isoformat(), python=sys.executable,
    wrapper_sha256=sha(snapshot_dir / 'span_refinement_model.py'), proposal_sha256=sha(snapshot_dir / 'METHOD_PROPOSAL.md'),
    verified_r1_artifacts_unchanged=len(r1_manifest), r2_input_files=len(r2_hashes),
    scope='AST and hash comparisons only; no new numerical diagnostic or runtime model execution',
    native_positional_dict_signature='PASS', parent_forward_call_count=1,
    source_mode_payload=True, parent_identity_payload=True, optimizer_payload=True, update_step_payload=True,
    checks_before_state_loading=['source_mode equality', 'parent_identity equality'], head_state_loading_strict=True,
    r1_apply_span_mixer_unchanged=True, r1_constructor_unchanged=True,
    mixer_and_objective_bytes_unchanged=True, diagnostic_inputs_bytes_unchanged=True,
    neural_architecture_reused_from_r1=dict(parameters=29793, trainable_tensors=14),
    actual_model_builds=0, actual_neural_forwards=0, actual_optimizer_updates=0,
    actual_serialization_or_restore=False, actual_gradient_witness=False,
    actual_M0='NOT_AVAILABLE', runner_present=False, launch_approved=False,
    diagnostic_replayed=False, r1_failed_invocations_preserved=True, blocking_findings=[])
write('R2_STATIC_VERIFICATION.json', result)
print(json.dumps(result))
