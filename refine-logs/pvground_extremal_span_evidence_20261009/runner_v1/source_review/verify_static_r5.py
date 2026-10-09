"""Narrow R5 stdlib-only source/argv verification, not a runtime witness."""
import ast
import copy
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import sys

out=Path(__file__).resolve().parent
runner=out.parent
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def assigned(module,name):
    return next(node for node in ast.walk(module) if isinstance(node,ast.Assign) and
        len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name)
def replace_once(source,before,after):
    assert source.count(before)==1,before
    return source.replace(before,after)
def adump(node):
    return ast.dump(node,include_attributes=False)

manifest=read_json(out/'INPUT_MANIFEST_R5.json')
preserved=read_json(out/'R5_PRIOR_PRESERVATION_BASELINE.json')
for item in preserved:
    assert sha(Path(item['preserved_path']))==item['sha256'],item['preserved_path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
assert (out/'R5_EXACT_REQUEST.txt').read_bytes()==(runner/'LAUNCH_R5_REQUEST.txt').read_bytes()
assert sha(out/'LAUNCH_SOURCE_REVIEW_R4.json')=='711ff42a4d7c1e77d5636dc9adbaf101f23cc2529bd888bea402cb33a1589cdb'
assert sha(out/'LAUNCH_SOURCE_REVIEW_R4.md')=='726f0390ad06cab10beb95c354420c03324d6dba507ed583f69957d589f774a0'
assert (out/'LAUNCH_SOURCE_REVIEW_R4.md').read_bytes()==(out/'RAW_RESPONSE_R4.md').read_bytes()
old_manifest=read_json(out/'INPUT_MANIFEST_R4.json')
changed={'launch_span_authorized.py','observe_span_authorized.py','LAUNCH_TOOL_PREPARATION.json'}
unchanged=0
for item in old_manifest:
    path=Path(item['path'])
    if path.parent.resolve()==runner and path.name in changed:
        continue
    assert sha(path)==item['sha256'],str(path)
    unchanged+=1
expected_hashes={
'finalize_parent.py':'0b7218d1a2802977984e06eddcab95f67a14b9a09c78e9bed1074a683ba0e5f4',
'launch_span_authorized.py':'4cfe370c1a404012fab4d82ae37b7dba48292bd0d994fc094d34c2e4a3bfb975',
'observe_span_authorized.py':'8789bd232e261ff13cbe2fbcdba3855c98f233d14d895ace4e7bc3cc1530f7f0',
'collect_span_authorized.py':'25f847c47ad76131b58cbab83ba6475ab346e6ef7bd9f0ccc3d62d4975f5b6d6'}
prep=read_json(runner/'LAUNCH_TOOL_PREPARATION.json')
assert prep['source_sha256']==expected_hashes
for name,digest in expected_hashes.items():
    assert sha(runner/name)==digest
old_prep=read_json(out/'r4_snapshots/018_LAUNCH_TOOL_PREPARATION.json')
normalized=copy.deepcopy(prep)
normalized['source_sha256']=old_prep['source_sha256']
normalized['status']=old_prep['status']
assert normalized==old_prep
r3_manifest=read_json(out/'INPUT_MANIFEST_R3.json')
canonical=[item for item in r3_manifest if Path(item['path']).parent.resolve()==runner and
    (Path(item['path']).suffix=='.py' or Path(item['path']).name in ('pair_spec_template.json','RUNNER_PREPARATION.json'))]
assert len(canonical)==13
for item in canonical:
    assert sha(Path(item['path']))==item['sha256']
python_items=[item for item in manifest if Path(item['path']).suffix=='.py']
for item in python_items:
    ast.parse((out/item['snapshot']).read_bytes(),filename=item['path'])
old_launch=(out/'r4_snapshots/002_launch_span_authorized.py').read_text(encoding='utf-8')
new_launch=(runner/'launch_span_authorized.py').read_text(encoding='utf-8')
expected=replace_once(old_launch,
    "command = shlex.join(['flock', '-n', env['resource_limits']['gpu_lock'],\n    spec['runtime'] + '/venv/bin/python', '-B', '-u', spec['root'] + '/span_controller.py', '--phase', phase])",
    "controller_argv = [spec['runtime'] + '/venv/bin/python', '-B', '-u',\n                   spec['root'] + '/span_controller.py', '--phase', phase]\ncommand = shlex.join(['flock', '-n', env['resource_limits']['gpu_lock']] + controller_argv)")
expected=replace_once(expected,"process=process, screen=screen, resources=json.loads(raw),",
    "process=process, controller_argv=controller_argv, screen=screen, resources=json.loads(raw),")
assert expected==new_launch
old_observer=(out/'r4_snapshots/003_observe_span_authorized.py').read_text(encoding='utf-8')
new_observer=(runner/'observe_span_authorized.py').read_text(encoding='utf-8')
expected=replace_once(old_observer,
    "root=Path(sys.argv[1]);pid=sys.argv[2];phase=sys.argv[3];proc=Path('/proc')/pid;alive=proc.exists()",
    "root=Path(sys.argv[1]);pid=sys.argv[2];phase=sys.argv[3];expected=json.loads(sys.argv[4]);proc=Path('/proc')/pid;alive=proc.exists()")
expected=replace_once(expected,
    "command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode() if alive else ''",
    "argv=[part.decode() for part in (proc/'cmdline').read_bytes().split(bytes([0]))[:-1]] if alive else []")
expected=replace_once(expected,
    "assert not alive or str(root)+'/span_controller.py --phase '+phase in command",
    "assert not alive or argv==expected")
expected=replace_once(expected,
    "'-B', '-c', code, launch['root'], str(launch['controller_pid']), phase]), timeout=30)",
    "'-B', '-c', code, launch['root'], str(launch['controller_pid']), phase,\n        json.dumps(launch['controller_argv'])]), timeout=30)")
assert expected==new_observer
diff=''.join(difflib.unified_diff(old_launch.splitlines(True),new_launch.splitlines(True),
    fromfile='R4/launch_span_authorized.py',tofile='R5/launch_span_authorized.py'))
diff+=''.join(difflib.unified_diff(old_observer.splitlines(True),new_observer.splitlines(True),
    fromfile='R4/observe_span_authorized.py',tofile='R5/observe_span_authorized.py'))
(out/'R5_SOURCE.diff').write_text(diff,encoding='utf-8')
old_tree=ast.parse(old_launch)
launch=ast.parse(new_launch)
observe=ast.parse(new_observer)
old_args=assigned(old_tree,'command').value.args[0]
argv_expr=assigned(launch,'controller_argv').value
assert adump(argv_expr)==adump(ast.List(elts=old_args.elts[3:],ctx=ast.Load()))
joined=assigned(launch,'command').value.args[0]
assert isinstance(joined,ast.BinOp) and isinstance(joined.op,ast.Add)
assert adump(joined.left)==adump(ast.List(elts=old_args.elts[:3],ctx=ast.Load()))
assert isinstance(joined.right,ast.Name) and joined.right.id=='controller_argv'
receipt=assigned(launch,'receipt').value
argv_field=next(kw for kw in receipt.keywords if kw.arg=='controller_argv')
assert isinstance(argv_field.value,ast.Name) and argv_field.value.id=='controller_argv'
code=ast.literal_eval(assigned(observe,'code').value)
remote=ast.parse(code,filename='observer::<source-not-executed>')
identity=next(node for node in ast.walk(remote) if isinstance(node,ast.Assert))
assert isinstance(identity.test,ast.BoolOp) and isinstance(identity.test.op,ast.Or)
comparison=identity.test.values[1]
assert isinstance(comparison,ast.Compare) and len(comparison.ops)==1 and isinstance(comparison.ops[0],ast.Eq)
assert isinstance(comparison.left,ast.Name) and comparison.left.id=='argv'
assert isinstance(comparison.comparators[0],ast.Name) and comparison.comparators[0].id=='expected'
expected_expr=assigned(remote,'expected').value
assert adump(expected_expr)==adump(ast.parse('json.loads(sys.argv[4])',mode='eval').body)
query=next(node for node in ast.walk(observe) if isinstance(node,ast.Call) and
    isinstance(node.func,ast.Attribute) and node.func.attr=='exec_command')
last_arg=query.args[0].args[0].elts[-1]
assert adump(last_arg)==adump(ast.parse("json.dumps(launch['controller_argv'])",mode='eval').body)
loop=next(node for node in observe.body if isinstance(node,ast.While))
stores={node.id for node in ast.walk(loop) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}
assert 'code' not in stores and 'launch' not in stores and 'exit_code' in stores
# The two phases use actual template literals, solely for JSON/NUL metadata encoding.
spec=read_json(runner/'pair_spec_template.json')
roundtrips={}
for phase in ('preflight','fit'):
    argv=eval(compile(ast.Expression(argv_expr),'<template-controller-argv-metadata>','eval'),{'spec':spec,'phase':phase})
    encoded=b''.join(value.encode()+bytes([0]) for value in argv)
    decoded=[part.decode() for part in encoded.split(bytes([0]))[:-1]]
    assert decoded==json.loads(json.dumps(argv))
    roundtrips[phase]=dict(argv=argv,json_and_NUL_metadata_roundtrip_exact=True,actual_process_observed=False)
pending={}
for name in ('run_span_pair.py','span_controller.py'):
    module=ast.parse((runner/name).read_bytes())
    node=next(n for n in ast.walk(module) if isinstance(n,ast.Assert) and 'parent_selection_status' in adump(n.test))
    assert eval(compile(ast.Expression(node.test),'<pending-parent-predicate>','eval'),{'spec':spec}) is False
    pending[name]=dict(line=node.lineno,accepted=False,entry_executed=False)
state={name:(runner/name).exists() for name in
    ('pair_spec.json','FINAL_SPEC_RECEIPT.json','preflight_launch.json','preflight_wait.json',
     'preflight_complete/preflight.json','preflight_results/EXPERIMENT_AUDIT.json','fit_launch.json')}
assert not any(state.values())
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',generated_at=datetime.datetime.utcnow().isoformat()+'Z',
    checks_passed=True,python_executable=sys.executable,python_version=sys.version,
    prior_artifacts_preserved=len(preserved),prior_aliases_preserved_as_R4_files=2,
    inputs_snapshotted=len(manifest),python_files_parsed=len(python_items),
    other_R4_inputs_unchanged=unchanged,canonical_R3_publication_files_unchanged=13,
    new_tool_hashes=expected_hashes,exact_source_changes_only=True,
    launched_controller_argv_equals_previous_inline_argv=True,
    receipt_and_remote_expected_argv_bound=True,observer_identity_comparison='argv == expected',
    observer_remote_source_not_overwritten_during_polling=True,
    metadata_roundtrips_not_process_witnesses=roundtrips,pending_parent_predicates=pending,
    prior_gate_timing_failure_and_collection_paths_unchanged=True,
    local_pending_state=state,actual_wrong_process_observed=False,
    actual_deployment=False,actual_M0=False,actual_metrics=False,launch_approved=False,
    torch_imported='torch' in sys.modules,paramiko_imported='paramiko' in sys.modules,
    numpy_imported='numpy' in sys.modules,project_module_imports=0)
assert not result['torch_imported'] and not result['paramiko_imported'] and not result['numpy_imported']
(out/'R5_STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
print(diff)
