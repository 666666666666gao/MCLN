"""R4 static-only hashes, AST/control-flow, schema and arithmetic checks."""
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

out=Path(__file__).resolve().parent
runner=out.parent
prior=runner.parents[1]/'pvground_selected_mask_training_20261009'
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def tree(path):
    return ast.parse(path.read_bytes(),filename=str(path))
def statement(module,line):
    return next(node for node in ast.walk(module) if isinstance(node,ast.stmt) and node.lineno==line)
def simple_assignment(module,name):
    return next(node for node in module.body if isinstance(node,ast.Assign) and
        len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name)
def attr_calls(module,attribute):
    return [node for node in ast.walk(module) if isinstance(node,ast.Call) and
        isinstance(node.func,ast.Attribute) and node.func.attr==attribute]

manifest=read_json(out/'INPUT_MANIFEST_R4.json')
preserved=read_json(out/'R4_PRIOR_PRESERVATION_BASELINE.json')
for item in preserved:
    assert sha(Path(item['path']))==item['sha256'],item['path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
assert (out/'R4_EXACT_REQUEST.txt').read_bytes()==(runner/'LAUNCH_SOURCE_AUDIT_REQUEST.txt').read_bytes()
python_items=[item for item in manifest if Path(item['path']).suffix=='.py']
for item in python_items:
    tree(out/item['snapshot'])
tools_expected={
'finalize_parent.py':'0b7218d1a2802977984e06eddcab95f67a14b9a09c78e9bed1074a683ba0e5f4',
'launch_span_authorized.py':'1e061846d96f2107f5a760e937182d925964e989713739dea49aabdb66175328',
'observe_span_authorized.py':'ac50ac13b92f8ebab6d6c9cd68d3117357a2899a134c232aff8082f4a1d83d5f',
'collect_span_authorized.py':'25f847c47ad76131b58cbab83ba6475ab346e6ef7bd9f0ccc3d62d4975f5b6d6'}
prep=read_json(runner/'LAUNCH_TOOL_PREPARATION.json')
assert prep['source_sha256']==tools_expected
for name,digest in tools_expected.items():
    assert sha(runner/name)==digest
spec=read_json(runner/'pair_spec_template.json')
canonical=read_json(runner/'RUNNER_PREPARATION.json')
r3_manifest=read_json(out/'INPUT_MANIFEST_R3.json')
r3_bound={Path(row['path']).name:row['sha256'] for row in r3_manifest
    if Path(row['path']).parent.resolve()==runner and
       (Path(row['path']).suffix=='.py' or Path(row['path']).name in ('pair_spec_template.json','RUNNER_PREPARATION.json'))}
assert len(r3_bound)==13
for name,digest in spec['new_runner_files'].items():
    assert sha(runner/name)==digest==canonical['files'][name]==r3_bound[name]
for name in ('pair_spec_template.json','RUNNER_PREPARATION.json'):
    assert sha(runner/name)==r3_bound[name]
assert len(spec['new_runner_files'])==11 and spec['parent_selection_status'].startswith('PENDING_')
prior_spec=read_json(prior/'pair_spec.json')
assert prior_spec['support_modes']==['content','selected_query']
assert prior_spec['env_spec_sha256']==spec['env_spec_sha256']
postrun=read_json(prior/'postrun_source_review_20261009/SOURCE_REVIEW_R3.json')
assert postrun['execution_scope']=='SOURCE_ONLY' and postrun['verdict']=='PASS' and not postrun['blocking_findings']
postrun_bound={Path(row['path']).resolve():row['sha256'] for row in postrun['reviewed_files']}
for name in ('analyze_selected_pair.py','inspect_closed_terminals_authorized.py','collect_closed_pair_authorized.py'):
    path=prior/'postrun_source'/name
    assert sha(path)==postrun_bound[path.resolve()]

finalizer=tree(runner/'finalize_parent.py')
launcher=tree(runner/'launch_span_authorized.py')
observer=tree(runner/'observe_span_authorized.py')
collector=tree(runner/'collect_span_authorized.py')
controller=tree(runner/'span_controller.py')
# All explicit finalizer prerequisites precede its first output write.
first_final_write=min(node.lineno for node in attr_calls(finalizer,'write_text'))
assert first_final_write==50
assert all(isinstance(statement(finalizer,line),ast.Assert) for line in
    (9,11,12,14,15,17,18,21,23,25,26,29,30,31,32,33,34,35,38,41,43))
assert all(line<first_final_write for line in (35,41,43))
update=next(node for node in attr_calls(finalizer,'update') if node.lineno==44)
updated_fields=[kw.arg for kw in update.keywords]
assert set(updated_fields)==set(('parent_selection_status','parent_support_terminal',
    'parent_support_terminal_sha256','starting_hits','parent_decision_sha256','previous_completed_experiment'))
rank_expr=simple_assignment(finalizer,'rank').value
assert isinstance(rank_expr,ast.Lambda) and isinstance(rank_expr.body,ast.Tuple)
rank=eval(compile(ast.Expression(rank_expr),'<actual-template-hit-ranking>','eval'),{})
assert rank(spec['starting_hits'])==(True,5599,4859)
# Current final spec is deliberately unavailable; no top-level script is run.
local_state={name:(runner/name).exists() for name in
    ('pair_spec.json','FINAL_SPEC_RECEIPT.json','preflight_launch.json','preflight_wait.json',
     'preflight_complete/preflight.json','preflight_results/EXPERIMENT_AUDIT.json','fit_launch.json')}
assert not any(local_state.values())
first_connect=min(node.lineno for node in attr_calls(launcher,'connect'))
assert first_connect==63
assert all(isinstance(statement(launcher,line),ast.Assert) for line in
    (19,22,24,26,28,29,32,34,36,39,40,42,43,46,48,50,51,52,53))
assert all(line<first_connect for line in (24,26,34,36,40,48,50,52,53))
fit_branch=next(node for node in launcher.body if isinstance(node,ast.If) and node.lineno==37)
assert 'fit' in ast.dump(fit_branch.test)
files=list(spec['new_runner_files'])+['pair_spec.json','FINAL_SPEC_RECEIPT.json','LAUNCH_SOURCE_REVIEW.json']
assert len(files)==len(set(files))==14 and files.count('span_controller.py')==1
assert not set(files).intersection(tools_expected)
upload_branch=next(node for node in launcher.body if isinstance(node,ast.If) and node.lineno==110)
assert 'preflight' in ast.dump(upload_branch.test)
assert len([node for node in ast.walk(upload_branch) if isinstance(node,ast.For)])==1
for name in spec['new_runner_files']:
    assert 'MCLN_SSH_PASSWORD' not in (runner/name).read_text(encoding='utf-8')
embedded={}
for name,module in (('launcher',launcher),('observer',observer),('collector',collector)):
    code=ast.literal_eval(simple_assignment(module,'code').value)
    embedded[name]=ast.parse(code,filename=name+'::<remote-source-not-executed>')
    assert not any(isinstance(node,(ast.Import,ast.ImportFrom)) and
        any(alias.name.split('.')[0] in ('torch','numpy') for alias in node.names)
        for node in ast.walk(embedded[name]))
remote_launcher=embedded['launcher']
assert all(isinstance(statement(remote_launcher,line),ast.Assert) for line in (4,5,6,7,8,10,12,13,18))
assert 'spec_sha256' in ast.dump(remote_launcher) and 'new_runner_files' in ast.dump(remote_launcher)
formal_expr=simple_assignment(launcher,'formal_bytes').value
formal_bytes=eval(compile(ast.Expression(formal_expr),'<static-storage-arithmetic>','eval'),{'math':math})
assert formal_bytes==321733888
preflight_base=256*1024**2
fit_base=formal_bytes+130*1024**2
# Explicit raw exit receipts precede success assertions.
failure_receipts={}
for name,module,triples in (
    ('launcher',launcher,((106,107,108,109),(124,125,126,127),(132,133,134,135))),
    ('observer',observer,((51,52,53,54),))):
    for lines in triples:
        assert all(isinstance(statement(module,line),ast.Expr) for line in lines[:-1])
        assert isinstance(statement(module,lines[-1]),ast.Assert)
        failure_receipts[name+'_'+str(lines[-1])]=list(lines)
assert isinstance(statement(controller,57),ast.Expr) and isinstance(statement(controller,58),ast.If)
assert isinstance(statement(controller,61),ast.Raise)
loop=next(node for node in observer.body if isinstance(node,ast.While))
stores={node.id for node in ast.walk(loop) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}
assert 'code' not in stores and 'exit_code' in stores
assert [node.lineno for node in attr_calls(observer,'sleep')]==[26,70]
assert ast.literal_eval(attr_calls(observer,'sleep')[-1].args[0])==240
breaks=[node for node in ast.walk(loop) if isinstance(node,ast.Break)]
assert len(breaks)==1 and breaks[0].lineno==63
remote_observer=embedded['observer']
identity=statement(remote_observer,5)
assert isinstance(identity,ast.Assert)
assert any(isinstance(node,ast.Compare) and any(isinstance(op,ast.In) for op in node.ops)
    for node in ast.walk(identity.test))
identity_finding=dict(id='R4-B1',line=31,result='SUBSTRING_MEMBERSHIP_NOT_EXACT_COMMAND_MATCH',
    actual_wrong_process_observed=False,requires_minimal_exact_argv_binding=True)
assert all(isinstance(statement(collector,line),ast.Assert) for line in (20,21,24,49,50,55,57,59,68,70,75))
assert max((20,21))<min(node.lineno for node in attr_calls(collector,'connect'))
assert statement(collector,63).items[0].context_expr.args[0].s=='xb'
remote_collector=embedded['collector']
assert isinstance(statement(remote_collector,4),ast.Assert) and isinstance(statement(remote_collector,5),ast.Assert)
# Timing provenance is an existing local receipt, not a new remote observation.
historical=read_json(prior/'fit_launch.json')
historical_first=datetime.datetime.fromisoformat(historical['time_cst'])+datetime.timedelta(seconds=historical['first_check_seconds'])
assert historical['later_poll_seconds']==240
assert historical_first.hour==17 and historical_first.minute==14 and historical_first.second==27
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',checks_passed=True,
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',python_version=sys.version,
    python_executable=sys.executable,prior_artifacts_preserved=len(preserved),
    inputs_snapshotted=len(manifest),python_files_parsed=len(python_items),
    remote_strings_parsed_not_executed=list(embedded),canonical_R3_files_unchanged=len(r3_bound),
    new_tool_hashes=tools_expected,postrun_three_source_hashes_match_R3_review=True,
    final_spec_updated_fields=updated_fields,actual_template_parent_rank=list(rank(spec['starting_hits'])),
    finalizer_writes_only_after_static_gate_lines=True,source_and_actual_fit_gates_precede_SSH=True,
    uploaded_files_if_future_preflight=files,source_upload_once_preflight_only=True,
    credential_accessing_local_tools_not_uploaded=True,remote_source_string_preserved_across_observer_iterations=True,
    failure_receipts_before_assert=failure_receipts,observer_identity_finding=identity_finding,
    formal_array_and_archive_reserve_bytes=formal_bytes,preflight_base_reserve_bytes=preflight_base,
    fit_base_reserve_bytes=fit_base,upload_bytes_added_separately=True,
    preflight_estimate_seconds=480,preflight_first_check_seconds=300,
    fit_estimate_source='actual prior complete_fit/fit_status.json elapsed_seconds; not yet available',
    fit_first_check_before_estimated_end_seconds=300,later_poll_seconds=240,
    historical_original_first_observation_cst=historical_first.isoformat(),
    historical_receipt_read_only=True,no_current_remote_query=True,local_pending_state=local_state,
    torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
    paramiko_imported='paramiko' in sys.modules,project_modules_imported=False,
    actual_deployment=False,actual_M0=False,actual_metrics=False,launch_approved=False)
assert not result['torch_imported'] and not result['numpy_imported'] and not result['paramiko_imported']
(out/'R4_STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
