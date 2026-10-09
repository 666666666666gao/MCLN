"""Source-only retention verification; no result, model, SSH or deletion execution."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

out=Path(__file__).resolve().parent
toolroot=out.parent
local=toolroot.parent
span=local.parent/'pvground_extremal_span_evidence_20261009/runner_v1'
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def parse(path):
    return ast.parse(path.read_bytes(),filename=str(path))
def at(module,line,kind):
    return next(node for node in ast.walk(module) if isinstance(node,kind) and node.lineno==line)
def assign(module,name):
    return next(node for node in ast.walk(module) if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id==name for target in node.targets))
def calls(module,attribute):
    return [node for node in ast.walk(module) if isinstance(node,ast.Call) and
        isinstance(node.func,ast.Attribute) and node.func.attr==attribute]
def dict_keys(call):
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='dict'
    return {kw.arg for kw in call.keywords}
def string_keys(module,name):
    result=set()
    for node in ast.walk(module):
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id==name:
            if isinstance(node.slice,ast.Index) and isinstance(node.slice.value,ast.Str):
                result.add(node.slice.value.s)
    return result

manifest=read_json(out/'INPUT_MANIFEST.json')
baseline=read_json(out/'PRIOR_AUDIT_PRESERVATION_BASELINE.json')
for item in baseline:
    assert sha(Path(item['path']))==item['sha256'],item['path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
assert (out/'EXACT_REQUEST.txt').read_bytes()==(toolroot/'SOURCE_AUDIT_REQUEST.txt').read_bytes()
expected_hashes={
'prepare_retention_plan.py':'f29885a3cad0eb014f0c138719b74de8f02ede92d04b483159e9b69d4cff8292',
'retire_closed_pair_authorized.py':'9b4a6e40730fc66ebf92e7f40a140d25e88a95987ca701312f0aefc3cfc3c206'}
prep=read_json(toolroot/'TOOL_PREPARATION.json')
assert prep['source_sha256']==expected_hashes
for name,digest in expected_hashes.items():
    assert sha(toolroot/name)==digest
python_inputs=[row for row in manifest if Path(row['path']).suffix=='.py']
for row in python_inputs:
    parse(out/row['snapshot'])
reviewed=read_json(local/'postrun_source_review_20261009/SOURCE_REVIEW_R3.json')
bound={Path(row['path']).resolve():row['sha256'] for row in reviewed['reviewed_files']}
for name in ('analyze_selected_pair.py','inspect_closed_terminals_authorized.py','collect_closed_pair_authorized.py'):
    path=local/'postrun_source'/name
    assert sha(path)==bound[path.resolve()]
span_review=read_json(span/'source_review/LAUNCH_SOURCE_REVIEW_R5.json')
span_bound={Path(row['path']).resolve():row['sha256'] for row in span_review['reviewed_files']}
assert sha(span/'finalize_parent.py')==span_bound[(span/'finalize_parent.py').resolve()]
proposal=parse(toolroot/'prepare_retention_plan.py')
retire=parse(toolroot/'retire_closed_pair_authorized.py')
finalizer=parse(span/'finalize_parent.py')
analyzer=parse(local/'postrun_source/analyze_selected_pair.py')
inspector=parse(local/'postrun_source/inspect_closed_terminals_authorized.py')
collector=parse(local/'postrun_source/collect_closed_pair_authorized.py')
remote=ast.parse(ast.literal_eval(assign(retire,'code').value),filename='retirement::<not-executed>')
inspection_remote=ast.parse(ast.literal_eval(assign(inspector,'code').value),filename='inspector::<not-executed>')
# Actual prerequisites, all before the proposal's archive creation.
assert all(at(proposal,line,ast.Assert) for line in (10,12,13,16,17,20,21,24,26,29,30,33,35))
archive_creation=at(proposal,39,ast.Expr)
assert all(line<archive_creation.lineno for line in (12,13,17,20,24,26,30,35))
assert not calls(proposal,'connect') and not calls(proposal,'unlink')
assert all(at(proposal,line,ast.Assert) for line in (42,45,49))
assert at(proposal,48,ast.Expr).value.func.attr=='write_bytes'
choice_fields=dict_keys(at(proposal,50,ast.Expr).value.args[0])
assert {'arm','hits','remote_parent_path','parent_sha256','local_parent_path',
    'optimizer_preserved','full_cpu_state_tensors','total_support_updates'}.issubset(choice_fields)
first_choice=dict_keys(assign(proposal,'choices').value.elts[0])
assert {'arm','hits','remote_parent_path','parent_sha256','local_parent_path'}.issubset(first_choice)
assert next(kw.value.s for kw in assign(proposal,'choices').value.elts[0].keywords if kw.arg=='arm')=='protected_prior'
plan_fields=dict_keys(assign(proposal,'plan').value)
assert {'choices','proposed_parent','cpu_recount_sha256','checkpoint_inspection_sha256',
    'actual_audit_sha256','archived_terminals','automatic_promotion','current_best_marker_changed'}.issubset(plan_fields)
chosen=assign(proposal,'chosen').value
assert isinstance(chosen.func,ast.Name) and chosen.func.id=='max'
assert isinstance(chosen.args[0],ast.Name) and chosen.args[0].id=='choices'
assert next(kw.value.id for kw in chosen.keywords if kw.arg=='key')=='rank'
spec=read_json(local/'pair_spec.json')
template=read_json(span/'pair_spec_template.json')
assert spec['root']=='/root/autodl-tmp/pvground_selected_mask_training_20261009'
assert spec['support_modes']==['content','selected_query']
assert spec['warm_support_terminal_sha256']==template['parent_support_terminal_sha256']
assert spec['starting_hits']==template['starting_hits']==[5599,4859]
proposal_rank=eval(compile(ast.Expression(assign(proposal,'rank').value),'<static-ranking-lambda>','eval'),{})
finalizer_rank=eval(compile(ast.Expression(assign(finalizer,'rank').value),'<static-finalizer-ranking-lambda>','eval'),{})
assert proposal_rank({'hits':spec['starting_hits']})==finalizer_rank(spec['starting_hits'])==(True,5599,4859)
# Source schemas, not fabricated actual receipts.
producer=next(node for node in ast.walk(inspection_remote) if isinstance(node,ast.Call) and
    isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and
    node.func.value.id=='weights' and node.func.attr=='append')
inspection_weight_fields=dict_keys(producer.args[0])
assert {'arm','path','bytes','sha256','full_state_exact','optimizer_states',
    'full_cpu_state_tensors','total_support_updates'}.issubset(inspection_weight_fields)
recount_fields=dict_keys(assign(analyzer,'report').value)
assert {'status','rows','table','cumulative_support_updates','source_hashes'}.issubset(recount_fields)
required_decision={'status','current_pair_closed','automatic_promotion','parent_sha256',
    'remote_parent_path','local_parent_path','parent_hits','cpu_recount_sha256',
    'checkpoint_inspection_sha256','actual_audit_sha256'}
assert string_keys(finalizer,'decision')==required_decision
assert string_keys(retire,'decision').issubset(required_decision)
assert all(at(retire,line,ast.Assert) for line in (15,17,18,21,23,25,26,28,29,31,35,37,39,40,42,44,45,46,49,52,53,58,59,63,64,66))
assert max((17,23,25,31,35,40,42,45,46,63,66))<min(node.lineno for node in calls(retire,'connect'))
# Extract the actual lexical filter and exact filename-set assertion.
selection=assign(retire,'chosen').value
pattern=ast.literal_eval(selection.generators[0].ifs[0].args[0])
assert pattern==r'formal/batch_[0-9]{5}\.npz'
names=eval(compile(ast.Expression(at(retire,53,ast.Assert).test.comparators[0]),
    '<exact-filename-contract>','eval'),{})
assert len(names)==1189 and min(names)=='formal/batch_00000.npz' and max(names)=='formal/batch_09504.npz'
assert all(re.fullmatch(pattern,name) for name in names)
invalid=['/formal/batch_00000.npz','formal/sub/batch_00000.npz','formal/batch_00000.npz.tmp',
    'formal/batch_00000xnpz','formal/batch_0000.npz','content/terminal.pth','formal/rows.jsonl','../formal/batch_00000.npz']
assert not any(re.fullmatch(pattern,name) for name in invalid)
assert 'formal/batch_09500.npz' not in names and 'formal/batch_99999.npz' not in names
# Nonselected terminal names derive only from the proposal's checked two arms.
terminal_loop=at(retire,54,ast.For)
assert isinstance(terminal_loop.iter,ast.Subscript)
assert at(retire,55,ast.If).test.ops[0].__class__ is ast.NotEq
assert ast.literal_eval(at(proposal,42,ast.Assert).test.comparators[0])==('content','selected_query')
assert at(retire,60,ast.Expr).value.func.attr=='append'
# The embedded remote code has only one unlink call, fed by fully resolved checked paths.
unlinks=calls(remote,'unlink')
assert len(unlinks)==1
unlink=unlinks[0]
assert isinstance(unlink.func.value,ast.Name) and unlink.func.value.id=='path'
delete_loop=next(node for node in remote.body if isinstance(node,ast.For) and
    any(child is unlink for child in ast.walk(node)))
assert isinstance(delete_loop.iter,ast.Name) and delete_loop.iter.id=='paths'
validation=next(node for node in remote.body if isinstance(node,ast.For) and node is not delete_loop)
assert validation.lineno<delete_loop.lineno
path_assignment=next(node for node in validation.body if isinstance(node,ast.Assign))
assert isinstance(path_assignment.value,ast.Call) and path_assignment.value.func.attr=='resolve'
validation_asserts=[node for node in validation.body if isinstance(node,ast.Assert)]
assert len(validation_asserts)==2
assert "attr='parents'" in ast.dump(validation_asserts[0].test)
assert "id='protected'" in ast.dump(validation_asserts[0].test)
append=next(node for node in validation.body if isinstance(node,ast.Expr) and
    isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='append')
assert append.lineno>=max(node.lineno for node in validation_asserts)
protection_asserts=[node for node in remote.body if isinstance(node,ast.Assert) and
    any(isinstance(child,ast.Name) and child.id=='protected' for child in ast.walk(node))]
assert len(protection_asserts)==2
assert protection_asserts[0].lineno<delete_loop.lineno<protection_asserts[1].lineno
assert not calls(remote,'glob') and not calls(remote,'rglob') and not calls(remote,'rmtree')
assert not calls(remote,'run') and not calls(remote,'Popen') and not calls(remote,'system')
assert all(at(retire,line,ast.Expr) for line in (103,104,105))
assert at(retire,107,ast.Assert).lineno>105
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',checks_passed=True,
    python_executable=sys.executable,python_version=sys.version,
    prior_audit_artifacts_preserved=len(baseline),inputs_snapshotted=len(manifest),
    python_inputs_parsed=len(python_inputs),remote_retirement_source_parsed_not_executed=True,
    tool_hashes=expected_hashes,source_contract_hashes_match_prior_reviews=True,
    closed_audit_recount_inspection_gates_precede_archive=True,
    explicit_decision_source_and_recovery_gates_precede_SSH=True,
    inspected_terminal_fields=sorted(inspection_weight_fields),plan_fields=sorted(plan_fields),
    explicit_executor_decision_required_keys=sorted(required_decision),
    rank_uses_gate_then_wide_then_strict=True,protected_choice_first=True,
    exact_ties_use_stable_first_maximum=True,actual_score_ties_observed=False,
    archive_copies_complete_inspected_terminal_bytes_including_optimizer=True,
    filename_contract_count=len(names),filename_contract_first=min(names),filename_contract_last=max(names),
    request_end_offset_wording_corrected_from=9500,actual_last_offset=9504,
    negative_lexical_examples_checked=len(invalid),arbitrary_five_digit_offsets_rejected_by_exact_set=True,
    deletion_only_canonical_batches_and_unselected_inspected_terminal_paths=True,
    remote_fixed_root_and_all_paths_resolved_before_unlink=True,
    protected_parent_hashes_checked_before_and_after=True,
    recursive_or_shell_deletion=False,failure_stdout_stderr_exit_saved_before_assert=True,
    actual_results_read=False,actual_plan_created=False,actual_decision_created=False,
    actual_archive_copies=0,actual_deletions=0,actual_model_or_runtime_witness=False,
    actual_SSH=False,actual_GPU=False,torch_imported='torch' in sys.modules,
    numpy_imported='numpy' in sys.modules,paramiko_imported='paramiko' in sys.modules,
    project_imports=0,experimental_source_edits=0,launch_approved=False)
assert not result['torch_imported'] and not result['numpy_imported'] and not result['paramiko_imported']
(out/'STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
