"""Narrow R3 verification: no experiment, Torch, NumPy or project imports."""
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
assert sys.version_info[:2] == (3,7), sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=read_json(out/'INPUT_MANIFEST_R3.json')
prior=read_json(out/'R3_PRIOR_PRESERVATION_BASELINE.json')
for item in prior:
    assert sha(Path(item['path'])) == item['sha256'], item['path']
for item in manifest:
    assert sha(Path(item['path'])) == item['sha256'] == sha(out/item['snapshot']), item['path']
old_path=out/'r2_snapshots'/'007_paired_span_loop.py'
new_path=runner/'paired_span_loop.py'
old=old_path.read_text(encoding='utf-8')
new=new_path.read_text(encoding='utf-8')
before="selected = int(scores[bid].argmax())"
after="selected = int(scores[bid].argsort(descending=True)[0])"
assert old.count(before)==1 and new.count(after)==1
assert old.replace(before,after)==new
assert sha(new_path)=='33cec963d2a1277a85c861dfd8dd1049604e9123e75c6887c4c23209f6272154'
assert sha(old_path)=='c6b8328aee7ae7369d5636b11ce31eb06b53267af1358e4667cc3e8e88df0d5d'
diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
    fromfile='R2/paired_span_loop.py',tofile='R3/paired_span_loop.py'))
(out/'R3_LOOP.diff').write_text(diff,encoding='utf-8')
spec=read_json(runner/'pair_spec_template.json')
prep=read_json(runner/'RUNNER_PREPARATION.json')
oldspec=read_json(out/'r2_snapshots'/'012_pair_spec_template.json')
oldprep=read_json(out/'r2_snapshots'/'013_RUNNER_PREPARATION.json')
assert len(spec['new_runner_files'])==11
for name,expected in spec['new_runner_files'].items():
    assert sha(runner/name)==expected==prep['files'][name]
    ast.parse((runner/name).read_bytes(),filename=name)
    if name!='paired_span_loop.py':
        assert expected==oldspec['new_runner_files'][name]
normalized=copy.deepcopy(spec)
normalized['new_runner_files']['paired_span_loop.py']=oldspec['new_runner_files']['paired_span_loop.py']
assert normalized==oldspec
normalized=copy.deepcopy(prep)
normalized['files']['paired_span_loop.py']=oldprep['files']['paired_span_loop.py']
normalized['time_cst']=oldprep['time_cst']
assert normalized==oldprep
pending={}
for name in ('run_span_pair.py','span_controller.py'):
    tree=ast.parse((runner/name).read_bytes())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and 'parent_selection_status' in ast.dump(n.test))
    assert eval(compile(ast.Expression(node.test),'<static-parent-predicate>','eval'),{'spec':spec}) is False
    pending[name]={'line':node.lineno,'predicate':False,'entry_executed':False}
tree=ast.parse(new)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
witness=methods['support_input_witness']
evaluate=methods['evaluate']
selected=next(n.value.args[0] for n in ast.walk(witness) if isinstance(n,ast.Assign) and
    len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='selected')
ranked=next(n.value for n in ast.walk(evaluate) if isinstance(n,ast.Assign) and
    len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ranked')
assert isinstance(selected,ast.Subscript)
assert ast.dump(selected.value)==ast.dump(ranked)
assert ast.literal_eval(selected.slice.value)==0
assert "self.support_input_witness(parent, pair, inputs, batch, indices, targets) if preflight else None" in new
assert "raw_logits_scope='native winner plus every original matched query; foreground/members cover all256'" in new
publication=[n for n in manifest if Path(n['path']).parent==runner and
    (n['path'].endswith('.py') or Path(n['path']).name in ('pair_spec_template.json','RUNNER_PREPARATION.json'))]
assert len(publication)==13
result=dict(scope='STATIC_ONLY',generated_at=datetime.datetime.utcnow().isoformat()+'Z',
    python_version=sys.version,python_executable=sys.executable,
    prior_R1_R2_artifacts_preserved=len(prior),R3_inputs_snapshotted=len(manifest),
    runner_python_files_parsed=11,runner_source_hashes_bound=11,publication_files=publication,
    exact_one_line_source_change=True,new_selector_ranking_AST_equals_formal=True,selected_rank=0,
    no_new_tie_branch_or_fallback=True,other_10_sources_unchanged=True,
    only_spec_change='paired loop hash',only_preparation_changes=['paired loop hash','time_cst'],
    pending_parent_rejections=pending,preflight_only_call_and_explicit_logit_scope_unchanged=True,
    actual_tie_or_disagreement_observed=False,actual_npz_capture=False,actual_M0=False,
    actual_gradient=False,actual_save_restore=False,actual_metrics=False,launch_approved=False,
    torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,checks_passed=True)
assert not result['torch_imported'] and not result['numpy_imported']
(out/'R3_STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
print(diff)

