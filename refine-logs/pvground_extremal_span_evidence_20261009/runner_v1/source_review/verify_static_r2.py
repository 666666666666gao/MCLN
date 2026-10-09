"""R2 STATIC_ONLY review verifier; stdlib AST/hash/diff only."""
import ast
import copy
import datetime
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import sys

out = Path(__file__).resolve().parent
runner = out.parent
assert sys.version_info[:2] == (3, 7), sys.version

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = read_json(out / 'INPUT_MANIFEST_R2.json')
preserved = read_json(out / 'R2_R1_PRESERVATION_BASELINE.json')
for item in preserved:
    assert digest(Path(item['path'])) == item['sha256'], item['path']
for item in manifest:
    assert digest(Path(item['path'])) == item['sha256'] == digest(out / item['snapshot']), item['path']

old_manifest = read_json(out / 'INPUT_MANIFEST.json')
old_by_path = {item['path']: item for item in old_manifest}
old_by_name = {Path(item['path']).name: item for item in old_manifest}
parsed = []
for item in manifest:
    if item['path'].endswith('.py'):
        ast.parse(Path(item['path']).read_bytes(), filename=item['path'])
        parsed.append(item['path'])

old_path = out / 'snapshots' / '009_paired_span_loop.py'
new_path = runner / 'paired_span_loop.py'
old_text = old_path.read_text(encoding='utf-8')
new_text = new_path.read_text(encoding='utf-8')
assert digest(old_path) == 'e5ae46b9688cf9a2867a21c34d818172491a7b2bf0cb06a0f82e20bb3ec50e8a'
assert digest(new_path) == 'c6b8328aee7ae7369d5636b11ce31eb06b53267af1358e4667cc3e8e88df0d5d'
(out / 'R2_LOOP.diff').write_text(''.join(difflib.unified_diff(old_text.splitlines(True),
    new_text.splitlines(True), fromfile='R1/paired_span_loop.py', tofile='R2/paired_span_loop.py')), encoding='utf-8')

old_tree, new_tree = ast.parse(old_text), ast.parse(new_text)
old_class = next(n for n in old_tree.body if isinstance(n, ast.ClassDef))
new_class = next(n for n in new_tree.body if isinstance(n, ast.ClassDef))
old_methods = {n.name: n for n in old_class.body if isinstance(n, ast.FunctionDef)}
new_methods = {n.name: n for n in new_class.body if isinstance(n, ast.FunctionDef)}
assert set(new_methods) - set(old_methods) == {'support_input_witness'}
unchanged_methods = [name for name in old_methods if ast.dump(old_methods[name]) == ast.dump(new_methods[name])]
assert set(old_methods) - set(unchanged_methods) == {'step'}

class StripReceiptAdditions(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'support_input_witness':
            return None
        return self.generic_visit(node)
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'support_witness':
            return None
        return self.generic_visit(node)
    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [item for item in node.keywords if item.arg not in
                         ('upstream_group_grad_norms', 'support_input_witness')]
        return node

stripped = StripReceiptAdditions().visit(copy.deepcopy(new_tree))
assert ast.dump(stripped) == ast.dump(old_tree), 'Unexpected computation change'

spec = read_json(runner / 'pair_spec_template.json')
preparation = read_json(runner / 'RUNNER_PREPARATION.json')
old_spec = read_json(out / old_by_name['pair_spec_template.json']['snapshot'])
old_preparation = read_json(out / old_by_name['RUNNER_PREPARATION.json']['snapshot'])
assert len(spec['new_runner_files']) == 11
for name, expected in spec['new_runner_files'].items():
    assert digest(runner / name) == expected == preparation['files'][name]
    if name != 'paired_span_loop.py':
        assert digest(runner / name) == old_spec['new_runner_files'][name]
normalized_spec = copy.deepcopy(spec)
normalized_spec['new_runner_files']['paired_span_loop.py'] = old_spec['new_runner_files']['paired_span_loop.py']
assert normalized_spec == old_spec
normalized_preparation = copy.deepcopy(preparation)
normalized_preparation['files']['paired_span_loop.py'] = old_preparation['files']['paired_span_loop.py']
normalized_preparation['time_cst'] = old_preparation['time_cst']
assert normalized_preparation == old_preparation

pending_rejections = {}
for name in ('run_span_pair.py', 'span_controller.py'):
    module = ast.parse((runner / name).read_bytes())
    test = next(n for n in ast.walk(module) if isinstance(n, ast.Assert) and
                'parent_selection_status' in ast.dump(n.test))
    assert eval(compile(ast.Expression(test.test), '<static-parent-predicate-only>', 'eval'), {'spec': spec}) is False
    pending_rejections[name] = dict(line=test.lineno, actual_entry_executed=False, predicate=False)

witness = new_methods['support_input_witness']
step = new_methods['step']
witness_assignment = next(n for n in step.body if isinstance(n, ast.Assign) and
    isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'support_witness')
assert isinstance(witness_assignment.value, ast.IfExp)
assert ast.dump(witness_assignment.value.test) == ast.dump(ast.Name(id='preflight', ctx=ast.Load()))
assert isinstance(witness_assignment.value.orelse, ast.NameConstant) and witness_assignment.value.orelse.value is None
witness_calls = [n for n in ast.walk(new_tree) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Attribute) and n.func.attr == 'support_input_witness']
assert len(witness_calls) == 1
array_call = next(n.value for n in ast.walk(witness) if isinstance(n, ast.Assign) and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'arrays')
array_keys = [n.arg for n in array_call.keywords]
required_arrays = ['raw_points', 'superpoints', 'scores', 'foreground_sp', 'reference_valid',
    'observed_queries', 'observed_text_logits', 'observed_query_logits', 'alpha', 'mask_center',
    'mask_size', 'native_center', 'native_size', 'matched_queries', 'matched_gt_ids',
    'actual_gt_boxes', 'actual_gt_masks']
assert set(array_keys) == set(required_arrays)
assert "raw_logits_scope='native winner plus every original matched query; foreground/members cover all256'" in new_text
assert "arrays.update({'member_' + name: np.asarray(value) for name, value in geometry.items()})" in new_text
assert "for arm in ARMS})" in new_text
gradient_keyword = next(n for n in ast.walk(step) if isinstance(n, ast.keyword) and n.arg == 'upstream_group_grad_norms')
groups = ast.literal_eval(gradient_keyword.value.generators[0].iter)
assert groups == ('query_projection', 'support_projection', 'face_encoder', 'axis_decoder')
assert all('upstream_group_grad_norms' not in ast.dump(n.test) for n in ast.walk(step) if isinstance(n, ast.Assert))

model_item = next(n for n in manifest if n['path'].endswith('models\\pv_ground.py'))
model_text = Path(model_item['path']).read_text(encoding='utf-8')
assert "prediction_masks.append(pred_masks.expand(1,256,pred_masks.shape[-1]))" in model_text
assert "end_points['sp_last_pred_masks'] = corrected" in model_text
assert "parent['last_pred_masks'][bid][0]" in new_text
assert "parent['sp_last_pred_masks'][bid]" in new_text
assert "text[observed_queries].cpu().numpy()" in new_text
assert "own[observed_queries].cpu().numpy()" in new_text
assert "selected = int(scores[bid].argmax())" in new_text
assert "ranked = scores[bid].argsort(descending=True)" in new_text

result = dict(
    scope='STATIC_ONLY_NO_PROJECT_IMPORT', generated_at=datetime.datetime.utcnow().isoformat()+'Z',
    python_version=sys.version, python_executable=sys.executable,
    r1_files_preserved=len(preserved), r2_inputs_and_snapshots_exact=len(manifest),
    python_files_parsed=parsed, current_source_sha256=digest(new_path), prior_source_sha256=digest(old_path),
    current_runner_hashes_bound=11, other_10_runner_sources_unchanged=True,
    only_spec_change='new_runner_files.paired_span_loop.py', only_preparation_changes=['files.paired_span_loop.py','time_cst'],
    unchanged_methods=unchanged_methods, only_existing_changed_method='step', only_added_method='support_input_witness',
    neural_control_AST_identical_after_removing_receipt_additions=True,
    support_witness_only_call_guarded_by_preflight=True, same_forward_fit_restore_formal_AST_unchanged=True,
    pending_parent_rejection=pending_rejections,
    witness_array_keys=required_arrays, full_geometry_dictionary_saved=True,
    observed_logit_scope_explicit=True, gradient_group_names=list(groups), gradient_group_summary_is_observation=True,
    native_text_indexing_static_shape='list[B] of (1,256,S); [bid][0] gives (256,S); observed slice gives (K,S)',
    native_query_indexing_static_shape='list[B] of (256,S); observed slice gives (K,S)',
    source_fraction_static_shape='(256,3,2) per arm per row',
    selector_note=dict(witness='argmax', formal='argsort(descending=True)[0]',
                       actual_ties_or_disagreement_observed=False, exact_tie_identity_not_proven=True),
    actual_support_npz_generated=False, actual_cpu_reconstruction_of_witness=False,
    actual_M0=False, actual_neural_forwards=0, actual_gradients=False,
    actual_save_restore=False, actual_formal_metrics=False, launch_approved=False,
    imported_torch=any(n == 'torch' or n.startswith('torch.') for n in sys.modules),
    imported_numpy='numpy' in sys.modules, checks_passed=True)
assert not result['imported_torch'] and not result['imported_numpy']
(out / 'R2_STATIC_VERIFICATION.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2, sort_keys=True))

