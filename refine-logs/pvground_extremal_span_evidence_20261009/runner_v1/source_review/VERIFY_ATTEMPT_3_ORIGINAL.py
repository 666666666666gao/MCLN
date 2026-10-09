"""STATIC_ONLY, stdlib-only audit: no imported project or neural modules."""
import ast
import datetime
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

out = Path(__file__).resolve().parent
runner = out.parent
manifest = json.loads((out / 'INPUT_MANIFEST.json').read_text(encoding='utf-8-sig'))
by_name = {}
for item in manifest:
    by_name.setdefault(Path(item['path']).name, []).append(item)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def source(name, index=0):
    item = by_name[name][index]
    return (out / item['snapshot']).read_text(encoding='utf-8-sig')

def tree(name, index=0):
    return ast.parse(source(name, index), filename=by_name[name][index]['path'])

def method(module, cls, name):
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)

def fullname(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return fullname(node.value) + '.' + node.attr
    return None

def calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and fullname(n.func) == name]

def signature(node):
    assert not node.args.vararg and not node.args.kwarg and not node.args.kwonlyargs
    defaults = [inspect.Parameter.empty] * (len(node.args.args) - len(node.args.defaults))
    defaults += [ast.literal_eval(n) for n in node.args.defaults]
    return inspect.Signature([inspect.Parameter(n.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=d)
                              for n, d in zip(node.args.args, defaults)])

assert sys.version_info[:2] == (3, 7), sys.version
checked = []
for item in manifest:
    actual = Path(item['path']).read_bytes()
    copied = (out / item['snapshot']).read_bytes()
    assert sha(actual) == item['sha256'] == sha(copied), item['path']
    assert len(actual) == item['bytes']
    if item['path'].endswith('.py'):
        ast.parse(actual, filename=item['path'])
        checked.append(item['path'])

spec = json.loads(source('pair_spec_template.json'))
preparation = json.loads(source('RUNNER_PREPARATION.json'))
provenance = json.loads(source('ENTRY_PROVENANCE.json'))
runner_files = sorted(p.name for p in runner.glob('*.py'))
assert len(runner_files) == 11
assert set(runner_files) == set(spec['new_runner_files']) == set(preparation['files'])
for name in runner_files:
    assert sha((runner / name).read_bytes()) == spec['new_runner_files'][name] == preparation['files'][name], name

dependency_hashes = {}
for name in ('readback_preflight_checks.py', 'whole_model_preflight_checks.py',
             'native_root_bbs.py', 'whole_mask_range.py', 'pvground_boundary_box_refiner.py'):
    digest = sha(Path(by_name[name][0]['path']).read_bytes())
    assert digest == spec['runner_files'][name]
    dependency_hashes[name] = digest
assert sha(Path(by_name['grounding_evaluator.py'][0]['path']).read_bytes()) == spec['native_evaluator_sha256']

entry = tree('run_span_pair.py')
controller = tree('span_controller.py')
wrapper = tree('span_refinement_model.py')
loop = tree('paired_span_loop.py')
native = tree('pv_ground.py')
forward = method(wrapper, 'SpanRefinementModel', 'forward')
native_forward = method(native, 'PVGround', 'forward')
assert str(signature(forward)) == '(self, inputs)'
assert str(signature(native_forward)) == '(self, inputs)'
parent_calls = calls(forward, 'self.parent')
assert len(parent_calls) == 1 and len(parent_calls[0].args) == 1 and not parent_calls[0].keywords
signature(native_forward).bind(object(), object())

gates = {}
for name, module in (('run_span_pair.py', entry), ('span_controller.py', controller)):
    gate = next(n for n in ast.walk(module) if isinstance(n, ast.Assert)
                and 'parent_selection_status' in ast.dump(n.test))
    expression = ast.Expression(body=gate.test)
    assert eval(compile(expression, '<only-static-parent-comparison>', 'eval'), {'spec': spec}) is False
    gates[name] = dict(line=gate.lineno, pending_predicate=False, actual_entry_executed=False)
assert gates['run_span_pair.py']['line'] < next(n.lineno for n in ast.walk(entry)
    if isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names))
assert gates['span_controller.py']['line'] < calls(controller, 'subprocess.Popen')[0].lineno

prepare = tree('prepare_runner_entry.py')
replacements = ast.literal_eval(next(n.value for n in prepare.body
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'replacements' for t in n.targets)))
prior = source('run_mask_support_pair.py')
generated = prior
historic = prior
for before, after in replacements:
    assert generated.count(before) == 1
    generated = generated.replace(before, after)
    if 'candidate_gate_hits' not in after:
        assert historic.count(before) == 1
        historic = historic.replace(before, after)
assert len(replacements) == 10
current = source('run_span_pair.py')
assert current == generated
historic_bytes = historic.replace('\n', os.linesep).encode('utf-8')
generated_bytes = generated.replace('\n', os.linesep).encode('utf-8')
assert sha(historic_bytes) == provenance['prepared_sha256']
assert sha(generated_bytes) == spec['new_runner_files']['run_span_pair.py']
assert provenance['replacement_count'] == 9
(out / 'ENTRY_FROM_PRIOR.diff').write_text(''.join(difflib.unified_diff(
    prior.splitlines(True), current.splitlines(True), fromfile='run_mask_support_pair.py',
    tofile='run_span_pair.py')), encoding='utf-8')
(out / 'HISTORICAL_ENTRY_TO_CURRENT.diff').write_text(''.join(difflib.unified_diff(
    historic.splitlines(True), current.splitlines(True), fromfile='historical_generated_entry',
    tofile='current_bound_entry')), encoding='utf-8')
protocol = {}
prior_tree = tree('run_mask_support_pair.py')
for name in ('FitDataset', 'FormalDataset', 'loader', 'prepare', 'box_iou'):
    candidates = [[n for n in ast.walk(t) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
                  for t in (entry, prior_tree)]
    assert len(candidates[0]) == len(candidates[1]) == 1
    assert ast.dump(candidates[0][0]) == ast.dump(candidates[1][0])
    protocol[name] = 'identical_executable_AST'
old_spec = json.loads(source('pair_spec.json'))
keys = ('batch_size', 'seed', 'lr', 'weight_decay', 'clip_norm', 'fit_passes', 'updates',
        'input_manifest', 'preflight_batch_index', 'preflight_reference_rows', 'reference_fixtures',
        'primary_mode', 'primary_threshold', 'native_evaluator_sha256', 'no_multiseed',
        'runner_files', 'model_source', 'source_port', 'source_port_sha256',
        'base_terminal', 'base_terminal_sha256', 'selected_terminal', 'selected_terminal_sha256',
        'checkpoint_sha256', 'env_spec_sha256')
assert all(spec[k] == old_spec[k] for k in keys)

mixer = tree('extremal_span_mixer.py')
initialization = method(mixer, 'ExtremalSpanMixer', '__init__')
dimensions = [tuple(ast.literal_eval(a) for a in n.args) for n in calls(initialization, 'nn.Linear')]
params = sum(i * o + o for i, o in dimensions)
assert params == 29793 and 2 * len(dimensions) == 14
mode_branches = [n.lineno for n in ast.walk(mixer) if isinstance(n, ast.If)
                 and 'source_mode' in ast.dump(n.test)]
assert mode_branches == [69]
pair_forward = method(loop, 'PairedSpanRun', 'forward_pair')
step = method(loop, 'PairedSpanRun', 'step')
assert len(calls(pair_forward, 'observed_readback_forward')) == 1
assert len(calls(step, 'self.forward_pair')) == 1
assert len(calls(step, 'frozen_parent_assignments')) == 1
assignment = calls(step, 'frozen_parent_assignments')[0]
arm_loop = next(n for n in step.body if isinstance(n, ast.For))
assert assignment.lineno < arm_loop.lineno
assert not calls(arm_loop, 'frozen_parent_assignments')
witness = method(loop, 'PairedSpanRun', 'restore_and_integration_witness')
integrated = calls(witness, 'deployed')
assert len(integrated) == 1 and fullname(integrated[0].args[0].func) == 'dict'
assert len(calls(witness, 'optimizer_restore_exact')) == 2
assert len(calls(witness, 'torch.set_rng_state')) == 1
assert len(calls(witness, 'torch.cuda.set_rng_state_all')) == 1
assert len(calls(witness, 'np.random.set_state')) == 1
assert len(calls(witness, 'random.setstate')) == 1

all_mkdir = [{'line': n.lineno, 'call': ast.dump(n)} for n in ast.walk(loop)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'mkdir']
report = dict(
    verification_scope='STATIC_ONLY_STDLIB_NO_PROJECT_IMPORT',
    timestamp_utc=datetime.datetime.utcnow().isoformat()+'Z',
    python_version=sys.version, python_executable=sys.executable,
    neural_imported=any(k == 'torch' or k.startswith('torch.') for k in sys.modules),
    project_modules_imported=False, parsed_python_files=checked,
    input_snapshots_and_current_hashes_equal=len(manifest),
    bound_runner_sources=runner_files, dependency_hashes=dependency_hashes,
    native_evaluator_hash_matches=True, wrapper_signature=str(signature(forward)),
    native_signature=str(signature(native_forward)), parent_call_count_ast=1,
    pending_gate_predicates=gates,
    entry_provenance=dict(current_ten_replacements_reproduce_bytes=True,
                         historical_nine_replacements_reproduce_recorded_hash=True,
                         historical_prepared_sha256=sha(historic_bytes),
                         current_sha256=sha(generated_bytes),
                         active_source_modified=False),
    copied_protocol_AST=protocol, unchanged_spec_contract_keys=list(keys),
    linear_shapes=dimensions, static_parameter_count=params, static_state_tensor_count=2*len(dimensions),
    mode_branch_lines=mode_branches,
    shared_parent_and_assignment_once_AST=True,
    native_wrapper_same_cache_integration_wired_AST=True,
    CPU_GPU_optimizer_and_four_RNG_restore_calls_present=True,
    loop_mkdir_calls=all_mkdir,
    actual_M0=False, actual_gradients=False, actual_neural_save_restore=False,
    actual_formal_results=False, runtime_or_launch_approval=False,
    checks_passed=True)
assert report['neural_imported'] is False
(out / 'STATIC_VERIFICATION_R1.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2, sort_keys=True))

