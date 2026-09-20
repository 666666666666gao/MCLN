import ast, datetime, hashlib, json, os, pickle, re, shutil, subprocess
from pathlib import Path

old = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v1')
r = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v2')
assert (old / 'controller.exit').read_text().strip() == '1'
assert not (old / 'fit').exists()
assert not (old / 'preflight/receipt.json').exists()
assert (old / 'preflight/updates.jsonl').stat().st_size == 0
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
r.mkdir()
shutil.copytree(str(old / 'source'), str(r / 'source'), ignore=shutil.ignore_patterns('__pycache__'))
p = r / 'source/src/joint_det_dataset.py'
before = p.read_text()
tree = ast.parse(before)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Joint3DDataset')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['_augment_nr3d', '_is_view_dep']]
assert len(methods) == 2
lines = before.splitlines(keepends=True)
for node in sorted(methods, key=lambda n: n.lineno, reverse=True):
    assert not node.decorator_list
    lines.insert(node.lineno - 1, '    @staticmethod\n')
after = ''.join(lines)
p.write_text(after)
tree = ast.parse(after)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Joint3DDataset')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['_augment_nr3d', '_is_view_dep']]
assert all(len(n.decorator_list) == 1 and n.decorator_list[0].id == 'staticmethod' for n in methods)
test = ast.Module(body=[ast.ClassDef(name='Joint3DDataset', bases=[], keywords=[], body=methods, decorator_list=[])])
namespace = {'re': re}
exec(compile(ast.fix_missing_locations(test), str(p), 'exec'), namespace)
test_cls = namespace['Joint3DDataset']
instance = test_cls()
with Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/train_input_cache/nr3d_annotations.pkl').open('rb') as f:
    annotations = pickle.load(f)
for row in annotations:
    text = row['utterance']
    assert instance._augment_nr3d(text) == test_cls._augment_nr3d(text)
    assert instance._is_view_dep(text) == test_cls._is_view_dep(text)
    assert instance._augment_nr3d(text) == (not instance._is_view_dep(text))
assert not instance._augment_nr3d('Facing the windows, choose the desk.')
assert instance._augment_nr3d('The red chair next to the table.')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for name in ['train.py', 'controller.py', 'evaluate_adapted.py', 'audit_adapted.py', 'training_order.npy', 'analyze_pair.py', 'wait_pair_analysis.py', 'view_fix_rows.json']:
    shutil.copyfile(str(old / name), str(r / name))
spec = json.loads((old / 'spec.json').read_text())
spec['source'] = str(r / 'source')
spec['source_files']['src/joint_det_dataset.py'] = sha(p)
old_order = spec['order_path']
spec['order_path'] = str(r / 'training_order.npy')
spec['input_hashes'][spec['order_path']] = spec['input_hashes'].pop(old_order)
assert sha(spec['order_path']) == spec['input_hashes'][spec['order_path']]
spec['state_root'] = '/root/mcln_eg3dvg_nr3d_adapt_states_20260920_v2'
assert shutil.disk_usage('/root').free > 4000000000
Path(spec['state_root']).mkdir()
spec['replaces_failed_preflight'] = str(old)
spec['repair'] = 'Restore staticmethod declarations omitted by Python 3.7 AST source replacement; no optimization had occurred.'
(r / 'spec.json').write_text(json.dumps(spec, indent=2))
report = {'status': 'pass', 'previous_root': str(old), 'new_root': str(r), 'previous_optimizer_steps': 0,
          'checked_rows': len(annotations), 'class_and_instance_calls_identical': True,
          'before_sha256': hashlib.sha256(before.encode()).hexdigest(), 'after_sha256': sha(p),
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
(r / 'static_method_repair.json').write_text(json.dumps(report, indent=2))
shutil.copyfile(__file__, str(r / 'repair_and_launch.py'))
env = dict(os.environ)
env.update({'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
with (r / 'controller.log').open('xb') as log:
    proc = subprocess.Popen([spec['runtime'], '-u', str(r / 'controller.py')], cwd=spec['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
with (r / 'pair_wait.log').open('xb') as log:
    pair = subprocess.Popen([spec['runtime'], '-u', str(r / 'wait_pair_analysis.py')], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(report, pid=proc.pid, pair_pid=pair.pid, spec_sha256=sha(r / 'spec.json'), fit_updates_planned=5614, fit_rows_planned=44909)
(r / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
