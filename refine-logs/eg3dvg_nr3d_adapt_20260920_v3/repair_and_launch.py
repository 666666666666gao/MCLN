import datetime, hashlib, json, os, shutil, subprocess
from pathlib import Path

old = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v2')
r = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3')
assert (old / 'controller.exit').read_text().strip() == '1'
assert not (old / 'fit').exists()
assert (old / 'preflight/updates.jsonl').stat().st_size == 0
assert "KeyError: 'super_xyz_list'" in (old / 'preflight.log').read_text()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
r.mkdir()
shutil.copytree(str(old / 'source'), str(r / 'source'), ignore=shutil.ignore_patterns('__pycache__'))
p = r / 'source/models/eg.py'
before = p.read_text()
marker = '        # STEP 5. Query Points Generation\n'
assert before.count(marker) == 1
assert "end_points['super_xyz_list']" not in before
after = before.replace(marker, "        end_points['super_xyz_list'] = super_xyz_list\n\n" + marker)
compile(after, str(p), 'exec')
p.write_text(after)
assert "output['super_xyz_list'] = end_points['super_xyz_list']" in (r / 'source/models/losses.py').read_text()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for name in ['train.py', 'controller.py', 'evaluate_adapted.py', 'audit_adapted.py', 'training_order.npy', 'analyze_pair.py', 'wait_pair_analysis.py', 'view_fix_rows.json']:
    shutil.copyfile(str(old / name), str(r / name))
spec = json.loads((old / 'spec.json').read_text())
spec['source'] = str(r / 'source')
spec['source_files']['models/eg.py'] = sha(p)
old_order = spec['order_path']
spec['order_path'] = str(r / 'training_order.npy')
spec['input_hashes'][spec['order_path']] = spec['input_hashes'].pop(old_order)
assert sha(spec['order_path']) == spec['input_hashes'][spec['order_path']]
spec['state_root'] = '/root/mcln_eg3dvg_nr3d_adapt_states_20260920_v3'
assert shutil.disk_usage('/root').free > 4000000000
Path(spec['state_root']).mkdir()
spec['replaces_failed_preflight'] = str(old)
spec['repair'] = 'Export already computed super_xyz_list required by the unmodified author loss; no forward numeric or loss definition change.'
(r / 'spec.json').write_text(json.dumps(spec, indent=2))
report = {'status': 'patched', 'previous_root': str(old), 'new_root': str(r), 'previous_optimizer_steps': 0,
          'before_sha256': hashlib.sha256(before.encode()).hexdigest(), 'after_sha256': sha(p),
          'change': "end_points['super_xyz_list'] = super_xyz_list", 'loss_source_unchanged': True,
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
(r / 'loss_export_repair.json').write_text(json.dumps(report, indent=2))
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
