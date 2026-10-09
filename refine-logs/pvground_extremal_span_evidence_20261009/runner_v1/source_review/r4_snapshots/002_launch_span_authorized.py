"""Admit an audited span phase after real closure; no automatic successor."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex

import paramiko


runner = Path(__file__).resolve().parent
prior = runner.parents[1] / 'pvground_selected_mask_training_20261009'
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('preflight', 'fit'), required=True)
phase = parser.parse_args().phase
assert not (runner / (phase + '_launch.json')).exists()
spec_path = runner / 'pair_spec.json'
spec = json.loads(spec_path.read_bytes())
assert spec['parent_selection_status'] == 'FINALIZED_AFTER_PRIOR_PAIR_CLOSED'
bound = json.loads((runner / 'FINAL_SPEC_RECEIPT.json').read_bytes())
assert bound['pair_spec_sha256'] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
for name, digest in bound['prior_evidence'].items():
    assert hashlib.sha256((prior / name).read_bytes()).hexdigest() == digest
review = json.loads((runner / 'LAUNCH_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN')
assert not review['blocking_findings']
required = {(runner / name).resolve() for name in
            ('finalize_parent.py', 'launch_span_authorized.py', 'span_controller.py', 'pair_spec_template.json')}
assert required.issubset({Path(row['path']).resolve() for row in review['reviewed_files']})
for row in review['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
for name, digest in spec['new_runner_files'].items():
    assert hashlib.sha256((runner / name).read_bytes()).hexdigest() == digest
if phase == 'fit':
    wait = json.loads((runner / 'preflight_wait.json').read_bytes())
    assert wait['observer_closed'] and not wait['terminal']['controller_alive']
    assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
    actual = json.loads((runner / 'preflight_results/EXPERIMENT_AUDIT.json').read_bytes())
    assert actual['execution_scope'] == 'ACTUAL_PREFLIGHT' and actual['fresh_context'] is True
    assert actual['verdict'] in ('PASS', 'WARN') and not actual['blocking_findings']
    required = {(runner / name).resolve() for name in
                ('preflight_wait.json', 'preflight_complete/preflight.json', 'pair_spec.json')}
    assert required.issubset({Path(row['path']).resolve() for row in actual['reviewed_files']})
    for row in actual['reviewed_files']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
    proof = json.loads((runner / 'preflight_complete/preflight.json').read_bytes())
    assert proof['status'] == 'pass' and proof['optimizer_steps_per_arm'] == 2
    assert proof['weight_files_created'] == 0 and proof['separate_optimizers_and_gradients']
    assert proof['spec_sha256'] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert all(row['actual_native_dictionary_model_call'] and row['updated_same_cache_output_exact']
               for row in proof['restore'].values())
files = list(spec['new_runner_files']) + ['pair_spec.json', 'FINAL_SPEC_RECEIPT.json', 'LAUNCH_SOURCE_REVIEW.json']
formal_bytes = math.ceil(9508 / 8) * (8 * (5 * 256 * 6 + 256 + 6) * 4 + 8 * 8 + 16384)
reserve = (256 * 1024**2 if phase == 'preflight' else formal_bytes + 130 * 1024**2)
reserve += sum((runner / name).stat().st_size for name in files)
estimated = 480 if phase == 'preflight' else math.ceil(json.loads((prior / 'complete_fit/fit_status.json').read_bytes())['elapsed_seconds'])
first = max(180, estimated - (180 if phase == 'preflight' else 300))
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
code = r'''import hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);spec=b['spec'];root=Path(spec['root']);old=Path(spec['previous_completed_experiment'])
assert root==Path('/root/autodl-tmp/pvground_extremal_span_pair_20261009')
assert root.parent.resolve()==Path('/root/autodl-tmp')
assert (old/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((old/'fit_status.json').read_bytes())['status']=='complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[value.strip() for value in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and 40000<=int(total)<=45000 and int(used)<500
assert shutil.disk_usage(root.parent).free>b['reserve']
environment=json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
parents={Path(environment['weight_dirs']['scanrefer']['path']):spec['checkpoint_sha256']}
parents.update({Path(spec[key]):spec[key+'_sha256'] for key in
               ('base_terminal','selected_terminal','parent_support_terminal')})
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in parents.items())
if b['phase']=='preflight':
 assert not root.exists();root.mkdir()
 for arm in spec['support_modes']:(root/arm).mkdir()
else:
 assert (root/'preflight_controller.exit').read_text().strip()=='0'
 assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
 assert not (root/'fit_status.json').exists()
 assert hashlib.sha256((root/'pair_spec.json').read_bytes()).hexdigest()==b['spec_sha256']
 assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in spec['new_runner_files'].items())
print(json.dumps(dict(GPU_idle=True,data_free_bytes=shutil.disk_usage(root).free,
    system_free_bytes=shutil.disk_usage('/').free,reserve=b['reserve'],phase=b['phase'],
    previous_fit_closed=True,protected_parents_exact=True,new_weights=0,deletions=0)))
'''
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python', '-B', '-c', code]), timeout=120)
stdin.write(json.dumps(dict(spec=spec, phase=phase, reserve=reserve,
    spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest())))
stdin.channel.shutdown_write()
raw, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(runner / (phase + '_admission.stdout.json')).write_bytes(raw)
(runner / (phase + '_admission.stderr.txt')).write_bytes(error)
(runner / (phase + '_admission.exit')).write_text(str(status) + '\n')
assert status == 0, error.decode()
if phase == 'preflight':
    for name in files:
        data = (runner / name).read_bytes()
        with sftp.open(spec['root'] + '/' + name, 'wx') as stream:
            stream.write(data)
        with sftp.open(spec['root'] + '/' + name, 'rb') as stream:
            assert stream.read() == data
command = shlex.join(['flock', '-n', env['resource_limits']['gpu_lock'],
    spec['runtime'] + '/venv/bin/python', '-B', '-u', spec['root'] + '/span_controller.py', '--phase', phase])
shell = command + ' > ' + shlex.quote(spec['root'] + '/' + phase + '_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(spec['root'] + '/' + phase + '_controller.exit') + '; exit "$code"'
screen = 'pvg_extremal_span_' + phase + '_20261009'
_, stdout, stderr = client.exec_command(shlex.join(['screen', '-dmS', screen, 'bash', '-c', shell]), timeout=30)
launch_stdout, launch_error = stdout.read(), stderr.read()
launch_code = stdout.channel.recv_exit_status()
(runner / (phase + '_screen.stdout.txt')).write_bytes(launch_stdout)
(runner / (phase + '_screen.stderr.txt')).write_bytes(launch_error)
(runner / (phase + '_screen.exit')).write_text(str(launch_code) + '\n')
assert launch_code == 0, launch_error.decode()
pattern = '^' + spec['runtime'] + '/venv/bin/python -B -u ' + spec['root'] + '/span_controller.py --phase ' + phase + '$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process, error = stdout.read().decode().strip(), stderr.read()
process_code = stdout.channel.recv_exit_status()
(runner / (phase + '_process.stdout.txt')).write_text(process + '\n')
(runner / (phase + '_process.stderr.txt')).write_bytes(error)
(runner / (phase + '_process.exit')).write_text(str(process_code) + '\n')
assert process_code == 0 and len(process.splitlines()) == 1, error.decode()
started = datetime.datetime.now().astimezone()
receipt = dict(status=phase.upper() + '_STARTED_NOT_COMPLETED', phase=phase,
    time_cst=started.isoformat(), root=spec['root'], controller_pid=int(process.split()[0]),
    process=process, screen=screen, resources=json.loads(raw),
    estimated_seconds=estimated, first_check_seconds=first, later_poll_seconds=240,
    first_observation_cst=(started + datetime.timedelta(seconds=first)).isoformat(),
    estimate_basis='preflight480s from actual previous paired M0; fit uses completed prior frozen-parent paired fit/formal elapsed, heads/output differ so timing remains estimated',
    seed=2027, physical_batch=8, effective_batch=8, fresh_optimizers=True,
    optimizer_steps_planned_per_arm=2 if phase == 'preflight' else 3723,
    parent_sha256=spec['parent_support_terminal_sha256'], accuracy_result=False,
    full_goal_complete=False, automatic_successor=False)
path = runner / (phase + '_launch.json')
path.write_text(json.dumps(receipt, indent=2) + '\n')
with sftp.open(spec['root'] + '/' + path.name, 'wx') as stream:
    stream.write(path.read_bytes())
sftp.close()
client.close()
print(json.dumps(receipt), flush=True)
