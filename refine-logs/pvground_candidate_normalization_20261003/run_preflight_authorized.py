"""Reuse the certified environment for native CPU checks and real batch8 sanity."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and review['blocking_findings'] == []
assert not (local / 'preflight_intake.json').exists()
remote = '/root/autodl-tmp/pvground_candidate_normalization_20261003'
spec = json.loads((local / 'preflight_spec.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
python = spec['runtime'] + '/venv/bin/python'
probe = '''
import json, shutil, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); assert not root.exists()
free=shutil.disk_usage(root.parent).free
assert free >= 1024**3, 'insufficient output-directory save space'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU already has a compute process'
print(json.dumps(dict(directory_free_bytes=free,GPU_idle=True,environment_reused=True)))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', probe, remote]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
(local / 'preflight_resource_check.json').write_bytes(raw)
sftp.mkdir(remote)
modules = ['pvground_semantic_assignment.py', 'pvground_candidate_consistency.py',
           'pvground_task_observation_query.py', 'pvground_observation_query.py', 'pvground_source_query.py']
for name in ['run.py', 'controller.py', 'cpu_test.py', 'EXPERIMENT_PLAN.md',
             'EXPERIMENT_TRACKER.md', 'research_contract.md', 'EXPERIMENT_CODE_REVIEW.json'] + modules:
    sftp.put(str(local / name), remote + '/' + name)
for arm in ('preflight', 'normalized'):
    sftp.mkdir(remote + '/' + arm)
    sftp.put(str(local / (arm + '_spec.json')), remote + '/' + arm + '/spec.json')
variables = dict(environment['env'])
variables['PYTHONPATH'] = remote + ':' + variables['PYTHONPATH']
prefix = 'env ' + shlex.join([key + '=' + value for key, value in variables.items()]) + ' '
commands = [
    ('cpu', [python, '-u', remote + '/cpu_test.py', '--native-loss',
             spec['model_source'] + '/models/losses.py', '--output', remote + '/cpu_test.json']),
    ('preflight', ['flock', '-n', environment['resource_limits']['gpu_lock'], python,
                   '-u', remote + '/run.py', '--spec', remote + '/preflight/spec.json', '--mode', 'preflight']),
]
for phase, command in commands:
    _, stdout, _ = client.exec_command(prefix + shlex.join(command) + ' 2>&1', timeout=60)
    stdout.channel.settimeout(None)
    with (local / (phase + '.log')).open('x', encoding='utf-8') as output:
        for line in stdout:
            output.write(line)
            output.flush()
            if phase == 'cpu' or line.startswith(('G_CANDIDATE_CONSISTENCY_PREFLIGHT_PASS ', 'PVG_FINETUNE_DATASET_LOADING ')):
                print(line.strip(), flush=True)
    code = stdout.channel.recv_exit_status()
    (local / (phase + '.exit')).write_text(str(code) + '\n', encoding='utf-8')
    if code:
        print(json.dumps(dict(status='failed', phase=phase, exit_code=code, log=str(local / (phase + '.log')))), flush=True)
        raise SystemExit(code)
    if phase == 'cpu':
        with sftp.open(remote + '/cpu_test.json', 'rb') as stream:
            (local / 'cpu_test.json').write_bytes(stream.read())
for name in ('preflight.json', 'load.json', 'imports.json'):
    with sftp.open(remote + '/preflight/' + name, 'rb') as stream:
        (local / name).write_bytes(stream.read())
sftp.close()
client.close()
record = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
              CPU_test=True, GPU_batch_size=8, preflight_optimizer_steps=2,
              environment_reused=True, full_training_started=False)
(local / 'preflight_intake.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
