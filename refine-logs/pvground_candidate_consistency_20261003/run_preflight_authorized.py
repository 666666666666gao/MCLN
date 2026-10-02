"""Deploy reviewed semantic-target-only code, then run CPU and real GPU sanity."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and review['blocking_findings'] == []
remote = '/root/autodl-tmp/pvground_candidate_consistency_20261003'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
modules = ['pvground_semantic_assignment.py', 'pvground_candidate_consistency.py',
           'pvground_task_observation_query.py', 'pvground_observation_query.py', 'pvground_source_query.py']
for name in ['run.py', 'pair.py', 'cpu_test.py', 'EXPERIMENT_PLAN.md', 'EXPERIMENT_TRACKER.md'] + modules:
    sftp.put(str(local / name), remote + '/' + name)
for arm in ('preflight', 'g_control', 'g_consistent'):
    sftp.mkdir(remote + '/' + arm)
    sftp.put(str(local / (arm + '_spec.json')), remote + '/' + arm + '/spec.json')
    for name in modules:
        sftp.put(str(local / name), remote + '/' + arm + '/' + name)
spec = json.loads((local / 'preflight_spec.json').read_bytes())
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
python = spec['runtime'] + '/venv/bin/python'
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
            output.write(line); output.flush()
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
client.close()
record = dict(status='pass', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              CPU_test=True, GPU_batch_size=8, preflight_optimizer_steps=2, full_training_started=False)
(local / 'preflight_intake.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
