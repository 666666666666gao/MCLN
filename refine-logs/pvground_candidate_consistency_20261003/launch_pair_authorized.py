"""Start the existing reviewed pair only after actual save space is sufficient."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
assert not (local / 'pair_launch.json').exists()
for name in ('EXPERIMENT_CODE_REVIEW.json', 'RUNTIME_OBSERVATION_REVIEW.json'):
    review = json.loads((local / name).read_bytes())
    assert review['verdict'] in ('PASS', 'WARN') and review['blocking_findings'] == []
preflight = json.loads((local / 'preflight.json').read_bytes())
assert preflight['status'] == 'pass' and preflight['optimizer_steps'] == 2
assert preflight['new_model_states'] == 0 and preflight['batch_size'] == 8
assert (local / 'preflight.exit').read_text().strip() == '0'
spec = json.loads((local / 'g_control_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_candidate_consistency_20261003'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
with sftp.open(root + '/preflight/preflight.json', 'rb') as stream:
    assert json.loads(stream.read()) == preflight
python = spec['runtime'] + '/venv/bin/python'
code = '''
import json, shutil, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); required=int(sys.argv[2])
assert not any((root/name).exists() for name in ('pair_status.json','pair_launch.json','pair.exit'))
free=shutil.disk_usage(root).free
assert free>=required, 'insufficient output-directory checkpoint space: %d < %d' % (free,required)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU already has a compute process'
print(json.dumps(dict(directory_free_bytes=free,required_bytes=required,GPU_idle=True)))
'''
required = 3 * preflight['serialization_bytes'] + 256 * 1024**2
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', code, root, str(required)]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
variables = dict(environment['env'])
variables['PYTHONPATH'] = root + ':' + variables['PYTHONPATH']
command = 'env ' + shlex.join([key + '=' + value for key, value in variables.items()]) + ' ' + shlex.join(
    [python, '-u', root + '/pair.py', '--root', root])
inner = command + ' > ' + shlex.quote(root + '/pair.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/pair.exit') + '; exit "$code"'
screen = 'pvg_candidate_consistency_20261003'
_, stdout, stderr = client.exec_command('screen -dmS ' + screen + ' bash -c ' + shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote('^' + python + ' -u ' + root + '/pair.py --root ' + root + '$'), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
receipt = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root,
    process=process, screen=screen, resources=resources,
    order=['G control train', 'G consistent train', 'G control formal9508', 'G consistent formal9508'],
    start='same original G with fresh AdamW', updates_per_arm=3723,
    new_accuracy_available=False, deletion_executed=False)
raw = (json.dumps(receipt, indent=2) + '\n').encode()
(local / 'pair_launch.json').write_bytes(raw)
with sftp.open(root + '/pair_launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(receipt))
