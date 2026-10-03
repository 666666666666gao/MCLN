"""Launch the reviewed one-arm normalization experiment after actual GPU/save checks."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
assert not (local / 'launch.json').exists()
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and review['blocking_findings'] == []
preflight = json.loads((local / 'preflight.json').read_bytes())
assert preflight['status'] == 'pass' and preflight['optimizer_steps'] == 2
assert preflight['new_model_states'] == 0 and preflight['batch_size'] == 8
assert preflight['capacity']['expanded_count_normalization']
assert (local / 'preflight.exit').read_text().strip() == '0'
spec = json.loads((local / 'normalized_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_candidate_normalization_20261003'
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
probe = '''
import json, shutil, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); required=int(sys.argv[2])
assert not any((root/name).exists() for name in ('status.json','launch.json','controller.exit'))
free=shutil.disk_usage(root).free
assert free>=required, 'insufficient output-directory checkpoint space: %d < %d' % (free,required)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU already has a compute process'
print(json.dumps(dict(directory_free_bytes=free,required_bytes=required,GPU_idle=True)))
'''
required = 2 * preflight['serialization_bytes'] + 256 * 1024**2
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', probe, root, str(required)]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
variables = dict(environment['env'])
variables['PYTHONPATH'] = root + ':' + variables['PYTHONPATH']
command = 'env ' + shlex.join([key + '=' + value for key, value in variables.items()]) + ' ' + shlex.join(
    [python, '-u', root + '/controller.py', '--root', root])
inner = command + ' > ' + shlex.quote(root + '/controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/controller.exit') + '; exit "$code"'
screen = 'pvg_candidate_normalization_20261003'
_, stdout, stderr = client.exec_command('screen -dmS ' + screen + ' bash -c ' + shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote('^' + python + ' -u ' + root + '/controller.py --root ' + root + '$'), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
started = datetime.datetime.now().astimezone()
receipt = dict(time_cst=started.isoformat(), root=root, process=process, screen=screen, resources=resources,
    order=['normalized train', 'normalized formal9508'], start='original G with fresh AdamW',
    updates=3723, batch_size=8, expanded_count_normalization=True,
    estimated_finish_cst=(started + datetime.timedelta(seconds=10515+1042)).isoformat(),
    estimate_basis='measured preceding one-arm train and formal runtime',
    new_accuracy_available=False, deletion_executed=False)
raw = (json.dumps(receipt, indent=2) + '\n').encode()
(local / 'launch.json').write_bytes(raw)
with sftp.open(root + '/launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(receipt), flush=True)
