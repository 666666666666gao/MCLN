"""Launch one reviewed CPU data replay in the existing authorized runtime."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'launch.json').exists()
review = json.loads((local / 'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
spec = json.loads((local / 'spec.json').read_bytes())
remote = spec['root']
assert remote == '/root/autodl-tmp/pvground_target_jitter_20261005'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
for name in ('run_label_probe.py', 'spec.json', 'EXPERIMENT_PLAN.md', 'SOURCE_REVIEW.json', 'SOURCE_REVIEW.md'):
    raw = (local / name).read_bytes()
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == raw
python = spec['runtime'] + '/venv/bin/python'
command = shlex.join([python, '-B', '-u', remote + '/run_label_probe.py', '--spec', remote + '/spec.json'])
inner = command + ' > ' + shlex.quote(remote + '/run.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(remote + '/run.exit') + '; exit "$code"'
screen = 'pvg_target_jitter_20261005'
_, stdout, stderr = client.exec_command(shlex.join(['screen', '-dmS', screen, 'bash', '-c', inner]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=remote, screen=screen,
    status='LAUNCHED_NOT_COMPLETED', model_forwards=0, weights_loaded=0, optimizer_steps=0,
    estimated_seconds=300, first_check_seconds=180, later_poll_seconds=240, accuracy_result=False)
(local / 'launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(record), flush=True)
