"""Deploy only the freshly reviewed source to the existing warm environment."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'readback_preflight_launch.json').exists()
review = json.loads((local / 'READBACK_FULL_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    raw = Path(item['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
face = local.parent / 'pvground_face_conditioned_20261004'
intake = json.loads((face / 'complete/INTAKE.json').read_bytes())
audit = json.loads((face / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert intake['controller_exit'] == 0 and not intake['controller_alive']
assert intake['status']['status'] == 'complete' and not audit['blocking_issues']
source = json.loads((local / 'PREFLIGHT_BUNDLE_SOURCE_CHECK.json').read_bytes())
template = json.loads((local / 'evidence_visible_preflight_template.json').read_bytes())
root = '/root/autodl-tmp/pvground_readback_preflight_20261004'
assert template['root'] == root + '/evidence_visible'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(template['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
python = template['runtime'] + '/venv/bin/python'
probe = '''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);closed=Path(sys.argv[2]);best=Path(sys.argv[3])
assert not root.exists()
assert json.loads((closed/'status.json').read_bytes())['status']=='complete'
assert (closed/'controller.exit').read_text().strip()=='0'
assert hashlib.sha256(best.read_bytes()).hexdigest()==sys.argv[4]
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU already has a compute process'
free=shutil.disk_usage(root.parent).free
assert free>=128*1024**2
print(json.dumps(dict(GPU_idle=True,prior_controller_closed=True,directory_free_bytes=free,
    system_free_bytes=shutil.disk_usage('/').free,weight_reserve_bytes=0,log_reserve_bytes=128*1024**2)))
'''
command = shlex.join([python, '-c', probe, root, '/root/autodl-tmp/pvground_face_fit_20261004',
    template['geometry_terminal'], template['geometry_terminal_sha256']])
_, stdout, stderr = client.exec_command(command, timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local / 'readback_preflight_resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for directory in ('overrides', 'overrides/models', 'evidence_hidden', 'evidence_visible'):
    sftp.mkdir(root + '/' + directory)
payloads = {name: (local / name).read_bytes() for name in ('readback_preflight_controller.py',
    'create_remote_readback_source.py', 'EXPERIMENT_PLAN_READBACK.md', 'READBACK_FULL_SOURCE_REVIEW.json',
    'evidence_hidden_preflight_template.json', 'evidence_visible_preflight_template.json')}
for name, digest in source['model_overrides'].items():
    raw = (local / 'source_preview/PV-Ground' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    payloads['overrides/' + name] = raw
for arm in ('evidence_hidden', 'evidence_visible'):
    for name, digest in source['runner_files'].items():
        raw = (local / 'runtime_bundle' / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        payloads[arm + '/' + name] = raw
for name, raw in payloads.items():
    with sftp.open(root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
_, stdout, stderr = client.exec_command(shlex.join([python, '-B', root + '/create_remote_readback_source.py']), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
sealed = json.loads(raw)
assert sealed['overrides'] == source['model_overrides'] and sealed['original_source_unchanged']
(local / 'readback_remote_source_receipt.json').write_bytes(raw)
for remote_name, local_name in [('source_port.json', 'SEALED_READBACK_SOURCE_PORT.json'),
        ('evidence_hidden/spec.json', 'evidence_hidden_preflight_spec.json'),
        ('evidence_visible/spec.json', 'evidence_visible_preflight_spec.json')]:
    with sftp.open(root + '/' + remote_name, 'rb') as stream:
        (local / local_name).write_bytes(stream.read())
controller = shlex.join(['flock', '-n', environment['resource_limits']['gpu_lock'], python,
    '-B', '-u', root + '/readback_preflight_controller.py'])
inner = controller + ' > ' + shlex.quote(root + '/controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/controller.exit') + '; exit "$code"'
screen = 'pvg_readback_preflight_20261004'
_, stdout, stderr = client.exec_command('screen -dmS ' + screen + ' bash -c ' + shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/readback_preflight_controller.py$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root, process=process,
    screen=screen, resources=resources, sealed_source=sealed, order=['evidence_hidden', 'evidence_visible'],
    execution_status='LAUNCHED_NOT_COMPLETED', cpu_factory_complete=False, gpu_preflight_complete=False,
    formal_training_started=False, accuracy_result=False, weight_files_created=0,
    estimate_seconds=1100, first_check_seconds=850, later_poll_seconds=240,
    estimate_basis='two warm real-model probes; prior flat probe about450seconds, plus CPU factories and new checks')
raw = (json.dumps(record, indent=2) + '\n').encode()
(local / 'readback_preflight_launch.json').write_bytes(raw)
with sftp.open(root + '/launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(record), flush=True)
