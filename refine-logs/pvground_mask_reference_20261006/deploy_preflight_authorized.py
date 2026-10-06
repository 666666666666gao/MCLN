"""Deploy the reviewed auxiliary-target comparison's two-update sanity only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path(__file__).resolve().parent
previous=local.parent/'pvground_mask_extent_diagnostic_20261006'
audit=json.loads((previous/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert json.loads((previous/'analysis/SUMMARY.json').read_bytes())['protected_trained_model_hits']==[5616,4511]
review = json.loads((local / 'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
assert not (local / 'preflight_launch.json').exists()
root = '/root/autodl-tmp/pvground_mask_reference_20261006'
spec = json.loads((local / 'native_reference_spec.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    env = json.loads(stream.read())
python = spec['runtime'] + '/venv/bin/python'
probe = '''
import json,subprocess,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);closed=Path('/root/autodl-tmp/pvground_mask_extent_diagnostic_20261006')
assert not root.exists()
closed_status=json.loads((closed/'formal_status.json').read_bytes())
assert closed_status['completed'] and closed_status['exit_code']==0
assert (closed/'formal.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(root.parent).free>64*1024**2
print(json.dumps(dict(GPU_idle=True,prior_CPU_closed=True,data_free_bytes=shutil.disk_usage(root.parent).free,
    system_free_bytes=shutil.disk_usage('/').free,packages_installed=0)))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', probe, root]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
(local / 'resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for arm in ('native_reference','fused_mask_reference'):
    sftp.mkdir(root + '/' + arm)
for name in ('run_geometry_fit.py', 'query_supported_geometry.py', 'mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.npz','invalid_reference_fixture.json', 'native_reference_spec.json', 'fused_mask_reference_spec.json',
             'controller.py', 'EXPERIMENT_PLAN.md', 'SOURCE_REVIEW.md', 'SOURCE_REVIEW.json'):
    raw = (local / name).read_bytes()
    with sftp.open(root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['flock', '-n', env['resource_limits']['gpu_lock'], python, '-B', '-u',
                     root + '/controller.py', '--phase', 'preflight'])
inner = command + ' > ' + shlex.quote(root + '/preflight_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/preflight_controller.exit') + '; exit "$code"'
screen = 'pvg_mask_reference_preflight_20261006'
_, stdout, stderr = client.exec_command(shlex.join(['screen', '-dmS', screen, 'bash', '-c', inner]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/controller.py --phase preflight$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root, process=process, screen=screen,
    status='PREFLIGHT_LAUNCHED_NOT_COMPLETED', optimizer_steps_planned_per_arm=2, accuracy_result=False,
    weight_files_planned=0, estimated_seconds=900, first_check_seconds=720, later_poll_seconds=240,
    estimate_basis='Two warm protected4511 model reconstructions;2 updates each, all256 actual point extent check and39 recorded empty supports. No accuracy result.')
(local / 'preflight_launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(record), flush=True)
