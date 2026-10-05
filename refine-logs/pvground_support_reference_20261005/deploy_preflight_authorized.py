"""Deploy the reviewed auxiliary-target comparison's two-update sanity only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path(__file__).resolve().parent
previous=local.parent/'pvground_auxiliary_target_20261005'
audit=json.loads((previous/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert json.loads((previous/'terminal_publication.json').read_bytes())['retained_best']['hits']==[5616,4511]
review = json.loads((local / 'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
assert not (local / 'preflight_launch.json').exists()
root = '/root/autodl-tmp/pvground_support_reference_20261005'
spec = json.loads((local / 'control_spec.json').read_bytes())
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
root=Path(sys.argv[1]);closed=Path('/root/autodl-tmp/pvground_auxiliary_target_20261005')
assert not root.exists()
assert json.loads((closed/'fit_status.json').read_bytes())['status']=='complete'
assert (closed/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((closed/'weight_retention.json').read_bytes())['retained_best']['hits']==[5616,4511]
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
for arm in ('control', 'support_reference'):
    sftp.mkdir(root + '/' + arm)
for name in ('run_geometry_fit.py', 'query_supported_geometry.py', 'support_reference.py', 'control_spec.json', 'support_reference_spec.json',
             'controller.py', 'EXPERIMENT_PLAN.md', 'SOURCE_REVIEW.md', 'SOURCE_REVIEW.json'):
    raw = (local / name).read_bytes()
    with sftp.open(root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['flock', '-n', env['resource_limits']['gpu_lock'], python, '-B', '-u',
                     root + '/controller.py', '--phase', 'preflight'])
inner = command + ' > ' + shlex.quote(root + '/preflight_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/preflight_controller.exit') + '; exit "$code"'
screen = 'pvg_support_reference_preflight_20261005'
_, stdout, stderr = client.exec_command(shlex.join(['screen', '-dmS', screen, 'bash', '-c', inner]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/controller.py --phase preflight$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root, process=process, screen=screen,
    status='PREFLIGHT_LAUNCHED_NOT_COMPLETED', optimizer_steps_planned_per_arm=2, accuracy_result=False,
    weight_files_planned=0, estimated_seconds=1200, first_check_seconds=900, later_poll_seconds=240,
    estimate_basis='Two sequential protected4511 reconstructions; own-support reference and extra reference gradient/replay check. No formal accuracy.')
(local / 'preflight_launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(record), flush=True)
