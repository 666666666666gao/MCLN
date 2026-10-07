"""Stage the isolated source port and start only a source-accepted real GPU M0."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
assert not (local / 'preflight_launch.json').exists()
review = json.loads((local / 'SOURCE_REVIEW_SUPPLEMENT.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
spec = json.loads((local / 'pair_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_mask_support_correction_20261008'
assert spec['root'] == root and spec['shared_frozen_parent_forward']
assert json.loads((local.parent / 'pvground_support_boundary_cases_20261007/analysis/ACTUAL_EVIDENCE_REVIEW.json').read_bytes())['blocking_findings'] == []
assert json.loads((local.parent / 'pvground_face_support_20261007/weight_retention.json').read_bytes())['status'] == 'CLOSED_NONBEST_WEIGHTS_REMOVED'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
python = spec['runtime'] + '/venv/bin/python'
stage = '''import base64,hashlib,json,shutil,subprocess,sys
from pathlib import Path
spec=json.load(sys.stdin);root=Path(spec['root']);assert not root.exists()
old=Path(spec['old_model_source']);port_path=Path(spec['old_source_port'])
assert hashlib.sha256(port_path.read_bytes()).hexdigest()==spec['old_source_port_sha256']
port=json.loads(port_path.read_bytes());sources=[]
for name,digest in port['files'].items():
    source=old/name;assert old.resolve() in source.resolve().parents
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
    sources.append((name,source))
required=sum(path.stat().st_size for name,path in sources)+spec['payload_bytes']+256*1024**2
assert shutil.disk_usage(root.parent).free>required
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
previous=Path('/root/autodl-tmp/pvground_face_support_20261007')
assert (previous/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((previous/'fit_status.json').read_bytes())['status']=='complete'
best=Path(spec['selected_terminal'])
assert hashlib.sha256(best.read_bytes()).hexdigest()==spec['selected_terminal_sha256']
root.mkdir()
for arm in spec['support_modes']:(root/arm).mkdir()
for name,source in sources:
    dest=root/'PV-Ground'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if name=='models/pv_ground.py':
        dest.write_bytes(base64.b64decode(spec['pv_overlay']))
    else:shutil.copyfile(str(source),str(dest))
print(json.dumps(dict(GPU_idle=True,protected4848_exact=True,copied_source_files=len(sources),
    copied_source_bytes=sum(path.stat().st_size for name,path in sources),
    required_preflight_reserve_bytes=required,data_free_bytes=shutil.disk_usage(root).free,
    system_free_bytes=shutil.disk_usage('/').free,environment_rebuilt=False)))
'''
names = list(spec['new_runner_files']) + ['source_port.json','pair_spec.json','controller.py',
    'EXPERIMENT_PLAN.md','SOURCE_REVIEW.json','SOURCE_REVIEW.md',
    'SOURCE_REVIEW_SUPPLEMENT.json','SOURCE_REVIEW_SUPPLEMENT.md']
payload = dict(spec, payload_bytes=sum((local / name).stat().st_size for name in names),
    pv_overlay=base64.b64encode((local / 'runtime_overlay/PV-Ground/models/pv_ground.py').read_bytes()).decode())
stdin, stdout, stderr = client.exec_command(shlex.join([python, '-B', '-c', stage]), timeout=180)
stdin.write(json.dumps(payload))
stdin.channel.shutdown_write()
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local / 'resource_check.json').write_bytes(raw)
for name in names:
    data = (local / name).read_bytes()
    with sftp.open(root + '/' + name, 'wx') as stream:
        stream.write(data)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == data
command = shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],python,'-B','-u',
    root + '/controller.py','--phase','preflight'])
shell = command + ' > ' + shlex.quote(root + '/preflight_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/preflight_controller.exit') + '; exit "$code"'
screen = 'pvg_support_correction_preflight_20261008'
_, stdout, stderr = client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',shell]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/controller.py --phase preflight$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,
    controller_pid=int(process.split()[0]),process=process,screen=screen,resource=resources,
    status='PREFLIGHT_LAUNCHED_NOT_PASSED',optimizer_steps_planned_per_arm=2,accuracy_result=False,
    estimated_seconds=900,first_check_seconds=720,later_poll_seconds=240,
    estimate_basis='One protected PV reconstruction, two head updates, two full CPU state reconstructions and two native GPU route comparisons; previous shared-parent M0 plus extra forwards estimates15min. First check3min before endpoint.')
(local / 'preflight_launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with sftp.open(root + '/preflight_launch.json', 'wx') as stream:
    stream.write((local / 'preflight_launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record), flush=True)
