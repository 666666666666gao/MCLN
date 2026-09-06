import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
previous = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1'
formal = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3'
local = repo / 'refine-logs/scanrefer_local_visual_mesh_preflight_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_preflight_20260906_v1'
data_raw = (formal / 'data_inputs.json').read_bytes()
data = json.loads(data_raw)
assert data['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp'
data_root = data['data_root'] + '/'
assert {key: len(value) for key, value in data['superpoint_files'].items()} == {'train': 1201, 'val': 312}
manifest = json.loads((previous / 'input_manifest.json').read_bytes())
manifest.pop('native_probe_receipt')
manifest.pop('native_probe_receipt_sha256')
names = list(manifest['files']) + ['scripts/scanrefer_data_contract.py']
files = {name: (repo / name).read_bytes() for name in names}
plan = (repo / 'docs/SCANREFER_LOCAL_VISUAL_MESH_REPEAT_2026-09-06.md').read_bytes()
split = (previous / 'split_protocol.json').read_bytes()
assert hashlib.sha256(split).hexdigest() == manifest['split_protocol_sha256']
manifest.update(mode='preflight', data_root=data_root, superpoint_files=data['superpoint_files'],
    split_protocol=remote + '/split_protocol.json', plan_sha256=hashlib.sha256(plan).hexdigest(),
    files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    repeat_of='scanrefer_local_visual_pair_20260906_v1',
    changed_experimental_input='mesh-derived train and val superpoints; all model/loss/update settings fixed')
raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd ''' + shlex.quote(remote) + '''
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_local_visual_pair.py --manifest ''' + shlex.quote(remote + '/input_manifest.json') + '''
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
check = '''import datetime,json,os,shutil,subprocess
from pathlib import Path
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip()
free=shutil.disk_usage('/root/autodl-tmp').free
assert not apps, apps
assert int(memory.split(',')[0]) < 500, memory
assert free > 3 * 1024**3, free
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'uid':os.getuid(),'hostname':os.uname().nodename,'gpu_memory_mib':memory,'compute_processes':apps,'disk_free_bytes':free}))
'''
_, out, err = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(check), timeout=30)
raw_check = out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
resource_check = json.loads(raw_check)
sftp = client.open_sftp()
with sftp.open('/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v3/data_inputs.json', 'rb') as stream:
    assert stream.read() == data_raw
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
with sftp.open('/home/gb/new butd/butd_detr-main/MCLN-main/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md', 'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == hashlib.sha256(master.read_bytes()).hexdigest()
local.mkdir()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
uploads = dict(files)
uploads.update({'input_manifest.json': raw_manifest, 'plan.md': plan, 'split_protocol.json': split,
    'data_inputs.json': data_raw, 'controller.sh': controller.encode(), 'launch_helper.py': Path(__file__).read_bytes()})
for name, raw in uploads.items():
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    target = local / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).digest() == hashlib.sha256(raw).digest(), name
_, out, err = client.exec_command('bash -n ' + shlex.quote(remote + '/controller.sh'), timeout=30)
out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
command = 'screen -dmS mcln_scanrefer_mesh_preflight_v1 bash -lc ' + shlex.quote(
    'cd ' + shlex.quote(remote) + ' && bash controller.sh > run.log 2>&1')
_, out, err = client.exec_command(command, timeout=30)
out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = client.exec_command('screen -ls', timeout=30)
sessions = out.read().decode()
assert out.channel.recv_exit_status() == 0, err.read().decode()
matching = [line.strip() for line in sessions.splitlines() if 'mcln_scanrefer_mesh_preflight_v1' in line]
assert len(matching) == 1, sessions
receipt = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen_session': matching, 'command': command, 'resource_check': resource_check,
    'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(), 'plan_sha256': hashlib.sha256(plan).hexdigest(),
    'controller_sha256': hashlib.sha256(controller.encode()).hexdigest(), 'data_root': data_root,
    'scope': '16 fixed fit rows and two disposable updates; zero checkpoint writes',
    'status': 'launched; actual preflight pass still unproven'}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
with sftp.open(remote + '/launch.json', 'wx') as stream:
    stream.write(raw)
(local / 'launch.json').write_bytes(raw)
sftp.close()
client.close()
print(raw.decode())
