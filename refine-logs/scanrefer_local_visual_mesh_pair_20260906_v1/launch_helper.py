import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
probe = repo / 'refine-logs/scanrefer_local_visual_mesh_preflight_20260906_v1'
probe_raw = (probe / 'receipt.json').read_bytes()
checked = json.loads(probe_raw)
assert checked['status'] == 'pass' and checked['disposable_optimizer_steps'] == 2
assert checked['real_train_rows'] == 16 and checked['formal_rows'] == 0
assert checked['checkpoint_writes'] == 0
assert checked['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert all(row['zero_initialization_output_parity'] and row['v99_runtime_parity'] for row in checked['observations'])
local = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
manifest = json.loads((probe / 'input_manifest.json').read_bytes())
manifest.update(mode='train', split_protocol=remote + '/split_protocol.json',
                native_probe_receipt=remote + '/native_probe_receipt.json',
                native_probe_receipt_sha256=hashlib.sha256(probe_raw).hexdigest())
files = {name: (repo / name).read_bytes() for name in manifest['files']}
for name, raw in files.items():
    assert hashlib.sha256(raw).hexdigest() == manifest['files'][name], name
raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd ''' + remote + '''
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_local_visual_pair.py --manifest ''' + remote + '''/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
_, stdout, stderr = client.exec_command('nvidia-smi --query-compute-apps=pid --format=csv,noheader', timeout=30)
assert not stdout.read().strip() and stdout.channel.recv_exit_status() == 0
sftp = client.open_sftp()
with sftp.open('/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_preflight_20260906_v1/receipt.json', 'rb') as stream:
    assert stream.read() == probe_raw
local.mkdir()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
uploads = dict(files)
uploads.update({'input_manifest.json': raw_manifest, 'native_probe_receipt.json': probe_raw,
                'split_protocol.json': (probe / 'split_protocol.json').read_bytes(),
                'plan.md': (probe / 'plan.md').read_bytes(), 'controller.sh': controller.encode(),
                'data_inputs.json': (probe / 'data_inputs.json').read_bytes(),
                'launch_helper.py': Path(__file__).read_bytes()})
for name, raw in uploads.items():
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    target = local / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
command = 'screen -dmS mcln_scanrefer_local_visual_mesh_pair_v1 bash -lc ' + shlex.quote(
    'cd ' + shlex.quote(remote) + ' && bash controller.sh > run.log 2>&1')
_, stdout, stderr = client.exec_command(command, timeout=30)
stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = client.exec_command('screen -ls', timeout=30)
sessions = stdout.read().decode()
assert stdout.channel.recv_exit_status() == 0
matching = [line.strip() for line in sessions.splitlines() if 'mcln_scanrefer_local_visual_mesh_pair_v1' in line]
assert len(matching) == 1
receipt = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen_session': matching, 'command': command, 'steps_per_arm_planned': 2482,
    'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
    'controller_sha256': hashlib.sha256(controller.encode()).hexdigest(),
    'new_model_parameters': checked['local_parameters'],
    'preflight_model_not_reused': True, 'initialization': 'protected E71 plus zero-output local branch',
    'no_new_full_pretrained_weight_copy': True, 'data_root': manifest['data_root'], 'launch_does_not_prove_optimizer_update': True}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
with sftp.open(remote + '/launch.json', 'wx') as stream:
    stream.write(raw)
(local / 'launch.json').write_bytes(raw)
sftp.close()
client.close()
print(raw.decode())
