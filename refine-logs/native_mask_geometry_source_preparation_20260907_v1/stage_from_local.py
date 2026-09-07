import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/native_mask_geometry_source_preparation_20260907_v1'
remote = '/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
assert commit == '9c7c9f44ecfdada79cddfcfedfe2cfd47957b935'
overlay_names = ['main_utils.py', 'models/losses.py',
    'scripts/native_mask_geometry_supervision.py', 'scripts/native_teacher_box_transfer.py',
    'scripts/prototype_probability_geometry.py', 'tests/test_native_mask_geometry_training.py']
payloads = {name: subprocess.check_output(['git', 'show', commit + ':' + name], cwd=str(repo))
            for name in overlay_names}
manifest = {'schema': 'mcln-native-mask-geometry-source-preparation-input-v1',
    'parent_source': '/root/autodl-tmp/mcln_native_range_preparation_20260907_v1/model_source',
    'parent_manifest_sha256': '5d3dfcba7df8f34b31127e30b93e3e1ca52475e435e9ba6f2e25584330c562a7',
    'overlay_git_commit': commit,
    'overlays': {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
    'running_training': '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1',
    'running_manifest_sha256': '15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3',
    'gpu_training_authorized_at_this_stage': False, 'actual_endpoint_bound': False}
(local / 'input_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = c.open_sftp()
sftp.mkdir(remote)
sftp.mkdir(remote + '/overlays')
for folder in ['models', 'scripts', 'tests']:
    sftp.mkdir(remote + '/overlays/' + folder)
for name, raw in payloads.items():
    with sftp.open(remote + '/overlays/' + name, 'wb') as stream:
        stream.write(raw)
for name in ['prepare_source.py', 'check_complete_source.py', 'input_manifest.json']:
    sftp.put(str(local / name), remote + '/' + name)
command = 'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python ' + shlex.quote(remote + '/prepare_source.py') + ' ' + shlex.quote(remote)
_, stdout, stderr = c.exec_command(command, timeout=55)
body, error = stdout.read(), stderr.read()
code = stdout.channel.recv_exit_status()
(local / 'prepare_stdout.txt').write_bytes(body)
(local / 'prepare_stderr.txt').write_bytes(error)
print(body.decode(), flush=True)
print(error.decode(), flush=True)
for entry in sftp.listdir_attr(remote):
    if entry.filename in ['receipt.json', 'check_0.txt', 'check_1.txt']:
        sftp.get(remote + '/' + entry.filename, str(local / entry.filename))
sftp.get(remote + '/model_source/native_source_manifest.json', str(local / 'native_source_manifest.json'))
print(json.dumps({'prepare_exit': code, 'directory': remote,
                  'receipt_exists': (local / 'receipt.json').exists()}), flush=True)
sftp.close()
c.close()
raise SystemExit(code)
