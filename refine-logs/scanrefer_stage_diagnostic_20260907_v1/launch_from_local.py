import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
prepared = repo / 'refine-logs/scanrefer_stage_diagnostic_preparation_20260907_v2'
reference = repo / 'refine-logs/scanrefer_local_visual_mesh_official_20260906_v1'
acceptance = repo / 'refine-logs/scanrefer_local_visual_mesh_acceptance_queue_20260906_v1'
local = repo / 'refine-logs/scanrefer_stage_diagnostic_20260907_v1'
remote_reference = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

decision = json.loads((acceptance / 'decision.json').read_bytes())
assert (acceptance / 'controller.exit').read_text().strip() == '0'
assert not decision['promotion']['advance_to_nr3d_sr3d_rec']
assert not decision['native_gpu_preflight_launched'] and not decision['nr3d_sr3d_training_started']
reference_names = ['input_manifest.json', 'controller.exit', 'result/receipt.json',
                   'result/rows.json', 'result/native_rows.json', 'result/protocol.json',
                   'result/independent_audit.json']
reference_raw = {name: (reference / name).read_bytes() for name in reference_names}
assert sha(reference_raw['result/receipt.json']) == decision['formal_receipt_sha256']
assert sha(reference_raw['result/independent_audit.json']) == decision['formal_audit_sha256']
audit = json.loads(reference_raw['result/independent_audit.json'])
assert audit['integrity_pass'] and audit['formal_rows'] == 9508
base = json.loads(reference_raw['input_manifest.json'])
preparation = json.loads((prepared / 'manifest.json').read_bytes())
imports = json.loads((prepared / 'import_binding.json').read_bytes())
for name, digest in dict(preparation['files'], **imports['added_files']).items():
    assert sha((prepared / name).read_bytes()) == digest, name
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
for name, raw in reference_raw.items():
    with sftp.open(remote_reference + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw, name
_, output, error = client.exec_command('nvidia-smi --query-compute-apps=pid --format=csv,noheader', timeout=30)
gpu_processes = output.read().decode().strip()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert not gpu_processes, 'GPU is in use; do not launch a concurrent job.'
files = {}
for name, digest in base['files'].items():
    with sftp.open(remote_reference + '/' + name, 'rb') as stream:
        raw = stream.read()
    assert sha(raw) == digest, name
    files[name] = raw
for name in ['scripts/diagnose_scanrefer_readout_stages.py', 'scripts/scanrefer_stage_diagnostics.py',
             'scripts/trace_scanrefer_readout_stages.py']:
    assert name not in files
    files[name] = (prepared / name).read_bytes()
files['scripts/__init__.py'] = b''
manifest = {'schema': 'mcln-scanrefer-stage-diagnostic-input-v1',
            'reference_formal_directory': remote_reference,
            'reference_files': {name: sha(raw) for name, raw in reference_raw.items()},
            'training_directory': base['training_directory'],
            'training_receipt_sha256': base['training_receipt_sha256'],
            'data_root': base['data_root'], 'val_superpoint_files': base['val_superpoint_files'],
            'files': {name: sha(raw) for name, raw in files.items()},
            'formal_rows': 0, 'diagnostic_rows': 9508, 'optimizer_updates': 0,
            'checkpoint_writes': 0, 'used_for_promotion': False,
            'preparation_receipt_sha256': sha((prepared / 'receipt.json').read_bytes()),
            'runtime_import_receipt_sha256': sha((prepared / 'import_receipt.json').read_bytes()),
            'purpose': 'Locate native-to-Parent-to-Geometry-to-V99 changes for two fixed checkpoints;no new training or threshold search.'}
files['input_manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd DIRECTORY
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/diagnose_scanrefer_readout_stages.py --manifest MANIFEST
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.replace('DIRECTORY', shlex.quote(remote)).replace('MANIFEST', shlex.quote(remote + '/input_manifest.json'))
files['controller.sh'] = controller.encode()
files['launch_from_local.py'] = Path(__file__).read_bytes()
local.mkdir()
(local / 'scripts').mkdir()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
for name, raw in files.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == raw, name
_, output, error = client.exec_command('bash -n ' + shlex.quote(remote + '/controller.sh'), timeout=30)
assert output.channel.recv_exit_status() == 0, error.read().decode()
screen = 'mcln_scanrefer_stage_diagnostic_v1'
command = 'screen -dmS ' + screen + ' bash -lc ' + shlex.quote('cd ' + shlex.quote(remote) + ' && bash controller.sh > run.log 2>&1')
_, output, error = client.exec_command(command, timeout=30)
assert output.channel.recv_exit_status() == 0, error.read().decode()
_, output, error = client.exec_command('screen -ls', timeout=30)
sessions = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
selected = [line.strip() for line in sessions.splitlines() if '.' + screen in line]
assert len(selected) == 1
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
launch = {'time_cst': now.isoformat(), 'screen_session': selected,
          'manifest_sha256': sha(files['input_manifest.json']),
          'controller_sha256': sha(files['controller.sh']), 'command': command,
          'diagnostic_rows_planned': 9508, 'formal_rows': 0, 'optimizer_updates': 0,
          'checkpoint_writes': 0, 'used_for_promotion': False,
          'first_probe_cst': (now + datetime.timedelta(seconds=180)).isoformat(),
          'status': 'isolated diagnostic launched;first real-forward verification and metrics pending'}
raw = (json.dumps(launch, indent=2, sort_keys=True) + '\n').encode()
(local / 'launch.json').write_bytes(raw)
with sftp.open(remote + '/launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(launch))
