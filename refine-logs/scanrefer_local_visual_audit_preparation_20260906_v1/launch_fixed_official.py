import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
sys.path.insert(0, str(repo))

training_local = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1'
training_remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1'
receipt_raw = (training_local / 'receipt.json').read_bytes()
receipt = json.loads(receipt_raw)
assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482
assert receipt['formal_rows'] == 0 and receipt['fixed_endpoint_ready_for_official_evaluation']
audit_raw = (training_local / 'independent_audit.json').read_bytes()
audit = json.loads(audit_raw)
assert audit['schema'] == 'mcln-scanrefer-local-visual-independent-audit-v1'
assert audit['integrity_pass'] and audit['fit_traversal_and_paired_point_identity_verified']
assert audit['receipt_sha256'] == hashlib.sha256(receipt_raw).hexdigest()
assert audit['audit_script_sha256'] == hashlib.sha256((repo / 'scripts/audit_scanrefer_local_visual_pair.py').read_bytes()).hexdigest()
for arm, tensors in [('control', 66), ('local', 76)]:
    assert audit['checkpoints'][arm]['optimizer_parameter_tensors'] == tensors
    assert audit['checkpoints'][arm]['optimizer_steps'] == 2482
    assert audit['checkpoints'][arm]['frozen_core_and_buffers_unchanged']
    assert audit['checkpoints'][arm]['readout_parameters_and_metadata_unchanged']
prepared = json.loads((repo / 'refine-logs/scanrefer_local_visual_official_preparation_20260906_v1/receipt.json').read_bytes())
assert prepared['status'] == 'entry_checks_pass'
filenames = ['scripts/evaluate_scanrefer_local_visual_official.py',
             'scripts/scanrefer_joint_readout.py', 'scripts/scanrefer_rec_evaluation.py']
files = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in filenames}
assert prepared['files'][filenames[0]] == files[filenames[0]]
train_manifest = json.loads((training_local / 'input_manifest.json').read_bytes())
assert all(train_manifest['files'][name] == files[name] for name in filenames[1:])
local = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1'
manifest = {'schema': 'mcln-scanrefer-local-visual-official-input-v1',
    'training_directory': training_remote,
    'training_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
    'training_audit_sha256': hashlib.sha256(audit_raw).hexdigest(),
    'trained_checkpoint': receipt['checkpoints']['local'], 'files': files,
    'formal_rows': 9508, 'optimizer_steps': 0,
    'arms': ['protected_v99', 'local_v99'],
    'decision': 'Evaluate the single fixed local-visual endpoint after independent integrity audit;development metrics do not select epochs.',
    'scan_rec_historical_floor_hits': [5572, 4797],
    'scan_mask_paper_floor_percent': [58.70, 50.70, 44.72],
    'nr3d_sr3d_mask_gate': False}
raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
with s.open(training_remote + '/receipt.json') as stream:
    assert stream.read() == receipt_raw
with s.open(training_remote + '/independent_audit.json') as stream:
    assert stream.read() == audit_raw
with s.open(training_remote + '/controller.exit') as stream:
    assert stream.read().strip() == b'0'
local.mkdir()
s.mkdir(remote)
s.mkdir(remote + '/scripts')
uploads = {'input_manifest.json': raw}
uploads.update({name: (repo / name).read_bytes() for name in filenames})
for name, data in uploads.items():
    with s.open(remote + '/' + name, 'wx') as stream:
        stream.write(data)
    (local / Path(name).name).write_bytes(data)
native = '/root/miniconda3/envs/bdetr/bin/python -u scripts/evaluate_scanrefer_local_visual_official.py --manifest ' + remote + '/input_manifest.json'
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd ''' + remote + '''
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock ''' + native + '''
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
with s.open(remote + '/controller.sh', 'wx') as stream:
    stream.write(controller.encode())
(local / 'controller.sh').write_bytes(controller.encode())
command = 'screen -dmS mcln_scanrefer_local_visual_official_v1 bash -lc ' + shlex.quote('cd ' + shlex.quote(remote) + ' && bash controller.sh > run.log 2>&1')
_, out, err = c.exec_command(command, timeout=30)
out.read()
error = err.read().decode()
assert out.channel.recv_exit_status() == 0, error
_, out, err = c.exec_command('screen -ls', timeout=30)
sessions = out.read().decode()
assert out.channel.recv_exit_status() == 0 and 'mcln_scanrefer_local_visual_official_v1' in sessions
result = {'schema': 'mcln-scanrefer-local-visual-official-launch-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen_session': [line.strip() for line in sessions.splitlines() if 'mcln_scanrefer_local_visual_official_v1' in line],
    'manifest_sha256': hashlib.sha256(raw).hexdigest(),
    'controller_sha256': hashlib.sha256(controller.encode()).hexdigest(),
    'command': command, 'native_command': native, 'formal_rows_planned': 9508,
    'optimizer_steps': 0, 'launch_is_not_completed_evaluation': True}
data = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
with s.open(remote + '/launch.json', 'wx') as stream:
    stream.write(data)
(local / 'launch.json').write_bytes(data)
s.close()
c.close()
print(json.dumps(result))
