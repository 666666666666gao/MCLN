"""Run the independent CPU audit only after the actual formal job completes."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(remote + '/controller.exit', 'rb') as stream:
    assert stream.read().strip() == b'0'
with sftp.open(remote + '/result/receipt.json', 'rb') as stream:
    receipt = json.loads(stream.read())
assert receipt['status'] == 'complete' and receipt['formal_rows'] == 9508
assert 'independent_audit.json' not in sftp.listdir(remote + '/result')
audit_root = remote + '/audit_code'
sftp.mkdir(audit_root)
sftp.mkdir(audit_root + '/scripts')
local_code = local / 'audit_code'
(local_code / 'scripts').mkdir(parents=True)
names = ['scripts/audit_scanrefer_local_visual_official.py',
         'scripts/audit_scanrefer_joint_readout_pair.py',
         'scripts/evaluate_scanrefer_local_visual_official.py']
files = {name: (repo / name).read_bytes() for name in names}
files['scripts/__init__.py'] = b''
for name, raw in files.items():
    (local_code / name).write_bytes(raw)
    with sftp.open(audit_root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(audit_root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
(local_code / 'files.json').write_bytes(raw)
with sftp.open(audit_root + '/files.json', 'wx') as stream:
    stream.write(raw)
command = ('cd ' + shlex.quote(audit_root)
           + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
           + ' /root/miniconda3/envs/bdetr/bin/python -m scripts.audit_scanrefer_local_visual_official '
           + shlex.quote(remote) + ' ' + shlex.quote(remote + '/result/independent_audit.json')
           + ' > ' + shlex.quote(remote + '/result/independent_audit.log') + ' 2>&1')
_, output, error = client.exec_command(command, timeout=300)
output.read()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
with sftp.open(remote + '/result/independent_audit.log', 'rb') as stream:
    audit_log = stream.read()
(local / 'result/independent_audit_run.txt').write_bytes(audit_log)
assert status == 0, audit_log.decode() + error_text
downloaded = {}
names = ['controller.exit', 'result/receipt.json', 'result/rows.json', 'result/native_rows.json',
         'result/protocol.json', 'result/independent_audit.json', 'run.log']
for name in names:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    (local / ('run.txt' if name == 'run.log' else name)).write_bytes(raw)
    downloaded[name] = hashlib.sha256(raw).hexdigest()
sftp.close()
client.close()
audit = json.loads((local / 'result/independent_audit.json').read_bytes())
assert audit['integrity_pass']
assert audit['receipt_sha256'] == downloaded['result/receipt.json']
assert audit['audit_script_sha256'] == manifest['scripts/audit_scanrefer_local_visual_official.py']
print(json.dumps({'integrity_pass': True, 'metrics': audit['metrics'],
                  'native_rec_metrics': audit['native_rec_metrics'], 'promotion': audit['promotion'],
                  'system_effects': audit['system_local_minus_protected']['effects'],
                  'native_effects': audit['native_local_minus_protected']['effects'],
                  'downloaded_sha256': downloaded, 'weights_downloaded': 0}))
