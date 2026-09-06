import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1'
preparation = '/root/autodl-tmp/mcln_scanrefer_local_visual_audit_preparation_20260906_v1'
expected = json.loads((repo / 'refine-logs/scanrefer_local_visual_audit_preparation_20260906_v1/preparation_manifest.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(remote + '/controller.exit', 'rb') as stream:
    assert stream.read().strip() == b'0'
with sftp.open(remote + '/receipt.json', 'rb') as stream:
    receipt = json.loads(stream.read())
assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482
assert receipt['fixed_endpoint_ready_for_official_evaluation'] and receipt['formal_rows'] == 0
assert 'independent_audit.json' not in sftp.listdir(remote)
for name, digest in expected['files'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
    with sftp.open(preparation + '/' + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest, name
command = ('cd ' + shlex.quote(preparation)
           + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false"
           + ' /root/miniconda3/envs/bdetr/bin/python -m scripts.audit_scanrefer_local_visual_pair '
           + shlex.quote(remote) + ' ' + shlex.quote(remote + '/independent_audit.json')
           + ' > ' + shlex.quote(remote + '/independent_audit.log') + ' 2>&1')
_, stdout, stderr = client.exec_command(command, timeout=300)
stdout.read()
error = stderr.read().decode()
status = stdout.channel.recv_exit_status()
with sftp.open(remote + '/independent_audit.log', 'rb') as stream:
    output = stream.read()
(local / 'independent_audit_run.txt').write_bytes(output)
assert status == 0, output.decode() + error
names = ['controller.exit', 'receipt.json', 'protocol.json', 'fit_complete.json',
         'baseline_rows.json', 'terminal_rows.json', 'fit_point_batches.json',
         'baseline_metrics.json', 'terminal_metrics.json', 'independent_audit.json', 'run.log']
downloaded = {}
for name in names:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=sftp.stat(remote + '/' + name).st_size)
        raw = stream.read()
    (local / ('run.txt' if name == 'run.log' else name)).write_bytes(raw)
    downloaded[name] = hashlib.sha256(raw).hexdigest()
sftp.close()
client.close()
audit = json.loads((local / 'independent_audit.json').read_bytes())
assert audit['integrity_pass'] and audit['receipt_sha256'] == downloaded['receipt.json']
assert audit['audit_script_sha256'] == expected['files']['scripts/audit_scanrefer_local_visual_pair.py']
print(json.dumps({'integrity_pass': audit['integrity_pass'], 'metrics': audit['metrics'],
                  'development_dual_rec_nonregression': audit['development_dual_rec_nonregression'],
                  'downloaded_sha256': downloaded, 'weights_downloaded': 0}))
