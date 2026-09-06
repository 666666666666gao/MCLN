import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_mesh_official_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1'
manifest_raw = (local / 'input_manifest.json').read_bytes()
manifest = json.loads(manifest_raw)
post = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
preparation_raw = (post / 'preparation.json').read_bytes()
assert hashlib.sha256(preparation_raw).hexdigest() == '642f8facc62c30b623b37296672bc4c5385c947dabebc15d88eefc65712eec84'
preparation = json.loads(preparation_raw)
launch_raw = (post / 'executed.json').read_bytes()
launch = json.loads(launch_raw)
assert hashlib.sha256(manifest_raw).hexdigest() == launch['manifest_sha256']
assert manifest['training_directory'] == preparation['training_directory']
assert remote == preparation['formal_directory']
assert manifest['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert manifest['files'] == {name: preparation['files'][name]
                             for name in preparation['formal_files'] if name != 'scripts/__init__.py'}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(remote + '/input_manifest.json', 'rb') as stream:
    assert stream.read() == manifest_raw
with sftp.open('/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_posttraining_20260906_v1/executed.json', 'rb') as stream:
    assert stream.read() == launch_raw
with sftp.open(remote + '/controller.exit', 'rb') as stream:
    assert stream.read().strip() == b'0'
with sftp.open(remote + '/result/receipt.json', 'rb') as stream:
    receipt = json.loads(stream.read())
assert receipt['schema'] == 'mcln-scanrefer-local-visual-official-v2'
assert receipt['status'] == 'complete' and receipt['formal_rows'] == 9508
assert receipt['data_root'] == manifest['data_root']
assert 'independent_audit.json' not in sftp.listdir(remote + '/result')
for name, digest in manifest['files'].items():
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest, name
command = ('cd ' + shlex.quote(remote)
           + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
           + ' /root/miniconda3/envs/bdetr/bin/python -m scripts.audit_scanrefer_local_visual_official '
           + shlex.quote(remote) + ' ' + shlex.quote(remote + '/result/independent_audit.json')
           + ' > ' + shlex.quote(remote + '/result/independent_audit.log') + ' 2>&1')
_, output, error = client.exec_command(command, timeout=300)
output.read()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
with sftp.open(remote + '/result/independent_audit.log', 'rb') as stream:
    log = stream.read()
(local / 'result').mkdir(exist_ok=True)
(local / 'result/independent_audit_run.txt').write_bytes(log)
assert status == 0, log.decode() + error_text
downloaded = {}
for name in ['controller.exit', 'result/receipt.json', 'result/rows.json', 'result/native_rows.json',
             'result/protocol.json', 'result/independent_audit.json', 'run.log']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    (local / ('run.txt' if name == 'run.log' else name)).write_bytes(raw)
    downloaded[name] = hashlib.sha256(raw).hexdigest()
sftp.close()
client.close()
audit = json.loads((local / 'result/independent_audit.json').read_bytes())
assert audit['integrity_pass'] and audit['receipt_sha256'] == downloaded['result/receipt.json']
print(json.dumps({'integrity_pass': True, 'metrics': audit['metrics'],
                  'native_rec_metrics': audit['native_rec_metrics'], 'promotion': audit['promotion'],
                  'system_effects': audit['system_local_minus_protected']['effects'],
                  'native_effects': audit['native_local_minus_protected']['effects'],
                  'downloaded_sha256': downloaded, 'weights_downloaded': 0}))
