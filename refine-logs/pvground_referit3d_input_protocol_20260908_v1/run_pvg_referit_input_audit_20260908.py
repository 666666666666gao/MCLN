import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_referit3d_input_protocol_20260908_v1'
root = '/root/autodl-tmp/mcln_pvground_referit3d_input_protocol_20260908_v1'
source = (repo / 'scripts/audit_pvground_referit3d_inputs.py').read_bytes()
compile(source, 'audit_pvground_referit3d_inputs.py', 'exec')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(root)
with sftp.open(root + '/audit.py', 'wb') as stream:
    stream.write(source)
with sftp.open(root + '/audit.py', 'rb') as stream:
    assert stream.read() == source
argv = ['/root/miniconda3/envs/bdetr/bin/python', '-u', root + '/audit.py',
        '--runtime', '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1',
        '--input-manifest', '/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json',
        '--output', root + '/receipt.json']
_, stdout, stderr = client.exec_command(' '.join(shlex.quote(arg) for arg in argv), timeout=60)
raw = stdout.read()
err = stderr.read()
code = stdout.channel.recv_exit_status()
for name, data in [('audit.py', source), ('stdout.txt', raw), ('stderr.txt', err), ('audit.exit', (str(code) + '\n').encode())]:
    (archive / name).write_bytes(data)
    if name != 'audit.py':
        with sftp.open(root + '/' + name, 'wb') as stream:
            stream.write(data)
print(raw.decode(), end='')
print(err.decode(), end='')
assert code == 0, code
sftp.get(root + '/receipt.json', str(archive / 'receipt.json'))
receipt = json.loads((archive / 'receipt.json').read_bytes())
assert receipt['status'] == 'pass'
record = {'remote_root': root, 'argv': argv, 'audit_source_sha256': hashlib.sha256(source).hexdigest(),
          'receipt_sha256': hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest(), 'exit_code': code}
(archive / 'execution.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
