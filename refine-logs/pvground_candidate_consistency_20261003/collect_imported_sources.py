"""Collect the exact native sources recorded by the completed paired run."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path(__file__).parent
output = local / 'complete/source/imported'
assert not output.exists()
control = json.loads((local / 'complete/g_control/imports.json').read_bytes())
method = json.loads((local / 'complete/g_consistent/imports.json').read_bytes())
assert control == method
keys = ['evaluator', 'main_utils', 'models.losses', 'models.pv_ground', 'prepare_data', 'src.joint_det_dataset']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
data = {}
for key in keys:
    with sftp.open(control['files'][key], 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == control['sha256'][key]
    data[key + '.py'] = raw
client.close()
output.mkdir()
files = {}
for name, raw in data.items():
    (output / name).write_bytes(raw)
    files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
receipt = {'time_cst': datetime.datetime.now().astimezone().isoformat(), 'files': files,
           'matches_both_native_import_manifests': True, 'GPU_forwards': 0,
           'optimizer_updates': 0, 'remote_source_changed': False}
(output / 'INTAKE.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
