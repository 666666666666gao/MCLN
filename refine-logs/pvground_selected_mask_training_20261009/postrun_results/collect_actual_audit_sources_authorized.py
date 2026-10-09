"""Copy the exact imported source files from the completed actual run."""
import hashlib
import json
import os
from pathlib import Path

import paramiko


root = Path(__file__).resolve().parent
actual = root / 'complete_fit'
wait = json.loads((root / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
destination = root / 'postrun_results/audit_sources'
assert not destination.exists()
destination.mkdir(parents=True)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open('/root/autodl-tmp/pvground_selected_mask_training_20261009/imports.json', 'rb') as stream:
    imports_raw = stream.read()
imports = json.loads(imports_raw)
assert set(imports['files']) == set(imports['sha256'])
assert imports == json.loads((root / 'preflight_complete/imports.json').read_bytes())
(destination / 'SOURCE_IMPORTS.json').write_bytes(imports_raw)
rows = []
for module, remote_path in sorted(imports['files'].items()):
    with sftp.open(remote_path, 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == imports['sha256'][module]
    path = destination / (module.replace('.', '_') + '.py')
    path.write_bytes(raw)
    assert path.read_bytes() == raw
    rows.append(dict(module=module, remote_path=remote_path, local_path=str(path),
                     bytes=len(raw), sha256=imports['sha256'][module]))
sftp.close()
client.close()
record = dict(status='ACTUAL_IMPORTED_SOURCE_BYTES_COLLECTED', files=rows,
              imports_sha256=hashlib.sha256(imports_raw).hexdigest(),
              training_status_reads=0, new_neural_forwards=0, deletions=0)
(destination / 'SOURCE_INTAKE.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
