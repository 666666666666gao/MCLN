"""Read the closed formal receipt once while the full collection continues."""
import hashlib
import json
import os
from pathlib import Path
import paramiko

root = Path(__file__).resolve().parent
wait = json.loads((root / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
destination = root / 'postrun_results/FORMAL_PREVIEW.json'
assert not destination.exists()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open('/root/autodl-tmp/pvground_selected_mask_training_20261009/formal/receipt.json', 'rb') as stream:
    raw = stream.read()
sftp.close()
client.close()
record = json.loads(raw)
assert record['status'] == 'pass' and record['rows'] == record['formal_rows'] == 9508
destination.parent.mkdir(exist_ok=True)
destination.write_bytes(raw)
print(json.dumps(dict(status='CLOSED_FORMAL_RECEIPT_PREVIEW_ONLY', sha256=hashlib.sha256(raw).hexdigest(),
    metrics=record['metrics'], cpu_recount_pending=True, promotion=False)), flush=True)
