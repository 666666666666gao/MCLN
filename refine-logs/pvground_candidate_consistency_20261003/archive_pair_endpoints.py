"""Archive completed nonleading pair endpoints; deletion is a separate action."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import paramiko

local = Path(__file__).parent
archive = Path(r'C:\Users\gb\.codex\archives\pvg_candidate_consistency_20261003')
assert not archive.exists()
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
summary = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
assert intake['status']['status'] == 'complete'
assert summary['retention_recommendation']['score_leader'] == 'original_g'
assert set(summary['retention_recommendation']['nonleading_pair_endpoints']) == {'g_control', 'g_consistent'}
checkpoints = intake['checkpoints']
assert shutil.disk_usage(archive.parent).free > sum(item['bytes'] for item in checkpoints.values())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
archive.mkdir()
records = []
for arm in ('g_control', 'g_consistent'):
    item = checkpoints[arm]
    assert sftp.stat(item['path']).st_size == item['bytes']
    destination = archive / arm / 'terminal.pth'
    destination.parent.mkdir()
    sftp.get(item['path'], str(destination))
    assert destination.stat().st_size == item['bytes']
    with destination.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert digest == item['sha256']
    record = dict(item, arm=arm, local=str(destination), local_complete_sha256_verified=True)
    records.append(record)
    print(json.dumps({'archived': arm, 'bytes': item['bytes'], 'sha256': digest}), flush=True)
client.close()
receipt = {'time_cst': datetime.datetime.now().astimezone().isoformat(), 'files': records,
           'archive_complete': True, 'weights_deleted': False,
           'original_g_untouched': True, 'checkpoint_contents_changed': False}
raw = (json.dumps(receipt, indent=2) + '\n').encode('utf-8')
(archive / 'archive_receipt.json').write_bytes(raw)
(local / 'pair_endpoint_archive_receipt.json').write_bytes(raw)
print(json.dumps({'archive_complete': True, 'bytes': sum(item['bytes'] for item in records),
                  'weights_deleted': False}))
