import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

root = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scan_checkpoint_inspection_20260908_v1')
remote = '/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1'
plan = json.loads((root / 'plan.json').read_bytes())
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
with s.open(remote + '/cpu_inventory.exit', 'rb') as stream:
    assert stream.read().strip() == b'0'
with s.open(remote + '/controller.exit', 'rb') as stream:
    assert stream.read().strip() == b'1'
files = {}
for name in ('run.log', 'controller.exit', 'cpu_inventory.log', 'cpu_inventory.exit', 'receipt.json', 'state_inventory.json'):
    with s.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch()
        raw = stream.read()
    with (root / name).open('xb') as stream:
        stream.write(raw)
    files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
receipt = json.loads((root / 'receipt.json').read_bytes())
transfer = json.loads((root / 'transfer_receipt.json').read_bytes())
inventory = json.loads((root / 'state_inventory.json').read_bytes())
assert receipt['status'] == transfer['status'] == 'complete'
assert receipt['checkpoint_sha256'] == transfer['sha256'] == plan['sha256']
assert receipt['state_inventory_sha256'] == files['state_inventory.json']['sha256']
assert receipt['model_tensor_count'] == len(inventory)
assert sum(row['elements'] for row in inventory.values()) == sum(row['elements'] for row in receipt['parameter_prefixes'].values())
assert receipt['gpu_forwards'] == receipt['optimizer_steps'] == receipt['formal_rows'] == 0
assert s.stat(remote + '/' + plan['filename']).st_size == plan['bytes'] == transfer['bytes']
for name in ('local_download_receipt.json', 'transfer_receipt.json', 'cpu_inventory_launch.json'):
    with s.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == (root / name).read_bytes()
s.close()
c.close()
report = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'files': files, 'checkpoint_sha256': plan['sha256'], 'cpu_tensor_elements': sum(row['elements'] for row in inventory.values()),
          'original_transport_exit': 1, 'cpu_inventory_exit': 0, 'model_executed': False}
(root / 'local_verification.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
(root / 'collect_inventory.py').write_bytes(Path(__file__).read_bytes())
(root / 'remote_download_network_probe.json').write_bytes(Path('C:/Users/gb/.codex/tmp/pv_download_network_probe_20260908.json').read_bytes())
print(json.dumps(receipt))
print(json.dumps(report))
