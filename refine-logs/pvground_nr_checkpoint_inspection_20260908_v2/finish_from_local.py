"""Collect completed Nr3D CPU checks and release the verified local transfer copy."""
import hashlib
import json
import os
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
archive=repo/'refine-logs/pvground_nr_checkpoint_inspection_20260908_v2'
remote='/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v2'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for name in ['inventory.exit','inventory.log','receipt.json','state_inventory.json']:
    sftp.get(remote+'/'+name,str(archive/name))
assert (archive/'inventory.exit').read_text().strip()=='0'
inventory=json.loads((archive/'receipt.json').read_bytes())
assert inventory['status']=='complete' and not inventory['torch_cuda_initialized']
for name in ['strict_load.json','strict_load.exit','strict_load.log','strict_load.stderr','strict_load_execution.json','controller.exit','run.log','inventory.stderr','inventory_execution.json']:
    sftp.get(remote+'/'+name,str(archive/name))
assert (archive/'strict_load.exit').read_text().strip()=='0'
assert (archive/'controller.exit').read_text().strip()=='0'
r=json.loads((archive/'strict_load.json').read_bytes())
assert r['status']=='pass' and r['loaded_tensors_exact'] and r['model_forwards']==0
sftp.close();client.close()
# The only removed file is the verified local duplicate; the remote parent is retained.
cache=Path('C:/Users/gb/.codex/tmp/mcln_pv_nr_transfer_20260908').resolve()
local=(cache/'PV-Ground_NR3D.pth').resolve()
assert local.parent==cache and cache.parent==Path('C:/Users/gb/.codex/tmp').resolve()
digest=hashlib.sha256()
with local.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()==r['checkpoint_sha256']
released=local.stat().st_size;local.unlink();cache.rmdir()
(archive/'local_transfer_cleanup.json').write_text(json.dumps(dict(file=str(local),bytes_released=released,
    sha256=digest.hexdigest(),remote_parent_retained=True),indent=2)+'\n',encoding='utf-8')
(archive/'finish_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(inventory=inventory,strict_load=r,local_bytes_released=released)),flush=True)
