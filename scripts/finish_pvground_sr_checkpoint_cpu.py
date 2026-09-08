"""Collect real Sr terminal evidence, then delete only the verified local transfer duplicate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
archive = repo / 'refine-logs/pvground_sr_checkpoint_inspection_20260908_v1'
root = '/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
for name in ['inventory.exit', 'receipt.json', 'state_inventory.json', 'strict_load.exit', 'strict_load.json',
             'strict_load.log', 'strict_load.stderr', 'strict_controller.pid', 'strict_controller.py', 'strict_load.py', 'transfer.json']:
    s.get(root + '/' + name, str(archive / name))
assert (archive / 'inventory.exit').read_text().strip() == (archive / 'strict_load.exit').read_text().strip() == '0'
receipt = json.loads((archive / 'receipt.json').read_bytes())
strict = json.loads((archive / 'strict_load.json').read_bytes())
assert strict['status'] == 'pass' and strict['loaded_tensors_exact'] and strict['model_forwards'] == 0
assert strict['state_tensors'] == 1235 and not strict['torch_cuda_initialized']
assert strict['script_sha256'] == hashlib.sha256((archive / 'strict_load.py').read_bytes()).hexdigest()
remote = root + '/PV-Ground_SR3D.pth'
code = "from pathlib import Path;import hashlib,json;p=Path(" + repr(remote) + ");h=hashlib.sha256();f=p.open('rb');[h.update(b) for b in iter(lambda:f.read(8*1024*1024),b'')];f.close();print(json.dumps({'sha256':h.hexdigest(),'bytes':p.stat().st_size}))"
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(code), timeout=60)
actual = json.loads(out.read()); assert out.channel.recv_exit_status() == 0, err.read().decode()
assert actual['sha256'] == strict['checkpoint_sha256'] == receipt['checkpoint_sha256'] and actual['bytes'] == 829830168
s.close(); c.close()
cache = Path('C:/Users/gb/.codex/tmp/mcln_pv_sr_transfer_20260908').resolve()
local = (cache / 'PV-Ground_SR3D.pth').resolve()
assert local.parent == cache and cache.parent == Path('C:/Users/gb/.codex/tmp').resolve()
digest = hashlib.sha256()
with local.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
assert digest.hexdigest() == actual['sha256'] and local.stat().st_size == actual['bytes']
local.unlink()
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              file=str(local), bytes_released=actual['bytes'], sha256=actual['sha256'], remote_parent_retained=True,
              remote_parent_hash_reverified=True, inventory_pass=True, strict_load_pass=True, model_forwards=0, training_launched=False)
(archive / 'local_transfer_cleanup.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
