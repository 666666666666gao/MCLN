"""Archive the completed fit receipt while the same process evaluates its endpoint."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_mask_geometry_pair_20260907_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
with s.open(remote + '/fit_complete.json', 'rb') as stream:
    raw = stream.read()
fit = json.loads(raw)
assert fit['steps_per_arm'] == 2482
assert set(fit['checkpoints']) == {'native_gt', 'native_gt_mask_geometry'}
assert not (local / 'fit_complete.json').exists()
(local / 'fit_complete.json').write_bytes(raw)
code = '''import datetime,hashlib,json
from pathlib import Path
d=Path('/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1')
raw=(d/'fit_complete.json').read_bytes();fit=json.loads(raw)
checks={}
for arm,item in fit['checkpoints'].items():
 p=Path(item['path']);assert p.parent==d and p.stat().st_size==item['bytes']
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==item['sha256']
 checks[arm]={'bytes':item['bytes'],'sha256':item['sha256'],'changed_tensors_reported_by_runner':len(fit['changed_core_tensors'][arm]),'checkpoint_hash_verified':True}
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'steps_per_arm':fit['steps_per_arm'],'training_seconds':fit['training_seconds'],'fit_receipt_sha256':hashlib.sha256(raw).hexdigest(),'checkpoints':checks,'checkpoint_tensors_reaudited_in_this_collection':False,'terminal_receipt_exists':(d/'receipt.json').exists(),'checkpoints_downloaded':False}))
'''
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(code), timeout=30)
body = out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
result = json.loads(body)
assert result['fit_receipt_sha256'] == hashlib.sha256(raw).hexdigest()
with (local / 'fit_collection.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
s.close()
c.close()
