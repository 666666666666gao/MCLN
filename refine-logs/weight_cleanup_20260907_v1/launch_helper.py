import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/weight_cleanup_20260907_v1'
remote='/root/autodl-tmp/mcln_weight_cleanup_20260907_v1'
training=repo/'refine-logs/scanrefer_range_pair_20260907_v1'
formal=repo/'refine-logs/scanrefer_range_official_20260907_v1/result'
previous=repo/'refine-logs/weight_cleanup_20260906_v2'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
receipt=json.loads((training/'receipt.json').read_bytes())
plan={'delete':receipt['checkpoints'],'old_directory':'/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1',
    'formal_directory':'/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1',
    'new_directory':'/root/autodl-tmp/mcln_scanrefer_frozen_readout_probe_20260907_v1',
    'old_receipt_sha256':digest((training/'receipt.json').read_bytes()),
    'old_audit_sha256':digest((training/'independent_audit.json').read_bytes()),
    'formal_audit_sha256':digest((formal/'independent_audit.json').read_bytes()),
    'protected_files':json.loads((previous/'cleanup_plan.json').read_bytes())['protected_files'],
    'reason':'Both range endpoints fully audited and failed fixed formal REC;all dependent queues exited;next probe initializes protected E71 only.',
    'preserve':'all logs, row metrics, receipts, source manifests and six protected weight files',
    'user_authorized_unused_weight_cleanup':True}
assert not json.loads((formal/'receipt.json').read_bytes())['promotion']['advance_to_nr3d_sr3d_rec']
source=(previous/'cleanup.py').read_text().replace("arm+'_local_visual_state.pt'","arm+'_range_visual_state.pt'").replace('mcln-sealed-failed-local-weight-cleanup-v1','mcln-sealed-failed-range-weight-cleanup-v1')
local.mkdir()
files={'cleanup.py':source.encode(),'cleanup_plan.json':(json.dumps(plan,indent=2,sort_keys=True)+'\n').encode(),'launch_helper.py':Path(__file__).read_bytes()}
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
native='/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1'
with sftp.open(native+'/controller.exit','rb') as stream:assert stream.read().strip()==b'0'
with sftp.open(native+'/decision.json','rb') as stream:assert json.loads(stream.read())['status']=='scanrefer_not_promoted'
sftp.mkdir(remote)
for name,raw in files.items():
    (local/name).write_bytes(raw)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
_,output,error=client.exec_command('/root/miniconda3/envs/bdetr/bin/python '+shlex.quote(remote+'/cleanup.py'),timeout=60)
stdout,stderr=output.read(),error.read()
code=output.channel.recv_exit_status()
(local/'stdout.txt').write_bytes(stdout)
(local/'stderr.txt').write_bytes(stderr)
assert code==0,stderr.decode()
for name in ['verified_before_delete.json','receipt.json']:
    with sftp.open(remote+'/'+name,'rb') as stream:(local/name).write_bytes(stream.read())
result=json.loads((local/'receipt.json').read_bytes())
assert result['status']=='complete' and result['plan_sha256']==digest(files['cleanup_plan.json'])
assert len(result['deleted'])==2
sftp.close()
client.close()
print(json.dumps(result),flush=True)
