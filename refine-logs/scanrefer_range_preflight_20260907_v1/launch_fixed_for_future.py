import hashlib
import json
import os
from pathlib import Path

import paramiko

local=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_range_preflight_20260907_v1')
remote='/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1'
tests=json.loads((local/'unit_test_receipt.json').read_bytes())
assert tests['exit_code']==0
assert tests['manifest_sha256']==hashlib.sha256((local/'input_manifest.json').read_bytes()).hexdigest()
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code="""import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1')
assert not (root/'launch.json').exists() and not (root/'run.log').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']).decode().strip()
assert not gpu,gpu
manifest=json.loads((root/'input_manifest.json').read_text())
for name,expected in manifest['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
session='mcln_scanrefer_range_preflight_v1'
subprocess.run(['screen','-dmS',session,'bash','-c','exec bash '+str(root/'controller.sh')+' > '+str(root/'run.log')+' 2>&1'],check=True)
listing=subprocess.check_output(['screen','-ls']).decode()
matches=[line.split()[0] for line in listing.splitlines() if line.split() and line.split()[0].endswith('.'+session)]
assert len(matches)==1,matches
receipt={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'screen_session':matches,'manifest_sha256':hashlib.sha256((root/'input_manifest.json').read_bytes()).hexdigest(),'formal_rows':0,'training_started':False,'preflight_started':True,'checkpoint_writes':0,'controller_sha256':hashlib.sha256((root/'controller.sh').read_bytes()).hexdigest()}
with (root/'launch.json').open('x') as stream: json.dump(receipt,stream,indent=2,sort_keys=True)
print(json.dumps(receipt))
"""
_,out,err=client.exec_command('/root/miniconda3/envs/bdetr/bin/python - <<\'PY\'\n'+code+'\nPY',timeout=30)
raw=out.read()
assert out.channel.recv_exit_status()==0,err.read().decode()
receipt=json.loads(raw)
sftp=client.open_sftp()
with sftp.open(remote+'/launch.json','rb') as stream:
    (local/'launch.json').write_bytes(stream.read())
sftp.close()
client.close()
print(json.dumps(receipt))
