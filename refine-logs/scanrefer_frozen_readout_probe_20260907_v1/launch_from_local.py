import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_frozen_readout_probe_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_frozen_readout_probe_20260907_v1'
manifest_raw=(local/'input_manifest.json').read_bytes()
manifest=json.loads(manifest_raw)
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
assert not set(['launch.json','run.log','controller.exit']).intersection(sftp.listdir(remote))
for name in ['input_manifest.json','controller.sh']+list(manifest['files']):
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==(local/name).read_bytes(),name
_,output,error=client.exec_command('nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits',timeout=30)
gpu=output.read().decode().strip()
assert output.channel.recv_exit_status()==0,error.read().decode()
assert int(gpu.split(',')[2].strip())<500,gpu
command='screen -dmS mcln_frozen_readout_probe_v1 bash -c '+shlex.quote('exec bash '+remote+'/controller.sh > '+remote+'/run.log 2>&1')
_,output,error=client.exec_command(command,timeout=30)
assert output.channel.recv_exit_status()==0,error.read().decode()
_,output,error=client.exec_command('screen -ls',timeout=30)
sessions=output.read().decode()
assert output.channel.recv_exit_status()==0,error.read().decode()
matches=[line.strip() for line in sessions.splitlines() if '.mcln_frozen_readout_probe_v1' in line]
assert len(matches)==1,sessions
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen_session':matches,'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
    'gpu_before_launch':gpu,'real_train_rows':16,'backbone_forwards_planned':6,
    'disposable_optimizer_steps_per_arm':2,'checkpoint_writes_planned':0,
    'formal_training_started':False,'result_not_yet_available':True}
raw=(json.dumps(launch,indent=2,sort_keys=True)+'\n').encode()
for name,value in [('launch.json',raw),('launch_from_local.py',Path(__file__).read_bytes())]:
    (local/name).write_bytes(value)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(value)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==value
sftp.close()
client.close()
print(json.dumps(launch),flush=True)
