import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
prefix='/root/autodl-tmp/mcln_'
training='scanrefer_frozen_readout_pair_20260907_v1'
queue='scanrefer_frozen_readout_posttraining_20260907_v1'
local=repo/'refine-logs'/training
manifest_raw=(local/'input_manifest.json').read_bytes()
manifest=json.loads(manifest_raw)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
published=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0]
assert published==commit
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for name in [training,queue]:
    root=prefix+name
    assert not set(['launch.json','run.log','controller.exit']).intersection(sftp.listdir(root))
    mapping=['input_manifest.json','controller.sh']+(list(manifest['files']) if name==training else ['queue.py'])
    for item in mapping:
        with sftp.open(root+'/'+item,'rb') as stream:
            assert stream.read()==(repo/'refine-logs'/name/item).read_bytes(),item
    _,output,error=client.exec_command('bash -n '+shlex.quote(root+'/controller.sh'),timeout=30)
    assert output.channel.recv_exit_status()==0,error.read().decode()
_,output,error=client.exec_command('nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits',timeout=30)
gpu=output.read().decode().strip()
assert output.channel.recv_exit_status()==0,error.read().decode()
assert int(gpu.split(',')[2].strip())<500,gpu
launches={}
for name,session in [(training,'mcln_frozen_readout_pair_v1'),(queue,'mcln_frozen_readout_post_v1')]:
    root=prefix+name
    command='screen -dmS '+session+' bash -c '+shlex.quote('exec bash '+root+'/controller.sh > '+root+'/run.log 2>&1')
    _,output,error=client.exec_command(command,timeout=30)
    assert output.channel.recv_exit_status()==0,error.read().decode()
    _,output,error=client.exec_command('screen -ls',timeout=30)
    sessions=output.read().decode()
    assert output.channel.recv_exit_status()==0,error.read().decode()
    matches=[line.strip() for line in sessions.splitlines() if '.'+session in line]
    assert len(matches)==1,sessions
    raw_manifest=(repo/'refine-logs'/name/'input_manifest.json').read_bytes()
    launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'screen_session':matches,'manifest_sha256':sha(raw_manifest),'published_source_commit':commit,
        'gpu_before_pair_launch':gpu,'fixed_steps_per_arm':2482,'candidate_predeclared':'frozen_gt',
        'nr3d_sr3d_training_started':False,'new_quality_result_not_yet_available':True}
    raw=(json.dumps(launch,indent=2,sort_keys=True)+'\n').encode()
    for item,value in [('launch.json',raw),('launch_from_local.py',Path(__file__).read_bytes())]:
        (repo/'refine-logs'/name/item).write_bytes(value)
        with sftp.open(root+'/'+item,'wx') as stream:stream.write(value)
        with sftp.open(root+'/'+item,'rb') as stream:assert stream.read()==value
    launches[name]=launch
sftp.close()
client.close()
print(json.dumps(launches),flush=True)
