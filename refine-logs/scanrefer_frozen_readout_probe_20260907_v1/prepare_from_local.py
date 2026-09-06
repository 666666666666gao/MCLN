import ast
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
old=json.loads((repo/'refine-logs/scanrefer_range_preflight_20260907_v1/input_manifest.json').read_bytes())
formal=repo/'refine-logs/scanrefer_range_official_20260907_v1/result'
receipt=json.loads((formal/'receipt.json').read_bytes())
assert receipt['status']=='complete' and not receipt['promotion']['advance_to_nr3d_sr3d_rec']
names=['scripts/probe_scanrefer_frozen_readout_compatibility.py','scripts/scanrefer_data_contract.py',
       'scripts/scanrefer_joint_readout.py','scripts/audit_scanrefer_joint_readout_pair.py']
files={name:(repo/name).read_bytes() for name in names}
for name,raw in files.items():ast.parse(raw,filename=name)
files['scripts/__init__.py']=b''
digest=lambda raw:hashlib.sha256(raw).hexdigest()
manifest={'schema':'mcln-frozen-readout-compatibility-probe-v1','model_source':old['model_source'],
    'source_manifest_sha256':old['source_manifest_sha256'],'artifacts':old['artifacts'],
    'split_protocol':old['split_protocol'],'split_protocol_sha256':old['split_protocol_sha256'],'split_salt':old['split_salt'],
    'data_root':old['data_root'],'train_superpoint_files':old['superpoint_files']['train'],
    'files':{name:digest(raw) for name,raw in files.items()},'core_learning_rate':1e-6,'auxiliary_weight':1./3.,
    'formal_rows':0,'checkpoint_writes':0,'real_train_rows':16,'backbone_forwards':6,
    'disposable_optimizer_steps_per_arm':2,'readout_frozen':True,'new_network_modules':0,
    'upstream_formal_receipt_sha256':digest((formal/'receipt.json').read_bytes()),
    'upstream_formal_audit_sha256':digest((formal/'independent_audit.json').read_bytes()),
    'environment_reuse':'Existing bdetr Python3.7.11 Torch1.10.2cu111;no installs or rebuild;full actual Scan formal just passed execution audit.',
    'purpose':'Frozen old readout GT gradients versus native GT on protected E71;no failed range module or weights;engineering check only.',
    'training_budget_not_yet_locked':True}
files['input_manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd '''+remote+'''
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/probe_scanrefer_frozen_readout_compatibility.py --manifest '''+remote+'''/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
files['controller.sh']=controller.encode()
files['prepare_from_local.py']=Path(__file__).read_bytes()
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for path,expected in [('/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1/result/receipt.json',manifest['upstream_formal_receipt_sha256']),
                      ('/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1/result/independent_audit.json',manifest['upstream_formal_audit_sha256'])]:
    with sftp.open(path,'rb') as stream:assert digest(stream.read())==expected
probe="""import json,os,socket,subprocess,shutil
result={'uid':os.getuid(),'hostname':socket.gethostname(),'cwd':os.getcwd(),'disk_free':shutil.disk_usage('/root/autodl-tmp').free}
p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits'],stdout=subprocess.PIPE)
assert p.returncode==0
result['gpu']=p.stdout.decode().strip()
assert int(result['gpu'].split(',')[2].strip())<500
print(json.dumps(result))
"""
_,output,error=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
machine=json.loads(output.read())
assert output.channel.recv_exit_status()==0,error.read().decode()
assert machine['uid']==0 and machine['hostname']=='autodl-container-c7cb4299a4-24929f53'
local.mkdir()
sftp.mkdir(remote)
sftp.mkdir(remote+'/scripts')
for name,raw in files.items():
    target=local/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
command='cd '+shlex.quote(remote)+' && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python scripts/probe_scanrefer_frozen_readout_compatibility.py --help'
_,output,error=client.exec_command(command,timeout=30)
help_text=output.read().decode()
assert output.channel.recv_exit_status()==0,error.read().decode()
assert '--manifest' in help_text
preparation={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'manifest_sha256':digest(files['input_manifest.json']),'machine':machine,'original_python_cli_pass':True,
    'source_ast_pass':True,'uploaded_bytes_verified':True,'new_gpu_forwards':0,'probe_started':False,'formal_training_started':False}
raw=(json.dumps(preparation,indent=2,sort_keys=True)+'\n').encode()
(local/'preparation.json').write_bytes(raw)
with sftp.open(remote+'/preparation.json','wx') as stream:stream.write(raw)
sftp.close()
client.close()
print(json.dumps(preparation),flush=True)
