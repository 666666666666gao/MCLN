import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/native_transfer_input_audit_20260907_v1'
remote='/root/autodl-tmp/mcln_native_transfer_input_audit_20260907_v1'
pair_local=repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
pair_manifest=json.loads((pair_local/'input_manifest.json').read_bytes())
clean=json.loads((repo/'refine-logs/weight_cleanup_20260907_v1/cleanup_plan.json').read_bytes())
sha=lambda raw:hashlib.sha256(raw).hexdigest()
# Select the existing protected averaged Nr model by its already recorded exact digest.
nr=[{'path':path,'sha256':digest,'bytes':599090977} for path,digest in clean['protected_files'].items()
    if digest=='76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1']
assert len(nr)==1
manifest={'schema':'mcln-native-transfer-input-audit-v1','scan_backbone':pair_manifest['artifacts']['backbone'],
    'protected_nr':nr[0],'scan_source':pair_manifest['model_source'],
    'scan_source_manifest_sha256':pair_manifest['source_manifest_sha256'],
    'native_preparation':'/root/autodl-tmp/mcln_native_range_preparation_20260907_v1',
    'training_directory':'/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1',
    'training_manifest_sha256':sha((pair_local/'input_manifest.json').read_bytes()),
    'future_trained_candidate_not_available':True,'gpu_forwards':0,'checkpoint_writes':0,'optimizer_steps':0}
code='''import datetime,hashlib,json,os,time
from pathlib import Path
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
torch.set_num_threads(1)
root=Path(__file__).parent
m=json.loads((root/'input_manifest.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
started=time.time()
assert sha(Path(m['training_directory'])/'input_manifest.json')==m['training_manifest_sha256']
states={}
descriptions={}
for label in ['scan_backbone','protected_nr']:
 item=m[label]
 path=Path(item['path'])
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 payload=torch.load(str(path),map_location='cpu')
 raw=payload['model']
 assert raw and all(name.startswith('module.') for name in raw)
 states[label]={name[7:]:value for name,value in raw.items()}
 descriptions[label]={'path':str(path),'sha256':item['sha256'],'tensor_count':len(raw),
  'backbone_tensor_count':sum(name.startswith('module.backbone_net.') for name in raw),
  'selector_tensor_names':[name for name in states[label] if name.startswith('source_choice_selector.')],
  'has_range_or_local_state':any('local_visual.' in name for name in raw),
  'raw_native_prefix':'module.'}
 del payload
scan,nr=states['scan_backbone'],states['protected_nr']
common=set(scan)&set(nr)
mismatch=[{'name':name,'scan_shape':list(scan[name].shape),'nr_shape':list(nr[name].shape),
 'scan_dtype':str(scan[name].dtype),'nr_dtype':str(nr[name].dtype)} for name in sorted(common)
 if scan[name].shape!=nr[name].shape or scan[name].dtype!=nr[name].dtype]
source=Path(m['scan_source'])
assert sha(source/'local_visual_source_manifest.json')==m['scan_source_manifest_sha256']
sm=json.loads((source/'local_visual_source_manifest.json').read_text())
prep=Path(m['native_preparation'])
nm=json.loads((prep/'model_source/native_source_manifest.json').read_text())
selected=['main_utils.py','train_dist_mod.py','src/joint_det_dataset.py','src/grounding_evaluator.py']
sources={}
for name in selected:
 assert sha(source/name)==sm['files'][name]
 assert sha(prep/'model_source'/name)==nm['files'][name]
 sources[name]={'scan_sha256':sm['files'][name],'native_sha256':nm['files'][name],
   'equal':sm['files'][name]==nm['files'][name]}
annotation=json.loads((prep/'annotation_receipt.json').read_text())
for name,item in annotation['annotations_and_split_files'].items():assert sha(name)==item['sha256'],name
protocols={dset:{part:{key:value for key,value in data.items() if not isinstance(value,(list,dict))}
 for part,data in annotation['protocols'][dset].items() if isinstance(data,dict)} for dset in ['nr3d','sr3d']}
result={'schema':'mcln-native-transfer-input-audit-receipt-v1','status':'complete',
 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'elapsed_seconds':time.time()-started,'artifacts':descriptions,
 'only_scan_keys':sorted(set(scan)-set(nr)),'only_nr_keys':sorted(set(nr)-set(scan)),
 'common_shape_dtype_mismatches':mismatch,'identical_parameter_schema':set(scan)==set(nr) and not mismatch,
 'selected_runtime_source_files':sources,'native_annotation_protocol_counts':protocols,
 'annotations_and_splits_rehashed':len(annotation['annotations_and_split_files']),
 'gpu_forwards':0,'checkpoint_writes':0,'optimizer_steps':0,'formal_rows_evaluated':0,
 'future_trained_candidate_not_loaded':True,
 'limitations':'Actual protected model schemas and native data-file identities only;not an end-to-end transfer load,not GPUpreflight,not transfer performance;no promotion or native training authorization change.',
 'manifest_sha256':sha(root/'input_manifest.json'),'audit_script_sha256':sha(__file__)}
with (root/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
print(json.dumps(result))
'''
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
local.mkdir()
sftp.mkdir(remote)
files={'input_manifest.json':(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode(),
       'audit.py':code.encode(),'run_from_local.py':Path(__file__).read_bytes()}
for name,raw in files.items():
 (local/name).write_bytes(raw)
 with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
 with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python '+shlex.quote(remote+'/audit.py')
_,output,error=client.exec_command(command,timeout=60)
raw=output.read()
assert output.channel.recv_exit_status()==0,error.read().decode()
result=json.loads(raw)
with sftp.open(remote+'/receipt.json','rb') as stream:receipt_raw=stream.read()
assert json.loads(receipt_raw)==result
(local/'receipt.json').write_bytes(receipt_raw)
sftp.close()
client.close()
print(json.dumps(result),flush=True)
