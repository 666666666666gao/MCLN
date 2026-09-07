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
code='''import datetime,gc,hashlib,json,os,sys,time
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path(__file__).parent
m=json.loads((root/'input_manifest.json').read_text())
schema_audit=json.loads((root/'receipt.json').read_text())
assert schema_audit['identical_parameter_schema']
source=Path(m['native_preparation'])/'model_source'
os.chdir(str(source))
sys.path.insert(0,str(source))
import torch
from main_utils import BaseTrainTester,load_checkpoint,parse_option,prepare_source_moe_gate_checkpoint_config
from train_dist_mod import TrainTester
torch.set_num_threads(1)
assert not torch.cuda.is_available()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
native_manifest=json.loads((source/'native_source_manifest.json').read_text())
for name,digest in native_manifest['files'].items():assert sha(source/name)==digest,name
item=m['scan_backbone']
assert sha(item['path'])==item['sha256']
checkpoint=torch.load(item['path'],map_location='cpu')
expected=checkpoint['model']
assert len(expected)==1144 and sum(name.startswith('module.backbone_net.') for name in expected)==96
contract=json.loads((Path(m['native_preparation'])/'nr_contract.json').read_text())
data_root=json.loads((Path(m['native_preparation'])/'data_inputs.json').read_text())['data_root'].rstrip('/')+'/'
started=time.time()
results={}
for dset,count in [('nr3d',7899),('sr3d',17726)]:
 argv=list(contract['eval_argv'])
 argv.remove('--eval')
 for key,value in [('--dataset',dset),('--test_dataset',dset),('--expected_eval_sample_count',str(count)),
                   ('--data_root',data_root),('--checkpoint_path',item['path']),('--checkpoint_start_epoch','1'),
                   ('--start_epoch','1'),('--max_epoch','1')]:argv[argv.index(key)+1]=value
 argv+=['--model_only_initialization']
 sys.argv=['native-initialization-cpu-audit']+argv
 args=prepare_source_moe_gate_checkpoint_config(parse_option())
 assert args.model_only_initialization and not args.eval and args.max_epoch==1
 assert args.butd_cls and args.joint_det and not args.butd and not args.butd_gt
 assert args.use_source_choice_selector and args.eval_use_selector_choice_scores
 assert not args.use_candidate_local_visual
 # The full E71 state restores all96 backbone tensors; no second detector initialization is needed.
 args.pp_checkpoint=None
 args.lr=args.lr_backbone=args.source_choice_selector_lr=1e-6
 model=TrainTester.get_model(args)
 assert model.decoder[-1].local_visual is None
 wrapper=torch.nn.Module()
 wrapper.module=model
 assert set(wrapper.state_dict())==set(expected)
 optimizer=BaseTrainTester.get_optimizer(args,wrapper)
 scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2],gamma=.1)
 load_checkpoint(args,wrapper,optimizer,scheduler)
 actual=wrapper.state_dict()
 assert all(value.device.type=='cpu' and torch.equal(value,expected[name]) for name,value in actual.items())
 assert not optimizer.state and args.start_epoch==1
 assert all(not p.requires_grad for p in model.text_encoder.parameters())
 results[dset]={'native_model_factory_and_loader_used':True,'restored_tensors':len(actual),
  'restored_backbone_tensors':96,'all_tensor_values_equal_protected_e71':True,
  'optimizer_state_entries':len(optimizer.state),'start_epoch':args.start_epoch,
  'text_encoder_frozen':True,'source_choice_selector_enabled':True,'butd_cls':True,
  'local_or_range_module_installed':False,'expected_formal_rows':count,
  'optimizer_groups':[{'name':g['name'],'lr':g['lr'],'parameter_tensors':len(g['params'])} for g in optimizer.param_groups]}
 del actual,model,wrapper,optimizer,scheduler
 gc.collect()
result={'schema':'mcln-native-model-initialization-cpu-receipt-v1','status':'pass',
 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'elapsed_seconds':time.time()-started,'source_manifest_sha256':sha(source/'native_source_manifest.json'),
 'source_files_verified':len(native_manifest['files']),'checkpoint_sha256':item['sha256'],
 'protocols':results,'model_factories_executed':2,'native_checkpoint_loads':2,
 'gpu_forwards':0,'optimizer_steps':0,'checkpoint_writes':0,'formal_rows':0,
 'future_trained_candidate_not_loaded':True,
 'native_checkpoint_requires_module_prefix':True,
 'limitations':'Exact CPU model-only loading of protected E71 under both native dataset configurations. Future frozen-readout terminal is not yet available; its unprefixed model dict must be exported with the native module prefix and re-audited after Scanpromotion. No cross-dataset effectiveness claim.',
 'audit_script_sha256':sha(__file__),'upstream_input_audit_sha256':sha(root/'receipt.json')}
with (root/'native_initialization_receipt.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
print('NATIVE INITIALIZATION CPU COMPLETE '+json.dumps(result))
'''
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for name,raw in [('check_native_initialization.py',code.encode()),('run_initialization_from_local.py',Path(__file__).read_bytes())]:
 (local/name).write_bytes(raw)
 with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
 with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/bdetr/bin/python '+shlex.quote(remote+'/check_native_initialization.py')
_,output,error=client.exec_command(command,timeout=60)
raw=output.read()
err=error.read()
(local/'native_initialization_stdout.txt').write_bytes(raw)
(local/'native_initialization_stderr.txt').write_bytes(err)
status=output.channel.recv_exit_status()
assert status==0,err.decode()+raw.decode()[-5000:]
with sftp.open(remote+'/native_initialization_receipt.json','rb') as stream:receipt=stream.read()
(local/'native_initialization_receipt.json').write_bytes(receipt)
sftp.close()
client.close()
print(receipt.decode(),flush=True)
