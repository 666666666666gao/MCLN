import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
root=Path(__file__).parent
m=json.loads((root/'input_manifest.json').read_text())
source=Path(m['native_preparation'])/'model_source'
os.chdir(str(source));sys.path.insert(0,str(source))
import torch
from main_utils import BaseTrainTester,load_checkpoint,parse_option,prepare_source_moe_gate_checkpoint_config
from train_dist_mod import TrainTester
torch.set_num_threads(1)
assert not torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('export_fixture',root/'scripts/export_native_box_transfer_initialization.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
native_manifest=json.loads((source/'native_source_manifest.json').read_text())
for name,digest in native_manifest['files'].items():assert sha(source/name)==digest,name
item=m['scan_backbone'];assert sha(item['path'])==item['sha256']
backbone=torch.load(item['path'],map_location='cpu')
names=[name[7:] for name in backbone['model'] if name[7:].startswith(exporter.PREFIXES) and name.endswith(('.weight','.bias'))]
assert len(names)==16 and len(backbone['model'])==1144
endpoint={'schema':'mcln-scanrefer-native-box-head-state-v1','arm':'gt_teacher_box','steps':2482,
          'pretrained_artifacts':{'backbone':item},'manifest_sha256':'synthetic-format-test-not-a-training-run',
          'core_trainable_tensors':names,'head_parameters':{name:backbone['model']['module.'+name].clone()+1e-4 for name in names}}
payload=exporter.build_initialization(backbone,endpoint,item['sha256'])
payload['initialization_provenance']['synthetic_fixture_only']=True
fixture_path=root/'synthetic_native_initialization_cpu_fixture.pth'
with fixture_path.open('xb') as stream:torch.save(payload,stream)
expected=payload['model'];fixture_bytes=fixture_path.stat().st_size;fixture_sha=sha(fixture_path)
changed=[name for name in expected if not torch.equal(expected[name],backbone['model'][name])]
assert set(changed)=={'module.'+name for name in names}
del backbone,endpoint;gc.collect()
contract=json.loads((Path(m['native_preparation'])/'nr_contract.json').read_text())
data_root=json.loads((Path(m['native_preparation'])/'data_inputs.json').read_text())['data_root'].rstrip('/')+'/'
results={};started=time.time()
for dset,count in [('nr3d',7899),('sr3d',17726)]:
    argv=list(contract['eval_argv']);argv.remove('--eval')
    for key,value in [('--dataset',dset),('--test_dataset',dset),('--expected_eval_sample_count',str(count)),
                      ('--data_root',data_root),('--checkpoint_path',str(fixture_path)),('--checkpoint_start_epoch','1'),
                      ('--start_epoch','1'),('--max_epoch','1')]:argv[argv.index(key)+1]=value
    argv+=['--model_only_initialization'];sys.argv=['native-box-initialization-cpu-test']+argv
    args=prepare_source_moe_gate_checkpoint_config(parse_option())
    assert args.model_only_initialization and not args.eval and args.butd_cls and args.joint_det
    assert args.use_source_choice_selector and args.eval_use_selector_choice_scores
    assert not args.use_candidate_local_visual
    args.pp_checkpoint=None;args.lr=args.lr_backbone=args.source_choice_selector_lr=1e-6
    model=TrainTester.get_model(args);wrapper=torch.nn.Module();wrapper.module=model
    assert model.decoder[-1].local_visual is None and set(wrapper.state_dict())==set(expected)
    optimizer=BaseTrainTester.get_optimizer(args,wrapper)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2],gamma=.1)
    before_scheduler=scheduler.state_dict()
    load_checkpoint(args,wrapper,optimizer,scheduler)
    assert all(value.device.type=='cpu' and torch.equal(value,expected[name]) for name,value in wrapper.state_dict().items())
    assert not optimizer.state and args.start_epoch==1 and scheduler.state_dict()==before_scheduler
    results[dset]={'native_factory_and_checkpoint_loader_used':True,'all1144_tensors_equal_reconstructed_fixture':True,
                   'changed_box_parameters_loaded':16,'frozen_nonhead_tensors_preserved':1128,
                   'optimizer_state_entries':0,'scheduler_unchanged':True,'start_epoch':1,'expected_formal_rows':count}
    del model,wrapper,optimizer,scheduler;gc.collect()
assert fixture_path.resolve().parent==root.resolve()
assert sha(fixture_path)==fixture_sha
fixture_path.unlink()
result={'schema':'mcln-native-box-initialization-cpu-test-v1','status':'pass',
        'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'elapsed_seconds':time.time()-started,'protocols':results,'source_files_verified':len(native_manifest['files']),
        'backbone_sha256':item['sha256'],'export_script_sha256':sha(root/'scripts/export_native_box_transfer_initialization.py'),
        'synthetic_fixture_only':True,'actual_trained_endpoint_loaded':False,'actual_training_steps':0,
        'gpu_forwards':0,'formal_rows':0,'synthetic_fixture_bytes':fixture_bytes,'synthetic_fixture_sha256':fixture_sha,
        'temporary_fixture_deleted':not fixture_path.exists(),'remaining_new_weight_files':0,
        'no_cross_dataset_quality_claim':True}
with (root/'receipt.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True)
print('NATIVE BOX INITIALIZATION CPU TEST '+json.dumps(result),flush=True)
