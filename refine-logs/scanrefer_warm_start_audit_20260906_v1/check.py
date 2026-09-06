import datetime,hashlib,json,os,stat,sys
from pathlib import Path
directory=Path(sys.argv[1]);source=Path(sys.argv[2]);os.chdir(str(source));sys.path.insert(0,str(source))
assert os.environ['CUDA_VISIBLE_DEVICES']==''
import torch
expected=json.loads((directory/'expected.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
items={}
for arm,metadata in expected['artifacts'].items():
    path=Path(metadata['path']);digest=sha(path)
    assert digest==metadata['sha256'] and path.stat().st_size==metadata['bytes']
    assert stat.S_IMODE(path.stat().st_mode)==0o444
    payload=torch.load(str(path),map_location='cpu')
    info={'path':str(path),'sha256':digest,'bytes':path.stat().st_size,'mode':'0444','keys':sorted(payload)}
    if arm=='backbone':
        cfg=vars(payload['config'])
        names=['dataset','test_dataset','dataset_dict','butd','butd_cls','butd_gt','use_color','use_height',
            'use_multiview','num_queries','num_decoder_layers','d_model','data_root','use_source_moe',
            'source_choice_selector','source_choice_selector_sources','source_choice_selector_hidden_dim',
            'source_choice_selector_train_only','eval_use_selector_choice_scores','mask_loss_scale',
            'consistency_loss_scale','clip_norm','weight_decay','batch_size','learning_rate','lr_backbone',
            'lr_text_encoder','lr_decoder','num_points','detect_intermediate','checkpoint_path','pp_checkpoint']
        info['config']={name:cfg[name] for name in names if name in cfg}
        info['epoch']=payload.get('epoch')
        info['has_optimizer']='optimizer' in payload
        info['has_scheduler']='scheduler' in payload
        info['model_tensors']=len(payload['model'])
        info['model_tensor_parameters']=sum(value.numel() for value in payload['model'].values())
        info['selector_names']=[name for name in payload['model'] if 'source_choice_selector' in name]
    else:
        for name in ['schema','method','hidden_dim','dropout','margin','geometry_weight','feature_names',
                     'query_feature_names','variant_feature_names','contract']:
            if name in payload and isinstance(payload[name],(str,int,float,bool,list,dict)):
                info[name]=payload[name]
        info['state_tensors']={name:len(value) for name,value in payload.items()
                               if isinstance(value,dict) and value and all(torch.is_tensor(v) for v in value.values())}
    items[arm]=info
    del payload
receipt={'schema':'mcln-scanrefer-warm-start-audit-v1','status':'pass',
    'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'artifacts':items,'gpu_forwards':0,'optimizer_steps':0,'formal_rows':0,'checkpoint_writes':0,
    'python':sys.version,'torch':torch.__version__}
with (directory/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(receipt),flush=True)
