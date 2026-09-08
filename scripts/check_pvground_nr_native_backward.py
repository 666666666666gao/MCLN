"""One gated Nr3D native forward/backward, with no optimizer update or saved model."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--formal-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # This entry is prepared ahead of time; no Nr GPU work before the actual Scan gate.
    formal=args.formal_root
    assert (formal/'controller.exit').read_text().strip()=='0'
    audit=json.loads((formal/'audit.json').read_bytes())
    assert audit['integrity_pass'] and audit['formal_rows']==9508
    assert audit['receipt_sha256']==sha(formal/'receipt.json')
    assert audit['advance_to_nr3d_sr3d_rec'] and all(audit['checks'].values())
    runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
    source=Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground')
    parent=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/PV-Ground_NR3D.pth')
    strict_root=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v2')
    batch_root=Path('/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1')
    strict=json.loads((strict_root/'strict_load.json').read_bytes())
    batch_receipt=json.loads((batch_root/'receipt.json').read_bytes())
    assert strict['status']=='pass' and strict['state_tensors']==1235
    assert batch_receipt['status']=='pass'
    assert sha(parent)==strict['checkpoint_sha256']
    assert sha(batch_root/'native_batch.pt')==batch_receipt['input_sha256']
    env=json.loads((runtime/'env_spec.json').read_bytes())
    env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert env_sha==strict['env_spec_sha256']=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
    assert sha(source.parent/'source_port.json')==strict['model_source_port_sha256']
    upstream=json.loads((runtime/'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']
    port=json.loads((source.parent/'source_port.json').read_bytes())
    for name,entry in upstream['files'].items():
        expected=port['after_sha256'] if 'PV-Ground/'+name==port['file'] else entry['sha256']
        assert sha(source/name)==expected,name
    args.output.mkdir()
    os.chdir(str(source));sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from pcdet.config import cfg,cfg_from_yaml_file
    for name in ['models.pv_ground','models.losses','main_utils']:
        assert source in Path(sys.modules[name].__file__).resolve().parents
    random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(str(parent),map_location='cpu');config=saved['config']
    assert not config.butd and config.butd_cls and not config.butd_gt
    assert config.num_target==256 and config.num_decoder_layers==6
    assert config.use_soft_token_loss and config.use_contrastive_align
    assert all(name.startswith('module.') for name in saved['model'])
    state={name[7:]:value for name,value in saved['model'].items()};del saved
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model=PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,
        butd=config.butd or config.butd_cls or config.butd_gt,pointnet_ckpt=None,
        data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',self_attend=config.self_attend)
    assert set(model.state_dict())==set(state) and len(state)==1235
    model.load_state_dict(state,strict=True);model.cuda();model.train()
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    frozen={name:p for name,p in model.named_parameters() if not p.requires_grad}
    assert len(trainable)==783 and sum(p.numel() for p in trainable.values())==27959611
    assert len(frozen)==199 and all(name.startswith('text_encoder.') for name in frozen)
    payload=torch.load(str(batch_root/'native_batch.pt'),map_location='cpu')
    inputs={key:value.cuda() if torch.is_tensor(value) else value for key,value in payload['inputs'].items()}
    batch={key:value.cuda() if torch.is_tensor(value) else value for key,value in payload['native_batch'].items()}
    assert inputs['batch_size']==8 and batch['gt_masks'].dtype==torch.bool
    criterion,set_criterion=BaseTrainTester.get_criterion(config)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.time()
    output=model(inputs)
    assert not set(output).intersection(batch)
    output.update(batch)
    loss,output=criterion(output,6,set_criterion,query_points_obj_topk=config.query_points_obj_topk)
    assert torch.isfinite(loss)
    components={key:float(value) for key,value in output.items() if 'loss' in key and
                (isinstance(value,(float,int)) or (torch.is_tensor(value) and value.numel()==1))}
    assert all(np.isfinite(value) for value in components.values())
    loss.backward()
    groups=defaultdict(lambda:dict(tensors=0,nonzero_tensors=0,norm_squared_sum=0.0))
    missing=[]
    for name,p in trainable.items():
        if p.grad is None:
            missing.append(name)
            continue
        assert torch.isfinite(p.grad).all(),name
        norm=float(p.grad.norm());group=groups[name.split('.')[0]]
        group['tensors']+=1;group['nonzero_tensors']+=int(norm>0);group['norm_squared_sum']+=norm**2
    for name in ['backbone_net','cross_encoder','gumbel','decoder','prediction_heads','x_mask','x_query']:
        assert groups[name]['norm_squared_sum']>0,name
    assert all(p.grad is None for p in frozen.values())
    assert all(torch.equal(p.detach().cpu(),state[name]) for name,p in model.named_parameters())
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated();elapsed=time.time()-started
    parameter_names=set(dict(model.named_parameters()))
    changed_buffers=[name for name,value in model.state_dict().items()
                     if name not in parameter_names and not torch.equal(value.cpu(),state[name])]
    # Training-mode batch statistics can change in a no-update probe; discard them explicitly.
    model.load_state_dict(state,strict=True)
    assert all(torch.equal(value.cpu(),state[name]) for name,value in model.state_dict().items())
    assert sha(parent)==strict['checkpoint_sha256']
    record=dict(status='pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        model_forwards=1,backward_calls=1,optimizer_steps=0,formal_rows=0,new_checkpoints=0,
        formal_audit_sha256=sha(formal/'audit.json'),checkpoint_sha256=strict['checkpoint_sha256'],
        input_sha256=batch_receipt['input_sha256'],env_spec_sha256=env_sha,script_sha256=sha(__file__),
        losses=components,gradient_groups=dict(groups),missing_gradients=missing,
        all_parameters_unchanged=True,changed_buffers_before_restore=changed_buffers,
        full_state_restored=True,peak_allocated_bytes=peak,elapsed_seconds=elapsed,
        scope='one fixed augmented eight-row native backward; not training, formal accuracy or full-dataset capacity')
    (args.output/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('PVG_NR_NATIVE_BACKWARD_PASS '+json.dumps(record),flush=True)


if __name__=='__main__':
    main()
