"""One gated Sr3D F REC-competition forward/backward, with no optimizer update or saved model."""
import argparse
from collections import defaultdict
import datetime
import fcntl
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
    entry=Path(__file__).parent
    preparation=json.loads((entry/'spec.json').read_bytes())
    for name,digest in preparation['files'].items():assert sha(entry/name)==digest,name
    # This entry is prepared ahead of time; no Sr GPU work before the actual Scan gate.
    formal=args.formal_root.resolve()
    assert formal==Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_rec_competition_v1')
    assert (formal/'controller.exit').read_text().strip()=='0'
    audit=json.loads((formal/'audit.json').read_bytes())
    assert audit['integrity_pass'] and audit['formal_rows']==9508
    assert audit['receipt_sha256']==sha(formal/'receipt.json')
    assert audit['advance_to_nr3d_sr3d_rec'] and all(audit['checks'].values())
    formal_spec=json.loads((formal/'spec.json').read_bytes())
    for name,digest in formal_spec['files'].items():assert sha(formal/name)==digest,name
    training=Path(formal_spec['training_root'])
    assert sha(training/'spec.json')==formal_spec['training_spec_sha256']
    train_spec=json.loads((training/'spec.json').read_bytes())
    assert train_spec['task_read'] and train_spec['observation_state']
    assert train_spec['rec_competition'] and train_spec['competition_weight']==1.0
    assert audit['scanrefer_mask_gate'] is False
    assert sha(training/'pvground_rec_competition.py')==train_spec['competition_module_sha256']
    endpoint=json.loads((Path(formal_spec['training_audit_root'])/'audit.json').read_bytes())
    assert endpoint['integrity_pass'] and endpoint['rec_competition_verified']
    runtime=Path(train_spec['runtime'])
    source=Path(train_spec['model_source'])
    for name,field in [('pvground_source_query.py','source_query_module_sha256'),
                       ('pvground_observation_query.py','observation_module_sha256'),
                       ('pvground_task_observation_query.py','task_module_sha256')]:
        assert sha(formal/name)==train_spec[field],name
    gpu_lock=open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','a')
    fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_EX)
    parent=Path('/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1/PV-Ground_SR3D.pth')
    strict_root=Path('/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1')
    batch_root=Path('/root/autodl-tmp/mcln_pvground_sr_native_batch_20260917_v1')
    strict=json.loads((strict_root/'strict_load.json').read_bytes())
    batch_receipt=json.loads((batch_root/'receipt.json').read_bytes())
    assert strict['status']=='pass' and strict['state_tensors']==1235
    assert batch_receipt['status']=='pass'
    assert sha(parent)==strict['checkpoint_sha256']
    assert sha(batch_root/'native_batch.pt')==batch_receipt['input_sha256']
    env=json.loads((runtime/'env_spec.json').read_bytes())
    env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert env_sha==strict['env_spec_sha256']=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
    assert sha(train_spec['source_port'])==train_spec['source_port_sha256']
    port=json.loads(Path(train_spec['source_port']).read_bytes())
    for name,digest in port['files'].items():assert sha(source/name)==digest,name
    args.output.mkdir()
    os.chdir(str(source));sys.path[:0]=[str(formal),str(source),str(entry/'models'),str(training)]
    import numpy as np
    import torch
    from models.pv_ground import PVGround
    from pvground_task_observation_query import install_task_observation_query_read
    from main_utils import BaseTrainTester
    from pvground_referit_rec_competition import referit_competition_loss
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
    model.load_state_dict(state,strict=True)
    install_task_observation_query_read(model)
    added=set(model.state_dict())-set(state)
    assert len(added)==37 and all(name.startswith('decoder.5.source_query_read.') for name in added)
    assert all(torch.equal(model.state_dict()[name],value) for name,value in state.items())
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    assert len(state)==1272 and model.decoder[-1].task_read
    assert sum(p.numel() for p in model.decoder[-1].source_query_read.parameters())==923616
    model.cuda();model.train()
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    frozen={name:p for name,p in model.named_parameters() if not p.requires_grad}
    assert len(trainable)==820 and sum(p.numel() for p in trainable.values())==28883227
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
    matching=[]
    def capture_match(module,inputs,result):
        matching.append([(q.clone(),t.clone()) for q,t in result])
    hook=set_criterion.matcher.register_forward_hook(capture_match)
    native,output=criterion(output,6,set_criterion,query_points_obj_topk=config.query_points_obj_topk)
    hook.remove()
    assert len(matching)==7
    competition,competition_counts=referit_competition_loss(output,batch,matching[1],'sr3d')
    assert torch.isfinite(native) and torch.isfinite(competition)
    loss=native+competition
    gradient=torch.autograd.grad(competition,output['last_sem_cls_scores'],retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    competition_gradient_norm=float(gradient.norm())
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
    reader=model.decoder[-1].source_query_read
    assert reader.output.weight.grad is not None and torch.isfinite(reader.output.weight.grad).all()
    assert float(reader.output.weight.grad.norm())>0
    assert reader.task_queries.grad is not None and torch.isfinite(reader.task_queries.grad).all()
    task_gradient_norms=[float(reader.task_queries.grad[i].norm()) for i in [0,1]]
    # At the zero-output initialization, task-query gradients can be zero.
    # No update is performed here, so this does not test learned task separation.
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
        losses=components,loss_native=float(native),loss_rec_competition=float(competition),
        competition_pair_counts=competition_counts,competition_logit_gradient_norm=competition_gradient_norm,
        competition_module_sha256=train_spec['competition_module_sha256'],
        gradient_groups=dict(groups),missing_gradients=missing,
        task_gradient_norms=task_gradient_norms,expanded_state_tensors=1272,
        source_port_sha256=train_spec['source_port_sha256'],task_module_sha256=train_spec['task_module_sha256'],
        all_parameters_unchanged=True,changed_buffers_before_restore=changed_buffers,
        full_state_restored=True,peak_allocated_bytes=peak,elapsed_seconds=elapsed,
        scope='one fixed augmented eight-row F backward; not training, formal accuracy or full-dataset capacity')
    (args.output/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('PVG_SR_REC_COMPETITION_BACKWARD_PASS '+json.dumps(record),flush=True)


if __name__=='__main__':
    main()
