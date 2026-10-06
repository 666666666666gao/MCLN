"""Two real mixed-row updates from the corresponding author checkpoint, no Scan delta."""
import argparse
import copy
import datetime
import hashlib
import io
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


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.spec.read_bytes())
    output=Path(spec['root']);assert output==args.spec.parent
    start=time.perf_counter();dataset_name=spec['dataset']
    assert dataset_name in ('nr3d','sr3d') and spec['reference_mode'] in ('native','fused_mask')
    assert spec['seed']==2027 and spec['batch_size']==8 and spec['steps']==2
    assert spec['lr']==1e-5 and spec['weight_decay']==.0005 and spec['clip_norm']==.1
    for name,digest in spec['files'].items():assert sha(output.parent/name)==digest,name
    runtime=Path(spec['runtime']);env=json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',', ':')).encode()).hexdigest()==spec['env_spec_sha256']
    source=Path(spec['model_source']);port=json.loads(Path(spec['source_port']).read_bytes())
    assert sha(spec['source_port'])==spec['source_port_sha256']
    for name,digest in port['files'].items():assert sha(source/name)==digest,name
    for name,digest in spec['helper_files'].items():assert sha(Path(spec['helper_root'])/name)==digest,name
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    assert sha(spec['partition'])==spec['partition_sha256']
    manifest=json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source=Path(manifest['model_source'])
    assert sha(dataset_source/'appearance_source_manifest.json')==manifest['source_manifest_sha256']
    for name,digest in json.loads((dataset_source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source/name)==digest,name
    partition=json.loads(Path(spec['partition']).read_bytes())
    assert partition['dataset']==dataset_name and partition['detection_repeats']==10
    assert not set(partition['fit_ids']).intersection(partition['holdout_ids'])
    assert not set(partition['fit_physical_scenes']).intersection(partition['holdout_physical_scenes'])
    import fcntl
    lock=open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Subset
    def reset_rng():
        random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    reset_rng();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.chdir(str(dataset_source));sys.path.insert(0,str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve()==dataset_source/'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(source));sys.path.insert(0,spec['helper_root']);sys.path.insert(0,str(output.parent));sys.path.insert(0,str(source))
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg,cfg_from_yaml_file
    from pvground_task_observation_query import install_task_observation_query_read
    from pvground_boundary_box_refiner import install_boundary_refinement,distribution_loss
    from install_boundary_evidence_readback import install_boundary_evidence_readback
    from mask_reference import install_mask_reference,reference_bounds_witness
    from whole_model_preflight_checks import optimizer_restore_exact
    from pvground_referit_fit_dataset import make_fit_dataset_class
    from referit_training_targets import semantic_assignment_correction,verify_native_replacement,extra_geometry_loss
    imports={name:str(Path(sys.modules[name].__file__).resolve()) for name in
        ('src.joint_det_dataset','models.pv_ground','models.losses','main_utils','prepare_data')}
    write_json(output/'imports.json',dict(files=imports,sha256={name:sha(path) for name,path in imports.items()}))
    payload=torch.load(spec['checkpoint'],map_location='cpu');config=payload['config']
    assert not config.butd and config.butd_cls and not config.butd_gt
    assert config.use_soft_token_loss and config.use_contrastive_align
    official={key[7:]:value for key,value in payload['model'].items()}
    assert len(official)==1235 and all(key.startswith('module.') for key in payload['model'])
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model=PVGround(copy.deepcopy(cfg),num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=manifest['data_root'],self_attend=config.self_attend)
    assert set(model.state_dict())==set(official)
    model.load_state_dict(official,strict=True)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in official.items())
    install_task_observation_query_read(model)
    reader=model.decoder[-1].source_query_read
    model.decoder[-1].source_query_read=None
    model.decoder[-1].task_read=False
    FitDataset=make_fit_dataset_class(Joint3DDataset,partition,dataset_name)
    os.chdir(str(dataset_source))
    dataset=FitDataset(dataset_dict={dataset_name:1,'scannet':10},test_dataset=dataset_name,split='train',
        data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
        detect_intermediate=True,butd=False,butd_cls=True,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    assert len(dataset.annos)==partition['original_language_rows']+10*partition['original_detection_base_rows']
    fit_ids=partition['fit_ids'];nlanguage=partition['original_language_rows']
    rows=[index for index in fit_ids if index<nlanguage][:4]+[index for index in fit_ids if index>=nlanguage][:4]
    assert len(rows)==8 and len(set(rows))==8
    dataset.augment=True;dataset.augment_det=True;reset_rng()
    batch_cpu=next(iter(DataLoader(Subset(dataset,rows),batch_size=8,shuffle=False,num_workers=2,
        generator=torch.Generator().manual_seed(2027),pin_memory=True)))
    assert list(batch_cpu['sample_dataset'])==[dataset_name]*4+['scannet']*4
    assert all(name==dataset_name for name in batch_cpu['language_dataset'])
    processors=DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),True,6)
    voxel_rows=[processors.forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in batch_cpu['point_clouds']]
    voxels=processors.collate_batch(voxel_rows)
    assert np.array_equal(voxels['points'][:,1:].reshape(8,50000,6),batch_cpu['point_clouds'].numpy())
    batch={key:value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key,value in batch_cpu.items()}
    inputs={key:torch.from_numpy(voxels[key]).float().cuda() for key in ('points','voxels','voxel_coords','voxel_num_points')}
    inputs.update(batch_size=8,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
        det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
        superpoint=batch['superpoint'])
    model.cuda();model.eval();torch.cuda.reset_peak_memory_stats();reset_rng()
    with torch.no_grad():
        native=model(inputs)
        native_semantic=native['last_sem_cls_scores'].detach().clone()
        native_center=native['last_center'].detach().clone();native_size=native['last_pred_size'].detach().clone()
        native_masks=[value.detach().clone() for value in native['sp_last_pred_masks']]
    del native
    model.decoder[-1].source_query_read=reader
    model.decoder[-1].task_read=True
    install_boundary_refinement(model,'distribution')
    install_boundary_evidence_readback(model,True)
    for parameter in model.boundary_evidence_readback.parameters():parameter.requires_grad_(False)
    install_mask_reference(model,spec['reference_mode'])
    model.cuda();model.eval()
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    assert len(initial)==1305 and len(set(initial)-set(official))==70
    assert all(torch.equal(value,initial[key]) for key,value in official.items())
    reset_rng()
    with torch.no_grad():
        predictions=model(inputs)
        semantic_error=float((predictions['last_sem_cls_scores']-native_semantic).abs().max())
        assert torch.allclose(predictions['last_sem_cls_scores'],native_semantic,rtol=1e-5,atol=1e-5)
        assert all(torch.allclose(value,old,rtol=1e-5,atol=1e-5) for value,old in zip(predictions['sp_last_pred_masks'],native_masks))
        reference_witness=reference_bounds_witness(predictions,batch)
        assert torch.equal(predictions['last_center'],predictions['geometry_reference_center'])
        assert torch.equal(predictions['last_pred_size'],predictions['geometry_reference_size'].clamp(min=1e-6))
        if spec['reference_mode']=='native':
            assert torch.allclose(predictions['last_center'],native_center,rtol=1e-5,atol=1e-5)
            assert torch.allclose(predictions['last_pred_size'],native_size.clamp(min=1e-6),rtol=1e-5,atol=1e-5)
    del predictions,native_semantic,native_center,native_size,native_masks
    training=copy.copy(config);training.frozen=False;training.small_lr=False
    training.lr=training.lr_backbone=1e-5
    assert training.weight_decay==.0005 and training.clip_norm==.1
    criterion,set_criterion=BaseTrainTester.get_criterion(training)
    optimizer=BaseTrainTester.get_optimizer(training,model)
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    frozen={name:p for name,p in model.named_parameters() if not p.requires_grad}
    assert any(name.startswith('decoder.5.source_query_read.') for name in trainable)
    assert any(name.startswith('candidate_box_refiner.') for name in trainable)
    assert any(not name.startswith(('decoder.5.source_query_read.','candidate_box_refiner.')) for name in trainable)
    records=[];model.train();reset_rng()
    for step in (1,2):
        begin=time.perf_counter();predictions=model(inputs)
        matches=[]
        hook=set_criterion.matcher.register_forward_hook(lambda module,args,result:matches.append(
            [(queries.detach().clone(),targets.detach().clone()) for queries,targets in result]))
        assert not set(predictions).intersection(batch)
        predictions.update(batch)
        native,predictions=criterion(predictions,6,set_criterion,query_points_obj_topk=training.query_points_obj_topk)
        hook.remove();assert len(matches)==7
        correction,selected=semantic_assignment_correction(predictions,batch,matches[1],set_criterion.eos_coef,dataset_name)
        ce_witness=verify_native_replacement(predictions,batch,matches[1],set_criterion.eos_coef,
            dataset_name,correction,selected)
        edge,edge_counts=distribution_loss(predictions,batch,matches[1])
        extra,extra_counts,qualified=extra_geometry_loss(predictions,batch,matches[1],set_criterion,dataset_name)
        assert all(value.numel()==0 for value in qualified[4:])
        assert all(not set(value.tolist()).intersection(matches[1][bid][0].tolist()) for bid,value in enumerate(qualified))
        loss=native+correction+edge/7+extra
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1);assert torch.isfinite(norm)
        geometry_grad=float(model.candidate_box_refiner.output.weight.grad.norm())
        assert geometry_grad>0
        optimizer.step();torch.cuda.synchronize()
        record=dict(step=step,loss=float(loss),native=float(native),semantic_correction=float(correction),
            boundary=float(edge),extra_geometry=float(extra),qualified_semantic=int(selected.sum()),
            geometry_output_task_gradient=geometry_grad,grad_norm=float(norm),seconds=time.perf_counter()-begin,
            ce_witness=ce_witness,extra_counts=extra_counts,boundary_counts=edge_counts)
        records.append(record);print('REFERIT_REAL_PREFLIGHT_STEP '+json.dumps(record),flush=True)
        del predictions,native,correction,edge,extra,loss
    assert all(torch.equal(p.detach().cpu(),initial[name]) for name,p in frozen.items())
    final={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    checkpoint=io.BytesIO();torch.save(dict(model=final,optimizer=optimizer.state_dict(),step=2),checkpoint)
    checkpoint.seek(0);restored=torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(restored['model'],strict=True)
    assert set(model.state_dict())==set(final)
    assert all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in final.items())
    optimizer.load_state_dict(restored['optimizer']);optimizer_witness=optimizer_restore_exact(optimizer,restored['optimizer'])
    assert all(int(state['step'])==2 for state in restored['optimizer']['state'].values())
    changed=[name for name,p in trainable.items() if not torch.equal(p.detach().cpu(),initial[name])]
    assert any(name.startswith('candidate_box_refiner.') for name in changed)
    assert any(name.startswith('decoder.5.source_query_read.') for name in changed)
    receipt=dict(status='pass',dataset=dataset_name,reference_mode=spec['reference_mode'],
        time_cst=datetime.datetime.now().astimezone().isoformat(),elapsed_seconds=time.perf_counter()-start,
        official_checkpoint_sha256=spec['checkpoint_sha256'],corresponding_official_initial_states_exact=True,
        scanrefer_weights_loaded=False,initial_full_state_tensors=1305,new_state_tensors=70,
        trainable_tensors=len(trainable),trainable_parameters=sum(p.numel() for p in trainable.values()),
        rows=rows,sample_dataset=list(batch_cpu['sample_dataset']),
        gt_counts=batch_cpu['box_label_mask'].sum(-1).tolist(),native_semantic_initial_max_error=semantic_error,
        reference_witness=reference_witness,steps=records,actual_optimizer_steps=2,
        strict_full_model_and_optimizer_CPU_restore=True,optimizer_restore=optimizer_witness,
        checkpoint_serialization_bytes=checkpoint.getbuffer().nbytes,serialization_in_memory=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),frozen_parameters_exact=True,
        changed_parameter_tensors=len(changed),new_weights_saved=0,formal_rows=0,
        script_sha256=sha(__file__),spec_sha256=sha(args.spec))
    write_json(output/'receipt.json',receipt);print('REFERIT_REAL_PREFLIGHT_PASS '+json.dumps(receipt),flush=True)


if __name__=='__main__':main()
