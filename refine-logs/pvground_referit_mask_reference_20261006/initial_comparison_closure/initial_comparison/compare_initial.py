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
    hook_values=[]
    hook=model.x_query.register_forward_hook(lambda module,args,result:hook_values.append(result.detach().cpu().clone()))
    def capture(predictions):
        return dict(semantic=predictions['last_sem_cls_scores'].detach().cpu().clone(),
            center=predictions['last_center'].detach().cpu().clone(),size=predictions['last_pred_size'].detach().cpu().clone(),
            query=[x.detach().cpu().clone() for x in predictions['sp_last_pred_masks']],
            text=[x[0].detach().cpu().clone() for x in predictions['last_pred_masks']],
            alpha=[x.detach().cpu().clone() for x in predictions['adaptive_weights']],
            super_xyz=[x.detach().cpu().clone() for x in predictions['super_xyz_list']],
            x_query=hook_values[-1].clone(),query_xyz=predictions['query_points_xyz'].detach().cpu().clone())
    def differences(before,after):
        result=dict(semantic_max_abs=float((before['semantic']-after['semantic']).abs().max()),
            center_max_abs=float((before['center']-after['center']).abs().max()),
            size_max_abs=float((before['size']-after['size']).abs().max()),
            x_query_max_abs=float((before['x_query']-after['x_query']).abs().max()),
            query_xyz_max_abs=float((before['query_xyz']-after['query_xyz']).abs().max()),rows=[])
        for bid in range(8):
            a=before['query'][bid];b=after['query'][bid]
            fused_a=before['alpha'][bid]*before['text'][bid]+(1-before['alpha'][bid])*a
            fused_b=after['alpha'][bid]*after['text'][bid]+(1-after['alpha'][bid])*b
            result['rows'].append(dict(row_id=int(batch_cpu['local_training_id'][bid]),
                sample_dataset=batch_cpu['sample_dataset'][bid],query_logit_elements=a.numel(),
                query_mask_max_abs=float((a-b).abs().max()),query_mask_mean_abs=float((a-b).abs().mean()),
                query_mask_rms=float((a-b).square().mean().sqrt()),
                query_mask_original_allclose=bool(torch.allclose(a,b,rtol=1e-5,atol=1e-5)),
                query_mask_sign_changes=int((a.gt(0)!=b.gt(0)).sum()),
                fused_mask_sign_changes=int((fused_a.gt(0)!=fused_b.gt(0)).sum()),
                text_mask_max_abs=float((before['text'][bid]-after['text'][bid]).abs().max()),
                alpha_abs=float((before['alpha'][bid]-after['alpha'][bid]).abs()),
                super_xyz_max_abs=float((before['super_xyz'][bid]-after['super_xyz'][bid]).abs().max())))
        return result
    model.cuda();model.eval();torch.cuda.reset_peak_memory_stats();reset_rng()
    with torch.no_grad():
        native=model(dict(inputs))
        native_semantic=native['last_sem_cls_scores'].detach().clone()
        native_center=native['last_center'].detach().clone();native_size=native['last_pred_size'].detach().clone()
        native_masks=[value.detach().clone() for value in native['sp_last_pred_masks']]
        native_a=capture(native)
    del native
    reset_rng()
    with torch.no_grad():
        native_b_output=model(dict(inputs));native_b=capture(native_b_output)
    del native_b_output
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
        predictions=model(dict(inputs))
        installed=capture(predictions)
        reference_witness=reference_bounds_witness(predictions,batch)
    hook.remove();assert len(hook_values)==3
    record=dict(status='ACTUAL_ZERO_UPDATE_THREE_FORWARD_COMPARISON',
        time_cst=datetime.datetime.now().astimezone().isoformat(),dataset=dataset_name,reference_mode=spec['reference_mode'],
        model_forwards=3,optimizer_steps=0,formal_rows=0,new_weights_saved=0,
        identical_rng_before_each_forward=True,all_official_initial_states_exact=True,
        native_A_vs_native_B=differences(native_a,native_b),
        native_A_vs_installed_C=differences(native_a,installed),
        native_B_vs_installed_C=differences(native_b,installed),
        reference_witness=reference_witness,elapsed_seconds=time.perf_counter()-start,
        script_sha256=sha(__file__),spec_sha256=sha(args.spec))
    write_json(output/'comparison.json',record)
    print('ZERO_UPDATE_COMPARISON_COMPLETE '+json.dumps(record),flush=True)


if __name__=='__main__':main()
