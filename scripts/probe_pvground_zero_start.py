"""Paired same-tail raw versus native predicted fused-Mask support."""
import argparse
from collections import Counter
import copy
import datetime
import hashlib
import importlib.util
import json
import io
import math
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--mode', choices=['cpu', 'preflight', 'train', 'formal'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    assert spec['semantic_assignment'] and spec['assignment_threshold']==.5
    assert sha(output/'pvground_semantic_assignment.py')==spec['assignment_module_sha256']
    from pvground_semantic_assignment import semantic_assignment_correction, verify_native_replacement
    runtime = Path(spec['runtime'])
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source/'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((dataset_source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source/name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit'])==29778 and len(partitions['holdout'])==6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    env = json.loads((runtime/'env_spec.json').read_bytes())
    env_sha = hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert env_sha == spec['env_spec_sha256']
    interface = json.loads(Path(spec['training_interface_receipt']).read_bytes())
    assert interface['status']=='pass' and interface['optimizer_steps']==2
    assert interface['env_spec_sha256']==env_sha
    assert interface['source_query_read'] and spec['source_query_read']
    assert interface['observation_state'] and spec['observation_state']
    assert spec['observation_module_sha256']==sha(output/'pvground_observation_query.py')
    assert interface['task_read'] and spec['task_read']
    assert interface['module_sha256']==spec['task_module_sha256']==sha(output/'pvground_task_observation_query.py')
    assert spec['source_query_module_sha256']==sha(output/'pvground_source_query.py')
    assert interface['strict_cpu_restore'] and interface['direct_routing_verified']
    assert interface['added_state_tensors']==37 and interface['new_parameters']==923616
    # The existing D/G interface certifies its sealed parent source. The new
    # common source only adds P3 wiring and is checked separately below.
    assert interface['source_port_sha256']==sha(spec['parent_source_port'])
    upstream = json.loads((runtime/'source_bundle_receipt.json').read_bytes())
    model_source = Path(spec['model_source'])
    port = json.loads(Path(spec['source_port']).read_bytes())
    assert port['candidate_box_refinement'] and not port['p2']
    assert spec['p3'] and not spec['p2']
    assert sha(output/'pvground_candidate_box_refiner.py') == spec['p3_module_sha256']
    assert sha(output/'pvground_tail_support_box_refiner.py') == spec['tail_module_sha256']
    assert spec['support_arm'] in ('tail_raw','tail_fused')
    assert spec['fused_support'] == (spec['support_arm'] == 'tail_fused')
    assert port['call_position'] == 'after native Mask generation'
    for name,digest in port['files'].items():assert sha(model_source/name)==digest,name
    checkpoint=env['weight_dirs']['scanrefer']
    assert sha(checkpoint['path'])==checkpoint['sha256']==spec['checkpoint_sha256']
    assert spec['batch_size']==8 and spec['fit_passes']==1 and spec['seed']==2027
    assert spec['lr']==spec['lr_backbone']==1e-5

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    def reset_rng(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

    from pvground_tail_preflight import observed_forward, native_mask_loss_routes
    assert sha(output/'pvground_tail_preflight.py') == spec['tail_preflight_module_sha256']
    preflight_call_witnesses = []
    reset_rng(spec['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Keep the verified project's Dataset namespace, and import the official model
    # and evaluator from explicit source paths. Do not depend on accidental sys.path ordering.
    os.chdir(str(dataset_source))
    sys.path.insert(0,str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve()==dataset_source/'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    sys.path.insert(0,str(model_source))
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    evaluator_path=runtime/'PV-Ground/src/grounding_evaluator.py'
    module_spec=importlib.util.spec_from_file_location('pvground_official_evaluator',str(evaluator_path))
    evaluator_module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator=evaluator_module.GroundingEvaluator
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ['src.joint_det_dataset','models.pv_ground','models.losses','main_utils','prepare_data']}
    for name in ['models.pv_ground','models.losses','main_utils','prepare_data']:
        assert model_source in Path(imported[name]).parents
    imported['evaluator']=str(evaluator_path)
    write_json(output/'imports.json',{'files':imported,'sha256':{k:sha(v) for k,v in imported.items()}})

    payload=torch.load(checkpoint['path'],map_location='cpu')
    config=payload['config']
    initial={k[7:]:v for k,v in payload['model'].items()}
    assert all(k.startswith('module.') for k in payload['model']) and len(initial)==1234
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_soft_token_loss and config.use_contrastive_align
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model=PVGround(copy.deepcopy(cfg),num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=manifest['data_root'],self_attend=config.self_attend)
    position_ids=torch.arange(model.text_encoder.config.max_position_embeddings).expand((1,-1))
    assert set(model.state_dict())-set(initial)=={'text_encoder.embeddings.position_ids'}
    assert not set(initial)-set(model.state_dict())
    assert torch.equal(model.text_encoder.embeddings.position_ids,position_ids)
    model.text_encoder.embeddings.register_buffer('position_ids',model.text_encoder.embeddings.position_ids,persistent=False)
    model.load_state_dict(initial,strict=True)
    from pvground_task_observation_query import install_task_observation_query_read
    install_task_observation_query_read(model)
    added=set(model.state_dict())-set(initial)
    assert len(added)==37 and all(n.startswith('decoder.5.source_query_read.') for n in added)
    initial.update({n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n in added})
    assert sum(p.numel() for p in model.decoder[-1].source_query_read.parameters())==interface['new_parameters']
    # G is a trained delta on the author model plus the existing D/C reader.
    assert sha(spec['base_terminal']) == spec['base_terminal_sha256']
    base = torch.load(spec['base_terminal'], map_location='cpu')
    expected_delta = {n for n, p in model.named_parameters() if p.requires_grad}
    expected_delta.update(n for n, _ in model.named_buffers() if n in initial)
    assert set(base['state_delta']) == expected_delta and len(expected_delta) == 1072
    for name, value in base['state_delta'].items():
        assert value.shape == initial[name].shape and value.dtype == initial[name].dtype, name
    initial.update(base['state_delta'])
    model.load_state_dict(initial, strict=True)
    from pvground_tail_support_box_refiner import install_tail_support_refinement
    install_tail_support_refinement(model, spec['fused_support'])
    p3_names = set(model.state_dict()) - set(initial)
    assert len(p3_names) == 10 and all(n.startswith('candidate_box_refiner.') for n in p3_names)
    initial.update({n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n in p3_names})
    load_receipt = dict(status='pass', g_delta_tensors=1072, p3_states=len(p3_names),
                        base_terminal=spec['base_terminal'], fresh_optimizer=True,
                        p3=True, p2=False, support_arm=spec['support_arm'],
                        fused_support=spec['fused_support'], time_cst=now())
    write_json(output/'load.json', load_receipt)
    if args.mode == 'cpu':
        print('G_P3_CPU_RESTORE_PASS '+json.dumps(load_receipt), flush=True)
        return
    model.cuda()
    training=copy.copy(config)
    training.frozen=False;training.small_lr=False
    training.lr=spec['lr'];training.lr_backbone=spec['lr_backbone']
    assert training.weight_decay==0.0005 and training.clip_norm==0.1
    criterion,set_criterion=BaseTrainTester.get_criterion(training)
    processors={mode:DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),mode=='train',6)
                for mode in ['train','eval']}
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen={n:p for n,p in model.named_parameters() if not p.requires_grad}
    assert len(trainable)==interface['trainable_tensors'] + len(p3_names)

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            assert len(annos)==36665
            actual={'fit':[],'holdout':[]}
            for index,row in enumerate(annos):
                row['_local_training_id']=index
                code=(manifest['split_salt']+'\0'+row['scan_id'].split('_')[0]).encode()
                fold=int(hashlib.sha256(code).hexdigest()[:8],16)%5
                actual['holdout' if fold==0 else 'fit'].append(index)
            assert actual==partitions
            super()._scene_graph_parse(annos)

        def __getitem__(self,index):
            result=super().__getitem__(index)
            result['local_training_id']=self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'],[0,1]).all()
            result['gt_masks']=result['gt_masks'].astype(np.bool_)
            return result

    formal = args.mode == 'formal'
    if formal:
        terminal = torch.load(str(output/'terminal.pth'), map_location='cpu')
        assert terminal['base_terminal_sha256'] == spec['base_terminal_sha256']
        assert terminal['p3'] and not terminal['p2'] and terminal['step'] == 3723
        assert terminal['support_arm'] == spec['support_arm']
        assert terminal['tail_module_sha256'] == spec['tail_module_sha256']
        selected = set(trainable) | {n for n, _ in model.named_buffers() if n in initial}
        assert set(terminal['state_delta']) == selected
        restored = dict(initial)
        restored.update(terminal['state_delta'])
        model.load_state_dict(restored, strict=True)
    print('PVG_FINETUNE_DATASET_LOADING '+now(),flush=True)
    os.chdir(str(dataset_source))
    if formal:
        verify_scanrefer_superpoints(manifest['data_root'],'val',manifest['superpoint_files']['val'])
        class FormalDataset(Joint3DDataset):
            def _scene_graph_parse(self, annos):
                assert len(annos) == 9508
                for index, row in enumerate(annos):
                    row['_local_training_id'] = index
                super()._scene_graph_parse(annos)

            def __getitem__(self, index):
                result = super().__getitem__(index)
                result['local_training_id'] = self.annos[index]['_local_training_id']
                result['gt_masks'] = result['gt_masks'].astype(np.bool_)
                return result
        dataset = FormalDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='val',
            data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
            detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
            augment_det=False, skip_missing_superpoints=True)
        assert len(dataset) == 9508
        partitions = {'holdout': list(range(9508))}
    else:
        verify_scanrefer_superpoints(manifest['data_root'],'train',manifest['superpoint_files']['train'])
        dataset=FitDataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='train',
            data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
            detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
        assert [r['_local_training_id'] for r in dataset.annos]==list(range(36665))
        physical={part:{dataset.annos[i]['scan_id'].split('_')[0] for i in ids} for part,ids in partitions.items()}
        assert not physical['fit'].intersection(physical['holdout'])
        dataset.augment=False
        reference=json.loads((Path(spec['reference_fixtures'])/'receipt.json').read_bytes())
        reset_rng(spec['seed'])
        for row in reference['rows']:
            sample=dataset[row['training_row_id']]
            checks={'point_clouds':sample['point_clouds'],'det_boxes':sample['all_detected_boxes'],
                'det_bbox_label_mask':sample['all_detected_bbox_label_mask'],'det_class_ids':sample['all_detected_class_ids'],
                'superpoint':sample['superpoint'].numpy()}
            assert sample['utterances']==row['text']
            for name,array in checks.items():assert hashlib.sha256(array.tobytes()).hexdigest()==row['tensor_sha256'][name],name
        print('PVG_FULL_DATASET_INPUTS_PASS '+json.dumps({'fit':len(partitions['fit']),'holdout':len(partitions['holdout']),
            'physical_fit':len(physical['fit']),'physical_holdout':len(physical['holdout']),'reference_rows':4}),flush=True)

    def loader(part,shuffle,seed):
        return DataLoader(Subset(dataset,partitions[part]),batch_size=8,shuffle=shuffle,num_workers=2,
            generator=torch.Generator().manual_seed(seed),pin_memory=True,drop_last=False)

    def prepare(batch,mode):
        voxel_rows=[processors[mode].forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in batch['point_clouds']]
        voxels=processors[mode].collate_batch(voxel_rows)
        bs=len(batch['utterances'])
        assert np.array_equal(voxels['points'][:,1:].reshape(bs,50000,6),batch['point_clouds'].numpy())
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        inputs={k:torch.from_numpy(voxels[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs.update(batch_size=bs,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
            det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
            superpoint=batch['superpoint'])
        return inputs,batch

    def native_loss(predictions,batch):
        assert not set(predictions).intersection(batch)
        predictions.update(batch)
        loss,predictions=criterion(predictions,6,set_criterion,query_points_obj_topk=training.query_points_obj_topk)
        assert torch.isfinite(loss)
        return loss,predictions

    def step(batch,optimizer,update):
        begin=time.time()
        inputs,batch=prepare(batch,'train')
        if args.mode == 'preflight':
            predictions,call_witness = observed_forward(model,inputs)
            preflight_call_witnesses.append(call_witness)
        else:
            predictions=model(inputs)
        matching=[]
        def capture_match(module,inputs,result):
            matching.append([(q.clone(),t.clone()) for q,t in result])
        hook=set_criterion.matcher.register_forward_hook(capture_match)
        native,predictions=native_loss(predictions,batch)
        hook.remove()
        assert len(matching)==7
        correction,assignment_counts=semantic_assignment_correction(predictions,batch,matching[1],set_criterion.eos_coef)
        assert torch.isfinite(correction)
        loss=native+correction
        if args.mode == 'preflight':
            parameters = tuple(model.candidate_box_refiner.parameters())
            geometry_gradients = torch.autograd.grad(
                predictions['loss_bbox']+predictions['loss_giou'],
                parameters, retain_graph=True, allow_unused=True)
            geometry_output_gradient = float(geometry_gradients[-2].norm())
            assert geometry_output_gradient > 0
            semantic_gradients = torch.autograd.grad(
                predictions['loss_ce']+predictions['loss_sem_align']+correction,
                parameters, retain_graph=True, allow_unused=True)
            assert all(g is None or bool((g == 0).all()) for g in semantic_gradients)
            assignment_counts['direct_geometry_output_gradient'] = geometry_output_gradient
            assignment_counts['direct_semantic_to_p3_gradients_zero'] = True
            assignment_counts['native_mask_loss_to_refiner_gradients'] = native_mask_loss_routes(predictions,parameters)
            mask_outputs = tuple(predictions['last_pred_masks'] + predictions['sp_last_pred_masks'] + predictions['adaptive_weights'])
            mask_gradients = torch.autograd.grad(
                predictions['loss_bbox']+predictions['loss_giou'],
                mask_outputs,retain_graph=True,allow_unused=True)
            assignment_counts['geometry_to_mask_output_gradient'] = sum(
                float(g.norm()) for g in mask_gradients if g is not None)
            if not spec['fused_support']:
                assert assignment_counts['geometry_to_mask_output_gradient'] == 0
        if not update:
            assignment_counts['gradient_witness']=verify_native_replacement(predictions,batch,matching[1],set_criterion.eos_coef,correction)
        optimizer.zero_grad()
        loss.backward()
        for name,p in trainable.items():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),training.clip_norm)
        assert torch.isfinite(norm)
        if update:optimizer.step()
        torch.cuda.synchronize()
        return {'loss':float(loss),'loss_native':float(native),'loss_assignment_correction':float(correction),**assignment_counts,'grad_norm':float(norm),'seconds':time.time()-begin,
            'rows':batch['local_training_id'].cpu().tolist(),
            'loss_bbox':float(predictions['loss_bbox']),'loss_giou':float(predictions['loss_giou']),
            'loss_ce':float(predictions['loss_ce']),'loss_sem_align':float(predictions['loss_sem_align'])}

    if not formal:
        dataset.augment=True;dataset.augment_det=True
    reset_rng(spec['seed'])
    dataset.augment=False;dataset.augment_det=False
    model.eval();reset_rng(spec['seed'])
    batch=next(iter(loader('holdout',False,spec['seed'])))
    inputs,batch=prepare(batch,'eval');inputs['train']=False
    input_hash=hashlib.sha256(batch['point_clouds'].cpu().numpy().tobytes()).hexdigest()
    refiner=model.candidate_box_refiner
    assert bool((refiner.output.weight == 0).all())
    assert bool((refiner.output.bias == 0).all())
    input_snapshot={key:value.detach().cpu().clone() for key,value in inputs.items() if torch.is_tensor(value)}
    outputs={};rng_endpoints={};input_mutations={}
    labels=('fused_1','fused_2','fused_3','raw_1','bypass_1','bypass_2')
    with torch.no_grad():
        for label in labels:
            model.candidate_box_refiner=None if label.startswith('bypass') else refiner
            refiner.use_fused_support=not label.startswith('raw')
            fresh_inputs={key:value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                          for key,value in inputs.items()}
            assert all(torch.equal(fresh_inputs[key].detach().cpu(),value)
                       for key,value in input_snapshot.items())
            reset_rng(spec['seed'])
            prediction,witness=observed_forward(model,fresh_inputs)
            saved={key:prediction[key].detach().cpu().clone() for key in (
                'last_sem_cls_scores','last_center','last_pred_size','last_proj_queries',
                'seed_xyz','seed_features','query_points_xyz','query_points_feature')}
            if model.candidate_box_refiner is not None:
                assert torch.equal(prediction['last_center'],prediction['p3_coarse_center'])
                assert torch.equal(prediction['last_pred_size'],prediction['p3_coarse_size'])
            saved['call_witness']=witness
            outputs[label]=saved
            rng_endpoints[label]=dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=[hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in torch.cuda.get_rng_state_all()])
            input_mutations[label]=[key for key,value in input_snapshot.items()
                if not torch.equal(fresh_inputs[key].detach().cpu(),value)]
            del prediction,fresh_inputs
            model.candidate_box_refiner=refiner
            assert all(torch.equal(value.detach().cpu(),initial[name]) for name,value in model.state_dict().items())
    refiner.use_fused_support=spec['fused_support']
    differences={}
    for left,right in (('fused_1','fused_2'),('fused_2','fused_3'),
                       ('fused_3','raw_1'),('raw_1','bypass_1'),('bypass_1','bypass_2')):
        differences[left+'__'+right]={key:dict(
            exact=torch.equal(outputs[left][key],outputs[right][key]),
            max_abs=float((outputs[left][key]-outputs[right][key]).abs().max()),
            unequal_elements=int((outputs[left][key]!=outputs[right][key]).sum()))
            for key in ('last_sem_cls_scores','last_center','last_pred_size','last_proj_queries',
                        'seed_xyz','seed_features','query_points_xyz','query_points_feature')}
    receipt=dict(status='complete',time_cst=now(),script_sha256=sha(__file__),
        optimizer_updates=0,original_G_states_unchanged=True,zero_output_exactly_preserves_each_forward_box=True,
        row_ids=batch['local_training_id'].cpu().tolist(),point_hash=input_hash,
        differences=differences,rng_endpoints=rng_endpoints,input_tensor_mutations=input_mutations,
        call_witnesses={label:outputs[label]['call_witness'] for label in labels},
        backend_flags=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32))
    write_json(output/'zero_start_probe.json',receipt)
    print('PVG_ZERO_START_PROBE '+json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
