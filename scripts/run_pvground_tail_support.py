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
    if args.mode == 'preflight':
        assert spec['p3']
        probe_loader = loader('fit', True, spec['seed'])
        probe_batch = next(iter(probe_loader))
        # The loader generator is independent of this model RNG; reset before
        # EACH full forward so native Gumbel sampling sees identical randomness.
        dataset.augment=False;dataset.augment_det=False
        model.eval();reset_rng(spec['seed'])
        inputs, _ = prepare(probe_batch, 'eval')
        inputs['train'] = False
        with torch.no_grad():
            with_p3,call_witness = observed_forward(model,inputs)
            preflight_call_witnesses.append(call_witness)
            read = model.candidate_box_refiner
            model.candidate_box_refiner = None
            reset_rng(spec['seed'])
            without_p3,call_witness = observed_forward(model,inputs)
            preflight_call_witnesses.append(call_witness)
            model.candidate_box_refiner = read
        differences = {}
        for key in ['last_sem_cls_scores', 'last_center', 'last_pred_size']:
            differences[key] = float((with_p3[key] - without_p3[key]).abs().max())
            assert torch.equal(with_p3[key], without_p3[key]), key
        support = with_p3['p3_neighbor_distances'].reshape(-1)
        support_statistics = {str(q):float(torch.quantile(support,q)) for q in (0.,.5,.9,.99,1.)}
        support_statistics['negative_coarse_size_elements'] = int((with_p3['p3_coarse_size'] < 0).sum())
        raw_points = inputs['points'][:,1:].view(inputs['batch_size'],50000,6)
        coarse_center = with_p3['p3_coarse_center'].detach()
        layout_size = with_p3['p3_coarse_size'].detach().clamp(min=1e-6)
        locations = coarse_center[:,:,None] + .5*layout_size[:,:,None]*read.locations
        members,_ = read.nearest_members(raw_points[...,:3],locations.reshape(inputs['batch_size'],256*7,3))
        mask_support = read.member_support(members,with_p3,inputs['batch_size'],256)
        members = members.reshape(inputs['batch_size'],256,7,16)
        for bid in range(inputs['batch_size']):
            for qi in (0,42,255):
                sid = inputs['superpoint'][bid][members[bid,qi]]
                tl = with_p3['last_pred_masks'][bid][0,qi,sid]
                ql = with_p3['sp_last_pred_masks'][bid][qi,sid]
                alpha = with_p3['adaptive_weights'][bid]
                expected = torch.stack([tl.sigmoid(),ql.sigmoid(),
                    (tl.sigmoid()-ql.sigmoid()).abs(),(alpha*tl+(1-alpha)*ql).sigmoid()],dim=-1)
                assert torch.equal(mask_support[bid,qi],expected)
        del mask_support,members,locations,raw_points
        del with_p3, without_p3, inputs
        model.load_state_dict(initial, strict=True)
        dataset.augment=True;dataset.augment_det=True
        reset_rng(spec['seed']);model.train()
        optimizer=BaseTrainTester.get_optimizer(training,model)
        torch.cuda.reset_peak_memory_stats()
        steps=[]
        for index in range(2):
            record=step(probe_batch,optimizer,True)
            gradients={n: float(p.grad.norm()) for n,p in read.named_parameters() if p.grad is not None}
            assert gradients['output.weight'] > 0
            if index == 1:
                assert gradients['member.0.weight'] > 0
                assert gradients['condition.weight'] > 0
                assert gradients['aggregate.0.weight'] > 0
                if spec['fused_support']:
                    assert record['geometry_to_mask_output_gradient'] > 0
            record['p3_gradients']=gradients
            steps.append(record)
        # Real serialization + optimizer reload in memory; do not consume another
        # large disk checkpoint just for this disposable probe.
        selected=set(trainable)|{n for n,_ in model.named_buffers() if n in initial}
        stream=io.BytesIO()
        torch.save({'delta':{n:v.detach().cpu() for n,v in model.state_dict().items() if n in selected},
                    'optimizer':optimizer.state_dict()},stream)
        stream.seek(0);reloaded=torch.load(stream,map_location='cpu')
        restored=dict(initial);restored.update(reloaded['delta'])
        model.load_state_dict(restored,strict=True)
        optimizer.load_state_dict(reloaded['optimizer'])
        assert all(torch.equal(v.detach().cpu(),restored[n]) for n,v in model.state_dict().items())
        assert all(int(state['step'])==2 for state in optimizer.state.values())
        receipt=dict(status='pass',time_cst=now(),batch_size=8,optimizer_steps=2,
            p3=True,p2=False,direct_semantic_to_p3_gradients_zero=True,
            support_distance_quantiles=support_statistics,
            g_strict_restore=True,zero_output_differences=differences,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),steps=steps,
            serialization_bytes=stream.getbuffer().nbytes,optimizer_restore=True,formal_rows=0,
            support_arm=spec['support_arm'],fused_support=spec['fused_support'],
            member_mask_mapping_exact=True,tail_after_native_masks=True,
            native_call_order_verified=True,native_call_witnesses=preflight_call_witnesses,
            direct_native_mask_loss_to_refiner_gradients_zero=True)
        assert len(preflight_call_witnesses) == 4
        write_json(output/'preflight.json',receipt)
        print('G_P3_REAL_PREFLIGHT_PASS '+json.dumps(receipt),flush=True)
        return

    @torch.no_grad()
    def evaluate(stage):
        dataset.augment=False;dataset.augment_det=False
        model.eval();reset_rng(spec['seed'])
        evaluator=GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1,5,10],prefixes=['last_'],filter_non_gt_boxes=False,model='PVGround')
        directory=output/stage;directory.mkdir()
        n=len(partitions['holdout'])
        rows=[];begin=time.time()
        with (directory/'rows.jsonl').open('w') as stream:
            for batch in loader('holdout',False,spec['seed']):
                inputs,batch=prepare(batch,'eval');inputs['train']=False
                predictions=model(inputs)
                coarse_boxes=torch.cat([predictions['p3_coarse_center'],predictions['p3_coarse_size'].clamp(min=1e-6)],-1)
                _,predictions=native_loss(predictions,batch)
                for key in predictions:
                    if 'pred_size' in key:predictions[key]=predictions[key].clamp(min=1e-6)
                evaluator.evaluate(predictions,'last_')
                prob=predictions['last_sem_cls_scores'].softmax(-1)
                cr=(torch.matmul(predictions['last_proj_queries'],predictions['proj_tokens'].transpose(-1,-2))/0.07).softmax(-1)
                cp=torch.zeros_like(prob);cp[:,:,:cr.shape[-1]]=cr
                boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
                gt=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1)
                for bid in range(len(batch['utterances'])):
                    row_id=int(batch['local_training_id'][bid])
                    index=len(rows)
                    assert row_id==partitions['holdout'][index]
                    record={'row_id':row_id,'scan_id':batch['scan_ids'][bid],'target_id':int(batch['target_id'][bid]),
                        'root_box':gt[bid].cpu().tolist(),'point_sha256':hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()}
                    lo=torch.maximum(boxes[bid,:,:3]-boxes[bid,:,3:]/2,gt[bid,:3]-gt[bid,3:]/2)
                    hi=torch.minimum(boxes[bid,:,:3]+boxes[bid,:,3:]/2,gt[bid,:3]+gt[bid,3:]/2)
                    intersection=(hi-lo).clamp(min=0).prod(-1)
                    iou=intersection/(boxes[bid,:,3:].prod(-1)+gt[bid,3:].prod()-intersection)
                    assert torch.isfinite(iou).all()
                    coarse_lo=torch.maximum(coarse_boxes[bid,:,:3]-coarse_boxes[bid,:,3:]/2,gt[bid,:3]-gt[bid,3:]/2)
                    coarse_hi=torch.minimum(coarse_boxes[bid,:,:3]+coarse_boxes[bid,:,3:]/2,gt[bid,:3]+gt[bid,3:]/2)
                    coarse_intersection=(coarse_hi-coarse_lo).clamp(min=0).prod(-1)
                    coarse_iou=coarse_intersection/(coarse_boxes[bid,:,3:].prod(-1)+gt[bid,3:].prod()-coarse_intersection)
                    assert torch.isfinite(coarse_iou).all()

                    for mode_index,(mode,probabilities) in enumerate([('bbs',prob),('bbf',cp)]):
                        score=(probabilities[bid]*(batch['positive_map'][bid,0]>0)).sum(-1)
                        for name in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                            score=score+(probabilities[bid]*batch[name][bid,0]).sum(-1)
                        score=score-(probabilities[bid]*batch['other_entity_map'][bid,0]).sum(-1)
                        ranked=score.argsort(descending=True);q=int(ranked[0])
                        alpha=predictions['adaptive_weights'][bid]
                        mask=((alpha*predictions['last_pred_masks'][bid][0,q]+(1-alpha)*predictions['sp_last_pred_masks'][bid][q]).sigmoid()>.5)[predictions['superpoints'][bid]]
                        truth=batch['gt_masks'][bid,0].bool()
                        mask_iou=float((mask&truth).sum().float()/(mask|truth).sum())
                        record[mode]={'query':q,'box':boxes[bid,q].cpu().tolist(),'iou':float(iou[q]),'mask_iou':mask_iou,
                            'coarse_iou':float(coarse_iou[q]),'coarse_box':coarse_boxes[bid,q].cpu().tolist(),
                            'coarse_oracle25':int((coarse_iou>.25).any()),'coarse_oracle50':int((coarse_iou>.5).any()),
                            'oracle25':[int((iou[ranked[:k]]>.25).any()) for k in [16,32,64,256]],
                            'oracle50':[int((iou[ranked[:k]]>.5).any()) for k in [16,32,64,256]]}
                    rows.append(record);stream.write(json.dumps(record)+'\n')
                if len(rows)%512<8:
                    stream.flush();print('PVG_EVAL_PROGRESS '+json.dumps({'stage':stage,'rows':len(rows),'total':n,'seconds':time.time()-begin}),flush=True)
                del predictions,inputs,batch
        assert len(rows)==n
        metrics={}
        for mode in ['bbs','bbf']:
            hit25=sum(r[mode]['iou']>.25 for r in rows);hit50=sum(r[mode]['iou']>.5 for r in rows)
            assert hit25==evaluator.dets[('last_',.25,1,mode)] and hit50==evaluator.dets[('last_',.5,1,mode)]
            key='mask_pos' if mode=='bbs' else 'mask_sem'
            mask_sum=sum(r[mode]['mask_iou'] for r in rows)
            assert abs(mask_sum-float(evaluator.dets[key]))<1e-3
            metrics[mode]={'rec_hits25':hit25,'rec_hits50':hit50,'mask_hits25':sum(r[mode]['mask_iou']>.25 for r in rows),
                'mask_hits50':sum(r[mode]['mask_iou']>.5 for r in rows),'mask_iou_sum':mask_sum,'mask_miou':mask_sum/n*100}
        receipt={'status':'pass','stage':stage,'rows':n,'metrics':metrics,'elapsed_seconds':time.time()-begin,
            'time_cst':now(),'formal_rows':n if formal else 0,'rows_sha256':sha(directory/'rows.jsonl')}
        write_json(directory/'receipt.json',receipt)
        print('PVG_EVAL_COMPLETE '+json.dumps(receipt),flush=True)
        return rows,receipt

    if formal:
        evaluate('formal')
        return
    initial_rows,initial_receipt=evaluate('initial')
    reference_root=Path(spec['reused_control_root'])
    reference_initial=[json.loads(line) for line in (reference_root/'initial/rows.jsonl').read_text().splitlines()]
    assert len(reference_initial)==len(initial_rows)==6887
    for actual,reference in zip(initial_rows,reference_initial):
        for name in ('row_id','scan_id','target_id','root_box','point_sha256'):
            assert actual[name]==reference[name],name
        for mode in ('bbs','bbf'):
            for name in ('query','box','iou'):
                assert actual[mode][name]==reference[mode][name],(mode,name)
    initial_mask_differences = {mode:[a['row_id'] for a,r in zip(initial_rows,reference_initial)
        if a[mode]['mask_iou'] != r[mode]['mask_iou']] for mode in ('bbs','bbf')}
    write_json(output/'initial_control_comparison.json',dict(rec_inputs_exact=True,mask_difference_row_ids=initial_mask_differences))
    reference_training=[json.loads(line) for line in (reference_root/'train.jsonl').read_text().splitlines()]
    assert len(reference_training)==3723
    print('P3_INITIAL_REC_INPUTS_MATCH_CONTROL',flush=True)
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    reset_rng(spec['seed']);dataset.augment=True;dataset.augment_det=True;model.train()
    optimizer=BaseTrainTester.get_optimizer(training,model)
    selected_names=set(trainable)|{n for n,_ in model.named_buffers() if n in initial}

    def save_checkpoint(name,step_number,seen_rows):
        current=model.state_dict()
        delta={k:v.detach().cpu() for k,v in current.items() if k in selected_names}
        data={'parent_checkpoint_sha256':checkpoint['sha256'],'state_delta':delta,'optimizer':optimizer.state_dict(),
            'step':step_number,'row_ids':seen_rows,'spec_sha256':sha(args.spec),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        data.update(task_read=True,task_module_sha256=spec['task_module_sha256'],observation_state=True,observation_module_sha256=spec['observation_module_sha256'],source_query_read=True,source_query_module_sha256=spec['source_query_module_sha256'],source_port_sha256=sha(spec['source_port']))
        data.update(p3=True,p3_module_sha256=spec['p3_module_sha256'],
                    support_arm=spec['support_arm'],fused_support=spec['fused_support'],
                    tail_module_sha256=spec['tail_module_sha256'])
        data.update(semantic_assignment=True,assignment_module_sha256=spec['assignment_module_sha256'],
                    p2=spec['p2'],base_terminal_sha256=spec['base_terminal_sha256'])
        temporary=output/(name+'.tmp')
        torch.save(data,str(temporary));os.replace(str(temporary),str(output/name))

    seen=[];begin=time.time();total=math.ceil(len(partitions['fit'])/8)
    with (output/'train.jsonl').open('w') as stream:
        for index,batch in enumerate(loader('fit',True,spec['seed']),1):
            assert batch['local_training_id'].tolist()==reference_training[index-1]['rows']
            record=step(batch,optimizer,True);seen.extend(record['rows'])
            record.update(step=index,total_steps=total,cumulative_seconds=time.time()-begin)
            stream.write(json.dumps(record)+'\n')
            if index==1 or index%64==0:
                stream.flush();print('PVG_TRAIN_PROGRESS '+json.dumps(record),flush=True)
            if index%512==0:save_checkpoint('latest.pth',index,seen)
    assert index==3723 and Counter(seen)==Counter(partitions['fit'])
    assert not set(seen).intersection(partitions['holdout'])
    assert all(torch.equal(p.detach().cpu(),initial[n]) for n,p in frozen.items())
    save_checkpoint('latest.pth',index,seen)
    os.replace(str(output/'latest.pth'),str(output/'terminal.pth'))
    print('PVG_FIT_COMPLETE '+json.dumps({'steps':index,'rows':len(seen),'seconds':time.time()-begin,'time_cst':now()}),flush=True)
    final_rows,final_receipt=evaluate('terminal')
    transitions={}
    for mode in ['bbs','bbf']:
        transitions[mode]={}
        for threshold in [.25,.5]:
            fixes=breaks=0
            for old,new in zip(initial_rows,final_rows):
                assert old['row_id']==new['row_id'] and old['point_sha256']==new['point_sha256'] and old['root_box']==new['root_box']
                before=old[mode]['iou']>threshold;after=new[mode]['iou']>threshold
                fixes+=not before and after;breaks+=before and not after
            transitions[mode][str(threshold)]={'fixes':fixes,'breaks':breaks,'net':fixes-breaks}
    assert sha(checkpoint['path'])==checkpoint['sha256']
    changed=[n for n,p in trainable.items() if not torch.equal(p.detach().cpu(),initial[n])]
    receipt={'status':'complete','time_cst':now(),'training_steps':index,'fit_rows':len(seen),'holdout_rows':len(final_rows),
        'formal_rows':0,'primary_mode':'bbs','initial':initial_receipt['metrics'],'terminal':final_receipt['metrics'],
        'transitions':transitions,'primary_rec_nonregression':all(transitions['bbs'][str(t)]['net']>=0 for t in [.25,.5]),
        'checkpoint_sha256':checkpoint['sha256'],'terminal_sha256':sha(output/'terminal.pth'),
        'changed_parameter_tensors':len(changed),'frozen_parameters_unchanged':True,'fit_seen_exactly_once':True,
        'train_log_sha256':sha(output/'train.jsonl'),'spec_sha256':sha(args.spec),'script_sha256':sha(__file__)}
    receipt.update(task_read=True,task_module_sha256=spec['task_module_sha256'],observation_state=True,observation_module_sha256=spec['observation_module_sha256'],source_query_read=True,source_query_module_sha256=spec['source_query_module_sha256'],source_port_sha256=sha(spec['source_port']),added_state_tensors=len(added),new_parameters=interface['new_parameters'])
    receipt.update(p3=True,p3_module_sha256=spec['p3_module_sha256'],initial_rec_inputs_match_control=True,
                   fit_batch_row_order_matches_control=True,reused_control_root=spec['reused_control_root'])
    receipt.update(semantic_assignment=True,assignment_module_sha256=spec['assignment_module_sha256'],
                   p2=spec['p2'],base_terminal_sha256=spec['base_terminal_sha256'],fresh_optimizer=True)
    receipt.update(support_arm=spec['support_arm'],fused_support=spec['fused_support'],
                   tail_module_sha256=spec['tail_module_sha256'],primary_threshold=.5,
                   paired_control_required=True,tail_after_native_masks=True)
    write_json(output/'receipt.json',receipt)
    print('PVG_FINETUNE_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
