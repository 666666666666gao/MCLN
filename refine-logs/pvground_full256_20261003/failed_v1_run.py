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
    parser.add_argument('--mode', choices=['formal'], required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    parser.add_argument('--limit', type=int, choices=[8,9508], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    sys.path.insert(0,str(output))
    audit_output = args.audit_output
    audit_output.mkdir()
    assert args.mode == 'formal'
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
    write_json(audit_output/'imports.json',{'files':imported,'sha256':{k:sha(v) for k,v in imported.items()}})

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
                        base_terminal=spec['base_terminal'], fresh_optimizer=False, optimizer_constructed=False,
                        p3=True, p2=False, support_arm=spec['support_arm'],
                        fused_support=spec['fused_support'], time_cst=now())
    write_json(audit_output/'load.json', load_receipt)
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

    # Only this read-only evaluation runs. The original training/preflight and
    # optimizer branches are absent from the generated script.
    import base64
    torch.set_grad_enabled(False)
    dataset.augment=False;dataset.augment_det=False
    model.eval();reset_rng(spec['seed'])
    before = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    evaluator=GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1,5,10],
        prefixes=['last_'],filter_non_gt_boxes=False,model='PVGround')
    rows=[];begin=time.time();chunks=0

    def box_iou(boxes, targets):
        lo=torch.maximum(boxes[:,None,:3]-boxes[:,None,3:]/2,targets[None,:,:3]-targets[None,:,3:]/2)
        hi=torch.minimum(boxes[:,None,:3]+boxes[:,None,3:]/2,targets[None,:,:3]+targets[None,:,3:]/2)
        intersection=(hi-lo).clamp(min=0).prod(-1)
        result=intersection/(boxes[:,None,3:].prod(-1)+targets[None,:,3:].prod(-1)-intersection)
        assert bool(torch.isfinite(result).all())
        return result

    with (audit_output/'rows.jsonl').open('w') as stream:
        for batch in loader('holdout',False,spec['seed']):
            inputs,batch=prepare(batch,'eval');inputs['train']=False
            predictions=model(inputs)
            matching=[]
            def capture_match(module, inputs, result):
                matching.append([(q.clone(),t.clone()) for q,t in result])
            hook=set_criterion.matcher.register_forward_hook(capture_match)
            _,predictions=native_loss(predictions,batch)
            hook.remove()
            assert len(matching)==7
            last_matches=matching[1]  # native order: proposal, last, 0head,...
            for key in predictions:
                if 'pred_size' in key:predictions[key]=predictions[key].clamp(min=1e-6)
            evaluator.evaluate(predictions,'last_')
            boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
            coarse=torch.cat([predictions['p3_coarse_center'],predictions['p3_coarse_size'].clamp(min=1e-6)],-1)
            probabilities=predictions['last_sem_cls_scores'].softmax(-1)
            scores=(probabilities*(batch['positive_map'][:,0,None]>0)).sum(-1)
            for name in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                scores=scores+(probabilities*batch[name][:,0,None]).sum(-1)
            scores=scores-(probabilities*batch['other_entity_map'][:,0,None]).sum(-1)
            arrays={name:[] for name in ('boxes','coarse_boxes','bbs_scores','root_iou','root_mask_iou',
                'matched_GT_slot','best_scene_GT_id','best_scene_GT_class','best_scene_GT_iou',
                'root_joint_best_scene_overlap','no_object_probability','row_id')}
            for bid in range(len(batch['utterances'])):
                row_id=int(batch['local_training_id'][bid]);assert row_id==len(rows)
                assert bool(batch['box_label_mask'][bid,0])
                root=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])
                iou=box_iou(boxes[bid],root[None])[:,0]
                coarse_iou=box_iou(coarse[bid],root[None])[:,0]
                rank=scores[bid].argsort(descending=True);selected=int(rank[0])
                valid_slots=batch['box_label_mask'][bid].bool().nonzero().flatten()
                matched=torch.full((256,),-1,dtype=torch.int16,device=boxes.device)
                queries,targets=last_matches[bid]
                assert int((targets==0).sum())==1 and int(valid_slots[0])==0
                matched[queries]=valid_slots[targets].to(torch.int16)
                scene_ids=batch['all_bbox_label_mask'][bid].bool().nonzero().flatten()
                assert scene_ids.numel()>0
                scene_boxes=batch['all_bboxes'][bid,scene_ids]
                target_id=int(batch['target_id'][bid])
                root_scene_slot=(scene_ids==target_id).nonzero().flatten()
                assert root_scene_slot.numel()==1
                assert torch.equal(scene_boxes[root_scene_slot[0]],root)
                scene_overlap=box_iou(boxes[bid],scene_boxes)
                best_overlap,best_local=scene_overlap.max(-1)
                best_id=scene_ids[best_local]
                best_class=batch['all_class_ids'][bid,best_id]
                root_joint_best=scene_overlap[:,root_scene_slot[0]]==best_overlap

                # Count actual input members per superpoint; avoid Q x 50000
                # point-mask tensors. Keep native sigmoid > .5 exactly.
                point_sp=predictions['superpoints'][bid]
                text=predictions['last_pred_masks'][bid][0]
                query=predictions['sp_last_pred_masks'][bid]
                assert text.shape==query.shape and text.shape[0]==256
                truth=batch['gt_masks'][bid,0].bool()
                assert bool(truth.any())
                member_count=torch.bincount(point_sp,minlength=text.shape[-1]).float()
                root_count=torch.bincount(point_sp[truth],minlength=text.shape[-1]).float()
                alpha=predictions['adaptive_weights'][bid]
                support=(alpha*text+(1-alpha)*query).sigmoid()>.5
                intersection=support.float()@root_count
                union=support.float()@member_count+truth.sum()-intersection
                mask_iou=intersection/union
                assert bool(torch.isfinite(mask_iou).all())
                selected_mask=support[selected][point_sp]
                exact_mask=(selected_mask&truth).sum().float()/(selected_mask|truth).sum()
                assert float(mask_iou[selected])==float(exact_mask)

                first_rank={}
                error_candidates={}
                for threshold,suffix in ((.25,'25'),(.5,'50')):
                    good=iou>threshold
                    positions=good[rank].nonzero().flatten()
                    first_rank[suffix]=int(positions[0])+1 if positions.numel() else None
                    error_candidates[suffix]={
                        'matched_root':int((good&(matched==0)).sum()),
                        'matched_other':int((good&(matched>0)).sum()),
                        'unmatched':int((good&(matched<0)).sum()),
                        'qualified_in_topk':[int(good[rank[:k]].any()) for k in (16,32,64,256)],
                        'qualified_root_overlap_proxy':int((good&root_joint_best).sum())}
                record=dict(row_id=row_id,scan_id=batch['scan_ids'][bid],target_id=target_id,
                    utterance=batch['utterances'][bid],root_box=root.cpu().tolist(),
                    point_sha256=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest(),
                    selected_query=selected,selected_iou=float(iou[selected]),
                    selected_coarse_iou=float(coarse_iou[selected]),selected_mask_iou=float(mask_iou[selected]),
                    selected_matched_slot=int(matched[selected]),
                    selected_best_scene_GT_id=int(best_id[selected]),
                    selected_root_joint_best_overlap=bool(root_joint_best[selected]),
                    valid_native_GT_slots=valid_slots.cpu().tolist(),first_qualified_rank=first_rank,
                    geometric_qualification=error_candidates,candidate_count=256)
                rows.append(record);stream.write(json.dumps(record)+'\n')
                tensors={'boxes':boxes[bid],'coarse_boxes':coarse[bid],'bbs_scores':scores[bid],
                    'root_iou':iou,'root_mask_iou':mask_iou,'matched_GT_slot':matched,
                    'best_scene_GT_id':best_id.to(torch.int16),
                    'best_scene_GT_class':best_class.to(torch.int16),'best_scene_GT_iou':best_overlap,
                    'root_joint_best_scene_overlap':root_joint_best,
                    'no_object_probability':probabilities[bid,:,-1]}
                for name,value in tensors.items():arrays[name].append(value.cpu().numpy())
                arrays['row_id'].append(row_id)
            buffer=io.BytesIO()
            np.savez_compressed(buffer,**{name:np.stack(values) for name,values in arrays.items()})
            # Stream chunks directly to the local archive: no large remote cache.
            print('CANDIDATE_CHUNK '+str(chunks)+' '+base64.b64encode(buffer.getvalue()).decode('ascii'),flush=True)
            chunks+=1
            stream.flush()
            if len(rows)%512<8:print('CANDIDATE_AUDIT_PROGRESS '+json.dumps(dict(rows=len(rows),seconds=time.time()-begin)),flush=True)
            del predictions,inputs,batch
            if len(rows)==args.limit:break
    assert len(rows)==args.limit
    for name,value in model.state_dict().items():assert torch.equal(value.detach().cpu(),before[name]),name
    hits={suffix:sum(r['selected_iou']>threshold for r in rows) for suffix,threshold in [('25',.25),('50',.5)]}
    assert all(hits[suffix]==evaluator.dets[('last_',threshold,1,'bbs')] for suffix,threshold in [('25',.25),('50',.5)])
    assert abs(sum(r['selected_mask_iou'] for r in rows)-float(evaluator.dets['mask_pos']))<1e-3
    receipt=dict(status='pass',time_cst=now(),rows=len(rows),chunks=chunks,rec_hits=hits,
        elapsed_seconds=time.time()-begin,optimizer_updates=0,optimizer_constructed=False,
        all_candidates_retained=256,model_state_unchanged=True,official_evaluator_counts_exact=True,
        parent_arm=spec['support_arm'],terminal_step=int(terminal['step']),
        input_manifest=spec['input_manifest'],rows_sha256=sha(audit_output/'rows.jsonl'),
        evidence_limits=('Read-only full-candidate diagnostic, not a retrained ablation. '
            'Native matcher is applied to evaluation inputs; unmatched status is not physical background identity. '
            'IoU qualification and nearest annotated scene overlap are GT-only geometric proxies, not a deployed selection rule. '
            'All 256 are kept; no new pruning, supervision or optimizer update. Fresh forward may differ numerically from archived formal output.'))
    write_json(audit_output/'receipt.json',receipt)
    print('CANDIDATE_AUDIT_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
