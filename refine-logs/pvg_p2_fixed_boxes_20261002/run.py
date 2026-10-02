"""Same trained P2, fixed boxes: actual versus bypassed semantic reading only.
Loader/strict restore reused from sealed runner 9ffb44c9. No training.
This is a direct forward diagnostic, not a trained no-P2 ablation.
"""
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
    parser.add_argument('--mode', choices=['formal'], default='formal')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, choices=[8, 9508], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    source_output = Path(spec['root'])
    assert spec['p2']
    output = args.output
    assert output != source_output
    output.mkdir()
    assert spec['semantic_assignment'] and spec['assignment_threshold']==.5
    assert sha(source_output/'pvground_semantic_assignment.py')==spec['assignment_module_sha256']
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
    assert spec['observation_module_sha256']==sha(source_output/'pvground_observation_query.py')
    assert interface['task_read'] and spec['task_read']
    assert interface['module_sha256']==spec['task_module_sha256']==sha(source_output/'pvground_task_observation_query.py')
    assert spec['source_query_module_sha256']==sha(source_output/'pvground_source_query.py')
    assert interface['strict_cpu_restore'] and interface['direct_routing_verified']
    assert interface['added_state_tensors']==37 and interface['new_parameters']==923616
    # The existing D/G interface certifies its sealed parent source. The new
    # common source only adds P2 wiring and is checked separately below.
    assert interface['source_port_sha256']==sha(spec['parent_source_port'])
    upstream = json.loads((runtime/'source_bundle_receipt.json').read_bytes())
    model_source = Path(spec['model_source'])
    port = json.loads(Path(spec['source_port']).read_bytes())
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
    if spec['p2']:
        from pvground_expression_evidence import install_expression_evidence_read
        install_expression_evidence_read(model)
        p2_names = set(model.state_dict()) - set(initial)
        assert p2_names and all(n.startswith('decoder.5.candidate_evidence_read.') for n in p2_names)
        initial.update({n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in p2_names})
    else:
        p2_names = set()
    load_receipt = dict(status='pass', g_delta_tensors=1072, p2_states=len(p2_names),
                        base_terminal=spec['base_terminal'], fresh_optimizer=False, time_cst=now())
    write_json(output/'load.json', load_receipt)
    if args.mode == 'cpu':
        print('G_P2_CPU_RESTORE_PASS '+json.dumps(load_receipt), flush=True)
        return
    model.cuda()
    training=copy.copy(config)
    training.frozen=False;training.small_lr=False
    training.lr=spec['lr'];training.lr_backbone=spec['lr_backbone']
    assert training.weight_decay==0.0005 and training.clip_norm==0.1
    processors={mode:DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),mode=='train',6)
                for mode in ['train','eval']}
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen={n:p for n,p in model.named_parameters() if not p.requires_grad}
    assert len(trainable)==interface['trainable_tensors'] + len(p2_names)
    assert spec['p2'] or sum(p.numel() for p in trainable.values())==interface['trainable_parameters']

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
        terminal = torch.load(str(source_output/'terminal.pth'), map_location='cpu')
        assert terminal['base_terminal_sha256'] == spec['base_terminal_sha256']
        assert terminal['p2'] == spec['p2'] and terminal['step'] == 3723
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


    import gzip
    import models.encoder_decoder_layers as decoder_module

    terminal_sha = sha(source_output/'terminal.pth')
    assert terminal_sha == '60371c5ecf6127f4738aa39d77e754a414a03a302ccdbc605173e1612692d9e9'
    dataset.augment=False;dataset.augment_det=False
    model.eval();reset_rng(spec['seed'])
    partitions['holdout'] = list(range(args.rows))
    original = [json.loads(line) for line in (source_output/'formal/rows.jsonl').read_text().splitlines()]
    assert len(original) == 9508
    original = original[:args.rows]
    last = model.decoder[-1]
    assert last.task_read and last.candidate_evidence_read is not None
    original_finish = decoder_module.finish_task_queries
    cache = {}

    def capture_source(module, inputs, result):
        assert 'residuals' not in cache
        cache['residuals'] = result

    def finish_with_bypass(layer, query, visual, residuals):
        assert layer is last and 'bypass' not in cache
        actual = original_finish(layer, query, visual, residuals)
        bypass = original_finish(layer, query, visual, cache['residuals'])
        cache['actual'] = actual[0]
        cache['bypass'] = bypass[0]
        return actual

    hook = last.source_query_read.register_forward_hook(capture_source)
    decoder_module.finish_task_queries = finish_with_bypass
    buffers = {name:value.detach().cpu().clone() for name,value in model.named_buffers()}
    evaluators = {mode:GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1,5,10],
        prefixes=['last_'],filter_non_gt_boxes=False,model='PVGround') for mode in ('actual','bypass')}
    rows=[];begin=time.time();max_logit_change=0.0
    with torch.no_grad(), gzip.open(str(output/'rows.jsonl.gz'),'wt',encoding='utf-8') as stream:
        for batch in loader('holdout',False,spec['seed']):
            cache.clear()
            inputs,batch=prepare(batch,'eval');inputs['train']=False
            predictions=model(inputs)
            assert set(cache)=={'residuals','actual','bypass'}
            head=model.prediction_heads[-1].sem_cls_scores_head
            recomputed=head(cache['actual'].transpose(1,2).contiguous()).transpose(2,1)
            assert torch.equal(recomputed,predictions['last_sem_cls_scores'])
            alternative=head(cache['bypass'].transpose(1,2).contiguous()).transpose(2,1)
            assert torch.isfinite(alternative).all()
            max_logit_change=max(max_logit_change,float((alternative-recomputed).abs().max()))
            assert not set(predictions).intersection(batch)
            predictions.update(batch)
            for key in list(predictions):
                if 'pred_size' in key:predictions[key]=predictions[key].clamp(min=1e-6)
            counterfactual=dict(predictions)
            counterfactual['last_sem_cls_scores']=alternative
            assert counterfactual['last_center'] is predictions['last_center']
            assert counterfactual['last_pred_size'] is predictions['last_pred_size']
            assert counterfactual['last_pred_masks'] is predictions['last_pred_masks']
            assert counterfactual['sp_last_pred_masks'] is predictions['sp_last_pred_masks']
            boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
            fixed_boxes=boxes.clone()
            evaluators['actual'].evaluate(predictions,'last_')
            evaluators['bypass'].evaluate(counterfactual,'last_')
            assert torch.equal(boxes,fixed_boxes)
            assert torch.equal(boxes,torch.cat([predictions['last_center'],predictions['last_pred_size']],-1))
            gt=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1)
            for bid in range(len(batch['utterances'])):
                index=len(rows);row_id=int(batch['local_training_id'][bid])
                assert row_id==index
                reference=original[index]
                point_sha=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()
                root_box=gt[bid].cpu().tolist()
                assert reference['row_id']==row_id and reference['point_sha256']==point_sha
                assert reference['root_box']==root_box
                lo=torch.maximum(boxes[bid,:,:3]-boxes[bid,:,3:]/2,gt[bid,:3]-gt[bid,3:]/2)
                hi=torch.minimum(boxes[bid,:,:3]+boxes[bid,:,3:]/2,gt[bid,:3]+gt[bid,3:]/2)
                intersection=(hi-lo).clamp(min=0).prod(-1)
                iou=intersection/(boxes[bid,:,3:].prod(-1)+gt[bid,3:].prod()-intersection)
                assert torch.isfinite(iou).all()
                record={'row_id':row_id,'scan_id':batch['scan_ids'][bid],'target_id':int(batch['target_id'][bid]),
                    'text':batch['utterances'][bid],'point_sha256':point_sha,'root_box':root_box,
                    'boxes':boxes[bid].cpu().tolist(),'ious':iou.cpu().tolist()}
                for mode,logits in [('actual',predictions['last_sem_cls_scores']),('bypass',alternative)]:
                    prob=logits[bid].softmax(-1)
                    score=(prob*(batch['positive_map'][bid,0]>0)).sum(-1)
                    for name in ('modify_positive_map','pron_positive_map','rel_positive_map'):
                        score=score+(prob*batch[name][bid,0]).sum(-1)
                    score=score-(prob*batch['other_entity_map'][bid,0]).sum(-1)
                    assert torch.isfinite(score).all()
                    ranked=score.argsort(descending=True);q=int(ranked[0])
                    record[mode]={'query':q,'iou':float(iou[q]),'scores':score.cpu().tolist(),
                        'oracle25':[int((iou[ranked[:k]]>.25).any()) for k in [16,32,64,256]],
                        'oracle50':[int((iou[ranked[:k]]>.5).any()) for k in [16,32,64,256]]}
                record['historical_actual_query']=reference['bbs']['query']
                record['historical_actual_iou']=reference['bbs']['iou']
                compact={key:value for key,value in record.items() if key not in ('boxes','ious','text')}
                for mode in ('actual','bypass'):
                    compact[mode]={key:value for key,value in record[mode].items() if key!='scores'}
                rows.append(compact)
                stream.write(json.dumps(record,separators=(',',':'))+'\n')
            if len(rows)%512<8:
                stream.flush();print('FIXED_BOX_PROGRESS '+json.dumps({'rows':len(rows),'total':args.rows,
                    'elapsed_seconds':time.time()-begin}),flush=True)
            del predictions,counterfactual,inputs,batch,alternative,recomputed,boxes,fixed_boxes
    hook.remove();decoder_module.finish_task_queries=original_finish
    assert len(rows)==args.rows
    assert all(torch.equal(value.detach().cpu(),buffers[name]) for name,value in model.named_buffers())
    assert sha(source_output/'terminal.pth')==terminal_sha
    metrics={}
    for mode in ('actual','bypass'):
        hit25=sum(row[mode]['iou']>.25 for row in rows)
        hit50=sum(row[mode]['iou']>.5 for row in rows)
        assert hit25==evaluators[mode].dets[('last_',.25,1,'bbs')]
        assert hit50==evaluators[mode].dets[('last_',.5,1,'bbs')]
        metrics[mode]={'hits25':hit25,'hits50':hit50}
    transitions={}
    for threshold in (.25,.5):
        repairs=sum(row['actual']['iou']>threshold and row['bypass']['iou']<=threshold for row in rows)
        damages=sum(row['actual']['iou']<=threshold and row['bypass']['iou']>threshold for row in rows)
        transitions[str(threshold)]={'repairs':repairs,'damages':damages,'net':repairs-damages}
    historical_mismatches=sum(row['actual']['query']!=row['historical_actual_query'] or
        any((row['actual']['iou']>threshold)!=(row['historical_actual_iou']>threshold) for threshold in (.25,.5))
        for row in rows)
    result={'status':'complete' if historical_mismatches==0 else 'replay_mismatch',
        'time_cst':now(),'rows':len(rows),'formal_rows':len(rows) if len(rows)==9508 else 0,
        'primary_mode':'bbs','same_checkpoint_fixed_boxes':True,'checkpoint_sha256':terminal_sha,
        'spec_sha256':sha(args.spec),'script_sha256':sha(__file__),'metrics':metrics,'transitions':transitions,
        'changed_selected_queries':sum(row['actual']['query']!=row['bypass']['query'] for row in rows),
        'historical_actual_mismatches':historical_mismatches,'max_semantic_logit_change':max_logit_change,
        'buffers_unchanged':True,'optimizer_updates':0,'rows_sha256':sha(output/'rows.jsonl.gz'),
        'rows_bytes':(output/'rows.jsonl.gz').stat().st_size,'elapsed_seconds':time.time()-begin,
        'scope':'direct semantic forward effect on fixed P2 boxes; not trained no-P2 causal ablation'}
    write_json(output/'receipt.json',result)
    print('FIXED_BOX_DIAGNOSTIC_SAVED '+json.dumps(result),flush=True)
    assert historical_mismatches==0, 'historical actual REC replay differs; result retained as diagnostic'
    print('FIXED_BOX_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
