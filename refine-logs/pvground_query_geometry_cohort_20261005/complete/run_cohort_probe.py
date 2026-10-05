"""Read-only paired geometry heads on the existing augmentedfit64 support cohort."""
import argparse
from collections import Counter
import importlib.util
import math
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    begin = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    runtime = Path(spec['runtime'])
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['geometry_hits50'] == 4506 and spec['geometry_frozen']
    assert spec['use_geometry_evidence'] is True
    for name, digest in spec['runner_files'].items():
        assert sha(Path(spec['helper_root']) / name) == digest, name
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((dataset_source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source / name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    env = json.loads((runtime / 'env_spec.json').read_bytes())
    env_sha = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert env_sha == spec['env_spec_sha256']
    interface = json.loads(Path(spec['training_interface_receipt']).read_bytes())
    assert interface['status'] == 'pass' and interface['optimizer_steps'] == 2
    assert interface['env_spec_sha256'] == env_sha and interface['strict_cpu_restore']
    assert interface['direct_routing_verified'] and interface['added_state_tensors'] == 37
    assert interface['new_parameters'] == 923616
    assert interface['source_port_sha256'] == sha(spec['parent_source_port'])
    assert interface['module_sha256'] == spec['runner_files']['pvground_task_observation_query.py']
    model_source = Path(spec['model_source'])
    assert sha(spec['source_port']) == spec['source_port_sha256']
    port = json.loads(Path(spec['source_port']).read_bytes())
    assert port['boundary_evidence_readback'] and port['native_semantic_head_deferred']
    assert port['call_position'] == 'after native Mask generation'
    for name, digest in port['files'].items():
        assert sha(model_source / name) == digest, name
    official_weight = env['weight_dirs']['scanrefer']
    assert sha(official_weight['path']) == official_weight['sha256'] == spec['checkpoint_sha256']
    assert sha(spec['base_terminal']) == spec['base_terminal_sha256']
    assert sha(spec['geometry_terminal']) == spec['geometry_terminal_sha256']

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    torch.cuda.reset_peak_memory_stats()

    def reset_rng():
        random.seed(spec['seed'])
        np.random.seed(spec['seed'])
        torch.manual_seed(spec['seed'])
        torch.cuda.manual_seed_all(spec['seed'])

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(str(dataset_source))
    sys.path.insert(0, str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == dataset_source / 'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    sys.path.insert(0, spec['helper_root'])
    sys.path.insert(0, str(model_source))
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    from readback_model_factory import build_readback_model
    from readback_preflight_checks import observed_readback_forward
    from native_root_bbs import native_root_bbs
    from pvground_boundary_box_refiner import face_targets, KNOTS
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ('src.joint_det_dataset', 'models.pv_ground', 'models.losses', 'main_utils', 'prepare_data')}
    for name in ('models.pv_ground', 'models.losses', 'main_utils', 'prepare_data'):
        assert model_source in Path(imported[name]).parents
    write_json(output / 'imports.json', dict(files=imported, sha256={k: sha(v) for k, v in imported.items()}))
    cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
    model, config, initial, load = build_readback_model(cfg,
        torch.load(official_weight['path'], map_location='cpu'),
        torch.load(spec['base_terminal'], map_location='cpu'),
        torch.load(spec['geometry_terminal'], map_location='cpu'), manifest['data_root'],
        spec['use_geometry_evidence'])
    load.update(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
                spec_sha256=sha(args.spec), geometry_terminal=spec['geometry_terminal'])
    write_json(output / 'load.json', load)
    model.cuda()
    readback = model.boundary_evidence_readback
    training = copy.copy(config)
    training.frozen = False
    training.small_lr = False
    training.lr = spec['lr']
    training.lr_backbone = spec['lr']
    assert training.weight_decay == spec['weight_decay'] and training.clip_norm == spec['clip_norm']
    criterion, set_criterion = BaseTrainTester.get_criterion(training)
    processors = {mode: DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), mode == 'train', 6)
                  for mode in ('train', 'eval')}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    os.chdir(str(dataset_source))
    dataset = FitDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
        detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
        augment_det=False, skip_missing_superpoints=True)
    assert len(dataset) == 36665
    physical = {part: {dataset.annos[index]['scan_id'].split('_')[0] for index in ids}
                for part, ids in partitions.items()}
    assert not physical['fit'].intersection(physical['holdout'])
    dataset.augment = False
    reference = json.loads((Path(spec['reference_fixtures']) / 'receipt.json').read_bytes())
    reset_rng()
    for row in reference['rows']:
        sample = dataset[row['training_row_id']]
        checks = dict(point_clouds=sample['point_clouds'], det_boxes=sample['all_detected_boxes'],
            det_bbox_label_mask=sample['all_detected_bbox_label_mask'],
            det_class_ids=sample['all_detected_class_ids'], superpoint=sample['superpoint'].numpy())
        assert sample['utterances'] == row['text']
        for name, array in checks.items():
            assert hashlib.sha256(array.tobytes()).hexdigest() == row['tensor_sha256'][name], name
    def loader(part, shuffle):
        return DataLoader(Subset(dataset, partitions[part]), batch_size=8, shuffle=shuffle, num_workers=2,
            generator=torch.Generator().manual_seed(spec['seed']), pin_memory=True, drop_last=False)

    def prepare(batch, mode):
        voxel_rows = [processors[mode].forward(dict(points=pc.numpy().copy(), use_lead_xyz=True))
                      for pc in batch['point_clouds']]
        voxels = processors[mode].collate_batch(voxel_rows)
        size = len(batch['utterances'])
        assert np.array_equal(voxels['points'][:, 1:].reshape(size, 50000, 6), batch['point_clouds'].numpy())
        batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        inputs = {key: torch.from_numpy(voxels[key]).float().cuda()
                  for key in ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
        inputs.update(batch_size=size, text=batch['utterances'], superpoint=batch['superpoint'],
            train=False, det_boxes=batch['all_detected_boxes'],
            det_bbox_label_mask=batch['all_detected_bbox_label_mask'], det_class_ids=batch['all_detected_class_ids'])
        return inputs, batch

    def native_loss(predictions, batch):
        assert not set(predictions).intersection(batch)
        predictions.update(batch)
        native, predictions = criterion(predictions, 6, set_criterion,
            query_points_obj_topk=training.query_points_obj_topk)
        assert torch.isfinite(native)
        return native, predictions

    def box_iou(boxes, truth):
        low = torch.maximum(boxes[..., :3] - boxes[..., 3:] / 2, truth[:3] - truth[3:] / 2)
        high = torch.minimum(boxes[..., :3] + boxes[..., 3:] / 2, truth[:3] + truth[3:] / 2)
        intersection = (high - low).clamp(min=0).prod(-1)
        iou = intersection / (boxes[..., 3:].prod(-1) + truth[3:].prod() - intersection)
        assert torch.isfinite(iou).all()
        return iou

    assert spec['probe_batches']==8 and spec['probe_rows']==64
    geometry_head=model.candidate_box_refiner
    head_keys={name for name in model.state_dict() if name.startswith('candidate_box_refiner.')}
    assert len(head_keys)==10
    assert sum(parameter.numel() for parameter in geometry_head.parameters())==456102
    original={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    heads={'parent':{name:original[name] for name in head_keys}}
    for arm in ('control','query_supported'):
        item=spec['terminal_heads'][arm]
        assert sha(item['path'])==item['sha256']
        payload=torch.load(item['path'],map_location='cpu')
        assert payload['step']==3723 and payload['total_geometry_fit_updates']==7446
        assert payload['head_only'] and payload['extra_geometry_weight']==(0. if arm=='control' else 1.)
        assert payload['geometry_terminal_sha256']==spec['geometry_terminal_sha256']
        assert payload['spec_sha256']==item['spec_sha256']
        assert set(payload['state_delta'])==head_keys
        for key in head_keys:
            assert payload['state_delta'][key].shape==original[key].shape
            assert payload['state_delta'][key].dtype==original[key].dtype
        heads[arm]=payload['state_delta']
        del payload

    def load_head(arm):
        prefix='candidate_box_refiner.'
        geometry_head.load_state_dict({key[len(prefix):]:value for key,value in heads[arm].items()},strict=True)
        geometry_head.eval()

    dataset.augment=True
    dataset.augment_det=True
    reset_rng()
    reference=[json.loads(line) for line in Path(spec['reference_rows']).read_text().splitlines()]
    assert len(reference)==64
    records=[]
    start=time.time()
    with (output/'rows.jsonl').open('x') as stream:
        for index,batch_cpu in enumerate(loader('fit',True)):
            row_ids=batch_cpu['local_training_id'].tolist()
            assert row_ids==[row['row_id'] for row in reference[index*8:(index+1)*8]]
            inputs,batch=prepare(batch_cpu,'train')
            load_head('parent')
            head_inputs=[]
            head_hook=geometry_head.register_forward_pre_hook(lambda module,arguments:head_inputs.append(arguments))
            matches=[]
            matcher_hook=set_criterion.matcher.register_forward_hook(
                lambda module,arguments,result:matches.append([(q.clone(),t.clone()) for q,t in result]))
            with torch.no_grad():
                predictions,call=observed_readback_forward(model,inputs)
                assert torch.equal(predictions['last_semantic_query_before_readback'],
                    predictions['last_semantic_query_after_readback'])
                parent_boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1).clone()
                native,predictions=native_loss(predictions,batch)
            head_hook.remove()
            matcher_hook.remove()
            assert len(head_inputs)==1 and len(matches)==7
            assert call['final_semantic_head_calls']==1
            scores=native_root_bbs(predictions['last_sem_cls_scores'],batch).detach().clone()
            masks=[value.detach().clone() for value in
                predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights']]
            coarse_center=head_inputs[0][2].detach().clone()
            coarse_size=head_inputs[0][3].detach().clone()
            candidates={}
            with torch.no_grad():
                for arm in ('parent','control','query_supported'):
                    load_head(arm)
                    center,size=geometry_head(*head_inputs[0])
                    boxes=torch.cat([center,size],-1)
                    assert torch.isfinite(boxes).all() and (size>0).all()
                    if arm=='parent':
                        assert torch.equal(boxes,parent_boxes)
                    candidates[arm]=dict(boxes=boxes.detach().clone(),
                        logits=predictions['boundary_logits'].detach().clone())
                    assert torch.equal(native_root_bbs(predictions['last_sem_cls_scores'],batch),scores)
                    assert all(torch.equal(current,old) for current,old in zip(
                        predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights'],masks))
            arrays={name:[] for name in ('row_id','root_box','coarse_box','bbs','query_intersection','query_union',
                'fused_intersection','fused_union','matched_slot','face_target','outside')}
            for arm in candidates:
                for name in ('boxes','iou','max_face_error','max_face_move','dfl'):
                    arrays[arm+'_'+name]=[]
            for bid,row_id in enumerate(row_ids):
                prior=reference[index*8+bid]
                point_digest=hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest()
                assert point_digest==prior['point_sha256'] and batch['scan_ids'][bid]==prior['scan_id']
                valid=batch['box_label_mask'][bid].bool().nonzero().flatten()
                assert int(valid[0])==0
                truth=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])
                assert truth.cpu().tolist()==prior['root_box']
                queries,targets=matches[1][bid]
                matched=torch.full((256,),-1,dtype=torch.int16,device=truth.device)
                matched[queries]=valid[targets].to(torch.int16)
                point_sp=predictions['superpoints'][bid]
                text=predictions['last_pred_masks'][bid][0]
                query=predictions['sp_last_pred_masks'][bid]
                alpha=predictions['adaptive_weights'][bid]
                mask_truth=batch['gt_masks'][bid,0].bool()
                members=torch.bincount(point_sp,minlength=text.shape[-1]).float()
                root_members=torch.bincount(point_sp[mask_truth],minlength=text.shape[-1]).float()
                support={}
                values=dict(row_id=np.asarray(row_id),root_box=truth,
                    coarse_box=torch.cat([coarse_center[bid],coarse_size[bid].clamp(min=1e-6)],-1),
                    bbs=scores[bid],matched_slot=matched)
                for branch,logits in (('query',query),('fused',alpha*text+(1-alpha)*query)):
                    mask=(logits.sigmoid()>.5).float()
                    intersection=mask@root_members
                    union=mask@members+mask_truth.sum()-intersection
                    support[branch]=intersection/union
                    assert torch.isfinite(support[branch]).all()
                    values[branch+'_intersection']=intersection.to(torch.int64)
                    values[branch+'_union']=union.to(torch.int64)
                target=face_targets(coarse_center[bid],coarse_size[bid],
                    truth[:3].expand(256,3),truth[3:].expand(256,3)).detach()
                knots=target.new_tensor(KNOTS)
                outside=(target<knots[0])|(target>knots[-1])
                clipped=target.clamp(min=knots[0],max=knots[-1])
                left=(clipped[...,None]>=knots).sum(-1).sub(1).clamp(0,32-1)
                right_weight=(clipped-knots[left])/(knots[left+1]-knots[left])
                values.update(face_target=target,outside=outside)
                truth_faces=torch.cat([truth[:3]-truth[3:]/2,truth[:3]+truth[3:]/2])
                coarse=values['coarse_box']
                coarse_faces=torch.cat([coarse[:,:3]-coarse[:,3:]/2,coarse[:,:3]+coarse[:,3:]/2],-1)
                for arm,result in candidates.items():
                    boxes=result['boxes'][bid]
                    faces=torch.cat([boxes[:,:3]-boxes[:,3:]/2,boxes[:,:3]+boxes[:,3:]/2],-1)
                    log_probability=result['logits'][bid].log_softmax(-1)
                    dfl=-(1-right_weight)*log_probability.gather(-1,left[...,None]).squeeze(-1)
                    dfl-=right_weight*log_probability.gather(-1,(left+1)[...,None]).squeeze(-1)
                    values.update({arm+'_boxes':boxes,arm+'_iou':box_iou(boxes,truth),
                        arm+'_max_face_error':(faces-truth_faces).abs().max(-1).values,
                        arm+'_max_face_move':(faces-coarse_faces).abs().max(-1).values,
                        arm+'_dfl':dfl.mean(-1)})
                qualified=(matched<0)&(support['query']>.5)&(support['fused']>.5)&(values['parent_iou']<=.5)
                selected=int(scores[bid].argmax())
                record=dict(row_id=row_id,batch_index=index,scan_id=prior['scan_id'],point_sha256=point_digest,
                    root_box=truth.cpu().tolist(),valid_native_GT_slots=valid.cpu().tolist(),
                    parent_matched_queries=queries.cpu().tolist(),parent_matched_slots=valid[targets].cpu().tolist(),
                    selected_query=selected,parent_qualified_count=int(qualified.sum()),
                    parent_qualified_outside_count=int((qualified&outside.any(-1)).sum()),
                    selected_ious={arm:float(values[arm+'_iou'][selected]) for arm in candidates},
                    frozen_upstream_and_score=True,native_head_calls=1,head_replays=3)
                records.append(record)
                stream.write(json.dumps(record)+'\n')
                for name,value in values.items():
                    arrays[name].append(value.cpu().numpy() if torch.is_tensor(value) else value)
            np.savez_compressed(str(output/('batch_%02d.npz'%index)),
                **{name:np.stack(values) for name,values in arrays.items()})
            stream.flush()
            print('GEOMETRY_COHORT_BATCH '+json.dumps(dict(index=index,rows=len(records),
                elapsed_seconds=time.time()-start)),flush=True)
            del predictions,inputs,batch,head_inputs,candidates
            if index+1==spec['probe_batches']:
                break
    load_head('parent')
    assert len(records)==64
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value.detach().cpu(),original[name]) for name,value in model.state_dict().items())
    receipt=dict(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),rows=64,batches=8,
        candidates=256,optimizer_steps=0,optimizer_constructed=False,weight_files_created=0,
        model_state_restored=True,model_gradients_absent=True,accuracy_result=False,
        parent_qualification_fixed=True,model_mode='eval',augmentation=dict(points=True,detected_boxes=True),
        full_model_forwards=8,native_semantic_head_calls=8,geometry_head_replays=24,
        frozen_upstream_masks_and_score_exact=True,parent_cached_replay_exact=True,
        elapsed_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),spec_sha256=sha(args.spec),
        runner_sha256=sha(__file__),rows_sha256=sha(output/'rows.jsonl'),
        limitations='Augmented training64 only, fixed parent qualification/matching cohort; not formal accuracy, physical identity truth or inference GT. Cached head comparison does not establish cross-CUDA full-forward bitwise identity.')
    write_json(output/'receipt.json',receipt)
    print('GEOMETRY_COHORT_COMPLETE '+json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
