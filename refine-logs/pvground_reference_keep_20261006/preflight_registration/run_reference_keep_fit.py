"""Retained4848 complete Mask-reference state; original versus reference-preserving supervision."""
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
    parser.add_argument('--mode', choices=['preflight', 'train', 'initial_formal', 'formal'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    runtime = Path(spec['runtime'])
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['starting_hits']==[5598,4848] and spec['head_only']
    assert spec['use_geometry_evidence'] is True and spec['extra_geometry_weight']==1.0
    assert spec['auxiliary_target_mode']=='native_gt'
    assert spec['reference_mode']=='fused_mask' and spec['reference_keep_weight'] in (0.0,1.0)
    assert spec['common_output_reset']==['output.weight','output.bias']
    assert sha(output.parent/'mask_reference.py')==spec['mask_reference_sha256']
    assert sha(output.parent/'reference_keep.py')==spec['reference_keep_sha256']
    assert sha(output.parent/'selected_mask_reference_factory.py')==spec['selected_factory_sha256']
    for name, digest in spec['runner_files'].items():
        assert sha(Path(spec['helper_root']) / name)==digest,name
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
    assert sha(spec['selected_terminal'])==spec['selected_terminal_sha256']

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
    sys.path.insert(0,spec['helper_root'])
    sys.path.insert(0,str(output.parent))
    sys.path.insert(0,str(model_source))
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    from selected_mask_reference_factory import build_selected_mask_reference_model
    from readback_preflight_checks import observed_readback_forward
    from pvground_semantic_assignment import semantic_assignment_correction
    from pvground_boundary_box_refiner import distribution_loss, BoundaryBoxRefiner
    from mask_reference import install_mask_reference, reference_bounds_witness
    from query_supported_geometry import query_supported_geometry_loss
    from reference_keep import reference_keep_loss
    from check_invalid_reference import check_invalid_reference
    from whole_model_preflight_checks import optimizer_restore_exact
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ('src.joint_det_dataset', 'models.pv_ground', 'models.losses', 'main_utils', 'prepare_data')}
    for name in ('models.pv_ground', 'models.losses', 'main_utils', 'prepare_data'):
        assert model_source in Path(imported[name]).parents
    write_json(output / 'imports.json', dict(files=imported, sha256={k: sha(v) for k, v in imported.items()}))
    cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
    selected_payload=torch.load(spec['selected_terminal'],map_location='cpu')
    assert selected_payload['step']==0 and selected_payload['zero_update_architecture']
    assert selected_payload['reference_mode']=='fused_mask' and not selected_payload['optimizer']['state']
    model,config,load=build_selected_mask_reference_model(cfg,
        torch.load(official_weight['path'],map_location='cpu'),
        torch.load(spec['base_terminal'],map_location='cpu'),selected_payload,manifest['data_root'])
    assert load['full_state_tensors']==1304 and not load['old_geometry_checkpoint_required']
    assert torch.count_nonzero(model.candidate_box_refiner.output.weight)==0
    assert torch.count_nonzero(model.candidate_box_refiner.output.bias)==0
    load.update(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),
        spec_sha256=sha(args.spec),selected_terminal=spec['selected_terminal'],
        selected_terminal_sha256=spec['selected_terminal_sha256'],reference_keep_weight=spec['reference_keep_weight'],
        common_output_reset=spec['common_output_reset'],retained_hidden_state_tensors=8,reset_output_state_tensors=2)
    write_json(output/'load.json',load)
    model.cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry_head=model.candidate_box_refiner
    for parameter in geometry_head.parameters():
        parameter.requires_grad_(True)
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    model.eval()
    load.update(frozen_geometry_provider=False,frozen_parent_and_R=True,geometry_head_only=True,
        fresh_readback_optimizer_required=False,fresh_geometry_optimizer_required=True)
    write_json(output/'load.json',load)
    training = copy.copy(config)
    training.frozen = False
    training.small_lr = False
    training.lr = spec['lr']
    training.lr_backbone = spec['lr']
    assert training.weight_decay == spec['weight_decay'] and training.clip_norm == spec['clip_norm']
    criterion, set_criterion = BaseTrainTester.get_criterion(training)
    processors = {mode: DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), mode == 'train', 6)
                  for mode in ('train', 'eval')}
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_states=10
    expected_parameters=456102
    assert len(trainable)==expected_states and all(name.startswith('candidate_box_refiner.') for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values())==expected_parameters
    core_names=set(initial)-set(trainable)
    formal = args.mode in ('initial_formal','formal')
    assert spec['fit_passes'] == 1 and spec['updates'] == 3723
    assert spec['primary_mode'] == 'bbs' and spec['primary_threshold'] == .5
    verify_scanrefer_superpoints(manifest['data_root'], 'val' if formal else 'train',
                               manifest['superpoint_files']['val' if formal else 'train'])

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

        def _get_target_boxes(self, anno, scan):
            target = anno['target_id'][0] if isinstance(anno['target_id'], list) else anno['target_id']
            corners = scan.get_object_bbox(target).reshape(6).astype(np.float64)
            self.pre_jitter_root = np.concatenate(((corners[:3]+corners[3:])*.5,
                corners[3:]-corners[:3])).astype(np.float32)
            return super()._get_target_boxes(anno, scan)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['pre_jitter_root_box'] = self.pre_jitter_root.copy()
            result['local_training_id'] = self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    os.chdir(str(dataset_source))
    if formal:
        class FormalDataset(Joint3DDataset):
            def _scene_graph_parse(self, annos):
                assert len(annos) == 9508
                for index, row in enumerate(annos):
                    row['_local_training_id'] = index
                super()._scene_graph_parse(annos)

            def __getitem__(self, index):
                result = super().__getitem__(index)
                result['local_training_id'] = self.annos[index]['_local_training_id']
                assert np.isin(result['gt_masks'], [0, 1]).all()
                result['gt_masks'] = result['gt_masks'].astype(np.bool_)
                return result
        dataset = FormalDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='val',
            data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
            detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
            augment_det=False, skip_missing_superpoints=True)
        assert len(dataset) == 9508
        partitions = {'holdout': list(range(9508))}
    else:
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
    evaluator_path=runtime/'PV-Ground/src/grounding_evaluator.py'
    evaluator_spec=importlib.util.spec_from_file_location('pvground_official_evaluator',str(evaluator_path))
    evaluator_module=importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator=evaluator_module.GroundingEvaluator
    assert sha(evaluator_path)==spec['native_evaluator_sha256']
    imported['native_evaluator']=str(evaluator_path)
    write_json(output/'imports.json',dict(files=imported,sha256={key:sha(value) for key,value in imported.items()}))
    from native_root_bbs import native_root_bbs
    selected=set(trainable)
    optimizer=torch.optim.AdamW(tuple(trainable.values()),lr=spec['lr'],weight_decay=spec['weight_decay'])
    if args.mode=='formal':
        terminal=torch.load(str(output/'terminal.pth'),map_location='cpu')
        assert terminal['step']==3723 and terminal['spec_sha256']==sha(args.spec)
        assert terminal['head_only'] and terminal['extra_geometry_weight']==spec['extra_geometry_weight']
        assert terminal['auxiliary_target_mode']==spec['auxiliary_target_mode']
        assert terminal['reference_mode']==spec['reference_mode']
        assert terminal['reference_keep_weight']==spec['reference_keep_weight']
        assert terminal['common_output_reset']==spec['common_output_reset']
        assert terminal['mask_reference_sha256']==spec['mask_reference_sha256']
        for key in ('checkpoint_sha256','base_terminal_sha256','selected_terminal_sha256','source_port_sha256'):
            assert terminal[key]==spec[key]
        assert set(terminal['state_delta'])==selected
        assert Counter(terminal['row_ids'])==Counter(json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']['fit'])
        restored=dict(initial,**terminal['state_delta'])
        model.load_state_dict(restored,strict=True)
        optimizer.load_state_dict(terminal['optimizer'])
        assert all(int(state['step'])==3723 for state in optimizer.state.values())
        assert all(torch.equal(value.detach().cpu(),restored[name]) for name,value in model.state_dict().items())
        write_json(output/'formal_restore.json',dict(status='pass',terminal_sha256=sha(output/'terminal.pth'),
            optimizer=optimizer_restore_exact(optimizer,terminal['optimizer']),strict_model_restore=True))
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

    @torch.no_grad()
    def evaluate(stage):
        dataset.augment = False
        dataset.augment_det = False
        model.eval()
        reset_rng()
        evaluator = GroundingEvaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround')
        directory = output / stage
        directory.mkdir()
        rows = []
        begin = time.time()
        with (directory / 'rows.jsonl').open('w') as stream:
            for batch in loader('holdout', False):
                inputs, batch = prepare(batch, 'eval')
                predictions, call = observed_readback_forward(model, inputs)
                score = native_root_bbs(predictions['last_sem_cls_scores'], batch)
                boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
                reference_boxes=torch.cat([predictions['p3_coarse_center'],predictions['p3_coarse_size'].clamp(min=1e-6)],-1)
                coarse=torch.cat([predictions['native_coarse_center'],predictions['native_coarse_size'].clamp(min=1e-6)],-1)
                if formal:
                    np.savez_compressed(directory/('batch_%05d.npz'%len(rows)),
                        row_ids=batch['local_training_id'].cpu().numpy(),
                        original_prior=coarse.cpu().numpy(),reference=reference_boxes.cpu().numpy(),
                        final=boxes.cpu().numpy(),scores=score.cpu().numpy(),
                        reference_valid=predictions['mask_reference_valid'].cpu().numpy(),
                        root_gt=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1).cpu().numpy())
                assert (boxes[..., 3:] > 0).all()
                _, predictions = native_loss(predictions, batch)
                for key in predictions:
                    if 'pred_size' in key:
                        predictions[key] = predictions[key].clamp(min=1e-6)
                evaluator.evaluate(predictions, 'last_')
                truth = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
                for bid in range(len(batch['utterances'])):
                    row_id = int(batch['local_training_id'][bid])
                    assert row_id == partitions['holdout'][len(rows)]
                    iou = box_iou(boxes[bid], truth[bid])
                    coarse_iou = box_iou(coarse[bid], truth[bid])
                    reference_iou=box_iou(reference_boxes[bid],truth[bid])
                    ranked = score[bid].argsort(descending=True)
                    query = int(ranked[0])
                    alpha = predictions['adaptive_weights'][bid]
                    mask = ((alpha * predictions['last_pred_masks'][bid][0, query]
                        + (1 - alpha) * predictions['sp_last_pred_masks'][bid][query]).sigmoid() > .5)[predictions['superpoints'][bid]]
                    target_mask = batch['gt_masks'][bid, 0].bool()
                    mask_iou = float((mask & target_mask).sum().float() / (mask | target_mask).sum())
                    record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]),
                        root_box=truth[bid].cpu().tolist(),
                        point_sha256=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest(),
                        bbs=dict(query=query, box=boxes[bid, query].cpu().tolist(), iou=float(iou[query]), mask_iou=mask_iou,
                            coarse_box=coarse[bid, query].cpu().tolist(), coarse_iou=float(coarse_iou[query]),
                            reference_box=reference_boxes[bid,query].cpu().tolist(),reference_iou=float(reference_iou[query]),
                            reference_valid=bool(predictions['mask_reference_valid'][bid,query]),
                            oracle25=[int((iou[ranked[:count]] > .25).any()) for count in (16, 32, 64, 256)],
                            oracle50=[int((iou[ranked[:count]] > .5).any()) for count in (16, 32, 64, 256)]),
                        same_forward_geometry_exact=bool(call['fixed_geometry']), native_head_calls=call['final_semantic_head_calls'],
                        diagnostic_native_head_replay_calls=0)
                    rows.append(record)
                    stream.write(json.dumps(record) + '\n')
                if len(rows) % 512 < 8:
                    stream.flush()
                    print('READBACK_EVAL_PROGRESS ' + json.dumps(dict(stage=stage, rows=len(rows),
                        total=len(partitions['holdout']), seconds=time.time() - begin)), flush=True)
                del predictions, inputs, batch
        assert len(rows) == len(partitions['holdout'])
        hits25 = sum(row['bbs']['iou'] > .25 for row in rows)
        hits50 = sum(row['bbs']['iou'] > .5 for row in rows)
        assert hits25 == evaluator.dets[('last_', .25, 1, 'bbs')]
        assert hits50 == evaluator.dets[('last_', .5, 1, 'bbs')]
        mask_sum = sum(row['bbs']['mask_iou'] for row in rows)
        assert abs(mask_sum - float(evaluator.dets['mask_pos'])) < 1e-3
        metric = dict(rec_hits25=hits25, rec_hits50=hits50,
            mask_hits25=sum(row['bbs']['mask_iou'] > .25 for row in rows),
            mask_hits50=sum(row['bbs']['mask_iou'] > .5 for row in rows), mask_miou=mask_sum / len(rows) * 100)
        receipt = dict(status='pass', stage=stage, rows=len(rows), metrics={'bbs': metric},
            primary_mode='bbs',primary_threshold=.5,
            elapsed_seconds=time.time() - begin, time_cst=datetime.datetime.now().astimezone().isoformat(),
            formal_rows=len(rows) if formal else 0, rows_sha256=sha(directory / 'rows.jsonl'))
        write_json(directory / 'receipt.json', receipt)
        print('QUERY_SUPPORTED_GEOMETRY_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)
        return rows, receipt

    def step(batch_cpu, update):
        inputs,batch=prepare(batch_cpu,'train')
        head_inputs=[]
        readback_inputs=[]
        if args.mode=='preflight':
            head_hook=geometry_head.register_forward_pre_hook(lambda module,arguments:head_inputs.append(arguments))
            readback_hook=model.boundary_evidence_readback.register_forward_pre_hook(lambda module,arguments:readback_inputs.append(arguments))
        predictions,call=observed_readback_forward(model,inputs)
        if args.mode=='preflight':
            head_hook.remove();readback_hook.remove()
            assert len(head_inputs)==len(readback_inputs)==1
            before_bbs=native_root_bbs(predictions['last_sem_cls_scores'],batch).detach().clone()
            before_masks=[value.detach().clone() for value in (predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights'])]
        assert torch.equal(predictions['last_semantic_query_before_readback'],predictions['last_semantic_query_after_readback'])
        matches=[]
        hook=set_criterion.matcher.register_forward_hook(
            lambda module,arguments,result:matches.append([(q.clone(),t.clone()) for q,t in result]))
        native,predictions=native_loss(predictions,batch)
        hook.remove()
        assert len(matches)==7
        correction,assignment=semantic_assignment_correction(predictions,batch,matches[1],set_criterion.eos_coef)
        edge,edge_counts=distribution_loss(predictions,batch,matches[1])
        native_roots=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1)
        auxiliary_roots=native_roots
        extra,extra_counts,qualified=query_supported_geometry_loss(predictions,batch,matches[1],set_criterion,auxiliary_roots)
        keep,keep_counts,keep_selected=reference_keep_loss(predictions,batch,matches[1])
        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra+spec['reference_keep_weight']*keep
        assert torch.isfinite(loss)
        witness={}
        if args.mode=='preflight':
            witness.update(reference_bounds_witness(predictions,batch))
            keep_grad=torch.autograd.grad(keep,(predictions['last_center'],predictions['last_pred_size']),retain_graph=True)
            for bid,queries in enumerate(keep_selected):
                outside=torch.ones(256,dtype=torch.bool,device=queries.device);outside[queries]=False
                assert all((gradient[bid,outside]==0).all() for gradient in keep_grad)
            keep_output_grad=torch.autograd.grad(keep,geometry_head.output.weight,retain_graph=True)[0]
            witness.update(reference_keep_gradient_only_qualified=True,
                reference_keep_output_gradient_norm=float(keep_output_grad.norm()))
            if torch.count_nonzero(geometry_head.output.weight)==0 and torch.count_nonzero(geometry_head.output.bias)==0:
                assert torch.equal(predictions['last_center'],predictions['geometry_reference_center'])
                assert torch.equal(predictions['last_pred_size'],predictions['geometry_reference_size'].clamp(min=1e-6))
                witness['neutral_initial_decode_equals_reference']=True
                assert float(keep)==0.0
                witness['neutral_initial_reference_keep_exact_zero']=True
            # Reference changes the actual eligibility set. Keep the existing
            # empty-set loss and inspect its real output gradients without
            # imposing the old frame's nonempty/clipped-target fixture counts.
            output_gradients=torch.autograd.grad(extra+predictions['boundary_logits'].sum()*0,
                (predictions['last_center'],predictions['last_pred_size'],predictions['boundary_logits']),retain_graph=True)
            for bid,queries in enumerate(qualified):
                exclude=torch.ones(256,dtype=torch.bool,device=queries.device)
                exclude[queries]=False
                for gradient in output_gradients:
                    assert (gradient[bid,exclude]==0).all()
            isolated=torch.autograd.grad(extra,tuple(trainable.values()),retain_graph=True)
            witness.update(extra_alone_output_gradient=float(isolated[list(trainable).index('candidate_box_refiner.output.weight')].norm()),
                extra_direct_output_gradients_only_qualified=True)
            assert witness['extra_alone_output_gradient']>=0
        optimizer.zero_grad()
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
        assert all(parameter.grad is None for name,parameter in model.named_parameters() if name not in selected)
        if args.mode=='preflight':
            witness['raw_parameter_gradient_norms']={name:float(parameter.grad.norm()) for name,parameter in trainable.items()}
        norm=torch.nn.utils.clip_grad_norm_(tuple(trainable.values()),spec['clip_norm'])
        assert torch.isfinite(norm)
        if update:
            optimizer.step()
        if args.mode=='preflight':
            with torch.no_grad():
                center,size=geometry_head(*head_inputs[0])
                predictions['last_center']=center
                predictions['last_pred_size']=size
                replay_query=model.boundary_evidence_readback(*readback_inputs[0])
                assert torch.equal(replay_query,predictions['last_semantic_query_before_readback'])
                replay_logits=model.prediction_heads[-1].sem_cls_scores_head(replay_query.transpose(1,2).contiguous()).transpose(2,1)
                assert torch.equal(native_root_bbs(replay_logits,batch),before_bbs)
                current_masks=predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights']
                assert all(torch.equal(value,old) for value,old in zip(current_masks,before_masks))
                witness.update(cached_upstream_after_head_update_native_bbs_exact=True,
                    cached_upstream_after_head_update_masks_exact=True,
                    observed_empty_extra_rows=sum(count==0 for count in extra_counts['extra_row_counts']),
                    observed_clipped_extra_faces=extra_counts['extra_boundary_outside'],diagnostic_native_head_replay_calls=1)
        record=dict(loss=float(loss),native_loss=float(native),G_correction=float(correction),
            matched_boundary_loss=float(edge),extra_geometry_loss=float(extra),
            reference_keep_loss=float(keep),reference_keep_counts=keep_counts,reference_keep_weight=spec['reference_keep_weight'],
            reference_mode=spec['reference_mode'],
            extra_geometry_weight=spec['extra_geometry_weight'],extra_counts=extra_counts,
            auxiliary_target_mode=spec['auxiliary_target_mode'],
            auxiliary_native_max_face_shift=float((torch.cat([native_roots[:,:3]-native_roots[:,3:]/2,native_roots[:,:3]+native_roots[:,3:]/2],-1)
                -torch.cat([batch['pre_jitter_root_box'][:,:3]-batch['pre_jitter_root_box'][:,3:]/2,batch['pre_jitter_root_box'][:,:3]+batch['pre_jitter_root_box'][:,3:]/2],-1)).abs().max()),
            matched_boundary_counts=edge_counts,assignment_counts=assignment,gradient_norm=float(norm),
            native_head_calls=call['final_semantic_head_calls'],rows=batch['local_training_id'].cpu().tolist(),
            zero_R_semantic_exact=True,**witness)
        del predictions,inputs,batch
        return record

    if formal:
        evaluate(args.mode)
        if args.mode=='initial_formal':
            assert not optimizer.state
            torch.save(dict(state_delta={name:value.detach().cpu() for name,value in model.state_dict().items() if name in selected},
                optimizer=optimizer.state_dict(),step=0,zero_update_architecture=True,
                reference_mode=spec['reference_mode'],common_output_reset=spec['common_output_reset'],
                retained_hidden_prior_updates=11169,reset_output_total_updates=0,
                checkpoint_sha256=spec['checkpoint_sha256'],base_terminal_sha256=spec['base_terminal_sha256'],
                selected_terminal_sha256=spec['selected_terminal_sha256'],source_port_sha256=spec['source_port_sha256'],
                mask_reference_sha256=spec['mask_reference_sha256'],spec_sha256=sha(args.spec)),str(output/'initial.pth'))
        return
    dataset.augment=True
    dataset.augment_det=True
    model.eval()
    geometry_head.train()
    if args.mode=='preflight':
        reset_rng()
        for batch_index,batch_cpu in enumerate(loader('fit',True)):
            if batch_index==spec['preflight_batch_index']:
                break
        assert batch_index==spec['preflight_batch_index']
        reference_rows=[json.loads(line) for line in Path(spec['preflight_reference_rows']).read_text().splitlines()]
        for bid,row_id in enumerate(batch_cpu['local_training_id'].tolist()):
            reference=reference_rows[batch_index*8+bid]
            assert row_id==reference['row_id']
            assert hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest()==reference['point_sha256']
            native_root=torch.cat([batch_cpu['center_label'][bid,0,:3],batch_cpu['size_gts'][bid,0]])
            assert native_root.tolist()==reference['noisy_root_box']
            assert batch_cpu['pre_jitter_root_box'][bid].tolist()==reference['pre_jitter_root_box']
        witnesses=[step(batch_cpu,True) for _ in range(2)]
        invalid_fixture=check_invalid_reference(output.parent)
        assert witnesses[0]['neutral_initial_decode_equals_reference']
        assert witnesses[1]['raw_parameter_gradient_norms']['candidate_box_refiner.output.weight']>0
        assert all(value>0 for name,value in witnesses[1]['raw_parameter_gradient_norms'].items()
                   if not name.startswith('candidate_box_refiner.output.'))
        assert all(torch.equal(model.state_dict()[name].detach().cpu(),initial[name]) for name in core_names)
        memory=io.BytesIO()
        torch.save(dict(state_delta={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in selected},
            optimizer=optimizer.state_dict()),memory)
        size=memory.tell()
        memory.seek(0)
        restored=torch.load(memory,map_location='cpu')
        model.load_state_dict(dict(initial,**restored['state_delta']),strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        optimizer_check=optimizer_restore_exact(optimizer,restored['optimizer'])
        assert all(int(state['step'])==2 for state in optimizer.state.values())
        assert all(torch.equal(value.detach().cpu(),dict(initial,**restored['state_delta'])[name]) for name,value in model.state_dict().items())
        receipt=dict(status='pass',optimizer_steps=2,weight_files_created=0,accuracy_result=False,native_data_and_member_target_exact=True,
            head_parameters=expected_parameters,head_state_tensors=expected_states,all_parent_and_R_states_exact=True,
            qualified_extra_output_gradient_scope_verified=True,extra_geometry_weight=spec['extra_geometry_weight'],reference_keep_weight=spec['reference_keep_weight'],
            reference_mode=spec['reference_mode'],common_output_reset=spec['common_output_reset'],invalid_reference_fixture=invalid_fixture,
            optimizer_exact_check=optimizer_check,serialization_bytes=size,witnesses=witnesses,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.perf_counter()-begin,spec_sha256=sha(args.spec),runner_sha256=sha(__file__))
        write_json(output/'preflight.json',receipt)
        print('QUERY_SUPPORTED_GEOMETRY_PREFLIGHT_COMPLETE '+json.dumps(receipt),flush=True)
        return

    initial_rows,initial_receipt=evaluate('initial')
    dataset.augment=True
    dataset.augment_det=True
    model.eval()
    geometry_head.train()
    reset_rng()
    seen=[]
    start=time.time()
    total=math.ceil(len(partitions['fit'])/8)
    assert total==3723

    def save_checkpoint(name,step_number):
        payload=dict(state_delta={name:value.detach().cpu() for name,value in model.state_dict().items() if name in selected},
            optimizer=optimizer.state_dict(),step=step_number,row_ids=seen,head_only=True,
            boundary_mode='distribution',support_arm='whole_range',use_whole_range=True,boundary_loss_weight=1.0/7,
            retained_hidden_prior_updates=11169,retained_hidden_total_updates=11169+step_number,
            reset_output_total_updates=step_number,common_output_reset=spec['common_output_reset'],
            reference_mode=spec['reference_mode'],mask_reference_sha256=spec['mask_reference_sha256'],
            auxiliary_target_mode=spec['auxiliary_target_mode'],
            checkpoint_sha256=spec['checkpoint_sha256'],base_terminal_sha256=spec['base_terminal_sha256'],
            selected_terminal_sha256=spec['selected_terminal_sha256'],source_port_sha256=spec['source_port_sha256'],
            spec_sha256=sha(args.spec),extra_geometry_weight=spec['extra_geometry_weight'],reference_keep_weight=spec['reference_keep_weight'],
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate())
        temporary=output/(name+'.tmp')
        torch.save(payload,str(temporary))
        os.replace(str(temporary),str(output/name))

    with (output/'train.jsonl').open('w') as stream:
        for index,batch in enumerate(loader('fit',True),1):
            step_begin=time.time()
            record=step(batch,True)
            assert len(record['rows'])==(2 if index==total else 8)
            seen.extend(record['rows'])
            record.update(step=index,total_steps=total,seconds=time.time()-step_begin,cumulative_seconds=time.time()-start)
            stream.write(json.dumps(record)+'\n')
            if index==1 or index%64==0:
                stream.flush()
                print('QUERY_SUPPORTED_GEOMETRY_PROGRESS '+json.dumps(record),flush=True)
            if index%512==0:
                save_checkpoint('latest.pth',index)
    assert index==3723 and Counter(seen)==Counter(partitions['fit'])
    assert all(torch.equal(model.state_dict()[name].detach().cpu(),initial[name]) for name in core_names)
    save_checkpoint('latest.pth',index)
    os.replace(str(output/'latest.pth'),str(output/'terminal.pth'))
    final_rows,final_receipt=evaluate('terminal')
    transitions={}
    for threshold in (.25,.5):
        repairs=damages=0
        for old,new in zip(initial_rows,final_rows):
            assert old['row_id']==new['row_id'] and old['point_sha256']==new['point_sha256'] and old['root_box']==new['root_box']
            repairs+=old['bbs']['iou']<=threshold<new['bbs']['iou']
            damages+=new['bbs']['iou']<=threshold<old['bbs']['iou']
        transitions[str(threshold)]=dict(repairs=repairs,damages=damages,net=repairs-damages)
    receipt=dict(status='complete',training_steps=index,fit_rows=len(seen),holdout_rows=len(final_rows),formal_rows=0,
        initial=initial_receipt['metrics'],terminal=final_receipt['metrics'],transitions=transitions,
        physical_batch=8,effective_batch=8,accumulation=1,last_batch_rows=2,fit_seen_exactly_once=True,
        parent_and_zero_R_states_exact=True,head_parameters=expected_parameters,head_state_tensors=expected_states,fresh_optimizer=True,
        extra_geometry_weight=spec['extra_geometry_weight'],reference_keep_weight=spec['reference_keep_weight'],primary_mode='bbs',primary_threshold=.5,
        terminal_sha256=sha(output/'terminal.pth'),train_log_sha256=sha(output/'train.jsonl'),spec_sha256=sha(args.spec))
    write_json(output/'receipt.json',receipt)
    print('QUERY_SUPPORTED_GEOMETRY_FIT_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
