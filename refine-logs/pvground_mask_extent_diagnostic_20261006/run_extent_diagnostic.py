"""Protected4511 continuation; fixed versus own/fused-support reference."""
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
    parser.add_argument('--mode', choices=['preflight', 'formal'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root']) / args.mode
    output.mkdir()
    runtime = Path(spec['runtime'])
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['geometry_hits50']==4511 and spec['head_only']
    assert spec['use_geometry_evidence'] is True and spec['extra_geometry_weight']==1.0
    assert spec['auxiliary_target_mode']=='native_gt'
    assert isinstance(spec['reference_enabled'],bool)
    assert spec['reference_loss_weight']==(1.0 if spec['reference_enabled'] else 0.0)
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
    sys.path.insert(0,spec['helper_root'])
    sys.path.insert(0,str(output.parent))
    sys.path.insert(0,str(model_source))
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    from readback_model_factory import build_readback_model
    from readback_preflight_checks import observed_readback_forward
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
    assert not spec['reference_enabled']
    model.cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry_head=model.candidate_box_refiner
    assert not any(parameter.requires_grad for parameter in model.parameters())
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    model.eval()
    load.update(frozen_geometry_provider=True,frozen_parent_and_R=True,geometry_head_only=False,
        fresh_readback_optimizer_required=False,fresh_geometry_optimizer_required=False,read_only_diagnostic=True,optimizer_created=False)
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
    verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['superpoint_files']['val'])
    os.chdir(str(dataset_source))
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

    from extent_evidence import member_geometry, exact_box, quantile_box, box_iou as cpu_iou, QUANTILE
    from native_root_bbs import native_root_bbs
    assert spec['read_only_diagnostic'] and spec['quantile'] == QUANTILE
    assert sha(Path(spec['root']) / 'extent_evidence.py') == spec['extent_evidence_sha256']
    evaluator_path = runtime / 'PV-Ground/src/grounding_evaluator.py'
    assert sha(evaluator_path) == spec['native_evaluator_sha256']
    evaluator_spec = importlib.util.spec_from_file_location('pvground_official_evaluator', str(evaluator_path))
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    evaluator = evaluator_module.GroundingEvaluator(only_root=True, thresholds=[.25, .5],
        topks=[1, 5, 10], prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround')
    assert sha(spec['parent_formal_rows']) == spec['parent_formal_rows_sha256']
    parent_rows = [json.loads(line) for line in Path(spec['parent_formal_rows']).read_text().splitlines()]
    assert len(parent_rows) == 9508
    length = 8 if args.mode == 'preflight' else 9508
    dataset.augment = False
    dataset.augment_det = False
    model.eval()
    reset_rng()
    loader = DataLoader(Subset(dataset, list(range(length))), batch_size=8, shuffle=False,
        num_workers=2, generator=torch.Generator().manual_seed(spec['seed']), pin_memory=True, drop_last=False)
    rows = []
    preflight_formal_batch_bytes = None
    inference_start = time.perf_counter()
    with torch.no_grad(), (output / 'rows.jsonl').open('w') as stream:
        for batch_index, batch_cpu in enumerate(loader):
            inputs, batch = prepare(batch_cpu, 'eval')
            predictions, calls = observed_readback_forward(model, inputs)
            assert torch.equal(predictions['last_semantic_query_before_readback'],
                               predictions['last_semantic_query_after_readback'])
            scores = native_root_bbs(predictions['last_sem_cls_scores'], batch)
            boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
            assert (boxes[..., 3:] > 0).all()
            _, predictions = native_loss(predictions, batch)
            for key in predictions:
                if 'pred_size' in key:
                    predictions[key] = predictions[key].clamp(min=1e-6)
            evaluator.evaluate(predictions, 'last_')
            truths = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
            evidence = {}
            for bid in range(len(batch['utterances'])):
                row_id = int(batch['local_training_id'][bid])
                assert row_id == len(rows)
                query = int(scores[bid].argsort(descending=True)[0])
                points = batch['point_clouds'][bid].cpu().numpy()
                point_sha = hashlib.sha256(points.tobytes()).hexdigest()
                root_box = truths[bid].cpu().numpy()
                historic = parent_rows[row_id]
                assert historic['row_id'] == row_id and historic['scan_id'] == batch['scan_ids'][bid]
                assert historic['target_id'] == int(batch['target_id'][bid])
                assert historic['point_sha256'] == point_sha
                assert np.array_equal(np.asarray(historic['root_box']), root_box)
                superpoint = predictions['superpoints'][bid].cpu().numpy()
                target = batch['gt_masks'][bid, 0].bool().cpu().numpy()
                ids, inverse, count, low, high, target_count = member_geometry(points[:, :3], superpoint, target)
                text = predictions['last_pred_masks'][bid][0, query]
                own = predictions['sp_last_pred_masks'][bid][query]
                alpha = predictions['adaptive_weights'][bid]
                fused = alpha * text + (1 - alpha) * own
                assert text.shape == own.shape == fused.shape
                own_active = (own.sigmoid() > .5).cpu().numpy()[ids]
                active = (fused.sigmoid() > .5).cpu().numpy()[ids]
                point_active = active[inverse]
                exact = exact_box(active, low, high)
                trimmed, samples, ranks = quantile_box(points[:, :3], point_active)
                intersection = int(target_count[active].sum())
                foreground_count = int(count[active].sum())
                assert foreground_count == int(point_active.sum())
                mask_iou = intersection / (foreground_count + int(target.sum()) - intersection)
                own_intersection = int(target_count[own_active].sum())
                own_count = int(count[own_active].sum())
                own_iou = own_intersection / (own_count + int(target.sum()) - own_intersection)
                learned = boxes[bid, query].cpu().numpy()
                native_iou = float(box_iou(boxes[bid], truths[bid])[query])
                record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]),
                    query=query, root_box=root_box.tolist(), point_sha256=point_sha,
                    learned_box=learned.tolist(), learned_iou=native_iou,
                    exact_box=None if exact is None else exact.tolist(), exact_iou=cpu_iou(exact, root_box),
                    quantile_box=None if trimmed is None else trimmed.tolist(), quantile_iou=cpu_iou(trimmed, root_box),
                    foreground_count=foreground_count, fused_mask_iou=mask_iou, own_mask_iou=own_iou,
                    historic_query=historic['bbs']['query'], historic_learned_iou=historic['bbs']['iou'],
                    native_head_calls=calls['final_semantic_head_calls'], diagnostic_head_replay_calls=0,
                    same_forward_geometry_exact=calls['fixed_geometry'])
                rows.append(record)
                stream.write(json.dumps(record) + '\n')
                key = 'r' + str(row_id) + '__'
                arrays = dict(ids=ids, count=count, lower=low.astype(points.dtype), upper=high.astype(points.dtype), target_count=target_count,
                    own_active=own_active, fused_active=active, own_logits=own.cpu().numpy()[ids],
                    text_logits=text.cpu().numpy()[ids], fused_logits=fused.cpu().numpy()[ids],
                    quantile_order_values=samples, quantile_order_ranks=ranks,
                    alpha=alpha.cpu().numpy())
                if args.mode == 'preflight':
                    arrays.update(point_clouds=points, superpoint=superpoint, gt_point_mask=target)
                evidence.update({key + name: value for name, value in arrays.items()})
            np.savez_compressed(str(output / ('batch_%04d.npz' % batch_index)), **evidence)
            if args.mode == 'preflight':
                formal_payload = {name:value for name,value in evidence.items()
                    if not name.endswith(('__point_clouds','__superpoint','__gt_point_mask'))}
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **formal_payload)
                preflight_formal_batch_bytes = buffer.tell()
            if len(rows) % 512 < 8:
                stream.flush()
                print('EXTENT_PROGRESS ' + json.dumps(dict(rows=len(rows), total=length,
                    seconds=time.perf_counter()-inference_start)), flush=True)
            del inputs, batch, predictions, evidence
    assert len(rows) == length
    counts = {name: {str(t): sum(row[name + '_iou'] > t for row in rows) for t in (.25, .5)}
              for name in ('learned', 'exact', 'quantile')}
    for threshold in (.25, .5):
        assert counts['learned'][str(threshold)] == evaluator.dets[('last_', threshold, 1, 'bbs')]
    assert abs(sum(row['fused_mask_iou'] for row in rows) - float(evaluator.dets['mask_pos'])) < 1e-3
    assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
    receipt = dict(status='pass', mode=args.mode, rows=length, hits=counts,
        inference_seconds=time.perf_counter()-inference_start, total_seconds=time.perf_counter()-begin,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        preflight_formal_batch_bytes=preflight_formal_batch_bytes,
        evidence_bytes_written=sum(path.stat().st_size for path in output.glob('batch_*.npz')),
        model_states_unchanged=True, optimizer_created=False, optimizer_updates=0, new_weights=0,
        all256_candidates_retained=True, native_head_calls_per_batch=1, scoring_or_box_outputs_overwritten=False,
        native_mask_hits50=sum(row['fused_mask_iou'] > .5 for row in rows),
        history_query_changes=sum(row['query'] != row['historic_query'] for row in rows),
        history_hit50_changes=sum((row['learned_iou'] > .5) != (row['historic_learned_iou'] > .5) for row in rows),
        invalid_exact=sum(row['exact_box'] is None for row in rows),
        invalid_quantile=sum(row['quantile_box'] is None for row in rows),
        quantile=QUANTILE, quantile_precision='NumPy float64 linear interpolation from actual float32 XYZ',
        geometry_terminal=spec['geometry_terminal'], geometry_terminal_sha256=spec['geometry_terminal_sha256'],
        spec_sha256=sha(args.spec), rows_sha256=sha(output / 'rows.jsonl'),
        time_cst=datetime.datetime.now().astimezone().isoformat())
    write_json(output / 'receipt.json', receipt)
    print('EXTENT_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
