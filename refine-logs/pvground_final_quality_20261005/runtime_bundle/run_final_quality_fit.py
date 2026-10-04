"""Frozen4506 R fit with training-only native final-IoU differences.

Implementation draft; source review and two-step preflight precede launch.
"""
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
    parser.add_argument('--mode', choices=['preflight', 'train', 'formal'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    runtime = Path(spec['runtime'])
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['geometry_hits50'] == 4506 and spec['geometry_frozen']
    assert spec['use_geometry_evidence'] is True and spec['quality_weight'] == 1.0
    for name, digest in spec['runner_files'].items():
        assert sha(output / name) == digest, name
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
    sys.path.insert(0, str(model_source))
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    from readback_model_factory import build_readback_model
    from readback_preflight_checks import (observed_readback_forward, repeated_forward_differences,
        zero_readback_cached_native_head, native_bbs_witness, readback_semantic_route)
    from pvground_semantic_assignment import semantic_assignment_correction
    from native_final_quality import native_final_quality_loss, verify_quality_loss
    from whole_model_preflight_checks import optimizer_restore_exact
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
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert len(trainable) == 23 and all(name.startswith('boundary_evidence_readback.') for name in trainable)
    core_names = set(initial)
    formal = args.mode == 'formal'
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

        def __getitem__(self, index):
            result = super().__getitem__(index)
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
    evaluator_path = runtime / 'PV-Ground/src/grounding_evaluator.py'
    evaluator_spec = importlib.util.spec_from_file_location('pvground_official_evaluator', str(evaluator_path))
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator = evaluator_module.GroundingEvaluator
    imported['evaluator'] = str(evaluator_path)
    write_json(output / 'imports.json', dict(files=imported, sha256={key: sha(path) for key, path in imported.items()}))
    from native_root_bbs import native_root_bbs
    selected = set(trainable)
    optimizer = torch.optim.AdamW(tuple(trainable.values()), lr=spec['lr'], weight_decay=spec['weight_decay'])
    if formal:
        terminal = torch.load(str(output / 'terminal.pth'), map_location='cpu')
        assert terminal['step'] == 3723 and terminal['spec_sha256'] == sha(args.spec)
        assert terminal['checkpoint_sha256'] == spec['checkpoint_sha256']
        assert terminal['base_terminal_sha256'] == spec['base_terminal_sha256']
        assert terminal['geometry_terminal_sha256'] == spec['geometry_terminal_sha256']
        assert terminal['source_port_sha256'] == spec['source_port_sha256']
        assert terminal['use_geometry_evidence'] == spec['use_geometry_evidence']
        assert terminal['quality_weight'] == spec['quality_weight']
        assert terminal['readback_only'] and set(terminal['state_delta']) == selected
        assert Counter(terminal['row_ids']) == Counter(json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']['fit'])
        restored = dict(initial)
        restored.update(terminal['state_delta'])
        model.load_state_dict(restored, strict=True)
        optimizer.load_state_dict(terminal['optimizer'])
        assert all(int(state['step']) == 3723 for state in optimizer.state.values())
        restore_check = optimizer_restore_exact(optimizer, terminal['optimizer'])
        assert all(torch.equal(value.detach().cpu(), restored[name]) for name, value in model.state_dict().items())
        write_json(output / 'formal_restore.json', dict(status='pass', terminal_sha256=sha(output / 'terminal.pth'),
            strict_model_restore=True, optimizer=restore_check, parents_protected=True, restored_steps=3723))

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
                # This is an explicitly separate fixed-Query diagnostic, not deployment.
                bypass_logits = model.prediction_heads[-1].sem_cls_scores_head(
                    predictions['last_semantic_query_before_readback'].transpose(1, 2).contiguous()).transpose(2, 1)
                bypass_score = native_root_bbs(bypass_logits, batch)
                boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
                coarse = torch.cat([predictions['p3_coarse_center'],
                                    predictions['p3_coarse_size'].clamp(min=1e-6)], -1)
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
                    ranked = score[bid].argsort(descending=True)
                    query = int(ranked[0])
                    bypass_query = int(bypass_score[bid].argsort(descending=True)[0])
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
                            oracle25=[int((iou[ranked[:count]] > .25).any()) for count in (16, 32, 64, 256)],
                            oracle50=[int((iou[ranked[:count]] > .5).any()) for count in (16, 32, 64, 256)]),
                        bypass_fixed_frame=dict(query=bypass_query, box=boxes[bid, bypass_query].cpu().tolist(),
                            iou=float(iou[bypass_query])),
                        same_forward_geometry_exact=bool(call['fixed_geometry']), native_head_calls=call['final_semantic_head_calls'],
                        diagnostic_native_head_replay_calls=1)
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
        direct = {}
        for threshold in (.25, .5):
            fixes = sum(row['bypass_fixed_frame']['iou'] <= threshold < row['bbs']['iou'] for row in rows)
            damages = sum(row['bbs']['iou'] <= threshold < row['bypass_fixed_frame']['iou'] for row in rows)
            direct[str(threshold)] = dict(fixes=fixes, damages=damages, net=fixes - damages)
        receipt = dict(status='pass', stage=stage, rows=len(rows), metrics={'bbs': metric},
            fixed_frame_readback_effect=direct, primary_mode='bbs', primary_threshold=.5,
            elapsed_seconds=time.time() - begin, time_cst=datetime.datetime.now().astimezone().isoformat(),
            formal_rows=len(rows) if formal else 0, rows_sha256=sha(directory / 'rows.jsonl'))
        write_json(directory / 'receipt.json', receipt)
        print('READBACK_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)
        return rows, receipt

    if formal:
        evaluate('formal')
        return
    if args.mode == 'preflight':
        dataset.augment = True
        dataset.augment_det = True
        reset_rng()
        batch_cpu = next(iter(loader('fit', True)))
        assert len(batch_cpu['utterances']) == 8
        model.eval()
        readback.train()
        witnesses = []
        for previous_updates in (0, 1):
            inputs, batch = prepare(batch_cpu, 'train')
            predictions, call = observed_readback_forward(model, inputs)
            zero = zero_readback_cached_native_head(model, predictions) if previous_updates == 0 else None
            matching = []
            hook = set_criterion.matcher.register_forward_hook(
                lambda module, arguments, result: matching.append([(q.clone(), t.clone()) for q, t in result]))
            native, predictions = native_loss(predictions, batch)
            hook.remove()
            assert len(matching) == 7
            correction, assignment = semantic_assignment_correction(predictions, batch, matching[1], set_criterion.eos_coef)
            quality, counts = native_final_quality_loss(predictions, batch, matching[1])
            route = verify_quality_loss(predictions, batch, matching[1], readback, previous_updates)
            semantic = readback_semantic_route(predictions, correction, readback, previous_updates)
            scoring = native_bbs_witness(predictions['last_sem_cls_scores'], batch)
            loss = native + correction + spec['quality_weight'] * quality
            optimizer.zero_grad()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
            assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in selected)
            norm = torch.nn.utils.clip_grad_norm_(tuple(trainable.values()), spec['clip_norm'])
            optimizer.step()
            witnesses.append(dict(previous_updates=previous_updates, loss=float(loss), quality=float(quality),
                counts=counts, quality_route=route, semantic_route=semantic, native_score=scoring,
                same_frame_call=call, zero_output=zero, gradient_norm=float(norm)))
            del inputs, predictions, batch
        assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)
        delta = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in selected}
        memory = io.BytesIO()
        torch.save(dict(state_delta=delta, optimizer=optimizer.state_dict()), memory)
        serialization_bytes = memory.tell()
        memory.seek(0)
        restored = torch.load(memory, map_location='cpu')
        with torch.no_grad():
            next(iter(trainable.values())).add_(1)
        model.load_state_dict(dict(initial, **restored['state_delta']), strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        optimizer_check = optimizer_restore_exact(optimizer, restored['optimizer'])
        assert all(int(state['step']) == 2 for state in optimizer.state.values())
        assert all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in restored['state_delta'].items())
        assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)
        receipt = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
            optimizer_steps=2, batch_size=8, accuracy_result=False, weight_files_created=0,
            readback_parameters=load['readback_parameters'], readback_state_tensors=23,
            geometry_provider_and_g_states_exact=True, quality_weight=spec['quality_weight'],
            initial_zero_output_native_exact=True, isolated_quality_route_verified=True,
            optimizer_exact_check=optimizer_check, serialization_bytes=serialization_bytes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), witnesses=witnesses,
            spec_sha256=sha(args.spec), runner_sha256=sha(__file__))
        write_json(output / 'preflight.json', receipt)
        print('FINAL_QUALITY_PREFLIGHT_COMPLETE ' + json.dumps(receipt), flush=True)
        return

    initial_rows, initial_receipt = evaluate('initial')
    reset_rng()
    dataset.augment = True
    dataset.augment_det = True
    model.eval()
    readback.train()
    seen = []
    begin = time.time()
    total = math.ceil(len(partitions['fit']) / spec['batch_size'])
    assert total == 3723

    def save_checkpoint(name, step):
        data = dict(state_delta={key: value.detach().cpu() for key, value in model.state_dict().items() if key in selected},
            optimizer=optimizer.state_dict(), step=step, row_ids=seen,
            checkpoint_sha256=spec['checkpoint_sha256'], base_terminal_sha256=spec['base_terminal_sha256'],
            geometry_terminal_sha256=spec['geometry_terminal_sha256'], source_port_sha256=spec['source_port_sha256'],
            spec_sha256=sha(args.spec), use_geometry_evidence=spec['use_geometry_evidence'], readback_only=True,
            quality_weight=spec['quality_weight'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        assert set(data['state_delta']) == selected
        temporary = output / (name + '.tmp')
        torch.save(data, str(temporary))
        os.replace(str(temporary), str(output / name))

    with (output / 'train.jsonl').open('w') as stream:
        for index, batch in enumerate(loader('fit', True), 1):
            step_begin = time.time()
            inputs, batch = prepare(batch, 'train')
            predictions, call = observed_readback_forward(model, inputs)
            assert not predictions['last_center'].requires_grad and not predictions['last_pred_size'].requires_grad
            matching = []
            hook = set_criterion.matcher.register_forward_hook(
                lambda module, arguments, result: matching.append([(q.clone(), t.clone()) for q, t in result]))
            native, predictions = native_loss(predictions, batch)
            hook.remove()
            assert len(matching) == 7
            correction, counts = semantic_assignment_correction(predictions, batch, matching[1], set_criterion.eos_coef)
            quality, quality_counts = native_final_quality_loss(predictions, batch, matching[1])
            loss = native + correction + spec['quality_weight'] * quality
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
            assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in selected)
            norm = torch.nn.utils.clip_grad_norm_(tuple(trainable.values()), spec['clip_norm'])
            assert torch.isfinite(norm)
            optimizer.step()
            torch.cuda.synchronize()
            ids = batch['local_training_id'].cpu().tolist()
            assert len(ids) == (2 if index == total else 8)
            seen.extend(ids)
            record = dict(step=index, total_steps=total, rows=ids, loss=float(loss), native_loss=float(native),
                assignment_correction=float(correction), assignment_counts=counts, gradient_norm=float(norm),
                quality_loss=float(quality), quality_weight=spec['quality_weight'], quality_counts=quality_counts,
                call_order=call['order'], native_head_calls=call['final_semantic_head_calls'],
                geometry_and_mask_same_frame_exact=True, seconds=time.time() - step_begin,
                cumulative_seconds=time.time() - begin)
            stream.write(json.dumps(record) + '\n')
            if index == 1 or index % 64 == 0:
                stream.flush()
                print('READBACK_TRAIN_PROGRESS ' + json.dumps(record), flush=True)
            if index % 512 == 0:
                save_checkpoint('latest.pth', index)
            del predictions, inputs, batch
    assert index == 3723 and Counter(seen) == Counter(partitions['fit'])
    assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)
    save_checkpoint('latest.pth', index)
    os.replace(str(output / 'latest.pth'), str(output / 'terminal.pth'))
    final_rows, final_receipt = evaluate('terminal')
    transitions = {}
    for threshold in (.25, .5):
        fixes = damages = 0
        for before, after in zip(initial_rows, final_rows):
            assert before['row_id'] == after['row_id'] and before['point_sha256'] == after['point_sha256']
            assert before['root_box'] == after['root_box']
            old = before['bbs']['iou'] > threshold
            new = after['bbs']['iou'] > threshold
            fixes += not old and new
            damages += old and not new
        transitions[str(threshold)] = dict(fixes=fixes, damages=damages, net=fixes - damages)
    receipt = dict(status='complete', time_cst=datetime.datetime.now().astimezone().isoformat(),
        training_steps=index, fit_rows=len(seen), holdout_rows=len(final_rows), formal_rows=0,
        initial=initial_receipt['metrics'], terminal=final_receipt['metrics'], transitions=transitions,
        primary_mode='bbs', primary_threshold=.5, frozen_parent_states_exact=True,
        readback_parameters=load['readback_parameters'], readback_state_tensors=23,
        fit_seen_exactly_once=True, physical_batch=8, effective_batch=8, last_batch_rows=2,
        fresh_optimizer=True, use_geometry_evidence=spec['use_geometry_evidence'], quality_weight=spec['quality_weight'],
        control_reused=spec['control_root'], quality_target='detached final IoU differences inside original G root pool',
        terminal_sha256=sha(output / 'terminal.pth'), train_log_sha256=sha(output / 'train.jsonl'),
        spec_sha256=sha(args.spec), source_port_sha256=spec['source_port_sha256'], script_sha256=sha(__file__),
        protected_geometry_parent_sha256=spec['geometry_terminal_sha256'])
    write_json(output / 'receipt.json', receipt)
    print('READBACK_FIT_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
