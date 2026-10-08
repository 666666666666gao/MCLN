"""Paired support-content learning on one frozen PV parent, native bbs only."""
import argparse
from collections import Counter
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


ARMS = ('shared_text', 'candidate_fused')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = Path(spec['root'])
    output = root / 'campaign'
    assert not output.exists()
    assert spec['seed'] == 2027 and spec['batch_size'] == 8
    assert spec['updates'] == 3723 and spec['fit_rows'] == 29778
    assert spec['starting_hits'] == [5599, 4859] and spec['target_hits'] == [5658, 4850]
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['arms'] == list(ARMS) and spec['accumulation'] == 1
    for name, digest in spec['files'].items():
        assert sha(root / name) == digest, name
    sanity = json.loads((root / 'preflight.json').read_bytes())
    assert sha(root / 'preflight.json') == spec['preflight_sha256']
    assert sanity['status'] == 'PASS_ACTUAL_B8_TWO_STEP_SUPPORT_IDENTITY_PREFLIGHT'
    assert sanity['optimizer_steps_per_arm'] == 2 and sanity['weight_files_created'] == 0
    assert sanity['parent_state_unchanged'] and not sanity['formal_accuracy_result']
    output.mkdir()
    parent_spec = json.loads(Path(spec['parent_spec']).read_bytes())
    assert sha(spec['parent_spec']) == spec['parent_spec_sha256']
    env = json.loads((Path(parent_spec['runtime']) / 'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == parent_spec['env_spec_sha256']
    manifest = json.loads(Path(parent_spec['input_manifest']).read_bytes())
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    source = Path(manifest['model_source'])
    model_source = Path(parent_spec['model_source'])
    assert sha(source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source / name) == digest, name
    assert sha(parent_spec['source_port']) == parent_spec['source_port_sha256']
    for name, digest in json.loads(Path(parent_spec['source_port']).read_bytes())['files'].items():
        assert sha(model_source / name) == digest, name
    for name, digest in parent_spec['runner_files'].items():
        assert sha(Path(parent_spec['helper_root']) / name) == digest, name
    for name in ('mask_support_model_factory.py', 'selected_mask_reference_factory.py', 'mask_support_corrector.py', 'mask_reference.py'):
        assert sha(Path(parent_spec['root']) / name) == parent_spec['new_runner_files'][name]
    official = env['weight_dirs']['scanrefer']['path']
    parents = {official: parent_spec['checkpoint_sha256'], parent_spec['base_terminal']: parent_spec['base_terminal_sha256'], parent_spec['selected_terminal']: parent_spec['selected_terminal_sha256'], spec['support_terminal']: spec['support_terminal_sha256']}
    assert all(sha(path) == digest for path, digest in parents.items())

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    def reset_rng():
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == source / 'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    for path in (parent_spec['helper_root'], parent_spec['root'], str(root), str(model_source)):
        sys.path.insert(0, path)
    from pcdet.config import cfg, cfg_from_yaml_file
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from mask_support_model_factory import build_support_model
    from support_identity_readout import SupportIdentityReadout
    from readback_preflight_checks import observed_readback_forward
    from native_root_bbs import native_root_bbs
    from pvground_semantic_assignment import semantic_assignment_correction
    for name in ('models.pv_ground', 'models.losses', 'main_utils', 'prepare_data'):
        assert model_source in Path(sys.modules[name].__file__).resolve().parents
    evaluator_path = Path(parent_spec['runtime']) / 'PV-Ground/src/grounding_evaluator.py'
    assert sha(evaluator_path) == parent_spec['native_evaluator_sha256']
    evaluator_spec = importlib.util.spec_from_file_location('pvground_native_evaluator', str(evaluator_path))
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator = evaluator_module.GroundingEvaluator
    cfg_from_yaml_file(str(Path(parent_spec['runtime']) / 'PV-Ground/wandb_config.yaml'), cfg)
    official_payload = torch.load(official, map_location='cpu')
    g_payload = torch.load(parent_spec['base_terminal'], map_location='cpu')
    reference_payload = torch.load(parent_spec['selected_terminal'], map_location='cpu')
    support_payload = torch.load(spec['support_terminal'], map_location='cpu')
    model, config, load = build_support_model(cfg, official_payload, g_payload, reference_payload, manifest['data_root'], support_payload)
    assert load['full_state_tensors'] == 1314 and support_payload['arm'] == 'content'
    assert support_payload['total_support_updates'] == 7446
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.cuda().eval()
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    heads = {ARMS[0]: SupportIdentityReadout(False).cuda()}
    heads[ARMS[1]] = copy.deepcopy(heads[ARMS[0]])
    heads[ARMS[1]].candidate_specific = True
    assert all(torch.equal(value, heads[ARMS[1]].state_dict()[name]) for name, value in heads[ARMS[0]].state_dict().items())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(heads[ARMS[0]].parameters(), heads[ARMS[1]].parameters()))
    optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay']) for arm, head in heads.items()}
    assert all(not value.state for value in optimizers.values())
    assert all(torch.count_nonzero(head.output.weight) == 0 and torch.count_nonzero(head.output.bias) == 0 for head in heads.values())
    training = copy.copy(config)
    training.frozen = False
    training.small_lr = False
    criterion, set_criterion = BaseTrainTester.get_criterion(training)
    processors = {mode: DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), mode == 'train', 6) for mode in ('train', 'eval')}
    steps = 0
    seen = []
    started = time.monotonic()
    progress = dict(status='running', mode='initial_formal', optimizer_steps_per_arm=0, fit_rows=0, started_cst=datetime.datetime.now().astimezone().isoformat(), parent_frozen=True, preflight_state_carried=False, seed=2027)

    def status(mode, **values):
        progress.update(mode=mode, **values)
        write_json(output / 'status.json', progress)

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_identity_training_id'] = index
                key = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(key).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_identity_training_id']
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    class FormalDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 9508
            for index, row in enumerate(annos):
                row['_identity_validation_id'] = index
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_identity_validation_id']
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    def dataset_for(split):
        verify_scanrefer_superpoints(manifest['data_root'], split, manifest['superpoint_files'][split])
        os.chdir(str(source))
        cls = FormalDataset if split == 'val' else FitDataset
        dataset = cls(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split=split, data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False, detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False, augment_det=False, skip_missing_superpoints=True)
        dataset.augment = False
        dataset.augment_det = False
        os.chdir(str(model_source))
        return dataset

    def loader(dataset, ids, shuffle):
        return DataLoader(Subset(dataset, ids), batch_size=8, shuffle=shuffle, num_workers=2, pin_memory=True, drop_last=False, generator=torch.Generator().manual_seed(2027))

    def prepare(raw, mode):
        processor = processors[mode]
        voxel = processor.collate_batch([processor.forward(dict(points=point.numpy().copy(), use_lead_xyz=True)) for point in raw['point_clouds']])
        size = len(raw['utterances'])
        assert np.array_equal(voxel['points'][:, 1:].reshape(size, 50000, 6), raw['point_clouds'].numpy())
        batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in raw.items()}
        inputs = {key: torch.from_numpy(voxel[key]).float().cuda() for key in ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
        inputs.update(batch_size=size, text=batch['utterances'], superpoint=batch['superpoint'], train=False, det_boxes=batch['all_detected_boxes'], det_bbox_label_mask=batch['all_detected_bbox_label_mask'], det_class_ids=batch['all_detected_class_ids'])
        return inputs, batch

    def pair(parent, raw_points):
        result = {}
        q = parent['last_semantic_query_before_readback'].detach()
        assert torch.equal(q, parent['last_semantic_query_after_readback'])
        for arm, head in heads.items():
            predictions = dict(parent)
            after = head(q, None, None, raw_points, predictions)
            logits = model.prediction_heads[-1].sem_cls_scores_head(after.transpose(1, 2).contiguous()).transpose(2, 1)
            if steps == 0:
                assert torch.equal(after, q) and torch.equal(logits, parent['last_sem_cls_scores'])
            predictions['last_sem_cls_scores'] = logits
            for key in ('last_center', 'last_pred_size', 'last_proj_queries'):
                assert torch.equal(predictions[key], parent[key])
            assert all(predictions[key] is parent[key] for key in ('last_pred_masks', 'sp_last_pred_masks', 'adaptive_weights'))
            result[arm] = predictions
        return result

    def iou(boxes, truth):
        lo = torch.maximum(boxes[..., :3] - boxes[..., 3:] / 2, truth[:3] - truth[3:] / 2)
        hi = torch.minimum(boxes[..., :3] + boxes[..., 3:] / 2, truth[:3] + truth[3:] / 2)
        intersection = (hi - lo).clamp(min=0).prod(-1)
        value = intersection / (boxes[..., 3:].prod(-1) + truth[3:].prod() - intersection)
        assert torch.isfinite(value).all()
        return value

    def evaluate(dataset, ids, stage):
        reset_rng()
        directory = output / stage
        directory.mkdir()
        evaluators = {name: GroundingEvaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10], prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround') for name in ('parent',) + ARMS}
        n = len(ids)
        boxes_array = np.lib.format.open_memmap(str(directory / 'boxes.npy'), mode='w+', dtype=np.float32, shape=(n, 256, 6))
        score_arrays = {name: np.lib.format.open_memmap(str(directory / (name + '_scores.npy')), mode='w+', dtype=np.float32, shape=(n, 256)) for name in evaluators}
        totals = {name: Counter() for name in evaluators}
        mask_sums = {name: 0. for name in evaluators}
        evaluated = 0
        phase_start = time.monotonic()
        with (directory / 'rows.jsonl').open('x', encoding='utf-8') as stream, torch.no_grad():
            for raw in loader(dataset, ids, False):
                inputs, batch = prepare(raw, 'eval')
                parent, call = observed_readback_forward(model, inputs)
                predictions = pair(parent, inputs['points'][:, 1:].reshape(len(batch['utterances']), 50000, 6))
                predictions['parent'] = parent
                final_boxes = torch.cat([parent['last_center'], parent['last_pred_size'].clamp(min=1e-6)], -1)
                truth = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
                boxes_array[evaluated:evaluated + len(truth)] = final_boxes.cpu().numpy()
                scores = {}
                for name, value in predictions.items():
                    assert not set(value).intersection(batch)
                    scores[name] = native_root_bbs(value['last_sem_cls_scores'], batch)
                    score_arrays[name][evaluated:evaluated + len(truth)] = scores[name].cpu().numpy()
                    value.update(batch)
                    for key in value:
                        if 'pred_size' in key:
                            value[key] = value[key].clamp(min=1e-6)
                    evaluators[name].evaluate(value, 'last_')
                for bid, root_box in enumerate(truth):
                    row_id = int(batch['local_training_id'][bid])
                    assert row_id == ids[evaluated]
                    overlap = iou(final_boxes[bid], root_box)
                    record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]), root_box=root_box.cpu().tolist(), text=batch['utterances'][bid], point_sha256=hashlib.sha256(raw['point_clouds'][bid].numpy().tobytes()).hexdigest(), detector_boxes_sha256=hashlib.sha256(raw['all_detected_boxes'][bid].numpy().tobytes()).hexdigest(), detector_class_ids_sha256=hashlib.sha256(raw['all_detected_class_ids'][bid].numpy().tobytes()).hexdigest(), detector_label_mask_sha256=hashlib.sha256(raw['all_detected_bbox_label_mask'][bid].numpy().tobytes()).hexdigest(), superpoint_sha256=hashlib.sha256(raw['superpoint'][bid].numpy().tobytes()).hexdigest(), all256_oracle25=bool((overlap > .25).any()), all256_oracle50=bool((overlap > .5).any()), arms={})
                    for name, value in predictions.items():
                        ranked = scores[name][bid].argsort(descending=True)
                        query = int(ranked[0])
                        alpha = parent['adaptive_weights'][bid]
                        mask = ((alpha * parent['last_pred_masks'][bid][0, query] + (1 - alpha) * parent['sp_last_pred_masks'][bid][query]).sigmoid() > .5)[parent['superpoints'][bid]]
                        target_mask = batch['gt_masks'][bid, 0].bool()
                        mask_iou = float((mask & target_mask).sum().float() / (mask | target_mask).sum())
                        box_iou = float(overlap[query])
                        item = dict(query=query, box=final_boxes[bid, query].cpu().tolist(), iou=box_iou, mask_iou=mask_iou, reference_valid=bool(parent['mask_reference_valid'][bid, query]), score=float(scores[name][bid, query]), oracle25=[bool((overlap[ranked[:count]] > .25).any()) for count in (16, 32, 64, 256)], oracle50=[bool((overlap[ranked[:count]] > .5).any()) for count in (16, 32, 64, 256)])
                        record['arms'][name] = item
                        totals[name].update(hits25=int(box_iou > .25), hits50=int(box_iou > .5), mask_hits25=int(mask_iou > .25), mask_hits50=int(mask_iou > .5))
                        mask_sums[name] += mask_iou
                    for arm in ARMS:
                        for threshold, suffix in ((.25, '25'), (.5, '50')):
                            a, b = record['arms']['parent']['iou'], record['arms'][arm]['iou']
                            totals[arm].update({'repairs' + suffix: int(a <= threshold < b), 'damages' + suffix: int(b <= threshold < a)})
                    stream.write(json.dumps(record) + '\n')
                    evaluated += 1
                if evaluated % 512 < 8 or evaluated == n:
                    stream.flush()
                    print('IDENTITY_EVAL_PROGRESS ' + json.dumps(dict(stage=stage, rows=evaluated, total=n, seconds=time.monotonic() - phase_start)), flush=True)
                del parent, predictions, inputs, batch
        assert evaluated == n
        boxes_array.flush()
        for value in score_arrays.values():
            value.flush()
        metrics = {}
        for name, evaluator in evaluators.items():
            assert totals[name]['hits25'] == evaluator.dets[('last_', .25, 1, 'bbs')]
            assert totals[name]['hits50'] == evaluator.dets[('last_', .5, 1, 'bbs')]
            assert abs(mask_sums[name] - float(evaluator.dets['mask_pos'])) < 1e-3
            metrics[name] = dict(totals[name], acc025=totals[name]['hits25'] / n * 100, acc050=totals[name]['hits50'] / n * 100, mask_miou=mask_sums[name] / n * 100)
        if stage == 'initial_formal':
            assert all(metrics[arm]['hits25'] == metrics['parent']['hits25'] and metrics[arm]['hits50'] == metrics['parent']['hits50'] for arm in ARMS)
            assert all(np.array_equal(score_arrays[arm], score_arrays['parent']) for arm in ARMS)
        receipt = dict(status='complete', stage=stage, rows=n, formal_rows=n if stage in ('initial_formal', 'formal') else 0, metrics=metrics, frozen_parent_forwards=math.ceil(n / 8), parent_final_native_head_calls_per_forward=1, cached_counterfactual_native_head_calls_per_batch=2, actual_deployment_native_head_calls=1, same_selected_query_box_and_mask=True, all256_retained=True, shared_boxes_and_masks=True, optimizer_steps_per_arm=steps, seconds=time.monotonic() - phase_start, historical_best_hits=[5599, 4859], historical_best_updated=False)
        write_json(directory / 'receipt.json', receipt)
        print('IDENTITY_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)
        return receipt

    status('initial_formal')
    validation = dataset_for('val')
    initial_formal = evaluate(validation, list(range(9508)), 'initial_formal')
    status('train')
    train_dataset = dataset_for('train')
    physical = {part: {train_dataset.annos[index]['scan_id'].split('_')[0] for index in ids} for part, ids in partitions.items()}
    assert not physical['fit'].intersection(physical['holdout'])
    train_dataset.augment = True
    train_dataset.augment_det = False
    reset_rng()
    phase_start = time.monotonic()
    with (output / 'train.jsonl').open('x', encoding='utf-8') as stream:
        for raw in loader(train_dataset, partitions['fit'], True):
            inputs, batch = prepare(raw, 'train')
            with torch.no_grad():
                parent, call = observed_readback_forward(model, inputs)
            predictions = pair(parent, inputs['points'][:, 1:].reshape(len(batch['utterances']), 50000, 6))
            records = {}
            for arm, value in predictions.items():
                assert not set(value).intersection(batch)
                value.update(batch)
                matching = []
                hook = set_criterion.matcher.register_forward_hook(lambda module, arguments, result: matching.append([(a.clone(), b.clone()) for a, b in result]))
                native, value = criterion(value, 6, set_criterion, query_points_obj_topk=training.query_points_obj_topk)
                hook.remove()
                assert len(matching) == 7
                correction, assignment = semantic_assignment_correction(value, batch, matching[1], set_criterion.eos_coef)
                loss = native + correction
                optimizer, head = optimizers[arm], heads[arm]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                assert all(p.grad is None for p in model.parameters())
                other = ARMS[1 - ARMS.index(arm)]
                assert all(p.grad is None for p in heads[other].parameters())
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), .1)
                assert torch.isfinite(norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                records[arm] = dict(loss=float(loss), final_native_ce=float(value['last__loss_ce']), g_correction=float(correction), gradient_norm=float(norm), support_mass_zero_queries=int((value['identity_support_mass'] == 0).sum()), assignment=assignment)
            steps += 1
            row_ids = batch['local_training_id'].tolist()
            assert len(row_ids) == (2 if steps == 3723 else 8)
            seen.extend(row_ids)
            record = dict(step=steps, row_ids=row_ids, arms=records, parent_forwards=1, parent_native_head_calls=call['final_semantic_head_calls'], seconds=time.monotonic() - phase_start)
            stream.write(json.dumps(record) + '\n')
            if steps == 1 or steps % 128 == 0 or steps == 3723:
                stream.flush()
                status('train', optimizer_steps_per_arm=steps, fit_rows=len(seen), elapsed_training_seconds=time.monotonic() - phase_start)
                print('IDENTITY_TRAIN_PROGRESS ' + json.dumps(record), flush=True)
            del parent, predictions, inputs, batch
    assert steps == 3723 and Counter(seen) == Counter(partitions['fit'])
    assert all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in frozen.items())
    for arm, head in heads.items():
        directory = output / arm
        directory.mkdir()
        payload = dict(state_delta={'boundary_evidence_readback.' + name: value.detach().cpu().clone() for name, value in head.state_dict().items()}, optimizer=optimizers[arm].state_dict(), arm=arm, candidate_specific=arm == 'candidate_fused', step=steps, row_ids=seen, parent_support_terminal_sha256=spec['support_terminal_sha256'], parent_support_updates=7446, new_identity_updates=3723, optimizer_reinitialized=True, preflight_state_carried=False, parent_state_tensors=1314, deployed_full_state_tensors=1299, head_parameters=107040, head_state_tensors=8, seed=2027, spec_sha256=sha(args.spec), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        temporary = directory / 'terminal.pth.tmp'
        torch.save(payload, temporary)
        temporary.replace(directory / 'terminal.pth')
        saved = torch.load(directory / 'terminal.pth', map_location='cpu')
        assert all(torch.equal(value.detach().cpu(), saved['state_delta']['boundary_evidence_readback.' + name]) for name, value in head.state_dict().items())
    train_dataset.augment = False
    status('holdout')
    holdout = evaluate(train_dataset, partitions['holdout'], 'holdout')
    del train_dataset
    status('formal')
    formal = evaluate(validation, list(range(9508)), 'formal')
    assert all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in frozen.items())
    assert all(sha(path) == digest for path, digest in parents.items())
    result = dict(status='complete', initial_formal=initial_formal, holdout=holdout, formal=formal, optimizer_steps_per_arm=3723, fit_examples_per_arm=29778, fit_examples_seen_exactly_once=True, parent_state_exact=True, preflight_updates_not_carried=True, model_selection_not_yet_audited=True, full_goal_complete=False, elapsed_seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(output / 'receipt.json', result)
    status('complete', status='complete', exit_code=0, finished_cst=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=result['elapsed_seconds'])
    print('IDENTITY_CAMPAIGN_COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
