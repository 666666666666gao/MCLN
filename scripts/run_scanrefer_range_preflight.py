"""Real train-input parity, observation coverage and disposable-gradient checks."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    assert manifest['schema'] == 'mcln-scanrefer-range-preflight-v1'
    assert manifest['formal_rows'] == manifest['checkpoint_writes'] == 0
    source = Path(manifest['model_source'])
    assert sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, expected in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == expected, name
    for name, expected in manifest['files'].items():
        assert sha(directory / name) == expected, name
    for name, item in manifest['artifacts'].items():
        assert sha(item['path']) == item['sha256'], name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    split = json.loads(Path(manifest['split_protocol']).read_text())
    selected_ids = split['selected_ids']
    assert len(selected_ids) == 16 and set(selected_ids).issubset(split['row_ids']['fit'])
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import copy
    import random
    import numpy as np
    import torch
    import scripts
    import models
    scripts.__path__ = [str(directory / 'scripts'), str(source / 'scripts')]
    models.__path__ = [str(directory / 'models'), str(source / 'models')]
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.rec_reranker import compute_query_ious
    import models.candidate_range_visual as range_module
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints
    from scripts.scanrefer_joint_readout import JointRecReadout

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    command = set_scanrefer_data_root(build_authoritative_command(directory / 'unused_output'), manifest['data_root'])
    data_inputs = {part: verify_scanrefer_superpoints(manifest['data_root'], part, files)
                   for part, files in manifest['superpoint_files'].items()}
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls and not args.butd_gt
    assert args.use_color and not args.use_height and not args.use_multiview
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    initial = {name[7:]: value for name, value in torch.load(args.checkpoint_path, map_location='cpu')['model'].items()}
    first = TrainTester.get_model(args).cuda().eval()
    first.load_state_dict(initial, strict=True)
    for name, parameter in first.named_parameters():
        parameter.requires_grad_(name.startswith(('decoder.5.', 'prediction_heads.5.')))
    core_names = [name for name, parameter in first.named_parameters() if parameter.requires_grad]
    first.decoder[-1].local_visual = range_module.CandidateRangeVisual('center').cuda().eval()
    models_by_arm = {'center': first, 'extent': copy.deepcopy(first)}
    models_by_arm['extent'].decoder[-1].local_visual.sampling = 'extent'
    reader_initial = {name: value.detach().cpu().clone() for name, value in first.state_dict().items()
                      if name.startswith('decoder.5.local_visual.')}
    assert sum(parameter.numel() for parameter in first.decoder[-1].local_visual.parameters()) == 145008
    artifacts = {name: torch.load(item['path'], map_location='cpu')
                 for name, item in manifest['artifacts'].items() if name != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    readout_state = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}
    criterion, set_criterion = TrainTester.get_criterion(args)
    dataset = Joint3DDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(dataset.annos) == 36665
    dataset.augment = False
    partitions = {'fit': [], 'holdout': []}
    for index, row in enumerate(dataset.annos):
        code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
        part = 'holdout' if int(hashlib.sha256(code).hexdigest()[:8], 16) % 5 == 0 else 'fit'
        partitions[part].append(index)
    assert partitions == split['row_ids']
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected_ids), batch_size=12,
        shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(0))
    observations, coverage = [], {arm: [] for arm in models_by_arm}
    original_sampler = range_module.candidate_region_points
    capture = {}

    def record_sampler(xyz, boxes, sampling):
        selected, valid, regions = original_sampler(xyz, boxes, sampling)
        coordinates = xyz.gather(1, selected.reshape(len(xyz), -1, 1).expand(-1, -1, 3)).reshape(*selected.shape, 3)
        capture[sampling] = (boxes.detach(), selected.detach(), valid.detach(), regions.detach(), coordinates.detach())
        return selected, valid, regions

    range_module.candidate_region_points = record_sampler
    keys = ['last_center', 'last_pred_size', 'last_sem_cls_scores', 'last_proj_queries',
            'sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']
    start = time.time()
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
        root_valid = batch['box_label_mask'][:, :1].bool()
        reader = first.decoder[-1].local_visual
        first.decoder[-1].local_visual = None
        with torch.no_grad():
            reference = first(inputs)
            reference_runtime = readout(reference, inputs)['runtime']
        first.decoder[-1].local_visual = reader
        for arm, model in models_by_arm.items():
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            began = time.time()
            candidate = model(inputs)
            torch.cuda.synchronize()
            forward_seconds = time.time() - began
            for key in keys:
                if torch.is_tensor(reference[key]):
                    assert torch.equal(reference[key], candidate[key]), (arm, key)
                else:
                    assert len(reference[key]) == len(candidate[key])
                    assert all(torch.equal(a, b) for a, b in zip(reference[key], candidate[key])), (arm, key)
            with torch.no_grad():
                candidate_runtime = readout(candidate, inputs)['runtime']
                for key, value in reference_runtime.items():
                    assert torch.equal(value, candidate_runtime[key]) if torch.is_tensor(value) else value == candidate_runtime[key], (arm, key)
                boxes, indices, valid, regions, coordinates = capture[arm]
                ious = compute_query_ious(boxes, roots, root_valid)
                normalized = (coordinates - boxes[:, :, None, :3]) / (boxes[..., 3:] * .5).clamp_min(.05)[:, :, None]
                indices_cpu, valid_cpu, regions_cpu, normalized_cpu, ious_cpu = [value.cpu().numpy()
                    for value in (indices, valid, regions, normalized, ious)]
                for b in range(len(boxes)):
                    for q in range(256):
                        mask = valid_cpu[b, q]
                        point_ids = indices_cpu[b, q, mask]
                        assert len(np.unique(point_ids)) == len(point_ids)
                        point_values = normalized_cpu[b, q, mask]
                        assert (np.abs(point_values) <= 1.5).all()
                        counts = [int(((regions_cpu[b, q] == region) & mask).sum()) for region in range(8)]
                        coverage[arm].append({'row_id': selected_ids[batch_index * 12 + b],
                            'scan_id': raw['scan_ids'][b], 'query_slot': q, 'input_box_iou': float(ious_cpu[b, q]),
                            'valid_points': int(mask.sum()), 'occupied_regions': sum(count > 0 for count in counts),
                            'region_counts': counts, 'context_points': int((np.abs(point_values) > 1).any(axis=-1).sum()),
                            'normalized_span': (point_values.max(axis=0) - point_values.min(axis=0)).tolist() if len(point_values) else [0., 0., 0.]})
            candidate.update(batch)
            loss, candidate = TrainTester._compute_loss(candidate, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            gradient = float(model.decoder[-1].local_visual.output_projection.weight.grad.norm())
            assert gradient > 0
            observation = {'arm': arm, 'batch': batch_index, 'rows': len(raw['scan_ids']),
                'forward_seconds_including_coverage_capture': forward_seconds,
                'zero_start_native_and_v99_parity': True, 'loss': float(loss), 'output_gradient': gradient,
                'point_sha256': [hashlib.sha256(point.cpu().numpy().tobytes()).hexdigest() for point in inputs['point_clouds']]}
            observations.append(observation)
            print('SCANREFER RANGE PREFLIGHT', json.dumps(observation), flush=True)
            del candidate, candidate_runtime, loss
        del reference, reference_runtime
    range_module.candidate_region_points = original_sampler
    capture.clear()
    gradient_checks = {}
    for arm, model in models_by_arm.items():
        core = [p for name, p in model.named_parameters() if name in core_names]
        optimizer = torch.optim.AdamW([{'params': core, 'lr': 1e-6},
            {'params': model.decoder[-1].local_visual.parameters(), 'lr': 1e-4}], weight_decay=.0005)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            outputs.update(batch)
            loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            parameters = [p for p in model.parameters() if p.requires_grad]
            assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(parameters, .1)
            optimizer.step()
            del outputs, loss
        gradient_checks[arm] = {name: float(p.grad.norm()) for name, p in model.decoder[-1].local_visual.named_parameters()}
        for name in ['point_encoder.0.weight', 'query_projection.weight', 'key_projection.weight', 'value_projection.weight']:
            assert gradient_checks[arm][name] > 0, (arm, name)
        for name, value in model.state_dict().items():
            if name not in core_names and name not in reader_initial:
                assert torch.equal(value.cpu(), initial[name]), (arm, name)
        del optimizer
    assert all(torch.equal(value.cpu(), readout_state[name]) for name, value in readout.state_dict().items())
    write(directory / 'coverage_rows.json', coverage)
    summary = {}
    for arm, rows in coverage.items():
        summary[arm] = {}
        for name, subset in [('all', rows), ('input_box_iou_gt025', [r for r in rows if r['input_box_iou'] > .25])]:
            assert subset
            summary[arm][name] = {'queries': len(subset),
                'mean_valid_points': sum(r['valid_points'] for r in subset) / len(subset),
                'mean_occupied_regions': sum(r['occupied_regions'] for r in subset) / len(subset),
                'empty_queries': sum(r['valid_points'] == 0 for r in subset),
                'mean_normalized_span': np.asarray([r['normalized_span'] for r in subset]).mean(axis=0).tolist()}
    receipt = {'schema': 'mcln-scanrefer-range-preflight-result-v1', 'status': 'pass',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'formal_rows': 0, 'train_rows': 16, 'selected_train_ids': selected_ids,
        'disposable_optimizer_steps_per_arm': 2, 'checkpoint_writes': 0,
        'reader_parameters_per_arm': 145008, 'slot_budget': 64, 'valid_points_are_not_assumed_to_equal_slots': True,
        'source_manifest_sha256': manifest['source_manifest_sha256'], 'manifest_sha256': sha(option.manifest),
        'data_root': args.data_root, 'superpoint_inputs': data_inputs, 'observations': observations,
        'coverage_summary': summary, 'coverage_rows_sha256': sha(directory / 'coverage_rows.json'),
        'gradients_after_second_step': gradient_checks, 'frozen_parameters_buffers_readouts_unchanged': True,
        'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2, 'elapsed_seconds': time.time() - start,
        'quality_improvement_not_tested': True, 'preflight_weights_discarded': True}
    write(directory / 'receipt.json', receipt)
    print('SCANREFER RANGE PREFLIGHT COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
