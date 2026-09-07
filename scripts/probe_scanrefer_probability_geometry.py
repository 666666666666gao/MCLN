"""Actual fit-input check of a continuous probability-geometry prototype.

No optimizer, validation, parameter updates, or production-path replacement.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.resolve().parent
    manifest = json.loads(option.manifest.read_text())
    assert manifest['schema'] == 'mcln-probability-geometry-probe-v1'
    source = Path(manifest['model_source'])
    assert sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert sha(directory / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert sha(item['path']) == item['sha256'], name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    split = json.loads(Path(manifest['split_protocol']).read_text())
    selected_ids = split['selected_ids']
    assert len(selected_ids) == 16 and set(selected_ids).issubset(split['row_ids']['fit'])
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts'), str(source / 'scripts')]
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.rec_reranker import compute_query_ious
    from models.rec_mask_geometry import normalize_mcln_mask_logits, _find_superpoint_map
    from scripts.prototype_probability_geometry import probability_quantile_boxes
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
    verified_data = verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls and not args.butd_gt
    initial = {name[7:]: value for name, value in torch.load(args.checkpoint_path, map_location='cpu')['model'].items()}
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict(initial, strict=True)
    assert model.decoder[-1].local_visual is None and model.query_mask_fusion_calibrator is None
    prefixes = ('decoder.5.', 'prediction_heads.5.', 'x_mask.', 'x_query.', 'rel_encoder.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    artifacts = {name: torch.load(item['path'], map_location='cpu') for name, item in manifest['artifacts'].items() if name != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    readout_state = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}

    class ProbeDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            for index, row in enumerate(annos):
                row['_geometry_gradient_id'] = index
            scenes = {annos[i]['scan_id'] for i in selected_ids}
            annos[:] = [row for row in annos if row['scan_id'] in scenes]
            super()._scene_graph_parse(annos)

    dataset = ProbeDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    by_id = {row['_geometry_gradient_id']: row for row in dataset.annos}
    dataset.annos = [by_id[i] for i in selected_ids]
    dataset.augment = False
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(0))
    rows = []
    gradient_records = []
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        outputs = model(inputs)
        with torch.no_grad():
            trace = readout(outputs, inputs)
        query_indices = trace['parent']['candidate_batch']['query_indices']
        hard_boxes = trace['runtime']['rec_geometry_boxes'].reshape(4, 16, 7, 6)[:, :, 2]
        valid = trace['runtime']['rec_geometry_valid_mask'].reshape(4, 16, 7)[:, :, 2]
        roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
        root_valid = batch['box_label_mask'][:, :1]
        superpoints, _ = _find_superpoint_map(outputs, inputs)
        soft_list, mass_ratios = [], []
        for index in range(4):
            _, _, fused, _ = normalize_mcln_mask_logits(outputs, index, query_indices[index])
            ids = superpoints[index].long().reshape(-1)
            coords = inputs['point_clouds'][index, :, :3]
            soft, weights = probability_quantile_boxes(coords, ids, fused, .005)
            excluded = fused.detach().index_select(1, ids) <= 0.
            mass_ratios.append((weights.detach() * excluded).sum(-1) / weights.detach().sum(-1))
            soft_list.append(soft)
        soft_boxes = torch.stack(soft_list)
        soft_ious = compute_query_ious(soft_boxes.detach(), roots, root_valid)
        hard_ious = compute_query_ious(hard_boxes, roots, root_valid)
        objective = (soft_boxes - roots).abs().mean(-1)[valid].mean()
        original_masks = list(outputs['sp_last_pred_masks'])
        gradients = torch.autograd.grad(objective, tuple(original_masks) + tuple(parameters.values()), allow_unused=True)
        assert all(torch.isfinite(grad).all() for grad in gradients if grad is not None)
        mask_norm = sum(float(grad.double().square().sum()) for grad in gradients[:4] if grad is not None) ** .5
        core_norm = sum(float(grad.double().square().sum()) for grad in gradients[4:] if grad is not None) ** .5
        assert mask_norm > 0. and core_norm > 0.
        gradient_records.append({'batch': batch_index, 'coordinate_l1': float(objective),
            'original_query_mask_gradient_norm': mask_norm, 'native_parameter_gradient_norm': core_norm})
        for index in range(4):
            rows.append({'row_id': selected_ids[batch_index * 4 + index], 'scan_id': raw['scan_ids'][index],
                'point_sha256': hashlib.sha256(inputs['point_clouds'][index].cpu().numpy().tobytes()).hexdigest(),
                'query_indices': query_indices[index].cpu().tolist(),
                'root_box': roots[index, 0].cpu().tolist(), 'hard_boxes': hard_boxes[index].cpu().tolist(),
                'soft_boxes': soft_boxes[index].detach().cpu().tolist(), 'valid': valid[index].cpu().tolist(),
                'hard_ious': hard_ious[index].cpu().tolist(), 'soft_ious': soft_ious[index].cpu().tolist(),
                'threshold_excluded_probability_mass': mass_ratios[index].cpu().tolist()})
        print('PROBABILITY GEOMETRY BATCH', json.dumps(gradient_records[-1]), flush=True)
        del outputs, trace, hard_boxes, valid, soft_boxes, roots, fused, soft, weights, soft_list, gradients, objective, original_masks
    assert len(rows) == 16 and len(set(row['row_id'] for row in rows)) == 16
    assert all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value.cpu(), readout_state[name]) for name, value in readout.state_dict().items())
    paired_valid = np.asarray([row['valid'] for row in rows], dtype=bool)
    hard = np.asarray([row['hard_ious'] for row in rows])
    soft = np.asarray([row['soft_ious'] for row in rows])
    mass = np.asarray([row['threshold_excluded_probability_mass'] for row in rows])
    summary = {'valid_candidate_pairs': int(paired_valid.sum()),
        'hard_candidate_hits': [int(((hard > threshold) & paired_valid).sum()) for threshold in [.25, .5]],
        'soft_candidate_hits': [int(((soft > threshold) & paired_valid).sum()) for threshold in [.25, .5]],
        'hard_row_oracle_hits': [int(((hard > threshold) & paired_valid).any(1).sum()) for threshold in [.25, .5]],
        'soft_row_oracle_hits': [int(((soft > threshold) & paired_valid).any(1).sum()) for threshold in [.25, .5]],
        'hard_mean_iou': float(hard[paired_valid].mean()), 'soft_mean_iou': float(soft[paired_valid].mean()),
        'threshold_excluded_probability_mass_mean': float(mass[paired_valid].mean()),
        'threshold_excluded_probability_mass_median': float(np.median(mass[paired_valid])),
        'pairs_with_excluded_mass_above_one_percent': int((mass[paired_valid] > .01).sum())}
    receipt = {'schema': 'mcln-probability-geometry-probe-result-v1', 'status': 'completed',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'input_manifest_sha256': sha(option.manifest), 'rows': rows, 'summary': summary, 'gradient_records': gradient_records,
        'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0, 'model_and_readouts_unchanged': True,
        'data_inputs': verified_data, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'scope': 'Fixed untrained readout feasibility on fit rows, not deployed REC or a trained method result.'}
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('PROBABILITY GEOMETRY COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
