"""Trace actual E71 Mask -> geometry and score gradients on 16 fixed fit rows.

No optimizer, weight writes, validation rows, surrogate gradients, or new model.
Coordinate L1 is a diagnostic probe, not a proposed training loss or matching rule.
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
    assert manifest['schema'] == 'mcln-geometry-gradient-probe-v1'
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
    from models.rec_mask_geometry import DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints
    from scripts.scanrefer_joint_readout import JointRecReadout, joint_rec_readout_loss

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
    criterion, set_criterion = TrainTester.get_criterion(args)

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
    variant_names = [item['name'] for item in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    observations = []
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        outputs = model(inputs)
        # These are the original graph ancestors, not copies or reconstructed fused tensors.
        groups = {
            'text_mask_outputs': list(outputs['last_pred_masks']),
            'query_mask_outputs': list(outputs['sp_last_pred_masks']),
            'native_center_output': [outputs['last_center']],
            'native_size_output': [outputs['last_pred_size']],
            'core_parameters': list(parameters.values()),
        }
        assert all(value.requires_grad for values in groups.values() for value in values)
        trace = readout(outputs, inputs)
        runtime = trace['runtime']
        boxes = runtime['rec_geometry_boxes'].reshape(4, 16, 7, 6)
        valid = runtime['rec_geometry_valid_mask'].reshape(4, 16, 7)
        roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
        root_valid = batch['box_label_mask'][:, :1]
        assert bool(root_valid.all())
        ious = compute_query_ious(boxes.detach().reshape(4, -1, 6), roots, root_valid).reshape(4, 16, 7)
        losses = {}
        for index, name in enumerate(variant_names):
            assert bool(valid[:, :, index].any()), name
            loss = (boxes[:, :, index] - roots).abs().mean(dim=-1)[valid[:, :, index]].mean()
            losses['coordinate_l1/' + name] = loss
        score_loss, score_stats = joint_rec_readout_loss(trace, roots, root_valid)
        losses['existing_readout_gt_loss'] = score_loss / 3.
        outputs.update(batch)
        native_loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        losses['native_gt_loss'] = native_loss
        tensors = tuple(value for values in groups.values() for value in values)
        summaries = {}
        for name, loss in losses.items():
            assert loss.requires_grad and bool(torch.isfinite(loss)), name
            gradients = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
            cursor = 0
            summary = {'loss': float(loss.detach()), 'groups': {}}
            for group, values in groups.items():
                grads = gradients[cursor:cursor + len(values)]
                cursor += len(values)
                present = [grad for grad in grads if grad is not None]
                assert all(bool(torch.isfinite(grad).all()) for grad in present), (name, group)
                summary['groups'][group] = {
                    'tensors': len(values), 'connected_tensors': len(present),
                    'nonzero_tensors': sum(int(bool(torch.count_nonzero(grad))) for grad in present),
                    'norm': sum(float(grad.double().square().sum()) for grad in present) ** .5,
                }
            summaries[name] = summary
            del gradients
        record = {
            'batch': batch_index, 'row_ids': selected_ids[batch_index * 4:(batch_index + 1) * 4],
            'scan_ids': raw['scan_ids'],
            'point_sha256': [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in inputs['point_clouds']],
            'query_indices': trace['parent']['candidate_batch']['query_indices'].detach().cpu().tolist(),
            'boxes': boxes.detach().cpu().tolist(), 'valid': valid.cpu().tolist(),
            'root_boxes': roots.cpu().tolist(), 'root_ious': ious.cpu().tolist(),
            'valid_per_variant': valid.sum(dim=(0, 1)).cpu().tolist(),
            'loss_gradients': summaries, 'readout_components': score_stats,
        }
        observations.append(record)
        print('GEOMETRY GRADIENT BATCH', json.dumps({
            'batch': batch_index, 'valid_per_variant': record['valid_per_variant'],
            'fused_box_mask_gradient': summaries['coordinate_l1/fused_t0_exact']['groups']['query_mask_outputs'],
            'readout_mask_gradient': summaries['existing_readout_gt_loss']['groups']['query_mask_outputs'],
        }), flush=True)
        del outputs, trace, runtime, boxes, valid, ious, native_loss, score_loss, loss, losses, tensors, groups
    assert len(observations) == 4
    assert all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value.cpu(), readout_state[name]) for name, value in readout.state_dict().items())
    for item in observations:
        gradients = item['loss_gradients']
        for name in variant_names:
            stats = gradients['coordinate_l1/' + name]['groups']
            assert stats['query_mask_outputs']['norm'] == 0. and stats['text_mask_outputs']['norm'] == 0.
        for name in variant_names[1:5]:
            assert gradients['coordinate_l1/' + name]['groups']['core_parameters']['norm'] == 0.
        assert gradients['coordinate_l1/regressed']['groups']['native_center_output']['norm'] > 0.
        assert gradients['native_gt_loss']['groups']['query_mask_outputs']['norm'] > 0.
        assert gradients['existing_readout_gt_loss']['groups']['query_mask_outputs']['norm'] > 0.
    receipt = {
        'schema': 'mcln-geometry-gradient-probe-result-v1', 'status': 'pass',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'input_manifest_sha256': sha(option.manifest), 'rows': 16, 'batch_size': 4,
        'variant_names': variant_names, 'observations': observations,
        'requires_grad_parameters': list(parameters), 'full_state_tensors_unchanged': len(initial),
        'readouts_unchanged': True, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0,
        'data_inputs': verified_data, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'scope': 'Autograd route diagnostic on actual fit forwards; not causal attribution of historical REC loss or a quality result.',
    }
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('GEOMETRY GRADIENT PROBE COMPLETE', json.dumps({'status': 'pass', 'rows': 16, 'optimizer_steps': 0}), flush=True)


if __name__ == '__main__':
    main()
