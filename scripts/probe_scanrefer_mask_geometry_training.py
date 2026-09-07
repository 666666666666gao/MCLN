"""Actual root-matched Mask geometry gradient and disposable update probe.

Two steps per arm on the first fixed fit batch; no saved weights or quality claim.
"""
import argparse
import copy
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
    assert manifest['schema'] == 'mcln-mask-geometry-training-probe-v1'
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
    from scripts.native_mask_geometry_supervision import native_mask_geometry_loss
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
    def losses(candidate, inputs, batch):
        outputs = candidate(inputs)
        boxes = torch.cat([outputs['last_center'], outputs['last_pred_size']], dim=-1)
        roots = torch.cat([batch['center_label'][:, 0], batch['size_gts'][:, 0]], dim=-1)
        captured = []
        def capture(module, arguments, result):
            if torch.equal(arguments[0]['pred_boxes'], boxes):
                assert all(torch.equal(target['boxes'][0], roots[i]) for i, target in enumerate(arguments[1]))
                captured.append(result)
        handle = set_criterion.matcher.register_forward_hook(capture)
        outputs.update(batch)
        native, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        handle.remove()
        assert len(captured) == 1
        auxiliary, stats = native_mask_geometry_loss(outputs, inputs, roots, captured[0])
        assert torch.isfinite(native) and torch.isfinite(auxiliary)
        return outputs, native, auxiliary, stats

    observations = []
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        if batch_index == 0:
            update_batch, update_inputs = batch, inputs
        outputs, native, auxiliary, stats = losses(model, inputs, batch)
        first = torch.autograd.grad(native, tuple(parameters.values()), retain_graph=True, allow_unused=True)
        second = torch.autograd.grad(auxiliary, tuple(parameters.values()), allow_unused=True)
        assert all(torch.isfinite(x).all() for x in first + second if x is not None)
        n1 = sum(float(x.double().square().sum()) for x in first if x is not None)
        n2 = sum(float(x.double().square().sum()) for x in second if x is not None)
        dot = sum(float((x.double() * y.double()).sum()) for x, y in zip(first, second) if x is not None and y is not None)
        assert n1 > 0 and n2 > 0
        record = {'batch': batch_index, 'row_ids': selected_ids[batch_index * 4:(batch_index + 1) * 4],
            'scan_ids': raw['scan_ids'], 'point_sha256': [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in inputs['point_clouds']],
            'native_loss': float(native), 'mask_geometry_loss': float(auxiliary),
            'native_gradient_norm': n1 ** .5, 'geometry_gradient_norm': n2 ** .5,
            'gradient_dot': dot, 'gradient_cosine': dot / (n1 * n2) ** .5,
            'auxiliary_connected_parameters': [name for name, grad in zip(parameters, second) if grad is not None],
            'target_stats': {name: value.cpu().tolist() for name, value in stats.items()}}
        observations.append(record)
        print('MASK GEOMETRY TRAIN GRADIENT', json.dumps({key: record[key] for key in ['batch', 'native_loss', 'mask_geometry_loss', 'gradient_cosine']}), flush=True)
        del outputs, native, auxiliary, stats, first, second
    assert len(observations) == 4
    assert all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items())
    updates = {}
    for arm in ['native_gt', 'native_gt_mask_geometry']:
        student = copy.deepcopy(model)
        trainable = [value for value in student.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-6, weight_decay=.0005)
        step_records = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs, native, auxiliary, stats = losses(student, update_inputs, update_batch)
            objective = native + (auxiliary if arm == 'native_gt_mask_geometry' else auxiliary * 0.)
            objective.backward()
            assert all(torch.isfinite(value.grad).all() for value in trainable if value.grad is not None)
            norm = torch.nn.utils.clip_grad_norm_(trainable, .1)
            optimizer.step()
            step_records.append({'step': step + 1, 'native_loss': float(native), 'mask_geometry_loss': float(auxiliary),
                'gradient_norm': float(norm), 'query_indices': stats['query_indices'].cpu().tolist()})
            del outputs, native, auxiliary, stats, objective
        changed = []
        for name, value in student.state_dict().items():
            same = torch.equal(value.cpu(), initial[name])
            if name not in parameters:
                assert same, (arm, name)
            elif not same:
                changed.append(name)
        assert changed and all(float(state['step']) == 2 for state in optimizer.state.values())
        with torch.no_grad():
            outputs, native, auxiliary, stats = losses(student, update_inputs, update_batch)
        updates[arm] = {'steps': step_records, 'changed_tensors': changed,
            'post_native_loss': float(native), 'post_geometry_loss': float(auxiliary),
            'post_query_indices': stats['query_indices'].cpu().tolist(),
            'frozen_state_unchanged': True}
        del student, optimizer, trainable, outputs, native, auxiliary, stats
    assert all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value.cpu(), readout_state[name]) for name, value in readout.state_dict().items())
    receipt = {'schema': 'mcln-mask-geometry-training-probe-result-v1', 'status': 'pass',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'input_manifest_sha256': sha(option.manifest), 'rows': 16, 'batch_size': 4,
        'observations': observations, 'updates': updates, 'requires_grad_parameters': list(parameters),
        'full_state_tensors': len(initial), 'readouts_unchanged': True,
        'disposable_optimizer_steps_per_arm': 2, 'checkpoint_writes': 0, 'formal_rows': 0,
        'data_inputs': verified_data, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'scope': 'Engineering execution on fixed fit rows; no quality acceptance or pretrained weight replacement.'}
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('MASK GEOMETRY TRAIN PROBE COMPLETE', json.dumps({'status': 'pass', 'rows': 16}), flush=True)


if __name__ == '__main__':
    main()
