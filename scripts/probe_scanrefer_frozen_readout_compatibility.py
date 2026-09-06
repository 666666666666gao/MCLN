"""Actual train-input gradients through frozen V99; disposable updates only."""

import argparse
import datetime
import hashlib
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
    assert manifest['schema'] == 'mcln-frozen-readout-compatibility-probe-v1'
    assert manifest['formal_rows'] == manifest['checkpoint_writes'] == 0
    assert manifest['core_learning_rate'] == 1e-6 and manifest['auxiliary_weight'] == 1. / 3.
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
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts'), str(source / 'scripts')]
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints
    from scripts.scanrefer_joint_readout import JointRecReadout, joint_rec_readout_loss
    from scripts.audit_scanrefer_joint_readout_pair import assert_metadata_equal

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    command = set_scanrefer_data_root(build_authoritative_command(directory / 'unused_output'), manifest['data_root'])
    data_inputs = verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls and not args.butd_gt
    assert args.use_color and not args.use_height and not args.use_multiview
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    initial = {name[7:]: value for name, value in torch.load(args.checkpoint_path, map_location='cpu')['model'].items()}
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict(initial, strict=True)
    assert model.decoder[-1].local_visual is None
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('decoder.5.', 'prediction_heads.5.')))
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    artifacts = {name: torch.load(item['path'], map_location='cpu')
                 for name, item in manifest['artifacts'].items() if name != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    readout_state = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}
    readout_metadata = copy.deepcopy(readout.metadata)
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
    observations = []
    begin = time.time()
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
        root_valid = batch['box_label_mask'][:, :1].bool()
        outputs = model(inputs)
        connected = readout(outputs, inputs)
        detached = readout(outputs, inputs, detach_visual=True)
        for key, value in connected['runtime'].items():
            assert torch.equal(value, detached['runtime'][key]) if torch.is_tensor(value) else value == detached['runtime'][key], key
        auxiliary, stats = joint_rec_readout_loss(connected, roots, root_valid)
        detached_auxiliary, detached_stats = joint_rec_readout_loss(detached, roots, root_valid)
        assert auxiliary.item() == detached_auxiliary.item() and stats == detached_stats
        assert auxiliary.requires_grad and not detached_auxiliary.requires_grad
        outputs.update(batch)
        native, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        native_grads = torch.autograd.grad(native, tuple(parameters.values()), retain_graph=True, allow_unused=True)
        auxiliary_grads = torch.autograd.grad(auxiliary, tuple(parameters.values()), allow_unused=True)
        records = {}
        norm_native = norm_auxiliary = dot = 0.
        for name, first, second in zip(parameters, native_grads, auxiliary_grads):
            for value in [first, second]:
                if value is not None:
                    assert torch.isfinite(value).all(), name
            records[name] = {'native_norm': None if first is None else float(first.norm()),
                             'frozen_readout_norm': None if second is None else float(second.norm())}
            norm_native += 0. if first is None else float(first.square().sum())
            norm_auxiliary += 0. if second is None else float(second.square().sum())
            if first is not None and second is not None:
                dot += float((first * second).sum())
        assert norm_native > 0 and norm_auxiliary > 0
        observation = {'batch': batch_index, 'rows': len(raw['scan_ids']),
            'row_ids': selected_ids[batch_index * 12:batch_index * 12 + len(raw['scan_ids'])],
            'scan_ids': raw['scan_ids'], 'point_sha256': [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in inputs['point_clouds']],
            'native_loss': float(native), 'frozen_readout_loss': float(auxiliary), 'readout_stats': stats,
            'native_gradient_norm': norm_native ** .5, 'frozen_readout_gradient_norm': norm_auxiliary ** .5,
            'gradient_cosine': dot / (norm_native * norm_auxiliary) ** .5, 'parameter_gradients': records,
            'frozen_connected_and_detached_runtime_equal': True, 'detached_gradient_absent': True}
        observations.append(observation)
        print('FROZEN READOUT GRADIENT', json.dumps({k: observation[k] for k in ['batch', 'rows', 'native_loss',
            'frozen_readout_loss', 'native_gradient_norm', 'frozen_readout_gradient_norm', 'gradient_cosine']}), flush=True)
        del outputs, connected, detached, auxiliary, detached_auxiliary, native, native_grads, auxiliary_grads
    assert len(observations) == 2 and sum(row['rows'] for row in observations) == 16
    assert all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items())
    updates = {}
    for arm in ['native_only', 'frozen_readout_gt']:
        candidate = copy.deepcopy(model)
        trainable = [value for value in candidate.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=manifest['core_learning_rate'], weight_decay=.0005)
        losses = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = candidate(inputs)
            current = readout(outputs, inputs, detach_visual=(arm == 'native_only'))
            auxiliary, stats = joint_rec_readout_loss(current, roots, root_valid)
            outputs.update(batch)
            native, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            loss = native + manifest['auxiliary_weight'] * auxiliary
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(value.grad).all() for value in trainable if value.grad is not None)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, .1)
            assert torch.isfinite(gradient_norm)
            optimizer.step()
            losses.append({'step': step + 1, 'native_loss': float(native), 'readout_loss': float(auxiliary),
                           'gradient_norm_before_clip': float(gradient_norm), 'readout_stats': stats})
            del outputs, current, auxiliary, native, loss
        changed = []
        for name, value in candidate.state_dict().items():
            same = torch.equal(value.cpu(), initial[name])
            if name not in parameters:
                assert same, (arm, name)
            elif not same:
                changed.append(name)
        assert changed and all(float(state['step']) == 2 for state in optimizer.state.values())
        updates[arm] = {'losses': losses, 'changed_core_tensors': changed,
                        'optimizer_parameter_tensors': len(optimizer.state), 'optimizer_steps': 2}
        del candidate, optimizer, trainable
    assert all(not p.requires_grad and p.grad is None for p in readout.parameters())
    assert all(torch.equal(value.cpu(), readout_state[name]) for name, value in readout.state_dict().items())
    assert_metadata_equal(readout.metadata, readout_metadata)
    receipt = {'schema': 'mcln-frozen-readout-compatibility-probe-receipt-v1', 'status': 'pass',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'manifest_sha256': sha(option.manifest), 'real_train_rows': 16, 'formal_rows': 0,
        'backbone_forwards': 6, 'disposable_optimizer_steps_per_arm': 2, 'checkpoint_writes': 0,
        'new_network_modules': 0, 'readout_frozen_and_unchanged': True, 'frozen_core_and_buffers_unchanged': True,
        'observations': observations, 'disposable_updates': updates, 'candidate_trainable_tensors': list(parameters),
        'data_inputs': data_inputs, 'python': sys.version, 'torch': torch.__version__,
        'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2, 'elapsed_seconds': time.time() - begin,
        'no_quality_or_generalization_conclusion': True}
    write(directory / 'receipt.json', receipt)
    print('FROZEN READOUT PROBE COMPLETE', json.dumps({k: v for k, v in receipt.items()
        if k not in ['observations', 'disposable_updates', 'candidate_trainable_tensors']}), flush=True)


if __name__ == '__main__':
    main()
