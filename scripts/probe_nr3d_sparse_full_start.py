"""Diagnostic only: unchanged V2 full initialization followed by five zero-update backwards."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    addon = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])
    assert manifest['epochs'] == 1 and manifest['steps_per_arm'] == 6687
    assert manifest['train_batch_size'] == 4 and manifest['eval_batch_size'] == 16
    assert manifest['native_learning_rate'] == 1e-5 and manifest['sparse_learning_rate'] == 1e-4
    assert manifest['no_augmentation']

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(addon / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
        assert file_sha(Path(manifest['sparse_preflight_receipt'])) == manifest['sparse_preflight_receipt_sha256']
        assert file_sha(Path(manifest['baseline_reference'])) == manifest['baseline_reference_sha256']
        assert file_sha(Path(manifest['warmup_regression_receipt'])) == manifest['warmup_regression_receipt_sha256']
        for name, digest in manifest['runtime_receipts'].items():
            assert file_sha(Path(name)) == digest, name

    verify_inputs()
    sparse_preflight = json.loads(Path(manifest['sparse_preflight_receipt']).read_text())
    assert sparse_preflight['status'] == 'complete' and sparse_preflight['optimizer_steps'] == 0
    assert sparse_preflight['zero_start_identity'] and sparse_preflight['native_mask_loss_connected']
    assert sparse_preflight['fixed_perturbation_rec_unchanged']
    warmup_regression = json.loads(Path(manifest['warmup_regression_receipt']).read_text())
    assert warmup_regression['status'] == 'complete' and warmup_regression['optimizer_steps'] == 0
    assert manifest['native_warmup_forwards'] == 1
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import spconv
    import cumm
    assert sys.prefix == manifest['python_prefix']
    assert torch.__version__ == '1.10.2+cu111' and spconv.__version__ == '2.3.6'
    assert cumm.__version__ == '0.4.11'
    import scripts
    scripts.__path__ = [str(addon / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual, SparseSuperpointIntervention
    from scripts.run_nr3d_view_pair_role import read_train_rows

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    partitions = {'fit': [], 'holdout': []}
    for i, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        partitions['holdout' if fold == 0 else 'fit'].append(i)
    assert len(partitions['fit']) == 26747 and len(partitions['holdout']) == 6172
    assert partitions == manifest['row_ids']
    scenes = {name: {raw_rows[i]['scan_id'] for i in ids} for name, ids in partitions.items()}
    assert not scenes['fit'].intersection(scenes['holdout'])
    assert len(scenes['fit']) == 413 and len(scenes['holdout']) == 98
    selected_ids = sorted(partitions['fit'] + partitions['holdout'])

    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert checkpoint['evaluation_only'] and 'optimizer' not in checkpoint
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint['config']))
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    assert args.clip_norm == .1 and args.weight_decay == .0005
    assert args.use_color and not args.use_height and not args.use_multiview
    initial_state = {name[7:]: value for name, value in checkpoint['model'].items()}
    del checkpoint
    models, optimizers, trainable = {}, {}, {}
    prefixes = ('x_query.', 'x_mask.', 'rel_encoder.')
    for arm in ['native', 'sparse']:
        model = TrainTester.get_model(args).cuda().eval()
        model.load_state_dict(initial_state, strict=True)
        assert model.decoder_query_adapter is None
        assert model.super_grouper.radius == .2 and model.super_grouper.nsample == 2
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert len(parameters) == 16
        if arm == 'native':
            # Same addon initialization order as the completed 16-row preflight.
            sparse_addon = SparsePointSuperpointResidual().cuda()
            initial_sparse_state = {name: value.detach().cpu().clone()
                                    for name, value in sparse_addon.state_dict().items()}
        else:
            sparse_attachment = SparseSuperpointIntervention(model, sparse_addon)
            parameters.update({'sparse_point.' + name: value
                               for name, value in sparse_addon.named_parameters()})
        models[arm] = model
        trainable[arm] = parameters
        groups = [{'params': [p for name, p in parameters.items() if not name.startswith('sparse_point.')],
                   'lr': manifest['native_learning_rate']}]
        if arm == 'sparse':
            groups.append({'params': list(sparse_addon.parameters()), 'lr': manifest['sparse_learning_rate']})
        optimizers[arm] = torch.optim.AdamW(groups, lr=1e-5, weight_decay=args.weight_decay)
    assert len(trainable['sparse']) == 33
    assert list(trainable['native']) == list(trainable['sparse'])[:16]
    assert sum(p.numel() for p in trainable['native'].values()) == 1348960
    assert sum(p.numel() for p in trainable['sparse'].values()) == 1616896
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)

    def verify_state(before_training=False):
        changed = {}
        for arm, model in models.items():
            changed[arm] = []
            for name, value in model.state_dict().items():
                same = torch.equal(value.detach().cpu(), initial_state[name])
                if before_training or name not in trainable[arm]:
                    assert same, (arm, name)
                elif not same:
                    changed[arm].append(name)
            assert all(parameter.grad is None for name, parameter in model.named_parameters()
                       if name not in trainable[arm])
            if arm == 'sparse':
                for name, value in sparse_addon.state_dict().items():
                    same = torch.equal(value.detach().cpu(), initial_sparse_state[name])
                    if before_training:
                        assert same, name
                    elif not same:
                        changed[arm].append('sparse_point.' + name)
        return changed

    class SelectedDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in selected_ids]
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['sparse_row_id'] = np.int64(self.sparse_ids[index])
            return item

    base = SelectedDataset(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                           data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
                           detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                           skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(base.annos) == len(selected_ids)
    by_id = {}
    for row_id, anno in zip(selected_ids, base.annos):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id'] and anno['target_id'] == int(raw_rows[row_id]['target_id'])
        by_id[row_id] = anno
    datasets = {}
    for name, ids in partitions.items():
        dataset = copy.copy(base)
        dataset.annos = [by_id[i] for i in ids]
        dataset.sparse_ids = ids
        dataset.augment = False
        datasets[name] = dataset

    def loader(name, seed, batch_size, shuffle):
        return torch.utils.data.DataLoader(datasets[name], batch_size=batch_size, shuffle=shuffle,
                 num_workers=0, generator=torch.Generator().manual_seed(seed))

    def inputs_for(raw):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        return inputs, batch

    def grounding_snapshot(outputs):
        return {key: outputs[key].detach().clone() for key in ['last_center', 'last_pred_size', 'selected_source_scores']}

    def require_same_grounding(first, second):
        for key in first:
            assert torch.equal(first[key], second[key]), key


    started = time.time()
    inputs, batch = inputs_for(next(iter(loader('fit', 0, 4, False))))
    expected_warmup = json.loads(Path(manifest['v2_warmup']).read_text())
    point_hashes = [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                    for cloud in inputs['point_clouds']]
    assert point_hashes == expected_warmup['input_point_sha256']
    gradients, snapshots, gpu_norms, losses = {}, {}, {}, {}
    cases = [('native_warmup', 'native'), ('native_check', 'native'),
             ('sparse_check', 'sparse'), ('native_repeat', 'native'), ('sparse_repeat', 'sparse')]
    for label, arm in cases:
        model = models[arm]
        outputs = model(inputs)
        snapshot = grounding_snapshot(outputs)
        mask_snapshot = {}
        for name in ['sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']:
            for i, value in enumerate(outputs[name]):
                mask_snapshot[name + '_' + str(i)] = value.detach().cpu().clone()
        outputs.update(batch)
        loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        loss.backward()
        shared = {name: trainable[arm][name].grad.detach().clone() for name in trainable['native']}
        norms = {name: float(parameter.grad.norm()) for name, parameter in trainable[arm].items()}
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in trainable[arm].values())
        for name, value in norms.items():
            if name.startswith('sparse_point.') and name != 'sparse_point.output.weight':
                assert value == 0, name
            else:
                assert value > 0, name
        optimizers[arm].zero_grad(set_to_none=True)
        gradients[label] = {name: value.cpu() for name, value in shared.items()}
        gpu_norms[label] = {name: norms[name] for name in trainable['native']}
        snapshots[label] = dict({name: value.cpu() for name, value in snapshot.items()}, **mask_snapshot)
        losses[label] = float(loss)
        print('FULL START PROBE', json.dumps({'label': label, 'loss': float(loss),
            'gradient_norms_gpu': gpu_norms[label], 'elapsed_seconds': time.time() - started}), flush=True)
        del shared, snapshot, mask_snapshot, outputs, loss
    verify_state(before_training=True)
    assert point_hashes == [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                           for cloud in inputs['point_clouds']]
    verify_inputs()
    comparisons = {}
    pairs = [('native_warmup', 'native_check'), ('native_check', 'sparse_check'),
             ('native_check', 'native_repeat'), ('sparse_check', 'sparse_repeat'),
             ('native_repeat', 'sparse_repeat')]
    for first, second in pairs:
        parameters = {}
        for name, original in gradients[first].items():
            current = gradients[second][name]
            delta = current.double() - original.double()
            parameters[name] = {'shape': list(original.shape),
                'max_abs_difference': float(delta.abs().max()),
                'relative_l2_difference': float(delta.norm() / original.double().norm()),
                'allclose_atol1e6_rtol1e5': torch.allclose(original, current, atol=1e-6, rtol=1e-5),
                'gpu_norm_exact': gpu_norms[first][name] == gpu_norms[second][name],
                'reference_gpu_norm': gpu_norms[first][name], 'current_gpu_norm': gpu_norms[second][name]}
        comparisons[first + '__' + second] = {'parameters': parameters,
            'loss_equal': losses[first] == losses[second],
            'all_output_tensors_exact': all(torch.equal(snapshots[first][name], current)
                for name, current in snapshots[second].items())}
    artifact = addon / 'shared_gradients.pt'
    assert not artifact.exists()
    torch.save({'gradients': gradients, 'gpu_norms': gpu_norms, 'losses': losses,
                'input_point_sha256': point_hashes, 'manifest_sha256': file_sha(options.manifest)}, artifact)
    receipt = {'schema': 'mcln-sparse-full-start-probe-v1', 'status': 'complete',
        'manifest_sha256': file_sha(options.manifest), 'native_forwards': 5, 'backwards': 5,
        'optimizer_steps': 0, 'evaluated_holdout_rows': 0, 'formal_rows': 0,
        'full_dataset_rows_initialized': 32919, 'fit_rows_exercised': 4,
        'full_v2_initialization_preserved': True, 'state_and_inputs_unchanged': True,
        'input_point_sha256': point_hashes, 'losses': losses, 'comparisons': comparisons,
        'gradient_artifact_sha256': file_sha(artifact), 'gradient_artifact_bytes': artifact.stat().st_size,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'elapsed_seconds': time.time() - started}
    write_json(addon / 'receipt.json', receipt)
    sparse_attachment.remove()
    print('FULL START PROBE COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
