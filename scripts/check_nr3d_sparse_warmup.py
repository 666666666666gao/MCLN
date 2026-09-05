"""Three first-batch backward comparisons with one warmup; no optimizer or quality evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    directory = options.manifest.parent
    m = json.loads(options.manifest.read_text())
    source = Path(m['model_source'])

    def verify():
        assert sha(source / 'g0_source_manifest.json') == m['source_manifest_sha256']
        for name, value in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
            assert sha(source / name) == value, name
        for name, value in m['files'].items():
            assert sha(directory / name) == value, name
        for name, value in m['data_files'].items():
            assert sha(Path(name)) == value['sha256'], name
        assert sha(Path(m['checkpoint'])) == m['checkpoint_sha256']
        assert sha(Path(m['m3_receipt'])) == m['m3_receipt_sha256']
        for name, value in m['runtime_receipts'].items():
            assert sha(Path(name)) == value, name

    verify()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual, SparseSuperpointIntervention
    from scripts.run_nr3d_view_pair_role import read_train_rows
    assert sys.prefix == m['python_prefix']
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parent = torch.load(m['checkpoint'], map_location='cpu')
    assert parent['evaluation_only'] and 'optimizer' not in parent
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(parent['config']))
    state = {name[7:]: value for name, value in parent['model'].items()}
    del parent
    models = {}
    shared = {}
    for label in ['A', 'B']:
        model = TrainTester.get_model(args).cuda().eval()
        model.load_state_dict(state, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('x_query.', 'x_mask.', 'rel_encoder.')))
        models[label] = model
        shared[label] = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert len(shared[label]) == 16
        if label == 'A':
            addon = SparsePointSuperpointResidual().cuda()
            addon_state = {name: p.detach().cpu().clone() for name, p in addon.state_dict().items()}
    criterion, set_criterion = TrainTester.get_criterion(args)
    row_ids = m['fit_row_ids'][:4]
    assert row_ids == [0, 1, 3, 4]
    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))

    class FirstBatch(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FirstBatch(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
        data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
        detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
        skip_missing_superpoints=args.skip_missing_superpoints)
    dataset.augment = False
    assert len(dataset) == 4
    for row_id, anno in zip(row_ids, dataset.annos):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id']
        assert anno['target_id'] == int(raw_rows[row_id]['target_id'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    batch = TrainTester._to_gpu(next(iter(loader)))
    inputs = TrainTester._get_inputs(batch)
    inputs['train'] = False
    m3 = json.loads(Path(m['m3_receipt']).read_text())
    expected = {row['fit_row_id']: row['input_point_cloud_sha256']
        for item in m3['batches'] for row in item['rows']}
    point_hashes = [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest() for cloud in inputs['point_clouds']]
    assert point_hashes == [expected[i] for i in row_ids]
    snapshots, gradients, losses, gpu_norms = {}, {}, {}, {}

    def capture(label, model_label):
        model = models[model_label]
        model.zero_grad(set_to_none=True)
        addon.zero_grad(set_to_none=True)
        output = model(inputs)
        snapshot = {}
        for name in ['last_center', 'last_pred_size', 'selected_source_scores']:
            snapshot[name] = output[name].detach().cpu().clone()
        for name in ['sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']:
            for i, value in enumerate(output[name]):
                snapshot[name + '_' + str(i)] = value.detach().cpu().clone()
        output.update(batch)
        loss, output = TrainTester._compute_loss(output, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        loss.backward()
        gpu_norms[label] = {name: float(p.grad.norm()) for name, p in shared[model_label].items()}
        gradients[label] = {name: p.grad.detach().cpu().clone() for name, p in shared[model_label].items()}
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients[label].values())
        losses[label] = float(loss)
        snapshots[label] = snapshot
        print('ZERO GRADIENT PROBE', json.dumps({'label': label, 'loss': float(loss),
            'gradient_norms_gpu': gpu_norms[label]}), flush=True)
        model.zero_grad(set_to_none=True)
        addon.zero_grad(set_to_none=True)

    capture('plain_A_first', 'A')
    capture('plain_A_repeat', 'A')
    attachment = SparseSuperpointIntervention(models['B'], addon)
    capture('sparse_zero_B', 'B')
    attachment.remove()

    reference = 'plain_A_repeat'
    comparisons = {}
    for label in ['plain_A_first', 'sparse_zero_B']:
        per_parameter = {}
        for name, original in gradients[reference].items():
            current = gradients[label][name]
            delta = current.double() - original.double()
            per_parameter[name] = {'shape': list(original.shape),
                'exact': torch.equal(original, current),
                'different_elements': int((original != current).sum()),
                'max_abs_difference': float(delta.abs().max()),
                'relative_l2_difference': float(delta.norm() / original.double().norm()),
                'reference_norm_float32_cpu': float(original.norm()),
                'current_norm_float32_cpu': float(current.norm()),
                'reference_norm_gpu': gpu_norms[reference][name],
                'current_norm_gpu': gpu_norms[label][name],
                'gpu_norm_exact': gpu_norms[reference][name] == gpu_norms[label][name],
                'allclose_atol1e6_rtol1e5': torch.allclose(original, current, atol=1e-6, rtol=1e-5)}
        comparisons[label] = {'loss_equal': losses[label] == losses[reference],
            'all_output_tensors_exact': all(torch.equal(snapshots[reference][name], value)
                                          for name, value in snapshots[label].items()),
            'parameters': per_parameter}
    regression = comparisons['sparse_zero_B']
    assert regression['loss_equal'] and regression['all_output_tensors_exact']
    assert all(p['gpu_norm_exact'] and p['allclose_atol1e6_rtol1e5'] for p in regression['parameters'].values())
    for model in models.values():
        assert all(torch.equal(p.detach().cpu(), state[name]) for name, p in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(p.detach().cpu(), addon_state[name]) for name, p in addon.state_dict().items())
    assert [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest() for cloud in inputs['point_clouds']] == point_hashes
    verify()
    artifact = directory / 'shared_gradients.pt'
    assert not artifact.exists()
    torch.save({'gradients': gradients, 'losses': losses, 'gpu_norms': gpu_norms, 'input_point_sha256': point_hashes,
        'manifest_sha256': sha(options.manifest)}, artifact)
    result = {'schema': 'mcln-sparse-warmup-regression-v1', 'status': 'complete',
        'manifest_sha256': sha(options.manifest), 'native_forwards': 3, 'backwards': 3,
        'fit_rows': 4, 'fit_row_ids': row_ids, 'point_hashes_match_m3': True,
        'optimizer_steps': 0, 'holdout_rows': 0, 'formal_rows': 0,
        'original_state_and_inputs_unchanged': True, 'losses': losses, 'comparisons': comparisons,
        'gradient_artifact_sha256': sha(artifact), 'gradient_artifact_bytes': artifact.stat().st_size,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    with (directory / 'receipt.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print('ZERO GRADIENT PROBE COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
