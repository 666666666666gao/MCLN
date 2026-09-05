"""Independent CPU checks of the completed V2 parameters and AdamW artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(directory):
    m = json.loads((directory / 'input_manifest.json').read_text())
    r = json.loads((directory / 'receipt.json').read_text())
    assert (directory / 'controller.exit').read_text().strip() == '0'
    assert r['schema'] == 'mcln-nr3d-sparse-point-pair-v2' and r['status'] == 'complete'
    assert r['optimizer_steps_per_arm'] == 6687 and r['formal_rows'] == 0
    assert file_sha(directory / 'input_manifest.json') == r['manifest_sha256']
    for name, digest in m['files'].items():
        assert file_sha(directory / name) == digest, name
    source = Path(m['model_source'])
    assert file_sha(source / 'g0_source_manifest.json') == m['source_manifest_sha256']
    for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    assert file_sha(Path(m['checkpoint'])) == m['checkpoint_sha256']
    assert file_sha(directory / 'baseline_rows.json') == r['baseline_rows_sha256']
    assert file_sha(directory / 'terminal_rows.json') == r['terminal_rows_sha256']
    assert file_sha(directory / 'fit_point_batches.json') == r['fit_point_batches_sha256']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    parent = torch.load(m['checkpoint'], map_location='cpu')
    initial = {name[7:]: value for name, value in parent['model'].items()}
    shared_set = set(m['shared_parameter_names'])
    # Averaging sorted parent keys; optimizer order follows model registration.
    shared_names = [prefix + '.' + str(layer) + '.' + kind
        for prefix, layers in [('rel_encoder', (0, 2)), ('x_mask', (0, 2, 4)), ('x_query', (0, 2, 4))]
        for layer in layers for kind in ['weight', 'bias']]
    assert len(shared_names) == 16 and set(shared_names) == shared_set
    model = SparsePointSuperpointResidual()
    new_shapes = {'sparse_point.' + name: tuple(p.shape) for name, p in model.named_parameters()}
    assert len(new_shapes) == 17
    output = {}
    for arm in ['native', 'sparse']:
        metadata = r['artifacts'][arm]
        path = Path(metadata['path'])
        assert path == directory / (arm + '_mask_state.pt')
        assert path.stat().st_size == metadata['bytes'] and file_sha(path) == metadata['sha256']
        artifact = torch.load(path, map_location='cpu')
        assert artifact['arm'] == arm and artifact['steps'] == 6687
        assert artifact['parent_checkpoint_sha256'] == m['checkpoint_sha256']
        parameters = artifact['mask_projection_state']
        expected = shared_names + (list(new_shapes) if arm == 'sparse' else [])
        assert list(parameters) == expected
        shared_changes = {}
        for name in shared_names:
            current = parameters[name]
            assert current.shape == initial[name].shape and torch.isfinite(current).all()
            assert not torch.equal(current, initial[name]), name
            shared_changes[name] = float((current - initial[name]).abs().max())
        if arm == 'sparse':
            for name, shape in new_shapes.items():
                assert tuple(parameters[name].shape) == shape and torch.isfinite(parameters[name]).all(), name
            assert parameters['sparse_point.output.weight'].count_nonzero() > 0
        optimizer = artifact['optimizer']
        groups = optimizer['param_groups']
        rates = [1e-5] if arm == 'native' else [1e-5, 1e-4]
        assert len(groups) == len(rates)
        assert [g['lr'] for g in groups] == rates
        assert all(g['weight_decay'] == .0005 and tuple(g['betas']) == (.9, .999) and g['eps'] == 1e-8 for g in groups)
        parameter_ids = [index for group in groups for index in group['params']]
        assert parameter_ids == list(range(len(expected)))
        assert len(groups[0]['params']) == 16
        assert set(optimizer['state']) == set(parameter_ids)
        for index, name in zip(parameter_ids, expected):
            state = optimizer['state'][index]
            assert float(state['step']) == 6687
            for key in ['exp_avg', 'exp_avg_sq']:
                assert state[key].shape == parameters[name].shape
                assert torch.isfinite(state[key]).all()
        output[arm] = {'artifact_sha256': metadata['sha256'], 'bytes': metadata['bytes'],
            'parameters': sum(p.numel() for p in parameters.values()), 'parameter_tensors': len(parameters),
            'shared_max_abs_changes': shared_changes, 'optimizer_steps': 6687,
            'optimizer_shapes_and_finiteness_pass': True, 'learning_rates': rates}
    return {'schema': 'mcln-sparse-point-artifact-verification-v1', 'status': 'pass',
        'manifest_sha256': r['manifest_sha256'], 'artifacts': output,
        'shared_parameter_changes_independently_verified': True,
        'new_non_output_initial_deltas_independently_verified': False,
        'new_non_output_change_evidence': 'Runtime initial-state comparison recorded in terminal receipt.',
        'gpu_forwards': 0, 'optimizer_steps_executed': 0, 'scientific_promotion_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = verify(args.directory.resolve())
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(result))
