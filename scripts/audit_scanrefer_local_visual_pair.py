"""Independent CPU audit of the fixed candidate-local ScanRefer experiment.

Run as a module: CUDA_VISIBLE_DEVICES='' python -m scripts.audit_scanrefer_local_visual_pair RUN OUTPUT
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

from scripts.audit_scanrefer_joint_readout_pair import (
    assert_metadata_equal, compare, file_sha, metrics,
)


def audit_rows(directory):
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert (directory / 'controller.exit').read_text().strip() == '0'
    assert receipt['schema'] == 'mcln-scanrefer-local-visual-pair-v1'
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0
    assert receipt['steps_per_arm'] == manifest['steps_per_arm'] == 2482
    assert manifest['epochs'] == 1 and manifest['batch_size'] == 12
    assert receipt['fit_rows'] == 29778 and receipt['holdout_rows'] == 6887
    assert receipt['manifest_sha256'] == file_sha(directory / 'input_manifest.json')
    for field, name in [('baseline_rows_sha256', 'baseline_rows.json'),
                        ('terminal_rows_sha256', 'terminal_rows.json'),
                        ('fit_batches_sha256', 'fit_point_batches.json')]:
        assert receipt[field] == file_sha(directory / name)
    assert protocol['source_protocol_sha256'] == manifest['split_protocol_sha256']
    assert file_sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    original_split = json.loads(Path(manifest['split_protocol']).read_text())
    assert protocol['row_ids'] == original_split['row_ids']
    fit_ids, holdout_ids = protocol['row_ids']['fit'], protocol['row_ids']['holdout']
    assert len(fit_ids) == 29778 and len(holdout_ids) == 6887
    assert len(set(fit_ids + holdout_ids)) == 36665
    spaces = protocol['physical_spaces']
    assert len(spaces['fit']) == 456 and len(spaces['holdout']) == 106
    assert not set(spaces['fit']).intersection(spaces['holdout'])
    batches = json.loads((directory / 'fit_point_batches.json').read_text())
    assert [item['step'] for item in batches] == list(range(1, 2483))
    assert Counter(row_id for item in batches for row_id in item['row_ids']) == Counter(fit_ids)
    assert all(len(item['row_ids']) == len(item['point_sha256']) == 12 for item in batches[:-1])
    assert len(batches[-1]['row_ids']) == len(batches[-1]['point_sha256']) == 6
    stages = {stage: json.loads((directory / (stage + '_rows.json')).read_text())
              for stage in ['baseline', 'terminal']}
    assert stages['baseline']['control'] == stages['baseline']['local']
    actual_metrics = {}
    for stage, records in stages.items():
        actual_metrics[stage] = {}
        for arm in ['control', 'local']:
            rows = records[arm]
            assert [row['row_id'] for row in rows] == holdout_ids
            assert sorted({row['physical_space'] for row in rows}) == spaces['holdout']
            value = metrics(rows)
            actual_metrics[stage][arm] = value
            recorded = receipt[stage + '_metrics'][arm]
            for key in value:
                if key == 'mask_miou':
                    # Different CPU Python summation versions already differed by 8.5e-14 pp.
                    assert abs(value[key] - recorded[key]) < 1e-8
                else:
                    assert value[key] == recorded[key], (stage, arm, key)
    baseline, terminal = stages['baseline'], stages['terminal']
    comparisons = {
        'local_minus_baseline': compare(baseline['local'], terminal['local']),
        'local_minus_control': compare(terminal['control'], terminal['local']),
        'control_minus_baseline': compare(baseline['control'], terminal['control']),
    }
    for name, reference in [('local_minus_baseline', 'baseline'), ('local_minus_control', 'control')]:
        for suffix, threshold in [('025', '0.25'), ('050', '0.5')]:
            assert comparisons[name]['effects']['rec' + suffix] == receipt['local_rec_effects'][reference][threshold]
    eligible = all(comparisons[name]['effects']['rec' + suffix]['net'] >= 0
                   for name in ['local_minus_baseline', 'local_minus_control']
                   for suffix in ['025', '050'])
    assert receipt['development_dual_rec_nonregression'] == eligible
    assert manifest['mode'] == 'train' and manifest['readouts_frozen']
    assert manifest['loss'] == 'native_gt_only' and receipt['readouts_unchanged']
    assert receipt['fixed_endpoint_ready_for_official_evaluation']
    return {'integrity_pass': True, 'metrics': actual_metrics, 'comparisons': comparisons,
            'development_dual_rec_nonregression': eligible,
            'fit_traversal_and_paired_point_identity_verified': True,
            'heldout_backbone_has_seen_scenes': True, 'formal_rows': 0}


def check_model_state(actual, initial, trainable, unused):
    import torch

    assert set(actual) == set(initial)
    changes = {}
    for name, value in actual.items():
        reference = initial[name]
        assert value.shape == reference.shape and value.dtype == reference.dtype, name
        assert torch.isfinite(value).all(), name
        same = torch.equal(value, reference)
        if name not in trainable or name in unused:
            assert same, name
        elif not same:
            changes[name] = float((value - reference).abs().max())
    return changes


def check_readouts(actual, original):
    import torch

    assert set(actual) == set(original)
    for name, reference in original.items():
        current = actual[name]
        assert set(current) == set(reference)
        for key in reference:
            if key == 'model_state_dict':
                # export_artifacts serializes this mapping as a plain dict.
                assert set(current[key]) == set(reference[key])
                for parameter, value in reference[key].items():
                    saved = current[key][parameter]
                    assert saved.shape == value.shape and saved.dtype == value.dtype
                    assert torch.equal(saved, value), (name, parameter)
            else:
                assert_metadata_equal(current[key], reference[key])


def check_optimizer(optimizer, model_state, core_names, local_names, unused, steps):
    import torch

    groups = optimizer['param_groups']
    expected_lrs = [1e-6, 1e-4] if local_names else [1e-6]
    assert len(groups) == len(expected_lrs)
    assert [group['lr'] for group in groups] == expected_lrs
    assert groups[0]['params'] == list(range(len(core_names)))
    if local_names:
        assert groups[1]['params'] == list(range(len(core_names), len(core_names) + len(local_names)))
    assert all(group['weight_decay'] == .0005 and tuple(group['betas']) == (.9, .999)
               and group['eps'] == 1e-8 for group in groups)
    names = core_names + local_names
    expected = {index for index, name in enumerate(names) if name not in unused}
    assert set(optimizer['state']) == expected
    for index, state in optimizer['state'].items():
        name = names[index]
        assert float(state['step']) == steps, (name, state['step'])
        for field in ['exp_avg', 'exp_avg_sq']:
            assert state[field].shape == model_state[name].shape, (name, field)
            assert state[field].dtype == model_state[name].dtype, (name, field)
            assert torch.isfinite(state[field]).all(), (name, field)
    return len(expected)


def reconstruct_initial_states(manifest):
    """Replay the trainer's CPU parameter construction before any data loading.

    The trainer's subsequent .cuda(), strict load and deepcopy do not consume
    the CPU RNG used to initialize CandidateLocalVisual.
    """
    import torch

    source = Path(manifest['model_source'])
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import scripts
    scripts.__path__ = [str(source / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from models.candidate_local_visual import CandidateLocalVisual

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    command = build_authoritative_command(source / 'unused_audit_output')
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls
    payload = torch.load(args.checkpoint_path, map_location='cpu')
    initial = {name[7:]: value for name, value in payload['model'].items()}
    model = TrainTester.get_model(args).eval()
    model.load_state_dict(initial, strict=True)
    core_names = [name for name, parameter in model.named_parameters()
                  if name.startswith(('decoder.5.', 'prediction_heads.5.'))]
    local = CandidateLocalVisual()
    local_initial = {'decoder.5.local_visual.' + name: value.detach().clone()
                     for name, value in local.state_dict().items()}
    assert len(core_names) == 68 and len(local_initial) == 10
    assert sum(value.numel() for value in local_initial.values()) == 145008
    return initial, local_initial, core_names


def audit_checkpoints(directory):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    import torch

    manifest = json.loads((directory / 'input_manifest.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    fit = json.loads((directory / 'fit_complete.json').read_text())
    assert fit['checkpoints'] == receipt['checkpoints']
    assert fit['steps_per_arm'] == 2482
    source = Path(manifest['model_source'])
    assert file_sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    for item in manifest['artifacts'].values():
        assert file_sha(item['path']) == item['sha256']
    assert file_sha(manifest['native_probe_receipt']) == manifest['native_probe_receipt_sha256']
    probe = json.loads(Path(manifest['native_probe_receipt']).read_text())
    assert probe['status'] == 'pass' and probe['disposable_optimizer_steps'] == 2
    initial, local_initial, core_names = reconstruct_initial_states(manifest)
    assert core_names == protocol['core_trainable_tensors'] == probe['core_trainable_tensors']
    assert list(local_initial) == protocol['local_trainable_tensors'] == probe['local_trainable_tensors']
    original_readouts = {name: torch.load(item['path'], map_location='cpu')
                         for name, item in manifest['artifacts'].items() if name != 'backbone'}
    # norm1 is defined but unused in the actual BiDecoderLayer forward.
    unused = {'decoder.5.norm1.weight', 'decoder.5.norm1.bias'}
    result = {}
    for arm in ['control', 'local']:
        metadata = receipt['checkpoints'][arm]
        path = directory / (arm + '_local_visual_state.pt')
        assert str(path) == metadata['path'] and path.stat().st_size == metadata['bytes']
        assert file_sha(path) == metadata['sha256']
        checkpoint = torch.load(path, map_location='cpu')
        assert checkpoint['schema'] == 'mcln-scanrefer-local-visual-trained-state-v1'
        assert checkpoint['arm'] == arm and checkpoint['steps'] == 2482
        assert checkpoint['manifest_sha256'] == receipt['manifest_sha256']
        assert checkpoint['pretrained_artifacts'] == manifest['artifacts']
        assert checkpoint['model_config'] == {'candidate_local_visual': arm == 'local'}
        assert checkpoint['core_trainable_tensors'] == core_names
        local_names = list(local_initial) if arm == 'local' else []
        assert checkpoint['local_trainable_tensors'] == local_names
        expected = dict(initial)
        if arm == 'local':
            expected.update(local_initial)
        changes = check_model_state(checkpoint['model'], expected, set(core_names + local_names), unused)
        assert changes
        assert set(changes) == set(receipt['changed_core_tensors'][arm]) == set(fit['changed_core_tensors'][arm])
        check_readouts(checkpoint['readout'], original_readouts)
        tensors = check_optimizer(checkpoint['optimizer'], checkpoint['model'], core_names, local_names, unused, 2482)
        result[arm] = {'checkpoint_sha256': metadata['sha256'], 'bytes': metadata['bytes'],
            'changed_parameter_max_abs': changes, 'optimizer_parameter_tensors': tensors,
            'optimizer_steps': 2482, 'frozen_core_and_buffers_unchanged': True,
            'readout_parameters_and_metadata_unchanged': True,
            'local_initial_reference': 'same seed0 CPU factory sequence, no GPU forward'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    directory = options.directory.resolve()
    output = options.output.resolve()
    result = audit_rows(directory)
    result['checkpoints'] = audit_checkpoints(directory)
    result.update(schema='mcln-scanrefer-local-visual-independent-audit-v1',
        receipt_sha256=file_sha(directory / 'receipt.json'), audit_script_sha256=file_sha(__file__),
        gpu_forwards=0, optimizer_steps_executed=0, python=sys.version.split()[0])
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['integrity_pass', 'metrics', 'development_dual_rec_nonregression']}))
