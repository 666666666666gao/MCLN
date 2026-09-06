"""Independent CPU audit of either outcome of the fixed ScanRefer pair."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows):
    result = {'rows': len(rows),
              'mask_miou': sum(row['mask_iou'] for row in rows) * 100. / len(rows)}
    for field, prefix in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
        values = [row[field] for row in rows]
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in values)
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            result[prefix + '_hits' + suffix] = sum(value > threshold for value in values)
    return result


def compare(reference, candidate):
    """Resample physical rooms, retaining all expressions in each sampled room."""
    assert len(reference) == len(candidate) and reference
    clusters = {}
    transitions = np.zeros((3, 3), dtype=np.int64)
    pairs = []
    for old, new in zip(reference, candidate):
        for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
            assert old[key] == new[key], (old['row_id'], key)
        assert old['physical_space'] == old['scan_id'].split('_')[0]
        delta = [1]
        for field in ['rec_iou', 'mask_iou']:
            delta.extend(int(new[field] > t) - int(old[field] > t) for t in [.25, .5])
        delta.append(new['mask_iou'] - old['mask_iou'])
        cluster = clusters.setdefault(old['physical_space'], np.zeros(6))
        cluster += delta
        old_band = int(old['rec_iou'] > .25) + int(old['rec_iou'] > .5)
        new_band = int(new['rec_iou'] > .25) + int(new['rec_iou'] > .5)
        transitions[old_band, new_band] += 1
        pairs.append((old, new))
    effects = {}
    for field, prefix in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            repair = sum(a[field] <= threshold < b[field] for a, b in pairs)
            damage = sum(b[field] <= threshold < a[field] for a, b in pairs)
            effects[prefix + suffix] = {'repair': repair, 'damage': damage,
                                      'net': repair - damage}
    values = np.asarray([clusters[name] for name in sorted(clusters)])
    rng = np.random.RandomState(0)
    draws = values[rng.randint(0, len(values), (2000, len(values)))].sum(axis=1)
    intervals = np.percentile(draws[:, 1:] / draws[:, :1] * 100., [2.5, 97.5], axis=0)
    return {'rows': len(pairs), 'physical_spaces': len(clusters), 'effects': effects,
            'mask_miou_delta_pp': sum(b['mask_iou'] - a['mask_iou'] for a, b in pairs)
                                  * 100. / len(pairs),
            'bootstrap': {'unit': 'physical_space', 'draws': 2000, 'seed': 0,
                          'intervals_95_percent_pp': dict(zip(
                              ['rec025', 'rec050', 'mask025', 'mask050', 'mask_miou'],
                              intervals.T.tolist())), 'is_promotion_gate': False},
            'rec_iou_transition_counts': transitions.tolist(),
            'transition_bands': ['[0,0.25]', '(0.25,0.50]', '(0.50,1]'],
            'selected_instance_identity_available': False}


def audit_rows(directory):
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert (directory / 'controller.exit').read_text().strip() == '0'
    assert receipt['schema'] == 'mcln-scanrefer-joint-readout-pair-v1'
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
    assert stages['baseline']['detached'] == stages['baseline']['joint']
    actual_metrics = {}
    for stage, records in stages.items():
        actual_metrics[stage] = {}
        for arm in ['detached', 'joint']:
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
        'joint_minus_baseline': compare(baseline['joint'], terminal['joint']),
        'joint_minus_detached': compare(terminal['detached'], terminal['joint']),
        'detached_minus_baseline': compare(baseline['detached'], terminal['detached']),
    }
    for name, reference in [('joint_minus_baseline', 'baseline'), ('joint_minus_detached', 'detached')]:
        for suffix, threshold in [('025', '0.25'), ('050', '0.5')]:
            assert comparisons[name]['effects']['rec' + suffix] == receipt['joint_rec_effects'][reference][threshold]
    eligible = all(comparisons[name]['effects']['rec' + suffix]['net'] >= 0
                   for name in ['joint_minus_baseline', 'joint_minus_detached']
                   for suffix in ['025', '050'])
    assert receipt['eligible_for_fixed_terminal_formal_evaluation'] == eligible
    return {'integrity_pass': True, 'metrics': actual_metrics, 'comparisons': comparisons,
            'eligible_for_fixed_terminal_formal_evaluation': eligible,
            'fit_traversal_and_paired_point_identity_verified': True,
            'heldout_backbone_has_seen_scenes': True, 'formal_rows': 0}


def audit_checkpoints(directory):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    import torch

    manifest = json.loads((directory / 'input_manifest.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    fit_complete = json.loads((directory / 'fit_complete.json').read_text())
    assert fit_complete['checkpoints'] == receipt['checkpoints']
    assert fit_complete['steps_per_arm'] == 2482
    source = Path(manifest['model_source'])
    assert file_sha(source / 'g0_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    for item in manifest['artifacts'].values():
        assert file_sha(item['path']) == item['sha256']
    parent = torch.load(manifest['artifacts']['backbone']['path'], map_location='cpu')
    initial = {name[7:]: value for name, value in parent['model'].items()}
    original_readouts = {name: torch.load(item['path'], map_location='cpu')
                         for name, item in manifest['artifacts'].items() if name != 'backbone'}
    core_names = protocol['core_trainable_tensors']
    readout_names = protocol['readout_trainable_tensors']
    assert len(core_names) == 68 and len(readout_names) == 42
    # Both actual native-probe batches observed these exact unused parameters.
    unused = {'decoder.5.norm1.weight', 'decoder.5.norm1.bias'}
    probe = json.loads(Path(manifest['native_probe_receipt']).read_text())
    assert file_sha(manifest['native_probe_receipt']) == manifest['native_probe_receipt_sha256']
    for observation in probe['observations']:
        for field in ['native_core_gradients', 'joint_readout_core_gradients']:
            assert {name for name, value in observation[field].items() if value is None} == unused
    output = {}
    for arm in ['detached', 'joint']:
        metadata = receipt['checkpoints'][arm]
        path = directory / (arm + '_joint_state.pt')
        assert str(path) == metadata['path']
        assert path.stat().st_size == metadata['bytes'] and file_sha(path) == metadata['sha256']
        checkpoint = torch.load(path, map_location='cpu')
        assert checkpoint['schema'] == 'mcln-scanrefer-joint-rec-trained-state-v1'
        assert checkpoint['arm'] == arm and checkpoint['steps'] == 2482
        assert checkpoint['manifest_sha256'] == receipt['manifest_sha256']
        assert checkpoint['pretrained_artifacts'] == manifest['artifacts']
        assert checkpoint['core_trainable_tensors'] == core_names
        assert set(checkpoint['model']) == set(initial)
        changes = {}
        for name, value in checkpoint['model'].items():
            assert value.shape == initial[name].shape and value.dtype == initial[name].dtype
            assert torch.isfinite(value).all(), (arm, name)
            same = torch.equal(value, initial[name])
            if name not in core_names or name in unused:
                assert same, (arm, name)
            elif not same:
                changes[name] = float((value - initial[name]).abs().max())
        assert set(changes) == set(receipt['changed_core_tensors'][arm]) == set(fit_complete['changed_core_tensors'][arm])
        assert changes
        readout_changes, parameters = {}, {name: checkpoint['model'][name] for name in core_names}
        for name, original in original_readouts.items():
            current = checkpoint['readout'][name]
            assert set(current) == set(original)
            for key in original:
                if key in ['feature_mean', 'feature_std']:
                    assert current[key].dtype == original[key].dtype
                    assert torch.equal(current[key], original[key]), (name, key)
                elif key != 'model_state_dict':
                    assert json.dumps(current[key], sort_keys=True) == json.dumps(original[key], sort_keys=True), (name, key)
            before, after = original['model_state_dict'], current['model_state_dict']
            assert set(before) == set(after)
            readout_changes[name] = []
            for key, value in after.items():
                assert value.shape == before[key].shape and value.dtype == before[key].dtype
                assert torch.isfinite(value).all(), (arm, name, key)
                parameters['scorers.' + name + '.' + key] = value
                if not torch.equal(value, before[key]):
                    readout_changes[name].append(key)
            assert readout_changes[name]
        optimizer = checkpoint['optimizer']
        groups = optimizer['param_groups']
        assert len(groups) == 2 and [item['lr'] for item in groups] == [1e-6, 1e-5]
        assert all(item['weight_decay'] == .0005 and tuple(item['betas']) == (.9, .999)
                   and item['eps'] == 1e-8 for item in groups)
        assert groups[0]['params'] == list(range(68)) and groups[1]['params'] == list(range(68, 110))
        names = core_names + readout_names
        expected = {index for index, name in enumerate(names) if name not in unused}
        assert set(optimizer['state']) == expected
        for index, state in optimizer['state'].items():
            assert float(state['step']) == 2482, (arm, names[index], state['step'])
            for key in ['exp_avg', 'exp_avg_sq']:
                assert state[key].shape == parameters[names[index]].shape
                assert torch.isfinite(state[key]).all(), (arm, names[index], key)
        output[arm] = {'checkpoint_sha256': metadata['sha256'], 'bytes': metadata['bytes'],
                       'changed_core_max_abs': changes, 'changed_readout_tensors': readout_changes,
                       'optimizer_parameter_tensors': len(expected), 'optimizer_steps': 2482,
                       'unused_parameters_unchanged': sorted(unused),
                       'frozen_core_parameters_and_buffers_unchanged': True,
                       'readout_metadata_unchanged': True}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    directory = options.directory.resolve()
    result = audit_rows(directory)
    result['checkpoints'] = audit_checkpoints(directory)
    result.update(schema='mcln-scanrefer-joint-pair-independent-audit-v1',
                  receipt_sha256=file_sha(directory / 'receipt.json'),
                  audit_script_sha256=file_sha(__file__), gpu_forwards=0,
                  optimizer_steps_executed=0, python=sys.version.split()[0])
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['integrity_pass', 'metrics',
                      'eligible_for_fixed_terminal_formal_evaluation']}))
