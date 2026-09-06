"""Independent rows, frozen artifacts and optimizer audit of the compatibility pair."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from scripts.audit_scanrefer_joint_readout_pair import file_sha, metrics, compare, assert_metadata_equal
from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
from scripts.audit_scanrefer_frozen_readout_official import native_metrics, rec_compare

def audit_rows(directory):
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert (directory / 'controller.exit').read_text().strip() == '0'
    assert receipt['schema'] == 'mcln-scanrefer-frozen-readout-pair-v1'
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
    assert stages['baseline']['native_only'] == stages['baseline']['frozen_gt']
    actual_metrics, actual_native = {}, {}
    native_stages = {}
    for stage, records in stages.items():
        actual_metrics[stage] = {}
        native_stages[stage], actual_native[stage] = {}, {}
        for arm in ['native_only', 'frozen_gt']:
            rows = records[arm]
            assert [row['row_id'] for row in rows] == holdout_ids
            assert sorted({row['physical_space'] for row in rows}) == spaces['holdout']
            value = metrics(rows)
            assert all(0 <= row['native_query_index'] < 256 for row in rows)
            native_rows = [dict(row, rec_iou=row['native_rec_iou']) for row in rows]
            native_stages[stage][arm] = native_rows
            actual_native[stage][arm] = native_metrics(native_rows)
            recorded_native = json.loads((directory / (stage + '_native_metrics.json')).read_text())
            assert actual_native[stage][arm] == recorded_native[arm]
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
        'frozen_gt_minus_baseline': compare(baseline['frozen_gt'], terminal['frozen_gt']),
        'frozen_gt_minus_native_only': compare(terminal['native_only'], terminal['frozen_gt']),
        'native_only_minus_baseline': compare(baseline['native_only'], terminal['native_only']),
    }
    for name, reference in [('frozen_gt_minus_baseline', 'baseline'), ('frozen_gt_minus_native_only', 'native_only')]:
        for suffix, threshold in [('025', '0.25'), ('050', '0.5')]:
            assert comparisons[name]['effects']['rec' + suffix] == receipt['frozen_readout_rec_effects'][reference][threshold]
    eligible = all(comparisons[name]['effects']['rec' + suffix]['net'] >= 0
                   for name in ['frozen_gt_minus_baseline', 'frozen_gt_minus_native_only']
                   for suffix in ['025', '050'])
    assert receipt['eligible_for_fixed_terminal_formal_evaluation'] == eligible
    native_comparisons = {
        'frozen_gt_minus_baseline': rec_compare(native_stages['baseline']['frozen_gt'], native_stages['terminal']['frozen_gt']),
        'frozen_gt_minus_native_only': rec_compare(native_stages['terminal']['native_only'], native_stages['terminal']['frozen_gt']),
        'native_only_minus_baseline': rec_compare(native_stages['baseline']['native_only'], native_stages['terminal']['native_only']),
    }
    return {'integrity_pass': True, 'metrics': actual_metrics, 'comparisons': comparisons,
            'native_rec_metrics': actual_native, 'native_rec_comparisons': native_comparisons,
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
    assert file_sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
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
    assert len(core_names) == 68 and len(readout_names) == 0
    # Both actual native-probe batches observed these exact unused parameters.
    unused = {'decoder.5.norm1.weight', 'decoder.5.norm1.bias'}
    probe = json.loads(Path(manifest['native_probe_receipt']).read_text())
    assert file_sha(manifest['native_probe_receipt']) == manifest['native_probe_receipt_sha256']
    for observation in probe['observations']:
        for field in ['native_norm', 'frozen_readout_norm']:
            assert {name for name, value in observation['parameter_gradients'].items() if value[field] is None} == unused
    output = {}
    for arm in ['native_only', 'frozen_gt']:
        metadata = receipt['checkpoints'][arm]
        path = directory / (arm + '_frozen_readout_state.pt')
        assert str(path) == metadata['path']
        assert path.stat().st_size == metadata['bytes'] and file_sha(path) == metadata['sha256']
        checkpoint = torch.load(path, map_location='cpu')
        assert checkpoint['schema'] == 'mcln-scanrefer-frozen-readout-trained-state-v1'
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
        parameters = {name: checkpoint['model'][name] for name in core_names}
        assert_metadata_equal(checkpoint['readout'], original_readouts)
        optimizer = checkpoint['optimizer']
        groups = optimizer['param_groups']
        assert len(groups) == 1 and groups[0]['lr'] == 1e-6
        assert all(item['weight_decay'] == .0005 and tuple(item['betas']) == (.9, .999)
                   and item['eps'] == 1e-8 for item in groups)
        assert groups[0]['params'] == list(range(68))
        names = core_names
        expected = {index for index, name in enumerate(names) if name not in unused}
        assert set(optimizer['state']) == expected
        for index, state in optimizer['state'].items():
            assert float(state['step']) == 2482, (arm, names[index], state['step'])
            for key in ['exp_avg', 'exp_avg_sq']:
                assert state[key].shape == parameters[names[index]].shape
                assert torch.isfinite(state[key]).all(), (arm, names[index], key)
        output[arm] = {'checkpoint_sha256': metadata['sha256'], 'bytes': metadata['bytes'],
                       'changed_core_max_abs': changes, 'all_readout_parameters_and_metadata_unchanged': True,
                       'optimizer_parameter_tensors': len(expected), 'optimizer_steps': 2482,
                       'unused_parameters_unchanged': sorted(unused),
                       'frozen_core_parameters_and_buffers_unchanged': True,
                       'readout_metadata_unchanged': True}
    data = verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    assert data == receipt['superpoint_inputs'] == protocol['superpoint_inputs']
    assert receipt['readout_parameters_and_metadata_unchanged']
    return {'arms': output, 'train_superpoint_inputs': data, 'core_trainable_tensors': core_names}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    directory = options.directory.resolve()
    result = audit_rows(directory)
    result['checkpoints'] = audit_checkpoints(directory)
    result.update(schema='mcln-scanrefer-frozen-readout-pair-independent-audit-v1',
                  receipt_sha256=file_sha(directory / 'receipt.json'),
                  audit_script_sha256=file_sha(__file__), gpu_forwards=0,
                  optimizer_steps_executed=0, python=sys.version.split()[0])
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['integrity_pass', 'metrics',
                      'eligible_for_fixed_terminal_formal_evaluation']}))
