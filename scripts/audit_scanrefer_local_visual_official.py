"""Audit the actual fixed ScanRefer result independently of promotion outcome."""

import argparse
import datetime
import json
import math
from pathlib import Path

from scripts.audit_scanrefer_joint_readout_pair import compare, file_sha, metrics
from scripts.evaluate_scanrefer_local_visual_official import promotion_check


ARMS = ['protected_v99', 'local_v99']
IDENTITY_FIELDS = ['row_id', 'scan_id', 'point_sha256']


def native_metrics(rows):
    values = [row['rec_iou'] for row in rows]
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
    return {'rows': len(rows), 'rec_hits025': sum(value > .25 for value in values),
            'rec_hits050': sum(value > .5 for value in values)}


def rec_compare(reference, candidate):
    assert len(reference) == len(candidate) and reference
    transitions = [[0] * 3 for _ in range(3)]
    effects = {suffix: {'repair': 0, 'damage': 0} for suffix in ['025', '050']}
    for old, new in zip(reference, candidate):
        assert all(old[key] == new[key] for key in IDENTITY_FIELDS)
        old_band = int(old['rec_iou'] > .25) + int(old['rec_iou'] > .5)
        new_band = int(new['rec_iou'] > .25) + int(new['rec_iou'] > .5)
        transitions[old_band][new_band] += 1
        for suffix, threshold in [('025', .25), ('050', .5)]:
            effects[suffix]['repair'] += int(old['rec_iou'] <= threshold < new['rec_iou'])
            effects[suffix]['damage'] += int(new['rec_iou'] <= threshold < old['rec_iou'])
    for value in effects.values():
        value['net'] = value['repair'] - value['damage']
    return {'rows': len(reference), 'effects': effects,
            'rec_iou_transition_counts': transitions,
            'transition_bands': ['[0,0.25]', '(0.25,0.50]', '(0.50,1]'],
            'selected_instance_identity_available': False}


def audit_rows(directory):
    result = directory / 'result'
    receipt = json.loads((result / 'receipt.json').read_text())
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    protocol = json.loads((result / 'protocol.json').read_text())
    assert (directory / 'controller.exit').read_text().strip() == '0'
    assert receipt['schema'] == 'mcln-scanrefer-local-visual-official-v1'
    assert receipt['status'] == 'complete'
    assert receipt['formal_rows'] == manifest['formal_rows'] == protocol['rows'] == 9508
    assert receipt['optimizer_steps'] == receipt['checkpoint_writes'] == 0
    assert receipt['all_model_states_unchanged'] and receipt['native_evaluators_match_row_metrics']
    assert receipt['evaluation_extent_policy'] == 'existing rec_candidate_adapter floor at 1e-6'
    assert receipt['manifest_sha256'] == file_sha(directory / 'input_manifest.json')
    for name in ['rows', 'native_rows']:
        assert receipt[name + '_sha256'] == file_sha(result / (name + '.json'))
    assert receipt['trained_checkpoint'] == manifest['trained_checkpoint']
    assert manifest['scan_rec_historical_floor_hits'] == [5572, 4797]
    assert manifest['scan_mask_paper_floor_percent'] == [58.70, 50.70, 44.72]
    assert not manifest['nr3d_sr3d_mask_gate']
    assert protocol['native_loader_and_worker_seeding']
    identities = protocol['identities']
    assert len(identities) == 9508
    records = json.loads((result / 'rows.json').read_text())
    native = json.loads((result / 'native_rows.json').read_text())
    assert set(records) == set(native) == set(ARMS)
    actual, actual_native = {}, {}
    for arm in ARMS:
        rows, native_rows = records[arm], native[arm]
        assert [row['row_id'] for row in rows] == list(range(9508))
        assert [row['row_id'] for row in native_rows] == list(range(9508))
        for system_row, native_row, identity in zip(rows, native_rows, identities):
            assert system_row['scan_id'] == identity[0]
            assert system_row['physical_space'] == system_row['scan_id'].split('_')[0]
            assert all(system_row[key] == native_row[key] for key in IDENTITY_FIELDS)
            assert 0 <= native_row['query_index'] < 256
        actual[arm], actual_native[arm] = metrics(rows), native_metrics(native_rows)
        for key, value in actual[arm].items():
            if key == 'mask_miou':
                # Existing CPU summation audits observed sub-1e-8 percentage-point differences.
                assert abs(value - receipt['metrics'][arm][key]) < 1e-8
            else:
                assert value == receipt['metrics'][arm][key], (arm, key)
        assert actual_native[arm] == receipt['native_rec_metrics'][arm]
    promotion = promotion_check(actual['protected_v99'], actual['local_v99'])
    assert promotion == receipt['promotion']
    return {'integrity_pass': True, 'formal_rows': 9508, 'metrics': actual,
            'native_rec_metrics': actual_native, 'promotion': promotion,
            'system_local_minus_protected': compare(records['protected_v99'], records['local_v99']),
            'native_local_minus_protected': rec_compare(native['protected_v99'], native['local_v99']),
            'system_minus_native': {arm: rec_compare(native[arm], records[arm]) for arm in ARMS},
            'paired_row_and_point_identity_verified': True,
            'native_mask_metrics_not_recorded': True,
            'receipt_sha256': file_sha(result / 'receipt.json'),
            'rows_sha256': receipt['rows_sha256'], 'native_rows_sha256': receipt['native_rows_sha256'],
            'protocol_sha256': file_sha(result / 'protocol.json')}


def audit_inputs(directory):
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    training = Path(manifest['training_directory'])
    for filename, field in [('receipt.json', 'training_receipt_sha256'),
                            ('independent_audit.json', 'training_audit_sha256')]:
        assert file_sha(training / filename) == manifest[field]
    training_receipt = json.loads((training / 'receipt.json').read_text())
    assert file_sha(training / 'input_manifest.json') == training_receipt['manifest_sha256']
    train_manifest = json.loads((training / 'input_manifest.json').read_text())
    assert manifest['trained_checkpoint'] == training_receipt['checkpoints']['local']
    source = Path(train_manifest['model_source'])
    source_manifest = source / 'local_visual_source_manifest.json'
    assert file_sha(source_manifest) == train_manifest['source_manifest_sha256']
    source_files = json.loads(source_manifest.read_text())['files']
    for name, digest in source_files.items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    artifacts = dict(train_manifest['artifacts'], trained_local=manifest['trained_checkpoint'])
    for name, item in artifacts.items():
        assert Path(item['path']).stat().st_size == item['bytes'], name
        assert file_sha(item['path']) == item['sha256'], name
    return {'source_files_verified': len(source_files),
            'source_manifest_sha256': train_manifest['source_manifest_sha256'],
            'manifest_sha256': file_sha(directory / 'input_manifest.json'),
            'training_audit_sha256': manifest['training_audit_sha256'],
            'artifacts_verified_after_evaluation': artifacts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    option = parser.parse_args()
    result = audit_rows(option.directory)
    result['inputs'] = audit_inputs(option.directory)
    result['schema'] = 'mcln-scanrefer-local-visual-official-audit-v1'
    result['time_cst'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    result['audit_script_sha256'] = file_sha(__file__)
    result['gpu_forwards'] = result['optimizer_steps'] = result['checkpoint_writes'] = 0
    with option.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
