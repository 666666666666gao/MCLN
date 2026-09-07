"""Independently recount saved stage evidence; no model or selection code is imported."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ('protected_v99', 'local_v99')
STAGES = ('native', 'parent', 'parent_after_geometry_validity',
          'geometry', 'v99_proposal', 'v99_final')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changes(before, after):
    result = {}
    for suffix, threshold in (('025', .25), ('050', .5)):
        repairs = sum(a <= threshold < b for a, b in zip(before, after))
        damage = sum(b <= threshold < a for a, b in zip(before, after))
        result[suffix] = {'repairs': repairs, 'damage': damage, 'net_hits': repairs - damage}
    return result


def hits(values):
    return {'hits025': int(sum(x > .25 for x in values)), 'hits050': int(sum(x > .5 for x in values))}


def box_iou(boxes, roots):
    """Independent NumPy float32 AABB calculation using the existing 1e-6 floors."""
    boxes = np.asarray(boxes, dtype=np.float32)
    roots = np.asarray(roots, dtype=np.float32)
    size = np.maximum(boxes[..., 3:], np.float32(1e-6))
    root_size = np.maximum(roots[..., 3:], np.float32(1e-6))
    lower, upper = boxes[..., :3] - size / 2, boxes[..., :3] + size / 2
    root_lower, root_upper = roots[..., :3] - root_size / 2, roots[..., :3] + root_size / 2
    overlap = np.maximum(np.minimum(upper, root_upper) - np.maximum(lower, root_lower), 0)
    a, b = np.maximum(upper - lower, 0), np.maximum(root_upper - root_lower, 0)
    intersection = overlap[..., 0] * overlap[..., 1] * overlap[..., 2]
    volume_a = a[..., 0] * a[..., 1] * a[..., 2]
    volume_b = b[..., 0] * b[..., 1] * b[..., 2]
    return intersection / np.maximum(volume_a + volume_b - intersection, np.float32(1e-6))


def audit(directory, reference):
    result = directory / 'diagnostic_result'
    manifest, receipt = read(directory / 'input_manifest.json'), read(result / 'receipt.json')
    collection = read(directory / 'collection_receipt.json')
    assert (directory / 'controller.exit').read_text().strip() == '0'
    assert receipt['status'] == 'complete'
    assert receipt['schema'] == 'mcln-scanrefer-stage-diagnostic-result-v1'
    assert receipt['formal_rows'] == manifest['formal_rows'] == 0
    assert receipt['diagnostic_rows'] == manifest['diagnostic_rows'] == 9508
    assert receipt['optimizer_steps'] == receipt['checkpoint_writes'] == 0
    assert not receipt['used_for_promotion'] and not receipt['selection_uses_gt']
    assert receipt['all_model_states_unchanged'] and receipt['all_forward_final_scores_verified_equal']
    assert receipt['native_evaluators_match_row_metrics']
    assert receipt['manifest_sha256'] == sha(directory / 'input_manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(directory / name) == expected, name
    for name, expected in manifest['reference_files'].items():
        assert sha(reference / name) == expected, name
    for name, expected in collection['downloaded_sha256'].items():
        assert sha(result / name) == expected, name
    for name in ('stage_rows', 'stage_summary', 'normalized_features', 'rows', 'native_rows'):
        assert receipt[name + '_sha256'] == sha(result / (name + '.json'))
    assert read(reference / 'result/independent_audit.json')['integrity_pass']
    protocol, old_protocol = read(result / 'protocol.json'), read(reference / 'result/protocol.json')
    assert protocol['identities'] == old_protocol['identities']
    assert protocol['data_root'] == receipt['data_root'] == manifest['data_root']
    assert protocol['superpoint_inputs'] == receipt['superpoint_inputs'] == old_protocol['superpoint_inputs']
    assert protocol['formal_rows'] == 0 and protocol['purpose'] == 'stage_diagnostic'
    rows, summary = read(result / 'stage_rows.json'), read(result / 'stage_summary.json')
    features = read(result / 'normalized_features.json')
    final, native = read(result / 'rows.json'), read(result / 'native_rows.json')
    old_final, old_native = read(reference / 'result/rows.json'), read(reference / 'result/native_rows.json')
    audit_result = {'arms': {}, 'paired_stage_changes': {}, 'reference_agreement': {}}
    for arm in ARMS:
        arm_rows = rows[arm]
        assert len(arm_rows) == len(final[arm]) == len(native[arm]) == 9508
        stage_values = {stage: [row['stages'][stage]['rec_iou'] for row in arm_rows] for stage in STAGES}
        for index, row in enumerate(arm_rows):
            assert row['row_id'] == final[arm][index]['row_id'] == native[arm][index]['row_id'] == index
            assert [row['scan_id'], row['target_id'], row['utterance']] == protocol['identities'][index]
            for key in ('scan_id', 'point_sha256'):
                assert row[key] == final[arm][index][key] == native[arm][index][key]
                assert row[key] == old_final[arm][index][key] == old_native[arm][index][key]
            assert set(row['stages']) == set(STAGES)
            assert len(row['top16_query_indices']) == len(row['top16_valid']) == len(row['top16_boxes']) == 16
            assert np.asarray(row['effective_variant_valid']).shape == (16, 7)
            assert len(row['deployed_variant_valid']) == 112
            for stage in STAGES:
                value = row['stages'][stage]
                assert 0 <= value['query_index'] < 256 and 0 <= value['rec_iou'] <= 1
                assert len(value['box']) == 6 and np.isfinite(value['box']).all()
            for stage, flat_key in (('geometry', 'geometry_flat_index'), ('v99_proposal', 'proposal_flat_index'),
                                    ('v99_final', 'final_flat_index')):
                flat = row[flat_key]
                assert row['deployed_variant_valid'][flat]
                assert row['stages'][stage]['query_index'] == row['top16_query_indices'][flat // 7]
                assert row['stages'][stage]['variant_index'] == flat % 7
            assert row['stages']['native']['query_index'] == native[arm][index]['query_index']
            assert row['stages']['native']['rec_iou'] == native[arm][index]['rec_iou']
            assert row['stages']['v99_final']['rec_iou'] == final[arm][index]['rec_iou']
            assert row['final_flat_index'] == final[arm][index]['selected_variant_position']
        actual = {'metrics': {name: hits(stage_values[name]) for name in STAGES}, 'consecutive_transitions': {}}
        for before, after in zip(STAGES[:-1], STAGES[1:]):
            value = changes(stage_values[before], stage_values[after])
            value['query_slot_changed'] = sum(r['stages'][before]['query_index'] != r['stages'][after]['query_index'] for r in arm_rows)
            value['same_query_slot_different_box'] = sum(
                r['stages'][before]['query_index'] == r['stages'][after]['query_index']
                and r['stages'][before]['box'] != r['stages'][after]['box'] for r in arm_rows)
            actual['consecutive_transitions'][before + '->' + after] = value
        assert actual == summary['arms'][arm]
        computed = box_iou([[r['stages'][s]['box'] for s in STAGES] for r in arm_rows],
                           [[r['root_box']] for r in arm_rows])
        recorded = np.asarray([[r['stages'][s]['rec_iou'] for s in STAGES] for r in arm_rows])
        assert np.isfinite(computed).all()
        threshold_mismatches = {str(t): int(((computed > t) != (recorded > t)).sum()) for t in (.25, .5)}
        assert not any(threshold_mismatches.values()), threshold_mismatches
        oracle = box_iou([r['top16_boxes'] for r in arm_rows], [[r['root_box']] for r in arm_rows])
        oracle[~np.asarray([r['top16_valid'] for r in arm_rows], dtype=bool)] = -1
        oracle = oracle.max(axis=1)
        recorded_oracle = np.asarray([r['top16_oracle_iou'] for r in arm_rows])
        for threshold in (.25, .5):
            assert np.array_equal(oracle > threshold, recorded_oracle > threshold)
        counts = {'parent': sum(sum(r['top16_valid']) for r in arm_rows),
                  'geometry': sum(int(np.asarray(r['effective_variant_valid']).sum()) for r in arm_rows)}
        for name, width in (('parent', 152), ('geometry', 179)):
            moment = features[arm][name]
            assert moment['valid_candidates'] == counts[name] and moment['feature_width'] == width
            for field in ('mean', 'root_mean_square', 'minimum', 'maximum'):
                assert len(moment[field]) == width and np.isfinite(moment[field]).all()
        agreement = {}
        for name, old in (('native', old_native[arm]), ('v99_final', old_final[arm])):
            old_values = [r['rec_iou'] for r in old]
            agreement[name] = dict(changes(old_values, stage_values[name]),
                max_absolute_iou_difference=max(abs(a-b) for a, b in zip(old_values, stage_values[name])))
        agreement['native_query_mismatches'] = sum(r['stages']['native']['query_index'] != o['query_index'] for r, o in zip(arm_rows, old_native[arm]))
        agreement['final_variant_position_mismatches'] = sum(r['final_flat_index'] != o['selected_variant_position'] for r, o in zip(arm_rows, old_final[arm]))
        assert agreement == summary['reference_agreement'][arm]
        audit_result['reference_agreement'][arm] = agreement
        actual.update({'top16_oracle_hits': hits(recorded_oracle), 'valid_feature_counts': counts,
                       'independent_box_iou_max_absolute_difference': float(np.abs(computed-recorded).max()),
                       'independent_oracle_max_absolute_difference': float(np.abs(oracle-recorded_oracle).max()),
                       'independent_iou_threshold_mismatches': threshold_mismatches})
        audit_result['arms'][arm] = actual
    for old, new in zip(rows[ARMS[0]], rows[ARMS[1]]):
        for key in ('row_id', 'scan_id', 'target_id', 'utterance', 'root_box', 'point_sha256'):
            assert old[key] == new[key]
    for stage in STAGES:
        actual = changes([r['stages'][stage]['rec_iou'] for r in rows[ARMS[0]]],
                         [r['stages'][stage]['rec_iou'] for r in rows[ARMS[1]]])
        assert actual == summary['paired_stage_changes'][stage]
        audit_result['paired_stage_changes'][stage] = actual
    audit_result.update({'integrity_pass': True, 'formal_rows': 0, 'diagnostic_rows': 9508,
        'used_for_promotion': False, 'optimizer_updates': 0, 'checkpoint_writes': 0,
        'receipt_sha256': sha(result / 'receipt.json'), 'auditor_sha256': sha(Path(__file__)),
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope': 'Independent saved-row, AABB, hash, identity, selection-index and statistic audit; model execution assertions come from the bound runtime receipt.',
        'reference_drift_never_rewrites_official': True, 'query_slots_are_not_instance_identity': True})
    return audit_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    output = audit(args.directory, args.reference)
    with (args.directory / 'diagnostic_result/independent_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(output), flush=True)
