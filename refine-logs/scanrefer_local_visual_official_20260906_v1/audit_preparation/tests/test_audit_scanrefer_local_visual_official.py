import json

import pytest

from scripts.audit_scanrefer_local_visual_official import (
    ARMS, audit_rows, file_sha, metrics, native_metrics, promotion_check, rec_compare,
)


def save(path, value):
    path.write_text(json.dumps(value))


def formal_fixture(directory, local_iou):
    """Synthetic unit-test rows; no model run or actual receipt is represented."""
    result = directory / 'result'
    result.mkdir()
    (directory / 'controller.exit').write_text('0\n')
    manifest = {'formal_rows': 9508, 'trained_checkpoint': {'fixture': True},
                'scan_rec_historical_floor_hits': [5572, 4797],
                'scan_mask_paper_floor_percent': [58.70, 50.70, 44.72],
                'nr3d_sr3d_mask_gate': False}
    save(directory / 'input_manifest.json', manifest)
    rows = [{'row_id': i, 'scan_id': 'scene%04d_00' % (i // 100),
             'physical_space': 'scene%04d' % (i // 100), 'point_sha256': str(i),
             'rec_iou': .8, 'mask_iou': .8} for i in range(9508)]
    records = {'protected_v99': rows,
               'local_v99': [dict(row, rec_iou=local_iou) for row in rows]}
    native = {arm: [{key: row[key] for key in ['row_id', 'scan_id', 'point_sha256']}
                    for row in rows] for arm in ARMS}
    for arm in ARMS:
        for row in native[arm]:
            row.update(query_index=0, rec_iou=.6)
    save(result / 'protocol.json', {'rows': 9508, 'native_loader_and_worker_seeding': True,
                                   'identities': [[row['scan_id'], 0, 'fixture'] for row in rows]})
    save(result / 'rows.json', records)
    save(result / 'native_rows.json', native)
    actual = {arm: metrics(records[arm]) for arm in ARMS}
    receipt = {'schema': 'mcln-scanrefer-local-visual-official-v1', 'status': 'complete',
               'formal_rows': 9508, 'optimizer_steps': 0, 'checkpoint_writes': 0,
               'all_model_states_unchanged': True, 'native_evaluators_match_row_metrics': True,
               'evaluation_extent_policy': 'existing rec_candidate_adapter floor at 1e-6',
               'trained_checkpoint': manifest['trained_checkpoint'],
               'manifest_sha256': file_sha(directory / 'input_manifest.json'),
               'rows_sha256': file_sha(result / 'rows.json'),
               'native_rows_sha256': file_sha(result / 'native_rows.json'),
               'metrics': actual, 'native_rec_metrics': {arm: native_metrics(native[arm]) for arm in ARMS},
               'promotion': promotion_check(actual['protected_v99'], actual['local_v99'])}
    save(result / 'receipt.json', receipt)
    return receipt


@pytest.mark.parametrize('local_iou,promoted', [(.8, True), (.4, False)])
def test_complete_pass_and_fail_results_both_receive_integrity_audit(tmp_path, local_iou, promoted):
    formal_fixture(tmp_path, local_iou)
    result = audit_rows(tmp_path)
    assert result['integrity_pass']
    assert result['promotion']['advance_to_nr3d_sr3d_rec'] == promoted
    assert result['native_mask_metrics_not_recorded']
    assert result['native_local_minus_protected']['effects']['050']['net'] == 0
    assert result['system_minus_native']['local_v99']['effects']['050']['net'] == (0 if promoted else -9508)


def test_rec_comparison_uses_strict_thresholds_without_borrowing_masks():
    old = [dict(row_id=i, scan_id='scene0000_00', point_sha256=str(i), rec_iou=value)
           for i, value in enumerate([.25, .5, .75])]
    new = [dict(row, rec_iou=value) for row, value in zip(old, [.5, .75, .25])]
    assert native_metrics(old) == {'rows': 3, 'rec_hits025': 2, 'rec_hits050': 1}
    result = rec_compare(old, new)
    assert result['effects']['025'] == result['effects']['050'] == {'repair': 1, 'damage': 1, 'net': 0}
    assert result['rec_iou_transition_counts'] == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]


@pytest.mark.parametrize('field', ['metrics', 'promotion', 'optimizer_steps'])
def test_receipt_cannot_override_rows_or_actual_evaluation_scope(tmp_path, field):
    receipt = formal_fixture(tmp_path, .4)
    if field == 'metrics':
        receipt['metrics']['local_v99']['rec_hits050'] += 1
    elif field == 'promotion':
        receipt['promotion']['advance_to_nr3d_sr3d_rec'] = True
    else:
        receipt['optimizer_steps'] = 1
    save(tmp_path / 'result/receipt.json', receipt)
    with pytest.raises(AssertionError):
        audit_rows(tmp_path)


@pytest.mark.parametrize('change', ['duplicate_id', 'native_point', 'invalid_native_iou', 'row_file_hash'])
def test_identity_invalid_values_and_changed_row_files_are_rejected(tmp_path, change):
    receipt = formal_fixture(tmp_path, .8)
    filename = 'native_rows' if change.startswith('native_') or change == 'invalid_native_iou' else 'rows'
    path = tmp_path / ('result/' + filename + '.json')
    rows = json.loads(path.read_text())
    if change == 'duplicate_id':
        rows['local_v99'][1]['row_id'] = 0
    elif change == 'native_point':
        rows['local_v99'][0]['point_sha256'] = 'different sampled cloud'
    elif change == 'invalid_native_iou':
        rows['local_v99'][0]['rec_iou'] = float('nan')
    else:
        rows['local_v99'][0]['rec_iou'] = .4
    save(path, rows)
    if change != 'row_file_hash':
        receipt[filename + '_sha256'] = file_sha(path)
        save(tmp_path / 'result/receipt.json', receipt)
    with pytest.raises(AssertionError):
        audit_rows(tmp_path)


def test_saved_loader_protocol_must_match_formal_rows(tmp_path):
    formal_fixture(tmp_path, .8)
    path = tmp_path / 'result/protocol.json'
    protocol = json.loads(path.read_text())
    protocol['identities'][0][0] = 'scene9999_00'
    save(path, protocol)
    with pytest.raises(AssertionError):
        audit_rows(tmp_path)
