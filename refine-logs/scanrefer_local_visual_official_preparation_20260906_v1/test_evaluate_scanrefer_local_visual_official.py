import json

import pytest

from scripts.evaluate_scanrefer_local_visual_official import (
    file_sha, promotion_check, row_metrics, verify_training_endpoint,
)


def floor_metrics():
    return {'rows': 9508, 'rec_hits025': 5572, 'rec_hits050': 4797,
            'mask_hits025': 5582, 'mask_hits050': 4821, 'mask_miou': 44.72}


def test_strict_iou_thresholds_and_percent_units():
    metrics = row_metrics([{'rec_iou': .25, 'mask_iou': .5},
                           {'rec_iou': .5, 'mask_iou': .75}])
    assert metrics == {'rows': 2, 'rec_hits025': 1, 'rec_hits050': 0,
                       'mask_hits025': 2, 'mask_hits050': 1, 'mask_miou': 62.5}


def test_scan_floor_passes_without_stretch_goals_or_nr_sr_masks():
    result = promotion_check(floor_metrics(), floor_metrics())
    assert result['advance_to_nr3d_sr3d_rec']


@pytest.mark.parametrize('field,value', [
    ('rec_hits025', 5571), ('rec_hits050', 4796),
    ('mask_hits025', 5581), ('mask_hits050', 4820), ('mask_miou', 44.719),
])
def test_each_required_scan_metric_can_block_promotion(field, value):
    candidate = floor_metrics()
    candidate[field] = value
    assert not promotion_check(floor_metrics(), candidate)['advance_to_nr3d_sr3d_rec']


def test_same_input_protected_improvement_is_also_preserved():
    protected = floor_metrics()
    protected['rec_hits050'] += 1
    assert not promotion_check(protected, floor_metrics())['advance_to_nr3d_sr3d_rec']


def test_development_row_count_cannot_be_labeled_formal():
    candidate = floor_metrics()
    candidate['rows'] = 6887
    with pytest.raises(AssertionError):
        promotion_check(floor_metrics(), candidate)


def screen_fixture(directory, joint_iou):
    def save(name, value):
        (directory / name).write_text(json.dumps(value))

    rows = [{'row_id': i, 'scan_id': 'scene0000_00', 'physical_space': 'scene0000',
             'point_sha256': str(i), 'rec_iou': .75, 'mask_iou': .8} for i in range(6887)]
    changed = [dict(row, rec_iou=joint_iou) for row in rows]
    baseline = {'control': rows, 'local': rows}
    terminal = {'control': rows, 'local': changed}
    save('input_manifest.json', {'steps_per_arm': 2482, 'epochs': 1, 'mode': 'train', 'readouts_frozen': True})
    save('baseline_rows.json', baseline)
    save('terminal_rows.json', terminal)
    (directory / 'synthetic_checkpoint').write_bytes(b'unit fixture, not a trained model')
    save('receipt.json', {
        'status': 'complete', 'formal_rows': 0, 'steps_per_arm': 2482, 'holdout_rows': 6887,
        'manifest_sha256': file_sha(directory / 'input_manifest.json'),
        'baseline_rows_sha256': file_sha(directory / 'baseline_rows.json'),
        'terminal_rows_sha256': file_sha(directory / 'terminal_rows.json'),
        'baseline_metrics': {arm: row_metrics(value) for arm, value in baseline.items()},
        'terminal_metrics': {arm: row_metrics(value) for arm, value in terminal.items()},
        'fixed_endpoint_ready_for_official_evaluation': True, 'readouts_unchanged': True,
        'checkpoints': {'local': {'path': str(directory / 'synthetic_checkpoint'),
                                  'sha256': file_sha(directory / 'synthetic_checkpoint')}},
    })


def test_eligible_endpoint_is_recomputed_from_rows(tmp_path):
    screen_fixture(tmp_path, .75)
    receipt, _ = verify_training_endpoint(tmp_path)
    assert receipt['steps_per_arm'] == 2482


def test_development_decline_is_diagnostic_and_does_not_select_an_epoch(tmp_path):
    screen_fixture(tmp_path, .4)
    receipt, _ = verify_training_endpoint(tmp_path)
    assert receipt["terminal_metrics"]["local"]["rec_hits050"] == 0


def test_changed_checkpoint_is_rejected_before_loading(tmp_path):
    screen_fixture(tmp_path, .75)
    (tmp_path / 'synthetic_checkpoint').write_bytes(b'changed')
    with pytest.raises(AssertionError):
        verify_training_endpoint(tmp_path)
