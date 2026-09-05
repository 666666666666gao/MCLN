import hashlib
import json

import pytest

from scripts.summarize_nr3d_object_appearance_pair import compare, verify_terminal_run


def rows(rec, mask, arm):
    return [{'id': i, 'scan_id': 'scene_' + str(i % 3),
             arm: {'rec_query': i, 'rec_box_iou': r, 'mask_iou': m}}
            for i, (r, m) in enumerate(zip(rec, mask))]


def test_ten_net_fixes_pass_but_nine_or_strict_rec_regression_fail():
    old = rows([None] * 10 + [.8] * 10, [.8] * 20, 'native')
    new = rows([.3] * 10 + [.8] * 10, [.8] * 20, 'appearance')
    assert compare(old, new)['fixed_screen_pass']
    new[0]['appearance']['rec_box_iou'] = .25
    assert not compare(old, new)['fixed_screen_pass']
    new[0]['appearance']['rec_box_iou'] = .3
    new[10]['appearance']['rec_box_iou'] = .5
    result = compare(old, new)
    assert result['thresholds']['rec025']['net'] == 10
    assert result['thresholds']['rec050']['net'] == -1 and not result['fixed_screen_pass']


def test_mask_mean_drop_fails_even_when_hit_counts_are_unchanged():
    old = rows([.2] * 10, [.8] * 10, 'native')
    new = rows([.3] * 10, [.79] * 10, 'appearance')
    result = compare(old, new)
    assert result['thresholds']['mask025']['net'] == result['thresholds']['mask050']['net'] == 0
    assert not result['fixed_screen_pass']


def fixture(directory, native_hits, protected_hits):
    directory.mkdir()
    fit = list(range(2048))
    order = fit * 2
    digest = hashlib.sha256(b'fixed').hexdigest()

    def value(i, hits):
        return {'rec_query': i, 'rec_box_iou': .3 if i < hits else .2,
                'mask_iou': .8, 'legal_box_oracle_iou': .8}

    baseline, terminal = [], []
    for i in range(6172):
        identity = {'id': i + 2048, 'scan_id': 'scene_' + str(i % 98), 'input_point_sha256': digest,
                    'early_queries_sha256': digest, 'frozen_text_mask_alpha_sha256': digest}
        baseline.append(dict(identity, native=value(i, protected_hits), appearance=value(i, protected_hits)))
        terminal.append(dict(identity, native=value(i, native_hits), appearance=value(i, 20)))
    batches = [{'step': i + 1, 'row_ids': order[i * 4:(i + 1) * 4], 'point_tensor_sha256': digest,
                'early_queries_sha256': [digest] * 4, 'frozen_text_mask_alpha_sha256': [digest] * 4}
               for i in range(1024)]
    manifest = {'row_ids': {'fit': fit, 'holdout': list(range(2048, 8220))}}
    (directory / 'input_manifest.json').write_text(json.dumps(manifest))
    receipt = {'schema': 'mcln-nr3d-object-appearance-pair-v1', 'status': 'complete',
               'optimizer_steps_per_arm': 1024, 'formal_rows': 0, 'formal_promotion': False,
               'frozen_parameters_and_buffers_unchanged': True, 'source_data_and_parent_checkpoint_unchanged': True,
               'early_queries_and_sampling_exactly_equal_to_start': True, 'text_mask_and_alpha_exactly_equal_to_start': True,
               'baseline_matches_protected_6172_rows': True, 'fit_order_ids': order,
               'fit_order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
               'manifest_sha256': hashlib.sha256((directory / 'input_manifest.json').read_bytes()).hexdigest()}
    for name, value in [('baseline_rows', baseline), ('terminal_rows', terminal), ('fit_point_batches', batches)]:
        path = directory / (name + '.json')
        path.write_text(json.dumps(value))
        receipt[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / 'receipt.json').write_text(json.dumps(receipt))


@pytest.mark.parametrize('native_hits,protected_hits,expected', [(10, 10, True), (11, 10, False), (10, 11, False)])
def test_full_receipt_requires_both_controls(tmp_path, native_hits, protected_hits, expected):
    directory = tmp_path / 'trial'
    fixture(directory, native_hits, protected_hits)
    result = verify_terminal_run(directory)
    assert result['integrity_pass'] and result['fixed_screen_pass'] is expected
    assert result['metrics']['terminal_appearance']['rec_hits_025'] == 20
    assert result['candidate_boxes_and_selections_allowed_to_change']
    assert result['formal_rows'] == 0 and not result['formal_promotion']


def test_modified_row_file_is_rejected(tmp_path):
    directory = tmp_path / 'trial'
    fixture(directory, 10, 10)
    path = directory / 'terminal_rows.json'
    records = json.loads(path.read_text())
    records[0]['appearance']['rec_box_iou'] = .9
    path.write_text(json.dumps(records))
    with pytest.raises(AssertionError):
        verify_terminal_run(directory)


def test_frozen_early_query_change_is_rejected_even_with_matching_file_hash(tmp_path):
    directory = tmp_path / 'trial'
    fixture(directory, 10, 10)
    path = directory / 'terminal_rows.json'
    records = json.loads(path.read_text())
    records[0]['early_queries_sha256'] = '0' * 64
    path.write_text(json.dumps(records))
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['terminal_rows_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        verify_terminal_run(directory)
