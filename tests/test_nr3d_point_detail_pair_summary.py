import copy

import pytest

from scripts.summarize_nr3d_point_detail_pair import (
    compare, good_box_cohort, metrics, passes_fixed_screen, require_identity)


def rows(values, arm):
    return [{'id': i, 'scan_id': 'scene' + str(i // 2), arm: {'mask_iou': value}}
            for i, value in enumerate(values)]


def test_mean_gain_cannot_override_a_strict_threshold_break():
    result = compare(rows([.51, .6, .6, .6], 'native'),
                     rows([.5, .62, .62, .62], 'detail'))
    assert result['delta_mask_mean_iou'] > .002
    assert result['thresholds']['050'] == {'fixes': 0, 'breaks': 1, 'net': -1}
    assert not passes_fixed_screen({'control': result})
    assert metrics(rows([.25, .5, .51], 'native'), 'native')['mask_hits_025'] == 2
    assert metrics(rows([.25, .5, .51], 'native'), 'native')['mask_hits_050'] == 1


def test_detail_must_improve_both_native_endpoint_and_protected_start():
    end = rows([.62] * 4, 'detail')
    control = compare(rows([.6] * 4, 'native'), end)
    start = compare(rows([.621] * 4, 'native'), end)
    assert passes_fixed_screen({'control': control})
    assert not passes_fixed_screen({'control': control, 'start': start})
    assert control['scene_count'] == 2
    assert all(value > 0 for value in control['paired_scene_bootstrap_95_ci']['mask_mean_iou'])


def test_query_diagnostic_uses_only_rows_with_that_query():
    before = rows([.2, None, .6], 'native')
    after = rows([.3, None, .7], 'detail')
    result = compare(before, after)
    assert result['rows'] == 3 and result['evaluated_rows'] == 2
    assert result['delta_mask_mean_iou'] == pytest.approx(.1)
    assert metrics(after, 'detail')['mask_mean_iou'] == .5
    with pytest.raises(AssertionError):
        compare(before, after[:2])


def test_good_box_cohort_is_fixed_by_start_boxes_and_strict_threshold():
    before = [{'id': i, 'native': {'legal_box_oracle_iou': value}}
              for i, value in enumerate([.5, .51, None, .8])]
    after = [{'id': i, 'detail': {'legal_box_oracle_iou': .9}} for i in range(4)]
    start, end = good_box_cohort(before, after)
    assert [row['id'] for row in start] == [row['id'] for row in end] == [1, 3]


def test_mask_improvement_does_not_authorize_changed_rec_or_inputs():
    value = {'rec_query': 2, 'rec_box_iou': .8, 'mask_query': 3,
             'legal_box_oracle_query': 4, 'legal_box_oracle_iou': .9, 'mask_iou': .4}
    before = [{'id': 1, 'scan_id': 'scene1', 'grounding_sha256': 'g',
               'input_point_sha256': 'p', 'native': copy.deepcopy(value), 'detail': copy.deepcopy(value)}]
    after = copy.deepcopy(before)
    after[0]['detail']['mask_iou'] = .7
    require_identity(before, after)
    after[0]['detail']['rec_query'] = 5
    with pytest.raises(AssertionError):
        require_identity(before, after)
    after = copy.deepcopy(before)
    after[0]['input_point_sha256'] = 'changed'
    with pytest.raises(AssertionError):
        require_identity(before, after)
