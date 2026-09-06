import copy
import json

import numpy as np
import pytest
import torch

from scripts.scanrefer_stage_diagnostics import FeatureMoments, summarize_stages, transitions
from scripts.trace_scanrefer_readout_stages import STAGES


def example_rows():
    values = [[.25, .8, .6], [.3, .4, .6], [.3, .4, .6],
              [.3, .4, .8], [.9, .55, .1], [.9, .4, .8]]
    records = {}
    for arm in ('protected_v99', 'local_v99'):
        rows = []
        for row_id in range(3):
            stages = {}
            for index, name in enumerate(STAGES):
                stages[name] = {'query_index': row_id, 'variant_index': max(0, index - 2),
                                'box': [float(index), 0., 0., 1., 1., 1.],
                                'rec_iou': values[index][row_id]}
            rows.append({'row_id': row_id, 'scan_id': 'unit_scene', 'point_sha256': str(row_id),
                         'target_id': row_id, 'utterance': 'unit fixture',
                         'root_box': [0., 0., 0., 1., 1., 1.], 'stages': stages,
                         'final_flat_index': row_id * 7})
        records[arm] = rows
    records['local_v99'][1]['stages']['v99_final']['rec_iou'] = .65
    native = {arm: [{'row_id': row['row_id'], 'scan_id': row['scan_id'],
                     'point_sha256': row['point_sha256'], 'query_index': row['row_id'],
                     'rec_iou': row['stages']['native']['rec_iou']} for row in rows]
              for arm, rows in records.items()}
    final = {arm: [{'row_id': row['row_id'], 'scan_id': row['scan_id'],
                    'point_sha256': row['point_sha256'],
                    'selected_variant_position': row['final_flat_index'],
                    'rec_iou': row['stages']['v99_final']['rec_iou']} for row in rows]
             for arm, rows in records.items()}
    return records, native, final


def test_feature_hook_ignores_invalid_values_and_preserves_forward():
    class Scorer(torch.nn.Module):
        def forward(self, features, valid):
            return features.sum(dim=-1).masked_fill(~valid, 0.)

    scorer = Scorer()
    features = torch.tensor([[[1., 2.], [1e20, 1e20], [3., 4.]]], requires_grad=True)
    valid = torch.tensor([[True, False, True]])
    expected = scorer(features, valid)
    moment = FeatureMoments(2)
    handle = scorer.register_forward_pre_hook(moment)
    observed = scorer(features, valid)
    scorer(torch.tensor([[[5., 6.]]]), torch.tensor([[True]]))
    handle.remove()
    assert torch.equal(observed, expected)
    assert observed.requires_grad and features.grad is None
    result = moment.export()
    assert result['valid_candidates'] == 3
    assert result['mean'] == [3., 4.]
    assert result['minimum'] == [1., 2.] and result['maximum'] == [5., 6.]
    assert np.allclose(result['root_mean_square'], np.sqrt([35. / 3., 56. / 3.]))


def test_thresholds_are_strict_and_repairs_subtract_damage():
    assert transitions([.25, .5, .8], [.3, .8, .5], .5) == {
        'repairs': 1, 'damage': 1, 'net_hits': 0}
    result = summarize_stages(*example_rows())
    parent = result['arms']['protected_v99']['consecutive_transitions']['native->parent']
    assert parent['025'] == {'repairs': 1, 'damage': 0, 'net_hits': 1}
    assert parent['050'] == {'repairs': 0, 'damage': 1, 'net_hits': -1}
    assert parent['query_slot_changed'] == 0
    assert parent['same_query_slot_different_box'] == 3
    assert result['paired_stage_changes']['v99_final']['050']['net_hits'] == 1
    assert result['query_slot_changes_do_not_prove_instance_changes']


def test_sorted_json_roundtrip_preserves_all_stage_metrics():
    values = example_rows()
    expected = summarize_stages(*values)
    roundtrip = json.loads(json.dumps(values, sort_keys=True))
    assert summarize_stages(*roundtrip) == expected


@pytest.mark.parametrize('source', ['reference', 'other_arm'])
def test_rejects_point_identity_mismatch(source):
    records, native, final = example_rows()
    if source == 'reference':
        native['protected_v99'][0]['point_sha256'] = 'wrong'
    else:
        records['local_v99'][0]['point_sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        summarize_stages(records, native, final)


def test_reference_prediction_drift_is_reported_without_rewriting_reference():
    values = example_rows()
    values[2]['local_v99'][0]['rec_iou'] = .2
    values[1]['local_v99'][0]['query_index'] = 10
    original = copy.deepcopy(values)
    result = summarize_stages(*values)
    agreement = result['reference_agreement']['local_v99']
    assert agreement['v99_final']['max_absolute_iou_difference'] == pytest.approx(.7)
    assert agreement['v99_final']['025']['net_hits'] == 1
    assert agreement['native_query_mismatches'] == 1
    assert values == original


def test_shared_query_slot_is_not_a_shared_root_identity():
    records, native, final = example_rows()
    records['local_v99'][0]['target_id'] = 99
    with pytest.raises(AssertionError):
        summarize_stages(records, native, final)
