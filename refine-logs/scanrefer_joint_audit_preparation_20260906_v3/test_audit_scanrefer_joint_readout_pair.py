import pytest

from scripts.audit_scanrefer_joint_readout_pair import compare, metrics


def row(index, room, rec, mask):
    return {'row_id': index, 'scan_id': room + '_00', 'physical_space': room,
            'point_sha256': str(index), 'rec_iou': rec, 'mask_iou': mask}


def test_exact_thresholds_and_paired_repairs_preserve_row_identity():
    before = [row(0, 'scene0000', .25, .5), row(1, 'scene0000', .75, .75),
              row(2, 'scene0001', .5, .25)]
    after = [row(0, 'scene0000', .5, .75), row(1, 'scene0000', .4, .5),
             row(2, 'scene0001', .75, .25)]
    value = compare(before, after)
    assert metrics(before)['rec_hits050'] == 1
    assert value['effects']['rec025'] == {'repair': 1, 'damage': 0, 'net': 1}
    assert value['effects']['rec050'] == {'repair': 1, 'damage': 1, 'net': 0}
    assert value['effects']['mask050'] == {'repair': 1, 'damage': 1, 'net': 0}
    assert value['rec_iou_transition_counts'] == [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
    assert value['physical_spaces'] == 2


def test_bootstrap_keeps_expressions_in_same_physical_room_together():
    before = [row(i, 'scene0000', .1, .2) for i in range(3)]
    after = [row(i, 'scene0000', .6, .4) for i in range(3)]
    value = compare(before, after)
    interval = value['bootstrap']['intervals_95_percent_pp']
    assert interval['rec025'] == [100., 100.]
    assert interval['mask_miou'] == pytest.approx([20., 20.])
    assert not value['bootstrap']['is_promotion_gate']


@pytest.mark.parametrize('field', ['row_id', 'point_sha256', 'physical_space'])
def test_changed_pair_identity_is_rejected(field):
    original = row(0, 'scene0000', .8, .8)
    changed = dict(original)
    changed[field] = 'changed'
    with pytest.raises(AssertionError):
        compare([original], [changed])
