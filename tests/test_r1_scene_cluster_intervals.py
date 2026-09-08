import pytest

from scripts.summarize_nr3d_reference_memory import paired_changes, scene_cluster_intervals


def test_factorial_interaction_keeps_scene_clusters_and_expression_weighting():
    rows = []
    for scene, count, primary_box, control_box, primary_mask in [
            ('large', 9, .8, .1, .75), ('small', 1, .1, .8, .25)]:
        for _ in range(count):
            rows.append({'scan_id': scene, 'protected_mask_iou': .5, 'scores': {
                **{mode: {'box_iou': control_box, 'mask_iou': .5}
                   for mode in ['query_global', 'query_pair', 'object_global']},
                'object_pair': {'box_iou': primary_box, 'mask_iou': primary_mask},
                'protected': {'box_iou': control_box, 'mask_iou': .1}}})
    result = scene_cluster_intervals(rows)
    assert result['scenes'] == 2 and result['rows'] == 10
    for effect in ['memory_with_pair', 'readout_with_object', 'interaction', 'primary_minus_protected']:
        values = result['effects'][effect]
        assert values['rec025']['estimate'] == pytest.approx(80)
        assert values['rec050']['estimate'] == pytest.approx(80)
        assert values['mask_mean_iou']['estimate'] == pytest.approx(20)
        assert values['rec025']['percentile_95_interval'] == [-100.0, 100.0]
    for effect in ['memory_with_global', 'readout_with_query']:
        for values in result['effects'][effect].values():
            assert values['estimate'] == 0
            assert values['percentile_95_interval'] == [0.0, 0.0]
    changes = paired_changes(rows, 'protected', 'object_pair')
    assert changes['rec025']['fixes'] == 9 and changes['rec025']['breaks'] == 1
    assert changes['mask025']['fixes'] == 0 and changes['mask025']['breaks'] == 1
    assert result['screening_gates_changed'] is False
