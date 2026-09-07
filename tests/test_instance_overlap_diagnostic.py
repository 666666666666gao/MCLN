import numpy as np

from scripts.analyze_scanrefer_instance_overlap import attribute_overlap, stages_with_original_boxes, transitions


def test_root_can_be_nearest_even_below_rec_threshold():
    result = attribute_overlap([.2, .1], [7, 42], ['chair', 'chair'], 7)
    assert result['category'] == 'root_unique_max' and result['best_object_id'] == 7


def test_wrong_same_class_is_not_root():
    result = attribute_overlap([.05, .8], [7, 42], ['chair', 'chair'], 7)
    assert result['category'] == 'other_same_label_unique_max' and result['best_object_id'] == 42


def test_object_permutation_preserves_attribution():
    original = attribute_overlap([.2, .7, .1], [7, 42, 11], ['chair', 'table', 'lamp'], 7)
    permuted = attribute_overlap([.1, .2, .7], [11, 7, 42], ['lamp', 'chair', 'table'], 7)
    assert original == permuted


def test_ties_and_no_overlap_do_not_invent_identity():
    tied = attribute_overlap(np.array([.5, .5], dtype=np.float32), [7, 42], ['chair', 'chair'], 7)
    assert tied['category'] == 'tied_max' and tied['best_object_id'] is None
    empty = attribute_overlap([0, 0], [7, 42], ['chair', 'chair'], 7)
    assert empty['category'] == 'no_overlap' and empty['best_object_id'] is None


def test_source_box_uses_query_mapping_not_topk_position():
    row = {'stages': {'geometry': {'query_index': 42}, 'v99_final': {'query_index': 7}},
           'top16_query_indices': [7, 42], 'top16_valid': [True, True],
           'top16_boxes': [[1]*6, [2]*6]}
    values = stages_with_original_boxes(row)
    assert values['geometry_query_native']['box'] == [2]*6
    assert values['final_query_native']['box'] == [1]*6


def test_strict_damage_with_same_nearest_object_is_retained():
    a = attribute_overlap([.7, .1], [7, 42], ['chair', 'chair'], 7)
    b = attribute_overlap([.3, .1], [7, 42], ['chair', 'chair'], 7)
    result = transitions([a], [b])
    assert result['025']['net'] == 0 and result['050']['net'] == -1
    assert result['050']['break_categories'] == {'root_unique_max->root_unique_max': 1}
