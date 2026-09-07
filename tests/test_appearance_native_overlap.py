import pytest

from scripts.analyze_scanrefer_appearance_native_overlap import annotate_row


def fixture():
    row = {'row_id': 8, 'scan_id': 'scene_a', 'point_sha256': 'points',
           'root_box': [0, 0, 0, 1, 1, 1], 'selected_box': [4, 0, 0, 1, 1, 1],
           'selected_query': 17, 'rec_iou': 0.0}
    annotation = {'scene_id': 'scene_a', 'object_id': '7', 'ann_id': '2', 'token': ['the', 'chair']}
    geometry = {'point_sha256': 'points', 'object_ids': [42, 7],
                'boxes': [[4, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], 'labels': ['chair', 'chair']}
    return row, annotation, geometry


def test_annotation_target_is_not_query_number_or_geometry_array_position():
    row, annotation, geometry = fixture()
    result = annotate_row(row, annotation, geometry)
    assert result['target_id'] == 7 and result['best_object_id'] == 42
    assert result['category'] == 'other_same_label_unique_max'


@pytest.mark.parametrize('mismatch', ['scene', 'points', 'root'])
def test_wrong_scene_input_or_target_geometry_rejected(mismatch):
    row, annotation, geometry = fixture()
    if mismatch == 'scene':
        annotation['scene_id'] = 'scene_b'
    elif mismatch == 'points':
        geometry['point_sha256'] = 'different_points'
    else:
        annotation['object_id'] = '42'
    with pytest.raises(AssertionError):
        annotate_row(row, annotation, geometry)
