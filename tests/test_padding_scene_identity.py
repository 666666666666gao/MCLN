from scripts.audit_nr3d_padding_scene import shared_query_indices


def test_permuted_queries_align_to_the_same_input_seed():
    before_ids, after_ids = [11, 22, 33], [33, 11, 22]
    ia, ib = shared_query_indices(before_ids, after_ids)
    assert ia == [0, 1, 2]
    assert ib == [1, 2, 0]
    before_boxes, after_boxes = [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]
    assert [before_boxes[i] for i in ia] == [after_boxes[i] for i in ib]
    # Equal selected Query indices are insufficient to claim equal identity.
    assert before_ids[0] != after_ids[0]


def test_replaced_seeds_are_excluded_from_aligned_box_comparison():
    assert shared_query_indices([11, 22, 33], [33, 44, 11]) == ([0, 2], [2, 0])
    assert shared_query_indices([11], [22]) == ([], [])
