from scripts.summarize_nr3d_official_candidate_audit import failure_partition


def row(rec_iou, top16, full_legal, full_raw):
    return {"rec_selection": {"box_iou": rec_iou}, "score_profiles": {
        "protected_selector": {"after_filter": {"top_16": {"hit025": top16},
                                                "top_256": {"hit025": full_legal}},
                               "before_filter": {"top_256": {"hit025": full_raw}}}}}


def test_topk_failure_is_not_automatically_a_proposal_failure():
    assert failure_partition(row(.3, True, True, True), 16) == "correct"
    assert failure_partition(row(.2, True, True, True), 16) == "reselectable_within_legal_topk"
    assert failure_partition(row(.2, False, True, True), 16) == "qualifying_box_only_beyond_legal_topk"
    assert failure_partition(row(.2, False, False, True), 16) == "qualifying_boxes_removed_by_filter"
    assert failure_partition(row(.2, False, False, False), 16) == "full256_no_qualifying_box"
