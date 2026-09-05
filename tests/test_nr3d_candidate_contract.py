import torch
import pytest

from scripts.nr3d_candidate_contract import ranked_oracle_profile


def test_top16_failure_can_have_full256_coverage():
    scores = torch.arange(256, 0, -1).float()
    ious = torch.zeros(256)
    ious[40] = .8
    result = ranked_oracle_profile(scores, ious, torch.ones(256, dtype=torch.bool))
    assert not result["top_16"]["hit025"]
    assert not result["top_32"]["hit025"]
    assert result["top_64"]["hit050"]
    assert result["top_256"]["hit050"]
    assert result["box_oracle_query"] == 40


def test_filter_loss_is_separate_from_raw_candidate_coverage():
    scores, ious = torch.tensor([.9, .8]), torch.tensor([.8, .1])
    before = ranked_oracle_profile(scores, ious, torch.tensor([True, True]))
    after = ranked_oracle_profile(scores, ious, torch.tensor([False, True]))
    assert before["top_256"]["hit050"]
    assert not after["top_256"]["hit025"]
    assert after["top_query"] == 1
    assert after["top_16"]["available"] == 1


def test_filtered_ranking_uses_native_mask_before_sort_order():
    from src.grounding_evaluator import GroundingEvaluator

    scores = torch.ones(256)
    valid = torch.arange(256) % 3 == 0
    ious = torch.arange(256).double() / 256
    native = GroundingEvaluator._position_top_indices(
        scores.unsqueeze(0), valid.unsqueeze(0), "default_query_axis", 256)[0]
    result = ranked_oracle_profile(scores, ious, valid)
    assert result["top_query"] == int(native[0])
    for count in (16, 32, 64, 256):
        for suffix, threshold in (("025", .25), ("050", .5)):
            assert result["top_{}".format(count)]["hit" + suffix] == bool(
                (ious[native[:count]] > threshold).any())


def test_empty_legal_set_has_no_query_and_thresholds_are_strict():
    scores, ious = torch.tensor([.9, .8]), torch.tensor([.25, .50])
    result = ranked_oracle_profile(scores, ious, torch.tensor([False, False]))
    assert result["top_query"] is None
    assert result["box_oracle_query"] is None
    assert not result["top_256"]["hit025"]
    result = ranked_oracle_profile(scores, ious, torch.tensor([True, True]))
    assert result["top_16"]["hit025"]
    assert not result["top_16"]["hit050"]


@pytest.mark.parametrize("raw_size", [0.0, -1.0])
def test_direct_forward_diagnostic_matches_formal_size_preprocessing(raw_size):
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates

    logits = torch.tensor([[-5., -5., 5., 5.], [5., 5., -5., -5.]])
    positive = torch.tensor([[[1., 0.]]])
    end_points = {
        "last_center": torch.tensor([[[10., 0., 0.], [0., 0., 0.]]]),
        "last_pred_size": torch.tensor([[[raw_size, 1., 1.], [1., 1., 1.]]]),
        "last_sem_cls_scores": torch.tensor([[[4., 0.], [3., 0.]]]),
        "selected_source_scores": torch.tensor([[.9, .8]]),
        "center_label": torch.zeros(1, 1, 3), "size_gts": torch.ones(1, 1, 3),
        "box_label_mask": torch.ones(1, 1, dtype=torch.bool),
        "positive_map": positive,
        "all_detected_boxes": torch.tensor([[[0., 0., 0., 1., 1., 1.]]]),
        "all_detected_bbox_label_mask": torch.ones(1, 1, dtype=torch.bool),
        "last_pred_masks": [logits.unsqueeze(0)], "sp_last_pred_masks": [logits],
        "adaptive_weights": [torch.tensor(.5)],
        "superpoints": torch.tensor([[0, 1, 2, 3]]),
        "gt_masks": torch.tensor([[[1, 1, 0, 0]]]),
    }
    for key in ["modify_positive_map", "pron_positive_map", "other_entity_map",
                "auxi_entity_positive_map", "rel_positive_map"]:
        end_points[key] = torch.zeros_like(positive)
    evaluator = GroundingEvaluator(
        only_root=True, prefixes=["last_"], topks=[1],
        filter_non_gt_boxes=True, eval_use_selector_choice_scores=True,
    )
    row = diagnose_root_candidates(end_points, evaluator)[0]
    assert row["raw_nonpositive_size_query_count"] == 1
    assert row["raw_prediction_size_min"] == raw_size
    assert row["rec_selection"] == {"query": 1, "box_iou": 1.0, "mask_iou": 1.0}
    assert row["score_profiles"]["default"]["before_filter"]["top_query"] == 0
    assert row["score_profiles"]["default"]["after_filter"]["top_query"] == 1

    # Apply the existing main evaluator's preprocessing, then run that evaluator.
    formal_inputs = dict(end_points)
    formal_inputs["last_pred_size"] = end_points["last_pred_size"].clamp(min=1e-6)
    evaluator.evaluate_bbox_by_pos_align(formal_inputs, "last_")
    assert evaluator.dets[("last_", .25, 1, "bbs")] == 1
    assert end_points["last_pred_size"][0, 0, 0] == raw_size
