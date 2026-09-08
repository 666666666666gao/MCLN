import copy

import torch

from models.candidate_edge_adapter import build_candidate_edge_inputs
from models.candidate_edge_direct_scorer import CandidateEdgeDirectScorer


def fixture():
    torch.manual_seed(31)
    points = {
        "last_center": torch.tensor([[[10., 0., 0.], [0., 0., 0.],
                                      [2., 0., 0.], [4., 0., 0.]]]),
        "last_pred_size": torch.ones(1, 4, 3),
        "last_sem_cls_scores": torch.tensor([[[4., 0., 0.], [3., 0., 0.],
                                              [2., 0., 0.], [1., 0., 0.]]]),
        "source_choice_candidate_feats": torch.full((1, 4, 64), -20.),
        "text_feats": torch.full((1, 3, 24), -10.),
        "text_memory": torch.randn(1, 3, 24),
        "text_attention_mask": torch.tensor([[False, False, True]]),
    }
    model_inputs = {
        "det_boxes": torch.tensor([[[0., 0., 0., 1., 1., 1.],
                                    [2., 0., 0., 1., 1., 1.],
                                    [4., 0., 0., 1., 1., 1.]]]),
        "det_bbox_label_mask": torch.ones(1, 3, dtype=torch.bool),
        "positive_map": torch.tensor([[[1., 0., 0.]]]),
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "target_id": torch.tensor([0]),
    }
    for name in ["modify_positive_map", "pron_positive_map",
                 "rel_positive_map", "other_entity_map"]:
        model_inputs[name] = torch.zeros(1, 1, 3)
    return points, model_inputs, torch.randn(1, 4, 24)


def test_adapter_filters_before_selection_and_preserves_full_query_and_text_stage():
    points, model_inputs, unprojected = fixture()
    adapted = build_candidate_edge_inputs(points, model_inputs, unprojected, top_k=2)
    assert adapted["query_indices"].tolist() == [[1, 2]]
    assert adapted["valid_query_mask"].tolist() == [[False, True, True, True]]
    assert adapted["candidate_feats"] is unprojected
    assert adapted["text_feats"] is points["text_memory"]
    assert adapted["text_padding_mask"] is points["text_attention_mask"]
    for mode in ["global", "pair"]:
        model = CandidateEdgeDirectScorer(mode, d_model=24, n_head=4).eval()
        result = model(**adapted)
        assert torch.isneginf(result["query_scores"][0, [0, 3]]).all()
        assert torch.count_nonzero(result["anchor_attention"][..., 0]) == 0
        assert (result["anchor_attention"][..., 3] > 0).all()


def test_root_target_supervision_does_not_change_forward_candidate_inputs():
    points, model_inputs, unprojected = fixture()
    before = build_candidate_edge_inputs(points, model_inputs, unprojected, top_k=2)
    changed = copy.deepcopy(model_inputs)
    changed["center_label"] += 100
    changed["size_gts"] *= 10
    changed["target_id"] += 2
    after = build_candidate_edge_inputs(points, changed, unprojected, top_k=2)
    for name in before:
        assert torch.equal(before[name], after[name])


def test_fewer_legal_targets_keep_invalid_compact_slots_excluded():
    points, model_inputs, unprojected = fixture()
    model_inputs["det_bbox_label_mask"][0, 2] = False
    adapted = build_candidate_edge_inputs(points, model_inputs, unprojected, top_k=3)
    assert adapted["query_indices"][0, :2].tolist() == [1, 2]
    assert adapted["valid_query_mask"].tolist() == [[False, True, True, False]]
    model = CandidateEdgeDirectScorer("pair", d_model=24, n_head=4).eval()
    result = model(**adapted)
    assert result["candidate_valid_mask"].tolist() == [[True, True, False]]
    assert torch.isfinite(result["candidate_logits"][0, :2]).all()
    assert torch.isneginf(result["candidate_logits"][0, 2])
