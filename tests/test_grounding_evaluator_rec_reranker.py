import torch
import pytest

import src.grounding_evaluator as grounding_evaluator
from models.rec_geometry_reranker import stable_query_descending_order
from src.grounding_evaluator import GroundingEvaluator


class DummyLogger:
    def info(self, *_args, **_kwargs):
        pass


def _end_points_with_reranker_scores():
    return {
        "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        "modify_positive_map": torch.zeros(1, 1, 4),
        "pron_positive_map": torch.zeros(1, 1, 4),
        "other_entity_map": torch.zeros(1, 1, 4),
        "auxi_entity_positive_map": torch.zeros(1, 1, 4),
        "rel_positive_map": torch.zeros(1, 1, 4),
        "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
        "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
        "box_label_mask": torch.tensor([[1]]),
        "last_center": torch.tensor([[
            [8.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]]),
        "last_pred_size": torch.tensor([[
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ]]),
        "last_sem_cls_scores": torch.tensor([[
            [8.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]]),
        "selected_source_scores": torch.tensor([[10.0, 0.0]]),
        "rec_reranker_scores": torch.tensor([[0.0, 10.0]]),
    }


def _evaluate(use_reranker=False, use_selector=False):
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_selector_choice_scores=use_selector,
        eval_use_rec_reranker_scores=use_reranker,
    )
    evaluator.evaluate_bbox_by_pos_align(
        _end_points_with_reranker_scores(), "last_"
    )
    return evaluator


def test_evaluator_uses_rec_reranker_scores_for_both_rec_thresholds():
    default = _evaluate()
    reranked = _evaluate(use_reranker=True)

    assert default.dets[("last_", 0.25, 1, "bbs")] == 0
    assert default.dets[("last_", 0.5, 1, "bbs")] == 0
    assert reranked.dets[("last_", 0.25, 1, "bbs")] == 1
    assert reranked.dets[("last_", 0.5, 1, "bbs")] == 1


def test_rec_reranker_scores_take_precedence_over_source_choice_scores():
    evaluator = _evaluate(use_reranker=True, use_selector=True)

    assert evaluator.dets[("last_", 0.25, 1, "bbs")] == 1
    assert evaluator.dets[("last_", 0.5, 1, "bbs")] == 1


def test_reranker_negative_infinity_is_safe_with_non_gt_box_filtering():
    end_points = _end_points_with_reranker_scores()
    end_points["rec_reranker_scores"] = torch.tensor([[
        -float("inf"), 10.0
    ]])
    end_points["all_detected_boxes"] = torch.tensor([[
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    ]])
    end_points["all_detected_bbox_label_mask"] = torch.tensor([[True]])
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        filter_non_gt_boxes=True,
        logger=DummyLogger(),
        eval_use_rec_reranker_scores=True,
    )

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert evaluator.dets[("last_", 0.25, 1, "bbs")] == 1
    assert evaluator.dets[("last_", 0.5, 1, "bbs")] == 1


def test_evaluator_fails_when_enabled_reranker_scores_are_missing():
    end_points = _end_points_with_reranker_scores()
    del end_points["rec_reranker_scores"]
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_rec_reranker_scores=True,
    )

    with pytest.raises(ValueError, match="rec_reranker_scores"):
        evaluator.evaluate_bbox_by_pos_align(end_points, "last_")


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
        torch.tensor([[-float("inf"), -float("inf")]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
    ],
)
def test_evaluator_rejects_invalid_reranker_scores(scores):
    end_points = _end_points_with_reranker_scores()
    end_points["rec_reranker_scores"] = scores
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_rec_reranker_scores=True,
    )

    with pytest.raises(ValueError, match="rec_reranker_scores"):
        evaluator.evaluate_bbox_by_pos_align(end_points, "last_")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_zero_weight_parent_axis_uses_canonical_query_tie_without_mutation(
        device, monkeypatch):
    end_points = _end_points_with_reranker_scores()
    end_points = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in end_points.items()
    }
    end_points["last_sem_cls_scores"] = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],
    ]], device=device)
    end_points["rec_reranker_scores"] = torch.tensor(
        [[7.0, 7.0]], device=device
    )
    end_points["rec_geometry_runtime_mode"] = "parent_query_axis"
    original_scores = end_points["rec_reranker_scores"].clone()
    calls = []

    def recording_order(scores):
        calls.append(scores.clone())
        return stable_query_descending_order(scores)

    monkeypatch.setattr(
        grounding_evaluator,
        "stable_query_descending_order",
        recording_order,
        raising=False,
    )
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_rec_reranker_scores=False,
        eval_use_rec_geometry_reranker_scores=True,
    )

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert len(calls) == 1
    assert torch.equal(end_points["rec_reranker_scores"], original_scores)
    assert evaluator.dets[("last_", 0.25, 1, "bbs")] == 0
