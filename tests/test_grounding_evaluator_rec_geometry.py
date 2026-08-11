import pytest
import torch

import src.grounding_evaluator as grounding_evaluator
from models.rec_geometry_reranker import stable_flat_descending_indices
from src.grounding_evaluator import GroundingEvaluator


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *_args, **_kwargs):
        self.messages.append(" ".join(str(arg) for arg in _args))


def _base_end_points(device="cpu"):
    device = torch.device(device)
    return {
        "positive_map": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]]], device=device
        ),
        "modify_positive_map": torch.zeros(1, 1, 4, device=device),
        "pron_positive_map": torch.zeros(1, 1, 4, device=device),
        "other_entity_map": torch.zeros(1, 1, 4, device=device),
        "auxi_entity_positive_map": torch.zeros(1, 1, 4, device=device),
        "rel_positive_map": torch.zeros(1, 1, 4, device=device),
        "center_label": torch.tensor(
            [[[0.0, 0.0, 0.0]]], device=device
        ),
        "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]], device=device),
        "box_label_mask": torch.tensor([[1]], device=device),
        "last_center": torch.tensor(
            [[[8.0, 0.0, 0.0], [12.0, 0.0, 0.0]]], device=device
        ),
        "last_pred_size": torch.tensor(
            [[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]], device=device
        ),
        "last_sem_cls_scores": torch.tensor(
            [[[8.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            device=device,
        ),
        "proj_tokens": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]],
            device=device,
        ),
        "last_proj_queries": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], device=device
        ),
        "selected_source_scores": torch.tensor(
            [[10.0, 0.0]], device=device
        ),
        "rec_reranker_scores": torch.tensor(
            [[10.0, 0.0]], device=device
        ),
        "is_unique": torch.tensor([True], device=device),
        "is_hard": torch.tensor([False], device=device),
        "is_view_dep": torch.tensor([True], device=device),
    }


def _attach_geometry(end_points, boxes=None, scores=None, valid=None,
                     fallback=0):
    device = end_points["last_center"].device
    if boxes is None:
        boxes = torch.tensor([[
            [8.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]], device=device)
    if scores is None:
        scores = torch.tensor([[0.0, 10.0, -float("inf")]], device=device)
    if valid is None:
        valid = torch.tensor([[True, True, False]], device=device)
    end_points.update({
        "rec_geometry_runtime_mode": "flat_geometry_axis",
        "rec_geometry_boxes": boxes,
        "rec_geometry_scores": scores,
        "rec_geometry_valid_mask": valid,
        "rec_geometry_fallback_index": torch.tensor(
            [fallback], dtype=torch.long, device=device
        ),
    })
    return end_points


def _geometry_evaluator(topks=(1,), filter_non_gt_boxes=False,
                        use_parent=True):
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=list(topks),
        prefixes=["last_"],
        filter_non_gt_boxes=filter_non_gt_boxes,
        logger=DummyLogger(),
        eval_use_selector_choice_scores=True,
        eval_use_rec_reranker_scores=use_parent,
        eval_use_rec_geometry_reranker_scores=True,
    )
    return evaluator


def _metric(evaluator, threshold=0.25, topk=1):
    return evaluator.dets[("last_", threshold, topk, "bbs")]


def _position_group(evaluator, group, threshold=0.25):
    key = ("position_subgroup", group, threshold)
    return evaluator.dets[key], evaluator.gts[key]


def test_geometry_updates_namespaced_position_unique_subgroups_only():
    end_points = _attach_geometry(_base_end_points())
    evaluator = _geometry_evaluator()

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _position_group(evaluator, "unique", 0.25) == (1, 1)
    assert _position_group(evaluator, "unique", 0.5) == (1, 1)
    assert _position_group(evaluator, "multiple", 0.25) == (0, 0)
    assert _position_group(evaluator, "easy", 0.25) == (1, 1)
    assert _position_group(evaluator, "hard", 0.25) == (0, 0)
    assert _position_group(
        evaluator, "view_dependent", 0.25
    ) == (1, 1)
    assert _position_group(
        evaluator, "view_independent", 0.25
    ) == (0, 0)
    assert evaluator.dets["unique"] == 0
    assert evaluator.gts["unique"] == pytest.approx(1e-14)


def test_geometry_updates_namespaced_position_multiple_subgroups_only():
    end_points = _attach_geometry(_base_end_points())
    end_points["is_unique"].fill_(False)
    end_points["is_hard"].fill_(True)
    end_points["is_view_dep"].fill_(False)
    evaluator = _geometry_evaluator()

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _position_group(evaluator, "multiple", 0.25) == (1, 1)
    assert _position_group(evaluator, "unique", 0.25) == (0, 0)
    assert _position_group(evaluator, "hard", 0.25) == (1, 1)
    assert _position_group(evaluator, "easy", 0.25) == (0, 0)
    assert _position_group(
        evaluator, "view_independent", 0.25
    ) == (1, 1)
    assert _position_group(
        evaluator, "view_dependent", 0.25
    ) == (0, 0)


def test_geometry_override_changes_position_but_not_semantic_subgroup():
    good_end_points = _attach_geometry(_base_end_points())
    bad_end_points = _attach_geometry(
        _base_end_points(), scores=torch.tensor([[10.0, 0.0, -float("inf")]])
    )
    text_map_keys = (
        "positive_map",
        "modify_positive_map",
        "pron_positive_map",
        "other_entity_map",
        "auxi_entity_positive_map",
        "rel_positive_map",
    )
    for end_points in (good_end_points, bad_end_points):
        for key in text_map_keys:
            padded = torch.zeros(1, 1, 256)
            padded[:, :, :4] = end_points[key]
            end_points[key] = padded
    good = _geometry_evaluator()
    bad = _geometry_evaluator()

    for evaluator, end_points in (
            (good, good_end_points), (bad, bad_end_points)):
        evaluator.evaluate_bbox_by_pos_align(end_points, "last_")
        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

    assert _position_group(good, "unique", 0.25) == (1, 1)
    assert _position_group(bad, "unique", 0.25) == (0, 1)
    assert good.dets["unique"] == bad.dets["unique"] == 0
    assert good.gts["unique"] == pytest.approx(bad.gts["unique"])


@pytest.mark.parametrize("threshold", [0.25, 0.5])
def test_position_subgroup_threshold_equality_is_a_miss(
        monkeypatch, threshold):
    end_points = _attach_geometry(_base_end_points())
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[threshold],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_rec_geometry_reranker_scores=True,
    )

    def exact_boundary_iou(_gt_boxes, _pred_boxes):
        return torch.tensor([[threshold]]), None

    monkeypatch.setattr(
        grounding_evaluator, "_iou3d_par", exact_boundary_iou
    )

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _metric(evaluator, threshold) == 0
    assert _position_group(evaluator, "unique", threshold) == (0, 1)


def test_position_subgroup_stats_print_exact_counts_and_ratio():
    evaluator = _geometry_evaluator()
    key = ("position_subgroup", "unique", 0.25)
    evaluator.dets[key] = 2
    evaluator.gts[key] = 3

    evaluator.print_stats()

    assert (
        "position subgroup unique Acc0.25: "
        "hits=2, total=3, accuracy=0.666666666667"
    ) in evaluator.logger.messages


def test_position_subgroup_counters_merge_between_processes(monkeypatch):
    evaluator = _geometry_evaluator()
    key = ("position_subgroup", "multiple", 0.5)
    first_dets = dict(evaluator.dets)
    second_dets = dict(evaluator.dets)
    first_gts = dict(evaluator.gts)
    second_gts = dict(evaluator.gts)
    first_dets[key], second_dets[key] = 2, 3
    first_gts[key], second_gts[key] = 4, 5
    gathered = [
        [first_dets, second_dets],
        [first_gts, second_gts],
    ]

    monkeypatch.setattr(
        grounding_evaluator.misc,
        "all_gather",
        lambda _value: gathered.pop(0),
    )
    monkeypatch.setattr(
        grounding_evaluator.misc, "is_main_process", lambda: True
    )

    evaluator.synchronize_between_processes()

    assert evaluator.dets[key] == 5
    assert evaluator.gts[key] == 9


def test_missing_subgroup_metadata_preserves_existing_position_metric():
    end_points = _attach_geometry(_base_end_points())
    for key in ("is_unique", "is_hard", "is_view_dep"):
        del end_points[key]
    evaluator = _geometry_evaluator()

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _metric(evaluator, 0.25) == 1
    assert _position_group(evaluator, "unique", 0.25) == (0, 0)
    assert _position_group(evaluator, "multiple", 0.25) == (0, 0)


def test_geometry_flag_defaults_off_and_can_be_enabled():
    default = GroundingEvaluator()
    enabled = GroundingEvaluator(eval_use_rec_geometry_reranker_scores=True)

    assert default.eval_use_rec_geometry_reranker_scores is False
    assert enabled.eval_use_rec_geometry_reranker_scores is True


def test_flat_geometry_axis_takes_precedence_over_parent_and_default_scores():
    end_points = _attach_geometry(_base_end_points())
    evaluator = _geometry_evaluator()

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _metric(evaluator, 0.25) == 1
    assert _metric(evaluator, 0.5) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__(
            "rec_geometry_boxes", value["rec_geometry_boxes"][:, :2]
        ),
        lambda value: value.__setitem__(
            "rec_geometry_scores", value["rec_geometry_scores"][:, :2]
        ),
        lambda value: value.__setitem__(
            "rec_geometry_valid_mask",
            value["rec_geometry_valid_mask"][:, :2],
        ),
        lambda value: value["rec_geometry_scores"].__setitem__(
            (0, 0), float("nan")
        ),
        lambda value: value["rec_geometry_scores"].__setitem__(
            (0, 0), float("inf")
        ),
        lambda value: value["rec_geometry_boxes"].__setitem__(
            (0, 0, 0), float("nan")
        ),
        lambda value: value["rec_geometry_boxes"].__setitem__(
            (0, 0, 3), 0.0
        ),
        lambda value: value["rec_geometry_fallback_index"].__setitem__(0, 3),
        lambda value: value["rec_geometry_fallback_index"].__setitem__(0, 2),
        lambda value: (
            value["rec_geometry_valid_mask"].fill_(False),
            value["rec_geometry_scores"].fill_(-float("inf")),
        ),
    ],
)
def test_flat_geometry_axis_rejects_malformed_candidate_tensors(mutation):
    end_points = _attach_geometry(_base_end_points())
    mutation(end_points)

    with pytest.raises(ValueError, match="geometry"):
        _geometry_evaluator().evaluate_bbox_by_pos_align(end_points, "last_")


@pytest.mark.parametrize(
    "missing_key",
    [
        "rec_geometry_boxes",
        "rec_geometry_scores",
        "rec_geometry_valid_mask",
        "rec_geometry_fallback_index",
    ],
)
def test_flat_geometry_axis_rejects_partial_attachment(missing_key):
    end_points = _attach_geometry(_base_end_points())
    del end_points[missing_key]

    with pytest.raises(ValueError, match="geometry"):
        _geometry_evaluator().evaluate_bbox_by_pos_align(end_points, "last_")


def test_enabled_geometry_rejects_missing_runtime_mode():
    with pytest.raises(ValueError, match="geometry.*mode"):
        _geometry_evaluator().evaluate_bbox_by_pos_align(
            _base_end_points(), "last_"
        )


def test_parent_runtime_mode_rejects_geometry_tensors():
    end_points = _attach_geometry(_base_end_points())
    end_points["rec_geometry_runtime_mode"] = "parent_query_axis"

    with pytest.raises(ValueError, match="geometry"):
        _geometry_evaluator().evaluate_bbox_by_pos_align(end_points, "last_")


def test_detector_filter_uses_active_geometry_boxes():
    end_points = _attach_geometry(_base_end_points())
    end_points["all_detected_boxes"] = torch.tensor([[
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    ]])
    end_points["all_detected_bbox_label_mask"] = torch.tensor([[True]])
    evaluator = _geometry_evaluator(filter_non_gt_boxes=True)

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _metric(evaluator, 0.25) == 1
    assert _metric(evaluator, 0.5) == 1


def test_detector_filter_fails_closed_when_no_geometry_candidate_remains():
    end_points = _attach_geometry(_base_end_points())
    end_points["all_detected_boxes"] = torch.tensor([[
        [30.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    ]])
    end_points["all_detected_bbox_label_mask"] = torch.tensor([[True]])

    with pytest.raises(ValueError, match="no valid.*candidate"):
        _geometry_evaluator(
            filter_non_gt_boxes=True
        ).evaluate_bbox_by_pos_align(end_points, "last_")


def test_invalid_geometry_candidates_never_fill_top5_or_top10():
    end_points = _base_end_points()
    end_points["last_center"][0, 0].zero_()
    boxes = torch.zeros(1, 12, 6)
    boxes[0, :2] = torch.tensor([
        [8.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        [12.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    ])
    boxes[0, 2:] = torch.tensor(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    )
    scores = torch.tensor([[
        2.0, 1.0, 100.0, 99.0, 98.0, 97.0,
        96.0, 95.0, 94.0, 93.0, 92.0, 91.0,
    ]])
    valid = torch.tensor([[
        True, True, False, False, False, False,
        False, False, False, False, False, False,
    ]])
    _attach_geometry(
        end_points, boxes=boxes, scores=scores, valid=valid, fallback=0
    )
    evaluator = _geometry_evaluator(topks=(1, 5, 10))

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert _metric(evaluator, topk=1) == 0
    assert _metric(evaluator, topk=5) == 0
    assert _metric(evaluator, topk=10) == 0


def test_flat_geometry_topk_calls_stable_order_and_breaks_ties_by_flat_index(
        monkeypatch):
    end_points = _base_end_points()
    boxes = torch.tensor([[
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        [8.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    ]])
    scores = torch.tensor([[5.0, 5.0]])
    valid = torch.tensor([[True, True]])
    _attach_geometry(
        end_points, boxes=boxes, scores=scores, valid=valid, fallback=0
    )
    calls = []

    def recording_order(actual_scores, actual_valid):
        calls.append((actual_scores.clone(), actual_valid.clone()))
        return stable_flat_descending_indices(actual_scores, actual_valid)

    monkeypatch.setattr(
        grounding_evaluator,
        "stable_flat_descending_indices",
        recording_order,
        raising=False,
    )
    evaluator = _geometry_evaluator()

    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

    assert len(calls) == 1
    assert _metric(evaluator) == 1


def test_geometry_attachment_is_ignored_outside_last_position_axis():
    end_points = _attach_geometry(_base_end_points())
    end_points.update({
        "proposal_center": torch.tensor([[
            [0.0, 0.0, 0.0], [8.0, 0.0, 0.0]
        ]]),
        "proposal_pred_size": torch.tensor([[
            [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]
        ]]),
        "proposal_sem_cls_scores": torch.tensor([[
            [8.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        ]]),
    })
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["proposal_"],
        logger=DummyLogger(),
        eval_use_rec_geometry_reranker_scores=True,
    )

    evaluator.evaluate_bbox_by_pos_align(end_points, "proposal_")

    assert evaluator.dets[("proposal_", 0.25, 1, "bbs")] == 1


def test_default_query_topk_does_not_rebuild_indices_from_host(
        monkeypatch):
    scores = torch.tensor([
        [1.0, 4.0, 3.0, 2.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    valid = torch.tensor([
        [True, False, True, True],
        [True, True, False, True],
    ])
    expected = torch.tensor([[2, 3], [0, 1]])

    def forbidden_host_rebuild(*_args, **_kwargs):
        raise AssertionError("default ordering must stay on its source device")

    monkeypatch.setattr(
        grounding_evaluator.torch, "tensor", forbidden_host_rebuild
    )

    actual = GroundingEvaluator._position_top_indices(
        scores, valid, "default_query_axis", max_topk=2
    )

    assert torch.equal(actual, expected)
