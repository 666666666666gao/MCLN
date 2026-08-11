import sys
from types import SimpleNamespace

import pytest
import torch

from main_utils import parse_option
from src.grounding_evaluator import GroundingEvaluator
from models.rec_geometry_reranker import build_deployed_parent_state
from train_dist_mod import (
    TrainTester,
    build_rec_geometry_parent_query_indices,
)


class DummyLogger:
    def info(self, *_args, **_kwargs):
        pass


def _mask_end_points():
    """Build a small mask batch with an intentionally non-contiguous mapping."""
    query_count = 4
    token_count = 256
    points = 2
    end_points = {
        "positive_map": torch.cat([
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            torch.zeros(1, 1, token_count - 4),
        ], dim=-1),
        "modify_positive_map": torch.zeros(1, 1, token_count),
        "pron_positive_map": torch.zeros(1, 1, token_count),
        "other_entity_map": torch.zeros(1, 1, token_count),
        "auxi_entity_positive_map": torch.zeros(1, 1, token_count),
        "rel_positive_map": torch.zeros(1, 1, token_count),
        "gt_masks": torch.tensor([[[1, 0]]], dtype=torch.bool),
        "box_label_mask": torch.tensor([[1]], dtype=torch.bool),
        "is_view_dep": torch.tensor([True]),
        "is_unique": torch.tensor([True]),
        "is_hard": torch.tensor([False]),
        "last_center": torch.zeros(1, query_count, 3),
        "last_pred_size": torch.ones(1, query_count, 3),
        # Query 3 wins legacy semantic scoring; query 0 is the geometry parent.
        "last_sem_cls_scores": torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ]]),
        "proj_tokens": torch.cat([
            torch.tensor([[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]]),
            torch.zeros(1, token_count - 4, 2),
        ], dim=1),
        "last_proj_queries": torch.tensor([[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]]),
        "last_pred_masks": torch.tensor([[
            [[10.0, -10.0]],
            [[-10.0, 10.0]],
            [[-10.0, 10.0]],
            [[-10.0, 10.0]],
        ]]).reshape(1, query_count, points),
        "sp_last_pred_masks": torch.tensor([[
            [[10.0, -10.0]],
            [[-10.0, 10.0]],
            [[-10.0, 10.0]],
            [[-10.0, 10.0]],
        ]]).reshape(1, query_count, points),
        "adaptive_weights": torch.tensor([0.5]),
        "superpoints": torch.tensor([[0, 1]], dtype=torch.long),
        "rec_geometry_runtime_mode": "flat_geometry_axis",
        "rec_geometry_boxes": torch.ones(1, 14, 6),
        "rec_geometry_scores": torch.tensor([[
            9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]]),
        "rec_geometry_valid_mask": torch.ones(1, 14, dtype=torch.bool),
        "rec_geometry_fallback_index": torch.tensor([0], dtype=torch.long),
        # Compact candidates [query 3, query 0], repeated query-major over 7 variants.
        "rec_geometry_parent_query_indices": torch.tensor([[
            3, 3, 3, 3, 3, 3, 3,
            0, 0, 0, 0, 0, 0, 0,
        ]], dtype=torch.long),
        "rec_joint_selected_flat_index": torch.tensor([7], dtype=torch.long),
        "rec_joint_selected_parent_position": torch.tensor(
            [1], dtype=torch.long
        ),
        "rec_joint_mask_policy_index": torch.tensor([12], dtype=torch.long),
        "rec_joint_mask_source_index": torch.tensor([2], dtype=torch.long),
        "rec_joint_mask_threshold_index": torch.tensor([2], dtype=torch.long),
        "rec_joint_mask_threshold": torch.tensor([0.0]),
    }
    return end_points


def _evaluator(*, joint=False):
    return GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        model="MCLN",
        eval_use_rec_geometry_reranker_scores=True,
        eval_use_rec_joint_box_mask=joint,
    )


def test_joint_box_mask_flag_defaults_off_and_parser_can_enable_it(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dist_mod.py"])
    args = parse_option()
    assert args.eval_use_rec_joint_box_mask is False
    assert GroundingEvaluator().eval_use_rec_joint_box_mask is False

    monkeypatch.setattr(sys, "argv", [
        "train_dist_mod.py", "--eval_use_rec_joint_box_mask",
    ])
    enabled_args = parse_option()
    assert enabled_args.eval_use_rec_joint_box_mask is True
    assert GroundingEvaluator(
        eval_use_rec_joint_box_mask=True
    ).eval_use_rec_joint_box_mask is True


def test_both_mask_evaluators_use_geometry_parent_query_when_enabled():
    legacy_pos = _evaluator()
    legacy_sem = _evaluator()
    joint_pos = _evaluator(joint=True)
    joint_sem = _evaluator(joint=True)

    legacy_pos.evaluate_masks_by_pos_align(_mask_end_points(), "last_")
    legacy_sem.evaluate_masks_by_sem_align(_mask_end_points(), "last_")
    joint_pos.evaluate_masks_by_pos_align(_mask_end_points(), "last_")
    joint_sem.evaluate_masks_by_sem_align(_mask_end_points(), "last_")

    # Legacy semantic ranking selects query 3 (the wrong mask); the flat
    # geometry winner is index 7, whose original parent is query 0.
    assert legacy_pos.dets["mask_pos"] == 0.0
    assert legacy_sem.dets["mask_sem"] == 0.0
    assert joint_pos.dets["mask_pos"] == 1.0
    assert joint_sem.dets["mask_sem"] == 1.0


def test_joint_mask_policy_applies_learned_source_not_legacy_fusion():
    end_points = _mask_end_points()
    # Query 0's query-mask source is wrong and alpha=0 makes legacy fusion
    # follow it.  The learned policy selects the correct text source at 0.0.
    end_points["sp_last_pred_masks"][0, 0] = torch.tensor([-10.0, 10.0])
    end_points["adaptive_weights"] = torch.tensor([0.0])
    end_points["rec_joint_mask_policy_index"] = torch.tensor(
        [2], dtype=torch.long
    )
    end_points["rec_joint_mask_source_index"] = torch.tensor(
        [0], dtype=torch.long
    )
    end_points["rec_joint_mask_threshold_index"] = torch.tensor(
        [2], dtype=torch.long
    )

    evaluator = _evaluator(joint=True)
    evaluator.evaluate_masks_by_sem_align(end_points, "last_")

    assert evaluator.dets["mask_sem"] == 1.0


def test_joint_mask_policy_fails_closed_when_payload_is_missing():
    end_points = _mask_end_points()
    end_points.pop("rec_joint_mask_source_index")
    with pytest.raises(ValueError, match="mask policy payload"):
        _evaluator(joint=True).evaluate_masks_by_sem_align(
            end_points, "last_"
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.pop("rec_geometry_parent_query_indices"),
         "parent.*mapping"),
        (lambda value: value.__setitem__(
            "rec_geometry_parent_query_indices",
            value["rec_geometry_parent_query_indices"].to(torch.int32),
        ), "int64"),
        (lambda value: value.__setitem__(
            "rec_geometry_parent_query_indices",
            value["rec_geometry_parent_query_indices"][:, :-1],
        ), "shape"),
        (lambda value: value["rec_geometry_parent_query_indices"].__setitem__(
            (0, 7), 4
        ), "range"),
    ],
)
def test_joint_box_mask_fails_closed_on_malformed_parent_mapping(
        mutation, match):
    end_points = _mask_end_points()
    mutation(end_points)
    with pytest.raises((TypeError, ValueError), match=match):
        _evaluator(joint=True).evaluate_masks_by_pos_align(
            end_points, "last_"
        )


def test_joint_box_mask_rejects_parent_axis_without_flat_candidates():
    end_points = _mask_end_points()
    end_points["rec_geometry_runtime_mode"] = "parent_query_axis"
    for key in (
            "rec_geometry_boxes", "rec_geometry_scores",
            "rec_geometry_valid_mask", "rec_geometry_fallback_index"):
        end_points.pop(key)
    with pytest.raises(ValueError, match="flat.*geometry|parent.*mapping"):
        _evaluator(joint=True).evaluate_masks_by_sem_align(
            end_points, "last_"
        )


def _runtime_parent_outputs(query_indices):
    batch_size, candidate_count = query_indices.shape
    valid = torch.ones(batch_size, candidate_count, dtype=torch.bool)
    compact_scores = torch.zeros(batch_size, candidate_count)
    parent_state = build_deployed_parent_state(
        compact_scores, query_indices, valid, 256
    )
    return {
        "candidate_batch": {
            "features": torch.zeros(batch_size, candidate_count, 1),
            "feature_names": ("feature",),
            "query_indices": query_indices,
            "valid_mask": valid,
            "num_queries": 256,
        },
        "compact_scores": compact_scores,
        "query_scores": parent_state["query_scores"],
    }


def _runtime_flat_outputs(batch_size=1):
    return {
        "rec_reranker_scores": torch.zeros(batch_size, 256),
        "rec_geometry_runtime_mode": "flat_geometry_axis",
        "rec_geometry_boxes": torch.ones(batch_size, 112, 6),
        "rec_geometry_scores": torch.zeros(batch_size, 112),
        "rec_geometry_valid_mask": torch.ones(
            batch_size, 112, dtype=torch.bool
        ),
        "rec_geometry_fallback_index": torch.zeros(
            batch_size, dtype=torch.long
        ),
    }


def test_runtime_parent_mapping_repeats_original_query_indices_and_excludes_targets():
    query_indices = torch.tensor([[17, 3, 91, 5, 8, 13, 21, 34, 55, 2, 7, 11,
                                   19, 23, 29, 31]], dtype=torch.long)
    outputs = build_rec_geometry_parent_query_indices(
        _runtime_parent_outputs(query_indices), _runtime_flat_outputs()
    )
    assert outputs.shape == (1, 112)
    assert torch.equal(outputs[0, :7], query_indices[0, :1].expand(7))
    assert torch.equal(outputs[0, 7:14], query_indices[0, 1:2].expand(7))
    assert isinstance(outputs, torch.Tensor)


def test_runtime_parent_mapping_rejects_gt_or_iou_payloads():
    parent = _runtime_parent_outputs(torch.arange(16).reshape(1, 16))
    parent["candidate_batch"]["candidate_ious"] = torch.zeros(1, 16)
    with pytest.raises(ValueError, match="GT|IoU|target"):
        build_rec_geometry_parent_query_indices(parent, _runtime_flat_outputs())

    runtime = _runtime_flat_outputs()
    runtime["candidate_ious"] = torch.zeros(1, 112)
    with pytest.raises(ValueError, match="GT|IoU|target|schema"):
        build_rec_geometry_parent_query_indices(
            _runtime_parent_outputs(torch.arange(16).reshape(1, 16)),
            runtime,
        )


def test_train_tester_attaches_mapping_after_validated_runtime_payload(
        monkeypatch):
    query_indices = torch.tensor([[
        17, 3, 91, 5, 8, 13, 21, 34,
        55, 2, 7, 11, 19, 23, 29, 31,
    ]], dtype=torch.long)
    parent_outputs = _runtime_parent_outputs(query_indices)
    runtime_outputs = _runtime_flat_outputs()
    tester = TrainTester.__new__(TrainTester)
    tester.rec_reranker = object()
    tester.rec_reranker_artifact = object()
    tester.rec_geometry_reranker = object()
    tester.rec_geometry_reranker_artifact = object()
    monkeypatch.setattr(
        "train_dist_mod.validate_rec_geometry_runtime_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tester,
        "_ensure_rec_geometry_runtime_loaded",
        lambda *_args, **_kwargs: {"allow_tf32": True},
    )
    monkeypatch.setattr(
        tester,
        "_ensure_rec_joint_box_mask_runtime_loaded",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        "train_dist_mod.build_rec_reranker_outputs",
        lambda *_args, **_kwargs: parent_outputs,
    )
    monkeypatch.setattr(
        "train_dist_mod.build_rec_geometry_runtime_outputs",
        lambda *_args, **_kwargs: runtime_outputs,
    )
    end_points = {"last_center": torch.zeros(1, 256, 3)}
    args = SimpleNamespace(
        eval_use_rec_reranker_scores=True,
        eval_use_rec_geometry_reranker_scores=True,
        eval_use_rec_joint_box_mask=True,
        rec_joint_box_mask_checkpoint="adapter.pth",
        eval_use_rec_selective_residual_scores=False,
        eval_use_rec_hierarchical_reranker_scores=False,
    )

    tester._attach_rec_reranker_scores(
        end_points, {}, args, batch_idx=0, num_batches=1
    )

    assert torch.equal(
        end_points["rec_geometry_parent_query_indices"][:, :7],
        query_indices[:, :1].expand(1, 7),
    )
    assert not {
        "candidate_ious", "geometry_ious", "gt_masks", "target_iou"
    }.intersection(end_points)


def test_grounding_builder_wires_joint_box_mask_flag():
    tester = TrainTester.__new__(TrainTester)
    tester.logger = DummyLogger()
    args = SimpleNamespace(
        butd_cls=False,
        model="MCLN",
        eval_use_selector_choice_scores=False,
        eval_use_rec_reranker_scores=False,
        eval_use_rec_geometry_reranker_scores=True,
        eval_use_rec_joint_box_mask=True,
    )
    evaluator = tester._build_grounding_evaluator(args, ["last_"])
    assert evaluator.eval_use_rec_joint_box_mask is True
