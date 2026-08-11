import torch

from src.grounding_evaluator import GroundingEvaluator


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *_args, **_kwargs):
        if _args:
            self.messages.append(str(_args[0]))


def _end_points_with_selected_scores():
    return {
        "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        "modify_positive_map": torch.zeros(1, 1, 4),
        "pron_positive_map": torch.zeros(1, 1, 4),
        "other_entity_map": torch.zeros(1, 1, 4),
        "auxi_entity_positive_map": torch.zeros(1, 1, 4),
        "rel_positive_map": torch.zeros(1, 1, 4),
        "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
        "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
        "box_label_mask": torch.tensor([[1]]),
        "last_center": torch.tensor([[[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]]]),
        "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        "last_sem_cls_scores": torch.tensor([[[8.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        "selected_source_scores": torch.tensor([[0.0, 10.0]]),
    }


def test_evaluator_uses_selected_source_scores_for_bbox_ranking():
    default_eval = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )
    default_eval.evaluate_bbox_by_pos_align(_end_points_with_selected_scores(), "last_")
    assert default_eval.dets[("last_", 0.25, 1, "bbs")] == 0

    selector_eval = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_selector_choice_scores=True,
    )
    selector_eval.evaluate_bbox_by_pos_align(_end_points_with_selected_scores(), "last_")
    assert selector_eval.dets[("last_", 0.25, 1, "bbs")] == 1


def test_evaluator_reports_fixed_learned_and_oracle_source_choice_metrics():
    end_points = _end_points_with_selected_scores()
    end_points["source_choice_source_scores"] = {
        "default": torch.tensor([[10.0, 0.0]]),
        "mask_text": torch.tensor([[0.0, 10.0]]),
    }
    end_points["selector_choice_source_names"] = ["default", "mask_text"]

    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    evaluator.evaluate_source_choice_diagnostics(end_points, "last_")

    assert evaluator.dets[("source_choice", "fixed_default", 0.25, 1)] == 0
    assert evaluator.dets[("source_choice", "fixed_mask_text", 0.25, 1)] == 1
    assert evaluator.dets[("source_choice", "learned_selector", 0.25, 1)] == 1
    assert evaluator.dets[("source_choice", "oracle", 0.25, 1)] == 1
    assert evaluator.dets[("source_choice", "fixed_mask_text", 0.5, 1)] == 1
    assert evaluator.gts[("source_choice", "oracle", 0.25, 1)] == 1


def test_evaluator_records_source_choice_gap_diagnostics():
    end_points = _end_points_with_selected_scores()
    end_points["source_choice_source_scores"] = {
        "default": torch.tensor([[10.0, 0.0]]),
        "mask_text": torch.tensor([[0.0, 10.0]]),
    }
    end_points["selector_choice_source_names"] = ["default", "mask_text"]
    end_points["selected_source_id"] = torch.tensor([1])

    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    evaluator.evaluate_source_choice_diagnostics(end_points, "last_")

    assert evaluator.gts[("source_choice_mean_iou", "fixed_default")] == 1
    assert evaluator.dets[("source_choice_mean_iou", "fixed_default")] == 0
    assert evaluator.dets[("source_choice_mean_iou", "fixed_mask_text")] == 1
    assert evaluator.dets[("source_choice_mean_iou", "learned_selector")] == 1
    assert evaluator.dets[("source_choice_selected_source", "mask_text")] == 1
    assert evaluator.gts[("source_choice_selected_source", "mask_text")] == 1
    assert evaluator.dets[("source_choice_effect", 0.25, "selector_fix")] == 1
    assert evaluator.dets[("source_choice_effect", 0.25, "selector_break")] == 0
    assert evaluator.dets[("source_choice_effect", 0.25, "oracle_headroom")] == 1

    evaluator.print_stats()
    assert any("Source choice mean IoU" in msg for msg in evaluator.logger.messages)
    assert any("Source choice threshold effects" in msg for msg in evaluator.logger.messages)


def test_evaluator_reports_gate_candidate_set_oracle_headroom():
    end_points = _end_points_with_selected_scores()
    end_points["last_center"] = torch.tensor([[
        [5.0, 5.0, 5.0],
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0],
    ]])
    end_points["last_pred_size"] = torch.ones(1, 3, 3)
    end_points["last_sem_cls_scores"] = torch.zeros(1, 3, 4)
    end_points["selected_source_scores"] = torch.tensor([[10.0, 0.0, -1.0]])
    end_points["source_choice_source_scores"] = {
        "default": torch.tensor([[10.0, 0.0, -1.0]]),
        "mask_text": torch.tensor([[0.0, 10.0, -1.0]]),
    }
    end_points["selector_choice_source_names"] = ["default", "mask_text"]
    end_points["moe_gate_candidate_mask"] = torch.tensor([
        [False, False, True],
    ])
    end_points["moe_gate_default_query"] = torch.tensor([0])
    end_points["moe_valid_mask"] = torch.ones(1, 3, dtype=torch.bool)

    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    evaluator.evaluate_source_choice_diagnostics(end_points, "last_")

    assert evaluator.dets[(
        "source_choice", "oracle", 0.25, 1
    )] == 0
    assert evaluator.dets[(
        "source_choice", "gate_candidate_oracle", 0.25, 1
    )] == 1
    assert evaluator.dets[(
        "source_choice_effect", 0.25, "gate_oracle_headroom"
    )] == 1
    assert evaluator.dets[(
        "source_choice_mean_iou", "gate_candidate_oracle"
    )] == 1

    diagnostics = evaluator.export_source_choice_diagnostics(
        expected_sample_count=1
    )
    assert diagnostics == {
        "schema": "mcln-source-choice-diagnostics-v1",
        "sample_count": 1,
        "gate_candidate_oracle": {
            "hits025": 1,
            "hits050": 1,
            "iou_sum": 1.0,
            "miou": 1.0,
        },
        "gate_oracle_headroom": {
            "hits025": 1,
            "hits050": 1,
            "rate025": 1.0,
            "rate050": 1.0,
        },
    }


def test_gate_candidate_oracle_includes_dynamic_cascade_anchor():
    end_points = _end_points_with_selected_scores()
    end_points["source_choice_source_scores"] = {
        "default": torch.tensor([[10.0, 0.0]]),
    }
    end_points["moe_gate_candidate_mask"] = torch.tensor([[True, False]])
    end_points["moe_gate_default_query"] = torch.tensor([0])
    end_points["moe_gate_action_anchor_query"] = torch.tensor([1])
    end_points["moe_valid_mask"] = torch.ones(1, 2, dtype=torch.bool)
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    evaluator.evaluate_source_choice_diagnostics(end_points, "last_")

    assert evaluator.dets[(
        "source_choice", "gate_candidate_oracle", 0.25, 1
    )] == 1
    assert evaluator.dets[(
        "source_choice", "gate_candidate_oracle", 0.5, 1
    )] == 1


def test_gate_candidate_oracle_prefers_deployed_v19_fallback():
    end_points = _end_points_with_selected_scores()
    end_points["source_choice_source_scores"] = {
        "default": torch.tensor([[10.0, 0.0]]),
    }
    end_points["moe_gate_candidate_mask"] = torch.tensor([[True, False]])
    end_points["moe_gate_default_query"] = torch.tensor([0])
    end_points["moe_gate_action_anchor_query"] = torch.tensor([0])
    end_points["moe_gate_supervision_fallback_query"] = torch.tensor([1])
    end_points["moe_valid_mask"] = torch.ones(1, 2, dtype=torch.bool)
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    evaluator.evaluate_source_choice_diagnostics(end_points, "last_")

    assert evaluator.dets[(
        "source_choice", "gate_candidate_oracle", 0.25, 1
    )] == 1
    assert evaluator.dets[(
        "source_choice", "gate_candidate_oracle", 0.5, 1
    )] == 1


def test_evaluator_returns_no_gate_diagnostics_without_gate_metadata():
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    assert evaluator.export_source_choice_diagnostics() is None


def test_evaluator_rejects_partial_gate_oracle_metadata():
    end_points = _end_points_with_selected_scores()
    end_points["source_choice_source_scores"] = {
        "default": torch.tensor([[10.0, 0.0]]),
    }
    end_points["moe_gate_candidate_mask"] = torch.tensor([[False, True]])
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
    )

    try:
        evaluator.evaluate_source_choice_diagnostics(end_points, "last_")
    except ValueError as error:
        assert "candidate mask and default query" in str(error)
    else:
        raise AssertionError("partial gate oracle metadata must fail closed")


def test_learned_mask_query_uses_same_query_as_learned_box_ranking():
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_selector_choice_scores=True,
    )
    end_points = {
        "box_label_mask": torch.ones(2, 1),
        "selected_source_scores": torch.tensor([
            [0.0, 3.0, 1.0],
            [2.0, 0.0, 1.0],
        ]),
        "moe_valid_mask": torch.tensor([
            [True, True, True],
            [True, False, True],
        ]),
    }
    selected = evaluator._resolve_learned_mask_queries(
        end_points, "last_", 3, torch.device("cpu")
    )
    assert torch.equal(selected, torch.tensor([1, 0]))


def test_learned_mask_query_is_disabled_for_nonfinal_layers():
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25],
        topks=[1],
        prefixes=["last_"],
        logger=DummyLogger(),
        eval_use_selector_choice_scores=True,
    )
    assert evaluator._resolve_learned_mask_queries(
        {}, "0head_", 3, torch.device("cpu")
    ) is None
