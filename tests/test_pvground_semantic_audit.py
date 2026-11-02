import ast
from collections import defaultdict
import io
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from tools.pvground_semantic_audit import PVGroundSemanticAudit


def reference_functions():
    """Execute the tracked evaluator/math functions without CUDA-only imports."""
    root = Path(__file__).resolve().parents[1]
    namespace = {"torch": torch}
    tree = ast.parse((root / "models/losses.py").read_text(encoding="utf-8"))
    names = {
        "box_cxcyczwhd_to_xyzxyz", "_volume_par", "_intersect_par", "_iou3d_par"
    }
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(tree, "tracked_iou_functions", "exec"), namespace)
    tree = ast.parse((root / "src/grounding_evaluator.py").read_text(encoding="utf-8"))
    evaluator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroundingEvaluator"
    )
    tree.body = [
        node for node in evaluator.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_parse_gt", "evaluate_bbox_by_sem_align")
    ]
    exec(compile(tree, "tracked_semantic_evaluator", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_rows_match_tracked_evaluator_selection_iou_and_counts(seed):
    torch.manual_seed(seed)
    batch, queries, tokens = 4, 12, 6
    end_points = {
        "last_proj_queries": torch.randn(batch, queries, 8),
        "proj_tokens": torch.randn(batch, tokens, 8),
        "last_center": torch.randn(batch, queries, 3) * 0.3,
        "last_pred_size": torch.rand(batch, queries, 3) + 0.7,
        "center_label": torch.zeros(batch, 1, 3),
        "size_gts": torch.ones(batch, 1, 3),
        "box_label_mask": torch.ones(batch, 1),
        "point_clouds": torch.randn(batch, 8, 6),
        "scan_ids": ["scene" + str(index) for index in range(batch)],
        "target_id": torch.arange(batch),
        "utterances": ["the object"] * batch,
        "is_unique": torch.tensor([True, False, True, False]),
        "is_view_dep": torch.zeros(batch, dtype=torch.bool),
        "is_hard": torch.zeros(batch, dtype=torch.bool),
        "mask_reference_valid": torch.ones(batch, queries, dtype=torch.bool),
    }
    for name in (
        "positive_map", "modify_positive_map", "pron_positive_map",
        "other_entity_map", "rel_positive_map", "auxi_entity_positive_map",
    ):
        value = torch.zeros(batch, 1, 256)
        value[..., :tokens] = torch.rand(batch, 1, tokens)
        end_points[name] = value
    end_points["native_coarse_center"] = end_points["last_center"] + 0.1
    end_points["native_coarse_size"] = end_points["last_pred_size"] + 0.1
    end_points["mask_reference_center"] = end_points["last_center"] - 0.1
    end_points["mask_reference_size"] = end_points["last_pred_size"] - 0.1

    namespace = reference_functions()
    iou = namespace["_iou3d_par"]
    selected_boxes, selected_ious = [], []

    def capture_iou(gt, predictions):
        result = iou(gt, predictions)
        selected_boxes.append(predictions[0])
        selected_ious.append(float(result[0][0, 0]))
        return result

    namespace["_iou3d_par"] = capture_iou
    evaluator = SimpleNamespace(
        only_root=True, thresholds=[0.25, 0.5], topks=[1, 5, 10],
        filter_non_gt_boxes=False, bad_case_visualization=False,
        kps_points_visualization=False,
        dets=defaultdict(int), gts=defaultdict(int),
    )
    evaluator._parse_gt = MethodType(namespace["_parse_gt"], evaluator)
    namespace["evaluate_bbox_by_sem_align"](evaluator, end_points, "last_")

    stream = io.StringIO()
    audit = PVGroundSemanticAudit(
        stream, iou, namespace["box_cxcyczwhd_to_xyzxyz"]
    )
    raw_gate = torch.randn(batch, queries, 3)
    gate_evidence = [
        {"raw": values, "clamped": values.clamp(0, 1)}
        for values in raw_gate
    ]
    audit.update(end_points, gate_evidence)
    audit.verify(evaluator)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    for index, row in enumerate(rows):
        assert row["selected_iou"] == pytest.approx(selected_ious[index])
        final = torch.tensor(row["final_box"])
        assert torch.allclose(
            namespace["box_cxcyczwhd_to_xyzxyz"](final), selected_boxes[index]
        )
        assert row["scan_id"] == end_points["scan_ids"][index]
        assert row["native_center_error"] == pytest.approx(
            end_points["native_coarse_center"][index, row["selected_query"]].tolist()
        )
        assert row["oracle_iou"] >= row["selected_iou"]
        assert row["selected_raw_gate"] == raw_gate[index, row["selected_query"]].tolist()
        assert len(row["point_sha256"]) == 64
    evaluator.dets[("last_", 0.25, 1, "bbf")] += 1
    with pytest.raises(ValueError, match="disagrees"):
        audit.verify(evaluator)


def test_rank_uses_strict_threshold_and_reports_missing_coverage():
    ordered = torch.tensor([2, 0, 1])
    ious = torch.tensor([0.25, 0.5, 0.1])
    assert PVGroundSemanticAudit._first_good(ious, ordered, 0.25) == 3
    assert PVGroundSemanticAudit._first_good(ious, ordered, 0.5) is None
