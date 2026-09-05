"""CPU counterexample for the existing REC/Mask candidate-filter contracts.

This uses synthetic tensors and the real evaluator, not a benchmark dataset.
It documents current behavior; it does not prescribe an evaluation-rule change.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    import torch
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates

    class CapturingEvaluator(GroundingEvaluator):
        def _position_top_indices(self, scores, valid, axis_mode, max_topk):
            top = super()._position_top_indices(scores, valid, axis_mode, max_topk)
            self.rec_query = int(top[0, 0])
            return top

        def _resolve_learned_mask_queries(self, end_points, prefix, count, device):
            top = super()._resolve_learned_mask_queries(end_points, prefix, count, device)
            self.mask_query = int(top[0])
            return top

    def run_case(filter_boxes, explicit_valid):
        logits = torch.tensor([[-5., -5., 5., 5.], [5., 5., -5., -5.]])
        positive = torch.tensor([[[1., 0.]]])
        end_points = {
            "last_center": torch.tensor([[[10., 0., 0.], [0., 0., 0.]]]),
            "last_pred_size": torch.ones(1, 2, 3),
            "last_sem_cls_scores": torch.zeros(1, 2, 2),
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
        for key in ("modify_positive_map", "pron_positive_map", "other_entity_map",
                    "auxi_entity_positive_map", "rel_positive_map"):
            end_points[key] = torch.zeros_like(positive)
        if explicit_valid:
            end_points["moe_valid_mask"] = torch.tensor([[False, True]])
        evaluator = CapturingEvaluator(
            only_root=True, thresholds=[.25, .5], topks=[1], prefixes=["last_"],
            filter_non_gt_boxes=filter_boxes, eval_use_selector_choice_scores=True)
        evaluator.evaluate_bbox_by_pos_align(end_points, "last_")
        evaluator.evaluate_masks_by_pos_align(end_points, "last_")
        point_masks, _ = evaluator._build_mask_point_predictions(end_points, "last_")
        contract = diagnose_root_candidates(end_points, evaluator)[0]
        assert contract["rec_selection"]["query"] == evaluator.rec_query
        assert contract["mask_selection"]["query"] == evaluator.mask_query
        assert contract["mask_selection"]["mask_iou"] == evaluator.dets["mask_pos"]
        return {
            "filter_non_gt_boxes": filter_boxes, "explicit_moe_valid_mask": explicit_valid,
            "rec_query": evaluator.rec_query, "mask_query": evaluator.mask_query,
            "rec_hit025": evaluator.dets[("last_", .25, 1, "bbs")],
            "evaluator_mask_iou": float(evaluator.dets["mask_pos"]),
            "mask_iou_at_rec_query": float(evaluator.calculate_masks_iou(
                point_masks[0, evaluator.rec_query], end_points["gt_masks"][0, 0])),
            "candidate_contract": contract,
        }

    cases = {"no_overlap_filter": run_case(False, False),
             "rec_overlap_filter_only": run_case(True, False),
             "explicit_shared_validity": run_case(True, True)}
    assert cases["no_overlap_filter"]["rec_query"] == cases["no_overlap_filter"]["mask_query"] == 0
    assert cases["rec_overlap_filter_only"]["rec_query"] == 1
    assert cases["rec_overlap_filter_only"]["mask_query"] == 0
    assert cases["rec_overlap_filter_only"]["evaluator_mask_iou"] == 0.
    assert cases["rec_overlap_filter_only"]["mask_iou_at_rec_query"] == 1.
    assert cases["explicit_shared_validity"]["rec_query"] == cases["explicit_shared_validity"]["mask_query"] == 1
    evaluator_source = Path(inspect.getfile(GroundingEvaluator))
    result = {"schema": "mcln-rec-mask-selection-counterexample-v2",
              "synthetic_only": True, "benchmark_rows": 0, "optimizer_steps": 0,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "candidate_contract_sha256": hashlib.sha256(Path(inspect.getfile(diagnose_root_candidates)).read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(evaluator_source.read_bytes()).hexdigest(),
              "evaluator_source": str(evaluator_source), "cases": cases}
    with opt.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
