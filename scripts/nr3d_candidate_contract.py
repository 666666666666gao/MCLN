"""Read-only root-target coverage and REC/Mask query diagnostics."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from models.rec_evaluator_filter import build_detector_overlap_valid
from models.source_choice_adapter import compute_default_source_scores


def ranked_oracle_profile(scores, root_ious, valid):
    order = scores.argsort(descending=True)
    order = order[valid[order]]
    profile = {"candidate_count": int(order.numel()), "top_query": None,
               "top_iou": None, "box_oracle_query": None, "box_oracle_iou": None}
    if order.numel():
        best = order[root_ious[order].argmax()]
        profile.update(top_query=int(order[0]), top_iou=float(root_ious[order[0]]),
                       box_oracle_query=int(best), box_oracle_iou=float(root_ious[best]))
    for count in (16, 32, 64, 256):
        indices = order[:count]
        profile["top_{}".format(count)] = {
            "available": int(indices.numel()),
            "hit025": bool((root_ious[indices] > .25).any()),
            "hit050": bool((root_ious[indices] > .50).any()),
        }
    return profile


def diagnose_root_candidates(end_points, evaluator):
    """Measure the protected selector protocol without changing its outputs.

    Detector-object coverage is only a potential-anchor availability proxy.
    No human-labelled text-anchor mapping is available through this interface.
    """
    assert evaluator.only_root and evaluator.eval_use_selector_choice_scores
    assert not evaluator.eval_use_rec_reranker_scores
    assert not evaluator.eval_use_rec_geometry_reranker_scores
    assert not evaluator.eval_use_rec_joint_box_mask
    boxes = torch.cat([end_points["last_center"], end_points["last_pred_size"]], -1)
    all_valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
    filtered = build_detector_overlap_valid(
        boxes, all_valid, end_points["all_detected_boxes"],
        end_points["all_detected_bbox_label_mask"].bool(), iou_threshold=.25)
    _, _, _, _, _, _, root_boxes = evaluator._parse_gt(end_points)
    point_masks, _ = evaluator._build_mask_point_predictions(end_points, "last_")
    mask_queries = evaluator._resolve_learned_mask_queries(
        end_points, "last_", boxes.shape[1], boxes.device)
    score_sources = {"protected_selector": end_points["selected_source_scores"],
                     "default": compute_default_source_scores(end_points, end_points)}
    rows = []
    for row in range(boxes.shape[0]):
        root_ious = _iou3d_par(box_cxcyczwhd_to_xyzxyz(root_boxes[row]),
                                box_cxcyczwhd_to_xyzxyz(boxes[row]))[0][0]
        profiles = {
            name: {"before_filter": ranked_oracle_profile(scores[row], root_ious, all_valid[row]),
                   "after_filter": ranked_oracle_profile(scores[row], root_ious, filtered[row])}
            for name, scores in score_sources.items()}
        active = profiles["protected_selector"][
            "after_filter" if evaluator.filter_non_gt_boxes else "before_filter"]
        rec_query = active["top_query"]
        mask_query = int(mask_queries[row])
        target_mask = end_points["gt_masks"][row, 0].bool()
        predicted = point_masks[row].bool()
        intersection = (predicted & target_mask.unsqueeze(0)).sum(-1).float()
        union = (predicted | target_mask.unsqueeze(0)).sum(-1).float()
        mask_ious = intersection / union

        def query_quality(index):
            if index is None:
                return None
            return {"query": int(index), "box_iou": float(root_ious[index]),
                    "mask_iou": float(mask_ious[index])}

        detector_ids = end_points["all_detected_bbox_label_mask"][row].nonzero().reshape(-1)
        detector_ious = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(end_points["all_detected_boxes"][row, detector_ids]),
            box_cxcyczwhd_to_xyzxyz(boxes[row]))[0]
        default_order = score_sources["default"][row].argsort(descending=True)
        default_order = default_order[filtered[row, default_order]]
        candidate_sets = {"full_256": all_valid[row].nonzero().reshape(-1),
                          "valid_queries": filtered[row].nonzero().reshape(-1),
                          "valid_default_top32": default_order[:32]}
        object_coverage = {
            name: detector_ids[(detector_ious[:, indices] > .25).any(-1)].tolist()
            for name, indices in candidate_sets.items()}
        rows.append({
            "score_profiles": profiles,
            "root_target_input_points": int(target_mask.sum()),
            "rec_selection": query_quality(rec_query),
            "mask_selection": query_quality(mask_query),
            "rec_and_mask_same_query": rec_query == mask_query,
            "box_oracle_before_filter": query_quality(profiles["protected_selector"]["before_filter"]["box_oracle_query"]),
            "box_oracle_after_filter": query_quality(profiles["protected_selector"]["after_filter"]["box_oracle_query"]),
            "mask_oracle_all_queries": query_quality(mask_ious.argmax()),
            "detector_object_coverage": {
                "active_detector_slots": detector_ids.tolist(),
                "covered_slots": object_coverage,
                "is_text_anchor_ground_truth": False,
            },
        })
    return rows
