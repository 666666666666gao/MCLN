"""Observe existing text, Query and fused Masks without changing predictions."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from models.mask_fusion import as_query_mask_logits, fuse_query_mask_logits
from models.rec_evaluator_filter import build_detector_overlap_valid
from scripts.nr3d_candidate_contract import diagnose_root_candidates as diagnose_candidates


def superpoint_mask_ious(logits, superpoints, target_mask):
    """Exact point-count IoU of fixed logit>0 masks, without dense expansion."""
    total = torch.bincount(superpoints, minlength=logits.shape[-1]).double()
    positive = torch.bincount(superpoints[target_mask], minlength=logits.shape[-1]).double()
    foreground = (logits > 0).double()
    intersection = foreground @ positive
    union = target_mask.sum().double() + foreground @ (total - positive)
    return intersection / union


def summarize_branches(ious, box_ious, legal, alpha, selected_queries):
    """Keep actual selections and GT-only branch oracles explicitly separate."""
    good_box = legal & (box_ious > .5)

    def at(index):
        if index is None:
            return None
        return {'query': int(index), 'box_iou': float(box_ious[index]),
                'mask_iou_by_branch': {name: float(values[index]) for name, values in ious.items()}}

    return {
        'alpha': float(alpha), 'logit_threshold': 0.0,
        'selected_queries': {name: at(index) for name, index in selected_queries.items()},
        'all_query_oracles': {name: at(int(values.argmax())) for name, values in ious.items()},
        'good_legal_box_queries': int(good_box.sum()),
        'good_legal_box_mask_oracles': {
            name: at(int(values.masked_fill(~good_box, -1).argmax())) if good_box.any() else None
            for name, values in ious.items()},
        'text_mask_iou': float(ious['text'][0]),
        'any_raw_query_mask_gt050': bool((ious['query'] > .5).any()),
        'any_fused_mask_gt050': bool((ious['fused'] > .5).any()),
        'text_gt050': bool(ious['text'][0] > .5),
        'oracle_uses_gt': True,
    }


def diagnose_root_candidates(end_points, evaluator):
    """Extend the native P1 observer with previously unrecorded branch evidence."""
    rows = diagnose_candidates(end_points, evaluator)
    native_point_masks, _ = evaluator._build_mask_point_predictions(end_points, 'last_')
    boxes = torch.cat([end_points['last_center'], end_points['last_pred_size'].clamp(min=1e-6)], -1)
    legal = build_detector_overlap_valid(
        boxes, torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device),
        end_points['all_detected_boxes'], end_points['all_detected_bbox_label_mask'].bool(),
        iou_threshold=.25)
    _, _, _, _, _, _, root_boxes = evaluator._parse_gt(end_points)
    for bid, row in enumerate(rows):
        text = as_query_mask_logits(end_points['last_pred_masks'][bid])
        query = as_query_mask_logits(end_points['sp_last_pred_masks'][bid])
        alpha = end_points['adaptive_weights'][bid]
        assert alpha.numel() == 1
        assert torch.equal(text, text[:1].expand_as(text))
        fused = fuse_query_mask_logits(text, query, alpha)
        superpoints = end_points['superpoints'][bid].long()
        target = end_points['gt_masks'][bid, 0].bool()
        assert target.any()
        assert torch.equal((fused > 0)[:, superpoints], native_point_masks[bid].bool())
        ious = {name: superpoint_mask_ious(logits, superpoints, target)
                for name, logits in [('text', text), ('query', query), ('fused', fused)]}
        assert all(torch.isfinite(values).all() for values in ious.values())
        assert float(ious['fused'].max()) == row['mask_oracle_all_queries']['mask_iou']
        root_ious = _iou3d_par(box_cxcyczwhd_to_xyzxyz(root_boxes[bid]),
                              box_cxcyczwhd_to_xyzxyz(boxes[bid]))[0][0]
        selected = {name: None if row[key] is None else row[key]['query']
                    for name, key in [('native_rec', 'rec_selection'), ('native_mask', 'mask_selection'),
                                      ('best_legal_box', 'box_oracle_after_filter')]}
        row['mask_branches'] = summarize_branches(ious, root_ious, legal[bid], alpha, selected)
    return rows
