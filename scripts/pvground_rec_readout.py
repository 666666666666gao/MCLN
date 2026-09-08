"""Expose the pinned PV-Ground root REC decisions without adding a new score.

Derived from AaNnWwTt/PV-Ground 262e259, src/grounding_evaluator.py.
Keep this separate from the protected MCLN selector and its candidate masking.
"""
import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz


def native_root_rec_readout(end_points, filter_non_gt_boxes):
    """Return bbs/bbf scores, all Query ranks and the author's overlap mask.

    Only the six-layer model's final, 256-token root readout is supported.
    GT boxes and masks do not enter this function. Maps are the existing text
    component maps used by the author's Evaluator.
    """
    raw_boxes = torch.cat([end_points['last_center'], end_points['last_pred_size']], -1)
    boxes = torch.cat([end_points['last_center'], end_points['last_pred_size'].clamp(min=1e-6)], -1)
    semantic = end_points['last_sem_cls_scores'].softmax(-1)
    projected = (torch.matmul(end_points['last_proj_queries'], end_points['proj_tokens'].transpose(-1, -2)) / .07).softmax(-1)
    assert semantic.shape[-1] == end_points['positive_map'].shape[-1] == 256
    contrastive = torch.zeros_like(semantic)
    contrastive[:, :, :projected.shape[-1]] = projected
    main_map = end_points['positive_map'][:, 0].clone()
    main_map[main_map > 0] = 1
    sources = []
    for probability in [semantic, contrastive]:
        score = (probability * main_map[:, None]).sum(-1)
        for name in ['modify_positive_map', 'pron_positive_map', 'rel_positive_map']:
            score = score + (probability * end_points[name][:, :1]).sum(-1)
        score = score - (probability * end_points['other_entity_map'][:, :1]).sum(-1)
        sources.append(score)
    raw_scores = torch.stack(sources, dim=1)
    overlap_valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
    if filter_non_gt_boxes:
        for bid in range(len(boxes)):
            objects = end_points['all_detected_boxes'][bid][end_points['all_detected_bbox_label_mask'][bid]]
            overlap = _iou3d_par(box_cxcyczwhd_to_xyzxyz(objects), box_cxcyczwhd_to_xyzxyz(boxes[bid]))[0]
            overlap_valid[bid] = overlap.max(0)[0] > .25
    # The upstream code zeros a score; it does not remove that Query. Preserve
    # both negative-score competition and ties instead of introducing -inf.
    scores = raw_scores * overlap_valid[:, None].to(raw_scores.dtype)
    return {'raw_boxes': raw_boxes, 'boxes': boxes, 'raw_scores': raw_scores,
            'overlap_valid': overlap_valid, 'scores': scores,
            'ranks': scores.argsort(dim=-1, descending=True), 'modes': ['bbs', 'bbf']}
