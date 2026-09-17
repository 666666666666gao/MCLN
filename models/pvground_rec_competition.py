"""Training-only margin on native bbs scores, anchored to the native root match."""
import torch


def bbs_scores(logits, batch):
    probabilities = logits.softmax(-1)
    score = (probabilities * (batch['positive_map'][:, 0] > 0).unsqueeze(1)).sum(-1)
    for key in ['modify_positive_map', 'pron_positive_map', 'rel_positive_map']:
        score = score + (probabilities * batch[key][:, 0].unsqueeze(1)).sum(-1)
    return score - (probabilities * batch['other_entity_map'][:, 0].unsqueeze(1)).sum(-1)


def root_iou(boxes, root):
    # Selection/quality targets are labels, not a route for moving boxes to
    # escape the comparison. Native regression still supplies box gradients.
    boxes = boxes.detach()
    root = root.detach()
    size = boxes[..., 3:].clamp(min=1e-6)
    root_size = root[:, None, 3:].clamp(min=1e-6)
    lo = torch.maximum(boxes[..., :3] - size / 2, root[:, None, :3] - root_size / 2)
    hi = torch.minimum(boxes[..., :3] + size / 2, root[:, None, :3] + root_size / 2)
    intersection = (hi - lo).clamp(min=0).prod(-1)
    return intersection / (size.prod(-1) + root_size.prod(-1) - intersection).clamp(min=1e-6)


def competition_loss(predictions, batch, last_indices):
    scores = bbs_scores(predictions['last_sem_cls_scores'], batch)
    assert bool(batch['box_label_mask'][:, 0].all())
    root_queries = []
    for queries, targets in last_indices:
        matched = queries[targets == 0]
        assert matched.numel() == 1
        root_queries.append(int(matched.item()))
    ids = torch.arange(scores.shape[0], device=scores.device)
    positives = torch.tensor(root_queries, device=scores.device, dtype=torch.long)
    boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
    root = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
    iou = root_iou(boxes, root)
    assert bool(torch.isfinite(iou).all())
    positive_iou = iou[ids, positives]
    positive_score = scores[ids, positives]
    terms = []
    counts = {}
    for threshold, label in [(.25, '25'), (.5, '50')]:
        negative_mask = iou <= threshold
        eligible = (positive_iou > threshold) & negative_mask.any(-1)
        negative = scores.detach().masked_fill(~negative_mask, -torch.inf).argmax(-1)
        negative_score = scores[ids, negative]
        gap = positive_iou - iou[ids, negative]
        margin_violation = gap + negative_score - positive_score
        term = torch.where(eligible, margin_violation.relu(), torch.zeros_like(margin_violation))
        terms.append(term)
        counts['eligible' + label] = int(eligible.sum())
        counts['active' + label] = int((eligible & (margin_violation > 0)).sum())
    # Fixed 2*batch normalization; missing eligible pairs do not upweight others.
    return torch.stack(terms).mean(), counts
