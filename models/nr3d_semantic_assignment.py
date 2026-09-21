"""Nr3D training-only replacement of qualified unmatched token targets.

This returns an unscaled CE difference. The native criterion keeps responsibility
for its decoder-layer and contrastive-loss coefficients. Geometry and matching
are not changed. IoU qualification is a geometric proxy, not an identity label.
"""
import torch


def semantic_assignment_delta(outputs, targets, indices, num_boxes,
                              sample_datasets, eos_coef):
    """Replace eos by the native root token target only in Nr3D samples."""
    logits = outputs['pred_logits']
    boxes = outputs['pred_boxes'].detach()
    assert len(targets) == len(indices) == len(sample_datasets) == logits.shape[0]
    assert all(name in ('nr3d', 'scannet') for name in sample_datasets)
    assert all(name == 'nr3d' for name in outputs['language_dataset'])
    assert float(num_boxes) > 0 and eos_coef == .1
    selected = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.bool)
    labels = torch.zeros_like(logits[:, 0])
    for bid, dataset in enumerate(sample_datasets):
        if dataset != 'nr3d':
            continue
        target = targets[bid]
        query_ids, target_ids = indices[bid]
        assert int((target_ids == 0).sum()) == 1
        root = target['boxes'][0].detach()
        size = boxes[bid, :, 3:].clamp(min=1e-6)
        root_size = root[3:].clamp(min=1e-6)
        lo = torch.maximum(boxes[bid, :, :3] - size / 2, root[:3] - root_size / 2)
        hi = torch.minimum(boxes[bid, :, :3] + size / 2, root[:3] + root_size / 2)
        intersection = (hi - lo).clamp(min=0).prod(-1)
        iou = intersection / (size.prod(-1) + root_size.prod() - intersection).clamp(min=1e-6)
        assert bool(torch.isfinite(iou).all())
        selected[bid] = iou > .5
        selected[bid, query_ids] = False
        labels[bid] = (target['positive_map'][0] * .6
                       + target['modify_positive_map'][0] * .2
                       + target['pron_positive_map'][0] * .2
                       + target['rel_positive_map'][0] * .1).detach()
        assert labels[bid, -1] == 0 and labels[bid].sum() > 0
    logp = logits.log_softmax(-1)
    new = (labels * torch.log(labels + 1e-6)).sum(-1)[:, None]
    new = new - (logp * labels[:, None]).sum(-1)
    old = torch.log(torch.ones_like(logp[..., -1]) + 1e-6) - logp[..., -1]
    delta = (new - old)[selected].sum() * eos_coef / num_boxes
    return delta, selected


class LastLayerSemanticAssignment:
    """Attach to one criterion; bind each batch before the native loss call."""

    def __init__(self, criterion):
        self.criterion = criterion
        self.original = criterion.loss_pos_align
        criterion.loss_pos_align = self.loss_pos_align
        self.last_logits = None
        self.sample_datasets = None
        self.records = []

    def bind(self, predictions, sample_datasets):
        self.last_logits = predictions['last_sem_cls_scores']
        self.sample_datasets = sample_datasets
        self.records = []

    def loss_pos_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        losses = self.original(outputs, targets, indices, num_boxes, auxi_indices)
        if outputs['pred_logits'] is self.last_logits:
            delta, selected = semantic_assignment_delta(
                outputs, targets, indices, num_boxes, self.sample_datasets,
                self.criterion.eos_coef)
            self.records.append({'selected': selected.detach(), 'delta': delta,
                                 'native_ce': losses['loss_ce']})
            losses['loss_ce'] = losses['loss_ce'] + delta
        return losses

    def remove(self):
        self.criterion.loss_pos_align = self.original
