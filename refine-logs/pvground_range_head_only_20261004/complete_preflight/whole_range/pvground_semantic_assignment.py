"""Training-only last-layer label replacement for unmatched root-qualified boxes.

Hungarian assignments and all regression losses are unchanged. This replaces,
rather than adds to, the existing eos-weighted token loss of qualified slots.
"""
import torch


def qualified_unmatched(predictions, batch, indices):
    boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1).detach()
    root = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1).detach()
    size = boxes[..., 3:].clamp(min=1e-6)
    root_size = root[:, None, 3:].clamp(min=1e-6)
    lo = torch.maximum(boxes[..., :3] - size / 2, root[:, None, :3] - root_size / 2)
    hi = torch.minimum(boxes[..., :3] + size / 2, root[:, None, :3] + root_size / 2)
    intersection = (hi - lo).clamp(min=0).prod(-1)
    iou = intersection / (size.prod(-1) + root_size.prod(-1) - intersection).clamp(min=1e-6)
    assert bool(batch['box_label_mask'][:, 0].all()) and bool(torch.isfinite(iou).all())
    matched = torch.zeros_like(iou, dtype=torch.bool)
    for bid, (queries, targets) in enumerate(indices):
        assert int((targets == 0).sum()) == 1
        matched[bid, queries] = True
    return (iou > .5) & ~matched


def root_token_target(batch):
    # Exact ScanRefer native weighting, including its unnormalized target mass.
    return (batch['positive_map'][:, 0] * .6
            + batch['modify_positive_map'][:, 0] * .2
            + batch['pron_positive_map'][:, 0] * .2
            + batch['rel_positive_map'][:, 0] * .1).detach()


def semantic_assignment_correction(predictions, batch, indices, eos_coef):
    assert eos_coef == .1 and all(x == 'scanrefer' for x in batch['language_dataset'])
    logp = predictions['last_sem_cls_scores'].log_softmax(-1)
    selected = qualified_unmatched(predictions, batch, indices)
    target = root_token_target(batch)
    assert bool((target[:, -1] == 0).all()) and bool((target.sum(-1) > 0).all())
    num_boxes = sum(len(queries) for queries, _ in indices)
    entropy = (torch.log(target + 1e-6) * target).sum(-1)
    replacement = entropy[:, None] - (logp * target[:, None]).sum(-1)
    # Native eos target has mass 1; retain its small log(1+1e-6) constant too.
    original = torch.log(torch.ones_like(logp[..., -1]) + 1e-6) - logp[..., -1]
    old_ce = original[selected].sum() * eos_coef / num_boxes
    new_ce = replacement[selected].sum() * eos_coef / num_boxes
    correction = (new_ce - old_ce) * (.5 / 7)
    return correction, dict(reassigned_queries=int(selected.sum()),
        reassigned_samples=int(selected.any(-1).sum()), matched_queries=num_boxes,
        replaced_ce_old=float(old_ce), replaced_ce_new=float(new_ce))


def verify_native_replacement(predictions, batch, indices, eos_coef, correction):
    """Zero-update real-batch witness against the actual native criterion tensor."""
    logits = predictions['last_sem_cls_scores']
    logp = logits.log_softmax(-1)
    target = torch.zeros_like(logp)
    target[..., -1] = 1
    weights = torch.full_like(logp[..., -1], eos_coef)
    for bid, (queries, targets) in enumerate(indices):
        valid = batch['box_label_mask'][bid].bool()
        maps = [batch[key][bid][valid][targets] for key in
                ['positive_map', 'modify_positive_map', 'pron_positive_map', 'rel_positive_map']]
        target[bid, queries] = maps[0] * .6 + maps[1] * .2 + maps[2] * .2 + maps[3] * .1
        weights[bid, queries] = 1
    denominator = sum(len(q) for q, _ in indices)

    def ce(labels):
        return (((labels * torch.log(labels + 1e-6) - labels * logp).sum(-1)) * weights).sum() / denominator

    old = ce(target)
    actual = predictions['last__loss_ce']
    assert torch.allclose(old, actual, rtol=1e-6, atol=1e-6)
    selected = qualified_unmatched(predictions, batch, indices)
    changed = target.clone()
    changed[selected] = root_token_target(batch)[:, None].expand_as(changed)[selected]
    expected = ce(changed) * (.5 / 7)
    corrected = actual * (.5 / 7) + correction
    assert torch.allclose(expected, corrected, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad(corrected, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    difference = torch.autograd.grad(correction, logits, retain_graph=True)[0]
    assert torch.allclose(grad, expected_grad, rtol=1e-5, atol=1e-7)
    assert bool((difference[~selected] == 0).all())
    assert bool(selected.any()) and bool((grad[..., -1][selected] > 0).all())
    geometry = torch.autograd.grad(correction,
        [predictions['last_center'], predictions['last_pred_size']], retain_graph=True, allow_unused=True)
    assert all(g is None for g in geometry)
    return dict(native_ce_reconstruction_max_error=float((old - actual).abs()),
        corrected_ce_error=float((expected - corrected).abs()),
        corrected_gradient_max_error=float((grad - expected_grad).abs().max()),
        untouched_query_gradient_unchanged=True, geometry_target_detached=True,
        qualified_null_gradient_positive=True)
