"""Training-only final-box quality differences on the existing root pool.

No score is added at inference. Native bbs is signed token evidence, so the
loss fits within-pool differences without treating it as a probability.
"""
import torch

from native_root_bbs import native_root_bbs
from pvground_semantic_assignment import qualified_unmatched


def final_quality_pool(predictions, batch, indices):
    assert all(value == 'scanrefer' for value in batch['language_dataset'])
    pool = qualified_unmatched(predictions, batch, indices)
    for bid, (queries, targets) in enumerate(indices):
        root = queries[targets == 0]
        assert len(root) == 1
        pool[bid, root] = True
    boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1).detach()
    truth = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1).detach()
    assert boxes.shape[1:] == (256, 6)
    assert bool((boxes[..., 3:] > 0).all()) and bool((truth[..., 3:] > 0).all())
    low = torch.maximum(boxes[..., :3] - boxes[..., 3:] / 2,
                        truth[:, None, :3] - truth[:, None, 3:] / 2)
    high = torch.minimum(boxes[..., :3] + boxes[..., 3:] / 2,
                         truth[:, None, :3] + truth[:, None, 3:] / 2)
    intersection = (high - low).clamp(min=0).prod(-1)
    quality = intersection / (boxes[..., 3:].prod(-1) + truth[:, None, 3:].prod(-1) - intersection)
    assert bool(torch.isfinite(quality).all()) and bool(pool.any(-1).all())
    return pool, quality


def centered_quality_error(score, quality, pool):
    """Per-row mean squared error of score differences vs final IoU differences."""
    weight = pool.to(score.dtype) / pool.sum(-1, keepdim=True)
    error = score - quality
    centered = error - (weight * error).sum(-1, keepdim=True)
    return (weight * centered.square()).sum(-1).mean()


def native_final_quality_loss(predictions, batch, indices):
    pool, quality = final_quality_pool(predictions, batch, indices)
    score = native_root_bbs(predictions['last_sem_cls_scores'], batch)
    loss = centered_quality_error(score, quality, pool)
    assert bool(torch.isfinite(loss))
    weight = pool.to(score.dtype) / pool.sum(-1, keepdim=True)
    quality_mean = (weight * quality).sum(-1, keepdim=True)
    score_mean = (weight * score).sum(-1, keepdim=True)
    counts = dict(pool_queries=int(pool.sum()), pool_per_sample=pool.sum(-1).tolist(),
        final_iou_variance=float((weight * (quality - quality_mean).square()).sum(-1).mean()),
        native_score_variance=float((weight * (score - score_mean).square()).sum(-1).mean()),
        final_iou_detached=not quality.requires_grad, other_matched_queries_excluded=True)
    return loss, counts


def verify_quality_loss(predictions, batch, indices, readback, previous_updates):
    """Real-batch independent pairwise, gradient-scope and optimizer-route witness."""
    pool, quality = final_quality_pool(predictions, batch, indices)
    logits = predictions['last_sem_cls_scores']
    score = native_root_bbs(logits, batch)
    loss = centered_quality_error(score, quality, pool)
    pairs = []
    for bid in range(len(score)):
        selected_score = score[bid, pool[bid]]
        selected_quality = quality[bid, pool[bid]]
        differences = (selected_score[:, None] - selected_score[None, :]
                       - selected_quality[:, None] + selected_quality[None, :])
        pairs.append(differences.square().mean() / 2)
    independent = torch.stack(pairs).mean()
    assert torch.allclose(loss, independent, rtol=1e-5, atol=1e-7)
    direct, = torch.autograd.grad(loss, logits, retain_graph=True)
    reference, = torch.autograd.grad(independent, logits, retain_graph=True)
    assert torch.allclose(direct, reference, rtol=1e-5, atol=1e-7)
    assert bool((direct[~pool] == 0).all())
    assert bool(torch.isfinite(direct).all()) and float(direct.norm()) > 0
    for bid, (queries, targets) in enumerate(indices):
        assert not bool(pool[bid, queries[targets != 0]].any())

    # With equal scores, the gradient must promote high-IoU members and reduce
    # low-IoU members. This tests the desired direction on this real GT pool.
    equal = torch.zeros_like(score, requires_grad=True)
    direction, = torch.autograd.grad(centered_quality_error(equal, quality, pool), equal)
    weight = pool.to(score.dtype) / pool.sum(-1, keepdim=True)
    centered_quality = quality - (weight * quality).sum(-1, keepdim=True)
    expected = -2 * weight * centered_quality / len(score)
    assert torch.allclose(direction, expected, rtol=1e-5, atol=1e-7)
    assert float(direction.norm()) > 0

    named = tuple(readback.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(value for _, value in named), retain_graph=True)
    norms = {name: float(value.norm()) for (name, _), value in zip(named, gradients)}
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    assert norms['output.weight'] > 0
    if previous_updates == 1:
        for prefix in ('query.', 'text.', 'evidence.', 'language_read.', 'face_read.'):
            assert sum(value for name, value in norms.items() if name.startswith(prefix)) > 0, prefix
    assert not predictions['last_center'].requires_grad
    assert not predictions['last_pred_size'].requires_grad
    return dict(previous_updates=previous_updates, isolated_quality_loss=float(loss),
        independent_pairwise_error=float((loss - independent).abs()),
        independent_logit_gradient_error=float((direct - reference).abs().max()),
        nonpool_direct_gradient_zero=True, other_matched_direct_gradient_zero=True,
        equal_score_quality_direction_verified=True, final_geometry_target_detached=True,
        parameter_gradient_norms=norms)
