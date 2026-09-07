"""Distinguish hard geometry support from differentiable Mask statistics."""
import torch

from models.rec_mask_geometry import mask_logits_to_point_aabbs


def fixture():
    coords = torch.tensor([[float(i), float(i % 3), float(i % 4)] for i in range(12)])
    ids = torch.arange(12)
    logits = torch.tensor([[2.] * 5 + [-2.] * 7], requires_grad=True)
    return coords, ids, logits


def test_hard_mask_bounds_have_no_logit_autograd_route():
    coords, ids, logits = fixture()
    boxes, valid, _ = mask_logits_to_point_aabbs(coords, ids, logits, 0.)
    assert valid.all() and boxes.shape == (1, 2, 6)
    assert logits.requires_grad and not boxes.requires_grad
    assert torch.equal(boxes[0, 0], torch.tensor([2., 1., 1.5, 4., 2., 3.]))


def test_foreground_statistics_still_have_nonzero_logit_gradients():
    coords, ids, logits = fixture()
    _, _, stats = mask_logits_to_point_aabbs(coords, ids, logits, 0.)
    gradient, = torch.autograd.grad(stats['foreground_logit_mean'].sum(), logits)
    assert torch.allclose(gradient[0, :5], torch.full((5,), .2))
    assert torch.count_nonzero(gradient[0, 5:]) == 0


def test_crossing_threshold_changes_box_despite_zero_autograd_route():
    coords, ids, logits = fixture()
    before, _, _ = mask_logits_to_point_aabbs(coords, ids, logits, 0.)
    unchanged, _, _ = mask_logits_to_point_aabbs(coords, ids, logits + .1, 0.)
    changed = logits.detach().clone()
    changed[0, 11] = 2.
    after, valid, _ = mask_logits_to_point_aabbs(coords, ids, changed, 0.)
    assert torch.equal(before, unchanged)
    assert valid.all() and not torch.equal(before, after)
    assert after[0, 0, 3] == 11.
