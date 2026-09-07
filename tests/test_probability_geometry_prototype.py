import torch

from scripts.prototype_probability_geometry import probability_quantile_boxes


def inputs():
    coords = torch.tensor([[0., 0., 0.], [1., 2., 3.], [2., 4., 6.], [4., 8., 12.]], dtype=torch.double)
    logits = torch.tensor([[.2, -.3, .8, -.1]], dtype=torch.double, requires_grad=True)
    return coords, torch.arange(4), logits


def test_inverse_cdf_uniform_weights():
    coords, ids, _ = inputs()
    boxes, _ = probability_quantile_boxes(coords, ids, torch.zeros(1, 4, dtype=torch.double), .25)
    assert torch.allclose(boxes, torch.tensor([[1.75, 3.5, 5.25, 2.5, 5., 7.5]], dtype=torch.double))


def test_weight_gradient_matches_finite_differences():
    coords, ids, logits = inputs()
    assert torch.autograd.gradcheck(lambda values: probability_quantile_boxes(coords, ids, values, .25)[0], (logits,))


def test_permutation_and_affine_geometry():
    coords, ids, logits = inputs()
    original, _ = probability_quantile_boxes(coords, ids, logits, .25)
    order = torch.tensor([3, 0, 2, 1])
    permuted, _ = probability_quantile_boxes(coords[order], ids[order], logits, .25)
    assert torch.allclose(original, permuted)
    transformed, _ = probability_quantile_boxes(coords * 3. + 7., ids, logits, .25)
    expected = original * 3.
    expected[:, :3] += 7.
    assert torch.allclose(transformed, expected)


def test_background_probability_can_expand_the_support():
    coords = torch.stack([torch.arange(100., dtype=torch.double)] * 3, dim=-1)
    logits = torch.full((1, 100), -5., dtype=torch.double)
    logits[0, 45:55] = 5.
    boxes, _ = probability_quantile_boxes(coords, torch.arange(100), logits)
    assert boxes[0, 3] > 70.  # Hard-threshold foreground spans only 9 units.
