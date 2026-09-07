import torch

from scripts.native_mask_geometry_supervision import native_mask_geometry_loss


def test_current_root_matching_controls_mask_gradient_and_gt_is_not_learned():
    coords = torch.stack([torch.arange(100.)] * 3, dim=-1)
    mask = torch.full((2, 100), -5.)
    mask[:, 45:55] = 5.
    mask.requires_grad_(True)
    text = torch.zeros(1, 2, 100, requires_grad=True)
    outputs = {'last_pred_masks': [text], 'sp_last_pred_masks': [mask],
               'adaptive_weights': [torch.tensor(0.)]}
    inputs = {'point_clouds': coords[None], 'superpoint': torch.arange(100)[None]}
    roots = torch.tensor([[49.5, 49.5, 49.5, 9., 9., 9.]], requires_grad=True)
    # Target 0 is assigned to Query 1, not the first Query or a cached index.
    matches = [(torch.tensor([0, 1]), torch.tensor([1, 0]))]
    loss, stats = native_mask_geometry_loss(outputs, inputs, roots, matches)
    gradient, target_gradient = torch.autograd.grad(loss, (mask, roots), allow_unused=True)
    assert torch.isfinite(loss) and loss > 0.
    assert stats['query_indices'].tolist() == [1]
    assert torch.count_nonzero(gradient[0]) == 0
    assert torch.count_nonzero(gradient[1]) > 0
    assert torch.isfinite(gradient).all() and target_gradient is None


def test_query_permutation_preserves_the_geometry_objective():
    coords = torch.stack([torch.arange(100.)] * 3, dim=-1)
    first = torch.linspace(-6., 3., 100)
    second = torch.linspace(3., -6., 100)
    masks = torch.stack([first, second])
    inputs = {'point_clouds': coords[None], 'superpoint': torch.arange(100)[None]}
    roots = torch.tensor([[49.5, 49.5, 49.5, 20., 20., 20.]])
    def evaluate(values, root_index):
        outputs = {'last_pred_masks': [torch.zeros(1, 2, 100)], 'sp_last_pred_masks': [values],
                   'adaptive_weights': [torch.tensor(0.)]}
        return native_mask_geometry_loss(outputs, inputs, roots,
            [(torch.tensor([root_index]), torch.tensor([0]))])[0]
    assert torch.equal(evaluate(masks, 1), evaluate(masks.flip(0), 0))
