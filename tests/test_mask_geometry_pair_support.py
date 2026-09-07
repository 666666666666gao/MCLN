import io

import torch
from scripts.mask_geometry_pair_support import optimizer_presence_counts, hard_root_geometry


def test_saved_optimizer_matches_intermittent_and_absent_gradients():
    parameters = [torch.nn.Parameter(torch.tensor([1.])) for _ in range(3)]
    names = ['always', 'intermittent', 'unused']
    optimizer = torch.optim.AdamW(parameters, lr=1e-6)
    masks = []
    for index in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = parameters[0].square().sum()
        if index == 1:
            loss = loss + parameters[1].square().sum()
        loss.backward()
        masks.append(''.join('1' if value.grad is not None else '0' for value in parameters))
        optimizer.step()
    counts = optimizer_presence_counts(masks, names)
    assert counts == {'always': 3, 'intermittent': 1, 'unused': 0}
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream)
    ids = saved['param_groups'][0]['params']
    assert set(saved['state']) == set(ids[:2])
    assert [float(saved['state'][i]['step']) for i in ids[:2]] == [3, 1]
    assert parameters[2].item() == 1.


def test_hard_root_geometry_uses_current_query_and_reports_empty_support():
    coords = torch.stack([torch.arange(100.)] * 3, -1)
    masks = torch.full((2, 100), -8.)
    masks[1, 40:60] = 8.
    outputs = {'last_pred_masks': [torch.zeros(1, 2, 100)],
               'sp_last_pred_masks': [masks], 'adaptive_weights': [torch.tensor(0.)]}
    inputs = {'point_clouds': coords[None], 'superpoint': torch.arange(100)[None]}
    box, valid = hard_root_geometry(outputs, inputs, torch.tensor([1]))
    assert valid.tolist() == [True]
    torch.testing.assert_allclose(box, torch.tensor([[49.5, 49.5, 49.5, 18.81, 18.81, 18.81]]))
    empty, valid = hard_root_geometry(outputs, inputs, torch.tensor([0]))
    assert valid.tolist() == [False] and torch.count_nonzero(empty) == 0
