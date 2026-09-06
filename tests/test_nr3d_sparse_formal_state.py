"""Synthetic CPU endpoint loading and restoration; no native model forward."""

import hashlib

import pytest
import torch
from torch import nn

from scripts.nr3d_sparse_formal_state import load_mask_state, MaskProjectionSwitch


def artifact(tmp_path, arm='sparse'):
    shared = {'x_mask.weight': torch.zeros(2, 3), 'x_mask.bias': torch.zeros(2)}
    addon = {'output.weight': torch.zeros(2, 4)}
    values = {name: value + 1 for name, value in shared.items()}
    if arm == 'sparse':
        values.update({'sparse_point.' + name: value + 2 for name, value in addon.items()})
    state = {'arm': arm, 'steps': 6687, 'parent_checkpoint_sha256': 'protected-parent',
             'mask_projection_state': values}
    path = tmp_path / (arm + '.pt')
    return path, state, shared, addon


def save(path, state):
    torch.save(state, path)
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def test_loads_separate_native_and_sparse_endpoints(tmp_path):
    for arm in ['native', 'sparse']:
        path, state, shared, addon = artifact(tmp_path, arm)
        loaded, new = load_mask_state(save(path, state), 'protected-parent', arm, shared, addon)
        assert set(loaded) == set(shared)
        assert all(torch.equal(value, shared[name] + 1) for name, value in loaded.items())
        if arm == 'sparse':
            assert torch.equal(new['output.weight'], addon['output.weight'] + 2)
        else:
            assert new == {}


@pytest.mark.parametrize('changed', ['parent', 'steps', 'arm', 'shape', 'nonfinite', 'extra_key'])
def test_rejects_wrong_or_incomplete_endpoint_even_with_matching_file_hash(tmp_path, changed):
    path, state, shared, addon = artifact(tmp_path)
    if changed == 'parent':
        state['parent_checkpoint_sha256'] = 'other-parent'
    elif changed == 'steps':
        state['steps'] = 1024
    elif changed == 'arm':
        state['arm'] = 'native'
    elif changed == 'shape':
        state['mask_projection_state']['x_mask.weight'] = torch.ones(2, 4)
    elif changed == 'nonfinite':
        state['mask_projection_state']['sparse_point.output.weight'][0, 0] = float('nan')
    else:
        state['mask_projection_state']['box_head.weight'] = torch.ones(1)
    with pytest.raises(AssertionError):
        load_mask_state(save(path, state), 'protected-parent', 'sparse', shared, addon)


def test_modified_artifact_cannot_reuse_old_hash(tmp_path):
    path, state, shared, addon = artifact(tmp_path)
    metadata = save(path, state)
    state['mask_projection_state']['x_mask.weight'].add_(1)
    save(path, state)
    with pytest.raises(AssertionError):
        load_mask_state(metadata, 'protected-parent', 'sparse', shared, addon)


def test_projection_switch_restores_protected_model_between_batches():
    model = nn.Module()
    model.x_mask = nn.Linear(3, 2)
    model.box_head = nn.Linear(3, 6)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    mask = {name: value for name, value in original.items() if name.startswith('x_mask.')}
    endpoints = {arm: {name: value + delta for name, value in mask.items()}
                 for arm, delta in [('native', 1), ('sparse', 2)]}
    switch = MaskProjectionSwitch(model, endpoints)
    cloud = torch.tensor([[1., 2., 3.]])
    protected = model.x_mask(cloud).detach().clone()
    for _ in range(2):
        for arm in ['native', 'sparse']:
            switch.apply(arm)
            assert not torch.equal(model.x_mask(cloud), protected)
            with pytest.raises(AssertionError):
                switch.require_protected()
            for name, value in model.box_head.state_dict().items():
                assert torch.equal(value, original['box_head.' + name])
        switch.apply('protected')
        switch.require_protected()
        assert torch.equal(model.x_mask(cloud), protected)
        assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
