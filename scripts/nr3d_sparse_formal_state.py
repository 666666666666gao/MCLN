"""Load the two fixed Mask endpoints and restore the protected projections."""

import hashlib
from pathlib import Path

import torch


def load_mask_state(metadata, parent_sha256, arm, shared_reference, addon_reference):
    path = Path(metadata['path'])
    raw = path.read_bytes()
    assert len(raw) == metadata['bytes']
    assert hashlib.sha256(raw).hexdigest() == metadata['sha256']
    state = torch.load(str(path), map_location='cpu')
    assert state['arm'] == arm and state['steps'] == 6687
    assert state['parent_checkpoint_sha256'] == parent_sha256
    parameters = state['mask_projection_state']
    expected = dict(shared_reference)
    if arm == 'sparse':
        expected.update({'sparse_point.' + name: value
                         for name, value in addon_reference.items()})
    assert set(parameters) == set(expected)
    for name, reference in expected.items():
        value = parameters[name]
        assert value.shape == reference.shape and value.dtype == reference.dtype, name
        assert torch.isfinite(value).all(), name
    shared = {name: parameters[name] for name in shared_reference}
    addon = {name: parameters['sparse_point.' + name]
             for name in addon_reference} if arm == 'sparse' else {}
    return shared, addon


class MaskProjectionSwitch:
    """Use one native model; only the registered Mask projections can change."""

    def __init__(self, model, endpoints):
        parameters = dict(model.named_parameters())
        names = list(endpoints['native'])
        assert set(names) == set(endpoints['sparse'])
        self.parameters = {name: parameters[name] for name in names}
        self.states = {'protected': {name: parameter.detach().clone()
                                    for name, parameter in self.parameters.items()}}
        for arm in ['native', 'sparse']:
            self.states[arm] = {}
            for name, parameter in self.parameters.items():
                value = endpoints[arm][name]
                assert value.shape == parameter.shape and value.dtype == parameter.dtype
                self.states[arm][name] = value.to(parameter.device)

    @torch.no_grad()
    def apply(self, arm):
        for name, parameter in self.parameters.items():
            parameter.copy_(self.states[arm][name])

    def require_protected(self):
        for name, parameter in self.parameters.items():
            assert torch.equal(parameter, self.states['protected'][name]), name
