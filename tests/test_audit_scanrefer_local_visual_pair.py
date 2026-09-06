from collections import OrderedDict
import copy

import pytest
import torch

from scripts.audit_scanrefer_local_visual_pair import (
    check_model_state, check_optimizer, check_readouts,
)


def test_frozen_running_buffer_drift_is_rejected_even_with_valid_trainable_updates():
    initial = {'head.weight': torch.zeros(2), 'bn.running_mean': torch.zeros(2)}
    updated = {name: value.clone() for name, value in initial.items()}
    updated['head.weight'][0] = .1
    assert set(check_model_state(updated, initial, {'head.weight'}, set())) == {'head.weight'}
    updated['bn.running_mean'][1] = .01
    with pytest.raises(AssertionError):
        check_model_state(updated, initial, {'head.weight'}, set())


def test_unused_parameter_cannot_change_despite_being_in_the_optimizer_group():
    initial = {'norm1.weight': torch.ones(2)}
    with pytest.raises(AssertionError):
        check_model_state({'norm1.weight': torch.zeros(2)}, initial, set(initial), set(initial))


@pytest.mark.parametrize('changed_field', ['weight', 'normalization'])
def test_frozen_readout_weight_and_nested_normalization_changes_are_rejected(changed_field):
    original = {'parent': {'model_state_dict': OrderedDict(weight=torch.ones(2)),
                            'normalization': {'mean': torch.zeros(2)}}}
    actual = copy.deepcopy(original)
    actual['parent']['model_state_dict'] = dict(actual['parent']['model_state_dict'])
    check_readouts(actual, original)
    if changed_field == 'weight':
        actual['parent']['model_state_dict']['weight'][0] += .1
    else:
        actual['parent']['normalization']['mean'][0] += .1
    with pytest.raises(AssertionError):
        check_readouts(actual, original)


def optimizer_fixture():
    state = {'core': torch.zeros(2), 'unused': torch.zeros(1), 'local': torch.zeros(3)}
    common = {'weight_decay': .0005, 'betas': (.9, .999), 'eps': 1e-8}
    optimizer = {'param_groups': [dict(common, lr=1e-6, params=[0, 1]),
                                   dict(common, lr=1e-4, params=[2])],
        'state': {index: {'step': 2482, 'exp_avg': torch.zeros_like(state[name]),
                         'exp_avg_sq': torch.zeros_like(state[name])}
                  for index, name in [(0, 'core'), (2, 'local')]}}
    return optimizer, state


def test_local_optimizer_short_update_count_is_rejected():
    optimizer, state = optimizer_fixture()
    args = (state, ['core', 'unused'], ['local'], {'unused'}, 2482)
    assert check_optimizer(optimizer, *args) == 2
    optimizer['state'][2]['step'] = 2481
    with pytest.raises(AssertionError):
        check_optimizer(optimizer, *args)


def test_optimizer_moments_must_belong_to_the_corresponding_parameter():
    optimizer, state = optimizer_fixture()
    optimizer['state'][2]['exp_avg'] = torch.zeros_like(state['core'])
    with pytest.raises(AssertionError):
        check_optimizer(optimizer, state, ['core', 'unused'], ['local'], {'unused'}, 2482)
