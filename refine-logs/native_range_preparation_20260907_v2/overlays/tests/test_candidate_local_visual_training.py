import copy
import sys

import pytest
import torch

from main_utils import load_checkpoint, parse_option
from models.candidate_local_visual import CandidateLocalVisual
from models.candidate_local_visual_training import (
    local_visual_state_keys, validate_local_visual_checkpoint,
)
from models.mcln_training_groups import build_mcln_optimizer_param_groups


class LocalToy(torch.nn.Module):
    def __init__(self, local=True):
        super().__init__()
        self.module = torch.nn.Module()
        self.module.decoder = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(6)])
        self.module.backbone_net = torch.nn.Linear(2, 2)
        self.module.x_mask = torch.nn.Linear(2, 2)
        self.module.source_choice_selector = torch.nn.Linear(2, 2)
        if local:
            self.module.decoder[-1].local_visual = CandidateLocalVisual(
                d_model=2, point_dim=2, hidden_dim=4, heads=1, points_per_query=3)


def parameter_groups(model, local_lr=1e-4):
    return build_mcln_optimizer_param_groups(
        model, decoder_lr=1e-6, backbone_lr=2e-6, selector_lr=4e-6,
        mask_head_lr_multiplier=3, candidate_local_visual_lr=local_lr)


def test_native_cli_keeps_local_reading_disabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train_dist_mod.py'])
    args = parse_option()
    assert not args.use_candidate_local_visual
    assert args.candidate_local_visual_lr == 1e-4
    assert args.candidate_local_visual_variant == 'local'


def test_old_full_core_can_initialize_only_the_new_ten_tensors():
    current = LocalToy().state_dict()
    new = local_visual_state_keys(current)
    assert len(new) == 10
    saved = {name: value.clone() for name, value in current.items() if name not in new}
    validate_local_visual_checkpoint(current, saved, True)
    with pytest.raises(ValueError, match='key mismatch'):
        validate_local_visual_checkpoint(current, saved, False)
    del saved['module.decoder.0.weight']
    with pytest.raises(ValueError, match='key mismatch'):
        validate_local_visual_checkpoint(current, saved, True)


@pytest.mark.parametrize('change', ['shape', 'dtype'])
def test_incompatible_core_cannot_be_partially_loaded(change):
    current = LocalToy().state_dict()
    saved = {name: value.clone() for name, value in current.items()
             if name not in local_visual_state_keys(current)}
    name = 'module.decoder.0.weight'
    saved[name] = saved[name][:1] if change == 'shape' else saved[name].double()
    with pytest.raises(ValueError, match='shape/dtype mismatch'):
        validate_local_visual_checkpoint(current, saved, True)


def test_partial_trained_reader_is_rejected_even_for_weights_only_loading():
    current = LocalToy().state_dict()
    saved = dict(current)
    del saved[next(iter(local_visual_state_keys(saved)))]
    with pytest.raises(ValueError, match='key mismatch'):
        validate_local_visual_checkpoint(current, saved, True)


def test_new_learning_rate_group_preserves_exact_parameter_coverage():
    model = LocalToy()
    groups = parameter_groups(model)
    assert [group['name'] for group in groups] == [
        'decoder', 'backbone', 'mask_head', 'selector', 'candidate_local_visual']
    assert [group['lr'] for group in groups] == [1e-6, 2e-6, 3e-6, 4e-6, 1e-4]
    assert len(groups[-1]['params']) == len(groups[-1]['parameter_names']) == 10
    actual = [id(parameter) for group in groups for parameter in group['params']]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(parameter) for parameter in model.parameters()}
    assert all(name.startswith('decoder.5.local_visual.') for name in groups[-1]['parameter_names'])
    old_groups = parameter_groups(LocalToy(local=False), None)
    assert [group['name'] for group in old_groups] == [group['name'] for group in groups[:-1]]
    assert [group['lr'] for group in old_groups] == [group['lr'] for group in groups[:-1]]


def native_args(monkeypatch, path, enabled=True):
    argv = ['train_dist_mod.py', '--model', 'MCLN', '--checkpoint_path', str(path),
            '--use_source_choice_selector']
    if enabled:
        argv += ['--use_candidate_local_visual']
    monkeypatch.setattr(sys, 'argv', argv)
    return parse_option()


def test_native_loader_rejects_shape_changes_before_legacy_partial_load(monkeypatch, tmp_path):
    model = LocalToy()
    before = copy.deepcopy(model.state_dict())
    saved = copy.deepcopy(before)
    saved['module.decoder.0.weight'] = torch.zeros(1, 2)
    path = tmp_path / 'incompatible.pt'
    torch.save({'epoch': 7, 'model': saved}, str(path))
    args = native_args(monkeypatch, path)
    args.model_only_initialization = True
    args.checkpoint_start_epoch = 1
    with pytest.raises(ValueError, match='shape/dtype mismatch'):
        load_checkpoint(args, model, None, None)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_native_loader_does_not_silently_drop_trained_reader(monkeypatch, tmp_path):
    path = tmp_path / 'trained_local.pt'
    torch.save({'epoch': 7, 'model': LocalToy().state_dict()}, str(path))
    args = native_args(monkeypatch, path, enabled=False)
    with pytest.raises(ValueError, match='requires --use_candidate_local_visual'):
        load_checkpoint(args, LocalToy(local=False), None, None)


@pytest.mark.parametrize('saved_variant,current_variant', [
    ('local', 'extent'), ('center', 'extent'), ('extent', 'local')])
def test_same_shape_reader_cannot_resume_under_another_computation(
        monkeypatch, tmp_path, saved_variant, current_variant):
    model = LocalToy()
    before = copy.deepcopy(model.state_dict())
    path = tmp_path / 'reader.pt'
    torch.save({'epoch': 7, 'model': before,
                'config': {'candidate_local_visual_variant': saved_variant}}, str(path))
    args = native_args(monkeypatch, path)
    args.candidate_local_visual_variant = current_variant
    with pytest.raises(ValueError, match='reader variant mismatch'):
        load_checkpoint(args, model, None, None)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('variant', ['local', 'extent'])
def test_native_resume_restores_all_five_optimizer_groups(monkeypatch, tmp_path, variant):
    original = LocalToy()
    optimizer = torch.optim.AdamW(parameter_groups(original), weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 4], gamma=.1)
    for parameter in original.parameters():
        parameter.grad = torch.full_like(parameter, .01)
    optimizer.step()
    scheduler.step()
    path = tmp_path / 'complete_native.pt'
    torch.save({'epoch': 7, 'model': original.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': {'candidate_local_visual_variant': variant}}, str(path))
    model = LocalToy()
    restored = torch.optim.AdamW(parameter_groups(model), weight_decay=.0005)
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored, milestones=[1, 4], gamma=.1)
    args = native_args(monkeypatch, path)
    args.candidate_local_visual_variant = variant
    load_checkpoint(args, model, restored, restored_scheduler)
    assert args.start_epoch == 8
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in model.state_dict().items())
    expected = optimizer.state_dict()
    actual = restored.state_dict()
    assert actual['param_groups'] == expected['param_groups']
    assert set(actual['state']) == set(expected['state'])
    for identifier, state in actual['state'].items():
        assert state['step'] == expected['state'][identifier]['step']
        for field in ['exp_avg', 'exp_avg_sq']:
            assert torch.equal(state[field], expected['state'][identifier][field])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
