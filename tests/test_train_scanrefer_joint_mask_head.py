import importlib
import copy
import math
import traceback
from collections import OrderedDict

import pytest
import torch
from torch import nn


class _ToyMCLN(nn.Module):
    def __init__(self):
        super(_ToyMCLN, self).__init__()
        self.detector = nn.Linear(4, 4)
        self.x_mask = nn.Linear(4, 4)
        self.text_query_proj = nn.Linear(4, 4)
        self.swa_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.swa_ffn_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.out_norm = nn.LayerNorm(4)
        self.out_score = nn.Linear(4, 1)
        self.x_query = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(8, 2),
        )
        self.x_query.register_buffer("frozen_scale", torch.tensor([1.0]))
        self.register_buffer("frozen_counter", torch.tensor([3], dtype=torch.long))
        self.register_buffer(
            "nonpersistent_counter", torch.tensor([11], dtype=torch.long),
            persistent=False,
        )
        self.large_frozen = nn.Parameter(torch.ones(2048))

    def forward(self, inputs):
        return self.x_query(self.detector(inputs))


def test_configure_query_mask_head_trainability_is_exact_and_deterministic():
    trainer = importlib.import_module(
        "scripts.train_scanrefer_joint_mask_head"
    )
    torch.manual_seed(17)
    model = _ToyMCLN().train()

    group = trainer.configure_query_mask_head_trainability(model)

    trainable_names = tuple(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert trainable_names
    assert trainable_names == group["names"]
    assert all(name.startswith("x_query.") for name in trainable_names)
    assert set(trainable_names) == {
        "x_query.0.weight",
        "x_query.0.bias",
        "x_query.3.weight",
        "x_query.3.bias",
    }
    assert model.training is False
    assert model.x_query[2].training is False
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("x_query.")
    )

    inputs = torch.randn(3, 4)
    first = model(inputs)
    second = model(inputs)
    assert torch.equal(first, second)


def _trainer():
    return importlib.import_module("scripts.train_scanrefer_joint_mask_head")


def _state_clone(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_nested_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert tuple(actual.shape) == tuple(expected.shape)
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert list(actual.keys()) == list(expected.keys())
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_training_step_changes_x_query_only_and_keeps_eval_deterministic():
    trainer = _trainer()
    torch.manual_seed(23)
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    frozen_snapshot = trainer.snapshot_query_mask_frozen_state(model)
    before = _state_clone(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 2)
    eval_seen = []

    model.train()

    def loss_closure(active_model):
        eval_seen.append(
            active_model.training is False
            and active_model.x_query[2].training is False
        )
        prediction = active_model(inputs)
        return (prediction - targets).pow(2).mean()

    record = trainer.run_query_mask_training_step(
        model, optimizer, loss_closure, frozen_snapshot
    )

    assert eval_seen == [True]
    assert record["gradient_tensor_count"] == len(group["parameters"])
    assert record["loss"] >= 0.0
    assert model.training is False
    after = model.state_dict()
    changed = [
        name for name in group["names"]
        if not torch.equal(before[name], after[name].detach().cpu())
    ]
    assert changed
    for name, value in after.items():
        if name not in group["names"]:
            assert torch.equal(before[name], value.detach().cpu()), name
    trainer.verify_query_mask_frozen_state(model, frozen_snapshot)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_step_clips_global_x_query_gradient_and_reports_norms():
    trainer = _trainer()
    torch.manual_seed(29)
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    learning_rate = 0.25
    max_grad_norm = 0.05
    optimizer = torch.optim.SGD(group["parameters"], lr=learning_rate)
    inputs = torch.full((6, 4), 4.0)
    targets = torch.full((6, 2), -3.0)

    loss = (model(inputs) - targets).pow(2).mean()
    loss.backward()
    expected_gradients = tuple(
        parameter.grad.detach().clone() for parameter in group["parameters"]
    )
    pre_clip_norm = math.sqrt(sum(
        float(gradient.double().pow(2).sum().item())
        for gradient in expected_gradients
    ))
    assert pre_clip_norm > max_grad_norm
    clip_scale = max_grad_norm / (pre_clip_norm + 1e-6)
    before = tuple(
        parameter.detach().clone() for parameter in group["parameters"]
    )
    optimizer.zero_grad()

    record = trainer.run_query_mask_training_step(
        model,
        optimizer,
        lambda active_model: (
            active_model(inputs) - targets
        ).pow(2).mean(),
        snapshot,
        max_grad_norm=max_grad_norm,
    )

    for parameter, original, gradient in zip(
            group["parameters"], before, expected_gradients):
        expected = original - learning_rate * gradient * clip_scale
        assert torch.allclose(parameter, expected, rtol=1e-5, atol=1e-7)
    assert record["max_grad_norm"] == max_grad_norm
    assert isinstance(record["pre_clip_grad_norm"], float)
    assert isinstance(record["post_clip_grad_norm"], float)
    assert math.isfinite(record["pre_clip_grad_norm"])
    assert math.isfinite(record["post_clip_grad_norm"])
    assert record["pre_clip_grad_norm"] == pytest.approx(pre_clip_norm)
    assert record["post_clip_grad_norm"] <= max_grad_norm + 1e-7


def test_training_step_clips_only_revalidated_x_query_between_backward_and_step(
        monkeypatch):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    events = []

    class OrderedSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append(("step", tuple(
                parameter for parameter in model.parameters()
                if parameter.grad is not None
            )))
            return super(OrderedSGD, self).step(closure=closure)

    optimizer = OrderedSGD(group["parameters"], lr=0.1)
    original_clip = torch.nn.utils.clip_grad_norm_
    backward_handle = group["parameters"][0].register_hook(
        lambda gradient: events.append(("backward",)) or gradient
    )

    def clip_spy(parameters, max_norm, *args, **kwargs):
        parameters = tuple(parameters)
        assert all(parameter.grad is not None for parameter in parameters)
        events.append(("clip", parameters, max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)

    try:
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: active_model(torch.ones(2, 4)).sum(),
            snapshot,
        )
    finally:
        backward_handle.remove()

    assert [event[0] for event in events] == ["backward", "clip", "step"]
    assert events[1][1] == group["parameters"]
    assert events[1][2] == 0.1
    assert events[2][1] == group["parameters"]
    clipped_ids = {id(parameter) for parameter in events[1][1]}
    assert all(
        id(parameter) not in clipped_ids
        for name, parameter in model.named_parameters()
        if not name.startswith("x_query.")
    )


@pytest.mark.parametrize(
    "invalid_value, error_type",
    [
        (True, TypeError),
        (False, TypeError),
        (None, TypeError),
        ("0.1", TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (0.0, ValueError),
        (-0.1, ValueError),
    ],
)
def test_invalid_max_grad_norm_is_rejected_without_any_side_effect(
        invalid_value, error_type):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1, momentum=0.9)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))
    model.train()
    torch.manual_seed(31)
    before_rng = torch.get_rng_state().clone()
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_groups = optimizer.param_groups
    before_state = optimizer.state
    before_gradients = tuple(
        parameter.grad.detach().clone() for parameter in model.parameters()
    )
    before_modes = tuple(module.training for module in model.modules())
    before_requires_grad = tuple(
        parameter.requires_grad for parameter in model.parameters()
    )
    closure_calls = []

    with pytest.raises(
            error_type, match="max_grad_norm must be a finite positive real"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: closure_calls.append(active_model),
            snapshot,
            max_grad_norm=invalid_value,
        )

    assert closure_calls == []
    _assert_nested_equal(_state_clone(model), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert optimizer.param_groups is before_groups
    assert optimizer.state is before_state
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert tuple(module.training for module in model.modules()) == before_modes
    assert tuple(
        parameter.requires_grad for parameter in model.parameters()
    ) == before_requires_grad
    for parameter, expected in zip(model.parameters(), before_gradients):
        assert torch.equal(parameter.grad, expected)


class _MissingXQuery(nn.Module):
    def __init__(self):
        super(_MissingXQuery, self).__init__()
        self.detector = nn.Linear(2, 2)


class _EmptyXQuery(nn.Module):
    def __init__(self):
        super(_EmptyXQuery, self).__init__()
        self.detector = nn.Linear(2, 2)
        self.x_query = nn.Identity()


class _AliasedXQuery(nn.Module):
    def __init__(self):
        super(_AliasedXQuery, self).__init__()
        self.x_query = nn.Linear(2, 2)
        self.detector = self.x_query


class _PartiallyInvalidXQuery(nn.Module):
    def __init__(self):
        super(_PartiallyInvalidXQuery, self).__init__()
        self.x_query = nn.Module()
        self.x_query.register_parameter(
            "float_weight", nn.Parameter(torch.ones(2))
        )
        self.x_query.register_parameter(
            "integer_weight",
            nn.Parameter(torch.ones(2, dtype=torch.long), requires_grad=False),
        )


@pytest.mark.parametrize(
    "model_type, message",
    [
        (_MissingXQuery, "registered x_query"),
        (_EmptyXQuery, "at least one parameter"),
    ],
)
def test_configuration_fails_closed_for_missing_or_empty_x_query(
        model_type, message):
    trainer = _trainer()
    model = model_type().train()

    with pytest.raises(ValueError, match=message):
        trainer.configure_query_mask_head_trainability(model)

    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_configuration_rejects_x_query_parameters_shared_with_detector():
    trainer = _trainer()
    model = _AliasedXQuery().train()

    with pytest.raises(ValueError, match="shared|unique"):
        trainer.configure_query_mask_head_trainability(model)

    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_configuration_rolls_back_a_partial_unfreeze_failure():
    trainer = _trainer()
    model = _PartiallyInvalidXQuery().train()

    with pytest.raises(RuntimeError, match="gradient|requires_grad"):
        trainer.configure_query_mask_head_trainability(model)

    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_snapshot_rejects_an_extra_unfrozen_parameter():
    trainer = _trainer()
    model = _ToyMCLN()
    trainer.configure_query_mask_head_trainability(model)
    model.detector.weight.requires_grad_(True)

    with pytest.raises(RuntimeError, match="only exact x_query"):
        trainer.snapshot_query_mask_frozen_state(model)


@pytest.mark.parametrize("frozen_kind", ["parameter", "buffer"])
def test_snapshot_rejects_trainable_frozen_storage_alias_without_side_effects(
        frozen_kind):
    trainer = _trainer()
    model = _ToyMCLN()
    trainable = model.x_query[0].weight
    alias = trainable.detach().view_as(trainable)
    if frozen_kind == "parameter":
        model.register_parameter(
            "aliased_frozen", nn.Parameter(alias, requires_grad=False)
        )
    else:
        model.register_buffer("aliased_frozen", alias)
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    forward_calls = []
    handle = model.register_forward_pre_hook(
        lambda *_args: forward_calls.append(True)
    )
    before_state = _state_clone(model)
    before_trainability = tuple(
        (name, parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )
    trainable_pointer = trainable.storage().data_ptr()
    alias_pointer = model.aliased_frozen.storage().data_ptr()
    assert trainable_pointer == alias_pointer

    try:
        with pytest.raises(ValueError, match="storage.*alias|alias.*storage"):
            trainer.snapshot_query_mask_frozen_state(model)
    finally:
        handle.remove()

    _assert_nested_equal(_state_clone(model), before_state)
    assert tuple(
        (name, parameter.requires_grad)
        for name, parameter in model.named_parameters()
    ) == before_trainability
    assert trainable.storage().data_ptr() == trainable_pointer
    assert model.aliased_frozen.storage().data_ptr() == alias_pointer
    assert forward_calls == []
    assert optimizer.step_calls == 0


def test_runtime_storage_alias_is_rejected_before_step_and_pointer_restored():
    trainer = _trainer()
    model = _ToyMCLN()
    trainable = model.x_query[0].weight
    model.register_buffer(
        "runtime_frozen_alias", trainable.detach().clone()
    )
    frozen = model.runtime_frozen_alias
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    frozen_before = frozen.detach().clone()
    frozen_data_pointer = frozen.data_ptr()
    frozen_storage_pointer = frozen.storage().data_ptr()
    frozen_storage_offset = frozen.storage_offset()
    frozen_stride = frozen.stride()
    assert frozen_storage_pointer != trainable.storage().data_ptr()

    def introduce_alias_and_return(active_model):
        active_model.runtime_frozen_alias.data = (
            active_model.x_query[0].weight.data
        )
        assert (
            active_model.runtime_frozen_alias.storage().data_ptr()
            == active_model.x_query[0].weight.storage().data_ptr()
        )
        return active_model(torch.ones(2, 4)).sum()

    with pytest.raises(ValueError, match="storage.*alias|alias.*storage"):
        trainer.run_query_mask_training_step(
            model, optimizer, introduce_alias_and_return, snapshot
        )

    assert optimizer.step_calls == 0
    assert model.runtime_frozen_alias is frozen
    assert frozen.data_ptr() == frozen_data_pointer
    assert frozen.storage().data_ptr() == frozen_storage_pointer
    assert frozen.storage_offset() == frozen_storage_offset
    assert frozen.stride() == frozen_stride
    assert frozen.storage().data_ptr() != trainable.storage().data_ptr()
    assert torch.equal(frozen, frozen_before)

    trainer.run_query_mask_training_step(
        model,
        optimizer,
        lambda active_model: active_model(torch.ones(2, 4)).sum(),
        snapshot,
    )
    assert optimizer.step_calls == 1


@pytest.mark.parametrize(
    "state_name",
    ["detector.weight", "frozen_counter", "x_query.frozen_scale"],
)
def test_frozen_state_verifier_rejects_changed_parameters_and_buffers(
        state_name):
    trainer = _trainer()
    model = _ToyMCLN()
    trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)

    with torch.no_grad():
        model.state_dict()[state_name].add_(1)

    with pytest.raises(RuntimeError, match=state_name.replace(".", r"\.")):
        trainer.verify_query_mask_frozen_state(model, snapshot)


def test_bitwise_view_of_noncontiguous_tensor_reuses_storage_without_materializing():
    trainer = _trainer()
    value = torch.arange(48, dtype=torch.float32).reshape(8, 6)[:, 1::2]
    assert not value.is_contiguous()
    storage_pointer = value.storage().data_ptr()

    bitwise = trainer._bitwise_tensor(value, "noncontiguous tensor")

    assert bitwise.dtype == torch.int32
    assert bitwise.data_ptr() == value.data_ptr()
    assert bitwise.storage().data_ptr() == storage_pointer
    assert bitwise.storage_offset() == value.storage_offset()
    assert bitwise.stride() == value.stride()


def test_training_step_never_materializes_large_noncontiguous_frozen_buffer(
        monkeypatch):
    trainer = _trainer()
    model = _ToyMCLN()
    backing = torch.arange(32768, dtype=torch.float32).reshape(8192, 4)
    model.register_buffer("large_noncontiguous", backing[:, 1])
    frozen = model.large_noncontiguous
    assert not frozen.is_contiguous()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.05)
    data_pointer = frozen.data_ptr()
    storage_pointer = frozen.storage().data_ptr()
    storage_offset = frozen.storage_offset()
    stride = frozen.stride()
    materializations = []
    original_contiguous = torch.Tensor.contiguous
    original_clone = torch.Tensor.clone

    def guarded_contiguous(value, *args, **kwargs):
        if value.storage().data_ptr() == storage_pointer:
            materializations.append("contiguous")
            raise AssertionError("frozen tensor must not be made contiguous")
        return original_contiguous(value, *args, **kwargs)

    def guarded_clone(value, *args, **kwargs):
        if value.storage().data_ptr() == storage_pointer:
            materializations.append("clone")
            raise AssertionError("frozen tensor must not be cloned per step")
        return original_clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", guarded_contiguous)
    monkeypatch.setattr(torch.Tensor, "clone", guarded_clone)

    trainer.verify_query_mask_frozen_state(model, snapshot)
    trainer.run_query_mask_training_step(
        model,
        optimizer,
        lambda active_model: active_model(torch.ones(2, 4)).sum(),
        snapshot,
    )

    assert materializations == []
    assert frozen.data_ptr() == data_pointer
    assert frozen.storage().data_ptr() == storage_pointer
    assert frozen.storage_offset() == storage_offset
    assert frozen.stride() == stride


def test_frozen_state_verifier_rejects_trainable_collection_drift():
    trainer = _trainer()
    model = _ToyMCLN()
    trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    model.x_query[0].weight.requires_grad_(False)

    with pytest.raises(RuntimeError, match="trainable parameter set drift"):
        trainer.verify_query_mask_frozen_state(model, snapshot)


@pytest.mark.parametrize("failure", ["missing", "nan", "frozen"])
def test_gradient_validation_fails_closed(failure):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    for parameter in group["parameters"]:
        parameter.grad = torch.ones_like(parameter)

    if failure == "missing":
        group["parameters"][0].grad = None
        expected = "has no gradient"
    elif failure == "nan":
        group["parameters"][0].grad.fill_(float("nan"))
        expected = "non-finite gradient"
    else:
        model.detector.weight.grad = torch.zeros_like(model.detector.weight)
        expected = "frozen parameter.*gradient"

    with pytest.raises((RuntimeError, FloatingPointError), match=expected):
        trainer.validate_query_mask_gradients(model)


def test_training_step_rejects_optimizer_parameter_drift_before_forward():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(
        group["parameters"] + (model.detector.weight,), lr=0.1
    )
    called = []

    with pytest.raises(ValueError, match="optimizer parameter set"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: called.append(True) or active_model(
                torch.ones(1, 4)
            ).sum(),
            snapshot,
        )

    assert called == []


def test_training_step_rejects_closure_that_switches_model_to_train():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)

    def loss_closure(active_model):
        active_model.train()
        return active_model(torch.ones(2, 4)).sum()

    with pytest.raises(RuntimeError, match="closure.*train|eval mode"):
        trainer.run_query_mask_training_step(
            model, optimizer, loss_closure, snapshot
        )

    assert all(not module.training for module in model.modules())


class _ClosureFailure(Exception):
    pass


class _InjectedEvalFailureModule(nn.Module):
    def train(self, mode=True):
        raise RuntimeError("injected module eval failed")


def test_training_step_restores_eval_when_closure_raises():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)

    def loss_closure(active_model):
        active_model.train()
        raise _ClosureFailure("closure failed")

    with pytest.raises(_ClosureFailure, match="closure failed"):
        trainer.run_query_mask_training_step(
            model, optimizer, loss_closure, snapshot
        )

    assert all(not module.training for module in model.modules())


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_training_step_fully_rolls_back_base_exception_and_preserves_traceback(
        exception_type):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    first = group["parameters"][0]
    state_tensor = torch.tensor([2.5], dtype=torch.float64)
    shared_list = [state_tensor, "stable"]
    optimizer.state[first]["tensor"] = state_tensor
    optimizer.state[first]["shared"] = shared_list
    optimizer.defaults["shared"] = shared_list
    optimizer.param_groups[0]["shared"] = shared_list
    optimizer_state = optimizer.state
    optimizer_defaults = optimizer.defaults
    optimizer_groups = optimizer.param_groups
    optimizer_group = optimizer.param_groups[0]
    optimizer_params = optimizer_group["params"]
    state_entry = optimizer.state[first]
    module = model.detector
    hook_handle = module.register_forward_hook(lambda *_args: None)
    hook_registry = module._forward_hooks
    hook_entries = tuple(hook_registry.items())
    pre_hook_registries = tuple(
        (active_module, active_module._forward_pre_hooks,
         tuple(active_module._forward_pre_hooks.items()))
        for active_module in model.modules()
    )
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_requires_grad = tuple(
        parameter.requires_grad for parameter in model.parameters()
    )
    torch.manual_seed(443)
    before_rng = torch.get_rng_state().clone()
    sentinel = exception_type("base exception sentinel")

    def mutate_everything_then_abort(active_model):
        torch.rand(19)
        with torch.no_grad():
            active_model.x_query[0].weight.add_(7.0)
            active_model.detector.weight.add_(11.0)
            active_model.frozen_counter.add_(3)
            state_tensor.add_(13.0)
        shared_list.append("injected")
        state_entry["injected"] = torch.tensor([17.0])
        optimizer.state = {"replacement": {}}
        optimizer.defaults = {"replacement": True}
        optimizer.param_groups = [{"params": []}]
        module._forward_hooks = OrderedDict()
        active_model.register_buffer(
            "injected_buffer", torch.tensor([23.0])
        )
        active_model.detector.weight.requires_grad_(True)
        active_model.x_query[0].weight.requires_grad_(False)
        active_model.train()
        for parameter in active_model.parameters():
            parameter.grad = torch.ones_like(parameter)
        raise sentinel

    try:
        with pytest.raises(exception_type) as caught:
            trainer.run_query_mask_training_step(
                model, optimizer, mutate_everything_then_abort, snapshot
            )

        assert caught.value is sentinel
        traceback_names = tuple(
            frame.name for frame in traceback.extract_tb(
                caught.value.__traceback__
            )
        )
        assert "mutate_everything_then_abort" in traceback_names
        _assert_nested_equal(_state_clone(model), before_model)
        _assert_nested_equal(optimizer.state_dict(), before_optimizer)
        assert optimizer.state is optimizer_state
        assert optimizer.defaults is optimizer_defaults
        assert optimizer.param_groups is optimizer_groups
        assert optimizer.param_groups[0] is optimizer_group
        assert optimizer_group["params"] is optimizer_params
        assert optimizer.state[first] is state_entry
        assert optimizer.state[first]["tensor"] is state_tensor
        assert optimizer.state[first]["shared"] is shared_list
        assert optimizer.defaults["shared"] is shared_list
        assert optimizer_group["shared"] is shared_list
        assert shared_list == [state_tensor, "stable"]
        assert module._forward_hooks is hook_registry
        assert tuple(hook_registry.items()) == hook_entries
        for active_module, registry, entries in pre_hook_registries:
            assert active_module._forward_pre_hooks is registry
            assert tuple(registry.items()) == entries
        assert "injected_buffer" not in model._buffers
        assert torch.equal(torch.get_rng_state(), before_rng)
        assert tuple(
            parameter.requires_grad for parameter in model.parameters()
        ) == before_requires_grad
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(not active_module.training for active_module in model.modules())
    finally:
        hook_handle.remove()


@pytest.mark.parametrize("cleanup_layer", ["inner", "outer"])
def test_base_exception_primary_is_not_replaced_by_cleanup_error(
        monkeypatch, cleanup_layer):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    before = _state_clone(model)
    sentinel = KeyboardInterrupt("primary interrupt")

    if cleanup_layer == "inner":
        original_cleanup = trainer._remove_forward_guards

        def fail_after_guard_cleanup(guards):
            original_cleanup(guards)
            raise SystemExit("inner cleanup failure")

        monkeypatch.setattr(
            trainer, "_remove_forward_guards", fail_after_guard_cleanup
        )
    else:
        original_cleanup = trainer._clear_model_gradients
        cleanup_calls = []

        def fail_during_final_cleanup(active_model, parameter_records=()):
            cleanup_calls.append(True)
            original_cleanup(active_model, parameter_records)
            if len(cleanup_calls) == 2:
                raise SystemExit("outer cleanup failure")

        monkeypatch.setattr(
            trainer, "_clear_model_gradients", fail_during_final_cleanup
        )

    def mutate_then_interrupt(active_model):
        with torch.no_grad():
            active_model.x_query[0].weight.add_(5.0)
        raise sentinel

    with pytest.raises(KeyboardInterrupt) as caught:
        trainer.run_query_mask_training_step(
            model, optimizer, mutate_then_interrupt, snapshot
        )

    assert caught.value is sentinel
    _assert_nested_equal(_state_clone(model), before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not active_module.training for active_module in model.modules())


def test_base_exception_and_rollback_failure_are_reported_together(monkeypatch):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    sentinel = SystemExit("primary system exit")
    rollback_failure = KeyboardInterrupt("rollback interrupt")

    def fail_rollback(*_args, **_kwargs):
        raise rollback_failure

    monkeypatch.setattr(
        trainer, "_restore_training_transaction", fail_rollback
    )

    with pytest.raises(trainer.QueryMaskRollbackError) as caught:
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda _active_model: (_ for _ in ()).throw(sentinel),
            snapshot,
        )

    assert caught.value.original_error is sentinel
    assert caught.value.rollback_error is rollback_failure


def test_cleanup_eval_failure_does_not_replace_primary_closure_exception():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    module_hook = model.register_forward_hook(lambda *_args: None)
    tensor = model.x_query[0].weight
    tensor_hook = tensor.register_hook(lambda gradient: gradient)
    module_hook_registry = model._forward_hooks
    module_hook_entries = tuple(module_hook_registry.items())
    tensor_hook_registry = tensor._backward_hooks
    tensor_hook_entries = tuple(tensor_hook_registry.items())
    modules_container = model._modules
    module_entries = tuple(modules_container.items())
    optimizer_state = optimizer.state
    optimizer_defaults = optimizer.defaults
    optimizer_groups = optimizer.param_groups
    optimizer_group = optimizer_groups[0]
    optimizer_params = optimizer_group["params"]
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    torch.manual_seed(433)
    before_rng = torch.get_rng_state().clone()
    tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
    tensor_topology = tuple(
        (
            name,
            value,
            value.data_ptr(),
            value.storage().data_ptr(),
            value.storage_offset(),
            value.stride(),
        )
        for name, value in tensors
    )
    sentinel = _ClosureFailure("sentinel closure failure")

    def inject_eval_failure_then_raise(active_model):
        torch.rand(17)
        active_model.add_module(
            "injected_eval_failure", _InjectedEvalFailureModule()
        )
        raise sentinel

    try:
        with pytest.raises(_ClosureFailure) as caught:
            trainer.run_query_mask_training_step(
                model,
                optimizer,
                inject_eval_failure_then_raise,
                snapshot,
            )

        assert caught.value is sentinel
        assert optimizer.step_calls == 0
        assert "injected_eval_failure" not in model._modules
        _assert_nested_equal(_state_clone(model), before_model)
        _assert_nested_equal(optimizer.state_dict(), before_optimizer)
        assert optimizer.state is optimizer_state
        assert optimizer.defaults is optimizer_defaults
        assert optimizer.param_groups is optimizer_groups
        assert optimizer.param_groups[0] is optimizer_group
        assert optimizer_group["params"] is optimizer_params
        assert model._modules is modules_container
        assert tuple(model._modules.items()) == module_entries
        assert model._forward_hooks is module_hook_registry
        assert tuple(module_hook_registry.items()) == module_hook_entries
        assert tensor._backward_hooks is tensor_hook_registry
        assert tuple(tensor_hook_registry.items()) == tensor_hook_entries
        assert torch.equal(torch.get_rng_state(), before_rng)
        for (
                name, value, data_pointer, storage_pointer,
                storage_offset, stride) in tensor_topology:
            assert dict(tensors)[name] is value
            assert value.data_ptr() == data_pointer
            assert value.storage().data_ptr() == storage_pointer
            assert value.storage_offset() == storage_offset
            assert value.stride() == stride
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(not module.training for module in model.modules())
    finally:
        module_hook.remove()
        tensor_hook.remove()


def test_outer_cleanup_preserves_combined_rollback_error():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    sentinel = _ClosureFailure("persistent eval primary failure")
    rollback_failure = RuntimeError("persistent eval rollback failure")

    def broken_eval():
        raise rollback_failure

    def override_eval_then_raise(active_model):
        active_model.eval = broken_eval
        raise sentinel

    try:
        with pytest.raises(trainer.QueryMaskRollbackError) as caught:
            trainer.run_query_mask_training_step(
                model, optimizer, override_eval_then_raise, snapshot
            )
    finally:
        model.__dict__.pop("eval", None)

    assert caught.value.original_error is sentinel
    assert caught.value.rollback_error is rollback_failure
    assert all(parameter.grad is None for parameter in model.parameters())


def test_outer_eval_failure_does_not_replace_primary_after_rollback():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    sentinel = _ClosureFailure("outer eval primary failure")
    outer_failure = RuntimeError("outer final eval failure")
    eval_calls = []

    def install_flaky_eval_then_raise(active_model):
        original_eval = active_model.eval

        def flaky_eval():
            eval_calls.append(True)
            if len(eval_calls) == 3:
                raise outer_failure
            return original_eval()

        active_model.eval = flaky_eval
        raise sentinel

    try:
        with pytest.raises(_ClosureFailure) as caught:
            trainer.run_query_mask_training_step(
                model,
                optimizer,
                install_flaky_eval_then_raise,
                snapshot,
            )
    finally:
        model.__dict__.pop("eval", None)

    assert caught.value is sentinel
    assert eval_calls == [True, True, True]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_success_path_outer_eval_cleanup_failure_is_raised():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    cleanup_failure = RuntimeError("success outer final eval failure")
    eval_calls = []

    def install_flaky_eval_and_return(active_model):
        original_eval = active_model.eval

        def flaky_eval():
            eval_calls.append(True)
            if len(eval_calls) == 4:
                raise cleanup_failure
            return original_eval()

        active_model.eval = flaky_eval
        return active_model(torch.ones(2, 4)).sum()

    try:
        with pytest.raises(RuntimeError) as caught:
            trainer.run_query_mask_training_step(
                model,
                optimizer,
                install_flaky_eval_and_return,
                snapshot,
            )
    finally:
        model.__dict__.pop("eval", None)

    assert caught.value is cleanup_failure
    assert eval_calls == [True, True, True, True]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_unrecoverable_rollback_reports_primary_and_restore_errors_together():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)

    def unavailable_state_dict():
        raise RuntimeError("state_dict unavailable during rollback")

    def make_rollback_unrecoverable(active_model):
        active_model.state_dict = unavailable_state_dict
        raise _ClosureFailure("primary closure failure")

    try:
        with pytest.raises(RuntimeError) as caught:
            trainer.run_query_mask_training_step(
                model, optimizer, make_rollback_unrecoverable, snapshot
            )
    finally:
        model.__dict__.pop("state_dict", None)

    message = str(caught.value)
    assert "primary closure failure" in message
    assert "state_dict unavailable during rollback" in message
    assert isinstance(caught.value.original_error, _ClosureFailure)
    assert str(caught.value.original_error) == "primary closure failure"
    assert isinstance(caught.value.rollback_error, RuntimeError)
    assert str(caught.value.rollback_error) == (
        "state_dict unavailable during rollback"
    )


def test_failed_training_step_restores_cpu_rng_state():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    torch.manual_seed(401)
    before = torch.get_rng_state().clone()

    def consume_rng_then_raise(_active_model):
        torch.rand(37)
        raise _ClosureFailure("cpu rng failure")

    with pytest.raises(_ClosureFailure, match="cpu rng failure"):
        trainer.run_query_mask_training_step(
            model, optimizer, consume_rng_then_raise, snapshot
        )

    assert torch.equal(torch.get_rng_state(), before)


def test_successful_training_step_keeps_advanced_cpu_rng_state():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    torch.manual_seed(409)
    before = torch.get_rng_state().clone()

    def consume_rng_and_return(active_model):
        torch.rand(37)
        return active_model(torch.ones(2, 4)).sum()

    trainer.run_query_mask_training_step(
        model, optimizer, consume_rng_and_return, snapshot
    )

    assert not torch.equal(torch.get_rng_state(), before)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA RNG state requires CUDA"
)
def test_failed_training_step_restores_all_cuda_rng_states():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    torch.cuda.manual_seed_all(419)
    before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    def consume_cuda_rng_then_raise(_active_model):
        for device_index in range(torch.cuda.device_count()):
            torch.rand(37, device=torch.device("cuda", device_index))
        raise _ClosureFailure("cuda rng failure")

    with pytest.raises(_ClosureFailure, match="cuda rng failure"):
        trainer.run_query_mask_training_step(
            model, optimizer, consume_cuda_rng_then_raise, snapshot
        )

    after = tuple(torch.cuda.get_rng_state_all())
    assert len(after) == len(before)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(after, before)
    )


class _ParamGroupDriftSGD(torch.optim.SGD):
    def __init__(self, parameters, extra_parameter, **kwargs):
        super(_ParamGroupDriftSGD, self).__init__(parameters, **kwargs)
        self.extra_parameter = extra_parameter

    def step(self, closure=None):
        result = super(_ParamGroupDriftSGD, self).step(closure=closure)
        self.param_groups[0]["params"].append(self.extra_parameter)
        return result


def test_training_step_rechecks_optimizer_parameters_after_step():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _ParamGroupDriftSGD(
        group["parameters"], model.detector.weight, lr=0.1
    )

    with pytest.raises(ValueError, match="optimizer parameter set"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: active_model(torch.ones(2, 4)).sum(),
            snapshot,
        )


class _MutatingOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, model, raises=False):
        super(_MutatingOptimizer, self).__init__(parameters, {"lr": 0.1})
        self.model = model
        self.raises = raises

    def step(self, closure=None):
        with torch.no_grad():
            for parameter in self.param_groups[0]["params"]:
                parameter.add_(1.0)
            self.model.detector.weight.add_(7.0)
            self.model.frozen_counter.add_(3)
            self.model.x_query.frozen_scale.add_(5.0)
        first = self.param_groups[0]["params"][0]
        self.state[first]["unexpected"] = torch.tensor([9.0])
        self.param_groups[0]["params"].append(self.model.detector.weight)
        if self.raises:
            raise RuntimeError("optimizer step failed after mutation")


def test_training_step_rolls_back_after_backward_validation_failure():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())

    def incomplete_loss(active_model):
        with torch.no_grad():
            active_model.x_query[0].weight.add_(4.0)
        return active_model.x_query[0](
            active_model.detector(torch.ones(2, 4))
        ).sum()

    with pytest.raises(RuntimeError, match="has no gradient"):
        trainer.run_query_mask_training_step(
            model, optimizer, incomplete_loss, snapshot
        )

    _assert_nested_equal(_state_clone(model), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_training_step_rolls_back_after_post_step_validation_failure():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _MutatingOptimizer(group["parameters"], model)
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(ValueError, match="optimizer parameter set"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: active_model(torch.ones(2, 4)).sum(),
            snapshot,
        )

    _assert_nested_equal(_state_clone(model), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_training_step_rolls_back_when_optimizer_step_raises():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _MutatingOptimizer(group["parameters"], model, raises=True)
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(RuntimeError, match="step failed after mutation"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: active_model(torch.ones(2, 4)).sum(),
            snapshot,
        )

    _assert_nested_equal(_state_clone(model), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super(_CountingSGD, self).__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super(_CountingSGD, self).step(closure=closure)


def test_train_then_forward_then_eval_inside_closure_is_rejected_and_rolled_back():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    hook_counts = tuple(len(module._forward_pre_hooks) for module in model.modules())

    def hidden_train_forward(active_model):
        active_model.train()
        output = active_model(torch.ones(2, 4))
        active_model.eval()
        return output.sum()

    with pytest.raises(RuntimeError, match="train mode.*forward|forward.*train"):
        trainer.run_query_mask_training_step(
            model, optimizer, hidden_train_forward, snapshot
        )

    _assert_nested_equal(_state_clone(model), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert tuple(
        len(module._forward_pre_hooks) for module in model.modules()
    ) == hook_counts
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_removed_eval_guards_are_rejected_before_backward_and_fully_rolled_back():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    module_hook = model.register_forward_hook(lambda *_args: None)
    tensor = model.x_query[0].weight
    tensor_hook = tensor.register_hook(lambda gradient: gradient)
    module_hook_registry = model._forward_hooks
    module_hook_entries = tuple(module_hook_registry.items())
    tensor_hook_registry = tensor._backward_hooks
    tensor_hook_entries = tuple(tensor_hook_registry.items())
    modules_container = model._modules
    module_entries = tuple(modules_container.items())
    optimizer_state = optimizer.state
    optimizer_defaults = optimizer.defaults
    optimizer_groups = optimizer.param_groups
    optimizer_group = optimizer_groups[0]
    optimizer_params = optimizer_group["params"]
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    torch.manual_seed(431)
    before_rng = torch.get_rng_state().clone()
    tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
    tensor_topology = tuple(
        (
            name,
            value,
            value.data_ptr(),
            value.storage().data_ptr(),
            value.storage_offset(),
            value.stride(),
        )
        for name, value in tensors
    )

    def remove_guards_train_forward_and_return(active_model):
        for module in active_model.modules():
            module._forward_pre_hooks.clear()
        active_model.train()
        output = active_model(torch.ones(2, 4))
        active_model.eval()
        return output.sum()

    try:
        with pytest.raises(RuntimeError, match="guard"):
            trainer.run_query_mask_training_step(
                model,
                optimizer,
                remove_guards_train_forward_and_return,
                snapshot,
            )

        assert optimizer.step_calls == 0
        _assert_nested_equal(_state_clone(model), before_model)
        _assert_nested_equal(optimizer.state_dict(), before_optimizer)
        assert optimizer.state is optimizer_state
        assert optimizer.defaults is optimizer_defaults
        assert optimizer.param_groups is optimizer_groups
        assert optimizer.param_groups[0] is optimizer_group
        assert optimizer_group["params"] is optimizer_params
        assert model._modules is modules_container
        assert tuple(model._modules.items()) == module_entries
        assert model._forward_hooks is module_hook_registry
        assert tuple(module_hook_registry.items()) == module_hook_entries
        assert tensor._backward_hooks is tensor_hook_registry
        assert tuple(tensor_hook_registry.items()) == tensor_hook_entries
        assert torch.equal(torch.get_rng_state(), before_rng)
        for (
                name, value, data_pointer, storage_pointer,
                storage_offset, stride) in tensor_topology:
            assert dict(tensors)[name] is value
            assert value.data_ptr() == data_pointer
            assert value.storage().data_ptr() == storage_pointer
            assert value.storage_offset() == storage_offset
            assert value.stride() == stride
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(not module.training for module in model.modules())
    finally:
        module_hook.remove()
        tensor_hook.remove()


@pytest.mark.parametrize("tamper_mode", ["clear_restore", "rebind_restore"])
def test_temporarily_removed_eval_guards_cannot_be_restored_to_hide_forward(
        tamper_mode):
    trainer = _trainer()
    model = _ToyMCLN()
    other_model = _ToyMCLN().train()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    module_hook = model.register_forward_hook(lambda *_args: None)
    tensor = model.x_query[0].weight
    tensor_hook = tensor.register_hook(lambda gradient: gradient)
    module_hook_registry = model._forward_hooks
    module_hook_entries = tuple(module_hook_registry.items())
    tensor_hook_registry = tensor._backward_hooks
    tensor_hook_entries = tuple(tensor_hook_registry.items())
    local_pre_hook_registries = tuple(
        (module, module._forward_pre_hooks,
         tuple(module._forward_pre_hooks.items()))
        for module in model.modules()
    )
    global_registry = torch.nn.modules.module._global_forward_pre_hooks
    global_entries = tuple(global_registry.items())
    modules_container = model._modules
    module_entries = tuple(modules_container.items())
    optimizer_state = optimizer.state
    optimizer_defaults = optimizer.defaults
    optimizer_groups = optimizer.param_groups
    optimizer_group = optimizer_groups[0]
    optimizer_params = optimizer_group["params"]
    before_model = _state_clone(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    torch.manual_seed(439)
    before_rng = torch.get_rng_state().clone()
    tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
    tensor_topology = tuple(
        (
            name,
            value,
            value.data_ptr(),
            value.storage().data_ptr(),
            value.storage_offset(),
            value.stride(),
        )
        for name, value in tensors
    )

    def temporarily_remove_guards(active_model):
        active_registries = tuple(
            (module, module._forward_pre_hooks,
             tuple(module._forward_pre_hooks.items()))
            for module in active_model.modules()
        )
        if tamper_mode == "clear_restore":
            for _module, registry, _entries in active_registries:
                registry.clear()
        else:
            for module, _registry, _entries in active_registries:
                module._forward_pre_hooks = OrderedDict()
        try:
            other_model.train()
            other_model(torch.ones(2, 4))
            active_model.train()
            return active_model(torch.ones(2, 4)).sum()
        finally:
            active_model.eval()
            for module, registry, entries in active_registries:
                registry.clear()
                registry.update(entries)
                module._forward_pre_hooks = registry

    try:
        with pytest.raises(
                RuntimeError, match="train mode.*forward|forward.*train|guard"):
            trainer.run_query_mask_training_step(
                model, optimizer, temporarily_remove_guards, snapshot
            )

        assert optimizer.step_calls == 0
        _assert_nested_equal(_state_clone(model), before_model)
        _assert_nested_equal(optimizer.state_dict(), before_optimizer)
        assert optimizer.state is optimizer_state
        assert optimizer.defaults is optimizer_defaults
        assert optimizer.param_groups is optimizer_groups
        assert optimizer.param_groups[0] is optimizer_group
        assert optimizer_group["params"] is optimizer_params
        assert model._modules is modules_container
        assert tuple(model._modules.items()) == module_entries
        assert model._forward_hooks is module_hook_registry
        assert tuple(module_hook_registry.items()) == module_hook_entries
        assert tensor._backward_hooks is tensor_hook_registry
        assert tuple(tensor_hook_registry.items()) == tensor_hook_entries
        for module, registry, entries in local_pre_hook_registries:
            assert module._forward_pre_hooks is registry
            assert tuple(registry.items()) == entries
        assert (
            torch.nn.modules.module._global_forward_pre_hooks
            is global_registry
        )
        assert tuple(global_registry.items()) == global_entries
        assert torch.equal(torch.get_rng_state(), before_rng)
        for (
                name, value, data_pointer, storage_pointer,
                storage_offset, stride) in tensor_topology:
            assert dict(tensors)[name] is value
            assert value.data_ptr() == data_pointer
            assert value.storage().data_ptr() == storage_pointer
            assert value.storage_offset() == storage_offset
            assert value.stride() == stride
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(not module.training for module in model.modules())
        other_model.train()
        other_model(torch.ones(2, 4))
    finally:
        module_hook.remove()
        tensor_hook.remove()


def test_nonpersistent_buffer_mutation_is_rejected_before_backward_and_restored():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    before = model.nonpersistent_counter.detach().clone()

    def mutate_buffer(active_model):
        active_model.nonpersistent_counter.add_(1)
        return active_model(torch.ones(2, 4)).sum()

    with pytest.raises(RuntimeError, match="nonpersistent_counter|buffer"):
        trainer.run_query_mask_training_step(
            model, optimizer, mutate_buffer, snapshot
        )

    assert optimizer.step_calls == 0
    assert torch.equal(model.nonpersistent_counter, before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_nonpersistent_buffer_is_restored_when_closure_raises():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    before = model.nonpersistent_counter.detach().clone()

    def mutate_then_raise(active_model):
        active_model.nonpersistent_counter.add_(7)
        raise _ClosureFailure("nonpersistent closure failure")

    with pytest.raises(_ClosureFailure, match="nonpersistent closure failure"):
        trainer.run_query_mask_training_step(
            model, optimizer, mutate_then_raise, snapshot
        )

    assert torch.equal(model.nonpersistent_counter, before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_registered_buffer_injection_is_removed_and_original_error_preserved():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    before_keys = tuple(model._buffers.keys())

    def inject_then_raise(active_model):
        active_model.register_buffer("injected_buffer", torch.tensor([4.0]))
        raise _ClosureFailure("registry closure failure")

    with pytest.raises(_ClosureFailure, match="registry closure failure"):
        trainer.run_query_mask_training_step(
            model, optimizer, inject_then_raise, snapshot
        )

    assert tuple(model._buffers.keys()) == before_keys
    assert "injected_buffer" not in model._buffers
    assert "injected_buffer" not in model.state_dict()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_registry_drift_is_rejected_before_backward_and_restored():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _CountingSGD(group["parameters"], lr=0.1)
    before_keys = tuple(model._buffers.keys())

    def inject_and_return(active_model):
        active_model.register_buffer("injected_buffer", torch.tensor([4.0]))
        return active_model(torch.ones(2, 4)).sum()

    with pytest.raises(RuntimeError, match="registry|structure"):
        trainer.run_query_mask_training_step(
            model, optimizer, inject_and_return, snapshot
        )

    assert optimizer.step_calls == 0
    assert tuple(model._buffers.keys()) == before_keys
    assert "injected_buffer" not in model._buffers


_MODULE_HOOK_REGISTRY_NAMES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


@pytest.mark.parametrize("outcome", ["raise", "return"])
def test_module_hook_registry_drift_is_rejected_and_restored(outcome):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    module = model.detector
    original_calls = []
    injected_calls = []

    def original_hook(*_args, **_kwargs):
        original_calls.append(True)

    def injected_hook(*_args, **_kwargs):
        injected_calls.append(True)

    original_registries = {}
    for index, name in enumerate(_MODULE_HOOK_REGISTRY_NAMES):
        if not hasattr(module, name):
            setattr(module, name, OrderedDict())
        registry = getattr(module, name)
        registry[10000 + index] = original_hook
        original_registries[name] = (registry, tuple(registry.items()))
    original_full_backward_hook = module._is_full_backward_hook

    def mutate_hooks(active_model):
        for index, name in enumerate(_MODULE_HOOK_REGISTRY_NAMES):
            registry = getattr(active_model.detector, name)
            registry.clear()
            setattr(
                active_model.detector,
                name,
                OrderedDict([(20000 + index, injected_hook)]),
            )
        active_model.detector._is_full_backward_hook = True
        if outcome == "raise":
            raise _ClosureFailure("module hook registry failure")
        return active_model(torch.ones(2, 4)).sum()

    if outcome == "raise":
        expected_error = _ClosureFailure
        expected_message = "module hook registry failure"
    else:
        expected_error = RuntimeError
        expected_message = "hook"
    with pytest.raises(expected_error, match=expected_message):
        trainer.run_query_mask_training_step(
            model, optimizer, mutate_hooks, snapshot
        )

    for name, (registry, entries) in original_registries.items():
        assert getattr(module, name) is registry
        assert tuple(registry.items()) == entries
    assert module._is_full_backward_hook is original_full_backward_hook
    injected_calls[:] = []
    model(torch.ones(1, 4))
    assert injected_calls == []


@pytest.mark.parametrize("owner_name", ["parameter", "buffer"])
@pytest.mark.parametrize("outcome", ["raise", "return"])
def test_tensor_backward_hook_registry_drift_is_rejected_and_restored(
        owner_name, outcome):
    trainer = _trainer()
    model = _ToyMCLN()
    model.x_query.frozen_scale.requires_grad_(True)
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    if owner_name == "parameter":
        tensor = model.x_query[0].weight
    else:
        tensor = model.x_query.frozen_scale
    original_calls = []
    injected_calls = []
    handle = tensor.register_hook(
        lambda gradient: original_calls.append(True) or gradient
    )
    registry = tensor._backward_hooks
    entries = tuple(registry.items())

    def injected_hook(gradient):
        injected_calls.append(True)
        return gradient

    def mutate_hooks(active_model):
        registry.clear()
        tensor._backward_hooks = OrderedDict([(30000, injected_hook)])
        if outcome == "raise":
            raise _ClosureFailure("tensor hook registry failure")
        return active_model(torch.ones(2, 4)).sum()

    if outcome == "raise":
        expected_error = _ClosureFailure
        expected_message = "tensor hook registry failure"
    else:
        expected_error = RuntimeError
        expected_message = "hook"
    try:
        with pytest.raises(expected_error, match=expected_message):
            trainer.run_query_mask_training_step(
                model, optimizer, mutate_hooks, snapshot
            )

        assert tensor._backward_hooks is registry
        assert tuple(registry.items()) == entries
        original_calls[:] = []
        injected_calls[:] = []
        tensor.sum().backward()
        assert original_calls == [True]
        assert injected_calls == []
    finally:
        handle.remove()
        tensor.grad = None


class _NoLoadMutatingOptimizer(_MutatingOptimizer):
    def load_state_dict(self, state_dict):
        raise AssertionError("optimizer.load_state_dict must not be called")


def test_optimizer_float64_state_rolls_back_without_load_state_dict():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _NoLoadMutatingOptimizer(group["parameters"], model)
    first = group["parameters"][0]
    optimizer.state[first]["float64_state"] = torch.tensor(
        [1.125, -2.5], dtype=torch.float64
    )
    optimizer.state[first]["nested"] = {
        "values": [torch.tensor([7], dtype=torch.int64)]
    }
    before = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(ValueError, match="optimizer parameter set"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: active_model(torch.ones(2, 4)).sum(),
            snapshot,
        )

    _assert_nested_equal(optimizer.state_dict(), before)
    restored = optimizer.state[first]["float64_state"]
    assert restored.dtype == torch.float64
    assert restored.device == first.device


def test_optimizer_rollback_preserves_complete_object_graph_identity_and_aliases():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = _NoLoadMutatingOptimizer(group["parameters"], model)
    first = group["parameters"][0]

    state_container = optimizer.state
    groups_container = optimizer.param_groups
    defaults_container = optimizer.defaults
    group_container = groups_container[0]
    params_container = group_container["params"]
    shared_tensor = torch.tensor([1.125, -2.5], dtype=torch.float64)
    shared_before = shared_tensor.detach().clone()
    nested_list = [shared_tensor, "stable"]
    nested_tuple = (nested_list, shared_tensor)
    nested_set = {"alpha", "beta"}
    state_entry = {
        "a": shared_tensor,
        "b": shared_tensor,
        "nested": nested_list,
        "tuple": nested_tuple,
        "set": nested_set,
    }
    state_container[first] = state_entry
    defaults_container["shared"] = nested_list
    group_container["shared"] = nested_list
    group_container["tuple"] = nested_tuple

    def mutate_optimizer_graph_then_raise(_active_model):
        shared_tensor.fill_(91.0)
        nested_list[:] = [torch.tensor([-7.0])]
        nested_set.clear()
        nested_set.add("injected")
        state_entry.clear()
        state_entry["injected"] = torch.tensor([8.0])
        params_container[:] = [model.detector.weight]
        group_container.clear()
        group_container["injected"] = True
        state_container.clear()
        state_container["injected"] = {}
        defaults_container.clear()
        defaults_container["injected"] = True
        groups_container[:] = [{"params": []}]
        optimizer.state = {"replacement": {}}
        optimizer.defaults = {"replacement": True}
        optimizer.param_groups = [{"params": []}]
        raise _ClosureFailure("optimizer object graph failure")

    with pytest.raises(_ClosureFailure, match="optimizer object graph failure"):
        trainer.run_query_mask_training_step(
            model, optimizer, mutate_optimizer_graph_then_raise, snapshot
        )

    assert optimizer.state is state_container
    assert optimizer.param_groups is groups_container
    assert optimizer.defaults is defaults_container
    assert groups_container == [group_container]
    assert groups_container[0] is group_container
    assert group_container["params"] is params_container
    assert tuple(params_container) == tuple(group["parameters"])
    assert state_container[first] is state_entry
    assert state_entry["nested"] is nested_list
    assert state_entry["tuple"] is nested_tuple
    assert state_entry["set"] is nested_set
    assert defaults_container["shared"] is nested_list
    assert group_container["shared"] is nested_list
    assert group_container["tuple"] is nested_tuple
    assert state_entry["a"] is shared_tensor
    assert state_entry["b"] is shared_tensor
    assert nested_list[0] is shared_tensor
    assert nested_tuple[0] is nested_list
    assert nested_tuple[1] is shared_tensor
    assert nested_list == [shared_tensor, "stable"]
    assert nested_set == {"alpha", "beta"}
    assert shared_tensor.dtype == torch.float64
    assert torch.equal(shared_tensor, shared_before)
    assert "injected" not in state_entry
    assert "injected" not in group_container
    assert "injected" not in defaults_container


def test_transaction_snapshot_clones_only_trainable_and_optimizer_tensors(
        monkeypatch):
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    state_tensor = torch.tensor([3.25], dtype=torch.float64)
    optimizer.state[group["parameters"][0]]["state_tensor"] = state_tensor
    cloned_sources = []

    def record_clone(value):
        cloned_sources.append(value)
        return value.detach().clone()

    monkeypatch.setattr(
        trainer, "_clone_transaction_tensor", record_clone, raising=False
    )
    trainer._snapshot_training_transaction(model, optimizer)

    source_ids = {id(value) for value in cloned_sources}
    allowed_ids = {id(parameter) for parameter in group["parameters"]}
    allowed_ids.add(id(state_tensor))
    assert source_ids == allowed_ids
    assert id(model.large_frozen) not in source_ids
    expected_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in group["parameters"]
    ) + state_tensor.numel() * state_tensor.element_size()
    assert sum(
        value.numel() * value.element_size() for value in cloned_sources
    ) == expected_bytes


def test_frozen_parameter_data_reassignment_restores_original_exception_and_tensor():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    parameter = model.detector.weight
    before = parameter.detach().clone()
    before_shape = tuple(parameter.shape)
    before_dtype = parameter.dtype
    before_device = parameter.device

    def replace_data_then_raise(active_model):
        active_model.detector.weight.data = torch.ones(
            1, dtype=torch.float64, device=before_device
        )
        raise _ClosureFailure("malicious data replacement")

    with pytest.raises(_ClosureFailure, match="malicious data replacement"):
        trainer.run_query_mask_training_step(
            model, optimizer, replace_data_then_raise, snapshot
        )

    assert model.detector.weight is parameter
    assert tuple(parameter.shape) == before_shape
    assert parameter.dtype == before_dtype
    assert parameter.device == before_device
    assert torch.equal(parameter, before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_trainable_parameter_data_long_reassignment_restores_original_exception_and_tensor():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(group["parameters"], lr=0.1)
    parameter = model.x_query[0].weight
    before = parameter.detach().clone()
    before_shape = tuple(parameter.shape)
    before_dtype = parameter.dtype
    before_device = parameter.device
    before_requires_grad = parameter.requires_grad

    def replace_trainable_data_then_raise(active_model):
        active_model.x_query[0].weight.data = torch.ones(
            before_shape, dtype=torch.long, device=before_device
        )
        raise _ClosureFailure("trainable data replacement")

    with pytest.raises(_ClosureFailure, match="trainable data replacement"):
        trainer.run_query_mask_training_step(
            model, optimizer, replace_trainable_data_then_raise, snapshot
        )

    assert model.x_query[0].weight is parameter
    assert tuple(parameter.shape) == before_shape
    assert parameter.dtype == before_dtype
    assert parameter.device == before_device
    assert parameter.requires_grad is before_requires_grad
    assert torch.equal(parameter, before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())


def test_optimizer_precheck_failure_clears_all_gradients_and_restores_eval():
    trainer = _trainer()
    model = _ToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    snapshot = trainer.snapshot_query_mask_frozen_state(model)
    optimizer = torch.optim.SGD(
        group["parameters"] + (model.detector.weight,), lr=0.1
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.train()

    with pytest.raises(ValueError, match="optimizer parameter set"):
        trainer.run_query_mask_training_step(
            model,
            optimizer,
            lambda active_model: active_model(torch.ones(2, 4)).sum(),
            snapshot,
        )

    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())
