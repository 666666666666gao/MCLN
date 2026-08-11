import hashlib
import json
import math
import re
import struct

import pytest
import torch

from models import rec_source_gate
from models.rec_source_gate import (
    SOURCE_GATE_TRAINABLE_PREFIX,
    attach_full_query_targets,
    build_rec_source_gate_optimizer,
    clip_rec_source_gate_gradients,
    compute_rec_source_gate_loss,
    configure_rec_source_gate_trainability,
    set_rec_source_gate_eval_mode,
    set_rec_source_gate_train_mode,
)


_THRESHOLD_STATS = {
    "informative_rows",
    "active_violations",
    "no_positive_rows",
    "too_few_negative_rows",
    "positive_count",
    "mean_positive_cutoff_gap",
}


def _assert_detached_scalar(value):
    assert torch.is_tensor(value)
    assert value.dim() == 0
    assert torch.isfinite(value)
    assert not value.requires_grad
    assert value.grad_fn is None


def _assert_exact_stats_schema(stats):
    assert set(stats) == {
        "loss025", "loss050", "loss_total",
        "threshold025", "threshold050",
    }
    for key in ("loss025", "loss050", "loss_total"):
        _assert_detached_scalar(stats[key])
    for key in ("threshold025", "threshold050"):
        assert set(stats[key]) == _THRESHOLD_STATS
        for value in stats[key].values():
            _assert_detached_scalar(value)


def _source_gate_inputs(batch_size=1, num_queries=9):
    scores = torch.zeros(batch_size, num_queries, requires_grad=True)
    ious = torch.zeros(batch_size, num_queries)
    ious[:, 0] = 0.8
    valid = torch.ones(batch_size, num_queries, dtype=torch.bool)
    return scores, ious, valid


class _FakePredictionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sem_cls_scores_head = torch.nn.Sequential(
            torch.nn.Linear(4, 1),
            torch.nn.Dropout(p=0.0),
        )
        self.center_head = torch.nn.Linear(4, 3)
        self.size_head = torch.nn.Linear(4, 3)
        self.objectness_head = torch.nn.Linear(4, 1)
        self.mask_head = torch.nn.Linear(4, 2)
        self.projection_head = torch.nn.Linear(4, 4)
        self.source_selector = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.center_head.weight.zero_()
            self.center_head.weight[:, :3].copy_(torch.eye(3))
            self.center_head.bias.zero_()
            self.size_head.weight.zero_()
            self.size_head.bias.fill_(2.0)

    def forward(self, query_features):
        return {
            "scores": self.sem_cls_scores_head(query_features).squeeze(-1),
            "center": self.center_head(query_features),
            "size": self.size_head(query_features),
            "objectness": self.objectness_head(query_features),
            "mask": self.mask_head(query_features),
            "projection": self.projection_head(query_features),
            "source": self.source_selector(query_features),
        }


class _FakeSourceGateMCLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_net = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Dropout(p=0.4),
        )
        self.text_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Dropout(p=0.3),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Dropout(p=0.2),
        )
        self.decoder_query_proj = torch.nn.Linear(4, 4)
        self.proposal_head = _FakePredictionHead()
        self.prediction_heads = torch.nn.ModuleList([
            _FakePredictionHead() for _index in range(6)
        ])
        self.mask_projection = torch.nn.Linear(4, 4)
        self.source_selector = torch.nn.Linear(4, 2)

    def forward(self, query_features):
        return self.prediction_heads[5](query_features)


def _source_gate_models():
    return (
        _FakeSourceGateMCLN(),
        torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.Dropout(p=0.2),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(5, 2),
            torch.nn.Dropout(p=0.3),
        ),
    )


def _parameter_ids(parameters):
    return tuple(id(parameter) for parameter in parameters)


def _all_state_dict_tensors(mcln, parent, geometry):
    return tuple(
        ("{}.{}".format(model_name, tensor_name), tensor)
        for model_name, model in (
            ("mcln", mcln),
            ("parent", parent),
            ("geometry", geometry),
        )
        for tensor_name, tensor in model.state_dict().items()
    )


def _reachable_modules(root):
    modules = []
    stack = [root]
    seen = set()
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        modules.append(module)
        stack.extend(
            child for child in module._modules.values()
            if child is not None
        )
    return tuple(modules)


def _assert_all_frozen(mcln, parent, geometry):
    seen = set()
    for model in (mcln, parent, geometry):
        for module in _reachable_modules(model):
            for parameter in module._parameters.values():
                if parameter is None or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                assert not parameter.requires_grad


def _assert_all_eval(mcln, parent, geometry):
    assert all(
        module.training is False
        for model in (mcln, parent, geometry)
        for module in _reachable_modules(model)
    )


def test_source_gate_trainable_prefix_is_the_exact_production_contract():
    assert SOURCE_GATE_TRAINABLE_PREFIX == (
        "prediction_heads.5.sem_cls_scores_head."
    )


def test_configure_source_gate_returns_only_final_semantic_classifier():
    mcln, parent, geometry = _source_gate_models()
    expected = tuple(
        (name, parameter)
        for name, parameter in mcln.named_parameters()
        if name.startswith(SOURCE_GATE_TRAINABLE_PREFIX)
    )

    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )

    assert parameters.names == tuple(name for name, _ in expected)
    assert _parameter_ids(parameters.parameters) == _parameter_ids(
        tuple(parameter for _, parameter in expected)
    )
    assert parameters.mcln is mcln
    assert parameters.parent is parent
    assert parameters.geometry is geometry
    assert parameters.parameters
    assert len(set(parameters.names)) == len(parameters.names)
    assert len(set(_parameter_ids(parameters.parameters))) == len(
        parameters.parameters
    )

    selected_ids = set(_parameter_ids(parameters.parameters))
    for name, parameter in mcln.named_parameters():
        assert parameter.requires_grad is name.startswith(
            SOURCE_GATE_TRAINABLE_PREFIX
        )
        assert (id(parameter) in selected_ids) is name.startswith(
            SOURCE_GATE_TRAINABLE_PREFIX
        )
    assert all(not parameter.requires_grad for parameter in parent.parameters())
    assert all(
        not parameter.requires_grad for parameter in geometry.parameters()
    )


def test_configure_source_gate_rejects_missing_final_semantic_prefix_closed():
    mcln, parent, geometry = _source_gate_models()
    mcln.prediction_heads[5].sem_cls_scores_head = torch.nn.Dropout(p=0.0)

    with pytest.raises(ValueError, match="trainable prefix"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)


@pytest.mark.parametrize("invalid_root", ["mcln", "parent", "geometry"])
def test_configure_source_gate_invalid_root_closes_all_valid_models(
        invalid_root):
    mcln, parent, geometry = _source_gate_models()
    models = {"mcln": mcln, "parent": parent, "geometry": geometry}
    arguments = dict(models)
    arguments[invalid_root] = object()

    with pytest.raises(ValueError, match="Module"):
        configure_rec_source_gate_trainability(**arguments)

    for name, model in models.items():
        if name != invalid_root:
            assert all(
                not parameter.requires_grad
                for parameter in model.parameters()
            )
            assert all(
                module.training is False for module in model.modules()
            )


def test_configure_source_gate_rejects_cross_boundary_parameter_alias_closed():
    mcln, parent, geometry = _source_gate_models()
    shared = mcln.prediction_heads[5].sem_cls_scores_head[0].weight
    mcln.backbone_net.register_parameter("shared_semantic_weight", shared)

    with pytest.raises(ValueError, match="parameter.*allowlist boundary"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)


def test_configure_source_gate_rejects_duplicate_allowed_registration_closed():
    mcln, parent, geometry = _source_gate_models()
    classifier = mcln.prediction_heads[5].sem_cls_scores_head[0]
    classifier.register_parameter("weight_alias", classifier.weight)

    with pytest.raises(ValueError, match="duplicate"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)


@pytest.mark.parametrize("overlap_model", ["parent", "geometry"])
def test_configure_source_gate_rejects_selected_reranker_overlap_closed(
        overlap_model):
    mcln, parent, geometry = _source_gate_models()
    shared = mcln.prediction_heads[5].sem_cls_scores_head[0].weight
    target = parent if overlap_model == "parent" else geometry
    target.register_parameter("shared_semantic_weight", shared)

    with pytest.raises(ValueError, match="overlap"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)


def test_configure_rejects_selected_parameter_registered_as_parent_buffer():
    mcln, parent, geometry = _source_gate_models()
    selected = mcln.prediction_heads[5].sem_cls_scores_head[0].weight
    parent.register_buffer("selected_alias", selected)

    with pytest.raises(ValueError, match="tensor.*overlap"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)
    _assert_all_eval(mcln, parent, geometry)


@pytest.mark.parametrize("frozen_root", ["mcln", "parent"])
def test_configure_rejects_different_parameter_with_shared_selected_storage(
        frozen_root):
    mcln, parent, geometry = _source_gate_models()
    selected = mcln.prediction_heads[5].sem_cls_scores_head[0].weight
    shared_storage = torch.nn.Parameter(selected.detach())
    assert shared_storage is not selected
    assert shared_storage.storage().data_ptr() == selected.storage().data_ptr()
    if frozen_root == "mcln":
        mcln.backbone_net.register_parameter(
            "shared_selected_storage", shared_storage
        )
    else:
        parent.register_parameter("shared_selected_storage", shared_storage)

    with pytest.raises(ValueError, match="storage.*overlap"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)
    _assert_all_eval(mcln, parent, geometry)


def test_configure_rejects_shared_storage_between_selected_parameters():
    mcln, parent, geometry = _source_gate_models()
    classifier = mcln.prediction_heads[5].sem_cls_scores_head[0]
    classifier.bias = torch.nn.Parameter(
        classifier.weight.detach().reshape(-1)[:1]
    )
    assert classifier.bias.storage().data_ptr() == (
        classifier.weight.storage().data_ptr()
    )

    with pytest.raises(ValueError, match="selected.*storage.*overlap"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)
    _assert_all_eval(mcln, parent, geometry)


def test_empty_registered_tensors_do_not_create_false_storage_overlap():
    mcln, parent, geometry = _source_gate_models()
    classifier = mcln.prediction_heads[5].sem_cls_scores_head[0]
    classifier.weight = torch.nn.Parameter(torch.empty(0))
    parent.register_buffer("empty_buffer", torch.empty(0))

    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )

    assert id(classifier.weight) in _parameter_ids(parameters.parameters)


def test_unsupported_registered_tensor_storage_fails_closed():
    mcln, parent, geometry = _source_gate_models()
    parent.register_buffer(
        "sparse_buffer",
        torch.sparse_coo_tensor(
            torch.tensor([[0], [0]]),
            torch.tensor([1.0]),
            (2, 2),
        ),
    )

    with pytest.raises(ValueError, match="non-strided tensor"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)
    _assert_all_eval(mcln, parent, geometry)


@pytest.mark.parametrize(
    "operation",
    [build_rec_source_gate_optimizer, clip_rec_source_gate_gradients],
    ids=["optimizer", "clip"],
)
@pytest.mark.parametrize("alias_kind", ["buffer-identity", "shared-storage"])
def test_contract_validation_rechecks_registered_tensor_overlap(
        operation, alias_kind):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    selected = parameters.parameters[0]
    if alias_kind == "buffer-identity":
        parent.register_buffer("late_selected_alias", selected)
    else:
        parent.register_parameter(
            "late_shared_storage",
            torch.nn.Parameter(selected.detach(), requires_grad=False),
        )

    with pytest.raises(ValueError, match="tensor.*overlap|storage.*overlap"):
        operation(parameters)


def test_configure_cycle_fails_closed_without_recursive_cleanup():
    mcln, parent, geometry = _source_gate_models()
    parent.add_module("loop", parent)

    with pytest.raises(ValueError, match="cyclic module registration"):
        configure_rec_source_gate_trainability(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)
    _assert_all_eval(mcln, parent, geometry)


def test_source_gate_modes_train_only_final_semantic_classifier_subtree():
    mcln, parent, geometry = _source_gate_models()
    configure_rec_source_gate_trainability(mcln, parent, geometry)

    set_rec_source_gate_train_mode(mcln, parent, geometry)

    allowed_module = SOURCE_GATE_TRAINABLE_PREFIX[:-1]
    for name, module in mcln.named_modules():
        expected_training = (
            name == allowed_module
            or name.startswith(allowed_module + ".")
        )
        assert module.training is expected_training, name
    assert mcln.training is False
    assert mcln.prediction_heads.training is False
    assert mcln.prediction_heads[5].training is False
    assert parent.training is False
    assert geometry.training is False
    assert all(module.training is False for module in parent.modules())
    assert all(module.training is False for module in geometry.modules())

    set_rec_source_gate_eval_mode(mcln, parent, geometry)

    _assert_all_eval(mcln, parent, geometry)


def test_source_gate_train_mode_rejects_cross_boundary_module_alias_closed():
    mcln, parent, geometry = _source_gate_models()
    configure_rec_source_gate_trainability(mcln, parent, geometry)
    mcln.frozen_semantic_alias = (
        mcln.prediction_heads[5].sem_cls_scores_head
    )

    with pytest.raises(AssertionError, match="module.*train/eval boundary"):
        set_rec_source_gate_train_mode(mcln, parent, geometry)

    _assert_all_eval(mcln, parent, geometry)
    _assert_all_frozen(mcln, parent, geometry)


def test_source_gate_train_mode_cycle_fails_closed_after_configuration():
    mcln, parent, geometry = _source_gate_models()
    configure_rec_source_gate_trainability(mcln, parent, geometry)
    parent.add_module("loop", parent)

    with pytest.raises(ValueError, match="cyclic module registration"):
        set_rec_source_gate_train_mode(mcln, parent, geometry)

    _assert_all_frozen(mcln, parent, geometry)
    _assert_all_eval(mcln, parent, geometry)


@pytest.mark.parametrize("invalid_root", ["mcln", "parent", "geometry"])
def test_source_gate_train_mode_invalid_root_closes_all_valid_models(
        invalid_root):
    mcln, parent, geometry = _source_gate_models()
    configure_rec_source_gate_trainability(mcln, parent, geometry)
    models = {"mcln": mcln, "parent": parent, "geometry": geometry}
    arguments = dict(models)
    arguments[invalid_root] = object()

    with pytest.raises(ValueError, match="Module"):
        set_rec_source_gate_train_mode(**arguments)

    for name, model in models.items():
        if name != invalid_root:
            assert all(
                not parameter.requires_grad
                for parameter in model.parameters()
            )
            assert all(
                module.training is False for module in model.modules()
            )


@pytest.mark.parametrize(
    "drift",
    ["selected-frozen", "mcln-frozen-trainable", "parent-trainable",
     "geometry-trainable"],
)
def test_clip_source_gate_rejects_requires_grad_allowlist_drift(drift):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    if drift == "selected-frozen":
        parameters.parameters[0].requires_grad_(False)
    elif drift == "mcln-frozen-trainable":
        next(
            parameter for name, parameter in mcln.named_parameters()
            if not name.startswith(SOURCE_GATE_TRAINABLE_PREFIX)
        ).requires_grad_(True)
    elif drift == "parent-trainable":
        next(parent.parameters()).requires_grad_(True)
    else:
        next(geometry.parameters()).requires_grad_(True)

    with pytest.raises(RuntimeError, match="trainability"):
        clip_rec_source_gate_gradients(parameters)


@pytest.mark.parametrize("model_name", ["mcln", "parent", "geometry"])
def test_clip_source_gate_rejects_any_frozen_parameter_gradient(model_name):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    models = {"mcln": mcln, "parent": parent, "geometry": geometry}
    if model_name == "mcln":
        frozen = next(
            parameter for name, parameter in mcln.named_parameters()
            if not name.startswith(SOURCE_GATE_TRAINABLE_PREFIX)
        )
    else:
        frozen = next(models[model_name].parameters())
    frozen.grad = torch.zeros_like(frozen)

    with pytest.raises(RuntimeError, match="frozen.*gradient"):
        clip_rec_source_gate_gradients(parameters)


@pytest.mark.parametrize(
    "operation",
    [build_rec_source_gate_optimizer, clip_rec_source_gate_gradients],
    ids=["optimizer", "clip"],
)
def test_source_gate_rejects_illegal_parameter_contract_schema(operation):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    valid_dict = {
        "names": parameters.names,
        "parameters": parameters.parameters,
        "mcln": parameters.mcln,
        "parent": parameters.parent,
        "geometry": parameters.geometry,
    }
    missing = dict(valid_dict)
    missing.pop("geometry")
    extra = dict(valid_dict)
    extra["unexpected"] = object()
    names_list = dict(valid_dict)
    names_list["names"] = list(names_list["names"])
    parameter_list = dict(valid_dict)
    parameter_list["parameters"] = list(parameter_list["parameters"])
    bad_contracts = (
        object(),
        parameters.parameters,
        valid_dict,
        missing,
        extra,
        names_list,
        parameter_list,
    )

    for bad_contract in bad_contracts:
        with pytest.raises(ValueError):
            operation(bad_contract)


@pytest.mark.parametrize(
    "operation",
    [build_rec_source_gate_optimizer, clip_rec_source_gate_gradients],
    ids=["optimizer", "clip"],
)
def test_source_gate_rejects_raw_tuple_contract_schema(operation):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    raw_tuple = (parameters.parameters[0],) * 2

    with pytest.raises(ValueError):
        operation(raw_tuple)


def test_source_gate_contract_rejects_replaced_model_reference():
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    next(parent.parameters()).requires_grad_(True)
    replaced = {
        "names": parameters.names,
        "parameters": parameters.parameters,
        "mcln": parameters.mcln,
        "parent": torch.nn.Module(),
        "geometry": parameters.geometry,
    }

    with pytest.raises(ValueError):
        clip_rec_source_gate_gradients(replaced)
    with pytest.raises(AttributeError):
        parameters.parent = torch.nn.Module()


def test_source_gate_contract_rejects_object_setattr_root_tampering():
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    next(parent.parameters()).requires_grad_(True)
    object.__setattr__(parameters, "_parent", torch.nn.Module())

    with pytest.raises(ValueError, match="provenance"):
        clip_rec_source_gate_gradients(parameters)


def test_source_gate_contract_rejects_forged_exact_type_and_seal():
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    next(parent.parameters()).requires_grad_(True)
    forged = type(parameters)(
        parameters.names,
        parameters.parameters,
        parameters.mcln,
        torch.nn.Module(),
        parameters.geometry,
        parameters._seal,
    )

    with pytest.raises(ValueError, match="provenance"):
        clip_rec_source_gate_gradients(forged)


def test_source_gate_rejects_dict_schema_with_tuple_subclass():
    class TupleSubclass(tuple):
        pass

    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    replaced = {
        "names": parameters.names,
        "parameters": TupleSubclass(parameters.parameters),
        "mcln": parameters.mcln,
        "parent": parameters.parent,
        "geometry": parameters.geometry,
    }

    with pytest.raises(ValueError):
        build_rec_source_gate_optimizer(replaced)


def test_source_gate_optimizer_is_fresh_single_named_exact_adamw_group():
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )

    optimizer = build_rec_source_gate_optimizer(parameters)
    fresh_optimizer = build_rec_source_gate_optimizer(parameters)

    assert type(optimizer) is torch.optim.AdamW
    assert optimizer is not fresh_optimizer
    assert len(optimizer.state) == 0
    assert len(fresh_optimizer.state) == 0
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["name"] == "source_gate_semantic_classifier"
    assert group["lr"] == 1e-4
    assert group["weight_decay"] == 1e-4
    assert optimizer.defaults["lr"] == 1e-4
    assert optimizer.defaults["weight_decay"] == 1e-4
    assert _parameter_ids(group["params"]) == _parameter_ids(
        parameters.parameters
    )


@pytest.mark.parametrize("field", ["lr", "weight_decay"])
@pytest.mark.parametrize(
    "value",
    [True, "0.0001", float("nan"), float("inf"), -float("inf"),
     0.0, -1e-4, 1.000001e-4],
    ids=["bool", "string", "nan", "inf", "negative-inf", "zero",
         "negative", "non-exact"],
)
def test_source_gate_optimizer_rejects_non_exact_hyperparameters(field, value):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        build_rec_source_gate_optimizer(parameters, **kwargs)


@pytest.mark.parametrize(
    "max_norm",
    [True, "1.0", float("nan"), float("inf"), -float("inf"),
     0.0, -1.0, 0.999999, 1.000001],
    ids=["bool", "string", "nan", "inf", "negative-inf", "zero",
         "negative", "low", "high"],
)
def test_source_gate_clip_rejects_non_exact_max_norm(max_norm):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )

    with pytest.raises(ValueError, match="max_norm"):
        clip_rec_source_gate_gradients(parameters, max_norm=max_norm)


def test_source_gate_clip_uses_only_exact_selected_group(monkeypatch):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    calls = []

    def fake_clip(selected, max_norm):
        calls.append((_parameter_ids(selected), max_norm))
        return torch.tensor(0.75)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    diagnostics = clip_rec_source_gate_gradients(parameters)

    assert calls == [(_parameter_ids(parameters.parameters), 1.0)]
    assert diagnostics == {"source_gate_semantic_classifier": 0.75}


@pytest.mark.parametrize(
    "returned_norm", [float("nan"), float("inf"), -float("inf")]
)
def test_source_gate_clip_rejects_nonfinite_returned_norm(
        monkeypatch, returned_norm):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    monkeypatch.setattr(
        rec_source_gate.torch.nn.utils,
        "clip_grad_norm_",
        lambda _selected, _max_norm: torch.tensor(returned_norm),
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        clip_rec_source_gate_gradients(parameters)


@pytest.mark.parametrize(
    "gradient_value", [float("nan"), float("inf"), -float("inf")]
)
def test_source_gate_clip_rejects_real_nonfinite_selected_gradient(
        gradient_value):
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    selected = parameters.parameters[0]
    selected.grad = torch.full_like(selected, gradient_value)

    with pytest.raises(FloatingPointError, match="selected gradient"):
        clip_rec_source_gate_gradients(parameters)

    assert not bool(torch.isfinite(selected.grad).all().item())


def _synthetic_query_features():
    query_features = torch.zeros(1, 9, 4)
    query_features[0, :, 0] = torch.arange(9, dtype=torch.float32) * 4.0
    query_features[0, :, 3] = 1.0
    return query_features


def _tensor_sha256(value):
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_synthetic_source_gate_update_changes_only_final_semantic_head():
    torch.manual_seed(41)
    mcln, parent, geometry = _source_gate_models()
    parameters = configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    set_rec_source_gate_train_mode(mcln, parent, geometry)
    optimizer = build_rec_source_gate_optimizer(parameters)
    query_features = _synthetic_query_features()
    snapshots = {
        name: tensor.detach().clone()
        for name, tensor in _all_state_dict_tensors(mcln, parent, geometry)
    }

    before = mcln(query_features)
    end_points = {
        "center_label": before["center"][:, :1].detach().clone(),
        "size_gts": before["size"][:, :1].detach().clone(),
        "box_label_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    before_boxes = torch.cat((before["center"], before["size"]), dim=-1)
    before_ious = attach_full_query_targets(
        {"boxes": before_boxes}, end_points
    )
    before_digest = _tensor_sha256(before_ious)
    assert torch.equal(
        before_ious, torch.tensor([[1.0] + [0.0] * 8])
    )

    optimizer.zero_grad(set_to_none=True)
    loss, _stats = compute_rec_source_gate_loss(
        before["scores"],
        before_ious,
        torch.ones_like(before_ious, dtype=torch.bool),
    )
    loss.backward()
    diagnostics = clip_rec_source_gate_gradients(parameters)
    optimizer.step()

    assert math.isfinite(
        diagnostics["source_gate_semantic_classifier"]
    )
    selected_names = {
        "mcln.{}".format(name) for name in parameters.names
    }
    changed_selected = []
    after_state = dict(_all_state_dict_tensors(mcln, parent, geometry))
    assert set(after_state) == set(snapshots)
    for name, tensor in after_state.items():
        changed = not torch.equal(tensor.detach(), snapshots[name])
        if name in selected_names:
            changed_selected.append(changed)
        else:
            assert not changed, name
    assert any(changed_selected)

    after = mcln(query_features)
    assert torch.equal(after["center"], before["center"])
    assert torch.equal(after["size"], before["size"])
    after_boxes = torch.cat((after["center"], after["size"]), dim=-1)
    after_ious = attach_full_query_targets({"boxes": after_boxes}, end_points)
    assert _tensor_sha256(after_ious) == before_digest


def test_attach_full_query_targets_is_root_only_detached_and_non_mutating():
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                           [10.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                           [5.0, 0.0, 0.0, 2.0, 2.0, 2.0]]],
                         requires_grad=True)
    centers = torch.tensor([[[0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.0]]], requires_grad=True)
    sizes = torch.full((1, 2, 3), 2.0, requires_grad=True)
    full_state = {"boxes": boxes, "num_queries": 3}
    end_points = {
        "center_label": centers,
        "size_gts": sizes,
        "box_label_mask": torch.tensor([[True, True]]),
    }
    full_keys = set(full_state)
    endpoint_keys = set(end_points)
    snapshots = {
        "boxes": boxes.detach().clone(),
        "center_label": centers.detach().clone(),
        "size_gts": sizes.detach().clone(),
        "box_label_mask": end_points["box_label_mask"].clone(),
    }

    root_ious = attach_full_query_targets(full_state, end_points)
    all_ious = attach_full_query_targets(
        full_state, end_points, root_only=False
    )

    assert root_ious.shape == (1, 3)
    assert torch.allclose(root_ious, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(all_ious, torch.tensor([[1.0, 1.0, 0.0]]))
    assert not root_ious.requires_grad
    assert root_ious.grad_fn is None
    assert not all_ious.requires_grad
    assert all_ious.grad_fn is None
    assert set(full_state) == full_keys
    assert set(end_points) == endpoint_keys
    assert full_state["boxes"] is boxes
    assert end_points["center_label"] is centers
    assert end_points["size_gts"] is sizes
    assert torch.equal(boxes.detach(), snapshots["boxes"])
    assert torch.equal(centers.detach(), snapshots["center_label"])
    assert torch.equal(sizes.detach(), snapshots["size_gts"])
    assert torch.equal(
        end_points["box_label_mask"], snapshots["box_label_mask"]
    )


def test_loss_uses_strict_iou_boundaries_and_exact_detached_stats_schema():
    scores = torch.zeros(1, 12, requires_grad=True)
    ious = torch.tensor([[
        0.25, 0.25001, 0.50, 0.50001,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]])
    valid = torch.ones_like(ious, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    expected_threshold_loss = math.log(2.0)
    assert loss.item() == pytest.approx(3.0 * expected_threshold_loss)
    assert stats["loss025"].item() == pytest.approx(expected_threshold_loss)
    assert stats["loss050"].item() == pytest.approx(expected_threshold_loss)
    assert stats["loss_total"].item() == pytest.approx(loss.item())
    assert stats["threshold025"]["positive_count"].item() == 3
    assert stats["threshold050"]["positive_count"].item() == 1
    for key in ("threshold025", "threshold050"):
        assert stats[key]["informative_rows"].item() == 1
        assert stats[key]["active_violations"].item() == 1
        assert stats[key]["no_positive_rows"].item() == 0
        assert stats[key]["too_few_negative_rows"].item() == 0
        assert stats[key]["mean_positive_cutoff_gap"].item() == 0.0
    assert loss.requires_grad
    assert loss.dtype == scores.dtype
    assert loss.device == scores.device
    _assert_exact_stats_schema(stats)


def test_loss_uses_eighth_largest_negative_not_first_or_ninth():
    scores = torch.tensor([[
        0.0, 10.0, 9.0, 8.0, 7.0, 6.0,
        5.0, 4.0, 3.0, 2.0, 1.0,
    ]], requires_grad=True)
    ious = torch.tensor([[0.8] + [0.0] * 10])
    valid = torch.ones_like(ious, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)
    loss.backward()

    expected_threshold_loss = math.log1p(math.exp(3.0))
    assert loss.item() == pytest.approx(3.0 * expected_threshold_loss)
    assert stats["loss025"].item() == pytest.approx(expected_threshold_loss)
    assert stats["threshold025"][
        "mean_positive_cutoff_gap"
    ].item() == pytest.approx(-3.0)
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 8].item() > 0.0
    assert scores.grad[0, 1].item() == 0.0
    assert scores.grad[0, 9].item() == 0.0
    assert scores.grad[0, 10].item() == 0.0


def test_invalid_high_score_and_high_iou_do_not_enter_cutoff_or_positives():
    scores = torch.tensor([[
        0.0, 8.0, 7.0, 6.0, 5.0,
        4.0, 3.0, 2.0, 1.0, 1000.0,
    ]], requires_grad=True)
    ious = torch.tensor([[0.8] + [0.0] * 8 + [0.99]])
    valid = torch.tensor([[
        True, True, True, True, True,
        True, True, True, True, False,
    ]])

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)
    loss.backward()

    expected = 3.0 * math.log1p(math.exp(1.0))
    assert loss.item() == pytest.approx(expected)
    assert stats["threshold025"]["positive_count"].item() == 1
    assert stats["threshold050"]["positive_count"].item() == 1
    assert scores.grad[0, 9].item() == 0.0


def test_nonfinite_and_out_of_range_invalid_entries_are_ignored():
    scores = torch.tensor([[
        0.0, 8.0, 7.0, 6.0, 5.0,
        4.0, 3.0, 2.0, 1.0, float("nan"),
    ]], requires_grad=True)
    ious = torch.tensor([[0.8] + [0.0] * 8 + [float("inf")]])
    valid = torch.tensor([[
        True, True, True, True, True,
        True, True, True, True, False,
    ]])

    loss, _stats = compute_rec_source_gate_loss(scores, ious, valid)
    loss.backward()

    assert loss.item() == pytest.approx(
        3.0 * math.log1p(math.exp(1.0))
    )
    assert scores.grad[0, 9].item() == 0.0
    assert torch.isfinite(scores.grad).all()


def test_fewer_than_eight_negatives_and_no_positive_return_graph_zero():
    scores = torch.arange(18.0).reshape(2, 9).requires_grad_()
    ious = torch.zeros(2, 9)
    ious[0, 0] = 0.8
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, 8] = False

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert loss.grad_fn is not None
    for key in ("threshold025", "threshold050"):
        threshold_stats = stats[key]
        assert threshold_stats["informative_rows"].item() == 0
        assert threshold_stats["active_violations"].item() == 0
        assert threshold_stats["no_positive_rows"].item() == 1
        assert threshold_stats["too_few_negative_rows"].item() == 1
        assert threshold_stats["positive_count"].item() == 1
        assert torch.isfinite(
            threshold_stats["mean_positive_cutoff_gap"]
        )
        assert threshold_stats["mean_positive_cutoff_gap"].item() == 0.0
    loss.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))
    _assert_exact_stats_schema(stats)


def test_no_informative_rows_with_max_finite_scores_return_finite_zero():
    finfo_max = torch.finfo(torch.float32).max
    scores = torch.full((1, 9), finfo_max, requires_grad=True)
    ious = torch.zeros(1, 9)
    valid = torch.ones(1, 9, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    _assert_exact_stats_schema(stats)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_informative_finite_extremes_keep_loss_stats_and_gradients_finite():
    finfo_max = torch.finfo(torch.float32).max
    scores = torch.tensor(
        [[-finfo_max] + [finfo_max] * 8], requires_grad=True
    )
    ious = torch.tensor([[0.8] + [0.0] * 8])
    valid = torch.ones(1, 9, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    assert loss.dtype == scores.dtype
    assert loss.device == scores.device
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    _assert_exact_stats_schema(stats)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_extreme_row_is_averaged_in_float64_before_dtype_saturation():
    finfo_max = torch.finfo(torch.float32).max
    scores = torch.full((8, 9), finfo_max)
    scores[0, 0] = -finfo_max
    scores.requires_grad_()
    ious = torch.zeros(8, 9)
    ious[:, 0] = 0.8
    valid = torch.ones(8, 9, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    expected_threshold_loss = 0.25 * finfo_max
    assert stats["loss025"].item() == pytest.approx(
        expected_threshold_loss, rel=1e-6
    )
    assert stats["loss050"].item() == pytest.approx(
        expected_threshold_loss, rel=1e-6
    )
    assert loss.item() == pytest.approx(0.75 * finfo_max, rel=1e-6)
    _assert_exact_stats_schema(stats)

    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, 0].item() == pytest.approx(-3.0 / 8.0)
    cutoff_gradients = scores.grad[0, 1:]
    assert torch.count_nonzero(cutoff_gradients).item() == 1
    assert cutoff_gradients.sum().item() == pytest.approx(3.0 / 8.0)


def test_cpu_float16_informative_loss_runs_in_float64_working_precision():
    scores = torch.tensor([[
        0.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ]], dtype=torch.float16, requires_grad=True)
    ious = torch.tensor(
        [[0.8] + [0.0] * 8], dtype=torch.float16
    )
    valid = torch.ones(1, 9, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    expected_threshold = torch.nn.functional.softplus(
        torch.tensor(1.0, dtype=torch.float64)
    )
    expected_total = (3.0 * expected_threshold).to(scores.dtype)
    assert loss.dtype == scores.dtype
    assert loss.device == scores.device
    assert loss.item() == expected_total.item()
    assert stats["loss025"].item() == expected_threshold.to(
        scores.dtype
    ).item()
    assert stats["loss050"].item() == expected_threshold.to(
        scores.dtype
    ).item()
    assert stats["threshold025"][
        "mean_positive_cutoff_gap"
    ].item() == -1.0
    _assert_exact_stats_schema(stats)
    for key in ("loss025", "loss050", "loss_total"):
        assert stats[key].dtype == scores.dtype
        assert stats[key].device == scores.device
    for threshold_key in ("threshold025", "threshold050"):
        for value in stats[threshold_key].values():
            assert value.device == scores.device
        assert stats[threshold_key][
            "mean_positive_cutoff_gap"
        ].dtype == scores.dtype

    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 8].item() > 0.0


def test_swapping_positive_and_eighth_negative_reverses_gradients():
    base_scores = torch.tensor([[0.0, 7.0, 6.0, 5.0, 4.0,
                                 3.0, 2.0, 1.0, 0.0]])
    first_scores = base_scores.clone().requires_grad_()
    second_scores = base_scores.clone().requires_grad_()
    first_ious = torch.tensor([[0.8] + [0.0] * 8])
    second_ious = torch.tensor([[0.0] * 8 + [0.8]])
    valid = torch.ones(1, 9, dtype=torch.bool)

    first_loss, _ = compute_rec_source_gate_loss(
        first_scores, first_ious, valid
    )
    second_loss, _ = compute_rec_source_gate_loss(
        second_scores, second_ious, valid
    )
    first_loss.backward()
    second_loss.backward()

    assert first_loss.item() == pytest.approx(3.0 * math.log(2.0))
    assert second_loss.item() == pytest.approx(first_loss.item())
    assert first_scores.grad[0, 0].item() == pytest.approx(-1.5)
    assert first_scores.grad[0, 8].item() == pytest.approx(1.5)
    assert second_scores.grad[0, 0].item() == pytest.approx(1.5)
    assert second_scores.grad[0, 8].item() == pytest.approx(-1.5)


def test_informative_rows_are_equally_weighted_despite_positive_counts():
    scores = torch.tensor([
        [0.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], requires_grad=True)
    ious = torch.tensor([
        [0.8] + [0.0] * 9,
        [0.8, 0.8] + [0.0] * 8,
    ])
    valid = torch.ones_like(ious, dtype=torch.bool)

    loss, stats = compute_rec_source_gate_loss(scores, ious, valid)

    expected_per_threshold = (
        math.log1p(math.exp(2.0)) + math.log1p(math.exp(-2.0))
    ) / 2.0
    assert loss.item() == pytest.approx(3.0 * expected_per_threshold)
    for key in ("threshold025", "threshold050"):
        assert stats[key]["informative_rows"].item() == 2
        assert stats[key]["positive_count"].item() == 3
        assert stats[key]["mean_positive_cutoff_gap"].item() == pytest.approx(
            0.0
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk": 7},
        {"topk": 9},
        {"topk": 8.0},
        {"topk": True},
        {"thresholds": [0.25, 0.50]},
        {"thresholds": (0.25,)},
        {"thresholds": (0.50, 0.25)},
        {"thresholds": (0.25, 0.500001)},
        {"thresholds": (0.25, True)},
        {"thresholds": (0.25, float("nan"))},
        {"threshold_weights": [2.0, 1.0]},
        {"threshold_weights": (2.0,)},
        {"threshold_weights": (1.0, 2.0)},
        {"threshold_weights": (2.0, 1.000001)},
        {"threshold_weights": (2.0, True)},
        {"threshold_weights": (2.0, float("nan"))},
        {"margin": 0.1},
        {"margin": True},
        {"margin": float("nan")},
        {"margin": "0"},
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"temperature": 1.000001},
        {"temperature": True},
        {"temperature": float("nan")},
        {"temperature": "1"},
    ],
    ids=[
        "topk-seven", "topk-nine", "topk-float", "topk-bool",
        "threshold-list", "threshold-short", "threshold-reversed",
        "threshold-value", "threshold-bool", "threshold-nan",
        "weight-list", "weight-short", "weight-reversed", "weight-value",
        "weight-bool", "weight-nan", "margin-value", "margin-bool",
        "margin-nan", "margin-string", "temperature-zero",
        "temperature-negative", "temperature-value", "temperature-bool",
        "temperature-nan", "temperature-string",
    ],
)
def test_loss_rejects_non_exact_production_hyperparameters(kwargs):
    scores, ious, valid = _source_gate_inputs()

    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(scores, ious, valid, **kwargs)


@pytest.mark.parametrize(
    "scores,ious,valid",
    [
        (
            torch.zeros(9),
            torch.zeros(1, 9),
            torch.ones(1, 9, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 9, 1),
            torch.zeros(1, 9),
            torch.ones(1, 9, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 9),
            torch.zeros(1, 8),
            torch.ones(1, 9, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 9),
            torch.zeros(1, 9),
            torch.ones(1, 8, dtype=torch.bool),
        ),
        (
            torch.empty(0, 9),
            torch.empty(0, 9),
            torch.empty(0, 9, dtype=torch.bool),
        ),
        (
            torch.empty(1, 0),
            torch.empty(1, 0),
            torch.empty(1, 0, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 9, dtype=torch.long),
            torch.zeros(1, 9),
            torch.ones(1, 9, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 9),
            torch.zeros(1, 9, dtype=torch.long),
            torch.ones(1, 9, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 9),
            torch.zeros(1, 9),
            torch.ones(1, 9),
        ),
        (
            torch.zeros(1, 9, dtype=torch.float32),
            torch.zeros(1, 9, dtype=torch.float64),
            torch.ones(1, 9, dtype=torch.bool),
        ),
    ],
    ids=[
        "score-rank", "score-extra-axis", "iou-shape", "valid-shape",
        "empty-batch", "empty-query-axis", "score-dtype", "iou-dtype",
        "valid-dtype", "floating-dtype-mismatch",
    ],
)
def test_loss_rejects_shape_dtype_and_empty_contract_errors(
        scores, ious, valid):
    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(scores, ious, valid)


@pytest.mark.parametrize("field", ["scores", "ious", "valid"])
def test_loss_rejects_non_tensor_inputs(field):
    scores, ious, valid = _source_gate_inputs()
    values = {"scores": scores, "ious": ious, "valid": valid}
    values[field] = object()

    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(
            values["scores"], values["ious"], values["valid"]
        )


@pytest.mark.parametrize("field", ["ious", "valid"])
def test_loss_rejects_device_mismatch(field):
    scores, ious, valid = _source_gate_inputs()
    if field == "ious":
        ious = torch.empty(1, 9, device="meta")
    else:
        valid = torch.empty(1, 9, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(scores, ious, valid)


def test_loss_rejects_a_row_without_any_valid_query():
    scores, ious, valid = _source_gate_inputs(batch_size=2)
    valid[1].fill_(False)

    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(scores, ious, valid)


@pytest.mark.parametrize("field", ["scores", "ious"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_loss_rejects_nonfinite_valid_values(field, value):
    scores, ious, valid = _source_gate_inputs()
    if field == "scores":
        scores = scores.detach().clone()
        scores[0, 1] = value
        scores.requires_grad_()
    else:
        ious[0, 1] = value

    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(scores, ious, valid)


@pytest.mark.parametrize("value", [-0.000001, 1.000001])
def test_loss_rejects_valid_iou_outside_closed_unit_interval(value):
    scores, ious, valid = _source_gate_inputs()
    ious[0, 1] = value

    with pytest.raises(ValueError):
        compute_rec_source_gate_loss(scores, ious, valid)


_CALIBRATION_INDICES = (10, 3, 7, 12, 5)
_CALIBRATION_DIGEST_FORMAT = (
    "rec-source-gate-calibration-float32-sha256-v1"
)
_CALIBRATION_CURRENT = {
    "raw_query": (0.25, 0.50, 0.51, 0.90, 0.25),
    "default_top8": (0.25, 0.49, 0.51, 0.80, 0.20),
    "contrastive_top8": (0.20, 0.50, 0.40, 0.60, 0.25),
    "parent_candidate": (0.40, 0.50, 0.70, 0.20, 0.30),
    "geometry_candidate": (0.50, 0.51, 0.25, 0.80, 0.30),
    "default_top1": (0.25, 0.30, 0.50, 0.70, 0.10),
    "parent_top1": (0.25, 0.50, 0.60, 0.20, 0.30),
    "geometry_top1": (0.25, 0.51, 0.25, 0.50, 0.30),
}
_CALIBRATION_BASELINE = {
    "raw_query": (0.30, 0.50, 0.60, 0.90, 0.30),
    "default_top8": (0.30, 0.20, 0.60, 0.50, 0.30),
    "contrastive_top8": (0.30, 0.20, 0.60, 0.50, 0.10),
    "parent_candidate": (0.20, 0.60, 0.40, 0.30, 0.25),
    "geometry_candidate": (0.60, 0.40, 0.30, 0.50, 0.20),
    "default_top1": (0.30, 0.20, 0.60, 0.50, 0.10),
    "parent_top1": (0.20, 0.60, 0.25, 0.30, 0.10),
    "geometry_top1": (0.60, 0.25, 0.30, 0.50, 0.20),
}
_CALIBRATION_CURRENT_HITS = {
    "membership": {
        "default_top8": (3, 2),
        "contrastive_top8": (3, 1),
        "union_top16": (3, 2),
    },
    "candidate_oracle": {
        "raw_query": (3, 2),
        "union_query": (3, 2),
        "parent_candidate": (4, 1),
        "geometry_candidate": (4, 2),
    },
    "top1": {
        "default": (3, 1),
        "parent": (3, 1),
        "geometry": (3, 1),
    },
}
_CALIBRATION_TRANSITIONS = {
    "membership": {
        "default_top8": (1, 2, 1, 0),
        "contrastive_top8": (1, 1, 1, 1),
        "union_top16": (1, 2, 1, 0),
    },
    "candidate_oracle": {
        "raw_query": (0, 2, 0, 0),
        "union_query": (1, 2, 1, 0),
        "parent_candidate": (2, 1, 1, 1),
        "geometry_candidate": (1, 1, 2, 1),
    },
    "top1": {
        "default": (1, 1, 1, 1),
        "parent": (2, 1, 1, 1),
        "geometry": (2, 2, 1, 1),
    },
}


def _calibration_observation(branches=_CALIBRATION_CURRENT):
    sample_count = len(branches["raw_query"])
    full_query_ious = torch.zeros(sample_count, 18, dtype=torch.float32)
    full_query_ious[:, 0] = torch.tensor(branches["default_top1"])
    full_query_ious[:, 1] = torch.tensor(branches["default_top8"])
    full_query_ious[:, 8] = torch.tensor(branches["contrastive_top8"])
    full_query_ious[:, 17] = torch.tensor(branches["raw_query"])

    # Ties at Top-1 and the Top-8 boundary must prefer the lower query ID.
    default_row = torch.tensor([
        10.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0,
        1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
    ])
    contrastive_row = torch.tensor([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
        10.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, -9.0,
    ])
    default_scores = default_row.repeat(sample_count, 1)
    contrastive_scores = contrastive_row.repeat(sample_count, 1)
    compact_query_indices = torch.arange(16, dtype=torch.long).repeat(
        sample_count, 1
    )
    compact_valid_mask = torch.ones(
        sample_count, 16, dtype=torch.bool
    )

    parent_candidate_ious = torch.zeros(
        sample_count, 16, dtype=torch.float32
    )
    parent_candidate_ious[:, 0] = torch.tensor(branches["parent_top1"])
    parent_candidate_ious[:, 1] = torch.tensor(
        branches["parent_candidate"]
    )
    parent_top1_positions = torch.zeros(sample_count, dtype=torch.long)

    geometry_candidate_ious = torch.zeros(
        sample_count, 3, dtype=torch.float32
    )
    geometry_candidate_ious[:, 0] = torch.tensor(
        branches["geometry_top1"]
    )
    geometry_candidate_ious[:, 1] = torch.tensor(
        branches["geometry_candidate"]
    )
    geometry_valid_mask = torch.tensor(
        [[True, True, False]]
    ).repeat(sample_count, 1)

    return {
        "full_query_ious": full_query_ious,
        "default_scores": default_scores,
        "contrastive_scores": contrastive_scores,
        "compact_query_indices": compact_query_indices,
        "compact_valid_mask": compact_valid_mask,
        "parent_candidate_ious": parent_candidate_ious,
        "parent_valid_mask": compact_valid_mask.clone(),
        "parent_top1_positions": parent_top1_positions,
        "geometry_candidate_ious": geometry_candidate_ious,
        "geometry_valid_mask": geometry_valid_mask,
        "geometry_selected_ious": torch.tensor(
            branches["geometry_top1"], dtype=torch.float32
        ),
    }


def _slice_calibration_observation(observation, selection):
    return {
        name: value[selection].clone()
        for name, value in observation.items()
    }


def _run_calibration_accumulator(
        observation, indices=_CALIBRATION_INDICES, partitions=(2, 3),
        baseline=None):
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        indices, baseline=baseline
    )
    cursor = 0
    for batch_size in partitions:
        end = cursor + batch_size
        accumulator.update(
            indices[cursor:end],
            _slice_calibration_observation(observation, slice(cursor, end)),
        )
        cursor = end
    assert cursor == len(indices)
    return accumulator, accumulator.finalize(len(indices))


def _update_digest_frame(digest, name, payload):
    name = name if isinstance(name, bytes) else name.encode("ascii")
    digest.update(struct.pack("<Q", len(name)))
    digest.update(name)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _float32_payload(value):
    bits = value.detach().contiguous().view(torch.int32).reshape(-1).tolist()
    return b"".join(
        struct.pack("<I", int(item) & 0xFFFFFFFF) for item in bits
    )


def _expected_calibration_digest(field, indices, values):
    digest = hashlib.sha256()
    _update_digest_frame(
        digest, "schema", _CALIBRATION_DIGEST_FORMAT.encode("ascii")
    )
    _update_digest_frame(digest, "field", field.encode("ascii"))
    for dataset_index, value in zip(indices, values):
        _update_digest_frame(digest, "sample", b"")
        _update_digest_frame(
            digest, "dataset_index", str(dataset_index).encode("ascii")
        )
        _update_digest_frame(
            digest,
            "shape",
            json.dumps(list(value.shape), separators=(",", ":")).encode(
                "ascii"
            ),
        )
        _update_digest_frame(digest, "dtype", b"float32")
        _update_digest_frame(digest, "bytes", _float32_payload(value))
    return digest.hexdigest()


def _assert_json_primitives(value):
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for item in value:
            _assert_json_primitives(item)
        return
    assert type(value) is dict
    assert all(type(key) is str for key in value)
    for item in value.values():
        _assert_json_primitives(item)


def _iter_report_branches(report, section):
    for group_name, branches in report[section].items():
        for branch_name, record in branches.items():
            yield group_name, branch_name, record


def test_calibration_accumulator_exact_metrics_transitions_and_digests():
    baseline_observation = _calibration_observation(_CALIBRATION_BASELINE)
    baseline, baseline_report = _run_calibration_accumulator(
        baseline_observation, partitions=(1, 4)
    )
    current_observation = _calibration_observation()
    _accumulator, report = _run_calibration_accumulator(
        current_observation, partitions=(2, 3), baseline=baseline
    )

    assert set(report) == {
        "schema", "sample_count", "baseline_present", "metrics",
        "transitions", "digests",
    }
    assert report["schema"] == "rec-source-gate-calibration-v1"
    assert report["sample_count"] == 5
    assert report["baseline_present"] is True
    assert set(report["metrics"]) == {
        "membership", "candidate_oracle", "top1"
    }
    assert set(report["metrics"]["membership"]) == {
        "default_top8", "contrastive_top8", "union_top16"
    }
    assert set(report["metrics"]["candidate_oracle"]) == {
        "raw_query", "union_query", "parent_candidate",
        "geometry_candidate",
    }
    assert set(report["metrics"]["top1"]) == {
        "default", "parent", "geometry"
    }
    assert set(report["transitions"]) == set(report["metrics"])

    for group_name, expected_branches in _CALIBRATION_CURRENT_HITS.items():
        assert set(report["metrics"][group_name]) == set(expected_branches)
        for branch_name, (hits025, hits050) in expected_branches.items():
            metric = report["metrics"][group_name][branch_name]
            assert set(metric) == {
                "hits025", "hits050", "acc025", "acc050"
            }
            assert metric == {
                "hits025": hits025,
                "hits050": hits050,
                "acc025": hits025 / 5.0,
                "acc050": hits050 / 5.0,
            }

    for group_name, expected_branches in _CALIBRATION_TRANSITIONS.items():
        for branch_name, expected in expected_branches.items():
            transition = report["transitions"][group_name][branch_name]
            assert set(transition) == {
                "gained025", "lost025", "gained050", "lost050"
            }
            assert tuple(transition[name] for name in (
                "gained025", "lost025", "gained050", "lost050"
            )) == expected
            current_metric = report["metrics"][group_name][branch_name]
            baseline_metric = baseline_report["metrics"][group_name][
                branch_name
            ]
            for suffix in ("025", "050"):
                assert (
                    current_metric["hits" + suffix]
                    - baseline_metric["hits" + suffix]
                ) == (
                    transition["gained" + suffix]
                    - transition["lost" + suffix]
                )

    assert report["metrics"]["membership"]["union_top16"] == (
        report["metrics"]["candidate_oracle"]["union_query"]
    )
    assert set(report["digests"]) == {
        "canonical_format", "raw_query_ious_sha256",
        "geometry_selected_ious_sha256",
    }
    assert report["digests"]["canonical_format"] == (
        _CALIBRATION_DIGEST_FORMAT
    )
    assert report["digests"]["raw_query_ious_sha256"] == (
        _expected_calibration_digest(
            "raw_query_ious",
            _CALIBRATION_INDICES,
            current_observation["full_query_ious"],
        )
    )
    assert report["digests"]["geometry_selected_ious_sha256"] == (
        _expected_calibration_digest(
            "geometry_selected_ious",
            _CALIBRATION_INDICES,
            current_observation["geometry_selected_ious"],
        )
    )
    for digest_name in (
            "raw_query_ious_sha256", "geometry_selected_ious_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", report["digests"][digest_name])

    assert baseline_report["baseline_present"] is False
    assert all(
        value == 0
        for _group, _branch, transition in _iter_report_branches(
            baseline_report, "transitions"
        )
        for value in transition.values()
    )
    _assert_json_primitives(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert not any(
        type(value) is list
        for value in report.values()
    )


def test_calibration_digests_ignore_batch_partition_but_bind_order_and_bits():
    observation = _calibration_observation()
    _first, first = _run_calibration_accumulator(
        observation, partitions=(2, 3)
    )
    _second, second = _run_calibration_accumulator(
        observation, partitions=(1, 3, 1)
    )
    assert first == second

    reversed_positions = torch.arange(4, -1, -1)
    reversed_indices = tuple(reversed(_CALIBRATION_INDICES))
    reversed_observation = _slice_calibration_observation(
        observation, reversed_positions
    )
    _reordered, reordered = _run_calibration_accumulator(
        reversed_observation,
        indices=reversed_indices,
        partitions=(3, 2),
    )
    assert reordered["metrics"] == first["metrics"]
    assert reordered["digests"]["raw_query_ious_sha256"] != (
        first["digests"]["raw_query_ious_sha256"]
    )
    assert reordered["digests"]["geometry_selected_ious_sha256"] != (
        first["digests"]["geometry_selected_ious_sha256"]
    )

    raw_changed = _calibration_observation()
    raw_changed["full_query_ious"][0, 16] = torch.nextafter(
        torch.tensor(0.0), torch.tensor(1.0)
    )
    _raw_accumulator, raw_report = _run_calibration_accumulator(raw_changed)
    assert raw_report["metrics"] == first["metrics"]
    assert raw_report["digests"]["raw_query_ious_sha256"] != (
        first["digests"]["raw_query_ious_sha256"]
    )
    assert raw_report["digests"]["geometry_selected_ious_sha256"] == (
        first["digests"]["geometry_selected_ious_sha256"]
    )

    selected_changed = _calibration_observation()
    changed_value = torch.nextafter(torch.tensor(0.25), torch.tensor(0.0))
    selected_changed["geometry_selected_ious"][0] = changed_value
    selected_changed["geometry_candidate_ious"][0, 0] = changed_value
    _selected_accumulator, selected_report = _run_calibration_accumulator(
        selected_changed
    )
    assert selected_report["metrics"] == first["metrics"]
    assert selected_report["digests"]["raw_query_ious_sha256"] == (
        first["digests"]["raw_query_ious_sha256"]
    )
    assert selected_report["digests"][
        "geometry_selected_ious_sha256"
    ] != first["digests"]["geometry_selected_ious_sha256"]


@pytest.mark.parametrize(
    "expected_indices",
    [
        (),
        (1, 1),
        (1, -1),
        (1, True),
        (1, 2.0),
        {1, 2},
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([1.0, 2.0]),
    ],
    ids=[
        "empty", "duplicate", "negative", "bool", "float", "set",
        "tensor-rank", "tensor-dtype",
    ],
)
def test_calibration_constructor_rejects_invalid_expected_indices(
        expected_indices):
    with pytest.raises(ValueError):
        rec_source_gate.RecSourceGateCalibrationAccumulator(expected_indices)


@pytest.mark.parametrize(
    "batch_indices",
    [
        (10, 10),
        (3, 10),
        (10, 99),
        (10, 7),
        (),
    ],
    ids=["duplicate", "reordered", "out-of-range", "skipped", "empty"],
)
def test_calibration_update_rejects_non_exact_dataset_order(batch_indices):
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, len(batch_indices))
    )

    with pytest.raises(ValueError):
        accumulator.update(batch_indices, observation)


def test_calibration_finalize_enforces_lifecycle_and_exact_count():
    observation = _calibration_observation()
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        torch.tensor(_CALIBRATION_INDICES, dtype=torch.long)
    )
    accumulator.update(
        torch.tensor(_CALIBRATION_INDICES[:2], dtype=torch.int32),
        _slice_calibration_observation(observation, slice(0, 2)),
    )
    with pytest.raises(ValueError, match="incomplete"):
        accumulator.finalize(5)
    for invalid_count in (0, -1, 4, 6, 5.0, True):
        with pytest.raises(ValueError):
            accumulator.finalize(invalid_count)

    accumulator.update(
        _CALIBRATION_INDICES[2:],
        _slice_calibration_observation(observation, slice(2, 5)),
    )
    with pytest.raises(ValueError):
        accumulator.update(
            (99,), _slice_calibration_observation(observation, slice(0, 1))
        )
    report = accumulator.finalize(5)
    assert accumulator.finalize(5) == report
    with pytest.raises(RuntimeError, match="finalized"):
        accumulator.update(
            (99,), _slice_calibration_observation(observation, slice(0, 1))
        )


def test_calibration_baseline_must_be_finalized_complete_and_same_order():
    accumulator_type = rec_source_gate.RecSourceGateCalibrationAccumulator
    incomplete = accumulator_type(_CALIBRATION_INDICES)
    with pytest.raises(ValueError, match="finalized"):
        accumulator_type(_CALIBRATION_INDICES, baseline=incomplete)
    with pytest.raises(ValueError, match="baseline"):
        accumulator_type(_CALIBRATION_INDICES, baseline={})

    observation = _calibration_observation(_CALIBRATION_BASELINE)
    different_indices = tuple(reversed(_CALIBRATION_INDICES))
    different_observation = _slice_calibration_observation(
        observation, torch.arange(4, -1, -1)
    )
    different, _report = _run_calibration_accumulator(
        different_observation,
        indices=different_indices,
        partitions=(2, 3),
    )
    with pytest.raises(ValueError, match="expected_indices"):
        accumulator_type(_CALIBRATION_INDICES, baseline=different)


@pytest.mark.parametrize("schema_case", ["not-dict", "missing", "extra"])
def test_calibration_update_requires_exact_observation_schema(schema_case):
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 1)
    )
    if schema_case == "not-dict":
        observation = tuple(observation.values())
    elif schema_case == "missing":
        observation.pop("geometry_selected_ious")
    else:
        observation["extra"] = torch.zeros(1)

    with pytest.raises(ValueError, match="schema"):
        accumulator.update((_CALIBRATION_INDICES[0],), observation)


@pytest.mark.parametrize(
    "contract_case",
    [
        "full-rank", "default-shape", "compact-width", "parent-shape",
        "parent-top1-rank", "geometry-mask-shape", "selected-length",
        "full-float64", "all-float64", "compact-int32",
        "compact-mask-uint8", "parent-position-int32",
        "parent-mask-uint8", "geometry-mask-uint8", "device-mismatch",
    ],
)
def test_calibration_update_rejects_shape_dtype_and_device_errors(
        contract_case):
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 2)
    )
    if contract_case == "full-rank":
        observation["full_query_ious"] = observation[
            "full_query_ious"
        ][0]
    elif contract_case == "default-shape":
        observation["default_scores"] = observation["default_scores"][:, :-1]
    elif contract_case == "compact-width":
        observation["compact_query_indices"] = observation[
            "compact_query_indices"
        ][:, :-1]
        observation["compact_valid_mask"] = observation[
            "compact_valid_mask"
        ][:, :-1]
        observation["parent_candidate_ious"] = observation[
            "parent_candidate_ious"
        ][:, :-1]
        observation["parent_valid_mask"] = observation[
            "parent_valid_mask"
        ][:, :-1]
    elif contract_case == "parent-shape":
        observation["parent_candidate_ious"] = observation[
            "parent_candidate_ious"
        ][:, :-1]
    elif contract_case == "parent-top1-rank":
        observation["parent_top1_positions"] = observation[
            "parent_top1_positions"
        ].unsqueeze(1)
    elif contract_case == "geometry-mask-shape":
        observation["geometry_valid_mask"] = observation[
            "geometry_valid_mask"
        ][:, :-1]
    elif contract_case == "selected-length":
        observation["geometry_selected_ious"] = observation[
            "geometry_selected_ious"
        ][:1]
    elif contract_case == "full-float64":
        observation["full_query_ious"] = observation[
            "full_query_ious"
        ].double()
    elif contract_case == "all-float64":
        for name in (
                "full_query_ious", "default_scores", "contrastive_scores",
                "parent_candidate_ious", "geometry_candidate_ious",
                "geometry_selected_ious"):
            observation[name] = observation[name].double()
    elif contract_case == "compact-int32":
        observation["compact_query_indices"] = observation[
            "compact_query_indices"
        ].int()
    elif contract_case == "compact-mask-uint8":
        observation["compact_valid_mask"] = observation[
            "compact_valid_mask"
        ].to(torch.uint8)
    elif contract_case == "parent-position-int32":
        observation["parent_top1_positions"] = observation[
            "parent_top1_positions"
        ].int()
    elif contract_case == "parent-mask-uint8":
        observation["parent_valid_mask"] = observation[
            "parent_valid_mask"
        ].to(torch.uint8)
    elif contract_case == "geometry-mask-uint8":
        observation["geometry_valid_mask"] = observation[
            "geometry_valid_mask"
        ].to(torch.uint8)
    else:
        observation["contrastive_scores"] = torch.empty(
            observation["contrastive_scores"].shape,
            dtype=torch.float32,
            device="meta",
        )
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )

    with pytest.raises(ValueError):
        accumulator.update(_CALIBRATION_INDICES[:2], observation)


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("full_query_ious", (0, 2), float("nan")),
        ("default_scores", (0, 2), float("inf")),
        ("contrastive_scores", (0, 2), -float("inf")),
        ("parent_candidate_ious", (0, 15), float("nan")),
        ("geometry_candidate_ious", (0, 2), float("inf")),
        ("geometry_selected_ious", (0,), float("nan")),
        ("full_query_ious", (0, 2), -0.000001),
        ("parent_candidate_ious", (0, 2), 1.000001),
        ("geometry_candidate_ious", (0, 2), -0.000001),
        ("geometry_selected_ious", (0,), 1.000001),
    ],
)
def test_calibration_update_rejects_nonfinite_and_out_of_range_values(
        field, index, value):
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 1)
    )
    observation[field][index] = value
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )

    with pytest.raises(ValueError):
        accumulator.update(_CALIBRATION_INDICES[:1], observation)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "compact-out-of-range", "compact-tie-order", "compact-mask",
        "parent-mask", "geometry-empty", "parent-top1-negative",
        "parent-top1-out-of-range",
    ],
)
def test_calibration_update_rejects_invalid_indices_masks_and_top1(
        invalid_case):
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 1)
    )
    if invalid_case == "compact-out-of-range":
        observation["compact_query_indices"][0, 0] = 18
    elif invalid_case == "compact-tie-order":
        observation["compact_query_indices"][0, :2] = torch.tensor([1, 0])
    elif invalid_case == "compact-mask":
        observation["compact_valid_mask"][0, 15] = False
        observation["parent_valid_mask"][0, 15] = False
    elif invalid_case == "parent-mask":
        observation["parent_valid_mask"][0, 15] = False
    elif invalid_case == "geometry-empty":
        observation["geometry_valid_mask"].fill_(False)
    elif invalid_case == "parent-top1-negative":
        observation["parent_top1_positions"][0] = -1
    elif invalid_case == "parent-top1-out-of-range":
        observation["parent_top1_positions"][0] = 16
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )

    with pytest.raises(ValueError):
        accumulator.update(_CALIBRATION_INDICES[:1], observation)


def test_calibration_accepts_independently_computed_geometry_selected_iou():
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 1)
    )
    observation["geometry_selected_ious"][0] = 0.45
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator((10,))

    accumulator.update((10,), observation)
    report = accumulator.finalize(1)

    assert report["metrics"]["top1"]["geometry"] == {
        "hits025": 1,
        "hits050": 0,
        "acc025": 1.0,
        "acc050": 0.0,
    }


def test_calibration_parent_top1_position_must_point_to_valid_padding():
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 1)
    )
    observation["full_query_ious"] = observation["full_query_ious"][:, :10]
    observation["default_scores"] = observation["default_scores"][:, :10]
    observation["contrastive_scores"] = observation[
        "contrastive_scores"
    ][:, :10]
    observation["compact_query_indices"].zero_()
    observation["compact_query_indices"][0, :10] = torch.arange(10)
    observation["compact_valid_mask"][0, 10:] = False
    observation["parent_valid_mask"] = observation[
        "compact_valid_mask"
    ].clone()
    observation["parent_top1_positions"][0] = 12
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )

    with pytest.raises(ValueError, match="valid"):
        accumulator.update(_CALIBRATION_INDICES[:1], observation)


def test_calibration_update_does_not_mutate_or_retain_tensor_graphs():
    observation = _slice_calibration_observation(
        _calibration_observation(), slice(0, 2)
    )
    float_fields = (
        "full_query_ious", "default_scores", "contrastive_scores",
        "parent_candidate_ious", "geometry_candidate_ious",
        "geometry_selected_ious",
    )
    for name in float_fields:
        observation[name].requires_grad_()
    snapshots = {
        name: value.detach().clone() for name, value in observation.items()
    }
    accumulator = rec_source_gate.RecSourceGateCalibrationAccumulator(
        _CALIBRATION_INDICES
    )

    accumulator.update(_CALIBRATION_INDICES[:2], observation)

    for name, value in observation.items():
        assert torch.equal(value.detach(), snapshots[name])
        if name in float_fields:
            assert value.requires_grad
            assert value.grad is None

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if type(value) is dict:
            return any(contains_tensor(item) for item in value.values())
        if type(value) in (list, tuple):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(vars(accumulator))
