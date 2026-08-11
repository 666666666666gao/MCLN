#!/usr/bin/env python
"""Deterministic query-mask-only training safeguards for MCLN."""

import math
from numbers import Real

import torch


X_QUERY_PREFIX = "x_query."
FROZEN_SNAPSHOT_SCHEMA = "mcln-query-mask-frozen-state-v1"


class QueryMaskRollbackError(RuntimeError):
    """Report both a failed training step and its failed rollback."""

    def __init__(self, original_error, rollback_error):
        self.original_error = original_error
        self.rollback_error = rollback_error
        super(QueryMaskRollbackError, self).__init__(
            "query-mask training step failed with {}: {}; rollback failed "
            "with {}: {}".format(
                type(original_error).__name__,
                original_error,
                type(rollback_error).__name__,
                rollback_error,
            )
        )


class QueryMaskCheckpointRollbackError(RuntimeError):
    """Report a checkpoint load failure and every failed rollback action."""

    def __init__(self, original_error, rollback_errors):
        rollback_errors = tuple(rollback_errors)
        if not rollback_errors:
            raise ValueError("checkpoint rollback errors must not be empty")
        self.original_error = original_error
        self.rollback_errors = rollback_errors
        rollback_summary = "; ".join(
            "{}={}({})".format(action, type(error).__name__, error)
            for action, error in rollback_errors
        )
        super(QueryMaskCheckpointRollbackError, self).__init__(
            "query-mask checkpoint load failed with {}: {}; rollback "
            "failures: {}".format(
                type(original_error).__name__,
                original_error,
                rollback_summary,
            )
        )


_BITWISE_VIEW_DTYPES = {
    torch.bool: torch.uint8,
    torch.uint8: torch.uint8,
    torch.int8: torch.uint8,
    torch.int16: torch.int16,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.int32: torch.int32,
    torch.float32: torch.int32,
    torch.int64: torch.int64,
    torch.float64: torch.int64,
    torch.complex64: torch.int64,
}


def _freeze_model(model):
    if not isinstance(model, torch.nn.Module):
        raise TypeError("query-mask training requires a torch.nn.Module")
    model.requires_grad_(False)
    model.eval()


def _named_parameters_with_all_paths(module, prefix="", ancestors=()):
    """Yield parameter paths without relying on torch's deduplication API."""
    module_id = id(module)
    if module_id in ancestors:
        raise ValueError("model module graph contains a cycle")
    next_ancestors = ancestors + (module_id,)
    for name, parameter in module._parameters.items():
        if parameter is not None:
            yield prefix + name, parameter
    for name, child in module._modules.items():
        if child is not None:
            for path, parameter in _named_parameters_with_all_paths(
                    child, prefix + name + ".", next_ancestors):
                yield path, parameter


def _named_buffers_with_all_paths(module, prefix="", ancestors=()):
    """Yield every registered buffer path, including nonpersistent buffers."""
    module_id = id(module)
    if module_id in ancestors:
        raise ValueError("model module graph contains a cycle")
    next_ancestors = ancestors + (module_id,)
    for name, buffer in module._buffers.items():
        yield prefix + name, buffer
    for name, child in module._modules.items():
        if child is not None:
            for path, buffer in _named_buffers_with_all_paths(
                    child, prefix + name + ".", next_ancestors):
                yield path, buffer


def _x_query_inventory(model):
    x_query = model._modules.get("x_query")
    if not isinstance(x_query, torch.nn.Module):
        raise ValueError(
            "model must contain a registered x_query torch.nn.Module"
        )
    local_paths = tuple(_named_parameters_with_all_paths(x_query))
    local_parameters = tuple(parameter for _name, parameter in local_paths)
    if not local_parameters:
        raise ValueError("model.x_query must contain at least one parameter")
    if any(parameter.numel() <= 0 for parameter in local_parameters):
        raise ValueError("model.x_query parameters must be non-empty")
    local_ids = {id(parameter) for parameter in local_parameters}
    if len(local_ids) != len(local_parameters):
        raise ValueError(
            "x_query parameters must have unique exact x_query.* names"
        )

    all_paths = tuple(_named_parameters_with_all_paths(model))
    selected = tuple(
        (name, parameter)
        for name, parameter in all_paths
        if name.startswith(X_QUERY_PREFIX)
    )
    if not selected:
        raise ValueError("model has no parameters named x_query.*")
    selected_ids = {id(parameter) for _name, parameter in selected}
    if (len(local_ids) != len(local_parameters)
            or selected_ids != local_ids
            or len(selected_ids) != len(selected)):
        raise ValueError(
            "x_query parameters must have unique exact x_query.* names"
        )

    for name, parameter in all_paths:
        if id(parameter) in selected_ids and not name.startswith(
                X_QUERY_PREFIX):
            raise ValueError(
                "x_query parameter is shared outside the x_query.* namespace"
            )
    return selected


def configure_query_mask_head_trainability(model):
    """Freeze an MCLN-like model and enable only exact x_query parameters."""
    _freeze_model(model)
    try:
        selected = _x_query_inventory(model)
        for _name, parameter in selected:
            parameter.requires_grad_(True)

        trainable = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if _parameter_signature(trainable) != _parameter_signature(selected):
            raise RuntimeError(
                "query-mask trainable parameter set differs from exact "
                "x_query.*"
            )
    except Exception:
        model.requires_grad_(False)
        model.eval()
        raise
    model.eval()
    return {
        "names": tuple(name for name, _parameter in trainable),
        "parameters": tuple(parameter for _name, parameter in trainable),
    }


def _parameter_signature(named_parameters):
    return tuple(
        (name, id(parameter)) for name, parameter in named_parameters
    )


def _validated_trainable_inventory(model):
    if not isinstance(model, torch.nn.Module):
        raise TypeError("query-mask training requires a torch.nn.Module")
    selected = _x_query_inventory(model)
    actual = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    expected_signature = _parameter_signature(selected)
    actual_signature = _parameter_signature(actual)
    if actual_signature != expected_signature:
        raise RuntimeError(
            "trainable parameter set drift: only exact x_query.* "
            "parameters may require gradients"
        )
    if any(module.training for module in model.modules()):
        raise RuntimeError(
            "query-mask training requires the model and all submodules "
            "to remain in eval mode"
        )
    return selected


def _bitwise_tensor(value, label):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(label))
    if value.layout != torch.strided:
        raise TypeError("{} must use strided tensor storage".format(label))
    view_dtype = _BITWISE_VIEW_DTYPES.get(value.dtype)
    if view_dtype is None:
        raise TypeError(
            "{} has unsupported dtype {} for bitwise verification".format(
                label, value.dtype
            )
        )
    detached = value.detach()
    bitwise = detached.view(view_dtype)
    if (bitwise.element_size() != detached.element_size()
            or bitwise.data_ptr() != detached.data_ptr()
            or bitwise.storage_offset() != detached.storage_offset()
            or bitwise.stride() != detached.stride()):
        raise RuntimeError(
            "{} cannot be viewed bitwise without materialization".format(
                label
            )
        )
    return bitwise


def _clone_state_tensor(value, label):
    _bitwise_tensor(value, label)
    return value.detach().clone()


def _tensor_is_bitwise_equal(actual, expected, label):
    if (not isinstance(actual, torch.Tensor)
            or not isinstance(expected, torch.Tensor)
            or actual.dtype != expected.dtype
            or actual.device != expected.device
            or actual.layout != expected.layout
            or tuple(actual.shape) != tuple(expected.shape)):
        return False
    return bool(torch.equal(
        _bitwise_tensor(actual, label),
        _bitwise_tensor(expected, label),
    ))


def _tensor_storage_identity(value, label):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError("{} must use strided tensor storage".format(label))
    storage_pointer = value.detach().storage().data_ptr()
    if value.numel() > 0 and storage_pointer == 0:
        raise TypeError("{} has no addressable tensor storage".format(label))
    return (value.device.type, value.device.index, storage_pointer)


def _tensor_data_signature(value, label):
    return (
        value.dtype,
        value.device,
        value.layout,
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        _tensor_storage_identity(value, label),
    )


def _tensor_data_reference(value, label):
    _tensor_data_signature(value, label)
    return value.detach()


def _reject_trainable_frozen_storage_aliases(model, selected):
    selected_ids = {id(parameter) for _name, parameter in selected}
    trainable_storage = {
        _tensor_storage_identity(
            parameter, "trainable parameter {}".format(name)
        ): name
        for name, parameter in selected
    }
    frozen = tuple(
        (name, parameter)
        for name, parameter in _named_parameters_with_all_paths(model)
        if id(parameter) not in selected_ids
    ) + tuple(
        (name, buffer)
        for name, buffer in _named_buffers_with_all_paths(model)
        if buffer is not None
    )
    for name, tensor in frozen:
        storage_identity = _tensor_storage_identity(
            tensor, "frozen tensor {}".format(name)
        )
        if storage_identity in trainable_storage:
            raise ValueError(
                "trainable parameter {} has a storage alias with frozen "
                "tensor {}".format(trainable_storage[storage_identity], name)
            )

    non_x_query_storage = {
        _tensor_storage_identity(
            tensor, "non-x_query tensor {}".format(name)
        ): name
        for name, tensor in (
            tuple(_named_parameters_with_all_paths(model))
            + tuple(
                (buffer_name, buffer)
                for buffer_name, buffer in _named_buffers_with_all_paths(model)
                if buffer is not None
            )
        )
        if not name.startswith(X_QUERY_PREFIX)
    }
    for name, buffer in _named_buffers_with_all_paths(model):
        if buffer is None or not name.startswith(X_QUERY_PREFIX):
            continue
        storage_identity = _tensor_storage_identity(
            buffer, "x_query buffer {}".format(name)
        )
        if storage_identity in non_x_query_storage:
            raise ValueError(
                "x_query buffer {} has a storage alias with non-x_query "
                "tensor {}".format(
                    name, non_x_query_storage[storage_identity]
                )
            )


def snapshot_query_mask_frozen_state(model):
    """Clone frozen persistent state and every buffer on their live devices."""
    selected = _validated_trainable_inventory(model)
    _reject_trainable_frozen_storage_aliases(model, selected)
    trainable_names = tuple(name for name, _parameter in selected)
    state = model.state_dict()
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError("model state_dict contains an invalid name")
    missing = set(trainable_names).difference(state)
    if missing:
        raise ValueError(
            "x_query trainable parameters are missing from state_dict: {}"
            .format(", ".join(sorted(missing)))
        )
    frozen_state = {
        name: _clone_state_tensor(value, "state_dict[{}]".format(name))
        for name, value in state.items()
        if name not in trainable_names
    }
    registered_state = dict(_named_parameters_with_all_paths(model))
    registered_state.update(dict(_named_buffers_with_all_paths(model)))
    frozen_data = {}
    frozen_data_signatures = {}
    for name in frozen_state:
        tensor = registered_state.get(name)
        if tensor is None:
            frozen_data[name] = None
            frozen_data_signatures[name] = None
        else:
            label = "frozen state {}".format(name)
            frozen_data[name] = _tensor_data_reference(tensor, label)
            frozen_data_signatures[name] = _tensor_data_signature(
                tensor, label
            )
    buffers = tuple(_named_buffers_with_all_paths(model))
    buffer_signature = tuple(
        (name, None if buffer is None else id(buffer))
        for name, buffer in buffers
    )
    frozen_buffers = {}
    frozen_buffer_data = {}
    frozen_buffer_data_signatures = {}
    for name, buffer in buffers:
        if buffer is None:
            frozen_buffers[name] = None
            frozen_buffer_data[name] = None
            frozen_buffer_data_signatures[name] = None
        elif name in frozen_state:
            frozen_buffers[name] = frozen_state[name]
            frozen_buffer_data[name] = frozen_data[name]
            frozen_buffer_data_signatures[name] = frozen_data_signatures[name]
        else:
            label = "registered buffer {}".format(name)
            frozen_buffers[name] = _clone_state_tensor(
                buffer, label
            )
            frozen_buffer_data[name] = _tensor_data_reference(buffer, label)
            frozen_buffer_data_signatures[name] = _tensor_data_signature(
                buffer, label
            )
    return {
        "schema": FROZEN_SNAPSHOT_SCHEMA,
        "state_names": tuple(state.keys()),
        "parameter_signature": _parameter_signature(
            tuple(model.named_parameters())
        ),
        "trainable_signature": _parameter_signature(selected),
        "frozen_state": frozen_state,
        "frozen_data": frozen_data,
        "frozen_data_signatures": frozen_data_signatures,
        "buffer_signature": buffer_signature,
        "frozen_buffers": frozen_buffers,
        "frozen_buffer_data": frozen_buffer_data,
        "frozen_buffer_data_signatures": frozen_buffer_data_signatures,
    }


def _validate_snapshot(snapshot):
    required = {
        "schema",
        "state_names",
        "parameter_signature",
        "trainable_signature",
        "frozen_state",
        "frozen_data",
        "frozen_data_signatures",
        "buffer_signature",
        "frozen_buffers",
        "frozen_buffer_data",
        "frozen_buffer_data_signatures",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("query-mask frozen snapshot fields are invalid")
    if snapshot["schema"] != FROZEN_SNAPSHOT_SCHEMA:
        raise ValueError("query-mask frozen snapshot schema is invalid")
    if (not isinstance(snapshot["state_names"], tuple)
            or not isinstance(snapshot["parameter_signature"], tuple)
            or not isinstance(snapshot["trainable_signature"], tuple)
            or not isinstance(snapshot["frozen_state"], dict)
            or not isinstance(snapshot["frozen_data"], dict)
            or not isinstance(snapshot["frozen_data_signatures"], dict)
            or not isinstance(snapshot["buffer_signature"], tuple)
            or not isinstance(snapshot["frozen_buffers"], dict)
            or not isinstance(snapshot["frozen_buffer_data"], dict)
            or not isinstance(
                snapshot["frozen_buffer_data_signatures"], dict
            )):
        raise ValueError("query-mask frozen snapshot inventory is invalid")
    return snapshot


def verify_query_mask_frozen_state(model, snapshot):
    """Fail if trainability drifts or any frozen state bit changes."""
    snapshot = _validate_snapshot(snapshot)
    selected = _validated_trainable_inventory(model)
    if _parameter_signature(selected) != snapshot["trainable_signature"]:
        raise RuntimeError("query-mask trainable parameter identity drift")
    _reject_trainable_frozen_storage_aliases(model, selected)
    current_parameter_signature = _parameter_signature(
        tuple(model.named_parameters())
    )
    if current_parameter_signature != snapshot["parameter_signature"]:
        raise RuntimeError("model parameter collection drift")

    state = model.state_dict()
    if tuple(state.keys()) != snapshot["state_names"]:
        raise RuntimeError("model state_dict collection or order drift")
    trainable_names = {
        name for name, _identity in snapshot["trainable_signature"]
    }
    expected_frozen_names = tuple(
        name for name in snapshot["state_names"]
        if name not in trainable_names
    )
    if tuple(snapshot["frozen_state"].keys()) != expected_frozen_names:
        raise ValueError("query-mask frozen snapshot coverage is invalid")
    if (tuple(snapshot["frozen_data"].keys()) != expected_frozen_names
            or tuple(snapshot["frozen_data_signatures"].keys())
            != expected_frozen_names):
        raise ValueError("query-mask frozen data snapshot coverage is invalid")
    registered_state = dict(_named_parameters_with_all_paths(model))
    registered_state.update(dict(_named_buffers_with_all_paths(model)))
    for name, expected in snapshot["frozen_state"].items():
        expected_signature = snapshot["frozen_data_signatures"][name]
        if expected_signature is not None:
            actual_tensor = registered_state.get(name)
            if (actual_tensor is None
                    or _tensor_data_signature(
                        actual_tensor, "frozen state {}".format(name)
                    ) != expected_signature):
                raise RuntimeError(
                    "frozen state tensor {} storage identity drift".format(
                        name
                    )
                )
        if not _tensor_is_bitwise_equal(
                state[name], expected, "state_dict[{}]".format(name)):
            raise RuntimeError(
                "frozen state tensor {} changed bitwise".format(name)
            )
    buffers = tuple(_named_buffers_with_all_paths(model))
    buffer_signature = tuple(
        (name, None if buffer is None else id(buffer))
        for name, buffer in buffers
    )
    if buffer_signature != snapshot["buffer_signature"]:
        raise RuntimeError("registered buffer collection or identity drift")
    if tuple(snapshot["frozen_buffers"].keys()) != tuple(
            name for name, _buffer in buffers):
        raise ValueError("query-mask frozen buffer snapshot coverage is invalid")
    expected_buffer_names = tuple(name for name, _buffer in buffers)
    if (tuple(snapshot["frozen_buffer_data"].keys())
            != expected_buffer_names
            or tuple(snapshot["frozen_buffer_data_signatures"].keys())
            != expected_buffer_names):
        raise ValueError("query-mask frozen buffer data coverage is invalid")
    for name, buffer in buffers:
        expected = snapshot["frozen_buffers"][name]
        if buffer is None:
            if expected is not None:
                raise RuntimeError("registered buffer {} changed".format(name))
        else:
            label = "registered buffer {}".format(name)
            if (_tensor_data_signature(buffer, label)
                    != snapshot["frozen_buffer_data_signatures"][name]):
                raise RuntimeError(
                    "registered buffer {} storage identity drift".format(name)
                )
            if expected is None or not _tensor_is_bitwise_equal(
                    buffer, expected, label):
                raise RuntimeError(
                    "registered buffer {} changed bitwise".format(name)
                )
    return True


def _gradient_values(gradient):
    if gradient.is_sparse:
        return gradient.coalesce().values()
    return gradient


def validate_query_mask_gradients(model):
    """Require finite gradients for every x_query parameter and no others."""
    selected = _validated_trainable_inventory(model)
    frozen_with_gradients = tuple(
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    )
    if frozen_with_gradients:
        raise RuntimeError(
            "frozen parameter has a gradient: {}".format(
                ", ".join(frozen_with_gradients)
            )
        )

    element_count = 0
    for name, parameter in selected:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError("x_query parameter {} has no gradient".format(name))
        if (not isinstance(gradient, torch.Tensor)
                or tuple(gradient.shape) != tuple(parameter.shape)
                or gradient.device != parameter.device):
            raise RuntimeError(
                "x_query parameter {} has an invalid gradient".format(name)
            )
        values = _gradient_values(gradient)
        if values.numel() <= 0:
            raise RuntimeError(
                "x_query parameter {} has an empty gradient".format(name)
            )
        if not bool(torch.isfinite(values.detach()).all().item()):
            raise FloatingPointError(
                "x_query parameter {} has a non-finite gradient".format(name)
            )
        element_count += int(values.numel())
    return {
        "gradient_tensor_count": len(selected),
        "gradient_element_count": element_count,
    }


def _validated_max_grad_norm(max_grad_norm):
    message = "max_grad_norm must be a finite positive real number"
    if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, Real):
        raise TypeError(message)
    value = float(max_grad_norm)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(message)
    return value


def _query_mask_gradient_l2_norm(selected):
    total = 0.0
    for _name, parameter in selected:
        values = _gradient_values(parameter.grad).detach().abs()
        tensor_norm = float(torch.norm(values.double(), p=2).item())
        total = math.hypot(total, tensor_norm)
    return total


def _validate_optimizer_parameters(optimizer, selected):
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("query-mask optimizer must be a torch optimizer")
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    if (any(not isinstance(parameter, torch.nn.Parameter)
            for parameter in optimizer_parameters)
            or len({id(parameter) for parameter in optimizer_parameters})
            != len(optimizer_parameters)
            or {id(parameter) for parameter in optimizer_parameters}
            != {id(parameter) for _name, parameter in selected}):
        raise ValueError(
            "optimizer parameter set must equal exact x_query.* parameters"
        )


def _validated_scalar_loss(loss):
    if (not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not torch.is_floating_point(loss)
            or not loss.requires_grad):
        raise ValueError(
            "query-mask loss must be a differentiable floating scalar tensor"
        )
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("query-mask loss is non-finite")
    return float(loss.detach().item())


def _clone_transaction_tensor(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("transaction value must be a tensor")
    return value.detach().clone()


def _clone_trainable_parameters(selected):
    return {
        name: _clone_transaction_tensor(parameter)
        for name, parameter in selected
    }


def _validate_trainable_parameters_finite(selected):
    for name, parameter in selected:
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise FloatingPointError(
                "x_query parameter {} became non-finite".format(name)
            )


def _identity_mapping_matches(mapping, entries):
    if tuple(mapping.keys()) != tuple(name for name, _value in entries):
        return False
    return all(mapping[name] is value for name, value in entries)


_MODULE_HOOK_REGISTRY_NAMES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


def _snapshot_hook_attribute(owner, name, allow_value=False):
    if not hasattr(owner, name):
        return {"name": name, "kind": "absent"}
    value = getattr(owner, name)
    if isinstance(value, dict):
        return {
            "name": name,
            "kind": "mapping",
            "container": value,
            "entries": tuple(value.items()),
        }
    if isinstance(value, set):
        return {
            "name": name,
            "kind": "set",
            "container": value,
            "entries": tuple(value),
        }
    if value is None or allow_value:
        return {"name": name, "kind": "value", "value": value}
    raise TypeError(
        "hook registry {} has unsupported type {}".format(
            name, type(value).__name__
        )
    )


def _validate_hook_attribute(owner, record, label):
    name = record["name"]
    kind = record["kind"]
    if kind == "absent":
        matches = not hasattr(owner, name)
    elif not hasattr(owner, name):
        matches = False
    elif kind == "mapping":
        current = getattr(owner, name)
        matches = (
            current is record["container"]
            and _identity_mapping_matches(current, record["entries"])
        )
    elif kind == "set":
        current = getattr(owner, name)
        matches = (
            current is record["container"]
            and frozenset(current) == frozenset(record["entries"])
        )
    elif kind == "value":
        matches = getattr(owner, name) is record["value"]
    else:
        raise RuntimeError("invalid hook registry snapshot")
    if not matches:
        raise RuntimeError(
            "hook registry drift at {}.{}".format(label, name)
        )


def _restore_hook_attribute(owner, record):
    name = record["name"]
    kind = record["kind"]
    if kind == "absent":
        if name in getattr(owner, "__dict__", {}):
            del owner.__dict__[name]
    elif kind == "mapping":
        container = record["container"]
        container.clear()
        container.update(record["entries"])
        setattr(owner, name, container)
    elif kind == "set":
        container = record["container"]
        container.clear()
        container.update(record["entries"])
        setattr(owner, name, container)
    elif kind == "value":
        setattr(owner, name, record["value"])
    else:
        raise RuntimeError("invalid hook registry snapshot")


def _snapshot_tensor_hook_registries(model):
    records = []
    seen = set()
    registered = tuple(_named_parameters_with_all_paths(model)) + tuple(
        _named_buffers_with_all_paths(model)
    )
    for name, tensor in registered:
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        records.append({
            "name": name,
            "tensor": tensor,
            "hook": _snapshot_hook_attribute(tensor, "_backward_hooks"),
        })
    return tuple(records)


def _snapshot_module_registries(model):
    named_modules = tuple(model.named_modules())
    records = []
    for name, module in named_modules:
        records.append({
            "name": name,
            "module": module,
            "parameters_container": module._parameters,
            "parameters": tuple(module._parameters.items()),
            "buffers_container": module._buffers,
            "buffers": tuple(module._buffers.items()),
            "modules_container": module._modules,
            "modules": tuple(module._modules.items()),
            "nonpersistent_container": module._non_persistent_buffers_set,
            "nonpersistent": frozenset(module._non_persistent_buffers_set),
            "hook_registries": tuple(
                _snapshot_hook_attribute(module, name)
                for name in _MODULE_HOOK_REGISTRY_NAMES
            ),
            "full_backward_hook": _snapshot_hook_attribute(
                module, "_is_full_backward_hook", allow_value=True
            ),
        })
    return {
        "module_signature": tuple(
            (name, id(module)) for name, module in named_modules
        ),
        "records": tuple(records),
        "tensor_hooks": _snapshot_tensor_hook_registries(model),
    }


def _validate_module_registries(model, snapshot):
    current_signature = tuple(
        (name, id(module)) for name, module in model.named_modules()
    )
    if current_signature != snapshot["module_signature"]:
        raise RuntimeError("model module registry structure drift")
    for record in snapshot["records"]:
        module = record["module"]
        label = record["name"] or "<root>"
        if (module._parameters is not record["parameters_container"]
                or not _identity_mapping_matches(
                    module._parameters, record["parameters"]
                )):
            raise RuntimeError(
                "module registry parameter drift at {}".format(label)
            )
        if (module._buffers is not record["buffers_container"]
                or not _identity_mapping_matches(
                    module._buffers, record["buffers"]
                )):
            raise RuntimeError(
                "module registry buffer drift at {}".format(label)
            )
        if (module._modules is not record["modules_container"]
                or not _identity_mapping_matches(
                    module._modules, record["modules"]
                )):
            raise RuntimeError(
                "module registry child drift at {}".format(label)
            )
        if (module._non_persistent_buffers_set
                is not record["nonpersistent_container"]
                or frozenset(module._non_persistent_buffers_set)
                != record["nonpersistent"]):
            raise RuntimeError(
                "module registry nonpersistent-buffer drift at {}".format(
                    label
                )
            )
        for hook_record in record["hook_registries"]:
            _validate_hook_attribute(module, hook_record, label)
        _validate_hook_attribute(
            module, record["full_backward_hook"], label
        )
    for record in snapshot["tensor_hooks"]:
        _validate_hook_attribute(
            record["tensor"], record["hook"], record["name"]
        )


def _restore_module_registries(snapshot):
    for record in snapshot["records"]:
        module = record["module"]
        parameters = record["parameters_container"]
        parameters.clear()
        parameters.update(record["parameters"])
        module.__dict__["_parameters"] = parameters

        buffers = record["buffers_container"]
        buffers.clear()
        buffers.update(record["buffers"])
        module.__dict__["_buffers"] = buffers

        modules = record["modules_container"]
        modules.clear()
        modules.update(record["modules"])
        module.__dict__["_modules"] = modules

        nonpersistent = record["nonpersistent_container"]
        nonpersistent.clear()
        nonpersistent.update(record["nonpersistent"])
        module.__dict__["_non_persistent_buffers_set"] = nonpersistent
    for record in snapshot["records"]:
        module = record["module"]
        for hook_record in record["hook_registries"]:
            _restore_hook_attribute(module, hook_record)
        _restore_hook_attribute(module, record["full_backward_hook"])
    for record in snapshot["tensor_hooks"]:
        _restore_hook_attribute(record["tensor"], record["hook"])


def _optimizer_atomic_value(value):
    atomic_types = (
        bool,
        bytes,
        complex,
        float,
        int,
        range,
        slice,
        str,
        torch.device,
        torch.dtype,
        torch.layout,
        torch.memory_format,
    )
    return value is None or isinstance(value, atomic_types)


def _snapshot_optimizer_graph_node(value, memo, parameter_ids):
    if _optimizer_atomic_value(value):
        return {"kind": "reference", "object": value}

    identity = id(value)
    if identity in memo:
        return memo[identity]

    if isinstance(value, torch.nn.Parameter) and identity in parameter_ids:
        node = {"kind": "reference", "object": value}
        memo[identity] = node
        return node
    if isinstance(value, torch.Tensor):
        label = "optimizer state tensor"
        node = {
            "kind": "tensor",
            "object": value,
            "requires_grad": bool(value.requires_grad),
            "data": _tensor_data_reference(value, label),
            "data_signature": _tensor_data_signature(value, label),
        }
        memo[identity] = node
        node["value"] = _clone_transaction_tensor(value)
        return node
    if isinstance(value, dict):
        node = {
            "kind": "dict",
            "object": value,
            "default_factory": getattr(value, "default_factory", None),
            "items": None,
        }
        memo[identity] = node
        node["items"] = tuple(
            (
                _snapshot_optimizer_graph_node(key, memo, parameter_ids),
                _snapshot_optimizer_graph_node(nested, memo, parameter_ids),
            )
            for key, nested in value.items()
        )
        return node
    if isinstance(value, list):
        node = {"kind": "list", "object": value, "items": None}
        memo[identity] = node
        node["items"] = tuple(
            _snapshot_optimizer_graph_node(item, memo, parameter_ids)
            for item in value
        )
        return node
    if isinstance(value, tuple):
        node = {"kind": "tuple", "object": value, "items": None}
        memo[identity] = node
        node["items"] = tuple(
            _snapshot_optimizer_graph_node(item, memo, parameter_ids)
            for item in value
        )
        return node
    if isinstance(value, set):
        node = {"kind": "set", "object": value, "items": None}
        memo[identity] = node
        node["items"] = tuple(
            _snapshot_optimizer_graph_node(item, memo, parameter_ids)
            for item in value
        )
        return node
    if isinstance(value, frozenset):
        node = {"kind": "frozenset", "object": value, "items": None}
        memo[identity] = node
        node["items"] = tuple(
            _snapshot_optimizer_graph_node(item, memo, parameter_ids)
            for item in value
        )
        return node
    raise TypeError(
        "unsupported optimizer transaction value type {}".format(
            type(value).__name__
        )
    )


def _snapshot_optimizer_graph(optimizer):
    parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }
    memo = {}
    return {
        "state": _snapshot_optimizer_graph_node(
            optimizer.state, memo, parameter_ids
        ),
        "defaults": _snapshot_optimizer_graph_node(
            optimizer.defaults, memo, parameter_ids
        ),
        "param_groups": _snapshot_optimizer_graph_node(
            optimizer.param_groups, memo, parameter_ids
        ),
    }


def _snapshot_training_transaction(model, optimizer):
    selected = _validated_trainable_inventory(model)
    return {
        "trainable_state": _clone_trainable_parameters(selected),
        "trainable_data": {
            name: _tensor_data_reference(
                parameter, "trainable parameter {}".format(name)
            )
            for name, parameter in selected
        },
        "trainable_data_signatures": {
            name: _tensor_data_signature(
                parameter, "trainable parameter {}".format(name)
            )
            for name, parameter in selected
        },
        "parameters": tuple(
            (name, parameter, bool(parameter.requires_grad))
            for name, parameter in model.named_parameters()
        ),
        "module_registries": _snapshot_module_registries(model),
        "optimizer_graph": _snapshot_optimizer_graph(optimizer),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() else None
        ),
    }


def _clear_model_gradients(model, parameter_records=()):
    seen = set()
    for parameter in tuple(model.parameters()) + tuple(
            record[1] for record in parameter_records):
        if id(parameter) not in seen:
            parameter.grad = None
            seen.add(id(parameter))


def _copy_tensor_value(
        actual, expected, label, expected_data=None,
        expected_data_signature=None):
    if (not isinstance(actual, torch.Tensor)
            or not isinstance(expected, torch.Tensor)):
        raise RuntimeError("{} cannot be restored exactly".format(label))
    if ((expected_data is None) != (expected_data_signature is None)
            or (expected_data is not None
                and not isinstance(expected_data, torch.Tensor))):
        raise RuntimeError("{} data reference is invalid".format(label))
    with torch.no_grad():
        rebind = (actual.dtype != expected.dtype
                or actual.device != expected.device
                or actual.layout != expected.layout
                or tuple(actual.shape) != tuple(expected.shape))
        if (expected_data_signature is not None
                and _tensor_data_signature(actual, label)
                != expected_data_signature):
            rebind = True
        if rebind:
            actual.data = (
                expected_data
                if expected_data is not None
                else expected.detach().clone()
            )
        actual.copy_(expected)
    if (expected_data_signature is not None
            and _tensor_data_signature(actual, label)
            != expected_data_signature):
        raise RuntimeError(
            "{} data storage was not restored exactly".format(label)
        )
    if not _tensor_is_bitwise_equal(actual, expected, label):
        raise RuntimeError("{} was not restored bitwise".format(label))


def _restore_frozen_snapshot(model, frozen_snapshot):
    snapshot = _validate_snapshot(frozen_snapshot)
    state = model.state_dict()
    if tuple(state.keys()) != snapshot["state_names"]:
        raise RuntimeError("cannot restore changed model state structure")
    parameters = dict(_named_parameters_with_all_paths(model))
    buffers_by_name = dict(_named_buffers_with_all_paths(model))
    for name, expected in snapshot["frozen_state"].items():
        if name in parameters:
            actual = parameters[name]
        elif name in buffers_by_name:
            actual = buffers_by_name[name]
        else:
            actual = state[name]
        _copy_tensor_value(
            actual,
            expected,
            "frozen state {}".format(name),
            snapshot["frozen_data"][name],
            snapshot["frozen_data_signatures"][name],
        )

    buffers = tuple(_named_buffers_with_all_paths(model))
    signature = tuple(
        (name, None if buffer is None else id(buffer))
        for name, buffer in buffers
    )
    if signature != snapshot["buffer_signature"]:
        raise RuntimeError("cannot restore changed buffer structure")
    for name, buffer in buffers:
        expected = snapshot["frozen_buffers"][name]
        if buffer is None:
            if expected is not None:
                raise RuntimeError(
                    "registered buffer {} cannot be restored".format(name)
                )
        elif name not in snapshot["frozen_state"]:
            _copy_tensor_value(
                buffer,
                expected,
                "registered buffer {}".format(name),
                snapshot["frozen_buffer_data"][name],
                snapshot["frozen_buffer_data_signatures"][name],
            )


def _restore_optimizer_graph_node(node, restored):
    kind = node["kind"]
    value = node["object"]
    if kind == "reference":
        return value

    identity = id(node)
    if identity in restored:
        return value
    restored.add(identity)

    if kind == "tensor":
        _copy_tensor_value(
            value,
            node["value"],
            "optimizer state tensor",
            node["data"],
            node["data_signature"],
        )
        value.requires_grad_(node["requires_grad"])
    elif kind == "dict":
        value.clear()
        if hasattr(value, "default_factory"):
            value.default_factory = node["default_factory"]
        for key_node, nested_node in node["items"]:
            key = _restore_optimizer_graph_node(key_node, restored)
            nested = _restore_optimizer_graph_node(nested_node, restored)
            value[key] = nested
    elif kind == "list":
        value[:] = []
        value.extend(
            _restore_optimizer_graph_node(item, restored)
            for item in node["items"]
        )
    elif kind == "tuple":
        for item in node["items"]:
            _restore_optimizer_graph_node(item, restored)
    elif kind == "set":
        value.clear()
        for item in node["items"]:
            value.add(_restore_optimizer_graph_node(item, restored))
    elif kind == "frozenset":
        for item in node["items"]:
            _restore_optimizer_graph_node(item, restored)
    else:
        raise RuntimeError(
            "unsupported optimizer transaction node {}".format(kind)
        )
    return value


def _restore_optimizer_transaction(optimizer, transaction):
    graph = transaction["optimizer_graph"]
    restored = set()
    optimizer.state = _restore_optimizer_graph_node(
        graph["state"], restored
    )
    optimizer.defaults = _restore_optimizer_graph_node(
        graph["defaults"], restored
    )
    optimizer.param_groups = _restore_optimizer_graph_node(
        graph["param_groups"], restored
    )


def _restore_rng_transaction(transaction):
    torch.set_rng_state(transaction["cpu_rng_state"])
    cuda_rng_states = transaction["cuda_rng_states"]
    if cuda_rng_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA became unavailable while restoring RNG state"
            )
        if len(cuda_rng_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count changed while restoring RNG state"
            )
        torch.cuda.set_rng_state_all(list(cuda_rng_states))


def _restore_training_transaction(
        model, optimizer, transaction, frozen_snapshot):
    _restore_module_registries(transaction["module_registries"])
    current = tuple(model.named_parameters())
    expected_signature = tuple(
        (name, id(parameter))
        for name, parameter, _requires_grad in transaction["parameters"]
    )
    if _parameter_signature(current) != expected_signature:
        raise RuntimeError("cannot restore changed model parameter structure")

    current_by_name = dict(current)
    for name, expected in transaction["trainable_state"].items():
        _copy_tensor_value(
            current_by_name[name],
            expected,
            "trainable parameter {}".format(name),
            transaction["trainable_data"][name],
            transaction["trainable_data_signatures"][name],
        )
    _restore_frozen_snapshot(model, frozen_snapshot)
    for _name, parameter, requires_grad in transaction["parameters"]:
        parameter.requires_grad_(requires_grad)
    _restore_optimizer_transaction(optimizer, transaction)
    _clear_model_gradients(model, transaction["parameters"])
    model.eval()

    _validate_module_registries(model, transaction["module_registries"])
    verify_query_mask_frozen_state(model, frozen_snapshot)
    for name, expected in transaction["trainable_state"].items():
        if not _tensor_is_bitwise_equal(
                current_by_name[name], expected,
                "restored trainable parameter {}".format(name)):
            raise RuntimeError(
                "trainable parameter {} was not restored bitwise".format(name)
            )
    _restore_rng_transaction(transaction)


def _install_eval_forward_guards(model):
    modules = tuple(model.modules())
    active_module_ids = frozenset(id(module) for module in modules)

    def reject_train_mode(module, _inputs):
        if module.training:
            raise RuntimeError(
                "query-mask eval forward guard forbids loss_closure "
                "train mode forward"
            )

    def reject_active_train_mode(module, inputs):
        if id(module) in active_module_ids:
            return reject_train_mode(module, inputs)

    global_registry = torch.nn.modules.module._global_forward_pre_hooks
    global_entries = tuple(global_registry.items())
    global_handle = None
    local_guards = []
    try:
        global_handle = (
            torch.nn.modules.module.register_module_forward_pre_hook(
                reject_active_train_mode
            )
        )
        for module in modules:
            registry = module._forward_pre_hooks
            handle = module.register_forward_pre_hook(reject_train_mode)
            local_guards.append(
                (module, registry, handle, handle.id, reject_train_mode)
            )
    except Exception:
        if global_handle is not None:
            global_handle.remove()
        for _module, _registry, handle, _hook_id, _hook in local_guards:
            handle.remove()
        raise
    global_guard = (
        global_registry,
        global_entries,
        global_handle,
        global_handle.id,
        reject_active_train_mode,
    )
    return tuple(local_guards), global_guard


def _validate_eval_forward_guards(guards):
    local_guards, global_guard = guards
    for module, registry, _handle, hook_id, hook in local_guards:
        current = getattr(module, "_forward_pre_hooks", None)
        if (current is not registry
                or hook_id not in current
                or current[hook_id] is not hook):
            raise RuntimeError(
                "query-mask eval forward guard hook was removed or replaced"
            )
    registry, entries, _handle, hook_id, hook = global_guard
    current = torch.nn.modules.module._global_forward_pre_hooks
    if (current is not registry
            or not _identity_mapping_matches(
                current, entries + ((hook_id, hook),)
            )):
        raise RuntimeError(
            "query-mask global eval forward guard hook was removed or "
            "replaced"
        )


def _remove_forward_guards(guards):
    local_guards, global_guard = guards
    registry, entries, global_handle, _hook_id, _hook = global_guard
    global_handle.remove()
    for _module, _registry, handle, _hook_id, _hook in local_guards:
        handle.remove()
    current = torch.nn.modules.module._global_forward_pre_hooks
    if (current is not registry
            or not _identity_mapping_matches(current, entries)):
        raise RuntimeError(
            "query-mask global eval forward hook registry drift"
        )


def run_query_mask_training_step(
        model, optimizer, loss_closure, frozen_snapshot, max_grad_norm=0.1):
    """Run one deterministic, guarded optimizer step in whole-model eval mode."""
    max_grad_norm = _validated_max_grad_norm(max_grad_norm)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("query-mask training requires a torch.nn.Module")

    transaction = None
    pending_error = None
    try:
        if not callable(loss_closure):
            raise TypeError("query-mask loss_closure must be callable")
        model.eval()
        verify_query_mask_frozen_state(model, frozen_snapshot)
        selected = _validated_trainable_inventory(model)
        _validate_optimizer_parameters(optimizer, selected)
        transaction = _snapshot_training_transaction(model, optimizer)

        model.eval()
        optimizer.zero_grad()

        model.eval()
        guard_handles = _install_eval_forward_guards(model)
        closure_error = None
        try:
            loss = loss_closure(model)
            if any(module.training for module in model.modules()):
                raise RuntimeError(
                    "loss_closure switched the model or a submodule "
                    "to train mode"
                )
            _validate_eval_forward_guards(guard_handles)
        except BaseException as error:
            closure_error = error
            raise
        finally:
            cleanup_error = None
            try:
                _remove_forward_guards(guard_handles)
            except BaseException as error:
                cleanup_error = error
            try:
                model.eval()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            if closure_error is None and cleanup_error is not None:
                raise cleanup_error
        _validate_module_registries(model, transaction["module_registries"])
        verify_query_mask_frozen_state(model, frozen_snapshot)
        loss_value = _validated_scalar_loss(loss)
        selected = _validated_trainable_inventory(model)
        _validate_optimizer_parameters(optimizer, selected)
        loss.backward()
        model.eval()
        _validate_module_registries(model, transaction["module_registries"])
        verify_query_mask_frozen_state(model, frozen_snapshot)
        gradient_record = validate_query_mask_gradients(model)
        selected = _validated_trainable_inventory(model)
        _validate_optimizer_parameters(optimizer, selected)
        clip_parameters = tuple(
            parameter for _name, parameter in selected
        )
        pre_clip_norm_value = torch.nn.utils.clip_grad_norm_(
            clip_parameters, max_grad_norm
        )
        if (not isinstance(pre_clip_norm_value, torch.Tensor)
                or pre_clip_norm_value.numel() != 1):
            raise RuntimeError(
                "query-mask gradient clipping returned an invalid norm"
            )
        pre_clip_grad_norm = float(pre_clip_norm_value.detach().item())
        if not math.isfinite(pre_clip_grad_norm):
            raise FloatingPointError(
                "query-mask pre-clip gradient norm is non-finite"
            )
        post_clip_record = validate_query_mask_gradients(model)
        selected = _validated_trainable_inventory(model)
        _validate_optimizer_parameters(optimizer, selected)
        post_clip_grad_norm = _query_mask_gradient_l2_norm(selected)
        if not math.isfinite(post_clip_grad_norm):
            raise FloatingPointError(
                "query-mask post-clip gradient norm is non-finite"
            )
        clip_tolerance = max(1e-7, max_grad_norm * 1e-6)
        if post_clip_grad_norm > max_grad_norm + clip_tolerance:
            raise RuntimeError(
                "query-mask post-clip gradient norm exceeds max_grad_norm"
            )
        if post_clip_record != gradient_record:
            raise RuntimeError(
                "query-mask gradient contract changed during clipping"
            )

        before = _clone_trainable_parameters(selected)
        optimizer.step()
        model.eval()
        _validate_module_registries(model, transaction["module_registries"])
        selected = _validated_trainable_inventory(model)
        _validate_optimizer_parameters(optimizer, selected)
        _validate_trainable_parameters_finite(selected)
        changed_names = tuple(
            name
            for name, parameter in selected
            if not _tensor_is_bitwise_equal(
                parameter, before[name], "x_query parameter {}".format(name)
            )
        )
        if not changed_names:
            raise RuntimeError("optimizer step changed no x_query parameter")
        verify_query_mask_frozen_state(model, frozen_snapshot)
        return {
            "loss": loss_value,
            "max_grad_norm": max_grad_norm,
            "pre_clip_grad_norm": pre_clip_grad_norm,
            "post_clip_grad_norm": post_clip_grad_norm,
            "gradient_tensor_count": gradient_record[
                "gradient_tensor_count"
            ],
            "gradient_element_count": gradient_record[
                "gradient_element_count"
            ],
            "changed_names": changed_names,
        }
    except BaseException as original_error:
        pending_error = original_error
        if transaction is not None:
            try:
                _restore_training_transaction(
                    model, optimizer, transaction, frozen_snapshot
                )
            except BaseException as rollback_error:
                pending_error = QueryMaskRollbackError(
                    original_error, rollback_error
                )
                raise pending_error from original_error
        raise
    finally:
        parameter_records = (
            transaction["parameters"] if transaction is not None else ()
        )
        cleanup_error = None
        try:
            _clear_model_gradients(model, parameter_records)
        except BaseException as error:
            cleanup_error = error
        try:
            model.eval()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if pending_error is None and cleanup_error is not None:
            raise cleanup_error


# Train-only data and loss helpers are intentionally independent of the
# transaction safeguards above. They do not construct datasets or run models.
from collections.abc import Mapping as _Mapping
from numbers import Real as _Real

import math as _math
import numpy as _np
import re as _re


class RootMaskTrainDatasetView(torch.utils.data.Dataset):
    """Expose an owned first-instance mask from a train dataset sample."""

    def __init__(self, dataset):
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError("root-mask view requires a torch Dataset")
        if getattr(dataset, "split", None) != "train":
            raise ValueError("root-mask view requires split=train")
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dataset_index = int(index)
        item = self.dataset[index]
        if not isinstance(item, _Mapping):
            raise TypeError("root-mask dataset sample must be a mapping")
        if "gt_masks" not in item:
            raise ValueError("root-mask dataset sample is missing gt_masks")

        masks = item["gt_masks"]
        if not isinstance(masks, (torch.Tensor, _np.ndarray)):
            raise TypeError("gt_masks must be a Tensor or numpy ndarray")
        if masks.ndim != 2 or masks.shape[0] < 1 or masks.shape[1] < 1:
            raise ValueError("gt_masks must have shape [M,N] with M,N > 0")

        if "dataset_index" in item:
            existing = item["dataset_index"]
            if isinstance(existing, torch.Tensor):
                existing_value = (
                    existing.detach().cpu().item()
                    if existing.numel() == 1 else None
                )
            elif isinstance(existing, _np.ndarray):
                existing_value = (
                    existing.reshape(()).item()
                    if existing.size == 1 else None
                )
            else:
                existing_value = existing
            try:
                comparison = existing_value == dataset_index
                matches = (
                    isinstance(comparison, (bool, _np.bool_))
                    and not isinstance(existing_value, (bool, _np.bool_))
                    and bool(comparison)
                )
            except (TypeError, ValueError, OverflowError):
                matches = False
            if not matches:
                raise ValueError("sample dataset_index conflicts with index")

        result = dict(item)
        if isinstance(masks, torch.Tensor):
            result["gt_masks"] = masks[:1].clone().contiguous()
        else:
            result["gt_masks"] = _np.array(
                masks[:1], dtype=masks.dtype, copy=True, order="C"
            )
        result["dataset_index"] = dataset_index
        return result


def select_root_box_supervision_queries(candidate_batch, threshold=0.25):
    """Select the lowest-slot maximum strict-IoU candidate in each row."""
    if (not isinstance(threshold, _Real)
            or isinstance(threshold, (bool, _np.bool_))):
        raise TypeError("root supervision threshold must be numeric")
    if float(threshold) != 0.25:
        raise ValueError("root supervision threshold is fixed at 0.25")
    if not isinstance(candidate_batch, _Mapping):
        raise TypeError("candidate_batch must be a mapping")

    required = ("query_indices", "valid_mask", "candidate_ious")
    for key in required:
        if key not in candidate_batch:
            raise ValueError("candidate_batch is missing {}".format(key))
        if not isinstance(candidate_batch[key], torch.Tensor):
            raise TypeError("candidate_batch {} must be a tensor".format(key))
    query_indices = candidate_batch["query_indices"]
    valid_mask = candidate_batch["valid_mask"]
    candidate_ious = candidate_batch["candidate_ious"]

    if query_indices.dtype != torch.long:
        raise TypeError("query_indices must use int64 dtype")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    if not candidate_ious.is_floating_point():
        raise TypeError("candidate_ious must use a floating dtype")
    if (query_indices.dim() != 2
            or valid_mask.shape != query_indices.shape
            or candidate_ious.shape != query_indices.shape
            or query_indices.shape[0] < 1
            or query_indices.shape[1] < 1):
        raise ValueError(
            "candidate tensors must share a non-empty [B,K] shape"
        )
    if (valid_mask.device != query_indices.device
            or candidate_ious.device != query_indices.device):
        raise ValueError("candidate tensors must use the same device")
    if bool((query_indices < 0).any().item()):
        raise ValueError("query_indices must be non-negative")
    if bool(valid_mask.any().item()) and not bool(
            torch.isfinite(candidate_ious[valid_mask]).all().item()):
        raise ValueError("valid candidate_ious must be finite")

    eligible = valid_mask & (candidate_ious > 0.25)
    eligible_mask = eligible.any(dim=1)
    ranked_ious = candidate_ious.masked_fill(
        ~eligible, float("-inf")
    )
    candidate_slots = ranked_ious.argmax(dim=1)
    gathered_queries = query_indices.gather(
        1, candidate_slots.unsqueeze(1)
    ).squeeze(1)
    gathered_ious = candidate_ious.gather(
        1, candidate_slots.unsqueeze(1)
    ).squeeze(1)
    selected_slots = torch.where(
        eligible_mask,
        candidate_slots,
        torch.full_like(candidate_slots, -1),
    )
    selected_queries = torch.where(
        eligible_mask,
        gathered_queries,
        torch.full_like(gathered_queries, -1),
    )
    selected_ious = torch.where(
        eligible_mask,
        gathered_ious,
        torch.zeros_like(gathered_ious),
    )
    return {
        "selected_query_indices": selected_queries,
        "selected_candidate_slots": selected_slots,
        "selected_candidate_ious": selected_ious,
        "eligible_mask": eligible_mask,
    }


class NoEligibleRootMaskSupervisionError(RuntimeError):
    """Signal that a batch cannot produce a root-mask optimizer step."""


def _validate_root_mask_selection(selection, batch_size, device):
    if not isinstance(selection, _Mapping):
        raise TypeError("root-mask selection must be a mapping")
    required = (
        "selected_query_indices",
        "selected_candidate_slots",
        "selected_candidate_ious",
        "eligible_mask",
    )
    for key in required:
        value = selection.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError("root-mask selection {} must be a tensor".format(
                key
            ))
        if value.shape != (batch_size,):
            raise ValueError(
                "root-mask selection tensors must have shape [B]"
            )
        if value.device != device:
            raise ValueError(
                "root-mask selection tensors must use the loss device"
            )

    query_indices = selection["selected_query_indices"]
    candidate_slots = selection["selected_candidate_slots"]
    candidate_ious = selection["selected_candidate_ious"]
    eligible_mask = selection["eligible_mask"]
    if query_indices.dtype != torch.long:
        raise TypeError("selected_query_indices must use int64 dtype")
    if candidate_slots.dtype != torch.long:
        raise TypeError("selected_candidate_slots must use int64 dtype")
    if not candidate_ious.is_floating_point():
        raise TypeError("selected_candidate_ious must use a floating dtype")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must use bool dtype")
    if not bool(torch.isfinite(candidate_ious).all().item()):
        raise ValueError("selected_candidate_ious must be finite")

    if bool(eligible_mask.any().item()):
        if bool((query_indices[eligible_mask] < 0).any().item()):
            raise ValueError("eligible selected query index is negative")
        if bool((candidate_slots[eligible_mask] < 0).any().item()):
            raise ValueError("eligible selected candidate slot is negative")
        if not bool((candidate_ious[eligible_mask] > 0.25).all().item()):
            raise ValueError("eligible selected candidate IoU must exceed 0.25")
    ineligible_mask = ~eligible_mask
    if bool(ineligible_mask.any().item()):
        if (not bool((query_indices[ineligible_mask] == -1).all().item())
                or not bool(
                    (candidate_slots[ineligible_mask] == -1).all().item()
                )
                or not bool(
                    (candidate_ious[ineligible_mask] == 0).all().item()
                )):
            raise ValueError("ineligible root-mask selection sentinel is invalid")
    return query_indices, eligible_mask


def compute_point_count_weighted_root_mask_loss(
        sp_last_pred_masks, selection, gt_masks, superpoint):
    """Compute exact pointwise focal and Dice from superpoint counts."""
    if (not isinstance(sp_last_pred_masks, (list, tuple))
            or not sp_last_pred_masks):
        raise TypeError(
            "sp_last_pred_masks must be a non-empty list or tuple"
        )
    if not isinstance(gt_masks, torch.Tensor):
        raise TypeError("gt_masks must be a tensor")
    if not isinstance(superpoint, torch.Tensor):
        raise TypeError("superpoint must be a tensor")

    batch_size = len(sp_last_pred_masks)
    if (gt_masks.dim() != 3
            or gt_masks.shape[0] != batch_size
            or gt_masks.shape[1] != 1
            or gt_masks.shape[2] < 1):
        raise ValueError("gt_masks must have non-empty shape [B,1,N]")
    if (superpoint.dim() != 2
            or superpoint.shape != (batch_size, gt_masks.shape[2])):
        raise ValueError("superpoint must have shape [B,N]")
    if superpoint.dtype != torch.long:
        raise TypeError("superpoint must use int64 dtype")
    if superpoint.device != gt_masks.device:
        raise ValueError("gt_masks and superpoint must use the same device")
    if gt_masks.is_complex():
        raise TypeError("gt_masks must contain binary values")
    if gt_masks.is_floating_point() and not bool(
            torch.isfinite(gt_masks).all().item()):
        raise ValueError("gt_masks must be finite")
    if bool(((gt_masks != 0) & (gt_masks != 1)).any().item()):
        raise ValueError("gt_masks must contain only binary values")

    query_indices, eligible_mask = _validate_root_mask_selection(
        selection, batch_size, gt_masks.device
    )
    for batch_index, logits in enumerate(sp_last_pred_masks):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("each mask-logit row must be a tensor")
        if (logits.dim() != 2
                or logits.shape[0] < 1
                or logits.shape[1] < 1):
            raise ValueError("mask logits must have non-empty shape [Q,S_b]")
        if not logits.is_floating_point():
            raise TypeError("mask logits must use a floating dtype")
        if logits.device != gt_masks.device:
            raise ValueError("mask logits and targets must use the same device")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("mask logits must be finite")

        point_ids = superpoint[batch_index]
        if bool((point_ids < 0).any().item()):
            raise ValueError("superpoint ids must be non-negative")
        if bool((point_ids >= logits.shape[1]).any().item()):
            raise ValueError("superpoint id is out of range for mask logits")
        if bool(eligible_mask[batch_index].item()):
            selected_query = int(query_indices[batch_index].item())
            if selected_query >= logits.shape[0]:
                raise ValueError("selected query index is out of range")

    eligible_count = int(eligible_mask.sum().item())
    if eligible_count == 0:
        raise NoEligibleRootMaskSupervisionError(
            "batch has no eligible root-mask supervision row"
        )

    focal_losses = []
    dice_losses = []
    point_total = float(gt_masks.shape[2])
    for batch_index, logits in enumerate(sp_last_pred_masks):
        if not bool(eligible_mask[batch_index].item()):
            continue
        selected_logits = logits[query_indices[batch_index]]
        accumulation_dtype = (
            torch.float64
            if selected_logits.dtype == torch.float64 else torch.float32
        )
        accumulation_logits = selected_logits.to(dtype=accumulation_dtype)
        point_ids = superpoint[batch_index]
        target = gt_masks[batch_index, 0].to(dtype=accumulation_dtype)
        superpoint_count = accumulation_logits.shape[0]
        point_count = torch.bincount(
            point_ids, minlength=superpoint_count
        ).to(dtype=accumulation_dtype)
        positive_count = torch.bincount(
            point_ids, weights=target, minlength=superpoint_count
        )
        negative_count = point_count - positive_count

        probability = accumulation_logits.sigmoid()
        positive_bce = torch.nn.functional.softplus(-accumulation_logits)
        negative_bce = torch.nn.functional.softplus(accumulation_logits)
        positive_focal = (
            0.25 * positive_bce * (1.0 - probability).pow(2.0)
        )
        negative_focal = (
            0.75 * negative_bce * probability.pow(2.0)
        )
        focal_losses.append((
            positive_count * positive_focal
            + negative_count * negative_focal
        ).sum() / point_total)

        intersection = (probability * positive_count).sum()
        prediction_mass = (probability * point_count).sum()
        target_mass = positive_count.sum()
        dice_losses.append(
            1.0 - (2.0 * intersection + 1.0) / (
                prediction_mass + target_mass + 1.0
            )
        )

    focal = torch.stack(focal_losses).mean()
    dice = torch.stack(dice_losses).mean()
    return {
        "loss": focal + dice,
        "focal": focal,
        "dice": dice,
        "eligible_count": eligible_count,
    }


def compute_train_only_root_mask_supervision(
        end_points, batch, targeted_candidate_batch):
    """Compose train-only candidate selection and root-mask supervision."""
    if not isinstance(end_points, _Mapping):
        raise TypeError("end_points must be a mapping")
    if not isinstance(batch, _Mapping):
        raise TypeError("train batch must be a mapping")
    selection = select_root_box_supervision_queries(
        targeted_candidate_batch
    )
    sp_last_pred_masks = end_points["sp_last_pred_masks"]
    gt_masks = batch["gt_masks"]
    superpoint = batch["superpoint"]
    record = compute_point_count_weighted_root_mask_loss(
        sp_last_pred_masks, selection, gt_masks, superpoint
    )
    result = dict(record)
    result["selection"] = selection
    return result


X_QUERY_CHECKPOINT_SCHEMA = "mcln-x-query-checkpoint-v1"


def _cpu_checkpoint_clone(value, label):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, _Mapping):
        return {
            key: _cpu_checkpoint_clone(item, "{}[{}]".format(label, key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _cpu_checkpoint_clone(item, "{}[]".format(label))
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _cpu_checkpoint_clone(item, "{}[]".format(label))
            for item in value
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("{} contains unsupported checkpoint value".format(label))


def _validate_optimizer_checkpoint_tree(value, label):
    if isinstance(value, torch.Tensor):
        if (value.device.type != "cpu" or value.layout != torch.strided
                or value.numel() < 1):
            raise ValueError(
                "optimizer state tensor must be a non-empty CPU tensor"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all().item()):
            raise ValueError("optimizer state tensor must be finite")
        return
    if isinstance(value, _Mapping):
        for key, item in value.items():
            _validate_optimizer_checkpoint_tree(
                item, "{}[{}]".format(label, key)
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_optimizer_checkpoint_tree(
                item, "{}[{}]".format(label, index)
            )
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not _math.isfinite(value):
            raise ValueError("optimizer state scalar must be finite")
        return
    raise ValueError("optimizer state contains an unsupported value")


def _optimizer_type_contract(optimizer):
    optimizer_type = type(optimizer)
    module = getattr(optimizer_type, "__module__", None)
    qualname = getattr(optimizer_type, "__qualname__", None)
    if (not isinstance(module, str) or not module
            or not isinstance(qualname, str) or not qualname):
        raise ValueError("query-mask optimizer type is not identifiable")
    return {"module": module, "qualname": qualname}


def _optimizer_contract_key(value, label):
    if value is None:
        return ("none", None)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not _math.isfinite(value):
            raise ValueError("{} key must be finite".format(label))
        return ("float", value)
    if type(value) is str:
        return ("str", value)
    raise ValueError("{} has an unsupported mapping key".format(label))


def _optimizer_state_value_contract(value, parameter_shape, label):
    if isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
        if shape != () and shape != parameter_shape:
            raise ValueError(
                "optimizer state tensor {} shape {} is incompatible with "
                "parameter shape {}".format(label, shape, parameter_shape)
            )
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": shape,
        }
    if isinstance(value, _Mapping):
        return {
            "kind": "mapping",
            "items": tuple(
                (
                    _optimizer_contract_key(key, label),
                    _optimizer_state_value_contract(
                        item,
                        parameter_shape,
                        "{}[{}]".format(label, key),
                    ),
                )
                for key, item in value.items()
            ),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": tuple(
                _optimizer_state_value_contract(
                    item,
                    parameter_shape,
                    "{}[{}]".format(label, index),
                )
                for index, item in enumerate(value)
            ),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": tuple(
                _optimizer_state_value_contract(
                    item,
                    parameter_shape,
                    "{}[{}]".format(label, index),
                )
                for index, item in enumerate(value)
            ),
        }
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool"}
    if type(value) is int:
        return {"kind": "int"}
    if type(value) is float:
        if not _math.isfinite(value):
            raise ValueError("optimizer state scalar must be finite")
        return {"kind": "float"}
    if type(value) is str:
        return {"kind": "str"}
    raise ValueError("optimizer state contract contains an unsupported value")


def _optimizer_parameter_groups_contract(optimizer, selected):
    names_by_identity = {
        id(parameter): name for name, parameter in selected
    }
    parameter_groups = tuple(
        tuple(names_by_identity[id(parameter)] for parameter in group["params"])
        for group in optimizer.param_groups
    )
    flattened = tuple(
        name for group_names in parameter_groups for name in group_names
    )
    if (not parameter_groups
            or any(not group_names for group_names in parameter_groups)
            or flattened != tuple(name for name, _parameter in selected)):
        raise ValueError(
            "optimizer parameter group order must equal exact x_query.* "
            "parameter order"
        )
    return parameter_groups


def _optimizer_checkpoint_state_by_name(
        optimizer_state, parameter_groups, label):
    serialized_groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if (not isinstance(serialized_groups, (list, tuple))
            or not isinstance(state, _Mapping)
            or len(serialized_groups) != len(parameter_groups)):
        raise ValueError("{} parameter groups are invalid".format(label))
    identifiers_by_name = {}
    seen_identifiers = set()
    for group_index, (serialized_group, names) in enumerate(zip(
            serialized_groups, parameter_groups)):
        if not isinstance(serialized_group, _Mapping):
            raise ValueError("{} parameter group is invalid".format(label))
        identifiers = serialized_group.get("params")
        if (not isinstance(identifiers, (list, tuple))
                or len(identifiers) != len(names)):
            raise ValueError(
                "{} parameter group {} does not match contract".format(
                    label, group_index
                )
            )
        for identifier, name in zip(identifiers, names):
            if (not isinstance(identifier, int)
                    or isinstance(identifier, bool)
                    or identifier in seen_identifiers):
                raise ValueError(
                    "{} parameter identifiers are invalid".format(label)
                )
            seen_identifiers.add(identifier)
            identifiers_by_name[name] = identifier
    if any(identifier not in seen_identifiers for identifier in state):
        raise ValueError("{} contains state for an unknown parameter".format(
            label
        ))
    return {
        name: state[identifier]
        for name, identifier in identifiers_by_name.items()
        if identifier in state
    }


def _validated_optimizer_contract_structure(contract):
    if not isinstance(contract, _Mapping) or set(contract) != {
            "type", "parameter_groups", "state"}:
        raise ValueError("query-mask optimizer contract fields are invalid")
    optimizer_type = contract["type"]
    if (not isinstance(optimizer_type, _Mapping)
            or set(optimizer_type) != {"module", "qualname"}
            or any(not isinstance(value, str) or not value
                   for value in optimizer_type.values())):
        raise ValueError("query-mask optimizer type contract is invalid")
    parameter_groups = contract["parameter_groups"]
    if (not isinstance(parameter_groups, tuple)
            or not parameter_groups
            or any(not isinstance(group, tuple) or not group
                   for group in parameter_groups)):
        raise ValueError(
            "query-mask optimizer parameter group contract is invalid"
        )
    names = tuple(
        name for group_names in parameter_groups for name in group_names
    )
    if (any(not isinstance(name, str)
            or not name.startswith(X_QUERY_PREFIX) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError(
            "query-mask optimizer parameter group contract is invalid"
        )
    state = contract["state"]
    if (not isinstance(state, _Mapping)
            or any(name not in names for name in state)
            or tuple(state) != tuple(name for name in names if name in state)):
        raise ValueError("query-mask optimizer state contract is invalid")
    return contract


def _build_optimizer_checkpoint_contract(
        optimizer, optimizer_state, selected):
    parameter_groups = _optimizer_parameter_groups_contract(
        optimizer, selected
    )
    state_by_name = _optimizer_checkpoint_state_by_name(
        optimizer_state, parameter_groups, "optimizer state"
    )
    parameters_by_name = dict(selected)
    return {
        "type": _optimizer_type_contract(optimizer),
        "parameter_groups": parameter_groups,
        "state": {
            name: _optimizer_state_value_contract(
                state,
                tuple(parameters_by_name[name].shape),
                "{} state".format(name),
            )
            for name, state in state_by_name.items()
        },
    }


def _validate_optimizer_checkpoint_contract(
        optimizer, optimizer_state, contract, selected):
    contract = _validated_optimizer_contract_structure(contract)
    if _optimizer_type_contract(optimizer) != contract["type"]:
        raise ValueError("query-mask checkpoint optimizer type mismatch")
    current_groups = _optimizer_parameter_groups_contract(
        optimizer, selected
    )
    if current_groups != contract["parameter_groups"]:
        raise ValueError(
            "query-mask checkpoint optimizer parameter groups mismatch"
        )
    state_by_name = _optimizer_checkpoint_state_by_name(
        optimizer_state, current_groups, "optimizer state"
    )
    parameters_by_name = dict(selected)
    actual_state_contract = {
        name: _optimizer_state_value_contract(
            state,
            tuple(parameters_by_name[name].shape),
            "{} state".format(name),
        )
        for name, state in state_by_name.items()
    }
    if actual_state_contract != contract["state"]:
        raise ValueError("query-mask checkpoint optimizer state mismatch")
    return True


def _validated_base_checkpoint_binding(binding):
    required = {"path", "sha256", "size", "mode"}
    if not isinstance(binding, _Mapping) or set(binding) != required:
        raise ValueError("base checkpoint binding fields are invalid")
    path = binding["path"]
    digest = binding["sha256"]
    size = binding["size"]
    mode = binding["mode"]
    if not isinstance(path, str) or not path:
        raise ValueError("base checkpoint path is invalid")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("base checkpoint SHA-256 is invalid")
    try:
        int(digest, 16)
    except ValueError:
        raise ValueError("base checkpoint SHA-256 is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("base checkpoint size is invalid")
    if mode != "0444":
        raise ValueError("base checkpoint mode must be 0444")
    return dict(binding)


_RUN_ID_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_checkpoint_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_json_checkpoint_value(value, label):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise ValueError("{} must be finite".format(label))
        return
    if isinstance(value, _Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("{} has an invalid key".format(label))
            _validate_json_checkpoint_value(
                item, "{}[{}]".format(label, key)
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_checkpoint_value(
                item, "{}[{}]".format(label, index)
            )
        return
    raise ValueError("{} is not JSON metadata".format(label))


def _validate_checkpoint_metrics(value, label):
    if isinstance(value, bool):
        raise ValueError("{} must contain numeric metrics".format(label))
    if isinstance(value, int):
        return 1
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise ValueError("{} must contain finite metrics".format(label))
        return 1
    if isinstance(value, _Mapping) and value:
        count = 0
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("{} has an invalid key".format(label))
            count += _validate_checkpoint_metrics(
                item, "{}[{}]".format(label, key)
            )
        return count
    raise ValueError("{} must contain numeric metrics".format(label))


def _validated_query_mask_run_record(run_record):
    required = {
        "run_id",
        "step",
        "epoch",
        "config",
        "source_sha256",
        "split_digest",
        "protected_artifacts",
        "train_metrics",
    }
    if not isinstance(run_record, _Mapping) or set(run_record) != required:
        raise ValueError("query-mask run record fields are invalid")
    run_id = run_record["run_id"]
    if not isinstance(run_id, str) or _RUN_ID_RE.match(run_id) is None:
        raise ValueError("query-mask run record run_id is invalid")
    for key in ("step", "epoch"):
        value = run_record[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "query-mask run record {} is invalid".format(key)
            )
    config = run_record["config"]
    if not isinstance(config, _Mapping) or not config:
        raise ValueError("query-mask run record config is invalid")
    _validate_json_checkpoint_value(config, "run record config")
    train_metrics = run_record["train_metrics"]
    if not isinstance(train_metrics, _Mapping) or not train_metrics:
        raise ValueError("query-mask run record train_metrics is invalid")
    _validate_checkpoint_metrics(train_metrics, "run record train_metrics")

    sources = run_record["source_sha256"]
    if (not isinstance(sources, _Mapping) or not sources
            or any(not isinstance(path, str) or not path for path in sources)
            or any(not _is_checkpoint_sha256(digest)
                   for digest in sources.values())):
        raise ValueError("query-mask run record source_sha256 is invalid")
    if not _is_checkpoint_sha256(run_record["split_digest"]):
        raise ValueError("query-mask run record split_digest is invalid")
    protected = run_record["protected_artifacts"]
    if not isinstance(protected, _Mapping) or set(protected) != {
            "parent_reranker", "geometry_reranker"}:
        raise ValueError(
            "query-mask run record protected_artifacts are invalid"
        )
    for binding in protected.values():
        _validated_base_checkpoint_binding(binding)
    return _cpu_checkpoint_clone(run_record, "run record")


def build_query_mask_checkpoint(
        model, optimizer, *, base_checkpoint, run_record):
    """Build an owned CPU-only x_query training checkpoint payload."""
    selected = _validated_trainable_inventory(model)
    _validate_optimizer_parameters(optimizer, selected)
    base_checkpoint = _validated_base_checkpoint_binding(base_checkpoint)
    run_record = _validated_query_mask_run_record(run_record)

    model_state = model.state_dict()
    selected_names = tuple(name for name, _parameter in selected)
    patch_names = selected_names + tuple(sorted(
        name for name in model_state
        if name.startswith(X_QUERY_PREFIX) and name not in selected_names
    ))
    patch_state = {
        name: model_state[name].detach().to(device="cpu").clone()
        for name in patch_names
    }
    if not patch_state or any(name not in patch_state for name in selected_names):
        raise RuntimeError("x_query checkpoint state is incomplete")

    optimizer_state = _cpu_checkpoint_clone(
        optimizer.state_dict(), "optimizer state"
    )
    _validate_optimizer_checkpoint_tree(optimizer_state, "optimizer state")
    optimizer_contract = _build_optimizer_checkpoint_contract(
        optimizer, optimizer_state, selected
    )
    artifact = {
        "schema": X_QUERY_CHECKPOINT_SCHEMA,
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "base_checkpoint": _cpu_checkpoint_clone(
            base_checkpoint, "base checkpoint"
        ),
        "run_record": run_record,
        "x_query_state_dict": patch_state,
        "optimizer_state_dict": optimizer_state,
        "optimizer_contract": optimizer_contract,
    }
    _validated_query_mask_checkpoint_payload(artifact)
    return artifact


import hashlib as _hashlib
import io as _io
import json as _json
import errno as _errno
import os as _os
import secrets as _secrets
import stat as _stat
from pathlib import Path as _Path


X_QUERY_CHECKPOINT_RECEIPT_SCHEMA = (
    "mcln-x-query-checkpoint-receipt-v1"
)


_CHECKPOINT_DIRECTORY_FLAGS = (
    _os.O_RDONLY
    | getattr(_os, "O_CLOEXEC", 0)
    | getattr(_os, "O_DIRECTORY", 0)
    | getattr(_os, "O_NOFOLLOW", 0)
)
_CHECKPOINT_FILE_FLAGS = (
    _os.O_RDONLY
    | getattr(_os, "O_CLOEXEC", 0)
    | getattr(_os, "O_NOFOLLOW", 0)
)


def _checkpoint_inode_identity(file_stat):
    return (file_stat.st_dev, file_stat.st_ino)


def _checkpoint_file_signature(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        _stat.S_IFMT(file_stat.st_mode),
        _stat.S_IMODE(file_stat.st_mode),
        file_stat.st_size,
    )


def _open_checkpoint_child_directory(parent_fd, name, label):
    descriptor = None
    try:
        descriptor = _os.open(
            name, _CHECKPOINT_DIRECTORY_FLAGS, dir_fd=parent_fd
        )
        entry_stat = _os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False
        )
        descriptor_stat = _os.fstat(descriptor)
        if (not _stat.S_ISDIR(entry_stat.st_mode)
                or not _stat.S_ISDIR(descriptor_stat.st_mode)
                or _checkpoint_inode_identity(entry_stat)
                != _checkpoint_inode_identity(descriptor_stat)):
            raise RuntimeError(
                "{} ancestor changed while being opened".format(label)
            )
        return descriptor
    except OSError as error:
        if descriptor is not None:
            _close_checkpoint_descriptor_after_error(
                descriptor, error, "directory descriptor close"
            )
        try:
            entry_stat = _os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            if error.errno == _errno.ENOENT:
                raise error
            raise RuntimeError(
                "{} ancestor changed while being opened".format(label)
            )
        if _stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError("{} must not traverse a symlink".format(label))
        if not _stat.S_ISDIR(entry_stat.st_mode):
            raise ValueError(
                "{} ancestor must be a directory".format(label)
            )
        raise ValueError(
            "{} ancestor could not be opened without following links: {}"
            .format(label, error)
        )
    except BaseException as error:
        if descriptor is not None:
            _close_checkpoint_descriptor_after_error(
                descriptor, error, "directory descriptor close"
            )
        raise


def _open_checkpoint_parent(path, label, create):
    parent_parts = path.parent.parts
    components = (
        parent_parts[1:]
        if parent_parts and parent_parts[0] == path.parent.anchor
        else parent_parts
    )
    descriptor = None
    missing_index = None
    try:
        descriptor = _os.open(_os.sep, _CHECKPOINT_DIRECTORY_FLAGS)
        for index, component in enumerate(components):
            try:
                child = _open_checkpoint_child_directory(
                    descriptor, component, label
                )
            except FileNotFoundError:
                missing_index = index
                break
            previous = descriptor
            descriptor = child
            _os.close(previous)
        if missing_index is None:
            return descriptor
    except BaseException as error:
        if descriptor is not None:
            _close_checkpoint_descriptor_after_error(
                descriptor, error, "directory descriptor close"
            )
        raise

    if not create:
        try:
            raise ValueError(
                "{} parent directory is unavailable".format(label)
            )
        except BaseException as error:
            _close_checkpoint_descriptor_after_error(
                descriptor, error, "directory descriptor close"
            )

    previous = descriptor
    descriptor = None
    _os.close(previous)

    try:
        descriptor = _os.open(_os.sep, _CHECKPOINT_DIRECTORY_FLAGS)
        for component in components:
            try:
                child = _open_checkpoint_child_directory(
                    descriptor, component, label
                )
            except FileNotFoundError:
                try:
                    _os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = _open_checkpoint_child_directory(
                    descriptor, component, label
                )
            previous = descriptor
            descriptor = child
            _os.close(previous)
        return descriptor
    except BaseException as error:
        if descriptor is not None:
            _close_checkpoint_descriptor_after_error(
                descriptor, error, "directory descriptor close"
            )
        raise


def _sha256_checkpoint_handle(handle):
    digest = _hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _create_checkpoint_staging_file(
        parent_fd, prefix, *fdopen_args, **fdopen_kwargs):
    flags = (
        _os.O_RDWR
        | _os.O_CREAT
        | _os.O_EXCL
        | getattr(_os, "O_CLOEXEC", 0)
        | getattr(_os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = "{}.staging.{}.tmp".format(
            prefix, _secrets.token_hex(16)
        )
        descriptor = None
        identity = None
        try:
            descriptor = _os.open(
                name, flags, 0o600, dir_fd=parent_fd
            )
            identity = _checkpoint_inode_identity(
                _os.fstat(descriptor)
            )
            handle = _os.fdopen(
                descriptor, *fdopen_args, **fdopen_kwargs
            )
            descriptor = None
        except BaseException as primary_error:
            if descriptor is None:
                if isinstance(primary_error, FileExistsError):
                    continue
                raise
            primary_traceback = primary_error.__traceback__
            cleanup_errors = []
            if identity is None:
                try:
                    identity = _checkpoint_inode_identity(
                        _os.fstat(descriptor)
                    )
                except BaseException as error:
                    cleanup_errors.append(
                        ("staging identity recovery", error)
                    )
            try:
                _os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(("raw descriptor close", error))
            if identity is not None:
                try:
                    _cleanup_checkpoint_staging_file(
                        parent_fd, name, identity
                    )
                except BaseException as error:
                    cleanup_errors.append(("staging cleanup", error))
            _raise_checkpoint_error_after_cleanup(
                primary_error, primary_traceback, cleanup_errors
            )
        return handle, name, identity
    raise FileExistsError(
        "could not allocate query-mask checkpoint staging file"
    )


def _raise_checkpoint_error_after_cleanup(
        primary_error, primary_traceback, cleanup_errors):
    try:
        existing_errors = tuple(primary_error.cleanup_errors)
    except BaseException:
        existing_errors = ()
    try:
        primary_error.cleanup_errors = (
            existing_errors + tuple(cleanup_errors)
        )
    except BaseException:
        pass
    raise primary_error.with_traceback(primary_traceback)


def _close_checkpoint_descriptor_after_error(
        descriptor, primary_error, label="raw descriptor close"):
    primary_traceback = primary_error.__traceback__
    cleanup_errors = []
    try:
        _os.close(descriptor)
    except BaseException as error:
        cleanup_errors.append((label, error))
    _raise_checkpoint_error_after_cleanup(
        primary_error, primary_traceback, cleanup_errors
    )


def _fdopen_checkpoint_descriptor(descriptor, *args, **kwargs):
    try:
        return _os.fdopen(descriptor, *args, **kwargs)
    except BaseException as primary_error:
        _close_checkpoint_descriptor_after_error(
            descriptor, primary_error
        )


def _validated_query_mask_checkpoint_payload(artifact):
    required = {
        "schema",
        "validation_data_accessed",
        "inference_uses_ground_truth",
        "base_checkpoint",
        "run_record",
        "x_query_state_dict",
        "optimizer_state_dict",
        "optimizer_contract",
    }
    if not isinstance(artifact, _Mapping) or set(artifact) != required:
        raise ValueError("query-mask checkpoint fields are invalid")
    if artifact["schema"] != X_QUERY_CHECKPOINT_SCHEMA:
        raise ValueError("query-mask checkpoint schema is invalid")
    if (artifact["validation_data_accessed"] is not False
            or artifact["inference_uses_ground_truth"] is not False):
        raise ValueError("query-mask checkpoint provenance is invalid")
    _validated_base_checkpoint_binding(artifact["base_checkpoint"])

    _validated_query_mask_run_record(artifact["run_record"])

    patch_state = artifact["x_query_state_dict"]
    if not isinstance(patch_state, _Mapping) or not patch_state:
        raise ValueError("query-mask checkpoint patch state is invalid")
    for name, value in patch_state.items():
        if (not isinstance(name, str)
                or not name.startswith(X_QUERY_PREFIX)
                or not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.layout != torch.strided
                or value.numel() < 1):
            raise ValueError("query-mask checkpoint patch tensor is invalid")
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all().item()):
            raise ValueError("query-mask checkpoint patch tensor is non-finite")
    optimizer_state = artifact["optimizer_state_dict"]
    if (not isinstance(optimizer_state, _Mapping)
            or set(optimizer_state) != {"state", "param_groups"}):
        raise ValueError("query-mask checkpoint optimizer state is invalid")
    _validate_optimizer_checkpoint_tree(optimizer_state, "optimizer state")
    _validated_optimizer_contract_structure(artifact["optimizer_contract"])
    return artifact


def _checkpoint_entry_stat(parent_fd, name):
    try:
        return _os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _reject_existing_checkpoint_output(parent_fd, name, display_path):
    entry_stat = _checkpoint_entry_stat(parent_fd, name)
    if entry_stat is None:
        return
    if _stat.S_ISLNK(entry_stat.st_mode):
        raise ValueError(
            "query-mask checkpoint output must not be a symlink: {}".format(
                display_path
            )
        )
    raise FileExistsError(
        "query-mask checkpoint output already exists: {}".format(
            display_path
        )
    )


class _CheckpointStagingIdentityChanged(RuntimeError):
    def __init__(self, staged_name):
        super().__init__(
            "query-mask checkpoint staging identity changed"
        )
        self.staged_name = staged_name


def _publish_staged_file_noreplace(
        parent_fd, staged_name, destination_name, destination_path,
        expected_signature):
    staged_stat = _os.stat(
        staged_name, dir_fd=parent_fd, follow_symlinks=False
    )
    if _checkpoint_file_signature(staged_stat) != expected_signature:
        raise _CheckpointStagingIdentityChanged(staged_name)
    try:
        _os.link(
            staged_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        raise FileExistsError(
            "query-mask checkpoint output already exists: {}".format(
                destination_path
            )
        )
    published_stat = _os.stat(
        destination_name, dir_fd=parent_fd, follow_symlinks=False
    )
    if _checkpoint_file_signature(published_stat) != expected_signature:
        raise RuntimeError(
            "query-mask checkpoint publication identity changed"
        )
    return _checkpoint_inode_identity(published_stat)


def _validate_published_checkpoint_against_receipt(
        parent_fd, name, expected_signature, receipt):
    try:
        descriptor = _os.open(
            name, _CHECKPOINT_FILE_FLAGS, dir_fd=parent_fd
        )
    except OSError as error:
        raise RuntimeError(
            "published checkpoint could not be opened: {}".format(error)
        )
    try:
        with _fdopen_checkpoint_descriptor(descriptor, "rb") as handle:
            before = _os.fstat(handle.fileno())
            content = handle.read()
            after = _os.fstat(handle.fileno())
        path_stat = _os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False
        )
    except OSError as error:
        raise RuntimeError(
            "published checkpoint could not be verified: {}".format(error)
        )
    if (any(_checkpoint_file_signature(file_stat) != expected_signature
            for file_stat in (before, after, path_stat))
            or len(content) != expected_signature[-1]):
        raise RuntimeError("published checkpoint identity changed")
    digest = _hashlib.sha256(content).hexdigest()
    if (receipt["checkpoint_sha256"] != digest
            or receipt["checkpoint_size"] != len(content)
            or receipt["checkpoint_mode"] != "0444"
            or _stat.S_IMODE(after.st_mode) != 0o444):
        raise RuntimeError(
            "published checkpoint does not match its receipt"
        )
    return True


def _cleanup_checkpoint_staging_file(
        parent_fd, name, expected_identity):
    entry_stat = _checkpoint_entry_stat(parent_fd, name)
    if (entry_stat is None
            or _checkpoint_inode_identity(entry_stat) != expected_identity):
        return False
    _os.unlink(name, dir_fd=parent_fd)
    return True


def _fsync_checkpoint_directory(parent_fd):
    _os.fsync(parent_fd)


def publish_query_mask_checkpoint(destination, artifact):
    """Publish a checkpoint and receipt sealed read-only at publication.

    Same-owner processes or root can still modify or unlink both files.
    """
    artifact = _validated_query_mask_checkpoint_payload(artifact)
    destination = _logical_checkpoint_path(
        destination, "query-mask checkpoint output"
    )
    if destination.suffix != ".pth":
        raise ValueError("query-mask checkpoint path must end in .pth")
    receipt_path = destination.with_name(
        destination.name + ".receipt.json"
    )
    parent_fd = _open_checkpoint_parent(
        destination, "query-mask checkpoint output", create=True
    )
    checkpoint_staged = None
    receipt_staged = None
    checkpoint_staged_identity = None
    receipt_staged_identity = None
    checkpoint_published = False
    published_receipt = None
    primary_error = None
    primary_traceback = None
    try:
        _reject_existing_checkpoint_output(
            parent_fd, destination.name, destination
        )
        _reject_existing_checkpoint_output(
            parent_fd, receipt_path.name, receipt_path
        )

        checkpoint_handle, checkpoint_staged, checkpoint_staged_identity = (
            _create_checkpoint_staging_file(
                parent_fd, destination.name, "w+b"
            )
        )
        with checkpoint_handle as handle:
            torch.save(artifact, handle)
            handle.flush()
            checkpoint_sha256 = _sha256_checkpoint_handle(handle)
            _os.fchmod(handle.fileno(), 0o444)
            _os.fsync(handle.fileno())
            checkpoint_stat = _os.fstat(handle.fileno())
            checkpoint_size = checkpoint_stat.st_size
            checkpoint_staged_signature = _checkpoint_file_signature(
                checkpoint_stat
            )

        run_record = artifact["run_record"]
        receipt = {
            "schema": X_QUERY_CHECKPOINT_RECEIPT_SCHEMA,
            "checkpoint_name": destination.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size": checkpoint_size,
            "checkpoint_mode": "0444",
            "base_checkpoint_sha256": artifact["base_checkpoint"][
                "sha256"
            ],
            "run_id": run_record["run_id"],
            "step": run_record["step"],
            "validation_data_accessed": False,
            "inference_uses_ground_truth": False,
        }
        receipt_handle, receipt_staged, receipt_staged_identity = (
            _create_checkpoint_staging_file(
                parent_fd,
                receipt_path.name,
                "w",
                encoding="utf-8",
            )
        )
        with receipt_handle as handle:
            _json.dump(
                receipt, handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            _os.fchmod(handle.fileno(), 0o444)
            _os.fsync(handle.fileno())
            receipt_staged_signature = _checkpoint_file_signature(
                _os.fstat(handle.fileno())
            )

        _publish_staged_file_noreplace(
            parent_fd,
            checkpoint_staged,
            destination.name,
            destination,
            checkpoint_staged_signature,
        )
        checkpoint_published = True
        _publish_staged_file_noreplace(
            parent_fd,
            receipt_staged,
            receipt_path.name,
            receipt_path,
            receipt_staged_signature,
        )
        _validate_published_checkpoint_against_receipt(
            parent_fd,
            destination.name,
            checkpoint_staged_signature,
            receipt,
        )
        _fsync_checkpoint_directory(parent_fd)
        published_receipt = receipt
    except BaseException as error:
        primary_error = error
        primary_traceback = error.__traceback__

    cleanup_errors = []
    for label, staged_name, staged_identity in (
            ("checkpoint staging", checkpoint_staged,
             checkpoint_staged_identity),
            ("receipt staging", receipt_staged, receipt_staged_identity)):
        if staged_name is None or staged_identity is None:
            continue
        if (
                isinstance(
                    primary_error, _CheckpointStagingIdentityChanged
                )
                and primary_error.staged_name == staged_name):
            continue
        try:
            _cleanup_checkpoint_staging_file(
                parent_fd, staged_name, staged_identity
            )
        except BaseException as error:
            cleanup_errors.append((label, error))
    if checkpoint_published:
        try:
            _fsync_checkpoint_directory(parent_fd)
        except BaseException as error:
            cleanup_errors.append(("directory fsync", error))
    try:
        _os.close(parent_fd)
    except BaseException as error:
        cleanup_errors.append(("parent close", error))

    if primary_error is not None:
        try:
            primary_error.cleanup_errors = tuple(cleanup_errors)
        except BaseException:
            pass
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_errors:
        first_cleanup_error = cleanup_errors[0][1]
        try:
            first_cleanup_error.cleanup_errors = tuple(cleanup_errors)
        except BaseException:
            pass
        raise first_cleanup_error.with_traceback(
            first_cleanup_error.__traceback__
        )
    return published_receipt


def _logical_checkpoint_path(path, label):
    if (not isinstance(path, (str, _os.PathLike))
            or isinstance(path, bytes)):
        raise TypeError("{} must be path-like".format(label))
    expanded = _os.path.expanduser(_os.fspath(path))
    if not _os.path.isabs(expanded):
        expanded = _os.path.join(_os.getcwd(), expanded)
    if any(
            component in (".", "..")
            for component in expanded.split(_os.sep)):
        raise ValueError(
            "{} must use a canonical path without . or .. components"
            .format(label)
        )
    return _Path(expanded)


def _read_stable_read_only_file_at(parent_fd, name, label):
    try:
        descriptor = _os.open(
            name, _CHECKPOINT_FILE_FLAGS, dir_fd=parent_fd
        )
    except OSError as error:
        entry_stat = _checkpoint_entry_stat(parent_fd, name)
        if (entry_stat is not None
                and _stat.S_ISLNK(entry_stat.st_mode)):
            raise ValueError("{} must not be a symlink".format(label))
        raise ValueError("{} is unavailable: {}".format(label, error))
    try:
        before = _os.fstat(descriptor)
        if not _stat.S_ISREG(before.st_mode):
            raise ValueError("{} must be a regular file".format(label))
        if _stat.S_IMODE(before.st_mode) != 0o444:
            raise ValueError("{} must use mode 0444".format(label))
    except BaseException as error:
        _close_checkpoint_descriptor_after_error(descriptor, error)
    try:
        with _fdopen_checkpoint_descriptor(descriptor, "rb") as handle:
            content = handle.read()
            descriptor_stat = _os.fstat(handle.fileno())
        after = _os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False
        )
    except OSError as error:
        raise ValueError("{} could not be read: {}".format(label, error))
    signature = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if (signature(before) != signature(descriptor_stat)
            or signature(before) != signature(after)
            or len(content) != before.st_size):
        raise RuntimeError("{} changed while being read".format(label))
    return content, _hashlib.sha256(content).hexdigest()


def _read_stable_read_only_file(path, label):
    path = _logical_checkpoint_path(path, label)
    parent_fd = _open_checkpoint_parent(path, label, create=False)
    try:
        content, digest = _read_stable_read_only_file_at(
            parent_fd, path.name, label
        )
    finally:
        _os.close(parent_fd)
    return path, content, digest


def _validated_checkpoint_receipt(
        receipt, checkpoint_path, checkpoint_bytes, checkpoint_sha256):
    required = {
        "schema",
        "checkpoint_name",
        "checkpoint_sha256",
        "checkpoint_size",
        "checkpoint_mode",
        "base_checkpoint_sha256",
        "run_id",
        "step",
        "validation_data_accessed",
        "inference_uses_ground_truth",
    }
    if not isinstance(receipt, _Mapping) or set(receipt) != required:
        raise ValueError("query-mask checkpoint receipt fields are invalid")
    if receipt["schema"] != X_QUERY_CHECKPOINT_RECEIPT_SCHEMA:
        raise ValueError("query-mask checkpoint receipt schema is invalid")
    expected = {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": len(checkpoint_bytes),
        "checkpoint_mode": "0444",
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                "query-mask checkpoint receipt {} mismatch".format(key)
            )
    return receipt


def _restore_query_mask_patch_state(model_state, patch_state):
    with torch.no_grad():
        for name, value in patch_state.items():
            model_state[name].copy_(
                value.to(
                    device=model_state[name].device,
                    dtype=model_state[name].dtype,
                )
            )


def load_query_mask_checkpoint(
        path, model, optimizer, *, expected_base_checkpoint):
    """Verify and restore a checkpoint sealed read-only at publication.

    The restore is transactional, but same-owner processes or root can still
    modify or unlink the checkpoint files.
    """
    expected_base = _validated_base_checkpoint_binding(
        expected_base_checkpoint
    )
    checkpoint_path = _logical_checkpoint_path(
        path, "query-mask checkpoint"
    )
    receipt_path = checkpoint_path.with_name(
        checkpoint_path.name + ".receipt.json"
    )
    parent_fd = _open_checkpoint_parent(
        checkpoint_path, "query-mask checkpoint", create=False
    )
    try:
        checkpoint_bytes, checkpoint_sha256 = (
            _read_stable_read_only_file_at(
                parent_fd,
                checkpoint_path.name,
                "query-mask checkpoint",
            )
        )
        receipt_bytes, _receipt_sha256 = (
            _read_stable_read_only_file_at(
                parent_fd,
                receipt_path.name,
                "query-mask checkpoint receipt",
            )
        )
    finally:
        _os.close(parent_fd)
    try:
        receipt = _json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            "query-mask checkpoint receipt is invalid: {}".format(error)
        )
    _validated_checkpoint_receipt(
        receipt, checkpoint_path, checkpoint_bytes, checkpoint_sha256
    )
    if receipt["base_checkpoint_sha256"] != expected_base["sha256"]:
        raise ValueError("query-mask checkpoint receipt base SHA mismatch")

    try:
        artifact = torch.load(_io.BytesIO(checkpoint_bytes), map_location="cpu")
    except Exception as error:
        raise ValueError(
            "query-mask checkpoint could not be deserialized: {}".format(
                error
            )
        )
    _validated_query_mask_checkpoint_payload(artifact)
    if artifact["base_checkpoint"] != expected_base:
        raise ValueError("query-mask checkpoint base binding mismatch")
    if (receipt["run_id"] != artifact["run_record"]["run_id"]
            or receipt["step"] != artifact["run_record"]["step"]):
        raise ValueError("query-mask checkpoint receipt run mismatch")

    selected = _validated_trainable_inventory(model)
    _reject_trainable_frozen_storage_aliases(model, selected)
    _validate_optimizer_parameters(optimizer, selected)
    _validate_optimizer_checkpoint_contract(
        optimizer,
        artifact["optimizer_state_dict"],
        artifact["optimizer_contract"],
        selected,
    )
    model_state = model.state_dict()
    selected_names = tuple(name for name, _parameter in selected)
    expected_patch_names = selected_names + tuple(sorted(
        name for name in model_state
        if name.startswith(X_QUERY_PREFIX) and name not in selected_names
    ))
    patch_state = artifact["x_query_state_dict"]
    if tuple(patch_state) != expected_patch_names:
        raise ValueError("query-mask checkpoint patch names do not match model")
    for name, value in patch_state.items():
        current = model_state[name]
        if current.dtype != value.dtype or tuple(current.shape) != tuple(
                value.shape):
            raise ValueError(
                "query-mask checkpoint tensor {} does not match model".format(
                    name
                )
            )

    before_patch = {
        name: value.detach().clone() for name, value in model_state.items()
        if name.startswith(X_QUERY_PREFIX)
    }
    before_optimizer = _cpu_checkpoint_clone(
        optimizer.state_dict(), "optimizer rollback state"
    )
    try:
        _restore_query_mask_patch_state(model_state, patch_state)
        optimizer.load_state_dict(artifact["optimizer_state_dict"])
        _validate_optimizer_parameters(optimizer, selected)
        _validate_optimizer_checkpoint_contract(
            optimizer,
            optimizer.state_dict(),
            artifact["optimizer_contract"],
            selected,
        )
        model.eval()
    except BaseException as original_error:
        rollback_errors = []
        try:
            _restore_query_mask_patch_state(model_state, before_patch)
        except BaseException as error:
            rollback_errors.append(("patch", error))
        try:
            optimizer.load_state_dict(before_optimizer)
        except BaseException as error:
            rollback_errors.append(("optimizer", error))
        try:
            model.eval()
        except BaseException as error:
            rollback_errors.append(("eval", error))
        if rollback_errors:
            raise QueryMaskCheckpointRollbackError(
                original_error, rollback_errors
            ) from original_error
        raise
    return artifact
