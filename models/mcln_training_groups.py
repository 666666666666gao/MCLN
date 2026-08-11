"""Strict optimizer parameter groups for joint MCLN training."""

import copy
import math
from collections.abc import Mapping


MASK_HEAD_PREFIXES = (
    "x_mask.",
    "x_query.",
    "rel_encoder.",
    "swa_layers.",
    "swa_ffn_layers.",
    "out_norm.",
    "out_score.",
    "query_mask_fusion_calibrator.",
)

STRICT_GROUP_ORDER = ("decoder", "backbone", "mask_head", "selector")


def bare_parameter_name(name):
    return name[7:] if name.startswith("module.") else name


def parameter_group_name(name):
    name = bare_parameter_name(name)
    if (name.startswith("source_choice_selector.")
            or name.startswith("source_moe.")
            or name.startswith("joint_query_quality_reranker.")):
        return "selector"
    if name.startswith("backbone_net."):
        return "backbone"
    if any(name.startswith(prefix) for prefix in MASK_HEAD_PREFIXES):
        return "mask_head"
    if name.startswith("text_encoder."):
        return "frozen_text"
    return "decoder"


def _named_parameters_with_all_paths(module, prefix="", ancestors=()):
    """Yield registered parameter paths without PyTorch's alias deduplication."""
    module_identity = id(module)
    if module_identity in ancestors:
        raise ValueError(
            "model module graph contains a cycle at '{}'".format(
                prefix[:-1] or "<root>"
            )
        )
    next_ancestors = ancestors + (module_identity,)
    for name, parameter in module._parameters.items():
        if parameter is not None:
            yield prefix + name, parameter
    for name, child in module._modules.items():
        if child is not None:
            for path, parameter in _named_parameters_with_all_paths(
                child,
                prefix + name + ".",
                next_ancestors,
            ):
                yield path, parameter


def _positive_finite_float(label, value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("{} must be finite and positive".format(label))
    return float(value)


def build_mcln_optimizer_param_groups(
    model,
    decoder_lr,
    backbone_lr,
    selector_lr,
    mask_head_lr_multiplier,
    require_selector=True,
):
    decoder_lr = _positive_finite_float("decoder_lr", decoder_lr)
    backbone_lr = _positive_finite_float("backbone_lr", backbone_lr)
    selector_lr = _positive_finite_float("selector_lr", selector_lr)
    mask_head_lr_multiplier = _positive_finite_float(
        "mask_head_lr_multiplier", mask_head_lr_multiplier
    )
    mask_head_lr = _positive_finite_float(
        "mask_head_lr",
        decoder_lr * mask_head_lr_multiplier,
    )

    group_order = STRICT_GROUP_ORDER
    buckets = {name: [] for name in group_order}
    names = {name: [] for name in group_order}
    paths_by_identity = {}

    for name, parameter in _named_parameters_with_all_paths(model):
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        bare_name = bare_parameter_name(name)
        if identity in paths_by_identity:
            raise ValueError(
                "shared trainable parameter is registered at '{}' and '{}'".format(
                    paths_by_identity[identity], bare_name
                )
            )
        paths_by_identity[identity] = bare_name

        group_name = parameter_group_name(name)
        if group_name == "frozen_text":
            raise ValueError("text_encoder parameter is unexpectedly trainable")
        buckets[group_name].append(parameter)
        names[group_name].append(bare_name)

    if require_selector and not buckets["selector"]:
        raise ValueError(
            "complete innovation training requires selector parameters"
        )

    lr_by_group = {
        "decoder": decoder_lr,
        "backbone": backbone_lr,
        "mask_head": mask_head_lr,
        "selector": selector_lr,
    }
    groups = []
    for group_name in group_order:
        if not buckets[group_name]:
            continue
        named_parameters = tuple(
            sorted(zip(names[group_name], buckets[group_name]))
        )
        groups.append(
            {
                "name": group_name,
                "params": [parameter for _name, parameter in named_parameters],
                "parameter_names": tuple(
                    name for name, _parameter in named_parameters
                ),
                "lr": lr_by_group[group_name],
            }
        )

    actual = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    expected = set(paths_by_identity)
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer parameter coverage is not exact")
    return groups


def _legacy_migration_error(detail):
    return ValueError(
        "cannot migrate legacy 3-group source-choice optimizer: {}; "
        "restart with --reduce_lr to load model weights while intentionally "
        "resetting optimizer and scheduler state".format(detail)
    )


def _layout_proof_error(detail):
    return ValueError(
        "cannot prove optimizer checkpoint layout: {}; restart with "
        "--reduce_lr to load model weights while intentionally resetting "
        "optimizer and scheduler state".format(detail)
    )


def _validate_matching_optimizer_layout(current_groups, saved_groups):
    saved_identifiers = []
    current_parameter_names = []
    for index, (current_group, saved_group) in enumerate(
        zip(current_groups, saved_groups)
    ):
        if not isinstance(saved_group, Mapping):
            raise _layout_proof_error(
                "checkpoint parameter group {} is not a mapping".format(index)
            )
        current_name = current_group.get("name")
        saved_name = saved_group.get("name")
        if (
            not isinstance(current_name, str)
            or not isinstance(saved_name, str)
            or saved_name != current_name
        ):
            raise _layout_proof_error(
                "ordered group name mismatch at index {} (checkpoint {!r}, "
                "current {!r})".format(index, saved_name, current_name)
            )

        current_names = current_group.get("parameter_names")
        saved_names = saved_group.get("parameter_names")
        if (
            not isinstance(current_names, (list, tuple))
            or not isinstance(saved_names, (list, tuple))
            or tuple(saved_names) != tuple(current_names)
        ):
            raise _layout_proof_error(
                "ordered parameter_names mismatch in group {!r}".format(
                    current_name
                )
            )
        if any(not isinstance(name, str) for name in current_names):
            raise _layout_proof_error(
                "current parameter_names are invalid in group {!r}".format(
                    current_name
                )
            )
        current_parameter_names.extend(current_names)

        saved_parameters = saved_group.get("params")
        if not isinstance(saved_parameters, (list, tuple)):
            raise _layout_proof_error(
                "checkpoint parameter IDs are invalid in group {!r}".format(
                    current_name
                )
            )
        if (
            len(current_group["params"]) != len(current_names)
            or len(saved_parameters) != len(saved_names)
        ):
            raise _layout_proof_error(
                "parameter count does not match parameter_names in group "
                "{!r}".format(current_name)
            )
        for identifier in saved_parameters:
            if not isinstance(identifier, int) or isinstance(identifier, bool):
                raise _layout_proof_error(
                    "checkpoint parameter IDs are invalid in group {!r}".format(
                        current_name
                    )
                )
            saved_identifiers.append(identifier)

    if len(current_parameter_names) != len(set(current_parameter_names)):
        raise _layout_proof_error(
            "current parameter_names contain duplicates"
        )
    if len(saved_identifiers) != len(set(saved_identifiers)):
        raise _layout_proof_error(
            "checkpoint parameter IDs contain duplicates"
        )


def _legacy_optimizer_parameter_groups(model):
    named_parameters = tuple(model.named_parameters())
    return (
        tuple(
            parameter
            for name, parameter in named_parameters
            if "backbone_net" not in name
            and "text_encoder" not in name
            and parameter.requires_grad
        ),
        tuple(
            parameter
            for name, parameter in named_parameters
            if "backbone_net" in name and parameter.requires_grad
        ),
        tuple(
            parameter
            for name, parameter in named_parameters
            if "text_encoder" in name and parameter.requires_grad
        ),
    )


def _serialized_group_parameter_ids(group, group_index, expected_count):
    if not isinstance(group, Mapping):
        raise _legacy_migration_error(
            "checkpoint parameter group {} is not a mapping".format(
                group_index
            )
        )
    if "name" in group or "parameter_names" in group:
        raise _legacy_migration_error(
            "checkpoint parameter group {} contains non-legacy metadata".format(
                group_index
            )
        )
    identifiers = group.get("params")
    if not isinstance(identifiers, (list, tuple)):
        raise _legacy_migration_error(
            "checkpoint parameter group {} has invalid parameter IDs".format(
                group_index
            )
        )
    if len(identifiers) != expected_count:
        raise _legacy_migration_error(
            "checkpoint parameter group sizes do not match the historical "
            "layout (group {} has {}, expected {})".format(
                group_index, len(identifiers), expected_count
            )
        )
    for identifier in identifiers:
        if not isinstance(identifier, int) or isinstance(identifier, bool):
            raise _legacy_migration_error(
                "checkpoint parameter group {} has a non-integer parameter "
                "ID".format(group_index)
            )
    return tuple(identifiers)


def _learning_rate_scale(group, group_index):
    current_lr = group.get("lr")
    base_lr = group.get("initial_lr", current_lr)
    for label, value in (("lr", current_lr), ("initial_lr", base_lr)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise _legacy_migration_error(
                "checkpoint group {} {} is not finite and positive".format(
                    group_index, label
                )
            )
    return float(current_lr) / float(base_lr)


def load_mcln_optimizer_state(optimizer, optimizer_state, model):
    """Load an optimizer state, migrating the exact historical MCLN layout."""
    try:
        saved_groups = optimizer_state["param_groups"]
    except (KeyError, TypeError):
        optimizer.load_state_dict(optimizer_state)
        return None

    if not isinstance(saved_groups, (list, tuple)):
        raise _layout_proof_error("checkpoint param_groups is not a sequence")

    current_groups = optimizer.param_groups
    if len(saved_groups) == len(current_groups):
        _validate_matching_optimizer_layout(current_groups, saved_groups)
        optimizer.load_state_dict(optimizer_state)
        return None

    current_names = tuple(group.get("name") for group in current_groups)
    if len(saved_groups) != 3 or current_names != STRICT_GROUP_ORDER:
        raise _layout_proof_error(
            "checkpoint has {} groups, current optimizer has {}; only the "
            "exact historical 3-group to strict 4-group migration is "
            "supported".format(len(saved_groups), len(current_groups))
        )
    if not isinstance(optimizer_state.get("state"), Mapping):
        raise _legacy_migration_error("checkpoint state is not a mapping")

    legacy_parameter_groups = _legacy_optimizer_parameter_groups(model)
    expected_counts = tuple(len(group) for group in legacy_parameter_groups)
    saved_identifiers_by_group = tuple(
        _serialized_group_parameter_ids(group, index, expected_counts[index])
        for index, group in enumerate(saved_groups)
    )
    saved_identifiers = tuple(
        identifier
        for group in saved_identifiers_by_group
        for identifier in group
    )
    if len(saved_identifiers) != len(set(saved_identifiers)):
        raise _legacy_migration_error(
            "checkpoint parameter IDs are duplicated"
        )
    unknown_state_ids = set(optimizer_state["state"]) - set(saved_identifiers)
    if unknown_state_ids:
        raise _legacy_migration_error(
            "checkpoint contains state for unknown parameter IDs {}".format(
                tuple(sorted(unknown_state_ids))
            )
        )

    legacy_parameters = tuple(
        parameter
        for group in legacy_parameter_groups
        for parameter in group
    )
    current_parameters = tuple(
        parameter
        for group in current_groups
        for parameter in group["params"]
    )
    legacy_parameter_ids = tuple(id(parameter) for parameter in legacy_parameters)
    current_parameter_ids = tuple(id(parameter) for parameter in current_parameters)
    if (
        len(legacy_parameter_ids) != len(set(legacy_parameter_ids))
        or len(current_parameter_ids) != len(set(current_parameter_ids))
        or set(legacy_parameter_ids) != set(current_parameter_ids)
    ):
        raise _legacy_migration_error(
            "current optimizer parameters do not exactly match the historical "
            "trainable parameter inventory"
        )

    current_state = optimizer.state_dict()
    current_serialized_groups = current_state["param_groups"]
    parameter_to_current_identifier = {}
    for live_group, serialized_group in zip(
        current_groups, current_serialized_groups
    ):
        identifiers = serialized_group["params"]
        if len(identifiers) != len(live_group["params"]):
            raise _legacy_migration_error(
                "current optimizer serialization is internally inconsistent"
            )
        for parameter, identifier in zip(live_group["params"], identifiers):
            if id(parameter) in parameter_to_current_identifier:
                raise _legacy_migration_error(
                    "current optimizer contains a duplicated parameter"
                )
            parameter_to_current_identifier[id(parameter)] = identifier

    saved_identifier_to_parameter = {}
    for identifiers, parameters in zip(
        saved_identifiers_by_group, legacy_parameter_groups
    ):
        for identifier, parameter in zip(identifiers, parameters):
            saved_identifier_to_parameter[identifier] = parameter

    migrated_state = {}
    for saved_identifier, value in optimizer_state["state"].items():
        parameter = saved_identifier_to_parameter[saved_identifier]
        current_identifier = parameter_to_current_identifier[id(parameter)]
        migrated_state[current_identifier] = copy.deepcopy(value)

    legacy_source_by_group = {
        "decoder": 0,
        "backbone": 1,
        "mask_head": 0,
        "selector": 0,
    }
    migrated_groups = []
    base_lrs = []
    current_lrs = []
    protected_keys = {
        "params",
        "lr",
        "initial_lr",
        "name",
        "parameter_names",
    }
    for live_group, current_group in zip(
        current_groups, current_serialized_groups
    ):
        group_name = live_group["name"]
        legacy_index = legacy_source_by_group[group_name]
        legacy_group = saved_groups[legacy_index]
        for required_key in ("betas", "eps", "weight_decay", "amsgrad"):
            if required_key not in legacy_group:
                raise _legacy_migration_error(
                    "checkpoint group {} is not an AdamW parameter group "
                    "(missing '{}')".format(legacy_index, required_key)
                )
        migrated_group = copy.deepcopy(current_group)
        for key, value in legacy_group.items():
            if key not in protected_keys:
                migrated_group[key] = copy.deepcopy(value)

        base_lr = current_group.get("initial_lr", current_group["lr"])
        base_lr = _positive_finite_float(
            "current {} group base lr".format(group_name), base_lr
        )
        current_lr = base_lr * _learning_rate_scale(
            legacy_group, legacy_index
        )
        if not math.isfinite(current_lr) or current_lr <= 0.0:
            raise _legacy_migration_error(
                "migrated {} learning rate is not finite and positive".format(
                    group_name
                )
            )
        migrated_group["lr"] = current_lr
        if "initial_lr" in current_group:
            migrated_group["initial_lr"] = base_lr
        migrated_groups.append(migrated_group)
        base_lrs.append(base_lr)
        current_lrs.append(current_lr)

    optimizer.load_state_dict(
        {
            "state": migrated_state,
            "param_groups": migrated_groups,
        }
    )
    return {
        "base_lrs": tuple(base_lrs),
        "current_lrs": tuple(current_lrs),
    }


def migrate_mcln_scheduler_state(scheduler_state, migration):
    """Rebase legacy per-group scheduler lists onto the strict group layout."""
    strict_base_lrs = tuple(migration["base_lrs"])

    def read_lr_field(node, key, path):
        value = node.get(key)
        current_path = path + (key,)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                for item in value
            )
        ):
            raise _legacy_migration_error(
                "scheduler field '{}' does not match the historical "
                "3-group layout".format(".".join(current_path))
            )
        return value, tuple(float(item) for item in value)

    def replace_lr_fields(node, path, current_lrs):
        base_container, _legacy_base_lrs = read_lr_field(
            node, "base_lrs", path
        )
        last_container, _legacy_last_lrs = read_lr_field(
            node, "_last_lr", path
        )
        node["base_lrs"] = type(base_container)(strict_base_lrs)
        node["_last_lr"] = type(last_container)(current_lrs)

    def migrate_cosine(node, path):
        t_max = node.get("T_max")
        eta_min = node.get("eta_min")
        last_epoch = node.get("last_epoch")
        _base_container, legacy_base_lrs = read_lr_field(
            node, "base_lrs", path
        )
        _last_container, legacy_last_lrs = read_lr_field(
            node, "_last_lr", path
        )
        if (
            not isinstance(t_max, int)
            or isinstance(t_max, bool)
            or t_max <= 0
            or not isinstance(last_epoch, int)
            or isinstance(last_epoch, bool)
            or last_epoch < 0
            or not isinstance(eta_min, (int, float))
            or isinstance(eta_min, bool)
            or not math.isfinite(float(eta_min))
        ):
            raise _legacy_migration_error(
                "CosineAnnealingLR state is invalid"
            )
        cosine_factor = (
            1.0 + math.cos(math.pi * float(last_epoch) / float(t_max))
        ) / 2.0
        expected_legacy_lrs = tuple(
            float(eta_min)
            + (base_lr - float(eta_min)) * cosine_factor
            for base_lr in legacy_base_lrs
        )
        if any(
            not math.isclose(
                actual, expected, rel_tol=1e-9, abs_tol=1e-12
            )
            for actual, expected in zip(
                legacy_last_lrs, expected_legacy_lrs
            )
        ):
            raise _legacy_migration_error(
                "CosineAnnealingLR state is not on its closed-form schedule"
            )
        current_lrs = tuple(
            float(eta_min)
            + (base_lr - float(eta_min)) * cosine_factor
            for base_lr in strict_base_lrs
        )
        replace_lr_fields(node, path, current_lrs)
        return current_lrs

    def migrate_multistep(node, path):
        milestones = node.get("milestones")
        gamma = node.get("gamma")
        last_epoch = node.get("last_epoch")
        _base_container, legacy_base_lrs = read_lr_field(
            node, "base_lrs", path
        )
        _last_container, legacy_last_lrs = read_lr_field(
            node, "_last_lr", path
        )
        if (
            not isinstance(milestones, Mapping)
            or not isinstance(gamma, (int, float))
            or isinstance(gamma, bool)
            or not math.isfinite(float(gamma))
            or float(gamma) <= 0.0
            or not isinstance(last_epoch, int)
            or isinstance(last_epoch, bool)
            or last_epoch < 0
        ):
            raise _legacy_migration_error("MultiStepLR state is invalid")
        decay_count = 0
        for milestone, count in milestones.items():
            if (
                not isinstance(milestone, int)
                or isinstance(milestone, bool)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                raise _legacy_migration_error(
                    "MultiStepLR state is invalid"
                )
            if milestone <= last_epoch:
                decay_count += count
        decay_factor = float(gamma) ** decay_count
        expected_legacy_lrs = tuple(
            base_lr * decay_factor for base_lr in legacy_base_lrs
        )
        if any(
            not math.isclose(
                actual, expected, rel_tol=1e-9, abs_tol=1e-12
            )
            for actual, expected in zip(
                legacy_last_lrs, expected_legacy_lrs
            )
        ):
            raise _legacy_migration_error(
                "MultiStepLR state is not on its closed-form schedule"
            )
        current_lrs = tuple(
            base_lr * decay_factor for base_lr in strict_base_lrs
        )
        replace_lr_fields(node, path, current_lrs)
        return current_lrs

    def migrate_warmup(node, path):
        multiplier = node.get("multiplier")
        warmup_epoch = node.get("warmup_epoch")
        last_epoch = node.get("last_epoch")
        finished = node.get("finished")
        after_scheduler = node.get("after_scheduler")
        _base_container, legacy_base_lrs = read_lr_field(
            node, "base_lrs", path
        )
        _last_container, legacy_last_lrs = read_lr_field(
            node, "_last_lr", path
        )
        if (
            not isinstance(multiplier, (int, float))
            or isinstance(multiplier, bool)
            or not math.isfinite(float(multiplier))
            or float(multiplier) <= 1.0
            or not isinstance(warmup_epoch, int)
            or isinstance(warmup_epoch, bool)
            or warmup_epoch <= 0
            or not isinstance(last_epoch, int)
            or isinstance(last_epoch, bool)
            or last_epoch < 0
            or not isinstance(finished, bool)
            or not isinstance(after_scheduler, dict)
        ):
            raise _legacy_migration_error(
                "GradualWarmupScheduler state is invalid"
            )

        _after_base_container, legacy_after_base_lrs = read_lr_field(
            after_scheduler,
            "base_lrs",
            path + ("after_scheduler",),
        )
        if any(
            not math.isclose(
                outer, inner, rel_tol=1e-9, abs_tol=1e-12
            )
            for outer, inner in zip(
                legacy_base_lrs, legacy_after_base_lrs
            )
        ):
            raise _legacy_migration_error(
                "GradualWarmupScheduler base_lrs do not match its "
                "after_scheduler"
            )

        after_current_lrs = migrate_node(
            after_scheduler,
            path + ("after_scheduler",),
        )
        after_last_epoch = after_scheduler.get("last_epoch")
        expected_after_last_epoch = max(0, last_epoch - warmup_epoch)
        if after_last_epoch != expected_after_last_epoch:
            raise _legacy_migration_error(
                "GradualWarmupScheduler epoch does not match its "
                "after_scheduler"
            )

        if last_epoch <= warmup_epoch:
            warmup_factor = (
                (float(multiplier) - 1.0)
                * float(last_epoch)
                / float(warmup_epoch)
                + 1.0
            ) / float(multiplier)
            expected_legacy_lrs = tuple(
                base_lr * warmup_factor for base_lr in legacy_base_lrs
            )
            if any(
                not math.isclose(
                    actual, expected, rel_tol=1e-9, abs_tol=1e-12
                )
                for actual, expected in zip(
                    legacy_last_lrs, expected_legacy_lrs
                )
            ):
                raise _legacy_migration_error(
                    "GradualWarmupScheduler state is not on its warmup "
                    "schedule"
                )
            current_lrs = tuple(
                base_lr * warmup_factor for base_lr in strict_base_lrs
            )
        else:
            current_lrs = after_current_lrs

        replace_lr_fields(node, path, current_lrs)
        return current_lrs

    def migrate_node(node, path):
        if not isinstance(node, dict):
            raise _legacy_migration_error(
                "scheduler state is not a mapping"
            )
        if "after_scheduler" in node:
            return migrate_warmup(node, path)
        if "T_max" in node or "eta_min" in node:
            return migrate_cosine(node, path)
        if "milestones" in node or "gamma" in node:
            return migrate_multistep(node, path)
        raise _legacy_migration_error(
            "scheduler type is not safely supported for 3-group migration"
        )

    migrated = copy.deepcopy(scheduler_state)
    current_lrs = migrate_node(migrated, ())
    migration["current_lrs"] = current_lrs
    return migrated
