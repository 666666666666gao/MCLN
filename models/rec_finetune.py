"""Pure optimization contracts for one-epoch REC fine-tuning."""

import copy
from dataclasses import dataclass
import io
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import tempfile

import torch

from .rec_candidate_adapter import (
    FEATURE_SCHEMA_VERSION,
    attach_candidate_targets,
    build_rec_candidate_batch,
)
from .rec_geometry_reranker import (
    FLAT_PARENT_PRIOR_VERSION,
    REC_GEOMETRY_MODEL_SCHEMA_VERSION,
    blend_rec_geometry_scores,
    build_deployed_parent_state,
    build_rec_geometry_model_inputs,
)
from .rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
    attach_rec_mask_geometry_targets,
    build_rec_mask_geometry_candidates,
)
from .rec_reranker import (
    QueryReranker,
    blend_candidate_scores,
    compute_rec_reranker_loss,
)


MCLN_TRAINABLE_PREFIXES = (
    "decoder.",
    "decoder_query_proj.",
    "proposal_head.",
    "prediction_heads.",
)
_MCLN_TRAINABLE_MODULE_NAMES = tuple(
    prefix[:-1] for prefix in MCLN_TRAINABLE_PREFIXES
)

PRODUCTION_TRAIN_SAMPLE_COUNT = 33040
PRODUCTION_BATCH_SIZE = 18
PRODUCTION_MAX_STEPS = 1836
PRODUCTION_CALIBRATION_INTERVAL = 306
CALIBRATION_STEPS = (0, 306, 612, 918, 1224, 1530, 1836)

REC_FINETUNE_PARENT_SCHEMA = "rec-finetune-parent-v2"
REC_FINETUNE_GEOMETRY_SCHEMA = "rec-finetune-geometry-v2"
REC_FINETUNE_ARTIFACT_VERSION = 2
REC_FINETUNE_PARENT_INPUT_DIM = 152
REC_FINETUNE_GEOMETRY_INPUT_DIM = 179


def rec_finetune_ranking_objective_contract():
    """Return a fresh, exact description of both ranking objectives."""
    return {
        "parent": {
            "name": "single-best-iou-listwise-v1",
            "tier_pairwise_alpha": 0.0,
        },
        "geometry": {
            "name": "best-tier-pairwise-v1",
            "tier_pairwise_alpha": 1.0,
            "thresholds": [0.25, 0.50],
            "threshold_operator": "strict_gt",
            "positive_policy": "all_valid_candidates_in_best_tier",
            "negative_policy": "all_valid_candidates_below_best_tier",
            "loss": "softplus(negative_logit-positive_logit)",
            "pair_reduction": "mean_within_row",
            "row_reduction": "mean_over_informative_rows",
            "no_pair_policy": "differentiable_zero",
        },
    }
AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256 = (
    "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
)
AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256 = (
    "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f"
)

_MODEL_INPUT_KEYS = frozenset((
    "use_color", "use_height", "use_multiview",
    "butd", "butd_gt", "butd_cls",
))
_PRODUCTION_NO_GT_MODEL_INPUTS = {
    "butd": True,
    "butd_gt": False,
    "butd_cls": False,
}
_BACKBONE_CONFIG_KEYS = frozenset((
    "model", "num_target", "num_decoder_layers",
    "self_position_embedding", "self_attend",
    "use_soft_token_loss", "use_contrastive_align",
    "detect_intermediate", "use_source_choice_selector",
    "source_choice_selector_sources", "source_choice_selector_hidden_dim",
))
_CALIBRATION_METRIC_FIELDS = frozenset((
    "sample_count", "hits025", "hits050", "acc025", "acc050", "score",
))
_CALIBRATION_HISTORY_FIELDS = frozenset((
    "step", "metrics", "eligible", "regression", "action", "best_step",
))
_PROVENANCE_FIELDS = frozenset((
    "initial_backbone_sha256",
    "initial_parent_artifact_sha256",
    "initial_geometry_artifact_sha256",
    "authoritative_split_mapping_sha256",
    "selected_step",
    "validation_data_accessed",
    "normalization_policy",
    "parent_reranker_weight",
    "geometry_reranker_weight",
    "matcher_costs",
    "loss_scales",
    "reranker_loss_weights",
    "reranker_ranking_objectives",
    "optimizer_groups",
    "max_steps",
    "calibration_steps",
    "mcln_trainable_parameter_names",
    "calibration_history",
))
_MODEL_CONFIG_FIELDS = frozenset(("input_dim", "hidden_dim", "dropout"))
_PARENT_ARTIFACT_FIELDS = frozenset((
    "schema", "version", "model_state_dict", "model_state_sha256",
    "model_config", "adapter_schema_version", "input_dim",
    "feature_names", "feature_mean", "feature_std",
    "feature_mean_sha256", "feature_std_sha256", "candidate_rule",
    "checkpoint_sha256", "checkpoint_epoch", "target_iou_policy",
    "model_inputs", "backbone_config", "score_mode", "reranker_weight",
    "provenance", "calibration_metrics",
))
_PARENT_RUNTIME_BINDING_FIELDS = frozenset((
    "schema", "version", "model_state_sha256", "model_config",
    "adapter_schema_version", "input_dim", "feature_names",
    "feature_mean_sha256", "feature_std_sha256", "candidate_rule",
    "checkpoint_sha256", "checkpoint_epoch", "target_iou_policy",
    "model_inputs", "backbone_config", "score_mode", "reranker_weight",
))
_PARENT_INFERENCE_CONTRACT_FIELDS = frozenset((
    "schema", "version", "device_type", "device_index", "local_batch_size",
    "world_size", "row_order", "remainder_policy", "feature_source",
    "dtype", "autocast", "allow_tf32", "eval", "no_grad",
    "score_builder", "score_builder_version", "canonical_query_tie_policy",
    "content_digest_version", "row_count", "score_content_sha256",
))
_GEOMETRY_ARTIFACT_FIELDS = frozenset((
    "schema", "version", "model_state_dict", "model_state_sha256",
    "model_config", "model_schema_version", "geometry_schema_version",
    "input_dim", "feature_names", "feature_mean", "feature_std",
    "feature_mean_sha256", "feature_std_sha256", "variant_names",
    "variant_configs", "regressed_variant_index", "min_points",
    "max_point_fraction", "candidate_rule", "checkpoint_sha256",
    "checkpoint_epoch", "target_iou_policy", "model_inputs",
    "backbone_config", "parent_artifact_sha256", "parent_runtime_binding",
    "parent_inference_contract", "num_queries", "flat_parent_prior_version",
    "tie_policy", "score_mode", "geometry_weight",
    "evaluator_filter_policy", "filter_non_gt_boxes", "provenance",
    "calibration_metrics",
))

_NORMALIZATION_POLICY = "fixed-initial-artifact-v1"
_PARENT_SCORE_MODE = "rank_blend"
_PARENT_WEIGHT = 0.9
_GEOMETRY_SCORE_MODE = "parent-flat-rank-blend-v1"
_GEOMETRY_WEIGHT = 1.0
_GEOMETRY_TIE_POLICY = "score-desc-flat-index-asc-v1"
_EVALUATOR_FILTER_POLICY = "evaluator-valid-no-gt-filter-v1"
_TARGET_IOU_POLICY = "root_only"
_OPTIMIZER_CONTRACT = (
    ("mcln_decoder_box", 2e-5, 5e-4, 0.1),
    ("parent_reranker", 1e-3, 1e-4, 1.0),
    ("geometry_reranker", 3e-4, 1e-4, 1.0),
)

AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0 = {
    "split_seed": 0,
    "calibration_fraction": 0.10,
    "scene_count": 562,
    "fit_scene_count": 506,
    "calibration_scene_count": 56,
    "sample_count": 36665,
    "fit_sample_count": 33040,
    "calibration_sample_count": 3625,
    "fit_scene_sha256": (
        "790264c59d4e4f5937b49b0440c020d485c0929a843176a3a434f2ce8d797a17"
    ),
    "calibration_scene_sha256": (
        "f58524379488c4bd061849167f537ba3a10671317b30c89dd580ba147e8e5cdc"
    ),
    "mapping_sha256": (
        "72685aa01285dbe72b9e0331acd5f10457f773e9e158ae4f884b9c4176cf95bd"
    ),
}


def _stable_artifact_snapshot(path, label):
    if (not isinstance(path, (str, os.PathLike))
            or isinstance(path, bytes)):
        raise ValueError("{} path must be path-like".format(label))
    try:
        logical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("{} path must be path-like: {}".format(label, error))
    try:
        initial = os.lstat(str(logical))
    except OSError as error:
        raise ValueError("{} does not exist: {}".format(label, error))
    if stat.S_ISLNK(initial.st_mode):
        raise ValueError("{} must not be a symlink".format(label))
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("{} must be a regular file".format(label))

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(logical), flags)
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final = os.lstat(str(logical))
    except OSError as error:
        raise ValueError("could not read {}: {}".format(label, error))
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identities = (
        identity(initial), identity(before), identity(after), identity(final)
    )
    if (stat.S_ISLNK(final.st_mode) or not stat.S_ISREG(final.st_mode)
            or len(set(identities)) != 1):
        raise ValueError("{} changed during stable snapshot load".format(label))
    snapshot = b"".join(chunks)
    resolved = logical.resolve()
    return resolved, snapshot, hashlib.sha256(snapshot).hexdigest()


def rec_finetune_artifact_schema(path):
    """Return a stable artifact schema identifier without constructing a model."""
    _resolved, snapshot, _sha256 = _stable_artifact_snapshot(
        path, "REC reranker artifact"
    )
    try:
        artifact = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize REC reranker artifact: {}".format(
            error
        ))
    if not isinstance(artifact, dict):
        return None
    return artifact.get("schema")


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value, label):
    if not _is_sha256(value):
        raise ValueError("{} SHA-256 is invalid".format(label))
    return value


def _tensor_sha256(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("tensor digest input must be a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("model state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(name, str) or not name:
            raise ValueError("model state names are invalid")
        if (not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all().item())):
            raise ValueError("model state tensors must be finite CPU float32")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cpu_float_state_dict(model):
    if not isinstance(model, QueryReranker):
        raise ValueError("artifact model must be a QueryReranker")
    result = {
        name: value.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, value in model.state_dict().items()
    }
    _state_dict_sha256(result)
    return result


def _model_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            return float(module.p)
    return 0.0


def _model_config(model):
    if (not isinstance(model, QueryReranker)
            or not isinstance(model.input_dim, int)
            or not isinstance(model.hidden_dim, int)):
        raise ValueError("artifact model must be a configured QueryReranker")
    return {
        "input_dim": int(model.input_dim),
        "hidden_dim": int(model.hidden_dim),
        "dropout": _model_dropout(model),
    }


def _validate_model_config(value, input_dim, label):
    if (not isinstance(value, dict)
            or set(value) != _MODEL_CONFIG_FIELDS
            or value.get("input_dim") != input_dim
            or not isinstance(value.get("hidden_dim"), int)
            or isinstance(value.get("hidden_dim"), bool)
            or value["hidden_dim"] <= 0
            or not isinstance(value.get("dropout"), (int, float))
            or isinstance(value.get("dropout"), bool)
            or not math.isfinite(float(value["dropout"]))
            or not 0.0 <= float(value["dropout"]) < 1.0):
        raise ValueError("{} model configuration is invalid".format(label))
    return {
        "input_dim": input_dim,
        "hidden_dim": value["hidden_dim"],
        "dropout": float(value["dropout"]),
    }


def _normalization_from_initial(initial_artifact, input_dim, label):
    mean = initial_artifact.get("feature_mean")
    std = initial_artifact.get("feature_std")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or not torch.is_floating_point(mean)
            or not torch.is_floating_point(std)
            or tuple(mean.shape) != (input_dim,)
            or tuple(std.shape) != (input_dim,)
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(std).all().item())
            or bool((std <= 0.0).any().item())):
        raise ValueError("{} initial normalization is invalid".format(label))
    return (
        mean.detach().to(device="cpu", dtype=torch.float32).clone(),
        std.detach().to(device="cpu", dtype=torch.float32).clone(),
    )


def _validate_normalization(artifact, input_dim, label):
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or mean.device.type != "cpu" or std.device.type != "cpu"
            or mean.dtype != torch.float32 or std.dtype != torch.float32
            or tuple(mean.shape) != (input_dim,)
            or tuple(std.shape) != (input_dim,)
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(std).all().item())
            or bool((std <= 0.0).any().item())
            or artifact.get("feature_mean_sha256") != _tensor_sha256(mean)
            or artifact.get("feature_std_sha256") != _tensor_sha256(std)):
        raise ValueError("{} normalization binding is invalid".format(label))


def _validate_runtime_metadata(value, input_dim, label,
                               require_checkpoint_epoch=True):
    names = value.get("feature_names")
    if (not isinstance(input_dim, int) or isinstance(input_dim, bool)
            or input_dim <= 0
            or not isinstance(names, list)
            or len(names) != input_dim
            or len(set(names)) != input_dim
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("{} feature metadata is invalid".format(label))
    rule = value.get("candidate_rule")
    if (not isinstance(rule, dict)
            or set(rule) != {"topk_per_source", "max_candidates"}
            or not isinstance(rule.get("topk_per_source"), int)
            or isinstance(rule.get("topk_per_source"), bool)
            or rule["topk_per_source"] <= 0
            or rule.get("max_candidates") != 16):
        raise ValueError("{} candidate rule is invalid".format(label))
    model_inputs = value.get("model_inputs")
    backbone = value.get("backbone_config")
    if (not isinstance(model_inputs, dict)
            or set(model_inputs) != _MODEL_INPUT_KEYS
            or any(not isinstance(item, bool) for item in model_inputs.values())):
        raise ValueError("{} model inputs are invalid".format(label))
    if any(model_inputs[key] is not expected
           for key, expected in _PRODUCTION_NO_GT_MODEL_INPUTS.items()):
        raise ValueError("{} no-GT model inputs are invalid".format(label))
    if (not isinstance(backbone, dict)
            or set(backbone) != _BACKBONE_CONFIG_KEYS):
        raise ValueError("{} backbone configuration is invalid".format(label))
    if value.get("target_iou_policy") != _TARGET_IOU_POLICY:
        raise ValueError("{} target policy is invalid".format(label))
    _require_sha256(value.get("checkpoint_sha256"), label + " checkpoint")
    epoch = value.get("checkpoint_epoch")
    if (require_checkpoint_epoch
            and (not isinstance(epoch, int) or isinstance(epoch, bool)
                 or epoch < 0)):
        raise ValueError("{} checkpoint epoch is invalid".format(label))


def _validate_calibration_metrics(value):
    if not isinstance(value, dict) or set(value) != _CALIBRATION_METRIC_FIELDS:
        raise ValueError("calibration metrics fields do not match schema")
    count = value["sample_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("calibration metrics sample count is invalid")
    for key in ("hits025", "hits050"):
        hits = value[key]
        if (not isinstance(hits, int) or isinstance(hits, bool)
                or not 0 <= hits <= count):
            raise ValueError("calibration metrics hit count is invalid")
    if value["hits050"] > value["hits025"]:
        raise ValueError("calibration metrics thresholds are inconsistent")
    for key in ("acc025", "acc050", "score"):
        numeric = value[key]
        if (not isinstance(numeric, (int, float))
                or isinstance(numeric, bool)
                or not math.isfinite(float(numeric))):
            raise ValueError("calibration metrics must be finite")
    acc025 = value["hits025"] / float(count)
    acc050 = value["hits050"] / float(count)
    expected_score = _calibration_score(acc025, acc050)
    if (not math.isclose(float(value["acc025"]), acc025,
                         rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(value["acc050"]), acc050,
                                rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(value["score"]), expected_score,
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("calibration metrics are internally inconsistent")
    return {
        "sample_count": count,
        "hits025": value["hits025"],
        "hits050": value["hits050"],
        "acc025": float(value["acc025"]),
        "acc050": float(value["acc050"]),
        "score": float(value["score"]),
    }


def _exact_numeric_mapping(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("{} fields do not match schema".format(label))
    for key, required in expected.items():
        actual = value[key]
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(float(actual))
                or float(actual) != float(required)):
            raise ValueError("{} {} is invalid".format(label, key))


def _validate_provenance(value, calibration_metrics):
    if not isinstance(value, dict) or set(value) != _PROVENANCE_FIELDS:
        raise ValueError("online provenance fields do not match exact schema")
    expected_lineage = {
        "initial_backbone_sha256": (
            AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
        ),
        "initial_parent_artifact_sha256": (
            AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256
        ),
        "initial_geometry_artifact_sha256": (
            AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256
        ),
    }
    for key, expected in expected_lineage.items():
        _require_sha256(value[key], "online provenance " + key)
        if value[key] != expected:
            raise ValueError(
                "online provenance {} is not authoritative".format(key)
            )
    expected_mapping = AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0["mapping_sha256"]
    if value["authoritative_split_mapping_sha256"] != expected_mapping:
        raise ValueError("online provenance authoritative split is invalid")
    selected_step = value["selected_step"]
    if (not isinstance(selected_step, int) or isinstance(selected_step, bool)
            or selected_step not in CALIBRATION_STEPS):
        raise ValueError("online provenance selected step is invalid")
    if value["validation_data_accessed"] is not False:
        raise ValueError("online provenance validation data flag is invalid")
    if value["normalization_policy"] != _NORMALIZATION_POLICY:
        raise ValueError("online provenance normalization policy is invalid")
    _exact_numeric_mapping(
        {"parent": value["parent_reranker_weight"],
         "geometry": value["geometry_reranker_weight"]},
        {"parent": _PARENT_WEIGHT, "geometry": _GEOMETRY_WEIGHT},
        "online provenance blend weights",
    )
    _exact_numeric_mapping(
        value["matcher_costs"], {"class": 1.0, "bbox": 5.0, "giou": 2.0},
        "online provenance matcher costs",
    )
    _exact_numeric_mapping(
        value["loss_scales"],
        {"mask": 0.1, "consistency": 0.1, "source_choice": 0.0,
         "parent": 1.0, "geometry": 1.0},
        "online provenance loss scales",
    )
    _exact_numeric_mapping(
        value["reranker_loss_weights"],
        {"ranking": 1.0, "threshold": 1.0, "iou": 0.5},
        "online provenance reranker loss weights",
    )
    expected_ranking_objectives = {
        "parent": {
            "name": "single-best-iou-listwise-v1",
            "tier_pairwise_alpha": 0.0,
        },
        "geometry": {
            "name": "best-tier-pairwise-v1",
            "tier_pairwise_alpha": 1.0,
            "thresholds": [0.25, 0.50],
            "threshold_operator": "strict_gt",
            "positive_policy": "all_valid_candidates_in_best_tier",
            "negative_policy": "all_valid_candidates_below_best_tier",
            "loss": "softplus(negative_logit-positive_logit)",
            "pair_reduction": "mean_within_row",
            "row_reduction": "mean_over_informative_rows",
            "no_pair_policy": "differentiable_zero",
        },
    }
    if value["reranker_ranking_objectives"] != expected_ranking_objectives:
        raise ValueError("online provenance ranking objective is invalid")
    groups = value["optimizer_groups"]
    if not isinstance(groups, list) or len(groups) != len(_OPTIMIZER_CONTRACT):
        raise ValueError("online provenance optimizer groups are invalid")
    for group, expected in zip(groups, _OPTIMIZER_CONTRACT):
        name, lr, weight_decay, grad_clip = expected
        if not isinstance(group, dict) or set(group) != {
                "name", "lr", "weight_decay", "grad_clip"}:
            raise ValueError("online provenance optimizer group is invalid")
        if group["name"] != name:
            raise ValueError("online provenance optimizer group name is invalid")
        _exact_numeric_mapping(
            {key: group[key] for key in ("lr", "weight_decay", "grad_clip")},
            {"lr": lr, "weight_decay": weight_decay,
             "grad_clip": grad_clip},
            "online provenance optimizer group",
        )
    if value["max_steps"] != PRODUCTION_MAX_STEPS:
        raise ValueError("online provenance maximum step is invalid")
    if value["calibration_steps"] != list(CALIBRATION_STEPS):
        raise ValueError("online provenance calibration steps are invalid")
    names = value["mcln_trainable_parameter_names"]
    if (not isinstance(names, list) or not names or len(names) != len(set(names))
            or any(not isinstance(name, str) or not _is_allowed_mcln_name(name)
                   for name in names)):
        raise ValueError("online provenance MCLN allowlist names are invalid")

    history = value["calibration_history"]
    if (not isinstance(history, list) or not history
            or len(history) > len(CALIBRATION_STEPS)):
        raise ValueError("online provenance calibration history is invalid")
    baseline = None
    previous = None
    best_metrics = None
    best_step = None
    selected_metrics = None
    stopped = False
    for index, record in enumerate(history):
        if (not isinstance(record, dict)
                or set(record) != _CALIBRATION_HISTORY_FIELDS
                or record["step"] != CALIBRATION_STEPS[index]
                or stopped):
            raise ValueError("online provenance calibration history step is invalid")
        metrics = _validate_calibration_metrics(record["metrics"])
        if baseline is None:
            eligible = True
            regression = False
            best_metrics = metrics
            best_step = record["step"]
            baseline = metrics
        else:
            eligible = (
                metrics["acc025"] >= baseline["acc025"]
                and metrics["acc050"] >= baseline["acc050"]
            )
            regression = not eligible or metrics["score"] < previous["score"]
            if (not regression and eligible
                    and metrics["score"] > best_metrics["score"]):
                best_metrics = metrics
                best_step = record["step"]
        action = "stop" if regression else "continue"
        if (record["eligible"] is not eligible
                or record["regression"] is not regression
                or record["action"] != action
                or record["best_step"] != best_step):
            raise ValueError(
                "online provenance calibration history is inconsistent"
            )
        if record["step"] == selected_step:
            selected_metrics = metrics
        previous = metrics
        stopped = regression
    if (best_step != selected_step or selected_metrics is None
            or selected_metrics != _validate_calibration_metrics(
                calibration_metrics)):
        raise ValueError("online provenance selected step metrics are inconsistent")


def _validate_initial_common(initial_artifact, model, label,
                             require_checkpoint_epoch=True):
    if not isinstance(initial_artifact, dict):
        raise ValueError("{} initial artifact must be a mapping".format(label))
    input_dim = initial_artifact.get("input_dim")
    if (not isinstance(model, QueryReranker)
            or model.input_dim != input_dim):
        raise ValueError("{} model differs from initial input dimension".format(
            label
        ))
    _validate_runtime_metadata(
        initial_artifact,
        input_dim,
        label + " initial",
        require_checkpoint_epoch=require_checkpoint_epoch,
    )
    return input_dim


def _load_authoritative_initial_artifact(path, expected_sha256, label):
    if (not isinstance(path, (str, os.PathLike))
            or isinstance(path, bytes)):
        raise ValueError("{} initial artifact must be a path or path-like".format(
            label
        ))
    _resolved, snapshot, actual_sha256 = _stable_artifact_snapshot(
        path, "{} initial artifact".format(label)
    )
    if actual_sha256 != expected_sha256:
        raise ValueError("{} initial artifact SHA mismatch".format(label))
    return _deserialize_artifact(
        snapshot, "{} initial artifact".format(label)
    )


def _parent_runtime_binding(parent_artifact):
    return {
        key: copy.deepcopy(parent_artifact[key])
        for key in _PARENT_RUNTIME_BINDING_FIELDS
    }


def _parent_inference_contract(parent_binding, row_count):
    return {
        "schema": "rec-parent-inference-contract",
        "version": 1,
        "device_type": "cuda",
        "device_index": 0,
        "local_batch_size": 12,
        "world_size": 1,
        "row_order": "dataset-index-contiguous",
        "remainder_policy": "natural-remainder",
        "feature_source": "online-bound-backbone-features",
        "dtype": "float32",
        "autocast": False,
        "allow_tf32": True,
        "eval": True,
        "no_grad": True,
        "score_builder": "normalized-query-reranker-rank-blend",
        "score_builder_version": 1,
        "canonical_query_tie_policy": "score-desc-query-index-asc-v1",
        "content_digest_version": "parent-runtime-binding-canonical-json-v1",
        "row_count": int(row_count),
        "score_content_sha256": canonical_json_sha256(parent_binding),
    }


def build_rec_finetune_parent_artifact(
        model, initial_artifact, checkpoint_sha256, checkpoint_epoch,
        provenance, calibration_metrics):
    """Build one exact online-fine-tuned parent reranker artifact."""
    initial_artifact = _load_authoritative_initial_artifact(
        initial_artifact,
        AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256,
        "parent",
    )
    input_dim = _validate_initial_common(
        initial_artifact,
        model,
        "parent",
        require_checkpoint_epoch=False,
    )
    if input_dim != REC_FINETUNE_PARENT_INPUT_DIM:
        raise ValueError("parent artifact input dimension must be 152")
    if initial_artifact.get("adapter_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("parent initial adapter schema is invalid")
    if initial_artifact.get("score_mode") != _PARENT_SCORE_MODE:
        raise ValueError("parent initial score mode is invalid")
    if float(initial_artifact.get("reranker_weight", -1.0)) != _PARENT_WEIGHT:
        raise ValueError("parent initial blend weight is invalid")
    _require_sha256(checkpoint_sha256, "fine-tuned checkpoint")
    if (not isinstance(checkpoint_epoch, int) or isinstance(checkpoint_epoch, bool)
            or checkpoint_epoch < 0):
        raise ValueError("fine-tuned checkpoint epoch is invalid")
    metrics = _validate_calibration_metrics(calibration_metrics)
    provenance_copy = copy.deepcopy(provenance)
    _validate_provenance(provenance_copy, metrics)
    if (initial_artifact.get("checkpoint_sha256")
            != AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256):
        raise ValueError("parent initial backbone SHA lineage is invalid")
    mean, std = _normalization_from_initial(
        initial_artifact, input_dim, "parent"
    )
    state = _cpu_float_state_dict(model)
    artifact = {
        "schema": REC_FINETUNE_PARENT_SCHEMA,
        "version": REC_FINETUNE_ARTIFACT_VERSION,
        "model_state_dict": state,
        "model_state_sha256": _state_dict_sha256(state),
        "model_config": _model_config(model),
        "adapter_schema_version": initial_artifact["adapter_schema_version"],
        "input_dim": input_dim,
        "feature_names": copy.deepcopy(initial_artifact["feature_names"]),
        "feature_mean": mean,
        "feature_std": std,
        "feature_mean_sha256": _tensor_sha256(mean),
        "feature_std_sha256": _tensor_sha256(std),
        "candidate_rule": copy.deepcopy(initial_artifact["candidate_rule"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "target_iou_policy": initial_artifact["target_iou_policy"],
        "model_inputs": copy.deepcopy(initial_artifact["model_inputs"]),
        "backbone_config": copy.deepcopy(initial_artifact["backbone_config"]),
        "score_mode": _PARENT_SCORE_MODE,
        "reranker_weight": _PARENT_WEIGHT,
        "provenance": provenance_copy,
        "calibration_metrics": copy.deepcopy(metrics),
    }
    _validate_parent_artifact(artifact)
    return artifact


def build_rec_finetune_geometry_artifact(
        model, initial_artifact, parent_artifact, parent_artifact_sha256,
        checkpoint_sha256, checkpoint_epoch, provenance,
        calibration_metrics):
    """Build one geometry artifact bound to the new backbone and parent."""
    initial_artifact = _load_authoritative_initial_artifact(
        initial_artifact,
        AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256,
        "geometry",
    )
    input_dim = _validate_initial_common(initial_artifact, model, "geometry")
    if input_dim != REC_FINETUNE_GEOMETRY_INPUT_DIM:
        raise ValueError("geometry artifact input dimension must be 179")
    _validate_parent_artifact(parent_artifact)
    _require_sha256(parent_artifact_sha256, "parent artifact")
    _require_sha256(checkpoint_sha256, "fine-tuned checkpoint")
    if (not isinstance(checkpoint_epoch, int) or isinstance(checkpoint_epoch, bool)
            or checkpoint_epoch < 0):
        raise ValueError("fine-tuned checkpoint epoch is invalid")
    metrics = _validate_calibration_metrics(calibration_metrics)
    provenance_copy = copy.deepcopy(provenance)
    _validate_provenance(provenance_copy, metrics)
    if (initial_artifact.get("checkpoint_sha256")
            != AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256):
        raise ValueError("geometry initial backbone SHA lineage is invalid")
    expected_initial = {
        "model_schema_version": REC_GEOMETRY_MODEL_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "regressed_variant_index": 0,
        "num_queries": 256,
        "flat_parent_prior_version": FLAT_PARENT_PRIOR_VERSION,
        "tie_policy": _GEOMETRY_TIE_POLICY,
        "score_mode": _GEOMETRY_SCORE_MODE,
        "evaluator_filter_policy": _EVALUATOR_FILTER_POLICY,
        "filter_non_gt_boxes": False,
    }
    if any(initial_artifact.get(key) != expected
           for key, expected in expected_initial.items()):
        raise ValueError("geometry initial runtime policy is invalid")
    if float(initial_artifact.get("geometry_weight", -1.0)) != _GEOMETRY_WEIGHT:
        raise ValueError("geometry initial blend weight is invalid")
    variants = initial_artifact.get("variant_configs")
    variant_names = initial_artifact.get("variant_names")
    expected_variants = [dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    if (variants != expected_variants
            or variant_names != [value["name"] for value in expected_variants]
            or not isinstance(initial_artifact.get("min_points"), int)
            or isinstance(initial_artifact.get("min_points"), bool)
            or initial_artifact["min_points"] <= 0
            or not isinstance(initial_artifact.get("max_point_fraction"),
                              (int, float))
            or isinstance(initial_artifact.get("max_point_fraction"), bool)
            or not math.isfinite(float(initial_artifact["max_point_fraction"]))
            or not 0.0 < float(initial_artifact["max_point_fraction"]) <= 1.0):
        raise ValueError("geometry initial variant configuration is invalid")
    expected_names = (
        parent_artifact["feature_names"]
        + list(REC_MASK_GEOMETRY_FEATURE_NAMES)
        + ["parent_score", "parent_is_deployed_top1"]
    )
    if initial_artifact.get("feature_names") != expected_names:
        raise ValueError("geometry initial features do not bind the parent")
    for key in ("candidate_rule", "target_iou_policy", "model_inputs",
                "backbone_config"):
        if initial_artifact.get(key) != parent_artifact.get(key):
            raise ValueError("geometry runtime metadata differs from parent")
    if (checkpoint_sha256 != parent_artifact["checkpoint_sha256"]
            or checkpoint_epoch != parent_artifact["checkpoint_epoch"]
            or provenance_copy != parent_artifact["provenance"]
            or metrics != parent_artifact["calibration_metrics"]):
        raise ValueError("geometry fine-tune selection differs from parent")
    mean, std = _normalization_from_initial(
        initial_artifact, input_dim, "geometry"
    )
    state = _cpu_float_state_dict(model)
    parent_binding = _parent_runtime_binding(parent_artifact)
    artifact = {
        "schema": REC_FINETUNE_GEOMETRY_SCHEMA,
        "version": REC_FINETUNE_ARTIFACT_VERSION,
        "model_state_dict": state,
        "model_state_sha256": _state_dict_sha256(state),
        "model_config": _model_config(model),
        "model_schema_version": REC_GEOMETRY_MODEL_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "input_dim": input_dim,
        "feature_names": copy.deepcopy(initial_artifact["feature_names"]),
        "feature_mean": mean,
        "feature_std": std,
        "feature_mean_sha256": _tensor_sha256(mean),
        "feature_std_sha256": _tensor_sha256(std),
        "variant_names": copy.deepcopy(variant_names),
        "variant_configs": copy.deepcopy(variants),
        "regressed_variant_index": 0,
        "min_points": int(initial_artifact["min_points"]),
        "max_point_fraction": float(initial_artifact["max_point_fraction"]),
        "candidate_rule": copy.deepcopy(initial_artifact["candidate_rule"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "target_iou_policy": initial_artifact["target_iou_policy"],
        "model_inputs": copy.deepcopy(initial_artifact["model_inputs"]),
        "backbone_config": copy.deepcopy(initial_artifact["backbone_config"]),
        "parent_artifact_sha256": parent_artifact_sha256,
        "parent_runtime_binding": parent_binding,
        "parent_inference_contract": _parent_inference_contract(
            parent_binding, metrics["sample_count"]
        ),
        "num_queries": 256,
        "flat_parent_prior_version": FLAT_PARENT_PRIOR_VERSION,
        "tie_policy": _GEOMETRY_TIE_POLICY,
        "score_mode": _GEOMETRY_SCORE_MODE,
        "geometry_weight": _GEOMETRY_WEIGHT,
        "evaluator_filter_policy": _EVALUATOR_FILTER_POLICY,
        "filter_non_gt_boxes": False,
        "provenance": provenance_copy,
        "calibration_metrics": copy.deepcopy(metrics),
    }
    validate_rec_finetune_artifact_pair(
        parent_artifact, artifact, parent_artifact_sha256
    )
    return artifact


def _validate_parent_artifact(artifact):
    if not isinstance(artifact, dict) or set(artifact) != _PARENT_ARTIFACT_FIELDS:
        raise ValueError("fine-tune parent artifact fields do not match schema")
    if (artifact.get("schema") != REC_FINETUNE_PARENT_SCHEMA
            or artifact.get("version") != REC_FINETUNE_ARTIFACT_VERSION
            or isinstance(artifact.get("version"), bool)):
        raise ValueError("fine-tune parent artifact schema is invalid")
    input_dim = artifact.get("input_dim")
    if input_dim != REC_FINETUNE_PARENT_INPUT_DIM:
        raise ValueError("fine-tune parent input dimension must be 152")
    _validate_runtime_metadata(artifact, input_dim, "parent artifact")
    if (artifact.get("adapter_schema_version") != FEATURE_SCHEMA_VERSION
            or artifact.get("score_mode") != _PARENT_SCORE_MODE
            or artifact.get("reranker_weight") != _PARENT_WEIGHT):
        raise ValueError("fine-tune parent runtime policy is invalid")
    _validate_normalization(artifact, input_dim, "parent artifact")
    state_hash = _state_dict_sha256(artifact.get("model_state_dict"))
    if artifact.get("model_state_sha256") != state_hash:
        raise ValueError("parent artifact model state binding is invalid")
    config = _validate_model_config(
        artifact.get("model_config"), input_dim, "parent artifact"
    )
    metrics = _validate_calibration_metrics(artifact.get("calibration_metrics"))
    _validate_provenance(artifact.get("provenance"), metrics)
    return config


def _validate_parent_inference_contract(value, binding, row_count):
    expected = _parent_inference_contract(binding, row_count)
    if (not isinstance(value, dict)
            or set(value) != _PARENT_INFERENCE_CONTRACT_FIELDS
            or value != expected):
        raise ValueError("parent inference contract is not runtime-compatible")


def _validate_geometry_artifact(artifact):
    if (not isinstance(artifact, dict)
            or set(artifact) != _GEOMETRY_ARTIFACT_FIELDS):
        raise ValueError("fine-tune geometry artifact fields do not match schema")
    if (artifact.get("schema") != REC_FINETUNE_GEOMETRY_SCHEMA
            or artifact.get("version") != REC_FINETUNE_ARTIFACT_VERSION
            or isinstance(artifact.get("version"), bool)):
        raise ValueError("fine-tune geometry artifact schema is invalid")
    input_dim = artifact.get("input_dim")
    if input_dim != REC_FINETUNE_GEOMETRY_INPUT_DIM:
        raise ValueError("fine-tune geometry input dimension must be 179")
    _validate_runtime_metadata(artifact, input_dim, "geometry artifact")
    expected = {
        "model_schema_version": REC_GEOMETRY_MODEL_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "regressed_variant_index": 0,
        "num_queries": 256,
        "flat_parent_prior_version": FLAT_PARENT_PRIOR_VERSION,
        "tie_policy": _GEOMETRY_TIE_POLICY,
        "score_mode": _GEOMETRY_SCORE_MODE,
        "geometry_weight": _GEOMETRY_WEIGHT,
        "evaluator_filter_policy": _EVALUATOR_FILTER_POLICY,
        "filter_non_gt_boxes": False,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("fine-tune geometry runtime policy is invalid")
    expected_variants = [dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    if (artifact.get("variant_configs") != expected_variants
            or artifact.get("variant_names")
            != [value["name"] for value in expected_variants]
            or not isinstance(artifact.get("min_points"), int)
            or isinstance(artifact.get("min_points"), bool)
            or artifact["min_points"] <= 0
            or not isinstance(artifact.get("max_point_fraction"), (int, float))
            or isinstance(artifact.get("max_point_fraction"), bool)
            or not math.isfinite(float(artifact["max_point_fraction"]))
            or not 0.0 < float(artifact["max_point_fraction"]) <= 1.0):
        raise ValueError("fine-tune geometry variant configuration is invalid")
    _validate_normalization(artifact, input_dim, "geometry artifact")
    state_hash = _state_dict_sha256(artifact.get("model_state_dict"))
    if artifact.get("model_state_sha256") != state_hash:
        raise ValueError("geometry artifact model state binding is invalid")
    config = _validate_model_config(
        artifact.get("model_config"), input_dim, "geometry artifact"
    )
    _require_sha256(artifact.get("parent_artifact_sha256"), "parent artifact")
    binding = artifact.get("parent_runtime_binding")
    if (not isinstance(binding, dict)
            or set(binding) != _PARENT_RUNTIME_BINDING_FIELDS):
        raise ValueError("geometry parent runtime binding is invalid")
    metrics = _validate_calibration_metrics(artifact.get("calibration_metrics"))
    _validate_provenance(artifact.get("provenance"), metrics)
    _validate_parent_inference_contract(
        artifact.get("parent_inference_contract"), binding,
        metrics["sample_count"],
    )
    return config


def _validate_live_artifact_model(model, artifact, label):
    if not isinstance(model, QueryReranker):
        raise ValueError("live {} must be a QueryReranker".format(label))
    actual = model.state_dict()
    expected = artifact["model_state_dict"]
    if (set(actual) != set(expected)
            or any(not torch.equal(actual[name].detach().cpu(), expected[name])
                   for name in actual)):
        raise ValueError("live {} model state differs from artifact".format(label))
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("live {} must be frozen in eval mode".format(label))


def validate_rec_finetune_artifact_pair(
        parent_artifact, geometry_artifact, parent_artifact_sha256=None,
        parent_model=None, geometry_model=None):
    """Validate exact online artifacts and all cross-artifact bindings."""
    _validate_parent_artifact(parent_artifact)
    _validate_geometry_artifact(geometry_artifact)
    if parent_artifact_sha256 is not None:
        _require_sha256(parent_artifact_sha256, "parent artifact")
        if geometry_artifact["parent_artifact_sha256"] != parent_artifact_sha256:
            raise ValueError("geometry parent artifact SHA binding is invalid")
    if (geometry_artifact["checkpoint_sha256"]
            != parent_artifact["checkpoint_sha256"]
            or geometry_artifact["checkpoint_epoch"]
            != parent_artifact["checkpoint_epoch"]):
        raise ValueError("fine-tune checkpoint binding differs between artifacts")
    if (geometry_artifact["provenance"] != parent_artifact["provenance"]
            or geometry_artifact["calibration_metrics"]
            != parent_artifact["calibration_metrics"]):
        raise ValueError("fine-tune selection provenance differs between artifacts")
    expected_binding = _parent_runtime_binding(parent_artifact)
    if geometry_artifact["parent_runtime_binding"] != expected_binding:
        raise ValueError("geometry parent runtime metadata binding is invalid")
    expected_names = (
        parent_artifact["feature_names"]
        + list(REC_MASK_GEOMETRY_FEATURE_NAMES)
        + ["parent_score", "parent_is_deployed_top1"]
    )
    for key in ("candidate_rule", "target_iou_policy", "model_inputs",
                "backbone_config"):
        if geometry_artifact[key] != parent_artifact[key]:
            raise ValueError("geometry runtime metadata differs from parent")
    if geometry_artifact["feature_names"] != expected_names:
        raise ValueError("geometry features do not bind parent runtime metadata")
    if parent_model is not None:
        _validate_live_artifact_model(parent_model, parent_artifact, "parent")
    if geometry_model is not None:
        _validate_live_artifact_model(
            geometry_model, geometry_artifact, "geometry"
        )
    return {
        "parent_model_config": copy.deepcopy(parent_artifact["model_config"]),
        "geometry_model_config": copy.deepcopy(
            geometry_artifact["model_config"]
        ),
    }


def sha256_file(path):
    """Hash the same stable bytes used by artifact loaders."""
    _resolved, _snapshot, digest = _stable_artifact_snapshot(
        path, "REC artifact"
    )
    return digest


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_rec_finetune_artifact(path, artifact):
    """Validate and atomically publish one fine-tune artifact."""
    if isinstance(artifact, dict) and artifact.get("schema") == (
            REC_FINETUNE_PARENT_SCHEMA):
        _validate_parent_artifact(artifact)
    elif isinstance(artifact, dict) and artifact.get("schema") == (
            REC_FINETUNE_GEOMETRY_SCHEMA):
        _validate_geometry_artifact(artifact)
    else:
        raise ValueError("unsupported REC fine-tune artifact schema")
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("REC fine-tune artifact output must not be a symlink")
    if output.exists() and not output.is_file():
        raise ValueError("REC fine-tune artifact output must be a regular file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(artifact, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _resolved, snapshot, _digest = _stable_artifact_snapshot(
            temporary, "temporary REC fine-tune artifact"
        )
        try:
            reloaded = torch.load(io.BytesIO(snapshot), map_location="cpu")
        except Exception as error:
            raise ValueError("could not reload REC fine-tune artifact: {}".format(
                error
            ))
        if reloaded.get("schema") == REC_FINETUNE_PARENT_SCHEMA:
            _validate_parent_artifact(reloaded)
        else:
            _validate_geometry_artifact(reloaded)
        os.replace(str(temporary), str(output))
        _fsync_directory(output.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return artifact


atomic_save_rec_finetune_artifact = save_rec_finetune_artifact


def _resolve_device(device):
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("REC fine-tune artifacts support CPU or CUDA")
    return resolved


def _deserialize_artifact(snapshot, label):
    try:
        artifact = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize {}: {}".format(label, error))
    if not isinstance(artifact, dict):
        raise ValueError("{} must contain a mapping".format(label))
    return artifact


def load_rec_finetune_runtime_artifacts(parent_path, geometry_path, device):
    """Stable-load, cross-bind, freeze, and label the online artifact pair."""
    parent_resolved, parent_snapshot, parent_sha = _stable_artifact_snapshot(
        parent_path, "REC fine-tune parent artifact"
    )
    geometry_resolved, geometry_snapshot, geometry_sha = (
        _stable_artifact_snapshot(
            geometry_path, "REC fine-tune geometry artifact"
        )
    )
    parent_artifact = _deserialize_artifact(
        parent_snapshot, "REC fine-tune parent artifact"
    )
    geometry_artifact = _deserialize_artifact(
        geometry_snapshot, "REC fine-tune geometry artifact"
    )
    configs = validate_rec_finetune_artifact_pair(
        parent_artifact, geometry_artifact, parent_sha
    )
    parent_model = QueryReranker(**configs["parent_model_config"])
    geometry_model = QueryReranker(**configs["geometry_model_config"])
    try:
        parent_model.load_state_dict(
            parent_artifact["model_state_dict"], strict=True
        )
        geometry_model.load_state_dict(
            geometry_artifact["model_state_dict"], strict=True
        )
    except RuntimeError as error:
        raise ValueError("REC fine-tune model state is incompatible: {}".format(
            error
        ))
    resolved_device = _resolve_device(device)
    parent_model.to(resolved_device).eval().requires_grad_(False)
    geometry_model.to(resolved_device).eval().requires_grad_(False)
    parent_model._artifact_sha256 = parent_sha
    parent_model._artifact_path = str(parent_resolved)
    geometry_model._artifact_sha256 = geometry_sha
    geometry_model._artifact_path = str(geometry_resolved)
    validate_rec_finetune_artifact_pair(
        parent_artifact,
        geometry_artifact,
        parent_sha,
        parent_model=parent_model,
        geometry_model=geometry_model,
    )
    return parent_model, parent_artifact, geometry_model, geometry_artifact


_GROUP_SPECS = (
    ("mcln_decoder_box", "mcln_parameters", 2e-5, 5e-4, 0.1),
    ("parent_reranker", "parent_parameters", 1e-3, 1e-4, 1.0),
    ("geometry_reranker", "geometry_parameters", 3e-4, 1e-4, 1.0),
)

_TARGET_ONLY_KEYS = frozenset((
    "center_label",
    "size_gts",
    "sem_cls_label",
    "box_label_mask",
    "gt_masks",
    "point_instance_label",
    "all_bboxes",
    "candidate_ious",
    "geometry_ious",
    "threshold_labels",
))


def _require_deployable_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError("{} must be a mapping".format(name))
    target_keys = sorted(_TARGET_ONLY_KEYS.intersection(value))
    if target_keys:
        raise ValueError(
            "{} contains target-only fields: {}".format(
                name, ", ".join(target_keys)
            )
        )


def _require_artifact_weight(artifact, key):
    value = artifact.get(key)
    if (not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0):
        raise ValueError("artifact {} is invalid".format(key))
    return float(value)


def _normalize_artifact_features(features, valid_mask, feature_names,
                                 artifact, label):
    if (not isinstance(features, torch.Tensor) or features.dim() != 3
            or features.dtype != torch.float32):
        raise ValueError("{} features must be float32 [B,K,D]".format(label))
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or tuple(valid_mask.shape) != tuple(features.shape[:2])):
        raise ValueError("{} valid mask does not match features".format(label))
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("{} needs a valid candidate in every row".format(label))
    if (not isinstance(feature_names, (list, tuple))
            or len(feature_names) != features.shape[-1]
            or any(not isinstance(name, str) or not name
                   for name in feature_names)
            or list(feature_names) != artifact.get("feature_names")):
        raise ValueError("{} artifact feature names do not match".format(label))
    input_dim = artifact.get("input_dim")
    if (not isinstance(input_dim, int) or isinstance(input_dim, bool)
            or input_dim != features.shape[-1]):
        raise ValueError("{} artifact input dimension does not match".format(label))

    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    expected_shape = (features.shape[-1],)
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or mean.dtype != torch.float32
            or std.dtype != torch.float32
            or tuple(mean.shape) != expected_shape
            or tuple(std.shape) != expected_shape
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(std).all().item())
            or bool((std <= 0.0).any().item())):
        raise ValueError(
            "{} artifact normalization must be finite float32 with positive "
            "standard deviations".format(label)
        )
    if not bool(torch.isfinite(features[valid_mask]).all().item()):
        raise ValueError("{} valid features must be finite".format(label))

    local_mean = mean.detach().to(device=features.device)
    local_std = std.detach().to(device=features.device)
    safe_features = torch.where(
        valid_mask.unsqueeze(-1), features, torch.zeros_like(features)
    )
    normalized = (safe_features - local_mean) / local_std
    normalized = torch.where(
        valid_mask.unsqueeze(-1), normalized, torch.zeros_like(normalized)
    )
    if (normalized.dtype != torch.float32
            or not bool(torch.isfinite(normalized).all().item())):
        raise ValueError("normalized {} features must be finite float32".format(
            label
        ))
    return normalized


def _parent_candidate_rule(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("parent artifact must be a mapping")
    if artifact.get("adapter_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("parent artifact feature schema does not match adapter")
    candidate_rule = artifact.get("candidate_rule")
    if not isinstance(candidate_rule, dict):
        raise ValueError("parent artifact candidate rule is invalid")
    topk_per_source = candidate_rule.get("topk_per_source")
    max_candidates = candidate_rule.get("max_candidates")
    if (not isinstance(topk_per_source, int)
            or isinstance(topk_per_source, bool)
            or topk_per_source <= 0
            or not isinstance(max_candidates, int)
            or isinstance(max_candidates, bool)
            or max_candidates <= 0):
        raise ValueError("parent artifact candidate rule is invalid")
    return topk_per_source, max_candidates


def _validate_geometry_batch(candidate_batch, geometry_batch, artifact):
    if not isinstance(artifact, dict):
        raise ValueError("geometry artifact must be a mapping")
    if not isinstance(geometry_batch, dict):
        raise ValueError("geometry candidate batch must be a mapping")
    base_features = candidate_batch.get("features")
    geometry_features = geometry_batch.get("geometry_features")
    geometry_valid = geometry_batch.get("valid_mask")
    geometry_boxes = geometry_batch.get("boxes")
    if (not isinstance(base_features, torch.Tensor)
            or base_features.dtype != torch.float32
            or base_features.dim() != 3
            or not isinstance(geometry_features, torch.Tensor)
            or geometry_features.dtype != torch.float32
            or geometry_features.dim() != 4
            or not isinstance(geometry_valid, torch.Tensor)
            or geometry_valid.dtype != torch.bool
            or not isinstance(geometry_boxes, torch.Tensor)
            or geometry_boxes.dtype != torch.float32):
        raise ValueError("geometry feature tensors are malformed")
    batch_size, candidates, base_dim = base_features.shape
    if (candidates != 16 or candidate_batch.get("num_queries") != 256
            or geometry_features.shape[:2] != (batch_size, candidates)
            or tuple(geometry_features.shape[2:]) != (7, 25)
            or tuple(geometry_valid.shape) != (batch_size, candidates, 7)
            or tuple(geometry_boxes.shape) != (batch_size, candidates, 7, 6)):
        raise ValueError("geometry candidates do not match the deployed shape")
    if any(value.device != base_features.device for value in (
            geometry_features, geometry_valid, geometry_boxes)):
        raise ValueError("geometry tensors must share the parent feature device")
    expected_names = (
        list(candidate_batch.get("feature_names", ()))
        + list(geometry_batch.get("geometry_feature_names", ()))
        + ["parent_score", "parent_is_deployed_top1"]
    )
    if (base_dim != 152 or len(expected_names) != 179
            or artifact.get("input_dim") != 179
            or artifact.get("feature_names") != expected_names):
        raise ValueError("geometry artifact feature schema is incompatible")
    configs = artifact.get("variant_configs")
    try:
        max_point_fraction_matches = (
            float(geometry_batch.get("max_point_fraction"))
            == float(artifact.get("max_point_fraction"))
        )
    except (TypeError, ValueError, OverflowError):
        max_point_fraction_matches = False
    if (not isinstance(configs, list)
            or geometry_batch.get("variant_names")
            != tuple(artifact.get("variant_names", ()))
            or list(geometry_batch.get("variant_configs", ())) != configs
            or geometry_batch.get("min_points") != artifact.get("min_points")
            or not max_point_fraction_matches
            or artifact.get("regressed_variant_index") != 0):
        raise ValueError("geometry artifact variant schema is incompatible")
    return geometry_features, geometry_valid


def _build_runtime_outputs(parent_state, geometry_batch, geometry_valid,
                           geometry_logits, geometry_artifact):
    weight = _require_artifact_weight(geometry_artifact, "geometry_weight")
    with torch.no_grad():
        blended = blend_rec_geometry_scores(
            parent_state,
            geometry_logits.detach().float(),
            geometry_valid,
            weight,
            geometry_artifact["regressed_variant_index"],
        )
        if blended.get("use_parent_query_axis") is True:
            return {
                "rec_reranker_scores": parent_state["query_scores"].detach(),
                "rec_geometry_runtime_mode": "parent_query_axis",
            }
        if blended.get("use_parent_query_axis") is not False:
            raise RuntimeError("geometry blend returned an invalid output mode")

        fallback_positions = []
        for row_mask in parent_state["parent_top1_mask"]:
            positions = row_mask.nonzero(as_tuple=False).reshape(-1)
            if positions.numel() != 1:
                raise ValueError(
                    "canonical parent Top-1 needs one compact position"
                )
            fallback_positions.append(int(positions[0].item()))
        fallback = torch.tensor(
            fallback_positions,
            dtype=torch.long,
            device=geometry_valid.device,
        ) * geometry_valid.shape[2] + int(
            geometry_artifact["regressed_variant_index"]
        )
        return {
            "rec_reranker_scores": parent_state["query_scores"].detach(),
            "rec_geometry_runtime_mode": "flat_geometry_axis",
            "rec_geometry_boxes": geometry_batch["boxes"].detach().reshape(
                geometry_valid.shape[0], -1, 6
            ),
            "rec_geometry_scores": blended["flat_scores"],
            "rec_geometry_valid_mask": blended["flat_valid_mask"],
            "rec_geometry_fallback_index": fallback,
        }


def build_rec_finetune_forward(end_points, inputs, targets, parent,
                               parent_artifact, geometry,
                               geometry_artifact):
    """Build differentiable REC losses and the exact deployed score payload."""
    _require_deployable_mapping(end_points, "end_points")
    _require_deployable_mapping(inputs, "inputs")
    if not isinstance(targets, dict):
        raise ValueError("targets must be a mapping")

    topk_per_source, max_candidates = _parent_candidate_rule(
        parent_artifact
    )
    candidate_batch = build_rec_candidate_batch(
        end_points,
        inputs,
        topk_per_source=topk_per_source,
        max_candidates=max_candidates,
    )
    candidate_features = candidate_batch.get("features")
    candidate_valid = candidate_batch.get("valid_mask")
    normalized_parent = _normalize_artifact_features(
        candidate_features,
        candidate_valid,
        candidate_batch.get("feature_names"),
        parent_artifact,
        "parent",
    )
    parent_model_inputs = {
        "features": normalized_parent,
        "valid_mask": candidate_valid,
    }
    parent_scorer_outputs = parent(
        parent_model_inputs["features"],
        parent_model_inputs["valid_mask"],
    )
    if not isinstance(parent_scorer_outputs, dict):
        raise ValueError("parent scorer outputs must be a mapping")
    parent_logits = parent_scorer_outputs.get("ranking_logits")
    if (not isinstance(parent_logits, torch.Tensor)
            or tuple(parent_logits.shape) != tuple(candidate_valid.shape)):
        raise ValueError("parent scorer ranking logits are malformed")

    if parent_artifact.get("score_mode") != "rank_blend":
        raise ValueError("parent artifact score mode is unsupported")
    reranker_weight = _require_artifact_weight(
        parent_artifact, "reranker_weight"
    )
    with torch.no_grad():
        compact_scores = blend_candidate_scores(
            candidate_batch["default_scores"].detach(),
            parent_logits.detach().float(),
            candidate_valid,
            reranker_weight,
        )
        parent_state = build_deployed_parent_state(
            compact_scores,
            candidate_batch["query_indices"],
            candidate_valid,
            candidate_batch["num_queries"],
        )

    variant_config = {
        "variants": geometry_artifact.get("variant_configs"),
        "min_points": geometry_artifact.get("min_points"),
        "max_point_fraction": geometry_artifact.get("max_point_fraction"),
    }
    geometry_batch = build_rec_mask_geometry_candidates(
        end_points,
        inputs,
        candidate_batch,
        variant_config=variant_config,
    )
    geometry_features, geometry_valid = _validate_geometry_batch(
        candidate_batch, geometry_batch, geometry_artifact
    )
    raw_geometry_inputs = build_rec_geometry_model_inputs(
        candidate_features.float(),
        geometry_features.float(),
        parent_state["compact_scores"].float(),
        parent_state["parent_top1_mask"],
        geometry_valid,
        candidate_batch["feature_names"],
        geometry_batch["geometry_feature_names"],
    )
    if tuple(raw_geometry_inputs["features"].shape[1:]) != (112, 179):
        raise ValueError("geometry model inputs must have shape [B,112,179]")
    normalized_geometry = _normalize_artifact_features(
        raw_geometry_inputs["features"],
        raw_geometry_inputs["valid_mask"],
        raw_geometry_inputs["feature_names"],
        geometry_artifact,
        "geometry",
    )
    geometry_model_inputs = dict(raw_geometry_inputs)
    geometry_model_inputs["features"] = normalized_geometry
    geometry_scorer_outputs = geometry(
        geometry_model_inputs["features"],
        geometry_model_inputs["valid_mask"],
    )
    if not isinstance(geometry_scorer_outputs, dict):
        raise ValueError("geometry scorer outputs must be a mapping")
    geometry_logits = geometry_scorer_outputs.get("ranking_logits")
    if (not isinstance(geometry_logits, torch.Tensor)
            or tuple(geometry_logits.shape)
            != tuple(geometry_model_inputs["valid_mask"].shape)):
        raise ValueError("geometry scorer ranking logits are malformed")

    parent_targets = attach_candidate_targets(
        candidate_batch, targets, root_only=True
    )
    parent_candidate_ious = parent_targets["candidate_ious"].detach()
    geometry_targets = attach_rec_mask_geometry_targets(
        geometry_batch, targets, root_only=True
    )
    geometry_candidate_ious = geometry_targets["geometry_ious"].reshape(
        geometry_model_inputs["valid_mask"].shape
    ).detach()
    parent_loss, parent_loss_stats = compute_rec_reranker_loss(
        parent_scorer_outputs,
        parent_candidate_ious,
        candidate_valid,
        tier_pairwise_alpha=0.0,
    )
    geometry_loss, geometry_loss_stats = compute_rec_reranker_loss(
        geometry_scorer_outputs,
        geometry_candidate_ious,
        geometry_model_inputs["valid_mask"],
        tier_pairwise_alpha=1.0,
    )
    runtime_outputs = _build_runtime_outputs(
        parent_state,
        geometry_batch,
        geometry_valid,
        geometry_logits,
        geometry_artifact,
    )
    return {
        "parent_model_inputs": parent_model_inputs,
        "geometry_model_inputs": geometry_model_inputs,
        "parent_scorer_outputs": parent_scorer_outputs,
        "geometry_scorer_outputs": geometry_scorer_outputs,
        "parent_candidate_ious": parent_candidate_ious,
        "geometry_candidate_ious": geometry_candidate_ious,
        "parent_loss": parent_loss,
        "parent_loss_stats": parent_loss_stats,
        "geometry_loss": geometry_loss,
        "geometry_loss_stats": geometry_loss_stats,
        "parent_state": parent_state,
        "geometry_batch": geometry_batch,
        "runtime_outputs": runtime_outputs,
    }


def _freeze_parameters(named_parameter_groups):
    seen = set()
    for named_parameters in named_parameter_groups:
        for _name, parameter in named_parameters:
            identity = id(parameter)
            if identity not in seen:
                parameter.requires_grad_(False)
                seen.add(identity)


def _is_allowed_mcln_name(name):
    return any(name.startswith(prefix) for prefix in MCLN_TRAINABLE_PREFIXES)


def _is_allowed_mcln_module_name(name):
    return any(
        name == allowed_name or name.startswith(allowed_name + ".")
        for allowed_name in _MCLN_TRAINABLE_MODULE_NAMES
    )


def _registered_paths(root):
    """Return every registered module and parameter path, including aliases."""
    module_paths = []
    parameter_paths = []

    def visit(module, prefix, ancestor_ids):
        module_paths.append((prefix, module))
        next_ancestors = ancestor_ids + (id(module),)
        for name, parameter in module._parameters.items():
            if parameter is not None:
                full_name = "{}.{}".format(prefix, name) if prefix else name
                parameter_paths.append((full_name, parameter))
        for name, child in module._modules.items():
            if child is None:
                continue
            full_name = "{}.{}".format(prefix, name) if prefix else name
            if id(child) in next_ancestors:
                raise ValueError(
                    "cyclic module registration at {}".format(full_name)
                )
            visit(child, full_name, next_ancestors)

    visit(root, "", ())
    return tuple(module_paths), tuple(parameter_paths)


def _reject_cross_boundary_aliases(paths, is_allowed, kind, boundary,
                                   error_type):
    names_by_identity = {}
    for name, value in paths:
        names_by_identity.setdefault(id(value), []).append(name)
    for names in names_by_identity.values():
        if len({is_allowed(name) for name in names}) > 1:
            display_names = tuple(name or "<root>" for name in names)
            raise error_type(
                "{} aliases cross the MCLN {} boundary: {}".format(
                    kind, boundary, ", ".join(display_names)
                )
            )


def configure_rec_finetune_trainability(mcln, parent, geometry):
    """Apply the exact parameter allowlist and return disjoint named groups."""
    mcln_named = tuple(mcln.named_parameters())
    parent_named = tuple(parent.named_parameters())
    geometry_named = tuple(geometry.named_parameters())
    all_named = (mcln_named, parent_named, geometry_named)
    _freeze_parameters(all_named)

    _module_paths, registered_parameter_paths = _registered_paths(mcln)
    _reject_cross_boundary_aliases(
        registered_parameter_paths,
        _is_allowed_mcln_name,
        "parameter",
        "allowlist",
        ValueError,
    )

    mcln_selected = tuple(
        (name, parameter)
        for name, parameter in mcln_named
        if _is_allowed_mcln_name(name)
    )
    missing_prefixes = tuple(
        prefix for prefix in MCLN_TRAINABLE_PREFIXES
        if not any(name.startswith(prefix) for name, _parameter in mcln_named)
    )
    if missing_prefixes:
        raise ValueError(
            "MCLN trainable prefixes match no parameters: {}".format(
                ", ".join(missing_prefixes)
            )
        )
    for group_name, named_parameters in (
            ("mcln", mcln_selected),
            ("parent", parent_named),
            ("geometry", geometry_named)):
        if not named_parameters:
            raise ValueError(
                "{} parameter group must be non-empty".format(group_name)
            )

    identity_sets = tuple(
        {id(parameter) for _name, parameter in named_parameters}
        for named_parameters in (mcln_selected, parent_named, geometry_named)
    )
    if (not identity_sets[0].isdisjoint(identity_sets[1])
            or not identity_sets[0].isdisjoint(identity_sets[2])
            or not identity_sets[1].isdisjoint(identity_sets[2])):
        raise ValueError("REC fine-tuning parameter groups overlap")

    try:
        for _name, parameter in mcln_selected + parent_named + geometry_named:
            parameter.requires_grad_(True)
        for name, parameter in mcln_named:
            expected = _is_allowed_mcln_name(name)
            if parameter.requires_grad is not expected:
                raise RuntimeError(
                    "MCLN trainability does not match the approved allowlist"
                )
        if (not all(parameter.requires_grad for _name, parameter in parent_named)
                or not all(parameter.requires_grad
                           for _name, parameter in geometry_named)):
            raise RuntimeError("reranker parameters must all be trainable")
    except Exception:
        _freeze_parameters(all_named)
        raise

    return {
        "mcln_names": tuple(name for name, _parameter in mcln_selected),
        "mcln_parameters": tuple(
            parameter for _name, parameter in mcln_selected
        ),
        "parent_names": tuple(name for name, _parameter in parent_named),
        "parent_parameters": tuple(
            parameter for _name, parameter in parent_named
        ),
        "geometry_names": tuple(name for name, _parameter in geometry_named),
        "geometry_parameters": tuple(
            parameter for _name, parameter in geometry_named
        ),
    }


def set_rec_finetune_train_mode(mcln, parent, geometry):
    """Train only the allowlisted MCLN modules and both rerankers."""
    try:
        mcln.eval()
        module_paths, _parameter_paths = _registered_paths(mcln)
        _reject_cross_boundary_aliases(
            module_paths,
            _is_allowed_mcln_module_name,
            "module",
            "train/eval",
            AssertionError,
        )
        for module_name in _MCLN_TRAINABLE_MODULE_NAMES:
            module = getattr(mcln, module_name, None)
            if not isinstance(module, torch.nn.Module):
                raise ValueError(
                    "MCLN is missing trainable module {}".format(module_name)
                )
            module.train()

        for name, module in module_paths:
            if not _is_allowed_mcln_module_name(name) and module.training:
                raise AssertionError(
                    "frozen MCLN module {} entered train mode".format(
                        name or "<root>"
                    )
                )

        parent.train()
        geometry.train()
    except Exception:
        mcln.eval()
        parent.eval()
        geometry.eval()
        _freeze_parameters(tuple(
            tuple(model.named_parameters())
            for model in (mcln, parent, geometry)
        ))
        raise


def set_rec_finetune_eval_mode(mcln, parent, geometry):
    """Put every fine-tuning component in evaluation mode."""
    mcln.eval()
    parent.eval()
    geometry.eval()


def _validated_group_parameters(groups):
    if not isinstance(groups, dict):
        raise ValueError("REC fine-tuning groups must be a mapping")
    parameter_groups = []
    identity_sets = []
    for group_name, parameter_key, _lr, _weight_decay, _clip in _GROUP_SPECS:
        parameters = groups.get(parameter_key)
        if not isinstance(parameters, tuple) or not parameters:
            raise ValueError(
                "{} parameter group must be a non-empty tuple".format(
                    group_name
                )
            )
        if not all(isinstance(parameter, torch.nn.Parameter)
                   for parameter in parameters):
            raise ValueError("parameter groups must contain Parameters")
        identities = {id(parameter) for parameter in parameters}
        if len(identities) != len(parameters):
            raise ValueError("a parameter group contains duplicates")
        parameter_groups.append(parameters)
        identity_sets.append(identities)
    if (not identity_sets[0].isdisjoint(identity_sets[1])
            or not identity_sets[0].isdisjoint(identity_sets[2])
            or not identity_sets[1].isdisjoint(identity_sets[2])):
        raise ValueError("REC fine-tuning parameter groups overlap")
    return tuple(parameter_groups)


def build_rec_finetune_optimizer(groups):
    """Build a fresh constant-rate AdamW with the three approved groups."""
    parameter_groups = _validated_group_parameters(groups)
    optimizer_groups = []
    for spec, parameters in zip(_GROUP_SPECS, parameter_groups):
        group_name, _parameter_key, lr, weight_decay, _clip = spec
        optimizer_groups.append({
            "name": group_name,
            "params": parameters,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    return torch.optim.AdamW(optimizer_groups)


def clip_rec_finetune_gradients(groups):
    """Clip each parameter group independently and return finite norms."""
    parameter_groups = _validated_group_parameters(groups)
    diagnostics = {}
    for spec, parameters in zip(_GROUP_SPECS, parameter_groups):
        group_name, _parameter_key, _lr, _weight_decay, max_norm = spec
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        norm = float(total_norm)
        if not math.isfinite(norm):
            raise FloatingPointError(
                "{} gradient norm is non-finite".format(group_name)
            )
        diagnostics[group_name] = norm
    return diagnostics


def _require_positive_integer(name, value):
    if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError("{} must be a positive integer".format(name))


def natural_batch_count(sample_count, batch_size):
    """Return the natural drop_last=False batch count."""
    _require_positive_integer("sample_count", sample_count)
    _require_positive_integer("batch_size", batch_size)
    return (sample_count + batch_size - 1) // batch_size


def calibration_steps(max_steps, interval):
    """Return step zero, every full interval, and the final step."""
    _require_positive_integer("max_steps", max_steps)
    _require_positive_integer("interval", interval)
    steps = tuple(range(0, max_steps + 1, interval))
    if steps[-1] != max_steps:
        steps += (max_steps,)
    return steps


def canonical_json_sha256(payload):
    """Return a SHA-256 over the canonical JSON representation."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_calibration_fraction(value):
    if (not isinstance(value, (float, int)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0):
        raise ValueError(
            "calibration_fraction must lie strictly between 0 and 1"
        )
    return float(value)


def build_rec_finetune_scene_split(
        scan_ids, seed=0, calibration_fraction=0.10):
    """Build the deterministic scene-disjoint annotation index split."""
    if not isinstance(scan_ids, (list, tuple)) or not scan_ids:
        raise ValueError("scan_ids must be a non-empty annotation sequence")
    if any(not isinstance(scan_id, str) or not scan_id.strip()
           for scan_id in scan_ids):
        raise ValueError("every scan_id must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    fraction = _require_calibration_fraction(calibration_fraction)

    scenes = sorted(set(scan_ids))
    shuffled = list(scenes)
    random.Random(int(seed)).shuffle(shuffled)
    if len(scenes) == 1:
        calibration_count = 0
    else:
        calibration_count = int(round(len(scenes) * fraction))
        calibration_count = max(
            1, min(calibration_count, len(scenes) - 1)
        )
    calibration_scene_set = set(shuffled[:calibration_count])
    fit_indices = tuple(
        index for index, scan_id in enumerate(scan_ids)
        if scan_id not in calibration_scene_set
    )
    calibration_indices = tuple(
        index for index, scan_id in enumerate(scan_ids)
        if scan_id in calibration_scene_set
    )
    fit_scenes = tuple(sorted(set(scenes) - calibration_scene_set))
    calibration_scenes = tuple(sorted(calibration_scene_set))
    mapping = {
        "fit": list(fit_scenes),
        "calibration": list(calibration_scenes),
    }
    metadata = {
        "split_seed": int(seed),
        "calibration_fraction": fraction,
        "scene_count": len(scenes),
        "fit_scene_count": len(fit_scenes),
        "calibration_scene_count": len(calibration_scenes),
        "sample_count": len(scan_ids),
        "fit_sample_count": len(fit_indices),
        "calibration_sample_count": len(calibration_indices),
        "fit_scene_sha256": canonical_json_sha256(list(fit_scenes)),
        "calibration_scene_sha256": canonical_json_sha256(
            list(calibration_scenes)
        ),
        "mapping_sha256": canonical_json_sha256(mapping),
    }
    return {
        "fit_indices": fit_indices,
        "calibration_indices": calibration_indices,
        "fit_scenes": fit_scenes,
        "calibration_scenes": calibration_scenes,
        "metadata": metadata,
    }


_INDEX_DTYPES = frozenset((
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
))


def _ordered_dataset_indices(indices, name, allow_empty=False):
    if isinstance(indices, torch.Tensor):
        if indices.dim() != 1 or indices.dtype not in _INDEX_DTYPES:
            raise ValueError("{} must be a one-dimensional integer sequence".format(
                name
            ))
        values = tuple(indices.detach().cpu().tolist())
    elif isinstance(indices, (list, tuple)):
        values = tuple(indices)
    else:
        raise ValueError("{} must be an ordered index sequence".format(name))
    if not allow_empty and not values:
        raise ValueError("{} cannot be empty".format(name))
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in values):
        raise ValueError("{} contains an invalid dataset index".format(name))
    if len(set(values)) != len(values):
        raise ValueError("{} contains duplicate dataset indices".format(name))
    return values


def _calibration_score(acc025, acc050):
    return min(acc025 / 0.60, acc050 / 0.47) + 0.1 * (
        acc025 + acc050
    )


class CalibrationAccumulator:
    """Accumulate one fixed-order calibration pass without storing IoUs."""

    def __init__(self, expected_indices):
        self._expected_indices = _ordered_dataset_indices(
            expected_indices, "expected_indices"
        )
        self._cursor = 0
        self._hits025 = 0
        self._hits050 = 0

    @property
    def expected_indices(self):
        return self._expected_indices

    def update(self, indices, selected_ious):
        batch_indices = _ordered_dataset_indices(indices, "indices")
        if (not isinstance(selected_ious, torch.Tensor)
                or selected_ious.dim() != 1
                or not torch.is_floating_point(selected_ious)):
            raise ValueError("selected_ious must be a one-dimensional float tensor")
        if len(batch_indices) != selected_ious.numel():
            raise ValueError("indices and selected_ious length must match")
        values = selected_ious.detach()
        if (not bool(torch.isfinite(values).all().item())
                or bool(((values < 0.0) | (values > 1.0)).any().item())):
            raise ValueError("selected_ious must be finite and lie in [0, 1]")
        expected = self._expected_indices[
            self._cursor:self._cursor + len(batch_indices)
        ]
        if batch_indices != expected:
            raise ValueError("calibration dataset indices are out of order")
        self._hits025 += int((values > 0.25).sum().item())
        self._hits050 += int((values > 0.50).sum().item())
        self._cursor += len(batch_indices)

    def finalize(self):
        sample_count = len(self._expected_indices)
        if self._cursor != sample_count:
            raise ValueError("calibration pass is incomplete")
        acc025 = self._hits025 / float(sample_count)
        acc050 = self._hits050 / float(sample_count)
        return {
            "sample_count": sample_count,
            "hits025": self._hits025,
            "hits050": self._hits050,
            "acc025": acc025,
            "acc050": acc050,
            "score": _calibration_score(acc025, acc050),
        }


_CALIBRATION_DIAGNOSTIC_BRANCHES = (
    "default_top1",
    "source_selector_top1",
    "parent_top1",
    "geometry_top1",
    "raw_query_oracle",
    "parent_candidate_oracle",
    "geometry_candidate_oracle",
)
_CALIBRATION_ORACLE_ABS_TOLERANCE = 1e-6
_CALIBRATION_ORACLE_REQUIREMENTS = (
    (
        "raw_query_oracle",
        (
            "default_top1", "source_selector_top1", "parent_top1",
            "parent_candidate_oracle",
        ),
    ),
    (
        "parent_candidate_oracle",
        ("default_top1", "parent_top1"),
    ),
    (
        "geometry_candidate_oracle",
        (
            "default_top1", "parent_top1", "geometry_top1",
            "parent_candidate_oracle",
        ),
    ),
)
CALIBRATION_JOINT_STATE_TIERS = (
    ("s0_o0", 0, 0),
    ("s0_o1", 0, 1),
    ("s0_o2", 0, 2),
    ("s1_o1", 1, 1),
    ("s1_o2", 1, 2),
    ("s2_o2", 2, 2),
)
CALIBRATION_SELECTED_IOU_BIN_SPECS = (
    ("le_010", 0.00, 0.10, True, 0),
    ("gt_010_le_020", 0.10, 0.20, False, 0),
    ("gt_020_le_025", 0.20, 0.25, False, 0),
    ("gt_025_le_030", 0.25, 0.30, False, 1),
    ("gt_030_le_050", 0.30, 0.50, False, 1),
    ("gt_050_le_075", 0.50, 0.75, False, 2),
    ("gt_075_le_100", 0.75, 1.00, False, 2),
)
CALIBRATION_ORACLE_TIER_SPECS = (
    ("o0", 0.00, 0.25, True, 0),
    ("o1", 0.25, 0.50, False, 1),
    ("o2", 0.50, 1.00, False, 2),
)
CALIBRATION_REGRET_BAND_SPECS = (
    ("zero", 0.00, 0.00, True, True),
    ("gt_000_lt_005", 0.00, 0.05, False, False),
    ("ge_005_lt_010", 0.05, 0.10, True, False),
    ("ge_010", 0.10, 1.00, True, True),
)


@dataclass(frozen=True)
class CalibrationDiagnosticsTransitionState:
    """Private ordered IoUs needed to compare two calibration steps."""

    expected_indices: tuple
    selected_ious: tuple
    geometry_oracle_ious: tuple


@dataclass(frozen=True)
class CalibrationDiagnosticsResult:
    """Public aggregate diagnostics plus non-published transition state."""

    diagnostics: dict
    transition_state: CalibrationDiagnosticsTransitionState


def _diagnostic_threshold_metrics(values):
    sample_count = int(values.numel())
    hits025 = int((values > 0.25).sum().item())
    hits050 = int((values > 0.50).sum().item())
    return {
        "hits025": hits025,
        "hits050": hits050,
        "acc025": hits025 / float(sample_count),
        "acc050": hits050 / float(sample_count),
    }


def _diagnostic_effect_metrics(previous, current):
    result = {}
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        previous_hit = previous > threshold
        current_hit = current > threshold
        result["fixes" + suffix] = int(
            ((~previous_hit) & current_hit).sum().item()
        )
        result["breaks" + suffix] = int(
            (previous_hit & (~current_hit)).sum().item()
        )
    return result


def _bounded_interval_mask(values, lower, upper, lower_inclusive,
                           upper_inclusive=True):
    lower_mask = values >= lower if lower_inclusive else values > lower
    upper_mask = values <= upper if upper_inclusive else values < upper
    return lower_mask & upper_mask


def _selected_iou_bin_masks(values):
    return tuple(
        (
            name,
            _bounded_interval_mask(
                values, lower, upper, lower_inclusive, True
            ),
        )
        for name, lower, upper, lower_inclusive, _tier
        in CALIBRATION_SELECTED_IOU_BIN_SPECS
    )


def _selected_iou_bins(values):
    return {
        name: int(mask.sum().item())
        for name, mask in _selected_iou_bin_masks(values)
    }


def _oracle_tier_masks(values):
    return tuple(
        (
            name,
            _bounded_interval_mask(
                values, lower, upper, lower_inclusive, True
            ),
        )
        for name, lower, upper, lower_inclusive, _tier
        in CALIBRATION_ORACLE_TIER_SPECS
    )


def _regret_band_masks(values):
    return tuple(
        (
            name,
            _bounded_interval_mask(
                values, lower, upper, lower_inclusive, upper_inclusive
            ),
        )
        for name, lower, upper, lower_inclusive, upper_inclusive
        in CALIBRATION_REGRET_BAND_SPECS
    )


def _selected_oracle_regret_cells(selected, geometry_oracle):
    regret = geometry_oracle - selected
    cells = {}
    for selected_name, selected_mask in _selected_iou_bin_masks(selected):
        cells[selected_name] = {}
        for oracle_name, oracle_mask in _oracle_tier_masks(geometry_oracle):
            cells[selected_name][oracle_name] = {}
            for regret_name, regret_mask in _regret_band_masks(regret):
                values = regret[selected_mask & oracle_mask & regret_mask]
                cells[selected_name][oracle_name][regret_name] = {
                    "count": int(values.numel()),
                }
    return cells


def _canonicalize_calibration_diagnostic_oracles(values):
    canonical = dict(values)
    for oracle_name, required_stage_names in (
            _CALIBRATION_ORACLE_REQUIREMENTS):
        oracle = canonical[oracle_name].to(dtype=torch.float64)
        required = tuple(
            canonical[name].to(dtype=torch.float64)
            for name in required_stage_names
        )
        for stage_name, stage_values in zip(
                required_stage_names, required):
            if bool((
                    oracle + _CALIBRATION_ORACLE_ABS_TOLERANCE
                    < stage_values
            ).any().item()):
                raise ValueError(
                    "{} oracle is lower than {}".format(
                        oracle_name, stage_name
                    )
                )
        lower_bound = required[0]
        for stage_values in required[1:]:
            lower_bound = torch.maximum(lower_bound, stage_values)
        canonical[oracle_name] = torch.maximum(oracle, lower_bound)
    return canonical


class CalibrationDiagnosticsAccumulator:
    """Accumulate fixed-order, train-only calibration bottleneck metrics."""

    def __init__(self, expected_indices):
        self._expected_indices = _ordered_dataset_indices(
            expected_indices, "expected_indices"
        )
        self._cursor = 0
        self._branch_batches = {
            name: [] for name in _CALIBRATION_DIAGNOSTIC_BRANCHES
        }

    @property
    def expected_indices(self):
        return self._expected_indices

    def update(self, indices, branch_ious):
        batch_indices = _ordered_dataset_indices(indices, "indices")
        if (not isinstance(branch_ious, dict)
                or set(branch_ious) != set(_CALIBRATION_DIAGNOSTIC_BRANCHES)):
            raise ValueError(
                "calibration diagnostic branch fields do not match schema"
            )

        normalized = {}
        for name in _CALIBRATION_DIAGNOSTIC_BRANCHES:
            values = branch_ious[name]
            if (not isinstance(values, torch.Tensor)
                    or values.dim() != 1
                    or not torch.is_floating_point(values)):
                raise ValueError(
                    "{} must be a one-dimensional float tensor".format(name)
                )
            if values.numel() != len(batch_indices):
                raise ValueError(
                    "{} and indices length must match".format(name)
                )
            detached = values.detach()
            if not bool(torch.isfinite(detached).all().item()):
                raise ValueError("{} must be finite".format(name))
            if bool(((detached < 0.0) | (detached > 1.0)).any().item()):
                raise ValueError("{} must lie in [0, 1]".format(name))
            cpu_values = detached.to(device="cpu")
            if cpu_values.dtype in (torch.float16, torch.bfloat16):
                cpu_values = cpu_values.float()
            normalized[name] = cpu_values.clone()

        normalized = _canonicalize_calibration_diagnostic_oracles(
            normalized
        )

        expected = self._expected_indices[
            self._cursor:self._cursor + len(batch_indices)
        ]
        if batch_indices != expected:
            raise ValueError("calibration dataset indices are out of order")
        for name in _CALIBRATION_DIAGNOSTIC_BRANCHES:
            self._branch_batches[name].append(normalized[name])
        self._cursor += len(batch_indices)

    def finalize(self):
        sample_count = len(self._expected_indices)
        if self._cursor != sample_count:
            raise ValueError("calibration diagnostics pass is incomplete")
        values = {
            name: torch.cat(self._branch_batches[name], dim=0)
            for name in _CALIBRATION_DIAGNOSTIC_BRANCHES
        }
        selected = values["geometry_top1"]
        geometry_oracle = values["geometry_candidate_oracle"]
        regret = geometry_oracle - selected
        diagnostics = {
            "schema": "rec-finetune-calibration-diagnostics-v3",
            "sample_count": sample_count,
            "candidate_oracle": {
                "raw_query": _diagnostic_threshold_metrics(
                    values["raw_query_oracle"]
                ),
                "parent_candidate": _diagnostic_threshold_metrics(
                    values["parent_candidate_oracle"]
                ),
                "geometry_candidate": _diagnostic_threshold_metrics(
                    geometry_oracle
                ),
            },
            "stages": {
                name: _diagnostic_threshold_metrics(values[name])
                for name in (
                    "default_top1", "source_selector_top1",
                    "parent_top1", "geometry_top1",
                )
            },
            "effects": {
                "source_selector_vs_default": _diagnostic_effect_metrics(
                    values["default_top1"],
                    values["source_selector_top1"],
                ),
                "parent_vs_default": _diagnostic_effect_metrics(
                    values["default_top1"], values["parent_top1"]
                ),
                "geometry_vs_parent": _diagnostic_effect_metrics(
                    values["parent_top1"], selected
                ),
            },
            "selected_iou": {
                "bins": _selected_iou_bins(selected),
            },
            "geometry_oracle_selected_regret": {
                "positive_count": int((regret > 0.0).sum().item()),
                "ge005_count": int((regret >= 0.05).sum().item()),
                "ge010_count": int((regret >= 0.10).sum().item()),
            },
            "recoverable_misses": {
                "at025": int((
                    (selected <= 0.25) & (geometry_oracle > 0.25)
                ).sum().item()),
                "at050": int((
                    (selected <= 0.50) & (geometry_oracle > 0.50)
                ).sum().item()),
            },
            "selected_oracle_regret_cells": (
                _selected_oracle_regret_cells(selected, geometry_oracle)
            ),
        }
        transition_state = CalibrationDiagnosticsTransitionState(
            expected_indices=self._expected_indices,
            selected_ious=tuple(float(value) for value in selected.tolist()),
            geometry_oracle_ious=tuple(
                float(value) for value in geometry_oracle.tolist()
            ),
        )
        return CalibrationDiagnosticsResult(
            diagnostics=diagnostics,
            transition_state=transition_state,
        )


def _validated_calibration_transition_state(value, label):
    if not isinstance(value, CalibrationDiagnosticsTransitionState):
        raise ValueError("{} transition state is invalid".format(label))
    indices = _ordered_dataset_indices(
        value.expected_indices, label + " expected_indices"
    )
    sequences = {}
    for field in ("selected_ious", "geometry_oracle_ious"):
        sequence = getattr(value, field)
        if (not isinstance(sequence, tuple)
                or len(sequence) != len(indices)
                or any(not isinstance(item, (int, float))
                       or isinstance(item, bool)
                       or not math.isfinite(float(item))
                       or not 0.0 <= float(item) <= 1.0
                       for item in sequence)):
            raise ValueError(
                "{} {} transition values are invalid".format(label, field)
            )
        sequences[field] = torch.tensor(sequence, dtype=torch.float64)
    selected = sequences["selected_ious"]
    geometry_oracle = sequences["geometry_oracle_ious"]
    if bool((
            geometry_oracle + _CALIBRATION_ORACLE_ABS_TOLERANCE
            < selected
    ).any().item()):
        raise ValueError(
            "{} geometry oracle is lower than selected IoU".format(label)
        )
    sequences["geometry_oracle_ious"] = torch.maximum(
        geometry_oracle, selected
    )
    return indices, sequences


def calibration_selected_output_sha256(transition_state):
    """Hash ordered calibration indices and exact selected-IoU floats."""
    indices, _sequences = _validated_calibration_transition_state(
        transition_state, "selected output"
    )
    payload = {
        "schema": "rec-finetune-calibration-selected-output-v1",
        "rows": [
            [dataset_index, float(selected_iou).hex()]
            for dataset_index, selected_iou in zip(
                indices, transition_state.selected_ious
            )
        ],
    }
    return canonical_json_sha256(payload)


def _calibration_transition_counts(previous, current):
    result = {}
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        previous_hit = previous > threshold
        current_hit = current > threshold
        result["gained" + suffix] = int(
            ((~previous_hit) & current_hit).sum().item()
        )
        result["lost" + suffix] = int(
            (previous_hit & (~current_hit)).sum().item()
        )
    return result


def _calibration_iou_tiers(values):
    return (
        (values > 0.25).to(dtype=torch.long)
        + (values > 0.50).to(dtype=torch.long)
    )


def _calibration_joint_transition_counts(previous, current):
    state_names = tuple(
        name for name, _selected, _oracle in CALIBRATION_JOINT_STATE_TIERS
    )
    state_by_tiers = {
        (selected, oracle): name
        for name, selected, oracle in CALIBRATION_JOINT_STATE_TIERS
    }
    counts = {
        previous_name: {current_name: 0 for current_name in state_names}
        for previous_name in state_names
    }
    previous_selected = _calibration_iou_tiers(previous["selected_ious"])
    previous_oracle = _calibration_iou_tiers(
        previous["geometry_oracle_ious"]
    )
    current_selected = _calibration_iou_tiers(current["selected_ious"])
    current_oracle = _calibration_iou_tiers(
        current["geometry_oracle_ious"]
    )
    tier_rows = zip(
        previous_selected.tolist(), previous_oracle.tolist(),
        current_selected.tolist(), current_oracle.tolist(),
    )
    for previous_selected_tier, previous_oracle_tier, \
            current_selected_tier, current_oracle_tier in tier_rows:
        try:
            previous_name = state_by_tiers[
                (previous_selected_tier, previous_oracle_tier)
            ]
            current_name = state_by_tiers[
                (current_selected_tier, current_oracle_tier)
            ]
        except KeyError as error:
            raise ValueError(
                "calibration joint transition violates oracle containment"
            ) from error
        counts[previous_name][current_name] += 1
    return counts


def build_calibration_step_transition(
        previous_state, current_state, previous_step, current_step):
    """Summarize paired hit transitions for two fixed-order observations."""
    if (not isinstance(previous_step, int) or isinstance(previous_step, bool)
            or previous_step < 0
            or not isinstance(current_step, int)
            or isinstance(current_step, bool)
            or current_step <= previous_step):
        raise ValueError("calibration transition steps are invalid")
    previous_indices, previous = _validated_calibration_transition_state(
        previous_state, "previous"
    )
    current_indices, current = _validated_calibration_transition_state(
        current_state, "current"
    )
    if previous_indices != current_indices:
        raise ValueError("calibration transition indices differ")
    return {
        "schema": "rec-finetune-calibration-step-transition-v2",
        "previous_step": previous_step,
        "current_step": current_step,
        "sample_count": len(previous_indices),
        "selected": _calibration_transition_counts(
            previous["selected_ious"], current["selected_ious"]
        ),
        "geometry_oracle": _calibration_transition_counts(
            previous["geometry_oracle_ious"],
            current["geometry_oracle_ious"],
        ),
        "selected_oracle_joint": _calibration_joint_transition_counts(
            previous, current
        ),
    }


def _clone_cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise ValueError("snapshot tensors must be finite")
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {
            key: _clone_cpu_snapshot(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu_snapshot(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("snapshot contains an unsupported or non-finite value")


def _normalized_calibration_metrics(metrics, expected_sample_count):
    required = frozenset((
        "sample_count", "hits025", "hits050", "acc025", "acc050", "score",
    ))
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise ValueError("calibration metrics are malformed")
    sample_count = metrics["sample_count"]
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count <= 0
            or (expected_sample_count is not None
                and sample_count != expected_sample_count)):
        raise ValueError("calibration metrics sample_count is invalid")
    hits025 = metrics["hits025"]
    hits050 = metrics["hits050"]
    if any(not isinstance(hits, int) or isinstance(hits, bool)
           or not 0 <= hits <= sample_count for hits in (hits025, hits050)):
        raise ValueError("calibration metric hit counts are invalid")
    numeric = []
    for key in ("acc025", "acc050", "score"):
        value = metrics[key]
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            raise ValueError("calibration metric {} is invalid".format(key))
        numeric.append(float(value))
    acc025, acc050, score = numeric
    if not 0.0 <= acc025 <= 1.0 or not 0.0 <= acc050 <= 1.0:
        raise ValueError("calibration accuracies must lie in [0, 1]")
    if (not math.isclose(acc025, hits025 / float(sample_count),
                         rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(acc050, hits050 / float(sample_count),
                                rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(score, _calibration_score(acc025, acc050),
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("calibration metrics are internally inconsistent")
    return {
        "sample_count": sample_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": acc025,
        "acc050": acc050,
        "score": score,
    }


@dataclass(frozen=True)
class CalibrationDecision:
    """Immutable result of one calibration observation."""

    step: int
    action: str
    eligible: bool
    regression: bool
    best_step: int


class CalibrationSelector:
    """Select the earliest best eligible fixed-contract calibration pass."""

    def __init__(self, contract_steps=CALIBRATION_STEPS,
                 expected_sample_count=None):
        if (not isinstance(contract_steps, tuple) or not contract_steps
                or contract_steps[0] != 0
                or any(not isinstance(step, int) or isinstance(step, bool)
                       or step < 0 for step in contract_steps)
                or any(left >= right for left, right in zip(
                    contract_steps, contract_steps[1:]
                ))):
            raise ValueError(
                "contract_steps must be a strictly increasing tuple from zero"
            )
        if (expected_sample_count is not None
                and (not isinstance(expected_sample_count, int)
                     or isinstance(expected_sample_count, bool)
                     or expected_sample_count <= 0)):
            raise ValueError("expected_sample_count must be a positive integer")
        self._contract_steps = contract_steps
        self._expected_sample_count = expected_sample_count
        self._baseline_metrics = None
        self._previous_metrics = None
        self._best_metrics = None
        self._best_snapshot = None
        self._best_step = None
        self._history = []
        self._stopped = False

    @property
    def best_step(self):
        return self._best_step

    @property
    def best_metrics(self):
        if self._best_metrics is None:
            return None
        return dict(self._best_metrics)

    @property
    def best_snapshot(self):
        if self._best_snapshot is None:
            return None
        return _clone_cpu_snapshot(self._best_snapshot)

    @property
    def history(self):
        return tuple({
            **record,
            "metrics": dict(record["metrics"]),
        } for record in self._history)

    def observe(self, step, metrics, snapshot):
        if self._stopped:
            raise ValueError("calibration selection already stopped")
        if len(self._history) >= len(self._contract_steps):
            raise ValueError("all calibration contract steps were observed")
        expected_step = self._contract_steps[len(self._history)]
        if (not isinstance(step, int) or isinstance(step, bool)
                or step != expected_step):
            raise ValueError(
                "calibration step must be the next contract step {}".format(
                    expected_step
                )
            )
        normalized = _normalized_calibration_metrics(
            metrics, self._expected_sample_count
        )
        if not self._history:
            eligible = True
            regression = False
            replaces_best = True
        else:
            eligible = (
                normalized["acc025"] >= self._baseline_metrics["acc025"]
                and normalized["acc050"] >= self._baseline_metrics["acc050"]
            )
            regression = (
                not eligible
                or normalized["score"] < self._previous_metrics["score"]
            )
            replaces_best = (
                not regression
                and eligible
                and normalized["score"] > self._best_metrics["score"]
            )

        cloned_snapshot = None
        if replaces_best:
            cloned_snapshot = _clone_cpu_snapshot(snapshot)

        if self._expected_sample_count is None:
            self._expected_sample_count = normalized["sample_count"]
        if not self._history:
            self._baseline_metrics = dict(normalized)
            self._best_metrics = dict(normalized)
            self._best_snapshot = cloned_snapshot
            self._best_step = step
        elif replaces_best:
            self._best_metrics = dict(normalized)
            self._best_snapshot = cloned_snapshot
            self._best_step = step

        action = "stop" if regression else "continue"
        self._previous_metrics = dict(normalized)
        self._history.append({
            "step": step,
            "metrics": dict(normalized),
            "eligible": eligible,
            "regression": regression,
            "action": action,
            "best_step": self._best_step,
        })
        if regression:
            self._stopped = True
        return CalibrationDecision(
            step=step,
            action=action,
            eligible=eligible,
            regression=regression,
            best_step=self._best_step,
        )


if natural_batch_count(
        PRODUCTION_TRAIN_SAMPLE_COUNT,
        PRODUCTION_BATCH_SIZE) != PRODUCTION_MAX_STEPS:
    raise RuntimeError("production batch contract is inconsistent")
if calibration_steps(
        PRODUCTION_MAX_STEPS,
        PRODUCTION_CALIBRATION_INTERVAL) != CALIBRATION_STEPS:
    raise RuntimeError("production calibration contract is inconsistent")
