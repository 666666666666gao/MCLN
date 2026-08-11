#!/usr/bin/env python
"""Evaluate one frozen REC geometry artifact against one val sidecar once."""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import torch

from models.rec_geometry_reranker import (
    blend_rec_geometry_scores,
    stable_flat_descending_indices,
)
from scripts.rec_geometry_cache import (
    canonical_json_sha256,
    join_base_and_geometry_rows,
    load_bound_candidate_cache,
    load_geometry_cache,
    sha256_file,
)
from scripts.train_rec_geometry_reranker import (
    GEOMETRY_CANDIDATE_COUNT,
    GEOMETRY_INPUT_DIM,
    GEOMETRY_VARIANT_COUNT,
    PARENT_INFERENCE_CONTRACT_FIELDS,
    PARENT_INFERENCE_LOCAL_BATCH_SIZE,
    build_geometry_training_batch,
    load_geometry_reranker_artifact,
    load_parent_reranker_snapshot,
    materialize_parent_scores,
    _sealed_parent_materialization_metadata,
    validate_geometry_artifact,
)
from scripts.train_rec_reranker import normalize_features


FROZEN_VAL_SAMPLE_COUNT = 9508
FROZEN_EVAL_BATCH_SIZE = PARENT_INFERENCE_LOCAL_BATCH_SIZE
FROZEN_RECORD_SCHEMA = "rec-geometry-frozen-sidecar-evaluation"
FROZEN_RECORD_VERSION = 1
FROZEN_EVALUATION_CLAIM_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_artifacts/geometry_val_sidecar_once.claim"
)
FROZEN_RECORD_FIELDS = (
    "schema",
    "version",
    "sample_count",
    "hits025",
    "hits050",
    "acc025",
    "acc050",
    "parent_hits025",
    "parent_hits050",
    "parent_acc025",
    "parent_acc050",
    "fixes025",
    "breaks025",
    "fixes050",
    "breaks050",
    "geometry_weight",
    "selected_artifact_sha256",
    "parent_artifact_sha256",
    "backbone_checkpoint_sha256",
    "selection_record_sha256",
    "sidecar_evaluator_sha256",
    "record_schema_sha256",
    "base_cache_content_sha256",
    "base_cache_manifest_sha256",
    "geometry_cache_content_sha256",
    "geometry_cache_manifest_sha256",
    "geometry_cache_immutable_metadata_sha256",
    "val_parent_score_content_sha256",
    "parent_inference_contract",
    "selection_uses_validation",
    "inference_uses_ground_truth",
)
FROZEN_RECORD_SCHEMA_SHA256 = canonical_json_sha256({
    "schema": FROZEN_RECORD_SCHEMA,
    "version": FROZEN_RECORD_VERSION,
    "ordered_fields": list(FROZEN_RECORD_FIELDS),
})

_SHA_FIELDS = (
    "selected_artifact_sha256",
    "parent_artifact_sha256",
    "backbone_checkpoint_sha256",
    "selection_record_sha256",
    "sidecar_evaluator_sha256",
    "record_schema_sha256",
    "base_cache_content_sha256",
    "base_cache_manifest_sha256",
    "geometry_cache_content_sha256",
    "geometry_cache_manifest_sha256",
    "geometry_cache_immutable_metadata_sha256",
    "val_parent_score_content_sha256",
)

_SELECTION_FIELDS = frozenset((
    "candidate_count",
    "candidates",
    "code_sha256",
    "common_train_provenance",
    "created_at_utc",
    "selection_data_scope",
    "selection_rule",
    "selection_schema_version",
    "selection_uses_validation",
    "winner",
))
_SELECTION_WINNER_FIELDS = frozenset((
    "calibration_metrics",
    "candidate_order",
    "epoch",
    "geometry_weight",
    "selected_filename",
    "selected_sha256",
    "selection_score",
    "source_filename",
    "source_sha256",
))
_SELECTION_CANDIDATE_FIELDS = frozenset((
    "calibration_metrics",
    "candidate_order",
    "eligible_no_regression",
    "epoch",
    "filename",
    "geometry_weight",
    "model_config",
    "selection_score",
    "sha256",
    "training_args",
))
_SELECTION_COMMON_PROVENANCE_FIELDS = frozenset((
    "flat_parent_prior_version",
    "parent_artifact_sha256",
    "parent_inference_contract",
    "scene_split",
    "score_mode",
    "tie_policy",
    "train_base_cache_content_digest",
    "train_base_cache_manifest_digest",
    "train_geometry_cache_content_digest",
    "train_geometry_immutable_metadata_digest",
    "train_parent_score_content_sha256",
))
_SELECTION_RULE_FIELDS = frozenset((
    "eligibility", "objective", "tie_break",
))
_SELECTION_RULE = {
    "eligibility": (
        "acc025 >= frozen parent acc025 AND acc050 >= frozen parent acc050"
    ),
    "objective": (
        "min(acc025 / 0.60, acc050 / 0.47) + 0.1 * (acc025 + acc050)"
    ),
    "tie_break": "lower candidate_order (declared primary/sweep order)",
}
_SELECTION_CODE_PATHS = (
    "models/rec_geometry_reranker.py",
    "models/rec_mask_geometry.py",
    "scripts/train_rec_geometry_reranker.py",
)
_SELECTION_CODE_FIELDS = frozenset(_SELECTION_CODE_PATHS)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value):
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("JSON payload is not canonicalizable: {}".format(error))
    return (serialized + "\n").encode("utf-8")


def _file_identity(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _read_stable_file(path, label):
    resolved = Path(path).expanduser().resolve()
    try:
        initial = resolved.stat()
    except OSError as error:
        raise ValueError("{} does not exist: {}".format(label, error))
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("{} must be a regular file".format(label))
    try:
        with resolved.open("rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after = _file_identity(os.fstat(handle.fileno()))
        current = _file_identity(resolved.stat())
    except OSError as error:
        raise ValueError("could not read {}: {}".format(label, error))
    if before != after or after != current:
        raise ValueError("{} changed during stable read".format(label))
    return {
        "path": resolved,
        "bytes": snapshot,
        "sha256": hashlib.sha256(snapshot).hexdigest(),
        "identity": current,
    }


def _read_selection_record(path):
    snapshot = _read_stable_file(path, "selection record")
    try:
        record = json.loads(snapshot["bytes"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("selection record is not valid JSON: {}".format(error))
    if not isinstance(record, dict):
        raise ValueError("selection record must contain an object")
    snapshot["record"] = record
    return snapshot


def _reject_validation_selection_fields(value, path="selection"):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("selection record keys must be nonempty strings")
            lowered = key.lower()
            key_path = "{}.{}".format(path, key)
            if (key_path != "selection.selection_uses_validation"
                    and ("validation" in lowered
                         or lowered.startswith("val_"))):
                raise ValueError(
                    "selection record contains validation-derived field {}"
                    .format(key_path)
                )
            _reject_validation_selection_fields(
                item, key_path
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_validation_selection_fields(
                item, "{}[{}]".format(path, index)
            )


def _validate_no_ground_truth_artifact(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("selected geometry artifact is invalid")
    model_inputs = artifact.get("model_inputs")
    if (not isinstance(model_inputs, dict)
            or model_inputs.get("butd") is not True
            or model_inputs.get("butd_gt") is not False
            or model_inputs.get("butd_cls") is not False):
        raise ValueError(
            "selected artifact ground truth input configuration is invalid"
        )
    if (artifact.get("filter_non_gt_boxes") is not False
            or artifact.get("target_iou_policy") != "root_only"):
        raise ValueError(
            "selected artifact ground truth filter configuration is invalid"
        )
    weight = artifact.get("geometry_weight")
    if (not isinstance(weight, float)
            or not math.isfinite(weight)
            or not 0.0 <= weight <= 1.0):
        raise ValueError("selected artifact geometry weight is invalid")


def _selection_filename(value, label):
    if (not isinstance(value, str) or not value
            or Path(value).name != value
            or value in (".", "..")):
        raise ValueError("selection {} is invalid".format(label))
    return value


def calibration_score(acc025, acc050):
    """Return the exact train-only selection objective."""
    for value in (acc025, acc050):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0):
            raise ValueError("selection calibration accuracy is invalid")
    return min(float(acc025) / 0.60, float(acc050) / 0.47) + 0.1 * (
        float(acc025) + float(acc050)
    )


def _candidate_selection_outcome(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("selection candidate is invalid")
    order = candidate.get("candidate_order")
    if (not isinstance(order, int) or isinstance(order, bool) or order < 0):
        raise ValueError("selection candidate order is invalid")
    eligible = candidate.get("eligible_no_regression")
    if not isinstance(eligible, bool):
        raise ValueError("selection candidate eligibility is invalid")
    metrics = candidate.get("calibration_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("selection candidate calibration metrics are invalid")
    score = candidate.get("selection_score")
    if (not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(float(score))):
        raise ValueError("selection candidate score is invalid")

    accuracy_fields = ("acc025", "acc050", "parent_acc025", "parent_acc050")
    present = tuple(key in metrics for key in accuracy_fields)
    if any(present) and not all(present):
        raise ValueError("selection candidate calibration metrics are incomplete")
    if all(present):
        for key in accuracy_fields:
            value = metrics[key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0):
                raise ValueError(
                    "selection candidate calibration metrics are invalid"
                )
        expected_eligible = (
            float(metrics["acc025"]) >= float(metrics["parent_acc025"])
            and float(metrics["acc050"]) >= float(metrics["parent_acc050"])
        )
        expected_score = calibration_score(
            metrics["acc025"], metrics["acc050"]
        )
        if eligible is not expected_eligible:
            raise ValueError("selection candidate eligibility is inconsistent")
        if (float(score) != expected_score
                or metrics.get("score") != expected_score):
            raise ValueError("selection candidate objective score is inconsistent")
    elif metrics.get("score") != score:
        # Full production artifacts always take the branch above. This branch
        # keeps the selection helper usable with minimal synthetic artifacts.
        raise ValueError("selection candidate score differs from artifact")
    return eligible, float(score), order


def recompute_selection_winner(candidates):
    """Choose the best eligible score, breaking ties by lower declaration order."""
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("selection candidates are invalid")
    best = None
    seen_orders = set()
    for candidate in candidates:
        eligible, score, order = _candidate_selection_outcome(candidate)
        if order in seen_orders:
            raise ValueError("selection candidate orders are duplicated")
        seen_orders.add(order)
        if not eligible:
            continue
        ranking = (score, -order)
        if best is None or ranking > best[0]:
            best = (ranking, order)
    if best is None:
        raise ValueError("selection rule has no eligible candidate")
    return best[1]


def _snapshot_reference(snapshot):
    return {
        "path": snapshot["path"],
        "sha256": snapshot["sha256"],
        "identity": snapshot["identity"],
    }


def _snapshot_selection_code(selection):
    snapshots = []
    for relative_path in _SELECTION_CODE_PATHS:
        snapshot = _read_stable_file(
            _PROJECT_ROOT / relative_path,
            "selection code file {}".format(relative_path),
        )
        if snapshot["sha256"] != selection["code_sha256"][relative_path]:
            raise ValueError(
                "selection code SHA mismatch for {}".format(relative_path)
            )
        snapshots.append(_snapshot_reference(snapshot))
    return tuple(snapshots)


def _validate_selection_structure(selection, selection_path,
                                  selected_snapshot):
    if set(selection) != _SELECTION_FIELDS:
        raise ValueError("selection record fields do not match exact schema")
    if (not isinstance(selection.get("selection_schema_version"), int)
            or isinstance(selection.get("selection_schema_version"), bool)
            or selection["selection_schema_version"] != 1):
        raise ValueError("selection record version is invalid")
    if selection.get("selection_uses_validation") is not False:
        raise ValueError("selection must not use validation")
    if selection.get("selection_data_scope") != (
            "train fit/calibration scenes only"):
        raise ValueError("selection data scope is not train-only")
    if (not isinstance(selection.get("created_at_utc"), str)
            or not selection["created_at_utc"]):
        raise ValueError("selection creation timestamp is invalid")
    rule = selection.get("selection_rule")
    if (not isinstance(rule, dict) or set(rule) != _SELECTION_RULE_FIELDS
            or rule != _SELECTION_RULE):
        raise ValueError("selection rule is invalid")
    code_sha = selection.get("code_sha256")
    if (not isinstance(code_sha, dict) or set(code_sha) != _SELECTION_CODE_FIELDS
            or any(not _is_sha256(value) for value in code_sha.values())):
        raise ValueError("selection code SHA bindings are invalid")
    common = selection.get("common_train_provenance")
    if (not isinstance(common, dict)
            or set(common) != _SELECTION_COMMON_PROVENANCE_FIELDS):
        raise ValueError("selection common train provenance is invalid")

    candidates = selection.get("candidates")
    candidate_count = selection.get("candidate_count")
    if (not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count <= 0
            or not isinstance(candidates, list)
            or len(candidates) != candidate_count):
        raise ValueError("selection candidate count is invalid")
    by_order = {}
    candidate_snapshots = {}
    selection_directory = selection_path.parent
    for candidate in candidates:
        if (not isinstance(candidate, dict)
                or set(candidate) != _SELECTION_CANDIDATE_FIELDS):
            raise ValueError("selection candidate fields are invalid")
        order = candidate.get("candidate_order")
        if (not isinstance(order, int) or isinstance(order, bool)
                or order < 0 or order in by_order):
            raise ValueError("selection candidate order is invalid")
        filename = _selection_filename(
            candidate.get("filename"), "candidate filename"
        )
        if (not _is_sha256(candidate.get("sha256"))
                or not isinstance(
                    candidate.get("eligible_no_regression"), bool
                )):
            raise ValueError("selection candidate binding is invalid")
        candidate_snapshot = _read_stable_file(
            selection_directory / filename,
            "selection candidate artifact",
        )
        if candidate_snapshot["sha256"] != candidate["sha256"]:
            raise ValueError("selection candidate actual SHA mismatch")
        by_order[order] = candidate
        candidate_snapshots[order] = _snapshot_reference(candidate_snapshot)
    if set(by_order) != set(range(candidate_count)):
        raise ValueError("selection candidate orders are not contiguous")

    winner = selection.get("winner")
    if (not isinstance(winner, dict)
            or set(winner) != _SELECTION_WINNER_FIELDS):
        raise ValueError("selection winner fields are invalid")
    winner_order = winner.get("candidate_order")
    if winner_order not in by_order:
        raise ValueError("selection winner candidate order is invalid")
    candidate = by_order[winner_order]
    selected_filename = _selection_filename(
        winner.get("selected_filename"), "winner selected filename"
    )
    source_filename = _selection_filename(
        winner.get("source_filename"), "winner source filename"
    )
    if (selected_filename != selected_snapshot["path"].name
            or winner.get("selected_sha256")
            != selected_snapshot["sha256"]
            or source_filename != candidate["filename"]
            or winner.get("source_sha256") != candidate["sha256"]
            or winner.get("selected_sha256") != winner.get("source_sha256")
            or candidate.get("eligible_no_regression") is not True):
        raise ValueError("selection winner does not bind the selected artifact")
    winner_to_candidate = {
        "calibration_metrics": "calibration_metrics",
        "candidate_order": "candidate_order",
        "epoch": "epoch",
        "geometry_weight": "geometry_weight",
        "selection_score": "selection_score",
    }
    if any(winner.get(winner_key) != candidate.get(candidate_key)
           for winner_key, candidate_key in winner_to_candidate.items()):
        raise ValueError("selection winner metadata differs from candidate")
    return {
        "winner_candidate": candidate,
        "winner_order": winner_order,
        "candidates_by_order": by_order,
        "candidate_snapshots_by_order": candidate_snapshots,
    }


def _validate_selection_artifact_binding(selection, winner_candidate,
                                         artifact):
    common = selection["common_train_provenance"]
    if any(common.get(key) != artifact.get(key)
           for key in _SELECTION_COMMON_PROVENANCE_FIELDS):
        raise ValueError(
            "selection common train provenance differs from selected artifact"
        )
    candidate_bindings = (
        "calibration_metrics",
        "epoch",
        "geometry_weight",
        "model_config",
        "training_args",
    )
    if any(winner_candidate.get(key) != artifact.get(key)
           for key in candidate_bindings):
        raise ValueError("selection winner metadata differs from artifact")
    calibration_metrics = artifact.get("calibration_metrics")
    if (not isinstance(calibration_metrics, dict)
            or winner_candidate.get("selection_score")
            != calibration_metrics.get("score")):
        raise ValueError("selection score differs from selected artifact")
    _candidate_selection_outcome(winner_candidate)


def preflight_frozen_inputs(selection_record_path, selected_artifact_path,
                            parent_artifact_path, device="cuda:0"):
    """Validate all frozen, non-val inputs before acquiring the one-shot claim."""
    selection_snapshot = _read_selection_record(selection_record_path)
    selection = selection_snapshot["record"]
    _reject_validation_selection_fields(selection)

    selected_snapshot = _read_stable_file(
        selected_artifact_path, "selected geometry artifact"
    )
    parent_snapshot = _read_stable_file(
        parent_artifact_path, "parent reranker artifact"
    )
    selection_audit = _validate_selection_structure(
        selection, selection_snapshot["path"], selected_snapshot
    )
    code_snapshots = _snapshot_selection_code(selection)
    sidecar_snapshot = _read_stable_file(
        Path(__file__), "frozen sidecar evaluator"
    )

    parent = load_parent_reranker_snapshot(
        parent_snapshot["path"], device=device
    )
    parent_model, _ = parent
    if (getattr(parent_model, "_artifact_sha256", None)
            != parent_snapshot["sha256"]):
        raise ValueError("loaded parent model differs from actual artifact SHA")

    candidates = []
    for order in range(selection["candidate_count"]):
        candidate = selection_audit["candidates_by_order"][order]
        candidate_snapshot = selection_audit[
            "candidate_snapshots_by_order"
        ][order]
        candidate_model, candidate_artifact = (
            load_geometry_reranker_artifact(
                candidate_snapshot["path"], device="cpu"
            )
        )
        if (getattr(candidate_model, "_artifact_sha256", None)
                != candidate_snapshot["sha256"]):
            raise ValueError(
                "loaded selection candidate differs from actual artifact SHA"
            )
        if (candidate_artifact.get("parent_artifact_sha256")
                != parent_snapshot["sha256"]):
            raise ValueError("selection candidate parent SHA binding is invalid")
        validate_geometry_artifact(candidate_artifact, parent=parent)
        _validate_no_ground_truth_artifact(candidate_artifact)
        _validate_selection_artifact_binding(
            selection, candidate, candidate_artifact
        )
        candidates.append(candidate)

    recomputed_winner = recompute_selection_winner(candidates)
    if recomputed_winner != selection_audit["winner_order"]:
        raise ValueError("selection rule does not identify the recorded winner")

    geometry_model, geometry_artifact = load_geometry_reranker_artifact(
        selected_snapshot["path"], device=device
    )
    if (getattr(geometry_model, "_artifact_sha256", None)
            != selected_snapshot["sha256"]):
        raise ValueError("loaded geometry model differs from actual artifact SHA")
    if (geometry_artifact.get("parent_artifact_sha256")
            != parent_snapshot["sha256"]):
        raise ValueError("selected artifact parent SHA binding is invalid")
    validate_geometry_artifact(geometry_artifact, parent=parent)
    _validate_no_ground_truth_artifact(geometry_artifact)
    _validate_selection_artifact_binding(
        selection, selection_audit["winner_candidate"], geometry_artifact
    )
    geometry_model.eval().requires_grad_(False)
    parent_model.eval().requires_grad_(False)
    return {
        "selection_path": selection_snapshot["path"],
        "selected_path": selected_snapshot["path"],
        "parent_path": parent_snapshot["path"],
        "selection_record_sha256": selection_snapshot["sha256"],
        "selected_artifact_sha256": selected_snapshot["sha256"],
        "parent_artifact_sha256": parent_snapshot["sha256"],
        "sidecar_evaluator_sha256": sidecar_snapshot["sha256"],
        "record_schema_sha256": FROZEN_RECORD_SCHEMA_SHA256,
        "candidate_snapshots": tuple(
            selection_audit["candidate_snapshots_by_order"][order]
            for order in range(selection["candidate_count"])
        ),
        "code_snapshots": code_snapshots,
        "sidecar_snapshot": _snapshot_reference(sidecar_snapshot),
        "selection_uses_validation": False,
        "geometry_model": geometry_model,
        "geometry_artifact": geometry_artifact,
        "parent": parent,
    }


def _distributed_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def validate_production_runtime(device="cuda:0"):
    """Require the frozen sidecar scorer's production execution contract."""
    resolved = torch.device(device)
    if resolved != torch.device("cuda:0"):
        raise ValueError("frozen sidecar evaluation requires exactly cuda:0")
    if not torch.cuda.is_available():
        raise ValueError("frozen sidecar evaluation requires CUDA")
    if int(torch.cuda.current_device()) != 0:
        raise ValueError("frozen sidecar evaluation requires CUDA device 0")
    if _distributed_world_size() != 1:
        raise ValueError("frozen sidecar evaluation requires world_size=1")
    if torch.backends.cuda.matmul.allow_tf32 is not True:
        raise ValueError("frozen sidecar evaluation requires TF32")
    if torch.is_autocast_enabled():
        raise ValueError("frozen sidecar evaluation requires autocast disabled")
    return resolved


def _fsync_directory(path):
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_without_following_final(path):
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return absolute.parent.resolve() / absolute.name


def claim_path_for():
    """Return the authoritative one-shot goal claim registry path."""
    return _output_without_following_final(FROZEN_EVALUATION_CLAIM_PATH)


def _write_exclusive_file(path, payload, label):
    path = _output_without_following_final(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("{} already exists: {}".format(label, path))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o444)
    except FileExistsError:
        raise FileExistsError("{} already exists: {}".format(label, path))
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating {}".format(label))
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(str(path), 0o444)
    _fsync_directory(path.parent)
    return path


def acquire_evaluation_claim(output, preflight):
    claim = {
        "schema": "rec-geometry-frozen-sidecar-evaluation-claim",
        "version": 1,
        "output": str(_output_without_following_final(output)),
        "selected_artifact_sha256": preflight[
            "selected_artifact_sha256"
        ],
        "parent_artifact_sha256": preflight["parent_artifact_sha256"],
        "selection_record_sha256": preflight["selection_record_sha256"],
        "sidecar_evaluator_sha256": preflight["sidecar_evaluator_sha256"],
        "record_schema_sha256": preflight["record_schema_sha256"],
    }
    return _write_exclusive_file(
        claim_path_for(),
        _canonical_json_bytes(claim),
        "claim",
    )


def publish_immutable_json(path, payload):
    """Atomically publish canonical JSON without replacing any existing path."""
    output = _output_without_following_final(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("immutable output already exists: {}".format(output))
    data = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o444)
        try:
            os.link(str(temporary), str(output))
        except FileExistsError:
            raise FileExistsError(
                "immutable output already exists: {}".format(output)
            )
        _fsync_directory(output.parent)
        linked_identity = _file_identity(temporary.stat())
        expected_sha256 = hashlib.sha256(data).hexdigest()
        try:
            readback = _read_stable_file(output, "immutable output")
            output_stat = output.lstat()
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "immutable output read-back failed: {}".format(error)
            )
        if (output.is_symlink()
                or not stat.S_ISREG(output_stat.st_mode)
                or stat.S_IMODE(output_stat.st_mode) != 0o444
                or readback["bytes"] != data
                or readback["sha256"] != expected_sha256
                or readback["identity"] != linked_identity):
            raise RuntimeError(
                "immutable output read-back does not match publication"
            )
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_frozen_val_bundle(base_cache, geometry_cache):
    """Strict-load each complete val cache once and join exactly 9,508 rows."""
    base_path = Path(base_cache).expanduser().resolve()
    base_rows, base_manifest, base_binding = load_bound_candidate_cache(
        base_path, "val"
    )
    geometry_rows, geometry_manifest = load_geometry_cache(
        geometry_cache,
        "val",
        base_snapshot=(base_rows, base_manifest, base_binding),
    )
    geometry_binding = geometry_manifest.get("base_cache_binding")
    if (not isinstance(geometry_binding, dict)
            or geometry_binding.get("path") != str(base_path)
            or geometry_binding != base_binding):
        raise ValueError("geometry val cache base binding is invalid")
    rows = join_base_and_geometry_rows(
        base_rows,
        geometry_rows,
        base_manifest,
        geometry_manifest,
        verified_base_binding=base_binding,
    )
    if (len(rows) != FROZEN_VAL_SAMPLE_COUNT
            or geometry_manifest.get("sample_count")
            != FROZEN_VAL_SAMPLE_COUNT):
        raise ValueError("frozen sidecar evaluation requires exactly 9,508 rows")
    return {
        "rows": rows,
        "base_manifest": base_manifest,
        "base_binding": base_binding,
        "geometry_manifest": geometry_manifest,
        "geometry_manifest_sha256": canonical_json_sha256(
            geometry_manifest
        ),
    }


def validate_frozen_val_provenance(artifact, base_manifest, base_binding,
                                   geometry_manifest):
    """Bind the train-selected artifact to the val caches without GT choices."""
    if (base_manifest.get("split") != "val"
            or geometry_manifest.get("split") != "val"):
        raise ValueError("frozen sidecar caches must use split=val")
    if (base_manifest.get("sample_count") != FROZEN_VAL_SAMPLE_COUNT
            or geometry_manifest.get("sample_count")
            != FROZEN_VAL_SAMPLE_COUNT
            or base_binding.get("sample_count")
            != FROZEN_VAL_SAMPLE_COUNT):
        raise ValueError("frozen sidecar cache sample count is invalid")
    shared_fields = (
        "candidate_rule",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "model_inputs",
        "backbone_config",
        "target_iou_policy",
    )
    if any(base_manifest.get(key) != geometry_manifest.get(key)
           for key in shared_fields):
        raise ValueError("base and geometry val provenance differs")
    artifact_fields = shared_fields + (
        "geometry_cache_schema_version",
        "geometry_schema_version",
        "variant_names",
        "variant_configs",
        "regressed_variant_index",
        "min_points",
        "max_point_fraction",
        "filter_non_gt_boxes",
    )
    if any(artifact.get(key) != geometry_manifest.get(key)
           for key in artifact_fields):
        raise ValueError("selected artifact differs from geometry val provenance")
    if (artifact.get("base_cache_schema_version")
            != base_manifest.get("cache_schema_version")
            or artifact.get("base_feature_schema_version")
            != base_manifest.get("feature_schema_version")):
        raise ValueError("selected artifact differs from base val schema")
    _validate_no_ground_truth_artifact(artifact)


def _canonical_parent_flat_indices(parent_state, regressed_variant_index):
    query_indices = parent_state.get("query_indices")
    candidate_valid = parent_state.get("candidate_valid")
    top1 = parent_state.get("top1_query_index")
    if (not isinstance(query_indices, torch.Tensor)
            or not isinstance(candidate_valid, torch.Tensor)
            or not isinstance(top1, torch.Tensor)
            or query_indices.dim() != 2
            or candidate_valid.shape != query_indices.shape
            or top1.shape != (query_indices.shape[0],)):
        raise ValueError("parent state shape is invalid")
    matches = query_indices.eq(top1.unsqueeze(1)) & candidate_valid.bool()
    if not bool(matches.sum(dim=1).eq(1).all().item()):
        raise ValueError("canonical parent Top-1 is not uniquely compact")
    positions = matches.to(torch.long).argmax(dim=1)
    return positions * GEOMETRY_VARIANT_COUNT + int(regressed_variant_index)


def select_frozen_geometry_indices(model, artifact, features, valid_mask,
                                   parent_state):
    """Select Top-1 using only frozen scorer inputs, never labels or GT."""
    if (not isinstance(features, torch.Tensor)
            or features.dtype != torch.float32
            or features.dim() != 3
            or features.shape[1:] != (
                GEOMETRY_CANDIDATE_COUNT * GEOMETRY_VARIANT_COUNT,
                GEOMETRY_INPUT_DIM,
            )):
        raise ValueError("geometry scoring features have an invalid layout")
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != features.shape[:2]
            or valid_mask.device != features.device):
        raise ValueError("geometry scoring validity has an invalid layout")
    regressed = artifact.get("regressed_variant_index")
    if (not isinstance(regressed, int) or isinstance(regressed, bool)
            or not 0 <= regressed < GEOMETRY_VARIANT_COUNT):
        raise ValueError("artifact regressed variant index is invalid")
    parent_indices = _canonical_parent_flat_indices(parent_state, regressed)
    if parent_indices.device != features.device:
        parent_indices = parent_indices.to(features.device)
    if not bool(torch.gather(
            valid_mask, 1, parent_indices.unsqueeze(1)
    ).all().item()):
        raise ValueError("canonical parent candidate is not evaluator-valid")

    weight = artifact.get("geometry_weight")
    if (not isinstance(weight, float) or not math.isfinite(weight)
            or not 0.0 <= weight <= 1.0):
        raise ValueError("artifact geometry weight is invalid")
    if weight == 0.0:
        return parent_indices, parent_indices
    if model.training:
        raise ValueError("geometry scorer must be in eval mode")
    if torch.is_grad_enabled():
        raise ValueError("geometry scorer requires no_grad")
    normalized = normalize_features(
        features,
        valid_mask,
        artifact.get("feature_mean"),
        artifact.get("feature_std"),
    )
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("normalized geometry scoring features are non-finite")
    outputs = model(normalized, valid_mask)
    logits = outputs.get("ranking_logits") if isinstance(outputs, dict) else None
    if (not isinstance(logits, torch.Tensor)
            or logits.shape != valid_mask.shape):
        raise ValueError("geometry scorer ranking logits are invalid")
    geometry_valid = valid_mask.reshape(
        valid_mask.shape[0],
        GEOMETRY_CANDIDATE_COUNT,
        GEOMETRY_VARIANT_COUNT,
    )
    blended = blend_rec_geometry_scores(
        parent_state,
        logits,
        geometry_valid,
        geometry_weight=weight,
        regressed_variant_index=regressed,
    )
    if blended.get("use_parent_query_axis") is not False:
        raise RuntimeError("nonzero geometry weight did not build flat scores")
    orders = stable_flat_descending_indices(
        blended["flat_scores"], blended["flat_valid_mask"]
    )
    selected = torch.tensor(
        [order[0] for order in orders],
        dtype=torch.long,
        device=features.device,
    )
    return selected, parent_indices


def empty_metric_counts():
    return {
        "sample_count": 0,
        "hits025": 0,
        "hits050": 0,
        "parent_hits025": 0,
        "parent_hits050": 0,
        "fixes025": 0,
        "breaks025": 0,
        "fixes050": 0,
        "breaks050": 0,
    }


def accumulate_metric_counts(counts, selected_indices, parent_indices,
                             candidate_ious):
    if set(counts) != set(empty_metric_counts()):
        raise ValueError("frozen metric count fields are invalid")
    if (not isinstance(candidate_ious, torch.Tensor)
            or candidate_ious.dim() != 2
            or not torch.is_floating_point(candidate_ious)
            or not bool(torch.isfinite(candidate_ious).all().item())
            or bool(((candidate_ious < 0.0)
                     | (candidate_ious > 1.0)).any().item())):
        raise ValueError("candidate IoU labels are invalid")
    batch_size = candidate_ious.shape[0]
    for name, indices in (
            ("selected", selected_indices), ("parent", parent_indices)):
        if (not isinstance(indices, torch.Tensor)
                or indices.dtype != torch.long
                or indices.shape != (batch_size,)
                or indices.device != candidate_ious.device
                or bool((indices < 0).any().item())
                or bool((indices >= candidate_ious.shape[1]).any().item())):
            raise ValueError("{} indices are invalid".format(name))
    selected_ious = torch.gather(
        candidate_ious, 1, selected_indices.unsqueeze(1)
    ).squeeze(1)
    parent_ious = torch.gather(
        candidate_ious, 1, parent_indices.unsqueeze(1)
    ).squeeze(1)
    selected025 = selected_ious > 0.25
    selected050 = selected_ious > 0.50
    parent025 = parent_ious > 0.25
    parent050 = parent_ious > 0.50
    counts["sample_count"] += int(batch_size)
    counts["hits025"] += int(selected025.sum().item())
    counts["hits050"] += int(selected050.sum().item())
    counts["parent_hits025"] += int(parent025.sum().item())
    counts["parent_hits050"] += int(parent050.sum().item())
    counts["fixes025"] += int((selected025 & ~parent025).sum().item())
    counts["breaks025"] += int((~selected025 & parent025).sum().item())
    counts["fixes050"] += int((selected050 & ~parent050).sum().item())
    counts["breaks050"] += int((~selected050 & parent050).sum().item())
    return counts


def finalize_metric_counts(counts):
    if set(counts) != set(empty_metric_counts()):
        raise ValueError("frozen metric count fields are invalid")
    sample_count = counts.get("sample_count")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count <= 0):
        raise ValueError("frozen evaluation sample count is invalid")
    result = dict(counts)
    denominator = float(sample_count)
    result.update({
        "acc025": counts["hits025"] / denominator,
        "acc050": counts["hits050"] / denominator,
        "parent_acc025": counts["parent_hits025"] / denominator,
        "parent_acc050": counts["parent_hits050"] / denominator,
    })
    return result


def _disabled_autocast(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    if device.type == "cpu":
        cpu_amp = getattr(getattr(torch, "cpu", None), "amp", None)
        cpu_autocast = getattr(cpu_amp, "autocast", None)
        if cpu_autocast is not None:
            return cpu_autocast(enabled=False)
        return contextlib.nullcontext()
    raise ValueError("frozen sidecar scoring supports only CPU or CUDA")


def evaluate_selected_geometry_artifact(model, artifact, rows, parent,
                                        device="cuda:0"):
    """Evaluate only the artifact-selected weight in contiguous batches of 12."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("frozen sidecar rows cannot be empty")
    resolved = torch.device(device)
    model.to(device=resolved, dtype=torch.float32).eval().requires_grad_(False)
    materialize_parent_scores(
        rows,
        parent,
        device=resolved,
        local_batch_size=FROZEN_EVAL_BATCH_SIZE,
    )
    counts = empty_metric_counts()
    with torch.no_grad(), _disabled_autocast(resolved):
        for start in range(0, len(rows), FROZEN_EVAL_BATCH_SIZE):
            row_batch = rows[start:start + FROZEN_EVAL_BATCH_SIZE]
            batch = build_geometry_training_batch(row_batch, parent)
            features = batch["features"].to(
                device=resolved, dtype=torch.float32
            )
            valid = batch["valid_mask"].to(device=resolved)
            parent_state = {
                key: value.to(device=resolved)
                if isinstance(value, torch.Tensor) else value
                for key, value in batch["parent_state"].items()
            }
            selected, parent_indices = select_frozen_geometry_indices(
                model, artifact, features, valid, parent_state
            )
            # Labels are read only after frozen score selection has completed.
            candidate_ious = batch["candidate_ious"].to(device=resolved)
            accumulate_metric_counts(
                counts, selected, parent_indices, candidate_ious
            )
    return finalize_metric_counts(counts)


def sealed_parent_materialization_metadata(parent):
    return _sealed_parent_materialization_metadata(parent)


def _validate_parent_contract(contract, row_count, content_sha256):
    if (not isinstance(contract, dict)
            or set(contract) != set(PARENT_INFERENCE_CONTRACT_FIELDS)):
        raise ValueError("val parent inference contract fields are invalid")
    expected = {
        "schema": "rec-parent-inference-contract",
        "version": 1,
        "device_type": "cuda",
        "device_index": 0,
        "local_batch_size": 12,
        "world_size": 1,
        "row_order": "dataset-index-contiguous",
        "remainder_policy": "natural-remainder",
        "feature_source": "bound-base-cache-features",
        "dtype": "float32",
        "autocast": False,
        "allow_tf32": True,
        "eval": True,
        "no_grad": True,
        "score_builder": "normalized-query-reranker-rank-blend",
        "score_builder_version": 1,
        "canonical_query_tie_policy": (
            "score-desc-query-index-asc-v1"
        ),
        "content_digest_version": (
            "ordered-identity-raw-float32-sha256-v1"
        ),
        "row_count": row_count,
        "score_content_sha256": content_sha256,
    }
    if contract != expected:
        raise ValueError("val parent inference contract is not production exact")


def validate_frozen_record(record):
    if not isinstance(record, dict) or set(record) != set(FROZEN_RECORD_FIELDS):
        raise ValueError("frozen validation record fields do not match schema")
    if (record.get("schema") != FROZEN_RECORD_SCHEMA
            or not isinstance(record.get("version"), int)
            or isinstance(record.get("version"), bool)
            or record["version"] != FROZEN_RECORD_VERSION):
        raise ValueError("frozen validation record version is invalid")
    sample_count = record.get("sample_count")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count != FROZEN_VAL_SAMPLE_COUNT):
        raise ValueError(
            "frozen validation record requires exactly 9,508 samples"
        )
    count_fields = (
        "hits025", "hits050", "parent_hits025", "parent_hits050",
        "fixes025", "breaks025", "fixes050", "breaks050",
    )
    if any(not isinstance(record.get(key), int)
           or isinstance(record.get(key), bool)
           or not 0 <= record[key] <= sample_count for key in count_fields):
        raise ValueError("frozen validation record counts are invalid")
    if (record["hits050"] > record["hits025"]
            or record["parent_hits050"] > record["parent_hits025"]):
        raise ValueError("frozen validation threshold counts are inconsistent")
    for suffix in ("025", "050"):
        if record["hits" + suffix] != (
                record["parent_hits" + suffix]
                + record["fixes" + suffix]
                - record["breaks" + suffix]):
            raise ValueError("frozen validation fix/break identity is invalid")
    expected_rates = {
        "acc025": record["hits025"] / float(sample_count),
        "acc050": record["hits050"] / float(sample_count),
        "parent_acc025": record["parent_hits025"] / float(sample_count),
        "parent_acc050": record["parent_hits050"] / float(sample_count),
    }
    if any(not isinstance(record.get(key), float)
           or not math.isfinite(record[key])
           or record[key] != expected for key, expected in expected_rates.items()):
        raise ValueError("frozen validation record rates are invalid")
    weight = record.get("geometry_weight")
    if (not isinstance(weight, float) or not math.isfinite(weight)
            or not 0.0 <= weight <= 1.0):
        raise ValueError("frozen validation record weight is invalid")
    if any(not _is_sha256(record.get(key)) for key in _SHA_FIELDS):
        raise ValueError("frozen validation record SHA binding is invalid")
    if record["record_schema_sha256"] != FROZEN_RECORD_SCHEMA_SHA256:
        raise ValueError("frozen validation record schema digest is invalid")
    if (record.get("selection_uses_validation") is not False
            or record.get("inference_uses_ground_truth") is not False):
        raise ValueError("frozen validation isolation flags are invalid")
    _validate_parent_contract(
        record.get("parent_inference_contract"),
        sample_count,
        record["val_parent_score_content_sha256"],
    )
    return record


def _build_frozen_record(metrics, preflight, bundle, parent_contract,
                         parent_score_sha256):
    artifact = preflight["geometry_artifact"]
    geometry_manifest = bundle["geometry_manifest"]
    base_binding = bundle["base_binding"]
    record = {
        "schema": FROZEN_RECORD_SCHEMA,
        "version": FROZEN_RECORD_VERSION,
        "geometry_weight": float(artifact["geometry_weight"]),
        "selected_artifact_sha256": preflight[
            "selected_artifact_sha256"
        ],
        "parent_artifact_sha256": preflight["parent_artifact_sha256"],
        "backbone_checkpoint_sha256": artifact["checkpoint_sha256"],
        "selection_record_sha256": preflight["selection_record_sha256"],
        "sidecar_evaluator_sha256": preflight[
            "sidecar_evaluator_sha256"
        ],
        "record_schema_sha256": preflight["record_schema_sha256"],
        "base_cache_content_sha256": base_binding["content_sha256"],
        "base_cache_manifest_sha256": base_binding["manifest_sha256"],
        "geometry_cache_content_sha256": geometry_manifest[
            "cache_content_digest"
        ],
        "geometry_cache_manifest_sha256": bundle[
            "geometry_manifest_sha256"
        ],
        "geometry_cache_immutable_metadata_sha256": geometry_manifest[
            "immutable_metadata_digest"
        ],
        "val_parent_score_content_sha256": parent_score_sha256,
        "parent_inference_contract": parent_contract,
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
    }
    record.update(metrics)
    return validate_frozen_record(record)


def _require_unchanged_preflight_files(preflight):
    checks = (
        ("selection_path", "selection_record_sha256"),
        ("selected_path", "selected_artifact_sha256"),
        ("parent_path", "parent_artifact_sha256"),
    )
    for path_key, sha_key in checks:
        snapshot = _read_stable_file(
            preflight[path_key], "frozen preflight input"
        )
        if snapshot["sha256"] != preflight[sha_key]:
            raise ValueError("frozen preflight input changed during evaluation")
    retained = list(preflight.get("candidate_snapshots", ()))
    retained.extend(preflight.get("code_snapshots", ()))
    sidecar_snapshot = preflight.get("sidecar_snapshot")
    if sidecar_snapshot is not None:
        retained.append(sidecar_snapshot)
    for expected in retained:
        if (not isinstance(expected, dict)
                or "path" not in expected
                or not _is_sha256(expected.get("sha256"))):
            raise ValueError("frozen preflight snapshot is invalid")
        current = _read_stable_file(
            expected["path"], "frozen preflight snapshot"
        )
        if current["sha256"] != expected["sha256"]:
            raise ValueError("frozen preflight input changed during evaluation")


def _validate_output_separation(output, args):
    output = _output_without_following_final(output)
    input_files = (
        args.selection_record,
        args.selected_artifact,
        args.parent_artifact,
    )
    if any(output == Path(path).expanduser().resolve() for path in input_files):
        raise ValueError("frozen record output must not alias an input artifact")
    for cache in (args.base_cache, args.geometry_cache):
        cache_path = Path(cache).expanduser().resolve()
        try:
            within = os.path.commonpath([
                str(output), str(cache_path)
            ]) == str(cache_path)
        except ValueError:
            within = False
        if within:
            raise ValueError("frozen record output must be outside val caches")


def run_frozen_sidecar_evaluation(args):
    """Consume the one-shot claim, evaluate once, and freeze one exact record."""
    output = _output_without_following_final(args.output)
    _validate_output_separation(output, args)
    preflight = preflight_frozen_inputs(
        args.selection_record,
        args.selected_artifact,
        args.parent_artifact,
        device=args.device,
    )
    device = validate_production_runtime(args.device)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "immutable output already exists: {}".format(output)
        )
    acquire_evaluation_claim(output, preflight)

    # The claim deliberately remains after every failure below this point.
    bundle = load_frozen_val_bundle(args.base_cache, args.geometry_cache)
    validate_frozen_val_provenance(
        preflight["geometry_artifact"],
        bundle["base_manifest"],
        bundle["base_binding"],
        bundle["geometry_manifest"],
    )
    metrics = evaluate_selected_geometry_artifact(
        preflight["geometry_model"],
        preflight["geometry_artifact"],
        bundle["rows"],
        preflight["parent"],
        device=device,
    )
    if metrics.get("sample_count") != FROZEN_VAL_SAMPLE_COUNT:
        raise ValueError("frozen evaluation did not consume exactly 9,508 rows")
    parent_contract, parent_score_sha256 = (
        sealed_parent_materialization_metadata(preflight["parent"])
    )
    _require_unchanged_preflight_files(preflight)
    record = _build_frozen_record(
        metrics,
        preflight,
        bundle,
        parent_contract,
        parent_score_sha256,
    )
    publish_immutable_json(output, record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one train-selected REC geometry artifact against the "
            "complete val sidecar exactly once."
        )
    )
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--selected-artifact", required=True)
    parser.add_argument("--selection-record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_frozen_sidecar_evaluation(args)
    output = _output_without_following_final(args.output)
    print(
        "Frozen validation record: {} sha256={}".format(
            output, sha256_file(output)
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    main()
