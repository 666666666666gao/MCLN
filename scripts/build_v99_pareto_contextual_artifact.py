#!/usr/bin/env python
"""Refit and publish the frozen V99 Pareto contextual REC artifact."""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat

import torch

import scripts.run_v95_threshold_aligned_listwise_hierarchical as v95
from models.rec_pareto_contextual_hierarchy import (
    ParetoContextualHierarchicalReranker,
    V99_AGGREGATE_MARGIN,
    V99_ARTIFACT_SCHEMA,
    V99_ARTIFACT_VERSION,
    V99_DROPOUT,
    V99_HIDDEN_DIM,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    _normalization_sha256,
    _validate_hierarchical_normalization,
    build_hierarchical_feature_names,
    build_hierarchical_scene_folds,
    canonical_hierarchical_candidate_iou_sha256,
    canonical_hierarchical_deployable_sha256,
    canonical_hierarchical_scene_fold_sha256,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


V99_RESULT_SHA256 = (
    "db42ef5853fb36fba9bdc53afb719bff9eb5a3f9e772475a4c76c363db01572d"
)
V99_SCRIPT_SHA256 = (
    "78ff3fb141ab9aa8334285cd1d9e3c37845c7769710b166ee0fce00c33fac4a9"
)
ARTIFACT_FIELDS = {
    "schema", "version", "deployable", "validation_data_accessed",
    "input_sha256", "feature_names", "model_config", "model_state_dict",
    "normalization", "normalization_sha256", "training_contract", "policy",
    "fit", "oof_evidence",
}


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fit_model(records, statistics, device):
    original = v95.HierarchicalQueryVariantReranker
    v95.HierarchicalQueryVariantReranker = ParetoContextualHierarchicalReranker
    try:
        return v95.fit_graded_listwise_model(records, statistics, device)
    finally:
        v95.HierarchicalQueryVariantReranker = original


def _validate_oof_result(path):
    if _sha256(path) != V99_RESULT_SHA256:
        raise ValueError("V99 result SHA-256 changed")
    result = json.loads(Path(path).read_text(encoding="ascii"))
    oof = result.get("oof")
    diagnostics = oof.get("diagnostics") if isinstance(oof, dict) else None
    predicates = oof.get("predicates") if isinstance(oof, dict) else None
    if (result.get("schema")
            != "rec-pareto-contextual-hierarchical-train-only-v1"
            or result.get("validation_data_accessed") is not False
            or result.get("contaminated_calibration_accessed") is not False
            or not isinstance(oof, dict) or oof.get("passed") is not True
            or not isinstance(predicates, dict)
            or not predicates or not all(predicates.values())
            or not isinstance(diagnostics, dict)
            or diagnostics.get("delta_hits025") != 175
            or diagnostics.get("delta_hits050") != 474
            or diagnostics.get("switches") != 5186
            or result.get("protected_before") != result.get("protected_after")):
        raise ValueError("V99 OOF result does not satisfy frozen gate")
    return result


def validate_v99_artifact(
        artifact, expected_parent_sha256=None,
        expected_geometry_sha256=None, expected_feature_names=None):
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_FIELDS:
        raise ValueError("V99 artifact fields differ from schema")
    if (artifact.get("schema") != V99_ARTIFACT_SCHEMA
            or artifact.get("version") != V99_ARTIFACT_VERSION
            or artifact.get("deployable") is not True
            or artifact.get("validation_data_accessed") is not False):
        raise ValueError("V99 artifact top-level policy is invalid")
    inputs = artifact.get("input_sha256")
    if (not isinstance(inputs, dict) or not inputs
            or any(not _is_sha256(value) for value in inputs.values())):
        raise ValueError("V99 input provenance is invalid")
    if (expected_parent_sha256 is not None
            and inputs.get("parent") != expected_parent_sha256):
        raise ValueError("V99 parent artifact SHA mismatch")
    if (expected_geometry_sha256 is not None
            and inputs.get("geometry") != expected_geometry_sha256):
        raise ValueError("V99 geometry artifact SHA mismatch")
    features = artifact.get("feature_names")
    if not isinstance(features, dict):
        raise ValueError("V99 feature schema is invalid")
    try:
        expected_features = build_hierarchical_feature_names(
            features.get("geometry_input", ())
        )
    except (TypeError, ValueError):
        raise ValueError("V99 feature schema is invalid") from None
    if features != expected_features:
        raise ValueError("V99 feature schema is invalid")
    if (expected_feature_names is not None
            and features["geometry_input"] != list(expected_feature_names)):
        raise ValueError("V99 geometry feature binding changed")
    if artifact.get("model_config") != {
            "hidden_dim": V99_HIDDEN_DIM, "dropout": V99_DROPOUT}:
        raise ValueError("V99 model config changed")
    normalization = artifact.get("normalization")
    _validate_hierarchical_normalization(normalization)
    if (artifact.get("normalization_sha256") != normalization["sha256"]
            or normalization["sha256"]
            != _normalization_sha256(normalization)):
        raise ValueError("V99 normalization binding changed")
    expected_training = {
        "seed": 0,
        "epochs": 12,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gradient_clip_norm": 1.0,
        "dropout": V99_DROPOUT,
        "weight_decay": 1e-3,
        "objective": "bounded_iou_plus_2hit025_plus_hit050_soft_listwise",
        "target_temperature": 0.25,
        "architecture": {
            "query_context_layers": 1,
            "attention_heads": 4,
            "feedforward_dim": 256,
            "activation": "gelu",
            "permutation_equivariant": True,
            "padding_masked": True,
        },
    }
    if artifact.get("training_contract") != expected_training:
        raise ValueError("V99 training contract changed")
    if artifact.get("policy") != {
            "aggregate_margin": V99_AGGREGATE_MARGIN,
            "aggregate_weights": [2.0, 1.0],
            "require_positive_delta025": True,
            "require_positive_delta050": True,
        }:
        raise ValueError("V99 Pareto policy changed")
    fit = artifact.get("fit")
    if (not isinstance(fit, dict) or set(fit) != {
            "row_count", "scene_count", "scene_fold_sha256",
            "deployable_rows_sha256", "candidate_iou_sha256",
            "normalization_sha256", "final_epoch", "model_state_sha256",
        } or fit.get("row_count") != 33040
            or fit.get("scene_count") != 506
            or fit.get("normalization_sha256") != normalization["sha256"]
            or any(not _is_sha256(fit.get(name)) for name in (
                "scene_fold_sha256", "deployable_rows_sha256",
                "candidate_iou_sha256", "normalization_sha256",
                "model_state_sha256"))):
        raise ValueError("V99 fit evidence is invalid")
    evidence = artifact.get("oof_evidence")
    if (not isinstance(evidence, dict) or evidence != {
            "path": evidence.get("path"),
            "sha256": V99_RESULT_SHA256,
            "script_sha256": V99_SCRIPT_SHA256,
            "switches": 5186,
            "delta_hits025": 175,
            "delta_hits050": 474,
            "bootstrap025_lower_bound_95": 132,
            "bootstrap050_lower_bound_95": 385,
            "all_folds_positive": True,
        } or not isinstance(evidence.get("path"), str)
            or not Path(evidence["path"]).is_absolute()):
        raise ValueError("V99 OOF evidence is invalid")
    state = artifact.get("model_state_dict")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor)
                   or value.device.type != "cpu" for value in state.values())
            or _state_sha256(state) != fit["model_state_sha256"]):
        raise ValueError("V99 model state evidence is invalid")
    model = ParetoContextualHierarchicalReranker(**artifact["model_config"])
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("V99 model state is incompatible") from error
    return copy.deepcopy(artifact["model_config"])


def _stable_bytes(path):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(path)
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("V99 artifact must be a regular non-symlink file")
    if stat.S_IMODE(entry.st_mode) != 0o444:
        raise ValueError("V99 artifact mode must be 0444")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    live = os.stat(path, follow_symlinks=False)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(live):
        raise ValueError("V99 artifact changed during stable load")
    return path, payload, hashlib.sha256(payload).hexdigest()


def load_v99_artifact(
        path, device="cpu", expected_artifact_sha256=None,
        parent_sha256=None, geometry_sha256=None,
        expected_geometry_feature_names=None):
    resolved, payload, sha256 = _stable_bytes(path)
    if (expected_artifact_sha256 is not None
            and sha256 != expected_artifact_sha256):
        raise ValueError("V99 artifact SHA-256 mismatch")
    try:
        artifact = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize V99 artifact") from error
    config = validate_v99_artifact(
        artifact,
        expected_parent_sha256=parent_sha256,
        expected_geometry_sha256=geometry_sha256,
        expected_feature_names=expected_geometry_feature_names,
    )
    model = ParetoContextualHierarchicalReranker(**config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA V99 artifact load requested but unavailable")
    model.to(resolved_device).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = sha256
    return model, artifact


def _exclusive_write(path, payload):
    output = Path(path).expanduser().absolute()
    descriptor = os.open(
        str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("V99 artifact write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v99-result", required=True)
    parser.add_argument("--v99-script", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--artifact-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    artifact_output = Path(args.artifact_output).expanduser().absolute()
    receipt_output = Path(args.receipt_output).expanduser().absolute()
    if artifact_output.exists() or receipt_output.exists():
        raise FileExistsError("V99 artifact or receipt output already exists")
    if _sha256(args.v99_script) != V99_SCRIPT_SHA256:
        raise ValueError("V99 experiment script SHA-256 changed")
    result = _validate_oof_result(args.v99_result)
    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    loaded = load_residual_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    if loaded["input_sha256"] != result["input_sha256"]:
        raise ValueError("V99 live input provenance differs from OOF")
    split = split_residual_joined_rows(loaded["joined_rows"])
    records = materialize_hierarchical_rows(
        split["fit_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=False,
    )
    normalization = fit_hierarchical_normalization(records)
    model, epochs = _fit_model(records, normalization, args.device)
    model.eval().requires_grad_(False)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    diagnostics = result["oof"]["diagnostics"]
    artifact = {
        "schema": V99_ARTIFACT_SCHEMA,
        "version": V99_ARTIFACT_VERSION,
        "deployable": True,
        "validation_data_accessed": False,
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "feature_names": build_hierarchical_feature_names(
            loaded["geometry_artifact"]["feature_names"]
        ),
        "model_config": {
            "hidden_dim": V99_HIDDEN_DIM,
            "dropout": V99_DROPOUT,
        },
        "model_state_dict": state,
        "normalization": copy.deepcopy(normalization),
        "normalization_sha256": normalization["sha256"],
        "training_contract": {
            "seed": 0,
            "epochs": 12,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "gradient_clip_norm": 1.0,
            "dropout": V99_DROPOUT,
            "weight_decay": 1e-3,
            "objective": "bounded_iou_plus_2hit025_plus_hit050_soft_listwise",
            "target_temperature": 0.25,
            "architecture": {
                "query_context_layers": 1,
                "attention_heads": 4,
                "feedforward_dim": 256,
                "activation": "gelu",
                "permutation_equivariant": True,
                "padding_masked": True,
            },
        },
        "policy": {
            "aggregate_margin": V99_AGGREGATE_MARGIN,
            "aggregate_weights": [2.0, 1.0],
            "require_positive_delta025": True,
            "require_positive_delta050": True,
        },
        "fit": {
            "row_count": len(records),
            "scene_count": len(set(scene_folds)),
            "scene_fold_sha256": canonical_hierarchical_scene_fold_sha256(
                scene_folds
            ),
            "deployable_rows_sha256": (
                canonical_hierarchical_deployable_sha256(records)
            ),
            "candidate_iou_sha256": (
                canonical_hierarchical_candidate_iou_sha256(records)
            ),
            "normalization_sha256": normalization["sha256"],
            "final_epoch": epochs[-1],
            "model_state_sha256": _state_sha256(state),
        },
        "oof_evidence": {
            "path": str(Path(args.v99_result).expanduser().absolute()),
            "sha256": V99_RESULT_SHA256,
            "script_sha256": V99_SCRIPT_SHA256,
            "switches": diagnostics["switches"],
            "delta_hits025": diagnostics["delta_hits025"],
            "delta_hits050": diagnostics["delta_hits050"],
            "bootstrap025_lower_bound_95": diagnostics[
                "bootstrap025"
            ]["lower_bound_95"],
            "bootstrap050_lower_bound_95": diagnostics[
                "bootstrap050"
            ]["lower_bound_95"],
            "all_folds_positive": all(
                row["hits025"] > 0 and row["hits050"] > 0
                for row in diagnostics["fold_deltas"].values()
            ),
        },
    }
    validate_v99_artifact(
        artifact,
        expected_parent_sha256=loaded["input_sha256"]["parent"],
        expected_geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_feature_names=loaded["geometry_artifact"]["feature_names"],
    )
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_write(artifact_output, buffer.getvalue())
    artifact_sha = _sha256(artifact_output)
    loaded_model, reloaded = load_v99_artifact(
        artifact_output,
        device="cpu",
        expected_artifact_sha256=artifact_sha,
        parent_sha256=loaded["input_sha256"]["parent"],
        geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_geometry_feature_names=loaded[
            "geometry_artifact"
        ]["feature_names"],
    )
    if _state_sha256(loaded_model.state_dict()) != artifact[
            "fit"]["model_state_sha256"]:
        raise RuntimeError("V99 strict reload changed model state")
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V99 refit")
    receipt = {
        "schema": "rec-pareto-contextual-artifact-receipt-v1",
        "version": 1,
        "artifact": {
            "path": str(artifact_output),
            "sha256": artifact_sha,
            "mode": stat.S_IMODE(artifact_output.stat().st_mode),
            "size": artifact_output.stat().st_size,
        },
        "oof_result_sha256": V99_RESULT_SHA256,
        "validation_data_accessed": False,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "fit": copy.deepcopy(reloaded["fit"]),
    }
    receipt_payload = json.dumps(
        receipt, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    _exclusive_write(receipt_output, receipt_payload)
    print(json.dumps({
        "artifact": receipt["artifact"],
        "receipt": str(receipt_output),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "fit": receipt["fit"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
