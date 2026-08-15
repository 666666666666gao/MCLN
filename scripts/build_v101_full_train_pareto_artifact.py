#!/usr/bin/env python
"""Fit and freeze the V101 Pareto hierarchy on all train scenes."""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat

import torch

import scripts.build_v99_pareto_contextual_artifact as b99
from models.rec_pareto_contextual_hierarchy import (
    ParetoContextualHierarchicalReranker,
    V99_AGGREGATE_MARGIN,
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
)


V101_ARTIFACT_SCHEMA = "rec-pareto-contextual-full-train-artifact-v1"
V101_ARTIFACT_VERSION = 1
V101_RESULT_SHA256 = (
    "2cb453b130306449901bed9c337984aad7f8b7048d05bd4240b5077f0de9ac1e"
)
V101_SCRIPT_SHA256 = (
    "034291a86b08a2386b3861f8dbe732acd3ae34bacdf2f70145cf5ecde9e5af92"
)
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
EXPECTED_SWITCHES = 5882
EXPECTED_DELTA_025 = 159
EXPECTED_DELTA_050 = 520
ARTIFACT_FIELDS = {
    "schema", "version", "deployable", "validation_data_accessed",
    "input_sha256", "feature_names", "model_config", "model_state_dict",
    "normalization", "normalization_sha256", "training_contract", "policy",
    "fit", "oof_evidence",
}


def _validate_oof_result(path):
    if b99._sha256(path) != V101_RESULT_SHA256:
        raise ValueError("V101 result SHA-256 changed")
    result = json.loads(Path(path).read_text(encoding="ascii"))
    oof = result.get("oof")
    diagnostics = oof.get("diagnostics") if isinstance(oof, dict) else None
    predicates = oof.get("predicates") if isinstance(oof, dict) else None
    if (result.get("schema") != "rec-v99-pareto-full-train-scene-oof-v1"
            or result.get("validation_data_accessed") is not False
            or result.get("prior_calibration_used_for_selection") is not False
            or not isinstance(oof, dict) or oof.get("passed") is not True
            or not isinstance(predicates, dict) or not predicates
            or not all(predicates.values())
            or not isinstance(diagnostics, dict)
            or diagnostics.get("sample_count") != EXPECTED_ROW_COUNT
            or diagnostics.get("delta_hits025") != EXPECTED_DELTA_025
            or diagnostics.get("delta_hits050") != EXPECTED_DELTA_050
            or diagnostics.get("switches") != EXPECTED_SWITCHES
            or result.get("protected_before") != result.get("protected_after")):
        raise ValueError("V101 OOF result does not satisfy frozen gate")
    folds = diagnostics.get("fold_deltas", {}).values()
    if len(diagnostics.get("fold_deltas", {})) != 5 or not all(
            row["hits025"] > 0 and row["hits050"] > 0 for row in folds):
        raise ValueError("V101 OOF folds are not all strictly positive")
    return result


def _training_contract():
    return {
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
        "training_rows": "all_scanrefer_train_rows",
    }


def validate_v101_artifact(
        artifact, expected_parent_sha256=None,
        expected_geometry_sha256=None, expected_feature_names=None):
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_FIELDS:
        raise ValueError("V101 artifact fields differ from schema")
    if (artifact.get("schema") != V101_ARTIFACT_SCHEMA
            or artifact.get("version") != V101_ARTIFACT_VERSION
            or artifact.get("deployable") is not True
            or artifact.get("validation_data_accessed") is not False):
        raise ValueError("V101 artifact top-level policy is invalid")
    inputs = artifact.get("input_sha256")
    if (not isinstance(inputs, dict) or not inputs
            or any(not b99._is_sha256(value) for value in inputs.values())):
        raise ValueError("V101 input provenance is invalid")
    if expected_parent_sha256 is not None and inputs.get(
            "parent") != expected_parent_sha256:
        raise ValueError("V101 parent artifact SHA mismatch")
    if expected_geometry_sha256 is not None and inputs.get(
            "geometry") != expected_geometry_sha256:
        raise ValueError("V101 geometry artifact SHA mismatch")
    features = artifact.get("feature_names")
    if not isinstance(features, dict):
        raise ValueError("V101 feature schema is invalid")
    try:
        expected_features = build_hierarchical_feature_names(
            features.get("geometry_input", ())
        )
    except (TypeError, ValueError):
        raise ValueError("V101 feature schema is invalid") from None
    if features != expected_features:
        raise ValueError("V101 feature schema is invalid")
    if (expected_feature_names is not None
            and features["geometry_input"] != list(expected_feature_names)):
        raise ValueError("V101 geometry feature binding changed")
    if artifact.get("model_config") != {
            "hidden_dim": V99_HIDDEN_DIM, "dropout": V99_DROPOUT}:
        raise ValueError("V101 model config changed")
    normalization = artifact.get("normalization")
    _validate_hierarchical_normalization(normalization)
    if (artifact.get("normalization_sha256") != normalization["sha256"]
            or normalization["sha256"]
            != _normalization_sha256(normalization)):
        raise ValueError("V101 normalization binding changed")
    if artifact.get("training_contract") != _training_contract():
        raise ValueError("V101 training contract changed")
    if artifact.get("policy") != {
            "aggregate_margin": V99_AGGREGATE_MARGIN,
            "aggregate_weights": [2.0, 1.0],
            "require_positive_delta025": True,
            "require_positive_delta050": True,
        }:
        raise ValueError("V101 Pareto policy changed")
    fit = artifact.get("fit")
    if (not isinstance(fit, dict) or set(fit) != {
            "row_count", "scene_count", "scene_fold_sha256",
            "deployable_rows_sha256", "candidate_iou_sha256",
            "normalization_sha256", "final_epoch", "model_state_sha256",
        } or fit.get("row_count") != EXPECTED_ROW_COUNT
            or fit.get("scene_count") != EXPECTED_SCENE_COUNT
            or fit.get("normalization_sha256") != normalization["sha256"]
            or any(not b99._is_sha256(fit.get(name)) for name in (
                "scene_fold_sha256", "deployable_rows_sha256",
                "candidate_iou_sha256", "normalization_sha256",
                "model_state_sha256"))):
        raise ValueError("V101 fit evidence is invalid")
    evidence = artifact.get("oof_evidence")
    expected_evidence = {
        "path": evidence.get("path") if isinstance(evidence, dict) else None,
        "sha256": V101_RESULT_SHA256,
        "script_sha256": V101_SCRIPT_SHA256,
        "switches": EXPECTED_SWITCHES,
        "delta_hits025": EXPECTED_DELTA_025,
        "delta_hits050": EXPECTED_DELTA_050,
        "bootstrap025_lower_bound_95": 118,
        "bootstrap050_lower_bound_95": 421,
        "all_folds_strictly_positive": True,
    }
    if (evidence != expected_evidence
            or not isinstance(expected_evidence["path"], str)
            or not Path(expected_evidence["path"]).is_absolute()):
        raise ValueError("V101 OOF evidence is invalid")
    state = artifact.get("model_state_dict")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor)
                   or value.device.type != "cpu" for value in state.values())
            or b99._state_sha256(state) != fit["model_state_sha256"]):
        raise ValueError("V101 model state evidence is invalid")
    model = ParetoContextualHierarchicalReranker(**artifact["model_config"])
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("V101 model state is incompatible") from error
    return copy.deepcopy(artifact["model_config"])


def load_v101_artifact(
        path, device="cpu", expected_artifact_sha256=None,
        parent_sha256=None, geometry_sha256=None,
        expected_geometry_feature_names=None):
    resolved, payload, sha256 = b99._stable_bytes(path)
    if expected_artifact_sha256 is not None and sha256 != expected_artifact_sha256:
        raise ValueError("V101 artifact SHA-256 mismatch")
    try:
        artifact = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize V101 artifact") from error
    config = validate_v101_artifact(
        artifact,
        expected_parent_sha256=parent_sha256,
        expected_geometry_sha256=geometry_sha256,
        expected_feature_names=expected_geometry_feature_names,
    )
    model = ParetoContextualHierarchicalReranker(**config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA V101 artifact load requested but unavailable")
    model.to(resolved_device).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = sha256
    return model, artifact


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v101-result", required=True)
    parser.add_argument("--v101-script", required=True)
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
        raise FileExistsError("V101 artifact or receipt output already exists")
    if b99._sha256(args.v101_script) != V101_SCRIPT_SHA256:
        raise ValueError("V101 experiment script SHA-256 changed")
    result = _validate_oof_result(args.v101_result)
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
        raise ValueError("V101 live input provenance differs from OOF")
    records = materialize_hierarchical_rows(
        loaded["joined_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
    )
    if (len(records) != EXPECTED_ROW_COUNT
            or len(set(record["scan_id"] for record in records))
            != EXPECTED_SCENE_COUNT):
        raise ValueError("V101 full-train materialization changed")
    normalization = fit_hierarchical_normalization(records)
    model, epochs = b99._fit_model(records, normalization, args.device)
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
        "schema": V101_ARTIFACT_SCHEMA,
        "version": V101_ARTIFACT_VERSION,
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
        "training_contract": _training_contract(),
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
            "deployable_rows_sha256": canonical_hierarchical_deployable_sha256(
                records
            ),
            "candidate_iou_sha256": canonical_hierarchical_candidate_iou_sha256(
                records
            ),
            "normalization_sha256": normalization["sha256"],
            "final_epoch": epochs[-1],
            "model_state_sha256": b99._state_sha256(state),
        },
        "oof_evidence": {
            "path": str(Path(args.v101_result).expanduser().absolute()),
            "sha256": V101_RESULT_SHA256,
            "script_sha256": V101_SCRIPT_SHA256,
            "switches": diagnostics["switches"],
            "delta_hits025": diagnostics["delta_hits025"],
            "delta_hits050": diagnostics["delta_hits050"],
            "bootstrap025_lower_bound_95": diagnostics[
                "bootstrap025"
            ]["lower_bound_95"],
            "bootstrap050_lower_bound_95": diagnostics[
                "bootstrap050"
            ]["lower_bound_95"],
            "all_folds_strictly_positive": all(
                row["hits025"] > 0 and row["hits050"] > 0
                for row in diagnostics["fold_deltas"].values()
            ),
        },
    }
    validate_v101_artifact(
        artifact,
        expected_parent_sha256=loaded["input_sha256"]["parent"],
        expected_geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_feature_names=loaded["geometry_artifact"]["feature_names"],
    )
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    b99._exclusive_write(artifact_output, buffer.getvalue())
    artifact_sha = b99._sha256(artifact_output)
    loaded_model, reloaded = load_v101_artifact(
        artifact_output, device="cpu",
        expected_artifact_sha256=artifact_sha,
        parent_sha256=loaded["input_sha256"]["parent"],
        geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_geometry_feature_names=loaded[
            "geometry_artifact"
        ]["feature_names"],
    )
    if b99._state_sha256(loaded_model.state_dict()) != artifact[
            "fit"]["model_state_sha256"]:
        raise RuntimeError("V101 strict reload changed model state")
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V101 refit")
    receipt = {
        "schema": "rec-pareto-contextual-full-train-artifact-receipt-v1",
        "version": 1,
        "artifact": {
            "path": str(artifact_output),
            "sha256": artifact_sha,
            "mode": stat.S_IMODE(artifact_output.stat().st_mode),
            "size": artifact_output.stat().st_size,
        },
        "oof_result_sha256": V101_RESULT_SHA256,
        "validation_data_accessed": False,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "fit": copy.deepcopy(reloaded["fit"]),
    }
    receipt_payload = json.dumps(
        receipt, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    b99._exclusive_write(receipt_output, receipt_payload)
    print(json.dumps({
        "artifact": receipt["artifact"],
        "receipt": str(receipt_output),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "fit": receipt["fit"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
