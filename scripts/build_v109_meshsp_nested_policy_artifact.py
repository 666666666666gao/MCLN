#!/usr/bin/env python
"""Fit and freeze the V109 MeshSP nested-policy hierarchy."""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import stat

import torch

import scripts.build_v108_meshsp_pareto_artifact as b108
import scripts.build_v99_pareto_contextual_artifact as b99
from models.rec_pareto_contextual_hierarchy import (
    ParetoContextualHierarchicalReranker,
    V99_DROPOUT,
    V99_HIDDEN_DIM,
)
from scripts.run_v108_meshsp_pareto_oof import (
    load_v108_meshsp_training_inputs,
    validate_v108_materialization_artifact,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    build_hierarchical_feature_names,
    build_hierarchical_scene_folds,
    canonical_hierarchical_candidate_iou_sha256,
    canonical_hierarchical_deployable_sha256,
    canonical_hierarchical_scene_fold_sha256,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    materialize_hierarchical_rows,
)


V109_ARTIFACT_SCHEMA = (
    "rec-pareto-contextual-meshsp-nested-policy-full-train-artifact-v1"
)
V109_ARTIFACT_VERSION = 1
V109_OOF_SCHEMA = "rec-v109-meshsp-nested-policy-full-train-scene-oof-v1"
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
EXPECTED_CORRECTED_SCENE_COUNT = 361
EXPECTED_REGULAR_SCENE_COUNT = 201
MIN_DELTA_025 = 72
MIN_DELTA_050 = 74
EXPECTED_POLICY = {
    "aggregate_margin": 0.15,
    "aggregate_weights": [2.0, 1.0],
    "min_head_gain025": 0.02,
    "min_head_gain050": 0.0,
    "require_strict_head_gain025": True,
    "require_strict_head_gain050": True,
    "selection_procedure": "leave_one_model_oof_fold_out_meta_calibration",
}


def _validate_oof_result(path, script_path):
    result_sha256 = b99._sha256(path)
    script_sha256 = b99._sha256(script_path)
    result = json.loads(Path(path).read_text(encoding="ascii"))
    oof = result.get("oof")
    diagnostics = oof.get("diagnostics") if isinstance(oof, dict) else None
    predicates = oof.get("predicates") if isinstance(oof, dict) else None
    subgroups = oof.get("subgroups") if isinstance(oof, dict) else None
    global_selection = result.get("global_policy_selection")
    selected = (
        global_selection.get("selected")
        if isinstance(global_selection, dict) else None
    )
    selected_policy = (
        selected.get("policy") if isinstance(selected, dict) else None
    )
    expected_grid_policy = {
        "aggregate_margin": EXPECTED_POLICY["aggregate_margin"],
        "min_head_gain025": EXPECTED_POLICY["min_head_gain025"],
        "min_head_gain050": EXPECTED_POLICY["min_head_gain050"],
    }
    meta_folds = result.get("meta_folds")
    if (result.get("schema") != V109_OOF_SCHEMA
            or result.get("validation_data_accessed") is not False
            or result.get("prior_calibration_used_for_selection") is not False
            or result.get("protocol", {}).get(
                "held_labels_visible_to_policy_selection") is not False
            or not isinstance(oof, dict) or oof.get("passed") is not True
            or not isinstance(predicates, dict) or not predicates
            or not all(value is True for value in predicates.values())
            or not isinstance(diagnostics, dict)
            or diagnostics.get("sample_count") != EXPECTED_ROW_COUNT
            or diagnostics.get("delta_hits025", -1) < MIN_DELTA_025
            or diagnostics.get("delta_hits050", -1) < MIN_DELTA_050
            or diagnostics.get("bootstrap025", {}).get(
                "lower_bound_95", 0) <= 0
            or diagnostics.get("bootstrap050", {}).get(
                "lower_bound_95", 0) <= 0
            or selected_policy != expected_grid_policy
            or not isinstance(meta_folds, list) or len(meta_folds) != 5
            or any(row.get("selected_policy") != expected_grid_policy
                   for row in meta_folds)
            or result.get("source_sha256", {}).get("v109") != script_sha256
            or result.get("protected_before") != result.get("protected_after")
            or result.get("protected_metadata_before")
            != result.get("protected_metadata_after")):
        raise ValueError("V109 OOF result does not satisfy frozen gate")
    if len(diagnostics.get("fold_deltas", {})) != 5 or not all(
            row["hits025"] > 0 and row["hits050"] > 0
            for row in diagnostics["fold_deltas"].values()):
        raise ValueError("V109 held folds are not all strictly positive")
    if not isinstance(subgroups, dict) or set(subgroups) != {
            "corrected", "regular"}:
        raise ValueError("V109 subgroup evidence changed")
    expected_scenes = {
        "corrected": EXPECTED_CORRECTED_SCENE_COUNT,
        "regular": EXPECTED_REGULAR_SCENE_COUNT,
    }
    for name, scene_count in expected_scenes.items():
        subgroup = subgroups[name]
        subgroup_diagnostics = subgroup.get("diagnostics", {})
        if (subgroup.get("scene_count") != scene_count
                or subgroup_diagnostics.get("delta_hits025", -1) < 0
                or subgroup_diagnostics.get("delta_hits050", -1) < 0):
            raise ValueError("V109 subgroup gate changed")
    return result, result_sha256, script_sha256


def validate_v109_artifact(
        artifact, expected_parent_sha256=None,
        expected_geometry_sha256=None, expected_feature_names=None):
    if (not isinstance(artifact, dict)
            or artifact.get("schema") != V109_ARTIFACT_SCHEMA
            or artifact.get("version") != V109_ARTIFACT_VERSION
            or artifact.get("policy") != EXPECTED_POLICY):
        raise ValueError("V109 artifact top-level contract changed")
    probe = copy.deepcopy(artifact)
    probe["schema"] = b108.V108_ARTIFACT_SCHEMA
    probe["version"] = b108.V108_ARTIFACT_VERSION
    probe["policy"] = {
        "aggregate_margin": b108.V99_AGGREGATE_MARGIN,
        "aggregate_weights": [2.0, 1.0],
        "require_positive_delta025": True,
        "require_positive_delta050": True,
    }
    return b108.validate_v108_artifact(
        probe,
        expected_parent_sha256=expected_parent_sha256,
        expected_geometry_sha256=expected_geometry_sha256,
        expected_feature_names=expected_feature_names,
    )


def load_v109_artifact(
        path, device="cpu", expected_artifact_sha256=None,
        parent_sha256=None, geometry_sha256=None,
        expected_geometry_feature_names=None):
    resolved, payload, sha256 = b99._stable_bytes(path)
    if expected_artifact_sha256 is not None and sha256 != expected_artifact_sha256:
        raise ValueError("V109 artifact SHA-256 mismatch")
    try:
        artifact = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize V109 artifact") from error
    config = validate_v109_artifact(
        artifact,
        expected_parent_sha256=parent_sha256,
        expected_geometry_sha256=geometry_sha256,
        expected_feature_names=expected_geometry_feature_names,
    )
    model = ParetoContextualHierarchicalReranker(**config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA V109 artifact load requested but unavailable")
    model.to(resolved_device).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = sha256
    return model, artifact


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v109-result", required=True)
    parser.add_argument("--v109-script", required=True)
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
        raise FileExistsError("V109 artifact or receipt output already exists")
    result, result_sha256, script_sha256 = _validate_oof_result(
        args.v109_result, args.v109_script
    )
    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    metadata_paths = {
        "candidate_receipt": (
            Path(args.base_cache).expanduser().resolve().parent
            / "candidate_train_receipt.json"
        ),
        "geometry_receipt": (
            Path(args.base_cache).expanduser().resolve().parent
            / "geometry_train_receipt.json"
        ),
        "base_manifest": (
            Path(args.base_cache).expanduser().resolve() / "manifest.json"
        ),
        "geometry_manifest": (
            Path(args.geometry_cache).expanduser().resolve() / "manifest.json"
        ),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    metadata_before = {
        name: b99._sha256(path) for name, path in metadata_paths.items()
    }
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    if loaded["input_sha256"] != result["input_sha256"]:
        raise ValueError("V109 live input provenance differs from OOF")
    records = materialize_hierarchical_rows(
        loaded["joined_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    if (len(records) != EXPECTED_ROW_COUNT
            or len(set(record["scan_id"] for record in records))
            != EXPECTED_SCENE_COUNT):
        raise ValueError("V109 full-train materialization changed")
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
        "schema": V109_ARTIFACT_SCHEMA,
        "version": V109_ARTIFACT_VERSION,
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
        "training_contract": b108._training_contract(),
        "policy": copy.deepcopy(EXPECTED_POLICY),
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
            "path": str(Path(args.v109_result).expanduser().absolute()),
            "sha256": result_sha256,
            "script_sha256": script_sha256,
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
            "subgroups": {
                name: {
                    "scene_count": row["scene_count"],
                    "row_count": row["row_count"],
                    "delta_hits025": row["diagnostics"]["delta_hits025"],
                    "delta_hits050": row["diagnostics"]["delta_hits050"],
                }
                for name, row in result["oof"]["subgroups"].items()
            },
            "predicates": copy.deepcopy(result["oof"]["predicates"]),
        },
    }
    validate_v109_artifact(
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
    loaded_model, reloaded = load_v109_artifact(
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
        raise RuntimeError("V109 strict reload changed model state")
    protected_after = capture_immutable_artifact_identities(protected_paths)
    metadata_after = {
        name: b99._sha256(path) for name, path in metadata_paths.items()
    }
    if protected_after != protected_before or metadata_after != metadata_before:
        raise RuntimeError("V109 protected inputs changed during full fit")
    receipt = {
        "schema": "rec-v109-meshsp-nested-policy-artifact-receipt-v1",
        "version": 1,
        "artifact": {
            "path": str(artifact_output),
            "sha256": artifact_sha,
            "mode": stat.S_IMODE(artifact_output.stat().st_mode),
            "size": artifact_output.stat().st_size,
        },
        "oof_result_sha256": result_sha256,
        "oof_script_sha256": script_sha256,
        "validation_data_accessed": False,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_metadata_before": metadata_before,
        "protected_metadata_after": metadata_after,
        "fit": copy.deepcopy(reloaded["fit"]),
        "policy": copy.deepcopy(reloaded["policy"]),
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
        "policy": receipt["policy"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
