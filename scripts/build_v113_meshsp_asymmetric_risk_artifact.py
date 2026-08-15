#!/usr/bin/env python
"""Fit and freeze the V113 three-member asymmetric-risk committee."""

import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat

import torch

import scripts.build_v108_meshsp_pareto_artifact as b108
import scripts.build_v99_pareto_contextual_artifact as b99
from models.rec_pareto_contextual_hierarchy import (
    AsymmetricRiskContextualHierarchyCommittee,
    V99_DROPOUT,
    V99_HIDDEN_DIM,
    V113_ARTIFACT_SCHEMA,
    V113_ARTIFACT_VERSION,
    V113_MEMBER_COUNT,
)
from scripts.build_v109_meshsp_nested_policy_artifact import (
    load_v109_artifact,
)
from scripts.run_v108_meshsp_pareto_oof import (
    load_v108_meshsp_training_inputs,
    validate_v108_materialization_artifact,
)
from scripts.run_v111_meshsp_anchor_committee_oof import (
    ENSEMBLE_SEEDS,
    fit_seeded_model,
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


V113_OOF_SCHEMA = "rec-v113-meshsp-asymmetric-risk-cache-replay-v1"
V113_RESULT_SHA256 = (
    "ced399bca041cfa1f4213671100347f4a2423783aee4936ce7a82f785605e61d"
)
V113_SCRIPT_SHA256 = (
    "439c75c081c3f445564ad36a55dfb4ab92443061ee889301297081ab4b4a2ee3"
)
V112_RESULT_SHA256 = (
    "128ce636d27234db7fca4fb23bd5d30945928d9ac9dcd1cf8139c38670a41b96"
)
V109_ARTIFACT_SHA256 = (
    "20db69ddc27680a035384277bc48cd44109215e3d7d1158cdc4a4f21ff7c785b"
)
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
EXPECTED_CORRECTED_SCENE_COUNT = 361
EXPECTED_REGULAR_SCENE_COUNT = 201
EXPECTED_POLICY = {
    "aggregate_lcb_margin": 0.12,
    "aggregate_weights": [2.0, 1.0],
    "min_head_lcb025": 0.02,
    "min_head_lcb050": 0.0,
    "risk_lambda025": 0.5,
    "risk_lambda050": 0.25,
    "anchor_member_index": 0,
    "member_seeds": [0, 1, 2],
    "risk_definition": "rms_member_gain_deviation_from_anchor",
    "require_strict_head_lcb025": True,
    "require_strict_head_lcb050": True,
    "selection_procedure": "leave_one_model_oof_fold_out_meta_calibration",
}
ARTIFACT_FIELDS = frozenset({
    "schema", "version", "deployable", "validation_data_accessed",
    "input_sha256", "feature_names", "model_config",
    "model_state_dict", "normalization", "normalization_sha256",
    "training_contract", "policy", "fit", "oof_evidence",
})


def _validate_oof_result(path, script_path):
    result_sha256 = b99._sha256(path)
    script_sha256 = b99._sha256(script_path)
    if result_sha256 != V113_RESULT_SHA256:
        raise ValueError("V113 result SHA-256 changed")
    if script_sha256 != V113_SCRIPT_SHA256:
        raise ValueError("V113 script SHA-256 changed")
    result = json.loads(Path(path).read_text(encoding="ascii"))
    oof = result.get("oof")
    diagnostics = oof.get("diagnostics") if isinstance(oof, dict) else None
    predicates = oof.get("predicates") if isinstance(oof, dict) else None
    deployment = result.get("deployment_policy")
    policy = deployment.get("policy") if isinstance(deployment, dict) else None
    expected_replay_policy = {
        key: EXPECTED_POLICY[key] for key in (
            "aggregate_lcb_margin", "min_head_lcb025",
            "min_head_lcb050", "risk_lambda025", "risk_lambda050",
        )
    }
    source = result.get("source")
    v112_source = source.get("v112") if isinstance(source, dict) else None
    if (result.get("schema") != V113_OOF_SCHEMA
            or result.get("version") != 1
            or result.get("validation_data_accessed") is not False
            or result.get("deployable") is not False
            or result.get("protocol", {}).get(
                "held_labels_visible_to_margin_selection") is not False
            or not isinstance(oof, dict) or oof.get("passed") is not True
            or not isinstance(predicates, dict) or not predicates
            or not all(value is True for value in predicates.values())
            or not isinstance(diagnostics, dict)
            or diagnostics.get("sample_count") != EXPECTED_ROW_COUNT
            or diagnostics.get("switches") != 3553
            or diagnostics.get("delta_hits025") != 78
            or diagnostics.get("delta_hits050") != 240
            or diagnostics.get("bootstrap025", {}).get(
                "lower_bound_95") != 42
            or diagnostics.get("bootstrap050", {}).get(
                "lower_bound_95") != 184
            or not isinstance(deployment, dict)
            or policy != expected_replay_policy
            or deployment.get("winning_votes") != 4
            or result.get("source_sha256", {}).get("v113")
            != script_sha256
            or not isinstance(v112_source, dict)
            or v112_source.get("sha256") != V112_RESULT_SHA256
            or result.get("protected_before")
            != result.get("protected_after")):
        raise ValueError("V113 OOF result does not satisfy the frozen gate")
    fold_deltas = diagnostics.get("fold_deltas", {})
    if len(fold_deltas) != 5 or not all(
            row.get("hits025", 0) > 0 and row.get("hits050", 0) > 0
            for row in fold_deltas.values()):
        raise ValueError("V113 held folds are not all strictly positive")
    subgroups = oof.get("subgroups")
    expected_subgroups = {
        "corrected": (EXPECTED_CORRECTED_SCENE_COUNT, 57, 172, 27, 127),
        "regular": (EXPECTED_REGULAR_SCENE_COUNT, 21, 68, 2, 36),
    }
    if not isinstance(subgroups, dict) or set(subgroups) != set(
            expected_subgroups):
        raise ValueError("V113 subgroup evidence changed")
    for name, expected in expected_subgroups.items():
        row = subgroups[name]
        subgroup_diagnostics = row.get("diagnostics", {})
        actual = (
            row.get("scene_count"),
            subgroup_diagnostics.get("delta_hits025"),
            subgroup_diagnostics.get("delta_hits050"),
            subgroup_diagnostics.get("bootstrap025", {}).get(
                "lower_bound_95"),
            subgroup_diagnostics.get("bootstrap050", {}).get(
                "lower_bound_95"),
        )
        if actual != expected:
            raise ValueError("V113 {} subgroup gate changed".format(name))
    v112_path = Path(v112_source.get("path", "")).expanduser().resolve(
        strict=True
    )
    if b99._sha256(v112_path) != V112_RESULT_SHA256:
        raise ValueError("V112 source report SHA-256 changed")
    v112 = json.loads(v112_path.read_text(encoding="ascii"))
    inputs = v112.get("input_sha256")
    if (v112.get("validation_data_accessed") is not False
            or not isinstance(inputs, dict) or not inputs
            or any(not b99._is_sha256(value) for value in inputs.values())):
        raise ValueError("V112 input provenance changed")
    return result, result_sha256, script_sha256, copy.deepcopy(inputs)


def _member_state(state, index):
    prefix = "members.{}.".format(index)
    extracted = {
        name[len(prefix):]: value
        for name, value in state.items() if name.startswith(prefix)
    }
    if not extracted:
        raise ValueError("V113 committee member state is missing")
    return extracted


def _readonly_identity(path):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("V109 anchor must be a regular non-symlink file")
    mode = stat.S_IMODE(entry.st_mode)
    if mode != 0o444:
        raise ValueError("V109 anchor must have mode 0444")
    return {
        "path": str(path.resolve(strict=True)),
        "device": int(entry.st_dev),
        "inode": int(entry.st_ino),
        "mode": mode,
        "size": int(entry.st_size),
        "mtime_ns": int(entry.st_mtime_ns),
        "ctime_ns": int(entry.st_ctime_ns),
        "sha256": b99._sha256(path),
    }


def _valid_final_epoch(value):
    if not isinstance(value, dict) or set(value) != {
            "epoch", "loss", "query_loss", "variant_loss"}:
        return False
    if (not isinstance(value.get("epoch"), int)
            or isinstance(value.get("epoch"), bool)
            or value["epoch"] <= 0):
        return False
    for name in ("loss", "query_loss", "variant_loss"):
        item = value.get(name)
        if (not isinstance(item, (int, float)) or isinstance(item, bool)
                or not math.isfinite(float(item)) or float(item) <= 0.0):
            return False
    return True


def validate_v113_artifact(
        artifact, expected_parent_sha256=None,
        expected_geometry_sha256=None, expected_feature_names=None):
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_FIELDS:
        raise ValueError("V113 artifact fields differ from schema")
    if (artifact.get("schema") != V113_ARTIFACT_SCHEMA
            or artifact.get("version") != V113_ARTIFACT_VERSION
            or artifact.get("deployable") is not True
            or artifact.get("validation_data_accessed") is not False
            or artifact.get("policy") != EXPECTED_POLICY):
        raise ValueError("V113 artifact top-level contract changed")
    inputs = artifact.get("input_sha256")
    if (not isinstance(inputs, dict) or not inputs
            or any(not b99._is_sha256(value) for value in inputs.values())):
        raise ValueError("V113 input provenance is invalid")
    if (expected_parent_sha256 is not None
            and inputs.get("parent") != expected_parent_sha256):
        raise ValueError("V113 parent artifact SHA mismatch")
    if (expected_geometry_sha256 is not None
            and inputs.get("geometry") != expected_geometry_sha256):
        raise ValueError("V113 geometry artifact SHA mismatch")
    features = artifact.get("feature_names")
    if not isinstance(features, dict):
        raise ValueError("V113 feature schema is invalid")
    try:
        expected_features = build_hierarchical_feature_names(
            features.get("geometry_input", ())
        )
    except (TypeError, ValueError):
        raise ValueError("V113 feature schema is invalid") from None
    if features != expected_features:
        raise ValueError("V113 feature schema is invalid")
    if (expected_feature_names is not None
            and features["geometry_input"] != list(expected_feature_names)):
        raise ValueError("V113 geometry feature binding changed")
    expected_config = {
        "hidden_dim": V99_HIDDEN_DIM,
        "dropout": V99_DROPOUT,
        "member_count": V113_MEMBER_COUNT,
    }
    if artifact.get("model_config") != expected_config:
        raise ValueError("V113 model config changed")
    normalization = artifact.get("normalization")
    b108._validate_hierarchical_normalization(normalization)
    if (artifact.get("normalization_sha256") != normalization["sha256"]
            or normalization["sha256"]
            != b108._normalization_sha256(normalization)):
        raise ValueError("V113 normalization binding changed")
    if artifact.get("training_contract") != b108._training_contract():
        raise ValueError("V113 training contract changed")
    fit = artifact.get("fit")
    fit_fields = {
        "row_count", "scene_count", "scene_fold_sha256",
        "deployable_rows_sha256", "candidate_iou_sha256",
        "normalization_sha256", "member_seeds", "members",
        "model_state_sha256", "anchor_matches_v109",
    }
    if (not isinstance(fit, dict) or set(fit) != fit_fields
            or fit.get("row_count") != EXPECTED_ROW_COUNT
            or fit.get("scene_count") != EXPECTED_SCENE_COUNT
            or fit.get("normalization_sha256") != normalization["sha256"]
            or fit.get("member_seeds") != list(ENSEMBLE_SEEDS)
            or fit.get("anchor_matches_v109") is not True
            or any(not b99._is_sha256(fit.get(name)) for name in (
                "scene_fold_sha256", "deployable_rows_sha256",
                "candidate_iou_sha256", "normalization_sha256",
                "model_state_sha256"))):
        raise ValueError("V113 fit evidence is invalid")
    members = fit.get("members")
    if (not isinstance(members, list) or len(members) != V113_MEMBER_COUNT
            or [row.get("seed") for row in members]
            != list(ENSEMBLE_SEEDS)
            or any(set(row) != {"seed", "final_epoch", "state_sha256"}
                   or not _valid_final_epoch(row.get("final_epoch"))
                   or not b99._is_sha256(row.get("state_sha256"))
                   for row in members)):
        raise ValueError("V113 member fit evidence is invalid")
    evidence = artifact.get("oof_evidence")
    evidence_fields = {
        "path", "sha256", "script_sha256", "switches",
        "delta_hits025", "delta_hits050",
        "bootstrap025_lower_bound_95", "bootstrap050_lower_bound_95",
        "all_folds_strictly_positive", "subgroups", "predicates",
        "deployment_policy", "anchor_raw_prediction_sha256",
    }
    if (not isinstance(evidence, dict) or set(evidence) != evidence_fields
            or not isinstance(evidence.get("path"), str)
            or not Path(evidence["path"]).is_absolute()
            or evidence.get("sha256") != V113_RESULT_SHA256
            or evidence.get("script_sha256") != V113_SCRIPT_SHA256
            or evidence.get("switches") != 3553
            or evidence.get("delta_hits025") != 78
            or evidence.get("delta_hits050") != 240
            or evidence.get("bootstrap025_lower_bound_95") != 42
            or evidence.get("bootstrap050_lower_bound_95") != 184
            or evidence.get("all_folds_strictly_positive") is not True
            or evidence.get("deployment_policy") != EXPECTED_POLICY
            or not b99._is_sha256(
                evidence.get("anchor_raw_prediction_sha256"))
            or not isinstance(evidence.get("predicates"), dict)
            or not evidence["predicates"]
            or not all(value is True
                       for value in evidence["predicates"].values())):
        raise ValueError("V113 OOF evidence is invalid")
    subgroups = evidence.get("subgroups")
    if not isinstance(subgroups, dict) or set(subgroups) != {
            "corrected", "regular"}:
        raise ValueError("V113 OOF subgroup evidence is invalid")
    state = artifact.get("model_state_dict")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor)
                   or value.device.type != "cpu" for value in state.values())
            or b99._state_sha256(state) != fit["model_state_sha256"]):
        raise ValueError("V113 model state evidence is invalid")
    model = AsymmetricRiskContextualHierarchyCommittee(**expected_config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("V113 model state is incompatible") from error
    for index, row in enumerate(members):
        if b99._state_sha256(_member_state(state, index)) != row[
                "state_sha256"]:
            raise ValueError("V113 member state evidence changed")
    return copy.deepcopy(expected_config)


def load_v113_artifact(
        path, device="cpu", expected_artifact_sha256=None,
        parent_sha256=None, geometry_sha256=None,
        expected_geometry_feature_names=None):
    resolved, payload, sha256 = b99._stable_bytes(path)
    if (expected_artifact_sha256 is not None
            and sha256 != expected_artifact_sha256):
        raise ValueError("V113 artifact SHA-256 mismatch")
    try:
        artifact = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize V113 artifact") from error
    config = validate_v113_artifact(
        artifact,
        expected_parent_sha256=parent_sha256,
        expected_geometry_sha256=geometry_sha256,
        expected_feature_names=expected_geometry_feature_names,
    )
    model = AsymmetricRiskContextualHierarchyCommittee(**config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA V113 artifact load requested but unavailable")
    model.to(resolved_device).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = sha256
    return model, artifact


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--v113-result", required=True)
    parser.add_argument("--v113-script", required=True)
    parser.add_argument("--v109-anchor-artifact", required=True)
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
    if (artifact_output.exists() or artifact_output.is_symlink()
            or receipt_output.exists() or receipt_output.is_symlink()):
        raise FileExistsError("V113 artifact or receipt output already exists")
    result, result_sha, script_sha, expected_inputs = _validate_oof_result(
        args.v113_result, args.v113_script
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
    protected_before["v109_anchor"] = _readonly_identity(
        args.v109_anchor_artifact
    )
    if protected_before["v109_anchor"]["sha256"] != V109_ARTIFACT_SHA256:
        raise ValueError("V109 anchor artifact SHA-256 changed")
    metadata_before = {
        name: b99._sha256(path) for name, path in metadata_paths.items()
    }
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    if loaded["input_sha256"] != expected_inputs:
        raise ValueError("V113 live input provenance differs from OOF")
    records = materialize_hierarchical_rows(
        loaded["joined_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    scene_ids = {record["scan_id"] for record in records}
    if (len(records) != EXPECTED_ROW_COUNT
            or len(scene_ids) != EXPECTED_SCENE_COUNT):
        raise ValueError("V113 full-train materialization changed")
    normalization = fit_hierarchical_normalization(records)
    member_states = []
    member_evidence = []
    for seed in ENSEMBLE_SEEDS:
        model, epochs = fit_seeded_model(
            records, normalization, args.device, seed
        )
        model.eval().requires_grad_(False)
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        member_states.append(state)
        row = {
            "seed": int(seed),
            "final_epoch": copy.deepcopy(epochs[-1]),
            "state_sha256": b99._state_sha256(state),
        }
        member_evidence.append(row)
        print(json.dumps({
            "completed_fullfit_member": row
        }, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    anchor_model, anchor_artifact = load_v109_artifact(
        args.v109_anchor_artifact,
        device="cpu",
        expected_artifact_sha256=V109_ARTIFACT_SHA256,
        parent_sha256=loaded["input_sha256"]["parent"],
        geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_geometry_feature_names=loaded[
            "geometry_artifact"
        ]["feature_names"],
    )
    anchor_matches_v109 = all(
        name in anchor_artifact["model_state_dict"]
        and torch.equal(
            value, anchor_artifact["model_state_dict"][name]
        ) for name, value in member_states[0].items()
    ) and set(member_states[0]) == set(anchor_model.state_dict())
    if not anchor_matches_v109:
        raise RuntimeError("V113 seed-0 fullfit no longer equals V109")

    model_config = {
        "hidden_dim": V99_HIDDEN_DIM,
        "dropout": V99_DROPOUT,
        "member_count": V113_MEMBER_COUNT,
    }
    committee = AsymmetricRiskContextualHierarchyCommittee(**model_config)
    for member, state in zip(committee.members, member_states):
        member.load_state_dict(state, strict=True)
    committee.eval().requires_grad_(False)
    committee_state = {
        name: value.detach().cpu().clone()
        for name, value in committee.state_dict().items()
    }
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    diagnostics = result["oof"]["diagnostics"]
    subgroups = result["oof"]["subgroups"]
    artifact = {
        "schema": V113_ARTIFACT_SCHEMA,
        "version": V113_ARTIFACT_VERSION,
        "deployable": True,
        "validation_data_accessed": False,
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "feature_names": build_hierarchical_feature_names(
            loaded["geometry_artifact"]["feature_names"]
        ),
        "model_config": model_config,
        "model_state_dict": committee_state,
        "normalization": copy.deepcopy(normalization),
        "normalization_sha256": normalization["sha256"],
        "training_contract": b108._training_contract(),
        "policy": copy.deepcopy(EXPECTED_POLICY),
        "fit": {
            "row_count": len(records),
            "scene_count": len(scene_folds),
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
            "member_seeds": list(ENSEMBLE_SEEDS),
            "members": member_evidence,
            "model_state_sha256": b99._state_sha256(committee_state),
            "anchor_matches_v109": True,
        },
        "oof_evidence": {
            "path": str(Path(args.v113_result).expanduser().absolute()),
            "sha256": result_sha,
            "script_sha256": script_sha,
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
                for name, row in subgroups.items()
            },
            "predicates": copy.deepcopy(result["oof"]["predicates"]),
            "deployment_policy": copy.deepcopy(EXPECTED_POLICY),
            "anchor_raw_prediction_sha256": result[
                "anchor_raw_prediction_sha256"
            ],
        },
    }
    validate_v113_artifact(
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
    loaded_model, reloaded = load_v113_artifact(
        artifact_output,
        device="cpu",
        expected_artifact_sha256=artifact_sha,
        parent_sha256=loaded["input_sha256"]["parent"],
        geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_geometry_feature_names=loaded[
            "geometry_artifact"
        ]["feature_names"],
    )
    if b99._state_sha256(loaded_model.state_dict()) != artifact[
            "fit"]["model_state_sha256"]:
        raise RuntimeError("V113 strict reload changed committee state")
    protected_after = capture_immutable_artifact_identities(protected_paths)
    protected_after["v109_anchor"] = _readonly_identity(
        args.v109_anchor_artifact
    )
    metadata_after = {
        name: b99._sha256(path) for name, path in metadata_paths.items()
    }
    if protected_after != protected_before or metadata_after != metadata_before:
        raise RuntimeError("V113 protected inputs changed during full fit")
    receipt = {
        "schema": "rec-v113-meshsp-asymmetric-risk-artifact-receipt-v1",
        "version": 1,
        "artifact": {
            "path": str(artifact_output),
            "sha256": artifact_sha,
            "mode": stat.S_IMODE(artifact_output.stat().st_mode),
            "size": artifact_output.stat().st_size,
        },
        "oof_result_sha256": result_sha,
        "oof_script_sha256": script_sha,
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
