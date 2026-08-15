#!/usr/bin/env python
"""Nested scene-cross-fitted policy calibration for MeshSP V109."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

import scripts.run_v99_pareto_contextual_hierarchical as v99
from models.rec_hierarchical_reranker import build_hierarchical_scene_folds
from scripts.run_v108_meshsp_pareto_oof import (
    EXPECTED_CORRECTED_SCENE_COUNT,
    EXPECTED_FALLBACK_SCENE_COUNT,
    EXPECTED_REGULAR_SCENE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SCENE_COUNT,
    FALLBACK_MANIFEST_SHA256,
    file_sha256,
    load_v108_meshsp_training_inputs,
    validate_v108_materialization_artifact,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    capture_immutable_artifact_identities,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


V108_REPORT_SHA256 = (
    "72ca54b2db0bca829011a2f480c458c0a3e450a492dd77de9d8411e84f3e9162"
)
MIN_DELTA_025 = 72
MIN_DELTA_050 = 74
MARGINS = (0.10, 0.12, v99.V97_MARGIN, 0.15, 0.18, 0.22)
MIN_HEAD_025 = (0.0, 0.0025, 0.005, 0.01, 0.02)
POLICY_GRID = tuple(
    {
        "aggregate_margin": float(margin),
        "min_head_gain025": float(minimum),
        "min_head_gain050": 0.0,
    }
    for margin in MARGINS
    for minimum in MIN_HEAD_025
)
BASE_POLICY = {
    "aggregate_margin": float(v99.V97_MARGIN),
    "min_head_gain025": 0.0,
    "min_head_gain050": 0.0,
}


def policy_accept_mask(proposals, baselines, aggregate_gain, head_gain, policy):
    if policy not in POLICY_GRID:
        raise ValueError("V109 policy is outside the frozen grid")
    return (
        proposals.ne(baselines)
        & aggregate_gain.ge(policy["aggregate_margin"])
        & head_gain[:, 0].gt(policy["min_head_gain025"])
        & head_gain[:, 1].gt(policy["min_head_gain050"])
    )


def policy_summary(
        indices, proposals, baselines, aggregate_gain, head_gain,
        baseline_ious, proposal_ious, fold_ids, policy):
    indices = torch.as_tensor(indices, dtype=torch.long)
    accepted = policy_accept_mask(
        proposals[indices], baselines[indices], aggregate_gain[indices],
        head_gain[indices], policy,
    )
    selected_ious = torch.where(
        accepted, proposal_ious[indices], baseline_ious[indices]
    )
    base_ious = baseline_ious[indices]
    delta025_bits = (
        selected_ious.gt(0.25).long() - base_ious.gt(0.25).long()
    )
    delta050_bits = (
        selected_ious.gt(0.50).long() - base_ious.gt(0.50).long()
    )
    folds = sorted(set(int(value) for value in fold_ids[indices].tolist()))
    fold_deltas = {}
    for fold in folds:
        mask = fold_ids[indices].eq(fold)
        fold_deltas[str(fold)] = {
            "hits025": int(delta025_bits[mask].sum().item()),
            "hits050": int(delta050_bits[mask].sum().item()),
        }
    return {
        "policy": copy.deepcopy(policy),
        "row_count": int(indices.numel()),
        "switches": int(accepted.sum().item()),
        "delta_hits025": int(delta025_bits.sum().item()),
        "delta_hits050": int(delta050_bits.sum().item()),
        "fold_deltas": fold_deltas,
    }


def select_nested_policy(
        indices, proposals, baselines, aggregate_gain, head_gain,
        baseline_ious, proposal_ious, fold_ids):
    reference = policy_summary(
        indices, proposals, baselines, aggregate_gain, head_gain,
        baseline_ious, proposal_ious, fold_ids, BASE_POLICY,
    )
    minimum_delta050 = max(1, math.ceil(0.5 * reference["delta_hits050"]))
    candidates = []
    for policy in POLICY_GRID:
        summary = policy_summary(
            indices, proposals, baselines, aggregate_gain, head_gain,
            baseline_ious, proposal_ious, fold_ids, policy,
        )
        summary["eligible"] = bool(
            summary["delta_hits025"] > 0
            and summary["delta_hits050"] >= minimum_delta050
            and all(row["hits025"] >= 0 and row["hits050"] >= 0
                    for row in summary["fold_deltas"].values())
        )
        candidates.append(summary)
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("V109 frozen policy grid has no eligible candidate")
    winner = max(
        eligible,
        key=lambda row: (
            row["delta_hits025"],
            row["delta_hits050"],
            -row["switches"],
            row["policy"]["min_head_gain025"],
            row["policy"]["aggregate_margin"],
        ),
    )
    return {
        "selection_rule": (
            "maximize_delta025_then_delta050_then_fewer_switches_on_"
            "other_four_scene_oof_folds"
        ),
        "reference_delta050": reference["delta_hits050"],
        "minimum_delta050": minimum_delta050,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected": copy.deepcopy(winner),
        "candidates": candidates,
    }


def acceptance_gate(diagnostics, subgroups):
    folds = diagnostics["fold_deltas"].values()
    return {
        "delta025_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits025"] >= MIN_DELTA_025
        ),
        "delta050_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits050"] >= MIN_DELTA_050
        ),
        "all_folds_strictly_positive025": all(
            row["hits025"] > 0 for row in folds
        ),
        "all_folds_strictly_positive050": all(
            row["hits050"] > 0
            for row in diagnostics["fold_deltas"].values()
        ),
        "bootstrap025_lower_bound_positive": (
            diagnostics["bootstrap025"]["lower_bound_95"] > 0
        ),
        "bootstrap050_lower_bound_positive": (
            diagnostics["bootstrap050"]["lower_bound_95"] > 0
        ),
        "corrected_scenes_nonnegative025": (
            subgroups["corrected"]["diagnostics"]["delta_hits025"] >= 0
        ),
        "corrected_scenes_nonnegative050": (
            subgroups["corrected"]["diagnostics"]["delta_hits050"] >= 0
        ),
        "regular_scenes_nonnegative025": (
            subgroups["regular"]["diagnostics"]["delta_hits025"] >= 0
        ),
        "regular_scenes_nonnegative050": (
            subgroups["regular"]["diagnostics"]["delta_hits050"] >= 0
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v108-report", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--fallback-scenes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(str(output))

    source_path = Path(args.v108_report).expanduser().resolve(strict=True)
    if file_sha256(source_path) != V108_REPORT_SHA256:
        raise ValueError("V108 source report SHA-256 changed")
    source = json.loads(source_path.read_text(encoding="ascii"))
    source_predicates = source.get("oof", {}).get("predicates", {})
    if (source.get("validation_data_accessed") is not False
            or source.get("oof", {}).get("passed") is not False
            or source.get("oof", {}).get("diagnostics", {}).get(
                "delta_hits025") != 70
            or source.get("oof", {}).get("diagnostics", {}).get(
                "delta_hits050") != 245
            or source_predicates.get(
                "delta025_at_least_oracle_scaled_gap") is not False
            or any(value is not True for key, value in source_predicates.items()
                   if key != "delta025_at_least_oracle_scaled_gap")):
        raise ValueError("V108 source failure contract changed")

    fallback_path = Path(args.fallback_scenes).expanduser().resolve(strict=True)
    if file_sha256(fallback_path) != FALLBACK_MANIFEST_SHA256:
        raise ValueError("MeshSP fallback-scene manifest SHA-256 changed")
    fallback_scenes = {
        line.strip() for line in fallback_path.read_text(
            encoding="ascii"
        ).splitlines() if line.strip()
    }
    if len(fallback_scenes) != EXPECTED_FALLBACK_SCENE_COUNT:
        raise ValueError("MeshSP fallback-scene count changed")

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
        name: file_sha256(path) for name, path in metadata_paths.items()
    }
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    split = split_residual_joined_rows(loaded["joined_rows"])
    joined_rows = loaded["joined_rows"]
    if len(split["fit_rows"]) + len(split["calibration_rows"]) != len(
            joined_rows):
        raise RuntimeError("historical split no longer reconstructs full train")
    scan_ids = [row["base"]["scan_id"] for row in joined_rows]
    if (len(joined_rows) != EXPECTED_SAMPLE_COUNT
            or len(set(scan_ids)) != EXPECTED_SCENE_COUNT):
        raise ValueError("V109 full-train identity changed")
    training_scenes = set(scan_ids)
    corrected_scenes = training_scenes & fallback_scenes
    regular_scenes = training_scenes - fallback_scenes
    if (len(corrected_scenes) != EXPECTED_CORRECTED_SCENE_COUNT
            or len(regular_scenes) != EXPECTED_REGULAR_SCENE_COUNT):
        raise ValueError("V109 subgroup partition changed")

    records = materialize_hierarchical_rows(
        joined_rows, loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    fold_ids = torch.tensor([
        scene_folds[record["scan_id"]] for record in records
    ], dtype=torch.long)
    fold_indices = {
        fold: torch.nonzero(fold_ids.eq(fold), as_tuple=False).flatten()
        for fold in range(5)
    }
    if torch.cat([fold_indices[fold] for fold in range(5)]).sort().values.tolist(
            ) != list(range(EXPECTED_SAMPLE_COUNT)):
        raise RuntimeError("V109 folds do not partition all rows")

    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    proposals = baselines.clone()
    aggregate_gain = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.float32)
    head_gain = torch.zeros(EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32)
    model_folds = []
    for held in range(5):
        train_indices = torch.cat([
            fold_indices[fold] for fold in range(5) if fold != held
        ]).sort().values.tolist()
        held_rows = fold_indices[held].tolist()
        train_scenes = {records[index]["scan_id"] for index in train_indices}
        held_scenes = {records[index]["scan_id"] for index in held_rows}
        if train_scenes & held_scenes:
            raise RuntimeError("scene leakage in V109 model fold")
        model, statistics, epochs = v99.fit_v97(
            [records[index] for index in train_indices], args.device
        )
        prediction = v99.predict_pareto(
            model, [records[index] for index in held_rows],
            statistics, args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_rows]):
            raise RuntimeError("V109 OOF baseline identity changed")
        proposals[held_rows] = prediction["proposals"]
        aggregate_gain[held_rows] = prediction["aggregate_gain"]
        head_gain[held_rows] = prediction["head_gain"]
        row = {
            "held_fold": held,
            "train_row_count": len(train_indices),
            "held_row_count": len(held_rows),
            "train_scene_count": len(train_scenes),
            "held_scene_count": len(held_scenes),
            "normalization_sha256": statistics["sha256"],
            "final_epoch": epochs[-1],
        }
        model_folds.append(row)
        print(json.dumps({"completed_model_fold": row}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    candidate_ious = torch.stack([
        record["candidate_ious"].reshape(-1) for record in records
    ])
    rows = torch.arange(EXPECTED_SAMPLE_COUNT)
    baseline_ious = candidate_ious[rows, baselines]
    proposal_ious = candidate_ious[rows, proposals]
    selected = baselines.clone()
    meta_folds = []
    for held in range(5):
        held_rows = fold_indices[held]
        calibration_rows = torch.cat([
            fold_indices[fold] for fold in range(5) if fold != held
        ]).sort().values
        selection = select_nested_policy(
            calibration_rows, proposals, baselines, aggregate_gain, head_gain,
            baseline_ious, proposal_ious, fold_ids,
        )
        policy = selection["selected"]["policy"]
        accepted = policy_accept_mask(
            proposals[held_rows], baselines[held_rows],
            aggregate_gain[held_rows], head_gain[held_rows], policy,
        )
        selected[held_rows] = torch.where(
            accepted, proposals[held_rows], baselines[held_rows]
        )
        held_summary = policy_summary(
            held_rows, proposals, baselines, aggregate_gain, head_gain,
            baseline_ious, proposal_ious, fold_ids, policy,
        )
        meta_row = {
            "held_fold": held,
            "policy_calibration_folds": [
                fold for fold in range(5) if fold != held
            ],
            "policy_calibration_row_count": int(calibration_rows.numel()),
            "selected_policy": copy.deepcopy(policy),
            "calibration_selection": selection,
            "held_result": held_summary,
        }
        meta_folds.append(meta_row)
        print(json.dumps({"completed_meta_fold": meta_row}, sort_keys=True), flush=True)

    diagnostics = v99.build_diagnostics(records, selected, baselines)
    subgroups = {}
    for name, scenes in {
            "corrected": corrected_scenes,
            "regular": regular_scenes}.items():
        indices = [
            index for index, record in enumerate(records)
            if record["scan_id"] in scenes
        ]
        subgroups[name] = {
            "scene_count": len(scenes),
            "row_count": len(indices),
            "scene_sha256": hashlib.sha256(
                ("\n".join(sorted(scenes)) + "\n").encode("ascii")
            ).hexdigest(),
            "diagnostics": v99.build_diagnostics(
                [records[index] for index in indices],
                selected[indices], baselines[indices],
            ),
        }
    predicates = acceptance_gate(diagnostics, subgroups)
    global_policy_selection = select_nested_policy(
        torch.arange(EXPECTED_SAMPLE_COUNT), proposals, baselines,
        aggregate_gain, head_gain, baseline_ious, proposal_ious, fold_ids,
    )

    protected_after = capture_immutable_artifact_identities(protected_paths)
    metadata_after = {
        name: file_sha256(path) for name, path in metadata_paths.items()
    }
    if protected_after != protected_before or metadata_after != metadata_before:
        raise RuntimeError("V109 protected inputs changed during OOF")
    report = {
        "schema": "rec-v109-meshsp-nested-policy-full-train-scene-oof-v1",
        "version": 1,
        "validation_data_accessed": False,
        "prior_calibration_used_for_selection": False,
        "deployable": False,
        "source": {
            "path": str(source_path),
            "sha256": V108_REPORT_SHA256,
            "failed_gate": copy.deepcopy(source["oof"]),
        },
        "protocol": {
            "architecture": "V99 contextual hierarchy on MeshSP V108 features",
            "model_oof": "five_scene_disjoint_folds",
            "policy_oof": "leave_one_model_oof_fold_out_meta_calibration",
            "policy_grid": list(copy.deepcopy(POLICY_GRID)),
            "selection_objective": (
                "maximize_delta025_then_delta050_then_fewer_switches"
            ),
            "calibration_delta050_floor": "ceil_half_fixed_v108_policy_delta050",
            "held_labels_visible_to_policy_selection": False,
        },
        "model_folds": model_folds,
        "meta_folds": meta_folds,
        "global_policy_selection": global_policy_selection,
        "raw_prediction_sha256": v99.tensor_sha256(
            proposals, aggregate_gain, head_gain
        ),
        "oof": {
            "diagnostics": diagnostics,
            "subgroups": subgroups,
            "required_delta_hits025": MIN_DELTA_025,
            "required_delta_hits050": MIN_DELTA_050,
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "historical_split_metadata": copy.deepcopy(split["metadata"]),
        "source_sha256": {
            "v109": file_sha256(Path(__file__).resolve()),
            "v108": file_sha256(
                Path(__file__).resolve().parent
                / "run_v108_meshsp_pareto_oof.py"
            ),
            "v99": file_sha256(Path(v99.__file__).resolve()),
        },
        "fallback_scene_manifest": {
            "path": str(fallback_path),
            "sha256": FALLBACK_MANIFEST_SHA256,
            "scene_count": EXPECTED_FALLBACK_SCENE_COUNT,
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_metadata_before": metadata_before,
        "protected_metadata_after": metadata_after,
    }
    payload = json.dumps(
        report, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("V109 output write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "oof": report["oof"],
        "global_policy": global_policy_selection["selected"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
