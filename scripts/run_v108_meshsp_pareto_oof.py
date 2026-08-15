#!/usr/bin/env python
"""Scene-disjoint OOF audit after MeshSP-aligned ScanRefer train refitting."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

import scripts.run_v99_pareto_contextual_hierarchical as v99
from models.rec_hierarchical_reranker import (
    build_hierarchical_scene_folds,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    capture_immutable_artifact_identities,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)
from scripts.train_rec_geometry_reranker import (
    load_geometry_reranker_artifact,
    load_geometry_training_data,
)


V99_REPORT_SHA256 = (
    "db42ef5853fb36fba9bdc53afb719bff9eb5a3f9e772475a4c76c363db01572d"
)
EXPECTED_SAMPLE_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
MIN_DELTA_025 = 72
MIN_DELTA_050 = 74
FALLBACK_MANIFEST_SHA256 = (
    "caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079"
)
EXPECTED_FALLBACK_SCENE_COUNT = 789
EXPECTED_CORRECTED_SCENE_COUNT = 361
EXPECTED_REGULAR_SCENE_COUNT = 201
EXPECTED_CANDIDATE_RECEIPT_SHA256 = (
    "bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa"
)
EXPECTED_GEOMETRY_RECEIPT_SHA256 = (
    "e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443"
)
EXPECTED_PARENT_ARTIFACT_SHA256 = (
    "7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f"
)
EXPECTED_GEOMETRY_ARTIFACT_SHA256 = (
    "20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972"
)
EXPECTED_BACKBONE_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
EXPECTED_BASE_CACHE_CONTENT_SHA256 = (
    "0ff6855d9e144d07447aa6d6b1fd26b7dd6c4a52168f2b2a8d6729940c60f637"
)
EXPECTED_GEOMETRY_CACHE_CONTENT_SHA256 = (
    "7bd0634bb7a6faeece7399e81dc98987e562dc8eea2ee701de8e9535f9bbc91f"
)
EXPECTED_GEOMETRY_METADATA_SHA256 = (
    "d596a574342af7966fcd33380832aabd1c2f369a353f233fcc1c63bbd7196798"
)


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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_bound_receipt(path, expected_sha256, expected_schema):
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("V108 receipt must be a regular non-symlink file")
    if file_sha256(resolved) != expected_sha256:
        raise ValueError("V108 receipt SHA-256 changed: {}".format(resolved))
    receipt = json.loads(resolved.read_text(encoding="ascii"))
    if (receipt.get("schema") != expected_schema
            or receipt.get("validation_data_accessed") is not False
            or receipt.get("sample_count") != EXPECTED_SAMPLE_COUNT
            or receipt.get("scene_count") != EXPECTED_SCENE_COUNT):
        raise ValueError("V108 receipt policy changed: {}".format(resolved))
    return receipt


def load_v108_meshsp_training_inputs(
        base_cache, geometry_cache, parent_artifact_path,
        geometry_artifact_path, device="cuda:0"):
    """Load only the frozen MeshSP V108 caches and fitted sidecars."""
    base_path = Path(base_cache).expanduser().resolve(strict=True)
    geometry_path = Path(geometry_cache).expanduser().resolve(strict=True)
    parent_path = Path(parent_artifact_path).expanduser().resolve(strict=True)
    geometry_artifact_path = Path(
        geometry_artifact_path
    ).expanduser().resolve(strict=True)
    if (base_path.name != "train"
            or geometry_path.name != "geometry_train"
            or base_path.parent != geometry_path.parent
            or parent_path.parent != base_path.parent / "v108_artifacts"
            or geometry_artifact_path.parent != parent_path.parent):
        raise ValueError("V108 inputs do not have the frozen train-only layout")

    candidate_receipt_path = base_path.parent / "candidate_train_receipt.json"
    geometry_receipt_path = base_path.parent / "geometry_train_receipt.json"
    candidate_receipt = _load_bound_receipt(
        candidate_receipt_path,
        EXPECTED_CANDIDATE_RECEIPT_SHA256,
        "mcln-v108-meshsp-train-candidate-cache-receipt-v1",
    )
    geometry_receipt = _load_bound_receipt(
        geometry_receipt_path,
        EXPECTED_GEOMETRY_RECEIPT_SHA256,
        "mcln-v108-meshsp-train-geometry-cache-receipt-v1",
    )
    parent_sha256 = file_sha256(parent_path)
    geometry_sha256 = file_sha256(geometry_artifact_path)
    if parent_sha256 != EXPECTED_PARENT_ARTIFACT_SHA256:
        raise ValueError("V108 parent artifact SHA-256 changed")
    if geometry_sha256 != EXPECTED_GEOMETRY_ARTIFACT_SHA256:
        raise ValueError("V108 geometry artifact SHA-256 changed")

    joined, base_manifest, geometry_manifest, parent = (
        load_geometry_training_data(
            base_path, geometry_path, parent_path
        )
    )
    geometry_model, geometry_artifact = load_geometry_reranker_artifact(
        geometry_artifact_path,
        device=device,
        parent_artifact_path=parent_path,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
    )
    base_binding = geometry_manifest.get("base_cache_binding", {})
    expected = {
        "backbone": base_manifest.get("checkpoint_sha256"),
        "base_cache_content": base_binding.get("content_sha256"),
        "geometry_cache_content": geometry_manifest.get(
            "cache_content_digest"
        ),
        "geometry_metadata": geometry_manifest.get(
            "immutable_metadata_digest"
        ),
    }
    if (base_manifest.get("split") != "train"
            or geometry_manifest.get("split") != "train"
            or len(joined) != EXPECTED_SAMPLE_COUNT
            or base_manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT
            or geometry_manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT
            or candidate_receipt.get("checkpoint_sha256")
            != expected["backbone"]
            or candidate_receipt.get("manifest_sha256")
            != file_sha256(base_path / "manifest.json")
            or geometry_receipt.get("manifest_sha256")
            != file_sha256(geometry_path / "manifest.json")
            or geometry_receipt.get("base_cache_content_sha256")
            != expected["base_cache_content"]
            or geometry_receipt.get("cache_content_digest")
            != expected["geometry_cache_content"]
            or geometry_artifact.get("parent_artifact_sha256")
            != parent_sha256
            or geometry_artifact.get("train_base_cache_content_digest")
            != expected["base_cache_content"]
            or geometry_artifact.get("train_geometry_cache_content_digest")
            != expected["geometry_cache_content"]
            or geometry_artifact.get(
                "train_geometry_immutable_metadata_digest"
            ) != expected["geometry_metadata"]
            or float(geometry_artifact.get("geometry_weight", -1.0)) != 0.9
            or getattr(parent[0], "_artifact_sha256", None) != parent_sha256
            or getattr(geometry_model, "_artifact_sha256", None)
            != geometry_sha256):
        raise ValueError("V108 cache/artifact provenance binding changed")
    return {
        "joined_rows": joined,
        "base_manifest": base_manifest,
        "geometry_manifest": geometry_manifest,
        "parent": parent,
        "geometry_model": geometry_model,
        "geometry_artifact": geometry_artifact,
        "input_sha256": {
            **expected,
            "parent": parent_sha256,
            "geometry": geometry_sha256,
            "candidate_receipt": EXPECTED_CANDIDATE_RECEIPT_SHA256,
            "geometry_receipt": EXPECTED_GEOMETRY_RECEIPT_SHA256,
        },
        "validation_data_accessed": False,
    }


def validate_v108_materialization_artifact(artifact):
    """Accept only the already strict-loaded, frozen V108 geometry artifact."""
    expected = {
        "checkpoint_sha256": EXPECTED_BACKBONE_SHA256,
        "parent_artifact_sha256": EXPECTED_PARENT_ARTIFACT_SHA256,
        "train_base_cache_content_digest": (
            EXPECTED_BASE_CACHE_CONTENT_SHA256
        ),
        "train_geometry_cache_content_digest": (
            EXPECTED_GEOMETRY_CACHE_CONTENT_SHA256
        ),
        "train_geometry_immutable_metadata_digest": (
            EXPECTED_GEOMETRY_METADATA_SHA256
        ),
        "geometry_weight": 0.9,
        "regressed_variant_index": 0,
    }
    if (not isinstance(artifact, dict)
            or any(artifact.get(key) != value
                   for key, value in expected.items())):
        raise ValueError("V108 geometry materialization contract changed")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v99-report", required=True)
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
    source_path = Path(args.v99_report).expanduser().absolute()
    source_sha256 = file_sha256(source_path)
    if source_sha256 != V99_REPORT_SHA256:
        raise ValueError("V99 report SHA-256 changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source["oof"]["passed"]:
        raise ValueError("V99 source did not pass its train-only gate")
    fallback_path = Path(args.fallback_scenes).expanduser().absolute()
    if file_sha256(fallback_path) != FALLBACK_MANIFEST_SHA256:
        raise ValueError("MeshSP fallback-scene manifest SHA-256 changed")
    fallback_scenes = {
        line.strip() for line in fallback_path.read_text(
            encoding="ascii"
        ).splitlines() if line.strip()
    }
    if len(fallback_scenes) != EXPECTED_FALLBACK_SCENE_COUNT:
        raise ValueError("MeshSP fallback-scene manifest count changed")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_metadata_paths = {
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
    protected_metadata_before = {
        name: file_sha256(path)
        for name, path in protected_metadata_paths.items()
    }
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    split = split_residual_joined_rows(loaded["joined_rows"])
    joined_rows = loaded["joined_rows"]
    scan_ids = [row["base"]["scan_id"] for row in joined_rows]
    if len(joined_rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("V108 full-train sample count changed")
    if len(set(scan_ids)) != EXPECTED_SCENE_COUNT:
        raise ValueError("V108 full-train scene count changed")
    training_scenes = set(scan_ids)
    corrected_scenes = training_scenes & fallback_scenes
    regular_scenes = training_scenes - fallback_scenes
    if (len(corrected_scenes) != EXPECTED_CORRECTED_SCENE_COUNT
            or len(regular_scenes) != EXPECTED_REGULAR_SCENE_COUNT
            or corrected_scenes & regular_scenes
            or corrected_scenes | regular_scenes != training_scenes):
        raise ValueError("V108 corrected/regular train-scene partition changed")
    reconstructed_rows = split["fit_rows"] + split["calibration_rows"]
    if len(reconstructed_rows) != len(joined_rows):
        raise RuntimeError("fit and calibration rows do not reconstruct train rows")
    if set(scan_ids) != set(
            row["base"]["scan_id"] for row in reconstructed_rows):
        raise RuntimeError("split rows do not reconstruct all train scenes")
    records = materialize_hierarchical_rows(
        joined_rows, loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("materialization dropped full-train rows")
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    fold_indices = {
        fold: [
            index for index, record in enumerate(records)
            if scene_folds[record["scan_id"]] == fold
        ] for fold in range(5)
    }
    if sorted(index for rows in fold_indices.values() for index in rows) != list(
            range(EXPECTED_SAMPLE_COUNT)):
        raise RuntimeError("OOF folds do not partition all full-train rows")

    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    proposals = baselines.clone()
    selected = baselines.clone()
    aggregate_gain = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.float32)
    head_gain = torch.zeros(EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32)
    accepted = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.bool)
    folds = []
    for held in range(5):
        train_indices = sorted([
            index for fold in range(5) if fold != held
            for index in fold_indices[fold]
        ])
        held_indices = fold_indices[held]
        if set(records[index]["scan_id"] for index in train_indices) & set(
                records[index]["scan_id"] for index in held_indices):
            raise RuntimeError("scene leakage between train and held fold")
        model, statistics, epochs = v99.fit_v97(
            [records[index] for index in train_indices], args.device
        )
        prediction = v99.predict_pareto(
            model,
            [records[index] for index in held_indices],
            statistics,
            args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_indices]):
            raise RuntimeError("V108 OOF baseline identity changed")
        proposals[held_indices] = prediction["proposals"]
        selected[held_indices] = prediction["selected"]
        aggregate_gain[held_indices] = prediction["aggregate_gain"]
        head_gain[held_indices] = prediction["head_gain"]
        accepted[held_indices] = prediction["accepted"]
        fold_record = {
            "held_fold": held,
            "train_row_count": len(train_indices),
            "held_row_count": len(held_indices),
            "train_scene_count": len(set(
                records[index]["scan_id"] for index in train_indices
            )),
            "held_scene_count": len(set(
                records[index]["scan_id"] for index in held_indices
            )),
            "accepted_switches": int(prediction["accepted"].sum().item()),
            "normalization_sha256": statistics["sha256"],
            "final_epoch": epochs[-1],
        }
        folds.append(fold_record)
        print(json.dumps({"completed_fold": fold_record}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    diagnostics = v99.build_diagnostics(records, selected, baselines)
    subgroup_scene_sets = {
        "corrected": corrected_scenes,
        "regular": regular_scenes,
    }
    subgroups = {}
    for name, scenes in subgroup_scene_sets.items():
        indices = [
            index for index, record in enumerate(records)
            if record["scan_id"] in scenes
        ]
        subgroup_records = [records[index] for index in indices]
        subgroup_diagnostics = v99.build_diagnostics(
            subgroup_records, selected[indices], baselines[indices]
        )
        subgroups[name] = {
            "scene_count": len(scenes),
            "row_count": len(indices),
            "scene_sha256": hashlib.sha256(
                ("\n".join(sorted(scenes)) + "\n").encode("ascii")
            ).hexdigest(),
            "diagnostics": subgroup_diagnostics,
        }
    if sum(row["row_count"] for row in subgroups.values()) != len(records):
        raise RuntimeError("V108 subgroup rows do not partition train rows")
    predicates = acceptance_gate(diagnostics, subgroups)
    raw_switch = proposals.ne(baselines)
    original_margin_pass = raw_switch & aggregate_gain.ge(v99.V97_MARGIN)
    veto = original_margin_pass & ~accepted
    veto_reasons = {
        "original_margin_switches": int(original_margin_pass.sum().item()),
        "accepted_switches": int(accepted.sum().item()),
        "pareto_vetoes": int(veto.sum().item()),
        "nonpositive_delta025": int(
            (original_margin_pass & head_gain[:, 0].le(0.0)).sum().item()
        ),
        "nonpositive_delta050": int(
            (original_margin_pass & head_gain[:, 1].le(0.0)).sum().item()
        ),
    }
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V108")
    protected_metadata_after = {
        name: file_sha256(path)
        for name, path in protected_metadata_paths.items()
    }
    if protected_metadata_after != protected_metadata_before:
        raise RuntimeError("protected V108 cache metadata changed during OOF")
    script_sources = {
        "v108": str(Path(__file__).resolve()),
        "v99": str(Path(v99.__file__).resolve()),
    }
    report = {
        "schema": "rec-v108-meshsp-pareto-full-train-scene-oof-v1",
        "version": 1,
        "validation_data_accessed": False,
        "prior_calibration_used_for_selection": False,
        "deployable": False,
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "v99_oof": copy.deepcopy(source["oof"]),
        },
        "protocol": {
            "architecture": "V99 contextual query-set hierarchy refit on MeshSP-aligned features",
            "objective": "frozen V95 bounded threshold-aligned listwise",
            "proposal_margin": v99.V97_MARGIN,
            "acceptance": "aggregate_margin_and_positive_delta025_and_delta050",
            "grid_search": False,
            "selection_data": "all_train_rows_scene_disjoint_5_fold_oof_only",
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "scene_count": EXPECTED_SCENE_COUNT,
            "subgroup_policy": "corrected361_and_regular201_nonnegative_both_thresholds",
        },
        "folds": folds,
        "veto_diagnostics": veto_reasons,
        "prediction_sha256": v99.tensor_sha256(
            proposals, selected, aggregate_gain, head_gain, accepted
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
            name: file_sha256(path) for name, path in script_sources.items()
        },
        "fallback_scene_manifest": {
            "path": str(fallback_path),
            "sha256": FALLBACK_MANIFEST_SHA256,
            "scene_count": EXPECTED_FALLBACK_SCENE_COUNT,
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_metadata_before": protected_metadata_before,
        "protected_metadata_after": protected_metadata_after,
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
                raise OSError("V108 output write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "veto_diagnostics": veto_reasons,
        "oof": report["oof"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
