#!/usr/bin/env python
"""Scene-disjoint OOF audit of frozen V99 on all ScanRefer train scenes."""

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
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


V99_REPORT_SHA256 = (
    "db42ef5853fb36fba9bdc53afb719bff9eb5a3f9e772475a4c76c363db01572d"
)
EXPECTED_SAMPLE_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
MIN_DELTA_025 = 72
MIN_DELTA_050 = 74


def acceptance_gate(diagnostics):
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
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v99-report", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
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
    split = split_residual_joined_rows(loaded["joined_rows"])
    joined_rows = loaded["joined_rows"]
    scan_ids = [row["base"]["scan_id"] for row in joined_rows]
    if len(joined_rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("V101 full-train sample count changed")
    if len(set(scan_ids)) != EXPECTED_SCENE_COUNT:
        raise ValueError("V101 full-train scene count changed")
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
            raise RuntimeError("V101 OOF baseline identity changed")
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
    predicates = acceptance_gate(diagnostics)
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
        raise RuntimeError("protected artifacts changed during V101")
    script_sources = {
        "v101": str(Path(__file__).resolve()),
        "v99": str(Path(v99.__file__).resolve()),
    }
    report = {
        "schema": "rec-v99-pareto-full-train-scene-oof-v1",
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
            "architecture": "frozen V99 contextual query-set hierarchy",
            "objective": "frozen V95 bounded threshold-aligned listwise",
            "proposal_margin": v99.V97_MARGIN,
            "acceptance": "aggregate_margin_and_positive_delta025_and_delta050",
            "grid_search": False,
            "selection_data": "all_train_rows_scene_disjoint_5_fold_oof_only",
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "scene_count": EXPECTED_SCENE_COUNT,
        },
        "folds": folds,
        "veto_diagnostics": veto_reasons,
        "prediction_sha256": v99.tensor_sha256(
            proposals, selected, aggregate_gain, head_gain, accepted
        ),
        "oof": {
            "diagnostics": diagnostics,
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
        "protected_before": protected_before,
        "protected_after": protected_after,
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
                raise OSError("V101 output write made no progress")
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
