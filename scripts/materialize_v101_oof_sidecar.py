#!/usr/bin/env python
"""Reproduce and freeze row-level V101 OOF decisions for downstream Mask OOF."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import torch

import scripts.run_v101_full_train_pareto_oof as v101
from models.rec_hierarchical_reranker import build_hierarchical_scene_folds
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    capture_immutable_artifact_identities,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
)


V101_RESULT_SHA256 = (
    "2cb453b130306449901bed9c337984aad7f8b7048d05bd4240b5077f0de9ac1e"
)
EXPECTED_PREDICTION_SHA256 = (
    "b81664e65d64dad7058f8f252d990d4ab11dd8c00746c64a918bb120b6434c99"
)
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exclusive_torch_save(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("V101 sidecar write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--v101-result", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    result_path = Path(args.v101_result).expanduser().absolute()
    if _sha256(result_path) != V101_RESULT_SHA256:
        raise ValueError("V101 result SHA-256 changed")
    result = json.loads(result_path.read_text(encoding="ascii"))
    if result.get("prediction_sha256") != EXPECTED_PREDICTION_SHA256:
        raise ValueError("V101 result prediction digest changed")

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
        raise ValueError("V101 sidecar input provenance changed")
    records = materialize_hierarchical_rows(
        loaded["joined_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
    )
    if (len(records) != EXPECTED_ROW_COUNT
            or len(set(record["scan_id"] for record in records))
            != EXPECTED_SCENE_COUNT):
        raise ValueError("V101 sidecar train coverage changed")
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    fold_indices = {
        fold: [i for i, record in enumerate(records)
               if scene_folds[record["scan_id"]] == fold]
        for fold in range(5)
    }
    baselines = torch.tensor(
        [record["baseline_index"] for record in records], dtype=torch.long
    )
    proposals = baselines.clone()
    selected = baselines.clone()
    aggregate_gain = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.float32)
    head_gain = torch.zeros(EXPECTED_ROW_COUNT, 2, dtype=torch.float32)
    accepted = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.bool)
    fold_ids = torch.full((EXPECTED_ROW_COUNT,), -1, dtype=torch.int64)
    folds = []
    for held in range(5):
        train_indices = sorted([
            index for fold in range(5) if fold != held
            for index in fold_indices[fold]
        ])
        held_indices = fold_indices[held]
        model, statistics, epochs = v101.v99.fit_v97(
            [records[index] for index in train_indices], args.device
        )
        prediction = v101.v99.predict_pareto(
            model, [records[index] for index in held_indices],
            statistics, args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_indices]):
            raise RuntimeError("V101 sidecar baseline identity changed")
        proposals[held_indices] = prediction["proposals"]
        selected[held_indices] = prediction["selected"]
        aggregate_gain[held_indices] = prediction["aggregate_gain"]
        head_gain[held_indices] = prediction["head_gain"]
        accepted[held_indices] = prediction["accepted"]
        fold_ids[held_indices] = held
        folds.append({
            "held_fold": held,
            "train_row_count": len(train_indices),
            "held_row_count": len(held_indices),
            "normalization_sha256": statistics["sha256"],
            "final_epoch": epochs[-1],
        })
        print(json.dumps({"completed_fold": folds[-1]}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()
    if bool(fold_ids.lt(0).any().item()):
        raise RuntimeError("V101 sidecar missed OOF rows")
    prediction_sha = v101.v99.tensor_sha256(
        proposals, selected, aggregate_gain, head_gain, accepted
    )
    if prediction_sha != EXPECTED_PREDICTION_SHA256:
        raise RuntimeError("V101 OOF reproduction digest mismatch")
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V101 sidecar")
    sidecar = {
        "schema": "rec-v101-oof-row-decisions-v1",
        "version": 1,
        "validation_data_accessed": False,
        "row_count": EXPECTED_ROW_COUNT,
        "scene_count": EXPECTED_SCENE_COUNT,
        "v101_result_sha256": V101_RESULT_SHA256,
        "prediction_sha256": prediction_sha,
        "input_sha256": loaded["input_sha256"],
        "dataset_indices": torch.tensor(
            [record["dataset_index"] for record in records], dtype=torch.long
        ),
        "fold_ids": fold_ids,
        "baseline_indices": baselines,
        "proposal_indices": proposals,
        "selected_indices": selected,
        "selected_parent_positions": torch.div(
            selected, 7, rounding_mode="floor"
        ),
        "accepted": accepted,
        "aggregate_gain": aggregate_gain,
        "head_gain": head_gain,
        "folds": folds,
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sha = _exclusive_torch_save(output, sidecar)
    print(json.dumps({
        "output": str(output), "sha256": output_sha,
        "prediction_sha256": prediction_sha,
        "accepted_switches": int(accepted.sum().item()),
        "mode": output.stat().st_mode & 0o777,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
