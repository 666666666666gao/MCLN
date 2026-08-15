#!/usr/bin/env python
"""Train-only replay audit for the frozen V101 full-train artifact."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import torch

from models.rec_hierarchical_reranker import build_hierarchical_scene_folds
from scripts.build_v101_full_train_pareto_artifact import load_v101_artifact
from scripts.run_v99_pareto_contextual_hierarchical import (
    build_diagnostics,
    predict_pareto,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    canonical_hierarchical_candidate_iou_sha256,
    canonical_hierarchical_deployable_sha256,
    canonical_hierarchical_scene_fold_sha256,
    capture_immutable_artifact_identities,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
)


V101_ARTIFACT_SHA256 = (
    "2c969a6c28a0c9315b53f0f847567345e47da8c912091344b23612680643a2ae"
)
V101_OOF_SHA256 = (
    "2cb453b130306449901bed9c337984aad7f8b7048d05bd4240b5077f0de9ac1e"
)
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_identity(path):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("V101 protected artifact must be a regular file")
    mode = stat.S_IMODE(entry.st_mode)
    if mode != 0o444:
        raise ValueError("V101 protected artifact must have mode 0444")
    return {
        "path": str(path.resolve(strict=True)),
        "device": int(entry.st_dev),
        "inode": int(entry.st_ino),
        "mode": mode,
        "size": int(entry.st_size),
        "mtime_ns": int(entry.st_mtime_ns),
        "ctime_ns": int(entry.st_ctime_ns),
        "sha256": _sha256(path),
    }


def _write_readonly_json(path, value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("V101 replay output write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _fold_diagnostics(records, selected, baselines):
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    result = {}
    for fold in range(5):
        indices = [
            index for index, record in enumerate(records)
            if scene_folds[record["scan_id"]] == fold
        ]
        subset = [records[index] for index in indices]
        result[str(fold)] = build_diagnostics(
            subset, selected[indices], baselines[indices]
        )
    return scene_folds, result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--v101-artifact", required=True)
    parser.add_argument("--v101-oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    if _sha256(args.v101_artifact) != V101_ARTIFACT_SHA256:
        raise ValueError("V101 artifact SHA changed")
    if _sha256(args.v101_oof) != V101_OOF_SHA256:
        raise ValueError("V101 OOF SHA changed")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    protected_before["v101"] = _readonly_identity(args.v101_artifact)
    loaded = load_residual_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    records = materialize_hierarchical_rows(
        loaded["joined_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
    )
    model, artifact = load_v101_artifact(
        args.v101_artifact, device=args.device,
        expected_artifact_sha256=V101_ARTIFACT_SHA256,
        parent_sha256=loaded["input_sha256"]["parent"],
        geometry_sha256=loaded["input_sha256"]["geometry"],
        expected_geometry_feature_names=loaded["geometry_artifact"][
            "feature_names"
        ],
    )
    if artifact["input_sha256"] != loaded["input_sha256"]:
        raise ValueError("V101 artifact input binding differs from live caches")
    if (len(records) != EXPECTED_ROW_COUNT
            or len(records) != artifact["fit"]["row_count"]
            or len(set(record["scan_id"] for record in records))
            != EXPECTED_SCENE_COUNT):
        raise ValueError("V101 replay train coverage changed")
    row_sha = canonical_hierarchical_deployable_sha256(records)
    iou_sha = canonical_hierarchical_candidate_iou_sha256(records)
    if (row_sha != artifact["fit"]["deployable_rows_sha256"]
            or iou_sha != artifact["fit"]["candidate_iou_sha256"]):
        raise ValueError("V101 replay rows differ from artifact fit evidence")

    prediction = predict_pareto(
        model, records, artifact["normalization"], args.device
    )
    diagnostics = build_diagnostics(
        records, prediction["selected"], prediction["baselines"]
    )
    scene_folds, folds = _fold_diagnostics(
        records, prediction["selected"], prediction["baselines"]
    )
    fold_sha = canonical_hierarchical_scene_fold_sha256(scene_folds)
    if fold_sha != artifact["fit"]["scene_fold_sha256"]:
        raise ValueError("V101 replay scene fold identity changed")
    protected_after = capture_immutable_artifact_identities(protected_paths)
    protected_after["v101"] = _readonly_identity(args.v101_artifact)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V101 replay")

    oof = json.loads(Path(args.v101_oof).read_text(encoding="utf-8"))
    report = {
        "schema": "rec-v101-fullfit-train-replay-audit-v1",
        "version": 1,
        "validation_data_accessed": False,
        "weights_modified": False,
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "artifact_sha256": V101_ARTIFACT_SHA256,
        "oof_sha256": V101_OOF_SHA256,
        "row_count": len(records),
        "scene_count": len(scene_folds),
        "deployable_rows_sha256": row_sha,
        "candidate_iou_sha256": iou_sha,
        "scene_fold_sha256": fold_sha,
        "fullfit_replay": {
            "diagnostics": diagnostics,
            "fold_diagnostics": folds,
            "accepted_switches": int(prediction["accepted"].sum().item()),
            "proposal_switches": int(
                prediction["proposals"].ne(
                    prediction["baselines"]
                ).sum().item()
            ),
            "prediction_sha256": hashlib.sha256(b"".join(
                tensor.detach().cpu().contiguous().numpy().tobytes()
                for tensor in (
                    prediction["proposals"], prediction["selected"],
                    prediction["aggregate_gain"], prediction["head_gain"],
                    prediction["accepted"],
                )
            )).hexdigest(),
        },
        "oof_reference": copy.deepcopy(oof["oof"]["diagnostics"]),
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    output_sha = _write_readonly_json(output, report)
    print(json.dumps({
        "output": str(output), "sha256": output_sha,
        "accepted_switches": report["fullfit_replay"]["accepted_switches"],
        "delta_hits025": diagnostics["delta_hits025"],
        "delta_hits050": diagnostics["delta_hits050"],
        "fold_deltas": diagnostics["fold_deltas"],
    }, sort_keys=True), flush=True)
    if diagnostics["delta_hits025"] <= 0 or diagnostics["delta_hits050"] <= 0:
        raise RuntimeError("V101 fullfit replay is not positive at both thresholds")


if __name__ == "__main__":
    main()
