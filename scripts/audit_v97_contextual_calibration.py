#!/usr/bin/env python
"""One-shot train-only calibration of the frozen V97 OOF candidate."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import scripts.run_v95_threshold_aligned_listwise_hierarchical as base
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    build_hierarchical_cache_calibration_baseline,
    capture_immutable_artifact_identities,
    evaluate_hierarchical_cache_policy,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


SOURCE_SHA256 = "ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d"
EXPECTED_MARGIN = 0.13312220573425293
MIN_DELTA_025 = 8
MIN_DELTA_050 = 7


def load_source(path):
    resolved = Path(path).expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("V97 source must be a regular non-symlink file")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("V97 source SHA-256 mismatch")
    report = json.loads(payload.decode("ascii"))
    diagnostics = report["oof"]["diagnostics"]
    if (
        report.get("schema")
        != "rec-contextual-listwise-hierarchical-train-only-v1"
        or report.get("validation_data_accessed") is not False
        or report.get("deployable") is not False
        or report["calibration"].get("status") != "not_run"
        or report["oof"].get("margin") != EXPECTED_MARGIN
        or diagnostics.get("delta_hits025") != 174
        or diagnostics.get("delta_hits050") != 475
        or diagnostics.get("hidden_dim") != 128
        or diagnostics.get("weight_decay") != 0.001
        or diagnostics.get("margin_percentile") != 50.0
    ):
        raise ValueError("V97 frozen source contract changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
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
    source = load_source(args.source)
    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    loaded = load_residual_training_inputs(
        Path(args.base_cache),
        Path(args.geometry_cache),
        Path(args.parent_artifact),
        Path(args.geometry_artifact),
        device=args.device,
    )
    split = split_residual_joined_rows(loaded["joined_rows"])
    fit_records = materialize_hierarchical_rows(
        split["fit_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        device=args.device,
        require_contiguous=False,
    )
    statistics = fit_hierarchical_normalization(fit_records)
    original_model = base.HierarchicalQueryVariantReranker
    base.HierarchicalQueryVariantReranker = ContextualHierarchicalQueryVariantReranker
    try:
        model, epochs = base.fit_graded_listwise_model(
            fit_records, statistics, args.device
        )
    finally:
        base.HierarchicalQueryVariantReranker = original_model
    calibration_records = materialize_hierarchical_rows(
        split["calibration_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        device=args.device,
        require_contiguous=False,
    )
    fit_scenes = {record["scan_id"] for record in fit_records}
    calibration_scenes = {record["scan_id"] for record in calibration_records}
    if fit_scenes & calibration_scenes:
        raise RuntimeError("V97 fit and calibration scenes overlap")
    baseline = build_hierarchical_cache_calibration_baseline(calibration_records)
    metrics = evaluate_hierarchical_cache_policy(
        model,
        calibration_records,
        statistics,
        margin=EXPECTED_MARGIN,
        device=args.device,
    )
    delta025 = metrics["hits025"] - baseline["hits025"]
    delta050 = metrics["hits050"] - baseline["hits050"]
    predicates = {
        "delta025_at_least_oracle_scaled_gap": delta025 >= MIN_DELTA_025,
        "delta050_at_least_oracle_scaled_gap": delta050 >= MIN_DELTA_050,
        "bootstrap025_lower_bound_nonnegative": (
            metrics["bootstrap025"]["lower_bound_95"] >= 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            metrics["bootstrap050"]["lower_bound_95"] >= 0
        ),
    }
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V97 calibration")
    report = {
        "schema": "rec-contextual-listwise-calibration-audit-v1",
        "version": 1,
        "validation_data_accessed": False,
        "deployable": False,
        "source": {
            "path": str(Path(args.source).expanduser().absolute()),
            "sha256": SOURCE_SHA256,
            "oof": copy.deepcopy(source["oof"]),
        },
        "gate_contract": {
            "transfer_scale": "oracle-repair-headroom",
            "minimum_delta025": MIN_DELTA_025,
            "minimum_delta050": MIN_DELTA_050,
            "bootstrap_lower_bounds_nonnegative": True,
        },
        "training_contract": copy.deepcopy(source["training_contract"]),
        "normalization_sha256": statistics["sha256"],
        "epochs": epochs,
        "baseline": baseline,
        "metrics": metrics,
        "delta_hits025": delta025,
        "delta_hits050": delta050,
        "predicates": predicates,
        "passed": all(predicates.values()),
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "split": copy.deepcopy(split["metadata"]),
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    payload = json.dumps(
        report,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "delta_hits025": delta025,
        "delta_hits050": delta050,
        "bootstrap025": metrics["bootstrap025"],
        "bootstrap050": metrics["bootstrap050"],
        "effects025": {"fixes": metrics["fixes025"], "breaks": metrics["breaks025"]},
        "effects050": {"fixes": metrics["fixes050"], "breaks": metrics["breaks050"]},
        "switches": metrics["switches"],
        "passed": report["passed"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
