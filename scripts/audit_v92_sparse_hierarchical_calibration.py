#!/usr/bin/env python
"""Train-only calibration audit for the pre-registered sparse V92 policy."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    build_hierarchical_cache_calibration_baseline,
    capture_immutable_artifact_identities,
    evaluate_hierarchical_cache_policy,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    refit_hierarchical_reranker,
    split_residual_joined_rows,
    validate_hierarchical_result_receipt,
)


SOURCE_RECEIPT_SHA256 = (
    "5cc9c5ba94618bccd598b3d05f33d0dd1954602b9a2f54e246d0c19cef25ff6b"
)
MIN_OOF_DELTA_025 = 68
MIN_OOF_DELTA_050 = 38
MIN_CALIBRATION_DELTA_025 = 8
MIN_CALIBRATION_DELTA_050 = 5


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def load_source_receipt(path):
    resolved = Path(path).expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("source receipt must be a regular non-symlink file")
    payload = resolved.read_bytes()
    if sha256_bytes(payload) != SOURCE_RECEIPT_SHA256:
        raise ValueError("source receipt SHA-256 mismatch")
    receipt = json.loads(payload.decode("ascii"))
    validate_hierarchical_result_receipt(receipt)
    if receipt["validation_data_accessed"] is not False:
        raise ValueError("source receipt must be train-only")
    return receipt


def select_sparse_policy(receipt):
    candidates = receipt["oof"]["choice"]["candidate_diagnostics"]
    eligible = []
    for record in candidates:
        folds = record["fold_deltas"].values()
        if (
            record["no_switch"] is False
            and all(fold["hits025"] > 0 for fold in folds)
            and all(fold["hits050"] > 0 for fold in record["fold_deltas"].values())
            and record["bootstrap025"]["lower_bound_95"] > 0
            and record["bootstrap050"]["lower_bound_95"] > 0
            and record["delta_hits025"] >= MIN_OOF_DELTA_025
            and record["delta_hits050"] >= MIN_OOF_DELTA_050
        ):
            eligible.append(record)
    if not eligible:
        raise RuntimeError("no V91 OOF policy satisfies the V92 sparse gate")
    winner = min(
        eligible,
        key=lambda record: (
            record["switches"],
            -(2 * record["delta_hits025"] + record["delta_hits050"]),
            -record["delta_hits025"],
            -record["delta_hits050"],
            -record["margin"],
            record["hidden_dim"],
            -record["weight_decay"],
            -record["false_positive_cost"],
        ),
    )
    identity = (
        winner["hidden_dim"],
        winner["weight_decay"],
        winner["false_positive_cost"],
        winner["margin_percentile"],
    )
    matches = [
        record for record in candidates
        if (
            record["hidden_dim"],
            record["weight_decay"],
            record["false_positive_cost"],
            record["margin_percentile"],
        ) == identity
    ]
    if len(matches) != 1:
        raise RuntimeError("sparse policy identity is not unique")
    return copy.deepcopy(winner), len(eligible)


def compact_policy(record):
    fields = (
        "hidden_dim",
        "weight_decay",
        "false_positive_cost",
        "margin_percentile",
        "margin",
        "sample_count",
        "switches",
        "switch_rate",
        "delta_hits025",
        "delta_hits050",
        "fold_deltas",
        "effects",
        "bootstrap025",
        "bootstrap050",
        "transition_diagnostics",
    )
    return {name: copy.deepcopy(record[name]) for name in fields}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-receipt", required=True)
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
    receipt = load_source_receipt(args.source_receipt)
    choice, eligible_count = select_sparse_policy(receipt)
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
    model, normalization = refit_hierarchical_reranker(
        fit_records, choice, device=args.device
    )
    calibration_records = materialize_hierarchical_rows(
        split["calibration_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        device=args.device,
        require_contiguous=False,
    )
    baseline = build_hierarchical_cache_calibration_baseline(
        calibration_records
    )
    metrics = evaluate_hierarchical_cache_policy(
        model,
        calibration_records,
        normalization,
        margin=float(choice["margin"]),
        device=args.device,
    )
    delta025 = metrics["hits025"] - baseline["hits025"]
    delta050 = metrics["hits050"] - baseline["hits050"]
    predicates = {
        "delta025_at_least_scaled_target": delta025 >= MIN_CALIBRATION_DELTA_025,
        "delta050_at_least_scaled_target": delta050 >= MIN_CALIBRATION_DELTA_050,
        "bootstrap025_lower_bound_nonnegative": (
            metrics["bootstrap025"]["lower_bound_95"] >= 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            metrics["bootstrap050"]["lower_bound_95"] >= 0
        ),
    }
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V92 audit")
    report = {
        "schema": "rec-hierarchical-sparse-calibration-audit-v1",
        "version": 1,
        "validation_data_accessed": False,
        "deployable": False,
        "source_receipt": {
            "path": str(Path(args.source_receipt).expanduser().absolute()),
            "sha256": SOURCE_RECEIPT_SHA256,
        },
        "selection_rule": {
            "min_oof_delta025": MIN_OOF_DELTA_025,
            "min_oof_delta050": MIN_OOF_DELTA_050,
            "strict_positive_per_fold_both_thresholds": True,
            "strict_positive_bootstrap_lower_bound_both_thresholds": True,
            "primary_objective": "minimum_switches",
            "eligible_count": eligible_count,
        },
        "choice": compact_policy(choice),
        "calibration": {
            "baseline_hits025": baseline["hits025"],
            "baseline_hits050": baseline["hits050"],
            "hits025": metrics["hits025"],
            "hits050": metrics["hits050"],
            "delta_hits025": delta025,
            "delta_hits050": delta050,
            "required_delta_hits025": MIN_CALIBRATION_DELTA_025,
            "required_delta_hits050": MIN_CALIBRATION_DELTA_050,
            "switches": metrics["switches"],
            "switch_rate": metrics["switch_rate"],
            "fixes025": metrics["fixes025"],
            "breaks025": metrics["breaks025"],
            "fixes050": metrics["fixes050"],
            "breaks050": metrics["breaks050"],
            "bootstrap025": metrics["bootstrap025"],
            "bootstrap050": metrics["bootstrap050"],
            "transition_diagnostics": {
                name: metrics[name]
                for name in (
                    "selected_query_changes",
                    "same_query_variant_changes",
                    "wrong_query_recoveries025",
                    "wrong_query_recoveries050",
                    "wrong_variant_recoveries025",
                    "wrong_variant_recoveries050",
                )
            },
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "normalization_sha256": normalization["sha256"],
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
        "sha256": sha256_bytes(payload),
        "choice": compact_policy(choice),
        "calibration": report["calibration"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
