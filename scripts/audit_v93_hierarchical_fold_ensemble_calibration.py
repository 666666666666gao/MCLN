#!/usr/bin/env python
"""Train-only V93 audit of a fixed five-fold hierarchical logit ensemble."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from models.rec_hierarchical_reranker import (
    QUERY_COUNT,
    VARIANT_COUNT,
    build_hierarchical_scene_folds,
    choose_hierarchical_configuration,
    select_hierarchical_proposal,
)
from scripts.audit_v92_sparse_hierarchical_calibration import (
    MIN_CALIBRATION_DELTA_025,
    MIN_CALIBRATION_DELTA_050,
    SOURCE_RECEIPT_SHA256,
    compact_policy,
    load_source_receipt,
    select_sparse_policy,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_BATCH_SIZE,
    _fit_hierarchical_model,
    _normalized_hierarchical_model_batch,
    build_hierarchical_cache_calibration_baseline,
    build_hierarchical_policy_candidate,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def state_dict_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_fold_ensemble(fit_records, choice, device):
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in fit_records
    ])
    members = []
    records = []
    for held_out_fold in range(5):
        train_records = [
            record for record in fit_records
            if scene_folds[record["scan_id"]] != held_out_fold
        ]
        held_records = [
            record for record in fit_records
            if scene_folds[record["scan_id"]] == held_out_fold
        ]
        if not train_records or not held_records:
            raise RuntimeError("every V93 fold needs train and held rows")
        statistics = fit_hierarchical_normalization(train_records)
        model = _fit_hierarchical_model(
            train_records,
            statistics,
            hidden_dim=choice["hidden_dim"],
            weight_decay=choice["weight_decay"],
            false_positive_cost=choice["false_positive_cost"],
            device=torch.device(device),
            observer_context={
                "phase": "v93_fold_ensemble",
                "held_out_fold": held_out_fold,
            },
        )
        member_record = {
            "fold": held_out_fold,
            "train_scene_count": len({
                record["scan_id"] for record in train_records
            }),
            "train_row_count": len(train_records),
            "held_scene_count": len({
                record["scan_id"] for record in held_records
            }),
            "held_row_count": len(held_records),
            "normalization_sha256": statistics["sha256"],
            "state_dict_sha256": state_dict_sha256(model),
        }
        print(json.dumps({"trained_member": member_record}, sort_keys=True), flush=True)
        members.append((model, statistics))
        records.append(member_record)
    if len({record["state_dict_sha256"] for record in records}) != 5:
        raise RuntimeError("V93 fold members unexpectedly share state")
    return members, records, scene_folds


def predict_fold_ensemble(members, records, device):
    proposals = []
    gains = []
    resolved = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            query_logits = None
            variant_logits = None
            reference_batch = None
            for model, statistics in members:
                model_batch = _normalized_hierarchical_model_batch(
                    row_batch, statistics, resolved
                )
                outputs = model.to(resolved).eval()(**model_batch)
                if query_logits is None:
                    query_logits = outputs["query_logits"].clone()
                    variant_logits = outputs["variant_logits"].clone()
                    reference_batch = model_batch
                else:
                    query_logits.add_(outputs["query_logits"])
                    variant_logits.add_(outputs["variant_logits"])
            query_logits.div_(float(len(members)))
            variant_logits.div_(float(len(members)))
            selected = select_hierarchical_proposal(
                query_logits,
                variant_logits,
                reference_batch["query_valid"],
                reference_batch["variant_valid"],
            )
            flat_utility = selected["variant_utility"].reshape(
                len(row_batch), QUERY_COUNT * VARIANT_COUNT
            )
            baseline_indices = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=resolved)
            rows = torch.arange(len(row_batch), device=resolved)
            selected_indices = selected["flat_indices"]
            gain = (
                flat_utility[rows, selected_indices]
                - flat_utility[rows, baseline_indices]
            )
            proposals.append(selected_indices.detach().cpu().long())
            gains.append(gain.detach().cpu().float())
    return torch.cat(proposals), torch.cat(gains)


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
    members, member_records, scene_folds = train_fold_ensemble(
        fit_records, choice, args.device
    )
    calibration_records = materialize_hierarchical_rows(
        split["calibration_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        device=args.device,
        require_contiguous=False,
    )
    fit_scenes = {record["scan_id"] for record in fit_records}
    calibration_scenes = {
        record["scan_id"] for record in calibration_records
    }
    if fit_scenes & calibration_scenes:
        raise RuntimeError("V93 fit and calibration scenes overlap")
    proposals, gain = predict_fold_ensemble(
        members, calibration_records, args.device
    )
    config = {
        "hidden_dim": choice["hidden_dim"],
        "weight_decay": choice["weight_decay"],
        "false_positive_cost": choice["false_positive_cost"],
    }
    candidate = build_hierarchical_policy_candidate(
        calibration_records,
        proposals,
        gain,
        config,
        choice["margin_percentile"],
        choice["margin"],
    )
    diagnostics = choose_hierarchical_configuration([candidate])[
        "candidate_diagnostics"
    ][0]
    baseline = build_hierarchical_cache_calibration_baseline(
        calibration_records
    )
    delta025 = diagnostics["delta_hits025"]
    delta050 = diagnostics["delta_hits050"]
    predicates = {
        "delta025_at_least_scaled_target": delta025 >= MIN_CALIBRATION_DELTA_025,
        "delta050_at_least_scaled_target": delta050 >= MIN_CALIBRATION_DELTA_050,
        "bootstrap025_lower_bound_nonnegative": (
            diagnostics["bootstrap025"]["lower_bound_95"] >= 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            diagnostics["bootstrap050"]["lower_bound_95"] >= 0
        ),
    }
    calibration = {
        "sample_count": diagnostics["sample_count"],
        "baseline_hits025": baseline["hits025"],
        "baseline_hits050": baseline["hits050"],
        "hits025": diagnostics["proposed"]["0.25"]["hits"],
        "hits050": diagnostics["proposed"]["0.50"]["hits"],
        "delta_hits025": delta025,
        "delta_hits050": delta050,
        "required_delta_hits025": MIN_CALIBRATION_DELTA_025,
        "required_delta_hits050": MIN_CALIBRATION_DELTA_050,
        "switches": diagnostics["switches"],
        "switch_rate": diagnostics["switch_rate"],
        "effects": copy.deepcopy(diagnostics["effects"]),
        "bootstrap025": copy.deepcopy(diagnostics["bootstrap025"]),
        "bootstrap050": copy.deepcopy(diagnostics["bootstrap050"]),
        "transition_diagnostics": copy.deepcopy(
            diagnostics["transition_diagnostics"]
        ),
        "predicates": predicates,
        "passed": all(predicates.values()),
    }
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V93 audit")
    report = {
        "schema": "rec-hierarchical-fold-ensemble-calibration-audit-v1",
        "version": 1,
        "validation_data_accessed": False,
        "deployable": False,
        "source_receipt": {
            "path": str(Path(args.source_receipt).expanduser().absolute()),
            "sha256": SOURCE_RECEIPT_SHA256,
        },
        "ensemble_rule": {
            "member_count": 5,
            "member_weighting": "uniform-arithmetic-mean-logits",
            "selection_rule": "fixed-v92-sparse-policy",
            "eligible_sparse_policy_count": eligible_count,
            "scene_fold_mapping_sha256": hashlib.sha256(
                json.dumps(
                    scene_folds,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        },
        "choice": compact_policy(choice),
        "members": member_records,
        "calibration": calibration,
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
    descriptor = os.open(
        str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
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
        "calibration": calibration,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
