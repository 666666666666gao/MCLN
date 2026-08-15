#!/usr/bin/env python
"""Fixed train-only V95 threshold-aligned listwise experiment."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from models.rec_hierarchical_reranker import (
    HierarchicalQueryVariantReranker,
    build_hierarchical_scene_folds,
    choose_hierarchical_configuration,
    monotone_hit_probabilities,
)
from scripts.audit_v92_sparse_hierarchical_calibration import (
    SOURCE_RECEIPT_SHA256,
    load_source_receipt,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_BATCH_SIZE,
    HIERARCHICAL_DROPOUT,
    HIERARCHICAL_EPOCHS,
    HIERARCHICAL_GRAD_CLIP_NORM,
    HIERARCHICAL_LEARNING_RATE,
    _normalized_hierarchical_model_batch,
    _predict_hierarchical_proposals,
    _set_hierarchical_seed,
    build_hierarchical_cache_calibration_baseline,
    build_hierarchical_policy_candidate,
    capture_immutable_artifact_identities,
    evaluate_hierarchical_cache_policy,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    nearest_rank_hierarchical_margin,
    split_residual_joined_rows,
)


HIDDEN_DIM = 128
WEIGHT_DECAY = 1e-3
FALSE_POSITIVE_COST_METADATA = 4.0
MARGIN_PERCENTILE = 50.0
TARGET_TEMPERATURE = 0.25
MIN_OOF_DELTA_025 = 237
MIN_OOF_DELTA_050 = 133
MIN_CALIBRATION_DELTA_025 = 26
MIN_CALIBRATION_DELTA_050 = 15


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def graded_quality(candidate_ious):
    if candidate_ious.dtype != torch.float32 or candidate_ious.dim() != 3:
        raise ValueError("candidate_ious must be float32 [B,Q,V]")
    return (
        candidate_ious
        + 2.0 * candidate_ious.gt(0.25).to(candidate_ious.dtype)
        + candidate_ious.gt(0.50).to(candidate_ious.dtype)
    )


def masked_soft_listwise_cross_entropy(scores, target_quality, valid, dim):
    if scores.shape != target_quality.shape or scores.shape != valid.shape:
        raise ValueError("listwise score/target/valid shapes differ")
    if valid.dtype != torch.bool:
        raise TypeError("listwise valid mask must be boolean")
    if not bool(valid.any(dim=dim).all().item()):
        raise ValueError("every listwise group needs one valid item")
    masked_scores = scores.masked_fill(~valid, -float("inf"))
    masked_target = (target_quality / TARGET_TEMPERATURE).masked_fill(
        ~valid, -float("inf")
    )
    target_probability = torch.softmax(masked_target, dim=dim)
    log_probability = torch.log_softmax(masked_scores, dim=dim)
    safe_terms = torch.where(
        valid,
        target_probability * log_probability,
        torch.zeros_like(log_probability),
    )
    return -safe_terms.sum(dim=dim).mean()


def graded_listwise_loss(outputs, candidate_ious, query_valid, variant_valid):
    quality = graded_quality(candidate_ious)
    query_quality = quality.masked_fill(
        ~variant_valid, -float("inf")
    ).max(dim=2).values
    query_probability = monotone_hit_probabilities(outputs["query_logits"])
    variant_probability = monotone_hit_probabilities(
        outputs["variant_logits"]
    )
    query_utility = 2.0 * query_probability[..., 0] + query_probability[..., 1]
    variant_utility = (
        2.0 * variant_probability[..., 0] + variant_probability[..., 1]
    )
    query_loss = masked_soft_listwise_cross_entropy(
        query_utility, query_quality, query_valid, dim=1
    )

    batch_size, query_count, variant_count = quality.shape
    valid_query_rows = query_valid.reshape(-1)
    flat_variant_loss = masked_soft_listwise_cross_entropy(
        variant_utility.reshape(batch_size * query_count, variant_count)[
            valid_query_rows
        ],
        quality.reshape(batch_size * query_count, variant_count)[
            valid_query_rows
        ],
        variant_valid.reshape(batch_size * query_count, variant_count)[
            valid_query_rows
        ],
        dim=1,
    )
    return query_loss + flat_variant_loss, {
        "query_loss": float(query_loss.detach().item()),
        "variant_loss": float(flat_variant_loss.detach().item()),
    }


def fit_graded_listwise_model(records, statistics, device):
    resolved = torch.device(device)
    _set_hierarchical_seed(resolved)
    model = HierarchicalQueryVariantReranker(
        hidden_dim=HIDDEN_DIM, dropout=HIERARCHICAL_DROPOUT
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HIERARCHICAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    shuffle_state = random.Random(0)
    epoch_summaries = []
    for epoch in range(HIERARCHICAL_EPOCHS):
        model.train()
        order = list(range(len(records)))
        shuffle_state.shuffle(order)
        totals = {"loss": 0.0, "query_loss": 0.0, "variant_loss": 0.0}
        batches = 0
        for start in range(0, len(order), HIERARCHICAL_BATCH_SIZE):
            indices = order[start:start + HIERARCHICAL_BATCH_SIZE]
            row_batch = [records[index] for index in indices]
            model_batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            candidate_ious = torch.stack([
                record["candidate_ious"] for record in row_batch
            ]).to(resolved)
            outputs = model(**model_batch)
            loss, stats = graded_listwise_loss(
                outputs,
                candidate_ious,
                model_batch["query_valid"],
                model_batch["variant_valid"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), HIERARCHICAL_GRAD_CLIP_NORM)
            optimizer.step()
            totals["loss"] += float(loss.detach().item())
            totals["query_loss"] += stats["query_loss"]
            totals["variant_loss"] += stats["variant_loss"]
            batches += 1
        epoch_summaries.append({
            "epoch": epoch + 1,
            "loss": totals["loss"] / batches,
            "query_loss": totals["query_loss"] / batches,
            "variant_loss": totals["variant_loss"] / batches,
        })
    model.eval().requires_grad_(False)
    return model, epoch_summaries


def oof_gate(diagnostics):
    folds = list(diagnostics["fold_deltas"].values())
    predicates = {
        "all_folds_nonnegative025": all(row["hits025"] >= 0 for row in folds),
        "all_folds_nonnegative050": all(row["hits050"] >= 0 for row in folds),
        "delta025_at_least_scaled_official_gap": (
            diagnostics["delta_hits025"] >= MIN_OOF_DELTA_025
        ),
        "delta050_at_least_scaled_official_gap": (
            diagnostics["delta_hits050"] >= MIN_OOF_DELTA_050
        ),
        "bootstrap025_lower_bound_positive": (
            diagnostics["bootstrap025"]["lower_bound_95"] > 0
        ),
        "bootstrap050_lower_bound_positive": (
            diagnostics["bootstrap050"]["lower_bound_95"] > 0
        ),
    }
    return predicates


def calibration_gate(metrics, baseline):
    delta025 = metrics["hits025"] - baseline["hits025"]
    delta050 = metrics["hits050"] - baseline["hits050"]
    predicates = {
        "delta025_at_least_scaled_official_gap": (
            delta025 >= MIN_CALIBRATION_DELTA_025
        ),
        "delta050_at_least_scaled_official_gap": (
            delta050 >= MIN_CALIBRATION_DELTA_050
        ),
        "bootstrap025_lower_bound_nonnegative": (
            metrics["bootstrap025"]["lower_bound_95"] >= 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            metrics["bootstrap050"]["lower_bound_95"] >= 0
        ),
    }
    return predicates, delta025, delta050


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

    source_receipt = load_source_receipt(args.source_receipt)
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
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in fit_records
    ])
    proposals = torch.full((len(fit_records),), -1, dtype=torch.long)
    gain = torch.zeros(len(fit_records), dtype=torch.float32)
    fold_training = []
    for held_out_fold in range(5):
        train_indices = [
            index for index, record in enumerate(fit_records)
            if scene_folds[record["scan_id"]] != held_out_fold
        ]
        held_indices = [
            index for index, record in enumerate(fit_records)
            if scene_folds[record["scan_id"]] == held_out_fold
        ]
        train_records = [fit_records[index] for index in train_indices]
        held_records = [fit_records[index] for index in held_indices]
        statistics = fit_hierarchical_normalization(train_records)
        model, epochs = fit_graded_listwise_model(
            train_records, statistics, args.device
        )
        held_proposals, held_gain = _predict_hierarchical_proposals(
            model, held_records, statistics, torch.device(args.device)
        )
        proposals[held_indices] = held_proposals
        gain[held_indices] = held_gain
        fold_record = {
            "fold": held_out_fold,
            "train_row_count": len(train_records),
            "held_row_count": len(held_records),
            "normalization_sha256": statistics["sha256"],
            "epochs": epochs,
        }
        fold_training.append(fold_record)
        print(json.dumps({
            "completed_fold": held_out_fold,
            "train_row_count": len(train_records),
            "held_row_count": len(held_records),
            "final_epoch": epochs[-1],
        }, sort_keys=True), flush=True)
        del model
    if bool(proposals.lt(0).any().item()):
        raise RuntimeError("V95 OOF predictions are incomplete")
    margin = nearest_rank_hierarchical_margin(gain, MARGIN_PERCENTILE)
    if margin is None:
        raise RuntimeError("V95 has no positive OOF proposal gain")
    config = {
        "hidden_dim": HIDDEN_DIM,
        "weight_decay": WEIGHT_DECAY,
        "false_positive_cost": FALSE_POSITIVE_COST_METADATA,
    }
    candidate = build_hierarchical_policy_candidate(
        fit_records,
        proposals,
        gain,
        config,
        MARGIN_PERCENTILE,
        margin,
    )
    diagnostics = choose_hierarchical_configuration([candidate])[
        "candidate_diagnostics"
    ][0]
    predicates = oof_gate(diagnostics)
    calibration = {
        "status": "not_run",
        "reason": "oof_effect_gate_failed",
    }
    if all(predicates.values()):
        full_statistics = fit_hierarchical_normalization(fit_records)
        full_model, full_epochs = fit_graded_listwise_model(
            fit_records, full_statistics, args.device
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
            full_model,
            calibration_records,
            full_statistics,
            margin=float(margin),
            device=args.device,
        )
        cal_predicates, delta025, delta050 = calibration_gate(metrics, baseline)
        calibration = {
            "status": "run",
            "baseline": baseline,
            "metrics": metrics,
            "delta_hits025": delta025,
            "delta_hits050": delta050,
            "predicates": cal_predicates,
            "passed": all(cal_predicates.values()),
            "normalization_sha256": full_statistics["sha256"],
            "epochs": full_epochs,
        }

    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V95")
    report = {
        "schema": "rec-threshold-aligned-listwise-hierarchical-train-only-v1",
        "version": 1,
        "validation_data_accessed": False,
        "deployable": False,
        "source_receipt": {
            "path": str(Path(args.source_receipt).expanduser().absolute()),
            "sha256": SOURCE_RECEIPT_SHA256,
            "validated": source_receipt["validation_data_accessed"] is False,
        },
        "training_contract": {
            "objective": "two-level-graded-soft-listwise-cross-entropy",
            "quality": "iou+2*hit025+hit050",
            "target_temperature": TARGET_TEMPERATURE,
            "hidden_dim": HIDDEN_DIM,
            "weight_decay": WEIGHT_DECAY,
            "epochs": HIERARCHICAL_EPOCHS,
            "margin_percentile": MARGIN_PERCENTILE,
            "grid_search": False,
        },
        "fold_training": fold_training,
        "oof": {
            "margin": float(margin),
            "diagnostics": diagnostics,
            "required_delta_hits025": MIN_OOF_DELTA_025,
            "required_delta_hits050": MIN_OOF_DELTA_050,
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
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
        "oof": report["oof"],
        "calibration_status": calibration["status"],
        "calibration_passed": calibration.get("passed"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
