#!/usr/bin/env python
"""Train-only OOF audit for the V100 baseline-relative contextual ranker."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from models.rec_baseline_relative_contextual import (
    V100_DROPOUT,
    V100_HIDDEN_DIM,
    BaselineRelativeContextualReranker,
    apply_baseline_relative_policy,
    signed_effects,
)
from models.rec_hierarchical_reranker import (
    QUERY_COUNT,
    VARIANT_COUNT,
    build_hierarchical_scene_folds,
)
from scripts.run_v99_pareto_contextual_hierarchical import (
    build_diagnostics,
    tensor_sha256,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_BATCH_SIZE,
    HIERARCHICAL_GRAD_CLIP_NORM,
    HIERARCHICAL_LEARNING_RATE,
    _normalized_hierarchical_model_batch,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


EPOCHS = 12
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-3
SEED = 0
MIN_DELTA_025 = 65
MIN_DELTA_050 = 68
THRESHOLDS = (0.25, 0.50)


def set_seed(device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_relative_targets(candidate_ious, baseline_indices, variant_valid):
    if (candidate_ious.dtype != torch.float32
            or candidate_ious.dim() != 3
            or candidate_ious.shape != variant_valid.shape
            or variant_valid.dtype != torch.bool):
        raise ValueError("V100 target inputs are malformed")
    batch_size = candidate_ious.shape[0]
    if tuple(candidate_ious.shape[1:]) != (QUERY_COUNT, VARIANT_COUNT):
        raise ValueError("V100 target candidate axes changed")
    if (baseline_indices.dtype != torch.long
            or tuple(baseline_indices.shape) != (batch_size,)
            or baseline_indices.device != candidate_ious.device):
        raise ValueError("V100 target baselines are malformed")
    rows = torch.arange(batch_size, device=candidate_ious.device)
    flat_ious = candidate_ious.reshape(batch_size, -1)
    baseline_iou = flat_ious[rows, baseline_indices]
    candidate_hits = torch.stack(tuple(
        candidate_ious.gt(threshold) for threshold in THRESHOLDS
    ), dim=-1)
    baseline_hits = torch.stack(tuple(
        baseline_iou.gt(threshold) for threshold in THRESHOLDS
    ), dim=-1)
    signed = (
        candidate_hits.to(torch.long)
        - baseline_hits[:, None, None, :].to(torch.long)
    )
    classes = signed + 1
    if bool((classes[~variant_valid] != 1).any().item()):
        classes = torch.where(
            variant_valid.unsqueeze(-1), classes,
            torch.ones_like(classes),
        )
    baseline_classes = classes.reshape(batch_size, -1, 2)[
        rows, baseline_indices
    ]
    if not bool(baseline_classes.eq(1).all().item()):
        raise RuntimeError("V100 baseline target must be neutral")
    return {
        "classes": classes,
        "signed": signed.to(torch.float32),
        "baseline_hits": baseline_hits,
        "baseline_iou": baseline_iou,
    }


def _balanced_two_group_mean(values, group):
    if values.dim() != 1 or group.dtype != torch.bool or group.shape != values.shape:
        raise ValueError("V100 two-group reduction inputs are malformed")
    if not bool(group.any().item()) or not bool((~group).any().item()):
        raise ValueError("V100 batch must contain both balancing groups")
    return 0.5 * (values[group].mean() + values[~group].mean())


def _balanced_three_bin_mean(values, baseline_iou):
    if values.dim() != 1 or baseline_iou.shape != values.shape:
        raise ValueError("V100 three-bin reduction inputs are malformed")
    bins = (
        baseline_iou.le(0.25),
        baseline_iou.gt(0.25) & baseline_iou.le(0.50),
        baseline_iou.gt(0.50),
    )
    available = [values[mask].mean() for mask in bins if bool(mask.any().item())]
    if len(available) != 3:
        raise ValueError("V100 batch must contain all baseline-IoU bins")
    return sum(available) / 3.0


def stratified_epoch_batches(records, shuffle_state):
    """Use every row once while putting all baseline-IoU bins in each batch."""
    if not records:
        raise ValueError("V100 training records cannot be empty")
    batch_count = int(math.ceil(len(records) / float(BATCH_SIZE)))
    strata = ([], [], [])
    for index, record in enumerate(records):
        baseline_iou = float(
            record["candidate_ious"].reshape(-1)[record["baseline_index"]]
        )
        bin_index = 0 if baseline_iou <= 0.25 else (
            1 if baseline_iou <= 0.50 else 2
        )
        strata[bin_index].append(index)
    if any(len(values) < batch_count for values in strata):
        raise ValueError("V100 cannot stratify every mini-batch")
    buckets = [[] for _ in range(batch_count)]
    remaining = []
    for values in strata:
        shuffle_state.shuffle(values)
        for bucket, index in zip(buckets, values[:batch_count]):
            bucket.append(index)
        remaining.extend(values[batch_count:])
    shuffle_state.shuffle(remaining)
    for index in remaining:
        bucket = min(buckets, key=len)
        if len(bucket) >= BATCH_SIZE:
            raise RuntimeError("V100 stratified batch capacity changed")
        bucket.append(index)
    if (sum(len(bucket) for bucket in buckets) != len(records)
            or any(not bucket or len(bucket) > BATCH_SIZE for bucket in buckets)
            or sorted(index for bucket in buckets for index in bucket)
            != list(range(len(records)))):
        raise RuntimeError("V100 stratified batches do not partition records")
    for bucket in buckets:
        shuffle_state.shuffle(bucket)
    return buckets


def relative_effect_loss(outputs, candidate_ious, variant_valid):
    logits = outputs["relative_logits"]
    baseline_indices = outputs["baseline_indices"]
    targets = build_relative_targets(
        candidate_ious, baseline_indices, variant_valid
    )
    batch_size = candidate_ious.shape[0]
    flat_valid = variant_valid.reshape(batch_size, -1)
    flat_logits = logits.reshape(batch_size, -1, 2, 3)
    flat_classes = targets["classes"].reshape(batch_size, -1, 2)
    classification_losses = []
    for threshold_index in range(2):
        candidate_loss = F.cross_entropy(
            flat_logits[..., threshold_index, :].reshape(-1, 3),
            flat_classes[..., threshold_index].reshape(-1),
            reduction="none",
        ).reshape(batch_size, -1)
        row_loss = (
            candidate_loss * flat_valid.to(candidate_loss.dtype)
        ).sum(dim=1) / flat_valid.sum(dim=1).to(candidate_loss.dtype)
        classification_losses.append(_balanced_two_group_mean(
            row_loss, targets["baseline_hits"][:, threshold_index]
        ))
    classification_loss = sum(classification_losses) / 2.0

    predicted_effect = signed_effects(logits).reshape(batch_size, -1, 2)
    rows = torch.arange(batch_size, device=logits.device)
    predicted_effect = predicted_effect.clone()
    predicted_effect[rows, baseline_indices] = 0.0
    predicted_aggregate = (
        2.0 * predicted_effect[..., 0] + predicted_effect[..., 1]
    ).masked_fill(~flat_valid, -float("inf"))
    true_signed = targets["signed"].reshape(batch_size, -1, 2)
    true_aggregate = (
        2.0 * true_signed[..., 0] + true_signed[..., 1]
    ).masked_fill(~flat_valid, -float("inf"))
    true_max = true_aggregate.max(dim=1, keepdim=True).values
    winner = flat_valid & true_aggregate.eq(true_max)
    target_probability = winner.to(logits.dtype) / winner.sum(
        dim=1, keepdim=True
    ).to(logits.dtype)
    log_probability = torch.log_softmax(predicted_aggregate, dim=1)
    safe_terms = torch.where(
        flat_valid, target_probability * log_probability,
        torch.zeros_like(log_probability),
    )
    listwise_row_loss = -safe_terms.sum(dim=1)
    listwise_loss = _balanced_three_bin_mean(
        listwise_row_loss, targets["baseline_iou"]
    )
    loss = classification_loss + listwise_loss
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("V100 loss is non-finite")
    return loss, {
        "classification_loss": float(classification_loss.detach().item()),
        "listwise_loss": float(listwise_loss.detach().item()),
    }


def fit_model(records, statistics, device):
    resolved = torch.device(device)
    set_seed(resolved)
    model = BaselineRelativeContextualReranker(
        hidden_dim=V100_HIDDEN_DIM, dropout=V100_DROPOUT
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HIERARCHICAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    shuffle_state = random.Random(SEED)
    summaries = []
    for epoch in range(EPOCHS):
        model.train()
        totals = {"loss": 0.0, "classification_loss": 0.0, "listwise_loss": 0.0}
        batches = 0
        for indices in stratified_epoch_batches(records, shuffle_state):
            row_batch = [records[index] for index in indices]
            model_batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            candidate_ious = torch.stack([
                record["candidate_ious"] for record in row_batch
            ]).to(resolved)
            outputs = model(**model_batch)
            loss, parts = relative_effect_loss(
                outputs, candidate_ious, model_batch["variant_valid"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), HIERARCHICAL_GRAD_CLIP_NORM)
            optimizer.step()
            totals["loss"] += float(loss.detach().item())
            for name in ("classification_loss", "listwise_loss"):
                totals[name] += parts[name]
            batches += 1
        summaries.append({
            "epoch": epoch + 1,
            **{name: value / batches for name, value in totals.items()},
        })
    model.eval().requires_grad_(False)
    return model, summaries


def predict(model, records, statistics, device):
    resolved = torch.device(device)
    model.to(resolved).eval()
    pieces = {name: [] for name in (
        "selected_indices", "baseline_indices", "switch_mask",
        "signed_effects", "aggregate_effect",
    )}
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            outputs = model(**batch)
            policy = apply_baseline_relative_policy(
                outputs["relative_logits"], batch["variant_valid"],
                outputs["baseline_indices"],
            )
            for name in pieces:
                pieces[name].append(policy[name].detach().cpu())
    return {name: torch.cat(values) for name, values in pieces.items()}


def gate(diagnostics):
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


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
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
    records = materialize_hierarchical_rows(
        split["fit_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=False,
    )
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    fold_indices = {
        fold: [
            index for index, record in enumerate(records)
            if scene_folds[record["scan_id"]] == fold
        ] for fold in range(5)
    }
    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    selected = baselines.clone()
    signed = torch.zeros(len(records), QUERY_COUNT * VARIANT_COUNT, 2)
    aggregate = torch.zeros(len(records), QUERY_COUNT * VARIANT_COUNT)
    switches = torch.zeros(len(records), dtype=torch.bool)
    folds = []
    for held in range(5):
        train_indices = sorted([
            index for fold in range(5) if fold != held
            for index in fold_indices[fold]
        ])
        held_indices = fold_indices[held]
        train_records = [records[index] for index in train_indices]
        held_records = [records[index] for index in held_indices]
        statistics = fit_hierarchical_normalization(train_records)
        model, epochs = fit_model(train_records, statistics, args.device)
        prediction = predict(model, held_records, statistics, args.device)
        if not torch.equal(prediction["baseline_indices"], baselines[held_indices]):
            raise RuntimeError("V100 OOF baseline identity changed")
        selected[held_indices] = prediction["selected_indices"]
        signed[held_indices] = prediction["signed_effects"]
        aggregate[held_indices] = prediction["aggregate_effect"]
        switches[held_indices] = prediction["switch_mask"]
        record = {
            "held_fold": held,
            "train_row_count": len(train_indices),
            "train_scene_count": len({
                records[index]["scan_id"] for index in train_indices
            }),
            "held_row_count": len(held_indices),
            "held_scene_count": len({
                records[index]["scan_id"] for index in held_indices
            }),
            "normalization_sha256": statistics["sha256"],
            "accepted_switches": int(prediction["switch_mask"].sum().item()),
            "final_epoch": epochs[-1],
        }
        folds.append(record)
        print(json.dumps({"completed_fold": record}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    diagnostics = build_diagnostics(records, selected, baselines)
    predicates = gate(diagnostics)
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V100")
    report = {
        "schema": "rec-baseline-relative-contextual-train-only-v1",
        "version": 1,
        "validation_data_accessed": False,
        "contaminated_calibration_accessed": False,
        "deployable": False,
        "protocol": {
            "architecture": "baseline-conditioned contextual 112-candidate ranker",
            "objective": "balanced relative-effect 3-class CE plus signed listwise CE",
            "policy": "argmax doubly-positive signed aggregate against zero baseline anchor",
            "grid_search": False,
            "margin_search": False,
            "selection_data": "fit_rows_scene_disjoint_5_fold_oof_only",
        },
        "training_contract": {
            "hidden_dim": V100_HIDDEN_DIM,
            "dropout": V100_DROPOUT,
            "query_context_layers": 1,
            "attention_heads": 4,
            "feedforward_dim": 256,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": HIERARCHICAL_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": HIERARCHICAL_GRAD_CLIP_NORM,
            "seed": SEED,
        },
        "folds": folds,
        "prediction_sha256": tensor_sha256(
            selected, signed, aggregate, switches
        ),
        "oof": {
            "diagnostics": diagnostics,
            "required_delta_hits025": MIN_DELTA_025,
            "required_delta_hits050": MIN_DELTA_050,
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "split": copy.deepcopy(split["metadata"]),
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    payload = json.dumps(
        report, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("V100 output write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "oof": report["oof"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
