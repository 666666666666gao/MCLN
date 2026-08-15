#!/usr/bin/env python
"""Strict nested train-only verifier for frozen V97 proposals."""

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

import scripts.run_v95_threshold_aligned_listwise_hierarchical as v95
from models.rec_hierarchical_reranker import (
    apply_hierarchical_policy,
    build_hierarchical_scene_folds,
    choose_hierarchical_configuration,
)
from models.rec_selective_residual import (
    SelectiveResidualModel,
    compute_selective_residual_loss,
    expected_selective_gain,
)
from scripts.audit_v97_contextual_calibration import load_source
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    build_hierarchical_policy_candidate,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)
from scripts.train_scanrefer_rec_selective_residual import (
    materialize_residual_rows,
)


V97_MARGIN = 0.13312220573425293
VERIFIER_EPOCHS = 10
VERIFIER_BATCH_SIZE = 256
VERIFIER_LEARNING_RATE = 3e-4
VERIFIER_WEIGHT_DECAY = 1e-3
VERIFIER_BREAK_COST = 4.0
MIN_DELTA_025 = 65
MIN_DELTA_050 = 68


def fit_v97(records, device):
    statistics = fit_hierarchical_normalization(records)
    original = v95.HierarchicalQueryVariantReranker
    v95.HierarchicalQueryVariantReranker = ContextualHierarchicalQueryVariantReranker
    try:
        model, epochs = v95.fit_graded_listwise_model(
            records, statistics, device
        )
    finally:
        v95.HierarchicalQueryVariantReranker = original
    return model, statistics, epochs


def gated_v97_predictions(model, statistics, records, device):
    proposals, gain = v95._predict_hierarchical_proposals(
        model, records, statistics, torch.device(device)
    )
    base_scores = torch.stack([record["baseline_scores"] for record in records])
    variant_valid = torch.stack([record["variant_valid"] for record in records])
    policy = apply_hierarchical_policy(
        base_scores,
        proposals,
        gain,
        variant_valid,
        V97_MARGIN,
    )
    return policy["selected_indices"].cpu(), gain.cpu()


def target_for_pair(record, proposal):
    baseline_iou = float(record["candidate_ious"][record["baseline_index"]])
    proposal_iou = float(record["candidate_ious"][proposal])
    labels = []
    for threshold in (0.25, 0.50):
        before = baseline_iou > threshold
        after = proposal_iou > threshold
        labels.append(0 if before and not after else 2 if not before and after else 1)
    return labels


def pack_verifier_rows(residual_records, proposals, indices):
    features = []
    targets = []
    row_indices = []
    for index in indices:
        record = residual_records[index]
        proposal = int(proposals[index].item())
        if proposal == record["baseline_index"]:
            continue
        if not bool(record["pair_valid"][proposal].item()):
            raise RuntimeError("V98 proposal is not a valid residual pair")
        features.append(record["pair_features"][proposal])
        targets.append(target_for_pair(record, proposal))
        row_indices.append(index)
    if not features:
        raise RuntimeError("V98 verifier split has no proposed switches")
    return {
        "features": torch.stack(features).float(),
        "targets": torch.tensor(targets, dtype=torch.long),
        "row_indices": row_indices,
    }


def set_seed(device):
    random.seed(0)
    torch.manual_seed(0)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fit_verifier(packed, device):
    resolved = torch.device(device)
    set_seed(resolved)
    model = SelectiveResidualModel(
        input_dim=185, hidden_dim=64, dropout=0.1
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=VERIFIER_LEARNING_RATE,
        weight_decay=VERIFIER_WEIGHT_DECAY,
    )
    shuffle = random.Random(0)
    summaries = []
    for epoch in range(VERIFIER_EPOCHS):
        order = list(range(len(packed["row_indices"])))
        shuffle.shuffle(order)
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, len(order), VERIFIER_BATCH_SIZE):
            indices = order[start:start + VERIFIER_BATCH_SIZE]
            features = packed["features"][indices].to(resolved).unsqueeze(1)
            targets = packed["targets"][indices].to(resolved).unsqueeze(1)
            valid = torch.ones(
                len(indices), 1, dtype=torch.bool, device=resolved
            )
            logits = model(features, valid)
            loss, _stats = compute_selective_residual_loss(
                logits, targets, valid, break_cost=VERIFIER_BREAK_COST
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        summaries.append({"epoch": epoch + 1, "loss": total_loss / batches})
    model.eval().requires_grad_(False)
    return model, summaries


def predict_verifier(model, packed, total_rows, device):
    gain = torch.full((total_rows,), -1.0, dtype=torch.float32)
    resolved = torch.device(device)
    model.to(resolved).eval()
    with torch.no_grad():
        for start in range(0, len(packed["row_indices"]), VERIFIER_BATCH_SIZE):
            end = start + VERIFIER_BATCH_SIZE
            features = packed["features"][start:end].to(resolved).unsqueeze(1)
            valid = torch.ones(
                features.shape[0], 1, dtype=torch.bool, device=resolved
            )
            logits = model(features, valid)
            values = expected_selective_gain(logits)[:, 0].cpu()
            for row_index, value in zip(
                    packed["row_indices"][start:end], values):
                gain[row_index] = value
    return gain


def verifier_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_diagnostics(hierarchical_records, proposals, accepted_gain):
    config = {
        "hidden_dim": 64,
        "weight_decay": VERIFIER_WEIGHT_DECAY,
        "false_positive_cost": VERIFIER_BREAK_COST,
    }
    decision_gain = torch.where(
        accepted_gain > 0.0,
        torch.ones_like(accepted_gain),
        -torch.ones_like(accepted_gain),
    )
    candidate = build_hierarchical_policy_candidate(
        hierarchical_records,
        proposals,
        decision_gain,
        config,
        50.0,
        0.5,
    )
    return choose_hierarchical_configuration([candidate])[
        "candidate_diagnostics"
    ][0]


def gate(diagnostics):
    folds = diagnostics["fold_deltas"].values()
    predicates = {
        "delta025_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits025"] >= MIN_DELTA_025
        ),
        "delta050_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits050"] >= MIN_DELTA_050
        ),
        "all_folds_nonnegative025": all(row["hits025"] >= 0 for row in folds),
        "all_folds_nonnegative050": all(
            row["hits050"] >= 0 for row in diagnostics["fold_deltas"].values()
        ),
        "bootstrap025_lower_bound_positive": (
            diagnostics["bootstrap025"]["lower_bound_95"] > 0
        ),
        "bootstrap050_lower_bound_positive": (
            diagnostics["bootstrap050"]["lower_bound_95"] > 0
        ),
    }
    return predicates


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v97-source", required=True)
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
    source = load_source(args.v97_source)
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
    hierarchical = materialize_hierarchical_rows(
        split["fit_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=False,
    )
    residual = materialize_residual_rows(
        split["fit_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=False,
    )
    if [record["dataset_index"] for record in hierarchical] != [
            record["dataset_index"] for record in residual]:
        raise RuntimeError("V98 hierarchy/residual row identities differ")
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in hierarchical
    ])
    indices_by_fold = {
        fold: [
            index for index, record in enumerate(hierarchical)
            if scene_folds[record["scan_id"]] == fold
        ] for fold in range(5)
    }
    total = len(hierarchical)
    main_proposals = torch.tensor([
        record["baseline_index"] for record in residual
    ], dtype=torch.long)
    main_models = []
    for held in range(5):
        train_indices = sorted([
            index for fold in range(5) if fold != held
            for index in indices_by_fold[fold]
        ])
        model, statistics, epochs = fit_v97(
            [hierarchical[index] for index in train_indices], args.device
        )
        selected, _gain = gated_v97_predictions(
            model,
            statistics,
            [hierarchical[index] for index in indices_by_fold[held]],
            args.device,
        )
        main_proposals[indices_by_fold[held]] = selected
        main_models.append({
            "held_fold": held,
            "train_row_count": len(train_indices),
            "held_row_count": len(indices_by_fold[held]),
            "normalization_sha256": statistics["sha256"],
            "final_epoch": epochs[-1],
        })
        print(json.dumps({"completed_main_v97": main_models[-1]}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    pair_predictions = {}
    pair_models = []
    for first in range(5):
        for second in range(first + 1, 5):
            train_indices = sorted([
                index for fold in range(5) if fold not in (first, second)
                for index in indices_by_fold[fold]
            ])
            model, statistics, epochs = fit_v97(
                [hierarchical[index] for index in train_indices], args.device
            )
            pair_predictions[(first, second)] = {}
            for target_fold in (first, second):
                selected, _gain = gated_v97_predictions(
                    model,
                    statistics,
                    [hierarchical[index] for index in indices_by_fold[target_fold]],
                    args.device,
                )
                pair_predictions[(first, second)][target_fold] = selected
            record = {
                "excluded_folds": [first, second],
                "train_row_count": len(train_indices),
                "normalization_sha256": statistics["sha256"],
                "final_epoch": epochs[-1],
            }
            pair_models.append(record)
            print(json.dumps({"completed_pair_v97": record}, sort_keys=True), flush=True)
            del model
            torch.cuda.empty_cache()

    nested_gain = torch.full((total,), -1.0, dtype=torch.float32)
    ordinary_gain = torch.full((total,), -1.0, dtype=torch.float32)
    verifier_records = []
    for held in range(5):
        nested_train_proposals = main_proposals.clone()
        train_indices = []
        for row_fold in range(5):
            if row_fold == held:
                continue
            key = tuple(sorted((held, row_fold)))
            nested_train_proposals[indices_by_fold[row_fold]] = (
                pair_predictions[key][row_fold]
            )
            train_indices.extend(indices_by_fold[row_fold])
        train_indices.sort()
        held_indices = indices_by_fold[held]
        nested_packed = pack_verifier_rows(
            residual, nested_train_proposals, train_indices
        )
        held_packed = pack_verifier_rows(
            residual, main_proposals, held_indices
        )
        nested_model, nested_epochs = fit_verifier(nested_packed, args.device)
        nested_values = predict_verifier(
            nested_model, held_packed, total, args.device
        )
        nested_gain[held_indices] = nested_values[held_indices]

        ordinary_packed = pack_verifier_rows(
            residual, main_proposals, train_indices
        )
        ordinary_model, ordinary_epochs = fit_verifier(
            ordinary_packed, args.device
        )
        ordinary_values = predict_verifier(
            ordinary_model, held_packed, total, args.device
        )
        ordinary_gain[held_indices] = ordinary_values[held_indices]
        record = {
            "held_fold": held,
            "nested_train_switches": len(nested_packed["row_indices"]),
            "ordinary_train_switches": len(ordinary_packed["row_indices"]),
            "held_switches": len(held_packed["row_indices"]),
            "nested_state_sha256": verifier_state_sha256(nested_model),
            "ordinary_state_sha256": verifier_state_sha256(ordinary_model),
            "nested_final_epoch": nested_epochs[-1],
            "ordinary_final_epoch": ordinary_epochs[-1],
        }
        verifier_records.append(record)
        print(json.dumps({"completed_verifier": record}, sort_keys=True), flush=True)
        del nested_model, ordinary_model
        torch.cuda.empty_cache()

    nested_diagnostics = build_diagnostics(
        hierarchical, main_proposals, nested_gain
    )
    ordinary_diagnostics = build_diagnostics(
        hierarchical, main_proposals, ordinary_gain
    )
    predicates = gate(nested_diagnostics)
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V98")
    report = {
        "schema": "rec-nested-v97-proposal-verifier-train-only-v1",
        "version": 1,
        "validation_data_accessed": False,
        "deployable": False,
        "v97_source": {
            "path": str(Path(args.v97_source).expanduser().absolute()),
            "sha256": "ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d",
            "oof": copy.deepcopy(source["oof"]),
        },
        "protocol": {
            "nested": True,
            "main_v97_model_count": 5,
            "pair_exclusion_v97_model_count": 10,
            "verifier_model_count": 5,
            "hidden_dim": 64,
            "weight_decay": VERIFIER_WEIGHT_DECAY,
            "break_cost": VERIFIER_BREAK_COST,
            "epochs": VERIFIER_EPOCHS,
            "acceptance": "expected_signed_gain_gt_zero",
            "grid_search": False,
        },
        "main_v97_models": main_models,
        "pair_exclusion_v97_models": pair_models,
        "verifiers": verifier_records,
        "nested": {
            "diagnostics": nested_diagnostics,
            "required_delta_hits025": MIN_DELTA_025,
            "required_delta_hits050": MIN_DELTA_050,
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "ordinary_stacking_diagnostic_only": {
            "diagnostics": ordinary_diagnostics,
            "uses_outer_held_labels_in_training_proposal_generators": True,
        },
        "contaminated_calibration_accessed": False,
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
        "nested": report["nested"],
        "ordinary": report["ordinary_stacking_diagnostic_only"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
