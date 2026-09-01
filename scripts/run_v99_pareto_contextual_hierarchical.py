#!/usr/bin/env python
"""Train-only OOF audit of a Pareto-gated contextual REC hierarchy."""

import argparse
import copy
import hashlib
import json
import math
import os
from functools import partial
from pathlib import Path

import torch

import scripts.run_v95_threshold_aligned_listwise_hierarchical as v95
from models.rec_hierarchical_reranker import (
    build_hierarchical_scene_folds,
    choose_hierarchical_configuration,
    monotone_hit_probabilities,
    select_hierarchical_proposal,
)
from scripts.audit_v97_contextual_calibration import load_source
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_BATCH_SIZE,
    _normalized_hierarchical_model_batch,
    build_hierarchical_policy_candidate,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)
from scripts.train_scanrefer_rec_selective_residual import (
    _validate_materialization_artifact,
)


V97_SOURCE_SHA256 = (
    "ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d"
)
V97_MARGIN = 0.13312220573425293
MIN_DELTA_025 = 65
MIN_DELTA_050 = 68


def build_materialization_artifact_validator(
        portable_dataset_contract, protected_before):
    """Bind geometry validation to this dataset-specific artifact chain."""
    if not portable_dataset_contract:
        return None
    if not isinstance(protected_before, dict):
        raise ValueError("protected artifact identities must be an object")
    expected = {}
    for label in ("backbone", "parent"):
        identity = protected_before.get(label)
        sha256 = identity.get("sha256") if isinstance(identity, dict) else None
        if (not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in sha256)):
            raise ValueError(
                "protected {} SHA-256 is invalid".format(label)
            )
        expected[label] = sha256
    return partial(
        _validate_materialization_artifact,
        expected_backbone_sha256=expected["backbone"],
        expected_parent_artifact_sha256=expected["parent"],
        require_geometry_weight_one=False,
    )


def fit_v97(records, device):
    statistics = fit_hierarchical_normalization(records)
    original = v95.HierarchicalQueryVariantReranker
    v95.HierarchicalQueryVariantReranker = (
        ContextualHierarchicalQueryVariantReranker
    )
    try:
        model, epochs = v95.fit_graded_listwise_model(
            records, statistics, device
        )
    finally:
        v95.HierarchicalQueryVariantReranker = original
    return model, statistics, epochs


def pareto_accept_mask(proposals, baseline, aggregate_gain, head_gain):
    if (proposals.dtype != torch.long or baseline.dtype != torch.long
            or proposals.shape != baseline.shape):
        raise ValueError("proposal and baseline vectors must align as int64")
    if (aggregate_gain.dtype != torch.float32
            or aggregate_gain.shape != proposals.shape
            or head_gain.dtype != torch.float32
            or head_gain.shape != proposals.shape + (2,)):
        raise ValueError("V99 gain tensors have invalid shape or dtype")
    if (not bool(torch.isfinite(aggregate_gain).all().item())
            or not bool(torch.isfinite(head_gain).all().item())):
        raise ValueError("V99 gain tensors must be finite")
    return (
        proposals.ne(baseline)
        & aggregate_gain.ge(V97_MARGIN)
        & head_gain[..., 0].gt(0.0)
        & head_gain[..., 1].gt(0.0)
    )


def predict_pareto(model, records, statistics, device):
    resolved = torch.device(device)
    model.to(resolved).eval()
    all_proposals = []
    all_baselines = []
    all_aggregate_gain = []
    all_head_gain = []
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            outputs = model(**batch)
            selected = select_hierarchical_proposal(
                outputs["query_logits"],
                outputs["variant_logits"],
                batch["query_valid"],
                batch["variant_valid"],
            )
            probabilities = monotone_hit_probabilities(
                outputs["variant_logits"]
            ).reshape(len(row_batch), -1, 2)
            baselines = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=resolved)
            proposals = selected["flat_indices"]
            rows = torch.arange(len(row_batch), device=resolved)
            head_gain = (
                probabilities[rows, proposals]
                - probabilities[rows, baselines]
            )
            aggregate_gain = (
                2.0 * head_gain[..., 0] + head_gain[..., 1]
            )
            all_proposals.append(proposals.cpu().long())
            all_baselines.append(baselines.cpu().long())
            all_aggregate_gain.append(aggregate_gain.cpu().float())
            all_head_gain.append(head_gain.cpu().float())
    proposals = torch.cat(all_proposals)
    baselines = torch.cat(all_baselines)
    aggregate_gain = torch.cat(all_aggregate_gain)
    head_gain = torch.cat(all_head_gain)
    accepted = pareto_accept_mask(
        proposals, baselines, aggregate_gain, head_gain
    )
    selected = torch.where(accepted, proposals, baselines)
    return {
        "proposals": proposals,
        "baselines": baselines,
        "aggregate_gain": aggregate_gain,
        "head_gain": head_gain,
        "accepted": accepted,
        "selected": selected,
    }


def tensor_sha256(*values):
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_diagnostics(records, selected, baselines):
    accepted = selected.ne(baselines)
    decision_gain = torch.where(
        accepted,
        torch.ones(len(records), dtype=torch.float32),
        -torch.ones(len(records), dtype=torch.float32),
    )
    candidate = build_hierarchical_policy_candidate(
        records,
        selected,
        decision_gain,
        {
            "hidden_dim": 128,
            "weight_decay": 1e-3,
            "false_positive_cost": 4.0,
        },
        50.0,
        0.5,
    )
    return choose_hierarchical_configuration([candidate])[
        "candidate_diagnostics"
    ][0]


def acceptance_gate(diagnostics):
    folds = diagnostics["fold_deltas"].values()
    return {
        "delta025_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits025"] >= MIN_DELTA_025
        ),
        "delta050_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits050"] >= MIN_DELTA_050
        ),
        "all_folds_nonnegative025": all(
            row["hits025"] >= 0 for row in folds
        ),
        "all_folds_nonnegative050": all(
            row["hits050"] >= 0
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--v97-source", required=True)
    parser.add_argument(
        "--dataset", choices=("scanrefer", "nr3d", "sr3d"),
        default="scanrefer",
    )
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument(
        "--portable-dataset-contract",
        action="store_true",
        help=(
            "retain the exact V99 16x7 model/policy while binding OOF "
            "evidence to a new dataset-specific backbone"
        ),
    )
    parser.add_argument(
        "--backbone-joint-training",
        action="store_true",
        help="record that the frozen backbone used auxiliary ScanNet training rows",
    )
    parser.add_argument(
        "--inference-uses-ground-truth",
        action="store_true",
        help="record GT proposals with predicted classes at deployment",
    )
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    if args.portable_dataset_contract:
        if not args.backbone_checkpoint:
            parser.error(
                "--portable-dataset-contract requires --backbone-checkpoint"
            )
    elif (args.backbone_checkpoint is not None
          or args.dataset != "scanrefer"
          or args.backbone_joint_training
          or args.inference_uses_ground_truth):
        parser.error(
            "dataset/backbone overrides require --portable-dataset-contract"
        )
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(str(output))
    source_path = Path(args.v97_source).expanduser().absolute()
    source = load_source(source_path)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != V97_SOURCE_SHA256:
        raise ValueError("V97 source SHA-256 changed")

    protected_backbone = (
        Path(args.backbone_checkpoint).expanduser().resolve()
        if args.portable_dataset_contract
        else Path(AUTHORITATIVE_BACKBONE_PATH).resolve()
    )
    protected_paths = {
        "backbone": protected_backbone,
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    loaded = load_residual_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
        portable_provenance=args.portable_dataset_contract,
        expected_backbone_sha256=(
            protected_before["backbone"]["sha256"]
            if args.portable_dataset_contract else None
        ),
        expected_dataset=(
            args.dataset if args.portable_dataset_contract else None
        ),
    )
    split = split_residual_joined_rows(
        loaded["joined_rows"],
        portable_provenance=args.portable_dataset_contract,
    )
    records = materialize_hierarchical_rows(
        split["fit_rows"], loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=False,
        artifact_validator=build_materialization_artifact_validator(
            args.portable_dataset_contract, protected_before
        ),
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
    total = len(records)
    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    proposals = baselines.clone()
    selected = baselines.clone()
    aggregate_gain = torch.zeros(total, dtype=torch.float32)
    head_gain = torch.zeros(total, 2, dtype=torch.float32)
    accepted = torch.zeros(total, dtype=torch.bool)
    folds = []
    for held in range(5):
        train_indices = sorted([
            index for fold in range(5) if fold != held
            for index in fold_indices[fold]
        ])
        held_indices = fold_indices[held]
        model, statistics, epochs = fit_v97(
            [records[index] for index in train_indices], args.device
        )
        prediction = predict_pareto(
            model,
            [records[index] for index in held_indices],
            statistics,
            args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_indices]):
            raise RuntimeError("V99 OOF baseline identity changed")
        proposals[held_indices] = prediction["proposals"]
        selected[held_indices] = prediction["selected"]
        aggregate_gain[held_indices] = prediction["aggregate_gain"]
        head_gain[held_indices] = prediction["head_gain"]
        accepted[held_indices] = prediction["accepted"]
        fold_record = {
            "held_fold": held,
            "train_row_count": len(train_indices),
            "held_row_count": len(held_indices),
            "accepted_switches": int(prediction["accepted"].sum().item()),
            "normalization_sha256": statistics["sha256"],
            "final_epoch": epochs[-1],
        }
        folds.append(fold_record)
        print(json.dumps({"completed_fold": fold_record}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    diagnostics = build_diagnostics(records, selected, baselines)
    predicates = acceptance_gate(diagnostics)
    raw_switch = proposals.ne(baselines)
    original_margin_pass = raw_switch & aggregate_gain.ge(V97_MARGIN)
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
        raise RuntimeError("protected artifacts changed during V99")
    report = {
        "schema": "rec-pareto-contextual-hierarchical-train-only-v1",
        "version": 1,
        "validation_data_accessed": False,
        "contaminated_calibration_accessed": False,
        "deployable": False,
        "source": {
            "path": str(source_path),
            "sha256": V97_SOURCE_SHA256,
            "oof": copy.deepcopy(source["oof"]),
        },
        "protocol": {
            "architecture": "V97 contextual query-set hierarchy",
            "objective": "V95 bounded threshold-aligned listwise",
            "proposal_margin": V97_MARGIN,
            "acceptance": "aggregate_margin_and_positive_delta025_and_delta050",
            "grid_search": False,
            "selection_data": "fit_rows_scene_disjoint_5_fold_oof_only",
        },
        "folds": folds,
        "veto_diagnostics": veto_reasons,
        "prediction_sha256": tensor_sha256(
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
        "split": copy.deepcopy(split["metadata"]),
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    if args.portable_dataset_contract:
        report["dataset_contract"] = {
            "dataset": args.dataset,
            "reranker_dataset_only": True,
            "dataset_only": True,
            "joint_training": False,
            "backbone_training_dataset_only": not args.backbone_joint_training,
            "backbone_joint_training": args.backbone_joint_training,
            "inference_uses_ground_truth": args.inference_uses_ground_truth,
            "backbone_checkpoint": str(protected_backbone),
            "backbone_sha256": protected_before["backbone"]["sha256"],
            "query_count": 16,
            "variant_count": 7,
            "flat_candidate_count": 112,
            "method": "V99-contextual-pareto-16x7",
            "v99_script_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
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
                raise OSError("V99 output write made no progress")
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
