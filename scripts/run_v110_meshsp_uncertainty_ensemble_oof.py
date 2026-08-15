#!/usr/bin/env python
"""Train-only nested OOF audit of a seed-uncertainty MeshSP ensemble."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch

import scripts.run_v95_threshold_aligned_listwise_hierarchical as v95
import scripts.run_v99_pareto_contextual_hierarchical as v99
from models.rec_hierarchical_reranker import (
    VARIANT_COUNT,
    build_hierarchical_scene_folds,
    monotone_hit_probabilities,
    select_hierarchical_proposal,
)
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)
from scripts.run_v108_meshsp_pareto_oof import (
    EXPECTED_CORRECTED_SCENE_COUNT,
    EXPECTED_FALLBACK_SCENE_COUNT,
    EXPECTED_REGULAR_SCENE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SCENE_COUNT,
    FALLBACK_MANIFEST_SHA256,
    file_sha256,
    load_v108_meshsp_training_inputs,
    validate_v108_materialization_artifact,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_BATCH_SIZE,
    _normalized_hierarchical_model_batch,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)


V109_REPORT_SHA256 = (
    "37680aaa34757cf9bb2376e93629ae6b89aa6b8fac16960ac091305cc20146a1"
)
ENSEMBLE_SEEDS = (0, 1, 2)
PRIOR_DELTA_025 = 72
PRIOR_DELTA_050 = 246
MIN_DELTA_025 = PRIOR_DELTA_025 + 1
MIN_DELTA_050 = PRIOR_DELTA_050
PRIOR_BOOTSTRAP_LOWER = {"025": 34, "050": 183}
PRIOR_SUBGROUP_BOOTSTRAP_LOWER = {
    "corrected": {"025": 20, "050": 131},
    "regular": {"025": -1, "050": 29},
}
MARGINS = (0.10, float(v99.V97_MARGIN), 0.15, 0.18)
MIN_LCB_025 = (0.0, 0.01, 0.02)
UNCERTAINTY_PENALTIES = (0.0, 0.5, 1.0)
MIN_CONSENSUS = (2.0 / 3.0, 1.0)
POLICY_GRID = tuple(
    {
        "aggregate_lcb_margin": float(margin),
        "min_head_lcb025": float(minimum),
        "min_head_lcb050": 0.0,
        "uncertainty_penalty": float(penalty),
        "min_consensus": float(consensus),
    }
    for margin in MARGINS
    for minimum in MIN_LCB_025
    for penalty in UNCERTAINTY_PENALTIES
    for consensus in MIN_CONSENSUS
)


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def fit_seeded_model(records, statistics, device, seed):
    if seed not in ENSEMBLE_SEEDS:
        raise ValueError("V110 seed is outside the frozen ensemble")
    original_model = v95.HierarchicalQueryVariantReranker
    original_seed = v95._set_hierarchical_seed

    def set_seed(resolved):
        random.seed(seed)
        torch.manual_seed(seed)
        if resolved.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    v95.HierarchicalQueryVariantReranker = (
        ContextualHierarchicalQueryVariantReranker
    )
    v95._set_hierarchical_seed = set_seed
    try:
        model, epochs = v95.fit_graded_listwise_model(
            records, statistics, device
        )
    finally:
        v95.HierarchicalQueryVariantReranker = original_model
        v95._set_hierarchical_seed = original_seed
    return model, epochs


def _validate_policy_inputs(
        proposals, baselines, mean_head_gain, head_std, consensus):
    if (proposals.dtype != torch.long or baselines.dtype != torch.long
            or proposals.shape != baselines.shape or proposals.dim() != 1):
        raise ValueError("V110 proposals and baselines must align as int64")
    expected_heads = proposals.shape + (2,)
    if (mean_head_gain.dtype != torch.float32
            or mean_head_gain.shape != expected_heads
            or head_std.dtype != torch.float32
            or head_std.shape != expected_heads
            or consensus.dtype != torch.float32
            or consensus.shape != proposals.shape):
        raise ValueError("V110 uncertainty tensors have invalid shape or dtype")
    if (not bool(torch.isfinite(mean_head_gain).all().item())
            or not bool(torch.isfinite(head_std).all().item())
            or not bool(torch.isfinite(consensus).all().item())
            or bool(head_std.lt(0.0).any().item())
            or bool(consensus.lt(0.0).any().item())
            or bool(consensus.gt(1.0).any().item())):
        raise ValueError("V110 uncertainty tensors are invalid")


def policy_accept_mask(
        proposals, baselines, mean_head_gain, head_std, consensus, policy):
    if policy not in POLICY_GRID:
        raise ValueError("V110 policy is outside the frozen grid")
    _validate_policy_inputs(
        proposals, baselines, mean_head_gain, head_std, consensus
    )
    head_lcb = (
        mean_head_gain - policy["uncertainty_penalty"] * head_std
    )
    aggregate_lcb = 2.0 * head_lcb[:, 0] + head_lcb[:, 1]
    return (
        proposals.ne(baselines)
        & aggregate_lcb.ge(policy["aggregate_lcb_margin"])
        & head_lcb[:, 0].gt(policy["min_head_lcb025"])
        & head_lcb[:, 1].gt(policy["min_head_lcb050"])
        & consensus.ge(policy["min_consensus"])
    )


def policy_summary(
        indices, proposals, baselines, mean_head_gain, head_std, consensus,
        baseline_ious, proposal_ious, fold_ids, policy):
    indices = torch.as_tensor(indices, dtype=torch.long)
    accepted = policy_accept_mask(
        proposals[indices], baselines[indices], mean_head_gain[indices],
        head_std[indices], consensus[indices], policy,
    )
    selected_ious = torch.where(
        accepted, proposal_ious[indices], baseline_ious[indices]
    )
    base_ious = baseline_ious[indices]
    delta025_bits = (
        selected_ious.gt(0.25).long() - base_ious.gt(0.25).long()
    )
    delta050_bits = (
        selected_ious.gt(0.50).long() - base_ious.gt(0.50).long()
    )
    folds = sorted(set(int(value) for value in fold_ids[indices].tolist()))
    fold_deltas = {}
    for fold in folds:
        mask = fold_ids[indices].eq(fold)
        fold_deltas[str(fold)] = {
            "hits025": int(delta025_bits[mask].sum().item()),
            "hits050": int(delta050_bits[mask].sum().item()),
        }
    return {
        "policy": copy.deepcopy(policy),
        "row_count": int(indices.numel()),
        "switches": int(accepted.sum().item()),
        "delta_hits025": int(delta025_bits.sum().item()),
        "delta_hits050": int(delta050_bits.sum().item()),
        "fold_deltas": fold_deltas,
    }


def _policy_rank(row):
    policy = row["policy"]
    return (
        row["delta_hits025"],
        row["delta_hits050"],
        -row["switches"],
        policy["uncertainty_penalty"],
        policy["min_head_lcb025"],
        policy["min_consensus"],
        policy["aggregate_lcb_margin"],
    )


def select_nested_policy(
        indices, proposals, baselines, mean_head_gain, head_std, consensus,
        baseline_ious, proposal_ious, fold_ids):
    indices = torch.as_tensor(indices, dtype=torch.long)
    minimum_delta025 = math.ceil(
        MIN_DELTA_025 * int(indices.numel()) / EXPECTED_SAMPLE_COUNT
    )
    minimum_delta050 = math.ceil(
        MIN_DELTA_050 * int(indices.numel()) / EXPECTED_SAMPLE_COUNT
    )
    candidates = []
    for policy in POLICY_GRID:
        summary = policy_summary(
            indices, proposals, baselines, mean_head_gain, head_std, consensus,
            baseline_ious, proposal_ious, fold_ids, policy,
        )
        summary["eligible"] = bool(
            summary["delta_hits025"] >= minimum_delta025
            and summary["delta_hits050"] >= minimum_delta050
            and all(row["hits025"] >= 0 and row["hits050"] >= 0
                    for row in summary["fold_deltas"].values())
        )
        candidates.append(summary)
    eligible = [row for row in candidates if row["eligible"]]
    selection_feasible = bool(eligible)
    winner = max(eligible if eligible else candidates, key=_policy_rank)
    return {
        "selection_rule": (
            "require_v109_scaled_delta_floors_and_nonnegative_source_folds;"
            "maximize_delta025_then_delta050_then_fewer_switches_then_"
            "conservatism_on_other_four_scene_oof_folds"
        ),
        "minimum_delta025": minimum_delta025,
        "minimum_delta050": minimum_delta050,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selection_feasible": selection_feasible,
        "selected": copy.deepcopy(winner),
        "candidates": candidates,
    }


def predict_uncertainty_ensemble(models, records, statistics, device):
    if len(models) != len(ENSEMBLE_SEEDS):
        raise ValueError("V110 requires exactly three ensemble members")
    resolved = torch.device(device)
    for model in models:
        model.to(resolved).eval()
    all_proposals = []
    all_baselines = []
    all_mean_head_gain = []
    all_head_std = []
    all_consensus = []
    all_member_proposals = []
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            query_probabilities = []
            variant_probabilities = []
            member_proposals = []
            for model in models:
                outputs = model(**batch)
                query_probabilities.append(monotone_hit_probabilities(
                    outputs["query_logits"]
                ))
                variant_probabilities.append(monotone_hit_probabilities(
                    outputs["variant_logits"]
                ))
                member_proposals.append(select_hierarchical_proposal(
                    outputs["query_logits"], outputs["variant_logits"],
                    batch["query_valid"], batch["variant_valid"],
                )["flat_indices"])
            query_probability = torch.stack(
                query_probabilities, dim=0
            ).mean(dim=0)
            variant_probability = torch.stack(
                variant_probabilities, dim=0
            ).mean(dim=0)
            query_utility = (
                2.0 * query_probability[..., 0]
                + query_probability[..., 1]
            )
            variant_utility = (
                2.0 * variant_probability[..., 0]
                + variant_probability[..., 1]
            )
            selected_query = query_utility.masked_fill(
                ~batch["query_valid"], -float("inf")
            ).argmax(dim=1)
            rows = torch.arange(len(row_batch), device=resolved)
            selected_variant = variant_utility[
                rows, selected_query
            ].masked_fill(
                ~batch["variant_valid"][rows, selected_query], -float("inf")
            ).argmax(dim=1)
            proposals = selected_query * VARIANT_COUNT + selected_variant
            baselines = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=resolved)
            member_proposals = torch.stack(member_proposals, dim=0)
            consensus = member_proposals.eq(
                proposals.unsqueeze(0)
            ).float().mean(dim=0)
            member_variant_probability = torch.stack(
                variant_probabilities, dim=0
            ).reshape(len(models), len(row_batch), -1, 2)
            seed_rows = torch.arange(len(models), device=resolved).view(-1, 1)
            batch_rows = rows.view(1, -1)
            selected_probability = member_variant_probability[
                seed_rows, batch_rows, proposals.view(1, -1)
            ]
            baseline_probability = member_variant_probability[
                seed_rows, batch_rows, baselines.view(1, -1)
            ]
            member_head_gain = selected_probability - baseline_probability
            mean_head_gain = member_head_gain.mean(dim=0)
            head_std = member_head_gain.std(dim=0, unbiased=False)
            all_proposals.append(proposals.cpu().long())
            all_baselines.append(baselines.cpu().long())
            all_mean_head_gain.append(mean_head_gain.cpu().float())
            all_head_std.append(head_std.cpu().float())
            all_consensus.append(consensus.cpu().float())
            all_member_proposals.append(
                member_proposals.transpose(0, 1).cpu().long()
            )
    result = {
        "proposals": torch.cat(all_proposals),
        "baselines": torch.cat(all_baselines),
        "mean_head_gain": torch.cat(all_mean_head_gain),
        "head_std": torch.cat(all_head_std),
        "consensus": torch.cat(all_consensus),
        "member_proposals": torch.cat(all_member_proposals),
    }
    _validate_policy_inputs(
        result["proposals"], result["baselines"],
        result["mean_head_gain"], result["head_std"], result["consensus"],
    )
    return result


def acceptance_gate(diagnostics, subgroups, meta_folds):
    folds = diagnostics["fold_deltas"].values()
    policies = [row["selected_policy"] for row in meta_folds]
    return {
        "delta025_strictly_exceeds_v109": (
            diagnostics["delta_hits025"] >= MIN_DELTA_025
        ),
        "delta050_at_least_v109": (
            diagnostics["delta_hits050"] >= MIN_DELTA_050
        ),
        "all_folds_strictly_positive025": all(
            row["hits025"] > 0 for row in folds
        ),
        "all_folds_strictly_positive050": all(
            row["hits050"] > 0
            for row in diagnostics["fold_deltas"].values()
        ),
        "bootstrap025_lower_at_least_v109": (
            diagnostics["bootstrap025"]["lower_bound_95"]
            >= PRIOR_BOOTSTRAP_LOWER["025"]
        ),
        "bootstrap050_lower_at_least_v109": (
            diagnostics["bootstrap050"]["lower_bound_95"]
            >= PRIOR_BOOTSTRAP_LOWER["050"]
        ),
        "corrected_bootstrap025_lower_at_least_v109": (
            subgroups["corrected"]["diagnostics"]["bootstrap025"][
                "lower_bound_95"
            ] >= PRIOR_SUBGROUP_BOOTSTRAP_LOWER["corrected"]["025"]
        ),
        "corrected_bootstrap050_lower_at_least_v109": (
            subgroups["corrected"]["diagnostics"]["bootstrap050"][
                "lower_bound_95"
            ] >= PRIOR_SUBGROUP_BOOTSTRAP_LOWER["corrected"]["050"]
        ),
        "regular_bootstrap025_lower_strictly_positive": (
            subgroups["regular"]["diagnostics"]["bootstrap025"][
                "lower_bound_95"
            ] > 0
        ),
        "regular_bootstrap050_lower_at_least_v109": (
            subgroups["regular"]["diagnostics"]["bootstrap050"][
                "lower_bound_95"
            ] >= PRIOR_SUBGROUP_BOOTSTRAP_LOWER["regular"]["050"]
        ),
        "all_meta_policy_selections_feasible": all(
            row["calibration_selection"]["selection_feasible"]
            for row in meta_folds
        ),
        "meta_policy_consistent": all(
            policy == policies[0] for policy in policies[1:]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v109-report", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--fallback-scenes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(str(output))

    source_path = Path(args.v109_report).expanduser().resolve(strict=True)
    if file_sha256(source_path) != V109_REPORT_SHA256:
        raise ValueError("V109 source report SHA-256 changed")
    source = json.loads(source_path.read_text(encoding="ascii"))
    source_oof = source.get("oof", {})
    source_diagnostics = source_oof.get("diagnostics", {})
    source_subgroups = source_oof.get("subgroups", {})
    if (source.get("validation_data_accessed") is not False
            or source_oof.get("passed") is not True
            or source_diagnostics.get("delta_hits025") != PRIOR_DELTA_025
            or source_diagnostics.get("delta_hits050") != PRIOR_DELTA_050
            or source_diagnostics.get("bootstrap025", {}).get(
                "lower_bound_95") != PRIOR_BOOTSTRAP_LOWER["025"]
            or source_diagnostics.get("bootstrap050", {}).get(
                "lower_bound_95") != PRIOR_BOOTSTRAP_LOWER["050"]
            or source_subgroups.get("regular", {}).get(
                "diagnostics", {}).get("bootstrap025", {}).get(
                    "lower_bound_95") != -1
            or any(value is not True
                   for value in source_oof.get("predicates", {}).values())):
        raise ValueError("V109 source success contract changed")

    fallback_path = Path(args.fallback_scenes).expanduser().resolve(strict=True)
    if file_sha256(fallback_path) != FALLBACK_MANIFEST_SHA256:
        raise ValueError("MeshSP fallback-scene manifest SHA-256 changed")
    fallback_scenes = {
        line.strip() for line in fallback_path.read_text(
            encoding="ascii"
        ).splitlines() if line.strip()
    }
    if len(fallback_scenes) != EXPECTED_FALLBACK_SCENE_COUNT:
        raise ValueError("MeshSP fallback-scene count changed")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    metadata_paths = {
        "candidate_receipt": (
            Path(args.base_cache).expanduser().resolve().parent
            / "candidate_train_receipt.json"
        ),
        "geometry_receipt": (
            Path(args.base_cache).expanduser().resolve().parent
            / "geometry_train_receipt.json"
        ),
        "base_manifest": (
            Path(args.base_cache).expanduser().resolve() / "manifest.json"
        ),
        "geometry_manifest": (
            Path(args.geometry_cache).expanduser().resolve() / "manifest.json"
        ),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    metadata_before = {
        name: file_sha256(path) for name, path in metadata_paths.items()
    }
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    split = split_residual_joined_rows(loaded["joined_rows"])
    joined_rows = loaded["joined_rows"]
    if len(split["fit_rows"]) + len(split["calibration_rows"]) != len(
            joined_rows):
        raise RuntimeError("historical split no longer reconstructs full train")
    scan_ids = [row["base"]["scan_id"] for row in joined_rows]
    if (len(joined_rows) != EXPECTED_SAMPLE_COUNT
            or len(set(scan_ids)) != EXPECTED_SCENE_COUNT):
        raise ValueError("V110 full-train identity changed")
    training_scenes = set(scan_ids)
    corrected_scenes = training_scenes & fallback_scenes
    regular_scenes = training_scenes - fallback_scenes
    if (len(corrected_scenes) != EXPECTED_CORRECTED_SCENE_COUNT
            or len(regular_scenes) != EXPECTED_REGULAR_SCENE_COUNT):
        raise ValueError("V110 subgroup partition changed")

    records = materialize_hierarchical_rows(
        joined_rows, loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    fold_ids = torch.tensor([
        scene_folds[record["scan_id"]] for record in records
    ], dtype=torch.long)
    fold_indices = {
        fold: torch.nonzero(fold_ids.eq(fold), as_tuple=False).flatten()
        for fold in range(5)
    }
    if torch.cat([fold_indices[fold] for fold in range(5)]).sort().values.tolist(
            ) != list(range(EXPECTED_SAMPLE_COUNT)):
        raise RuntimeError("V110 folds do not partition all rows")

    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    proposals = baselines.clone()
    mean_head_gain = torch.zeros(
        EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32
    )
    head_std = torch.zeros(EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32)
    consensus = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.float32)
    member_proposals = torch.zeros(
        EXPECTED_SAMPLE_COUNT, len(ENSEMBLE_SEEDS), dtype=torch.long
    )
    model_folds = []
    for held in range(5):
        train_indices = torch.cat([
            fold_indices[fold] for fold in range(5) if fold != held
        ]).sort().values.tolist()
        held_rows = fold_indices[held].tolist()
        train_scenes = {records[index]["scan_id"] for index in train_indices}
        held_scenes = {records[index]["scan_id"] for index in held_rows}
        if train_scenes & held_scenes:
            raise RuntimeError("scene leakage in V110 model fold")
        fold_records = [records[index] for index in train_indices]
        statistics = fit_hierarchical_normalization(fold_records)
        models = []
        member_rows = []
        for seed in ENSEMBLE_SEEDS:
            model, epochs = fit_seeded_model(
                fold_records, statistics, args.device, seed
            )
            member_row = {
                "seed": seed,
                "state_sha256": model_state_sha256(model),
                "final_epoch": epochs[-1],
            }
            member_rows.append(member_row)
            models.append(model)
            print(json.dumps({
                "completed_model_member": {
                    "held_fold": held,
                    **member_row,
                }
            }, sort_keys=True), flush=True)
        prediction = predict_uncertainty_ensemble(
            models, [records[index] for index in held_rows],
            statistics, args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_rows]):
            raise RuntimeError("V110 OOF baseline identity changed")
        proposals[held_rows] = prediction["proposals"]
        mean_head_gain[held_rows] = prediction["mean_head_gain"]
        head_std[held_rows] = prediction["head_std"]
        consensus[held_rows] = prediction["consensus"]
        member_proposals[held_rows] = prediction["member_proposals"]
        row = {
            "held_fold": held,
            "train_row_count": len(train_indices),
            "held_row_count": len(held_rows),
            "train_scene_count": len(train_scenes),
            "held_scene_count": len(held_scenes),
            "normalization_sha256": statistics["sha256"],
            "members": member_rows,
            "prediction_sha256": v99.tensor_sha256(
                prediction["proposals"], prediction["mean_head_gain"],
                prediction["head_std"], prediction["consensus"],
                prediction["member_proposals"],
            ),
        }
        model_folds.append(row)
        print(json.dumps({"completed_model_fold": row}, sort_keys=True), flush=True)
        del models
        torch.cuda.empty_cache()

    candidate_ious = torch.stack([
        record["candidate_ious"].reshape(-1) for record in records
    ])
    rows = torch.arange(EXPECTED_SAMPLE_COUNT)
    baseline_ious = candidate_ious[rows, baselines]
    proposal_ious = candidate_ious[rows, proposals]
    selected = baselines.clone()
    meta_folds = []
    for held in range(5):
        held_rows = fold_indices[held]
        calibration_rows = torch.cat([
            fold_indices[fold] for fold in range(5) if fold != held
        ]).sort().values
        selection = select_nested_policy(
            calibration_rows, proposals, baselines, mean_head_gain, head_std,
            consensus, baseline_ious, proposal_ious, fold_ids,
        )
        policy = selection["selected"]["policy"]
        accepted = policy_accept_mask(
            proposals[held_rows], baselines[held_rows],
            mean_head_gain[held_rows], head_std[held_rows],
            consensus[held_rows], policy,
        )
        selected[held_rows] = torch.where(
            accepted, proposals[held_rows], baselines[held_rows]
        )
        held_summary = policy_summary(
            held_rows, proposals, baselines, mean_head_gain, head_std,
            consensus, baseline_ious, proposal_ious, fold_ids, policy,
        )
        meta_row = {
            "held_fold": held,
            "policy_calibration_folds": [
                fold for fold in range(5) if fold != held
            ],
            "policy_calibration_row_count": int(calibration_rows.numel()),
            "selected_policy": copy.deepcopy(policy),
            "calibration_selection": selection,
            "held_result": held_summary,
        }
        meta_folds.append(meta_row)
        print(json.dumps({"completed_meta_fold": meta_row}, sort_keys=True), flush=True)

    diagnostics = v99.build_diagnostics(records, selected, baselines)
    subgroups = {}
    for name, scenes in {
            "corrected": corrected_scenes,
            "regular": regular_scenes}.items():
        indices = [
            index for index, record in enumerate(records)
            if record["scan_id"] in scenes
        ]
        subgroups[name] = {
            "scene_count": len(scenes),
            "row_count": len(indices),
            "scene_sha256": hashlib.sha256(
                ("\n".join(sorted(scenes)) + "\n").encode("ascii")
            ).hexdigest(),
            "diagnostics": v99.build_diagnostics(
                [records[index] for index in indices],
                selected[indices], baselines[indices],
            ),
        }
    predicates = acceptance_gate(diagnostics, subgroups, meta_folds)
    global_policy_selection = select_nested_policy(
        torch.arange(EXPECTED_SAMPLE_COUNT), proposals, baselines,
        mean_head_gain, head_std, consensus, baseline_ious, proposal_ious,
        fold_ids,
    )

    protected_after = capture_immutable_artifact_identities(protected_paths)
    metadata_after = {
        name: file_sha256(path) for name, path in metadata_paths.items()
    }
    if protected_after != protected_before or metadata_after != metadata_before:
        raise RuntimeError("V110 protected inputs changed during OOF")
    report = {
        "schema": "rec-v110-meshsp-uncertainty-ensemble-full-train-scene-oof-v1",
        "version": 1,
        "validation_data_accessed": False,
        "prior_calibration_used_for_selection": False,
        "deployable": False,
        "source": {
            "path": str(source_path),
            "sha256": V109_REPORT_SHA256,
            "passed_gate": copy.deepcopy(source_oof),
        },
        "protocol": {
            "architecture": (
                "three-seed contextual-hierarchy deep ensemble on MeshSP V108 "
                "features with cross-seed lower-confidence-bound switching"
            ),
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "ensemble_probability": "arithmetic_mean_of_monotone_probabilities",
            "uncertainty": "population_std_of_seed_head_gain",
            "consensus": "fraction_of_seed_proposals_equal_to_ensemble_proposal",
            "model_oof": "five_scene_disjoint_folds",
            "policy_oof": "leave_one_model_oof_fold_out_meta_calibration",
            "policy_grid": list(copy.deepcopy(POLICY_GRID)),
            "selection_objective": (
                "v109_scaled_floors_then_maximize_delta025_then_delta050_"
                "then_fewer_switches_then_conservatism"
            ),
            "held_labels_visible_to_policy_selection": False,
        },
        "model_folds": model_folds,
        "meta_folds": meta_folds,
        "global_policy_selection": global_policy_selection,
        "raw_prediction_sha256": v99.tensor_sha256(
            proposals, mean_head_gain, head_std, consensus, member_proposals
        ),
        "oof": {
            "diagnostics": diagnostics,
            "subgroups": subgroups,
            "required_delta_hits025": MIN_DELTA_025,
            "required_delta_hits050": MIN_DELTA_050,
            "prior_bootstrap_lower": copy.deepcopy(PRIOR_BOOTSTRAP_LOWER),
            "prior_subgroup_bootstrap_lower": copy.deepcopy(
                PRIOR_SUBGROUP_BOOTSTRAP_LOWER
            ),
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "historical_split_metadata": copy.deepcopy(split["metadata"]),
        "source_sha256": {
            "v110": file_sha256(Path(__file__).resolve()),
            "v109": file_sha256(
                Path(__file__).resolve().parent
                / "run_v109_meshsp_nested_policy_oof.py"
            ),
            "v99": file_sha256(Path(v99.__file__).resolve()),
            "v95": file_sha256(Path(v95.__file__).resolve()),
        },
        "fallback_scene_manifest": {
            "path": str(fallback_path),
            "sha256": FALLBACK_MANIFEST_SHA256,
            "scene_count": EXPECTED_FALLBACK_SCENE_COUNT,
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_metadata_before": metadata_before,
        "protected_metadata_after": metadata_after,
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
                raise OSError("V110 output write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "oof": report["oof"],
        "global_policy": global_policy_selection["selected"],
    }, sort_keys=True), flush=True)
    if not report["oof"]["passed"]:
        raise SystemExit(76)


if __name__ == "__main__":
    main()
