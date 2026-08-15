#!/usr/bin/env python
"""Train-only nested OOF audit of an anchor-proposal seed committee."""

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import torch

import scripts.run_v95_threshold_aligned_listwise_hierarchical as v95
import scripts.run_v99_pareto_contextual_hierarchical as v99
from models.rec_hierarchical_reranker import (
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
V110_REPORT_SHA256 = (
    "7970a54bbf8a26ca09370be6a2413e436dbcb408dbc1dc1eec0a162ee40f8d48"
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
MARGINS = (0.10, 0.12, float(v99.V97_MARGIN), 0.15, 0.18, 0.22)
MIN_LCB_025 = (0.0, 0.0025, 0.005, 0.01, 0.02)
UNCERTAINTY_PENALTIES = (0.0, 0.5, 1.0, 2.0, 4.0)
POLICY_GRID = tuple(
    {
        "aggregate_lcb_margin": float(margin),
        "min_head_lcb025": float(minimum),
        "min_head_lcb050": 0.0,
        "uncertainty_penalty": float(penalty),
    }
    for margin in MARGINS
    for minimum in MIN_LCB_025
    for penalty in UNCERTAINTY_PENALTIES
)
BASE_POLICY = {
    "aggregate_lcb_margin": 0.15,
    "min_head_lcb025": 0.02,
    "min_head_lcb050": 0.0,
    "uncertainty_penalty": 0.0,
}


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
        raise ValueError("V111 seed is outside the frozen committee")
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
        proposals, baselines, anchor_head_gain, head_risk):
    if (proposals.dtype != torch.long or baselines.dtype != torch.long
            or proposals.shape != baselines.shape or proposals.dim() != 1):
        raise ValueError("V111 proposals and baselines must align as int64")
    expected_heads = proposals.shape + (2,)
    if (anchor_head_gain.dtype != torch.float32
            or anchor_head_gain.shape != expected_heads
            or head_risk.dtype != torch.float32
            or head_risk.shape != expected_heads):
        raise ValueError("V111 anchor-risk tensors have invalid shape or dtype")
    if (not bool(torch.isfinite(anchor_head_gain).all().item())
            or not bool(torch.isfinite(head_risk).all().item())
            or bool(head_risk.lt(0.0).any().item())):
        raise ValueError("V111 anchor-risk tensors are invalid")


def policy_accept_mask(
        proposals, baselines, anchor_head_gain, head_risk, policy):
    if policy not in POLICY_GRID:
        raise ValueError("V111 policy is outside the frozen grid")
    _validate_policy_inputs(
        proposals, baselines, anchor_head_gain, head_risk
    )
    head_lcb = (
        anchor_head_gain - policy["uncertainty_penalty"] * head_risk
    )
    aggregate_lcb = 2.0 * head_lcb[:, 0] + head_lcb[:, 1]
    return (
        proposals.ne(baselines)
        & aggregate_lcb.ge(policy["aggregate_lcb_margin"])
        & head_lcb[:, 0].gt(policy["min_head_lcb025"])
        & head_lcb[:, 1].gt(policy["min_head_lcb050"])
    )


def policy_summary(
        indices, proposals, baselines, anchor_head_gain, head_risk,
        baseline_ious, proposal_ious, fold_ids, policy):
    indices = torch.as_tensor(indices, dtype=torch.long)
    accepted = policy_accept_mask(
        proposals[indices], baselines[indices], anchor_head_gain[indices],
        head_risk[indices], policy,
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
    folds = list(row["fold_deltas"].values())
    return (
        min(fold["hits025"] for fold in folds),
        row["delta_hits025"],
        min(fold["hits050"] for fold in folds),
        row["delta_hits050"],
        -row["switches"],
        policy["uncertainty_penalty"],
        policy["min_head_lcb025"],
        policy["aggregate_lcb_margin"],
    )


def select_nested_policy(
        indices, proposals, baselines, anchor_head_gain, head_risk,
        baseline_ious, proposal_ious, fold_ids):
    indices = torch.as_tensor(indices, dtype=torch.long)
    reference = policy_summary(
        indices, proposals, baselines, anchor_head_gain, head_risk,
        baseline_ious, proposal_ious, fold_ids, BASE_POLICY,
    )
    candidates = []
    for policy in POLICY_GRID:
        summary = policy_summary(
            indices, proposals, baselines, anchor_head_gain, head_risk,
            baseline_ious, proposal_ious, fold_ids, policy,
        )
        summary["strict_improvement_eligible"] = bool(
            policy != BASE_POLICY
            and summary["delta_hits025"] > reference["delta_hits025"]
            and summary["delta_hits050"] >= reference["delta_hits050"]
            and all(row["hits025"] >= 0 and row["hits050"] >= 0
                    for row in summary["fold_deltas"].values())
        )
        candidates.append(summary)
    eligible = [
        row for row in candidates if row["strict_improvement_eligible"]
    ]
    winner = max(eligible, key=_policy_rank) if eligible else reference
    return {
        "selection_rule": (
            "strictly_improve_v109_anchor_delta025_without_reducing_delta050;"
            "require_nonnegative_source_folds;maximize_min_fold025_then_"
            "pooled025_then_min_fold050_then_pooled050_then_conservatism"
        ),
        "reference": copy.deepcopy(reference),
        "candidate_count": len(candidates),
        "strict_improvement_candidate_count": len(eligible),
        "selected_strict_improvement": bool(eligible),
        "selected": copy.deepcopy(winner),
        "candidates": candidates,
    }


def predict_anchor_committee(models, records, statistics, device):
    if len(models) != len(ENSEMBLE_SEEDS):
        raise ValueError("V111 requires exactly three committee members")
    resolved = torch.device(device)
    for model in models:
        model.to(resolved).eval()
    all_proposals = []
    all_baselines = []
    all_anchor_head_gain = []
    all_head_risk = []
    all_anchor_agreement = []
    all_member_proposals = []
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            variant_probabilities = []
            member_proposals = []
            for model in models:
                outputs = model(**batch)
                variant_probabilities.append(monotone_hit_probabilities(
                    outputs["variant_logits"]
                ))
                member_proposals.append(select_hierarchical_proposal(
                    outputs["query_logits"], outputs["variant_logits"],
                    batch["query_valid"], batch["variant_valid"],
                )["flat_indices"])
            rows = torch.arange(len(row_batch), device=resolved)
            member_proposals = torch.stack(member_proposals, dim=0)
            proposals = member_proposals[0]
            baselines = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=resolved)
            anchor_agreement = member_proposals.eq(
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
            anchor_head_gain = member_head_gain[0]
            head_risk = (
                (member_head_gain - anchor_head_gain.unsqueeze(0)).square()
                .mean(dim=0).sqrt()
            )
            all_proposals.append(proposals.cpu().long())
            all_baselines.append(baselines.cpu().long())
            all_anchor_head_gain.append(anchor_head_gain.cpu().float())
            all_head_risk.append(head_risk.cpu().float())
            all_anchor_agreement.append(anchor_agreement.cpu().float())
            all_member_proposals.append(
                member_proposals.transpose(0, 1).cpu().long()
            )
    result = {
        "proposals": torch.cat(all_proposals),
        "baselines": torch.cat(all_baselines),
        "anchor_head_gain": torch.cat(all_anchor_head_gain),
        "head_risk": torch.cat(all_head_risk),
        "anchor_agreement": torch.cat(all_anchor_agreement),
        "member_proposals": torch.cat(all_member_proposals),
    }
    _validate_policy_inputs(
        result["proposals"], result["baselines"],
        result["anchor_head_gain"], result["head_risk"],
    )
    if (not bool(torch.isfinite(result["anchor_agreement"]).all().item())
            or bool(result["anchor_agreement"].lt(0.0).any().item())
            or bool(result["anchor_agreement"].gt(1.0).any().item())):
        raise ValueError("V111 anchor-agreement tensor is invalid")
    return result


def acceptance_gate(diagnostics, subgroups, meta_folds, anchor_contract):
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
        "all_meta_selected_strict_improvement": all(
            row["calibration_selection"]["selected_strict_improvement"]
            for row in meta_folds
        ),
        "meta_policy_consistent": all(
            policy == policies[0] for policy in policies[1:]
        ),
        "anchor_raw_prediction_exactly_matches_v109": (
            anchor_contract["raw_prediction_sha256_match"]
        ),
        "anchor_base_policy_exactly_matches_v109": (
            anchor_contract["base_policy_delta_match"]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v109-report", required=True)
    parser.add_argument("--v110-report", required=True)
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

    v109_path = Path(args.v109_report).expanduser().resolve(strict=True)
    if file_sha256(v109_path) != V109_REPORT_SHA256:
        raise ValueError("V109 source report SHA-256 changed")
    v109_source = json.loads(v109_path.read_text(encoding="ascii"))
    source_oof = v109_source.get("oof", {})
    source_diagnostics = source_oof.get("diagnostics", {})
    source_subgroups = source_oof.get("subgroups", {})
    if (v109_source.get("validation_data_accessed") is not False
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

    v110_path = Path(args.v110_report).expanduser().resolve(strict=True)
    if file_sha256(v110_path) != V110_REPORT_SHA256:
        raise ValueError("V110 source report SHA-256 changed")
    v110_source = json.loads(v110_path.read_text(encoding="ascii"))
    v110_oof = v110_source.get("oof", {})
    if (v110_source.get("validation_data_accessed") is not False
            or v110_oof.get("passed") is not False
            or v110_oof.get("diagnostics", {}).get("delta_hits025") != 44
            or v110_oof.get("diagnostics", {}).get("delta_hits050") != 128
            or v110_oof.get("diagnostics", {}).get("fold_deltas", {}).get(
                "4", {}).get("hits025") != -1):
        raise ValueError("V110 source failure contract changed")

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
        raise ValueError("V111 full-train identity changed")
    training_scenes = set(scan_ids)
    corrected_scenes = training_scenes & fallback_scenes
    regular_scenes = training_scenes - fallback_scenes
    if (len(corrected_scenes) != EXPECTED_CORRECTED_SCENE_COUNT
            or len(regular_scenes) != EXPECTED_REGULAR_SCENE_COUNT):
        raise ValueError("V111 subgroup partition changed")

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
        raise RuntimeError("V111 folds do not partition all rows")

    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    proposals = baselines.clone()
    anchor_head_gain = torch.zeros(
        EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32
    )
    head_risk = torch.zeros(EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32)
    anchor_agreement = torch.zeros(
        EXPECTED_SAMPLE_COUNT, dtype=torch.float32
    )
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
            raise RuntimeError("scene leakage in V111 model fold")
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
        prediction = predict_anchor_committee(
            models, [records[index] for index in held_rows],
            statistics, args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_rows]):
            raise RuntimeError("V111 OOF baseline identity changed")
        proposals[held_rows] = prediction["proposals"]
        anchor_head_gain[held_rows] = prediction["anchor_head_gain"]
        head_risk[held_rows] = prediction["head_risk"]
        anchor_agreement[held_rows] = prediction["anchor_agreement"]
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
                prediction["proposals"], prediction["anchor_head_gain"],
                prediction["head_risk"], prediction["anchor_agreement"],
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
    anchor_aggregate_gain = (
        2.0 * anchor_head_gain[:, 0] + anchor_head_gain[:, 1]
    )
    anchor_raw_sha256 = v99.tensor_sha256(
        proposals, anchor_aggregate_gain, anchor_head_gain
    )
    anchor_base_summary = policy_summary(
        torch.arange(EXPECTED_SAMPLE_COUNT), proposals, baselines,
        anchor_head_gain, head_risk, baseline_ious, proposal_ious,
        fold_ids, BASE_POLICY,
    )
    anchor_contract = {
        "expected_raw_prediction_sha256": v109_source[
            "raw_prediction_sha256"
        ],
        "actual_raw_prediction_sha256": anchor_raw_sha256,
        "raw_prediction_sha256_match": (
            anchor_raw_sha256 == v109_source["raw_prediction_sha256"]
        ),
        "expected_delta_hits025": PRIOR_DELTA_025,
        "expected_delta_hits050": PRIOR_DELTA_050,
        "actual_base_policy": copy.deepcopy(anchor_base_summary),
        "base_policy_delta_match": (
            anchor_base_summary["delta_hits025"] == PRIOR_DELTA_025
            and anchor_base_summary["delta_hits050"] == PRIOR_DELTA_050
        ),
    }
    if (not anchor_contract["raw_prediction_sha256_match"]
            or not anchor_contract["base_policy_delta_match"]):
        raise RuntimeError("V111 seed-0 anchor does not exactly reproduce V109")
    selected = baselines.clone()
    meta_folds = []
    for held in range(5):
        held_rows = fold_indices[held]
        calibration_rows = torch.cat([
            fold_indices[fold] for fold in range(5) if fold != held
        ]).sort().values
        selection = select_nested_policy(
            calibration_rows, proposals, baselines, anchor_head_gain,
            head_risk, baseline_ious, proposal_ious, fold_ids,
        )
        policy = selection["selected"]["policy"]
        accepted = policy_accept_mask(
            proposals[held_rows], baselines[held_rows],
            anchor_head_gain[held_rows], head_risk[held_rows], policy,
        )
        selected[held_rows] = torch.where(
            accepted, proposals[held_rows], baselines[held_rows]
        )
        held_summary = policy_summary(
            held_rows, proposals, baselines, anchor_head_gain, head_risk,
            baseline_ious, proposal_ious, fold_ids, policy,
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
    predicates = acceptance_gate(
        diagnostics, subgroups, meta_folds, anchor_contract
    )
    global_policy_selection = select_nested_policy(
        torch.arange(EXPECTED_SAMPLE_COUNT), proposals, baselines,
        anchor_head_gain, head_risk, baseline_ious, proposal_ious, fold_ids,
    )

    protected_after = capture_immutable_artifact_identities(protected_paths)
    metadata_after = {
        name: file_sha256(path) for name, path in metadata_paths.items()
    }
    if protected_after != protected_before or metadata_after != metadata_before:
        raise RuntimeError("V111 protected inputs changed during OOF")
    report = {
        "schema": "rec-v111-meshsp-anchor-committee-full-train-scene-oof-v1",
        "version": 1,
        "validation_data_accessed": False,
        "prior_calibration_used_for_selection": False,
        "deployable": False,
        "source": {
            "v109": {
                "path": str(v109_path),
                "sha256": V109_REPORT_SHA256,
                "passed_gate": copy.deepcopy(source_oof),
            },
            "v110": {
                "path": str(v110_path),
                "sha256": V110_REPORT_SHA256,
                "failed_gate": copy.deepcopy(v110_oof),
            },
        },
        "protocol": {
            "architecture": (
                "V109 seed-0 anchor proposal with a three-seed contextual-"
                "hierarchy risk committee on MeshSP V108 features"
            ),
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "proposal": "seed0_anchor_exactly_reproducing_v109",
            "uncertainty": (
                "root_mean_square_member_head_gain_disagreement_from_seed0_"
                "for_the_fixed_anchor_proposal"
            ),
            "lambda_zero_contract": "exact_v109_proposal_gain_and_decision",
            "model_oof": "five_scene_disjoint_folds",
            "policy_oof": "leave_one_model_oof_fold_out_meta_calibration",
            "policy_grid": list(copy.deepcopy(POLICY_GRID)),
            "selection_objective": (
                "strictly_improve_anchor_delta025_without_reducing_delta050;"
                "maximize_min_fold025_then_pooled025_then_min_fold050_then_"
                "pooled050_then_conservatism"
            ),
            "held_labels_visible_to_policy_selection": False,
        },
        "model_folds": model_folds,
        "meta_folds": meta_folds,
        "anchor_contract": anchor_contract,
        "global_policy_selection": global_policy_selection,
        "raw_prediction_sha256": v99.tensor_sha256(
            proposals, anchor_head_gain, head_risk, anchor_agreement,
            member_proposals
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
            "v111": file_sha256(Path(__file__).resolve()),
            "v110": file_sha256(
                Path(__file__).resolve().parent
                / "run_v110_meshsp_uncertainty_ensemble_oof.py"
            ),
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
                raise OSError("V111 output write made no progress")
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
