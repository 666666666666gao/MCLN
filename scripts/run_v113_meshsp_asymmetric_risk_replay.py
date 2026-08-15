#!/usr/bin/env python
"""CPU-only nested replay of asymmetric threshold risk from V112 cache."""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from models.rec_hierarchical_reranker import (
    VARIANT_COUNT,
    hierarchical_scene_clustered_hit_delta_bootstrap,
)
from scripts.run_v108_meshsp_pareto_oof import (
    EXPECTED_CORRECTED_SCENE_COUNT,
    EXPECTED_FALLBACK_SCENE_COUNT,
    EXPECTED_REGULAR_SCENE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SCENE_COUNT,
    FALLBACK_MANIFEST_SHA256,
    file_sha256,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    capture_immutable_artifact_identities,
)


V109_REPORT_SHA256 = (
    "37680aaa34757cf9bb2376e93629ae6b89aa6b8fac16960ac091305cc20146a1"
)
V112_REPORT_SHA256 = (
    "128ce636d27234db7fca4fb23bd5d30945928d9ac9dcd1cf8139c38670a41b96"
)
V112_CACHE_SHA256 = (
    "1123df3d312e433bf14b83874de99742906907738802bf878056ca07caa7ffdd"
)
ANCHOR_RAW_SHA256 = (
    "bdcc8c01aabf5fe891f7789ca630ea533209209090d80f02852ea9e66184a57d"
)
MARGINS = (0.10, 0.12, 0.13312220573425293, 0.15, 0.18, 0.22)
RISK_LAMBDA_025 = 0.5
RISK_LAMBDA_050 = 0.25
MIN_LCB_025 = 0.02
MIN_LCB_050 = 0.0
CALIBRATION_050_RETENTION = 0.95
MIN_DELTA_025 = 77
MIN_DELTA_050 = 235
MIN_BOOTSTRAP_LOWER = {"025": 40, "050": 180}
MIN_SUBGROUP_BOOTSTRAP_LOWER = {
    "corrected": {"025": 25, "050": 125},
    "regular": {"025": 1, "050": 35},
}
POLICY_GRID = tuple({
    "aggregate_lcb_margin": float(margin),
    "min_head_lcb025": MIN_LCB_025,
    "min_head_lcb050": MIN_LCB_050,
    "risk_lambda025": RISK_LAMBDA_025,
    "risk_lambda050": RISK_LAMBDA_050,
} for margin in MARGINS)
BASE_POLICY = {
    "aggregate_lcb_margin": 0.15,
    "min_head_gain025": 0.02,
    "min_head_gain050": 0.0,
}


def tensor_sha256(*values):
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("V113 output write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_vectors(data):
    required = (
        "scan_ids", "fold_ids", "baselines", "proposals",
        "anchor_head_gain", "head_risk", "anchor_agreement",
        "baseline_ious", "proposal_ious",
    )
    if any(key not in data for key in required):
        raise ValueError("V113 cache is missing a required vector")
    if any(len(data[key]) != EXPECTED_SAMPLE_COUNT for key in required):
        raise ValueError("V113 cache vector length changed")
    scan_ids = data["scan_ids"]
    if (any(not isinstance(value, str) or not value for value in scan_ids)
            or len(set(scan_ids)) != EXPECTED_SCENE_COUNT):
        raise ValueError("V113 scan identity changed")
    values = {
        "fold_ids": torch.tensor(data["fold_ids"], dtype=torch.long),
        "baselines": torch.tensor(data["baselines"], dtype=torch.long),
        "proposals": torch.tensor(data["proposals"], dtype=torch.long),
        "anchor_head_gain": torch.tensor(
            data["anchor_head_gain"], dtype=torch.float32
        ),
        "head_risk": torch.tensor(data["head_risk"], dtype=torch.float32),
        "anchor_agreement": torch.tensor(
            data["anchor_agreement"], dtype=torch.float32
        ),
        "baseline_ious": torch.tensor(
            data["baseline_ious"], dtype=torch.float32
        ),
        "proposal_ious": torch.tensor(
            data["proposal_ious"], dtype=torch.float32
        ),
    }
    if set(values["fold_ids"].tolist()) != set(range(5)):
        raise ValueError("V113 cache fold IDs changed")
    if (values["anchor_head_gain"].shape != (EXPECTED_SAMPLE_COUNT, 2)
            or values["head_risk"].shape != (EXPECTED_SAMPLE_COUNT, 2)):
        raise ValueError("V113 cache head tensor shape changed")
    floats = (
        values["anchor_head_gain"], values["head_risk"],
        values["anchor_agreement"], values["baseline_ious"],
        values["proposal_ious"],
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in floats):
        raise ValueError("V113 cache contains a non-finite value")
    if (bool(values["head_risk"].lt(0).any().item())
            or bool(values["anchor_agreement"].lt(0).any().item())
            or bool(values["anchor_agreement"].gt(1).any().item())):
        raise ValueError("V113 cache risk/agreement values are invalid")
    return scan_ids, values


def base_accept(values):
    gain = values["anchor_head_gain"]
    aggregate = 2.0 * gain[:, 0] + gain[:, 1]
    return (
        values["proposals"].ne(values["baselines"])
        & aggregate.ge(BASE_POLICY["aggregate_lcb_margin"])
        & gain[:, 0].gt(BASE_POLICY["min_head_gain025"])
        & gain[:, 1].gt(BASE_POLICY["min_head_gain050"])
    )


def policy_accept(values, policy):
    if policy not in POLICY_GRID:
        raise ValueError("V113 policy is outside the frozen grid")
    gain = values["anchor_head_gain"]
    risk = values["head_risk"]
    lcb025 = gain[:, 0] - policy["risk_lambda025"] * risk[:, 0]
    lcb050 = gain[:, 1] - policy["risk_lambda050"] * risk[:, 1]
    aggregate = 2.0 * lcb025 + lcb050
    return (
        values["proposals"].ne(values["baselines"])
        & aggregate.ge(policy["aggregate_lcb_margin"])
        & lcb025.gt(policy["min_head_lcb025"])
        & lcb050.gt(policy["min_head_lcb050"])
    )


def decision_summary(indices, accepted, values, policy):
    indices = torch.as_tensor(indices, dtype=torch.long)
    baseline = values["baseline_ious"][indices]
    selected = torch.where(
        accepted[indices], values["proposal_ious"][indices], baseline
    )
    delta025 = selected.gt(0.25).long() - baseline.gt(0.25).long()
    delta050 = selected.gt(0.50).long() - baseline.gt(0.50).long()
    folds = values["fold_ids"][indices]
    fold_deltas = {}
    for fold in sorted(set(folds.tolist())):
        mask = folds.eq(fold)
        fold_deltas[str(fold)] = {
            "hits025": int(delta025[mask].sum().item()),
            "hits050": int(delta050[mask].sum().item()),
        }
    return {
        "policy": copy.deepcopy(policy),
        "row_count": int(indices.numel()),
        "switches": int(accepted[indices].sum().item()),
        "delta_hits025": int(delta025.sum().item()),
        "delta_hits050": int(delta050.sum().item()),
        "fold_deltas": fold_deltas,
    }


def _rank(summary):
    folds = list(summary["fold_deltas"].values())
    return (
        min(row["hits025"] for row in folds),
        summary["delta_hits025"],
        min(row["hits050"] for row in folds),
        summary["delta_hits050"],
        -summary["switches"],
        summary["policy"]["aggregate_lcb_margin"],
    )


def select_policy(indices, values, base_mask, policy_masks):
    reference = decision_summary(indices, base_mask, values, BASE_POLICY)
    minimum_delta050 = math.ceil(
        CALIBRATION_050_RETENTION * reference["delta_hits050"]
    )
    candidates = []
    for policy, accepted in zip(POLICY_GRID, policy_masks):
        summary = decision_summary(indices, accepted, values, policy)
        summary["eligible"] = bool(
            summary["delta_hits025"] > reference["delta_hits025"]
            and summary["delta_hits050"] >= minimum_delta050
            and all(row["hits025"] >= 0 and row["hits050"] >= 0
                    for row in summary["fold_deltas"].values())
        )
        candidates.append(summary)
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("V113 frozen margin grid has no eligible policy")
    winner = max(eligible, key=_rank)
    return {
        "selection_rule": (
            "fixed_asymmetric_risk_family;strictly_improve_anchor025;"
            "retain_95_percent_anchor050;maximize_min_fold025_then_"
            "pooled025_then_min_fold050_then_pooled050"
        ),
        "reference": reference,
        "minimum_delta050": minimum_delta050,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected": copy.deepcopy(winner),
        "candidates": candidates,
    }


def build_diagnostics(indices, accepted, scan_ids, values):
    indices = torch.as_tensor(indices, dtype=torch.long)
    baseline_ious = values["baseline_ious"][indices]
    selected_ious = torch.where(
        accepted[indices], values["proposal_ious"][indices], baseline_ious
    )
    baseline025 = baseline_ious.gt(0.25).long()
    baseline050 = baseline_ious.gt(0.50).long()
    proposed025 = selected_ious.gt(0.25).long()
    proposed050 = selected_ious.gt(0.50).long()
    local_scan_ids = [scan_ids[index] for index in indices.tolist()]
    bootstrap025 = hierarchical_scene_clustered_hit_delta_bootstrap(
        local_scan_ids, baseline025.tolist(), proposed025.tolist()
    )
    bootstrap050 = hierarchical_scene_clustered_hit_delta_bootstrap(
        local_scan_ids, baseline050.tolist(), proposed050.tolist()
    )
    folds = values["fold_ids"][indices]
    fold_deltas = {}
    for fold in sorted(set(folds.tolist())):
        mask = folds.eq(fold)
        fold_deltas[str(fold)] = {
            "hits025": int((proposed025[mask] - baseline025[mask]).sum()),
            "hits050": int((proposed050[mask] - baseline050[mask]).sum()),
        }

    def effects(base, proposed):
        switched = accepted[indices]
        return {
            "fixes": int((switched & base.eq(0) & proposed.eq(1)).sum()),
            "breaks": int((switched & base.eq(1) & proposed.eq(0)).sum()),
            "neutral_switches": int((switched & base.eq(proposed)).sum()),
            "kept_correct": int((~switched & base.eq(1)).sum()),
            "kept_wrong": int((~switched & base.eq(0)).sum()),
        }

    switches = int(accepted[indices].sum().item())
    return {
        "sample_count": int(indices.numel()),
        "switches": switches,
        "abstentions": int(indices.numel()) - switches,
        "switch_rate": switches / float(indices.numel()),
        "baseline": {
            "0.25": {"hits": int(baseline025.sum())},
            "0.50": {"hits": int(baseline050.sum())},
        },
        "proposed": {
            "0.25": {"hits": int(proposed025.sum())},
            "0.50": {"hits": int(proposed050.sum())},
        },
        "delta_hits025": int(proposed025.sum() - baseline025.sum()),
        "delta_hits050": int(proposed050.sum() - baseline050.sum()),
        "fold_deltas": fold_deltas,
        "bootstrap025": bootstrap025,
        "bootstrap050": bootstrap050,
        "effects": {
            "0.25": effects(baseline025, proposed025),
            "0.50": effects(baseline050, proposed050),
        },
    }


def acceptance_gate(diagnostics, subgroups, meta_folds, deployment):
    return {
        "delta025_at_least_frozen_floor": (
            diagnostics["delta_hits025"] >= MIN_DELTA_025
        ),
        "delta050_at_least_frozen_floor": (
            diagnostics["delta_hits050"] >= MIN_DELTA_050
        ),
        "all_folds_strictly_positive025": all(
            row["hits025"] > 0
            for row in diagnostics["fold_deltas"].values()
        ),
        "all_folds_strictly_positive050": all(
            row["hits050"] > 0
            for row in diagnostics["fold_deltas"].values()
        ),
        "bootstrap025_lower_at_least_frozen_floor": (
            diagnostics["bootstrap025"]["lower_bound_95"]
            >= MIN_BOOTSTRAP_LOWER["025"]
        ),
        "bootstrap050_lower_at_least_frozen_floor": (
            diagnostics["bootstrap050"]["lower_bound_95"]
            >= MIN_BOOTSTRAP_LOWER["050"]
        ),
        "corrected_bootstrap025_lower_at_least_frozen_floor": (
            subgroups["corrected"]["diagnostics"]["bootstrap025"][
                "lower_bound_95"
            ] >= MIN_SUBGROUP_BOOTSTRAP_LOWER["corrected"]["025"]
        ),
        "corrected_bootstrap050_lower_at_least_frozen_floor": (
            subgroups["corrected"]["diagnostics"]["bootstrap050"][
                "lower_bound_95"
            ] >= MIN_SUBGROUP_BOOTSTRAP_LOWER["corrected"]["050"]
        ),
        "regular_bootstrap025_lower_at_least_frozen_floor": (
            subgroups["regular"]["diagnostics"]["bootstrap025"][
                "lower_bound_95"
            ] >= MIN_SUBGROUP_BOOTSTRAP_LOWER["regular"]["025"]
        ),
        "regular_bootstrap050_lower_at_least_frozen_floor": (
            subgroups["regular"]["diagnostics"]["bootstrap050"][
                "lower_bound_95"
            ] >= MIN_SUBGROUP_BOOTSTRAP_LOWER["regular"]["050"]
        ),
        "all_meta_margin_selections_eligible": all(
            row["selection"]["eligible_candidate_count"] > 0
            for row in meta_folds
        ),
        "deployment_margin_has_strict_majority": (
            deployment["winning_votes"] >= 3
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v109-report", required=True)
    parser.add_argument("--v112-report", required=True)
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--fallback-scenes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(str(output))

    v109_path = Path(args.v109_report).expanduser().resolve(strict=True)
    v112_path = Path(args.v112_report).expanduser().resolve(strict=True)
    cache_path = Path(args.prediction_cache).expanduser().resolve(strict=True)
    fallback_path = Path(args.fallback_scenes).expanduser().resolve(strict=True)
    if file_sha256(v109_path) != V109_REPORT_SHA256:
        raise ValueError("V109 source report SHA-256 changed")
    if file_sha256(v112_path) != V112_REPORT_SHA256:
        raise ValueError("V112 source report SHA-256 changed")
    if file_sha256(cache_path) != V112_CACHE_SHA256:
        raise ValueError("V112 prediction-cache SHA-256 changed")
    if file_sha256(fallback_path) != FALLBACK_MANIFEST_SHA256:
        raise ValueError("fallback-scene manifest SHA-256 changed")
    v109 = json.loads(v109_path.read_text(encoding="ascii"))
    v112 = json.loads(v112_path.read_text(encoding="ascii"))
    if (v109.get("validation_data_accessed") is not False
            or v109.get("oof", {}).get("passed") is not True
            or v109.get("raw_prediction_sha256") != ANCHOR_RAW_SHA256):
        raise ValueError("V109 source success contract changed")
    if (v112.get("validation_data_accessed") is not False
            or v112.get("oof", {}).get("passed") is not False
            or v112.get("oof", {}).get("diagnostics", {}).get(
                "delta_hits025") != 74
            or v112.get("oof", {}).get("diagnostics", {}).get(
                "delta_hits050") != 231
            or [key for key, value in v112.get("oof", {}).get(
                "predicates", {}).items() if not value] != [
                    "regular_bootstrap025_lower_strictly_positive"]
            or v112.get("prediction_cache", {}).get(
                "sha256") != V112_CACHE_SHA256):
        raise ValueError("V112 source near-pass contract changed")
    cache = json.loads(gzip.decompress(cache_path.read_bytes()))
    if (cache.get("validation_data_accessed") is not False
            or cache.get("train_labels_only") is not True
            or cache.get("anchor_raw_prediction_sha256") != ANCHOR_RAW_SHA256
            or cache.get("schema")
            != "rec-v112-anchor-committee-train-oof-prediction-cache-v1"):
        raise ValueError("V112 prediction-cache contract changed")
    scan_ids, values = _validate_vectors(cache)
    base_mask = base_accept(values)
    anchor_aggregate = (
        2.0 * values["anchor_head_gain"][:, 0]
        + values["anchor_head_gain"][:, 1]
    )
    anchor_sha = tensor_sha256(
        values["proposals"], anchor_aggregate, values["anchor_head_gain"]
    )
    if anchor_sha != ANCHOR_RAW_SHA256:
        raise RuntimeError("V113 cache does not reproduce V109 anchor SHA")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    policy_masks = [policy_accept(values, policy) for policy in POLICY_GRID]
    fold_indices = {
        fold: torch.nonzero(
            values["fold_ids"].eq(fold), as_tuple=False
        ).flatten() for fold in range(5)
    }
    selected = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.bool)
    meta_folds = []
    for held in range(5):
        calibration = torch.cat([
            fold_indices[fold] for fold in range(5) if fold != held
        ]).sort().values
        selection = select_policy(
            calibration, values, base_mask, policy_masks
        )
        policy = selection["selected"]["policy"]
        policy_index = POLICY_GRID.index(policy)
        held_indices = fold_indices[held]
        selected[held_indices] = policy_masks[policy_index][held_indices]
        meta_folds.append({
            "held_fold": held,
            "selected_policy": copy.deepcopy(policy),
            "selection": selection,
            "held_result": decision_summary(
                held_indices, policy_masks[policy_index], values, policy
            ),
        })

    margins = [
        row["selected_policy"]["aggregate_lcb_margin"]
        for row in meta_folds
    ]
    margin_counts = {
        str(margin): margins.count(margin) for margin in sorted(set(margins))
    }
    deployment_margin = max(
        set(margins), key=lambda margin: (margins.count(margin), margin)
    )
    deployment = {
        "rule": "meta_fold_majority_margin_ties_choose_higher_margin",
        "margin_counts": margin_counts,
        "winning_votes": margins.count(deployment_margin),
        "policy": copy.deepcopy(next(
            policy for policy in POLICY_GRID
            if policy["aggregate_lcb_margin"] == deployment_margin
        )),
    }
    all_indices = torch.arange(EXPECTED_SAMPLE_COUNT)
    diagnostics = build_diagnostics(
        all_indices, selected, scan_ids, values
    )
    fallback_scenes = set(fallback_path.read_text(encoding="ascii").split())
    training_scenes = set(scan_ids)
    corrected_scenes = training_scenes & fallback_scenes
    regular_scenes = training_scenes - fallback_scenes
    if (len(fallback_scenes) != EXPECTED_FALLBACK_SCENE_COUNT
            or len(corrected_scenes) != EXPECTED_CORRECTED_SCENE_COUNT
            or len(regular_scenes) != EXPECTED_REGULAR_SCENE_COUNT):
        raise ValueError("V113 subgroup scene partition changed")
    subgroups = {}
    for name, scenes in {
            "corrected": corrected_scenes,
            "regular": regular_scenes}.items():
        indices = torch.tensor([
            index for index, scan_id in enumerate(scan_ids)
            if scan_id in scenes
        ], dtype=torch.long)
        subgroups[name] = {
            "scene_count": len(scenes),
            "row_count": int(indices.numel()),
            "scene_sha256": hashlib.sha256(
                ("\n".join(sorted(scenes)) + "\n").encode("ascii")
            ).hexdigest(),
            "diagnostics": build_diagnostics(
                indices, selected, scan_ids, values
            ),
        }
    predicates = acceptance_gate(
        diagnostics, subgroups, meta_folds, deployment
    )
    global_selection = select_policy(
        all_indices, values, base_mask, policy_masks
    )
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("V113 protected artifacts changed during replay")
    report = {
        "schema": "rec-v113-meshsp-asymmetric-risk-cache-replay-v1",
        "version": 1,
        "validation_data_accessed": False,
        "prior_train_oof_used_for_protocol_design": True,
        "deployable": False,
        "source": {
            "v109": {"path": str(v109_path), "sha256": V109_REPORT_SHA256},
            "v112": {"path": str(v112_path), "sha256": V112_REPORT_SHA256},
            "prediction_cache": {
                "path": str(cache_path), "sha256": V112_CACHE_SHA256,
            },
        },
        "protocol": {
            "architecture": "V109 anchor with asymmetric two-head seed risk",
            "risk_lambda025": RISK_LAMBDA_025,
            "risk_lambda050": RISK_LAMBDA_050,
            "minimum_lcb025": MIN_LCB_025,
            "minimum_lcb050": MIN_LCB_050,
            "margin_grid": list(MARGINS),
            "calibration_delta050_retention": CALIBRATION_050_RETENTION,
            "model_oof": "replay_exact_v112_five_scene_disjoint_predictions",
            "policy_oof": "leave_one_scene_oof_fold_out_margin_calibration",
            "held_labels_visible_to_margin_selection": False,
        },
        "anchor_raw_prediction_sha256": anchor_sha,
        "meta_folds": meta_folds,
        "deployment_policy": deployment,
        "global_policy_selection": global_selection,
        "nested_selection_sha256": tensor_sha256(selected.long()),
        "oof": {
            "diagnostics": diagnostics,
            "subgroups": subgroups,
            "required_delta_hits025": MIN_DELTA_025,
            "required_delta_hits050": MIN_DELTA_050,
            "required_bootstrap_lower": copy.deepcopy(MIN_BOOTSTRAP_LOWER),
            "required_subgroup_bootstrap_lower": copy.deepcopy(
                MIN_SUBGROUP_BOOTSTRAP_LOWER
            ),
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "source_sha256": {
            "v113": file_sha256(Path(__file__).resolve()),
            "v112": file_sha256(
                Path(__file__).resolve().parent
                / "run_v112_meshsp_anchor_committee_tradeoff_oof.py"
            ),
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    payload = json.dumps(
        report, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    write_exclusive(output, payload)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "oof": report["oof"],
        "deployment_policy": deployment,
    }, sort_keys=True), flush=True)
    if not report["oof"]["passed"]:
        raise SystemExit(76)


if __name__ == "__main__":
    main()
