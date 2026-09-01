#!/usr/bin/env python3
"""Audit V135 relation-counterfactual deployment and mining contracts."""

from __future__ import print_function

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from v133_receipt_utils import atomic_write_new_json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.sacr_relation_counterfactual import (  # noqa: E402
    RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES,
    apply_relation_counterfactual_refinement,
    compute_relation_counterfactual_loss,
)


SCHEMA = "mcln-v135-relation-counterfactual-contract-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(repo_root):
    paths = (
        "main_utils.py",
        "train_dist_mod.py",
        "models/losses.py",
        "models/mcln.py",
        "models/sacr_head.py",
        "models/sacr_relation_counterfactual.py",
        "src/grounding_evaluator.py",
        "scripts/audit_v135_relation_counterfactual_contract.py",
        "scripts/audit_v135_smoke_gate.py",
        "scripts/run_v135_relation_counterfactual.sh",
        "scripts/v133_receipt_utils.py",
        "tests/test_sacr_relation_counterfactual.py",
    )
    return {path: sha256(repo_root / path) for path in paths}


def fixture_inputs(relation_scores):
    return {
        "relation_scores": relation_scores,
        "geometry_signatures": torch.tensor([[
            [1.0] * 11, [0.0] * 11, [-1.0] * 11, [1.0] * 11,
        ]]),
        "relation_candidate_mask": torch.ones((1, 4), dtype=torch.bool),
        "target_affinity": torch.tensor([[0.78, 0.80, 0.77, 0.10]]),
        "attribute_affinity": torch.tensor([[0.48, 0.50, 0.52, 0.10]]),
        "attribute_present": torch.ones(1, dtype=torch.bool),
        "parent_scores": torch.tensor([[1.0, 0.90, 0.85, 0.80]]),
        "candidate_valid": torch.ones((1, 4), dtype=torch.bool),
        "structured_valid_mask": torch.ones(1, dtype=torch.bool),
    }


def audit_deployment():
    values = fixture_inputs(torch.tensor([[0.0, 0.4, -0.3, 0.8]]))
    output = apply_relation_counterfactual_refinement(
        **values,
        parse_confidence=torch.ones(1),
        anchor_top1_mass=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
        parent_top_k=4,
        target_tolerance=0.05,
        attribute_tolerance=0.05,
        relation_scale=4.0,
        deployment_threshold=0.05,
    )
    require(output["parent_indices"].tolist() == [0],
            "V135 selected the wrong parent")
    require(output["proposal_mask"].tolist()
            == [[False, True, True, False]],
            "V135 class/attribute/parent shortlist changed")
    require(output["promotion_mask"].tolist()
            == [[False, True, False, False]],
            "V135 relation-supported promotion changed")
    require(output["scores"][0, 0] == values["parent_scores"][0, 0],
            "V135 changed the parent score")
    require(float(output["residual"].abs().max()) <= 0.25,
            "V135 exceeded the residual bound")
    return {
        "parent_exact": True,
        "proposal_mask": output["proposal_mask"].tolist(),
        "promotion_mask": output["promotion_mask"].tolist(),
        "residual_abs_max": float(output["residual"].abs().max()),
    }


def audit_objective():
    relation_scores = torch.tensor(
        [[0.5, 0.1, -0.2, 0.9]], requires_grad=True
    )
    values = fixture_inputs(relation_scores)
    result = compute_relation_counterfactual_loss(
        **values,
        box_ious=torch.tensor([[0.10, 0.70, 0.20, 0.05]]),
        parent_top_k=4,
        target_tolerance=0.05,
        attribute_tolerance=0.05,
        geometry_threshold=0.08,
        iou_gap=0.10,
        pair_margin=0.25,
        max_negatives=4,
    )
    require(result["hard_negative_count_mean"] == 2.0,
            "V135 did not mine the expected target swaps")
    require(result["parent_hard_negative_ratio"] == 1.0,
            "V135 did not include the high-parent geometry negative")
    require(torch.isfinite(result["loss"]) and result["loss"] > 0.0,
            "V135 objective is inactive/non-finite")
    result["loss"].backward()
    require(torch.isfinite(relation_scores.grad).all(),
            "V135 produced non-finite gradients")

    extreme = torch.tensor([[1e30, -1e30, 1e30, -1e30]], requires_grad=True)
    extreme_result = compute_relation_counterfactual_loss(
        **fixture_inputs(extreme),
        box_ious=torch.tensor([[0.10, 0.70, 0.20, 0.05]]),
        parent_top_k=4,
        max_negatives=4,
    )
    require(torch.isfinite(extreme_result["loss"]),
            "V135 inherited raw score scale escaped bounded supervision")
    extreme_result["loss"].backward()
    require(torch.isfinite(extreme.grad).all(),
            "V135 extreme raw score gradient is non-finite")
    return {
        "hard_negative_count": float(result["hard_negative_count_mean"]),
        "parent_hard_negative_ratio": float(
            result["parent_hard_negative_ratio"]
        ),
        "loss": float(result["loss"]),
        "extreme_loss": float(extreme_result["loss"]),
    }


def audit_dataset_independence(repo_root):
    source = (repo_root / "models/sacr_relation_counterfactual.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = ("scanrefer", "nr3d", "sr3d", "unique", "multiple")
    require(not any(token in source for token in forbidden),
            "V135 mining module contains dataset/subgroup labels")
    forbidden_trainable = (
        "structured_slot_builder.global_attn.",
        "structured_slot_builder.target_attn.",
        "structured_slot_builder.attr_attn.",
        "sacr_head.target_attr_mlp.",
        "sacr_head.global_mlp.",
    )
    require(not any(
        any(prefix.startswith(forbidden) for forbidden in forbidden_trainable)
        for prefix in RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES
    ), "V135 trains a non-differentiable target-shortlist component")
    return {
        "forbidden_tokens": list(forbidden),
        "uses_dataset_or_subgroup_labels": False,
        "trainable_prefixes": list(
            RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--verify")
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    require(repo_root == REPO_ROOT,
            "--repo-root must contain the executing V135 audit script")
    require(bool(args.output) != bool(args.verify),
            "exactly one of --output or --verify is required")
    if args.verify:
        with open(args.verify, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        require(saved.get("schema") == SCHEMA,
                "unexpected V135 contract schema")
        require(saved.get("verdict") == "pass",
                "V135 contract receipt did not pass")
        require(saved.get("source_sha256") == source_hashes(repo_root),
                "V135 audited source hashes changed")
        print("VERIFIED {}".format(args.verify))
        return

    payload = {
        "schema": SCHEMA,
        "verdict": "pass",
        "source_sha256": source_hashes(repo_root),
        "deployment": audit_deployment(),
        "objective": audit_objective(),
        "dataset_independence": audit_dataset_independence(repo_root),
    }
    atomic_write_new_json(payload, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
