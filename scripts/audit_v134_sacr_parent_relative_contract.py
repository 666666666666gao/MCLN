#!/usr/bin/env python3
"""Audit V134 deployment, cross-dataset, and DDP loss contracts."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from v133_receipt_utils import atomic_write_new_json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.losses import (  # noqa: E402
    build_sacr_score_mask_supervision_mask,
    compute_sacr_score_parent_relative_loss,
)
from models.sacr_parent_relative import (  # noqa: E402
    SACRParentRelativeGate,
    apply_parent_relative_sacr_refinement,
)


SCHEMA = "mcln-v134-parent-relative-contract-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_hashes(repo_root):
    return {
        "main_utils.py": sha256(repo_root / "main_utils.py"),
        "train_dist_mod.py": sha256(repo_root / "train_dist_mod.py"),
        "models/losses.py": sha256(repo_root / "models/losses.py"),
        "models/mcln.py": sha256(repo_root / "models/mcln.py"),
        "models/sacr_head.py": sha256(repo_root / "models/sacr_head.py"),
        "models/sacr_parent_relative.py": sha256(
            repo_root / "models/sacr_parent_relative.py"
        ),
        "src/grounding_evaluator.py": sha256(
            repo_root / "src/grounding_evaluator.py"
        ),
        "scripts/audit_v134_sacr_parent_relative_contract.py": sha256(
            repo_root
            / "scripts/audit_v134_sacr_parent_relative_contract.py"
        ),
        "scripts/v133_receipt_utils.py": sha256(
            repo_root / "scripts/v133_receipt_utils.py"
        ),
    }


def loss_for(
        raw_scores, gate_logit, parent_scores, box_ious, structured,
        mask_ious=None, mask_rows=None):
    if mask_rows is None:
        mask_rows = torch.zeros_like(structured)
    candidate_valid = torch.ones_like(raw_scores, dtype=torch.bool)
    sample_gate = gate_logit.sigmoid().expand(raw_scores.shape[0])
    deployment = apply_parent_relative_sacr_refinement(
        raw_scores=raw_scores,
        parent_scores=parent_scores,
        candidate_valid=candidate_valid,
        structured_valid_mask=structured,
        sample_gate=sample_gate,
        max_delta=0.25,
        promotion_margin=0.01,
    )
    result = compute_sacr_score_parent_relative_loss(
        scores=deployment["scores"],
        parent_scores=parent_scores,
        relative_raw_scores=deployment["relative_raw_scores"],
        sample_gate=deployment["sample_gate"],
        parent_indices=deployment["parent_indices"],
        feasible_candidate_mask=deployment["feasible_candidate_mask"],
        box_ious=box_ious,
        valid_mask=deployment["apply_mask"],
        structured_valid_mask=structured,
        sample_mask=torch.ones_like(structured),
        mask_ious=mask_ious,
        mask_supervision_mask=mask_rows,
        temperature=0.1,
        mask_weight=0.25,
        max_delta=0.25,
        min_box_advantage=0.03,
        promotion_margin=0.01,
        mask_tolerance=0.02,
        raw_margin=0.1,
        dense_weight=0.25,
        preserve_weight=1.0,
        gate_weight=0.05,
    )
    return result, deployment


def audit_deployment():
    parent_scores = torch.tensor([[1.0, 0.9, 0.6]])
    uniform = apply_parent_relative_sacr_refinement(
        raw_scores=torch.full((1, 3), 2.0),
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
    )
    require(torch.equal(uniform["scores"], parent_scores),
            "uniform raw offsets did not cancel exactly")
    require(torch.equal(uniform["residual"], torch.zeros_like(parent_scores)),
            "uniform raw offsets produced a residual")
    output = apply_parent_relative_sacr_refinement(
        raw_scores=torch.tensor([[0.0, 2.0, 4.0]]),
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
    )
    require(output["parent_indices"].tolist() == [0],
            "deployment chose the wrong parent")
    require(output["feasible_candidate_mask"].tolist()
            == [[False, True, False]],
            "deployment feasibility mask changed")
    require(output["scores"][0, 0] == parent_scores[0, 0],
            "deployment changed the parent score")
    require(output["scores"][0, 2] == parent_scores[0, 2],
            "deployment changed an infeasible candidate")
    require(float(output["residual"].abs().max()) <= 0.25,
            "deployment exceeded the residual bound")
    return {
        "uniform_identity": True,
        "parent_exact": True,
        "infeasible_exact": True,
        "feasible_candidate_mask": output[
            "feasible_candidate_mask"
        ].tolist(),
        "residual_abs_max": float(output["residual"].abs().max()),
    }


def audit_reliability_gate():
    gate = SACRParentRelativeGate(hidden_dim=8, top_k_anchors=4)
    output = gate(
        raw_scores=torch.tensor([
            [0.0, 0.3, -0.2],
            [0.0, 0.4, 0.2],
        ]),
        parent_scores=torch.tensor([
            [1.0, 0.9, 0.6],
            [1.0, 0.6, 0.5],
        ]),
        candidate_valid=torch.ones((2, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(2, dtype=torch.bool),
        parse_confidence=torch.tensor([0.9, 0.8]),
        anchor_top1_mass=torch.tensor([0.7, 0.6]),
        anchor_entropy=torch.tensor([0.2, 0.3]),
        relation_active_ratio=torch.tensor([1.0, 0.5]),
        max_delta=0.25,
        promotion_margin=0.01,
    )
    require(output["features"].shape == (2, 8),
            "reliability feature contract changed")
    require(output["sample_gate"][0] > 0.0,
            "feasible row did not receive a trainable gate")
    require(output["sample_gate"][1] == 0.0,
            "row without a feasible candidate did not strictly abstain")
    require(not any(
        name == "sacr_score_gate" for name, _ in gate.named_parameters()
    ), "reliability gate retained the V133 global scalar")
    return {
        "feature_names": list(SACRParentRelativeGate.FEATURE_NAMES),
        "feature_count": int(output["features"].shape[1]),
        "initial_feasible_gate": float(output["sample_gate"][0]),
        "no_feasible_gate": float(output["sample_gate"][1]),
        "global_scalar_present": False,
    }


def row_loss_and_grad(box_ious, mask_ious, mask_row):
    raw_scores = torch.tensor([[0.0, 0.2, -0.1]], requires_grad=True)
    gate_logit = torch.tensor([1.0], requires_grad=True)
    result, _ = loss_for(
        raw_scores=raw_scores,
        gate_logit=gate_logit,
        parent_scores=torch.tensor([[1.0, 0.9, 0.8]]),
        box_ious=box_ious,
        mask_ious=mask_ious,
        mask_rows=torch.tensor([mask_row], dtype=torch.bool),
        structured=torch.ones(1, dtype=torch.bool),
    )
    result["loss"].backward()
    return (
        result["loss"].detach(),
        raw_scores.grad.detach().clone(),
        gate_logit.grad.detach().clone(),
    )


def audit_cross_dataset():
    datasets = ["scanrefer", "nr3d", "sr3d"]
    mask_rows = build_sacr_score_mask_supervision_mask(
        {"sample_dataset": datasets}, 3, torch.device("cpu")
    )
    require(mask_rows.tolist() == [True, False, False],
            "mask supervision is not ScanRefer-only")
    box_ious = torch.tensor([[0.40, 0.65, 0.50]])
    mask_ious = torch.tensor([[0.80, 0.10, 0.90]])
    box_only = row_loss_and_grad(box_ious, None, False)
    for dataset in ("nr3d", "sr3d"):
        with_unused_mask = row_loss_and_grad(
            box_ious, mask_ious, False
        )
        require(torch.equal(with_unused_mask[0], box_only[0]),
                "{} loss used mask quality".format(dataset))
        require(torch.equal(with_unused_mask[1], box_only[1]),
                "{} raw-score gradient used mask quality".format(dataset))
        require(torch.equal(with_unused_mask[2], box_only[2]),
                "{} gate gradient used mask quality".format(dataset))
    scanrefer = row_loss_and_grad(box_ious, mask_ious, True)
    require(abs(float(scanrefer[0] - box_only[0])) > 1e-6,
            "ScanRefer mask safety had no measurable effect")
    return {
        "datasets": datasets,
        "mask_supervision_mask": mask_rows.tolist(),
        "box_only_loss": float(box_only[0]),
        "scanrefer_mask_loss": float(scanrefer[0]),
        "scanrefer_mask_delta": float(scanrefer[0] - box_only[0]),
    }


def audit_objective():
    raw_scores = torch.tensor([[0.0, 0.2, -0.1]], requires_grad=True)
    gate_logit = torch.tensor([1.0], requires_grad=True)
    result, _ = loss_for(
        raw_scores=raw_scores,
        gate_logit=gate_logit,
        parent_scores=torch.tensor([[1.0, 0.9, 0.8]]),
        box_ious=torch.tensor([[0.40, 0.65, 0.50]]),
        mask_ious=torch.tensor([[0.50, 0.55, 0.45]]),
        mask_rows=torch.ones(1, dtype=torch.bool),
        structured=torch.ones(1, dtype=torch.bool),
    )
    component_names = (
        "dense_advantage_loss",
        "feasible_rank_loss",
        "promotion_loss",
        "preserve_loss",
        "abstention_loss",
        "saturation_loss",
    )
    require(all(name in result for name in component_names),
            "parent-relative objective is missing a required component")
    require(result["repairable_row_ratio"] == 1.0,
            "objective fixture did not expose one repairable row")
    require(result["feasible_rank_loss"] > 0.0,
            "feasible listwise ranking is inactive")
    result["loss"].backward()
    require(torch.isfinite(raw_scores.grad).all(),
            "objective produced a non-finite score gradient")
    require(torch.isfinite(gate_logit.grad).all(),
            "objective produced a non-finite gate gradient")
    extreme_raw_scores = torch.tensor(
        [[0.0, -1e15, 1e15]], requires_grad=True
    )
    extreme_result, _ = loss_for(
        raw_scores=extreme_raw_scores,
        gate_logit=torch.tensor([-4.0], requires_grad=True),
        parent_scores=torch.tensor([[1.0, 0.9, 0.8]]),
        box_ious=torch.tensor([[0.20, 0.60, 0.10]]),
        structured=torch.ones(1, dtype=torch.bool),
    )
    require(torch.isfinite(extreme_result["loss"]),
            "inherited V133 raw-score scale produced a non-finite loss")
    require(float(extreme_result["loss"]) < 100.0,
            "inherited V133 raw-score scale escaped bounded supervision")
    extreme_result["loss"].backward()
    require(torch.isfinite(extreme_raw_scores.grad).all(),
            "inherited V133 raw-score scale produced a non-finite gradient")
    audit = {
        name: float(result[name]) for name in component_names
    }
    audit["inherited_raw_scale_loss"] = float(extreme_result["loss"])
    return audit


def ddp_case(parameter, gate_logit, rank=None):
    parents = (
        torch.tensor([[1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]),
        torch.tensor([[1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]),
    )
    bases = (
        torch.tensor([[0.0, 0.1, -0.1], [0.0, -0.2, 0.2]]),
        torch.tensor([[0.0, 0.2, -0.1], [0.0, 0.3, 0.1]]),
    )
    qualities = (
        torch.tensor([[0.2, 0.8, 0.1], [0.3, 0.2, 0.9]]),
        torch.tensor([[0.4, 0.7, 0.5], [0.8, 0.5, 0.6]]),
    )
    structured = (
        torch.tensor([False, False]),
        torch.tensor([True, True]),
    )
    if rank is not None:
        result, _ = loss_for(
            raw_scores=bases[rank] + parameter.unsqueeze(0),
            gate_logit=gate_logit,
            parent_scores=parents[rank],
            box_ious=qualities[rank],
            structured=structured[rank],
        )
        return result["loss"]
    result, _ = loss_for(
        raw_scores=torch.cat(bases, dim=0) + parameter.unsqueeze(0),
        gate_logit=gate_logit,
        parent_scores=torch.cat(parents, dim=0),
        box_ious=torch.cat(qualities, dim=0),
        structured=torch.cat(structured, dim=0),
    )
    return result["loss"]


def _ddp_worker(rank, world_size, init_path, result_path):
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_path),
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.tensor([0.1, -0.05, 0.02], requires_grad=True)
        gate_logit = torch.tensor([0.7], requires_grad=True)
        loss = ddp_case(parameter, gate_logit, rank=rank)
        loss.backward()
        averaged = torch.cat((
            loss.detach().reshape(1),
            parameter.grad.detach(),
            gate_logit.grad.detach(),
        ))
        dist.all_reduce(averaged)
        averaged /= float(world_size)
        if rank == 0:
            with open(result_path, "w") as handle:
                json.dump({"values": averaged.tolist()}, handle)
    finally:
        dist.destroy_process_group()


def audit_ddp():
    directory = tempfile.mkdtemp(prefix="v134_ddp_contract_")
    init_path = os.path.join(directory, "init")
    result_path = os.path.join(directory, "result.json")
    mp.spawn(
        _ddp_worker,
        args=(2, init_path, result_path),
        nprocs=2,
        join=True,
    )
    with open(result_path, "r") as handle:
        distributed = torch.tensor(json.load(handle)["values"])
    parameter = torch.tensor([0.1, -0.05, 0.02], requires_grad=True)
    gate_logit = torch.tensor([0.7], requires_grad=True)
    reference_loss = ddp_case(parameter, gate_logit)
    reference_loss.backward()
    reference = torch.cat((
        reference_loss.detach().reshape(1),
        parameter.grad.detach(),
        gate_logit.grad.detach(),
    ))
    require(torch.allclose(distributed, reference, atol=1e-7, rtol=1e-6),
            "DDP loss/gradient differs from the global example mean")
    for path in (result_path, init_path):
        if os.path.exists(path):
            os.unlink(path)
    os.rmdir(directory)
    return {
        "global_supervised_rows": 2,
        "rank_supervised_rows": [0, 2],
        "distributed": distributed.tolist(),
        "reference": reference.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--verify")
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    require(repo_root == REPO_ROOT,
            "--repo-root must be the code imported by this audit script")
    require(bool(args.output) != bool(args.verify),
            "exactly one of --output or --verify is required")
    if args.verify:
        with open(args.verify, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        require(saved.get("schema") == SCHEMA,
                "V134 contract receipt schema changed")
        require(saved.get("verdict") == "pass",
                "V134 contract receipt did not pass")
        require(saved.get("source_sha256") == source_hashes(repo_root),
                "V134 contract receipt source hashes changed")
        require(
            saved.get("reliability_gate", {}).get(
                "global_scalar_present"
            ) is False,
            "V134 contract receipt retained a global scalar gate",
        )
        print(json.dumps({
            "receipt": str(Path(args.verify).resolve()),
            "sha256": sha256(args.verify),
            "verdict": "verified",
        }, sort_keys=True))
        return
    payload = {
        "schema": SCHEMA,
        "verdict": "pass",
        "deployment": audit_deployment(),
        "reliability_gate": audit_reliability_gate(),
        "cross_dataset": audit_cross_dataset(),
        "objective": audit_objective(),
        "ddp": audit_ddp(),
        "source_sha256": source_hashes(repo_root),
    }
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "receipt": str(output),
        "sha256": sha256(output),
        "verdict": "pass",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
