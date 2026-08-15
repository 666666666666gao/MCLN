#!/usr/bin/env python3
"""Audit V133 cross-dataset targets and DDP example normalization."""

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

from models.losses import (
    build_sacr_score_mask_supervision_mask,
    compute_sacr_score_refiner_listwise_loss,
)


SCHEMA = "mcln-v133-supervision-contract-v1"


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


def loss_for(scores, box_ious, mask_ious, mask_rows, structured=None):
    if structured is None:
        structured = torch.ones(
            scores.shape[0], dtype=torch.bool, device=scores.device
        )
    return compute_sacr_score_refiner_listwise_loss(
        scores=scores,
        box_ious=box_ious,
        mask_ious=mask_ious,
        valid_mask=torch.ones_like(scores, dtype=torch.bool),
        structured_valid_mask=structured,
        sample_mask=torch.ones_like(structured),
        mask_supervision_mask=mask_rows,
        temperature=0.1,
        mask_weight=0.25,
    )["loss"]


def audit_cross_dataset():
    datasets = ["scanrefer", "nr3d", "sr3d"]
    mask_rows = build_sacr_score_mask_supervision_mask(
        {"sample_dataset": datasets}, 3, torch.device("cpu")
    )
    require(mask_rows.tolist() == [True, False, False],
            "mask supervision is not ScanRefer-only")
    score_values = torch.tensor([
        [0.3, -0.4, 0.1],
        [-0.2, 0.5, 0.0],
        [0.1, 0.2, -0.5],
    ], dtype=torch.float32)
    box_ious = torch.tensor([
        [0.1, 0.7, 0.2],
        [0.8, 0.1, 0.3],
        [0.2, 0.4, 0.9],
    ], dtype=torch.float32)
    mask_ious = torch.tensor([
        [0.9, 0.1, 0.2],
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.1],
    ], dtype=torch.float32)
    batch_scores = score_values.clone().requires_grad_(True)
    batch_loss = loss_for(
        batch_scores, box_ious, mask_ious, mask_rows
    )
    batch_loss.backward()
    batch_grad = batch_scores.grad.detach().clone()
    row_losses = []
    row_grads = []
    for row in range(3):
        scores = score_values[row:row + 1].clone().requires_grad_(True)
        row_loss = loss_for(
            scores,
            box_ious[row:row + 1],
            mask_ious[row:row + 1],
            mask_rows[row:row + 1],
        )
        row_loss.backward()
        row_losses.append(row_loss.detach())
        row_grads.append(scores.grad.detach().clone())
    expected_loss = torch.stack(row_losses).mean()
    expected_grad = torch.cat(row_grads, dim=0) / 3.0
    require(torch.allclose(batch_loss.detach(), expected_loss, atol=1e-7,
                           rtol=1e-6),
            "mixed-dataset loss is not the example mean")
    require(torch.allclose(batch_grad, expected_grad, atol=1e-7, rtol=1e-6),
            "mixed-dataset gradient is not the example mean")
    for row, dataset in ((1, "nr3d"), (2, "sr3d")):
        scores_with = score_values[row:row + 1].clone().requires_grad_(True)
        with_mask = loss_for(
            scores_with,
            box_ious[row:row + 1],
            mask_ious[row:row + 1],
            mask_rows[row:row + 1],
        )
        with_mask.backward()
        scores_box = score_values[row:row + 1].clone().requires_grad_(True)
        box_only = loss_for(
            scores_box,
            box_ious[row:row + 1],
            None,
            mask_rows[row:row + 1],
        )
        box_only.backward()
        require(torch.equal(with_mask.detach(), box_only.detach()),
                "{} loss used mask quality".format(dataset))
        require(torch.equal(scores_with.grad, scores_box.grad),
                "{} gradient used mask quality".format(dataset))
    scan_box_only = loss_for(
        score_values[0:1], box_ious[0:1], None, mask_rows[0:1]
    )
    require(abs(float(row_losses[0] - scan_box_only)) > 1e-5,
            "ScanRefer mask term had no measurable effect")
    return {
        "datasets": datasets,
        "mask_supervision_mask": mask_rows.tolist(),
        "mixed_loss": float(batch_loss.detach()),
        "scanrefer_mask_delta": float(row_losses[0] - scan_box_only),
    }


def _ddp_worker(rank, world_size, init_path, result_path):
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_path),
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.tensor([0.2, -0.1, 0.05], requires_grad=True)
        bases = (
            torch.tensor([[0.1, 0.0, -0.2], [-0.3, 0.2, 0.4]]),
            torch.tensor([[0.0, 0.3, -0.1], [0.2, -0.4, 0.1]]),
        )
        qualities = (
            torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.6, 0.2]]),
            torch.tensor([[0.2, 0.9, 0.3], [0.7, 0.1, 0.4]]),
        )
        structured = (
            torch.tensor([False, False]),
            torch.tensor([True, True]),
        )
        scores = bases[rank] + parameter.unsqueeze(0)
        loss = loss_for(
            scores,
            qualities[rank],
            None,
            torch.zeros(2, dtype=torch.bool),
            structured=structured[rank],
        )
        loss.backward()
        averaged_loss = loss.detach().clone()
        averaged_grad = parameter.grad.detach().clone()
        dist.all_reduce(averaged_loss)
        dist.all_reduce(averaged_grad)
        averaged_loss /= float(world_size)
        averaged_grad /= float(world_size)
        if rank == 0:
            with open(result_path, "w") as handle:
                json.dump({
                    "loss": float(averaged_loss),
                    "grad": averaged_grad.tolist(),
                }, handle)
    finally:
        dist.destroy_process_group()


def audit_ddp():
    directory = tempfile.mkdtemp(prefix="v133_ddp_contract_")
    init_path = os.path.join(directory, "init")
    result_path = os.path.join(directory, "result.json")
    mp.spawn(
        _ddp_worker,
        args=(2, init_path, result_path),
        nprocs=2,
        join=True,
    )
    with open(result_path, "r") as handle:
        distributed = json.load(handle)
    parameter = torch.tensor([0.2, -0.1, 0.05], requires_grad=True)
    bases = torch.cat((
        torch.tensor([[0.1, 0.0, -0.2], [-0.3, 0.2, 0.4]]),
        torch.tensor([[0.0, 0.3, -0.1], [0.2, -0.4, 0.1]]),
    ), dim=0)
    qualities = torch.cat((
        torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.6, 0.2]]),
        torch.tensor([[0.2, 0.9, 0.3], [0.7, 0.1, 0.4]]),
    ), dim=0)
    reference_loss = loss_for(
        bases + parameter.unsqueeze(0),
        qualities,
        None,
        torch.zeros(4, dtype=torch.bool),
        structured=torch.tensor([False, False, True, True]),
    )
    reference_loss.backward()
    distributed_grad = torch.tensor(distributed["grad"])
    require(abs(distributed["loss"] - float(reference_loss)) <= 1e-7,
            "DDP averaged loss differs from the global example mean")
    require(torch.allclose(distributed_grad, parameter.grad, atol=1e-7,
                           rtol=1e-6),
            "DDP averaged gradient differs from the global example mean")
    for path in (result_path, init_path):
        if os.path.exists(path):
            os.unlink(path)
    os.rmdir(directory)
    return {
        "global_supervised_rows": 2,
        "rank_supervised_rows": [0, 2],
        "distributed_loss": distributed["loss"],
        "reference_loss": float(reference_loss.detach()),
        "distributed_grad": distributed["grad"],
        "reference_grad": parameter.grad.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    require(repo_root == REPO_ROOT,
            "--repo-root must be the code imported by this audit script")
    payload = {
        "schema": SCHEMA,
        "verdict": "pass",
        "cross_dataset": audit_cross_dataset(),
        "ddp": audit_ddp(),
        "source_sha256": {
            "models/losses.py": sha256(repo_root / "models/losses.py"),
            "scripts/audit_v133_sacr_supervision_contract.py": sha256(
                repo_root / "scripts/audit_v133_sacr_supervision_contract.py"
            ),
            "scripts/v133_receipt_utils.py": sha256(
                repo_root / "scripts/v133_receipt_utils.py"
            ),
        },
    }
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "receipt": str(output),
        "sha256": sha256(output),
        "verdict": "pass",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
