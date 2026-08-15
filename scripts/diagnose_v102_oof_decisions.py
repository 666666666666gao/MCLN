#!/usr/bin/env python
"""Read-only decision-level diagnosis for the frozen V102 OOF result."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from models.rec_joint_box_mask import (
    LEGACY_MASK_POLICY_INDEX,
    MASK_LOGIT_THRESHOLDS,
    MASK_POLICY_COUNT,
    MASK_SOURCE_NAMES,
)


EXPECTED_PREDICTION_SHA256 = (
    "f248e0ce2f853f59be0c09a1e9c80c846dc7e199cbefd89c2f770f06ebb4ab85"
)
EXPECTED_RESULT_SHA256 = (
    "4ed892f89bc04fecd0f1618dc415039614241d432b90b0010af3908ef719c1b0"
)
EXPECTED_ROWS = 36665


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_name(index):
    source = MASK_SOURCE_NAMES[index // len(MASK_LOGIT_THRESHOLDS)]
    threshold = MASK_LOGIT_THRESHOLDS[index % len(MASK_LOGIT_THRESHOLDS)]
    return "{}@{:+.1f}".format(source, threshold)


def summarize(before, after):
    before = torch.as_tensor(before, dtype=torch.float64).reshape(-1)
    after = torch.as_tensor(after, dtype=torch.float64).reshape(-1)
    result = {"count": int(before.numel())}
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        old = before.gt(threshold)
        new = after.gt(threshold)
        result.update({
            "rescues" + suffix: int((~old & new).sum().item()),
            "breaks" + suffix: int((old & ~new).sum().item()),
            "delta_hits" + suffix: int(new.sum().item() - old.sum().item()),
        })
    delta = after - before
    result.update({
        "delta_miou": float(delta.mean().item()) if delta.numel() else 0.0,
        "improved_iou": int(delta.gt(0).sum().item()),
        "equal_iou": int(delta.eq(0).sum().item()),
        "degraded_iou": int(delta.lt(0).sum().item()),
    })
    return result


def pearson(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator <= 0:
        return None
    return float((left * right).sum().div(denominator).item())


def quantile_calibration(predicted, actual, accepted, bins=10):
    predicted = torch.as_tensor(predicted, dtype=torch.float64).reshape(-1)
    actual = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    accepted = torch.as_tensor(accepted, dtype=torch.bool).reshape(-1)
    indices = accepted.nonzero(as_tuple=False).reshape(-1)
    order = indices[predicted[indices].argsort()]
    records = []
    for chunk in torch.tensor_split(order, bins):
        if not chunk.numel():
            continue
        records.append({
            "count": int(chunk.numel()),
            "predicted_mean": float(predicted[chunk].mean().item()),
            "actual_mean": float(actual[chunk].mean().item()),
            "actual_positive": int(actual[chunk].gt(0).sum().item()),
            "actual_negative": int(actual[chunk].lt(0).sum().item()),
        })
    return records


def write_exclusive_json(path, value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    descriptor = os.open(
        str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("diagnostic write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--result", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result_path = Path(args.result).expanduser().resolve()
    decision_path = Path(args.decisions).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    if file_sha256(result_path) != EXPECTED_RESULT_SHA256:
        raise ValueError("frozen V102 result SHA-256 changed")
    frozen = json.loads(result_path.read_text(encoding="ascii"))
    decisions = torch.load(decision_path, map_location="cpu")
    if (decisions.get("schema") != "rec-v102-mask-only-oof-decisions-v1"
            or decisions.get("row_count") != EXPECTED_ROWS
            or decisions.get("prediction_sha256") != EXPECTED_PREDICTION_SHA256
            or decisions.get("prediction_sha256")
            != frozen["diagnostics"]["prediction_sha256"]):
        raise ValueError("V102 decision replay identity changed")

    policies = torch.as_tensor(decisions["selected_policy_indices"]).long()
    proposals = torch.as_tensor(decisions["proposal_policy_indices"]).long()
    accepted = torch.as_tensor(decisions["accepted"]).bool()
    folds = torch.as_tensor(decisions["fold_ids"]).long()
    all_ious = torch.as_tensor(
        decisions["selected_parent_policy_ious"], dtype=torch.float64
    )
    before = torch.as_tensor(decisions["before_ious"], dtype=torch.float64)
    after = torch.as_tensor(decisions["after_ious"], dtype=torch.float64)
    predicted_iou = torch.as_tensor(
        decisions["predicted_delta_iou"], dtype=torch.float64
    )
    predicted_hits = torch.as_tensor(
        decisions["predicted_delta_hits"], dtype=torch.float64
    )
    if (policies.shape != (EXPECTED_ROWS,)
            or proposals.shape != (EXPECTED_ROWS,)
            or accepted.shape != (EXPECTED_ROWS,)
            or folds.shape != (EXPECTED_ROWS,)
            or all_ious.shape != (EXPECTED_ROWS, MASK_POLICY_COUNT)
            or not torch.equal(after, all_ious[
                torch.arange(EXPECTED_ROWS), policies
            ])):
        raise ValueError("V102 decision tensors do not align")
    actual_iou = after - before
    actual_hits = torch.stack((
        after.gt(0.25).double() - before.gt(0.25).double(),
        after.gt(0.50).double() - before.gt(0.50).double(),
    ), dim=-1)

    policy_records = []
    for policy in range(MASK_POLICY_COUNT):
        mask = accepted & policies.eq(policy)
        policy_records.append({
            "policy_index": policy,
            "policy": policy_name(policy),
            "selected_count": int(mask.sum().item()),
            **summarize(before[mask], after[mask]),
            "predicted_delta_iou_mean": (
                float(predicted_iou[mask].mean().item()) if mask.any() else 0.0
            ),
            "predicted_delta025_mean": (
                float(predicted_hits[mask, 0].mean().item()) if mask.any() else 0.0
            ),
            "predicted_delta050_mean": (
                float(predicted_hits[mask, 1].mean().item()) if mask.any() else 0.0
            ),
        })

    fold_policy_records = []
    for fold in range(5):
        for policy in range(MASK_POLICY_COUNT):
            mask = accepted & folds.eq(fold) & policies.eq(policy)
            if mask.any():
                fold_policy_records.append({
                    "fold": fold,
                    "policy_index": policy,
                    "policy": policy_name(policy),
                    **summarize(before[mask], after[mask]),
                })

    accepted_summary = summarize(before[accepted], after[accepted])
    proposal_changed = proposals.ne(LEGACY_MASK_POLICY_INDEX)
    proposal_after = all_ious[torch.arange(EXPECTED_ROWS), proposals]
    report = {
        "schema": "rec-v102-mask-only-oof-diagnosis-v1",
        "version": 1,
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "frozen_result_sha256": file_sha256(result_path),
        "decision_sidecar_sha256": file_sha256(decision_path),
        "prediction_sha256": decisions["prediction_sha256"],
        "coverage": {"rows": EXPECTED_ROWS, "accepted": int(accepted.sum())},
        "accepted_summary": accepted_summary,
        "all_proposal_changes_counterfactual": summarize(
            before[proposal_changed], proposal_after[proposal_changed]
        ),
        "prediction_calibration": {
            "pearson_delta_iou": pearson(
                predicted_iou[accepted], actual_iou[accepted]
            ),
            "pearson_delta025": pearson(
                predicted_hits[accepted, 0], actual_hits[accepted, 0]
            ),
            "pearson_delta050": pearson(
                predicted_hits[accepted, 1], actual_hits[accepted, 1]
            ),
            "delta_iou_deciles": quantile_calibration(
                predicted_iou, actual_iou, accepted
            ),
            "delta025_deciles": quantile_calibration(
                predicted_hits[:, 0], actual_hits[:, 0], accepted
            ),
            "delta050_deciles": quantile_calibration(
                predicted_hits[:, 1], actual_hits[:, 1], accepted
            ),
        },
        "per_policy": policy_records,
        "per_fold_policy": fold_policy_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_exclusive_json(output, report)
    print(json.dumps({
        "output": str(output), "sha256": digest,
        "accepted_summary": accepted_summary,
        "correlations": {
            key: value for key, value in report["prediction_calibration"].items()
            if key.startswith("pearson")
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
