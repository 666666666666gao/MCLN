#!/usr/bin/env python
"""Publish an immutable receipt for one dataset-specific exact V99 run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

import torch


DATASETS = {"scanrefer", "nr3d", "sr3d"}
V99_SCHEMA = "rec-pareto-contextual-hierarchical-v1"


def _sha256(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_snapshot(path, label):
    path = Path(path).expanduser().resolve()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("{} must be a regular non-symlink file".format(label))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": int(entry.st_size),
        "mode": stat.S_IMODE(entry.st_mode),
    }


def _torch_payload(snapshot, label):
    try:
        payload = torch.load(snapshot["path"], map_location="cpu")
    except Exception as error:
        raise ValueError("could not load {}: {}".format(label, error))
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a dictionary".format(label))
    return payload


def _json_payload(snapshot, label):
    try:
        payload = json.loads(Path(snapshot["path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("could not load {}: {}".format(label, error))
    if not isinstance(payload, dict):
        raise ValueError("{} must contain an object".format(label))
    return payload


def _extract_rec_metrics(metrics, expected_sample_count):
    """Read REC after the full V99 stack, not the pre-reranker selector."""
    if metrics.get("sample_count") != expected_sample_count:
        raise ValueError("official evaluation sample count changed")
    try:
        subgroups = metrics["position_subgroups"]
        if set(subgroups) != {"unique", "multiple"}:
            raise ValueError("unexpected final REC subgroup set")
        rows = [subgroups[name] for name in ("unique", "multiple")]
        counts = [
            (
                int(row["sample_count"]),
                int(row["hits025"]),
                int(row["hits050"]),
            )
            for row in rows
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("final deployed REC subgroup counters are missing") from error
    if any(not 0 <= hits050 <= hits025 <= total
           for total, hits025, hits050 in counts):
        raise ValueError("final deployed REC subgroup counters are invalid")
    sample_count = sum(total for total, _, _ in counts)
    hits025 = sum(value for _, value, _ in counts)
    hits050 = sum(value for _, _, value in counts)
    if sample_count != expected_sample_count:
        raise ValueError("final REC subgroup sample counts do not reconcile")
    if not 0 <= hits050 <= hits025 <= expected_sample_count:
        raise ValueError("final deployed REC counters are invalid")
    return {
        "counter_source": "final_deployed_position_subgroups",
        "sample_count": expected_sample_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": hits025 / float(expected_sample_count),
        "acc050": hits050 / float(expected_sample_count),
    }


def _exclusive_json(path, payload):
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        offset = 0
        while offset < len(encoded):
            count = os.write(descriptor, encoded[offset:])
            if count <= 0:
                raise OSError("receipt write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return hashlib.sha256(encoded).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--initialization-checkpoint", required=True)
    parser.add_argument("--expected-initialization-sha256", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--hierarchical-artifact", required=True)
    parser.add_argument("--audit-panel-preflight", required=True)
    parser.add_argument("--oof-result", required=True)
    parser.add_argument("--artifact-receipt", required=True)
    parser.add_argument("--eval-receipt", required=True)
    parser.add_argument("--official-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone-joint-training", action="store_true")
    parser.add_argument("--inference-uses-ground-truth", action="store_true")
    parser.add_argument("--no-task-checkpoint-transfer", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_sample_count <= 0:
        parser.error("--expected-sample-count must be positive")

    snapshots = {
        "initialization": _regular_snapshot(
            args.initialization_checkpoint, "initialization checkpoint"
        ),
        "backbone": _regular_snapshot(
            args.backbone_checkpoint, "dataset backbone"
        ),
        "parent": _regular_snapshot(args.parent_artifact, "parent artifact"),
        "geometry": _regular_snapshot(
            args.geometry_artifact, "geometry artifact"
        ),
        "hierarchical": _regular_snapshot(
            args.hierarchical_artifact, "hierarchical artifact"
        ),
        "audit_panel_preflight": _regular_snapshot(
            args.audit_panel_preflight, "audit panel preflight"
        ),
        "oof": _regular_snapshot(args.oof_result, "OOF result"),
        "artifact_receipt": _regular_snapshot(
            args.artifact_receipt, "artifact receipt"
        ),
        "evaluation": _regular_snapshot(
            args.eval_receipt, "evaluation receipt"
        ),
        "official_result": _regular_snapshot(
            args.official_result, "official result"
        ),
    }
    if snapshots["initialization"]["sha256"] != (
            args.expected_initialization_sha256):
        raise ValueError("initialization checkpoint SHA-256 changed")
    for name in ("backbone", "parent", "geometry", "hierarchical"):
        if snapshots[name]["mode"] != 0o444:
            raise ValueError("{} is not protected mode 0444".format(name))

    geometry = _torch_payload(snapshots["geometry"], "geometry artifact")
    hierarchical = _torch_payload(
        snapshots["hierarchical"], "hierarchical artifact"
    )
    oof = _json_payload(snapshots["oof"], "OOF result")
    artifact_receipt = _json_payload(
        snapshots["artifact_receipt"], "artifact receipt"
    )
    audit_panel_preflight = _json_payload(
        snapshots["audit_panel_preflight"], "audit panel preflight"
    )
    evaluation = _json_payload(snapshots["evaluation"], "evaluation receipt")
    official_result = _json_payload(
        snapshots["official_result"], "official result"
    )
    backbone_sha = snapshots["backbone"]["sha256"]
    parent_sha = snapshots["parent"]["sha256"]
    geometry_sha = snapshots["geometry"]["sha256"]
    if (geometry.get("checkpoint_sha256") != backbone_sha
            or geometry.get("parent_artifact_sha256") != parent_sha
            or len(geometry.get("variant_names", ())) != 7):
        raise ValueError("geometry artifact does not bind the 7-variant stack")
    inputs = hierarchical.get("input_sha256")
    if (hierarchical.get("schema") != V99_SCHEMA
            or hierarchical.get("deployable") is not True
            or not isinstance(inputs, dict)
            or inputs.get("backbone") != backbone_sha
            or inputs.get("parent") != parent_sha
            or inputs.get("geometry") != geometry_sha):
        raise ValueError("hierarchical artifact is not the bound V99 stack")
    contract = oof.get("dataset_contract")
    if (not isinstance(contract, dict)
            or contract.get("dataset") != args.dataset
            or contract.get("dataset_only") is not True
            or contract.get("joint_training") is not False
            or contract.get("reranker_dataset_only") is not True
            or contract.get("backbone_training_dataset_only") is not (
                not args.backbone_joint_training
            )
            or contract.get("backbone_joint_training") is not (
                args.backbone_joint_training
            )
            or contract.get("inference_uses_ground_truth") is not (
                args.inference_uses_ground_truth
            )
            or contract.get("variant_count") != 7
            or contract.get("flat_candidate_count") != 112
            or contract.get("backbone_sha256") != backbone_sha
            or oof.get("oof", {}).get("passed") is not True):
        raise ValueError("OOF dataset contract did not pass")
    if artifact_receipt.get("oof_result_sha256") != snapshots["oof"]["sha256"]:
        raise ValueError("artifact receipt OOF binding changed")
    if (audit_panel_preflight.get("schema")
            != "mcln-dataset-v99-geometry-panel-preflight-v1"
            or audit_panel_preflight.get("passed") is not True
            or audit_panel_preflight.get("dataset") != args.dataset
            or audit_panel_preflight.get("dataset_only") is not True
            or audit_panel_preflight.get("checkpoint_sha256") != backbone_sha
            or audit_panel_preflight.get("required_scene_count") != 64
            or audit_panel_preflight.get("eligible_scene_count", 0) < 64
            or audit_panel_preflight.get("expressions_per_scene") != 4
            or audit_panel_preflight.get("selected_sample_count") != 256):
        raise ValueError("geometry audit panel preflight did not pass")
    if (official_result.get("schema")
            != "mcln-dataset-v99-official-result-v1"
            or official_result.get("dataset") != args.dataset
            or official_result.get("sample_count")
            != args.expected_sample_count
            or official_result.get("dataset_only") is not (
                not args.backbone_joint_training
            )
            or official_result.get("joint_training") is not (
                args.backbone_joint_training
            )
            or official_result.get("inference_uses_ground_truth") is not (
                args.inference_uses_ground_truth
            )
            or official_result.get("metrics") != evaluation
            or official_result.get("eval_receipt", {}).get("sha256")
            != snapshots["evaluation"]["sha256"]
            or official_result.get("artifacts", {}).get(
                "backbone", {}
            ).get("sha256") != backbone_sha
            or official_result.get("artifacts", {}).get(
                "parent", {}
            ).get("sha256") != parent_sha
            or official_result.get("artifacts", {}).get(
                "geometry", {}
            ).get("sha256") != geometry_sha
            or official_result.get("artifacts", {}).get(
                "hierarchical", {}
            ).get("sha256") != snapshots["hierarchical"]["sha256"]):
        raise ValueError("official result does not bind the live V99 stack")
    rec = _extract_rec_metrics(evaluation, args.expected_sample_count)

    receipt = {
        "schema": "mcln-dataset-v99-pipeline-receipt-v1",
        "version": 1,
        "dataset": args.dataset,
        "dataset_only": not args.backbone_joint_training,
        "joint_training": args.backbone_joint_training,
        "inference_uses_ground_truth": args.inference_uses_ground_truth,
        "task_checkpoint_transfer": not args.no_task_checkpoint_transfer,
        "initialization_policy": (
            "Random MCLN task heads with GroupFree/PointNet detector "
            "pretraining; no task checkpoint transfer"
            if args.no_task_checkpoint_transfer else
            "Task-checkpoint initialization followed by dataset optimization"
        ),
        "method": {
            "name": "V99 contextual Pareto hierarchy",
            "query_count": 16,
            "variant_count": 7,
            "flat_candidate_count": 112,
            "aggregate_margin": hierarchical["policy"]["aggregate_margin"],
            "require_positive_delta025": True,
            "require_positive_delta050": True,
        },
        "artifacts": snapshots,
        "metrics": {
            "rec": rec,
            "official_eval_receipt": evaluation,
        },
    }
    receipt_sha = _exclusive_json(args.output, receipt)
    print(json.dumps({
        "output": str(Path(args.output).expanduser().absolute()),
        "sha256": receipt_sha,
        "dataset": args.dataset,
        "rec": rec,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
