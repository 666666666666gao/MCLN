#!/usr/bin/env python
"""Run and receipt one dataset-specific V99 official validation pass."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


RESULT_SCHEMA = "mcln-dataset-v99-official-result-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path, label):
    path = Path(path).expanduser().resolve()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("{} must be a regular non-symlink file".format(label))
    if stat.S_IMODE(entry.st_mode) != 0o444:
        raise ValueError("{} must have mode 0444".format(label))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": int(entry.st_size),
        "mode": stat.S_IMODE(entry.st_mode),
    }


def _exclusive_json(path, payload):
    path = Path(path)
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
                raise OSError("official receipt write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _load_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("could not read {}: {}".format(label, error))
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(label))
    return value


def _metric_receipts(output):
    """Return immutable metric receipts anywhere below the official run root."""
    output = Path(output).expanduser().resolve()
    receipts = []
    for candidate in output.rglob("eval_metrics_epoch_*.json"):
        entry = os.lstat(str(candidate))
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise ValueError(
                "official evaluation receipt must be a regular non-symlink file"
            )
        resolved = candidate.resolve()
        try:
            resolved.relative_to(output)
        except ValueError as error:
            raise ValueError(
                "official evaluation receipt escaped the output directory"
            ) from error
        receipts.append(resolved)
    return set(receipts)


def _new_metric_receipt(output, before):
    new_receipts = sorted(_metric_receipts(output) - set(before))
    if len(new_receipts) != 1:
        raise ValueError(
            "official evaluation must publish exactly one new metric receipt; "
            "found {}".format(len(new_receipts))
        )
    return new_receipts[0]


def _build_command(args, artifacts, output_dir):
    command = [
        str(Path(args.python_bin).expanduser().resolve()),
        "-m", "torch.distributed.launch",
        "--nproc_per_node", "1",
        "--master_port", str(args.master_port),
        "train_dist_mod.py",
        "--num_decoder_layers", "6",
        "--num_target", "256",
        "--model", "MCLN",
        "--use_color",
        "--self_attend",
        "--detect_intermediate",
        "--use_soft_token_loss",
        "--use_contrastive_align",
        "--use_source_choice_selector",
        "--source_choice_selector_sources",
        "default,default_rank_blend_contrastive010",
        "--source_choice_selector_hidden_dim", "288",
        "--skip_missing_superpoints",
        "--dataset", args.dataset,
        "--test_dataset", args.dataset,
        "--data_root", str(Path(args.data_root).expanduser().resolve()) + os.sep,
        "--batch_size", "12",
        "--num_workers", "2",
        "--print_freq", "100",
        "--checkpoint_path", artifacts["backbone"]["path"],
        "--rec_reranker_checkpoint", artifacts["parent"]["path"],
        "--rec_geometry_reranker_checkpoint", artifacts["geometry"]["path"],
        "--rec_hierarchical_reranker_checkpoint",
        artifacts["hierarchical"]["path"],
        "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
        "--eval_use_rec_hierarchical_reranker_scores",
        "--expected_eval_sample_count", str(args.expected_sample_count),
        "--log_dir", str(output_dir),
        "--exp", args.experiment,
        "--start_epoch", "0",
        "--eval",
    ]
    if args.backbone_joint_training:
        command.append("--joint_det")
    if args.inference_uses_ground_truth:
        command.append("--butd_cls")
    return command


def _validate_command(command, args):
    forbidden = {
        "--butd", "--butd_gt",
        "--eval_train", "--eval_use_ground_truth", "--use_gt_masks",
    }
    if forbidden.intersection(command):
        raise ValueError("official command contains forbidden input flags")
    required = {
        "--eval", "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
        "--eval_use_rec_hierarchical_reranker_scores",
    }
    if not required.issubset(command):
        raise ValueError("official command is missing V99 runtime flags")
    for flag, expected in (
            ("--joint_det", args.backbone_joint_training),
            ("--butd_cls", args.inference_uses_ground_truth)):
        if (flag in command) is not expected:
            raise ValueError(
                "official command provenance changed for {}".format(flag)
            )


def _validate_existing_result(result, args, artifacts, command):
    if (result.get("schema") != RESULT_SCHEMA
            or result.get("dataset") != args.dataset
            or result.get("sample_count") != args.expected_sample_count
            or result.get("dataset_only") is not (
                not args.backbone_joint_training
            )
            or result.get("joint_training") is not args.backbone_joint_training
            or result.get("inference_uses_ground_truth") is not (
                args.inference_uses_ground_truth
            )
            or result.get("artifacts") != artifacts
            or result.get("command") != command):
        raise ValueError("existing official result contract changed")
    receipt = result.get("eval_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("existing official result lacks eval receipt")
    path = Path(receipt.get("path", "")).expanduser().resolve()
    if (not path.is_file() or _sha256(path) != receipt.get("sha256")
            or _load_json(path, "evaluation receipt")
            != result.get("metrics")):
        raise ValueError("existing official evaluation receipt changed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("scanrefer", "nr3d", "sr3d"), required=True
    )
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--hierarchical-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone-joint-training", action="store_true")
    parser.add_argument("--inference-uses-ground-truth", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_sample_count <= 0 or not 1 <= args.master_port <= 65535:
        parser.error("sample count and master port are invalid")
    root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not (root / "train_dist_mod.py").is_file():
        raise ValueError("project root does not contain train_dist_mod.py")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "backbone": _snapshot(args.backbone_checkpoint, "backbone"),
        "parent": _snapshot(args.parent_artifact, "parent"),
        "geometry": _snapshot(args.geometry_artifact, "geometry"),
        "hierarchical": _snapshot(
            args.hierarchical_artifact, "hierarchical"
        ),
    }
    command = _build_command(args, artifacts, output)
    _validate_command(command, args)
    result_path = output / "official_result.json"
    if result_path.is_file():
        result = _validate_existing_result(
            _load_json(result_path, "official result"),
            args, artifacts, command,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    launch_path = output / "evaluation_launch.json"
    launch = {
        "schema": "mcln-dataset-v99-evaluation-launch-v1",
        "dataset": args.dataset,
        "sample_count": args.expected_sample_count,
        "dataset_only": not args.backbone_joint_training,
        "joint_training": args.backbone_joint_training,
        "inference_uses_ground_truth": args.inference_uses_ground_truth,
        "artifacts": artifacts,
        "command": command,
    }
    if launch_path.is_file():
        if _load_json(launch_path, "evaluation launch") != launch:
            raise ValueError("staged official evaluation launch changed")
    else:
        _exclusive_json(launch_path, launch)

    receipts_before = _metric_receipts(output)
    completed = subprocess.run(command, cwd=str(root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "official V99 evaluation exited {}".format(completed.returncode)
        )
    receipt = _new_metric_receipt(output, receipts_before)
    os.chmod(str(receipt), 0o444)
    metrics = _load_json(receipt, "evaluation receipt")
    if metrics.get("sample_count") != args.expected_sample_count:
        raise ValueError("official evaluation sample count changed")
    result = dict(launch)
    result["schema"] = RESULT_SCHEMA
    result["eval_receipt"] = {
        "path": str(receipt),
        "sha256": _sha256(receipt),
        "size": receipt.stat().st_size,
    }
    result["metrics"] = metrics
    _exclusive_json(result_path, result)
    _validate_existing_result(result, args, artifacts, command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
