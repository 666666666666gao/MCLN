#!/usr/bin/env python3
"""Build a full-state checkpoint with optimizer current LRs restored.

This tool does not alter model weights, optimizer moments, scheduler phase,
base learning rates, or milestones.  It is intended for a checkpoint whose
current learning rates were manually reduced before the scheduler milestone.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile

import torch


SCHEMA = "mcln-full-checkpoint-current-lr-restore-v1"


def _sha256_open_file(handle):
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path):
    with open(path, "rb") as handle:
        return _sha256_open_file(handle)


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _set_config_value(config, name, value):
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _require_close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0,
                        abs_tol=1e-15):
        raise ValueError("{} mismatch: {} != {}".format(
            label, actual, expected
        ))


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_overwrite(temp_path, final_path):
    if os.path.exists(final_path):
        raise FileExistsError("refusing to overwrite {}".format(final_path))
    os.link(temp_path, final_path)
    os.unlink(temp_path)
    _fsync_directory(os.path.dirname(final_path))


def _write_json_no_overwrite(payload, path):
    directory = os.path.dirname(path)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".lr_restore_manifest.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o444)
        _publish_no_overwrite(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-current-ratio", type=float, required=True)
    parser.add_argument("--expected-optimizer-groups", type=int, required=True)
    parser.add_argument("--expected-optimizer-states", type=int, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.output == args.manifest:
        raise ValueError("output and manifest must be different paths")
    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    manifest_path = os.path.abspath(args.manifest)
    if source in (output, manifest_path):
        raise ValueError("source, output, and manifest must be distinct")
    if not (0.0 < args.expected_current_ratio < 1.0):
        raise ValueError("expected-current-ratio must lie in (0,1)")
    if args.expected_epoch < 1:
        raise ValueError("expected-epoch must be positive")
    for path in (output, manifest_path):
        if os.path.exists(path):
            raise FileExistsError("refusing to overwrite {}".format(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Hash and load the exact same opened inode.
    with open(source, "rb") as handle:
        source_sha256 = _sha256_open_file(handle)
        if source_sha256 != args.source_sha256:
            raise ValueError("source checkpoint SHA-256 mismatch")
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu")

    if int(checkpoint.get("epoch", -1)) != args.expected_epoch:
        raise ValueError("source checkpoint epoch mismatch")
    for key in ("model", "optimizer", "scheduler", "config"):
        if key not in checkpoint:
            raise ValueError("source checkpoint is missing {}".format(key))
    optimizer = checkpoint["optimizer"]
    scheduler = checkpoint["scheduler"]
    config = checkpoint["config"]
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != args.expected_optimizer_groups:
        raise ValueError("optimizer parameter-group count mismatch")
    if not isinstance(states, dict) or len(states) != args.expected_optimizer_states:
        raise ValueError("optimizer state count mismatch")

    current_lrs = []
    initial_lrs = []
    for index, group in enumerate(groups):
        if "lr" not in group or "initial_lr" not in group:
            raise ValueError("optimizer group {} lacks LR provenance".format(index))
        current = float(group["lr"])
        initial = float(group["initial_lr"])
        if not (math.isfinite(current) and math.isfinite(initial)
                and current > 0.0 and initial > 0.0):
            raise ValueError("optimizer learning rates must be finite and positive")
        _require_close(
            current / initial,
            args.expected_current_ratio,
            "optimizer group {} current/initial ratio".format(index),
        )
        current_lrs.append(current)
        initial_lrs.append(initial)

    base_lrs = [float(value) for value in scheduler.get("base_lrs", [])]
    last_lrs = [float(value) for value in scheduler.get("_last_lr", [])]
    if len(base_lrs) != len(groups) or len(last_lrs) != len(groups):
        raise ValueError("scheduler learning-rate topology mismatch")
    for index, (base, initial, last, current) in enumerate(zip(
            base_lrs, initial_lrs, last_lrs, current_lrs)):
        _require_close(base, initial, "scheduler base LR {}".format(index))
        _require_close(last, current, "scheduler last LR {}".format(index))

    if _config_value(config, "lr_scheduler") != "step":
        raise ValueError("only a step-scheduler checkpoint is supported")
    if _config_value(config, "dataset") != "nr3d":
        raise ValueError("source checkpoint is not Nr3D")
    if not bool(_config_value(config, "joint_det", False)):
        raise ValueError("source checkpoint is not joint_det")
    if not bool(_config_value(config, "butd_cls", False)):
        raise ValueError("source checkpoint is not butd_cls")
    if not bool(_config_value(config, "use_source_choice_selector", False)):
        raise ValueError("source checkpoint does not enable the V99 selector")
    if bool(_config_value(config, "resume_current_lr_restore_applied", False)):
        raise ValueError("source checkpoint already records an LR restore")

    for group, restored in zip(groups, initial_lrs):
        group["lr"] = restored
    scheduler["_last_lr"] = list(initial_lrs)
    marker = {
        "schema": SCHEMA,
        "source_sha256": source_sha256,
        "source_epoch": args.expected_epoch,
        "expected_current_ratio": args.expected_current_ratio,
        "old_current_lrs": current_lrs,
        "restored_current_lrs": initial_lrs,
        "scheduler_last_epoch": int(scheduler.get("last_epoch", -1)),
        "scheduler_milestones": dict(scheduler.get("milestones", {})),
    }
    _set_config_value(config, "resume_current_lr_restore_applied", True)
    _set_config_value(config, "resume_current_lr_restore_receipt", marker)

    output_dir = os.path.dirname(output)
    descriptor, temp_output = tempfile.mkstemp(
        prefix=".lr_restored_checkpoint.", suffix=".tmp", dir=output_dir
    )
    os.close(descriptor)
    try:
        torch.save(checkpoint, temp_output)
        with open(temp_output, "rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temp_output, 0o444)
        output_sha256 = _sha256_path(temp_output)
        _publish_no_overwrite(temp_output, output)
    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)

    manifest = {
        "schema": SCHEMA,
        "source": {"path": source, "sha256": source_sha256},
        "output": {"path": output, "sha256": output_sha256},
        "epoch": args.expected_epoch,
        "model_state_modified": False,
        "optimizer_moments_modified": False,
        "optimizer_current_lrs_before": current_lrs,
        "optimizer_current_lrs_after": initial_lrs,
        "scheduler_base_lrs": base_lrs,
        "scheduler_last_epoch": int(scheduler.get("last_epoch", -1)),
        "scheduler_milestones": dict(scheduler.get("milestones", {})),
        "config_marker": marker,
        "resume_only": True,
    }
    _write_json_no_overwrite(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
