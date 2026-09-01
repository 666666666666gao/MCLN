#!/usr/bin/env python3
"""Build the one preregistered Nr3D E57/E69 interpolation for eval only."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_with_sha(path, expected_sha256, label):
    """Hash and deserialize the exact same opened file description."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise SystemExit("{} checkpoint SHA-256 mismatch".format(label))
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu")
    return checkpoint, observed


def finite_tensor(tensor):
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all().item())
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--other-weight", required=True, type=float)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--other-sha256", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not math.isclose(
            args.other_weight, 0.25, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("this one-shot builder requires other-weight=0.25")
    if os.path.realpath(args.output) == os.path.realpath(args.manifest):
        raise SystemExit("output checkpoint and manifest must differ")
    if os.path.exists(args.output) or os.path.exists(args.manifest):
        raise SystemExit("refusing to overwrite an existing candidate artifact")

    base, observed_base_sha = load_checkpoint_with_sha(
        args.base, args.base_sha256, "base"
    )
    other, observed_other_sha = load_checkpoint_with_sha(
        args.other, args.other_sha256, "other"
    )
    if int(base.get("epoch", -1)) != 57 or int(other.get("epoch", -1)) != 69:
        raise SystemExit("checkpoint epoch contract mismatch")
    if "model" not in base or "model" not in other:
        raise SystemExit("checkpoint does not contain a model state")
    if set(base["model"]) != set(other["model"]):
        raise SystemExit("model state keys differ")

    averaged = {}
    float_tensor_count = 0
    preserved_tensor_count = 0
    changed_float_tensor_count = 0
    for name in sorted(base["model"]):
        left = base["model"][name]
        right = other["model"][name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise SystemExit("non-tensor model state: {}".format(name))
        if left.shape != right.shape or left.dtype != right.dtype:
            raise SystemExit("tensor contract differs: {}".format(name))
        if not finite_tensor(left) or not finite_tensor(right):
            raise SystemExit("non-finite input tensor: {}".format(name))
        if left.is_floating_point() or left.is_complex():
            result = left.mul(1.0 - args.other_weight).add(
                right, alpha=args.other_weight
            )
            float_tensor_count += 1
            if not torch.equal(left, right):
                changed_float_tensor_count += 1
        else:
            # Integer counters are not parameters and cannot be interpolated.
            result = left.clone()
            preserved_tensor_count += 1
        if not finite_tensor(result):
            raise SystemExit("non-finite averaged tensor: {}".format(name))
        averaged[name] = result

    if changed_float_tensor_count == 0:
        raise SystemExit("the two checkpoints have no differing float tensors")

    provenance = {
        "schema": "mcln-nr3d-v99-e57-e69-weight-average-v1",
        "base_checkpoint": os.path.realpath(args.base),
        "base_epoch": 57,
        "base_sha256": observed_base_sha,
        "base_weight": 1.0 - args.other_weight,
        "other_checkpoint": os.path.realpath(args.other),
        "other_epoch": 69,
        "other_sha256": observed_other_sha,
        "other_weight": args.other_weight,
        "float_tensor_count": float_tensor_count,
        "changed_float_tensor_count": changed_float_tensor_count,
        "preserved_nonfloat_tensor_count": preserved_tensor_count,
        "purpose": "single_preregistered_eval_only_candidate",
    }
    base["model"] = averaged
    base["epoch"] = 57
    # A weight interpolation has no mathematically corresponding optimizer
    # moments or scheduler state. Omitting both makes accidental full-state
    # training resume fail closed while retaining eval/model-only loading.
    base.pop("optimizer", None)
    base.pop("scheduler", None)
    base["evaluation_only"] = True
    base["weight_average_provenance"] = provenance

    output_dir = os.path.dirname(os.path.realpath(args.output))
    manifest_dir = os.path.dirname(os.path.realpath(args.manifest))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    temporary_output = args.output + ".tmp.{}".format(os.getpid())
    try:
        torch.save(base, temporary_output)
        with open(temporary_output, "rb") as handle:
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if a concurrent writer
        # claimed the destination after the early readability check.
        os.link(temporary_output, args.output)
        os.unlink(temporary_output)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_output):
            os.unlink(temporary_output)

    provenance["output_checkpoint"] = os.path.realpath(args.output)
    provenance["output_sha256"] = sha256_file(args.output)
    temporary_manifest = os.path.join(
        manifest_dir,
        ".{}.tmp.{}".format(os.path.basename(args.manifest), os.getpid()),
    )
    try:
        with open(temporary_manifest, "w") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_manifest, args.manifest)
        os.unlink(temporary_manifest)
        directory_fd = os.open(manifest_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_manifest):
            os.unlink(temporary_manifest)
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
