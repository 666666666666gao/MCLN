#!/usr/bin/env python
"""Archive an exact model-only checkpoint delta against a protected parent."""

from __future__ import print_function

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v133_receipt_utils import atomic_write_new_json, fsync_directory


COMPACT_SCHEMA = "mcln-model-only-checkpoint-delta-v1"
RECEIPT_SCHEMA = "mcln-model-only-checkpoint-delta-receipt-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_evidence_bytes(path, required_mode=None):
    path = Path(path).expanduser().absolute()
    path_entry = os.lstat(str(path))
    require(not stat.S_ISLNK(path_entry.st_mode),
            "checkpoint evidence must not be a symlink: {}".format(path))
    require(stat.S_ISREG(path_entry.st_mode),
            "checkpoint evidence must be a regular file: {}".format(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        opened_entry = os.fstat(descriptor)
        require(
            (opened_entry.st_dev, opened_entry.st_ino)
            == (path_entry.st_dev, path_entry.st_ino),
            "checkpoint evidence changed while opening: {}".format(path),
        )
        require(stat.S_ISREG(opened_entry.st_mode),
                "opened checkpoint evidence is not regular: {}".format(path))
        mode = stat.S_IMODE(opened_entry.st_mode)
        if required_mode is not None:
            require(mode == required_mode,
                    "checkpoint evidence mode changed: {}".format(path))
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        final_entry = os.fstat(descriptor)
        require(
            (
                final_entry.st_dev,
                final_entry.st_ino,
                final_entry.st_size,
                final_entry.st_mtime_ns,
                final_entry.st_ctime_ns,
            )
            == (
                opened_entry.st_dev,
                opened_entry.st_ino,
                opened_entry.st_size,
                opened_entry.st_mtime_ns,
                opened_entry.st_ctime_ns,
            ),
            "checkpoint evidence changed while reading: {}".format(path),
        )
        require(len(payload) == opened_entry.st_size,
                "checkpoint evidence read was incomplete: {}".format(path))
    finally:
        os.close(descriptor)
    evidence = {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": int(opened_entry.st_size),
        "mode": format(mode, "04o"),
    }
    return evidence, payload


def snapshot(path, required_mode=None):
    evidence, _ = read_evidence_bytes(path, required_mode=required_mode)
    return evidence


def exact_model_state(checkpoint, label):
    require(isinstance(checkpoint, dict),
            "{} checkpoint is not a mapping".format(label))
    state = checkpoint.get("model")
    require(isinstance(state, dict),
            "{} checkpoint has no model state".format(label))
    exact = {}
    for name, value in state.items():
        require(isinstance(name, str),
                "{} model key is not a string".format(label))
        require(name not in exact,
                "{} model key is duplicated: {}".format(label, name))
        require(isinstance(value, torch.Tensor),
                "{} model state is not tensor-only: {}".format(label, name))
        require(value.layout == torch.strided,
                "{} model tensor is not dense: {}".format(label, name))
        exact[name] = value.detach().cpu()
    require(exact, "{} model state is empty".format(label))
    return exact


def tensor_bytes(tensor):
    contiguous = tensor.detach().cpu().contiguous()
    require(not contiguous.is_quantized,
            "quantized model tensors are not supported")
    byte_count = contiguous.numel() * contiguous.element_size()
    if byte_count == 0:
        return b""
    return ctypes.string_at(contiguous.data_ptr(), byte_count)


def model_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor_bytes(tensor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def build_delta(parent_state, child_state):
    overrides = {}
    changed_existing = 0
    added = 0
    for name, child_tensor in child_state.items():
        parent_tensor = parent_state.get(name)
        if parent_tensor is None:
            added += 1
            overrides[name] = child_tensor.clone()
            continue
        if (
            parent_tensor.shape != child_tensor.shape
            or parent_tensor.dtype != child_tensor.dtype
            or not torch.equal(parent_tensor, child_tensor)
        ):
            changed_existing += 1
            overrides[name] = child_tensor.clone()
    removed = tuple(sorted(set(parent_state) - set(child_state)))
    return overrides, removed, changed_existing, added


def reconstruct(parent_state, compact):
    require(compact.get("schema") == COMPACT_SCHEMA,
            "unexpected compact checkpoint schema")
    overrides = compact.get("model_overrides")
    removed = compact.get("removed_model_keys")
    require(isinstance(overrides, dict) and isinstance(removed, (list, tuple)),
            "compact checkpoint payload is incomplete")
    state = dict(parent_state)
    for name in removed:
        require(isinstance(name, str) and name in state,
                "compact checkpoint removes an unknown model key")
        del state[name]
    for name, tensor in overrides.items():
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor),
                "compact checkpoint override is invalid")
        state[name] = tensor.detach().cpu()
    return state


def atomic_write_new_torch(payload, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fsync_directory(output.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb+") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, str(output))
        fsync_directory(output.parent)
        os.unlink(temporary)
        fsync_directory(output.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
            fsync_directory(output.parent)
    return output


def load_checkpoint_payload(payload, label, allow_legacy_trusted_pickle):
    stream = io.BytesIO(payload)
    if "weights_only" in inspect.signature(torch.load).parameters:
        checkpoint = torch.load(
            stream, map_location="cpu", weights_only=True
        )
    else:
        require(
            allow_legacy_trusted_pickle,
            "{} requires --allow-legacy-trusted-pickle because this "
            "PyTorch cannot perform weights-only loading".format(label),
        )
        checkpoint = torch.load(stream, map_location="cpu")
    require(isinstance(checkpoint, dict),
            "{} checkpoint is not a mapping".format(label))
    return checkpoint


def load_checkpoint_evidence(
    path, label, required_mode=None, allow_legacy_trusted_pickle=False
):
    evidence, payload = read_evidence_bytes(
        path, required_mode=required_mode
    )
    checkpoint = load_checkpoint_payload(
        payload, label, allow_legacy_trusted_pickle
    )
    return evidence, checkpoint


def load_json_evidence(path, required_mode=None):
    evidence, payload = read_evidence_bytes(
        path, required_mode=required_mode
    )
    document = json.loads(payload.decode("utf-8"))
    require(isinstance(document, dict), "JSON evidence is not a mapping")
    return evidence, document


def build(args):
    output_path = Path(args.output).expanduser().resolve(strict=False)
    receipt_output_path = Path(args.receipt).expanduser().resolve(strict=False)
    require(output_path != receipt_output_path,
            "compact checkpoint and receipt paths must differ")
    require(not os.path.lexists(str(output_path)),
            "compact checkpoint output already exists")
    require(not os.path.lexists(str(receipt_output_path)),
            "compact receipt output already exists")
    parent_snapshot, parent_checkpoint = load_checkpoint_evidence(
        args.parent,
        "parent",
        required_mode=0o444,
        allow_legacy_trusted_pickle=args.allow_legacy_trusted_pickle,
    )
    child_snapshot, child_checkpoint = load_checkpoint_evidence(
        args.child,
        "child",
        allow_legacy_trusted_pickle=args.allow_legacy_trusted_pickle,
    )
    require(parent_snapshot["path"] != child_snapshot["path"],
            "parent and child checkpoint paths must differ")
    require(parent_snapshot["sha256"] != child_snapshot["sha256"],
            "parent and child checkpoints must differ")
    require(parent_snapshot["sha256"] == args.expected_parent_sha256,
            "parent checkpoint SHA-256 changed")
    require(child_snapshot["sha256"] == args.expected_child_sha256,
            "child checkpoint SHA-256 changed")
    child_epoch = child_checkpoint.get("epoch")
    require(
        isinstance(child_epoch, int)
        and not isinstance(child_epoch, bool)
        and child_epoch >= 0,
        "child checkpoint epoch is invalid",
    )
    require(child_epoch == args.expected_child_epoch,
            "child checkpoint epoch changed")
    parent_state = exact_model_state(parent_checkpoint, "parent")
    child_state = exact_model_state(child_checkpoint, "child")
    overrides, removed, changed_existing, added = build_delta(
        parent_state, child_state
    )
    parent_digest = model_digest(parent_state)
    child_digest = model_digest(child_state)
    source_sha256 = {
        "scripts/archive_checkpoint_model_delta.py": sha256(__file__),
        "scripts/v133_receipt_utils.py": sha256(
            ROOT / "scripts" / "v133_receipt_utils.py"
        ),
    }
    compact_payload = {
        "schema": COMPACT_SCHEMA,
        "parent_checkpoint_sha256": parent_snapshot["sha256"],
        "child_checkpoint_sha256": child_snapshot["sha256"],
        "parent_model_sha256": parent_digest,
        "child_model_sha256": child_digest,
        "child_epoch": child_epoch,
        "model_overrides": overrides,
        "removed_model_keys": removed,
        "recovery_scope": "exact_model_state_for_evaluation_only",
        "source_sha256": source_sha256,
    }
    reconstructed = reconstruct(parent_state, compact_payload)
    require(set(reconstructed) == set(child_state),
            "compact reconstruction model keys differ")
    require(model_digest(reconstructed) == child_digest,
            "compact reconstruction model tensors differ")
    compact_path = atomic_write_new_torch(compact_payload, args.output)
    compact_snapshot = snapshot(compact_path, required_mode=0o444)
    published_snapshot, published_compact = load_checkpoint_evidence(
        compact_path,
        "published compact",
        required_mode=0o444,
        allow_legacy_trusted_pickle=args.allow_legacy_trusted_pickle,
    )
    require(published_snapshot == compact_snapshot,
            "published compact checkpoint snapshot changed")
    published_reconstruction = reconstruct(parent_state, published_compact)
    require(model_digest(published_reconstruction) == child_digest,
            "published compact checkpoint failed reconstruction")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "parent_checkpoint": parent_snapshot,
        "child_checkpoint": child_snapshot,
        "child_epoch": child_epoch,
        "compact_checkpoint": compact_snapshot,
        "parent_model_tensor_count": len(parent_state),
        "child_model_tensor_count": len(child_state),
        "override_tensor_count": len(overrides),
        "changed_existing_tensor_count": changed_existing,
        "added_tensor_count": added,
        "removed_tensor_count": len(removed),
        "parent_model_sha256": parent_digest,
        "child_model_sha256": child_digest,
        "full_checkpoint_bytes": child_snapshot["size"],
        "compact_checkpoint_bytes": compact_snapshot["size"],
        "bytes_saved": child_snapshot["size"] - compact_snapshot["size"],
        "reconstruction_verified": True,
        "recovery_scope": "exact_model_state_for_evaluation_only",
        "optimizer_scheduler_recovery": False,
        "source_sha256": source_sha256,
    }
    receipt_path = atomic_write_new_json(receipt, args.receipt)
    return receipt_path, compact_snapshot, receipt


def verify_archive(args):
    receipt_snapshot, receipt = load_json_evidence(
        args.receipt, required_mode=0o444
    )
    require(receipt.get("schema") == RECEIPT_SCHEMA,
            "unexpected compact receipt schema")
    parent_snapshot, parent_checkpoint = load_checkpoint_evidence(
        args.parent,
        "parent",
        required_mode=0o444,
        allow_legacy_trusted_pickle=args.allow_legacy_trusted_pickle,
    )
    compact_snapshot, compact = load_checkpoint_evidence(
        args.compact,
        "compact",
        required_mode=0o444,
        allow_legacy_trusted_pickle=args.allow_legacy_trusted_pickle,
    )
    require(parent_snapshot["sha256"]
            == receipt["parent_checkpoint"]["sha256"],
            "live parent checkpoint changed")
    require(compact_snapshot == receipt["compact_checkpoint"],
            "live compact checkpoint changed")
    require(compact.get("parent_checkpoint_sha256")
            == parent_snapshot["sha256"],
            "compact checkpoint parent binding changed")
    require(compact.get("child_checkpoint_sha256")
            == receipt["child_checkpoint"]["sha256"],
            "compact checkpoint child binding changed")
    require(compact.get("child_epoch") == receipt.get("child_epoch"),
            "compact checkpoint child epoch changed")
    require(compact.get("source_sha256") == receipt.get("source_sha256"),
            "compact checkpoint source provenance changed")
    parent_state = exact_model_state(parent_checkpoint, "parent")
    require(model_digest(parent_state) == receipt["parent_model_sha256"],
            "live parent model state changed")
    reconstructed = reconstruct(parent_state, compact)
    require(set(reconstructed) == set(compact.get("model_overrides", {}))
            | (set(parent_state) - set(compact.get("removed_model_keys", ()))),
            "reconstructed model key set is inconsistent")
    require(model_digest(reconstructed) == receipt["child_model_sha256"],
            "compact checkpoint no longer reconstructs the child model")
    require(compact.get("child_model_sha256")
            == receipt["child_model_sha256"],
            "compact checkpoint child model digest changed")
    return (
        receipt_snapshot,
        compact_snapshot,
        receipt,
        reconstructed,
        compact["child_epoch"],
    )


def restore(args):
    output_path = Path(args.output).expanduser().resolve(strict=False)
    require(not os.path.lexists(str(output_path)),
            "restored checkpoint output already exists")
    (
        receipt_snapshot,
        compact_snapshot,
        receipt,
        reconstructed,
        verified_child_epoch,
    ) = verify_archive(args)
    restored_payload = {
        "model": reconstructed,
        "epoch": verified_child_epoch,
        "archive_provenance": {
            "schema": RECEIPT_SCHEMA,
            "receipt_sha256": receipt_snapshot["sha256"],
            "compact_checkpoint_sha256": compact_snapshot["sha256"],
            "original_child_checkpoint_sha256": receipt[
                "child_checkpoint"
            ]["sha256"],
            "recovery_scope": "exact_model_state_for_evaluation_only",
            "optimizer_scheduler_recovery": False,
        },
    }
    restored_path = atomic_write_new_torch(restored_payload, output_path)
    restored_snapshot = snapshot(restored_path, required_mode=0o444)
    published_restored_snapshot, restored_checkpoint = (
        load_checkpoint_evidence(
            restored_path,
            "restored",
            required_mode=0o444,
            allow_legacy_trusted_pickle=args.allow_legacy_trusted_pickle,
        )
    )
    require(published_restored_snapshot == restored_snapshot,
            "restored checkpoint snapshot changed")
    restored_state = exact_model_state(restored_checkpoint, "restored")
    require(model_digest(restored_state) == receipt["child_model_sha256"],
            "restored checkpoint model digest differs from child")
    return restored_snapshot, receipt


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--parent", required=True)
    build_parser.add_argument("--child", required=True)
    build_parser.add_argument("--expected-parent-sha256", required=True)
    build_parser.add_argument("--expected-child-sha256", required=True)
    build_parser.add_argument(
        "--expected-child-epoch", type=int, required=True
    )
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--receipt", required=True)
    build_parser.add_argument(
        "--allow-legacy-trusted-pickle",
        action="store_true",
        help=(
            "allow trusted local checkpoints on PyTorch versions without "
            "weights-only loading"
        ),
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--parent", required=True)
    verify_parser.add_argument("--compact", required=True)
    verify_parser.add_argument("--receipt", required=True)
    verify_parser.add_argument(
        "--allow-legacy-trusted-pickle",
        action="store_true",
        help=(
            "allow trusted local checkpoints on PyTorch versions without "
            "weights-only loading"
        ),
    )
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--parent", required=True)
    restore_parser.add_argument("--compact", required=True)
    restore_parser.add_argument("--receipt", required=True)
    restore_parser.add_argument("--output", required=True)
    restore_parser.add_argument(
        "--allow-legacy-trusted-pickle",
        action="store_true",
        help=(
            "allow trusted local checkpoints on PyTorch versions without "
            "weights-only loading"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "build":
        receipt_path, compact_snapshot, receipt = build(args)
        result = {
            "mode": "build",
            "receipt": str(receipt_path),
            "receipt_sha256": sha256(receipt_path),
            "compact": compact_snapshot,
            "child_model_sha256": receipt["child_model_sha256"],
            "override_tensor_count": receipt["override_tensor_count"],
        }
    elif args.mode == "verify":
        receipt_snapshot, compact_snapshot, receipt, _, _ = verify_archive(args)
        result = {
            "mode": "verify",
            "receipt": receipt_snapshot,
            "compact": compact_snapshot,
            "child_model_sha256": receipt["child_model_sha256"],
            "override_tensor_count": receipt["override_tensor_count"],
        }
    else:
        restored_snapshot, receipt = restore(args)
        result = {
            "mode": "restore",
            "restored": restored_snapshot,
            "child_model_sha256": receipt["child_model_sha256"],
            "original_child_checkpoint_sha256": receipt[
                "child_checkpoint"
            ]["sha256"],
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
