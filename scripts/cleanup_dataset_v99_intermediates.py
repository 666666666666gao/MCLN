#!/usr/bin/env python
"""Remove only reconstructible V99 caches and non-selected run weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path, payload):
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
                raise OSError("cleanup receipt write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--pipeline-receipt", required=True)
    parser.add_argument("--backbone-run-dir")
    parser.add_argument("--keep-checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    root = Path(args.pipeline_root).expanduser().resolve()
    receipt_path = Path(args.pipeline_receipt).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    if not root.is_dir() or not _within(receipt_path, root):
        raise ValueError("pipeline receipt must be inside pipeline root")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "mcln-dataset-v99-pipeline-receipt-v1":
        raise ValueError("pipeline receipt schema is invalid")
    if output.exists():
        raise FileExistsError(str(output))
    if not _within(output.resolve(), root):
        raise ValueError("cleanup receipt must be inside pipeline root")

    removed = []
    for name in ("candidate_cache", "geometry_cache", "geometry_audit"):
        target = (root / name).resolve()
        if target.parent != root or target.name != name:
            raise ValueError("unsafe cache target")
        if target.is_dir():
            size = sum(
                entry.stat().st_size for entry in target.rglob("*")
                if entry.is_file() and not entry.is_symlink()
            )
            shutil.rmtree(str(target))
            removed.append({"path": str(target), "bytes": int(size)})

    kept = None
    if args.backbone_run_dir or args.keep_checkpoint:
        if not args.backbone_run_dir or not args.keep_checkpoint:
            raise ValueError(
                "backbone pruning requires both run dir and keep checkpoint"
            )
        run_dir = Path(args.backbone_run_dir).expanduser().resolve()
        keep = Path(args.keep_checkpoint).expanduser().resolve()
        if (not run_dir.is_dir() or not keep.is_file()
                or not _within(keep, run_dir)):
            raise ValueError("selected checkpoint is outside backbone run dir")
        kept = {"path": str(keep), "sha256": _sha256(keep)}
        receipt_backbone = receipt.get("artifacts", {}).get("backbone", {})
        if (receipt_backbone.get("path") != str(keep)
                or receipt_backbone.get("sha256") != kept["sha256"]):
            raise ValueError("selected checkpoint differs from pipeline receipt")
        for path in sorted(run_dir.rglob("*.pth")):
            resolved = path.resolve()
            if resolved == keep:
                continue
            entry = os.lstat(str(path))
            if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
                raise ValueError("unexpected non-regular checkpoint path")
            removed.append({
                "path": str(path),
                "bytes": int(entry.st_size),
                "sha256": _sha256(path),
            })
            os.unlink(str(path))

    result = {
        "schema": "mcln-dataset-v99-cleanup-receipt-v1",
        "pipeline_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
        },
        "pipeline_root": str(root),
        "kept_checkpoint": kept,
        "removed": removed,
        "removed_logical_bytes": sum(item["bytes"] for item in removed),
        "recoverability": (
            "candidate/geometry caches are reconstructible from the protected "
            "backbone; removed checkpoints were not the selected REC@0.25 best"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output, result)
    print(json.dumps({
        "output": str(output),
        "removed_count": len(removed),
        "removed_logical_bytes": result["removed_logical_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
