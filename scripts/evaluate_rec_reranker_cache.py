#!/usr/bin/env python
"""Evaluate one train-selected REC reranker on a bound validation cache."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rec_geometry_cache import load_bound_candidate_cache
from scripts.train_rec_reranker import (
    evaluate_reranker,
    load_reranker_artifact,
    normalize_backbone_config,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_binding(artifact, manifest):
    exact_fields = (
        "feature_names",
        "candidate_rule",
        "checkpoint_sha256",
        "model_inputs",
        "target_iou_policy",
    )
    if any(artifact.get(key) != manifest.get(key) for key in exact_fields):
        raise ValueError("reranker artifact does not match val cache")
    if normalize_backbone_config(artifact.get("backbone_config")) != (
            normalize_backbone_config(manifest.get("backbone_config"))):
        raise ValueError("reranker backbone does not match val cache")


def evaluate(args):
    artifact_path = Path(args.artifact).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("evaluation output already exists: {}".format(
            output
        ))
    rows, manifest, binding = load_bound_candidate_cache(
        args.val_cache, "val"
    )
    model, artifact = load_reranker_artifact(
        artifact_path, device=args.device
    )
    _validate_binding(artifact, manifest)
    metrics = evaluate_reranker(
        model,
        rows,
        artifact["feature_mean"],
        artifact["feature_std"],
        batch_size=args.batch_size,
        device=args.device,
        reranker_weight=artifact["reranker_weight"],
    )
    record = {
        "schema": "rec-reranker-cache-evaluation-v1",
        "selection_uses_validation": False,
        "sample_count": len(rows),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "artifact_epoch": artifact["epoch"],
        "reranker_weight": artifact["reranker_weight"],
        "base_cache_path": binding["path"],
        "base_cache_content_sha256": binding["content_sha256"],
        "base_cache_manifest_sha256": binding["manifest_sha256"],
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    record = evaluate(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
