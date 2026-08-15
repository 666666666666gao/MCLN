#!/usr/bin/env python
"""Materialize train-only deployable mask-policy features for V102."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_joint_box_mask import (
    MASK_POLICY_FEATURE_SCHEMA_VERSION,
    build_mask_policy_feature_names,
    compute_mask_policy_inference_features,
)
from models.rec_mask_geometry import normalize_mcln_mask_logits


CACHE_SCHEMA = "rec-mask-policy-feature-cache-v1"
CACHE_VERSION = 1
CANDIDATE_COUNT = 16
FEATURE_DIM = 52
APPROVED_ROW_KEYS = frozenset({
    "dataset_index", "scan_id", "target_id", "query_indices",
    "candidate_valid", "mask_policy_features",
})
FORBIDDEN_ROW_KEYS = frozenset({
    "gt_masks", "candidate_ious", "geometry_ious", "mask_ious",
    "target_iou", "target_ious", "box_label_mask", "center_label",
    "size_gts", "threshold_labels",
})


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n").encode("ascii")
    descriptor = os.open(
        str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("manifest write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(str(temporary), str(path))


def validate_feature_row(row, expected_index):
    if not isinstance(row, dict):
        raise ValueError("mask feature row must be an object")
    if set(row).intersection(FORBIDDEN_ROW_KEYS):
        raise ValueError("mask feature row contains forbidden supervision")
    if set(row) != APPROVED_ROW_KEYS:
        raise ValueError("mask feature row fields differ from schema")
    if (not isinstance(row.get("dataset_index"), int)
            or isinstance(row.get("dataset_index"), bool)
            or row["dataset_index"] != int(expected_index)):
        raise ValueError("mask feature dataset index is not contiguous")
    if not isinstance(row.get("scan_id"), str) or not row["scan_id"]:
        raise ValueError("mask feature scan identity is invalid")
    if (not isinstance(row.get("target_id"), int)
            or isinstance(row.get("target_id"), bool)
            or row["target_id"] < 0):
        raise ValueError("mask feature target identity is invalid")
    query_indices = row.get("query_indices")
    valid = row.get("candidate_valid")
    features = row.get("mask_policy_features")
    if (not isinstance(query_indices, torch.Tensor)
            or query_indices.dtype != torch.long
            or query_indices.device.type != "cpu"
            or query_indices.shape != (CANDIDATE_COUNT,)):
        raise ValueError("mask feature query indices are invalid")
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.device.type != "cpu"
            or valid.shape != (CANDIDATE_COUNT,)
            or not bool(valid.any().item())):
        raise ValueError("mask feature candidate validity is invalid")
    if (not isinstance(features, torch.Tensor)
            or features.dtype != torch.float32
            or features.device.type != "cpu"
            or features.shape != (CANDIDATE_COUNT, FEATURE_DIM)
            or not bool(torch.isfinite(features).all().item())):
        raise ValueError("mask policy features are invalid")
    if not torch.equal(features[~valid], torch.zeros_like(features[~valid])):
        raise ValueError("invalid candidates must have zero mask features")
    return row


def validate_feature_manifest(manifest, require_full=False):
    if not isinstance(manifest, dict):
        raise ValueError("mask feature manifest must be an object")
    if (manifest.get("schema") != CACHE_SCHEMA
            or manifest.get("version") != CACHE_VERSION
            or manifest.get("split") != "train"
            or manifest.get("complete") is not True
            or manifest.get("validation_data_accessed") is not False
            or manifest.get("inference_uses_ground_truth") is not False):
        raise ValueError("mask feature manifest provenance is invalid")
    counts = [manifest.get(name) for name in (
        "sample_count", "dataset_size", "source_dataset_size"
    )]
    if (any(not isinstance(value, int) or isinstance(value, bool)
            or value <= 0 for value in counts)
            or counts[0] != counts[1] or counts[1] > counts[2]
            or (require_full and counts[1] != counts[2])):
        raise ValueError("mask feature cache coverage is invalid")
    if (manifest.get("feature_schema_version")
            != MASK_POLICY_FEATURE_SCHEMA_VERSION
            or manifest.get("candidate_count") != CANDIDATE_COUNT
            or manifest.get("feature_dim") != FEATURE_DIM
            or manifest.get("feature_names")
            != build_mask_policy_feature_names()):
        raise ValueError("mask feature schema changed")
    for name in (
            "checkpoint_sha256", "base_cache_manifest_sha256",
            "joint_label_manifest_sha256", "joint_label_content_sha256"):
        value = manifest.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)):
            raise ValueError("mask feature {} is invalid".format(name))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("mask feature shards are missing")
    total = 0
    for index, shard in enumerate(shards):
        if (not isinstance(shard, dict)
                or shard.get("name") != "shard_{:06d}.pt".format(index)
                or not isinstance(shard.get("row_count"), int)
                or shard["row_count"] <= 0
                or not isinstance(shard.get("sha256"), str)
                or len(shard["sha256"]) != 64):
            raise ValueError("mask feature shard manifest is invalid")
        total += shard["row_count"]
    if total != counts[0]:
        raise ValueError("mask feature shard coverage changed")
    content = manifest.get("content_sha256")
    expected = canonical_json_sha256({
        key: value for key, value in manifest.items()
        if key != "content_sha256"
    })
    if content != expected:
        raise ValueError("mask feature manifest content digest changed")
    return manifest


def load_mask_policy_feature_cache(output_dir, require_full=True):
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = validate_feature_manifest(
        json.loads((output_dir / "manifest.json").read_text(encoding="ascii")),
        require_full=require_full,
    )
    rows = []
    for shard in manifest["shards"]:
        path = output_dir / shard["name"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("mask feature shard SHA-256 changed")
        payload = torch.load(path, map_location="cpu")
        if (not isinstance(payload, dict)
                or payload.get("schema") != CACHE_SCHEMA
                or not isinstance(payload.get("rows"), list)
                or len(payload["rows"]) != shard["row_count"]):
            raise ValueError("mask feature shard payload is invalid")
        for row in payload["rows"]:
            validate_feature_row(row, len(rows))
            rows.append(row)
    if len(rows) != manifest["sample_count"]:
        raise RuntimeError("mask feature cache row count changed")
    return rows, manifest


def append_shard(output_dir, manifest, rows):
    if not rows:
        raise ValueError("cannot publish an empty mask feature shard")
    output_dir = Path(output_dir)
    index = len(manifest["shards"])
    name = "shard_{:06d}.pt".format(index)
    path = output_dir / name
    temporary = path.with_name(path.name + ".tmp")
    start = int(manifest["sample_count"])
    for offset, row in enumerate(rows):
        validate_feature_row(row, start + offset)
    torch.save({"schema": CACHE_SCHEMA, "rows": rows}, temporary)
    os.replace(str(temporary), str(path))
    updated = dict(manifest)
    updated["shards"] = list(manifest["shards"]) + [{
        "name": name,
        "row_count": len(rows),
        "sha256": sha256_file(path),
    }]
    updated["sample_count"] = start + len(rows)
    atomic_json(output_dir / "manifest.json", updated)
    return updated


def assert_runtime_candidate_identity(fresh, base_rows, indices,
                                      scan_ids, target_ids):
    query_indices = fresh.get("query_indices")
    valid = fresh.get("valid_mask")
    if (not isinstance(query_indices, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or query_indices.shape != valid.shape
            or query_indices.shape != (len(indices), CANDIDATE_COUNT)):
        raise ValueError("fresh candidate identity tensors are malformed")
    for batch_index, dataset_index in enumerate(indices):
        base = base_rows[dataset_index]
        target = int(torch.as_tensor(target_ids[batch_index]).item())
        if (base["dataset_index"] != dataset_index
                or base["scan_id"] != str(scan_ids[batch_index])
                or int(base["target_id"]) != target
                or not torch.equal(
                    query_indices[batch_index].detach().cpu().long(),
                    base["query_indices"],
                )
                or not torch.equal(
                    valid[batch_index].detach().cpu().bool(),
                    base["valid_mask"],
                )):
            raise RuntimeError(
                "fresh runtime candidate identity differs at row {}; "
                "base_identity={}; fresh_identity={}; base_queries={}; "
                "fresh_queries={}; base_valid={}; fresh_valid={}".format(
                    dataset_index,
                    (base["scan_id"], int(base["target_id"])),
                    (str(scan_ids[batch_index]), target),
                    base["query_indices"].tolist(),
                    query_indices[batch_index].detach().cpu().long().tolist(),
                    base["valid_mask"].tolist(),
                    valid[batch_index].detach().cpu().bool().tolist(),
                )
            )
    return True


def build_frozen_candidate_identity(base_rows, indices, scan_ids,
                                    target_ids, device):
    """Rehydrate the exact cache-bound query axis without re-ranking."""
    if len(indices) != len(scan_ids):
        raise ValueError("frozen candidate batch identities do not align")
    rows = []
    for batch_index, dataset_index in enumerate(indices):
        base = base_rows[dataset_index]
        target = int(torch.as_tensor(target_ids[batch_index]).item())
        if (base["dataset_index"] != dataset_index
                or base["scan_id"] != str(scan_ids[batch_index])
                or int(base["target_id"]) != target):
            raise RuntimeError(
                "frozen candidate source identity differs at row {}".format(
                    dataset_index
                )
            )
        rows.append(base)
    query_indices = torch.stack([
        row["query_indices"] for row in rows
    ]).to(device=device, dtype=torch.long)
    valid = torch.stack([
        row["valid_mask"] for row in rows
    ]).to(device=device, dtype=torch.bool)
    if (query_indices.shape != (len(rows), CANDIDATE_COUNT)
            or valid.shape != query_indices.shape
            or not bool(valid.any(dim=1).all().item())
            or bool((query_indices < 0).any().item())):
        raise ValueError("frozen candidate query axis is malformed")
    return {"query_indices": query_indices, "valid_mask": valid}


def compute_batch_mask_features(end_points, candidate_batch):
    query_indices = candidate_batch["query_indices"]
    valid = candidate_batch["valid_mask"].bool()
    rows = []
    for batch_index in range(query_indices.shape[0]):
        text, query, _fused, alpha = normalize_mcln_mask_logits(
            end_points, batch_index, query_indices[batch_index]
        )
        features = compute_mask_policy_inference_features(
            text.unsqueeze(0), query.unsqueeze(0), alpha,
            valid[batch_index].unsqueeze(0),
        )[0]
        rows.append(features.detach().cpu().float())
    return torch.stack(rows)


def build_feature_row(dataset_index, batch_data, candidate_batch,
                      feature_batch, batch_index):
    row = {
        "dataset_index": int(dataset_index),
        "scan_id": str(batch_data["scan_ids"][batch_index]),
        "target_id": int(torch.as_tensor(
            batch_data["target_id"][batch_index]
        ).item()),
        "query_indices": candidate_batch["query_indices"][batch_index]
        .detach().cpu().long().contiguous(),
        "candidate_valid": candidate_batch["valid_mask"][batch_index]
        .detach().cpu().bool().contiguous(),
        "mask_policy_features": feature_batch[batch_index]
        .detach().cpu().float().contiguous(),
    }
    return validate_feature_row(row, dataset_index)


def set_deterministic(seed, device):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--joint-label-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    for name in ("batch_size", "shard_size"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def run_extraction(args):
    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    base_cache = Path(args.base_cache).expanduser().resolve()
    joint_label_cache = Path(args.joint_label_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("mask feature output already exists")
    if (not data_root.is_dir() or not checkpoint_path.is_file()
            or not base_cache.is_dir() or not joint_label_cache.is_dir()):
        raise ValueError("V102 feature extraction input is missing")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    set_deterministic(args.seed, device)

    from scripts.cache_scanrefer_rec_candidates import (
        _build_dataset, _build_loader, _load_frozen_model,
        _move_batch_to_device, _normalized_data_root, _prepare_model_config,
    )
    from scripts.cache_scanrefer_joint_box_mask import (
        validate_joint_cache_manifest,
    )
    from scripts.rec_geometry_cache import load_bound_candidate_cache
    from train_dist_mod import TrainTester

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("backbone checkpoint must contain an object")
    config = _prepare_model_config(
        checkpoint, _normalized_data_root(data_root)
    )
    base_rows, base_manifest, _binding = load_bound_candidate_cache(
        base_cache, "train"
    )
    joint_manifest_path = joint_label_cache / "manifest.json"
    joint_manifest = validate_joint_cache_manifest(
        json.loads(joint_manifest_path.read_text(encoding="ascii")), "train"
    )
    base_manifest_sha = sha256_file(base_cache / "manifest.json")
    if (joint_manifest.get("base_cache_manifest_sha256")
            != base_manifest_sha):
        raise ValueError("joint labels bind a different base cache")
    checkpoint_sha = sha256_file(checkpoint_path)
    if ({checkpoint_sha, base_manifest.get("checkpoint_sha256"),
         joint_manifest.get("checkpoint_sha256")} != {checkpoint_sha}):
        raise ValueError("V102 caches bind different backbone checkpoints")
    dataset = _build_dataset(config, "train")
    if len(dataset) != len(base_rows):
        raise ValueError("train dataset and base cache size differ")
    limit = min(len(dataset), args.limit or len(dataset))
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "split": "train",
        "complete": False,
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "sample_count": 0,
        "dataset_size": int(limit),
        "source_dataset_size": int(len(dataset)),
        "feature_schema_version": MASK_POLICY_FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "feature_names": build_mask_policy_feature_names(),
        "candidate_count": CANDIDATE_COUNT,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "base_cache_manifest_sha256": base_manifest_sha,
        "joint_label_manifest_sha256": sha256_file(joint_manifest_path),
        "joint_label_content_sha256": joint_manifest["content_sha256"],
        "seed": int(args.seed),
        "shards": [],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    model = _load_frozen_model(checkpoint, config, device)
    del checkpoint
    loader = _build_loader(dataset, 0, limit, args, device)
    cursor = 0
    pending = []
    for batch_data in loader:
        batch_data = _move_batch_to_device(batch_data, device)
        inputs = TrainTester._get_inputs(batch_data)
        inputs["train"] = False
        with torch.inference_mode():
            end_points = model(inputs)
            indices = list(range(cursor, cursor + len(batch_data["scan_ids"])))
            frozen = build_frozen_candidate_identity(
                base_rows, indices, batch_data["scan_ids"],
                batch_data["target_id"], end_points["last_center"].device,
            )
            features = compute_batch_mask_features(end_points, frozen)
        for batch_index, dataset_index in enumerate(indices):
            pending.append(build_feature_row(
                dataset_index, batch_data, frozen, features, batch_index
            ))
        cursor += len(indices)
        while len(pending) >= args.shard_size:
            manifest = append_shard(
                output_dir, manifest, pending[:args.shard_size]
            )
            del pending[:args.shard_size]
        if cursor % max(args.shard_size, args.batch_size) == 0:
            print("V102 mask features {}/{}".format(cursor, limit), flush=True)
    if pending:
        manifest = append_shard(output_dir, manifest, pending)
    if manifest["sample_count"] != limit:
        raise RuntimeError("V102 mask feature extraction ended incomplete")
    manifest["complete"] = True
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    atomic_json(output_dir / "manifest.json", manifest)
    validate_feature_manifest(manifest, require_full=(limit == len(dataset)))
    print(json.dumps({
        "output": str(output_dir),
        "sample_count": limit,
        "full": limit == len(dataset),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "content_sha256": manifest["content_sha256"],
    }, sort_keys=True), flush=True)
    return manifest


def main(argv=None):
    return run_extraction(parse_args(argv))


if __name__ == "__main__":
    main()
