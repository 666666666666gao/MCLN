#!/usr/bin/env python
"""Materialize train-only query mask quality labels for the joint selector.

The cache deliberately stores labels, not deployable inference features.  The
base and geometry caches remain the canonical feature sources, while this
sidecar binds their identities to fresh frozen-backbone mask logits.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_joint_box_mask import (
    MASK_SOURCE_NAMES,
    JOINT_MASK_SCHEMA_VERSION,
    compress_point_mask_to_superpoints,
    compute_weighted_mask_candidate_targets,
)
from models.rec_mask_geometry import normalize_mcln_mask_logits


CACHE_SCHEMA = "rec-joint-box-mask-cache-v1"
DEFAULT_THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0)
CANDIDATE_COUNT = 16
VARIANT_COUNT = 7
SOURCE_COUNT = 3
RESERVE_BYTES = 4 * (1 << 30)
APPROVED_JOINT_ROW_KEYS = frozenset({
    "dataset_index",
    "scan_id",
    "target_id",
    "query_indices",
    "candidate_valid",
    "mask_ious",
})
INFERENCE_FORBIDDEN_KEYS = frozenset({
    "gt_masks",
    "center_label",
    "size_gts",
    "box_label_mask",
    "candidate_ious",
    "geometry_ious",
    "threshold_labels",
    "target_iou",
    "target_ious",
    "iou",
})


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def estimate_cache_capacity(sample_count, bytes_per_row, free_bytes,
                            reserve_bytes=RESERVE_BYTES):
    """Return a conservative materialization decision.

    ``projected_bytes`` is the usable free-space envelope after a one-GiB
    filesystem bookkeeping margin.  The actual decision retains a separate
    four-GiB reserve, so callers can report both values without silently
    consuming the reserve.
    """
    values = (
        (sample_count, "sample_count"),
        (bytes_per_row, "bytes_per_row"),
        (free_bytes, "free_bytes"),
        (reserve_bytes, "reserve_bytes"),
    )
    for value, name in values:
        if (not isinstance(value, int) or isinstance(value, bool)
                or value <= 0):
            raise ValueError("{} must be positive".format(name))
    required = int(sample_count) * int(bytes_per_row)
    projected = max(0, int(free_bytes) - (1 << 30))
    return {
        "sample_count": int(sample_count),
        "bytes_per_row": int(bytes_per_row),
        "required_bytes": required,
        "projected_bytes": projected,
        "free_bytes": int(free_bytes),
        "reserve_bytes": int(reserve_bytes),
        "can_materialize": int(free_bytes) >= required + int(reserve_bytes),
    }


def validate_joint_cache_manifest(manifest, expected_split="train"):
    """Validate the immutable train-only joint-label schema."""
    if not isinstance(manifest, dict):
        raise ValueError("joint cache manifest must be an object")
    if expected_split != "train":
        raise ValueError("joint cache validation only supports train")
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("joint cache schema is unsupported")
    if manifest.get("split") != "train":
        raise ValueError("joint cache must use the train split")
    if manifest.get("complete") is not True:
        raise ValueError("joint cache must be complete before loading")
    for key in ("sample_count", "dataset_size", "source_dataset_size"):
        value = manifest.get(key)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value <= 0):
            raise ValueError("joint cache {} is invalid".format(key))
    if len({manifest[key] for key in (
            "sample_count", "dataset_size", "source_dataset_size")}) != 1:
        raise ValueError("joint cache must cover the complete train source")
    if manifest.get("feature_schema_version") != "rec-query-v1":
        raise ValueError("joint cache feature schema is unsupported")
    if manifest.get("geometry_schema_version") != "rec-geometry-flat-v1":
        raise ValueError("joint cache geometry schema is unsupported")
    for key, expected in (
            ("candidate_count", CANDIDATE_COUNT),
            ("variant_count", VARIANT_COUNT),
            ("source_count", SOURCE_COUNT)):
        if manifest.get(key) != expected:
            raise ValueError("joint cache {} is incompatible".format(key))
    thresholds = manifest.get("thresholds")
    if thresholds is None or tuple(float(value) for value in thresholds) != DEFAULT_THRESHOLDS:
        raise ValueError("joint cache thresholds are incompatible")
    if list(manifest.get("source_names", ())) != list(MASK_SOURCE_NAMES):
        raise ValueError("joint cache source names are incompatible")
    for key in (
            "base_cache_manifest_sha256", "geometry_cache_manifest_sha256",
            "checkpoint_sha256"):
        value = manifest.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("joint cache {} is invalid".format(key))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("joint cache shards are missing")
    for index, name in enumerate(shards):
        if name != "shard_{:06d}.pt".format(index):
            raise ValueError("joint cache shard names are not contiguous")
    if manifest.get("validation_data_accessed") is not False:
        raise ValueError("joint cache must declare validation_data_accessed=false")
    return manifest


def _strict_int(value, name, minimum=0):
    if (not isinstance(value, int) or isinstance(value, bool)
            or value < minimum):
        raise ValueError("{} is invalid".format(name))
    return value


def validate_joint_cache_row(row, expected_index):
    """Validate one CPU label row and reject every inference-only field."""
    if not isinstance(row, dict):
        raise ValueError("joint cache row must be an object")
    forbidden = sorted(set(row).intersection(INFERENCE_FORBIDDEN_KEYS))
    if forbidden:
        raise ValueError(
            "joint cache row contains forbidden target/inference fields: {}"
            .format(", ".join(forbidden))
        )
    if set(row) != APPROVED_JOINT_ROW_KEYS:
        raise ValueError("joint cache row schema does not match approved keys")
    if _strict_int(row.get("dataset_index"), "dataset_index") != expected_index:
        raise ValueError("joint cache dataset indices are not contiguous")
    if not isinstance(row.get("scan_id"), str) or not row["scan_id"]:
        raise ValueError("joint cache scan_id is invalid")
    _strict_int(row.get("target_id"), "target_id")
    query_indices = row.get("query_indices")
    valid = row.get("candidate_valid")
    mask_ious = row.get("mask_ious")
    if (not isinstance(query_indices, torch.Tensor)
            or query_indices.dtype != torch.long
            or tuple(query_indices.shape) != (CANDIDATE_COUNT,)):
        raise ValueError("joint cache query_indices are invalid")
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or tuple(valid.shape) != (CANDIDATE_COUNT,)
            or not bool(valid.any().item())):
        raise ValueError("joint cache candidate_valid is invalid")
    if (not isinstance(mask_ious, torch.Tensor)
            or not mask_ious.is_floating_point()
            or tuple(mask_ious.shape) != (CANDIDATE_COUNT, SOURCE_COUNT, len(DEFAULT_THRESHOLDS))):
        raise ValueError("joint cache mask_ious shape is invalid")
    if not bool(torch.isfinite(mask_ious).all().item()):
        raise ValueError("joint cache mask_ious must be finite")
    if bool(((mask_ious < 0.0) | (mask_ious > 1.0)).any().item()):
        raise ValueError("joint cache mask_ious must lie in [0, 1]")
    if not bool(torch.equal(mask_ious[~valid], torch.zeros_like(mask_ious[~valid]))):
        raise ValueError("invalid candidates must have zero mask labels")
    return row


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path):
    with Path(path).open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON payload must be an object")
    return value


def _append_shard(output_dir, manifest, rows):
    if not rows:
        raise ValueError("cannot append an empty joint cache shard")
    output_dir = Path(output_dir)
    index = len(manifest.get("shards", []))
    name = "shard_{:06d}.pt".format(index)
    temporary = output_dir / (name + ".tmp")
    destination = output_dir / name
    for offset, row in enumerate(rows):
        validate_joint_cache_row(row, manifest["sample_count"] + offset)
    try:
        torch.save({"schema": CACHE_SCHEMA, "rows": rows}, temporary)
        os.replace(str(temporary), str(destination))
        updated = dict(manifest)
        updated["shards"] = list(manifest.get("shards", [])) + [name]
        updated["sample_count"] = manifest["sample_count"] + len(rows)
        _atomic_json(output_dir / "manifest.json", updated)
    finally:
        if temporary.exists():
            temporary.unlink()
    return updated


def load_joint_cache(output_dir):
    """Load and validate a completed CPU joint label cache."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = validate_joint_cache_manifest(
        _load_json(output_dir / "manifest.json"), "train"
    )
    rows = []
    for name in manifest["shards"]:
        payload = torch.load(output_dir / name, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
            raise ValueError("joint cache shard schema is invalid")
        shard_rows = payload.get("rows")
        if not isinstance(shard_rows, list) or not shard_rows:
            raise ValueError("joint cache shard rows are invalid")
        for row in shard_rows:
            validate_joint_cache_row(row, len(rows))
            rows.append(row)
    if len(rows) != manifest["sample_count"]:
        raise ValueError("joint cache row count does not match manifest")
    return rows, manifest


def _set_deterministic(seed, device):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _mask_superpoints(batch_data, batch_index):
    point_target = batch_data["gt_masks"][batch_index, 0].bool()
    key = "superpoint" if "superpoint" in batch_data else "superpoints"
    if key not in batch_data:
        raise ValueError("batch is missing superpoint IDs")
    superpoint_ids = batch_data[key][batch_index]
    return point_target, superpoint_ids


def compute_batch_mask_labels(end_points, batch_data, candidate_batch,
                              thresholds=DEFAULT_THRESHOLDS):
    """Compute exact fused/text/query mask IoU labels after the GT boundary."""
    query_indices = candidate_batch["query_indices"]
    valid = candidate_batch["valid_mask"].bool()
    if query_indices.dim() != 2 or tuple(valid.shape) != tuple(query_indices.shape):
        raise ValueError("candidate query identity tensors are malformed")
    labels = []
    for batch_index in range(query_indices.shape[0]):
        text, query, _fused, alpha = normalize_mcln_mask_logits(
            end_points, batch_index, query_indices[batch_index]
        )
        point_target, superpoint_ids = _mask_superpoints(batch_data, batch_index)
        compressed = compress_point_mask_to_superpoints(
            point_target, superpoint_ids, num_superpoints=text.shape[-1]
        )
        result = compute_weighted_mask_candidate_targets(
            text.unsqueeze(0), query.unsqueeze(0), alpha,
            compressed["point_counts"].unsqueeze(0),
            compressed["target_counts"].unsqueeze(0),
            valid[batch_index].unsqueeze(0),
            torch.as_tensor(thresholds, device=text.device, dtype=text.dtype),
        )
        labels.append(result["ious"][0].detach().cpu().float())
    return torch.stack(labels, dim=0)


def _build_row(dataset_index, batch_data, candidate_batch, mask_ious,
               batch_index):
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
        "mask_ious": mask_ious[batch_index].contiguous(),
    }
    validate_joint_cache_row(row, int(dataset_index))
    return row


def _load_checkpoint_and_config(checkpoint_path, data_root):
    from scripts.cache_scanrefer_rec_candidates import (
        _normalized_data_root, _prepare_model_config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain an object")
    return checkpoint, _prepare_model_config(
        checkpoint, _normalized_data_root(data_root)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Materialize train-only ScanRefer joint mask labels."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.split != "train":
        parser.error("--split must be train; validation labels are forbidden")
    for name in ("batch_size", "shard_size"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def run_extraction(args):
    if args.split != "train":
        raise ValueError("joint mask label extraction is train-only")
    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    base_cache = Path(args.base_cache).expanduser().resolve()
    geometry_cache = Path(args.geometry_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not data_root.is_dir() or not checkpoint_path.is_file():
        raise ValueError("data root and checkpoint must exist")
    if not base_cache.is_dir() or not geometry_cache.is_dir():
        raise ValueError("train base and geometry caches must exist")
    if output_dir.exists() and not args.overwrite:
        raise ValueError("joint cache output already exists")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(args.device)
    _set_deterministic(args.seed, device)

    from scripts.cache_scanrefer_rec_candidates import (
        _build_dataset, _build_loader, _load_frozen_model,
        _move_batch_to_device, _normalized_data_root,
    )
    from scripts.rec_geometry_cache import (
        load_bound_candidate_cache, validate_geometry_manifest,
    )
    from scripts.audit_scanrefer_mask_geometry import assert_candidate_cache_parity
    from models.rec_candidate_adapter import build_rec_candidate_batch
    from train_dist_mod import TrainTester

    checkpoint, config = _load_checkpoint_and_config(checkpoint_path, data_root)
    checkpoint_sha = _sha256_file(checkpoint_path)
    base_rows, base_manifest, _base_binding = load_bound_candidate_cache(
        base_cache, "train"
    )
    # Loading the geometry manifest binds identity/provenance; geometry labels
    # themselves remain in the canonical sidecar used by the trainer.
    # Only the immutable geometry manifest is needed for label provenance.  A
    # full row load here would spend minutes revalidating a multi-gigabyte
    # sidecar that the trainer will stream separately.
    geometry_manifest = _load_json(geometry_cache / "manifest.json")
    validate_geometry_manifest(
        geometry_manifest, "train", require_complete=True
    )
    if geometry_manifest.get("sample_count") != len(base_rows):
        raise ValueError("base and geometry train caches have different sizes")
    dataset = _build_dataset(config, "train")
    if len(dataset) != len(base_rows):
        raise ValueError("train dataset order/size differs from bound cache")
    limit = min(len(dataset), args.limit or len(dataset))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_dir.glob("shard_*.pt"):
            path.unlink()
        for name in ("manifest.json", "storage_decision.json"):
            path = output_dir / name
            if path.exists():
                path.unlink()

    free_bytes = shutil.disk_usage(str(output_dir)).free
    estimated_row_bytes = 2048
    capacity = estimate_cache_capacity(
        limit, estimated_row_bytes, free_bytes
    )
    _atomic_json(output_dir / "storage_decision.json", {
        "schema": CACHE_SCHEMA + "-storage-v1",
        "split": "train",
        "validation_data_accessed": False,
        "capacity": capacity,
        "streaming_fallback": not capacity["can_materialize"],
    })
    # A label row is small enough that streaming still produces a complete
    # deterministic cache; refuse only when even the conservative reserve is
    # unavailable rather than silently reducing the population.
    if not capacity["can_materialize"]:
        raise RuntimeError(
            "insufficient capacity for complete train joint cache: {}"
            .format(json.dumps(capacity, sort_keys=True))
        )

    manifest = {
        "schema": CACHE_SCHEMA,
        "joint_mask_schema_version": JOINT_MASK_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "sample_count": 0,
        "dataset_size": int(limit),
        "source_dataset_size": int(len(dataset)),
        "feature_schema_version": "rec-query-v1",
        "geometry_schema_version": "rec-geometry-flat-v1",
        "candidate_count": CANDIDATE_COUNT,
        "variant_count": VARIANT_COUNT,
        "source_count": SOURCE_COUNT,
        "thresholds": list(DEFAULT_THRESHOLDS),
        "source_names": list(MASK_SOURCE_NAMES),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "base_cache_manifest_sha256": _sha256_file(base_cache / "manifest.json"),
        "geometry_cache_manifest_sha256": _sha256_file(geometry_cache / "manifest.json"),
        "geometry_cache_content_digest": geometry_manifest.get("cache_content_digest"),
        "shards": [],
        "seed": int(args.seed),
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    model = _load_frozen_model(checkpoint, config, device)
    del checkpoint
    loader = _build_loader(dataset, 0, limit, args, device)
    pending = []
    cursor = 0
    for batch_data in loader:
        batch_data = _move_batch_to_device(batch_data, device)
        inputs = TrainTester._get_inputs(batch_data)
        inputs["train"] = False
        with torch.inference_mode():
            end_points = model(inputs)
            fresh = build_rec_candidate_batch(
                end_points, inputs, topk_per_source=8, max_candidates=16
            )
            # The existing cache is identity-only provenance here; fresh
            # runtime candidates are the labels' source of truth.
            indices = list(range(cursor, cursor + len(batch_data["scan_ids"])))
            # Candidate IoU is attached only after the deployable builder has
            # completed.  It is consumed by the identity-only audit helper,
            # never serialized into this inference-facing cache.
            from models.rec_candidate_adapter import attach_candidate_targets
            targeted = attach_candidate_targets(
                fresh, batch_data, root_only=True
            )
            assert_candidate_cache_parity(
                targeted, {row["dataset_index"]: row for row in base_rows},
                indices, batch_data["scan_ids"], batch_data["target_id"],
                identity_only=True,
            )
            mask_ious = compute_batch_mask_labels(
                end_points, batch_data, fresh, DEFAULT_THRESHOLDS
            )
        for batch_index, dataset_index in enumerate(indices):
            pending.append(_build_row(
                dataset_index, batch_data, fresh, mask_ious, batch_index
            ))
        cursor += len(indices)
        while len(pending) >= args.shard_size:
            manifest = _append_shard(
                output_dir, manifest, pending[:args.shard_size]
            )
            del pending[:args.shard_size]
        if cursor % max(args.shard_size, args.batch_size) == 0:
            print("Joint mask labels {}/{}".format(cursor, limit), flush=True)
    if pending:
        manifest = _append_shard(output_dir, manifest, pending)
    if manifest["sample_count"] != limit:
        raise RuntimeError("joint mask cache ended incomplete")
    manifest["complete"] = True
    manifest["content_sha256"] = _canonical_json_sha256({
        key: value for key, value in manifest.items()
        if key != "content_sha256"
    })
    _atomic_json(output_dir / "manifest.json", manifest)
    validate_joint_cache_manifest(manifest, "train")
    print("Published joint mask cache: {} rows at {}".format(limit, output_dir))
    return manifest


def main(argv=None):
    return run_extraction(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
