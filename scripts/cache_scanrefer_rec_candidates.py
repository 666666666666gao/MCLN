#!/usr/bin/env python
"""Cache deterministic ScanRefer REC candidates for reranker training."""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch


CACHE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
TOPK_PER_SOURCE = 8
ROOT = Path(__file__).resolve().parents[1]


def strip_module_prefix(state_dict):
    """Strip exactly one leading ``module.`` prefix from checkpoint keys."""
    stripped = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        if new_key in stripped:
            raise ValueError("state-dict key collision after stripping module prefix")
        stripped[new_key] = value
    return stripped


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    """Return the SHA-256 fingerprint of a checkpoint file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_manifest(output_dir):
    """Load a cache manifest from ``output_dir``."""
    manifest_path = Path(output_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("cache manifest does not exist: {}".format(manifest_path))
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("cache manifest must contain a JSON object")
    return manifest


def _manifest_metadata(manifest):
    return {
        key: value for key, value in manifest.items()
        if key not in ("sample_count", "shards")
    }


def _clear_known_cache_files(output_dir):
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    patterns = (
        "shard_[0-9][0-9][0-9][0-9][0-9][0-9].pt",
        "shard_[0-9][0-9][0-9][0-9][0-9][0-9].pt.tmp",
    )
    known_paths = [manifest_path, output_dir / (MANIFEST_NAME + ".tmp")]
    for pattern in patterns:
        known_paths.extend(output_dir.glob(pattern))
    for path in known_paths:
        if path.is_file():
            path.unlink()


def initialize_cache(output_dir, metadata, overwrite=False):
    """Create or resume a cache whose immutable metadata matches ``metadata``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if overwrite:
        _clear_known_cache_files(output_dir)
    elif manifest_path.is_file():
        manifest = load_manifest(output_dir)
        if _manifest_metadata(manifest) != dict(metadata):
            raise ValueError("existing cache metadata does not match requested metadata")
        cache_resume_state(output_dir, manifest)
        return manifest

    manifest = dict(metadata)
    manifest["sample_count"] = 0
    manifest["shards"] = []
    _atomic_write_json(manifest_path, manifest)
    return manifest


def cache_resume_state(output_dir, manifest):
    """Validate manifested shards and return sample count and next shard index."""
    output_dir = Path(output_dir)
    shards = manifest.get("shards")
    sample_count = manifest.get("sample_count")
    if not isinstance(shards, list) or not isinstance(sample_count, int):
        raise ValueError("cache manifest has invalid resume fields")
    for index, shard_name in enumerate(shards):
        expected = "shard_{:06d}.pt".format(index)
        if shard_name != expected:
            raise ValueError("cache manifest has non-contiguous shard names")
        if not (output_dir / shard_name).is_file():
            raise ValueError("missing shard listed in cache manifest: {}".format(shard_name))
    return sample_count, len(shards)


def append_cache_shard(output_dir, manifest, rows):
    """Atomically append one CPU shard and update its manifest."""
    if not rows:
        raise ValueError("cannot append an empty cache shard")
    output_dir = Path(output_dir)
    sample_count, shard_index = cache_resume_state(output_dir, manifest)
    shard_name = "shard_{:06d}.pt".format(shard_index)
    shard_path = output_dir / shard_name
    temporary = shard_path.with_name(shard_path.name + ".tmp")
    try:
        torch.save({"rows": rows}, temporary)
        os.replace(str(temporary), str(shard_path))
    finally:
        if temporary.exists():
            temporary.unlink()

    updated = dict(manifest)
    updated["shards"] = list(manifest["shards"]) + [shard_name]
    updated["sample_count"] = sample_count + len(rows)
    try:
        _atomic_write_json(output_dir / MANIFEST_NAME, updated)
    except Exception:
        if shard_path.exists():
            shard_path.unlink()
        raise
    return updated


def compute_batch_metric_counts(candidate_ious, valid_mask, query_indices,
                                default_top1_query_index):
    """Count strict default Top-1 and candidate-oracle hits for one batch."""
    if candidate_ious.dim() != 2:
        raise ValueError("candidate_ious must have shape [B,K]")
    if valid_mask.shape != candidate_ious.shape:
        raise ValueError("valid_mask must match candidate_ious")
    if query_indices.shape != candidate_ious.shape:
        raise ValueError("query_indices must match candidate_ious")
    if default_top1_query_index.shape != candidate_ious.shape[:1]:
        raise ValueError("default_top1_query_index must have shape [B]")

    valid = valid_mask.bool()
    default_matches = (
        query_indices == default_top1_query_index.unsqueeze(1)
    ) & valid
    if not default_matches.any(dim=1).all():
        raise ValueError("default Top-1 query is missing from valid candidates")
    default_ious = candidate_ious.masked_fill(~default_matches, -1.0).max(
        dim=1
    ).values
    oracle_ious = candidate_ious.masked_fill(~valid, -1.0).max(dim=1).values
    return {
        "sample_count": int(candidate_ious.shape[0]),
        "default_hits025": int((default_ious > 0.25).sum().item()),
        "default_hits050": int((default_ious > 0.50).sum().item()),
        "oracle_hits025": int((oracle_ious > 0.25).sum().item()),
        "oracle_hits050": int((oracle_ious > 0.50).sum().item()),
    }


def _batch_identity_value(value, index):
    if isinstance(value, torch.Tensor):
        value = value[index]
        return value.item() if value.numel() == 1 else value.detach().cpu()
    return value[index]


def build_cache_rows(dataset_indices, batch_data, candidate_batch):
    """Serialize one candidate batch as independent CPU cache rows."""
    dataset_indices = list(dataset_indices)
    batch_size = candidate_batch["features"].shape[0]
    if len(dataset_indices) != batch_size:
        raise ValueError("dataset_indices must match candidate batch size")
    tensor_keys = (
        "features",
        "boxes",
        "query_indices",
        "valid_mask",
        "default_scores",
        "contrastive_scores",
        "candidate_ious",
    )
    for key in tensor_keys:
        value = candidate_batch.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
            raise ValueError("{} must be a batched tensor".format(key))

    rows = []
    for index in range(batch_size):
        row = {
            "dataset_index": int(dataset_indices[index]),
            "scan_id": str(_batch_identity_value(batch_data["scan_ids"], index)),
            "target_id": int(_batch_identity_value(batch_data["target_id"], index)),
            "default_top1_query_index": int(
                candidate_batch["default_top1_query_index"][index].item()
            ),
        }
        for key in tensor_keys:
            row[key] = candidate_batch[key][index].detach().cpu().clone()
        rows.append(row)
    return rows


def parse_args(argv=None):
    """Parse candidate-cache command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cache deterministic ScanRefer REC reranker candidates."
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-oracle",
        type=float,
        nargs=2,
        metavar=("ACC025", "ACC050"),
        default=None,
    )
    args = parser.parse_args(argv)
    for name in ("batch_size", "shard_size", "max_candidates"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.require_oracle is not None and not all(
            0.0 <= value <= 1.0 for value in args.require_oracle):
        parser.error("--require-oracle values must lie in [0, 1]")
    return args


def oracle_gate_exit_code(metrics, requirement):
    """Return 2 when candidate oracle metrics miss a requested gate."""
    if requirement is None:
        return 0
    required025, required050 = requirement
    if (metrics["oracle_acc025"] < required025
            or metrics["oracle_acc050"] < required050):
        return 2
    return 0


def _add_metric_counts(total, update):
    for key in total:
        total[key] += int(update[key])


def _empty_metric_counts():
    return {
        "sample_count": 0,
        "default_hits025": 0,
        "default_hits050": 0,
        "oracle_hits025": 0,
        "oracle_hits050": 0,
    }


def _metric_rates(counts):
    denominator = max(counts["sample_count"], 1)
    return {
        "default_acc025": counts["default_hits025"] / float(denominator),
        "default_acc050": counts["default_hits050"] / float(denominator),
        "oracle_acc025": counts["oracle_hits025"] / float(denominator),
        "oracle_acc050": counts["oracle_hits050"] / float(denominator),
    }


def _counts_for_rows(rows):
    if not rows:
        return _empty_metric_counts()
    return compute_batch_metric_counts(
        torch.stack([row["candidate_ious"] for row in rows]),
        torch.stack([row["valid_mask"] for row in rows]),
        torch.stack([row["query_indices"] for row in rows]),
        torch.tensor([
            row["default_top1_query_index"] for row in rows
        ], dtype=torch.long),
    )


def _load_cached_metric_counts(output_dir, manifest):
    counts = _empty_metric_counts()
    expected_dataset_index = 0
    for shard_name in manifest["shards"]:
        payload = torch.load(
            Path(output_dir) / shard_name, map_location="cpu"
        )
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("cache shard {} has no row list".format(shard_name))
        for row in rows:
            if row.get("dataset_index") != expected_dataset_index:
                raise ValueError(
                    "cached dataset indices are not contiguous at {}".format(
                        expected_dataset_index
                    )
                )
            expected_dataset_index += 1
        _add_metric_counts(counts, _counts_for_rows(rows))
    if expected_dataset_index != manifest["sample_count"]:
        raise ValueError("manifest sample count does not match cached rows")
    return counts


def _ensure_project_imports():
    for path in (str(ROOT), str(ROOT / "pointnet2")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _normalized_data_root(path):
    value = str(Path(path).expanduser().resolve())
    return value if value.endswith(os.sep) else value + os.sep


def _prepare_model_config(checkpoint, data_root):
    config = checkpoint.get("config")
    if config is None or not hasattr(config, "__dict__"):
        raise ValueError("checkpoint does not contain an argparse-style config")
    config = copy.copy(config)
    defaults = {
        "butd": False,
        "butd_gt": False,
        "butd_cls": False,
        "detect_intermediate": False,
        "num_decoder_layers": 6,
        "num_target": 256,
        "self_attend": False,
        "self_position_embedding": "loc_learned",
        "use_color": False,
        "use_height": False,
        "use_multiview": False,
        "use_soft_token_loss": True,
        "use_contrastive_align": True,
        "use_source_choice_selector": False,
        "source_choice_selector_sources": "default,mask_text",
        "source_choice_selector_hidden_dim": 288,
        "use_source_moe": False,
        "source_moe_shared_source": "default",
        "source_moe_top_k": 2,
        "source_moe_balance_loss_weight": 0.01,
        "source_moe_query_layers": 1,
        "source_moe_query_heads": 4,
        "source_moe_query_dropout": 0.1,
        "source_moe_query_max_delta": 0.25,
        "source_moe_use_fallback_gate": False,
        "source_moe_gate_hidden_dim": 128,
        "source_moe_gate_candidate_top_k": 8,
        "source_moe_gate_break_cost": 2.0,
        "source_moe_gate_decision_margin": 0.0,
        "source_moe_gate_mask_utility_weight": 0.25,
        "source_moe_gate_uncertainty_weight": 0.0,
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_action_mode": "decision",
        "wo_obj_name": "None",
        "skip_missing_superpoints": False,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config.data_root = data_root
    config.pp_checkpoint = None
    config.model = "MCLN"
    config.eval = True
    return config


def _build_dataset(config, split):
    from src.joint_det_dataset import Joint3DDataset

    dataset = Joint3DDataset(
        dataset_dict={"scanrefer": 1},
        test_dataset="scanrefer",
        split=split,
        use_color=bool(config.use_color),
        use_height=bool(config.use_height),
        overfit=False,
        data_path=config.data_root,
        detect_intermediate=bool(config.detect_intermediate),
        use_multiview=bool(config.use_multiview),
        butd=bool(config.butd),
        butd_gt=bool(config.butd_gt),
        butd_cls=bool(config.butd_cls),
        augment_det=False,
        wo_obj_name=config.wo_obj_name,
        skip_missing_superpoints=bool(config.skip_missing_superpoints),
    )
    dataset.augment = False
    dataset.augment_det = False
    return dataset


def _load_frozen_model(checkpoint, config, device):
    from train_dist_mod import TrainTester

    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    model = TrainTester.get_model(config)
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def _seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _build_loader(dataset, start, stop, args, device):
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator()
    generator.manual_seed(0)
    subset = Subset(dataset, list(range(start, stop)))
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _move_batch_to_device(batch_data, device):
    return {
        key: (
            value.to(device, non_blocking=(device.type == "cuda"))
            if isinstance(value, torch.Tensor) else value
        )
        for key, value in batch_data.items()
    }


def _cache_metadata(args, fingerprint, checkpoint, config, dataset_size,
                    source_dataset_size, feature_dim, feature_names):
    backbone_config = {
        "model": str(config.model),
        "num_target": int(config.num_target),
        "num_decoder_layers": int(config.num_decoder_layers),
        "self_position_embedding": str(config.self_position_embedding),
        "self_attend": bool(config.self_attend),
        "use_soft_token_loss": bool(config.use_soft_token_loss),
        "use_contrastive_align": bool(config.use_contrastive_align),
        "detect_intermediate": bool(config.detect_intermediate),
        "use_source_choice_selector": bool(
            config.use_source_choice_selector
        ),
        "source_choice_selector_sources": str(
            config.source_choice_selector_sources
        ),
        "source_choice_selector_hidden_dim": int(
            config.source_choice_selector_hidden_dim
        ),
    }
    checkpoint_config = checkpoint.get("config")
    if (hasattr(checkpoint_config, "use_source_moe")
            or bool(getattr(config, "use_source_moe", False))):
        backbone_config.update({
            "use_source_moe": bool(config.use_source_moe),
            "source_moe_shared_source": str(
                config.source_moe_shared_source
            ),
            "source_moe_top_k": int(config.source_moe_top_k),
            "source_moe_balance_loss_weight": float(
                config.source_moe_balance_loss_weight
            ),
            "source_moe_query_layers": int(
                config.source_moe_query_layers
            ),
            "source_moe_query_heads": int(config.source_moe_query_heads),
            "source_moe_query_dropout": float(
                config.source_moe_query_dropout
            ),
            "source_moe_query_max_delta": float(
                config.source_moe_query_max_delta
            ),
            "source_moe_use_fallback_gate": bool(
                config.source_moe_use_fallback_gate
            ),
            "source_moe_gate_hidden_dim": int(
                config.source_moe_gate_hidden_dim
            ),
            "source_moe_gate_candidate_top_k": int(
                config.source_moe_gate_candidate_top_k
            ),
            "source_moe_gate_break_cost": float(
                config.source_moe_gate_break_cost
            ),
            "source_moe_gate_decision_margin": float(
                config.source_moe_gate_decision_margin
            ),
            "source_moe_gate_mask_utility_weight": float(
                config.source_moe_gate_mask_utility_weight
            ),
            "source_moe_gate_uncertainty_weight": float(getattr(
                config, "source_moe_gate_uncertainty_weight", 0.0
            )),
            "source_moe_gate_use_evidence_features": bool(
                config.source_moe_gate_use_evidence_features
            ),
            "source_moe_gate_action_mode": str(getattr(
                config, "source_moe_gate_action_mode", "decision"
            )),
        })
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "feature_schema_version": "rec-query-v1",
        "checkpoint_sha256": fingerprint,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "data_root": str(Path(config.data_root).expanduser().resolve()),
        "candidate_rule": {
            "topk_per_source": TOPK_PER_SOURCE,
            "max_candidates": args.max_candidates,
        },
        "feature_dim": int(feature_dim),
        "feature_names": list(feature_names),
        "target_iou_policy": "root_only",
        "dataset_size": int(dataset_size),
        "source_dataset_size": int(source_dataset_size),
        "model_inputs": {
            "use_color": bool(config.use_color),
            "use_height": bool(config.use_height),
            "use_multiview": bool(config.use_multiview),
            "butd": bool(config.butd),
            "butd_gt": bool(config.butd_gt),
            "butd_cls": bool(config.butd_cls),
        },
        "backbone_config": backbone_config,
        "deterministic": True,
    }


def _validate_candidate_batch(candidate_batch, feature_dim, feature_names):
    if candidate_batch["features"].shape[-1] != feature_dim:
        raise ValueError("candidate feature dimension changed during extraction")
    if list(candidate_batch["feature_names"]) != list(feature_names):
        raise ValueError("candidate feature schema changed during extraction")
    for key in ("features", "boxes", "candidate_ious"):
        if not torch.isfinite(candidate_batch[key]).all():
            raise ValueError("non-finite tensor in candidate field {}".format(key))


def _print_metrics(metrics, sample_count):
    print("Cached samples: {}".format(sample_count))
    print(
        "Default Top-1: Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            metrics["default_acc025"], metrics["default_acc050"]
        )
    )
    print(
        "Candidate oracle: Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            metrics["oracle_acc025"], metrics["oracle_acc050"]
        )
    )


def run_extraction(args):
    """Run deterministic model inference and append candidate cache shards."""
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint does not exist: {}".format(checkpoint_path))
    data_root = _normalized_data_root(args.data_root)
    if not Path(data_root).is_dir():
        raise ValueError("data root does not exist: {}".format(data_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")

    _ensure_project_imports()
    os.chdir(str(ROOT))
    random.seed(0)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    fingerprint = checkpoint_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    config = _prepare_model_config(checkpoint, data_root)
    dataset = _build_dataset(config, args.split)
    source_dataset_size = len(dataset)
    dataset_size = min(
        source_dataset_size,
        args.limit if args.limit is not None else source_dataset_size,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = None
    counts = _empty_metric_counts()
    start_index = 0
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file() and not args.overwrite:
        existing = load_manifest(output_dir)
        metadata = _cache_metadata(
            args,
            fingerprint,
            checkpoint,
            config,
            dataset_size,
            source_dataset_size,
            existing.get("feature_dim", -1),
            existing.get("feature_names", []),
        )
        manifest = initialize_cache(output_dir, metadata)
        start_index, _ = cache_resume_state(output_dir, manifest)
        if start_index > dataset_size:
            raise ValueError("cache contains more rows than requested dataset")
        counts = _load_cached_metric_counts(output_dir, manifest)
        if start_index == dataset_size:
            metrics = _metric_rates(counts)
            _print_metrics(metrics, counts["sample_count"])
            return oracle_gate_exit_code(metrics, args.require_oracle)

    model = _load_frozen_model(checkpoint, config, device)
    del checkpoint
    loader = _build_loader(dataset, start_index, dataset_size, args, device)

    from models.rec_candidate_adapter import (
        FEATURE_SCHEMA_VERSION,
        attach_candidate_targets,
        build_rec_candidate_batch,
    )
    if FEATURE_SCHEMA_VERSION != "rec-query-v1":
        raise ValueError("unsupported REC feature schema: {}".format(
            FEATURE_SCHEMA_VERSION
        ))
    from train_dist_mod import TrainTester

    feature_dim = manifest.get("feature_dim") if manifest is not None else None
    feature_names = manifest.get("feature_names") if manifest is not None else None
    pending_rows = []
    cursor = start_index
    with torch.no_grad():
        for batch_data in loader:
            batch_data = _move_batch_to_device(batch_data, device)
            inputs = TrainTester._get_inputs(batch_data)
            inputs["train"] = False
            end_points = model(inputs)
            candidate_batch = build_rec_candidate_batch(
                end_points,
                inputs,
                topk_per_source=TOPK_PER_SOURCE,
                max_candidates=args.max_candidates,
            )
            candidate_batch = attach_candidate_targets(
                candidate_batch, batch_data, root_only=True
            )
            if feature_dim is None:
                feature_dim = int(candidate_batch["features"].shape[-1])
                feature_names = list(candidate_batch["feature_names"])
                metadata = _cache_metadata(
                    args,
                    fingerprint,
                    {"epoch": checkpoint_epoch},
                    config,
                    dataset_size,
                    source_dataset_size,
                    feature_dim,
                    feature_names,
                )
                manifest = initialize_cache(
                    output_dir, metadata, overwrite=args.overwrite
                )
            _validate_candidate_batch(
                candidate_batch, feature_dim, feature_names
            )
            batch_counts = compute_batch_metric_counts(
                candidate_batch["candidate_ious"],
                candidate_batch["valid_mask"],
                candidate_batch["query_indices"],
                candidate_batch["default_top1_query_index"],
            )
            _add_metric_counts(counts, batch_counts)
            batch_size = candidate_batch["features"].shape[0]
            dataset_indices = range(cursor, cursor + batch_size)
            pending_rows.extend(build_cache_rows(
                dataset_indices, batch_data, candidate_batch
            ))
            cursor += batch_size
            while len(pending_rows) >= args.shard_size:
                shard_rows = pending_rows[:args.shard_size]
                del pending_rows[:args.shard_size]
                manifest = append_cache_shard(
                    output_dir, manifest, shard_rows
                )
                print(
                    "Cached {}/{} samples".format(
                        manifest["sample_count"], dataset_size
                    ),
                    flush=True,
                )

    if pending_rows:
        manifest = append_cache_shard(output_dir, manifest, pending_rows)
    if manifest is None or manifest["sample_count"] != dataset_size:
        raise RuntimeError("candidate extraction ended with an incomplete cache")
    if counts["sample_count"] != dataset_size:
        raise RuntimeError("metric sample count does not match cache")
    metrics = _metric_rates(counts)
    _print_metrics(metrics, counts["sample_count"])
    exit_code = oracle_gate_exit_code(metrics, args.require_oracle)
    if exit_code:
        print(
            "Candidate oracle missed required Acc@0.25={:.5f} "
            "Acc@0.50={:.5f}".format(*args.require_oracle),
            file=sys.stderr,
        )
    return exit_code


def main(argv=None):
    return run_extraction(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
