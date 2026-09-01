#!/usr/bin/env python
"""Cache deterministic full-dataset REC geometry sidecars."""

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_candidate_adapter import (
    attach_candidate_targets,
    build_rec_candidate_batch,
)
from models.rec_evaluator_filter import build_detector_overlap_valid
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
    attach_rec_mask_geometry_targets,
    build_rec_mask_geometry_candidates,
    project_variant_rejection_codes,
)
from scripts.audit_scanrefer_mask_geometry import (
    IOU_PARITY_ATOL,
    IOU_PARITY_RTOL,
    PARITY_ATOL,
    PARITY_RTOL,
    assert_candidate_cache_parity,
)
from scripts.cache_scanrefer_rec_candidates import (
    _build_dataset,
    _build_loader,
    _load_frozen_model,
    _move_batch_to_device,
    _normalized_data_root,
    _prepare_model_config,
)
from scripts.rec_geometry_cache import (
    GEOMETRY_CACHE_SCHEMA_VERSION,
    append_geometry_shard,
    canonical_json_sha256,
    finalize_geometry_cache,
    initialize_geometry_cache,
    load_bound_candidate_cache,
    validate_geometry_row,
)
from scripts.train_rec_reranker import normalize_backbone_config


EXPECTED_CHECKPOINT_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
EXPECTED_CHECKPOINT_EPOCH = 71
EXPECTED_DATASET_SIZES = {"train": 36665, "val": 9508}
EXPECTED_BASE_SHARD_COUNTS = {"train": 144, "val": 38}
EXPECTED_BASE_CONTENT_SHA256 = {
    "train": (
        "411ec7d5d80a7be9596de20b348667d529e6a8f568b8ab0c0e0922b8719f9045"
    ),
    "val": (
        "b2e6cf81ba8441d7e9ec04141e0d4fb73f61f069ea3ec99a2396ae68a6740ef3"
    ),
}
EXPECTED_BASE_MANIFEST_SHA256 = {
    "train": (
        "c8858036c3da0b25183f262c763e947a3dac77544ee3073623172716878cfabc"
    ),
    "val": (
        "695b5565de460f580a6d130d1b6465d309b65eef46502340f0fc8ea1235907d2"
    ),
}
EXPECTED_ANNOTATION_SHA256 = {
    "train": (
        "93b0bd2a884de8077ba659aa1ee341dd7f571b32c46986e0ce388d6aad349521"
    ),
    "val": (
        "c9a44fa2cfb83ea1893a76f4041f206bb0432614600b43d87b47ca2d791e76f3"
    ),
}
EXPECTED_AUDIT_SELECTION_SHA256 = (
    "1acb1325d2f4b3a78cdc33f06119771c82ef7ad359769bafea856d055bc7e4f5"
)
EXPECTED_BASE_FEATURE_DIM = 152
PRODUCTION_BATCH_SIZE = 12
PRODUCTION_NUM_WORKERS = 2
PRODUCTION_SHARD_SIZE = 252
PRODUCTION_DEVICE = "cuda:0"
PRODUCTION_TOPK_PER_SOURCE = 8
PRODUCTION_MAX_CANDIDATES = 16
PRODUCTION_MIN_POINTS = 5
PRODUCTION_MAX_POINT_FRACTION = 0.5

PARITY_FIELDS = (
    "boxes",
    "candidate_ious",
    "features",
    "default_scores",
    "contrastive_scores",
)
FORBIDDEN_DEPLOYABLE_KEYS = frozenset((
    "center_label",
    "size_gts",
    "box_label_mask",
    "gt_masks",
    "candidate_ious",
    "geometry_ious",
    "threshold_labels",
))
GEOMETRY_IMMUTABLE_METADATA_FIELDS = (
    "geometry_cache_schema_version",
    "geometry_schema_version",
    "geometry_feature_names",
    "variant_names",
    "variant_configs",
    "regressed_variant_index",
    "min_points",
    "max_point_fraction",
    "split",
    "dataset_size",
    "source_dataset_size",
    "candidate_rule",
    "target_iou_policy",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "model_inputs",
    "backbone_config",
    "extraction_batch_size",
    "num_workers",
    "shard_size",
    "base_cache_binding",
    "annotation_sha256",
    "audit_provenance",
    "filter_non_gt_boxes",
)


def _file_identity(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _read_stable_file_snapshot(path, label):
    """Read one immutable byte snapshot and verify its live path identity."""
    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open("rb") as handle:
            before_identity = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after_identity = _file_identity(os.fstat(handle.fileno()))
        path_identity = _file_identity(resolved.stat())
    except OSError as error:
        raise ValueError("could not read {}: {}".format(label, error))
    if (before_identity != after_identity
            or after_identity != path_identity):
        raise ValueError("{} changed during stable snapshot".format(label))
    return {
        "path": str(resolved),
        "bytes": snapshot,
        "sha256": hashlib.sha256(snapshot).hexdigest(),
        "identity": before_identity,
    }


def _load_checkpoint_snapshot(checkpoint_path):
    snapshot = _read_stable_file_snapshot(
        checkpoint_path, "checkpoint"
    )
    checkpoint_bytes = snapshot.pop("bytes")
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu"
        )
    except Exception as error:
        raise ValueError("checkpoint snapshot could not be loaded: {}".format(
            error
        ))
    finally:
        del checkpoint_bytes
    return checkpoint, snapshot["sha256"]


def _load_stable_json_snapshot(path, label):
    snapshot = _read_stable_file_snapshot(path, label)
    json_bytes = snapshot.pop("bytes")
    try:
        payload = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("{} JSON is malformed: {}".format(label, error))
    finally:
        del json_bytes
    return payload, snapshot


def _validated_annotation_snapshot(data_root, split):
    annotation_path = (
        Path(data_root).expanduser().resolve()
        / "scanrefer" / "ScanRefer_filtered_{}.json".format(split)
    )
    snapshot = _read_stable_file_snapshot(
        annotation_path, "{} annotation JSON".format(split)
    )
    del snapshot["bytes"]
    if snapshot["sha256"] != EXPECTED_ANNOTATION_SHA256[split]:
        raise ValueError(
            "annotation JSON SHA-256 is not authoritative for {}".format(
                split
            )
        )
    return snapshot


def _canonical_dataset_annotation_sha256(dataset):
    """Bind a dataset-only cache to the ordered live annotation identities."""
    annos = getattr(dataset, "annos", None)
    if not isinstance(annos, list) or not annos:
        raise ValueError("dataset annotations are unavailable")
    identities = []
    for index, annotation in enumerate(annos):
        if not isinstance(annotation, dict):
            raise ValueError("dataset annotation {} is malformed".format(index))
        try:
            identities.append({
                "dataset_index": index,
                "scan_id": str(annotation["scan_id"]),
                "target_id": int(annotation["target_id"]),
                "utterance": str(annotation.get(
                    "utterance", annotation.get("description", "")
                )),
            })
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "dataset annotation {} identity is invalid: {}".format(
                    index, error
                )
            )
    return canonical_json_sha256(identities)


def _build_validated_dataset(
        config, split, data_root, dataset_name="scanrefer",
        portable_provenance=False, dataset_builder=None):
    if dataset_builder is None:
        dataset_builder = _build_dataset
    if not portable_provenance:
        before = _validated_annotation_snapshot(data_root, split)
        dataset = dataset_builder(config, split)
        after = _validated_annotation_snapshot(data_root, split)
        if (before["sha256"] != after["sha256"]
                or before["identity"] != after["identity"]
                or before["path"] != after["path"]):
            raise ValueError(
                "annotation JSON changed during dataset construction"
            )
        return dataset, before["sha256"]
    dataset = dataset_builder(config, split, dataset_name)
    before = _canonical_dataset_annotation_sha256(dataset)
    after = _canonical_dataset_annotation_sha256(dataset)
    if before != after:
        raise ValueError("dataset annotation identities changed in memory")
    return dataset, before


def parse_args(argv=None):
    """Parse the fixed full-geometry extraction command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Cache deterministic ScanRefer mask geometry for one full split."
        )
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--dataset",
        choices=("scanrefer", "nr3d", "sr3d"),
        default="scanrefer",
        help="dataset-only source; non-ScanRefer use requires portable provenance",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-provenance", required=True)
    parser.add_argument(
        "--portable-provenance",
        action="store_true",
        help=(
            "Bind extraction to the supplied checkpoint and complete caches "
            "instead of the protected epoch-71 snapshot."
        ),
    )
    parser.add_argument(
        "--allow-butd-cls",
        action="store_true",
        help=(
            "Permit the baseline-compatible GT proposal / predicted-class "
            "input contract for portable Nr3D or Sr3D extraction."
        ),
    )
    parser.add_argument(
        "--audit-train-cache",
        default=None,
        help=(
            "Complete train cache used to create the audit panel; required "
            "for portable val extraction."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=PRODUCTION_BATCH_SIZE)
    parser.add_argument(
        "--num-workers", type=int, default=PRODUCTION_NUM_WORKERS
    )
    parser.add_argument("--shard-size", type=int, default=PRODUCTION_SHARD_SIZE)
    parser.add_argument("--device", default=PRODUCTION_DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--restart-building", action="store_true")
    parser.add_argument("--stop-after-shards", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.portable_provenance:
        fixed_values = (
            ("batch_size", PRODUCTION_BATCH_SIZE),
            ("num_workers", PRODUCTION_NUM_WORKERS),
            ("shard_size", PRODUCTION_SHARD_SIZE),
        )
        for name, expected in fixed_values:
            if getattr(args, name) != expected:
                parser.error(
                    "--{} must be exactly {}".format(
                        name.replace("_", "-"), expected
                    )
                )
    elif (args.batch_size <= 0 or args.num_workers < 0
          or args.shard_size <= 0
          or args.shard_size % args.batch_size != 0):
        parser.error(
            "portable extraction requires a positive batch/shard, "
            "non-negative workers, and shard-size divisible by batch-size"
        )
    if args.device != PRODUCTION_DEVICE:
        parser.error("--device must be exactly {}".format(PRODUCTION_DEVICE))
    if args.stop_after_shards is not None and args.stop_after_shards <= 0:
        parser.error("--stop-after-shards must be positive")
    if args.restart_building and not args.overwrite:
        parser.error("--restart-building requires --overwrite")
    if args.audit_train_cache is not None and not args.portable_provenance:
        parser.error(
            "--audit-train-cache requires --portable-provenance"
        )
    if args.allow_butd_cls and (
            not args.portable_provenance
            or args.dataset not in ("nr3d", "sr3d")):
        parser.error(
            "--allow-butd-cls requires portable Nr3D or Sr3D provenance"
        )
    if args.portable_provenance:
        if args.audit_train_cache is None:
            if args.split == "train":
                args.audit_train_cache = args.base_cache
            else:
                parser.error(
                    "portable val extraction requires --audit-train-cache"
                )
    elif args.dataset != "scanrefer":
        parser.error("non-ScanRefer geometry requires --portable-provenance")
    return args


def _model_inputs_from_config(config):
    try:
        return {
            "use_color": bool(config.use_color),
            "use_height": bool(config.use_height),
            "use_multiview": bool(config.use_multiview),
            "butd": bool(config.butd),
            "butd_gt": bool(config.butd_gt),
            "butd_cls": bool(config.butd_cls),
        }
    except AttributeError as error:
        raise ValueError(
            "checkpoint model input config is incomplete: {}".format(error)
        )


def _backbone_config_from_config(config):
    try:
        return {
            "model": str(config.model),
            "num_target": int(config.num_target),
            "num_decoder_layers": int(config.num_decoder_layers),
            "self_position_embedding": str(
                config.self_position_embedding
            ),
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
            "use_source_moe": bool(getattr(config, "use_source_moe", False)),
            "source_moe_shared_source": str(
                getattr(config, "source_moe_shared_source", "default")
            ),
            "source_moe_top_k": int(getattr(config, "source_moe_top_k", 2)),
            "source_moe_balance_loss_weight": float(
                getattr(config, "source_moe_balance_loss_weight", 0.01)
            ),
            "source_moe_query_layers": int(
                getattr(config, "source_moe_query_layers", 1)
            ),
            "source_moe_query_heads": int(getattr(
                config, "source_moe_query_heads", 4
            )),
            "source_moe_query_dropout": float(
                getattr(config, "source_moe_query_dropout", 0.1)
            ),
            "source_moe_query_max_delta": float(
                getattr(config, "source_moe_query_max_delta", 0.25)
            ),
            "source_moe_use_fallback_gate": bool(getattr(
                config, "source_moe_use_fallback_gate", False
            )),
            "source_moe_gate_hidden_dim": int(getattr(
                config, "source_moe_gate_hidden_dim", 128
            )),
            "source_moe_gate_candidate_top_k": int(getattr(
                config, "source_moe_gate_candidate_top_k", 8
            )),
            "source_moe_gate_break_cost": float(getattr(
                config, "source_moe_gate_break_cost", 2.0
            )),
            "source_moe_gate_decision_margin": float(getattr(
                config, "source_moe_gate_decision_margin", 0.0
            )),
            "source_moe_gate_mask_utility_weight": float(getattr(
                config, "source_moe_gate_mask_utility_weight", 0.25
            )),
            "source_moe_gate_uncertainty_weight": float(getattr(
                config, "source_moe_gate_uncertainty_weight", 0.0
            )),
            "source_moe_gate_use_evidence_features": bool(getattr(
                config, "source_moe_gate_use_evidence_features", False
            )),
            "source_moe_gate_action_mode": str(getattr(
                config, "source_moe_gate_action_mode", "decision"
            )),
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "checkpoint backbone config is incomplete: {}".format(error)
        )


def validate_source_provenance(
        split, base_binding, base_manifest, checkpoint_sha256,
        checkpoint_epoch, config, source_dataset_size, data_root,
        portable_provenance=False, dataset_name="scanrefer",
        allow_butd_cls=False):
    """Reject any source other than the approved complete production inputs."""
    if split not in EXPECTED_DATASET_SIZES:
        raise ValueError("split must be train or val")
    if (not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in checkpoint_sha256)):
        raise ValueError("checkpoint SHA-256 is invalid")
    if (not isinstance(checkpoint_epoch, int)
            or isinstance(checkpoint_epoch, bool) or checkpoint_epoch < 0):
        raise ValueError("checkpoint epoch is invalid")
    if not portable_provenance:
        if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(
                "checkpoint SHA-256 is not the approved epoch71 model"
            )
        if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
            raise ValueError(
                "checkpoint epoch is not the approved epoch71 model"
            )
    if not isinstance(base_binding, dict) or not isinstance(
            base_manifest, dict):
        raise ValueError("base cache provenance must contain mappings")
    if (base_binding.get("split") != split
            or base_manifest.get("split") != split):
        raise ValueError("base cache split does not match requested split")
    if not portable_provenance:
        if base_binding.get("content_sha256") != EXPECTED_BASE_CONTENT_SHA256[
                split]:
            raise ValueError("base cache content digest is not authoritative")
        if base_binding.get(
                "manifest_sha256") != EXPECTED_BASE_MANIFEST_SHA256[split]:
            raise ValueError(
                "base cache manifest digest is not authoritative"
            )
    declared_data_root = base_manifest.get("data_root")
    if not isinstance(declared_data_root, str) or not declared_data_root:
        raise ValueError("base cache data root is missing")
    try:
        resolved_data_root = Path(data_root).expanduser().resolve()
        resolved_declared_root = Path(
            declared_data_root
        ).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("base cache data root is invalid: {}".format(error))
    if resolved_data_root != resolved_declared_root:
        raise ValueError("live data root does not match base cache data root")

    if portable_provenance:
        declared_dataset = str(base_manifest.get("dataset", "scanrefer"))
        if declared_dataset != dataset_name:
            raise ValueError("base cache dataset does not match requested dataset")
        expected_size = base_manifest.get("dataset_size")
        if (not isinstance(expected_size, int)
                or isinstance(expected_size, bool) or expected_size <= 0):
            raise ValueError("portable base cache dataset size is invalid")
    else:
        if dataset_name != "scanrefer":
            raise ValueError("authoritative provenance is ScanRefer-only")
        expected_size = EXPECTED_DATASET_SIZES[split]
    manifest_sizes = (
        base_manifest.get("sample_count"),
        base_manifest.get("dataset_size"),
        base_manifest.get("source_dataset_size"),
        base_binding.get("sample_count"),
    )
    if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in manifest_sizes):
        raise ValueError("base cache sizes are invalid")
    if len(set(manifest_sizes)) != 1 or manifest_sizes[0] != expected_size:
        raise ValueError(
            "base cache must be a complete full-dataset cache"
        )
    if (not isinstance(source_dataset_size, int)
            or isinstance(source_dataset_size, bool)
            or source_dataset_size != expected_size):
        raise ValueError(
            "current dataset is not the approved full source dataset"
        )

    if portable_provenance:
        if (not isinstance(base_binding.get("shards"), list)
                or not base_binding["shards"]
                or not isinstance(base_manifest.get("shards"), list)
                or len(base_binding["shards"])
                != len(base_manifest["shards"])):
            raise ValueError("portable base cache shard binding is invalid")
    else:
        expected_shards = EXPECTED_BASE_SHARD_COUNTS[split]
        if (not isinstance(base_binding.get("shards"), list)
                or len(base_binding["shards"]) != expected_shards
                or not isinstance(base_manifest.get("shards"), list)
                or len(base_manifest["shards"]) != expected_shards):
            raise ValueError("base cache shard count is not the approved source")
    expected_rule = {
        "topk_per_source": PRODUCTION_TOPK_PER_SOURCE,
        "max_candidates": PRODUCTION_MAX_CANDIDATES,
    }
    if (base_binding.get("candidate_rule") != expected_rule
            or base_manifest.get("candidate_rule") != expected_rule):
        raise ValueError("base cache candidate rule must be Top-16")
    if (base_binding.get("feature_dim") != EXPECTED_BASE_FEATURE_DIM
            or base_manifest.get("feature_dim") != EXPECTED_BASE_FEATURE_DIM):
        raise ValueError("base cache feature dimension must be 152")
    if (base_binding.get("checkpoint_sha256") != checkpoint_sha256
            or base_manifest.get("checkpoint_sha256") != checkpoint_sha256):
        raise ValueError("checkpoint fingerprint does not match base cache")
    if base_manifest.get("checkpoint_epoch") != checkpoint_epoch:
        raise ValueError("checkpoint epoch does not match base cache")
    if base_manifest.get("target_iou_policy") != "root_only":
        raise ValueError("base cache target IoU policy must be root_only")
    if base_manifest.get("deterministic") is not True:
        raise ValueError("base cache must declare deterministic extraction")

    model_inputs = _model_inputs_from_config(config)
    approved_common_inputs = {
        "use_color": True,
        "use_height": False,
        "use_multiview": False,
        "butd_gt": False,
        "butd_cls": bool(allow_butd_cls),
    }
    if (
            any(model_inputs.get(key) is not value
                for key, value in approved_common_inputs.items())
            or not isinstance(model_inputs.get("butd"), bool)):
        raise ValueError(
            "checkpoint butd/model input config is not production: {}".format(
                model_inputs
            )
        )
    if (base_binding.get("model_inputs") != model_inputs
            or base_manifest.get("model_inputs") != model_inputs):
        raise ValueError(
            "checkpoint model inputs/butd do not match base cache"
        )
    backbone_config = _backbone_config_from_config(config)
    try:
        binding_backbone = normalize_backbone_config(
            base_binding.get("backbone_config")
        )
        manifest_backbone = normalize_backbone_config(
            base_manifest.get("backbone_config")
        )
    except ValueError:
        raise ValueError("base cache backbone config is invalid")
    if (binding_backbone != backbone_config
            or manifest_backbone != backbone_config):
        raise ValueError("checkpoint backbone config does not match base cache")


def _load_and_validate_audit_selection(
        selection, audit_sha256, data_root, split, base_binding,
        checkpoint_sha256, checkpoint_epoch, portable_provenance=False,
        audit_train_cache=None, dataset_name="scanrefer"):
    if (not portable_provenance
            and audit_sha256 != EXPECTED_AUDIT_SELECTION_SHA256):
        raise ValueError("audit selection SHA-256 is not authoritative")
    if not isinstance(selection, dict):
        raise ValueError("audit selection must contain an object")
    audit_batch_size = selection.get("cache_extraction_batch_size")
    if (not isinstance(audit_batch_size, int)
            or isinstance(audit_batch_size, bool)
            or audit_batch_size <= 0):
        raise ValueError("audit selection batch provenance is invalid")
    if (not portable_provenance
            and audit_batch_size != PRODUCTION_BATCH_SIZE):
        raise ValueError("audit selection batch provenance is not authoritative")
    if (selection.get("panel_schema_version")
            != "rec-mask-geometry-audit-panel-v1"
            or selection.get("population_estimate") is not False
            or selection.get("sample_count") != 256):
        raise ValueError("audit selection panel provenance is invalid")
    if selection.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("audit checkpoint provenance does not match")
    provenance = selection.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("audit selection provenance is missing")

    expected_variants = [
        dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    expected_variant_names = [value["name"] for value in expected_variants]
    if portable_provenance:
        if audit_train_cache is None:
            raise ValueError(
                "portable provenance requires the audit train cache"
            )
        try:
            expected_train_cache = Path(
                audit_train_cache
            ).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                "audit train cache path is invalid: {}".format(error)
            )
        train_manifest, _ = _load_stable_json_snapshot(
            expected_train_cache / "manifest.json",
            "audit train cache manifest",
        )
        if (not isinstance(train_manifest, dict)
                or train_manifest.get("split") != "train"
                or str(train_manifest.get("dataset", "scanrefer"))
                != dataset_name
                or train_manifest.get("sample_count")
                != train_manifest.get("dataset_size")
                or train_manifest.get("dataset_size")
                != train_manifest.get("source_dataset_size")
                or train_manifest.get("checkpoint_sha256")
                != checkpoint_sha256
                or train_manifest.get("checkpoint_epoch")
                != checkpoint_epoch):
            raise ValueError(
                "portable audit train cache manifest does not match"
            )
        expected_train_manifest_sha256 = canonical_json_sha256(
            train_manifest
        )
    else:
        expected_train_manifest_sha256 = EXPECTED_BASE_MANIFEST_SHA256[
            "train"
        ]

    exact_fields = {
        "panel_schema_version": "rec-mask-geometry-audit-panel-v1",
        "population_estimate": False,
        "split": "train",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "cache_extraction_batch_size": audit_batch_size,
        "candidate_rule": {
            "topk_per_source": PRODUCTION_TOPK_PER_SOURCE,
            "max_candidates": PRODUCTION_MAX_CANDIDATES,
        },
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "geometry_feature_names": list(REC_MASK_GEOMETRY_FEATURE_NAMES),
        "variant_names": expected_variant_names,
        "variant_configs": expected_variants,
        "min_points": PRODUCTION_MIN_POINTS,
        "max_point_fraction": PRODUCTION_MAX_POINT_FRACTION,
        "train_cache_manifest_sha256": expected_train_manifest_sha256,
    }
    for key, expected in exact_fields.items():
        if provenance.get(key) != expected:
            raise ValueError(
                "audit selection provenance {} does not match".format(key)
            )
    if portable_provenance:
        if (str(selection.get("provenance", {}).get(
                "dataset", "scanrefer")) != dataset_name
                or selection.get("sample_count") != 256):
            raise ValueError("portable audit dataset/sample provenance changed")
    try:
        audit_data_root = Path(
            provenance["data_root"]
        ).expanduser().resolve()
        live_data_root = Path(data_root).expanduser().resolve()
        audit_train_cache = Path(
            provenance["train_cache"]
        ).expanduser().resolve()
        top_train_cache = Path(
            selection["train_cache"]
        ).expanduser().resolve()
        if portable_provenance:
            pass
        elif split == "train":
            expected_train_cache = Path(
                base_binding["path"]
            ).expanduser().resolve()
        else:
            expected_train_cache = (
                live_data_root / "output" / "rec_reranker"
                / "e71_top16" / "train"
            ).resolve()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "audit selection path provenance is invalid: {}".format(error)
        )
    if audit_data_root != live_data_root:
        raise ValueError("audit selection data root does not match")
    if (audit_train_cache != expected_train_cache
            or top_train_cache != expected_train_cache):
        raise ValueError("audit selection train cache does not match")
    return selection


def build_geometry_immutable_metadata(
        args, base_binding, base_manifest, checkpoint_path,
        checkpoint_sha256, checkpoint_epoch, config,
        annotation_sha256=None):
    """Build the exact immutable sidecar manifest fields from live sources."""
    validate_source_provenance(
        args.split,
        base_binding,
        base_manifest,
        checkpoint_sha256,
        checkpoint_epoch,
        config,
        int(base_manifest.get("source_dataset_size", -1)),
        args.data_root,
        portable_provenance=bool(getattr(
            args, "portable_provenance", False
        )),
        dataset_name=str(getattr(args, "dataset", "scanrefer")),
        allow_butd_cls=bool(getattr(args, "allow_butd_cls", False)),
    )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint does not exist: {}".format(
            checkpoint_path
        ))
    data_root = Path(args.data_root).expanduser().resolve()
    audit_path = Path(args.audit_provenance).expanduser().resolve()
    if annotation_sha256 is None:
        annotation_sha256 = _validated_annotation_snapshot(
            data_root, args.split
        )["sha256"]
    if (not bool(getattr(args, "portable_provenance", False))
            and annotation_sha256 != EXPECTED_ANNOTATION_SHA256[args.split]):
        raise ValueError(
            "annotation JSON SHA-256 is not authoritative for {}".format(
                args.split
            )
        )
    audit_selection, audit_snapshot = _load_stable_json_snapshot(
        audit_path, "audit selection"
    )
    audit_sha256 = audit_snapshot["sha256"]
    _load_and_validate_audit_selection(
        audit_selection,
        audit_sha256,
        data_root,
        args.split,
        base_binding,
        checkpoint_sha256,
        checkpoint_epoch,
        portable_provenance=bool(getattr(
            args, "portable_provenance", False
        )),
        audit_train_cache=getattr(args, "audit_train_cache", None),
        dataset_name=str(getattr(args, "dataset", "scanrefer")),
    )
    variants = [
        dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    metadata = {
        "geometry_cache_schema_version": GEOMETRY_CACHE_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "geometry_feature_names": list(REC_MASK_GEOMETRY_FEATURE_NAMES),
        "variant_names": [value["name"] for value in variants],
        "variant_configs": variants,
        "regressed_variant_index": 0,
        "min_points": PRODUCTION_MIN_POINTS,
        "max_point_fraction": PRODUCTION_MAX_POINT_FRACTION,
        "split": args.split,
        "dataset_size": int(base_manifest["dataset_size"]),
        "source_dataset_size": int(base_manifest["source_dataset_size"]),
        "candidate_rule": copy.deepcopy(base_binding["candidate_rule"]),
        "target_iou_policy": "root_only",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_epoch": int(checkpoint_epoch),
        "model_inputs": copy.deepcopy(base_binding["model_inputs"]),
        "backbone_config": copy.deepcopy(base_binding["backbone_config"]),
        "extraction_batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "shard_size": int(args.shard_size),
        "base_cache_binding": copy.deepcopy(base_binding),
        "annotation_sha256": annotation_sha256,
        "audit_provenance": {
            "panel": str(audit_path),
            "sha256": audit_sha256,
        },
        "filter_non_gt_boxes": bool(getattr(config, "butd_cls", False)),
    }
    if set(metadata) != set(GEOMETRY_IMMUTABLE_METADATA_FIELDS):
        raise RuntimeError("geometry immutable metadata fields drifted")
    return metadata


def _forbidden_keys_in(value):
    if isinstance(value, dict):
        found = set(value).intersection(FORBIDDEN_DEPLOYABLE_KEYS)
        for child in value.values():
            found.update(_forbidden_keys_in(child))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for child in value:
            found.update(_forbidden_keys_in(child))
        return found
    return set()


def _require_deployable_payload(value, label):
    if not isinstance(value, dict):
        raise ValueError("{} deployable payload must be a mapping".format(label))
    forbidden = _forbidden_keys_in(value)
    if forbidden:
        raise ValueError(
            "{} deployable payload contains target fields: {}".format(
                label, ", ".join(sorted(forbidden))
            )
        )


def _base_rows_for_indices(base_rows_by_index, dataset_indices):
    rows = []
    for dataset_index in dataset_indices:
        index = int(dataset_index)
        if index not in base_rows_by_index:
            raise ValueError("dataset index {} is absent from base cache".format(
                index
            ))
        row = base_rows_by_index[index]
        if row.get("dataset_index") != index:
            raise ValueError("base cache index mapping is inconsistent")
        rows.append(row)
    return rows


def _canonicalize_candidate_parents(
        candidate_batch, base_rows_by_index, dataset_indices):
    """Return a target-free candidate copy whose geometry parents are cached."""
    rows = _base_rows_for_indices(base_rows_by_index, dataset_indices)
    for key in ("boxes", "query_indices", "valid_mask"):
        if not isinstance(candidate_batch.get(key), torch.Tensor):
            raise ValueError("fresh candidate {} must be a tensor".format(key))
        if candidate_batch[key].shape[0] != len(rows):
            raise ValueError("fresh candidate batch size does not match indices")
    boxes = torch.stack([
        torch.as_tensor(row["boxes"]).detach() for row in rows
    ]).to(device=candidate_batch["boxes"].device, dtype=torch.float32)
    query_indices = torch.stack([
        torch.as_tensor(row["query_indices"]).detach() for row in rows
    ]).to(device=candidate_batch["query_indices"].device, dtype=torch.int64)
    valid_mask = torch.stack([
        torch.as_tensor(row["valid_mask"]).detach() for row in rows
    ]).to(device=candidate_batch["valid_mask"].device, dtype=torch.bool)
    default_top1_query_index = torch.tensor([
        int(row["default_top1_query_index"]) for row in rows
    ], device=candidate_batch["query_indices"].device, dtype=torch.int64)
    if boxes.shape != candidate_batch["boxes"].shape:
        raise ValueError("cached candidate boxes do not match fresh shape")
    if query_indices.shape != candidate_batch["query_indices"].shape:
        raise ValueError("cached query indices do not match fresh shape")
    if valid_mask.shape != candidate_batch["valid_mask"].shape:
        raise ValueError("cached candidate validity does not match fresh shape")
    canonical = dict(candidate_batch)
    canonical["boxes"] = boxes.contiguous().clone()
    canonical["query_indices"] = query_indices.contiguous().clone()
    canonical["valid_mask"] = valid_mask.contiguous().clone()
    canonical["default_top1_query_index"] = (
        default_top1_query_index.contiguous().clone()
    )
    return canonical


def _validate_geometry_schema(geometry):
    if not isinstance(geometry, dict):
        raise ValueError("geometry builder output must be a mapping")
    if geometry.get("schema_version") != MASK_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("geometry schema version changed")
    expected_variants = [
        dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    expected_names = tuple(value["name"] for value in expected_variants)
    if tuple(geometry.get("variant_names", ())) != expected_names:
        raise ValueError("geometry variant names changed")
    configs = geometry.get("variant_configs")
    if (not isinstance(configs, (list, tuple))
            or [dict(value) for value in configs] != expected_variants):
        raise ValueError("geometry variant configs changed")
    if tuple(geometry.get("geometry_feature_names", ())) != tuple(
            REC_MASK_GEOMETRY_FEATURE_NAMES):
        raise ValueError("geometry feature names changed")
    if geometry.get("min_points") != PRODUCTION_MIN_POINTS:
        raise ValueError("geometry min_points changed")
    if float(geometry.get("max_point_fraction", float("nan"))) != (
            PRODUCTION_MAX_POINT_FRACTION):
        raise ValueError("geometry max_point_fraction changed")

    boxes = geometry.get("boxes")
    valid = geometry.get("valid_mask")
    features = geometry.get("geometry_features")
    if (not isinstance(boxes, torch.Tensor) or boxes.dim() != 4
            or boxes.shape[-2:] != (len(expected_variants), 6)):
        raise ValueError("geometry boxes do not match Kx7x6 schema")
    if not isinstance(valid, torch.Tensor) or valid.shape != boxes.shape[:3]:
        raise ValueError("geometry validity does not match boxes")
    expected_feature_shape = boxes.shape[:3] + (
        len(REC_MASK_GEOMETRY_FEATURE_NAMES),
    )
    if (not isinstance(features, torch.Tensor)
            or features.shape != expected_feature_shape):
        raise ValueError("geometry features do not match Kx7x25 schema")


def _validated_parity_maxima(value, allow_empty=True):
    if not isinstance(value, dict):
        raise ValueError("parity maxima must be a mapping")
    if not value and allow_empty:
        return {}
    if set(value) != set(PARITY_FIELDS):
        raise ValueError("parity fields do not match extraction schema")
    result = {}
    for name in PARITY_FIELDS:
        maximum = value[name]
        if (not isinstance(maximum, (int, float))
                or isinstance(maximum, bool)
                or not math.isfinite(float(maximum))
                or float(maximum) < 0.0):
            raise ValueError("parity maximum {} is invalid".format(name))
        result[name] = float(maximum)
    return result


def merge_parity_maxima(current, update):
    """Merge two exact parity diagnostic mappings by fieldwise maximum."""
    current = _validated_parity_maxima(current, allow_empty=True)
    update = _validated_parity_maxima(update, allow_empty=True)
    if not current:
        return dict(update)
    if not update:
        return dict(current)
    return {
        name: max(current[name], update[name]) for name in PARITY_FIELDS
    }


def _batch_identity(value, index):
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu()
    return value[index]


def _cpu_tensor(value, dtype):
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype)
    return tensor.contiguous().clone()


def build_evaluator_valid(geometry, inputs, filter_non_gt_boxes):
    """Build the exact deployable evaluator-valid geometry mask."""
    if type(filter_non_gt_boxes) is not bool:
        raise TypeError("filter_non_gt_boxes must be boolean")
    if not isinstance(geometry, dict) or not isinstance(inputs, dict):
        raise TypeError("geometry and inputs must be mappings")
    geometry_valid = geometry.get("valid_mask")
    geometry_boxes = geometry.get("boxes")
    if (not isinstance(geometry_valid, torch.Tensor)
            or geometry_valid.dtype != torch.bool
            or not isinstance(geometry_boxes, torch.Tensor)
            or geometry_boxes.shape[:-1] != geometry_valid.shape):
        raise ValueError("geometry evaluator tensors are malformed")
    if not filter_non_gt_boxes:
        return geometry_valid.clone()
    detected_boxes = inputs.get("det_boxes")
    detected_valid = inputs.get("det_bbox_label_mask")
    if not isinstance(detected_valid, torch.Tensor):
        raise ValueError("detector validity is missing from deployable inputs")
    return build_detector_overlap_valid(
        geometry_boxes,
        geometry_valid,
        detected_boxes,
        detected_valid.bool(),
        iou_threshold=0.25,
    )


def build_geometry_rows(
        dataset_indices, batch_data, targeted_candidates,
        targeted_geometry, base_rows_by_index, rejection_codes,
        parity_check=assert_candidate_cache_parity, manifest=None,
        evaluator_valid=None):
    """Parity-check and serialize one targeted geometry batch."""
    dataset_indices = [int(value) for value in dataset_indices]
    base_rows = _base_rows_for_indices(
        base_rows_by_index, dataset_indices
    )

    # Fail closed before reading or materializing any geometry row payload.
    parity_maxima = parity_check(
        targeted_candidates,
        base_rows_by_index,
        dataset_indices,
        batch_data["scan_ids"],
        batch_data["target_id"],
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        iou_atol=IOU_PARITY_ATOL,
        iou_rtol=IOU_PARITY_RTOL,
    )
    parity_maxima = _validated_parity_maxima(
        parity_maxima, allow_empty=False
    )

    _validate_geometry_schema(targeted_geometry)
    geometry_ious = targeted_geometry.get("geometry_ious")
    if not isinstance(geometry_ious, torch.Tensor):
        raise ValueError("targeted geometry is missing geometry_ious")
    boxes = targeted_geometry["boxes"]
    valid = targeted_geometry["valid_mask"]
    features = targeted_geometry["geometry_features"]
    if geometry_ious.shape != valid.shape:
        raise ValueError("geometry IoUs do not match geometry validity")
    if (not isinstance(rejection_codes, torch.Tensor)
            or rejection_codes.shape != valid.shape):
        raise ValueError("geometry rejection codes do not match validity")
    if boxes.shape[0] != len(dataset_indices):
        raise ValueError("geometry batch size does not match indices")

    # One batched transfer avoids synchronizing the GPU for every field of
    # every row while preserving the exact serialized per-row tensors.
    cpu_boxes = _cpu_tensor(boxes, torch.float32)
    cpu_valid = _cpu_tensor(valid, torch.bool)
    cpu_features = _cpu_tensor(features, torch.float32)
    cpu_ious = _cpu_tensor(geometry_ious, torch.float32)
    cpu_rejections = _cpu_tensor(rejection_codes, torch.int16)
    if evaluator_valid is None:
        cpu_evaluator_valid = cpu_valid.clone()
    else:
        if (not isinstance(evaluator_valid, torch.Tensor)
                or evaluator_valid.dtype != torch.bool
                or evaluator_valid.shape != valid.shape
                or evaluator_valid.device != valid.device
                or bool((evaluator_valid & ~valid).any().item())):
            raise ValueError(
                "evaluator validity must be a geometry-valid boolean tensor"
            )
        cpu_evaluator_valid = _cpu_tensor(evaluator_valid, torch.bool)

    rows = []
    for batch_index, base_row in enumerate(base_rows):
        candidate_valid = _cpu_tensor(
            base_row["valid_mask"], torch.bool
        )
        geometry_boxes = cpu_boxes[batch_index].contiguous().clone()
        geometry_valid = cpu_valid[batch_index].contiguous().clone()
        row_evaluator_valid = cpu_evaluator_valid[
            batch_index
        ].contiguous().clone()
        geometry_features = cpu_features[batch_index].contiguous().clone()
        row_ious = cpu_ious[batch_index].contiguous().clone()
        row_rejections = cpu_rejections[batch_index].contiguous().clone()

        # Canonical g0 is a bit-exact clone of the bound base row. Applying it
        # after target computation also prevents fresh numerical drift leaking
        # into the persisted regressed slice.
        geometry_boxes[:, 0] = _cpu_tensor(
            base_row["boxes"], torch.float32
        )
        geometry_valid[:, 0] = candidate_valid
        if row_ious.shape[1] > 1:
            row_ious[:, 1:].masked_fill_(
                ~geometry_valid[:, 1:], 0.0
            )
        row_ious[:, 0] = _cpu_tensor(
            base_row["candidate_ious"], torch.float32
        )
        row = {
            "dataset_index": int(base_row["dataset_index"]),
            "scan_id": str(base_row["scan_id"]),
            "target_id": int(base_row["target_id"]),
            "default_top1_query_index": int(
                base_row["default_top1_query_index"]
            ),
            "query_indices": _cpu_tensor(
                base_row["query_indices"], torch.int64
            ),
            "candidate_valid": candidate_valid,
            "geometry_boxes": geometry_boxes.contiguous(),
            "geometry_valid": geometry_valid.contiguous(),
            "evaluator_valid": row_evaluator_valid.contiguous(),
            "geometry_features": geometry_features.contiguous(),
            "geometry_ious": row_ious.contiguous(),
            "source_rejection_codes": row_rejections.contiguous(),
        }
        if manifest is not None:
            validate_geometry_row(row, manifest, base_row=base_row)
        rows.append(row)
    return rows, parity_maxima


def extract_geometry_batch(
        model, inputs, batch_data, dataset_indices, base_rows_by_index,
        manifest=None, candidate_builder=build_rec_candidate_batch,
        geometry_builder=build_rec_mask_geometry_candidates,
        candidate_target_attacher=attach_candidate_targets,
        geometry_target_attacher=attach_rec_mask_geometry_targets,
        rejection_projector=project_variant_rejection_codes,
        parity_check=assert_candidate_cache_parity,
        canonicalize_candidate_parity=False):
    """Run one inference batch while enforcing the GT/deployable boundary."""
    if type(canonicalize_candidate_parity) is not bool:
        raise TypeError("canonicalize_candidate_parity must be boolean")
    dataset_indices = [int(value) for value in dataset_indices]
    _require_deployable_payload(inputs, "model input")
    end_points = model(inputs)
    _require_deployable_payload(end_points, "model output")
    _require_deployable_payload(inputs, "candidate input")

    candidate_rule = (
        manifest["candidate_rule"] if manifest is not None else {
            "topk_per_source": PRODUCTION_TOPK_PER_SOURCE,
            "max_candidates": int(
                torch.as_tensor(
                    base_rows_by_index[dataset_indices[0]]["valid_mask"]
                ).numel()
            ),
        }
    )
    fresh_candidates = candidate_builder(
        end_points,
        inputs,
        topk_per_source=int(candidate_rule["topk_per_source"]),
        max_candidates=int(candidate_rule["max_candidates"]),
    )
    _require_deployable_payload(fresh_candidates, "candidate builder")
    canonical_candidates = _canonicalize_candidate_parents(
        fresh_candidates, base_rows_by_index, dataset_indices
    )
    _require_deployable_payload(end_points, "geometry model output")
    _require_deployable_payload(inputs, "geometry input")
    _require_deployable_payload(canonical_candidates, "canonical candidate")

    geometry = geometry_builder(
        end_points,
        inputs,
        canonical_candidates,
        variant_config={
            "min_points": PRODUCTION_MIN_POINTS,
            "max_point_fraction": PRODUCTION_MAX_POINT_FRACTION,
        },
    )
    _require_deployable_payload(geometry, "geometry builder")
    _validate_geometry_schema(geometry)
    rejection_codes = rejection_projector(geometry)
    filter_non_gt_boxes = (
        manifest.get("filter_non_gt_boxes", False)
        if isinstance(manifest, dict) else False
    )
    evaluator_valid = build_evaluator_valid(
        geometry,
        inputs,
        filter_non_gt_boxes=filter_non_gt_boxes,
    )

    # Ground-truth-bearing batch_data is deliberately introduced only after
    # both deployable builders and their diagnostics have completed.
    parity_candidates = (
        canonical_candidates
        if canonicalize_candidate_parity else fresh_candidates
    )
    targeted_candidates = candidate_target_attacher(
        parity_candidates, batch_data, root_only=True
    )
    targeted_geometry = geometry_target_attacher(
        geometry, batch_data, root_only=True
    )
    return build_geometry_rows(
        dataset_indices=dataset_indices,
        batch_data=batch_data,
        targeted_candidates=targeted_candidates,
        targeted_geometry=targeted_geometry,
        base_rows_by_index=base_rows_by_index,
        rejection_codes=rejection_codes,
        parity_check=parity_check,
        manifest=manifest,
        evaluator_valid=evaluator_valid,
    )


def _batch_size(batch_data):
    scan_ids = batch_data.get("scan_ids") if isinstance(
        batch_data, dict
    ) else None
    if scan_ids is None:
        raise ValueError("batch_data is missing scan_ids")
    try:
        size = len(scan_ids)
    except TypeError:
        raise ValueError("batch_data scan_ids must be batched")
    if size <= 0:
        raise ValueError("empty extraction batches are not allowed")
    return size


def process_geometry_loader(
        loader, model, base_rows_by_index, output_dir, manifest,
        get_inputs, device, stop_after_shards=None,
        extract_batch=extract_geometry_batch,
        append_shard=append_geometry_shard,
        finalize_cache=finalize_geometry_cache,
        move_batch=_move_batch_to_device,
        canonicalize_candidate_parity=False):
    """Consume a sequential loader and commit only lifecycle-valid shards."""
    if type(canonicalize_candidate_parity) is not bool:
        raise TypeError("canonicalize_candidate_parity must be boolean")
    if manifest.get("complete") is True:
        return manifest
    shard_size = manifest.get("shard_size")
    extraction_batch_size = manifest.get("extraction_batch_size")
    if (not isinstance(shard_size, int) or isinstance(shard_size, bool)
            or shard_size <= 0):
        raise ValueError("geometry manifest shard size must be positive")
    if (not isinstance(extraction_batch_size, int)
            or isinstance(extraction_batch_size, bool)
            or extraction_batch_size <= 0
            or shard_size % extraction_batch_size != 0):
        raise ValueError(
            "geometry extraction batch must be positive and divide shard size"
        )
    start_index = manifest.get("sample_count")
    dataset_size = manifest.get("dataset_size")
    if (not isinstance(start_index, int) or isinstance(start_index, bool)
            or start_index < 0
            or not isinstance(dataset_size, int)
            or isinstance(dataset_size, bool)
            or dataset_size <= 0
            or start_index > dataset_size):
        raise ValueError("geometry manifest resume state is invalid")
    if start_index % shard_size != 0:
        raise ValueError(
            "resume must start on a {}-row shard boundary".format(
                shard_size
            )
        )
    if stop_after_shards is not None and (
            not isinstance(stop_after_shards, int)
            or isinstance(stop_after_shards, bool)
            or stop_after_shards <= 0):
        raise ValueError("stop_after_shards must be positive")

    pending_rows = []
    pending_parity = {}
    cumulative_parity = _validated_parity_maxima(
        manifest.get("parity_maxima", {}), allow_empty=True
    )
    cursor = start_index
    newly_committed_shards = 0
    last_commit_time = time.monotonic()
    with torch.inference_mode():
        for batch_data in loader:
            batch_size = _batch_size(batch_data)
            remaining = dataset_size - cursor
            if remaining <= 0:
                raise RuntimeError("loader produced rows beyond dataset end")
            expected_batch_size = min(extraction_batch_size, remaining)
            if batch_size != expected_batch_size:
                if batch_size < extraction_batch_size and (
                        cursor + batch_size < dataset_size):
                    raise RuntimeError(
                        "DataLoader batch was short before dataset end"
                    )
                raise RuntimeError(
                    "DataLoader batch size does not match sequential source"
                )

            moved_batch = move_batch(batch_data, device)
            inputs = get_inputs(moved_batch)
            if not isinstance(inputs, dict):
                raise ValueError("TrainTester._get_inputs must return a mapping")
            inputs = dict(inputs)
            inputs["train"] = False
            dataset_indices = list(range(cursor, cursor + batch_size))
            extraction_kwargs = dict(
                model=model,
                inputs=inputs,
                batch_data=moved_batch,
                dataset_indices=dataset_indices,
                base_rows_by_index=base_rows_by_index,
                manifest=manifest,
            )
            if canonicalize_candidate_parity:
                extraction_kwargs["canonicalize_candidate_parity"] = True
            rows, batch_parity = extract_batch(**extraction_kwargs)
            if not isinstance(rows, list) or len(rows) != batch_size:
                raise RuntimeError("batch extraction returned the wrong row count")
            for offset, row in enumerate(rows):
                if (not isinstance(row, dict)
                        or row.get("dataset_index") != cursor + offset):
                    raise RuntimeError(
                        "batch extraction returned non-sequential indices"
                    )
            pending_rows.extend(rows)
            pending_parity = merge_parity_maxima(
                pending_parity, batch_parity
            )
            cursor += batch_size

            if len(pending_rows) > shard_size:
                raise RuntimeError("pending rows exceeded one geometry shard")
            if len(pending_rows) == shard_size:
                is_terminal = (
                    manifest["sample_count"] + shard_size
                    >= dataset_size
                )
                if not is_terminal:
                    shard_parity = dict(pending_parity)
                    previous_count = manifest["sample_count"]
                    manifest = append_shard(
                        output_dir,
                        manifest,
                        list(pending_rows),
                        shard_parity,
                    )
                    if (manifest.get("sample_count")
                            != previous_count + shard_size
                            or manifest.get("complete") is not False):
                        raise RuntimeError(
                            "geometry append returned an invalid resume state"
                        )
                    cumulative_parity = merge_parity_maxima(
                        cumulative_parity, shard_parity
                    )
                    pending_rows = []
                    pending_parity = {}
                    newly_committed_shards += 1
                    commit_time = time.monotonic()
                    elapsed = commit_time - last_commit_time
                    print(
                        "Committed geometry shard {:06d}: {} rows in "
                        "{:.2f}s ({:.2f} rows/s)".format(
                            len(manifest["shards"]) - 1,
                            shard_size,
                            elapsed,
                            shard_size / max(elapsed, 1e-12),
                        ),
                        flush=True,
                    )
                    last_commit_time = commit_time
                    if (stop_after_shards is not None
                            and newly_committed_shards
                            >= stop_after_shards):
                        return manifest

    if cursor != dataset_size:
        raise RuntimeError("geometry extraction ended before dataset end")
    if len(pending_rows) != dataset_size - manifest["sample_count"]:
        raise RuntimeError("terminal geometry rows do not match manifest tail")
    cumulative_parity = merge_parity_maxima(
        cumulative_parity, pending_parity
    )
    finalized = finalize_cache(
        output_dir,
        manifest,
        list(pending_rows),
        cumulative_parity,
    )
    if finalized.get("complete") is not True:
        raise RuntimeError("geometry finalization did not publish a complete cache")
    return finalized


def _set_deterministic(seed, device):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validate_dataset_order(dataset, base_rows):
    if len(dataset) != len(base_rows):
        raise ValueError("current dataset length does not match base rows")
    annos = getattr(dataset, "annos", None)
    if not isinstance(annos, list) or len(annos) != len(base_rows):
        raise ValueError("dataset annotations are unavailable")
    for index, (annotation, base_row) in enumerate(zip(annos, base_rows)):
        if base_row.get("dataset_index") != index:
            raise ValueError("base cache dataset indices are not sequential")
        if (str(annotation.get("scan_id")) != str(base_row.get("scan_id"))
                or int(annotation.get("target_id"))
                != int(base_row.get("target_id"))):
            raise ValueError(
                "current dataset order differs from base cache at {}".format(
                    index
                )
            )


def _index_base_rows(base_rows):
    indexed = {}
    for expected_index, row in enumerate(base_rows):
        if not isinstance(row, dict) or row.get(
                "dataset_index") != expected_index:
            raise ValueError("base cache rows are not sequential")
        if expected_index in indexed:
            raise ValueError("base cache dataset indices are not unique")
        indexed[expected_index] = row
    return indexed


def run_extraction(args):
    """Run frozen full-split inference and publish or resume its sidecar."""
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    base_cache = Path(args.base_cache).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint does not exist: {}".format(
            checkpoint_path
        ))
    if not base_cache.is_dir():
        raise ValueError("base cache does not exist: {}".format(base_cache))
    if not data_root.is_dir():
        raise ValueError("data root does not exist: {}".format(data_root))
    device = torch.device(args.device)
    if str(device) != PRODUCTION_DEVICE:
        raise ValueError("geometry extraction requires cuda:0")
    if not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")

    os.chdir(str(ROOT))
    _set_deterministic(0, device)
    checkpoint, fingerprint = _load_checkpoint_snapshot(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    checkpoint_epoch = checkpoint.get("epoch")
    if not isinstance(checkpoint_epoch, int) or isinstance(
            checkpoint_epoch, bool):
        raise ValueError("checkpoint epoch is missing")
    config = _prepare_model_config(
        checkpoint, _normalized_data_root(data_root)
    )

    base_rows, base_manifest, base_binding = load_bound_candidate_cache(
        base_cache, args.split
    )
    dataset, annotation_sha256 = _build_validated_dataset(
        config,
        args.split,
        data_root,
        dataset_name=str(getattr(args, "dataset", "scanrefer")),
        portable_provenance=bool(getattr(
            args, "portable_provenance", False
        )),
    )
    _validate_dataset_order(dataset, base_rows)
    validate_source_provenance(
        args.split,
        base_binding,
        base_manifest,
        fingerprint,
        checkpoint_epoch,
        config,
        len(dataset),
        data_root,
        portable_provenance=bool(getattr(
            args, "portable_provenance", False
        )),
        dataset_name=str(getattr(args, "dataset", "scanrefer")),
        allow_butd_cls=bool(getattr(args, "allow_butd_cls", False)),
    )
    metadata = build_geometry_immutable_metadata(
        args,
        base_binding,
        base_manifest,
        checkpoint_path,
        fingerprint,
        checkpoint_epoch,
        config,
        annotation_sha256=annotation_sha256,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = initialize_geometry_cache(
        output_dir,
        metadata,
        overwrite=bool(args.overwrite),
        restart_building=bool(args.restart_building),
    )
    if manifest["complete"]:
        print("Geometry cache already complete: {}".format(output_dir))
        return 0

    start_index = manifest["sample_count"]
    if start_index % int(manifest["shard_size"]) != 0:
        raise ValueError("geometry resume is not shard-aligned")
    base_rows_by_index = _index_base_rows(base_rows)
    loader = _build_loader(
        dataset, start_index, len(dataset), args, device
    )
    model = _load_frozen_model(checkpoint, config, device)
    del checkpoint
    from train_dist_mod import TrainTester

    final_manifest = process_geometry_loader(
        loader=loader,
        model=model,
        base_rows_by_index=base_rows_by_index,
        output_dir=output_dir,
        manifest=manifest,
        get_inputs=TrainTester._get_inputs,
        device=device,
        stop_after_shards=args.stop_after_shards,
        canonicalize_candidate_parity=bool(getattr(
            args, "portable_provenance", False
        )),
    )
    if final_manifest["complete"]:
        print(
            "Published complete geometry cache with {} rows: {}".format(
                final_manifest["sample_count"], output_dir
            ),
            flush=True,
        )
    else:
        print(
            "Stopped after {} committed rows; building remains incomplete: {}"
            .format(final_manifest["sample_count"], str(output_dir) + ".building"),
            flush=True,
        )
    return 0


def main(argv=None):
    return run_extraction(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
