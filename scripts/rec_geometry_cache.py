"""Integrity and schema validation for REC geometry sidecar caches."""

import copy
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat

import torch

from models.rec_candidate_adapter import FEATURE_SCHEMA_VERSION
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
)
from scripts.train_rec_reranker import (
    BACKBONE_CONFIG_KEYS,
    CACHE_SCHEMA_VERSION,
    MODEL_INPUT_KEYS,
    _validate_manifest as _validate_candidate_cache_manifest,
    _validate_row as _validate_candidate_cache_row,
    normalize_backbone_config,
)


GEOMETRY_CACHE_SCHEMA_VERSION = 1
BASE_CACHE_BINDING_VERSION = 1

_SHA256_LENGTH = 64
_SHARD_READ_SIZE = 1024 * 1024
_BASE_BINDING_CONTENT_KEYS = (
    "binding_version",
    "split",
    "sample_count",
    "cache_schema_version",
    "feature_schema_version",
    "feature_dim",
    "candidate_rule",
    "checkpoint_sha256",
    "model_inputs",
    "backbone_config",
    "manifest_sha256",
    "shards",
)
_BASE_BINDING_KEYS = frozenset(
    _BASE_BINDING_CONTENT_KEYS + ("path", "content_sha256")
)
_GEOMETRY_ROW_KEYS = frozenset((
    "dataset_index",
    "scan_id",
    "target_id",
    "default_top1_query_index",
    "query_indices",
    "candidate_valid",
    "geometry_boxes",
    "geometry_valid",
    "evaluator_valid",
    "geometry_features",
    "geometry_ious",
    "source_rejection_codes",
))
_BACKBONE_STRING_KEYS = (
    "model",
    "self_position_embedding",
    "source_choice_selector_sources",
)
_BACKBONE_POSITIVE_INT_KEYS = (
    "num_target",
    "num_decoder_layers",
    "source_choice_selector_hidden_dim",
)
_BACKBONE_BOOL_KEYS = (
    "self_attend",
    "use_soft_token_loss",
    "use_contrastive_align",
    "detect_intermediate",
    "use_source_choice_selector",
)
_GEOMETRY_IMMUTABLE_MANIFEST_FIELDS = (
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
_GEOMETRY_MUTABLE_MANIFEST_FIELDS = (
    "complete",
    "sample_count",
    "shards",
    "parity_maxima",
    "cache_content_digest",
)
_IMMUTABLE_METADATA_DIGEST_FIELD = "immutable_metadata_digest"
_GEOMETRY_MANIFEST_FIELDS = frozenset(
    _GEOMETRY_IMMUTABLE_MANIFEST_FIELDS
    + (_IMMUTABLE_METADATA_DIGEST_FIELD,)
    + _GEOMETRY_MUTABLE_MANIFEST_FIELDS
)
_GEOMETRY_SHARD_PAYLOAD_FIELDS = frozenset((
    "schema",
    "immutable_metadata_digest",
    "base_cache_content_digest",
    "shard_index",
    "row_start",
    "row_end",
    "rows",
    "parity_maxima",
))
_PUBLICATION_VERSION = 1
_PUBLICATION_FIELDS = frozenset((
    "publication_version",
    "stage",
    "old_manifest_digest",
    "new_manifest_digest",
))
_PUBLICATION_STAGES = frozenset(("prepared", "old_moved", "new_installed"))
_RESERVED_GEOMETRY_CACHE_SUFFIXES = (
    ".building",
    ".backup",
    ".publish.json",
    ".publish.json.tmp",
    ".building.lock",
)


class GeometryCacheDurabilityError(ValueError):
    """A rename completed but its parent-directory durability is uncertain."""


def canonical_json_sha256(payload):
    """Return the SHA-256 of the canonical JSON representation of payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path):
    """Hash an existing regular file in bounded-size chunks."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            "SHA-256 input is not an existing regular file: {}".format(
                resolved
            )
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_SHARD_READ_SIZE), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(
            "could not read SHA-256 input {}: {}".format(resolved, error)
        )
    return digest.hexdigest()


def geometry_immutable_metadata_digest(immutable_metadata):
    """Return the canonical digest for the fixed sidecar provenance fields."""
    if not isinstance(immutable_metadata, dict):
        raise ValueError("geometry immutable metadata must be an object")
    if set(immutable_metadata) != set(_GEOMETRY_IMMUTABLE_MANIFEST_FIELDS):
        raise ValueError("geometry immutable metadata fields do not match schema")
    try:
        return canonical_json_sha256(immutable_metadata)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "geometry immutable metadata is not canonical JSON: {}".format(
                error
            )
        )


def _geometry_immutable_metadata_from_manifest(manifest):
    return {
        key: copy.deepcopy(manifest[key])
        for key in _GEOMETRY_IMMUTABLE_MANIFEST_FIELDS
    }


def geometry_cache_content_digest(manifest):
    """Digest every manifest field other than the self-referential digest."""
    if not isinstance(manifest, dict):
        raise ValueError("geometry manifest must be an object")
    content = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "cache_content_digest"
    }
    try:
        return canonical_json_sha256(content)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "geometry cache content is not canonical JSON: {}".format(error)
        )


def _is_strict_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_strict_nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_exact_int(value, expected):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == expected
    )


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _canonical_json_equal(first, second):
    try:
        return canonical_json_sha256(first) == canonical_json_sha256(second)
    except (TypeError, ValueError):
        return False


def _base_binding_content(binding):
    return {
        key: copy.deepcopy(binding[key])
        for key in _BASE_BINDING_CONTENT_KEYS
    }


def _validate_candidate_rule(candidate_rule, context):
    if not isinstance(candidate_rule, dict):
        raise ValueError("{} candidate rule must be an object".format(context))
    for key in ("topk_per_source", "max_candidates"):
        if not _is_strict_positive_int(candidate_rule.get(key)):
            raise ValueError(
                "{} candidate rule {} must be positive".format(context, key)
            )


def _validate_model_provenance(model_inputs, backbone_config, context):
    if (not isinstance(model_inputs, dict)
            or any(not isinstance(model_inputs.get(key), bool)
                   for key in MODEL_INPUT_KEYS)):
        raise ValueError("{} model inputs are invalid".format(context))
    try:
        normalize_backbone_config(backbone_config)
    except ValueError:
        raise ValueError("{} backbone config is invalid".format(context))
    if any(
            not isinstance(backbone_config.get(key), str)
            or not backbone_config[key]
            for key in _BACKBONE_STRING_KEYS):
        raise ValueError("{} backbone config is invalid".format(context))
    if any(
            not _is_strict_positive_int(backbone_config.get(key))
            for key in _BACKBONE_POSITIVE_INT_KEYS):
        raise ValueError("{} backbone config is invalid".format(context))
    if any(
            not isinstance(backbone_config.get(key), bool)
            for key in _BACKBONE_BOOL_KEYS):
        raise ValueError("{} backbone config is invalid".format(context))


def _validate_base_binding_structure(binding, expected_split=None):
    if not isinstance(binding, dict):
        raise ValueError("base cache binding must be an object")
    if set(binding) != _BASE_BINDING_KEYS:
        raise ValueError("base cache binding fields do not match its schema")
    binding_version = binding.get("binding_version")
    if (not isinstance(binding_version, int)
            or isinstance(binding_version, bool)
            or binding_version != BASE_CACHE_BINDING_VERSION):
        raise ValueError("unsupported base cache binding version")

    path = binding.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("base cache binding path is invalid")
    try:
        normalized_path = str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("base cache binding path is invalid: {}".format(error))
    if normalized_path != path:
        raise ValueError("base cache binding path is not resolved")

    split = binding.get("split")
    if split not in ("train", "val"):
        raise ValueError("base cache binding split is invalid")
    if expected_split is not None and split != expected_split:
        raise ValueError("base cache binding split does not match expected split")
    if not _is_strict_positive_int(binding.get("sample_count")):
        raise ValueError("base cache binding sample count must be positive")
    cache_schema_version = binding.get("cache_schema_version")
    if (not isinstance(cache_schema_version, int)
            or isinstance(cache_schema_version, bool)
            or cache_schema_version != CACHE_SCHEMA_VERSION):
        raise ValueError("unsupported base cache binding cache schema")
    if binding.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported base cache binding feature schema")
    if not _is_strict_positive_int(binding.get("feature_dim")):
        raise ValueError("base cache binding feature dimension must be positive")
    _validate_candidate_rule(binding.get("candidate_rule"), "base cache binding")
    if not _is_sha256(binding.get("checkpoint_sha256")):
        raise ValueError("base cache binding checkpoint SHA-256 is invalid")
    _validate_model_provenance(
        binding.get("model_inputs"),
        binding.get("backbone_config"),
        "base cache binding",
    )
    if not _is_sha256(binding.get("manifest_sha256")):
        raise ValueError("base cache binding manifest SHA-256 is invalid")
    if not _is_sha256(binding.get("content_sha256")):
        raise ValueError("base cache binding content SHA-256 is invalid")

    shards = binding.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("base cache binding shards must be a nonempty list")
    total_rows = 0
    for index, descriptor in enumerate(shards):
        if not isinstance(descriptor, dict) or set(descriptor) != {
                "name", "row_count", "sha256"}:
            raise ValueError("base cache binding shard descriptor is invalid")
        expected_name = "shard_{:06d}.pt".format(index)
        if descriptor.get("name") != expected_name:
            raise ValueError(
                "base cache binding shard descriptors are not ordered"
            )
        if not _is_strict_positive_int(descriptor.get("row_count")):
            raise ValueError("base cache binding shard row count is invalid")
        if not _is_sha256(descriptor.get("sha256")):
            raise ValueError("base cache binding shard SHA-256 is invalid")
        total_rows += descriptor["row_count"]
    if total_rows != binding["sample_count"]:
        raise ValueError(
            "base cache binding shard row counts do not match sample count"
        )

    try:
        computed_content = canonical_json_sha256(
            _base_binding_content(binding)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "base cache binding content is not canonical JSON: {}".format(
                error
            )
        )
    if computed_content != binding["content_sha256"]:
        raise ValueError("base cache binding content digest mismatch")
    return binding


def _file_identity(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _hash_live_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            before_identity = _file_identity(os.fstat(handle.fileno()))
            for chunk in iter(lambda: handle.read(_SHARD_READ_SIZE), b""):
                digest.update(chunk)
            after_identity = _file_identity(os.fstat(handle.fileno()))
        path_identity = _file_identity(Path(path).stat())
    except OSError as error:
        raise ValueError(
            "could not read cache snapshot input {}: {}".format(path, error)
        )
    if (before_identity != after_identity
            or after_identity != path_identity):
        raise ValueError("base cache changed during validation")
    return digest.hexdigest(), after_identity


def _read_shard_snapshot(path, shard_name):
    try:
        with Path(path).open("rb") as handle:
            before_identity = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after_identity = _file_identity(os.fstat(handle.fileno()))
        path_identity = _file_identity(Path(path).stat())
    except OSError as error:
        raise ValueError(
            "could not load cache shard {}: {}".format(shard_name, error)
        )
    if (before_identity != after_identity
            or after_identity != path_identity):
        raise ValueError("base cache changed during validation")
    try:
        payload = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError(
            "could not load cache shard {}: {}".format(shard_name, error)
        )
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "cache shard {} does not contain a row list".format(shard_name)
        )
    return snapshot, rows, after_identity


def _read_manifest_snapshot(cache_dir):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "base cache manifest does not exist: {}".format(manifest_path)
        )
    try:
        with manifest_path.open("rb") as handle:
            before_identity = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after_identity = _file_identity(os.fstat(handle.fileno()))
        path_identity = _file_identity(manifest_path.stat())
        manifest = json.loads(snapshot.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("base cache manifest is malformed: {}".format(error))
    if not isinstance(manifest, dict):
        raise ValueError("base cache manifest must contain an object")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("base cache manifest must contain ordered shards")
    for index, shard_name in enumerate(shards):
        if shard_name != "shard_{:06d}.pt".format(index):
            raise ValueError("base cache manifest shards are not contiguous")
    if (before_identity != after_identity
            or after_identity != path_identity):
        raise ValueError("base cache changed during validation")
    return snapshot, manifest, after_identity


def _ordered_real_shards(cache_dir, manifest):
    expected_shards = []
    for index, shard_name in enumerate(manifest["shards"]):
        expected_name = "shard_{:06d}.pt".format(index)
        if shard_name != expected_name:
            raise ValueError("training cache has non-contiguous shard names")
        expected_shards.append(expected_name)
    actual_shards = sorted(
        path.name for path in cache_dir.glob("shard_*.pt") if path.is_file()
    )
    if actual_shards != expected_shards:
        raise ValueError("training cache shards do not match the manifest")
    return expected_shards


def _snapshot_candidate_cache(cache_dir, manifest, expected_split):
    if expected_split not in ("train", "val"):
        raise ValueError("expected_split must be either 'train' or 'val'")
    feature_dim, max_candidates = _validate_candidate_cache_manifest(
        manifest, expected_split
    )
    expected_shards = _ordered_real_shards(cache_dir, manifest)
    descriptors = []
    loaded_rows = []
    shard_identities = {}
    for shard_name in expected_shards:
        shard_path = cache_dir / shard_name
        snapshot, shard_rows, shard_identity = _read_shard_snapshot(
            shard_path, shard_name
        )
        for row in shard_rows:
            _validate_candidate_cache_row(
                row, len(loaded_rows), feature_dim, max_candidates
            )
            loaded_rows.append(row)
        descriptors.append({
            "name": shard_name,
            "row_count": len(shard_rows),
            "sha256": hashlib.sha256(snapshot).hexdigest(),
        })
        shard_identities[shard_name] = shard_identity
    if len(loaded_rows) != manifest["sample_count"]:
        raise ValueError("cache sample count does not match loaded rows")
    return descriptors, loaded_rows, shard_identities


def _verify_stable_cache_snapshot(cache_dir, manifest_snapshot,
                                  manifest_identity, descriptors,
                                  shard_identities):
    try:
        for descriptor in descriptors:
            current_sha = sha256_file(cache_dir / descriptor["name"])
            if current_sha != descriptor["sha256"]:
                raise ValueError("base cache changed during validation")
        current_manifest_snapshot, current_manifest, current_manifest_identity = (
            _read_manifest_snapshot(cache_dir)
        )
        if (current_manifest_snapshot != manifest_snapshot
                or current_manifest_identity != manifest_identity):
            raise ValueError("base cache changed during validation")
        _ordered_real_shards(cache_dir, current_manifest)
        for descriptor in descriptors:
            current_sha, current_identity = _hash_live_file(
                cache_dir / descriptor["name"]
            )
            if (current_sha != descriptor["sha256"]
                    or current_identity != shard_identities[descriptor["name"]]):
                raise ValueError("base cache changed during validation")
    except ValueError as error:
        if "changed during validation" in str(error):
            raise
        raise ValueError(
            "base cache changed during validation: {}".format(error)
        )


def _build_base_cache_binding(cache_dir, expected_split):
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_snapshot, snapshot_manifest, manifest_identity = (
        _read_manifest_snapshot(cache_dir)
    )
    manifest_digest = canonical_json_sha256(snapshot_manifest)
    shard_descriptors, loaded_rows, shard_identities = _snapshot_candidate_cache(
        cache_dir, snapshot_manifest, expected_split
    )

    _verify_stable_cache_snapshot(
        cache_dir,
        manifest_snapshot,
        manifest_identity,
        shard_descriptors,
        shard_identities,
    )

    described_rows = sum(
        descriptor["row_count"] for descriptor in shard_descriptors
    )
    if described_rows != len(loaded_rows):
        raise ValueError(
            "base cache binding shard row counts changed during validation"
        )

    binding = {
        "binding_version": BASE_CACHE_BINDING_VERSION,
        "path": str(cache_dir),
        "split": snapshot_manifest["split"],
        "sample_count": snapshot_manifest["sample_count"],
        "cache_schema_version": snapshot_manifest["cache_schema_version"],
        "feature_schema_version": snapshot_manifest["feature_schema_version"],
        "feature_dim": snapshot_manifest["feature_dim"],
        "candidate_rule": copy.deepcopy(snapshot_manifest["candidate_rule"]),
        "checkpoint_sha256": snapshot_manifest["checkpoint_sha256"],
        "model_inputs": copy.deepcopy(snapshot_manifest["model_inputs"]),
        "backbone_config": copy.deepcopy(snapshot_manifest["backbone_config"]),
        "manifest_sha256": manifest_digest,
        "shards": shard_descriptors,
    }
    binding["content_sha256"] = canonical_json_sha256(
        _base_binding_content(binding)
    )
    _validate_base_binding_structure(binding, expected_split)
    return binding, loaded_rows, snapshot_manifest


def build_base_cache_binding(cache_dir, expected_split):
    """Validate and cryptographically bind a complete candidate cache."""
    binding, _, _ = _build_base_cache_binding(cache_dir, expected_split)
    return binding


def load_bound_candidate_cache(cache_dir, expected_split):
    """Load rows, manifest, and binding from one stable cache snapshot."""
    binding, rows, manifest = _build_base_cache_binding(
        cache_dir, expected_split
    )
    return rows, manifest, binding


def _validate_and_load_base_cache_binding(cache_dir, binding, expected_split):
    _validate_base_binding_structure(binding, expected_split)
    actual, loaded_rows, manifest = _build_base_cache_binding(
        cache_dir, expected_split
    )
    if not _canonical_json_equal(binding, actual):
        raise ValueError("base cache binding does not match cache contents")
    return actual, loaded_rows, manifest


def validate_base_cache_binding(cache_dir, binding, expected_split):
    """Recompute a base binding and reject any caller or cache mismatch."""
    actual, _, _ = _validate_and_load_base_cache_binding(
        cache_dir, binding, expected_split
    )
    return actual


def _require_hash_field(manifest, key, label):
    if not _is_sha256(manifest.get(key)):
        raise ValueError("geometry manifest {} is invalid".format(label))


def _validate_geometry_shards(manifest):
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("geometry manifest shards must be a list")
    sample_count = manifest["sample_count"]
    shard_size = manifest["shard_size"]
    complete = manifest["complete"]
    total_rows = 0
    for index, descriptor in enumerate(shards):
        if not isinstance(descriptor, dict) or set(descriptor) != {
                "name", "row_count", "sha256"}:
            raise ValueError("geometry manifest shard descriptor is invalid")
        if descriptor.get("name") != "shard_{:06d}.pt".format(index):
            raise ValueError("geometry manifest shards are not contiguous")
        if not _is_strict_positive_int(descriptor.get("row_count")):
            raise ValueError("geometry manifest shard row count is invalid")
        if not _is_sha256(descriptor.get("sha256")):
            raise ValueError("geometry manifest shard SHA-256 is invalid")
        row_count = descriptor["row_count"]
        if not complete and row_count != shard_size:
            raise ValueError(
                "incomplete geometry cache shards must have the full shard "
                "size"
            )
        if complete:
            if index < len(shards) - 1 and row_count != shard_size:
                raise ValueError(
                    "only the final complete geometry shard may be short"
                )
            if row_count > shard_size:
                raise ValueError("geometry manifest shard exceeds shard size")
        total_rows += descriptor["row_count"]
    if total_rows != sample_count:
        raise ValueError(
            "geometry manifest shard rows do not match sample count"
        )
    if sample_count and not shards:
        raise ValueError("geometry manifest is missing committed shards")
    if not sample_count and shards:
        raise ValueError("empty geometry cache cannot contain committed shards")


def validate_geometry_manifest(manifest, expected_split,
                               require_complete=False):
    """Validate geometry cache provenance and committed state."""
    if not isinstance(manifest, dict):
        raise ValueError("geometry manifest must be an object")
    missing_fields = _GEOMETRY_MANIFEST_FIELDS - set(manifest)
    unexpected_fields = set(manifest) - _GEOMETRY_MANIFEST_FIELDS
    if missing_fields or unexpected_fields:
        details = []
        if missing_fields:
            details.append("missing {}".format(
                ", ".join(sorted(missing_fields))
            ))
        if unexpected_fields:
            details.append("unexpected {}".format(
                ", ".join(sorted(unexpected_fields))
            ))
        raise ValueError(
            "geometry manifest fields do not match its schema: {}".format(
                "; ".join(details)
            )
        )
    if expected_split not in ("train", "val"):
        raise ValueError("expected split must be either 'train' or 'val'")
    if not isinstance(require_complete, bool):
        raise ValueError("require_complete must be boolean")
    if not _is_exact_int(
            manifest.get("geometry_cache_schema_version"),
            GEOMETRY_CACHE_SCHEMA_VERSION):
        raise ValueError("unsupported geometry cache schema")
    if manifest.get("geometry_schema_version") != MASK_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported geometry feature schema")
    if manifest.get("geometry_feature_names") != list(
            REC_MASK_GEOMETRY_FEATURE_NAMES):
        raise ValueError("geometry feature names do not match schema")

    variant_names = manifest.get("variant_names")
    variant_configs = manifest.get("variant_configs")
    if (not isinstance(variant_names, list) or not variant_names
            or not all(isinstance(name, str) and name for name in variant_names)
            or len(set(variant_names)) != len(variant_names)
            or not isinstance(variant_configs, list)
            or len(variant_configs) != len(variant_names)
            or not all(isinstance(config, dict) for config in variant_configs)):
        raise ValueError("geometry variant schema is invalid")
    regressed_indices = [
        index for index, config in enumerate(variant_configs)
        if config.get("source") == "regressed"
    ]
    if (regressed_indices != [0]
            or not _is_exact_int(
                manifest.get("regressed_variant_index"), 0)):
        raise ValueError(
            "geometry manifest requires one regressed variant at index zero"
        )
    canonical_configs = [
        dict(config) for config in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    if variant_names != [config["name"] for config in canonical_configs]:
        raise ValueError("geometry variant names do not match schema")
    if not _canonical_json_equal(variant_configs, canonical_configs):
        raise ValueError("geometry variant configs do not match schema")

    if not _is_strict_positive_int(manifest.get("min_points")):
        raise ValueError("geometry manifest min_points must be positive")
    max_fraction = manifest.get("max_point_fraction")
    if (not isinstance(max_fraction, (int, float))
            or isinstance(max_fraction, bool)
            or not math.isfinite(float(max_fraction))
            or not 0.0 < float(max_fraction) <= 1.0):
        raise ValueError("geometry manifest max point fraction is invalid")
    if manifest.get("split") != expected_split:
        raise ValueError("geometry manifest does not match expected split")

    dataset_size = manifest.get("dataset_size")
    source_dataset_size = manifest.get("source_dataset_size")
    sample_count = manifest.get("sample_count")
    if not _is_strict_positive_int(dataset_size):
        raise ValueError("geometry manifest dataset size must be positive")
    if not _is_strict_positive_int(source_dataset_size):
        raise ValueError(
            "geometry manifest source dataset size must be a positive integer"
        )
    if source_dataset_size != dataset_size:
        raise ValueError(
            "geometry manifest source dataset size does not match dataset size"
        )
    if (not _is_strict_nonnegative_int(sample_count)
            or sample_count > dataset_size):
        raise ValueError("geometry manifest sample count is invalid")

    _validate_candidate_rule(
        manifest.get("candidate_rule"), "geometry manifest"
    )
    if manifest.get("target_iou_policy") != "root_only":
        raise ValueError("geometry manifest target IoU policy is invalid")
    if (not isinstance(manifest.get("checkpoint_path"), str)
            or not manifest["checkpoint_path"]):
        raise ValueError("geometry manifest checkpoint path is invalid")
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha):
        raise ValueError("geometry manifest checkpoint SHA-256 is invalid")
    if not _is_strict_nonnegative_int(manifest.get("checkpoint_epoch")):
        raise ValueError("geometry manifest checkpoint epoch is invalid")
    _validate_model_provenance(
        manifest.get("model_inputs"),
        manifest.get("backbone_config"),
        "geometry manifest",
    )

    if not _is_strict_positive_int(manifest.get("extraction_batch_size")):
        raise ValueError(
            "geometry manifest extraction batch size must be positive"
        )
    if not _is_strict_nonnegative_int(manifest.get("num_workers")):
        raise ValueError(
            "geometry manifest worker count must be non-negative"
        )
    if not _is_strict_positive_int(manifest.get("shard_size")):
        raise ValueError("geometry manifest shard size must be positive")
    if manifest["shard_size"] % manifest["extraction_batch_size"] != 0:
        raise ValueError(
            "geometry manifest shard size must align to extraction batch size"
        )

    binding = manifest.get("base_cache_binding")
    _validate_base_binding_structure(binding, expected_split)
    if binding["sample_count"] != dataset_size:
        raise ValueError("geometry manifest base binding size mismatch")
    if not _canonical_json_equal(
            binding["candidate_rule"], manifest["candidate_rule"]):
        raise ValueError("geometry manifest base candidate rule mismatch")
    if binding["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("geometry manifest base checkpoint mismatch")
    if not _canonical_json_equal(
            binding["model_inputs"], manifest["model_inputs"]):
        raise ValueError("geometry manifest base model inputs mismatch")
    if not _canonical_json_equal(
            binding["backbone_config"], manifest["backbone_config"]):
        raise ValueError("geometry manifest base backbone config mismatch")

    _require_hash_field(manifest, "annotation_sha256", "annotation SHA-256")
    audit = manifest.get("audit_provenance")
    if (not isinstance(audit, dict)
            or not isinstance(audit.get("panel"), str)
            or not audit["panel"]
            or not _is_sha256(audit.get("sha256"))):
        raise ValueError("geometry manifest audit provenance is invalid")
    if not isinstance(manifest.get("filter_non_gt_boxes"), bool):
        raise ValueError("geometry manifest filter policy must be boolean")

    complete = manifest.get("complete")
    if not isinstance(complete, bool):
        raise ValueError("geometry manifest complete state must be boolean")
    if require_complete and not complete:
        raise ValueError("geometry cache is not complete")
    if complete and sample_count != dataset_size:
        raise ValueError("complete geometry cache has an incomplete sample count")
    if not complete and sample_count % manifest["shard_size"] != 0:
        raise ValueError(
            "incomplete geometry cache sample count must align to shard size"
        )
    _validate_geometry_shards(manifest)

    parity = manifest.get("parity_maxima")
    if (not isinstance(parity, dict)
            or (not parity and sample_count != 0)):
        raise ValueError("geometry manifest parity maxima are invalid")
    for key, value in parity.items():
        if not isinstance(key, str) or not key:
            raise ValueError("geometry manifest parity maxima are invalid")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("geometry manifest parity maxima are invalid")
    immutable_digest = manifest.get(_IMMUTABLE_METADATA_DIGEST_FIELD)
    if not _is_sha256(immutable_digest):
        raise ValueError("geometry manifest immutable metadata digest is invalid")
    try:
        computed_immutable_digest = geometry_immutable_metadata_digest(
            _geometry_immutable_metadata_from_manifest(manifest)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "geometry manifest immutable metadata digest is invalid: {}".format(
                error
            )
        )
    if immutable_digest != computed_immutable_digest:
        raise ValueError("geometry manifest immutable metadata digest mismatch")
    _require_hash_field(
        manifest, "cache_content_digest", "cache content digest"
    )
    if manifest["cache_content_digest"] != geometry_cache_content_digest(
            manifest):
        raise ValueError("geometry manifest cache content digest mismatch")
    return manifest


def _validate_tensor(row, key, dtype, shape):
    value = row.get(key)
    if not isinstance(value, torch.Tensor) or value.dtype != dtype:
        raise ValueError(
            "geometry row {} has invalid {} dtype".format(
                row.get("dataset_index", "?"), key
            )
        )
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            "geometry row {} has invalid {} shape".format(
                row.get("dataset_index", "?"), key
            )
        )
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError(
            "geometry row {} {} must be contiguous CPU storage".format(
                row.get("dataset_index", "?"), key
            )
        )
    return value


def _validate_geometry_row(row, manifest, base_row=None,
                           manifest_validated=False):
    if not isinstance(row, dict):
        raise ValueError("geometry row must be an object")
    unexpected = set(row) - _GEOMETRY_ROW_KEYS
    if unexpected:
        raise ValueError(
            "geometry row contains duplicated base payload or target payload: "
            "{}".format(", ".join(sorted(unexpected)))
        )
    missing = _GEOMETRY_ROW_KEYS - set(row)
    if missing:
        raise ValueError(
            "geometry row schema is missing {}".format(
                ", ".join(sorted(missing))
            )
        )
    if not manifest_validated:
        validate_geometry_manifest(
            manifest, manifest.get("split") if isinstance(manifest, dict)
            else None, require_complete=False
        )

    index = row.get("dataset_index")
    if not _is_strict_nonnegative_int(index):
        raise ValueError("geometry row dataset index is invalid")
    if not isinstance(row.get("scan_id"), str) or not row["scan_id"]:
        raise ValueError("geometry row {} scan identity is invalid".format(index))
    for key in ("target_id", "default_top1_query_index"):
        if not isinstance(row.get(key), int) or isinstance(row.get(key), bool):
            raise ValueError(
                "geometry row {} identity field {} is invalid".format(
                    index, key
                )
            )

    num_candidates = manifest["candidate_rule"]["max_candidates"]
    num_variants = len(manifest["variant_names"])
    feature_dim = len(manifest["geometry_feature_names"])
    query_indices = _validate_tensor(
        row, "query_indices", torch.int64, (num_candidates,)
    )
    candidate_valid = _validate_tensor(
        row, "candidate_valid", torch.bool, (num_candidates,)
    )
    geometry_boxes = _validate_tensor(
        row, "geometry_boxes", torch.float32,
        (num_candidates, num_variants, 6),
    )
    geometry_valid = _validate_tensor(
        row, "geometry_valid", torch.bool,
        (num_candidates, num_variants),
    )
    evaluator_valid = _validate_tensor(
        row, "evaluator_valid", torch.bool,
        (num_candidates, num_variants),
    )
    geometry_features = _validate_tensor(
        row, "geometry_features", torch.float32,
        (num_candidates, num_variants, feature_dim),
    )
    geometry_ious = _validate_tensor(
        row, "geometry_ious", torch.float32,
        (num_candidates, num_variants),
    )
    rejection_codes = _validate_tensor(
        row, "source_rejection_codes", torch.int16,
        (num_candidates, num_variants),
    )

    if not bool(candidate_valid.any().item()):
        raise ValueError("geometry row has no valid candidates")
    valid_queries = query_indices[candidate_valid]
    if torch.unique(valid_queries).numel() != valid_queries.numel():
        raise ValueError("geometry row has duplicate valid query indices")
    default_matches = (
        (query_indices == row["default_top1_query_index"]) & candidate_valid
    )
    if int(default_matches.sum().item()) != 1:
        raise ValueError("geometry row default query is not uniquely valid")

    parent_valid = candidate_valid[:, None].expand_as(geometry_valid)
    if bool((geometry_valid & ~parent_valid).any().item()):
        raise ValueError("geometry validity includes an invalid parent")
    if not torch.equal(geometry_valid[:, 0], candidate_valid):
        raise ValueError(
            "geometry row regressed validity must equal candidate validity"
        )
    if bool((evaluator_valid & ~geometry_valid).any().item()):
        raise ValueError("geometry evaluator validity exceeds geometry validity")
    if (not manifest["filter_non_gt_boxes"]
            and not torch.equal(evaluator_valid, geometry_valid)):
        raise ValueError(
            "geometry evaluator validity must equal geometry validity"
        )
    if not bool(evaluator_valid.any().item()):
        raise ValueError("geometry row needs at least one evaluator candidate")

    for key, value in (
            ("geometry boxes", geometry_boxes),
            ("geometry features", geometry_features),
            ("geometry IoUs", geometry_ious)):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("{} must be finite".format(key))
    if bool((geometry_boxes[..., 3:][geometry_valid] <= 0.0).any().item()):
        raise ValueError("valid geometry boxes must have positive sizes")
    if bool(((geometry_ious < 0.0) | (geometry_ious > 1.0)).any().item()):
        raise ValueError("geometry IoU values must lie in [0, 1]")
    if bool((geometry_ious[~geometry_valid] != 0.0).any().item()):
        raise ValueError("invalid geometry IoUs must equal zero")
    if bool((rejection_codes < 0).any().item()):
        raise ValueError("geometry source rejection codes must be nonnegative")

    if base_row is not None:
        if not isinstance(base_row, dict):
            raise ValueError("base row must be an object")
        identity_keys = (
            "dataset_index", "scan_id", "target_id",
            "default_top1_query_index",
        )
        if any(row[key] != base_row.get(key) for key in identity_keys):
            raise ValueError("geometry/base row identity mismatch")
        if not torch.equal(query_indices, base_row.get("query_indices")):
            raise ValueError("geometry/base query indices mismatch")
        if not torch.equal(candidate_valid, base_row.get("valid_mask")):
            raise ValueError("geometry/base candidate validity mismatch")
        if not torch.equal(geometry_boxes[:, 0], base_row.get("boxes")):
            raise ValueError("geometry regressed boxes do not match base boxes")
        if not torch.equal(
                geometry_ious[:, 0], base_row.get("candidate_ious")):
            raise ValueError("geometry regressed IoUs do not match base IoUs")
    return row


def validate_geometry_row(row, manifest, base_row=None):
    """Validate one exact geometry sidecar row."""
    return _validate_geometry_row(row, manifest, base_row)


def _validate_join_provenance(base_manifest, geometry_manifest,
                              verified_base_binding=None):
    if not isinstance(base_manifest, dict):
        raise ValueError("base manifest must be an object")
    binding = geometry_manifest["base_cache_binding"]
    try:
        manifest_digest = canonical_json_sha256(base_manifest)
    except (TypeError, ValueError) as error:
        raise ValueError("base manifest cannot be canonicalized: {}".format(error))
    if manifest_digest != binding["manifest_sha256"]:
        raise ValueError("base manifest provenance does not match binding")
    if not _canonical_json_equal(
            base_manifest.get("checkpoint_epoch"),
            geometry_manifest["checkpoint_epoch"]):
        raise ValueError("base manifest checkpoint epoch provenance mismatch")
    comparisons = (
        ("split", geometry_manifest["split"]),
        ("sample_count", geometry_manifest["dataset_size"]),
        ("cache_schema_version", binding["cache_schema_version"]),
        ("feature_schema_version", binding["feature_schema_version"]),
        ("feature_dim", binding["feature_dim"]),
        ("candidate_rule", geometry_manifest["candidate_rule"]),
        ("checkpoint_sha256", geometry_manifest["checkpoint_sha256"]),
        ("model_inputs", geometry_manifest["model_inputs"]),
        ("backbone_config", geometry_manifest["backbone_config"]),
        ("target_iou_policy", geometry_manifest["target_iou_policy"]),
    )
    for key, expected in comparisons:
        if not _canonical_json_equal(base_manifest.get(key), expected):
            raise ValueError(
                "base manifest {} provenance mismatch".format(key)
            )
    if verified_base_binding is not None:
        _validate_base_binding_structure(
            verified_base_binding, geometry_manifest["split"]
        )
        if not _canonical_json_equal(verified_base_binding, binding):
            raise ValueError(
                "verified base binding does not match geometry cache binding"
            )
        return None
    _, bound_rows, _ = _validate_and_load_base_cache_binding(
        binding["path"], binding, geometry_manifest["split"]
    )
    return bound_rows


def _payloads_equal(first, second):
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and first.device.type == second.device.type
            and torch.equal(first, second)
        )
    if isinstance(first, dict) or isinstance(second, dict):
        return (
            isinstance(first, dict)
            and isinstance(second, dict)
            and set(first) == set(second)
            and all(_payloads_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(
                _payloads_equal(left, right)
                for left, right in zip(first, second)
            )
        )
    return type(first) is type(second) and first == second


def _rows_by_dataset_index(rows, label, expected_count):
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("{} row must be an object".format(label))
        index = row.get("dataset_index")
        if not _is_strict_nonnegative_int(index):
            raise ValueError("{} row dataset index is invalid".format(label))
        if index in indexed:
            raise ValueError("{} rows have duplicate dataset indices".format(label))
        indexed[index] = row
    expected_indices = set(range(expected_count))
    if set(indexed) != expected_indices:
        raise ValueError("{} rows do not cover contiguous dataset indices".format(
            label
        ))
    return indexed


def join_base_and_geometry_rows(base_rows, geometry_rows,
                                base_manifest, geometry_manifest,
                                verified_base_binding=None):
    """Validate provenance and return noncopying base/geometry row pairs."""
    expected_split = (
        geometry_manifest.get("split")
        if isinstance(geometry_manifest, dict) else None
    )
    validate_geometry_manifest(
        geometry_manifest, expected_split, require_complete=True
    )
    bound_base_rows = _validate_join_provenance(
        base_manifest,
        geometry_manifest,
        verified_base_binding=verified_base_binding,
    )
    if not isinstance(base_rows, list) or not isinstance(geometry_rows, list):
        raise ValueError("base and geometry rows must be lists")
    if len(base_rows) != len(geometry_rows):
        raise ValueError("base and geometry cache row counts differ")
    if len(base_rows) != geometry_manifest["sample_count"]:
        raise ValueError("joined rows do not match geometry sample count")
    if (bound_base_rows is not None
            and len(base_rows) != len(bound_base_rows)):
        raise ValueError("joined rows do not match bound base cache row count")

    caller_base_by_index = _rows_by_dataset_index(
        base_rows, "caller base", geometry_manifest["sample_count"]
    )
    geometry_by_index = _rows_by_dataset_index(
        geometry_rows, "geometry", geometry_manifest["sample_count"]
    )
    if bound_base_rows is None:
        feature_dim, max_candidates = _validate_candidate_cache_manifest(
            base_manifest, geometry_manifest["split"]
        )
        for index in range(geometry_manifest["sample_count"]):
            _validate_candidate_cache_row(
                caller_base_by_index[index], index, feature_dim, max_candidates
            )
        bound_base_by_index = caller_base_by_index
    else:
        bound_base_by_index = _rows_by_dataset_index(
            bound_base_rows, "bound base", geometry_manifest["sample_count"]
        )
    joined = []
    for index in range(geometry_manifest["sample_count"]):
        base_row = caller_base_by_index[index]
        geometry_row = geometry_by_index[index]
        bound_base_row = bound_base_by_index[index]
        if not _payloads_equal(base_row, bound_base_row):
            raise ValueError(
                "caller base row does not match bound base cache payload"
            )
        _validate_geometry_row(
            geometry_row,
            geometry_manifest,
            base_row=base_row,
            manifest_validated=True,
        )
        joined.append({"base": base_row, "geometry": geometry_row})
    return joined


def _geometry_cache_paths(output_dir, create_parent=False):
    raw_path = Path(output_dir).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    if raw_path.name in ("", ".", ".."):
        raise ValueError("geometry cache output directory name is invalid")
    if raw_path.name.endswith(_RESERVED_GEOMETRY_CACHE_SUFFIXES):
        raise ValueError("geometry cache output name uses a reserved suffix")
    try:
        parent = raw_path.parent.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("geometry cache output parent is invalid: {}".format(error))
    if create_parent:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(
                "could not create geometry cache parent {}: {}".format(
                    parent, error
                )
            )
    if not parent.is_dir():
        raise ValueError("geometry cache parent directory does not exist")
    final = parent / raw_path.name
    paths = {
        "final": final,
        "building": parent / (raw_path.name + ".building"),
        "backup": parent / (raw_path.name + ".backup"),
        "transaction": parent / (raw_path.name + ".publish.json"),
        "lock": parent / (raw_path.name + ".building.lock"),
    }
    parent_device = parent.stat().st_dev
    for path in paths.values():
        if path.parent != parent:
            raise ValueError("geometry cache paths do not share one parent")
        try:
            entry_stat = os.lstat(str(path))
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(
                "could not inspect geometry cache path {}: {}".format(
                    path, error
                )
            )
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError("geometry cache paths must not be symbolic links")
        if entry_stat.st_dev != parent_device:
            raise ValueError("geometry cache paths must share one filesystem")
    return paths


def _entry_kind(path):
    try:
        entry_stat = os.lstat(str(path))
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError("could not inspect {}: {}".format(path, error))
    if stat.S_ISLNK(entry_stat.st_mode):
        raise ValueError("geometry cache artifacts must not be symbolic links")
    if stat.S_ISDIR(entry_stat.st_mode):
        return "directory"
    if stat.S_ISREG(entry_stat.st_mode):
        return "file"
    raise ValueError("geometry cache artifact has an unsupported file type")


def _require_directory(path, label, required=True):
    kind = _entry_kind(path)
    if kind is None and not required:
        return False
    if kind != "directory":
        raise ValueError("geometry cache {} directory is invalid".format(label))
    return True


def _require_regular_file(path, label):
    if _entry_kind(path) != "file":
        raise ValueError("geometry cache {} file is invalid".format(label))


def _fsync_directory(directory):
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError as error:
        raise ValueError(
            "could not open geometry cache directory {}: {}".format(
                directory, error
            )
        )
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise ValueError(
            "could not fsync geometry cache directory {}: {}".format(
                directory, error
            )
        )
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _geometry_building_lock(paths):
    lock_path = paths["lock"]
    try:
        descriptor = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as error:
        raise ValueError("could not acquire geometry cache building lock: {}".format(
            error
        ))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("geometry cache building lock is already held")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()).encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(paths["final"].parent)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(descriptor)
        except OSError:
            pass


def _temporary_path(path):
    return path.with_name(path.name + ".tmp")


def _unlink_atomic_temp(path, fsync_parent):
    _require_regular_file(path, "atomic temporary")
    try:
        path.unlink()
    except OSError as error:
        raise ValueError("could not remove atomic temporary {}: {}".format(
            path, error
        ))
    _fsync_directory(fsync_parent)


def _cleanup_building_manifest_temp(paths):
    building = paths["building"]
    building_kind = _entry_kind(building)
    if building_kind is None:
        return
    if building_kind != "directory":
        raise ValueError("geometry building path is not a directory")
    manifest_path = building / "manifest.json"
    temporary = _temporary_path(manifest_path)
    temporary_kind = _entry_kind(temporary)
    if temporary_kind is None:
        return
    if temporary_kind != "file":
        raise ValueError("geometry manifest temporary is not a regular file")

    manifest_kind = _entry_kind(manifest_path)
    if manifest_kind == "file":
        _unlink_atomic_temp(temporary, building)
        return
    if manifest_kind is not None:
        raise ValueError("geometry manifest destination has an invalid type")
    try:
        actual_names = {entry.name for entry in building.iterdir()}
    except OSError as error:
        raise ValueError("could not inspect initial geometry build: {}".format(
            error
        ))
    if actual_names != {temporary.name}:
        raise ValueError(
            "geometry manifest temporary has no safe precommit state"
        )
    _unlink_atomic_temp(temporary, building)
    try:
        building.rmdir()
    except OSError as error:
        raise ValueError("could not remove empty precommit build: {}".format(
            error
        ))
    _fsync_directory(building.parent)


def _cleanup_publication_transaction_temp(paths):
    transaction = paths["transaction"]
    temporary = _temporary_path(transaction)
    temporary_kind = _entry_kind(temporary)
    if temporary_kind is None:
        return
    if temporary_kind != "file":
        raise ValueError("geometry publication temporary is not a regular file")
    transaction_kind = _entry_kind(transaction)
    if transaction_kind == "file":
        _unlink_atomic_temp(temporary, transaction.parent)
        return
    if transaction_kind is not None:
        raise ValueError("geometry publication destination has an invalid type")
    if (_entry_kind(paths["backup"]) is not None
            or _entry_kind(paths["building"]) != "directory"):
        raise ValueError(
            "geometry publication temporary has no safe precommit state"
        )
    _unlink_atomic_temp(temporary, transaction.parent)


def _cleanup_atomic_json_temps(paths):
    _cleanup_publication_transaction_temp(paths)
    _cleanup_building_manifest_temp(paths)


def _atomic_write_json(path, value):
    try:
        encoded = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("geometry cache JSON payload is invalid: {}".format(error))
    temporary = _temporary_path(path)
    if _entry_kind(temporary) is not None:
        raise ValueError("geometry cache temporary artifact already exists: {}".format(
            temporary
        ))
    descriptor = None
    try:
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        try:
            _fsync_directory(path.parent)
        except Exception as error:
            raise GeometryCacheDurabilityError(
                "geometry cache JSON rename completed but durability is "
                "uncertain for {}".format(path)
            ) from error
    except OSError:
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if _entry_kind(temporary) == "file":
            try:
                temporary.unlink()
            except OSError:
                pass


def _write_geometry_shard(building_dir, shard_name, payload):
    shard_path = building_dir / shard_name
    temporary = _temporary_path(shard_path)
    if _entry_kind(shard_path) is not None:
        raise ValueError("geometry cache shard destination already exists")
    if _entry_kind(temporary) is not None:
        raise ValueError("geometry cache shard temporary already exists")
    descriptor = None
    try:
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(shard_path))
        _fsync_directory(building_dir)
        return sha256_file(shard_path)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if _entry_kind(temporary) == "file":
            try:
                temporary.unlink()
            except OSError:
                pass


def _read_file_snapshot(path, label):
    _require_regular_file(path, label)
    try:
        with path.open("rb") as handle:
            before_identity = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after_identity = _file_identity(os.fstat(handle.fileno()))
        path_identity = _file_identity(path.stat())
    except OSError as error:
        raise ValueError("could not read geometry cache {}: {}".format(label, error))
    if (before_identity != after_identity or after_identity != path_identity):
        raise ValueError("geometry cache changed during validation")
    return snapshot, after_identity


def _read_geometry_manifest_snapshot(cache_dir):
    manifest_path = cache_dir / "manifest.json"
    snapshot, identity = _read_file_snapshot(manifest_path, "manifest")
    try:
        manifest = json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("geometry cache manifest is malformed: {}".format(error))
    if not isinstance(manifest, dict):
        raise ValueError("geometry cache manifest must contain an object")
    return snapshot, manifest, identity


def _geometry_manifest_identity(manifest):
    try:
        return canonical_json_sha256(manifest)
    except (TypeError, ValueError) as error:
        raise ValueError("geometry manifest cannot be canonicalized: {}".format(error))


def _manifested_geometry_shard_names(manifest):
    return [descriptor["name"] for descriptor in manifest["shards"]]


def _validate_geometry_directory_files(cache_dir, manifest, building,
                                       allow_next_orphan=False):
    _require_directory(cache_dir, "building" if building else "final")
    expected = {"manifest.json"}
    expected.update(_manifested_geometry_shard_names(manifest))
    next_name = "shard_{:06d}.pt".format(len(manifest["shards"]))
    allowed = set(expected)
    if building and not manifest["complete"] and allow_next_orphan:
        allowed.update((next_name, next_name + ".tmp"))
    try:
        actual = {entry.name for entry in cache_dir.iterdir()}
    except OSError as error:
        raise ValueError("could not list geometry cache directory: {}".format(error))
    unexpected = actual - allowed
    missing = expected - actual
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing {}".format(", ".join(sorted(missing))))
        if unexpected:
            details.append("unexpected {}".format(", ".join(sorted(unexpected))))
        raise ValueError("geometry cache files do not match manifest: {}".format(
            "; ".join(details)
        ))
    for name in expected:
        _require_regular_file(cache_dir / name, name)
    orphan_paths = []
    if building and not manifest["complete"] and allow_next_orphan:
        for name in (next_name, next_name + ".tmp"):
            if name in actual:
                _require_regular_file(cache_dir / name, name)
                orphan_paths.append(cache_dir / name)
        if len(orphan_paths) > 1:
            raise ValueError("geometry cache has ambiguous next shard artifacts")
    return orphan_paths


def _validate_parity_maxima(parity_maxima, context, expected_keys=None,
                            allow_empty=False):
    if not isinstance(parity_maxima, dict):
        raise ValueError("{} parity maxima must be an object".format(context))
    if not parity_maxima and not allow_empty:
        raise ValueError("{} parity maxima must not be empty".format(context))
    if expected_keys is not None and set(parity_maxima) != set(expected_keys):
        raise ValueError("{} parity maxima keys do not match".format(context))
    for key, value in parity_maxima.items():
        if not isinstance(key, str) or not key:
            raise ValueError("{} parity maxima keys are invalid".format(context))
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("{} parity maxima values are invalid".format(context))
    return copy.deepcopy(parity_maxima)


def _merge_parity_maxima(current, shard_parity):
    if not current:
        return copy.deepcopy(shard_parity)
    shard_parity = _validate_parity_maxima(
        shard_parity, "geometry shard", expected_keys=current
    )
    return {
        key: max(current[key], shard_parity[key])
        for key in current
    }


def _parity_maxima_equal(first, second):
    return (
        set(first) == set(second)
        and all(float(first[key]) == float(second[key]) for key in first)
    )


def _validate_geometry_shard_payload(payload, manifest, descriptor,
                                     shard_index, row_start, base_rows):
    if not isinstance(payload, dict) or set(payload) != _GEOMETRY_SHARD_PAYLOAD_FIELDS:
        raise ValueError("geometry cache shard payload fields are invalid")
    if not _is_exact_int(payload.get("schema"), GEOMETRY_CACHE_SCHEMA_VERSION):
        raise ValueError("geometry cache shard schema is invalid")
    if payload.get("immutable_metadata_digest") != manifest[
            _IMMUTABLE_METADATA_DIGEST_FIELD]:
        raise ValueError("geometry cache shard immutable metadata digest mismatch")
    if payload.get("base_cache_content_digest") != manifest[
            "base_cache_binding"]["content_sha256"]:
        raise ValueError("geometry cache shard base binding digest mismatch")
    if not _is_exact_int(payload.get("shard_index"), shard_index):
        raise ValueError("geometry cache shard index is invalid")
    if not _is_exact_int(payload.get("row_start"), row_start):
        raise ValueError("geometry cache shard row range is invalid")
    row_end = row_start + descriptor["row_count"]
    if not _is_exact_int(payload.get("row_end"), row_end):
        raise ValueError("geometry cache shard row range is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != descriptor["row_count"]:
        raise ValueError("geometry cache shard row count is invalid")
    if not rows:
        raise ValueError("geometry cache shard cannot be empty")
    shard_parity = _validate_parity_maxima(
        payload.get("parity_maxima"), "geometry shard"
    )
    if base_rows is not None and row_end > len(base_rows):
        raise ValueError("geometry cache shard row range exceeds base cache")
    for offset, row in enumerate(rows):
        expected_index = row_start + offset
        if row.get("dataset_index") != expected_index:
            raise ValueError(
                "geometry cache dataset indices are not contiguous at {}".format(
                    expected_index
                )
            )
        _validate_geometry_row(
            row,
            manifest,
            base_row=(
                base_rows[expected_index]
                if base_rows is not None else None
            ),
            manifest_validated=True,
        )
    return rows, shard_parity


def _load_bound_base_rows(manifest):
    binding = manifest["base_cache_binding"]
    _, rows, _ = _validate_and_load_base_cache_binding(
        binding["path"], binding, manifest["split"]
    )
    if len(rows) != manifest["dataset_size"]:
        raise ValueError("geometry cache base binding row count changed")
    return rows


def _validate_preloaded_base_snapshot(manifest, base_snapshot):
    if (not isinstance(base_snapshot, (tuple, list))
            or len(base_snapshot) != 3):
        raise ValueError(
            "preloaded base snapshot must contain rows, manifest, and binding"
        )
    rows, base_manifest, binding = base_snapshot
    if (not isinstance(rows, list) or not isinstance(base_manifest, dict)
            or not isinstance(binding, dict)):
        raise ValueError("preloaded base snapshot is invalid")
    _validate_join_provenance(
        base_manifest, manifest, verified_base_binding=binding
    )
    if len(rows) != manifest["dataset_size"]:
        raise ValueError("preloaded base snapshot row count is invalid")
    feature_dim, max_candidates = _validate_candidate_cache_manifest(
        base_manifest, manifest["split"]
    )
    indexed = _rows_by_dataset_index(
        rows, "preloaded base", manifest["dataset_size"]
    )
    for index in range(manifest["dataset_size"]):
        _validate_candidate_cache_row(
            indexed[index], index, feature_dim, max_candidates
        )
    return [indexed[index] for index in range(manifest["dataset_size"])]


def _validate_geometry_bundle(cache_dir, expected_split=None,
                              require_complete=True, building=False,
                              allow_next_orphan=False, retain_rows=True,
                              base_snapshot=None):
    if not isinstance(retain_rows, bool):
        raise ValueError("geometry row retention flag must be boolean")
    cache_dir = Path(cache_dir)
    manifest_snapshot, manifest, manifest_identity = (
        _read_geometry_manifest_snapshot(cache_dir)
    )
    if expected_split is None:
        expected_split = manifest.get("split") if isinstance(manifest, dict) else None
    validate_geometry_manifest(manifest, expected_split, require_complete)
    orphan_paths = _validate_geometry_directory_files(
        cache_dir, manifest, building, allow_next_orphan
    )
    base_rows = (
        _load_bound_base_rows(manifest)
        if base_snapshot is None
        else _validate_preloaded_base_snapshot(manifest, base_snapshot)
    )
    loaded_rows = [] if retain_rows else None
    loaded_row_count = 0
    expected_parity = {}
    shard_identities = {}
    row_start = 0
    for shard_index, descriptor in enumerate(manifest["shards"]):
        shard_path = cache_dir / descriptor["name"]
        snapshot, shard_identity = _read_file_snapshot(
            shard_path, descriptor["name"]
        )
        digest = hashlib.sha256(snapshot).hexdigest()
        if digest != descriptor["sha256"]:
            raise ValueError("geometry cache shard SHA-256 does not match manifest")
        try:
            payload = torch.load(io.BytesIO(snapshot), map_location="cpu")
        except Exception as error:
            raise ValueError("could not load geometry cache shard {}: {}".format(
                descriptor["name"], error
            ))
        shard_rows, shard_parity = _validate_geometry_shard_payload(
            payload,
            manifest,
            descriptor,
            shard_index,
            row_start,
            base_rows,
        )
        expected_parity = _merge_parity_maxima(expected_parity, shard_parity)
        loaded_row_count += len(shard_rows)
        if retain_rows:
            loaded_rows.extend(shard_rows)
        row_start += descriptor["row_count"]
        shard_identities[descriptor["name"]] = shard_identity
        del payload, shard_rows, shard_parity, snapshot
    if loaded_row_count != manifest["sample_count"]:
        raise ValueError("geometry cache sample count does not match loaded rows")
    if not _parity_maxima_equal(expected_parity, manifest["parity_maxima"]):
        raise ValueError("geometry cache parity maxima do not match shard payloads")

    # Row/base cross-validation is complete; publication callers only retain
    # the manifest, while the public loader opts into the geometry row list.
    del base_rows

    # Recheck the exact files and manifest after loading snapshot bytes.
    _validate_geometry_directory_files(
        cache_dir, manifest, building, allow_next_orphan
    )
    for descriptor in manifest["shards"]:
        current_digest, current_identity = _hash_live_file(
            cache_dir / descriptor["name"]
        )
        if (current_digest != descriptor["sha256"]
                or current_identity != shard_identities[descriptor["name"]]):
            raise ValueError("geometry cache changed during validation")
    current_snapshot, current_manifest, current_identity = (
        _read_geometry_manifest_snapshot(cache_dir)
    )
    if (current_snapshot != manifest_snapshot
            or current_identity != manifest_identity
            or not _canonical_json_equal(current_manifest, manifest)):
        raise ValueError("geometry cache changed during validation")
    return manifest, loaded_rows, orphan_paths


def _refresh_geometry_manifest_digests(manifest):
    manifest[_IMMUTABLE_METADATA_DIGEST_FIELD] = (
        geometry_immutable_metadata_digest(
            _geometry_immutable_metadata_from_manifest(manifest)
        )
    )
    manifest["cache_content_digest"] = geometry_cache_content_digest(manifest)
    return manifest


def _new_building_geometry_manifest(immutable_metadata):
    immutable = copy.deepcopy(immutable_metadata)
    geometry_immutable_metadata_digest(immutable)
    manifest = immutable
    manifest.update({
        _IMMUTABLE_METADATA_DIGEST_FIELD: "0" * _SHA256_LENGTH,
        "complete": False,
        "sample_count": 0,
        "shards": [],
        "parity_maxima": {},
        "cache_content_digest": "0" * _SHA256_LENGTH,
    })
    _refresh_geometry_manifest_digests(manifest)
    validate_geometry_manifest(manifest, manifest["split"], require_complete=False)
    return manifest


def _same_geometry_immutable_metadata(manifest, immutable_metadata):
    try:
        expected_digest = geometry_immutable_metadata_digest(immutable_metadata)
        actual = _geometry_immutable_metadata_from_manifest(manifest)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        manifest.get(_IMMUTABLE_METADATA_DIGEST_FIELD) == expected_digest
        and _canonical_json_equal(actual, immutable_metadata)
    )


def _cleanup_exact_next_orphan(building_dir, manifest, orphan_paths):
    if not orphan_paths:
        return
    expected_name = "shard_{:06d}.pt".format(len(manifest["shards"]))
    expected_paths = {
        building_dir / expected_name,
        building_dir / (expected_name + ".tmp"),
    }
    if len(orphan_paths) != 1 or orphan_paths[0] not in expected_paths:
        raise ValueError("geometry cache orphan is not the exact next shard")
    orphan_paths[0].unlink()
    _fsync_directory(building_dir)


def _read_current_building_manifest(paths, allow_orphan_cleanup=False):
    _require_directory(paths["building"], "building")
    manifest, _, orphan_paths = _validate_geometry_bundle(
        paths["building"],
        expected_split=None,
        require_complete=False,
        building=True,
        allow_next_orphan=allow_orphan_cleanup,
        retain_rows=False,
    )
    if orphan_paths:
        _cleanup_exact_next_orphan(paths["building"], manifest, orphan_paths)
        manifest, _, orphan_paths = _validate_geometry_bundle(
            paths["building"],
            expected_split=manifest["split"],
            require_complete=False,
            building=True,
            allow_next_orphan=False,
            retain_rows=False,
        )
    return manifest, _load_bound_base_rows(manifest)


def _manifest_matches_disk(paths, caller_manifest, require_complete=False):
    validate_geometry_manifest(
        caller_manifest,
        caller_manifest.get("split") if isinstance(caller_manifest, dict) else None,
        require_complete,
    )
    disk_manifest, base_rows = _read_current_building_manifest(
        paths,
        allow_orphan_cleanup=False,
    )
    if not _canonical_json_equal(caller_manifest, disk_manifest):
        raise ValueError("geometry cache caller manifest is stale")
    return disk_manifest, base_rows


def _append_manifest_matches_disk(paths, caller_manifest):
    expected_split = (
        caller_manifest.get("split")
        if isinstance(caller_manifest, dict) else None
    )
    validate_geometry_manifest(
        caller_manifest, expected_split, require_complete=False
    )
    building = paths["building"]
    _require_directory(building, "building")
    manifest_snapshot, disk_manifest, manifest_identity = (
        _read_geometry_manifest_snapshot(building)
    )
    validate_geometry_manifest(
        disk_manifest, disk_manifest.get("split"), require_complete=False
    )
    if not _canonical_json_equal(caller_manifest, disk_manifest):
        raise ValueError("geometry cache caller manifest is stale")
    _validate_geometry_directory_files(
        building,
        disk_manifest,
        building=True,
        allow_next_orphan=False,
    )

    current_snapshot, current_manifest, current_identity = (
        _read_geometry_manifest_snapshot(building)
    )
    if (current_snapshot != manifest_snapshot
            or current_identity != manifest_identity
            or not _canonical_json_equal(current_manifest, disk_manifest)):
        raise ValueError("geometry cache changed during append preflight")
    _validate_geometry_directory_files(
        building,
        disk_manifest,
        building=True,
        allow_next_orphan=False,
    )
    return disk_manifest


def _build_geometry_shard_payload(manifest, shard_index, row_start, rows,
                                  parity_maxima):
    return {
        "schema": GEOMETRY_CACHE_SCHEMA_VERSION,
        "immutable_metadata_digest": manifest[_IMMUTABLE_METADATA_DIGEST_FIELD],
        "base_cache_content_digest": manifest["base_cache_binding"][
            "content_sha256"
        ],
        "shard_index": shard_index,
        "row_start": row_start,
        "row_end": row_start + len(rows),
        "rows": rows,
        "parity_maxima": copy.deepcopy(parity_maxima),
    }


def _manifest_after_shard(manifest, descriptor, parity_maxima, complete):
    updated = copy.deepcopy(manifest)
    updated["shards"].append(descriptor)
    updated["sample_count"] += descriptor["row_count"]
    updated["complete"] = complete
    updated["parity_maxima"] = copy.deepcopy(parity_maxima)
    _refresh_geometry_manifest_digests(updated)
    validate_geometry_manifest(updated, updated["split"], require_complete=complete)
    return updated


def _manifest_after_finalization(manifest, parity_maxima):
    updated = copy.deepcopy(manifest)
    updated["complete"] = True
    updated["parity_maxima"] = copy.deepcopy(parity_maxima)
    _refresh_geometry_manifest_digests(updated)
    validate_geometry_manifest(updated, updated["split"], require_complete=True)
    return updated


def _manifest_commit_state(building_dir, previous_manifest, expected_manifest):
    try:
        _, actual, _ = _read_geometry_manifest_snapshot(building_dir)
        validate_geometry_manifest(actual, actual.get("split"), require_complete=False)
    except ValueError:
        return "ambiguous", None
    if _canonical_json_equal(actual, expected_manifest):
        return "committed", actual
    if _canonical_json_equal(actual, previous_manifest):
        return "uncommitted", actual
    return "ambiguous", actual


def _rollback_unmanifested_shard(building_dir, shard_name, previous_manifest,
                                 expected_manifest):
    state, observed = _manifest_commit_state(
        building_dir, previous_manifest, expected_manifest
    )
    if state == "committed":
        return observed
    if state != "uncommitted":
        raise ValueError("geometry manifest commit outcome is ambiguous")
    shard_path = building_dir / shard_name
    if _entry_kind(shard_path) == "file":
        shard_path.unlink()
        _fsync_directory(building_dir)
    return None


def _commit_manifest_after_shard(building_dir, previous_manifest,
                                 updated_manifest, shard_name):
    try:
        _atomic_write_json(building_dir / "manifest.json", updated_manifest)
        return updated_manifest
    except GeometryCacheDurabilityError:
        raise
    except Exception:
        committed = _rollback_unmanifested_shard(
            building_dir, shard_name, previous_manifest, updated_manifest
        )
        if committed is not None:
            return committed
        raise


def _commit_manifest_without_shard(building_dir, previous_manifest,
                                   updated_manifest):
    try:
        _atomic_write_json(building_dir / "manifest.json", updated_manifest)
        return updated_manifest
    except GeometryCacheDurabilityError:
        raise
    except Exception:
        state, observed = _manifest_commit_state(
            building_dir, previous_manifest, updated_manifest
        )
        if state == "committed":
            return observed
        if state != "uncommitted":
            raise ValueError("geometry manifest commit outcome is ambiguous")
        raise


def _rollback_prepublished_shard(building_dir, shard_name, previous_manifest):
    try:
        _, observed, _ = _read_geometry_manifest_snapshot(building_dir)
        validate_geometry_manifest(
            observed, observed.get("split"), require_complete=False
        )
    except ValueError:
        raise ValueError("geometry shard publication outcome is ambiguous")
    if not _canonical_json_equal(observed, previous_manifest):
        raise ValueError("geometry shard publication outcome is ambiguous")
    shard_path = building_dir / shard_name
    if _entry_kind(shard_path) == "file":
        shard_path.unlink()
        _fsync_directory(building_dir)


def _validate_complete_geometry_directory(
        cache_dir, expected_identity=None, retain_rows=True):
    manifest, rows, orphan_paths = _validate_geometry_bundle(
        cache_dir,
        expected_split=None,
        require_complete=True,
        building=False,
        allow_next_orphan=False,
        retain_rows=retain_rows,
    )
    if orphan_paths:
        raise ValueError("complete geometry cache has orphan artifacts")
    if (expected_identity is not None
            and _geometry_manifest_identity(manifest) != expected_identity):
        raise ValueError("geometry publication manifest identity mismatch")
    return manifest, rows


def _publication_record(old_manifest_digest, new_manifest_digest, stage):
    record = {
        "publication_version": _PUBLICATION_VERSION,
        "stage": stage,
        "old_manifest_digest": old_manifest_digest,
        "new_manifest_digest": new_manifest_digest,
    }
    _validate_publication_record(record)
    return record


def _validate_publication_record(record):
    if not isinstance(record, dict) or set(record) != _PUBLICATION_FIELDS:
        raise ValueError("geometry publication record fields are invalid")
    if not _is_exact_int(record.get("publication_version"), _PUBLICATION_VERSION):
        raise ValueError("geometry publication record version is invalid")
    if record.get("stage") not in _PUBLICATION_STAGES:
        raise ValueError("geometry publication record stage is invalid")
    old_digest = record.get("old_manifest_digest")
    if old_digest is not None and not _is_sha256(old_digest):
        raise ValueError("geometry publication old manifest identity is invalid")
    if not _is_sha256(record.get("new_manifest_digest")):
        raise ValueError("geometry publication new manifest identity is invalid")
    return record


def _write_publication_record(paths, record):
    _validate_publication_record(record)
    _atomic_write_json(paths["transaction"], record)


def _read_publication_record(paths):
    _require_regular_file(paths["transaction"], "publication transaction")
    snapshot, _ = _read_file_snapshot(paths["transaction"], "publication transaction")
    try:
        record = json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("geometry publication transaction is malformed: {}".format(
            error
        ))
    return _validate_publication_record(record)


def _remove_publication_record(paths):
    transaction = paths["transaction"]
    if _entry_kind(transaction) == "file":
        transaction.unlink()
        _fsync_directory(paths["final"].parent)


def _remove_validated_directory(path):
    _require_directory(path, "publication backup")
    try:
        shutil.rmtree(str(path))
    except OSError as error:
        raise ValueError("could not remove geometry cache directory {}: {}".format(
            path, error
        ))
    _fsync_directory(path.parent)


def _load_publication_bundle(path, label, expected_identity):
    if _entry_kind(path) is None:
        return None
    _require_directory(path, label)
    manifest, _ = _validate_complete_geometry_directory(
        path, expected_identity=expected_identity, retain_rows=False
    )
    return manifest


def _finish_initial_publication(paths, record):
    final_bundle = _load_publication_bundle(
        paths["final"], "final", record["new_manifest_digest"]
    )
    building_bundle = _load_publication_bundle(
        paths["building"], "building", record["new_manifest_digest"]
    )
    if _entry_kind(paths["backup"]) is not None:
        raise ValueError("initial geometry publication has an unexpected backup")
    if (final_bundle is not None and building_bundle is None
            and record["stage"] in ("prepared", "new_installed")):
        _remove_publication_record(paths)
        return final_bundle
    if (final_bundle is None and building_bundle is not None
            and record["stage"] == "prepared"):
        os.replace(str(paths["building"]), str(paths["final"]))
        _fsync_directory(paths["final"].parent)
        _write_publication_record(
            paths,
            _publication_record(
                None, record["new_manifest_digest"], "new_installed"
            ),
        )
        _remove_publication_record(paths)
        return _load_publication_bundle(
            paths["final"], "final", record["new_manifest_digest"]
        )
    raise ValueError("geometry initial publication state is impossible")


def _finish_replacement_publication(paths, record):
    old_digest = record["old_manifest_digest"]
    new_digest = record["new_manifest_digest"]
    final_kind = _entry_kind(paths["final"])
    building_kind = _entry_kind(paths["building"])
    backup_kind = _entry_kind(paths["backup"])
    if final_kind not in (None, "directory"):
        raise ValueError("geometry replacement final state is invalid")
    if building_kind not in (None, "directory"):
        raise ValueError("geometry replacement building state is invalid")
    if backup_kind not in (None, "directory"):
        raise ValueError("geometry replacement backup state is invalid")

    final_manifest = None
    if final_kind is not None:
        try:
            final_manifest = _load_publication_bundle(
                paths["final"], "final", new_digest
            )
        except ValueError:
            final_manifest = _load_publication_bundle(
                paths["final"], "final", old_digest
            )
    building_manifest = (
        _load_publication_bundle(paths["building"], "building", new_digest)
        if building_kind is not None else None
    )
    backup_manifest = (
        _load_publication_bundle(paths["backup"], "backup", old_digest)
        if backup_kind is not None else None
    )

    final_identity = (
        _geometry_manifest_identity(final_manifest)
        if final_manifest is not None else None
    )
    if (final_identity == old_digest and building_manifest is not None
            and backup_manifest is None and record["stage"] == "prepared"):
        # No directory move has committed. Keep the old final and retry later.
        _remove_publication_record(paths)
        return final_manifest
    if (final_manifest is None and building_manifest is not None
            and backup_manifest is not None
            and record["stage"] in ("prepared", "old_moved")):
        # The old final moved to B; W is already fully validated.
        os.replace(str(paths["building"]), str(paths["final"]))
        _fsync_directory(paths["final"].parent)
        _write_publication_record(
            paths, _publication_record(old_digest, new_digest, "new_installed")
        )
        final_manifest = _load_publication_bundle(
            paths["final"], "final", new_digest
        )
        _remove_validated_directory(paths["backup"])
        _remove_publication_record(paths)
        return final_manifest
    if (final_identity == new_digest and building_manifest is None
            and backup_manifest is not None
            and record["stage"] in ("prepared", "old_moved", "new_installed")):
        _remove_validated_directory(paths["backup"])
        _remove_publication_record(paths)
        return final_manifest
    if (final_identity == new_digest and building_manifest is None
            and backup_manifest is None and record["stage"] == "new_installed"):
        _remove_publication_record(paths)
        return final_manifest
    raise ValueError("geometry replacement publication state is impossible")


def _recover_geometry_publication(paths):
    if _entry_kind(paths["transaction"]) is None:
        if _entry_kind(paths["backup"]) is not None:
            raise ValueError("geometry cache backup exists without publication record")
        return None
    record = _read_publication_record(paths)
    if record["old_manifest_digest"] is None:
        return _finish_initial_publication(paths, record)
    return _finish_replacement_publication(paths, record)


def recover_geometry_publication(output_dir):
    """Finish or fail closed on an interrupted final-directory publication."""
    paths = _geometry_cache_paths(output_dir, create_parent=False)
    with _geometry_building_lock(paths):
        _cleanup_atomic_json_temps(paths)
        return _recover_geometry_publication(paths)


def _publish_complete_geometry_bundle(paths, expected_manifest):
    building_manifest, _ = _validate_complete_geometry_directory(
        paths["building"],
        expected_identity=_geometry_manifest_identity(expected_manifest),
        retain_rows=False,
    )
    if not _canonical_json_equal(building_manifest, expected_manifest):
        raise ValueError("geometry building manifest changed before publication")
    if _entry_kind(paths["transaction"]) is not None:
        raise ValueError("geometry publication transaction already exists")
    if _entry_kind(paths["backup"]) is not None:
        raise ValueError("geometry publication backup already exists")

    new_digest = _geometry_manifest_identity(building_manifest)
    if _entry_kind(paths["final"]) is None:
        _write_publication_record(
            paths, _publication_record(None, new_digest, "prepared")
        )
        os.replace(str(paths["building"]), str(paths["final"]))
        _fsync_directory(paths["final"].parent)
        _write_publication_record(
            paths, _publication_record(None, new_digest, "new_installed")
        )
        _remove_publication_record(paths)
        return _load_publication_bundle(paths["final"], "final", new_digest)

    old_manifest, _ = _validate_complete_geometry_directory(
        paths["final"], retain_rows=False
    )
    old_digest = _geometry_manifest_identity(old_manifest)
    _write_publication_record(
        paths, _publication_record(old_digest, new_digest, "prepared")
    )
    os.replace(str(paths["final"]), str(paths["backup"]))
    _fsync_directory(paths["final"].parent)
    _write_publication_record(
        paths, _publication_record(old_digest, new_digest, "old_moved")
    )
    try:
        os.replace(str(paths["building"]), str(paths["final"]))
        _fsync_directory(paths["final"].parent)
    except Exception:
        # Never delete B here. Recovery classifies the actual durable state.
        return _recover_geometry_publication(paths)
    _write_publication_record(
        paths, _publication_record(old_digest, new_digest, "new_installed")
    )
    published = _load_publication_bundle(paths["final"], "final", new_digest)
    _load_publication_bundle(paths["backup"], "backup", old_digest)
    _remove_validated_directory(paths["backup"])
    _remove_publication_record(paths)
    return published


def _create_building_directory(paths, manifest):
    if _entry_kind(paths["building"]) is not None:
        raise ValueError("geometry building directory already exists")
    try:
        paths["building"].mkdir(mode=0o700)
    except OSError as error:
        raise ValueError("could not create geometry building directory: {}".format(
            error
        ))
    _fsync_directory(paths["final"].parent)
    try:
        _atomic_write_json(paths["building"] / "manifest.json", manifest)
    except Exception:
        try:
            paths["building"].rmdir()
            _fsync_directory(paths["final"].parent)
        except OSError:
            pass
        raise


def _remove_recognized_building_directory(paths):
    _read_current_building_manifest(
        paths, allow_orphan_cleanup=False
    )
    _remove_validated_directory(paths["building"])


def _validate_staged_rows(rows, manifest, row_start, base_rows=None):
    if not isinstance(rows, list):
        raise ValueError("geometry shard rows must be a list")
    for offset, row in enumerate(rows):
        expected_index = row_start + offset
        if not isinstance(row, dict) or row.get("dataset_index") != expected_index:
            raise ValueError(
                "geometry cache dataset indices are not contiguous at {}".format(
                    expected_index
                )
            )
        _validate_geometry_row(
            row,
            manifest,
            base_row=(
                base_rows[expected_index]
                if base_rows is not None else None
            ),
            manifest_validated=True,
        )


def initialize_geometry_cache(output_dir, immutable_metadata, overwrite=False,
                              restart_building=False):
    """Create or resume W without exposing it as the final cache directory."""
    if not isinstance(overwrite, bool) or not isinstance(restart_building, bool):
        raise ValueError("geometry cache overwrite and restart flags must be boolean")
    if restart_building and not overwrite:
        raise ValueError("restarting a geometry build requires overwrite=True")
    paths = _geometry_cache_paths(output_dir, create_parent=True)
    requested_manifest = _new_building_geometry_manifest(immutable_metadata)
    # Initialization is a provenance checkpoint even when a matching F exists.
    _load_bound_base_rows(requested_manifest)

    with _geometry_building_lock(paths):
        _cleanup_atomic_json_temps(paths)
        _recover_geometry_publication(paths)
        building_kind = _entry_kind(paths["building"])
        if building_kind is not None and building_kind != "directory":
            raise ValueError("geometry building path is not a directory")
        if building_kind == "directory":
            building_manifest, _ = _read_current_building_manifest(
                paths, allow_orphan_cleanup=True
            )
            if not _same_geometry_immutable_metadata(
                    building_manifest, immutable_metadata):
                if not overwrite:
                    raise ValueError("geometry building immutable metadata mismatch")
                _remove_recognized_building_directory(paths)
            elif restart_building:
                _remove_recognized_building_directory(paths)
            elif building_manifest["complete"]:
                return _publish_complete_geometry_bundle(paths, building_manifest)
            else:
                return building_manifest

        final_kind = _entry_kind(paths["final"])
        if final_kind is not None and final_kind != "directory":
            raise ValueError("geometry final path is not a directory")
        if final_kind == "directory":
            final_manifest, _ = _validate_complete_geometry_directory(
                paths["final"], retain_rows=False
            )
            if (not overwrite and _same_geometry_immutable_metadata(
                    final_manifest, immutable_metadata)):
                return final_manifest
            if not overwrite:
                raise ValueError("geometry final immutable metadata mismatch")

        _create_building_directory(paths, requested_manifest)
        return requested_manifest


def append_geometry_shard(output_dir, manifest, rows, shard_parity_maxima):
    """Commit one nonterminal, full 252-row shard to the building cache."""
    paths = _geometry_cache_paths(output_dir, create_parent=False)
    with _geometry_building_lock(paths):
        _cleanup_atomic_json_temps(paths)
        _recover_geometry_publication(paths)
        current = _append_manifest_matches_disk(paths, manifest)
        if current["complete"]:
            raise ValueError("cannot append to a complete geometry cache")
        shard_size = current["shard_size"]
        if not isinstance(rows, list) or len(rows) != shard_size:
            raise ValueError("geometry append requires exactly 252 rows")
        if current["sample_count"] + shard_size >= current["dataset_size"]:
            raise ValueError(
                "terminal geometry shard must be committed by finalization"
            )
        row_start = current["sample_count"]
        _validate_staged_rows(rows, current, row_start, base_rows=None)
        shard_parity = _validate_parity_maxima(
            shard_parity_maxima,
            "geometry shard",
            expected_keys=current["parity_maxima"] or None,
        )
        merged_parity = _merge_parity_maxima(
            current["parity_maxima"], shard_parity
        )
        shard_index = len(current["shards"])
        shard_name = "shard_{:06d}.pt".format(shard_index)
        payload = _build_geometry_shard_payload(
            current, shard_index, row_start, rows, shard_parity
        )
        try:
            shard_digest = _write_geometry_shard(
                paths["building"], shard_name, payload
            )
        except Exception:
            _rollback_prepublished_shard(
                paths["building"], shard_name, current
            )
            raise
        descriptor = {
            "name": shard_name,
            "row_count": shard_size,
            "sha256": shard_digest,
        }
        updated = _manifest_after_shard(
            current, descriptor, merged_parity, complete=False
        )
        return _commit_manifest_after_shard(
            paths["building"], current, updated, shard_name
        )


def finalize_geometry_cache(output_dir, manifest, final_rows,
                            final_parity_maxima):
    """Commit the terminal 0..252 rows and publish a complete final bundle."""
    paths = _geometry_cache_paths(output_dir, create_parent=False)
    with _geometry_building_lock(paths):
        _cleanup_atomic_json_temps(paths)
        _recover_geometry_publication(paths)
        current, base_rows = _manifest_matches_disk(
            paths,
            manifest,
            require_complete=False,
        )
        if current["complete"]:
            raise ValueError("geometry cache is already complete")
        remaining = current["dataset_size"] - current["sample_count"]
        if remaining < 0 or remaining > current["shard_size"]:
            raise ValueError("geometry finalization requires only the terminal tail")
        if not isinstance(final_rows, list) or len(final_rows) != remaining:
            raise ValueError("geometry final rows do not match the remaining tail")
        _validate_staged_rows(
            final_rows, current, current["sample_count"], base_rows=base_rows
        )
        del base_rows
        final_parity = _validate_parity_maxima(
            final_parity_maxima,
            "geometry final",
            expected_keys=current["parity_maxima"] or None,
        )
        if current["parity_maxima"] and any(
                float(final_parity[key]) < float(current["parity_maxima"][key])
                for key in current["parity_maxima"]):
            raise ValueError("geometry final parity maxima regress committed maxima")

        if remaining:
            shard_index = len(current["shards"])
            shard_name = "shard_{:06d}.pt".format(shard_index)
            payload = _build_geometry_shard_payload(
                current,
                shard_index,
                current["sample_count"],
                final_rows,
                final_parity,
            )
            try:
                shard_digest = _write_geometry_shard(
                    paths["building"], shard_name, payload
                )
            except Exception:
                _rollback_prepublished_shard(
                    paths["building"], shard_name, current
                )
                raise
            descriptor = {
                "name": shard_name,
                "row_count": remaining,
                "sha256": shard_digest,
            }
            updated = _manifest_after_shard(
                current, descriptor, final_parity, complete=True
            )
            committed = _commit_manifest_after_shard(
                paths["building"], current, updated, shard_name
            )
        else:
            updated = _manifest_after_finalization(current, final_parity)
            committed = _commit_manifest_without_shard(
                paths["building"], current, updated
            )
        complete_manifest, _ = _validate_complete_geometry_directory(
            paths["building"],
            expected_identity=_geometry_manifest_identity(committed),
            retain_rows=False,
        )
        return _publish_complete_geometry_bundle(paths, complete_manifest)


def load_geometry_cache(cache_dir, expected_split, base_snapshot=None):
    """Load only one complete final geometry cache, with strict file validation."""
    paths = _geometry_cache_paths(cache_dir, create_parent=False)
    _require_directory(paths["final"], "final")
    manifest, rows, orphan_paths = _validate_geometry_bundle(
        paths["final"],
        expected_split=expected_split,
        require_complete=True,
        building=False,
        allow_next_orphan=False,
        base_snapshot=base_snapshot,
    )
    if orphan_paths:
        raise ValueError("complete geometry cache contains orphan artifacts")
    return rows, manifest
