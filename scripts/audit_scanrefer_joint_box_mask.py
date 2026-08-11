#!/usr/bin/env python
"""Train-only headroom audit for query-consistent ScanRefer box and masks."""

import argparse
from collections.abc import Mapping
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from types import MappingProxyType

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_joint_box_mask import (
    MASK_SOURCE_NAMES,
    compress_point_mask_to_superpoints,
    compute_weighted_mask_candidate_targets,
    select_joint_oracle,
    stage0_gate,
    summarize_joint_oracle,
)
from scripts.audit_scanrefer_mask_geometry import (
    _atomic_torch_save,
    _atomic_write_json,
    _build_replay_loader,
    _manifest_sha256,
    _panel_bucket_counts,
    _prune_dataset_scenes,
    _select_batch_mapping,
    _set_deterministic,
    assert_candidate_cache_parity,
    build_cache_replay_groups,
    load_selected_cache_rows,
    load_train_cache_panel_records,
    select_baseline_stratified_panel,
)


AUDIT_SCHEMA_VERSION = "rec-joint-box-mask-audit-v1"
PROTECTED_ARTIFACT_CONTRACT = MappingProxyType({
    "checkpoint": MappingProxyType({
        "path": "/root/autodl-tmp/DATA_ROOT/output/preserved_best/"
                "mcln_pair_sweep/"
                "mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_"
                "0.57993.pth",
        "sha256": "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208",
        "size": 794125833,
        "mode": 0o444,
    }),
    "parent": MappingProxyType({
        "path": "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/"
                "e71_top16/artifacts/"
                "reranker_h256_d010_lr1e3_seed0_final_contract.pth",
        "sha256": "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b",
        "size": 611713,
        "mode": 0o444,
    }),
    "geometry": MappingProxyType({
        "path": "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/"
                "e71_top16/geometry_artifacts/selected_geometry_reranker.pth",
        "sha256": "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f",
        "size": 704449,
        "mode": 0o444,
    }),
})
# Keep the historical digest-only name available to callers that import it.
EXPECTED_ARTIFACT_SHA256 = {
    name: contract["sha256"]
    for name, contract in PROTECTED_ARTIFACT_CONTRACT.items()
}
INFERENCE_FORBIDDEN_KEYS = frozenset((
    "gt_masks", "candidate_ious", "geometry_ious", "center_label",
    "size_gts", "box_label_mask",
))
SCENE_ID_RE = re.compile(r"\Ascene[0-9]{4}_[0-9]{2}\Z")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FILE_BINDING_FIELDS = ("path", "sha256", "size", "mode")
FILE_IDENTITY_FIELDS = ("device", "inode", "mtime_ns", "ctime_ns")
OUTPUT_NAMES = (
    "selection.json", "rows.pt", "summary.json", "stdout.log",
)
SOURCE_SNAPSHOT_SCHEMA = "runtime-code-snapshot-v1"
TOKENIZER_ASSET_NAMES = (
    "added_tokens.json", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
)
TOKENIZER_REQUIRED_ASSET_NAMES = frozenset((
    "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json",
))
PROJECT_ASSET_NAMES = (
    "data/class_embeddings3d.npy",
    "data/cls_results.json",
    "data/meta_data/scannetv2-labels.combined.tsv",
)
PROJECT_REQUIRED_ASSET_NAMES = frozenset((
    "data/class_embeddings3d.npy",
    "data/meta_data/scannetv2-labels.combined.tsv",
))
ROWS_TOP_LEVEL_KEYS = frozenset((
    "schema", "split", "validation_data_accessed", "logit_thresholds",
    "source_names", "rows",
))
ROW_KEYS = frozenset((
    "dataset_index", "scan_id", "target_id", "query_indices",
    "candidate_valid", "geometry_valid", "geometry_ious",
    "candidate_mask_ious", "legacy_semantic_query", "legacy_mask_iou",
    "geometry_parent_mask_iou", "baseline_flat_index", "baseline_box_iou",
    "joint_oracle_flat_index", "joint_oracle_mask_iou", "selected_box_iou",
    "selected_mask_iou", "selected_uses_joint_query",
))


def _assert_historical_cache_identity(
        candidate_batch, cached_rows, dataset_indices, scan_ids, target_ids):
    """Use the legacy cache for panel identity, not current runtime truth."""
    return assert_candidate_cache_parity(
        candidate_batch,
        cached_rows,
        dataset_indices,
        scan_ids,
        target_ids,
        identity_only=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit frozen ScanRefer joint box-mask headroom on train."
    )
    parser.add_argument("--data-root", default="/root/autodl-tmp/DATA_ROOT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-count", type=int, default=64)
    parser.add_argument("--expressions-per-scene", type=int, default=16)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--cache-extraction-batch-size", type=int, default=12)
    parser.add_argument("--cache-replay-boundaries", type=int, nargs="+", default=[0])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--logit-thresholds", type=float, nargs="+",
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    args = parser.parse_args(argv)
    for name in (
            "scene_count", "expressions_per_scene", "batch_size",
            "cache_extraction_batch_size"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.expressions_per_scene < 3:
        parser.error("--expressions-per-scene must cover all three buckets")
    if args.batch_size != args.cache_extraction_batch_size:
        parser.error("--batch-size must match --cache-extraction-batch-size")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not args.logit_thresholds or any(
            not math.isfinite(value) for value in args.logit_thresholds):
        parser.error("--logit-thresholds must be finite")
    if 0.0 not in args.logit_thresholds:
        parser.error("--logit-thresholds must include the legacy zero threshold")
    if (not args.cache_replay_boundaries
            or args.cache_replay_boundaries[0] != 0
            or sorted(set(args.cache_replay_boundaries))
            != args.cache_replay_boundaries):
        parser.error("--cache-replay-boundaries must be sorted and start at zero")
    return args


def _stat_fingerprint(stat_result):
    return (
        int(stat_result.st_dev), int(stat_result.st_ino),
        int(stat_result.st_mode), int(stat_result.st_size),
        int(stat_result.st_mtime_ns), int(stat_result.st_ctime_ns),
    )


def _read_stable_regular_file(path, label, capture=False):
    """Hash one regular file while rejecting replacement or mutation."""
    lexical_path = Path(os.path.abspath(os.path.expanduser(str(path))))
    try:
        path_before = os.lstat(str(lexical_path))
    except OSError as error:
        raise ValueError("{} does not exist: {}".format(
            label, lexical_path
        )) from error
    if stat.S_ISLNK(path_before.st_mode):
        raise ValueError("{} must not be a symlink: {}".format(
            label, lexical_path
        ))
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError("{} must be a regular file: {}".format(
            label, lexical_path
        ))
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(lexical_path), flags)
    except OSError as error:
        raise ValueError("{} could not be opened safely: {}".format(
            label, lexical_path
        )) from error
    chunks = [] if capture else None
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (_stat_fingerprint(path_before) != _stat_fingerprint(opened)
                or not stat.S_ISREG(opened.st_mode)):
            raise RuntimeError("{} changed before hashing".format(label))
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            if capture:
                chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(str(lexical_path))
    except OSError as error:
        raise RuntimeError("{} changed while hashing".format(label)) from error
    fingerprint = _stat_fingerprint(path_before)
    if (fingerprint != _stat_fingerprint(descriptor_after)
            or fingerprint != _stat_fingerprint(path_after)):
        raise RuntimeError("{} changed while hashing".format(label))
    resolved_path = os.path.realpath(str(lexical_path))
    binding = {
        "path": resolved_path,
        "sha256": digest.hexdigest(),
        "size": int(path_before.st_size),
        "mode": int(stat.S_IMODE(path_before.st_mode)),
        "device": int(path_before.st_dev),
        "inode": int(path_before.st_ino),
        "mtime_ns": int(path_before.st_mtime_ns),
        "ctime_ns": int(path_before.st_ctime_ns),
    }
    content = b"".join(chunks) if capture else None
    return binding, content


def _stable_file_binding(path, label="file"):
    return _read_stable_regular_file(path, label, capture=False)[0]


def _sha256(path):
    return _stable_file_binding(path)["sha256"]


def _portable_file_binding(binding, relative_path):
    return {
        "path": str(relative_path),
        "sha256": binding["sha256"],
        "size": binding["size"],
        "mode": binding["mode"],
    }


def _validate_file_binding(binding, label, require_identity=False):
    if not isinstance(binding, dict):
        raise ValueError("{} binding must be a mapping".format(label))
    for key in FILE_BINDING_FIELDS:
        if key not in binding:
            raise ValueError("{} binding is missing {}".format(label, key))
    if not isinstance(binding["path"], str) or not binding["path"]:
        raise ValueError("{} binding path is invalid".format(label))
    if (not isinstance(binding["sha256"], str)
            or SHA256_RE.match(binding["sha256"]) is None):
        raise ValueError("{} binding SHA-256 is invalid".format(label))
    for key in ("size", "mode"):
        value = binding[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("{} binding {} is invalid".format(label, key))
    if binding["mode"] > 0o7777:
        raise ValueError("{} binding mode is invalid".format(label))
    identity_present = any(
        key in binding for key in FILE_IDENTITY_FIELDS
    )
    if require_identity or identity_present:
        for key in FILE_IDENTITY_FIELDS:
            value = binding.get(key)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ValueError(
                    "{} binding {} is invalid".format(label, key)
                )


def _verify_file_binding(binding, label, actual_path=None,
                         require_identity=False):
    _validate_file_binding(binding, label, require_identity=require_identity)
    path = binding["path"] if actual_path is None else actual_path
    actual = _stable_file_binding(path, label)
    fields = list(FILE_BINDING_FIELDS[1:])
    if require_identity or any(
            key in binding for key in FILE_IDENTITY_FIELDS):
        fields.extend(FILE_IDENTITY_FIELDS)
    mismatches = [
        key for key in fields if binding.get(key) != actual.get(key)
    ]
    if mismatches:
        raise RuntimeError("{} binding changed ({})".format(
            label, ", ".join(mismatches)
        ))
    return actual


def _safe_relative_path(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{} path is invalid".format(label))
    path = Path(value)
    if (path.is_absolute() or value != path.as_posix()
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ValueError("{} path is unsafe".format(label))
    return path


def _canonical_mapping_sha256(mapping):
    payload = json.dumps(
        mapping, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_scene_ids(scene_ids):
    if not isinstance(scene_ids, (list, tuple)) or not scene_ids:
        raise ValueError("selected scene IDs must be a non-empty sequence")
    normalized = []
    for scene_id in scene_ids:
        if (not isinstance(scene_id, str)
                or SCENE_ID_RE.match(scene_id) is None):
            raise ValueError("selected scene ID is unsafe: {!r}".format(
                scene_id
            ))
        normalized.append(scene_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected scene IDs must be unique")
    return sorted(normalized)


def _canonical_path(path):
    return Path(os.path.realpath(os.path.abspath(
        os.path.expanduser(str(path))
    )))


def _directory_scene_ids(directory, suffix, label):
    """Return the exact scene-file inventory without following links."""
    directory = Path(directory)
    try:
        directory_stat = os.lstat(str(directory))
    except OSError as error:
        raise ValueError("{} directory does not exist: {}".format(
            label, directory
        )) from error
    if stat.S_ISLNK(directory_stat.st_mode):
        raise ValueError("{} directory must not be a symlink: {}".format(
            label, directory
        ))
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("{} directory is not a directory: {}".format(
            label, directory
        ))
    try:
        with os.scandir(str(directory)) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as error:
        raise ValueError("{} directory could not be enumerated".format(
            label
        )) from error
    scene_ids = []
    for entry in entries:
        name = entry.name
        if entry.is_symlink():
            raise ValueError("{} contains a symlink: {}".format(label, name))
        if not name.endswith(suffix):
            raise ValueError("{} contains an unexpected file: {}".format(
                label, name
            ))
        scene_id = name[:-len(suffix)]
        if SCENE_ID_RE.match(scene_id) is None:
            raise ValueError("{} contains an unsafe scene ID: {}".format(
                label, scene_id
            ))
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("{} entry could not be inspected: {}".format(
                label, name
            )) from error
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ValueError("{} entry must be a regular file: {}".format(
                label, name
            ))
        scene_ids.append(scene_id)
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("{} scene-file inventory is invalid".format(label))
    return sorted(scene_ids)


def _optional_file_binding(path, label):
    """Bind an optional file while recording absence as ``None``."""
    path = Path(path)
    try:
        path_stat = os.lstat(str(path))
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError("{} could not be inspected".format(label)) from error
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError("{} must not be a symlink".format(label))
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("{} must be a regular file".format(label))
    return _stable_file_binding(path, label)


def _train_scene_ids_from_list(path):
    _binding, content = _read_stable_regular_file(
        path, "ScanRefer train scene list", capture=True
    )
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("ScanRefer train scene list is not UTF-8") from error
    scene_ids = [line.strip() for line in lines if line.strip()]
    if not scene_ids:
        raise ValueError("ScanRefer train scene list is empty")
    return _validated_scene_ids(scene_ids)


def _build_dataset_bindings(data_root, scene_ids, replay_scene_ids=None,
                            project_root=ROOT):
    selected = _validated_scene_ids(scene_ids)
    replay = _validated_scene_ids(
        selected if replay_scene_ids is None else replay_scene_ids
    )
    if not set(selected).issubset(replay):
        raise ValueError("replay scene IDs must include selected scenes")
    root = _canonical_path(data_root)
    project = _canonical_path(project_root)
    scanrefer_name = "ScanRefer" if (root / "ScanRefer").is_dir() \
        else "scanrefer"
    scanrefer_root = root / scanrefer_name
    scene_list_path = scanrefer_root / "ScanRefer_filtered_train.txt"
    listed_train_scene_ids = _train_scene_ids_from_list(scene_list_path)
    superpoint_root = root / "superpoints" / "train"
    discovered_scene_ids = _directory_scene_ids(
        superpoint_root, "_superpoint.pth", "train superpoints"
    )
    all_train_scene_ids = sorted(set(
        listed_train_scene_ids + discovered_scene_ids + replay
    ))
    superpoints_train = {}
    for scene_id in all_train_scene_ids:
        superpoints_train[scene_id] = _stable_file_binding(
            superpoint_root / (scene_id + "_superpoint.pth"),
            "{} train superpoint".format(scene_id),
        )

    tokenizer_root = root / "roberta-base"
    tokenizer_assets = {}
    for relative_name in TOKENIZER_ASSET_NAMES:
        tokenizer_assets[relative_name] = _optional_file_binding(
            tokenizer_root / relative_name,
            "tokenizer asset {}".format(relative_name),
        )
    for relative_name in TOKENIZER_REQUIRED_ASSET_NAMES:
        if tokenizer_assets[relative_name] is None:
            raise ValueError("required tokenizer asset is missing: {}".format(
                relative_name
            ))

    project_assets = {}
    for relative_name in PROJECT_ASSET_NAMES:
        relative = _safe_relative_path(relative_name, "project asset")
        project_assets[relative_name] = _optional_file_binding(
            project / relative,
            "project asset {}".format(relative_name),
        )
    for relative_name in PROJECT_REQUIRED_ASSET_NAMES:
        if project_assets[relative_name] is None:
            raise ValueError("required project asset is missing: {}".format(
                relative_name
            ))
    missing_replay_superpoints = sorted(
        set(replay).difference(superpoints_train)
    )
    if missing_replay_superpoints:
        raise ValueError(
            "replay scenes have no train superpoint: {}".format(
                ", ".join(missing_replay_superpoints)
            )
        )

    bindings = {
        "data_root": str(root),
        "project_root": str(project),
        "scanrefer_directory": scanrefer_name,
        "annotation_json": _stable_file_binding(
            scanrefer_root / "ScanRefer_filtered_train.json",
            "ScanRefer train annotations",
        ),
        "scene_list": _stable_file_binding(
            scene_list_path,
            "ScanRefer train scene list",
        ),
        "train_scans": _stable_file_binding(
            root / "train_v3scans.pkl", "train scans"
        ),
        "scenes": {},
        "superpoints_train": superpoints_train,
        "tokenizer_assets": tokenizer_assets,
        "project_assets": project_assets,
    }
    for scene_id in replay:
        bindings["scenes"][scene_id] = {
            "superpoint": superpoints_train[scene_id],
            "predicted_bboxes": _stable_file_binding(
                root / "group_free_pred_bboxes"
                / "group_free_pred_bboxes_train" / (scene_id + ".npy"),
                "{} train predicted boxes".format(scene_id),
            ),
        }
    return bindings


def _expected_absolute_path(path):
    return os.path.realpath(os.path.abspath(str(path)))


def _lexical_absolute_path(path):
    value = os.path.expanduser(str(path))
    if not os.path.isabs(value):
        value = os.path.join(os.getcwd(), value)
    return os.path.normpath(value)


def _verify_expected_binding(binding, expected_path, label,
                             require_identity=True):
    expected = _expected_absolute_path(expected_path)
    _validate_file_binding(
        binding, label, require_identity=require_identity
    )
    if binding["path"] != expected:
        raise ValueError("{} binding path does not match receipt".format(label))
    _verify_file_binding(
        binding, label, actual_path=expected,
        require_identity=require_identity,
    )


def _verify_optional_expected_binding(binding, expected_path, label):
    expected = Path(expected_path)
    try:
        path_stat = os.lstat(str(expected))
    except FileNotFoundError:
        if binding is not None:
            raise RuntimeError("{} optional file disappeared".format(label))
        return
    except OSError as error:
        raise RuntimeError("{} optional file cannot be inspected".format(
            label
        )) from error
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError("{} must not be a symlink".format(label))
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("{} must be a regular file".format(label))
    if binding is None:
        raise RuntimeError("{} optional file appeared".format(label))
    _verify_expected_binding(binding, expected, label)


def _verify_dataset_bindings(bindings, selected_scene_ids,
                             replay_scene_ids=None):
    if not isinstance(bindings, dict):
        raise ValueError("dataset_inputs must be a mapping")
    expected_keys = {
        "data_root", "project_root", "scanrefer_directory", "annotation_json",
        "scene_list", "train_scans", "scenes", "superpoints_train",
        "tokenizer_assets", "project_assets",
    }
    if set(bindings) != expected_keys:
        raise ValueError("dataset input binding keys are invalid")
    selected = _validated_scene_ids(selected_scene_ids)
    if list(selected_scene_ids) != selected:
        raise ValueError("selected train scene IDs must be sorted and unique")
    replay_input = _validated_scene_ids(
        selected if replay_scene_ids is None else replay_scene_ids
    )
    if replay_scene_ids is not None and list(replay_scene_ids) != replay_input:
        raise ValueError("replay scene IDs must be sorted and unique")
    if not set(selected).issubset(replay_input):
        raise ValueError("replay scene IDs must include selected scenes")
    root_value = bindings.get("data_root")
    if not isinstance(root_value, str) or not os.path.isabs(root_value):
        raise ValueError("dataset input root is invalid")
    root = Path(_expected_absolute_path(root_value))
    if str(root) != root_value:
        raise ValueError("dataset input root must be canonical")
    project_value = bindings.get("project_root")
    if not isinstance(project_value, str) or not os.path.isabs(project_value):
        raise ValueError("project root is invalid")
    project = Path(_expected_absolute_path(project_value))
    if str(project) != project_value:
        raise ValueError("project root must be canonical")
    runtime_scanrefer = "ScanRefer" if (root / "ScanRefer").is_dir() \
        else "scanrefer"
    if bindings.get("scanrefer_directory") != runtime_scanrefer:
        raise RuntimeError("ScanRefer directory selection changed")
    scanrefer = root / runtime_scanrefer
    scenes = bindings.get("scenes")
    if not isinstance(scenes, dict) or sorted(scenes) != replay_input:
        raise ValueError("dataset scene bindings do not match replay scenes")
    expected_scene_keys = {"superpoint", "predicted_bboxes"}
    for scene_id in replay_input:
        scene = scenes.get(scene_id)
        if not isinstance(scene, dict):
            raise ValueError("dataset scene binding must be a mapping")
        if set(scene) != expected_scene_keys:
            raise ValueError("dataset scene binding keys are invalid")

    superpoints_train = bindings.get("superpoints_train")
    if not isinstance(superpoints_train, dict):
        raise ValueError("train superpoint bindings are invalid")
    discovered = _directory_scene_ids(
        root / "superpoints" / "train",
        "_superpoint.pth", "train superpoints",
    )
    if list(superpoints_train) != discovered:
        raise ValueError("train superpoint bindings do not match file set")
    for scene_id in discovered:
        _verify_expected_binding(
            superpoints_train[scene_id],
            root / "superpoints" / "train"
            / (scene_id + "_superpoint.pth"),
            "{} train superpoint".format(scene_id),
        )

    tokenizer_assets = bindings.get("tokenizer_assets")
    if (not isinstance(tokenizer_assets, dict)
            or list(tokenizer_assets) != list(TOKENIZER_ASSET_NAMES)):
        raise ValueError("tokenizer asset bindings are invalid")
    for relative_name in TOKENIZER_ASSET_NAMES:
        _verify_optional_expected_binding(
            tokenizer_assets[relative_name],
            root / "roberta-base" / relative_name,
            "tokenizer asset {}".format(relative_name),
        )
    for relative_name in TOKENIZER_REQUIRED_ASSET_NAMES:
        if tokenizer_assets[relative_name] is None:
            raise ValueError("required tokenizer asset binding is missing")

    project_assets = bindings.get("project_assets")
    if (not isinstance(project_assets, dict)
            or list(project_assets) != list(PROJECT_ASSET_NAMES)):
        raise ValueError("project asset bindings are invalid")
    for relative_name in PROJECT_ASSET_NAMES:
        relative = _safe_relative_path(relative_name, "project asset")
        _verify_optional_expected_binding(
            project_assets[relative_name], project / relative,
            "project asset {}".format(relative_name),
        )
    for relative_name in PROJECT_REQUIRED_ASSET_NAMES:
        if project_assets[relative_name] is None:
            raise ValueError("required project asset binding is missing")
    _verify_expected_binding(
        bindings.get("annotation_json"),
        scanrefer / "ScanRefer_filtered_train.json",
        "ScanRefer train annotations",
    )
    _verify_expected_binding(
        bindings.get("scene_list"),
        scanrefer / "ScanRefer_filtered_train.txt",
        "ScanRefer train scene list",
    )
    _verify_expected_binding(
        bindings.get("train_scans"), root / "train_v3scans.pkl",
        "train scans",
    )
    for scene_id in replay_input:
        scene = scenes.get(scene_id)
        _verify_expected_binding(
            scene.get("superpoint"),
            root / "superpoints" / "train"
            / (scene_id + "_superpoint.pth"),
            "{} train superpoint".format(scene_id),
        )
        _verify_expected_binding(
            scene.get("predicted_bboxes"),
            root / "group_free_pred_bboxes"
            / "group_free_pred_bboxes_train" / (scene_id + ".npy"),
            "{} train predicted boxes".format(scene_id),
        )


def _validate_cache_shard_names(shards):
    if not isinstance(shards, list) or not shards:
        raise ValueError("train cache shards must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in shards):
        raise ValueError("train cache shard names must be non-empty strings")
    if len(set(shards)) != len(shards):
        raise ValueError("train cache shard names must be unique")
    for index, shard_name in enumerate(shards):
        expected = "shard_{:06d}.pt".format(index)
        if not isinstance(shard_name, str) or shard_name != expected:
            raise ValueError(
                "train cache shard names must be safe and contiguous"
            )
    return list(shards)


def _load_train_cache_binding(cache_dir):
    root = Path(_expected_absolute_path(os.path.expanduser(str(cache_dir))))
    manifest_binding, manifest_bytes = _read_stable_regular_file(
        root / "manifest.json", "train cache manifest", capture=True
    )
    try:
        physical_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("train cache manifest bytes are invalid JSON") from error
    if not isinstance(physical_manifest, dict):
        raise ValueError("train cache manifest bytes must contain a mapping")
    shard_names = _validate_cache_shard_names(
        physical_manifest.get("shards")
    )
    shard_bindings = {}
    for shard_name in shard_names:
        shard_bindings[shard_name] = _stable_file_binding(
            root / shard_name, "train cache shard {}".format(shard_name)
        )
    return physical_manifest, {
        "root": str(root),
        "logical_manifest_sha256": _manifest_sha256(physical_manifest),
        "manifest": manifest_binding,
        "shards": shard_bindings,
    }


def _build_train_cache_binding(cache_dir, loaded_manifest):
    physical_manifest, binding = _load_train_cache_binding(cache_dir)
    if (not isinstance(loaded_manifest, dict)
            or _manifest_sha256(physical_manifest)
            != _manifest_sha256(loaded_manifest)):
        raise ValueError(
            "loaded train cache manifest does not match manifested bytes"
        )
    return binding


def _verify_train_cache_binding(binding):
    if not isinstance(binding, dict):
        raise ValueError("train_cache binding must be a mapping")
    root_value = binding.get("root")
    if not isinstance(root_value, str) or not os.path.isabs(root_value):
        raise ValueError("train cache root is invalid")
    root = Path(_expected_absolute_path(root_value))
    if str(root) != root_value:
        raise ValueError("train cache root must be canonical")
    manifest_binding = binding.get("manifest")
    _verify_expected_binding(
        manifest_binding, root / "manifest.json", "train cache manifest"
    )
    _manifest_actual, manifest_bytes = _read_stable_regular_file(
        root / "manifest.json", "train cache manifest", capture=True
    )
    try:
        physical_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("train cache manifest bytes are invalid JSON") from error
    if not isinstance(physical_manifest, dict):
        raise ValueError("train cache manifest bytes must contain a mapping")
    logical_digest = binding.get("logical_manifest_sha256")
    if (not isinstance(logical_digest, str)
            or SHA256_RE.match(logical_digest) is None
            or logical_digest != _manifest_sha256(physical_manifest)):
        raise RuntimeError(
            "train cache logical manifest does not match manifested bytes"
        )
    shard_names = _validate_cache_shard_names(
        physical_manifest.get("shards")
    )
    shards = binding.get("shards")
    if not isinstance(shards, dict) or list(shards) != shard_names:
        raise ValueError("train cache shard bindings do not match manifest")
    for shard_name in shard_names:
        _verify_expected_binding(
            shards[shard_name], root / shard_name,
            "train cache shard {}".format(shard_name),
        )


def _path_within(path, parent):
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def _collect_source_files(root, excluded_root=None):
    root = Path(_expected_absolute_path(root))
    excluded = None if excluded_root is None else Path(
        _expected_absolute_path(excluded_root)
    )
    candidates = []
    for current, directory_names, file_names in os.walk(str(root)):
        current_path = Path(current)
        kept_directories = []
        for directory_name in sorted(directory_names):
            directory = current_path / directory_name
            if directory.is_symlink():
                raise ValueError(
                    "source snapshot rejects symlink: {}".format(directory)
                )
            if directory_name == "__pycache__":
                continue
            if excluded is not None and _path_within(directory, excluded):
                continue
            kept_directories.append(directory_name)
        directory_names[:] = kept_directories
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.is_symlink():
                raise ValueError(
                    "source snapshot rejects symlink: {}".format(path)
                )
            if path.suffix in (".py", ".so"):
                candidates.append(path)
    return root, sorted(candidates, key=lambda path: path.relative_to(root).as_posix())


def _copy_stable_source(source, destination, label):
    source_before = _stable_file_binding(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError("source snapshot destination already exists")
    shutil.copyfile(str(source), str(destination))
    os.chmod(str(destination), source_before["mode"])
    source_after = _stable_file_binding(source, label)
    if source_before != source_after:
        raise RuntimeError("{} changed while copying".format(label))
    copied = _stable_file_binding(destination, label + " snapshot")
    for key in ("sha256", "size", "mode"):
        if copied[key] != source_before[key]:
            raise RuntimeError("{} snapshot does not match source".format(label))
    return source_before, copied


def _build_source_snapshot(staging, root=ROOT):
    staging = Path(_expected_absolute_path(staging))
    source_root, source_files = _collect_source_files(
        root, excluded_root=staging
    )
    relative_paths = [
        path.relative_to(source_root).as_posix() for path in source_files
    ]
    required = (
        "scripts/audit_scanrefer_joint_box_mask.py", "train_dist_mod.py",
    )
    missing = [name for name in required if name not in relative_paths]
    if missing:
        raise ValueError("runtime code snapshot is missing {}".format(
            ", ".join(missing)
        ))
    snapshot_root = staging / "source_snapshot"
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise ValueError("source snapshot destination already exists")
    snapshot_root.mkdir(parents=True)
    files = {}
    source_bindings = {}
    for source, relative_path in zip(source_files, relative_paths):
        source_binding, copied_binding = _copy_stable_source(
            source, snapshot_root / relative_path,
            "runtime source {}".format(relative_path),
        )
        source_bindings[relative_path] = source_binding
        files[relative_path] = _portable_file_binding(
            copied_binding, "source_snapshot/" + relative_path
        )
    return ({
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "root": "source_snapshot",
        "aggregate_sha256": _canonical_mapping_sha256(files),
        "files": files,
    }, source_bindings)


def _snapshot_actual_files(snapshot_root):
    actual = []
    for current, directory_names, file_names in os.walk(str(snapshot_root)):
        current_path = Path(current)
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink() or directory_name == "__pycache__":
                raise ValueError("source snapshot contains an invalid directory")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink():
                raise ValueError("source snapshot contains a symlink")
            actual.append(path.relative_to(snapshot_root).as_posix())
    return sorted(actual)


def _verify_code_snapshot(staging, snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError("code_snapshot must be a mapping")
    if (snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA
            or snapshot.get("root") != "source_snapshot"):
        raise ValueError("code snapshot schema is invalid")
    files = snapshot.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("code snapshot files must be a non-empty mapping")
    if list(files) != sorted(files):
        raise ValueError("code snapshot paths must be sorted")
    for required in (
            "scripts/audit_scanrefer_joint_box_mask.py", "train_dist_mod.py"):
        if required not in files:
            raise ValueError("code snapshot is missing {}".format(required))
    aggregate = snapshot.get("aggregate_sha256")
    if (not isinstance(aggregate, str) or SHA256_RE.match(aggregate) is None
            or aggregate != _canonical_mapping_sha256(files)):
        raise RuntimeError("code snapshot aggregate SHA-256 is invalid")
    staging = Path(_expected_absolute_path(staging))
    snapshot_root = staging / "source_snapshot"
    expected_paths = []
    for relative_path, binding in files.items():
        relative = _safe_relative_path(relative_path, "code snapshot")
        if relative.suffix not in (".py", ".so") \
                or "__pycache__" in relative.parts:
            raise ValueError("code snapshot path is invalid")
        manifested_path = "source_snapshot/" + relative_path
        _validate_file_binding(binding, "code snapshot {}".format(relative_path))
        if binding["path"] != manifested_path:
            raise ValueError("code snapshot binding path is invalid")
        _verify_file_binding(
            binding, "code snapshot {}".format(relative_path),
            actual_path=snapshot_root / relative,
        )
        expected_paths.append(relative_path)
    if _snapshot_actual_files(snapshot_root) != sorted(expected_paths):
        raise RuntimeError("code snapshot file set does not match manifest")


def _verify_source_bindings(source_bindings, root=ROOT):
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("runtime source bindings must be a non-empty mapping")
    source_root = Path(_expected_absolute_path(root))
    for relative_path in sorted(source_bindings):
        relative = _safe_relative_path(relative_path, "runtime source")
        _verify_expected_binding(
            source_bindings[relative_path], source_root / relative,
            "runtime source {}".format(relative_path),
        )


def _build_output_bindings(staging, names=OUTPUT_NAMES):
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("output names must be a non-empty sequence")
    if len(set(names)) != len(names):
        raise ValueError("output names must be unique")
    staging = Path(_expected_absolute_path(staging))
    outputs = {}
    for name in sorted(names):
        relative = _safe_relative_path(name, "output")
        if len(relative.parts) != 1:
            raise ValueError("audit outputs must be direct staging files")
        binding = _stable_file_binding(staging / relative, "output {}".format(name))
        outputs[name] = _portable_file_binding(binding, name)
    return outputs


def _verify_output_bindings(staging, outputs):
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_NAMES):
        raise ValueError("output bindings are incomplete")
    staging = Path(_expected_absolute_path(staging))
    for name in OUTPUT_NAMES:
        binding = outputs[name]
        _validate_file_binding(binding, "output {}".format(name))
        if binding["path"] != name:
            raise ValueError("output binding path is invalid")
        _verify_file_binding(
            binding, "output {}".format(name), actual_path=staging / name
        )


def _load_staged_json(path, label):
    _binding, content = _read_stable_regular_file(path, label, capture=True)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("{} is invalid JSON".format(label)) from error
    if not isinstance(value, dict):
        raise ValueError("{} must contain a mapping".format(label))
    return value


def _load_staged_torch(path, label):
    _binding, content = _read_stable_regular_file(path, label, capture=True)
    try:
        return torch.load(io.BytesIO(content), map_location="cpu")
    except Exception as error:
        raise ValueError("{} is not a valid torch payload".format(label)) \
            from error


def _reject_nonfinite_scalars(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("{} contains a non-finite scalar".format(label))
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_scalars(
                child, "{}.{}".format(label, key)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_scalars(
                child, "{}[{}]".format(label, index)
            )


def _reject_nonfinite_rows_values(value, label):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("{} tensor must be on CPU".format(label))
        if (torch.is_floating_point(value) or value.is_complex()) \
                and not bool(torch.isfinite(value).all().item()):
            raise ValueError("{} contains a non-finite tensor".format(label))
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("{} contains a non-finite scalar".format(label))
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_rows_values(
                child, "{}.{}".format(label, key)
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite_rows_values(
                child, "{}[{}]".format(label, index)
            )


def _row_identity(row, label):
    if not isinstance(row, dict):
        raise ValueError("{} must be a mapping".format(label))
    dataset_index = row.get("dataset_index")
    if (not isinstance(dataset_index, int) or isinstance(dataset_index, bool)
            or dataset_index < 0):
        raise ValueError("{} dataset index is invalid".format(label))
    scan_id = row.get("scan_id")
    if not isinstance(scan_id, str) or SCENE_ID_RE.match(scan_id) is None:
        raise ValueError("{} scene ID is invalid".format(label))
    target_id = row.get("target_id")
    if (not isinstance(target_id, int) or isinstance(target_id, bool)
            or target_id < 0):
        raise ValueError("{} target ID is invalid".format(label))
    return dataset_index, scan_id, target_id


def _identity_set(rows, label):
    identities = []
    for index, row in enumerate(rows):
        identities.append(_row_identity(
            row, "{} row {}".format(label, index)
        ))
    if len(set(identities)) != len(identities):
        raise ValueError("{} row identities must be unique".format(label))
    return set(identities)


def _finite_number(value, label):
    finite = (
        isinstance(value, int) and not isinstance(value, bool)
    ) or (
        isinstance(value, float) and math.isfinite(value)
    )
    if not finite:
        raise ValueError("{} must be a finite number".format(label))


def _nonnegative_integer(value, label):
    if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError("{} must be a non-negative integer".format(label))


def _validate_result_row(row, index):
    label = "rows.pt row {}".format(index)
    _row_identity(row, label)
    if set(row) != ROW_KEYS:
        raise ValueError("{} keys are invalid".format(label))
    query_indices = row["query_indices"]
    candidate_valid = row["candidate_valid"]
    geometry_valid = row["geometry_valid"]
    geometry_ious = row["geometry_ious"]
    candidate_mask_ious = row["candidate_mask_ious"]
    tensor_fields = (
        (query_indices, "query_indices"),
        (candidate_valid, "candidate_valid"),
        (geometry_valid, "geometry_valid"),
        (geometry_ious, "geometry_ious"),
        (candidate_mask_ious, "candidate_mask_ious"),
    )
    for value, name in tensor_fields:
        if not isinstance(value, torch.Tensor):
            raise ValueError("{}.{} must be a tensor".format(label, name))
        if value.device.type != "cpu":
            raise ValueError("{}.{} must be on CPU".format(label, name))
    if (query_indices.dim() != 1 or query_indices.numel() == 0
            or torch.is_floating_point(query_indices)
            or query_indices.dtype == torch.bool
            or bool((query_indices < 0).any().item())):
        raise ValueError("{}.query_indices is invalid".format(label))
    candidate_count = query_indices.shape[0]
    if (candidate_valid.dtype != torch.bool
            or candidate_valid.shape != query_indices.shape):
        raise ValueError("{}.candidate_valid is invalid".format(label))
    if (geometry_valid.dtype != torch.bool
            or geometry_valid.dim() not in (1, 2)
            or geometry_valid.shape[0] != candidate_count
            or any(size <= 0 for size in geometry_valid.shape)):
        raise ValueError("{}.geometry_valid is invalid".format(label))
    if (not torch.is_floating_point(geometry_ious)
            or geometry_ious.dim() != 2
            or geometry_ious.shape[0] != candidate_count
            or (geometry_valid.dim() == 2
                and geometry_ious.shape != geometry_valid.shape)):
        raise ValueError("{}.geometry_ious is invalid".format(label))
    if (not torch.is_floating_point(candidate_mask_ious)
            or candidate_mask_ious.dim() != 3
            or candidate_mask_ious.shape[0] != candidate_count
            or any(size <= 0 for size in candidate_mask_ious.shape)):
        raise ValueError("{}.candidate_mask_ious is invalid".format(label))
    for name in (
            "legacy_semantic_query", "baseline_flat_index",
            "joint_oracle_flat_index"):
        _nonnegative_integer(row[name], "{}.{}".format(label, name))
    for name in (
            "legacy_mask_iou", "geometry_parent_mask_iou",
            "baseline_box_iou", "joint_oracle_mask_iou",
            "selected_box_iou", "selected_mask_iou"):
        _finite_number(row[name], "{}.{}".format(label, name))
    if type(row["selected_uses_joint_query"]) is not bool:
        raise ValueError(
            "{}.selected_uses_joint_query must be boolean".format(label)
        )


def _verify_rows_payload(payload, selection, summary):
    if not isinstance(payload, dict):
        raise ValueError("rows.pt torch payload must contain a mapping")
    forbidden = set(payload).intersection(INFERENCE_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError("rows.pt contains forbidden top-level keys")
    if set(payload) != ROWS_TOP_LEVEL_KEYS:
        raise ValueError("rows.pt top-level keys are invalid")
    if (payload.get("schema") != AUDIT_SCHEMA_VERSION
            or payload.get("split") != "train"
            or payload.get("validation_data_accessed") is not False):
        raise ValueError("rows.pt receipt fields are invalid")
    thresholds = payload.get("logit_thresholds")
    if (not isinstance(thresholds, (list, tuple)) or not thresholds
            or any(
                not (
                    (isinstance(value, int) and not isinstance(value, bool))
                    or (isinstance(value, float) and math.isfinite(value))
                ) for value in thresholds
            )):
        raise ValueError("rows.pt logit thresholds are invalid")
    source_names = payload.get("source_names")
    if (not isinstance(source_names, (list, tuple)) or not source_names
            or any(not isinstance(name, str) or not name
                   for name in source_names)
            or len(set(source_names)) != len(source_names)):
        raise ValueError("rows.pt source names are invalid")
    summary_thresholds = summary.get("logit_thresholds")
    if summary_thresholds is not None:
        if (not isinstance(summary_thresholds, (list, tuple))
                or list(summary_thresholds) != list(thresholds)):
            raise ValueError("rows.pt thresholds do not match summary")
    summary_source_names = summary.get("source_names")
    if summary_source_names is not None:
        if (not isinstance(summary_source_names, (list, tuple))
                or list(summary_source_names) != list(source_names)):
            raise ValueError("rows.pt source names do not match summary")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows.pt rows are invalid")
    _reject_nonfinite_rows_values(payload, "rows.pt")
    for index, row in enumerate(rows):
        _validate_result_row(row, index)

    summary_count = summary.get("sample_count")
    if (not isinstance(summary_count, int) or isinstance(summary_count, bool)
            or summary_count <= 0 or summary_count != len(rows)):
        raise ValueError("summary sample_count does not match rows.pt")
    selection_rows = selection.get("rows")
    if not isinstance(selection_rows, list) or not selection_rows:
        raise ValueError("selection rows are invalid")
    selection_count = selection.get("sample_count")
    if selection_count is not None and (
            not isinstance(selection_count, int)
            or isinstance(selection_count, bool)
            or selection_count != len(selection_rows)):
        raise ValueError("selection sample_count is invalid")
    if len(selection_rows) != summary_count:
        raise ValueError("selection row count does not match summary")
    if _identity_set(rows, "rows.pt") != _identity_set(
            selection_rows, "selection"):
        raise ValueError("rows.pt identities do not match selection")


def _validate_receipt_header(receipt):
    if receipt.get("schema") != AUDIT_SCHEMA_VERSION:
        raise ValueError("receipt schema is invalid")
    if receipt.get("split") != "train":
        raise ValueError("receipt split must be train")
    if receipt.get("validation_data_accessed") is not False:
        raise ValueError("receipt must declare no validation data access")
    if receipt.get("population_estimate") is not False:
        raise ValueError("receipt population_estimate must be false")
    elapsed = receipt.get("elapsed_seconds")
    if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed < 0.0):
        raise ValueError("receipt elapsed_seconds is invalid")


def _verify_protected_bindings(protected_before, protected_after):
    expected_names = {"checkpoint", "parent", "geometry"}
    if (not isinstance(protected_before, dict)
            or set(protected_before) != expected_names
            or not isinstance(protected_after, dict)
            or protected_before != protected_after):
        raise ValueError("protected artifact bindings are invalid")
    for name in sorted(expected_names):
        _verify_file_binding(
            protected_before[name], name, require_identity=True
        )


def _verify_staged_manifest(staging):
    """Verify a complete staged receipt and every byte binding it names."""
    staging = Path(_expected_absolute_path(staging))
    receipt = _load_staged_json(staging / "manifest.json", "audit manifest")
    _reject_nonfinite_scalars(receipt, "manifest")
    _validate_receipt_header(receipt)
    selected = receipt.get("selected_train_scene_ids")
    normalized = _validated_scene_ids(selected)
    if selected != normalized:
        raise ValueError("selected train scene IDs must be sorted and unique")
    replay_input = receipt.get("replay_input_scene_ids")
    if not isinstance(replay_input, (list, tuple)) or not replay_input:
        raise ValueError("replay input scene IDs are invalid")
    replay_normalized = _validated_scene_ids(replay_input)
    if replay_input != replay_normalized:
        raise ValueError("replay input scene IDs must be sorted and unique")
    if not set(selected).issubset(replay_input):
        raise ValueError("replay input scene IDs must include selected scenes")
    _verify_dataset_bindings(
        receipt.get("dataset_inputs"), selected, replay_input
    )
    _verify_train_cache_binding(receipt.get("train_cache"))
    logical_digest = receipt["train_cache"]["logical_manifest_sha256"]
    if receipt.get("train_cache_manifest_sha256") != logical_digest:
        raise ValueError("train cache logical manifest digest is inconsistent")
    _verify_protected_bindings(
        receipt.get("protected_before"), receipt.get("protected_after")
    )
    outputs = receipt.get("outputs")
    _verify_output_bindings(staging, outputs)
    expected_hashes = {
        name: binding["sha256"] for name, binding in outputs.items()
    }
    if receipt.get("outputs_sha256") != expected_hashes:
        raise ValueError("legacy output hashes do not match output bindings")
    _verify_code_snapshot(staging, receipt.get("code_snapshot"))
    source_hashes = receipt.get("source_sha256")
    expected_source_names = {
        "models/rec_joint_box_mask.py",
        "scripts/audit_scanrefer_joint_box_mask.py",
    }
    if not isinstance(source_hashes, dict) \
            or set(source_hashes) != expected_source_names:
        raise ValueError("legacy source hashes are incomplete")
    snapshot_files = receipt["code_snapshot"]["files"]
    for name in expected_source_names:
        if (name not in snapshot_files
                or source_hashes[name] != snapshot_files[name]["sha256"]):
            raise ValueError("legacy source hashes do not match code snapshot")
    summary = _load_staged_json(staging / "summary.json", "audit summary")
    selection = _load_staged_json(
        staging / "selection.json", "audit selection"
    )
    _reject_nonfinite_scalars(summary, "summary")
    _reject_nonfinite_scalars(selection, "selection")
    for payload, label in ((summary, "summary"), (selection, "selection")):
        if (payload.get("schema") != AUDIT_SCHEMA_VERSION
                or payload.get("split") != "train"
                or payload.get("validation_data_accessed") is not False
                or payload.get("population_estimate") is not False):
            raise ValueError("audit {} receipt fields are invalid".format(label))
    if summary.get("elapsed_seconds") != receipt["elapsed_seconds"]:
        raise ValueError("manifest elapsed_seconds does not match summary")
    if summary.get("stage0_gate") != receipt.get("stage0_gate"):
        raise ValueError("manifest stage0 gate does not match summary")
    rows = selection.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection rows are invalid")
    row_scene_values = []
    for row in rows:
        scene_id = row.get("scan_id") if isinstance(row, dict) else None
        if (not isinstance(scene_id, str)
                or SCENE_ID_RE.match(scene_id) is None):
            raise ValueError("selection scene ID is invalid")
        row_scene_values.append(scene_id)
    row_scene_ids = sorted(set(row_scene_values))
    if row_scene_ids != selected:
        raise ValueError("manifest selected scenes do not match selection")
    if selection.get("replay_input_scene_ids") != replay_input:
        raise ValueError("selection replay input scenes are inconsistent")
    if selection.get("train_cache_manifest_sha256") != logical_digest:
        raise ValueError("selection train cache digest is inconsistent")
    rows_payload = _load_staged_torch(
        staging / "rows.pt", "audit rows.pt"
    )
    _verify_rows_payload(rows_payload, selection, summary)
    return receipt


def _artifact_snapshot(path, expected_sha256, label):
    """Snapshot only the exact immutable artifact named by the contract."""
    contract = PROTECTED_ARTIFACT_CONTRACT.get(label)
    if not isinstance(contract, Mapping):
        raise ValueError("{} is not in the protected artifact contract".format(
            label
        ))
    contract_path = contract.get("path")
    contract_size = contract.get("size")
    contract_mode = contract.get("mode")
    contract_sha256 = contract.get("sha256")
    if (not isinstance(contract_path, str)
            or not os.path.isabs(contract_path)
            or not isinstance(contract_size, int)
            or isinstance(contract_size, bool) or contract_size < 0
            or not isinstance(contract_mode, int)
            or isinstance(contract_mode, bool)
            or not isinstance(contract_sha256, str)
            or SHA256_RE.match(contract_sha256) is None):
        raise ValueError("{} protected artifact contract is invalid".format(
            label
        ))
    canonical_path = _lexical_absolute_path(contract_path)
    requested_path = _lexical_absolute_path(path)
    if (requested_path != canonical_path
            or _expected_absolute_path(requested_path) != canonical_path):
        raise ValueError(
            "{} path is not the canonical protected artifact".format(label)
        )
    if expected_sha256 != contract_sha256:
        raise ValueError("{} SHA-256 contract is inconsistent".format(label))
    snapshot = _stable_file_binding(canonical_path, label)
    if snapshot["sha256"] != contract_sha256:
        raise ValueError("{} SHA-256 is not the protected artifact".format(label))
    if snapshot["size"] != contract_size:
        raise ValueError("{} size is not the protected artifact size".format(
            label
        ))
    if snapshot["mode"] != 0o444 or snapshot["mode"] != contract_mode:
        raise ValueError("{} mode must be exactly 0444".format(label))
    return snapshot


def validate_manifest(manifest, checkpoint_sha256):
    """Reject non-train, incomplete, or differently bound candidate caches."""
    if not isinstance(manifest, dict):
        raise ValueError("cache manifest must be a mapping")
    if manifest.get("split") != "train":
        raise ValueError("joint audit requires a train cache")
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("train cache checkpoint fingerprint does not match")
    counts = [manifest.get(name) for name in (
        "sample_count", "dataset_size", "source_dataset_size"
    )]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in counts) or len(set(counts)) != 1:
        raise ValueError("train cache must cover the complete source dataset")
    cache_schema_version = manifest.get("cache_schema_version")
    if (not isinstance(cache_schema_version, int)
            or isinstance(cache_schema_version, bool)
            or cache_schema_version != 1
            or manifest.get("feature_schema_version") != "rec-query-v1"):
        raise ValueError("train cache schema is unsupported")
    if manifest.get("target_iou_policy") != "root_only":
        raise ValueError("train cache target policy is unsupported")
    if manifest.get("deterministic") is not True:
        raise ValueError("train cache must declare deterministic extraction")
    candidate_rule = manifest.get("candidate_rule")
    if candidate_rule != {"topk_per_source": 8, "max_candidates": 16}:
        raise ValueError("train cache candidate rule is unsupported")
    _validate_cache_shard_names(manifest.get("shards"))
    return manifest


def _aligned_text_map(mapping, token_count, device, dtype):
    mapping = torch.as_tensor(mapping, device=device, dtype=dtype)
    if mapping.dim() == 3:
        mapping = mapping[:, 0]
    if mapping.dim() != 2:
        raise ValueError("language maps must have shape [B,T] or [B,N,T]")
    aligned = torch.zeros(
        mapping.shape[0], token_count, device=device, dtype=dtype
    )
    aligned[:, :min(token_count, mapping.shape[1])] = mapping[
        :, :min(token_count, mapping.shape[1])
    ]
    return aligned


def semantic_mask_query_indices(end_points, language_maps):
    """Reproduce the formal semantic-mask Top-1 query selection."""
    tokens = end_points.get("proj_tokens")
    queries = end_points.get("last_proj_queries")
    if not isinstance(tokens, torch.Tensor) or not isinstance(queries, torch.Tensor):
        raise ValueError("semantic query selection requires projected tensors")
    if tokens.dim() != 3 or queries.dim() != 3 or tokens.shape[0] != queries.shape[0] \
            or tokens.shape[-1] != queries.shape[-1]:
        raise ValueError("projected text and query tensors are malformed")
    semantic = torch.matmul(queries, tokens.transpose(-1, -2))
    semantic = (semantic / 0.07).softmax(-1)
    token_count = semantic.shape[-1]
    maps = {}
    for key in (
            "positive_map", "modify_positive_map", "pron_positive_map",
            "other_entity_map", "rel_positive_map"):
        if key not in language_maps:
            raise ValueError("language maps are missing {}".format(key))
        maps[key] = _aligned_text_map(
            language_maps[key], token_count, semantic.device, semantic.dtype
        )
    maps["positive_map"] = (maps["positive_map"] > 0).to(semantic.dtype)

    def score(key):
        return (semantic * maps[key].unsqueeze(1)).sum(-1)

    scores = (
        score("positive_map") + score("modify_positive_map")
        + score("pron_positive_map") + score("rel_positive_map")
        - score("other_entity_map")
    )
    return scores.argmax(dim=1)


def _assert_inference_payload(payload, label):
    if not isinstance(payload, dict):
        raise ValueError("{} must be a mapping".format(label))
    forbidden = sorted(set(payload).intersection(INFERENCE_FORBIDDEN_KEYS))
    if forbidden:
        raise ValueError("{} contains target-only fields: {}".format(
            label, ", ".join(forbidden)
        ))


def _root_box_ious(flat_boxes, valid_mask, batch_data):
    from models.rec_reranker import compute_query_ious

    gt_boxes = torch.cat((
        batch_data["center_label"][:, :1, :3].float(),
        batch_data["size_gts"][:, :1].float(),
    ), dim=-1)
    gt_valid = batch_data["box_label_mask"][:, :1]
    ious = compute_query_ious(flat_boxes, gt_boxes, gt_valid)
    return ious.masked_fill(~valid_mask, 0.0)


def _one_mask_target(end_points, batch_data, batch_index, query_indices,
                     thresholds):
    from models.rec_mask_geometry import normalize_mcln_mask_logits

    text, query, _fused, alpha = normalize_mcln_mask_logits(
        end_points, batch_index, query_indices
    )
    point_target = batch_data["gt_masks"][batch_index, 0].bool()
    superpoint_ids = batch_data["superpoint"][batch_index]
    compressed = compress_point_mask_to_superpoints(
        point_target, superpoint_ids, num_superpoints=text.shape[-1]
    )
    return compute_weighted_mask_candidate_targets(
        text.unsqueeze(0), query.unsqueeze(0), alpha,
        compressed["point_counts"].unsqueeze(0),
        compressed["target_counts"].unsqueeze(0),
        torch.ones(1, text.shape[0], dtype=torch.bool, device=text.device),
        torch.as_tensor(thresholds, device=text.device, dtype=text.dtype),
    )


def _threshold_index(thresholds, value):
    values = torch.as_tensor(thresholds, dtype=torch.float64)
    matches = torch.isclose(values, values.new_tensor(value), atol=0.0, rtol=0.0)
    indices = matches.nonzero(as_tuple=False).reshape(-1)
    if indices.numel() != 1:
        raise ValueError("threshold {} is absent or duplicated".format(value))
    return int(indices.item())


def _cpu_clone(value, dtype=None):
    tensor = torch.as_tensor(value).detach().cpu()
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.contiguous().clone()


def _build_joint_rows(selected_indices, selected_end_points, selected_inputs,
                      selected_batch_data, parent_outputs, runtime_outputs,
                      thresholds):
    candidates = parent_outputs["candidate_batch"]
    query_indices = candidates["query_indices"]
    candidate_valid = candidates["valid_mask"].bool()
    batch_size, candidate_count = query_indices.shape
    geometry_valid = runtime_outputs["rec_geometry_valid_mask"].reshape(
        batch_size, candidate_count, -1
    )
    variant_count = geometry_valid.shape[-1]
    geometry_boxes = runtime_outputs["rec_geometry_boxes"]
    geometry_ious = _root_box_ious(
        geometry_boxes, runtime_outputs["rec_geometry_valid_mask"],
        selected_batch_data,
    ).reshape(batch_size, candidate_count, variant_count)
    baseline_flat = runtime_outputs["rec_geometry_scores"].argmax(1)
    semantic_queries = semantic_mask_query_indices(
        selected_end_points, selected_inputs
    )
    zero_index = _threshold_index(thresholds, 0.0)
    rows = []
    for row_index, dataset_index in enumerate(selected_indices):
        candidate_targets = _one_mask_target(
            selected_end_points, selected_batch_data, row_index,
            query_indices[row_index], thresholds,
        )
        candidate_mask_ious = candidate_targets["ious"][0]
        query_oracle_mask = candidate_mask_ious.amax(dim=(1, 2))
        expanded_mask = query_oracle_mask.unsqueeze(-1).expand(
            candidate_count, variant_count
        ).unsqueeze(0)
        oracle = select_joint_oracle(
            geometry_ious[row_index:row_index + 1],
            expanded_mask,
            baseline_flat[row_index:row_index + 1],
            valid_mask=geometry_valid[row_index:row_index + 1],
        )
        semantic_target = _one_mask_target(
            selected_end_points, selected_batch_data, row_index,
            semantic_queries[row_index:row_index + 1], thresholds,
        )
        legacy_mask_iou = semantic_target["ious"][0, 0, 2, zero_index]
        baseline_parent_position = int(
            baseline_flat[row_index].item() // variant_count
        )
        geometry_parent_mask_iou = candidate_mask_ious[
            baseline_parent_position, 2, zero_index
        ]
        use_joint = bool(
            oracle["selected_mask_iou"].item() > legacy_mask_iou.item()
        )
        baseline_box_iou = oracle["baseline_box_iou"][0]
        selected_box_iou = (
            oracle["selected_box_iou"][0] if use_joint else baseline_box_iou
        )
        selected_mask_iou = (
            oracle["selected_mask_iou"][0] if use_joint else legacy_mask_iou
        )
        row = {
            "dataset_index": int(dataset_index),
            "scan_id": str(selected_batch_data["scan_ids"][row_index]),
            "target_id": int(selected_batch_data["target_id"][row_index].item()),
            "query_indices": _cpu_clone(query_indices[row_index], torch.long),
            "candidate_valid": _cpu_clone(candidate_valid[row_index], torch.bool),
            "geometry_valid": _cpu_clone(geometry_valid[row_index], torch.bool),
            "geometry_ious": _cpu_clone(geometry_ious[row_index], torch.float32),
            "candidate_mask_ious": _cpu_clone(candidate_mask_ious, torch.float32),
            "legacy_semantic_query": int(semantic_queries[row_index].item()),
            "legacy_mask_iou": float(legacy_mask_iou.item()),
            "geometry_parent_mask_iou": float(geometry_parent_mask_iou.item()),
            "baseline_flat_index": int(baseline_flat[row_index].item()),
            "baseline_box_iou": float(baseline_box_iou.item()),
            "joint_oracle_flat_index": int(oracle["selected_flat_index"].item()),
            "joint_oracle_mask_iou": float(oracle["selected_mask_iou"].item()),
            "selected_box_iou": float(selected_box_iou.item()),
            "selected_mask_iou": float(selected_mask_iou.item()),
            "selected_uses_joint_query": use_joint,
        }
        rows.append(row)
    return rows


def _summarize_rows(rows, thresholds):
    if not rows:
        raise ValueError("joint audit produced no rows")
    selection = {
        key: torch.tensor([row[key] for row in rows], dtype=torch.float64)
        for key in (
            "baseline_box_iou", "selected_box_iou", "legacy_mask_iou",
            "selected_mask_iou")
    }
    strict_selection = {
        "baseline_box_iou": selection["baseline_box_iou"],
        "selected_box_iou": selection["selected_box_iou"],
        "baseline_mask_iou": selection["legacy_mask_iou"],
        "selected_mask_iou": selection["selected_mask_iou"],
    }
    summary = summarize_joint_oracle(strict_selection)
    geometry_parent = torch.tensor([
        row["geometry_parent_mask_iou"] for row in rows
    ], dtype=torch.float64)
    joint_oracle = torch.tensor([
        row["joint_oracle_mask_iou"] for row in rows
    ], dtype=torch.float64)

    def metrics(values):
        return {
            "hits025": int((values > 0.25).sum().item()),
            "hits050": int((values > 0.50).sum().item()),
            "acc025": float((values > 0.25).double().mean().item()),
            "acc050": float((values > 0.50).double().mean().item()),
            "miou": float(values.mean().item()),
        }

    summary["legacy_semantic_mask"] = metrics(selection["legacy_mask_iou"])
    summary["geometry_parent_fused_t0_mask"] = metrics(geometry_parent)
    summary["joint_query_oracle_mask"] = metrics(joint_oracle)
    summary["fallback_aware_selected_mask"] = metrics(
        selection["selected_mask_iou"]
    )
    summary["joint_switch_count"] = sum(
        bool(row["selected_uses_joint_query"]) for row in rows
    )
    summary["logit_thresholds"] = list(thresholds)
    summary["source_names"] = list(MASK_SOURCE_NAMES)
    summary["stage0_gate"] = stage0_gate(summary)
    return summary


def _prepare_staging(output_dir):
    final_path = Path(output_dir).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        raise ValueError("audit output already exists: {}".format(final_path))
    staging = Path(tempfile.mkdtemp(
        prefix=final_path.name + ".staging.", dir=str(final_path.parent)
    ))
    return final_path, staging


def _publish_staging_no_clobber(staging, final_path):
    """Atomically publish a directory while refusing every existing target."""
    staging = Path(staging)
    final_path = Path(final_path)
    if not staging.is_dir():
        raise ValueError("audit staging directory does not exist")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-clobber directory publish is unsupported")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(str(staging)),
        -100,
        os.fsencode(str(final_path)),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number, os.strerror(error_number), str(final_path)
        )
    raise OSError(error_number, os.strerror(error_number), str(final_path))


def _build_parent_geometry_targets(
        selected_end_points, selected_inputs, selected_batch,
        parent_model, parent_artifact, geometry_model, geometry_artifact):
    """Keep target-only tensors beyond the complete inference boundary."""
    from models.rec_candidate_adapter import attach_candidate_targets
    from train_dist_mod import (
        build_rec_geometry_runtime_outputs,
        build_rec_reranker_outputs,
    )

    parent = build_rec_reranker_outputs(
        selected_end_points, selected_inputs, parent_model, parent_artifact
    )
    _assert_inference_payload(parent["candidate_batch"], "parent candidates")
    runtime = build_rec_geometry_runtime_outputs(
        selected_end_points, selected_inputs, parent,
        geometry_model, geometry_artifact,
    )
    _assert_inference_payload(runtime, "geometry runtime")
    targeted_candidates = attach_candidate_targets(
        parent["candidate_batch"], selected_batch, root_only=True
    )
    return parent, runtime, targeted_candidates


def _replay_scene_ids_from_records(records, replay_groups):
    """Resolve every scene forwarded by the contiguous replay batches."""
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("cache records must be a non-empty list")
    records_by_index = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("cache record must be a mapping")
        dataset_index = record.get("dataset_index")
        if (not isinstance(dataset_index, int)
                or isinstance(dataset_index, bool) or dataset_index < 0
                or dataset_index in records_by_index):
            raise ValueError("cache record dataset index is invalid")
        scan_id = record.get("scan_id")
        if not isinstance(scan_id, str) or SCENE_ID_RE.match(scan_id) is None:
            raise ValueError("cache record scene ID is invalid")
        records_by_index[dataset_index] = scan_id
    replay_indices = []
    for group in replay_groups:
        if not isinstance(group, dict):
            raise ValueError("replay group must be a mapping")
        batch_indices = group.get("batch_indices")
        if (not isinstance(batch_indices, (list, tuple, range))
                or not batch_indices):
            raise ValueError("replay group batch indices are invalid")
        for dataset_index in batch_indices:
            if (not isinstance(dataset_index, int)
                    or isinstance(dataset_index, bool)
                    or dataset_index not in records_by_index):
                raise ValueError("replay batch index is absent from cache")
            replay_indices.append(dataset_index)
    if not replay_indices:
        raise ValueError("replay groups contain no input rows")
    return sorted(set(records_by_index[index] for index in replay_indices))


def _run(args, staging):
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    protected_before = {
        name: _artifact_snapshot(
            getattr(args, {
                "checkpoint": "checkpoint",
                "parent": "parent_checkpoint",
                "geometry": "geometry_checkpoint",
            }[name]),
            PROTECTED_ARTIFACT_CONTRACT[name]["sha256"], name,
        )
        for name in ("checkpoint", "parent", "geometry")
    }
    code_snapshot, source_bindings = _build_source_snapshot(staging)
    checkpoint_sha256 = protected_before["checkpoint"]["sha256"]
    bound_manifest, train_cache_binding = _load_train_cache_binding(
        args.train_cache
    )
    validate_manifest(bound_manifest, checkpoint_sha256)
    manifest, records = load_train_cache_panel_records(
        args.train_cache, checkpoint_sha256
    )
    if (_manifest_sha256(manifest)
            != _manifest_sha256(bound_manifest)):
        raise RuntimeError(
            "loaded train cache manifest changed after byte binding"
        )
    validate_manifest(manifest, checkpoint_sha256)
    panel = select_baseline_stratified_panel(
        records, scene_count=args.scene_count,
        expressions_per_scene=args.expressions_per_scene,
        seed=args.selection_seed,
    )
    selected_scene_ids = sorted(set(
        str(row["scan_id"]) for row in panel
    ))
    _validated_scene_ids(selected_scene_ids)
    selected_indices = [row["dataset_index"] for row in panel]
    replay_groups = build_cache_replay_groups(
        selected_indices, manifest["source_dataset_size"],
        args.cache_extraction_batch_size, args.cache_replay_boundaries,
    )
    replay_input_scene_ids = _replay_scene_ids_from_records(
        records, replay_groups
    )
    dataset_inputs = _build_dataset_bindings(
        args.data_root, selected_scene_ids,
        replay_scene_ids=replay_input_scene_ids,
        project_root=ROOT,
    )
    cache_rows = load_selected_cache_rows(
        args.train_cache, manifest, selected_indices
    )
    selection = {
        "schema": AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "population_estimate": False,
        "selection_seed": args.selection_seed,
        "scene_count": args.scene_count,
        "expressions_per_scene": args.expressions_per_scene,
        "sample_count": len(panel),
        "replay_batch_count": len(replay_groups),
        "bucket_counts": _panel_bucket_counts(panel),
        "selected_train_scene_ids": selected_scene_ids,
        "replay_input_scene_ids": replay_input_scene_ids,
        "rows": panel,
        "train_cache_manifest_sha256": _manifest_sha256(manifest),
        "protected_artifacts": protected_before,
    }
    _atomic_write_json(staging / "selection.json", selection)

    from scripts.cache_scanrefer_rec_candidates import (
        _build_dataset,
        _load_frozen_model,
        _move_batch_to_device,
        _normalized_data_root,
        _prepare_model_config,
    )
    from train_dist_mod import (
        TrainTester,
        load_rec_geometry_runtime_artifacts,
    )

    _set_deterministic(args.selection_seed, device)
    checkpoint = torch.load(
        protected_before["checkpoint"]["path"], map_location="cpu"
    )
    config = _prepare_model_config(
        checkpoint, _normalized_data_root(args.data_root)
    )
    dataset = _build_dataset(config, "train")
    replay_indices = {
        index for group in replay_groups for index in group["batch_indices"]
    }
    replay_scan_ids = {
        str(dataset.annos[index]["scan_id"]) for index in replay_indices
    }
    if sorted(replay_scan_ids) != replay_input_scene_ids:
        raise RuntimeError("runtime replay scene IDs do not match cache records")
    _prune_dataset_scenes(dataset, replay_scan_ids)
    model = _load_frozen_model(checkpoint, config, device)
    del checkpoint
    parent_model, parent_artifact, geometry_model, geometry_artifact = (
        load_rec_geometry_runtime_artifacts(
            protected_before["parent"]["path"],
            protected_before["geometry"]["path"], device
        )
    )
    loader = _build_replay_loader(dataset, replay_groups, args, device)
    rows = []
    parity_maxima = {}
    messages = []
    with torch.inference_mode():
        for batch_number, (group, batch_data) in enumerate(
                zip(replay_groups, loader), start=1):
            batch_size = len(batch_data["scan_ids"])
            batch_data = _move_batch_to_device(batch_data, device)
            inputs = dict(TrainTester._get_inputs(batch_data))
            _assert_inference_payload(inputs, "model inputs")
            inputs["train"] = False
            end_points = model(inputs)
            _assert_inference_payload(end_points, "model outputs")
            positions = group["selected_positions"]
            indices = group["selected_indices"]
            selected_batch = _select_batch_mapping(
                batch_data, positions, batch_size
            )
            selected_inputs = _select_batch_mapping(inputs, positions, batch_size)
            selected_end_points = _select_batch_mapping(
                end_points, positions, batch_size
            )
            parent, runtime, targeted_candidates = (
                _build_parent_geometry_targets(
                    selected_end_points, selected_inputs, selected_batch,
                    parent_model, parent_artifact,
                    geometry_model, geometry_artifact,
                )
            )
            parity = _assert_historical_cache_identity(
                targeted_candidates, cache_rows, indices,
                selected_batch["scan_ids"], selected_batch["target_id"],
            )
            for key, value in parity.items():
                parity_maxima[key] = max(parity_maxima.get(key, 0.0), value)
            rows.extend(_build_joint_rows(
                indices, selected_end_points, selected_inputs,
                selected_batch, parent, runtime, args.logit_thresholds,
            ))
            message = "Audited {}/{} rows (replay batch {}/{})".format(
                len(rows), len(panel), batch_number, len(replay_groups)
            )
            print(message, flush=True)
            messages.append(message)
    if len(rows) != len(panel):
        raise RuntimeError("audit replay ended before all selected rows")
    summary = _summarize_rows(rows, args.logit_thresholds)
    summary.update({
        "schema": AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "population_estimate": False,
        "sample_count": len(rows),
        "scene_count": args.scene_count,
        "expressions_per_scene": args.expressions_per_scene,
        "replay_batch_count": len(replay_groups),
        "historical_cache_drift_diagnostics": parity_maxima,
        "elapsed_seconds": float(time.time() - started),
    })
    protected_after = {
        name: _artifact_snapshot(
            protected_before[name]["path"],
            PROTECTED_ARTIFACT_CONTRACT[name]["sha256"], name
        ) for name in protected_before
    }
    if protected_before != protected_after:
        raise RuntimeError("a protected artifact changed during audit")
    _atomic_torch_save(staging / "rows.pt", {
        "schema": AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "logit_thresholds": tuple(args.logit_thresholds),
        "source_names": MASK_SOURCE_NAMES,
        "rows": rows,
    })
    _atomic_write_json(staging / "summary.json", summary)
    (staging / "stdout.log").write_text("\n".join(messages) + "\n")
    output_bindings = _build_output_bindings(staging)
    output_hashes = {
        name: binding["sha256"]
        for name, binding in output_bindings.items()
    }
    source_hashes = {
        name: code_snapshot["files"][name]["sha256"] for name in (
            "models/rec_joint_box_mask.py",
            "scripts/audit_scanrefer_joint_box_mask.py",
        )
    }
    _atomic_write_json(staging / "manifest.json", {
        "schema": AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "population_estimate": False,
        "elapsed_seconds": summary["elapsed_seconds"],
        "selected_train_scene_ids": selected_scene_ids,
        "replay_input_scene_ids": replay_input_scene_ids,
        "dataset_inputs": dataset_inputs,
        "train_cache": train_cache_binding,
        "train_cache_manifest_sha256": _manifest_sha256(manifest),
        "protected_before": protected_before,
        "protected_after": protected_after,
        "outputs": output_bindings,
        "outputs_sha256": output_hashes,
        "code_snapshot": code_snapshot,
        "source_sha256": source_hashes,
        "stage0_gate": summary["stage0_gate"],
    })
    _verify_source_bindings(source_bindings)
    _verify_staged_manifest(staging)
    return summary


def run_audit(args):
    final_path, staging = _prepare_staging(args.output_dir)
    try:
        summary = _run(args, staging)
        _publish_staging_no_clobber(staging, final_path)
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("Audit artifacts: {}".format(final_path), flush=True)
    return summary


def main(argv=None):
    run_audit(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
