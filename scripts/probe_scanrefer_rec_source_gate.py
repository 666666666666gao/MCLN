#!/usr/bin/env python
"""Run a train-only, nondeployable ScanRefer REC source-gate probe."""

import argparse
import copy
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import platform
import random
import secrets
import shutil
import stat
import struct
import sys
import tempfile
import time

import torch

from models import rec_finetune, rec_source_gate
from models.rec_candidate_adapter import (
    build_full_rec_query_state,
    compact_rec_query_state,
)
from scripts import train_scanrefer_rec_finetune as legacy


PRODUCTION_DEVICE = "cuda:0"
PRODUCTION_PROBE_STEPS = 306
PRODUCTION_SEED = 0
PRODUCTION_BATCH_SIZE = 18
PRODUCTION_FIT_SAMPLE_COUNT = 33040
PRODUCTION_CALIBRATION_SAMPLE_COUNT = 3625
SOURCE_GATE_TRAIN_DATA_SCHEMA = (
    "scanrefer-rec-source-gate-train-data-v1"
)
PROTECTED_LEGACY_PATHS = legacy.PROTECTED_LEGACY_PATHS
RUNTIME_ENVIRONMENT_ALLOWLIST = legacy.RUNTIME_ENVIRONMENT_ALLOWLIST
STATE_DIGEST_FORMAT = "rec-source-gate-state-digest-v1"
CALIBRATION_HIT_BITS_DIGEST_FORMAT = (
    "rec-source-gate-calibration-hit-bits-sha256-v1"
)
CALIBRATION_BINDING_FORMAT = (
    "rec-source-gate-calibration-transition-binding-sha256-v1"
)
CALIBRATION_EVIDENCE_SCHEMA = (
    "rec-source-gate-calibration-evidence-v1"
)
_FILESYSTEM_EINTR_ATTEMPTS = 8

_CALIBRATION_GROUPS = {
    "membership": ("default_top8", "contrastive_top8", "union_top16"),
    "candidate_oracle": (
        "raw_query", "union_query", "parent_candidate",
        "geometry_candidate",
    ),
    "top1": ("default", "parent", "geometry"),
}
_CALIBRATION_METRIC_FIELDS = {
    "hits025", "hits050", "acc025", "acc050",
}
_CALIBRATION_TRANSITION_FIELDS = {
    "gained025", "lost025", "gained050", "lost050",
}
_CALIBRATION_BRANCH_BINDINGS = {
    "membership": {
        "default_top8": "default_top8",
        "contrastive_top8": "contrastive_top8",
        "union_top16": "union_query",
    },
    "candidate_oracle": {
        "raw_query": "raw_query",
        "union_query": "union_query",
        "parent_candidate": "parent_candidate",
        "geometry_candidate": "geometry_candidate",
    },
    "top1": {
        "default": "default_top1",
        "parent": "parent_top1",
        "geometry": "geometry_top1",
    },
}
_CALIBRATION_INTERNAL_BRANCHES = {
    internal
    for branches in _CALIBRATION_BRANCH_BINDINGS.values()
    for internal in branches.values()
}
_REQUIRED_CODE_MANIFEST_ENTRIES = {
    "source_gate_runner", "rec_source_gate", "rec_finetune",
    "rec_reranker", "rec_candidate_adapter", "rec_mask_geometry",
    "rec_geometry_reranker", "source_choice_adapter",
    "source_choice_selector", "train_rec_finetune",
}
_SOURCE_GATE_MANIFEST_PRUNED_DIRECTORIES = frozenset((
    ".pytest_cache", "__pycache__",
))
_SOURCE_GATE_MANIFEST_EXCLUDED_FILES = frozenset((
    "pointnet2/pointnet2_test.py",
    "scripts/cache_scanrefer_rec_mask_geometry.py",
))
_CLI_OPTIONS = {
    "--data-root", "--backbone-checkpoint", "--parent-reranker",
    "--geometry-reranker", "--output-dir", "--device", "--probe-steps",
}


@dataclass(frozen=True)
class SourceGateRuntimePaths:
    """Normalized immutable inputs and a still-absent output directory."""

    data_root: Path
    backbone_checkpoint: Path
    parent_reranker: Path
    geometry_reranker: Path
    output_dir: Path
    output_parent: Path
    output_parent_device: int
    output_parent_inode: int


def _probe_steps(value):
    try:
        steps = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("probe steps must be an integer")
    if str(steps) != str(value) or steps not in (1, PRODUCTION_PROBE_STEPS):
        raise argparse.ArgumentTypeError("probe steps must be exactly 1 or 306")
    return steps


def parse_args(argv=None):
    """Parse the exact source-gate probe command line."""
    parser = argparse.ArgumentParser(
        description="Run the train-only ScanRefer REC source-gate probe.",
        allow_abbrev=False,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--parent-reranker", required=True)
    parser.add_argument("--geometry-reranker", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=PRODUCTION_DEVICE)
    parser.add_argument(
        "--probe-steps", type=_probe_steps, default=PRODUCTION_PROBE_STEPS
    )
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    option_tokens = [
        token for token in raw_arguments
        if isinstance(token, str) and token.startswith("--")
    ]
    if (any(token not in _CLI_OPTIONS for token in option_tokens)
            or any(option_tokens.count(option) > 1
                   for option in _CLI_OPTIONS)):
        parser.error("only exact, non-duplicated source-gate options are allowed")
    args = parser.parse_args(raw_arguments)
    if args.device != PRODUCTION_DEVICE:
        parser.error("--device must be cuda:0")
    return args


def _logical_absolute(value, label):
    if (not isinstance(value, (str, os.PathLike))
            or isinstance(value, bytes)):
        raise ValueError("{} must be path-like".format(label))
    try:
        return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("{} is invalid: {}".format(label, error))


def _reject_symlink_ancestors(path, label, include_leaf=True):
    current = Path(path) if include_leaf else Path(path).parent
    while True:
        try:
            metadata = os.lstat(str(current))
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ValueError(
                "could not inspect {} ancestors: {}".format(label, error)
            )
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    "{} has a symlink ancestor: {}".format(label, current)
                )
        parent = current.parent
        if parent == current:
            break
        current = parent


def _comparison_path(path):
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("could not normalize path: {}".format(error))


def _paths_overlap(first, second):
    first_value = str(_comparison_path(first))
    second_value = str(_comparison_path(second))
    try:
        common = os.path.commonpath((first_value, second_value))
    except ValueError:
        return False
    return common in (first_value, second_value)


def _same_existing_file(first, second):
    if _comparison_path(first) == _comparison_path(second):
        return True
    try:
        return os.path.samefile(str(first), str(second))
    except OSError:
        return False


def _require_read_only_input(path, label):
    logical = _logical_absolute(path, label)
    _reject_symlink_ancestors(logical, label)
    try:
        metadata = os.stat(str(logical), follow_symlinks=False)
    except OSError as error:
        raise ValueError(
            "{} does not exist as a regular file: {}".format(label, error)
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("{} must be a regular non-symlink file".format(label))
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        raise ValueError("{} must be read-only".format(label))
    return logical.resolve(strict=True)


def validate_runtime_paths(args):
    """Validate immutable inputs and output isolation without creating paths."""
    if args is None:
        raise ValueError("runner arguments are required")
    data_logical = _logical_absolute(
        getattr(args, "data_root", None), "data root"
    )
    _reject_symlink_ancestors(data_logical, "data root")
    if not data_logical.is_dir():
        raise ValueError("data root must be an existing non-symlink directory")
    data_root = data_logical.resolve(strict=True)

    backbone = _require_read_only_input(
        getattr(args, "backbone_checkpoint", None), "backbone checkpoint"
    )
    parent = _require_read_only_input(
        getattr(args, "parent_reranker", None), "parent reranker"
    )
    geometry = _require_read_only_input(
        getattr(args, "geometry_reranker", None), "geometry reranker"
    )
    inputs = (backbone, parent, geometry)
    for index, first in enumerate(inputs):
        for second in inputs[index + 1:]:
            if _same_existing_file(first, second):
                raise ValueError("runner input files must be distinct")

    output_logical = _logical_absolute(
        getattr(args, "output_dir", None), "output directory"
    )
    protected = (
        (data_root,) + inputs + tuple(path.parent for path in inputs)
        + tuple(PROTECTED_LEGACY_PATHS)
    )
    for path in protected:
        if _paths_overlap(output_logical, path):
            raise ValueError(
                "output directory overlaps protected tree: {}".format(path)
            )
    _reject_symlink_ancestors(
        output_logical, "output directory", include_leaf=True
    )
    if output_logical.exists() or output_logical.is_symlink():
        raise FileExistsError(
            "final output directory must not exist: {}".format(output_logical)
        )
    output_parent = output_logical.parent
    if not output_parent.is_dir():
        raise ValueError("output parent must be an existing directory")
    _reject_symlink_ancestors(output_parent, "output parent")
    output = _comparison_path(output_logical)
    try:
        output_parent_metadata = os.stat(
            str(output_parent), follow_symlinks=False
        )
    except OSError as error:
        raise ValueError(
            "could not capture output parent identity: {}".format(error)
        )
    if not stat.S_ISDIR(output_parent_metadata.st_mode):
        raise ValueError("output parent must be a real directory")
    _reject_symlink_ancestors(output_parent, "output parent")
    return SourceGateRuntimePaths(
        data_root=data_root,
        backbone_checkpoint=backbone,
        parent_reranker=parent,
        geometry_reranker=geometry,
        output_dir=output,
        output_parent=output.parent,
        output_parent_device=int(output_parent_metadata.st_dev),
        output_parent_inode=int(output_parent_metadata.st_ino),
    )


def _file_hash_record(path):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "sha256": rec_finetune.sha256_file(path),
    }


def _iter_source_gate_python_files(directory):
    """Return Python sources without descending into runtime cache trees."""
    root = Path(directory).resolve(strict=True)
    result = []
    for current, directory_names, filenames in os.walk(str(root)):
        directory_names[:] = sorted(
            name for name in directory_names
            if name not in _SOURCE_GATE_MANIFEST_PRUNED_DIRECTORIES
        )
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                result.append(Path(current) / filename)
    return tuple(sorted(result))


def _source_gate_code_paths():
    root = Path(__file__).resolve().parents[1]
    models_root = Path(rec_source_gate.__file__).resolve().parent
    paths = {
        "runner": Path(legacy.__file__).resolve(),
        "rec_finetune": Path(rec_finetune.__file__).resolve(),
        "rec_reranker": models_root / "rec_reranker.py",
        "rec_candidate_adapter": models_root / "rec_candidate_adapter.py",
        "rec_mask_geometry": models_root / "rec_mask_geometry.py",
        "rec_geometry_reranker": models_root / "rec_geometry_reranker.py",
        "source_choice_adapter": models_root / "source_choice_adapter.py",
        "source_choice_selector": models_root / "source_choice_selector.py",
        "losses": models_root / "losses.py",
        "cache_rec_candidates": (
            root / "scripts" / "cache_scanrefer_rec_candidates.py"
        ),
        "train_rec_geometry_reranker": (
            root / "scripts" / "train_rec_geometry_reranker.py"
        ),
        "train_dist_mod": root / "train_dist_mod.py",
    }
    for directory_name in (
            "models", "src", "data", "utils", "pointnet2",
            "sng_parser", "scripts"):
        directory = root / directory_name
        for path in _iter_source_gate_python_files(directory):
            relative = path.relative_to(root).as_posix()
            if relative in _SOURCE_GATE_MANIFEST_EXCLUDED_FILES:
                continue
            paths["source/" + relative] = path
    for filename in ("main_utils.py", "train_dist_mod.py"):
        paths["source/" + filename] = root / filename
    for path in sorted((root / "pointnet2").glob("_ext*.so")):
        relative = path.relative_to(root).as_posix()
        paths["binary/" + relative] = path
    return paths


def read_source_gate_input_identity(path, label):
    """Read a stable public identity for one immutable input artifact."""
    resolved = _require_read_only_input(path, label)
    stable_path, _snapshot, digest = rec_finetune._stable_artifact_snapshot(
        resolved, label
    )
    metadata = os.stat(str(stable_path), follow_symlinks=False)
    return {
        "path": str(stable_path),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "size": int(metadata.st_size),
        "sha256": digest,
    }


def _capture_input_identities(paths, identity_reader):
    result = {}
    for name, path in (
            ("backbone_checkpoint", paths.backbone_checkpoint),
            ("parent_reranker", paths.parent_reranker),
            ("geometry_reranker", paths.geometry_reranker)):
        identity = identity_reader(path, name.replace("_", " "))
        _validate_public_input_identity(identity, name)
        result[name] = copy.deepcopy(identity)
    if len({identity["path"] for identity in result.values()}) != 3:
        raise ValueError("source-gate input identities must be distinct")
    return result


def build_source_gate_code_manifest():
    """Capture every publication dependency before model/data construction."""
    paths = _source_gate_code_paths()
    paths["source_gate_runner"] = Path(__file__).resolve()
    paths["rec_source_gate"] = Path(rec_source_gate.__file__).resolve()
    paths["train_rec_finetune"] = Path(legacy.__file__).resolve()
    manifest = {
        name: _file_hash_record(path) for name, path in paths.items()
    }
    if not _REQUIRED_CODE_MANIFEST_ENTRIES.issubset(manifest):
        raise RuntimeError("source-gate code manifest is incomplete")
    return copy.deepcopy(manifest)


def _canonical_json_bytes(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("value is not canonical JSON: {}".format(error))


def _manifest_sha256(manifest):
    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source-gate code manifest must be a non-empty mapping")
    for name, record in manifest.items():
        if (not isinstance(name, str) or not name
                or not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or not isinstance(record["path"], str)
                or not record["path"]
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64):
            raise ValueError("source-gate code manifest record is invalid")
        try:
            int(record["sha256"], 16)
        except ValueError:
            raise ValueError("source-gate code manifest digest is invalid")
    return copy.deepcopy(manifest)


def _validate_train_only_data(data):
    expected = {
        "dataset", "split", "fit_view", "calibration_view",
        "fit_loader", "calibration_loader",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(
            "train-only data contains validation, test, cache, or unknown objects"
        )


def build_source_gate_train_only_data(config, device):
    """Build the legacy train split with source-gate-only unpinned loaders."""
    if torch.device(device).type != "cuda":
        raise ValueError("source-gate data loaders require a CUDA device")

    def build_unpinned_loader(dataset, **kwargs):
        if (type(kwargs.get("num_workers")) is not int
                or kwargs["num_workers"] != legacy.PRODUCTION_NUM_WORKERS
                or kwargs.get("pin_memory") is not True):
            raise ValueError(
                "legacy loader request differs from the source-gate baseline"
            )
        loader_kwargs = dict(kwargs)
        loader_kwargs["pin_memory"] = False
        return legacy.DataLoader(dataset, **loader_kwargs)

    return legacy.build_train_only_data(
        config, device, loader_factory=build_unpinned_loader
    )


def _source_gate_loader_execution(data):
    if not isinstance(data, dict):
        raise ValueError("source-gate live loader data is unavailable")
    execution = {}
    for public_name, data_name in (
            ("fit", "fit_loader"),
            ("calibration", "calibration_loader")):
        loader = data.get(data_name)
        num_workers = getattr(loader, "num_workers", None)
        pin_memory = getattr(loader, "pin_memory", None)
        if (type(num_workers) is not int
                or type(pin_memory) is not bool):
            raise ValueError(
                "source-gate live loader execution settings are invalid"
            )
        execution[public_name] = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
    return execution


def build_source_gate_train_data_contract(initialized):
    """Bind the legacy train split plus both live loader execution modes."""
    base = legacy.build_rec_finetune_train_data_contract(initialized)
    legacy._validate_train_data_contract(base)
    contract = copy.deepcopy(base)
    contract["schema"] = SOURCE_GATE_TRAIN_DATA_SCHEMA
    contract["loader_execution"] = _source_gate_loader_execution(
        initialized.get("data")
    )
    return _validate_train_data_contract(contract)


def _validate_train_data_contract(contract):
    if (not isinstance(contract, dict)
            or contract.get("schema") != SOURCE_GATE_TRAIN_DATA_SCHEMA):
        raise ValueError("source-gate train-only data contract is inexact")
    execution = contract.get("loader_execution")
    if (not isinstance(execution, dict)
            or set(execution) != {"fit", "calibration"}):
        raise ValueError("source-gate loader execution contract is invalid")
    for name in ("fit", "calibration"):
        record = execution[name]
        if (not isinstance(record, dict)
                or set(record) != {"num_workers", "pin_memory"}
                or type(record["num_workers"]) is not int
                or record["num_workers"] != legacy.PRODUCTION_NUM_WORKERS
                or type(record["pin_memory"]) is not bool
                or record["pin_memory"] is not False):
            raise ValueError(
                "source-gate loader execution contract is invalid"
            )
    legacy_contract = copy.deepcopy(contract)
    del legacy_contract["loader_execution"]
    legacy_contract["schema"] = "scanrefer-rec-finetune-train-data-v1"
    legacy._validate_train_data_contract(legacy_contract)
    if (contract["fit_sample_count"] != PRODUCTION_FIT_SAMPLE_COUNT
            or contract["calibration_sample_count"]
            != PRODUCTION_CALIBRATION_SAMPLE_COUNT
            or contract["batch_size"] != PRODUCTION_BATCH_SIZE
            or contract["drop_last"] is not False
            or contract["calibration_augment"] is not False
            or contract["calibration_augment_det"] is not False
            or contract["validation_data_accessed"] is not False
            or contract["validation_data_objects_present"] is not False):
        raise ValueError("source-gate train-only data contract is inexact")
    return copy.deepcopy(contract)


def validate_live_source_gate_data_contract(initialized):
    """Recheck the sealed source-gate loader settings against live objects."""
    if not isinstance(initialized, dict):
        raise ValueError("live source-gate data state must be a mapping")
    contract = _validate_train_data_contract(
        initialized.get("train_data_contract")
    )
    if (_source_gate_loader_execution(initialized.get("data"))
            != contract["loader_execution"]):
        raise RuntimeError("live source-gate loader contract changed")
    return contract


def _require_fresh_legacy_optimizer(state):
    if not isinstance(state, dict):
        raise ValueError("legacy initial state must be a mapping")
    if "groups" not in state or "optimizer" not in state:
        raise ValueError("legacy initial state has no joint optimizer")
    optimizer = state["optimizer"]
    try:
        optimizer_state = optimizer.state
        parameter_groups = optimizer.param_groups
    except AttributeError as error:
        raise ValueError("legacy joint optimizer is invalid") from error
    if (len(optimizer_state) != 0
            or not isinstance(parameter_groups, list)
            or not parameter_groups):
        raise ValueError("legacy joint optimizer must be fresh with zero state")
    del state["optimizer"]
    del state["groups"]


def initialize_source_gate_probe(
        args, *, device=None, manifest_builder=None,
        initial_state_loader=None, data_builder=None,
        data_contract_builder=None, trainability_configurer=None,
        optimizer_builder=None, identity_reader=None):
    """Initialize only train data and replace the unused legacy optimizer."""
    paths = validate_runtime_paths(args)
    runtime_device = str(device if device is not None else args.device)
    try:
        torch.device(runtime_device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError("source-gate device is invalid") from error

    manifest_builder = manifest_builder or build_source_gate_code_manifest
    initial_state_loader = (
        initial_state_loader or legacy.load_rec_finetune_initial_state
    )
    data_builder = data_builder or build_source_gate_train_only_data
    data_contract_builder = (
        data_contract_builder or build_source_gate_train_data_contract
    )
    trainability_configurer = (
        trainability_configurer
        or rec_source_gate.configure_rec_source_gate_trainability
    )
    optimizer_builder = (
        optimizer_builder or rec_source_gate.build_rec_source_gate_optimizer
    )
    identity_reader = identity_reader or read_source_gate_input_identity
    hooks = (
        manifest_builder, initial_state_loader, data_builder,
        data_contract_builder, trainability_configurer, optimizer_builder,
        identity_reader,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("source-gate initialization hooks must be callable")

    input_identities_before = _capture_input_identities(
        paths, identity_reader
    )
    publication_code_hashes = _validate_manifest(manifest_builder())
    state = initial_state_loader(
        paths.backbone_checkpoint,
        paths.parent_reranker,
        paths.geometry_reranker,
        paths.data_root,
        device=runtime_device,
    )
    if not isinstance(state, dict):
        raise ValueError("legacy initial state must be a mapping")
    _require_fresh_legacy_optimizer(state)
    required_state = {
        "config", "mcln", "parent", "parent_artifact",
        "geometry", "geometry_artifact",
    }
    if not required_state.issubset(state):
        raise ValueError("legacy initial model state is incomplete")

    source_parameters = trainability_configurer(
        state["mcln"], state["parent"], state["geometry"]
    )
    source_optimizer = optimizer_builder(source_parameters)

    data = data_builder(state["config"], runtime_device)
    _validate_train_only_data(data)
    initialized = {
        "paths": paths,
        "initial_state": state,
        "data": data,
        "source_parameters": source_parameters,
        "source_optimizer": source_optimizer,
        "publication_code_hashes": publication_code_hashes,
        "publication_code_manifest_sha256": _manifest_sha256(
            publication_code_hashes
        ),
        "probe_steps": args.probe_steps,
        "device": runtime_device,
        "seed": PRODUCTION_SEED,
        "legacy_joint_optimizer_updates": 0,
        "input_identities_before": input_identities_before,
    }
    contract = data_contract_builder(initialized)
    initialized["train_data_contract"] = _validate_train_data_contract(
        contract
    )
    return initialized


def _digest_frame(digest, label, payload):
    label_bytes = label if isinstance(label, bytes) else label.encode("ascii")
    digest.update(struct.pack("<Q", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _tensor_raw_bytes(value):
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("state digest tensors need dense real storage")
    byte_count = int(tensor.numel()) * int(tensor.element_size())
    if byte_count == 0:
        return b""
    pointer = int(tensor.data_ptr())
    if pointer == 0:
        raise ValueError("state digest tensor has no readable storage")
    return ctypes.string_at(pointer, byte_count)


def canonical_state_digest(state, *, excluded_prefixes=()):
    """Hash sorted tensor keys, dtypes, shapes, and exact contiguous bytes."""
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise ValueError("state digest input must be a state mapping or Module")
    if (type(excluded_prefixes) is not tuple
            or any(not isinstance(prefix, str) or not prefix
                   for prefix in excluded_prefixes)):
        raise ValueError("state digest excluded prefixes must be exact strings")
    selected = {
        name: value for name, value in state.items()
        if not any(name.startswith(prefix) for prefix in excluded_prefixes)
    }
    if any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("state digest keys must be non-empty strings")
    if any(not isinstance(value, torch.Tensor) for value in selected.values()):
        raise ValueError("state digest values must be tensors")

    digest = hashlib.sha256()
    _digest_frame(digest, "schema", STATE_DIGEST_FORMAT.encode("ascii"))
    for name in sorted(selected):
        value = selected[name].detach().to(device="cpu").contiguous()
        _digest_frame(digest, "tensor", b"")
        _digest_frame(digest, "key", name.encode("utf-8"))
        _digest_frame(digest, "dtype", str(value.dtype).encode("ascii"))
        _digest_frame(
            digest,
            "shape",
            json.dumps(
                list(value.shape), separators=(",", ":")
            ).encode("ascii"),
        )
        _digest_frame(digest, "bytes", _tensor_raw_bytes(value))
    return digest.hexdigest()


def source_gate_state_digests(mcln, parent, geometry):
    """Return full and frozen canonical state digests for the three models."""
    for label, model in (
            ("mcln", mcln), ("parent", parent), ("geometry", geometry)):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("{} must be a Module".format(label))
    return {
        "mcln_full": canonical_state_digest(mcln),
        "mcln_frozen": canonical_state_digest(
            mcln,
            excluded_prefixes=(
                rec_source_gate.SOURCE_GATE_TRAINABLE_PREFIX,
            ),
        ),
        "parent": canonical_state_digest(parent),
        "geometry": canonical_state_digest(geometry),
    }


def _require_calibration_initialized(initialized):
    if not isinstance(initialized, dict):
        raise ValueError("initialized source-gate probe must be a mapping")
    state = initialized.get("initial_state")
    data = initialized.get("data")
    contract = initialized.get("train_data_contract")
    if (not isinstance(state, dict) or not isinstance(data, dict)
            or not isinstance(contract, dict)):
        raise ValueError("initialized source-gate calibration is incomplete")
    required_state = {
        "mcln", "parent", "parent_artifact", "geometry",
        "geometry_artifact",
    }
    if not required_state.issubset(state):
        raise ValueError("source-gate calibration model state is incomplete")
    calibration_view = data.get("calibration_view")
    expected_indices = tuple(getattr(calibration_view, "indices", ()))
    expected_count = contract.get("calibration_sample_count")
    if (not expected_indices
            or type(expected_count) is not int
            or expected_count != len(expected_indices)):
        raise ValueError("source-gate calibration index contract is invalid")
    if "calibration_loader" not in data:
        raise ValueError("source-gate calibration loader is unavailable")
    return state, data, expected_indices, expected_count


def _as_calibration_float(value, label):
    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a tensor".format(label))
    return value.detach().float()


def calibrate_source_gate_probe(
        initialized, *, baseline=None, move_batch=None, input_builder=None,
        full_state_builder=None, compact_state_builder=None,
        target_attacher=None, forward_fn=None, selected_iou_builder=None,
        eval_mode_setter=None, accumulator_factory=None):
    """Run one exact ordered train-calibration pass with frozen rerankers."""
    state, data, expected_indices, expected_count = (
        _require_calibration_initialized(initialized)
    )
    move_batch = move_batch or legacy._move_batch_to_device
    input_builder = input_builder or legacy.build_rec_finetune_inputs
    full_state_builder = full_state_builder or build_full_rec_query_state
    compact_state_builder = (
        compact_state_builder or compact_rec_query_state
    )
    target_attacher = (
        target_attacher or rec_source_gate.attach_full_query_targets
    )
    forward_fn = forward_fn or rec_finetune.build_rec_finetune_forward
    selected_iou_builder = (
        selected_iou_builder or legacy._selected_calibration_ious
    )
    eval_mode_setter = (
        eval_mode_setter or rec_source_gate.set_rec_source_gate_eval_mode
    )
    accumulator_factory = (
        accumulator_factory
        or rec_source_gate.RecSourceGateCalibrationAccumulator
    )
    hooks = (
        move_batch, input_builder, full_state_builder,
        compact_state_builder, target_attacher, forward_fn,
        selected_iou_builder, eval_mode_setter, accumulator_factory,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("source-gate calibration hooks must be callable")

    runtime_device = torch.device(initialized["device"])
    accumulator = accumulator_factory(expected_indices, baseline=baseline)
    mcln = state["mcln"]
    parent = state["parent"]
    geometry = state["geometry"]
    eval_mode_setter(mcln, parent, geometry)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for batch in data["calibration_loader"]:
            moved_batch = move_batch(batch, runtime_device)
            if not isinstance(moved_batch, dict):
                raise ValueError("calibration batch must be a mapping")
            inputs = input_builder(moved_batch)
            if not isinstance(inputs, dict):
                raise ValueError("calibration inputs must be a mapping")
            legacy._reject_rec_target_only_fields(
                inputs, "source-gate calibration inputs"
            )
            inputs["train"] = False
            end_points = mcln(inputs)
            if not isinstance(end_points, dict):
                raise ValueError("MCLN output must be a mapping")
            legacy._reject_rec_target_only_fields(
                inputs, "post-MCLN source-gate calibration inputs"
            )
            legacy._reject_rec_target_only_fields(
                end_points, "source-gate calibration MCLN outputs"
            )

            full_state = full_state_builder(end_points, inputs)
            if not isinstance(full_state, dict):
                raise ValueError("full source-gate state must be a mapping")
            compact_state = compact_state_builder(full_state)
            if not isinstance(compact_state, dict):
                raise ValueError("compact source-gate state must be a mapping")
            full_ious = target_attacher(
                full_state, moved_batch, root_only=True
            )
            forward_state = forward_fn(
                end_points,
                inputs,
                moved_batch,
                parent,
                state["parent_artifact"],
                geometry,
                state["geometry_artifact"],
            )
            if not isinstance(forward_state, dict):
                raise ValueError("frozen REC forward state must be a mapping")
            parent_state = forward_state.get("parent_state")
            parent_inputs = forward_state.get("parent_model_inputs")
            geometry_inputs = forward_state.get("geometry_model_inputs")
            if (not isinstance(parent_state, dict)
                    or not isinstance(parent_inputs, dict)):
                raise ValueError("frozen REC parent candidate state is incomplete")

            compact_indices = compact_state.get("query_indices")
            compact_valid = compact_state.get("valid_mask")
            parent_indices = parent_state.get("query_indices")
            parent_state_valid = parent_state.get("candidate_valid")
            parent_valid = parent_inputs.get("valid_mask")
            tensors = (
                compact_indices, compact_valid, parent_indices,
                parent_state_valid, parent_valid,
            )
            if (not all(isinstance(value, torch.Tensor) for value in tensors)
                    or not torch.equal(compact_indices, parent_indices)
                    or not torch.equal(compact_valid, parent_state_valid)
                    or not torch.equal(compact_valid, parent_valid)):
                raise ValueError(
                    "independently built compact candidates differ from parent state"
                )
            if not isinstance(geometry_inputs, dict):
                raise ValueError(
                    "frozen REC geometry candidate state is incomplete"
                )
            parent_top1_mask = parent_state.get("parent_top1_mask")
            if (not isinstance(parent_top1_mask, torch.Tensor)
                    or parent_top1_mask.dtype != torch.bool
                    or parent_top1_mask.shape != compact_indices.shape
                    or not bool(
                        (parent_top1_mask.sum(dim=1) == 1).all().item()
                    )):
                raise ValueError("parent compact Top-1 mask is invalid")
            parent_top1_positions = parent_top1_mask.long().argmax(dim=1)

            observation = {
                "full_query_ious": _as_calibration_float(
                    full_ious, "full-query IoUs"
                ),
                "default_scores": _as_calibration_float(
                    full_state.get("default_scores"), "default scores"
                ),
                "contrastive_scores": _as_calibration_float(
                    full_state.get("contrastive_scores"),
                    "contrastive scores",
                ),
                "compact_query_indices": compact_indices.detach().long(),
                "compact_valid_mask": compact_valid.detach().bool(),
                "parent_candidate_ious": _as_calibration_float(
                    forward_state.get("parent_candidate_ious"),
                    "parent candidate IoUs",
                ),
                "parent_valid_mask": parent_valid.detach().bool(),
                "parent_top1_positions": parent_top1_positions.detach().long(),
                "geometry_candidate_ious": _as_calibration_float(
                    forward_state.get("geometry_candidate_ious"),
                    "geometry candidate IoUs",
                ),
                "geometry_valid_mask": geometry_inputs[
                    "valid_mask"
                ].detach().bool(),
                "geometry_selected_ious": _as_calibration_float(
                    selected_iou_builder(forward_state),
                    "geometry selected IoUs",
                ),
            }
            indices = moved_batch.get("dataset_index")
            if indices is None:
                raise ValueError("calibration batch has no dataset_index")
            accumulator.update(indices, observation)
    return {
        "accumulator": accumulator,
        "report": accumulator.finalize(expected_count),
    }


def _is_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or value.lower() != value):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_exact_int(value, label, minimum=0, maximum=None):
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError("{} is not an exact integer count".format(label))
    return value


def _require_exact_finite_float(value, label):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("{} is not an exact finite float".format(label))
    return value


def validate_source_gate_calibration_report(
        report, *, baseline_report=None, expected_sample_count=None):
    """Strictly validate Task4 metrics, transitions, and digest schema."""
    top_fields = {
        "schema", "sample_count", "baseline_present", "metrics",
        "transitions", "digests",
    }
    if (not isinstance(report, dict) or set(report) != top_fields
            or report.get("schema") != "rec-source-gate-calibration-v1"):
        raise ValueError("source-gate calibration report schema is invalid")
    sample_count = _require_exact_int(
        report.get("sample_count"), "calibration sample_count", minimum=1
    )
    if (expected_sample_count is not None
            and (type(expected_sample_count) is not int
                 or sample_count != expected_sample_count)):
        raise ValueError("calibration sample_count differs from contract")
    baseline_present = report.get("baseline_present")
    if type(baseline_present) is not bool:
        raise ValueError("calibration baseline_present must be boolean")
    if baseline_present is not (baseline_report is not None):
        raise ValueError("calibration baseline presence is incoherent")

    metrics = report.get("metrics")
    transitions = report.get("transitions")
    if (not isinstance(metrics, dict)
            or set(metrics) != set(_CALIBRATION_GROUPS)
            or not isinstance(transitions, dict)
            or set(transitions) != set(_CALIBRATION_GROUPS)):
        raise ValueError("calibration report groups are invalid")
    for group, expected_branches in _CALIBRATION_GROUPS.items():
        group_metrics = metrics[group]
        group_transitions = transitions[group]
        if (not isinstance(group_metrics, dict)
                or set(group_metrics) != set(expected_branches)
                or not isinstance(group_transitions, dict)
                or set(group_transitions) != set(expected_branches)):
            raise ValueError("calibration report branches are invalid")
        for branch in expected_branches:
            metric = group_metrics[branch]
            transition = group_transitions[branch]
            if (not isinstance(metric, dict)
                    or set(metric) != _CALIBRATION_METRIC_FIELDS
                    or not isinstance(transition, dict)
                    or set(transition) != _CALIBRATION_TRANSITION_FIELDS):
                raise ValueError("calibration branch schema is invalid")
            for suffix in ("025", "050"):
                hits = _require_exact_int(
                    metric["hits" + suffix],
                    "calibration hits" + suffix,
                    maximum=sample_count,
                )
                accuracy = _require_exact_finite_float(
                    metric["acc" + suffix],
                    "calibration acc" + suffix,
                )
                if accuracy != hits / float(sample_count):
                    raise ValueError(
                        "calibration accuracy differs from integer hits"
                    )
                gained = _require_exact_int(
                    transition["gained" + suffix],
                    "calibration gained" + suffix,
                    maximum=sample_count,
                )
                lost = _require_exact_int(
                    transition["lost" + suffix],
                    "calibration lost" + suffix,
                    maximum=sample_count,
                )
                if baseline_report is None:
                    if gained != 0 or lost != 0:
                        raise ValueError(
                            "step-0 calibration transitions must be zero"
                        )
            if metric["hits050"] > metric["hits025"]:
                raise ValueError(
                    "calibration threshold hits are not nested"
                )

    if (metrics["membership"]["union_top16"]
            != metrics["candidate_oracle"]["union_query"]
            or transitions["membership"]["union_top16"]
            != transitions["candidate_oracle"]["union_query"]):
        raise ValueError("calibration union branches are not identical")

    digests = report.get("digests")
    if (not isinstance(digests, dict)
            or set(digests) != {
                "canonical_format", "raw_query_ious_sha256",
                "geometry_selected_ious_sha256",
            }
            or digests.get("canonical_format")
            != "rec-source-gate-calibration-float32-sha256-v1"
            or not _is_sha256(digests.get("raw_query_ious_sha256"))
            or not _is_sha256(
                digests.get("geometry_selected_ious_sha256")
            )):
        raise ValueError("calibration report digests are invalid")

    if baseline_report is not None:
        baseline = validate_source_gate_calibration_report(
            baseline_report, expected_sample_count=sample_count
        )
        for group, branches in _CALIBRATION_GROUPS.items():
            for branch in branches:
                current_metric = metrics[group][branch]
                previous_metric = baseline["metrics"][group][branch]
                transition = transitions[group][branch]
                for suffix in ("025", "050"):
                    current_hits = current_metric["hits" + suffix]
                    previous_hits = previous_metric["hits" + suffix]
                    gained = transition["gained" + suffix]
                    lost = transition["lost" + suffix]
                    if (current_hits - previous_hits != gained - lost
                            or gained > sample_count - previous_hits
                            or gained > current_hits
                            or lost > previous_hits
                            or lost > sample_count - current_hits):
                        raise ValueError(
                            "calibration transition equation is invalid"
                        )
    try:
        _canonical_json_bytes(report)
    except ValueError as error:
        raise ValueError("calibration report is not public JSON") from error
    return copy.deepcopy(report)


def _validated_private_calibration_bits(accumulator, report):
    accumulator_type = rec_source_gate.RecSourceGateCalibrationAccumulator
    if (type(accumulator) is not accumulator_type
            or getattr(accumulator, "_finalized", None) is not True
            or getattr(accumulator, "_report", None) != report):
        raise ValueError(
            "calibration accumulator is not the exact finalized report source"
        )
    sample_count = report["sample_count"]
    expected_indices = getattr(accumulator, "_expected_indices", None)
    if (type(expected_indices) is not tuple
            or len(expected_indices) != sample_count
            or len(set(expected_indices)) != sample_count
            or any(type(index) is not int or index < 0
                   for index in expected_indices)):
        raise ValueError("calibration accumulator index order is invalid")
    hit_bits = getattr(accumulator, "_hit_bits", None)
    if (not isinstance(hit_bits, dict)
            or set(hit_bits) != _CALIBRATION_INTERNAL_BRANCHES):
        raise ValueError("calibration accumulator branches are invalid")
    frozen = {}
    for name in sorted(_CALIBRATION_INTERNAL_BRANCHES):
        branch = hit_bits[name]
        if not isinstance(branch, dict) or set(branch) != {"025", "050"}:
            raise ValueError("calibration accumulator threshold schema is invalid")
        frozen[name] = {}
        for suffix in ("025", "050"):
            bits = branch[suffix]
            if (type(bits) is not tuple or len(bits) != sample_count
                    or any(type(bit) is not bool for bit in bits)):
                raise ValueError(
                    "calibration accumulator hit bits are invalid"
                )
            frozen[name][suffix] = bits
        if any(
                bit050 and not bit025
                for bit025, bit050 in zip(
                    frozen[name]["025"], frozen[name]["050"]
                )):
            raise ValueError(
                "calibration accumulator threshold bits are not nested"
            )
    for group, branches in _CALIBRATION_BRANCH_BINDINGS.items():
        if set(branches) != set(_CALIBRATION_GROUPS[group]):
            raise RuntimeError("calibration branch binding is incomplete")
        for public_name, internal_name in branches.items():
            metric = report["metrics"][group][public_name]
            if any(
                    sum(frozen[internal_name][suffix])
                    != metric["hits" + suffix]
                    for suffix in ("025", "050")):
                raise ValueError(
                    "calibration accumulator bits differ from report metrics"
                )
    return expected_indices, frozen


def _expected_bit_transitions(current_bits, baseline_bits):
    result = {}
    for suffix in ("025", "050"):
        if baseline_bits is None:
            gained = 0
            lost = 0
        else:
            gained = sum(
                (not previous) and current
                for previous, current in zip(
                    baseline_bits[suffix], current_bits[suffix]
                )
            )
            lost = sum(
                previous and (not current)
                for previous, current in zip(
                    baseline_bits[suffix], current_bits[suffix]
                )
            )
        result["gained" + suffix] = gained
        result["lost" + suffix] = lost
    return result


def _source_gate_calibration_hit_bits_sha256(
        accumulator, report, *, baseline_accumulator=None):
    """Hash finalized ordered hit bits without publishing the bit arrays."""
    if baseline_accumulator is None:
        validated_report = validate_source_gate_calibration_report(report)
        baseline_bits = None
    else:
        baseline_report = getattr(baseline_accumulator, "_report", None)
        validated_baseline = validate_source_gate_calibration_report(
            baseline_report
        )
        validated_report = validate_source_gate_calibration_report(
            report, baseline_report=validated_baseline
        )
        baseline_indices, baseline_bits = (
            _validated_private_calibration_bits(
                baseline_accumulator, validated_baseline
            )
        )
    expected_indices, current_bits = _validated_private_calibration_bits(
        accumulator, validated_report
    )
    if (baseline_accumulator is not None
            and baseline_indices != expected_indices):
        raise ValueError("calibration baseline index order changed")
    for group, branches in _CALIBRATION_BRANCH_BINDINGS.items():
        for public_name, internal_name in branches.items():
            expected_transition = _expected_bit_transitions(
                current_bits[internal_name],
                None if baseline_bits is None else baseline_bits[internal_name],
            )
            if (validated_report["transitions"][group][public_name]
                    != expected_transition):
                raise ValueError(
                    "calibration accumulator bits differ from transitions"
                )
    private_payload = {
        "canonical_format": CALIBRATION_HIT_BITS_DIGEST_FORMAT,
        "dataset_indices": list(expected_indices),
        "sample_count": validated_report["sample_count"],
        "branches": {
            name: {
                suffix: list(current_bits[name][suffix])
                for suffix in ("025", "050")
            }
            for name in sorted(_CALIBRATION_INTERNAL_BRANCHES)
        },
    }
    return hashlib.sha256(_canonical_json_bytes(private_payload)).hexdigest()


def _source_gate_calibration_binding_sha256(
        step0_hit_bits_sha256, current_hit_bits_sha256, report):
    if (not _is_sha256(step0_hit_bits_sha256)
            or not _is_sha256(current_hit_bits_sha256)
            or not isinstance(report, dict)
            or set(report) != {
                "schema", "sample_count", "baseline_present", "metrics",
                "transitions", "digests",
            }):
        raise ValueError("calibration binding inputs are invalid")
    payload = {
        "canonical_format": CALIBRATION_BINDING_FORMAT,
        "step0_hit_bits_sha256": step0_hit_bits_sha256,
        "current_hit_bits_sha256": current_hit_bits_sha256,
        "sample_count": report["sample_count"],
        "metrics": report["metrics"],
        "transitions": report["transitions"],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _calibration_evidence_record(
        step0_hit_bits_sha256, current_hit_bits_sha256, report):
    return {
        "hit_bits_sha256": current_hit_bits_sha256,
        "binding_sha256": _source_gate_calibration_binding_sha256(
            step0_hit_bits_sha256, current_hit_bits_sha256, report
        ),
    }


def _validate_state_digest_bundle(value, label):
    fields = {"mcln_full", "mcln_frozen", "parent", "geometry"}
    if (not isinstance(value, dict) or set(value) != fields
            or any(not _is_sha256(value[name]) for name in fields)):
        raise ValueError("{} state digest bundle is invalid".format(label))
    return value


def evaluate_source_gate_gate(
        step0_report, final_report, *, baseline_state_digests,
        final_state_digests, training_diagnostics, final_step=306):
    """Recompute the source-gate eligibility decision from strict evidence."""
    if type(final_step) is not int or final_step not in (1, 306):
        raise ValueError("source-gate final step must be exactly 1 or 306")
    step0 = validate_source_gate_calibration_report(step0_report)
    final = validate_source_gate_calibration_report(
        final_report,
        baseline_report=step0,
        expected_sample_count=step0["sample_count"],
    )
    baseline_state = _validate_state_digest_bundle(
        baseline_state_digests, "step-0"
    )
    final_state = _validate_state_digest_bundle(
        final_state_digests, "final"
    )
    if not isinstance(training_diagnostics, dict):
        raise ValueError("training diagnostics must be a mapping")
    informative = training_diagnostics.get("informative_rows_total")
    if type(informative) is not int or informative < 0:
        raise ValueError("training informative row count is invalid")

    initial_metrics = step0["metrics"]
    final_metrics = final["metrics"]
    checks = {
        "geometry_top1_hits025_nonregression": (
            final_metrics["top1"]["geometry"]["hits025"]
            >= initial_metrics["top1"]["geometry"]["hits025"]
        ),
        "geometry_top1_hits050_nonregression": (
            final_metrics["top1"]["geometry"]["hits050"]
            >= initial_metrics["top1"]["geometry"]["hits050"]
        ),
        "parent_candidate_hits025_nonregression": (
            final_metrics["candidate_oracle"]["parent_candidate"][
                "hits025"
            ] >= initial_metrics["candidate_oracle"]["parent_candidate"][
                "hits025"
            ]
        ),
        "parent_candidate_hits050_nonregression": (
            final_metrics["candidate_oracle"]["parent_candidate"][
                "hits050"
            ] >= initial_metrics["candidate_oracle"]["parent_candidate"][
                "hits050"
            ]
        ),
        "geometry_candidate_hits025_nonregression": (
            final_metrics["candidate_oracle"]["geometry_candidate"][
                "hits025"
            ] >= initial_metrics["candidate_oracle"]["geometry_candidate"][
                "hits025"
            ]
        ),
        "geometry_candidate_hits050_nonregression": (
            final_metrics["candidate_oracle"]["geometry_candidate"][
                "hits050"
            ] >= initial_metrics["candidate_oracle"]["geometry_candidate"][
                "hits050"
            ]
        ),
        "raw_query_hits025_equal": (
            final_metrics["candidate_oracle"]["raw_query"]["hits025"]
            == initial_metrics["candidate_oracle"]["raw_query"]["hits025"]
        ),
        "raw_query_hits050_equal": (
            final_metrics["candidate_oracle"]["raw_query"]["hits050"]
            == initial_metrics["candidate_oracle"]["raw_query"]["hits050"]
        ),
        "raw_query_digest_equal": (
            final["digests"]["raw_query_ious_sha256"]
            == step0["digests"]["raw_query_ious_sha256"]
        ),
        "default_top8_hits025_improved": (
            final_metrics["membership"]["default_top8"]["hits025"]
            > initial_metrics["membership"]["default_top8"]["hits025"]
        ),
        "parent_candidate_hits025_improved": (
            final_metrics["candidate_oracle"]["parent_candidate"][
                "hits025"
            ] > initial_metrics["candidate_oracle"]["parent_candidate"][
                "hits025"
            ]
        ),
        "geometry_top1_hits025_improved": (
            final_metrics["top1"]["geometry"]["hits025"]
            > initial_metrics["top1"]["geometry"]["hits025"]
        ),
        "mcln_frozen_state_unchanged": (
            final_state["mcln_frozen"] == baseline_state["mcln_frozen"]
        ),
        "parent_state_unchanged": (
            final_state["parent"] == baseline_state["parent"]
        ),
        "geometry_state_unchanged": (
            final_state["geometry"] == baseline_state["geometry"]
        ),
        "training_informative_rows_positive": informative > 0,
    }
    checks["strict_improvement_present"] = any((
        checks["default_top8_hits025_improved"],
        checks["parent_candidate_hits025_improved"],
        checks["geometry_top1_hits025_improved"],
    ))
    required = (
        "geometry_top1_hits025_nonregression",
        "geometry_top1_hits050_nonregression",
        "parent_candidate_hits025_nonregression",
        "parent_candidate_hits050_nonregression",
        "geometry_candidate_hits025_nonregression",
        "geometry_candidate_hits050_nonregression",
        "raw_query_hits025_equal",
        "raw_query_hits050_equal",
        "raw_query_digest_equal",
        "strict_improvement_present",
        "mcln_frozen_state_unchanged",
        "parent_state_unchanged",
        "geometry_state_unchanged",
        "training_informative_rows_positive",
    )
    reasons = [name for name in required if not checks[name]]
    eligible = not reasons
    return {
        "schema": "rec-source-gate-decision-v1",
        "checks": checks,
        "reasons": reasons,
        "eligible": eligible,
        "selected_step": final_step if eligible else 0,
    }


def _source_gate_models(initialized):
    if not isinstance(initialized, dict):
        raise ValueError("initialized source-gate probe must be a mapping")
    state = initialized.get("initial_state")
    if not isinstance(state, dict):
        raise ValueError("source-gate initial model state is invalid")
    models = tuple(state.get(name) for name in ("mcln", "parent", "geometry"))
    if not all(isinstance(model, torch.nn.Module) for model in models):
        raise ValueError("source-gate models must be Modules")
    return models


def validate_live_source_gate_contract(initialized):
    """Revalidate the sealed allowlist and the final all-eval model mode."""
    if not isinstance(initialized, dict):
        raise ValueError("live source-gate state must be a mapping")
    contract = initialized.get("source_parameters")
    selected = rec_source_gate._validated_source_gate_contract(contract)
    models = (contract.mcln, contract.parent, contract.geometry)
    if ("initial_state" in initialized
            and models != _source_gate_models(initialized)):
        raise ValueError("source-gate contract roots changed")
    if any(module.training for model in models for module in model.modules()):
        raise RuntimeError("source-gate final model mode is not all-eval")
    if (tuple(id(parameter) for parameter in selected)
            != tuple(id(parameter) for parameter in contract.parameters)):
        raise ValueError("source-gate selected parameter contract changed")
    return contract


def validate_live_source_gate_probe_contract(initialized):
    """Revalidate both trainability and loader execution contracts."""
    contract = validate_live_source_gate_contract(initialized)
    validate_live_source_gate_data_contract(initialized)
    return contract


def _repair_source_gate_trainability(initialized):
    contract = initialized.get("source_parameters")
    models = _source_gate_models(initialized)
    if (getattr(contract, "mcln", None) is models[0]
            and getattr(contract, "parent", None) is models[1]
            and getattr(contract, "geometry", None) is models[2]):
        initialized["source_parameters"] = (
            rec_source_gate.configure_rec_source_gate_trainability(*models)
        )


def _cleanup_source_training_state(
        initialized, *, eval_mode_setter, clear_optimizer=True):
    mcln, parent, geometry = _source_gate_models(initialized)
    optimizer = initialized.get("source_optimizer")
    if clear_optimizer:
        if optimizer is None or not callable(getattr(optimizer, "zero_grad", None)):
            raise ValueError("source-gate optimizer is invalid")
        optimizer.zero_grad(set_to_none=True)
        state = getattr(optimizer, "state", None)
        if state is None or not callable(getattr(state, "clear", None)):
            raise ValueError("source-gate optimizer state is invalid")
        state.clear()
    for model in (mcln, parent, geometry):
        for parameter in model.parameters():
            parameter.grad = None
    eval_mode_setter(mcln, parent, geometry)


def _scalar_value(value, label):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("{} must be scalar".format(label))
        value = value.detach().cpu().item()
    if (not isinstance(value, numbers.Real) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError("{} must be finite".format(label))
    return float(value)


def _integer_stat(value, label):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("{} must be scalar".format(label))
        value = value.detach().cpu().item()
    if type(value) is not int or value < 0:
        raise ValueError("{} must be a nonnegative integer".format(label))
    return value


def _new_training_diagnostics():
    return {
        "losses": [],
        "gradient_norms": [],
        "informative_rows025": 0,
        "informative_rows050": 0,
        "active_violations025": 0,
        "active_violations050": 0,
        "no_positive_rows025": 0,
        "no_positive_rows050": 0,
        "too_few_negative_rows025": 0,
        "too_few_negative_rows050": 0,
        "positive_count025": 0,
        "positive_count050": 0,
        "_positive_cutoff_gap_weighted_sum025": 0.0,
        "_positive_cutoff_gap_weighted_sum050": 0.0,
    }


def _update_training_diagnostics(aggregate, loss, stats, gradient):
    if (not isinstance(stats, dict)
            or set(stats) != {
                "loss025", "loss050", "loss_total",
                "threshold025", "threshold050",
            }):
        raise ValueError("source-gate loss stats schema is invalid")
    aggregate["losses"].append(_scalar_value(loss, "source-gate loss"))
    aggregate["gradient_norms"].append(
        _scalar_value(gradient, "source-gate gradient norm")
    )
    fields = (
        "informative_rows", "active_violations", "no_positive_rows",
        "too_few_negative_rows", "positive_count",
    )
    for suffix in ("025", "050"):
        threshold = stats.get("threshold" + suffix)
        if (not isinstance(threshold, dict)
                or set(threshold) != {
                    "informative_rows", "active_violations",
                    "no_positive_rows", "too_few_negative_rows",
                    "positive_count", "mean_positive_cutoff_gap",
                }):
            raise ValueError("source-gate threshold stats schema is invalid")
        informative_rows = None
        for field in fields:
            count = _integer_stat(
                threshold[field], "{}{}".format(field, suffix)
            )
            aggregate[field + suffix] += count
            if field == "informative_rows":
                informative_rows = count
        mean_gap = _scalar_value(
            threshold["mean_positive_cutoff_gap"],
            "mean_positive_cutoff_gap" + suffix,
        )
        weighted_name = "_positive_cutoff_gap_weighted_sum" + suffix
        weighted_sum = aggregate[weighted_name] + (
            mean_gap * informative_rows
        )
        if not math.isfinite(weighted_sum):
            raise ValueError(
                "mean_positive_cutoff_gap{} aggregate is non-finite".format(
                    suffix
                )
            )
        aggregate[weighted_name] = weighted_sum


def _summary(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("training diagnostic sequence is empty or non-finite")
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
        "last": float(values[-1]),
    }


def _finalize_training_diagnostics(aggregate, completed_steps):
    if (type(completed_steps) is not int or completed_steps <= 0
            or len(aggregate["losses"]) != completed_steps
            or len(aggregate["gradient_norms"]) != completed_steps):
        raise ValueError("source-gate update diagnostics are incomplete")
    result = {
        "schema": "rec-source-gate-training-diagnostics-v1",
        "update_count": completed_steps,
        "loss": _summary(aggregate["losses"]),
        "gradient_norm": _summary(aggregate["gradient_norms"]),
    }
    for name, value in aggregate.items():
        if (name in ("losses", "gradient_norms")
                or name.startswith("_positive_cutoff_gap_weighted_sum")):
            continue
        result[name] = value
    for suffix in ("025", "050"):
        informative_rows = result["informative_rows" + suffix]
        weighted_sum = aggregate[
            "_positive_cutoff_gap_weighted_sum" + suffix
        ]
        mean_gap = (
            weighted_sum / informative_rows
            if informative_rows else 0.0
        )
        if not math.isfinite(mean_gap):
            raise ValueError(
                "mean positive cutoff gap aggregate is non-finite"
            )
        result["mean_positive_cutoff_gap" + suffix] = float(mean_gap)
    result["informative_rows_total"] = (
        result["informative_rows025"] + result["informative_rows050"]
    )
    return result


def _require_calibration_result(value, label):
    if (not isinstance(value, dict)
            or set(value) != {"accumulator", "report"}
            or not isinstance(value["report"], dict)):
        raise ValueError("{} calibration result is invalid".format(label))
    return value


def _restore_snapshot(
        initialized, snapshot, *, restorer, eval_mode_setter,
        clear_optimizer=True):
    mcln, parent, geometry = _source_gate_models(initialized)
    restorer(mcln, parent, geometry, snapshot)
    _repair_source_gate_trainability(initialized)
    _cleanup_source_training_state(
        initialized,
        eval_mode_setter=eval_mode_setter,
        clear_optimizer=clear_optimizer,
    )


def run_source_gate_probe(
        initialized, *, calibration_fn=None, move_batch=None,
        input_builder=None, full_state_builder=None, target_attacher=None,
        loss_fn=None, train_mode_setter=None, eval_mode_setter=None,
        gradient_clipper=None, snapshotter=None, restorer=None,
        state_digest_builder=None, gate_evaluator=None,
        calibration_kwargs=None, live_contract_validator=None):
    """Fit the bounded probe, select earliest-safe state, and reproduce it."""
    mcln, parent, geometry = _source_gate_models(initialized)
    source_parameters = initialized.get("source_parameters")
    optimizer = initialized.get("source_optimizer")
    data = initialized.get("data")
    probe_steps = initialized.get("probe_steps")
    if (source_parameters is None or optimizer is None
            or not isinstance(data, dict) or "fit_loader" not in data
            or type(probe_steps) is not int or probe_steps not in (1, 306)):
        raise ValueError("initialized source-gate loop contract is invalid")
    if (not callable(getattr(optimizer, "zero_grad", None))
            or not callable(getattr(optimizer, "step", None))):
        raise ValueError("source-gate optimizer operations are invalid")

    calibration_fn = calibration_fn or calibrate_source_gate_probe
    move_batch = move_batch or legacy._move_batch_to_device
    input_builder = input_builder or legacy.build_rec_finetune_inputs
    full_state_builder = full_state_builder or build_full_rec_query_state
    target_attacher = (
        target_attacher or rec_source_gate.attach_full_query_targets
    )
    loss_fn = loss_fn or rec_source_gate.compute_rec_source_gate_loss
    train_mode_setter = (
        train_mode_setter or rec_source_gate.set_rec_source_gate_train_mode
    )
    eval_mode_setter = (
        eval_mode_setter or rec_source_gate.set_rec_source_gate_eval_mode
    )
    gradient_clipper = (
        gradient_clipper or rec_source_gate.clip_rec_source_gate_gradients
    )
    snapshotter = snapshotter or legacy.snapshot_rec_finetune_state
    restorer = restorer or legacy.restore_rec_finetune_state
    state_digest_builder = state_digest_builder or source_gate_state_digests
    gate_evaluator = gate_evaluator or evaluate_source_gate_gate
    live_contract_validator = (
        live_contract_validator or validate_live_source_gate_probe_contract
    )
    hooks = (
        calibration_fn, move_batch, input_builder, full_state_builder,
        target_attacher, loss_fn, train_mode_setter, eval_mode_setter,
        gradient_clipper, snapshotter, restorer, state_digest_builder,
        gate_evaluator, live_contract_validator,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("source-gate loop hooks must be callable")
    if calibration_kwargs is None:
        calibration_kwargs = {}
    if not isinstance(calibration_kwargs, dict):
        raise ValueError("calibration kwargs must be a mapping")
    live_contract_validator(initialized)

    runtime_device = torch.device(initialized["device"])
    step0_snapshot = None
    completed_steps = 0
    try:
        step0_snapshot = snapshotter(mcln, parent, geometry)
        initialized["_step0_snapshot"] = step0_snapshot
        step0_state_digests = state_digest_builder(mcln, parent, geometry)
        _validate_state_digest_bundle(step0_state_digests, "step-0")
        step0_calibration = _require_calibration_result(
            calibration_fn(
                initialized, baseline=None, **calibration_kwargs
            ),
            "step-0",
        )
        if state_digest_builder(mcln, parent, geometry) != step0_state_digests:
            raise RuntimeError("step-0 calibration mutated model state")
        step0_hit_bits_sha256 = (
            _source_gate_calibration_hit_bits_sha256(
                step0_calibration["accumulator"],
                step0_calibration["report"],
            )
        )

        aggregate = _new_training_diagnostics()
        fit_iterator = iter(data["fit_loader"])
        for _step in range(1, probe_steps + 1):
            try:
                batch = next(fit_iterator)
            except StopIteration:
                raise ValueError(
                    "fit loader exhausted before requested probe steps"
                )
            train_mode_setter(mcln, parent, geometry)
            moved_batch = move_batch(batch, runtime_device)
            if not isinstance(moved_batch, dict):
                raise ValueError("fit batch must be a mapping")
            inputs = input_builder(moved_batch)
            if not isinstance(inputs, dict):
                raise ValueError("fit inputs must be a mapping")
            legacy._reject_rec_target_only_fields(
                inputs, "source-gate fit inputs"
            )
            with torch.cuda.amp.autocast(enabled=False):
                end_points = mcln(inputs)
                if not isinstance(end_points, dict):
                    raise ValueError("MCLN output must be a mapping")
                legacy._reject_rec_target_only_fields(
                    inputs, "post-MCLN source-gate fit inputs"
                )
                legacy._reject_rec_target_only_fields(
                    end_points, "source-gate fit MCLN outputs"
                )
                full_state = full_state_builder(end_points, inputs)
                if not isinstance(full_state, dict):
                    raise ValueError("full source-gate fit state is invalid")
                query_ious = target_attacher(
                    full_state, moved_batch, root_only=True
                )
                default_scores = full_state.get("default_scores")
                if (not isinstance(default_scores, torch.Tensor)
                        or not isinstance(query_ious, torch.Tensor)
                        or query_ious.requires_grad
                        or default_scores.shape != query_ious.shape):
                    raise ValueError(
                        "source-gate fit scores or IoUs are invalid"
                    )
                valid = torch.ones_like(query_ious, dtype=torch.bool)
                loss, loss_stats = loss_fn(
                    default_scores, query_ious, valid
                )
                if (not isinstance(loss, torch.Tensor)
                        or loss.numel() != 1
                        or not loss.requires_grad
                        or not bool(torch.isfinite(loss.detach()).item())):
                    raise ValueError(
                        "source-gate loss is not finite and trainable"
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = gradient_clipper(source_parameters)
            if (not isinstance(gradients, dict)
                    or set(gradients)
                    != {"source_gate_semantic_classifier"}):
                raise ValueError("source-gate gradient diagnostics are invalid")
            gradient = gradients["source_gate_semantic_classifier"]
            optimizer.step()
            completed_steps += 1
            _update_training_diagnostics(
                aggregate, loss.detach(), loss_stats, gradient
            )
        diagnostics = _finalize_training_diagnostics(
            aggregate, completed_steps
        )

        final_snapshot = snapshotter(mcln, parent, geometry)
        final_state_digests = state_digest_builder(mcln, parent, geometry)
        _validate_state_digest_bundle(final_state_digests, "final")
        final_calibration = _require_calibration_result(
            calibration_fn(
                initialized,
                baseline=step0_calibration["accumulator"],
                **calibration_kwargs
            ),
            "final",
        )
        if state_digest_builder(mcln, parent, geometry) != final_state_digests:
            raise RuntimeError("final calibration mutated model state")
        final_hit_bits_sha256 = (
            _source_gate_calibration_hit_bits_sha256(
                final_calibration["accumulator"],
                final_calibration["report"],
                baseline_accumulator=step0_calibration["accumulator"],
            )
        )
        decision = gate_evaluator(
            step0_calibration["report"],
            final_calibration["report"],
            baseline_state_digests=step0_state_digests,
            final_state_digests=final_state_digests,
            training_diagnostics=diagnostics,
            final_step=probe_steps,
        )
        if (not isinstance(decision, dict)
                or type(decision.get("eligible")) is not bool
                or decision.get("selected_step")
                != (probe_steps if decision["eligible"] else 0)):
            raise ValueError("source-gate decision is invalid")

        if decision["eligible"]:
            selected_snapshot = final_snapshot
            selected_report = final_calibration["report"]
            selected_digests = final_state_digests
            selected_hit_bits_sha256 = final_hit_bits_sha256
            reproduction_baseline = step0_calibration["accumulator"]
        else:
            selected_snapshot = step0_snapshot
            selected_report = step0_calibration["report"]
            selected_digests = step0_state_digests
            selected_hit_bits_sha256 = step0_hit_bits_sha256
            reproduction_baseline = None
        _restore_snapshot(
            initialized,
            selected_snapshot,
            restorer=restorer,
            eval_mode_setter=eval_mode_setter,
        )
        restored_digests = state_digest_builder(mcln, parent, geometry)
        if restored_digests != selected_digests:
            raise RuntimeError("selected snapshot restore is not bitwise exact")
        live_contract_validator(initialized)
        reproduced = _require_calibration_result(
            calibration_fn(
                initialized,
                baseline=reproduction_baseline,
                **calibration_kwargs
            ),
            "reproduced",
        )
        if reproduced["report"] != selected_report:
            raise RuntimeError(
                "selected source-gate calibration did not reproduce exactly"
            )
        reproduced_hit_bits_sha256 = (
            _source_gate_calibration_hit_bits_sha256(
                reproduced["accumulator"],
                reproduced["report"],
                baseline_accumulator=reproduction_baseline,
            )
        )
        if reproduced_hit_bits_sha256 != selected_hit_bits_sha256:
            raise RuntimeError(
                "selected calibration hit bits did not reproduce exactly"
            )
        reproduced_digests = state_digest_builder(mcln, parent, geometry)
        if reproduced_digests != selected_digests:
            raise RuntimeError("reproduction mutated the restored model state")
        live_contract_validator(initialized)

        return {
            "schema": "rec-source-gate-probe-run-v1",
            "requested_steps": probe_steps,
            "completed_steps": completed_steps,
            "step0_report": copy.deepcopy(step0_calibration["report"]),
            "final_report": copy.deepcopy(final_calibration["report"]),
            "reproduced_report": copy.deepcopy(reproduced["report"]),
            "calibration_evidence": {
                "schema": CALIBRATION_EVIDENCE_SCHEMA,
                "canonical_format": CALIBRATION_BINDING_FORMAT,
                "step0": _calibration_evidence_record(
                    step0_hit_bits_sha256,
                    step0_hit_bits_sha256,
                    step0_calibration["report"],
                ),
                "final": _calibration_evidence_record(
                    step0_hit_bits_sha256,
                    final_hit_bits_sha256,
                    final_calibration["report"],
                ),
                "reproduced": _calibration_evidence_record(
                    step0_hit_bits_sha256,
                    reproduced_hit_bits_sha256,
                    reproduced["report"],
                ),
            },
            "decision": copy.deepcopy(decision),
            "training_diagnostics": copy.deepcopy(diagnostics),
            "state_digests": {
                "step0": copy.deepcopy(step0_state_digests),
                "final": copy.deepcopy(final_state_digests),
                "selected": copy.deepcopy(selected_digests),
                "restored": copy.deepcopy(restored_digests),
                "reproduced": copy.deepcopy(reproduced_digests),
            },
            "restore": {
                "target_step": decision["selected_step"],
                "target_digests": copy.deepcopy(selected_digests),
                "actual_digests": copy.deepcopy(restored_digests),
                "bitwise_verified": True,
            },
            "reproduction_matches": True,
            "legacy_joint_optimizer_updates": initialized.get(
                "legacy_joint_optimizer_updates"
            ),
            "_step0_snapshot": step0_snapshot,
        }
    except BaseException as error:
        if step0_snapshot is not None:
            try:
                _restore_snapshot(
                    initialized,
                    step0_snapshot,
                    restorer=restorer,
                    eval_mode_setter=eval_mode_setter,
                )
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "source-gate step-0 rollback failed"
                ) from cleanup_error
        raise


def _require_absolute_path(value, label):
    if (not isinstance(value, str) or not value
            or not Path(value).is_absolute()):
        raise ValueError("{} must be a non-empty absolute path".format(label))
    return value


def _validate_public_input_identity(value, label):
    fields = {"path", "device", "inode", "mode", "size", "sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("{} identity schema is invalid".format(label))
    _require_absolute_path(value["path"], label + " path")
    for name in ("device", "inode", "mode", "size"):
        _require_exact_int(value[name], label + " " + name)
    if (not stat.S_ISREG(value["mode"])
            or stat.S_IMODE(value["mode"]) & 0o222):
        raise ValueError("{} identity is not a read-only regular file".format(label))
    if not _is_sha256(value["sha256"]):
        raise ValueError("{} identity SHA-256 is invalid".format(label))
    return value


def _validate_receipt_inputs(value):
    names = (
        "backbone_checkpoint", "parent_reranker", "geometry_reranker",
    )
    if (not isinstance(value, dict)
            or set(value) != {"data_root"}.union(names)):
        raise ValueError("receipt input schema is invalid")
    _require_absolute_path(value["data_root"], "receipt data root")
    paths = []
    for name in names:
        pair = value[name]
        if (not isinstance(pair, dict)
                or set(pair) != {"before", "after"}):
            raise ValueError("receipt input identity pair is invalid")
        before = _validate_public_input_identity(pair["before"], name)
        after = _validate_public_input_identity(pair["after"], name)
        if before != after:
            raise ValueError("receipt input changed during the probe")
        paths.append(before["path"])
    if len(set(paths)) != len(paths):
        raise ValueError("receipt input paths must be distinct")


def _validate_receipt_code(value):
    if (not isinstance(value, dict)
            or set(value) != {"hashes", "manifest_sha256"}):
        raise ValueError("receipt code schema is invalid")
    hashes = _validate_manifest(value["hashes"])
    if not _REQUIRED_CODE_MANIFEST_ENTRIES.issubset(hashes):
        raise ValueError("receipt code manifest is incomplete")
    for record in hashes.values():
        _require_absolute_path(record["path"], "code manifest path")
        if not _is_sha256(record["sha256"]):
            raise ValueError("code manifest SHA-256 is invalid")
    if (not _is_sha256(value["manifest_sha256"])
            or value["manifest_sha256"] != _manifest_sha256(hashes)):
        raise ValueError("code manifest digest is invalid")


def _parse_utc(value, label):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("{} must be a UTC timestamp".format(label))
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("{} timestamp is invalid".format(label)) from error
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("{} must be a UTC timestamp".format(label))
    return parsed


def validate_source_gate_runtime(runtime):
    fields = {
        "schema", "started_utc", "finished_utc", "elapsed_seconds",
        "command", "interpreter", "versions", "device",
        "peak_cuda_memory", "environment",
    }
    if (not isinstance(runtime, dict) or set(runtime) != fields
            or runtime.get("schema") != "rec-source-gate-runtime-v1"):
        raise ValueError("source-gate runtime schema is invalid")
    started = _parse_utc(runtime["started_utc"], "runtime start")
    finished = _parse_utc(runtime["finished_utc"], "runtime finish")
    if finished < started:
        raise ValueError("runtime finish precedes start")
    elapsed = runtime["elapsed_seconds"]
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("runtime elapsed_seconds is invalid")
    command = runtime["command"]
    if (not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item for item in command)):
        raise ValueError("runtime command is invalid")
    interpreter = runtime["interpreter"]
    if (not isinstance(interpreter, dict)
            or set(interpreter) != {"logical_path", "resolved_path"}):
        raise ValueError("runtime interpreter identity is invalid")
    logical = _require_absolute_path(
        interpreter["logical_path"], "runtime interpreter logical path"
    )
    resolved = _require_absolute_path(
        interpreter["resolved_path"], "runtime interpreter resolved path"
    )
    if command[0] != logical or str(Path(logical).resolve()) != resolved:
        raise ValueError("runtime interpreter and command are incoherent")
    versions = runtime["versions"]
    if (not isinstance(versions, dict)
            or set(versions) != {"python", "torch", "cuda", "cudnn"}
            or any(not isinstance(versions[name], str) or not versions[name]
                   for name in ("python", "torch", "cuda"))
            or type(versions["cudnn"]) is not int
            or versions["cudnn"] <= 0):
        raise ValueError("runtime version identity is invalid")
    device = runtime["device"]
    if (not isinstance(device, dict)
            or set(device)
            != {"type", "index", "name", "total_memory_bytes"}
            or device.get("type") != "cuda"
            or type(device.get("index")) is not int
            or device["index"] != 0
            or not isinstance(device.get("name"), str)
            or not device["name"]
            or type(device.get("total_memory_bytes")) is not int
            or device["total_memory_bytes"] <= 0):
        raise ValueError("runtime CUDA device identity is invalid")
    peak = runtime["peak_cuda_memory"]
    if (not isinstance(peak, dict)
            or set(peak) != {"allocated_bytes", "reserved_bytes"}
            or any(type(peak[name]) is not int or peak[name] < 0
                   for name in peak)
            or peak["allocated_bytes"] > peak["reserved_bytes"]):
        raise ValueError("runtime peak CUDA memory is invalid")
    environment = runtime["environment"]
    if (not isinstance(environment, dict)
            or set(environment) != set(RUNTIME_ENVIRONMENT_ALLOWLIST)
            or any(item is not None and not isinstance(item, str)
                   for item in environment.values())):
        raise ValueError("runtime environment allowlist is invalid")
    return copy.deepcopy(runtime)


def _validate_trainability_contract(value):
    fields = {
        "mode", "allowed_prefix", "parameter_names", "parameter_count",
        "parameter_elements", "parameter_names_sha256",
    }
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("mode") != "final-semantic-classifier-only-v1"
            or value.get("allowed_prefix")
            != rec_source_gate.SOURCE_GATE_TRAINABLE_PREFIX):
        raise ValueError("receipt trainability contract is invalid")
    names = value["parameter_names"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str)
                   or not name.startswith(value["allowed_prefix"])
                   for name in names)
            or len(set(names)) != len(names)
            or type(value["parameter_count"]) is not int
            or value["parameter_count"] != len(names)
            or type(value["parameter_elements"]) is not int
            or value["parameter_elements"] <= 0
            or not _is_sha256(value["parameter_names_sha256"])
            or value["parameter_names_sha256"]
            != _manifest_sha256(names)):
        raise ValueError("receipt trainable parameter inventory is invalid")


def _validate_loss_contract(value):
    fields = {
        "name", "topk", "strict_gt", "thresholds",
        "threshold_weights", "margin", "temperature", "reduction",
    }
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("name") != "strict-top8-membership-v1"
            or type(value.get("topk")) is not int or value["topk"] != 8
            or value.get("strict_gt") is not True
            or value.get("reduction")
            != "per-threshold-informative-row-mean-sum"):
        raise ValueError("receipt loss contract is invalid")
    for name, expected in (
            ("thresholds", [0.25, 0.5]),
            ("threshold_weights", [2.0, 1.0])):
        values = value[name]
        if (not isinstance(values, list) or len(values) != len(expected)
                or any(type(item) is not float
                       or item != required
                       for item, required in zip(values, expected))):
            raise ValueError("receipt loss {} is invalid".format(name))
    if (type(value["margin"]) is not float or value["margin"] != 0.0
            or type(value["temperature"]) is not float
            or value["temperature"] != 1.0):
        raise ValueError("receipt loss scalar contract is invalid")


def _validate_optimizer_contract(value):
    fields = {
        "type", "group_count", "group", "gradient_clip_max_norm",
        "scheduler", "legacy_joint_optimizer_updates",
    }
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("type") != "AdamW"
            or type(value.get("group_count")) is not int
            or value["group_count"] != 1
            or value.get("scheduler") is not None
            or type(value.get("gradient_clip_max_norm")) is not float
            or value["gradient_clip_max_norm"] != 1.0
            or type(value.get("legacy_joint_optimizer_updates")) is not int
            or value["legacy_joint_optimizer_updates"] != 0):
        raise ValueError("receipt optimizer contract is invalid")
    group = value["group"]
    if (not isinstance(group, dict)
            or set(group) != {"name", "lr", "weight_decay"}
            or group.get("name") != "source_gate_semantic_classifier"
            or type(group.get("lr")) is not float
            or group["lr"] != 1e-4
            or type(group.get("weight_decay")) is not float
            or group["weight_decay"] != 1e-4):
        raise ValueError("receipt optimizer group is invalid")


def _validate_diagnostic_summary(value, expected_count, label):
    if (not isinstance(value, dict)
            or set(value) != {"count", "min", "max", "mean", "last"}
            or type(value["count"]) is not int
            or value["count"] != expected_count):
        raise ValueError("{} summary schema is invalid".format(label))
    numbers_value = []
    for name in ("min", "max", "mean", "last"):
        item = value[name]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError("{} summary is non-finite".format(label))
        numbers_value.append(item)
    minimum, maximum, mean, last = numbers_value
    if minimum > maximum or not minimum <= mean <= maximum \
            or not minimum <= last <= maximum:
        raise ValueError("{} summary bounds are incoherent".format(label))


def validate_source_gate_training_diagnostics(value, expected_updates):
    count_fields = {
        "informative_rows025", "informative_rows050",
        "active_violations025", "active_violations050",
        "no_positive_rows025", "no_positive_rows050",
        "too_few_negative_rows025", "too_few_negative_rows050",
        "positive_count025", "positive_count050",
        "informative_rows_total",
    }
    fields = {
        "schema", "update_count", "loss", "gradient_norm",
        "mean_positive_cutoff_gap025", "mean_positive_cutoff_gap050",
    }.union(count_fields)
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("schema")
            != "rec-source-gate-training-diagnostics-v1"
            or type(value.get("update_count")) is not int
            or value["update_count"] != expected_updates):
        raise ValueError("training diagnostics schema is invalid")
    _validate_diagnostic_summary(value["loss"], expected_updates, "loss")
    _validate_diagnostic_summary(
        value["gradient_norm"], expected_updates, "gradient norm"
    )
    for name in count_fields:
        _require_exact_int(value[name], "training " + name)
    for suffix in ("025", "050"):
        mean_gap = _require_exact_finite_float(
            value["mean_positive_cutoff_gap" + suffix],
            "training mean_positive_cutoff_gap" + suffix,
        )
        if value["informative_rows" + suffix] == 0 and mean_gap != 0.0:
            raise ValueError(
                "zero-row training cutoff gap must be canonical zero"
            )
    if (value["informative_rows_total"]
            != value["informative_rows025"]
            + value["informative_rows050"]
            or value["active_violations025"]
            > value["informative_rows025"]
            or value["active_violations050"]
            > value["informative_rows050"]):
        raise ValueError("training diagnostic counts are incoherent")
    return copy.deepcopy(value)


def validate_source_gate_calibration_evidence(
        value, *, step0_report, final_report, reproduced_report,
        selected_report_name):
    evidence_fields = {
        "schema", "canonical_format", "step0", "final", "reproduced",
    }
    if (not isinstance(value, dict) or set(value) != evidence_fields
            or value.get("schema") != CALIBRATION_EVIDENCE_SCHEMA
            or value.get("canonical_format") != CALIBRATION_BINDING_FORMAT
            or selected_report_name not in ("step0", "final")):
        raise ValueError("calibration binding evidence schema is invalid")
    reports = {
        "step0": step0_report,
        "final": final_report,
        "reproduced": reproduced_report,
    }
    records = {}
    for name in ("step0", "final", "reproduced"):
        record = value[name]
        if (not isinstance(record, dict)
                or set(record) != {
                    "hit_bits_sha256", "binding_sha256",
                }
                or not _is_sha256(record.get("hit_bits_sha256"))
                or not _is_sha256(record.get("binding_sha256"))):
            raise ValueError(
                "calibration binding evidence record is invalid"
            )
        records[name] = record
    step0_hit_bits_sha256 = records["step0"]["hit_bits_sha256"]
    for name in ("step0", "final", "reproduced"):
        expected_binding = _source_gate_calibration_binding_sha256(
            step0_hit_bits_sha256,
            records[name]["hit_bits_sha256"],
            reports[name],
        )
        if records[name]["binding_sha256"] != expected_binding:
            raise ValueError("calibration transition binding differs")
    if (records["reproduced"]["hit_bits_sha256"]
            != records[selected_report_name]["hit_bits_sha256"]):
        raise ValueError(
            "selected calibration hit-bit digest did not reproduce"
        )
    return copy.deepcopy(value)


def _validate_receipt_output_isolation(receipt):
    output = Path(receipt["output_dir"])
    inputs = receipt["inputs"]
    protected = [Path(inputs["data_root"])]
    for name in (
            "backbone_checkpoint", "parent_reranker", "geometry_reranker"):
        path = Path(inputs[name]["before"]["path"])
        protected.extend((path, path.parent))
    protected.extend(PROTECTED_LEGACY_PATHS)
    if any(_paths_overlap(output, path) for path in protected):
        raise ValueError("receipt output overlaps a protected input tree")


def _validate_receipt_runtime_command(receipt):
    command = receipt["runtime"]["command"]
    if (len(command) < 2
            or _comparison_path(command[1]) != Path(__file__).resolve()):
        raise ValueError("runtime command does not name this source-gate runner")
    arguments = command[2:]
    if len(arguments) % 2 != 0:
        raise ValueError("runtime command option/value pairs are incomplete")
    options = {}
    for offset in range(0, len(arguments), 2):
        option = arguments[offset]
        value = arguments[offset + 1]
        if (option not in _CLI_OPTIONS or option in options
                or not isinstance(value, str) or not value):
            raise ValueError("runtime command contains an invalid option")
        options[option] = value
    required = {
        "--data-root", "--backbone-checkpoint", "--parent-reranker",
        "--geometry-reranker", "--output-dir",
    }
    if not required.issubset(options):
        raise ValueError("runtime command omits a required source-gate option")
    if options.get("--device", PRODUCTION_DEVICE) != PRODUCTION_DEVICE:
        raise ValueError("runtime command device is not cuda:0")
    try:
        command_steps = _probe_steps(
            options.get("--probe-steps", str(PRODUCTION_PROBE_STEPS))
        )
    except argparse.ArgumentTypeError as error:
        raise ValueError("runtime command probe steps are invalid") from error
    if command_steps != receipt["probe"]["requested_steps"]:
        raise ValueError("runtime command probe steps differ from receipt")
    expected_paths = {
        "--data-root": receipt["inputs"]["data_root"],
        "--backbone-checkpoint": receipt["inputs"][
            "backbone_checkpoint"
        ]["before"]["path"],
        "--parent-reranker": receipt["inputs"]["parent_reranker"][
            "before"
        ]["path"],
        "--geometry-reranker": receipt["inputs"]["geometry_reranker"][
            "before"
        ]["path"],
        "--output-dir": receipt["output_dir"],
    }
    for option, expected in expected_paths.items():
        if _comparison_path(options[option]) != _comparison_path(expected):
            raise ValueError(
                "runtime command {} differs from receipt".format(option)
            )


def validate_source_gate_receipt(receipt):
    """Validate every receipt field and recompute all derived claims."""
    fields = {
        "schema", "version", "deployable", "checkpoint_written",
        "output_dir", "output_files", "validation_data_accessed",
        "validation_data_objects_present", "inputs", "code", "runtime",
        "data_contract", "trainability", "loss_contract",
        "optimizer_contract", "probe", "calibration",
        "calibration_evidence", "decision", "state", "restore",
    }
    if (not isinstance(receipt, dict) or set(receipt) != fields
            or receipt.get("schema")
            != "rec-source-gate-probe-receipt-v1"
            or type(receipt.get("version")) is not int
            or receipt["version"] != 1
            or receipt.get("deployable") is not False
            or receipt.get("checkpoint_written") is not False
            or receipt.get("output_files") != ["smoke-receipt.json"]
            or receipt.get("validation_data_accessed") is not False
            or receipt.get("validation_data_objects_present") is not False):
        raise ValueError("source-gate receipt top-level schema is invalid")
    _require_absolute_path(receipt["output_dir"], "receipt output directory")
    _validate_receipt_inputs(receipt["inputs"])
    _validate_receipt_output_isolation(receipt)
    _validate_receipt_code(receipt["code"])
    validate_source_gate_runtime(receipt["runtime"])
    _validate_train_data_contract(receipt["data_contract"])
    _validate_trainability_contract(receipt["trainability"])
    _validate_loss_contract(receipt["loss_contract"])
    _validate_optimizer_contract(receipt["optimizer_contract"])

    probe_record = receipt["probe"]
    probe_fields = {
        "requested_steps", "completed_steps", "seed", "batch_size",
        "dataset_split", "fit_sample_count", "calibration_sample_count",
        "training_diagnostics",
    }
    if (not isinstance(probe_record, dict)
            or set(probe_record) != probe_fields
            or type(probe_record.get("requested_steps")) is not int
            or probe_record["requested_steps"] not in (1, 306)
            or type(probe_record.get("completed_steps")) is not int
            or probe_record["completed_steps"]
            != probe_record["requested_steps"]
            or type(probe_record.get("seed")) is not int
            or probe_record["seed"] != 0
            or type(probe_record.get("batch_size")) is not int
            or probe_record["batch_size"] != PRODUCTION_BATCH_SIZE
            or probe_record.get("dataset_split") != "train"
            or type(probe_record.get("fit_sample_count")) is not int
            or probe_record["fit_sample_count"]
            != PRODUCTION_FIT_SAMPLE_COUNT
            or type(probe_record.get("calibration_sample_count")) is not int
            or probe_record["calibration_sample_count"]
            != PRODUCTION_CALIBRATION_SAMPLE_COUNT):
        raise ValueError("receipt probe contract is invalid")
    diagnostics = validate_source_gate_training_diagnostics(
        probe_record["training_diagnostics"],
        probe_record["completed_steps"],
    )
    _validate_receipt_runtime_command(receipt)

    calibration = receipt["calibration"]
    if (not isinstance(calibration, dict)
            or set(calibration) != {"step0", "final", "reproduced"}):
        raise ValueError("receipt calibration schema is invalid")
    step0 = validate_source_gate_calibration_report(
        calibration["step0"],
        expected_sample_count=PRODUCTION_CALIBRATION_SAMPLE_COUNT,
    )
    final = validate_source_gate_calibration_report(
        calibration["final"],
        baseline_report=step0,
        expected_sample_count=PRODUCTION_CALIBRATION_SAMPLE_COUNT,
    )

    state = receipt["state"]
    state_fields = {
        "canonical_format", "step0", "final", "selected", "restored",
        "reproduced",
    }
    if (not isinstance(state, dict) or set(state) != state_fields
            or state.get("canonical_format") != STATE_DIGEST_FORMAT):
        raise ValueError("receipt state digest schema is invalid")
    for name in state_fields - {"canonical_format"}:
        _validate_state_digest_bundle(state[name], name)
    recomputed_decision = evaluate_source_gate_gate(
        step0,
        final,
        baseline_state_digests=state["step0"],
        final_state_digests=state["final"],
        training_diagnostics=diagnostics,
        final_step=probe_record["requested_steps"],
    )
    if receipt["decision"] != recomputed_decision:
        raise ValueError("receipt decision does not recompute exactly")
    selected_report = final if recomputed_decision["eligible"] else step0
    selected_state = (
        state["final"] if recomputed_decision["eligible"] else state["step0"]
    )
    if (calibration["reproduced"] != selected_report
            or state["selected"] != selected_state
            or state["restored"] != selected_state
            or state["reproduced"] != selected_state):
        raise ValueError("receipt selected state or reproduction is inexact")
    validate_source_gate_calibration_evidence(
        receipt["calibration_evidence"],
        step0_report=step0,
        final_report=final,
        reproduced_report=calibration["reproduced"],
        selected_report_name=(
            "final" if recomputed_decision["eligible"] else "step0"
        ),
    )

    restore = receipt["restore"]
    restore_fields = {
        "target_step", "target_digests", "actual_digests",
        "bitwise_verified", "reproduction_matches",
    }
    if (not isinstance(restore, dict) or set(restore) != restore_fields
            or type(restore.get("target_step")) is not int
            or restore["target_step"] != recomputed_decision["selected_step"]
            or restore.get("target_digests") != selected_state
            or restore.get("actual_digests") != selected_state
            or restore.get("bitwise_verified") is not True
            or restore.get("reproduction_matches") is not True):
        raise ValueError("receipt restore proof is invalid")
    _validate_state_digest_bundle(restore["target_digests"], "restore target")
    _validate_state_digest_bundle(restore["actual_digests"], "restore actual")
    try:
        _canonical_json_bytes(receipt)
    except ValueError as error:
        raise ValueError("receipt is not public canonical JSON") from error
    return copy.deepcopy(receipt)


def _require_unchanged_inputs(initialized, identity_reader):
    before = initialized.get("input_identities_before")
    paths = initialized.get("paths")
    expected_names = {
        "backbone_checkpoint", "parent_reranker", "geometry_reranker",
    }
    if (not isinstance(before, dict) or set(before) != expected_names
            or paths is None):
        raise ValueError("initial immutable input identities are unavailable")
    after = _capture_input_identities(paths, identity_reader)
    if before != after:
        raise RuntimeError("an immutable source-gate input changed")
    return copy.deepcopy(before), after


def _require_unchanged_manifest(initialized, manifest_builder):
    initial = initialized.get("publication_code_hashes")
    if not isinstance(initial, dict):
        raise ValueError("initial source-gate code manifest is unavailable")
    current = _validate_manifest(manifest_builder())
    if initial != current:
        raise RuntimeError("source-gate code changed during the probe")
    return copy.deepcopy(current)


def _require_live_runtime_identity(runtime, runtime_snapshot_reader):
    if not callable(runtime_snapshot_reader):
        raise ValueError("runtime snapshot reader must be callable")
    snapshot = runtime_snapshot_reader()
    stable_fields = {
        "interpreter", "versions", "device", "peak_cuda_memory",
        "environment",
    }
    if (not isinstance(snapshot, dict)
            or not stable_fields.issubset(snapshot)
            or any(snapshot[name] != runtime[name] for name in stable_fields)):
        raise RuntimeError("receipt runtime identity differs from live runtime")


def _build_trainability_receipt(initialized):
    contract = validate_live_source_gate_contract(initialized)
    names = getattr(contract, "names", None)
    parameters = getattr(contract, "parameters", None)
    if (type(names) is not tuple or type(parameters) is not tuple
            or not names or len(names) != len(parameters)
            or any(not isinstance(name, str) for name in names)
            or any(not isinstance(parameter, torch.nn.Parameter)
                   for parameter in parameters)):
        raise ValueError("live source-gate parameter contract is invalid")
    public_names = list(names)
    return {
        "mode": "final-semantic-classifier-only-v1",
        "allowed_prefix": rec_source_gate.SOURCE_GATE_TRAINABLE_PREFIX,
        "parameter_names": public_names,
        "parameter_count": len(public_names),
        "parameter_elements": sum(
            int(parameter.numel()) for parameter in parameters
        ),
        "parameter_names_sha256": _manifest_sha256(public_names),
    }


def _build_optimizer_receipt(initialized):
    optimizer = initialized.get("source_optimizer")
    contract = initialized.get("source_parameters")
    if (type(optimizer) is not torch.optim.AdamW
            or len(optimizer.param_groups) != 1):
        raise ValueError("live source-gate optimizer is not exact AdamW")
    group = optimizer.param_groups[0]
    expected_parameters = tuple(getattr(contract, "parameters", ()))
    if (group.get("name") != "source_gate_semantic_classifier"
            or tuple(group.get("params", ())) != expected_parameters
            or type(group.get("lr")) is not float
            or group["lr"] != 1e-4
            or type(group.get("weight_decay")) is not float
            or group["weight_decay"] != 1e-4):
        raise ValueError("live source-gate optimizer group is inexact")
    return {
        "type": "AdamW",
        "group_count": 1,
        "group": {
            "name": "source_gate_semantic_classifier",
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "gradient_clip_max_norm": 1.0,
        "scheduler": None,
        "legacy_joint_optimizer_updates": initialized.get(
            "legacy_joint_optimizer_updates"
        ),
    }


def build_source_gate_receipt(
        initialized, run_result, runtime, *, manifest_builder=None,
        identity_reader=None, runtime_snapshot_reader=None):
    """Build and self-validate the sole nondeployable probe receipt."""
    if not isinstance(initialized, dict) or not isinstance(run_result, dict):
        raise ValueError("receipt inputs must be mappings")
    manifest_builder = manifest_builder or build_source_gate_code_manifest
    identity_reader = identity_reader or read_source_gate_input_identity
    runtime_snapshot_reader = (
        runtime_snapshot_reader or capture_source_gate_runtime_snapshot
    )
    if (not callable(manifest_builder) or not callable(identity_reader)
            or not callable(runtime_snapshot_reader)):
        raise ValueError("receipt provenance readers must be callable")
    code_hashes = _require_unchanged_manifest(initialized, manifest_builder)
    identities_before, identities_after = _require_unchanged_inputs(
        initialized, identity_reader
    )
    paths = initialized.get("paths")
    if paths is None:
        raise ValueError("receipt runtime paths are unavailable")
    validate_source_gate_runtime(runtime)
    _require_live_runtime_identity(runtime, runtime_snapshot_reader)
    contract = validate_live_source_gate_data_contract(initialized)

    required_run = {
        "requested_steps", "completed_steps", "step0_report",
        "final_report", "reproduced_report", "decision",
        "training_diagnostics", "calibration_evidence",
        "state_digests", "restore",
        "reproduction_matches", "legacy_joint_optimizer_updates",
    }
    if not required_run.issubset(run_result):
        raise ValueError("source-gate run result is incomplete")
    state_digests = run_result["state_digests"]
    if (not isinstance(state_digests, dict)
            or set(state_digests)
            != {"step0", "final", "selected", "restored", "reproduced"}):
        raise ValueError("source-gate run state digests are incomplete")
    restore = run_result["restore"]
    if not isinstance(restore, dict):
        raise ValueError("source-gate restore result is invalid")

    receipt = {
        "schema": "rec-source-gate-probe-receipt-v1",
        "version": 1,
        "deployable": False,
        "checkpoint_written": False,
        "output_dir": str(Path(paths.output_dir).resolve()),
        "output_files": ["smoke-receipt.json"],
        "validation_data_accessed": False,
        "validation_data_objects_present": False,
        "inputs": {
            "data_root": str(Path(paths.data_root).resolve()),
            **{
                name: {
                    "before": copy.deepcopy(identities_before[name]),
                    "after": copy.deepcopy(identities_after[name]),
                }
                for name in (
                    "backbone_checkpoint", "parent_reranker",
                    "geometry_reranker",
                )
            },
        },
        "code": {
            "hashes": code_hashes,
            "manifest_sha256": _manifest_sha256(code_hashes),
        },
        "runtime": copy.deepcopy(runtime),
        "data_contract": contract,
        "trainability": _build_trainability_receipt(initialized),
        "loss_contract": {
            "name": "strict-top8-membership-v1",
            "topk": 8,
            "strict_gt": True,
            "thresholds": [0.25, 0.5],
            "threshold_weights": [2.0, 1.0],
            "margin": 0.0,
            "temperature": 1.0,
            "reduction": "per-threshold-informative-row-mean-sum",
        },
        "optimizer_contract": _build_optimizer_receipt(initialized),
        "probe": {
            "requested_steps": run_result["requested_steps"],
            "completed_steps": run_result["completed_steps"],
            "seed": initialized.get("seed"),
            "batch_size": contract["batch_size"],
            "dataset_split": contract["dataset_split"],
            "fit_sample_count": contract["fit_sample_count"],
            "calibration_sample_count": contract[
                "calibration_sample_count"
            ],
            "training_diagnostics": copy.deepcopy(
                run_result["training_diagnostics"]
            ),
        },
        "calibration": {
            "step0": copy.deepcopy(run_result["step0_report"]),
            "final": copy.deepcopy(run_result["final_report"]),
            "reproduced": copy.deepcopy(run_result["reproduced_report"]),
        },
        "calibration_evidence": copy.deepcopy(
            run_result["calibration_evidence"]
        ),
        "decision": copy.deepcopy(run_result["decision"]),
        "state": {
            "canonical_format": STATE_DIGEST_FORMAT,
            **copy.deepcopy(state_digests),
        },
        "restore": {
            "target_step": restore.get("target_step"),
            "target_digests": copy.deepcopy(
                restore.get("target_digests")
            ),
            "actual_digests": copy.deepcopy(
                restore.get("actual_digests")
            ),
            "bitwise_verified": restore.get("bitwise_verified"),
            "reproduction_matches": run_result.get(
                "reproduction_matches"
            ),
        },
    }
    return validate_source_gate_receipt(receipt)


def _reject_duplicate_json_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key: {}".format(name))
        result[name] = value
    return result


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def load_strict_source_gate_receipt(path):
    """Strict-load canonical receipt bytes and rerun the exact validator."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("receipt must be a regular non-symlink file")
    try:
        encoded = path.read_bytes()
        text = encoded.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("receipt JSON reload failed: {}".format(error))
    if _canonical_json_bytes(value) != encoded:
        raise ValueError("receipt bytes are not exact canonical ASCII JSON")
    return validate_source_gate_receipt(value)


def _write_atomic_canonical_json(path, payload):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("receipt path already exists: {}".format(path))
    encoded = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        legacy._fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _rename_directory_noreplace(source, destination):
    return legacy._rename_directory_noreplace(source, destination)


def _publication_parent_contract(paths, output_dir):
    try:
        parent = Path(paths.output_parent)
        device = paths.output_parent_device
        inode = paths.output_parent_inode
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "initial publication output parent identity is unavailable"
        ) from error
    if (not parent.is_absolute() or parent != output_dir.parent
            or type(device) is not int or device < 0
            or type(inode) is not int or inode <= 0):
        raise ValueError(
            "initial publication output parent identity is invalid"
        )
    return parent, (device, inode)


def _logical_directory_identity(path, label):
    _reject_symlink_ancestors(path, label)
    try:
        metadata = os.stat(str(path), follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(
            "{} is unavailable: {}".format(label, error)
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("{} is not a real directory".format(label))
    _reject_symlink_ancestors(path, label)
    return int(metadata.st_dev), int(metadata.st_ino)


def _require_logical_parent_identity(parent, expected_identity):
    actual = _logical_directory_identity(
        parent, "publication output parent"
    )
    if actual != expected_identity:
        raise RuntimeError("publication output parent identity changed")


def _open_bound_publication_parent(parent, expected_identity):
    flags = _nofollow_directory_open_flags()
    _require_logical_parent_identity(parent, expected_identity)
    try:
        descriptor = os.open(str(parent), flags)
    except OSError as error:
        raise RuntimeError(
            "could not bind publication output parent: {}".format(error)
        )
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISDIR(metadata.st_mode)
                or (int(metadata.st_dev), int(metadata.st_ino))
                != expected_identity):
            raise RuntimeError("publication output parent identity changed")
        _require_logical_parent_identity(parent, expected_identity)
        proc_path = Path("/proc/self/fd") / str(descriptor)
        if not proc_path.is_dir():
            raise RuntimeError("bound publication parent is inaccessible")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _nofollow_directory_open_flags():
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Linux no-follow directory descriptors are required")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _require_child_name(name, label):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or os.path.basename(name) != name
            or os.sep in name
            or (os.altsep is not None and os.altsep in name)):
        raise ValueError("{} must be a single child name".format(label))
    return name


def _bound_child_path(parent_fd, name):
    name = _require_child_name(name, "bound publication child")
    return Path("/proc/self/fd") / str(parent_fd) / name


def _bound_child_metadata(parent_fd, name):
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_bound_directory_identity(
        parent_fd, name, expected_identity, label):
    metadata = _bound_child_metadata(parent_fd, name)
    if (metadata is None or not stat.S_ISDIR(metadata.st_mode)
            or (int(metadata.st_dev), int(metadata.st_ino))
            != expected_identity):
        raise RuntimeError("{} identity changed".format(label))


def _create_unique_staging_directory(parent_fd, output_name):
    output_name = _require_child_name(output_name, "publication output")
    prefix = ".{}.staging-".format(output_name)
    for _attempt in range(128):
        name = prefix + secrets.token_hex(12)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        identity = None
        try:
            metadata = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("created staging entry is not a directory")
            identity = (
                int(metadata.st_dev), int(metadata.st_ino),
            )
            os.fsync(parent_fd)
            return name, identity
        except BaseException as error:
            try:
                if identity is None:
                    metadata = os.stat(
                        name, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise RuntimeError(
                            "created staging entry identity is unavailable"
                        )
                    identity = (
                        int(metadata.st_dev), int(metadata.st_ino),
                    )
                if not _remove_bound_directory_if_owned(
                        parent_fd, name, identity):
                    raise RuntimeError(
                        "created staging directory ownership changed"
                    )
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "could not clean failed staging directory creation"
                ) from cleanup_error
            raise
    raise FileExistsError("could not allocate a unique staging directory")


def _metadata_identity(metadata):
    return int(metadata.st_dev), int(metadata.st_ino)


def _open_bound_child_directory(parent_fd, name, expected_identity):
    flags = _nofollow_directory_open_flags()
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            return None
        raise
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISDIR(metadata.st_mode)
                or _metadata_identity(metadata) != expected_identity):
            os.close(descriptor)
            return None
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_open_directory_still_named(
        parent_fd, name, directory_fd, expected_identity):
    opened_metadata = os.fstat(directory_fd)
    named_metadata = _bound_child_metadata(parent_fd, name)
    if (not stat.S_ISDIR(opened_metadata.st_mode)
            or _metadata_identity(opened_metadata) != expected_identity
            or named_metadata is None
            or not stat.S_ISDIR(named_metadata.st_mode)
            or _metadata_identity(named_metadata) != expected_identity):
        raise RuntimeError("owned publication directory identity changed")


def _unlink_at_retry(directory_fd, name):
    last_error = None
    for _attempt in range(_FILESYSTEM_EINTR_ATTEMPTS):
        try:
            os.unlink(name, dir_fd=directory_fd)
            return
        except InterruptedError as error:
            last_error = error
    raise last_error


def _rmdir_at_retry(directory_fd, name):
    last_error = None
    for _attempt in range(_FILESYSTEM_EINTR_ATTEMPTS):
        try:
            os.rmdir(name, dir_fd=directory_fd)
            return
        except InterruptedError as error:
            last_error = error
    raise last_error


def _clear_bound_directory_contents(directory_fd):
    for name in os.listdir(directory_fd):
        _require_child_name(name, "owned publication entry")
        try:
            metadata = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            identity = _metadata_identity(metadata)
            child_fd = _open_bound_child_directory(
                directory_fd, name, identity
            )
            if child_fd is None:
                raise RuntimeError(
                    "owned publication child identity changed"
                )
            try:
                _clear_bound_directory_contents(child_fd)
                _require_open_directory_still_named(
                    directory_fd, name, child_fd, identity
                )
                _rmdir_at_retry(directory_fd, name)
            finally:
                os.close(child_fd)
        else:
            _unlink_at_retry(directory_fd, name)


def _remove_bound_directory_if_owned(
        parent_fd, name, expected_identity):
    directory_fd = _open_bound_child_directory(
        parent_fd, name, expected_identity
    )
    if directory_fd is None:
        return False
    try:
        _clear_bound_directory_contents(directory_fd)
        _require_open_directory_still_named(
            parent_fd, name, directory_fd, expected_identity
        )
        _rmdir_at_retry(parent_fd, name)
        return True
    finally:
        os.close(directory_fd)


def _rename_directory_noreplace_at(
        parent_fd, source_name, destination_name):
    """Atomically rename two children of one bound parent directory."""
    if type(parent_fd) is not int or parent_fd < 0:
        raise ValueError("publication parent descriptor is invalid")
    source_name = _require_child_name(source_name, "staging directory")
    destination_name = _require_child_name(
        destination_name, "output directory"
    )
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic dirfd RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            "final output directory already exists: {}".format(
                destination_name
            )
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("atomic dirfd RENAME_NOREPLACE is unavailable")
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


def _strict_directory_file(path, expected_receipt, expected_sha256):
    directory = path.parent
    entries = list(directory.iterdir())
    if (len(entries) != 1 or entries[0].name != "smoke-receipt.json"
            or entries[0].is_symlink()):
        raise RuntimeError("receipt directory contains unexpected files")
    if stat.S_IMODE(os.stat(str(path), follow_symlinks=False).st_mode) != 0o444:
        raise RuntimeError("published receipt is not mode 0444")
    loaded = load_strict_source_gate_receipt(path)
    if (loaded != expected_receipt
            or rec_finetune.sha256_file(path) != expected_sha256
            or path.read_bytes() != _canonical_json_bytes(expected_receipt)):
        raise RuntimeError("published receipt failed final parity checks")


def _revalidate_publication_output_path(
        output_dir, expected_resolved, *, allow_exists):
    output_dir = Path(output_dir)
    _reject_symlink_ancestors(output_dir, "publication output directory")
    if str(output_dir.resolve(strict=False)) != expected_resolved:
        raise ValueError("publication output path identity changed")
    parent = output_dir.parent
    _reject_symlink_ancestors(parent, "publication output parent")
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("publication output parent must be a real directory")
    exists = output_dir.exists() or output_dir.is_symlink()
    if exists is not allow_exists:
        if exists:
            raise FileExistsError("final output directory already exists")
        raise RuntimeError("committed output directory is missing")


def publish_source_gate_receipt(
        initialized, receipt, *, writer=None, failure_injector=None,
        manifest_builder=None, identity_reader=None, rename_fn=None,
        runtime_snapshot_reader=None):
    """Atomically publish only the immutable nondeployable receipt."""
    if not isinstance(initialized, dict):
        raise ValueError("publication initialized state is invalid")
    validated = validate_source_gate_receipt(receipt)
    writer = writer or _write_atomic_canonical_json
    failure_injector = failure_injector or (lambda _stage: None)
    manifest_builder = manifest_builder or build_source_gate_code_manifest
    identity_reader = identity_reader or read_source_gate_input_identity
    rename_fn = rename_fn or _rename_directory_noreplace_at
    runtime_snapshot_reader = (
        runtime_snapshot_reader or capture_source_gate_runtime_snapshot
    )
    hooks = (
        writer, failure_injector, manifest_builder, identity_reader, rename_fn,
        runtime_snapshot_reader,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("receipt publication hooks must be callable")
    paths = initialized.get("paths")
    if paths is None:
        raise ValueError("publication runtime paths are unavailable")
    output_dir = Path(paths.output_dir)
    _revalidate_publication_output_path(
        output_dir, validated["output_dir"], allow_exists=False
    )
    output_parent, expected_parent_identity = (
        _publication_parent_contract(paths, output_dir)
    )
    output_name = _require_child_name(
        output_dir.name, "publication output"
    )
    expected_sha256 = hashlib.sha256(
        _canonical_json_bytes(validated)
    ).hexdigest()
    parent_fd = _open_bound_publication_parent(
        output_parent, expected_parent_identity
    )
    staging_name = None
    staging_identity = None
    succeeded = False
    try:
        initial_manifest = _require_unchanged_manifest(
            initialized, manifest_builder
        )
        _require_live_runtime_identity(
            validated["runtime"], runtime_snapshot_reader
        )
        identities_before, identities_after = _require_unchanged_inputs(
            initialized, identity_reader
        )
        if (validated["code"]["hashes"] != initial_manifest
                or any(validated["inputs"][name]["before"]
                       != identities_before[name]
                       or validated["inputs"][name]["after"]
                       != identities_after[name]
                       for name in identities_before)):
            raise ValueError("receipt provenance differs from live inputs")
        _revalidate_publication_output_path(
            output_dir, validated["output_dir"], allow_exists=False
        )
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        staging_name, staging_identity = _create_unique_staging_directory(
            parent_fd, output_name
        )
        staging = _bound_child_path(parent_fd, staging_name)
        staged_receipt = staging / "smoke-receipt.json"
        bound_output = _bound_child_path(parent_fd, output_name)
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        _require_bound_directory_identity(
            parent_fd, staging_name, staging_identity,
            "receipt staging directory",
        )
        writer(staged_receipt, validated)
        failure_injector("written")
        _revalidate_publication_output_path(
            output_dir, validated["output_dir"], allow_exists=False
        )
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        _require_bound_directory_identity(
            parent_fd, staging_name, staging_identity,
            "receipt staging directory",
        )
        if ({entry.name for entry in staging.iterdir()}
                != {"smoke-receipt.json"}):
            raise RuntimeError("staging directory contains unexpected files")
        reloaded = load_strict_source_gate_receipt(staged_receipt)
        if (reloaded != validated
                or rec_finetune.sha256_file(staged_receipt)
                != expected_sha256):
            raise RuntimeError("staged receipt differs from validated payload")
        failure_injector("validated")
        _revalidate_publication_output_path(
            output_dir, validated["output_dir"], allow_exists=False
        )
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        _require_bound_directory_identity(
            parent_fd, staging_name, staging_identity,
            "receipt staging directory",
        )
        staged_receipt.chmod(0o444)
        legacy._fsync_file(staged_receipt)
        legacy._fsync_directory(staging)
        _strict_directory_file(
            staged_receipt, validated, expected_sha256
        )
        _require_unchanged_manifest(initialized, manifest_builder)
        _require_unchanged_inputs(initialized, identity_reader)
        _require_live_runtime_identity(
            validated["runtime"], runtime_snapshot_reader
        )
        failure_injector("finalize")
        _revalidate_publication_output_path(
            output_dir, validated["output_dir"], allow_exists=False
        )
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        _require_bound_directory_identity(
            parent_fd, staging_name, staging_identity,
            "receipt staging directory",
        )
        rename_fn(parent_fd, staging_name, output_name)
        os.fsync(parent_fd)
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        failure_injector("committed")
        _revalidate_publication_output_path(
            output_dir, validated["output_dir"], allow_exists=True
        )
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        _require_bound_directory_identity(
            parent_fd, output_name, staging_identity,
            "committed receipt directory",
        )
        bound_final_path = bound_output / "smoke-receipt.json"
        _strict_directory_file(
            bound_final_path, validated, expected_sha256
        )
        _require_unchanged_manifest(initialized, manifest_builder)
        _require_unchanged_inputs(initialized, identity_reader)
        _require_live_runtime_identity(
            validated["runtime"], runtime_snapshot_reader
        )
        _require_logical_parent_identity(
            output_parent, expected_parent_identity
        )
        succeeded = True
        return {
            "output_dir": output_dir,
            "path": output_dir / "smoke-receipt.json",
            "sha256": expected_sha256,
            "receipt": copy.deepcopy(validated),
        }
    finally:
        try:
            if not succeeded and staging_identity is not None:
                removed_output = _remove_bound_directory_if_owned(
                    parent_fd, output_name, staging_identity
                )
                removed_staging = _remove_bound_directory_if_owned(
                    parent_fd, staging_name, staging_identity
                )
                if removed_output or removed_staging:
                    os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def capture_source_gate_runtime_snapshot():
    if not torch.cuda.is_available():
        raise RuntimeError("production source-gate probe requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    cudnn = torch.backends.cudnn.version()
    if torch.version.cuda is None or cudnn is None:
        raise RuntimeError("production CUDA runtime versions are unavailable")
    return {
        "interpreter": {
            "logical_path": str(sys.executable),
            "resolved_path": str(Path(sys.executable).resolve(strict=True)),
        },
        "versions": {
            "python": str(platform.python_version()),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": int(cudnn),
        },
        "device": {
            "type": "cuda",
            "index": 0,
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
        },
        "peak_cuda_memory": {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        },
        "environment": {
            name: os.environ.get(name)
            for name in RUNTIME_ENVIRONMENT_ALLOWLIST
        },
    }


def build_source_gate_runtime_provenance(
        *, started_utc, finished_utc, elapsed_seconds, command,
        snapshot=None, snapshot_builder=None):
    snapshot_builder = snapshot_builder or capture_source_gate_runtime_snapshot
    if snapshot is None:
        if not callable(snapshot_builder):
            raise ValueError("runtime snapshot builder must be callable")
        snapshot = snapshot_builder()
    if not isinstance(snapshot, dict) or set(snapshot) != {
            "interpreter", "versions", "device", "peak_cuda_memory",
            "environment"}:
        raise ValueError("source-gate runtime snapshot is invalid")
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(item, str) for item in command)):
        raise ValueError("source-gate runtime command is invalid")
    runtime = {
        "schema": "rec-source-gate-runtime-v1",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "command": list(command),
        **copy.deepcopy(snapshot),
    }
    return validate_source_gate_runtime(runtime)


def _runtime_utc_now():
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _runtime_monotonic():
    return time.monotonic()


def _runtime_command(argv):
    if argv is None:
        arguments = [str(item) for item in sys.argv]
    else:
        arguments = [str(Path(__file__).resolve())] + [
            str(item) for item in argv
        ]
    return [str(sys.executable)] + arguments


def _set_production_determinism():
    legacy._set_production_determinism()


def _reset_peak_cuda_memory():
    legacy._reset_cuda_peak_memory_stats()


def main(
        argv=None, *, utc_now=None, monotonic=None,
        determinism_setter=None, peak_memory_resetter=None,
        initializer=None, runner=None, runtime_builder=None,
        receipt_builder=None, publisher=None):
    """Run the probe and publish its sole nondeployable smoke receipt."""
    args = parse_args(argv)
    utc_now = utc_now or _runtime_utc_now
    monotonic = monotonic or _runtime_monotonic
    determinism_setter = determinism_setter or _set_production_determinism
    peak_memory_resetter = (
        peak_memory_resetter or _reset_peak_cuda_memory
    )
    initializer = initializer or initialize_source_gate_probe
    runner = runner or run_source_gate_probe
    runtime_builder = (
        runtime_builder or build_source_gate_runtime_provenance
    )
    receipt_builder = receipt_builder or build_source_gate_receipt
    publisher = publisher or publish_source_gate_receipt
    hooks = (
        utc_now, monotonic, determinism_setter, peak_memory_resetter,
        initializer, runner, runtime_builder, receipt_builder, publisher,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("source-gate main hooks must be callable")

    command = _runtime_command(argv)
    started_utc = utc_now()
    started_monotonic = monotonic()
    determinism_setter()
    peak_memory_resetter()
    initialized = None
    run_result = None
    try:
        initialized = initializer(args)
        run_result = runner(initialized)
        if not isinstance(run_result, dict):
            raise ValueError("source-gate runner result must be a mapping")
        finished_utc = utc_now()
        finished_monotonic = monotonic()
        elapsed = finished_monotonic - started_monotonic
        runtime = runtime_builder(
            started_utc=started_utc,
            finished_utc=finished_utc,
            elapsed_seconds=elapsed,
            command=command,
        )
        receipt = receipt_builder(initialized, run_result, runtime)
        return publisher(initialized, receipt)
    except BaseException:
        if (isinstance(initialized, dict)
                and isinstance(run_result, dict)
                and run_result.get("_step0_snapshot") is not None):
            _restore_snapshot(
                initialized,
                run_result["_step0_snapshot"],
                restorer=legacy.restore_rec_finetune_state,
                eval_mode_setter=rec_source_gate.set_rec_source_gate_eval_mode,
            )
        raise


if __name__ == "__main__":
    main()
