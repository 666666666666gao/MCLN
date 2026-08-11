#!/usr/bin/env python
"""Launch and seal the one frozen official REC geometry evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from datetime import datetime

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from scripts.evaluate_frozen_rec_geometry_sidecar import (
    preflight_frozen_inputs,
    validate_frozen_record,
)


OFFICIAL_SAMPLE_COUNT = 9508
OFFICIAL_DATASET = "scanrefer"
OFFICIAL_EXPERIMENT = "epoch71_geometry_official"
OFFICIAL_DATA_ROOT = Path("/root/autodl-tmp/DATA_ROOT")
OFFICIAL_PYTHON_EXECUTABLE = Path(
    "/root/miniconda3/envs/bdetr/bin/python"
)
OFFICIAL_PYTHON_LINK_TARGET = "python3.7"
OFFICIAL_PYTHON_TARGET_SHA256 = (
    "8fca22177830bb3165a16e3137be525f8073921a7ba686ab91e0affb40fea4f6"
)
OFFICIAL_PYTHON_TARGET_SIZE = 12830744
OFFICIAL_PYTHON_TARGET_MODE = 0o755
OFFICIAL_MASTER_PORT = 29671
OFFICIAL_RESULT_SCHEMA = "rec-geometry-official-validation-result"
OFFICIAL_RESULT_VERSION = 1
COMPARISON_RESULT_SCHEMA = "rec-geometry-official-sidecar-comparison"
COMPARISON_RESULT_VERSION = 1
OFFICIAL_GATE_HITS025 = 5705
OFFICIAL_GATE_HITS050 = 4469
OFFICIAL_CLAIM_REGISTRY = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_artifacts"
)
OFFICIAL_GOAL_NAME = "epoch71_rec_geometry_official_validation_once"
OFFICIAL_SELECTION_RECORD_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_artifacts/selection.json"
)
OFFICIAL_SELECTED_ARTIFACT_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_artifacts/selected_geometry_reranker.pth"
)
OFFICIAL_PARENT_ARTIFACT_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth"
)
OFFICIAL_CHECKPOINT_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/"
    "mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
)
OFFICIAL_RUN_ROOT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_official_val"
)
OFFICIAL_CODE_ROOT = Path(os.path.abspath(__file__)).parents[1]
OFFICIAL_SIDECAR_RECORD_PATH = OFFICIAL_RUN_ROOT / (
    "geometry_val_sidecar_once.json"
)
OFFICIAL_SELECTION_RECORD_SHA256 = (
    "b1edc6cf01886929b005832c609a9d7e736e3585e3a9b5dd9421b50a5c55d1a4"
)
OFFICIAL_SELECTED_ARTIFACT_SHA256 = (
    "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f"
)
OFFICIAL_PARENT_ARTIFACT_SHA256 = (
    "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
OFFICIAL_SIDECAR_RECORD_SHA256 = (
    "c46bcd1c7ca3f67de1ba10555512c05672a67415ca66d3edb4d0b29e4aba6168"
)
OFFICIAL_RESULT_FIELDS = (
    "schema",
    "version",
    "created_at_utc",
    "sample_count",
    "printed_acc025",
    "printed_acc050",
    "hits025",
    "hits050",
    "gate025_pass",
    "gate050_pass",
    "acceptance_gate_pass",
    "inference_uses_ground_truth",
    "run",
    "files",
    "artifacts",
    "code",
    "launch",
)
COMPARISON_RESULT_FIELDS = (
    "schema",
    "version",
    "created_at_utc",
    "sample_count",
    "official_hits025",
    "official_hits050",
    "sidecar_hits025",
    "sidecar_hits050",
    "delta025",
    "delta050",
    "hits025_match",
    "hits050_match",
    "hit_counts_match",
    "artifact_provenance_match",
    "official_acceptance_gate_pass",
    "selection_uses_validation",
    "inference_uses_ground_truth",
    "acceptance",
    "files",
    "artifacts",
)
_ACCURACY_TOKEN = re.compile(r"^(?:0|1)\.[0-9]{5}$")
_METRIC_PATTERN = re.compile(
    r"last_ position alignment Acc0\.(25|50): Top-1: "
    r"((?:0|1)\.[0-9]{5})(?=,|\r?$)",
    re.MULTILINE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CRITICAL_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
    "PYTHONHOME",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
)


def recover_exact_hits(token, sample_count=OFFICIAL_SAMPLE_COUNT):
    """Recover the unique integer count represented by a five-decimal rate."""
    if (not isinstance(token, str) or _ACCURACY_TOKEN.fullmatch(token) is None
            or not isinstance(sample_count, int)
            or isinstance(sample_count, bool) or sample_count <= 0):
        raise ValueError("official accuracy does not identify a unique hit count")
    matches = [
        hits for hits in range(sample_count + 1)
        if "%.5f" % (hits / float(sample_count)) == token
    ]
    if len(matches) != 1:
        raise ValueError("official accuracy does not identify a unique hit count")
    return matches[0]


def _metric_tokens(text, label):
    if not isinstance(text, str):
        raise ValueError("{} metrics must be text".format(label))
    matches = _METRIC_PATTERN.findall(text)
    result = {}
    for threshold in ("25", "50"):
        values = [value for key, value in matches if key == threshold]
        if len(values) != 1:
            raise ValueError(
                "{} must contain each official metric exactly once".format(
                    label
                )
            )
        result[threshold] = values[0]
    return result


def parse_official_metrics(log_text, stdout_text):
    """Parse two independently preserved renderings of the official metrics."""
    log_tokens = _metric_tokens(log_text, "official log")
    stdout_tokens = _metric_tokens(stdout_text, "official stdout")
    if log_tokens != stdout_tokens:
        raise ValueError("official log and stdout metrics must agree")
    hits025 = recover_exact_hits(log_tokens["25"])
    hits050 = recover_exact_hits(log_tokens["50"])
    if hits050 > hits025:
        raise ValueError("official threshold hit counts are inconsistent")
    return {
        "printed_acc025": log_tokens["25"],
        "printed_acc050": log_tokens["50"],
        "hits025": hits025,
        "hits050": hits050,
    }


def build_authoritative_command():
    """Build the sole command accepted for the frozen official launch."""
    selected = Path(OFFICIAL_SELECTED_ARTIFACT_PATH).expanduser().resolve()
    parent = Path(OFFICIAL_PARENT_ARTIFACT_PATH).expanduser().resolve()
    backbone = Path(OFFICIAL_CHECKPOINT_PATH).expanduser().resolve()
    output = Path(OFFICIAL_RUN_ROOT).expanduser().resolve()
    return [
        str(OFFICIAL_PYTHON_EXECUTABLE),
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        "1",
        "--master_port",
        str(OFFICIAL_MASTER_PORT),
        "train_dist_mod.py",
        "--num_decoder_layers",
        "6",
        "--num_target",
        "256",
        "--model",
        "MCLN",
        "--use_color",
        "--butd",
        "--self_attend",
        "--detect_intermediate",
        "--joint_det",
        "--use_soft_token_loss",
        "--use_contrastive_align",
        "--use_source_choice_selector",
        "--source_choice_selector_sources",
        "default,default_rank_blend_contrastive010",
        "--source_choice_selector_hidden_dim",
        "288",
        "--skip_missing_superpoints",
        "--dataset",
        OFFICIAL_DATASET,
        "--test_dataset",
        OFFICIAL_DATASET,
        "--data_root",
        str(OFFICIAL_DATA_ROOT) + os.sep,
        "--batch_size",
        "12",
        "--num_workers",
        "2",
        "--print_freq",
        "100",
        "--checkpoint_path",
        str(backbone),
        "--rec_reranker_checkpoint",
        str(parent),
        "--rec_geometry_reranker_checkpoint",
        str(selected),
        "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
        "--log_dir",
        str(output),
        "--exp",
        OFFICIAL_EXPERIMENT,
        "--eval",
    ]


def claim_path_for():
    """Return the fixed authoritative claim for the official goal."""
    descriptor, registry = _open_secure_registry()
    os.close(descriptor)
    return registry / (OFFICIAL_GOAL_NAME + ".claim.json")


def official_stdout_path(run_root=None):
    root = OFFICIAL_RUN_ROOT if run_root is None else run_root
    return Path(root).expanduser().resolve() / "official_stdout.log"


def official_result_path(run_root=None):
    root = OFFICIAL_RUN_ROOT if run_root is None else run_root
    return Path(root).expanduser().resolve() / "official_result.json"


def comparison_result_path(run_root=None):
    root = OFFICIAL_RUN_ROOT if run_root is None else run_root
    return (
        Path(root).expanduser().resolve()
        / "official_sidecar_comparison.json"
    )


def receipt_path_for_claim(claim):
    claim = Path(claim)
    suffix = ".claim.json"
    if not claim.name.endswith(suffix):
        raise ValueError("official claim path is invalid")
    return claim.with_name(claim.name[:-len(suffix)] + ".receipt.json")


def timestamp_run_root(run_root):
    return (
        Path(run_root).expanduser().resolve()
        / OFFICIAL_DATASET / OFFICIAL_EXPERIMENT
    )


def snapshot_timestamp_runs(run_root):
    root = timestamp_run_root(run_root)
    if not root.exists():
        return frozenset()
    if not root.is_dir():
        raise ValueError("official timestamp root must be a directory")
    return frozenset(
        child.name for child in root.iterdir()
        if child.name.isdigit() and child.is_dir() and not child.is_symlink()
    )


def discover_unique_timestamp_run(run_root, preexisting):
    if not isinstance(preexisting, (set, frozenset)):
        raise ValueError("pre-existing timestamp snapshot is invalid")
    current = snapshot_timestamp_runs(run_root)
    if not preexisting.issubset(current):
        raise ValueError("a pre-existing timestamp run disappeared")
    created = current - preexisting
    if len(created) != 1:
        raise ValueError("official launch must create exactly one timestamp run")
    return timestamp_run_root(run_root) / next(iter(created))


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _canonical_json_bytes(value):
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not canonical JSON: {}".format(error))
    return (serialized + "\n").encode("utf-8")


def _file_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _identity_vector(value):
    return list(_file_identity(value))


def _is_identity_vector(value):
    return (isinstance(value, list) and len(value) == 5
            and all(type(component) is int for component in value))


def _stable_snapshot(path, label):
    resolved = Path(path).expanduser().resolve()
    try:
        initial = resolved.stat()
    except OSError as error:
        raise ValueError("{} does not exist: {}".format(label, error))
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("{} must be a regular file".format(label))
    try:
        with resolved.open("rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()))
            payload = handle.read()
            after = _file_identity(os.fstat(handle.fileno()))
        current_stat = resolved.stat()
        current = _file_identity(current_stat)
    except OSError as error:
        raise ValueError("could not read {}: {}".format(label, error))
    if before != after or after != current:
        raise ValueError("{} changed during stable read".format(label))
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "bytes": payload,
        "mode": stat.S_IMODE(current_stat.st_mode),
        "identity": _identity_vector(current_stat),
    }


def _snapshot_open_descriptor(descriptor, path, label):
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("{} descriptor must be regular".format(label))
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if (_file_identity(before) != _file_identity(after)
            or after.st_size != sum(len(chunk) for chunk in chunks)):
        raise ValueError("{} changed during descriptor read".format(label))
    payload = b"".join(chunks)
    snapshot = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "bytes": payload,
        "mode": stat.S_IMODE(after.st_mode),
        "identity": _identity_vector(after),
    }
    return snapshot, after


def _require_descriptor_path_identity(path, descriptor_stat, label):
    try:
        path_stat = os.lstat(str(path))
    except OSError as error:
        raise ValueError(
            "{} pathname identity is unavailable: {}".format(label, error)
        )
    if (not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_dev != descriptor_stat.st_dev
            or path_stat.st_ino != descriptor_stat.st_ino):
        raise ValueError("{} pathname identity changed".format(label))
    return path_stat


def _fsync_directory(path):
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_without_following_final(path):
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return absolute.parent.resolve() / absolute.name


def _logical_absolute_path(path):
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink_components(path, label):
    absolute = _logical_absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(str(current))
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("{} must not contain symlinks".format(label))
    return absolute


def _open_secure_registry():
    registry = _logical_absolute_path(OFFICIAL_CLAIM_REGISTRY)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in registry.parts[1:]:
            try:
                metadata = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise ValueError(
                    "official claim registry does not exist"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    "official claim registry contains a symlink"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    "official claim registry component is not a directory"
                )
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, registry
    except BaseException:
        os.close(descriptor)
        raise


def _registry_entry_exists(name):
    descriptor, _registry = _open_secure_registry()
    try:
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(descriptor)


def _write_registry_file(name, payload, label):
    descriptor, registry = _open_secure_registry()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            output = os.open(name, flags, 0o444, dir_fd=descriptor)
        except FileExistsError:
            raise FileExistsError(
                "{} already exists: {}".format(label, registry / name)
            )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    raise OSError(
                        "short write while creating {}".format(label)
                    )
                view = view[written:]
            os.fsync(output)
            os.fchmod(output, 0o444)
        finally:
            os.close(output)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return registry / name


def _write_registry_json(name, payload, label):
    return _write_registry_file(
        name, _canonical_json_bytes(payload), label
    )


def _load_registry_snapshot(name, label):
    descriptor, registry = _open_secure_registry()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            source = os.open(name, flags, dir_fd=descriptor)
        except OSError as error:
            raise ValueError("could not read {}: {}".format(label, error))
        try:
            before_stat = os.fstat(source)
            if not stat.S_ISREG(before_stat.st_mode):
                raise ValueError("{} must be a regular file".format(label))
            chunks = []
            while True:
                chunk = os.read(source, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after_stat = os.fstat(source)
        finally:
            os.close(source)
    finally:
        os.close(descriptor)
    if _file_identity(before_stat) != _file_identity(after_stat):
        raise ValueError("{} changed during stable read".format(label))
    payload = b"".join(chunks)
    return {
        "path": str(registry / name),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "bytes": payload,
        "mode": stat.S_IMODE(after_stat.st_mode),
    }


def _load_registry_json_snapshot(name, label):
    snapshot = _load_registry_snapshot(name, label)
    try:
        value = json.loads(snapshot["bytes"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("{} is invalid JSON: {}".format(label, error))
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(label))
    snapshot["value"] = value
    return snapshot


def _publish_launch_receipt(receipt_path, receipt):
    _validate_receipt_schema(receipt)
    _write_registry_json(
        receipt_path.name, receipt, "official receipt"
    )
    published = _load_registry_json_snapshot(
        receipt_path.name, "official receipt"
    )
    _require_canonical_immutable_json_snapshot(
        published, "official receipt"
    )
    if published["value"] != receipt:
        raise ValueError("published official receipt is not exact")
    return published


def _write_exclusive_file(path, payload, label):
    path = _path_without_following_final(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("{} already exists: {}".format(label, path))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o444)
    except FileExistsError:
        raise FileExistsError("{} already exists: {}".format(label, path))
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating {}".format(label))
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(str(path), 0o444)
    _fsync_directory(path.parent)
    return path


def _write_exclusive_json(path, payload, label):
    return _write_exclusive_file(
        path, _canonical_json_bytes(payload), label
    )


def _require_published_json(path, expected, label):
    snapshot = _load_json_snapshot(path, label)
    _require_canonical_immutable_json_snapshot(snapshot, label)
    if snapshot["value"] != expected:
        raise ValueError("published {} is not exact".format(label))
    return snapshot


def _require_canonical_immutable_json_snapshot(snapshot, label):
    if (not isinstance(snapshot, dict)
            or "value" not in snapshot
            or snapshot.get("bytes")
            != _canonical_json_bytes(snapshot["value"])
            or snapshot.get("mode") != 0o444):
        raise ValueError(
            "{} must have canonical bytes and immutable mode".format(label)
        )
    return snapshot


def snapshot_code_tree(code_root):
    configured_root = _reject_symlink_components(
        code_root, "runtime code tree"
    )
    root = configured_root.resolve()
    if not root.is_dir():
        raise ValueError("code root must be a directory")
    for current, directories, filenames in os.walk(
            str(root), topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directories + filenames:
            if (current_path / name).is_symlink():
                raise ValueError(
                    "runtime code tree must not contain symlinks"
                )
        directories[:] = [
            name for name in directories if name != "__pycache__"
        ]
    files = {}
    runtime_files = set(root.rglob("*.py"))
    runtime_files.update(root.rglob("*.so"))
    for path in sorted(runtime_files):
        if "__pycache__" in path.parts:
            continue
        snapshot = _stable_snapshot(path, "runtime code")
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "sha256": snapshot["sha256"],
            "size": snapshot["size"],
            "identity": snapshot["identity"],
        }
    if not files:
        raise ValueError("code manifest cannot be empty")
    digest = hashlib.sha256(_canonical_json_bytes(files)).hexdigest()
    return {"root": str(root), "files": files, "sha256": digest}


_PYTHON_MANIFEST_FIELDS = frozenset((
    "logical_path", "link_target", "resolved_path", "sha256", "size",
    "mode", "identity", "link_identity",
))


def snapshot_authoritative_python():
    logical = _logical_absolute_path(OFFICIAL_PYTHON_EXECUTABLE)
    _reject_symlink_components(
        logical.parent, "authoritative interpreter parent"
    )
    try:
        logical_metadata = os.lstat(str(logical))
    except OSError as error:
        raise ValueError(
            "authoritative interpreter symlink is unavailable: {}".format(
                error
            )
        )
    if not stat.S_ISLNK(logical_metadata.st_mode):
        raise ValueError(
            "authoritative interpreter must be the fixed symlink"
        )
    link_target = os.readlink(str(logical))
    if link_target != OFFICIAL_PYTHON_LINK_TARGET:
        raise ValueError("authoritative interpreter link target is invalid")
    target = logical.parent / link_target
    _reject_symlink_components(target, "authoritative interpreter target")
    try:
        target_metadata = os.lstat(str(target))
    except OSError as error:
        raise ValueError(
            "authoritative interpreter target is unavailable: {}".format(
                error
            )
        )
    if not stat.S_ISREG(target_metadata.st_mode):
        raise ValueError(
            "authoritative interpreter target must be a regular file"
        )
    snapshot = _stable_snapshot(target, "authoritative interpreter target")
    manifest = {
        "logical_path": str(logical),
        "link_target": link_target,
        "resolved_path": snapshot["path"],
        "sha256": snapshot["sha256"],
        "size": snapshot["size"],
        "mode": snapshot["mode"],
        "identity": snapshot["identity"],
        "link_identity": _identity_vector(logical_metadata),
    }
    if (not _is_sha256(OFFICIAL_PYTHON_TARGET_SHA256)
            or manifest["sha256"] != OFFICIAL_PYTHON_TARGET_SHA256):
        raise ValueError("authoritative interpreter SHA mismatch")
    if (type(OFFICIAL_PYTHON_TARGET_SIZE) is not int
            or manifest["size"] != OFFICIAL_PYTHON_TARGET_SIZE):
        raise ValueError("authoritative interpreter size mismatch")
    if (type(OFFICIAL_PYTHON_TARGET_MODE) is not int
            or manifest["mode"] != OFFICIAL_PYTHON_TARGET_MODE):
        raise ValueError("authoritative interpreter mode mismatch")
    return manifest


def _validate_python_manifest(manifest):
    if (not isinstance(manifest, dict)
            or set(manifest) != _PYTHON_MANIFEST_FIELDS
            or manifest.get("logical_path")
            != str(_logical_absolute_path(OFFICIAL_PYTHON_EXECUTABLE))
            or manifest.get("link_target") != OFFICIAL_PYTHON_LINK_TARGET
            or manifest.get("resolved_path") != str(
                _logical_absolute_path(OFFICIAL_PYTHON_EXECUTABLE).parent
                / OFFICIAL_PYTHON_LINK_TARGET
            )
            or manifest.get("sha256") != OFFICIAL_PYTHON_TARGET_SHA256
            or manifest.get("size") != OFFICIAL_PYTHON_TARGET_SIZE
            or manifest.get("mode") != OFFICIAL_PYTHON_TARGET_MODE
            or not _is_identity_vector(manifest.get("identity"))
            or manifest["identity"][2] != manifest["size"]
            or not _is_identity_vector(manifest.get("link_identity"))):
        raise ValueError("official interpreter manifest is invalid")
    return manifest


def _require_unchanged_python(expected):
    _validate_python_manifest(expected)
    current = snapshot_authoritative_python()
    if current["link_identity"] != expected["link_identity"]:
        raise ValueError(
            "official interpreter symlink identity changed after launch"
        )
    if current["identity"] != expected["identity"]:
        raise ValueError("official interpreter identity changed after launch")
    if current != expected:
        raise ValueError("official interpreter changed after launch")
    return current


def _read_selection(snapshot):
    try:
        selection = json.loads(snapshot["bytes"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("selection record is invalid JSON: {}".format(error))
    if not isinstance(selection, dict):
        raise ValueError("selection record must contain an object")
    return selection


def _load_json_snapshot(path, label):
    snapshot = _stable_snapshot(path, label)
    try:
        value = json.loads(snapshot["bytes"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("{} is invalid JSON: {}".format(label, error))
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(label))
    snapshot["value"] = value
    return snapshot


def _validate_selection_bindings(selection, selected, parent):
    winner = selection.get("winner")
    common = selection.get("common_train_provenance")
    if (selection.get("selection_uses_validation") is not False
            or not isinstance(winner, dict)
            or winner.get("selected_filename")
            != Path(selected["path"]).name
            or winner.get("selected_sha256") != selected["sha256"]
            or not isinstance(common, dict)
            or common.get("parent_artifact_sha256") != parent["sha256"]):
        raise ValueError("selection record does not bind frozen artifacts")


def _validate_launch_contract(command, environment):
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(value, str) or not value for value in command)):
        raise ValueError("official command must be a nonempty argv sequence")
    values = []
    for index, value in enumerate(command):
        if value in ("--nproc_per_node", "--nproc-per-node"):
            if index + 1 >= len(command):
                raise ValueError("official launch requires world_size=1")
            values.append(command[index + 1])
        elif value.startswith("--nproc_per_node="):
            values.append(value.split("=", 1)[1])
        elif value.startswith("--nproc-per-node="):
            values.append(value.split("=", 1)[1])
    if values != ["1"]:
        raise ValueError("official launch requires world_size=1")
    code_root = Path(OFFICIAL_CODE_ROOT).expanduser().resolve()
    expected_pythonpath = os.pathsep.join((
        str(code_root), str(code_root / "pointnet2")
    ))
    if (not isinstance(environment, dict)
            or environment.get("CUDA_VISIBLE_DEVICES") != "0"
            or environment.get("OMP_NUM_THREADS") != "1"
            or environment.get("PYTHONPATH") != expected_pythonpath
            or any(environment.get(key) is not None for key in (
                "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"
            ))):
        raise ValueError("official launch environment must bind CUDA 0 and OMP 1")


def _validate_authoritative_command(command):
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("official launch requires the authoritative command")
    expected = build_authoritative_command()
    if list(command) != expected:
        raise ValueError("official launch requires the authoritative command")
    return list(command)


def _input_snapshots(selection_record, selected_artifact, parent_artifact,
                     checkpoint):
    snapshots = {
        "selection_record": _stable_snapshot(
            selection_record, "selection record"
        ),
        "selected_artifact": _stable_snapshot(
            selected_artifact, "selected artifact"
        ),
        "parent_artifact": _stable_snapshot(
            parent_artifact, "parent artifact"
        ),
        "checkpoint": _stable_snapshot(checkpoint, "backbone checkpoint"),
    }
    selection = _read_selection(snapshots["selection_record"])
    _validate_selection_bindings(
        selection,
        snapshots["selected_artifact"],
        snapshots["parent_artifact"],
    )
    for snapshot in snapshots.values():
        snapshot.pop("bytes")
        snapshot.pop("mode")
    return snapshots


def _authoritative_input_snapshots():
    paths = {
        "selection_record": OFFICIAL_SELECTION_RECORD_PATH,
        "selected_artifact": OFFICIAL_SELECTED_ARTIFACT_PATH,
        "parent_artifact": OFFICIAL_PARENT_ARTIFACT_PATH,
        "checkpoint": OFFICIAL_CHECKPOINT_PATH,
    }
    expected_shas = {
        "selection_record": OFFICIAL_SELECTION_RECORD_SHA256,
        "selected_artifact": OFFICIAL_SELECTED_ARTIFACT_SHA256,
        "parent_artifact": OFFICIAL_PARENT_ARTIFACT_SHA256,
        "checkpoint": OFFICIAL_CHECKPOINT_SHA256,
    }
    if any(not _is_sha256(value) for value in expected_shas.values()):
        raise ValueError("authoritative input SHA constant is invalid")
    snapshots = _input_snapshots(
        paths["selection_record"],
        paths["selected_artifact"],
        paths["parent_artifact"],
        paths["checkpoint"],
    )
    for name, expected in expected_shas.items():
        if snapshots[name]["sha256"] != expected:
            raise ValueError(
                "authoritative {} SHA mismatch".format(
                    name.replace("_", " ")
                )
            )

    try:
        preflight = preflight_frozen_inputs(
            paths["selection_record"],
            paths["selected_artifact"],
            paths["parent_artifact"],
            device="cpu",
        )
    except ValueError as error:
        raise ValueError(
            "authoritative preflight failed: {}".format(error)
        )
    expected_preflight = {
        "selection_record_sha256": expected_shas["selection_record"],
        "selected_artifact_sha256": expected_shas["selected_artifact"],
        "parent_artifact_sha256": expected_shas["parent_artifact"],
    }
    if not isinstance(preflight, dict):
        raise ValueError("authoritative preflight binding is invalid")
    geometry_artifact = preflight.get("geometry_artifact")
    if (any(preflight.get(key) != value
                   for key, value in expected_preflight.items())
            or preflight.get("selection_uses_validation") is not False
            or not isinstance(geometry_artifact, dict)
            or geometry_artifact.get("checkpoint_sha256")
            != expected_shas["checkpoint"]):
        raise ValueError("authoritative preflight binding is invalid")
    model_inputs = geometry_artifact.get("model_inputs")
    if (not isinstance(model_inputs, dict)
            or model_inputs.get("butd") is not True
            or model_inputs.get("butd_gt") is not False
            or model_inputs.get("butd_cls") is not False
            or geometry_artifact.get("filter_non_gt_boxes") is not False
            or geometry_artifact.get("target_iou_policy") != "root_only"):
        raise ValueError("authoritative preflight no-GT binding is invalid")
    return snapshots


def run_official_launch():
    """Consume the fixed claim and capture exactly one official subprocess."""
    claim = claim_path_for()
    if _registry_entry_exists(claim.name):
        raise FileExistsError("official claim already exists: {}".format(claim))

    run_root = Path(OFFICIAL_RUN_ROOT).expanduser().resolve()
    stdout_path = official_stdout_path()
    result_path = official_result_path()
    receipt_path = receipt_path_for_claim(claim)
    for path, label in (
            (stdout_path, "official stdout"),
            (result_path, "official result")):
        if path.exists() or path.is_symlink():
            raise FileExistsError("{} already exists: {}".format(label, path))
    if _registry_entry_exists(receipt_path.name):
        raise FileExistsError(
            "official receipt already exists: {}".format(receipt_path)
        )

    code_root = _logical_absolute_path(OFFICIAL_CODE_ROOT)
    environment = _authoritative_environment(code_root)
    command = build_authoritative_command()
    _validate_launch_contract(command, environment)
    command = _validate_authoritative_command(command)
    python_manifest = snapshot_authoritative_python()
    inputs = _authoritative_input_snapshots()
    code = snapshot_code_tree(code_root)
    entrypoint = Path(code["root"]) / "train_dist_mod.py"
    if (not entrypoint.is_file() or entrypoint.is_symlink()
            or "train_dist_mod.py" not in code["files"]):
        raise ValueError("official runtime entrypoint is missing")
    preexisting = snapshot_timestamp_runs(run_root)
    critical_environment = {
        key: environment.get(key)
        for key in _CRITICAL_ENVIRONMENT_KEYS
    }
    claim_payload = {
        "schema": "rec-geometry-official-launch-claim",
        "version": 1,
        "goal": OFFICIAL_GOAL_NAME,
        "created_at_utc": _utc_now(),
        "cwd": code["root"],
        "run_root": str(Path(run_root).expanduser().resolve()),
        "stdout_path": str(stdout_path),
        "result_path": str(result_path),
        "command": list(command),
        "python": python_manifest,
        "environment": critical_environment,
        "preexisting_timestamp_runs": sorted(preexisting),
        "inputs": inputs,
        "code": code,
    }
    claim_bytes = _canonical_json_bytes(claim_payload)
    _write_registry_file(claim.name, claim_bytes, "official claim")
    claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()

    completed = None
    failure = None
    stdout_snapshot = None
    stdout_status = "not-created"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(stdout_path), flags, 0o444)
    except BaseException as error:
        failure = error
    else:
        try:
            with os.fdopen(
                    descriptor, "wb", closefd=False) as stdout_handle:
                completed = subprocess.run(
                    command,
                    stdout=stdout_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    env=environment,
                    cwd=code["root"],
                )
                stdout_handle.flush()
                os.fsync(stdout_handle.fileno())
        except BaseException as error:
            failure = error
        finalization_error = None
        try:
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        except BaseException as error:
            finalization_error = error
            if failure is None:
                failure = error
        descriptor_stat = None
        try:
            stdout_snapshot, descriptor_stat = _snapshot_open_descriptor(
                descriptor, stdout_path, "official stdout"
            )
        except BaseException as error:
            stdout_status = "snapshot-failed"
            if failure is None:
                failure = error
        identity_error = None
        if descriptor_stat is not None:
            try:
                _require_descriptor_path_identity(
                    stdout_path, descriptor_stat, "official stdout"
                )
            except BaseException as error:
                identity_error = error
                if failure is None:
                    failure = error
        try:
            _fsync_directory(stdout_path.parent)
        except BaseException as error:
            if finalization_error is None:
                finalization_error = error
            if failure is None:
                failure = error
        try:
            os.close(descriptor)
        except BaseException as error:
            if finalization_error is None:
                finalization_error = error
            if failure is None:
                failure = error
        if identity_error is not None:
            stdout_status = "identity-failed"
        elif stdout_snapshot is None:
            stdout_status = "snapshot-failed"
        elif finalization_error is not None:
            stdout_status = "finalization-failed"
        else:
            stdout_status = "captured"

    run_path = None
    discovery_failure = None
    discovery_error = None
    try:
        run_path = discover_unique_timestamp_run(run_root, preexisting)
    except BaseException as error:
        discovery_failure = error
        discovery_error = "{}: {}".format(type(error).__name__, error)

    returncode = None
    if completed is not None:
        candidate_returncode = getattr(completed, "returncode", None)
        if type(candidate_returncode) is int:
            returncode = candidate_returncode
        elif failure is None:
            failure = ValueError("subprocess return code is invalid")
    if failure is None and returncode != 0:
        failure = subprocess.CalledProcessError(returncode, list(command))
    if failure is None and discovery_failure is not None:
        failure = discovery_failure

    for require_unchanged, expected in (
            (_require_unchanged_python, python_manifest),
            (_require_unchanged_inputs, inputs),
            (_require_unchanged_code, code)):
        try:
            require_unchanged(expected)
        except BaseException as error:
            if failure is None:
                failure = error

    launcher_error = None
    if failure is not None:
        launcher_error = {
            "type": type(failure).__name__,
            "message": str(failure) or type(failure).__name__,
        }
    success = failure is None
    receipt = {
        "schema": "rec-geometry-official-launch-receipt",
        "version": 1,
        "created_at_utc": _utc_now(),
        "cwd": code["root"],
        "claim_path": str(claim),
        "claim_sha256": claim_sha256,
        "success": success,
        "status": "success" if success else "failure",
        "launcher_error": launcher_error,
        "returncode": returncode,
        "run_path": None if run_path is None else str(run_path),
        "run_discovery_error": discovery_error,
        "stdout_path": str(stdout_path),
        "stdout_sha256": (
            None if stdout_snapshot is None else stdout_snapshot["sha256"]
        ),
        "stdout_status": stdout_status,
    }
    _publish_launch_receipt(receipt_path, receipt)
    if failure is not None:
        raise failure
    result = dict(receipt)
    result.update({
        "claim_path": str(claim),
        "receipt_path": str(receipt_path),
    })
    return result


_CLAIM_FIELDS = frozenset((
    "schema", "version", "goal", "created_at_utc", "cwd", "run_root",
    "stdout_path", "result_path", "command", "python", "environment",
    "preexisting_timestamp_runs", "inputs", "code",
))
_RECEIPT_FIELDS = frozenset((
    "schema", "version", "created_at_utc", "cwd", "claim_path",
    "claim_sha256", "success", "status", "launcher_error",
    "returncode", "run_path", "run_discovery_error", "stdout_path",
    "stdout_sha256", "stdout_status",
))
_CONFIG_VALUES = {
    "eval": True,
    "eval_train": False,
    "dataset": [OFFICIAL_DATASET],
    "test_dataset": OFFICIAL_DATASET,
    "exp": OFFICIAL_EXPERIMENT,
    "batch_size": 12,
    "num_workers": 2,
    "print_freq": 100,
    "local_rank": 0,
    "num_target": 256,
    "num_decoder_layers": 6,
    "model": "MCLN",
    "butd": True,
    "butd_gt": False,
    "butd_cls": False,
    "use_color": True,
    "use_height": False,
    "use_multiview": False,
    "joint_det": True,
    "self_attend": True,
    "detect_intermediate": True,
    "use_soft_token_loss": True,
    "use_contrastive_align": True,
    "use_source_choice_selector": True,
    "source_choice_selector_sources": (
        "default,default_rank_blend_contrastive010"
    ),
    "source_choice_selector_hidden_dim": 288,
    "skip_missing_superpoints": True,
    "eval_use_rec_reranker_scores": True,
    "eval_use_rec_geometry_reranker_scores": True,
}


def _validate_claim(claim):
    if (not isinstance(claim, dict) or set(claim) != _CLAIM_FIELDS
            or claim.get("schema") != "rec-geometry-official-launch-claim"
            or claim.get("version") != 1
            or claim.get("goal") != OFFICIAL_GOAL_NAME):
        raise ValueError("official launch claim schema is invalid")
    if (not isinstance(claim.get("preexisting_timestamp_runs"), list)
            or any(not isinstance(value, str) or not value.isdigit()
                   for value in claim["preexisting_timestamp_runs"])
            or not isinstance(claim.get("inputs"), dict)
            or not isinstance(claim.get("code"), dict)):
        raise ValueError("official launch claim payload is invalid")
    _validate_code_manifest(claim["code"])
    if (not isinstance(claim["code"].get("root"), str)
            or claim.get("cwd") != claim["code"]["root"]):
        raise ValueError("official launch claim cwd is invalid")
    inputs = claim["inputs"]
    input_names = {
        "selection_record", "selected_artifact", "parent_artifact",
        "checkpoint",
    }
    if (set(inputs) != input_names
            or any(
                not isinstance(inputs.get(name), dict)
                or set(inputs[name]) != {
                    "path", "sha256", "size", "identity",
                }
                or not isinstance(inputs[name].get("path"), str)
                or not _is_sha256(inputs[name].get("sha256"))
                or type(inputs[name].get("size")) is not int
                or inputs[name]["size"] < 0
                or not _is_identity_vector(inputs[name].get("identity"))
                or inputs[name]["identity"][2] != inputs[name]["size"]
                for name in input_names
            )):
        raise ValueError("official launch claim payload is invalid")
    _validate_launch_contract(claim.get("command"), claim.get("environment"))
    _validate_authoritative_command(claim["command"])
    _validate_python_manifest(claim.get("python"))
    expected_inputs = {
        "selection_record": (
            OFFICIAL_SELECTION_RECORD_PATH,
            OFFICIAL_SELECTION_RECORD_SHA256,
        ),
        "selected_artifact": (
            OFFICIAL_SELECTED_ARTIFACT_PATH,
            OFFICIAL_SELECTED_ARTIFACT_SHA256,
        ),
        "parent_artifact": (
            OFFICIAL_PARENT_ARTIFACT_PATH,
            OFFICIAL_PARENT_ARTIFACT_SHA256,
        ),
        "checkpoint": (OFFICIAL_CHECKPOINT_PATH, OFFICIAL_CHECKPOINT_SHA256),
    }
    if any(
            inputs.get(name, {}).get("path")
            != str(Path(path).expanduser().resolve())
            or inputs.get(name, {}).get("sha256") != expected_sha
            for name, (path, expected_sha) in expected_inputs.items()):
        raise ValueError("official launch claim input binding is invalid")
    if (claim.get("run_root")
            != str(Path(OFFICIAL_RUN_ROOT).expanduser().resolve())
            or claim.get("cwd")
            != str(Path(OFFICIAL_CODE_ROOT).expanduser().resolve())
            or claim.get("stdout_path") != str(official_stdout_path())
            or claim.get("result_path") != str(official_result_path())):
        raise ValueError("official launch claim authoritative path is invalid")
    return claim


def _validate_receipt_schema(receipt):
    if (not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS
            or receipt.get("schema")
            != "rec-geometry-official-launch-receipt"
            or receipt.get("version") != 1
            or not isinstance(receipt.get("created_at_utc"), str)
            or not isinstance(receipt.get("cwd"), str)
            or not isinstance(receipt.get("claim_path"), str)
            or not _is_sha256(receipt.get("claim_sha256"))
            or type(receipt.get("success")) is not bool
            or receipt.get("status") not in ("success", "failure")
            or (receipt.get("returncode") is not None
                and type(receipt.get("returncode")) is not int)
            or (receipt.get("run_path") is not None
                and not isinstance(receipt.get("run_path"), str))
            or (receipt.get("run_discovery_error") is not None
                and not isinstance(receipt.get("run_discovery_error"), str))
            or not isinstance(receipt.get("stdout_path"), str)
            or receipt.get("stdout_status") not in (
                "captured", "not-created", "snapshot-failed",
                "finalization-failed", "identity-failed",
            )
            or (receipt.get("stdout_sha256") is not None
                and not _is_sha256(receipt.get("stdout_sha256")))):
        raise ValueError("official launch receipt schema is invalid")
    launcher_error = receipt.get("launcher_error")
    if (launcher_error is not None
            and (not isinstance(launcher_error, dict)
                 or set(launcher_error) != {"type", "message"}
                 or not isinstance(launcher_error.get("type"), str)
                 or not launcher_error["type"]
                 or not isinstance(launcher_error.get("message"), str)
                 or not launcher_error["message"])):
        raise ValueError("official launch receipt error is invalid")
    if (receipt["stdout_status"] == "captured"
            and not _is_sha256(receipt["stdout_sha256"])):
        raise ValueError("official launch receipt stdout binding is invalid")
    if (receipt["stdout_status"] in ("not-created", "snapshot-failed")
            and receipt["stdout_sha256"] is not None):
        raise ValueError("official launch receipt stdout binding is invalid")
    if ((receipt["status"] == "success" and launcher_error is not None)
            or (receipt["status"] == "failure" and launcher_error is None)):
        raise ValueError("official launch receipt error status is invalid")
    expected_success = (
        receipt["status"] == "success"
        and launcher_error is None
        and receipt["returncode"] == 0
        and isinstance(receipt["run_path"], str)
        and receipt["run_discovery_error"] is None
        and receipt["stdout_status"] == "captured"
        and _is_sha256(receipt["stdout_sha256"])
    )
    if receipt["success"] is not expected_success:
        raise ValueError("official launch receipt status is inconsistent")
    return receipt


def _validate_receipt(receipt, claim_path, claim_sha256, cwd):
    _validate_receipt_schema(receipt)
    if (receipt.get("cwd") != cwd
            or receipt.get("claim_path") != str(claim_path)
            or receipt.get("claim_sha256") != claim_sha256
            or receipt.get("success") is not True):
        raise ValueError("official launch receipt is not successful")
    return receipt


def _snapshot_without_bytes(snapshot):
    return {
        "path": snapshot["path"],
        "sha256": snapshot["sha256"],
        "size": snapshot["size"],
    }


def _require_unchanged_inputs(expected):
    names = (
        "selection_record", "selected_artifact", "parent_artifact",
        "checkpoint",
    )
    if (not isinstance(expected, dict) or set(expected) != set(names)):
        raise ValueError("official frozen input manifest is invalid")
    current = {}
    for name in names:
        value = expected.get(name)
        if (not isinstance(value, dict) or set(value)
                != {"path", "sha256", "size", "identity"}
                or not isinstance(value.get("path"), str)
                or not _is_sha256(value.get("sha256"))
                or type(value.get("size")) is not int
                or value["size"] < 0
                or not _is_identity_vector(value.get("identity"))
                or value["identity"][2] != value["size"]):
            raise ValueError("official frozen input manifest is invalid")
        snapshot = _stable_snapshot(value["path"], name.replace("_", " "))
        current[name] = {
            "path": snapshot["path"],
            "sha256": snapshot["sha256"],
            "size": snapshot["size"],
            "identity": snapshot["identity"],
        }
        if current[name]["identity"] != value["identity"]:
            raise ValueError(
                "official frozen input identity changed after launch"
            )
        if current[name] != value:
            raise ValueError("official frozen input changed after launch")
    return current


def _require_unchanged_code(expected):
    _validate_code_manifest(expected)
    current = snapshot_code_tree(expected["root"])
    if set(current["files"]) == set(expected["files"]):
        for name, binding in current["files"].items():
            if binding["identity"] != expected["files"][name]["identity"]:
                raise ValueError(
                    "official runtime code identity changed after launch"
                )
    if current != expected:
        raise ValueError("official runtime code changed after launch")
    return current


def _validate_config(config, claim, run_path):
    if not isinstance(config, dict):
        raise ValueError("official config must contain an object")
    mismatches = [
        key for key, expected in _CONFIG_VALUES.items()
        if (type(config.get(key)) is not type(expected)
            or config.get(key) != expected)
    ]
    paths = {
        "log_dir": str(run_path),
        "data_root": str(OFFICIAL_DATA_ROOT.resolve()),
        "checkpoint_path": claim["inputs"]["checkpoint"]["path"],
        "rec_reranker_checkpoint": claim["inputs"][
            "parent_artifact"
        ]["path"],
        "rec_geometry_reranker_checkpoint": claim["inputs"][
            "selected_artifact"
        ]["path"],
    }
    for key, expected in paths.items():
        value = config.get(key)
        if not isinstance(value, str) or str(
                Path(value).expanduser().resolve()) != expected:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "official config contract mismatch: {}".format(
                ", ".join(sorted(set(mismatches)))
            )
        )
    return config


def _is_sha256(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_code_manifest(manifest):
    if (not isinstance(manifest, dict)
            or set(manifest) != {"root", "files", "sha256"}
            or not isinstance(manifest.get("root"), str)
            or not manifest["root"]
            or not isinstance(manifest.get("files"), dict)
            or not manifest["files"]
            or "train_dist_mod.py" not in manifest["files"]):
        raise ValueError("official code manifest is invalid")
    for relative, binding in manifest["files"].items():
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (not isinstance(relative, str) or not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.suffix not in (".py", ".so")
                or not isinstance(binding, dict)
                or set(binding) != {"sha256", "size", "identity"}
                or not _is_sha256(binding.get("sha256"))
                or type(binding.get("size")) is not int
                or binding["size"] < 0
                or not _is_identity_vector(binding.get("identity"))
                or binding["identity"][2] != binding["size"]):
            raise ValueError("official code manifest is invalid")
    expected_sha = hashlib.sha256(
        _canonical_json_bytes(manifest["files"])
    ).hexdigest()
    if manifest.get("sha256") != expected_sha:
        raise ValueError("official code manifest digest is invalid")
    return manifest


def validate_official_result(record):
    if (not isinstance(record, dict)
            or set(record) != set(OFFICIAL_RESULT_FIELDS)
            or record.get("schema") != OFFICIAL_RESULT_SCHEMA
            or type(record.get("version")) is not int
            or record.get("version") != OFFICIAL_RESULT_VERSION
            or type(record.get("sample_count")) is not int
            or record.get("sample_count") != OFFICIAL_SAMPLE_COUNT):
        raise ValueError("official result fields do not match exact schema")
    for name in ("hits025", "hits050"):
        if (not isinstance(record.get(name), int)
                or isinstance(record.get(name), bool)
                or not 0 <= record[name] <= OFFICIAL_SAMPLE_COUNT):
            raise ValueError("official result hit count is invalid")
    if (record["hits050"] > record["hits025"]
            or recover_exact_hits(record.get("printed_acc025"))
            != record["hits025"]
            or recover_exact_hits(record.get("printed_acc050"))
            != record["hits050"]):
        raise ValueError("official result metrics are inconsistent")
    expected_booleans = {
        "gate025_pass": record["hits025"] >= OFFICIAL_GATE_HITS025,
        "gate050_pass": record["hits050"] >= OFFICIAL_GATE_HITS050,
        "acceptance_gate_pass": (
            record["hits025"] >= OFFICIAL_GATE_HITS025
            and record["hits050"] >= OFFICIAL_GATE_HITS050
        ),
        "inference_uses_ground_truth": False,
    }
    if any(record.get(key) is not value
           for key, value in expected_booleans.items()):
        raise ValueError("official result acceptance flags are inconsistent")
    if (not isinstance(record.get("created_at_utc"), str)
            or not record["created_at_utc"].endswith("Z")):
        raise ValueError("official result creation time is invalid")
    nested = {
        "run": {"path", "timestamp", "dataset", "experiment"},
        "files": {"config", "log", "stdout"},
        "artifacts": {
            "selection_record", "selected_artifact", "parent_artifact",
            "checkpoint",
        },
        "code": {"root", "files", "sha256"},
        "launch": {
            "claim_path", "claim_sha256", "receipt_path",
            "receipt_sha256", "command", "environment", "world_size",
            "local_rank", "batch_size", "cwd", "python",
        },
    }
    if any(not isinstance(record.get(name), dict)
           or set(record[name]) != fields for name, fields in nested.items()):
        raise ValueError("official result nested schema is invalid")
    for value in record["files"].values():
        if (not isinstance(value, dict)
                or set(value) != {"path", "sha256", "size"}
                or not isinstance(value["path"], str)
                or not _is_sha256(value["sha256"])
                or type(value["size"]) is not int
                or value["size"] < 0):
            raise ValueError("official result file binding is invalid")
    for value in record["artifacts"].values():
        if (not isinstance(value, dict)
                or set(value) != {"path", "sha256", "size", "identity"}
                or not isinstance(value["path"], str)
                or not _is_sha256(value["sha256"])
                or type(value["size"]) is not int
                or value["size"] < 0
                or not _is_identity_vector(value.get("identity"))
                or value["identity"][2] != value["size"]):
            raise ValueError("official result artifact binding is invalid")
    run = record["run"]
    timestamp = run.get("timestamp")
    if (type(timestamp) is not int or timestamp < 0
            or run.get("dataset") != OFFICIAL_DATASET
            or run.get("experiment") != OFFICIAL_EXPERIMENT):
        raise ValueError("official result run identity is invalid")
    expected_run_path = timestamp_run_root(OFFICIAL_RUN_ROOT) / str(timestamp)
    if run.get("path") != str(expected_run_path):
        raise ValueError("official result run path is invalid")
    expected_file_paths = {
        "config": expected_run_path / "config.json",
        "log": expected_run_path / "log.txt",
        "stdout": official_stdout_path(),
    }
    if any(record["files"][name]["path"] != str(path)
           for name, path in expected_file_paths.items()):
        raise ValueError("official result evidence path is invalid")
    expected_artifacts = {
        "selection_record": (
            OFFICIAL_SELECTION_RECORD_PATH,
            OFFICIAL_SELECTION_RECORD_SHA256,
        ),
        "selected_artifact": (
            OFFICIAL_SELECTED_ARTIFACT_PATH,
            OFFICIAL_SELECTED_ARTIFACT_SHA256,
        ),
        "parent_artifact": (
            OFFICIAL_PARENT_ARTIFACT_PATH,
            OFFICIAL_PARENT_ARTIFACT_SHA256,
        ),
        "checkpoint": (OFFICIAL_CHECKPOINT_PATH, OFFICIAL_CHECKPOINT_SHA256),
    }
    if any(
            record["artifacts"][name]["path"]
            != str(Path(path).expanduser().resolve())
            or record["artifacts"][name]["sha256"] != expected_sha
            for name, (path, expected_sha) in expected_artifacts.items()):
        raise ValueError("official result authoritative artifact is invalid")
    code = record["code"]
    expected_code_root = str(Path(OFFICIAL_CODE_ROOT).expanduser().resolve())
    _validate_code_manifest(code)
    if code.get("root") != expected_code_root:
        raise ValueError("official result code manifest is invalid")
    launch = record["launch"]
    if (launch.get("cwd") != expected_code_root
            or launch.get("claim_path") != str(claim_path_for())
            or launch.get("receipt_path") != str(
                receipt_path_for_claim(claim_path_for())
            )
            or set(launch.get("environment", {}))
            != set(_CRITICAL_ENVIRONMENT_KEYS)):
        raise ValueError("official result launch provenance is invalid")
    _validate_launch_contract(
        launch.get("command"), launch.get("environment")
    )
    _validate_authoritative_command(launch["command"])
    if (not _is_sha256(record["code"].get("sha256"))
            or not _is_sha256(record["launch"].get("claim_sha256"))
            or not _is_sha256(record["launch"].get("receipt_sha256"))
            or not isinstance(record["launch"].get("cwd"), str)
            or record["launch"].get("cwd") != record["code"].get("root")
            or record["launch"].get("world_size") != 1
            or record["launch"].get("local_rank") != 0
            or record["launch"].get("batch_size") != 12):
        raise ValueError("official result provenance is invalid")
    _validate_python_manifest(record["launch"].get("python"))
    return record


def _validate_sidecar_core(record):
    try:
        return validate_frozen_record(record)
    except ValueError as error:
        raise ValueError("sidecar record is invalid: {}".format(error))


def validate_comparison_result(record):
    if (not isinstance(record, dict)
            or set(record) != set(COMPARISON_RESULT_FIELDS)
            or record.get("schema") != COMPARISON_RESULT_SCHEMA
            or type(record.get("version")) is not int
            or record.get("version") != COMPARISON_RESULT_VERSION
            or type(record.get("sample_count")) is not int
            or record.get("sample_count") != OFFICIAL_SAMPLE_COUNT):
        raise ValueError("comparison result fields do not match exact schema")
    hit_names = (
        "official_hits025", "official_hits050",
        "sidecar_hits025", "sidecar_hits050",
    )
    if any(type(record.get(name)) is not int
           or not 0 <= record[name] <= OFFICIAL_SAMPLE_COUNT
           for name in hit_names):
        raise ValueError("comparison result hit count is invalid")
    if (record["official_hits050"] > record["official_hits025"]
            or record["sidecar_hits050"] > record["sidecar_hits025"]):
        raise ValueError("comparison result hit counts are inconsistent")
    expected_delta025 = (
        record["official_hits025"] - record["sidecar_hits025"]
    )
    expected_delta050 = (
        record["official_hits050"] - record["sidecar_hits050"]
    )
    if (type(record.get("delta025")) is not int
            or type(record.get("delta050")) is not int
            or record["delta025"] != expected_delta025
            or record["delta050"] != expected_delta050):
        raise ValueError("comparison result deltas are inconsistent")
    hits025_match = expected_delta025 == 0
    hits050_match = expected_delta050 == 0
    expected_booleans = {
        "hits025_match": hits025_match,
        "hits050_match": hits050_match,
        "hit_counts_match": hits025_match and hits050_match,
        "artifact_provenance_match": True,
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
    }
    if any(record.get(key) is not value
           for key, value in expected_booleans.items()):
        raise ValueError("comparison result flags are inconsistent")
    expected_official_gate = (
        record["official_hits025"] >= OFFICIAL_GATE_HITS025
        and record["official_hits050"] >= OFFICIAL_GATE_HITS050
    )
    if (record.get("official_acceptance_gate_pass")
            is not expected_official_gate):
        raise ValueError("comparison result official gate is invalid")
    expected_acceptance = (
        record["official_acceptance_gate_pass"]
        and record["hit_counts_match"]
        and record["artifact_provenance_match"]
    )
    if record.get("acceptance") is not expected_acceptance:
        raise ValueError("comparison result acceptance is inconsistent")
    if (not isinstance(record.get("created_at_utc"), str)
            or not record["created_at_utc"].endswith("Z")):
        raise ValueError("comparison result creation time is invalid")
    files = record.get("files")
    if (not isinstance(files, dict)
            or set(files) != {"official_result", "sidecar_record"}):
        raise ValueError("comparison result file schema is invalid")
    for value in files.values():
        if (not isinstance(value, dict)
                or set(value) != {"path", "sha256", "size"}
                or not isinstance(value.get("path"), str)
                or not _is_sha256(value.get("sha256"))
                or type(value.get("size")) is not int
                or value["size"] < 0):
            raise ValueError("comparison result file binding is invalid")
    expected_file_paths = {
        "official_result": official_result_path(),
        "sidecar_record": Path(
            OFFICIAL_SIDECAR_RECORD_PATH
        ).expanduser().resolve(),
    }
    if any(files[name]["path"] != str(path)
           for name, path in expected_file_paths.items()):
        raise ValueError("comparison result file path is invalid")
    artifacts = record.get("artifacts")
    artifact_fields = {
        "selected_artifact_sha256",
        "parent_artifact_sha256",
        "backbone_checkpoint_sha256",
        "selection_record_sha256",
    }
    if (not isinstance(artifacts, dict)
            or set(artifacts) != artifact_fields
            or any(not _is_sha256(value) for value in artifacts.values())):
        raise ValueError("comparison result artifact schema is invalid")
    expected_artifacts = {
        "selected_artifact_sha256": OFFICIAL_SELECTED_ARTIFACT_SHA256,
        "parent_artifact_sha256": OFFICIAL_PARENT_ARTIFACT_SHA256,
        "backbone_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "selection_record_sha256": OFFICIAL_SELECTION_RECORD_SHA256,
    }
    if artifacts != expected_artifacts:
        raise ValueError("comparison result artifact provenance is invalid")
    return record


def _load_launch_evidence():
    claim_path = claim_path_for()
    claim_snapshot = _load_registry_json_snapshot(
        claim_path.name, "official claim"
    )
    _require_canonical_immutable_json_snapshot(
        claim_snapshot, "official claim"
    )
    claim = _validate_claim(claim_snapshot["value"])
    python_manifest = _require_unchanged_python(claim["python"])
    receipt_path = receipt_path_for_claim(claim_path)
    receipt_snapshot = _load_registry_json_snapshot(
        receipt_path.name, "official receipt"
    )
    _require_canonical_immutable_json_snapshot(
        receipt_snapshot, "official receipt"
    )
    receipt = _validate_receipt(
        receipt_snapshot["value"], claim_path, claim_snapshot["sha256"],
        claim["cwd"],
    )
    preexisting = frozenset(claim["preexisting_timestamp_runs"])
    run_path = discover_unique_timestamp_run(claim["run_root"], preexisting)
    if str(run_path) != receipt["run_path"]:
        raise ValueError("official receipt identifies a different run")

    artifacts = _require_unchanged_inputs(claim["inputs"])
    code = _require_unchanged_code(claim["code"])
    config_snapshot = _load_json_snapshot(
        run_path / "config.json", "official config"
    )
    config = _validate_config(config_snapshot["value"], claim, run_path)
    log_snapshot = _stable_snapshot(run_path / "log.txt", "official log")
    stdout_snapshot = _stable_snapshot(
        claim["stdout_path"], "official stdout"
    )
    if (stdout_snapshot["sha256"] != receipt["stdout_sha256"]
            or stdout_snapshot["path"] != receipt["stdout_path"]):
        raise ValueError("official stdout changed after launch")
    try:
        log_text = log_snapshot["bytes"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("official log is not UTF-8: {}".format(error))
    stdout_text = stdout_snapshot["bytes"].decode(
        "utf-8", errors="replace"
    )
    if len(re.findall(
            r"length of testing dataset:\s*9508(?:\D|$)", log_text)) != 1:
        raise ValueError("official log must prove the 9,508-row dataset once")
    metrics = parse_official_metrics(log_text, stdout_text)
    return {
        "claim_path": claim_path,
        "claim_snapshot": claim_snapshot,
        "claim": claim,
        "python": python_manifest,
        "receipt_path": receipt_path,
        "receipt_snapshot": receipt_snapshot,
        "receipt": receipt,
        "run_path": run_path,
        "artifacts": artifacts,
        "code": code,
        "config_snapshot": config_snapshot,
        "config": config,
        "log_snapshot": log_snapshot,
        "stdout_snapshot": stdout_snapshot,
        "metrics": metrics,
    }


def _official_record_from_evidence(evidence, created_at_utc):
    metrics = evidence["metrics"]
    run_path = evidence["run_path"]
    claim = evidence["claim"]
    config = evidence["config"]
    return {
        "schema": OFFICIAL_RESULT_SCHEMA,
        "version": OFFICIAL_RESULT_VERSION,
        "created_at_utc": created_at_utc,
        "sample_count": OFFICIAL_SAMPLE_COUNT,
        "printed_acc025": metrics["printed_acc025"],
        "printed_acc050": metrics["printed_acc050"],
        "hits025": metrics["hits025"],
        "hits050": metrics["hits050"],
        "gate025_pass": metrics["hits025"] >= OFFICIAL_GATE_HITS025,
        "gate050_pass": metrics["hits050"] >= OFFICIAL_GATE_HITS050,
        "acceptance_gate_pass": (
            metrics["hits025"] >= OFFICIAL_GATE_HITS025
            and metrics["hits050"] >= OFFICIAL_GATE_HITS050
        ),
        "inference_uses_ground_truth": False,
        "run": {
            "path": str(run_path),
            "timestamp": int(run_path.name),
            "dataset": OFFICIAL_DATASET,
            "experiment": OFFICIAL_EXPERIMENT,
        },
        "files": {
            "config": _snapshot_without_bytes(
                evidence["config_snapshot"]
            ),
            "log": _snapshot_without_bytes(evidence["log_snapshot"]),
            "stdout": _snapshot_without_bytes(
                evidence["stdout_snapshot"]
            ),
        },
        "artifacts": evidence["artifacts"],
        "code": evidence["code"],
        "launch": {
            "claim_path": str(evidence["claim_path"]),
            "claim_sha256": evidence["claim_snapshot"]["sha256"],
            "receipt_path": str(evidence["receipt_path"]),
            "receipt_sha256": evidence["receipt_snapshot"]["sha256"],
            "cwd": claim["cwd"],
            "command": claim["command"],
            "python": evidence["python"],
            "environment": claim["environment"],
            "world_size": 1,
            "local_rank": config["local_rank"],
            "batch_size": config["batch_size"],
        },
    }


def seal_official_result():
    """Seal the sole successful run bound by the fixed selection claim."""
    evidence = _load_launch_evidence()
    claim = evidence["claim"]
    output = Path(claim["result_path"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("official result already exists: {}".format(output))
    if output != official_result_path(claim["run_root"]):
        raise ValueError("official result path is not claim-derived")
    record = _official_record_from_evidence(evidence, _utc_now())
    validate_official_result(record)
    _write_exclusive_json(output, record, "official result")
    _require_published_json(output, record, "official result")
    return record


def seal_sidecar_comparison():
    """Seal one independent comparison with the fixed official result."""
    evidence = _load_launch_evidence()
    claim = evidence["claim"]
    official_path = official_result_path(claim["run_root"])
    if official_path != Path(claim["result_path"]):
        raise ValueError("official result path is not claim-derived")
    output = comparison_result_path(claim["run_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "official comparison already exists: {}".format(output)
        )

    official_snapshot = _load_json_snapshot(
        official_path, "official result"
    )
    _require_canonical_immutable_json_snapshot(
        official_snapshot, "official result"
    )
    official_record = validate_official_result(
        official_snapshot["value"]
    )
    expected_official = _official_record_from_evidence(
        evidence, official_record["created_at_utc"]
    )
    if official_record != expected_official:
        raise ValueError("official result provenance does not match evidence")

    sidecar_snapshot = _load_json_snapshot(
        OFFICIAL_SIDECAR_RECORD_PATH, "sidecar record"
    )
    _require_canonical_immutable_json_snapshot(
        sidecar_snapshot, "sidecar record"
    )
    if (not _is_sha256(OFFICIAL_SIDECAR_RECORD_SHA256)
            or sidecar_snapshot["sha256"]
            != OFFICIAL_SIDECAR_RECORD_SHA256):
        raise ValueError("sidecar record SHA is not authoritative")
    sidecar = _validate_sidecar_core(sidecar_snapshot["value"])
    artifact_values = {
        "selected_artifact_sha256": official_record["artifacts"][
            "selected_artifact"
        ]["sha256"],
        "parent_artifact_sha256": official_record["artifacts"][
            "parent_artifact"
        ]["sha256"],
        "backbone_checkpoint_sha256": official_record["artifacts"][
            "checkpoint"
        ]["sha256"],
        "selection_record_sha256": official_record["artifacts"][
            "selection_record"
        ]["sha256"],
    }
    if any(sidecar.get(name) != value
           for name, value in artifact_values.items()):
        raise ValueError(
            "sidecar artifact provenance does not match official result"
        )

    delta025 = official_record["hits025"] - sidecar["hits025"]
    delta050 = official_record["hits050"] - sidecar["hits050"]
    hits025_match = delta025 == 0
    hits050_match = delta050 == 0
    hit_counts_match = hits025_match and hits050_match
    record = {
        "schema": COMPARISON_RESULT_SCHEMA,
        "version": COMPARISON_RESULT_VERSION,
        "created_at_utc": _utc_now(),
        "sample_count": OFFICIAL_SAMPLE_COUNT,
        "official_hits025": official_record["hits025"],
        "official_hits050": official_record["hits050"],
        "sidecar_hits025": sidecar["hits025"],
        "sidecar_hits050": sidecar["hits050"],
        "delta025": delta025,
        "delta050": delta050,
        "hits025_match": hits025_match,
        "hits050_match": hits050_match,
        "hit_counts_match": hit_counts_match,
        "artifact_provenance_match": True,
        "official_acceptance_gate_pass": official_record[
            "acceptance_gate_pass"
        ],
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
        "acceptance": (
            official_record["acceptance_gate_pass"] and hit_counts_match
        ),
        "files": {
            "official_result": _snapshot_without_bytes(official_snapshot),
            "sidecar_record": _snapshot_without_bytes(sidecar_snapshot),
        },
        "artifacts": artifact_values,
    }
    validate_comparison_result(record)
    _write_exclusive_json(output, record, "official comparison")
    published = _require_published_json(
        output, record, "official comparison"
    )["value"]
    validate_comparison_result(published)
    if published != record:
        raise ValueError("official comparison changed while publishing")
    return record


def _authoritative_environment(code_root, base_environment=None):
    root = Path(code_root).expanduser().resolve()
    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    pythonpath = [str(root), str(root / "pointnet2")]
    for key in ("PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(pythonpath),
    })
    return environment


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Launch and seal the frozen official REC geometry run."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    subparsers.add_parser("launch")
    subparsers.add_parser("seal")
    subparsers.add_parser("compare")
    return parser


def main(argv=None):
    args = _build_argument_parser().parse_args(argv)
    if args.operation == "launch":
        result = run_official_launch()
    elif args.operation == "seal":
        result = seal_official_result()
    else:
        result = seal_sidecar_comparison()
    sys.stdout.write(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
