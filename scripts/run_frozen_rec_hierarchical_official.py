#!/usr/bin/env python
"""Run and seal one frozen official hierarchical ScanRefer evaluation."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import torch

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from scripts import run_frozen_rec_geometry_official as geometry_official
from scripts import run_rec_hierarchical_online_calibration as online
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_SHA256,
    AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
    load_hierarchical_artifact,
    validate_hierarchical_result_receipt,
)


OFFICIAL_SAMPLE_COUNT = 9508
OFFICIAL_DATASET = "scanrefer"
OFFICIAL_EXPERIMENT = "epoch71_hierarchical_official"
OFFICIAL_DATA_ROOT = Path("/root/autodl-tmp/DATA_ROOT")
OFFICIAL_PYTHON_EXECUTABLE = Path(
    "/root/miniconda3/envs/bdetr/bin/python"
)
OFFICIAL_MASTER_PORT = 29671
OFFICIAL_GATE_HITS025 = 5705
OFFICIAL_GATE_HITS050 = 4469
OFFICIAL_RESULT_SCHEMA = "rec-hierarchical-official-validation-result-v1"
OFFICIAL_RESULT_VERSION = 1
OFFICIAL_CLAIM_NAME = "hierarchical-official.claim.json"
OFFICIAL_STDOUT_NAME = "official_stdout.log"
OFFICIAL_RESULT_NAME = "official_result.json"
OFFICIAL_CODE_ROOT = Path(os.path.abspath(__file__)).parents[1]

OFFICIAL_CONFIG_VALUES = {
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
    "eval_use_rec_hierarchical_reranker_scores": True,
    "eval_use_rec_selective_residual_scores": False,
}

_SNAPSHOT_FIELDS = {
    "path", "device", "inode", "mode", "size", "mtime_ns", "ctime_ns",
    "sha256",
}
_PROTECTED_ONLINE_NAMES = {
    "backbone", "parent", "geometry", "staged_hierarchical",
    "staged_result_receipt", "source_gate_baseline",
}
_ARTIFACT_NAMES = _PROTECTED_ONLINE_NAMES | {
    "hierarchical", "online_calibration",
}
_RESULT_FIELDS = {
    "schema", "version", "created_at_utc", "sample_count",
    "printed_acc025", "printed_acc050", "hits025", "hits050",
    "gate025_pass", "gate050_pass", "acceptance_gate_pass",
    "position_subgroups", "inference_uses_ground_truth", "run", "files",
    "artifacts", "code", "launch",
}
_ACCURACY_TOKEN = re.compile(r"^(?:0|1)\.[0-9]{5}$")
_METRIC_PATTERN = re.compile(
    r"last_ position alignment Acc0\.(25|50): Top-1: "
    r"((?:0|1)\.[0-9]{5})(?=,|\r?$)",
    re.MULTILINE,
)
_SUBGROUP_PATTERN = re.compile(
    r"position subgroup (unique|multiple) Acc(0\.(?:25|50)): "
    r"hits=([0-9]+), total=([0-9]+), "
    r"accuracy=((?:0|1)\.[0-9]{12})(?:\r?$)",
    re.MULTILINE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CRITICAL_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH", "PYTHONHOME",
    "LD_PRELOAD", "LD_LIBRARY_PATH",
)


def _is_sha256(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


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


def _absolute_path(value, label):
    if (not isinstance(value, (str, os.PathLike))
            or isinstance(value, bytes)):
        raise ValueError("{} must be path-like".format(label))
    try:
        return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("{} is invalid: {}".format(label, error))


def _reject_symlink_components(path, label, include_leaf=True):
    current = Path(path) if include_leaf else Path(path).parent
    while True:
        try:
            metadata = os.lstat(str(current))
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ValueError(
                "could not inspect {} path: {}".format(label, error)
            )
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("{} path contains a symlink".format(label))
        parent = current.parent
        if parent == current:
            break
        current = parent


def _file_identity(metadata):
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stable_read_only_snapshot(path, label, include_bytes=False):
    logical = _absolute_path(path, label)
    _reject_symlink_components(logical, label)
    try:
        initial = os.lstat(str(logical))
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("{} must be a regular file".format(label))
        if stat.S_IMODE(initial.st_mode) != 0o444:
            raise ValueError("{} must have immutable mode 0444".format(label))
        descriptor = os.open(
            str(logical), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            before = os.fstat(descriptor)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final = os.lstat(str(logical))
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("could not read {}: {}".format(label, error))
    identities = tuple(map(_file_identity, (initial, before, after, final)))
    if len(set(identities)) != 1:
        raise ValueError("{} changed during stable snapshot".format(label))
    payload = b"".join(chunks)
    snapshot = {
        "path": str(logical.resolve(strict=True)),
        "device": int(final.st_dev),
        "inode": int(final.st_ino),
        "mode": stat.S_IMODE(final.st_mode),
        "size": int(final.st_size),
        "mtime_ns": int(final.st_mtime_ns),
        "ctime_ns": int(final.st_ctime_ns),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if include_bytes:
        snapshot["bytes"] = payload
    return snapshot


def _snapshot_without_bytes(snapshot):
    result = dict(snapshot)
    result.pop("bytes", None)
    return result


def _validate_snapshot(value, label):
    if (not isinstance(value, dict) or set(value) != _SNAPSHOT_FIELDS
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or type(value.get("device")) is not int
            or type(value.get("inode")) is not int
            or value.get("mode") != 0o444
            or type(value.get("size")) is not int
            or type(value.get("mtime_ns")) is not int
            or type(value.get("ctime_ns")) is not int
            or not _is_sha256(value.get("sha256"))):
        raise ValueError("{} snapshot is invalid".format(label))
    return copy.deepcopy(value)


def recover_exact_hits(token, sample_count=OFFICIAL_SAMPLE_COUNT):
    """Recover the unique integer represented by a five-decimal accuracy."""
    if (not isinstance(token, str)
            or _ACCURACY_TOKEN.fullmatch(token) is None
            or type(sample_count) is not int or sample_count <= 0):
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


def _position_subgroups(text, label):
    matches = _SUBGROUP_PATTERN.findall(text)
    if not matches:
        return None
    expected = {
        (group, threshold)
        for group in ("unique", "multiple")
        for threshold in ("0.25", "0.50")
    }
    grouped = {}
    for group, threshold, hits_token, total_token, accuracy in matches:
        key = (group, threshold)
        if key in grouped:
            raise ValueError(
                "{} position subgroup is duplicated".format(label)
            )
        hits = int(hits_token)
        total = int(total_token)
        if (total <= 0 or not 0 <= hits <= total
                or accuracy != "{:.12f}".format(hits / float(total))):
            raise ValueError(
                "{} position subgroup values are invalid".format(label)
            )
        grouped[key] = {"hits": hits, "total": total}
    if set(grouped) != expected:
        raise ValueError("{} position subgroup set is incomplete".format(label))
    return {
        group: {
            threshold: grouped[(group, threshold)]
            for threshold in ("0.25", "0.50")
        }
        for group in ("unique", "multiple")
    }


def _validate_subgroup_reconciliation(subgroups, hits025, hits050):
    if subgroups is None:
        return None
    for threshold, overall_hits in (
            ("0.25", hits025), ("0.50", hits050)):
        unique = subgroups["unique"][threshold]
        multiple = subgroups["multiple"][threshold]
        if (unique["total"] + multiple["total"] != OFFICIAL_SAMPLE_COUNT
                or unique["hits"] + multiple["hits"] != overall_hits):
            raise ValueError("position subgroup counts do not reconcile")
    return copy.deepcopy(subgroups)


def parse_official_evidence(log_text, stdout_text):
    """Parse two independent metric renderings and reconcile subgroup counts."""
    log_tokens = _metric_tokens(log_text, "official log")
    stdout_tokens = _metric_tokens(stdout_text, "official stdout")
    if log_tokens != stdout_tokens:
        raise ValueError("official log and stdout metrics must agree")
    hits025 = recover_exact_hits(log_tokens["25"])
    hits050 = recover_exact_hits(log_tokens["50"])
    if hits050 > hits025:
        raise ValueError("official threshold hit counts are inconsistent")
    log_groups = _position_subgroups(log_text, "official log")
    stdout_groups = _position_subgroups(stdout_text, "official stdout")
    if log_groups != stdout_groups:
        raise ValueError("official log and stdout subgroup evidence must agree")
    groups = _validate_subgroup_reconciliation(
        log_groups, hits025, hits050
    )
    return {
        "printed_acc025": log_tokens["25"],
        "printed_acc050": log_tokens["50"],
        "hits025": hits025,
        "hits050": hits050,
        "position_subgroups": groups,
    }


def acceptance_gate_pass(hits025, hits050, sample_count,
                         inference_uses_ground_truth):
    """Apply the exact integer paper-goal gate."""
    return (
        type(hits025) is int
        and type(hits050) is int
        and type(sample_count) is int
        and type(inference_uses_ground_truth) is bool
        and sample_count == OFFICIAL_SAMPLE_COUNT
        and 0 <= hits050 <= hits025 <= sample_count
        and hits025 >= OFFICIAL_GATE_HITS025
        and hits050 >= OFFICIAL_GATE_HITS050
        and inference_uses_ground_truth is False
    )


def build_authoritative_command(artifacts, run_root):
    """Build the geometry command plus only the deployed hierarchy surface."""
    if not isinstance(artifacts, dict):
        raise ValueError("official artifacts are required")
    for name in ("backbone", "parent", "geometry", "hierarchical"):
        _validate_snapshot(artifacts.get(name), name)
    output = _absolute_path(run_root, "official run root")
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
        artifacts["backbone"]["path"],
        "--rec_reranker_checkpoint",
        artifacts["parent"]["path"],
        "--rec_geometry_reranker_checkpoint",
        artifacts["geometry"]["path"],
        "--rec_hierarchical_reranker_checkpoint",
        artifacts["hierarchical"]["path"],
        "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
        "--eval_use_rec_hierarchical_reranker_scores",
        "--log_dir",
        str(output),
        "--exp",
        OFFICIAL_EXPERIMENT,
        "--eval",
    ]


def _default_online_receipt_loader(path):
    return online._load_strict_canonical_json(
        path, "online calibration receipt"
    )


def _default_code_manifest_builder(root):
    return geometry_official.snapshot_code_tree(root)


def _default_python_manifest_builder():
    return geometry_official.snapshot_authoritative_python()


def _default_environment_builder(root):
    return geometry_official._authoritative_environment(root)


def _prepare_absent_run_root(path, protected_paths):
    output = _absolute_path(path, "official run root")
    _reject_symlink_components(output, "official run root")
    if output.exists() or output.is_symlink():
        raise FileExistsError("official run root already exists: {}".format(
            output
        ))
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("official run-root parent must be a real directory")
    output_value = str(output)
    for protected in protected_paths:
        protected_value = str(Path(protected))
        try:
            common = os.path.commonpath((output_value, protected_value))
        except ValueError:
            continue
        if common in (output_value, protected_value):
            raise ValueError("official run root overlaps protected input")
    return output


def _require_online_record_policy(record, online_path):
    if (not isinstance(record, dict)
            or record.get("schema") != online.ONLINE_CALIBRATION_SCHEMA
            or record.get("version") != online.ONLINE_CALIBRATION_VERSION
            or record.get("sample_count") != online.ONLINE_SAMPLE_COUNT
            or record.get("validation_data_accessed") is not False
            or record.get("inference_uses_ground_truth") is not False
            or not isinstance(record.get("gate"), dict)
            or record["gate"].get("passed") is not True):
        raise ValueError("online calibration did not authorize official launch")
    deployed = record.get("deployed_artifact")
    staged = record.get("staged_artifact")
    if (not isinstance(deployed, dict)
            or deployed.get("deployable") is not True
            or not isinstance(deployed.get("path"), str)
            or not _is_sha256(deployed.get("sha256"))
            or not isinstance(staged, dict)
            or staged.get("deployable") is not False
            or not _is_sha256(staged.get("sha256"))):
        raise ValueError("online deployed artifact binding is invalid")
    deployed_path = _absolute_path(deployed["path"], "deployed hierarchy")
    if (deployed_path.parent != online_path.parent
            or deployed_path.name != online.DEPLOYED_ARTIFACT_NAME):
        raise ValueError("online receipt and deployed hierarchy are not colocated")
    return staged, deployed


def preflight_official_inputs(
        online_calibration_receipt, run_root, *,
        online_receipt_loader=None, online_record_validator=None,
        staged_receipt_loader=None, staged_receipt_validator=None,
        hierarchy_loader=None, code_manifest_builder=None,
        python_manifest_builder=None, environment_builder=None):
    """Validate all train-only gates and frozen inputs before claiming."""
    online_path = _absolute_path(
        online_calibration_receipt, "online calibration receipt"
    )
    if online_path.name != online.ONLINE_CALIBRATION_NAME:
        raise ValueError("online calibration receipt name is not authoritative")
    online_snapshot_with_bytes = _stable_read_only_snapshot(
        online_path, "online calibration receipt", include_bytes=True
    )
    online_receipt_loader = (
        online_receipt_loader or _default_online_receipt_loader
    )
    online_record_validator = (
        online_record_validator or online.validate_online_calibration_record
    )
    staged_receipt_loader = (
        staged_receipt_loader or online.load_strict_hierarchical_result_receipt
    )
    staged_receipt_validator = (
        staged_receipt_validator or validate_hierarchical_result_receipt
    )
    hierarchy_loader = hierarchy_loader or load_hierarchical_artifact
    code_manifest_builder = (
        code_manifest_builder or _default_code_manifest_builder
    )
    python_manifest_builder = (
        python_manifest_builder or _default_python_manifest_builder
    )
    environment_builder = (
        environment_builder or _default_environment_builder
    )
    hooks = (
        online_receipt_loader, online_record_validator,
        staged_receipt_loader, staged_receipt_validator, hierarchy_loader,
        code_manifest_builder, python_manifest_builder, environment_builder,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("official preflight hooks must be callable")

    record = online_receipt_loader(online_path)
    if _canonical_json_bytes(record) != online_snapshot_with_bytes["bytes"]:
        raise ValueError("online calibration receipt bytes changed")
    staged_binding, deployed_binding = _require_online_record_policy(
        record, online_path
    )
    record = online_record_validator(
        record, expected_staged_sha256=staged_binding["sha256"]
    )

    provenance = record.get("provenance")
    before = (
        provenance.get("protected_before")
        if isinstance(provenance, dict) else None
    )
    after = (
        provenance.get("protected_after")
        if isinstance(provenance, dict) else None
    )
    if (not isinstance(before, dict)
            or set(before) != _PROTECTED_ONLINE_NAMES
            or before != after):
        raise ValueError("online protected provenance is incomplete")
    artifacts = {}
    for name in sorted(_PROTECTED_ONLINE_NAMES):
        expected = _validate_snapshot(before[name], "online " + name)
        current = _stable_read_only_snapshot(expected["path"], name)
        if current != expected:
            raise ValueError("online protected {} changed".format(name))
        artifacts[name] = current
    expected_shas = {
        "backbone": AUTHORITATIVE_BACKBONE_SHA256,
        "parent": AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        "geometry": AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    }
    for name, expected in expected_shas.items():
        if artifacts[name]["sha256"] != expected:
            raise ValueError("official {} SHA-256 mismatch".format(name))
    if (artifacts["staged_hierarchical"]["path"]
            != staged_binding["path"]
            or artifacts["staged_hierarchical"]["sha256"]
            != staged_binding["sha256"]):
        raise ValueError("staged hierarchy binding changed")

    staged_receipt = staged_receipt_loader(
        artifacts["staged_result_receipt"]["path"]
    )
    staged_receipt = staged_receipt_validator(staged_receipt)
    if (not isinstance(staged_receipt, dict)
            or staged_receipt.get("selected") != "staged_hierarchical"
            or staged_receipt.get("deployable") is not False
            or staged_receipt.get("validation_data_accessed") is not False
            or not isinstance(staged_receipt.get("calibration"), dict)
            or not isinstance(
                staged_receipt["calibration"].get("gate"), dict
            )
            or staged_receipt["calibration"]["gate"].get("passed") is not True
            or staged_receipt.get("artifact") != {
                "name": Path(staged_binding["path"]).name,
                "sha256": staged_binding["sha256"],
            }):
        raise ValueError("cache calibration receipt did not authorize launch")

    deployed_snapshot = _stable_read_only_snapshot(
        deployed_binding["path"], "deployed hierarchy"
    )
    if deployed_snapshot["sha256"] != deployed_binding["sha256"]:
        raise ValueError("deployed hierarchy SHA-256 mismatch")
    artifacts["hierarchical"] = deployed_snapshot
    artifacts["online_calibration"] = _snapshot_without_bytes(
        online_snapshot_with_bytes
    )
    model, hierarchical_artifact = hierarchy_loader(
        deployed_snapshot["path"],
        device="cpu",
        expected_artifact_sha256=deployed_snapshot["sha256"],
        parent_sha256=expected_shas["parent"],
        geometry_sha256=expected_shas["geometry"],
        expected_deployable=True,
    )
    if not isinstance(model, torch.nn.Module):
        raise ValueError("deployed hierarchy loader returned no model")
    model.eval().requires_grad_(False)
    if (model.training
            or any(parameter.requires_grad for parameter in model.parameters())
            or not isinstance(hierarchical_artifact, dict)
            or hierarchical_artifact.get("deployable") is not True
            or hierarchical_artifact.get("validation_data_accessed") is not False
            or getattr(model, "_artifact_sha256", deployed_snapshot["sha256"])
            != deployed_snapshot["sha256"]):
        raise ValueError("deployed hierarchy strict preflight failed")
    del model

    output = _prepare_absent_run_root(
        run_root, [binding["path"] for binding in artifacts.values()]
    )
    claim_path = online_path.parent / OFFICIAL_CLAIM_NAME
    _reject_symlink_components(claim_path, "official claim")
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError("official claim already exists: {}".format(
            claim_path
        ))
    code_root = Path(OFFICIAL_CODE_ROOT).resolve(strict=True)
    code = code_manifest_builder(code_root)
    python_manifest = python_manifest_builder()
    subprocess_environment = environment_builder(code_root)
    if not isinstance(subprocess_environment, dict):
        raise ValueError("official environment must be a mapping")
    environment = {
        key: subprocess_environment.get(key)
        for key in _CRITICAL_ENVIRONMENT_KEYS
    }
    expected_pythonpath = os.pathsep.join((
        str(code_root), str(code_root / "pointnet2")
    ))
    if (environment["CUDA_VISIBLE_DEVICES"] != "0"
            or environment["OMP_NUM_THREADS"] != "1"
            or not isinstance(environment["PYTHONPATH"], str)
            or not environment["PYTHONPATH"]
            or any(environment[key] is not None for key in (
                "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"))):
        raise ValueError("official CUDA environment is not isolated")
    if environment_builder is _default_environment_builder:
        if environment["PYTHONPATH"] != expected_pythonpath:
            raise ValueError("official PYTHONPATH is not authoritative")
    command = build_authoritative_command(artifacts, output)
    return {
        "online_record": copy.deepcopy(record),
        "staged_receipt": copy.deepcopy(staged_receipt),
        "online_receipt_path": online_path,
        "run_root": output,
        "claim_path": claim_path,
        "stdout_path": output / OFFICIAL_STDOUT_NAME,
        "result_path": output / OFFICIAL_RESULT_NAME,
        "artifacts": artifacts,
        "code_root": code_root,
        "code": code,
        "python": python_manifest,
        "environment": environment,
        "subprocess_environment": subprocess_environment,
        "command": command,
        "inference_uses_ground_truth": False,
    }


def _write_exclusive(path, payload, mode, label):
    path = _absolute_path(path, label)
    _reject_symlink_components(path, label)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _write_exclusive_json(path, value, label):
    return _write_exclusive(path, _canonical_json_bytes(value), 0o444, label)


def _fsync_directory(path):
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _timestamp_base(run_root):
    return Path(run_root) / OFFICIAL_DATASET / OFFICIAL_EXPERIMENT


def _discover_one_timestamp_run(run_root):
    base = _timestamp_base(run_root)
    if not base.is_dir():
        raise ValueError("official timestamp run root was not created")
    runs = sorted(
        path for path in base.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if len(runs) != 1:
        raise ValueError("official launch must create exactly one timestamp run")
    return runs[0].resolve(strict=True)


def _build_claim(context, created_at_utc):
    return {
        "schema": "rec-hierarchical-official-claim-v1",
        "version": 1,
        "created_at_utc": created_at_utc,
        "sample_count": OFFICIAL_SAMPLE_COUNT,
        "required_hits025": OFFICIAL_GATE_HITS025,
        "required_hits050": OFFICIAL_GATE_HITS050,
        "online_calibration_receipt": context["artifacts"][
            "online_calibration"
        ],
        "artifacts": copy.deepcopy(context["artifacts"]),
        "code": copy.deepcopy(context["code"]),
        "python": copy.deepcopy(context["python"]),
        "command": list(context["command"]),
        "environment": copy.deepcopy(context["environment"]),
        "cwd": str(context["code_root"]),
        "run_root": str(context["run_root"]),
        "stdout_path": str(context["stdout_path"]),
        "result_path": str(context["result_path"]),
        "world_size": 1,
        "inference_uses_ground_truth": False,
    }


def _verify_official_inputs_unchanged(context):
    current_artifacts = {
        name: _stable_read_only_snapshot(binding["path"], name)
        for name, binding in context["artifacts"].items()
    }
    if current_artifacts != context["artifacts"]:
        raise RuntimeError("official protected artifacts changed during launch")
    current_code = _default_code_manifest_builder(context["code_root"])
    if current_code != context["code"]:
        raise RuntimeError("official runtime code changed during launch")
    current_python = _default_python_manifest_builder()
    if current_python != context["python"]:
        raise RuntimeError("official Python interpreter changed during launch")
    return context


def _freeze_output_file(path, label):
    logical = _absolute_path(path, label)
    _reject_symlink_components(logical, label)
    initial = os.lstat(str(logical))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(logical), flags)
    try:
        before = os.fstat(descriptor)
        if (stat.S_ISLNK(initial.st_mode)
                or not stat.S_ISREG(initial.st_mode)
                or not stat.S_ISREG(before.st_mode)):
            raise ValueError("{} must be a regular file".format(label))
        if (initial.st_dev, initial.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("{} changed before freezing".format(label))
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(str(logical))
    if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or _file_identity(after) != _file_identity(final)
            or stat.S_IMODE(final.st_mode) != 0o444):
        raise ValueError("{} identity changed while freezing".format(label))
    _fsync_directory(logical.parent)
    snapshot = _stable_read_only_snapshot(
        logical, label, include_bytes=True
    )
    if tuple(snapshot[key] for key in (
            "device", "inode", "mode", "size", "mtime_ns", "ctime_ns",
    )) != (
            int(after.st_dev), int(after.st_ino), stat.S_IMODE(after.st_mode),
            int(after.st_size), int(after.st_mtime_ns), int(after.st_ctime_ns),
    ):
        raise ValueError("{} changed after freezing".format(label))
    return snapshot


def _validate_config(config, context, run_path):
    if not isinstance(config, dict):
        raise ValueError("official config must contain an object")
    mismatches = [
        key for key, expected in OFFICIAL_CONFIG_VALUES.items()
        if (type(config.get(key)) is not type(expected)
            or config.get(key) != expected)
    ]
    expected_paths = {
        "log_dir": str(run_path),
        "data_root": str(OFFICIAL_DATA_ROOT.resolve()),
        "checkpoint_path": context["artifacts"]["backbone"]["path"],
        "rec_reranker_checkpoint": context["artifacts"]["parent"]["path"],
        "rec_geometry_reranker_checkpoint": context["artifacts"][
            "geometry"
        ]["path"],
        "rec_hierarchical_reranker_checkpoint": context["artifacts"][
            "hierarchical"
        ]["path"],
    }
    for key, expected in expected_paths.items():
        value = config.get(key)
        if (not isinstance(value, str)
                or str(Path(value).expanduser().resolve()) != expected):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "official config contract mismatch: {}".format(
                ", ".join(sorted(set(mismatches)))
            )
        )
    return copy.deepcopy(config)


def _require_dataset_marker(text, label):
    if len(re.findall(
            r"length of testing dataset:\s*9508(?:\D|$)", text)) != 1:
        raise ValueError("{} must prove the 9,508-row dataset once".format(label))


def _result_record(context, claim_snapshot, run_path, files, metrics,
                   created_at_utc):
    hits025 = metrics["hits025"]
    hits050 = metrics["hits050"]
    return {
        "schema": OFFICIAL_RESULT_SCHEMA,
        "version": OFFICIAL_RESULT_VERSION,
        "created_at_utc": created_at_utc,
        "sample_count": OFFICIAL_SAMPLE_COUNT,
        "printed_acc025": metrics["printed_acc025"],
        "printed_acc050": metrics["printed_acc050"],
        "hits025": hits025,
        "hits050": hits050,
        "gate025_pass": hits025 >= OFFICIAL_GATE_HITS025,
        "gate050_pass": hits050 >= OFFICIAL_GATE_HITS050,
        "acceptance_gate_pass": acceptance_gate_pass(
            hits025, hits050, OFFICIAL_SAMPLE_COUNT, False
        ),
        "position_subgroups": copy.deepcopy(metrics["position_subgroups"]),
        "inference_uses_ground_truth": False,
        "run": {
            "path": str(run_path),
            "timestamp": int(run_path.name),
            "dataset": OFFICIAL_DATASET,
            "experiment": OFFICIAL_EXPERIMENT,
        },
        "files": copy.deepcopy(files),
        "artifacts": copy.deepcopy(context["artifacts"]),
        "code": copy.deepcopy(context["code"]),
        "launch": {
            "claim_path": claim_snapshot["path"],
            "claim_sha256": claim_snapshot["sha256"],
            "online_calibration_receipt": context["artifacts"][
                "online_calibration"
            ],
            "cwd": str(context["code_root"]),
            "command": list(context["command"]),
            "python": copy.deepcopy(context["python"]),
            "environment": copy.deepcopy(context["environment"]),
            "world_size": 1,
            "local_rank": 0,
            "batch_size": 12,
        },
    }


def validate_official_result(record):
    """Validate all derived official metrics and immutable bindings."""
    if (not isinstance(record, dict) or set(record) != _RESULT_FIELDS
            or record.get("schema") != OFFICIAL_RESULT_SCHEMA
            or type(record.get("version")) is not int
            or record["version"] != OFFICIAL_RESULT_VERSION
            or record.get("sample_count") != OFFICIAL_SAMPLE_COUNT
            or record.get("inference_uses_ground_truth") is not False):
        raise ValueError("official result top-level contract is invalid")
    hits025 = record.get("hits025")
    hits050 = record.get("hits050")
    if (type(hits025) is not int or type(hits050) is not int
            or not 0 <= hits050 <= hits025 <= OFFICIAL_SAMPLE_COUNT
            or recover_exact_hits(record.get("printed_acc025")) != hits025
            or recover_exact_hits(record.get("printed_acc050")) != hits050
            or record.get("gate025_pass")
            is not (hits025 >= OFFICIAL_GATE_HITS025)
            or record.get("gate050_pass")
            is not (hits050 >= OFFICIAL_GATE_HITS050)
            or record.get("acceptance_gate_pass")
            is not acceptance_gate_pass(hits025, hits050, 9508, False)):
        raise ValueError("official result metrics or integer gate changed")
    _validate_subgroup_reconciliation(
        record.get("position_subgroups"), hits025, hits050
    )
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACT_NAMES:
        raise ValueError("official result artifact set changed")
    for name, binding in artifacts.items():
        _validate_snapshot(binding, name)
    files = record.get("files")
    if (not isinstance(files, dict)
            or set(files) != {"claim", "stdout", "config", "log"}):
        raise ValueError("official result file set changed")
    for name, binding in files.items():
        _validate_snapshot(binding, name)
    run = record.get("run")
    if (not isinstance(run, dict)
            or set(run) != {"path", "timestamp", "dataset", "experiment"}
            or run.get("dataset") != OFFICIAL_DATASET
            or run.get("experiment") != OFFICIAL_EXPERIMENT
            or type(run.get("timestamp")) is not int
            or Path(run.get("path", "")).name != str(run["timestamp"])):
        raise ValueError("official result run binding changed")
    launch = record.get("launch")
    if (not isinstance(launch, dict)
            or launch.get("world_size") != 1
            or launch.get("local_rank") != 0
            or launch.get("batch_size") != 12
            or launch.get("claim_path") != files["claim"]["path"]
            or launch.get("claim_sha256") != files["claim"]["sha256"]
            or launch.get("online_calibration_receipt")
            != artifacts["online_calibration"]
            or not isinstance(launch.get("command"), list)
            or "--eval_use_rec_hierarchical_reranker_scores"
            not in launch["command"]
            or "--eval_use_rec_selective_residual_scores"
            in launch["command"]):
        raise ValueError("official result launch binding changed")
    if not isinstance(record.get("code"), dict):
        raise ValueError("official result code manifest is invalid")
    _canonical_json_bytes(record)
    return copy.deepcopy(record)


def run_official_evaluation(
        online_calibration_receipt, run_root, *, preflight_builder=None,
        subprocess_runner=None, postflight_verifier=None, utc_now=None):
    """Claim, run, verify, and seal the sole official evaluation."""
    preflight_builder = preflight_builder or preflight_official_inputs
    subprocess_runner = subprocess_runner or subprocess.run
    postflight_verifier = (
        postflight_verifier or _verify_official_inputs_unchanged
    )
    utc_now = utc_now or _utc_now
    if not all(callable(value) for value in (
            preflight_builder, subprocess_runner, postflight_verifier,
            utc_now)):
        raise ValueError("official runner hooks must be callable")
    context = preflight_builder(online_calibration_receipt, run_root)
    if not isinstance(context, dict):
        raise ValueError("official preflight returned invalid context")
    claim_path = Path(context["claim_path"])
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError("official claim already exists: {}".format(
            claim_path
        ))
    if Path(context["run_root"]).exists():
        raise FileExistsError("official run root already exists")
    claim = _build_claim(context, utc_now())
    _write_exclusive_json(claim_path, claim, "official claim")
    claim_snapshot_with_bytes = _stable_read_only_snapshot(
        claim_path, "official claim", include_bytes=True
    )
    if claim_snapshot_with_bytes["bytes"] != _canonical_json_bytes(claim):
        raise RuntimeError("official claim readback changed")
    claim_snapshot = _snapshot_without_bytes(claim_snapshot_with_bytes)

    os.mkdir(str(context["run_root"]), 0o700)
    stdout_path = Path(context["stdout_path"])
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(stdout_path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stdout_handle:
            completed = subprocess_runner(
                list(context["command"]),
                cwd=str(context["code_root"]),
                env=dict(context.get(
                    "subprocess_environment", context["environment"]
                )),
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            stdout_handle.flush()
            os.fsync(stdout_handle.fileno())
    finally:
        os.close(descriptor)
    if (not hasattr(completed, "returncode")
            or type(completed.returncode) is not int
            or completed.returncode != 0):
        raise RuntimeError("official subprocess failed")

    stdout_snapshot_with_bytes = _freeze_output_file(
        stdout_path, "official stdout"
    )
    run_path = _discover_one_timestamp_run(context["run_root"])
    config_snapshot_with_bytes = _freeze_output_file(
        run_path / "config.json", "official config"
    )
    log_snapshot_with_bytes = _freeze_output_file(
        run_path / "log.txt", "official log"
    )
    postflight_verifier(context)
    try:
        config = json.loads(
            config_snapshot_with_bytes["bytes"].decode("utf-8")
        )
        log_text = log_snapshot_with_bytes["bytes"].decode("utf-8")
        stdout_text = stdout_snapshot_with_bytes["bytes"].decode(
            "utf-8", errors="replace"
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("official output decoding failed: {}".format(error))
    _validate_config(config, context, run_path)
    _require_dataset_marker(log_text, "official log")
    _require_dataset_marker(stdout_text, "official stdout")
    metrics = parse_official_evidence(log_text, stdout_text)
    files = {
        "claim": claim_snapshot,
        "stdout": _snapshot_without_bytes(stdout_snapshot_with_bytes),
        "config": _snapshot_without_bytes(config_snapshot_with_bytes),
        "log": _snapshot_without_bytes(log_snapshot_with_bytes),
    }
    result = _result_record(
        context, claim_snapshot, run_path, files, metrics, utc_now()
    )
    result = validate_official_result(result)
    _write_exclusive_json(context["result_path"], result, "official result")
    result_snapshot = _stable_read_only_snapshot(
        context["result_path"], "official result", include_bytes=True
    )
    if result_snapshot["bytes"] != _canonical_json_bytes(result):
        raise RuntimeError("official result readback changed")
    _fsync_directory(context["run_root"])
    return result


_CLI_OPTIONS = {
    "--online-calibration-receipt", "--run-root",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one sealed frozen hierarchical official evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--online-calibration-receipt", required=True)
    parser.add_argument("--run-root", required=True)
    raw = list(sys.argv[1:] if argv is None else argv)
    option_tokens = [
        token for token in raw
        if isinstance(token, str) and token.startswith("--")
    ]
    if (any(token not in _CLI_OPTIONS for token in option_tokens)
            or any(option_tokens.count(option) != 1
                   for option in _CLI_OPTIONS)):
        parser.error("exactly one of each official path option is required")
    return parser.parse_args(raw)


def main(argv=None):
    args = parse_args(argv)
    result = run_official_evaluation(
        args.online_calibration_receipt, args.run_root
    )
    sys.stdout.write(_canonical_json_bytes(result).decode("ascii") + "\n")
    return result


if __name__ == "__main__":
    main()
