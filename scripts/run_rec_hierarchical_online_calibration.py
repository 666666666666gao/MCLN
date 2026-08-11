#!/usr/bin/env python
"""Calibrate and promote a frozen hierarchical REC reranker on train only."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from models import rec_source_gate
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_SHA256,
    AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
    load_hierarchical_artifact,
    validate_hierarchical_artifact,
    validate_hierarchical_result_receipt,
)


ONLINE_CALIBRATION_SCHEMA = "rec-hierarchical-online-calibration-v1"
ONLINE_CALIBRATION_VERSION = 1
ONLINE_CALIBRATION_NAME = "online-calibration.json"
DEPLOYED_ARTIFACT_NAME = "deployed_hierarchical_reranker.pth"
ONLINE_SAMPLE_COUNT = 3625
ONLINE_BASELINE_HITS025 = 3461
ONLINE_BASELINE_HITS050 = 3316
ONLINE_REQUIRED_HITS025 = 3524
ONLINE_REQUIRED_HITS050 = 3316
ONLINE_ORACLE_HITS025 = 3606
ONLINE_ORACLE_HITS050 = 3588
ONLINE_RAW_QUERY_HITS025 = 3615
ONLINE_RAW_QUERY_HITS050 = 3559
ONLINE_RAW_QUERY_IOU_SHA256 = (
    "7a75f033a7afb2b1871e971b2797e544b"
    "ac8bb59e6a20aa68735eb23842d5751"
)
ONLINE_BASELINE_SELECTED_IOU_SHA256 = (
    "2e8c815bdc2151f05358fad9007b1ad0"
    "ad9075a367b20afed7bf504a78fb23fb"
)
AUTHORITATIVE_SOURCE_GATE_RECEIPT_SHA256 = (
    "a57e34b356bf1bc04afdf8a968bc474a"
    "7f8e52a54811f8f263c830f302322f2c"
)
ONLINE_INVARIANT_FIELDS = (
    "oracle_hits025",
    "oracle_hits050",
    "raw_query_hits025",
    "raw_query_hits050",
    "candidate_iou_sha256",
    "raw_query_iou_sha256",
    "row_materialization_sha256",
)
_ONLINE_EVIDENCE_FIELDS = {
    "sample_count",
    "hits025",
    "hits050",
    "oracle_hits025",
    "oracle_hits050",
    "raw_query_hits025",
    "raw_query_hits050",
    "candidate_iou_sha256",
    "raw_query_iou_sha256",
    "row_materialization_sha256",
    "selected_iou_sha256",
}
_ONLINE_RECORD_FIELDS = {
    "schema",
    "version",
    "sample_count",
    "baseline",
    "candidate",
    "staged_artifact",
    "deployed_artifact",
    "gate",
    "provenance",
    "validation_data_accessed",
    "inference_uses_ground_truth",
}
_ONLINE_PROVENANCE_FIELDS = {
    "command",
    "environment",
    "code",
    "protected_before",
    "protected_after",
    "source_gate_baseline_receipt",
    "staged_result_receipt",
}
_ONLINE_PROTECTED_NAMES = {
    "backbone",
    "parent",
    "geometry",
    "staged_hierarchical",
    "staged_result_receipt",
    "source_gate_baseline",
}
_ONLINE_SNAPSHOT_FIELDS = {
    "path",
    "device",
    "inode",
    "mode",
    "size",
    "mtime_ns",
    "ctime_ns",
    "sha256",
}
_ONLINE_OBSERVATION_FIELDS = {
    "baseline_selected_ious",
    "candidate_selected_ious",
    "candidate_ious",
    "candidate_valid",
    "raw_query_ious",
    "row_materialization",
}


@dataclass(frozen=True)
class OnlineCalibrationGateResult:
    passed: bool
    failures: tuple
    required_hits025: int = ONLINE_REQUIRED_HITS025
    required_hits050: int = ONLINE_REQUIRED_HITS050


def _is_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or value.lower() != value):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_evidence(value, label):
    if not isinstance(value, dict) or set(value) != _ONLINE_EVIDENCE_FIELDS:
        raise ValueError("{} online evidence fields are invalid".format(label))
    for field in (
            "sample_count", "hits025", "hits050", "oracle_hits025",
            "oracle_hits050", "raw_query_hits025", "raw_query_hits050"):
        item = value[field]
        if (type(item) is not int or item < 0
                or (field != "sample_count" and item > value["sample_count"])):
            raise ValueError(
                "{} {} is invalid".format(label, field)
            )
    if value["sample_count"] <= 0:
        raise ValueError("{} sample_count must be positive".format(label))
    if value["hits050"] > value["hits025"]:
        raise ValueError("{} Top-1 threshold hits are not nested".format(label))
    if value["oracle_hits050"] > value["oracle_hits025"]:
        raise ValueError("{} oracle threshold hits are not nested".format(label))
    if value["raw_query_hits050"] > value["raw_query_hits025"]:
        raise ValueError(
            "{} raw-query threshold hits are not nested".format(label)
        )
    for field in (
            "candidate_iou_sha256", "raw_query_iou_sha256",
            "row_materialization_sha256", "selected_iou_sha256"):
        if not _is_sha256(value[field]):
            raise ValueError("{} {} is invalid".format(label, field))
    return copy.deepcopy(value)


def online_calibration_gate(baseline, candidate):
    """Apply the fixed online train-calibration gate without tuning."""
    baseline = _validate_evidence(baseline, "baseline")
    candidate = _validate_evidence(candidate, "candidate")
    failures = []
    if baseline["sample_count"] != ONLINE_SAMPLE_COUNT:
        failures.append("baseline_sample_count")
    if baseline["hits025"] != ONLINE_BASELINE_HITS025:
        failures.append("baseline_hits025")
    if baseline["hits050"] != ONLINE_BASELINE_HITS050:
        failures.append("baseline_hits050")
    if baseline["oracle_hits025"] != ONLINE_ORACLE_HITS025:
        failures.append("baseline_oracle_hits025")
    if baseline["oracle_hits050"] != ONLINE_ORACLE_HITS050:
        failures.append("baseline_oracle_hits050")
    if baseline["raw_query_hits025"] != ONLINE_RAW_QUERY_HITS025:
        failures.append("baseline_raw_query_hits025")
    if baseline["raw_query_hits050"] != ONLINE_RAW_QUERY_HITS050:
        failures.append("baseline_raw_query_hits050")
    if baseline["raw_query_iou_sha256"] != ONLINE_RAW_QUERY_IOU_SHA256:
        failures.append("baseline_raw_query_iou_sha256")
    if (baseline["selected_iou_sha256"]
            != ONLINE_BASELINE_SELECTED_IOU_SHA256):
        failures.append("baseline_selected_iou_sha256")
    if candidate["sample_count"] != ONLINE_SAMPLE_COUNT:
        failures.append("sample_count")
    if candidate["hits025"] < ONLINE_REQUIRED_HITS025:
        failures.append("hits025")
    if candidate["hits050"] < ONLINE_REQUIRED_HITS050:
        failures.append("hits050")
    for field in ONLINE_INVARIANT_FIELDS:
        if candidate[field] != baseline[field]:
            failures.append(field)
    failures = tuple(dict.fromkeys(failures))
    return OnlineCalibrationGateResult(not failures, failures)


def validate_source_gate_baseline_evidence(receipt):
    """Extract the exact restored source-gate calibration reproduction."""
    if (not isinstance(receipt, dict)
            or receipt.get("validation_data_accessed") is not False
            or receipt.get("checkpoint_written") is not False
            or receipt.get("deployable") is not False):
        raise ValueError("source-gate baseline receipt policy is invalid")
    calibration = receipt.get("calibration")
    reproduced = (
        calibration.get("reproduced")
        if isinstance(calibration, dict) else None
    )
    metrics = reproduced.get("metrics") if isinstance(reproduced, dict) else None
    top1 = metrics.get("top1") if isinstance(metrics, dict) else None
    candidate_oracle = (
        metrics.get("candidate_oracle") if isinstance(metrics, dict) else None
    )
    geometry = top1.get("geometry") if isinstance(top1, dict) else None
    oracle = (
        candidate_oracle.get("geometry_candidate")
        if isinstance(candidate_oracle, dict) else None
    )
    raw = (
        candidate_oracle.get("raw_query")
        if isinstance(candidate_oracle, dict) else None
    )
    digests = (
        reproduced.get("digests") if isinstance(reproduced, dict) else None
    )
    try:
        evidence = {
            "sample_count": reproduced["sample_count"],
            "hits025": geometry["hits025"],
            "hits050": geometry["hits050"],
            "oracle_hits025": oracle["hits025"],
            "oracle_hits050": oracle["hits050"],
            "raw_query_hits025": raw["hits025"],
            "raw_query_hits050": raw["hits050"],
            "raw_query_iou_sha256": digests[
                "raw_query_ious_sha256"
            ],
            "selected_iou_sha256": digests[
                "geometry_selected_ious_sha256"
            ],
        }
    except (KeyError, TypeError):
        raise ValueError("source-gate reproduced evidence is incomplete")
    expected = {
        "sample_count": ONLINE_SAMPLE_COUNT,
        "hits025": ONLINE_BASELINE_HITS025,
        "hits050": ONLINE_BASELINE_HITS050,
        "oracle_hits025": ONLINE_ORACLE_HITS025,
        "oracle_hits050": ONLINE_ORACLE_HITS050,
        "raw_query_hits025": ONLINE_RAW_QUERY_HITS025,
        "raw_query_hits050": ONLINE_RAW_QUERY_HITS050,
        "raw_query_iou_sha256": ONLINE_RAW_QUERY_IOU_SHA256,
        "selected_iou_sha256": ONLINE_BASELINE_SELECTED_IOU_SHA256,
    }
    if any(type(evidence[name]) is not int for name in (
            "sample_count", "hits025", "hits050", "oracle_hits025",
            "oracle_hits050", "raw_query_hits025", "raw_query_hits050")):
        raise ValueError("source-gate reproduced counts are invalid")
    if evidence != expected:
        raise ValueError("source-gate reproduced evidence is not authoritative")
    return copy.deepcopy(evidence)


def _reject_duplicate_json_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key: {}".format(name))
        result[name] = value
    return result


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


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


def _load_strict_canonical_json(path, label):
    resolved, _digest, _identity = _regular_file_sha256(
        path, label, required_mode=0o444
    )
    try:
        encoded = resolved.read_bytes()
        value = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("{} reload failed: {}".format(label, error))
    if _canonical_json_bytes(value) != encoded:
        raise ValueError("{} bytes are not canonical ASCII JSON".format(label))
    return value


def load_strict_hierarchical_result_receipt(path):
    """Strict-load and validate one cache-gated staged result receipt."""
    return validate_hierarchical_result_receipt(
        _load_strict_canonical_json(path, "staged result receipt")
    )


def load_strict_source_gate_baseline_receipt(path):
    """Strict-load the authoritative source-gate receipt."""
    from scripts.probe_scanrefer_rec_source_gate import (
        load_strict_source_gate_receipt,
    )

    return load_strict_source_gate_receipt(path)


def _capture_protected_snapshot(path, label):
    resolved, digest, identity = _regular_file_sha256(
        path, label, required_mode=0o444
    )
    metadata = os.lstat(str(resolved))
    current_identity = (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )
    if current_identity != tuple(int(value) for value in identity):
        raise ValueError("{} changed after stable hash".format(label))
    snapshot = {
        "path": str(resolved),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "sha256": digest,
    }
    return _validate_protected_snapshot(snapshot, label)


def _capture_online_protected_inputs(paths):
    if not isinstance(paths, dict) or set(paths) != _ONLINE_PROTECTED_NAMES:
        raise ValueError("online protected paths are incomplete")
    snapshots = {
        name: _capture_protected_snapshot(paths[name], name.replace("_", " "))
        for name in sorted(paths)
    }
    if len({record["path"] for record in snapshots.values()}) != len(snapshots):
        raise ValueError("online protected inputs must be distinct")
    return snapshots


def build_online_calibration_code_manifest():
    """Hash the online runner and every imported inference dependency."""
    from scripts.probe_scanrefer_rec_source_gate import (
        build_source_gate_code_manifest,
    )

    files = build_source_gate_code_manifest()
    for name, path in (
            ("online_runner", Path(__file__).resolve()),
            ("hierarchical_trainer", Path(
                sys.modules[
                    "scripts.train_scanrefer_rec_hierarchical_reranker"
                ].__file__
            ).resolve())):
        resolved, digest, _identity = _regular_file_sha256(path, name)
        files[name] = {"path": str(resolved), "sha256": digest}
    return {"files": files, "sha256": _canonical_json_sha256(files)}


def build_online_calibration_command(args):
    """Return the exact train-only online-calibration command."""
    return [
        str(Path(sys.executable).absolute()),
        str(Path(__file__).absolute()),
        "--data-root", str(args.data_root),
        "--backbone-checkpoint", str(args.backbone_checkpoint),
        "--parent-artifact", str(args.parent_artifact),
        "--geometry-artifact", str(args.geometry_artifact),
        "--staged-hierarchical-artifact",
        str(args.staged_hierarchical_artifact),
        "--staged-result-receipt", str(args.staged_result_receipt),
        "--source-gate-baseline-receipt",
        str(args.source_gate_baseline_receipt),
        "--output-dir", str(args.output_dir),
        "--device", str(args.device),
    ]


def build_online_calibration_environment(args):
    """Capture the fixed CUDA execution environment used by calibration."""
    return {
        "device": str(args.device),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "python": str(Path(sys.executable).absolute()),
    }


def _absolute_directory(path, label):
    try:
        logical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("{} is invalid: {}".format(label, error))
    if logical.is_symlink() or not logical.is_dir():
        raise ValueError("{} must be an existing non-symlink directory".format(
            label
        ))
    return logical.resolve(strict=True)


def _absolute_absent_output(path):
    try:
        output = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("output directory is invalid: {}".format(error))
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "online output directory must not exist: {}".format(output)
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("online output parent must be a real directory")
    return output.absolute()


def _freeze_online_model(model, label):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("{} must be a torch module".format(label))
    model.eval().requires_grad_(False)
    if model.training or any(
            parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("{} did not freeze for calibration".format(label))
    return model


def _validate_staged_receipt_binding(receipt, staged_path, staged_sha256):
    if (not isinstance(receipt, dict)
            or receipt.get("selected") != "staged_hierarchical"
            or receipt.get("deployable") is not False
            or receipt.get("validation_data_accessed") is not False):
        raise ValueError("staged result receipt policy is invalid")
    artifact = receipt.get("artifact")
    calibration = receipt.get("calibration")
    gate = calibration.get("gate") if isinstance(calibration, dict) else None
    if (not isinstance(artifact, dict)
            or artifact.get("name") != staged_path.name
            or artifact.get("sha256") != staged_sha256
            or not isinstance(gate, dict)
            or gate.get("passed") is not True):
        raise ValueError("staged result receipt binding is invalid")
    return copy.deepcopy(receipt)


def initialize_live_online_calibration(
        args, *, initial_state_loader=None, data_builder=None,
        data_contract_builder=None, hierarchy_loader=None,
        staged_receipt_loader=None, staged_receipt_validator=None,
        source_receipt_loader=None, code_manifest_builder=None,
        command_builder=None, environment_builder=None):
    """Preflight immutable inputs and build only the train calibration view."""
    if args is None or getattr(args, "device", None) != "cuda:0":
        raise ValueError("online calibration requires device cuda:0")
    runtime_device = str(args.device)
    data_root = _absolute_directory(args.data_root, "data root")
    output_dir = _absolute_absent_output(args.output_dir)
    protected_paths = {
        "backbone": Path(args.backbone_checkpoint).expanduser().absolute(),
        "parent": Path(args.parent_artifact).expanduser().absolute(),
        "geometry": Path(args.geometry_artifact).expanduser().absolute(),
        "staged_hierarchical": Path(
            args.staged_hierarchical_artifact
        ).expanduser().absolute(),
        "staged_result_receipt": Path(
            args.staged_result_receipt
        ).expanduser().absolute(),
        "source_gate_baseline": Path(
            args.source_gate_baseline_receipt
        ).expanduser().absolute(),
    }
    before = _capture_online_protected_inputs(protected_paths)
    expected_hashes = {
        "backbone": AUTHORITATIVE_BACKBONE_SHA256,
        "parent": AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        "geometry": AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        "source_gate_baseline": AUTHORITATIVE_SOURCE_GATE_RECEIPT_SHA256,
    }
    for name, expected in expected_hashes.items():
        if before[name]["sha256"] != expected:
            raise ValueError("{} SHA-256 is not authoritative".format(name))
    protected_roots = (data_root,) + tuple(
        Path(record["path"]) for record in before.values()
    )
    for protected in protected_roots:
        try:
            common = os.path.commonpath((str(output_dir), str(protected)))
        except ValueError:
            continue
        if common in (str(output_dir), str(protected)):
            raise ValueError("online output overlaps protected input")

    from scripts import probe_scanrefer_rec_source_gate as source_gate_probe
    from scripts import train_scanrefer_rec_finetune as legacy

    initial_state_loader = (
        initial_state_loader or legacy.load_rec_finetune_initial_state
    )
    data_builder = (
        data_builder or source_gate_probe.build_source_gate_train_only_data
    )
    data_contract_builder = (
        data_contract_builder
        or source_gate_probe.build_source_gate_train_data_contract
    )
    hierarchy_loader = hierarchy_loader or load_hierarchical_artifact
    staged_receipt_loader = (
        staged_receipt_loader or load_strict_hierarchical_result_receipt
    )
    staged_receipt_validator = (
        staged_receipt_validator or validate_hierarchical_result_receipt
    )
    source_receipt_loader = (
        source_receipt_loader or load_strict_source_gate_baseline_receipt
    )
    code_manifest_builder = (
        code_manifest_builder or build_online_calibration_code_manifest
    )
    command_builder = command_builder or build_online_calibration_command
    environment_builder = (
        environment_builder or build_online_calibration_environment
    )
    hooks = (
        initial_state_loader, data_builder, data_contract_builder,
        hierarchy_loader, staged_receipt_loader, staged_receipt_validator,
        source_receipt_loader, code_manifest_builder, command_builder,
        environment_builder,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("online initialization hooks must be callable")

    code = code_manifest_builder()
    staged_receipt = staged_receipt_validator(
        staged_receipt_loader(protected_paths["staged_result_receipt"])
    )
    staged_path = Path(before["staged_hierarchical"]["path"])
    staged_sha256 = before["staged_hierarchical"]["sha256"]
    _validate_staged_receipt_binding(
        staged_receipt, staged_path, staged_sha256
    )
    source_receipt = source_receipt_loader(
        protected_paths["source_gate_baseline"]
    )
    source_baseline = validate_source_gate_baseline_evidence(source_receipt)

    state = initial_state_loader(
        Path(before["backbone"]["path"]),
        Path(before["parent"]["path"]),
        Path(before["geometry"]["path"]),
        data_root,
        device=runtime_device,
    )
    if not isinstance(state, dict) or not {
            "config", "mcln", "parent", "parent_artifact",
            "geometry", "geometry_artifact",
    }.issubset(state):
        raise ValueError("online initial frozen state is incomplete")
    if (state.get("checkpoint_sha256", expected_hashes["backbone"])
            != expected_hashes["backbone"]):
        raise ValueError("loaded backbone SHA-256 changed")
    for name, model, expected in (
            ("parent", state["parent"], expected_hashes["parent"]),
            ("geometry", state["geometry"], expected_hashes["geometry"])):
        loaded_sha = getattr(model, "_artifact_sha256", expected)
        if loaded_sha != expected:
            raise ValueError("loaded {} SHA-256 changed".format(name))
    _freeze_online_model(state["mcln"], "MCLN")
    _freeze_online_model(state["parent"], "parent reranker")
    _freeze_online_model(state["geometry"], "geometry reranker")
    state.pop("groups", None)
    state.pop("optimizer", None)

    hierarchical_model, hierarchical_artifact = hierarchy_loader(
        staged_path,
        device=runtime_device,
        expected_geometry_feature_names=state[
            "geometry_artifact"
        ].get("feature_names"),
        expected_artifact_sha256=staged_sha256,
        parent_sha256=expected_hashes["parent"],
        geometry_sha256=expected_hashes["geometry"],
        expected_deployable=False,
    )
    if (not isinstance(hierarchical_artifact, dict)
            or hierarchical_artifact.get("deployable") is not False
            or getattr(hierarchical_model, "_artifact_sha256", staged_sha256)
            != staged_sha256):
        raise ValueError("loaded staged hierarchy binding changed")
    _freeze_online_model(hierarchical_model, "hierarchical reranker")

    data = data_builder(state["config"], runtime_device)
    expected_data_fields = {
        "dataset", "split", "fit_view", "calibration_view",
        "fit_loader", "calibration_loader",
    }
    if not isinstance(data, dict) or set(data) != expected_data_fields:
        raise ValueError("online data is not the sole train-only view")
    initialized = {
        "paths": {
            "data_root": data_root,
            "output_dir": output_dir,
            **protected_paths,
        },
        "device": runtime_device,
        "initial_state": state,
        "data": data,
        "hierarchical_model": hierarchical_model,
        "hierarchical_artifact": hierarchical_artifact,
        "staged_artifact_path": staged_path,
        "staged_artifact_sha256": staged_sha256,
        "staged_result_receipt": staged_receipt,
        "source_gate_receipt": source_receipt,
        "source_gate_baseline": source_baseline,
        "protected_paths": protected_paths,
    }
    contract = data_contract_builder(initialized)
    if (not isinstance(contract, dict)
            or contract.get("calibration_sample_count") != ONLINE_SAMPLE_COUNT
            or contract.get("validation_data_accessed") is not False
            or len(tuple(getattr(data["calibration_view"], "indices", ())))
            != ONLINE_SAMPLE_COUNT):
        raise ValueError("online train calibration contract is inexact")
    initialized["train_data_contract"] = copy.deepcopy(contract)

    after = _capture_online_protected_inputs(protected_paths)
    if after != before:
        raise RuntimeError("online protected inputs changed during initialization")
    provenance = {
        "command": command_builder(args),
        "environment": environment_builder(args),
        "code": code,
        "protected_before": before,
        "protected_after": after,
        "source_gate_baseline_receipt": {
            "path": before["source_gate_baseline"]["path"],
            "sha256": before["source_gate_baseline"]["sha256"],
        },
        "staged_result_receipt": {
            "path": before["staged_result_receipt"]["path"],
            "sha256": before["staged_result_receipt"]["sha256"],
        },
    }
    initialized["provenance"] = _validate_online_provenance(
        provenance,
        {"path": str(staged_path), "sha256": staged_sha256,
         "deployable": False},
    )
    return initialized


def _new_digest(label):
    digest = hashlib.sha256()
    payload = label.encode("ascii")
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)
    return digest


def _digest_tensor(digest, dataset_index, value):
    tensor = value.detach().to(device="cpu").contiguous()
    digest.update(struct.pack("<Q", int(dataset_index)))
    dtype = str(tensor.dtype).encode("ascii")
    digest.update(struct.pack("<Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<Q", tensor.dim()))
    for dimension in tensor.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    if tensor.numel():
        digest.update(tensor.numpy().tobytes(order="C"))


def _ordered_indices(value, label):
    if isinstance(value, torch.Tensor):
        if value.dim() != 1:
            raise ValueError("{} must be one-dimensional".format(label))
        value = value.detach().cpu().tolist()
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError("{} must contain exact integers".format(label))
    if (not result or any(isinstance(item, bool) for item in value)
            or len(set(result)) != len(result)):
        raise ValueError("{} must contain distinct integers".format(label))
    return result


class OnlineCalibrationAccumulator:
    """Accumulate ordered live baseline/candidate evidence in one pass."""

    def __init__(self, expected_indices):
        self._expected_indices = _ordered_indices(
            expected_indices, "expected indices"
        )
        self._cursor = 0
        self._counts = {
            "baseline025": 0,
            "baseline050": 0,
            "candidate025": 0,
            "candidate050": 0,
            "oracle025": 0,
            "oracle050": 0,
            "raw025": 0,
            "raw050": 0,
        }
        self._digests = {
            "candidate": _new_digest("candidate_ious_and_valid-v1"),
            "raw": rec_source_gate._new_calibration_digest(
                "raw_query_ious"
            ),
            "materialization": _new_digest("row_materialization-v1"),
            "baseline_selected": rec_source_gate._new_calibration_digest(
                "geometry_selected_ious"
            ),
            "candidate_selected": rec_source_gate._new_calibration_digest(
                "geometry_selected_ious"
            ),
        }
        self._candidate_width = None
        self._raw_width = None
        self._materialization_shape = None
        self._finalized = False
        self._report = None

    def update(self, dataset_indices, observation):
        if self._finalized:
            raise RuntimeError("online calibration is finalized")
        indices = _ordered_indices(dataset_indices, "dataset indices")
        expected = self._expected_indices[
            self._cursor:self._cursor + len(indices)
        ]
        if indices != expected:
            raise ValueError("online calibration dataset index order changed")
        if (not isinstance(observation, dict)
                or set(observation) != _ONLINE_OBSERVATION_FIELDS):
            raise ValueError("online calibration observation fields changed")
        values = observation
        batch_size = len(indices)
        baseline = values["baseline_selected_ious"]
        candidate = values["candidate_selected_ious"]
        candidate_ious = values["candidate_ious"]
        candidate_valid = values["candidate_valid"]
        raw_ious = values["raw_query_ious"]
        materialization = values["row_materialization"]
        tensors = (
            baseline, candidate, candidate_ious, candidate_valid,
            raw_ious, materialization,
        )
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("online calibration observations must be tensors")
        if (tuple(baseline.shape) != (batch_size,)
                or tuple(candidate.shape) != (batch_size,)
                or candidate_ious.dim() != 2
                or candidate_ious.shape[0] != batch_size
                or candidate_valid.dtype != torch.bool
                or candidate_valid.shape != candidate_ious.shape
                or raw_ious.dim() != 2 or raw_ious.shape[0] != batch_size
                or materialization.dim() < 2
                or materialization.shape[0] != batch_size):
            raise ValueError("online calibration tensor shapes are invalid")
        if not bool(candidate_valid.any(dim=1).all().item()):
            raise ValueError("every online row needs a valid candidate")
        finite_values = (
            baseline, candidate, candidate_ious[candidate_valid], raw_ious,
            materialization,
        )
        if any(not bool(torch.isfinite(value).all().item())
               for value in finite_values):
            raise ValueError("online calibration tensors must be finite")
        if (bool((candidate_ious[candidate_valid] < 0.0).any().item())
                or bool((candidate_ious[candidate_valid] > 1.0).any().item())
                or bool((raw_ious < 0.0).any().item())
                or bool((raw_ious > 1.0).any().item())):
            raise ValueError("online calibration IoUs must lie in [0,1]")
        widths = (int(candidate_ious.shape[1]), int(raw_ious.shape[1]))
        materialization_shape = tuple(materialization.shape[1:])
        if self._candidate_width is None:
            self._candidate_width, self._raw_width = widths
            self._materialization_shape = materialization_shape
        elif (widths != (self._candidate_width, self._raw_width)
              or materialization_shape != self._materialization_shape):
            raise ValueError("online calibration tensor layout changed")

        masked_candidates = candidate_ious.masked_fill(
            ~candidate_valid, -float("inf")
        )
        oracle = masked_candidates.max(dim=1).values
        raw = raw_ious.max(dim=1).values
        self._counts["baseline025"] += int(baseline.gt(0.25).sum().item())
        self._counts["baseline050"] += int(baseline.gt(0.50).sum().item())
        self._counts["candidate025"] += int(candidate.gt(0.25).sum().item())
        self._counts["candidate050"] += int(candidate.gt(0.50).sum().item())
        self._counts["oracle025"] += int(oracle.gt(0.25).sum().item())
        self._counts["oracle050"] += int(oracle.gt(0.50).sum().item())
        self._counts["raw025"] += int(raw.gt(0.25).sum().item())
        self._counts["raw050"] += int(raw.gt(0.50).sum().item())
        for offset, dataset_index in enumerate(indices):
            _digest_tensor(
                self._digests["candidate"], dataset_index,
                candidate_ious[offset],
            )
            _digest_tensor(
                self._digests["candidate"], dataset_index,
                candidate_valid[offset],
            )
            raw_shape, raw_payload = \
                rec_source_gate._calibration_digest_record(
                    raw_ious[offset]
                )
            rec_source_gate._append_calibration_digest_record(
                self._digests["raw"], dataset_index,
                raw_shape, raw_payload,
            )
            _digest_tensor(
                self._digests["materialization"], dataset_index,
                materialization[offset],
            )
            for name, value in (
                    ("baseline_selected", baseline[offset]),
                    ("candidate_selected", candidate[offset])):
                shape, payload = rec_source_gate._calibration_digest_record(
                    value
                )
                rec_source_gate._append_calibration_digest_record(
                    self._digests[name], dataset_index, shape, payload
                )
        self._cursor += batch_size

    def finalize(self, expected_sample_count):
        if (type(expected_sample_count) is not int
                or expected_sample_count <= 0
                or expected_sample_count != len(self._expected_indices)):
            raise ValueError("online expected sample count is invalid")
        if self._cursor != expected_sample_count:
            raise ValueError("online calibration pass is incomplete")
        if self._finalized:
            return copy.deepcopy(self._report)
        invariant = {
            "oracle_hits025": self._counts["oracle025"],
            "oracle_hits050": self._counts["oracle050"],
            "raw_query_hits025": self._counts["raw025"],
            "raw_query_hits050": self._counts["raw050"],
            "candidate_iou_sha256": self._digests[
                "candidate"
            ].hexdigest(),
            "raw_query_iou_sha256": self._digests["raw"].hexdigest(),
            "row_materialization_sha256": self._digests[
                "materialization"
            ].hexdigest(),
        }
        baseline = {
            "sample_count": expected_sample_count,
            "hits025": self._counts["baseline025"],
            "hits050": self._counts["baseline050"],
            "selected_iou_sha256": self._digests[
                "baseline_selected"
            ].hexdigest(),
        }
        candidate = {
            "sample_count": expected_sample_count,
            "hits025": self._counts["candidate025"],
            "hits050": self._counts["candidate050"],
            "selected_iou_sha256": self._digests[
                "candidate_selected"
            ].hexdigest(),
        }
        baseline.update(invariant)
        candidate.update(invariant)
        self._report = {"baseline": baseline, "candidate": candidate}
        self._finalized = True
        return copy.deepcopy(self._report)


def _require_flat_runtime_outputs(value, label):
    if not isinstance(value, dict):
        raise ValueError("{} runtime outputs must be a mapping".format(label))
    expected = {
        "rec_reranker_scores",
        "rec_geometry_runtime_mode",
        "rec_geometry_boxes",
        "rec_geometry_scores",
        "rec_geometry_valid_mask",
        "rec_geometry_fallback_index",
    }
    if (set(value) != expected
            or value.get("rec_geometry_runtime_mode")
            != "flat_geometry_axis"):
        raise ValueError("{} runtime output schema changed".format(label))
    boxes = value["rec_geometry_boxes"]
    scores = value["rec_geometry_scores"]
    valid = value["rec_geometry_valid_mask"]
    parent_scores = value["rec_reranker_scores"]
    if (not isinstance(boxes, torch.Tensor) or boxes.dim() != 3
            or boxes.shape[-1] != 6 or boxes.dtype != torch.float32
            or not isinstance(scores, torch.Tensor) or scores.dim() != 2
            or scores.shape != boxes.shape[:2]
            or scores.dtype != torch.float32
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool or valid.shape != scores.shape
            or not isinstance(parent_scores, torch.Tensor)
            or parent_scores.dim() != 2
            or parent_scores.shape[0] != scores.shape[0]
            or parent_scores.dtype != torch.float32
            or not bool(valid.any(dim=1).all().item())
            or not bool(torch.isfinite(boxes).all().item())
            or not bool(torch.isfinite(scores[valid]).all().item())
            or not bool(torch.isfinite(parent_scores).all().item())):
        raise ValueError("{} runtime tensors are malformed".format(label))
    return value


def evaluate_live_online_calibration(
        initialized, hierarchical_model, hierarchical_artifact, *,
        move_batch=None, input_builder=None, mcln_forward=None,
        full_state_builder=None, parent_output_builder=None,
        geometry_output_builder=None, full_target_attacher=None,
        geometry_target_attacher=None):
    """Evaluate baseline and hierarchy before attaching train-only targets."""
    if not isinstance(initialized, dict):
        raise ValueError("online initialized state must be a mapping")
    state = initialized.get("initial_state")
    data = initialized.get("data")
    contract = initialized.get("train_data_contract")
    if (not isinstance(state, dict) or not isinstance(data, dict)
            or not isinstance(contract, dict)
            or contract.get("validation_data_accessed") is not False):
        raise ValueError("online train-only state is incomplete")
    required_state = {
        "mcln", "parent", "parent_artifact", "geometry",
        "geometry_artifact",
    }
    if not required_state.issubset(state):
        raise ValueError("online frozen model state is incomplete")
    calibration_view = data.get("calibration_view")
    expected_indices = tuple(getattr(calibration_view, "indices", ()))
    expected_count = contract.get("calibration_sample_count")
    if (not expected_indices or type(expected_count) is not int
            or expected_count != len(expected_indices)
            or "calibration_loader" not in data):
        raise ValueError("online calibration data contract is invalid")

    from models import rec_source_gate
    from models.rec_candidate_adapter import build_full_rec_query_state
    from models.rec_mask_geometry import attach_rec_mask_geometry_targets
    from scripts import train_scanrefer_rec_finetune as legacy
    from train_dist_mod import (
        build_rec_geometry_runtime_outputs,
        build_rec_reranker_outputs,
    )

    move_batch = move_batch or legacy._move_batch_to_device
    input_builder = input_builder or legacy.build_rec_finetune_inputs
    full_state_builder = full_state_builder or build_full_rec_query_state
    parent_output_builder = (
        parent_output_builder or build_rec_reranker_outputs
    )
    geometry_output_builder = (
        geometry_output_builder or build_rec_geometry_runtime_outputs
    )
    full_target_attacher = (
        full_target_attacher or rec_source_gate.attach_full_query_targets
    )
    geometry_target_attacher = (
        geometry_target_attacher or attach_rec_mask_geometry_targets
    )
    if mcln_forward is None:
        mcln_forward = state["mcln"]
    hooks = (
        move_batch, input_builder, mcln_forward, full_state_builder,
        parent_output_builder, geometry_output_builder,
        full_target_attacher, geometry_target_attacher,
    )
    if not all(callable(hook) for hook in hooks):
        raise ValueError("online calibration hooks must be callable")

    runtime_device = torch.device(initialized.get("device"))
    accumulator = OnlineCalibrationAccumulator(expected_indices)
    with torch.no_grad():
        for batch in data["calibration_loader"]:
            moved = move_batch(batch, runtime_device)
            if not isinstance(moved, dict):
                raise ValueError("online calibration batch must be a mapping")
            inputs = input_builder(moved)
            if not isinstance(inputs, dict):
                raise ValueError("online calibration inputs must be a mapping")
            legacy._reject_rec_target_only_fields(
                inputs, "online calibration inputs"
            )
            inputs["train"] = False
            end_points = mcln_forward(inputs)
            if not isinstance(end_points, dict):
                raise ValueError("online MCLN outputs must be a mapping")
            legacy._reject_rec_target_only_fields(
                inputs, "post-MCLN online inputs"
            )
            legacy._reject_rec_target_only_fields(
                end_points, "online MCLN outputs"
            )
            full_state = full_state_builder(end_points, inputs)
            if not isinstance(full_state, dict):
                raise ValueError("online raw-query state must be a mapping")
            parent_outputs = parent_output_builder(
                end_points,
                inputs,
                state["parent"],
                state["parent_artifact"],
            )
            baseline = _require_flat_runtime_outputs(
                geometry_output_builder(
                    end_points,
                    inputs,
                    parent_outputs,
                    state["geometry"],
                    state["geometry_artifact"],
                ),
                "baseline",
            )
            candidate = _require_flat_runtime_outputs(
                geometry_output_builder(
                    end_points,
                    inputs,
                    parent_outputs,
                    state["geometry"],
                    state["geometry_artifact"],
                    hierarchical_model=hierarchical_model,
                    hierarchical_artifact=hierarchical_artifact,
                ),
                "candidate",
            )
            if (not torch.equal(
                    baseline["rec_geometry_boxes"],
                    candidate["rec_geometry_boxes"])
                    or not torch.equal(
                        baseline["rec_geometry_valid_mask"],
                        candidate["rec_geometry_valid_mask"])
                    or not torch.equal(
                        baseline["rec_reranker_scores"],
                        candidate["rec_reranker_scores"])
                    or not torch.equal(
                        baseline["rec_geometry_fallback_index"],
                        candidate["rec_geometry_fallback_index"])):
                raise RuntimeError(
                    "hierarchy changed frozen candidate materialization"
                )
            legacy._reject_rec_target_only_fields(
                inputs, "pre-target online inputs"
            )
            legacy._reject_rec_target_only_fields(
                end_points, "pre-target online outputs"
            )

            full_ious = full_target_attacher(
                full_state, moved, root_only=True
            )
            boxes = baseline["rec_geometry_boxes"]
            valid = baseline["rec_geometry_valid_mask"]
            batch_size, candidate_count = valid.shape
            if candidate_count != 16 * 7:
                raise ValueError("online geometry axis must contain 112 rows")
            geometry_targets = geometry_target_attacher(
                {
                    "boxes": boxes.reshape(batch_size, 16, 7, 6),
                    "valid_mask": valid.reshape(batch_size, 16, 7),
                },
                moved,
                root_only=True,
            )
            candidate_ious = (
                geometry_targets.get("geometry_ious")
                if isinstance(geometry_targets, dict) else None
            )
            if (not isinstance(full_ious, torch.Tensor)
                    or full_ious.dim() != 2
                    or full_ious.shape[0] != batch_size
                    or not isinstance(candidate_ious, torch.Tensor)
                    or tuple(candidate_ious.shape)
                    != (batch_size, 16, 7)):
                raise ValueError("online target IoUs are malformed")
            flat_ious = candidate_ious.reshape(batch_size, candidate_count)
            baseline_scores = baseline["rec_geometry_scores"].masked_fill(
                ~valid, -float("inf")
            )
            candidate_scores = candidate["rec_geometry_scores"].masked_fill(
                ~valid, -float("inf")
            )
            rows = torch.arange(batch_size, device=valid.device)
            baseline_selected = flat_ious[
                rows, baseline_scores.argmax(dim=1)
            ]
            candidate_selected = flat_ious[
                rows, candidate_scores.argmax(dim=1)
            ]
            materialization = torch.cat((
                boxes.reshape(batch_size, -1),
                valid.float(),
                baseline_scores,
                baseline["rec_reranker_scores"],
            ), dim=1)
            dataset_indices = moved.get("dataset_index")
            if dataset_indices is None:
                raise ValueError("online batch has no dataset_index")
            accumulator.update(dataset_indices, {
                "baseline_selected_ious": baseline_selected.float(),
                "candidate_selected_ious": candidate_selected.float(),
                "candidate_ious": flat_ious.float(),
                "candidate_valid": valid,
                "raw_query_ious": full_ious.float(),
                "row_materialization": materialization.float(),
            })
    return accumulator.finalize(expected_count)


def _validate_artifact_binding(value, label, deployable):
    if (not isinstance(value, dict)
            or set(value) != {"path", "sha256", "deployable"}
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or not _is_sha256(value.get("sha256"))
            or value.get("deployable") is not deployable):
        raise ValueError("{} artifact binding is invalid".format(label))
    return copy.deepcopy(value)


def _canonical_json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_protected_snapshot(value, label):
    if (not isinstance(value, dict)
            or set(value) != _ONLINE_SNAPSHOT_FIELDS
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or type(value.get("device")) is not int
            or value["device"] < 0
            or type(value.get("inode")) is not int
            or value["inode"] < 0
            or value.get("mode") != 0o444
            or type(value.get("size")) is not int
            or value["size"] < 0
            or type(value.get("mtime_ns")) is not int
            or value["mtime_ns"] < 0
            or type(value.get("ctime_ns")) is not int
            or value["ctime_ns"] < 0
            or not _is_sha256(value.get("sha256"))):
        raise ValueError("{} protected snapshot is invalid".format(label))
    return copy.deepcopy(value)


def _validate_online_provenance(value, staged):
    if (not isinstance(value, dict)
            or set(value) != _ONLINE_PROVENANCE_FIELDS):
        raise ValueError("online provenance fields are invalid")
    command = value.get("command")
    if (not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item for item in command)
            or any(
                token in item.lower()
                for item in command
                for token in ("--official", "--validation", "--val-")
            )):
        raise ValueError("online command is invalid or accesses validation")
    environment = value.get("environment")
    if (not isinstance(environment, dict)
            or set(environment) != {
                "device", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                "PYTHONPATH", "python",
            }
            or environment.get("device") != "cuda:0"
            or environment.get("CUDA_VISIBLE_DEVICES") != "0"
            or environment.get("OMP_NUM_THREADS") != "1"
            or not isinstance(environment.get("PYTHONPATH"), str)
            or not environment["PYTHONPATH"]
            or not isinstance(environment.get("python"), str)
            or not Path(environment["python"]).is_absolute()):
        raise ValueError("online environment is not authoritative")
    code = value.get("code")
    files = code.get("files") if isinstance(code, dict) else None
    if (not isinstance(code, dict) or set(code) != {"files", "sha256"}
            or not isinstance(files, dict) or not files
            or any(
                not isinstance(name, str) or not name
                or not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or not isinstance(record.get("path"), str)
                or not Path(record["path"]).is_absolute()
                or not _is_sha256(record.get("sha256"))
                for name, record in files.items()
            )
            or code.get("sha256") != _canonical_json_sha256(files)):
        raise ValueError("online code manifest is invalid")
    before = value.get("protected_before")
    after = value.get("protected_after")
    if (not isinstance(before, dict) or set(before) != _ONLINE_PROTECTED_NAMES
            or not isinstance(after, dict)
            or set(after) != _ONLINE_PROTECTED_NAMES):
        raise ValueError("online protected input set changed")
    for name in sorted(_ONLINE_PROTECTED_NAMES):
        _validate_protected_snapshot(before[name], "before " + name)
        _validate_protected_snapshot(after[name], "after " + name)
    if before != after:
        raise ValueError("online protected inputs changed during calibration")
    if (before["staged_hierarchical"]["path"] != staged["path"]
            or before["staged_hierarchical"]["sha256"] != staged["sha256"]):
        raise ValueError("online staged snapshot binding changed")
    source = value.get("source_gate_baseline_receipt")
    expected_source = {
        "path": before["source_gate_baseline"]["path"],
        "sha256": before["source_gate_baseline"]["sha256"],
    }
    if source != expected_source:
        raise ValueError("source-gate baseline receipt binding changed")
    staged_receipt = value.get("staged_result_receipt")
    expected_staged_receipt = {
        "path": before["staged_result_receipt"]["path"],
        "sha256": before["staged_result_receipt"]["sha256"],
    }
    if staged_receipt != expected_staged_receipt:
        raise ValueError("staged result receipt binding changed")
    return copy.deepcopy(value)


def _gate_record(gate):
    return {
        "passed": gate.passed,
        "failures": list(gate.failures),
        "required_hits025": gate.required_hits025,
        "required_hits050": gate.required_hits050,
    }


def validate_online_calibration_record(record, expected_staged_sha256=None):
    """Validate exact online evidence and its staged/deployed bindings."""
    if not isinstance(record, dict) or set(record) != _ONLINE_RECORD_FIELDS:
        raise ValueError("online calibration record fields differ from schema")
    if (record.get("schema") != ONLINE_CALIBRATION_SCHEMA
            or record.get("version") != ONLINE_CALIBRATION_VERSION
            or record.get("sample_count") != ONLINE_SAMPLE_COUNT
            or record.get("validation_data_accessed") is not False
            or record.get("inference_uses_ground_truth") is not False):
        raise ValueError("online calibration top-level policy is invalid")
    baseline = _validate_evidence(record.get("baseline"), "baseline")
    candidate = _validate_evidence(record.get("candidate"), "candidate")
    if (baseline["sample_count"] != ONLINE_SAMPLE_COUNT
            or candidate["sample_count"] != ONLINE_SAMPLE_COUNT):
        raise ValueError("online calibration evidence count changed")
    staged = _validate_artifact_binding(
        record.get("staged_artifact"), "staged", False
    )
    if (expected_staged_sha256 is not None
            and (not _is_sha256(expected_staged_sha256)
                 or staged["sha256"] != expected_staged_sha256)):
        raise ValueError("online staged artifact SHA mismatch")
    _validate_online_provenance(record.get("provenance"), staged)
    deployed = record.get("deployed_artifact")
    if deployed is not None:
        _validate_artifact_binding(deployed, "deployed", True)
    gate = online_calibration_gate(baseline, candidate)
    if record.get("gate") != _gate_record(gate):
        raise ValueError("online calibration gate record changed")
    return copy.deepcopy(record)


def build_deployed_hierarchical_copy(staged_artifact):
    """Return a validated deployed copy without changing staged state."""
    geometry_names = (
        staged_artifact.get("feature_names", {}).get("geometry_input", ())
        if isinstance(staged_artifact, dict) else ()
    )
    validate_hierarchical_artifact(
        staged_artifact,
        expected_geometry_feature_names=geometry_names,
        expected_deployable=False,
    )
    deployed = copy.deepcopy(staged_artifact)
    deployed["deployable"] = True
    validate_hierarchical_artifact(
        deployed,
        expected_geometry_feature_names=geometry_names,
        expected_deployable=True,
    )
    return deployed


def serialize_deployed_hierarchical_artifact(staged_artifact):
    """Serialize a validated deployed copy while preserving staged bytes."""
    from scripts.train_scanrefer_rec_hierarchical_reranker import (
        _serialize_hierarchical_artifact,
    )

    deployed = build_deployed_hierarchical_copy(staged_artifact)
    return _serialize_hierarchical_artifact(
        deployed, expected_deployable=True
    )


def build_online_calibration_record(initialized, evidence):
    """Bind one live evidence pair to the staged artifact and provenance."""
    if not isinstance(initialized, dict):
        raise ValueError("online initialized publication state is invalid")
    if not isinstance(evidence, dict) or set(evidence) != {
            "baseline", "candidate"}:
        raise ValueError("online evaluator evidence fields changed")
    staged_path = initialized.get("staged_artifact_path")
    staged_sha = initialized.get("staged_artifact_sha256")
    provenance = initialized.get("provenance")
    if (not isinstance(staged_path, Path) or not staged_path.is_absolute()
            or not _is_sha256(staged_sha)):
        raise ValueError("online staged publication binding is invalid")
    baseline = _validate_evidence(evidence["baseline"], "baseline")
    candidate = _validate_evidence(evidence["candidate"], "candidate")
    gate = online_calibration_gate(baseline, candidate)
    record = {
        "schema": ONLINE_CALIBRATION_SCHEMA,
        "version": ONLINE_CALIBRATION_VERSION,
        "sample_count": ONLINE_SAMPLE_COUNT,
        "baseline": baseline,
        "candidate": candidate,
        "staged_artifact": {
            "path": str(staged_path),
            "sha256": staged_sha,
            "deployable": False,
        },
        "deployed_artifact": None,
        "gate": _gate_record(gate),
        "provenance": copy.deepcopy(provenance),
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
    }
    return validate_online_calibration_record(
        record, expected_staged_sha256=staged_sha
    )


def finalize_live_online_calibration(initialized):
    """Recheck every protected byte after the sole live evaluation pass."""
    if not isinstance(initialized, dict):
        raise ValueError("online initialized state must be a mapping")
    protected_paths = initialized.get("protected_paths")
    provenance = initialized.get("provenance")
    staged_path = initialized.get("staged_artifact_path")
    staged_sha256 = initialized.get("staged_artifact_sha256")
    if (not isinstance(protected_paths, dict)
            or not isinstance(provenance, dict)
            or not isinstance(staged_path, Path)
            or not staged_path.is_absolute()
            or not _is_sha256(staged_sha256)):
        raise ValueError("online finalization state is incomplete")
    before = provenance.get("protected_before")
    after = _capture_online_protected_inputs(protected_paths)
    if after != before:
        raise RuntimeError(
            "online protected inputs changed during live evaluation"
        )
    finalized = dict(initialized)
    finalized_provenance = copy.deepcopy(provenance)
    finalized_provenance["protected_after"] = after
    finalized["provenance"] = _validate_online_provenance(
        finalized_provenance,
        {
            "path": str(staged_path),
            "sha256": staged_sha256,
            "deployable": False,
        },
    )
    return finalized


def run_online_calibration(
        args, *, initializer=None, evaluator=None, finalizer=None,
        deployed_serializer=None, publisher=None):
    """Run exactly one online calibration and publish its immutable result."""
    initializer = initializer or initialize_live_online_calibration
    evaluator = evaluator or evaluate_live_online_calibration
    finalizer = finalizer or finalize_live_online_calibration
    deployed_serializer = (
        deployed_serializer or serialize_deployed_hierarchical_artifact
    )
    publisher = publisher or publish_online_calibration
    if not all(callable(value) for value in (
            initializer, evaluator, finalizer, deployed_serializer,
            publisher)):
        raise ValueError("online calibration orchestration hooks are invalid")
    initialized = initializer(args)
    if not isinstance(initialized, dict):
        raise ValueError("online initializer returned invalid state")
    hierarchical_model = initialized.get("hierarchical_model")
    hierarchical_artifact = initialized.get("hierarchical_artifact")
    if hierarchical_model is None or not isinstance(hierarchical_artifact, dict):
        raise ValueError("online staged hierarchy is unavailable")
    evidence = evaluator(
        initialized, hierarchical_model, hierarchical_artifact
    )
    finalized = finalizer(initialized)
    if not isinstance(finalized, dict):
        raise ValueError("online finalizer returned invalid state")
    record = build_online_calibration_record(finalized, evidence)
    payload = None
    if record["gate"]["passed"]:
        payload = deployed_serializer(hierarchical_artifact)
    return publisher(
        args.output_dir,
        record,
        payload,
        expected_staged_path=initialized["staged_artifact_path"],
        expected_staged_sha256=initialized["staged_artifact_sha256"],
    )


def _regular_file_sha256(path, label, required_mode=None):
    path = Path(path).expanduser().absolute()
    metadata = os.lstat(str(path))
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("{} must be a regular non-symlink file".format(label))
    if required_mode is not None and stat.S_IMODE(metadata.st_mode) != required_mode:
        raise ValueError("{} mode changed".format(label))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.lstat(str(path))
    before_identity = (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns,
    )
    after_identity = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError("{} changed during stable hash".format(label))
    return path, digest.hexdigest(), before_identity


def _write_exclusive(path, payload, mode):
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


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_online_calibration(
        output_dir, record, deployed_payload, *, expected_staged_path,
        expected_staged_sha256):
    """Publish a deployed copy followed by the immutable completion record."""
    validated = validate_online_calibration_record(
        record, expected_staged_sha256=expected_staged_sha256
    )
    gate_passed = validated["gate"]["passed"] is True
    if gate_passed:
        if not isinstance(deployed_payload, bytes) or not deployed_payload:
            raise ValueError("deployed artifact payload must be nonempty bytes")
    elif deployed_payload is not None:
        raise ValueError("failed online gate cannot carry a deployed payload")
    staged_path, staged_sha, staged_identity = _regular_file_sha256(
        expected_staged_path, "staged hierarchical artifact", required_mode=0o444
    )
    if (staged_path != Path(validated["staged_artifact"]["path"])
            or staged_sha != expected_staged_sha256
            or staged_sha != validated["staged_artifact"]["sha256"]):
        raise ValueError("staged hierarchical artifact binding changed")

    output = Path(output_dir).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(str(output), 0o700)
    finalized = copy.deepcopy(validated)
    if gate_passed:
        deployed_path = output / DEPLOYED_ARTIFACT_NAME
        _write_exclusive(deployed_path, deployed_payload, 0o444)
        deployed_sha = hashlib.sha256(deployed_payload).hexdigest()
        finalized["deployed_artifact"] = {
            "path": str(deployed_path),
            "sha256": deployed_sha,
            "deployable": True,
        }
    finalized = validate_online_calibration_record(
        finalized, expected_staged_sha256=expected_staged_sha256
    )
    payload = json.dumps(
        finalized, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    _write_exclusive(output / ONLINE_CALIBRATION_NAME, payload, 0o444)
    _fsync_directory(output)
    _, final_staged_sha, final_staged_identity = _regular_file_sha256(
        staged_path, "staged hierarchical artifact", required_mode=0o444
    )
    if (final_staged_sha != staged_sha
            or final_staged_identity != staged_identity):
        raise RuntimeError("staged hierarchical artifact changed during promotion")
    return finalized


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one frozen train-only hierarchical calibration."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--staged-hierarchical-artifact", required=True)
    parser.add_argument("--staged-result-receipt", required=True)
    parser.add_argument("--source-gate-baseline-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None, *, runner=None):
    """Run the sole live train calibration and print its canonical receipt."""
    args = parse_args(argv)
    runner = runner or run_online_calibration
    if not callable(runner):
        raise ValueError("online calibration runner must be callable")
    result = runner(args)
    sys.stdout.write(_canonical_json_bytes(result).decode("ascii") + "\n")
    return result


__all__ = [
    "DEPLOYED_ARTIFACT_NAME",
    "ONLINE_CALIBRATION_NAME",
    "ONLINE_CALIBRATION_SCHEMA",
    "ONLINE_CALIBRATION_VERSION",
    "ONLINE_BASELINE_SELECTED_IOU_SHA256",
    "ONLINE_INVARIANT_FIELDS",
    "ONLINE_RAW_QUERY_IOU_SHA256",
    "OnlineCalibrationAccumulator",
    "OnlineCalibrationGateResult",
    "build_deployed_hierarchical_copy",
    "finalize_live_online_calibration",
    "initialize_live_online_calibration",
    "main",
    "online_calibration_gate",
    "parse_args",
    "publish_online_calibration",
    "validate_online_calibration_record",
    "validate_source_gate_baseline_evidence",
]


if __name__ == "__main__":
    main()
