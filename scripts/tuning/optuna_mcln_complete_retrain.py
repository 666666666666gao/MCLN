#!/usr/bin/env python
"""Resumable train-only Optuna orchestration for complete MCLN retraining."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import optuna
from optuna.trial import TrialState

from scripts.tuning.mcln_optuna_contract import (
    assess_trial_metrics,
    cleanup_trial_checkpoints,
    count_completed_trials,
    require_minimum_free_space,
    seed_presets,
    select_best_trial,
    suggest_trial_params,
    validate_metrics_receipt,
)
from scripts.tuning.train_mcln_optuna_trial import (
    SOURCE_PAIR,
    TRIAL_RECEIPT_SCHEMA,
    atomic_write_json,
    file_sha256,
    validate_loss_receipt,
    validate_study_binding,
)
from scripts.tuning.scanrefer_train_only import (
    AUTHORITATIVE_SCANREFER_SPLIT_METADATA,
)


DEFAULT_BASE_SHA256 = (
    "a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d"
)
MAX_PROCESS_ATTEMPTS = 60
STUDY_CONTRACT_SCHEMA = "mcln-complete-retrain-study-contract-v2"
BASELINE_ARTIFACT_SCHEMA = "mcln-optuna-baseline-artifact-v2"
STUDY_BINDING_SCHEMA = "mcln-optuna-study-binding-v1"
_VOLATILE_STUDY_CONTRACT_FIELDS = frozenset(("contract_digest",))
LONG_DISPATCH_SCHEMA = "mcln-complete-long-dispatch-v1"
LONG_STARTUP_ACK_SCHEMA = "mcln-complete-long-startup-ack-v1"
LONG_COMPLETION_SCHEMA = "mcln-complete-long-completion-v1"
LONG_STARTUP_GRACE_SECONDS = 120.0


def repo_root():
    return Path(__file__).resolve().parents[2]


def _output_root(args):
    return Path(args.output_root)


def _trial_directory(args, trial_number):
    return _output_root(args) / "trials" / "trial_{:04d}".format(
        trial_number
    )


def _sqlite_storage_identity(storage):
    prefix = "sqlite:///"
    if not isinstance(storage, str) or not storage.startswith(prefix):
        raise ValueError("formal study storage must use sqlite:///PATH")
    database = storage[len(prefix):]
    if not database or "?" in database or "#" in database:
        raise ValueError("formal SQLite storage path is invalid")
    return str(Path(database).resolve())


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in value
        )
    )


def canonical_json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_study_contract_payload(contract):
    if not isinstance(contract, dict):
        raise ValueError("study contract must be a mapping")
    return {
        key: json.loads(json.dumps(value, sort_keys=True))
        for key, value in contract.items()
        if key not in _VOLATILE_STUDY_CONTRACT_FIELDS
    }


def _write_json_no_replace(path, payload):
    """Atomically publish one complete JSON file without replacing a peer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            return False
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_study_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("study contract must be a mapping")
    if contract.get("schema") != STUDY_CONTRACT_SCHEMA:
        raise ValueError("study contract schema is invalid")
    digest = contract.get("contract_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("study contract digest is invalid")
    actual = canonical_json_sha256(
        canonical_study_contract_payload(contract)
    )
    if actual != digest:
        raise ValueError("study contract digest does not match its payload")
    return contract


def study_receipt_binding(contract):
    contract = _validated_study_contract(contract)
    binding = {
        "study_contract_digest": contract["contract_digest"],
        "source_snapshot_digest": contract["source_snapshot_digest"],
        "base_checkpoint_sha256": contract["base_checkpoint_sha256"],
        "pointnet_checkpoint_sha256": contract[
            "pointnet_checkpoint_sha256"
        ],
        "repo_root": contract["repo_root"],
        "data_root": contract["data_root"],
        "python_bin": contract["python_bin"],
        "run_manifest_sha256": contract["run_manifest_sha256"],
        "environment_sha256": contract["environment_sha256"],
        "inputs_sha256": contract["inputs_sha256"],
        "study_name": contract["study_name"],
        "storage_identity": contract["storage_identity"],
        "split_metadata": contract["fit_calibration"],
        "split_metadata_sha256": contract["split_metadata_sha256"],
        "data_digests": contract["data_digests"],
        "continuation_horizon": contract["fixed"][
            "continuation_horizon"
        ],
    }
    return validate_study_binding(binding)


def _current_study_contract(args):
    path = _output_root(args) / "study_contract.json"
    if not path.is_file():
        raise ValueError("study contract has not been established")
    return _validated_study_contract(_load_json(path))


def _expected_study_binding(args):
    return study_receipt_binding(_current_study_contract(args))


def _common_runner_command(args, mode, receipt_path, master_port):
    study_binding = study_receipt_binding(write_study_contract(args))
    command = [
        args.python_bin,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        "1",
        "--master_port",
        str(master_port),
        "scripts/tuning/train_mcln_optuna_trial.py",
        "--mode",
        mode,
        "--receipt-path",
        str(receipt_path),
        "--base-checkpoint",
        args.base_checkpoint,
        "--expected-base-sha256",
        args.base_sha256,
        "--split-seed",
        "0",
        "--calibration-fraction",
        "0.10",
        "--continuation-horizon",
        "46",
        "--study-binding-json",
        json.dumps(
            study_binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        "--num_decoder_layers",
        "6",
        "--use_color",
        "--weight_decay",
        "0.0005",
        "--data_root",
        args.data_root,
        "--val_freq",
        "1",
        "--batch_size",
        "18",
        "--save_freq",
        "1",
        "--print_freq",
        "400",
        "--dataset",
        "scanrefer",
        "--test_dataset",
        "scanrefer",
        "--detect_intermediate",
        "--joint_det",
        "--use_soft_token_loss",
        "--use_contrastive_align",
        "--log_dir",
        str(_output_root(args) / "logs"),
        "--clip_norm",
        "0.1",
        "--pp_checkpoint",
        args.pp_checkpoint,
        "--butd",
        "--self_attend",
        "--augment_det",
        "--max_epoch",
        "56",
        "--model",
        "MCLN",
        "--checkpoint_path",
        args.base_checkpoint,
        "--reduce_lr",
        "--skip_missing_superpoints",
        "--use_source_choice_selector",
        "--source_choice_selector_sources",
        SOURCE_PAIR,
        "--source_choice_selector_default_source",
        "default",
        "--source_choice_selector_choice_target",
        "precision_gain_default_sourcewise_focal_bce",
        "--source_choice_selector_hidden_dim",
        "288",
        "--eval_use_selector_choice_scores",
        "--num_workers",
        "4",
        "--rng_seed",
        "0",
    ]
    return command


def command_for_baseline(args):
    receipt_path = _output_root(args) / "baseline" / "receipt.json"
    command = _common_runner_command(
        args, "baseline", receipt_path, args.master_port_base
    )
    command.extend(["--exp", "calibration_baseline_epoch54"])
    return command


def command_for_trial(args, params, trial_number):
    params = dict(params)
    if set(params) != set(seed_presets()[0]):
        raise ValueError("trial parameters do not match the approved space")
    trial_dir = _trial_directory(args, trial_number)
    receipt_path = trial_dir / "receipt.json"
    checkpoint_path = trial_dir / "epoch56.pth"
    command = _common_runner_command(
        args,
        "trial",
        receipt_path,
        args.master_port_base + 1 + (trial_number % 1000),
    )
    command.extend([
        "--checkpoint-output",
        str(checkpoint_path),
        "--exp",
        "optuna_trial_{:04d}".format(trial_number),
        "--lr",
        str(params["decoder_lr"]),
        "--lr_backbone",
        str(params["decoder_lr"] * 10.0),
        "--source_choice_selector_lr",
        str(params["selector_lr"]),
        "--source_choice_selector_loss_weight",
        str(params["selector_loss_weight"]),
        "--source_choice_selector_min_iou_gap",
        str(params["selector_min_iou_gap"]),
        "--mask_head_lr_multiplier",
        str(params["mask_head_lr_multiplier"]),
        "--mask_loss_scale",
        str(params["mask_loss_scale"]),
        "--consistency_loss_scale",
        str(params["consistency_loss_scale"]),
        "--decoder-lr",
        str(params["decoder_lr"]),
        "--mask-head-lr-multiplier",
        str(params["mask_head_lr_multiplier"]),
        "--selector-lr",
        str(params["selector_lr"]),
        "--mask-loss-scale",
        str(params["mask_loss_scale"]),
        "--consistency-loss-scale",
        str(params["consistency_loss_scale"]),
        "--selector-loss-weight",
        str(params["selector_loss_weight"]),
        "--selector-min-iou-gap",
        str(params["selector_min_iou_gap"]),
    ])
    return command


def enqueue_seed_presets_if_new(study):
    if study.trials:
        return 0
    presets = seed_presets()
    for preset in presets:
        study.enqueue_trial(dict(preset))
    return len(presets)


def remaining_successful_trials(
        study, target_successful_trials, receipt_is_valid=None):
    if (
        not isinstance(target_successful_trials, int)
        or isinstance(target_successful_trials, bool)
        or target_successful_trials <= 0
    ):
        raise ValueError("target_successful_trials must be positive")
    completed = count_completed_trials(
        study.trials, receipt_is_valid=receipt_is_valid
    )
    return max(0, target_successful_trials - completed)


def _load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_trial_params(params):
    if not isinstance(params, dict):
        raise ValueError("trial_params must be a mapping")
    expected_names = set(seed_presets()[0])
    if set(params) != expected_names:
        raise ValueError("trial_params fields differ from the approved space")
    resolved = {}
    for name, value in params.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("{} is invalid".format(name))
        resolved[name] = float(value)
    bounds = {
        "decoder_lr": (5e-6, 4e-5),
        "selector_lr": (2e-4, 2e-3),
        "mask_loss_scale": (0.5, 4.0),
        "consistency_loss_scale": (0.1, 2.0),
        "selector_loss_weight": (0.1, 1.0),
    }
    for name, (minimum, maximum) in bounds.items():
        if not minimum <= resolved[name] <= maximum:
            raise ValueError("{} is outside the approved space".format(name))
    categorical = {
        "mask_head_lr_multiplier": (1.0, 2.0, 4.0),
        "selector_min_iou_gap": (0.02, 0.03, 0.05, 0.08),
    }
    for name, choices in categorical.items():
        if resolved[name] not in choices:
            raise ValueError("{} is outside the approved space".format(name))
    return resolved


def _validate_optimizer_groups(groups, params):
    if not isinstance(groups, (list, tuple)):
        raise ValueError("optimizer_groups must be a sequence")
    expected_names = ("decoder", "backbone", "mask_head", "selector")
    actual_names = tuple(
        group.get("name") if isinstance(group, dict) else None
        for group in groups
    )
    if actual_names != expected_names:
        raise ValueError("optimizer group names are invalid")
    expected_lrs = (
        params["decoder_lr"],
        params["decoder_lr"] * 10.0,
        params["decoder_lr"] * params["mask_head_lr_multiplier"],
        params["selector_lr"],
    )
    result = []
    all_parameter_names = []
    for group, expected_lr in zip(groups, expected_lrs):
        learning_rate = group.get("initial_lr")
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) != float(expected_lr)
        ):
            raise ValueError("optimizer group learning rate is invalid")
        parameter_names = group.get("parameter_names")
        if (
            not isinstance(parameter_names, (list, tuple))
            or not parameter_names
            or any(
                not isinstance(name, str) or not name
                for name in parameter_names
            )
        ):
            raise ValueError("optimizer group parameter_names are invalid")
        parameter_names = tuple(parameter_names)
        if parameter_names != tuple(sorted(parameter_names)):
            raise ValueError("optimizer group parameter_names are not sorted")
        all_parameter_names.extend(parameter_names)
        result.append({
            "name": group["name"],
            "initial_lr": float(learning_rate),
            "parameter_names": parameter_names,
        })
    if len(all_parameter_names) != len(set(all_parameter_names)):
        raise ValueError("optimizer parameter_names are not disjoint")
    return tuple(result)


def _validate_receipt_study_binding(receipt, expected=None):
    try:
        binding = validate_study_binding(receipt.get("study_binding"))
    except ValueError as error:
        raise ValueError("receipt study binding is invalid: {}".format(error))
    if expected is not None and binding != expected:
        raise ValueError("receipt study binding differs from the study contract")
    return binding


def _validate_checkpoint_binding(receipt, checkpoint):
    digest = receipt.get("checkpoint_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("trial checkpoint SHA-256 is invalid")
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file() and file_sha256(checkpoint_path) != digest:
        raise ValueError("trial checkpoint SHA-256 differs from its receipt")
    return digest


def candidate_from_trial_receipt(
        trial_number, baseline_receipt, trial_receipt, receipt_path,
        expected_study_binding=None, expected_trial_params=None):
    if (
        not isinstance(trial_number, int)
        or isinstance(trial_number, bool)
        or trial_number < 0
    ):
        raise ValueError("trial_number must be non-negative")
    if not isinstance(baseline_receipt, dict):
        raise ValueError("baseline receipt must be a mapping")
    if baseline_receipt.get("schema") != TRIAL_RECEIPT_SCHEMA:
        raise ValueError("baseline receipt schema is invalid")
    if (
        baseline_receipt.get("mode") != "baseline"
        or baseline_receipt.get("selection_epoch") != 54
    ):
        raise ValueError("baseline receipt contract is invalid")
    baseline_binding = _validate_receipt_study_binding(
        baseline_receipt, expected=expected_study_binding
    )
    if baseline_receipt.get("checkpoint") is not None:
        raise ValueError("baseline receipt checkpoint is invalid")
    if baseline_receipt.get("trial_params") is not None:
        raise ValueError("baseline receipt trial_params are invalid")
    if baseline_receipt.get("optimizer_groups") not in ([], ()):
        raise ValueError("baseline receipt optimizer groups are invalid")
    if "losses" in baseline_receipt:
        raise ValueError("baseline receipt must not contain training losses")
    baseline_metrics = baseline_receipt.get("metrics")
    if not isinstance(baseline_metrics, dict) or set(baseline_metrics) != {
        "epoch_54"
    }:
        raise ValueError("baseline receipt metrics are invalid")
    baseline = validate_metrics_receipt(baseline_metrics["epoch_54"])

    if not isinstance(trial_receipt, dict):
        raise ValueError("trial receipt must be a mapping")
    if trial_receipt.get("schema") != TRIAL_RECEIPT_SCHEMA:
        raise ValueError("trial receipt schema is invalid")
    if (
        trial_receipt.get("mode") != "trial"
        or trial_receipt.get("selection_epoch") != 56
    ):
        raise ValueError("trial receipt epoch contract is invalid")
    trial_binding = _validate_receipt_study_binding(
        trial_receipt, expected=expected_study_binding
    )
    if trial_binding != baseline_binding:
        raise ValueError("baseline and trial study bindings differ")
    metrics = trial_receipt.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "epoch_55", "epoch_56"
    }:
        raise ValueError("trial receipt metrics are invalid")
    validate_metrics_receipt(metrics["epoch_55"])
    selected_metrics = validate_metrics_receipt(metrics["epoch_56"])
    losses = trial_receipt.get("losses")
    if not isinstance(losses, dict) or set(losses) != {
        "epoch_55", "epoch_56"
    }:
        raise ValueError("trial receipt training loss epochs are invalid")
    validated_losses = {
        epoch_key: validate_loss_receipt(losses[epoch_key])
        for epoch_key in ("epoch_55", "epoch_56")
    }
    params = _validate_trial_params(trial_receipt.get("trial_params"))
    if expected_trial_params is not None and params != dict(
            expected_trial_params):
        raise ValueError("trial receipt params differ from the Optuna trial")
    optimizer_groups = _validate_optimizer_groups(
        trial_receipt.get("optimizer_groups"), params
    )
    checkpoint = trial_receipt.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("trial receipt checkpoint path is invalid")
    checkpoint_sha256 = _validate_checkpoint_binding(
        trial_receipt, checkpoint
    )

    assessment = assess_trial_metrics(baseline, selected_metrics)
    return {
        "trial_number": trial_number,
        "metrics": selected_metrics,
        "losses": validated_losses,
        "feasible": assessment["feasible"],
        "objective": assessment["objective"],
        "deltas": assessment["deltas"],
        "constraint_failures": assessment["constraint_failures"],
        "trial_params": params,
        "optimizer_groups": optimizer_groups,
        "optimizer_groups_sha256": canonical_json_sha256(optimizer_groups),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "receipt": os.fspath(receipt_path),
    }


def _stable_checkpoint_path(args):
    return _output_root(args) / "checkpoints" / "optuna_best_trial.pth"


def _same_file(first, second):
    try:
        return os.path.samefile(str(first), str(second))
    except (FileNotFoundError, OSError):
        return False


def _publish_best_checkpoint(args, candidate):
    stable = _stable_checkpoint_path(args)
    stable.parent.mkdir(parents=True, exist_ok=True)
    expected_digest = candidate.get("checkpoint_sha256")
    if not _valid_sha256(expected_digest):
        raise ValueError("selected checkpoint SHA-256 digest is invalid")
    expected_digest = expected_digest.lower()
    source = Path(candidate["checkpoint"])
    if not source.is_file():
        existing_best = _output_root(args) / "best.json"
        if not existing_best.is_file() or not stable.is_file():
            raise ValueError("selected short-trial checkpoint is missing")
        if file_sha256(stable) != expected_digest:
            raise ValueError("existing stable checkpoint digest differs")
        receipt = _load_json(existing_best)
        if receipt.get("trial_number") != candidate["trial_number"]:
            raise ValueError("existing checkpoint trial binding differs")
        if receipt.get("receipt") != candidate.get("receipt"):
            raise ValueError("existing checkpoint receipt binding differs")
        try:
            existing_binding = validate_study_binding(
                receipt.get("study_binding")
            )
        except ValueError as error:
            raise ValueError(
                "existing checkpoint study binding is invalid: {}".format(
                    error
                )
            )
        if existing_binding != _expected_study_binding(args):
            raise ValueError("existing checkpoint study binding differs")
        return stable, "existing"
    if file_sha256(source) != expected_digest:
        raise ValueError("selected checkpoint SHA-256 digest differs")
    if stable.is_file() and _same_file(source, stable):
        if file_sha256(stable) != expected_digest:
            raise ValueError("stable checkpoint SHA-256 digest differs")
        return stable, "existing"

    temporary = stable.with_name(".{}-{}.tmp".format(stable.name, os.getpid()))
    if temporary.exists():
        temporary.unlink()
    method = "hardlink"
    try:
        try:
            os.link(str(source), str(temporary))
        except OSError:
            shutil.copy2(str(source), str(temporary))
            method = "copy"
        if file_sha256(temporary) != expected_digest:
            raise ValueError("staged checkpoint SHA-256 digest differs")
        os.replace(str(temporary), str(stable))
        if file_sha256(stable) != expected_digest:
            stable.unlink()
            raise ValueError("published checkpoint SHA-256 digest differs")
    finally:
        if temporary.exists():
            temporary.unlink()
    return stable, method


def long_dispatch_binding(best):
    if not isinstance(best, dict):
        raise ValueError("long dispatch best selection must be a mapping")
    if best.get("selection_status") != "feasible_best":
        raise ValueError("long dispatch requires a feasible best selection")
    trial_number = best.get("trial_number")
    if (
        not isinstance(trial_number, int)
        or isinstance(trial_number, bool)
        or trial_number < 0
    ):
        raise ValueError("long dispatch best trial number is invalid")
    checkpoint = best.get("checkpoint")
    checkpoint_sha256 = best.get("checkpoint_sha256")
    receipt = best.get("receipt")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("long dispatch best checkpoint is invalid")
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError("long dispatch best checkpoint digest is invalid")
    if not isinstance(receipt, str) or not receipt:
        raise ValueError("long dispatch best receipt is invalid")
    study_binding = validate_study_binding(best.get("study_binding"))
    return {
        "schema": "mcln-complete-long-binding-v1",
        "trial_number": trial_number,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "receipt": receipt,
        "study_binding": study_binding,
    }


def _valid_dispatch_token(token):
    return (
        isinstance(token, str)
        and len(token) == 32
        and all(character in "0123456789abcdef" for character in token)
    )


def command_for_long_run(
        args, best_json, dispatch_token=None, startup_ack=None,
        completion_receipt=None):
    command = [
        args.python_bin,
        str(Path(args.repo_root) / "scripts/tuning/train_mcln_complete_long.py"),
        "--best-json",
        str(best_json),
        "--output-root",
        str(_output_root(args) / "long_run"),
        "--base-checkpoint",
        args.base_checkpoint,
        "--expected-base-sha256",
        args.base_sha256,
        "--data-root",
        args.data_root,
        "--pp-checkpoint",
        args.pp_checkpoint,
        "--gpu",
        str(args.gpu),
    ]
    dispatch_values = (dispatch_token, startup_ack, completion_receipt)
    if any(value is not None for value in dispatch_values):
        if not all(value is not None for value in dispatch_values):
            raise ValueError("long dispatch command state paths are incomplete")
        if not _valid_dispatch_token(dispatch_token):
            raise ValueError("long dispatch command token is invalid")
        command.extend([
            "--dispatch-token",
            dispatch_token,
            "--startup-ack",
            str(startup_ack),
            "--completion-receipt",
            str(completion_receipt),
        ])
    return command


def _validated_dispatch_binding(payload, expected_binding):
    binding = payload.get("binding") if isinstance(payload, dict) else None
    if binding != expected_binding:
        raise ValueError("long dispatch state binding differs")
    return binding


def _validated_dispatch_state(payload, expected_binding):
    if not isinstance(payload, dict):
        raise ValueError("long dispatch state must be a mapping")
    if payload.get("schema") != LONG_DISPATCH_SCHEMA:
        raise ValueError("long dispatch state schema is invalid")
    if payload.get("status") not in ("starting", "running", "failed"):
        raise ValueError("long dispatch state status is invalid")
    token = payload.get("token")
    if not _valid_dispatch_token(token):
        raise ValueError("long dispatch state token is invalid")
    attempt = payload.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt <= 0
    ):
        raise ValueError("long dispatch attempt is invalid")
    created_at = payload.get("created_at")
    if (
        not isinstance(created_at, (int, float))
        or isinstance(created_at, bool)
        or not math.isfinite(float(created_at))
        or created_at < 0
    ):
        raise ValueError("long dispatch creation time is invalid")
    _validated_dispatch_binding(payload, expected_binding)
    return dict(payload)


def _validated_startup_ack(payload, expected_binding):
    if not isinstance(payload, dict):
        raise ValueError("long startup ack must be a mapping")
    if payload.get("schema") != LONG_STARTUP_ACK_SCHEMA:
        raise ValueError("long startup ack schema is invalid")
    if not _valid_dispatch_token(payload.get("token")):
        raise ValueError("long startup ack token is invalid")
    _validated_dispatch_binding(payload, expected_binding)
    pid = payload.get("pid")
    start_time = payload.get("process_start_time")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(start_time, str)
        or not start_time
    ):
        raise ValueError("long startup ack process identity is invalid")
    return dict(payload)


def _validated_completion_receipt(
        payload, expected_binding, summary_path, handoff_path):
    if not isinstance(payload, dict):
        raise ValueError("long completion receipt must be a mapping")
    if payload.get("schema") != LONG_COMPLETION_SCHEMA:
        raise ValueError("long completion receipt schema is invalid")
    if not _valid_dispatch_token(payload.get("token")):
        raise ValueError("long completion receipt token is invalid")
    _validated_dispatch_binding(payload, expected_binding)
    pid = payload.get("pid")
    start_time = payload.get("process_start_time")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(start_time, str)
        or not start_time
    ):
        raise ValueError("long completion process identity is invalid")
    if payload.get("summary") != str(summary_path):
        raise ValueError("long completion summary path differs")
    if payload.get("handoff") != str(handoff_path):
        raise ValueError("long completion handoff path differs")
    if not summary_path.is_file() or not handoff_path.is_file():
        raise ValueError("long completion artifacts are missing")
    if payload.get("summary_sha256") != file_sha256(summary_path):
        raise ValueError("long completion summary digest differs")
    if payload.get("handoff_sha256") != file_sha256(handoff_path):
        raise ValueError("long completion handoff digest differs")
    result = dict(payload)
    result["status"] = "completed"
    result["already_dispatched"] = True
    return result


def _live_process_identity(pid, expected_start_time):
    return (
        isinstance(expected_start_time, str)
        and expected_start_time
        and _process_start_time(pid) == expected_start_time
    )


def _dispatch_long_run(args, best_json, popen_factory):
    best_json = Path(best_json)
    best = _load_json(best_json)
    binding = long_dispatch_binding(best)
    long_root = _output_root(args) / "long_run"
    dispatch_path = long_root / "dispatch.json"
    ack_path = long_root / "startup_ack.json"
    completion_path = long_root / "completion.json"
    summary_path = long_root / "long_summary.json"
    handoff_path = long_root / "sidecar_handoff.json"
    long_root.mkdir(parents=True, exist_ok=True)

    if completion_path.is_file():
        return _validated_completion_receipt(
            _load_json(completion_path),
            binding,
            summary_path,
            handoff_path,
        )

    existing = None
    prior_attempt = 0
    if dispatch_path.is_file():
        raw_dispatch = _load_json(dispatch_path)
        if (
            isinstance(raw_dispatch, dict)
            and raw_dispatch.get("status") == "launched"
            and raw_dispatch.get("schema") is None
        ):
            legacy_pid = raw_dispatch.get("pid")
            if _process_start_time(legacy_pid) is not None:
                raise RuntimeError(
                    "live legacy long dispatch lacks exact process identity"
                )
            prior_attempt = 1
        else:
            existing = _validated_dispatch_state(raw_dispatch, binding)
            prior_attempt = existing["attempt"]
    if existing is not None:
        matching_ack = None
        if ack_path.is_file():
            ack = _validated_startup_ack(_load_json(ack_path), binding)
            if ack["token"] == existing["token"]:
                matching_ack = ack
        if matching_ack is not None and _live_process_identity(
                matching_ack["pid"],
                matching_ack["process_start_time"]):
            result = dict(existing)
            result.update({
                "status": "running",
                "pid": matching_ack["pid"],
                "process_start_time": matching_ack[
                    "process_start_time"
                ],
                "already_dispatched": True,
            })
            atomic_write_json(dispatch_path, result)
            return result
        if (
            existing["status"] in ("starting", "running")
            and _live_process_identity(
                existing.get("pid"),
                existing.get("process_start_time"),
            )
        ):
            result = dict(existing)
            result["already_dispatched"] = True
            return result
        if (
            existing["status"] == "starting"
            and matching_ack is None
            and time.time() - float(existing["created_at"])
            < LONG_STARTUP_GRACE_SECONDS
        ):
            result = dict(existing)
            result["already_dispatched"] = True
            return result

    attempt = prior_attempt + 1
    token = uuid.uuid4().hex
    command = command_for_long_run(
        args,
        best_json,
        dispatch_token=token,
        startup_ack=ack_path,
        completion_receipt=completion_path,
    )
    stdout_path = long_root / "long_stdout.log"
    starting = {
        "schema": LONG_DISPATCH_SCHEMA,
        "status": "starting",
        "token": token,
        "binding": binding,
        "attempt": attempt,
        "created_at": time.time(),
        "command": command,
        "stdout": str(stdout_path),
        "already_dispatched": False,
    }
    atomic_write_json(dispatch_path, starting)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    with stdout_path.open("a", encoding="utf-8", errors="replace") as stdout:
        process = popen_factory(
            command,
            cwd=args.repo_root,
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start_time = _process_start_time(int(process.pid))
    result = dict(starting)
    result.update({
        "status": "running" if start_time is not None else "failed",
        "pid": int(process.pid),
        "process_start_time": start_time,
        "already_dispatched": False,
    })
    atomic_write_json(dispatch_path, result)
    return result


def publish_final_selection(
        args, candidates, popen_factory=subprocess.Popen,
        dispatch_long=True):
    output_root = _output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    best_json = output_root / "best.json"
    best = select_best_trial(candidates)
    study_binding = study_receipt_binding(write_study_contract(args))
    if best is None:
        receipt = {
            "selection_status": "no_feasible_trial",
            "feasible_trial_count": 0,
            "long_dispatched": False,
            "study_binding": study_binding,
        }
        receipt.update(_provenance_binding(args))
        atomic_write_json(best_json, receipt)
        return receipt

    stable, link_method = _publish_best_checkpoint(args, best)
    receipt = dict(best)
    receipt["study_binding"] = study_binding
    receipt["selection_status"] = "feasible_best"
    receipt["short_checkpoint_source"] = receipt["checkpoint"]
    receipt["checkpoint"] = str(stable)
    receipt["checkpoint_sha256"] = file_sha256(stable)
    receipt["checkpoint_publication"] = link_method
    receipt["long_dispatched"] = False
    receipt.update(_provenance_binding(args))
    atomic_write_json(best_json, receipt)

    if dispatch_long:
        dispatch = _dispatch_long_run(args, best_json, popen_factory)
        receipt["long_dispatch"] = dispatch
        receipt["long_dispatched"] = True
        atomic_write_json(best_json, receipt)
    return receipt


def _provenance_binding(args):
    manifest_path = getattr(args, "provenance_manifest", None)
    if not manifest_path:
        raise ValueError("a run provenance manifest is required")
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    expected_manifest_fields = {
        "schema",
        "repo_root",
        "output_root",
        "data_root",
        "base_checkpoint",
        "pointnet_checkpoint",
        "source_snapshot",
        "environment",
        "environment_sha256",
        "inputs",
        "inputs_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("run provenance manifest fields are invalid")
    if manifest.get("schema") != "mcln-retrain-run-provenance-v1":
        raise ValueError("run provenance manifest schema is invalid")
    resolved_paths = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
    }
    for field, expected in resolved_paths.items():
        value = manifest.get(field)
        if not isinstance(value, str) or str(Path(value).resolve()) != expected:
            raise ValueError(
                "run provenance {} path differs".format(field)
            )
    snapshot = manifest.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("run provenance source snapshot is missing")
    digest = snapshot.get("manifest_sha256")
    if (
        not _valid_sha256(digest)
    ):
        raise ValueError("run provenance source snapshot digest is invalid")
    base = manifest.get("base_checkpoint")
    pointnet = manifest.get("pointnet_checkpoint")
    if not isinstance(base, dict) or not isinstance(pointnet, dict):
        raise ValueError("run provenance checkpoint inputs are missing")
    base_digest = base.get("sha256")
    pointnet_digest = pointnet.get("sha256")
    for label, value in (
            ("base checkpoint", base_digest),
            ("PointNet checkpoint", pointnet_digest)):
        if (
            not _valid_sha256(value)
        ):
            raise ValueError(
                "run provenance {} digest is invalid".format(label)
            )
    if base_digest.lower() != args.base_sha256.lower():
        raise ValueError("run provenance base checkpoint digest differs")
    inputs_path = manifest.get("inputs")
    environment_path = manifest.get("environment")
    if not isinstance(inputs_path, str) or not isinstance(
            environment_path, str):
        raise ValueError("run provenance referenced paths are invalid")
    inputs_path = Path(inputs_path).resolve()
    environment_path = Path(environment_path).resolve()
    for label, path, expected_digest in (
            ("inputs", inputs_path, manifest.get("inputs_sha256")),
            ("environment", environment_path,
             manifest.get("environment_sha256"))):
        if not _valid_sha256(expected_digest):
            raise ValueError(
                "run provenance {} digest is invalid".format(label)
            )
        if not path.is_file() or file_sha256(path) != expected_digest.lower():
            raise ValueError(
                "run provenance {} artifact differs".format(label)
            )

    inputs = _load_json(inputs_path)
    if set(inputs) != {
            "schema", "data_root", "base_checkpoint",
            "pointnet_checkpoint"}:
        raise ValueError("run provenance inputs fields are invalid")
    if inputs.get("schema") != "mcln-retrain-inputs-v1":
        raise ValueError("run provenance inputs schema is invalid")
    inputs_data_root = inputs.get("data_root")
    if (
        not isinstance(inputs_data_root, str)
        or str(Path(inputs_data_root).resolve()) != resolved_paths["data_root"]
    ):
        raise ValueError("run provenance inputs data root differs")
    if inputs.get("base_checkpoint") != base:
        raise ValueError("run provenance inputs base checkpoint differs")
    if inputs.get("pointnet_checkpoint") != pointnet:
        raise ValueError("run provenance inputs PointNet checkpoint differs")

    for label, record, requested_path in (
            ("base checkpoint", base, args.base_checkpoint),
            ("PointNet checkpoint", pointnet, args.pp_checkpoint)):
        recorded_path = record.get("path")
        if (
            recorded_path is not None
            and (
                not isinstance(recorded_path, str)
                or str(Path(recorded_path).resolve())
                != str(Path(requested_path).resolve())
            )
        ):
            raise ValueError(
                "run provenance {} path differs".format(label)
            )

    environment = _load_json(environment_path)
    python_executable = environment.get("python_executable")
    resolved_python = str(Path(args.python_bin).resolve())
    if (
        not isinstance(python_executable, str)
        or str(Path(python_executable).resolve()) != resolved_python
    ):
        raise ValueError("run provenance environment python differs")
    return {
        "provenance_manifest": str(manifest_path.resolve()),
        "repo_root": resolved_paths["repo_root"],
        "data_root": resolved_paths["data_root"],
        "python_bin": resolved_python,
        "run_manifest_sha256": file_sha256(manifest_path),
        "environment_sha256": manifest["environment_sha256"].lower(),
        "inputs_sha256": manifest["inputs_sha256"].lower(),
        "source_snapshot_digest": digest,
        "pointnet_checkpoint_sha256": pointnet_digest,
    }


def _receipt_path_for_trial(args, frozen_trial):
    relative = frozen_trial.user_attrs.get("receipt")
    if not isinstance(relative, str) or not relative:
        raise ValueError("complete trial lacks a receipt path")
    path = Path(relative)
    if not path.is_absolute():
        path = _output_root(args) / path
    return path


def _candidate_for_frozen_trial(args, baseline_receipt, frozen_trial):
    if frozen_trial.state != TrialState.COMPLETE:
        raise ValueError("trial is not COMPLETE")
    receipt_path = _receipt_path_for_trial(args, frozen_trial)
    trial_receipt = _load_json(receipt_path)
    candidate = candidate_from_trial_receipt(
        frozen_trial.number,
        baseline_receipt,
        trial_receipt,
        receipt_path=os.path.relpath(str(receipt_path), str(_output_root(args))),
        expected_study_binding=_expected_study_binding(args),
        expected_trial_params=frozen_trial.params,
    )
    expected_attrs = {
        "study_contract_digest": trial_receipt[
            "study_binding"
        ]["study_contract_digest"],
        "receipt_sha256": file_sha256(receipt_path),
        "optimizer_groups_sha256": candidate[
            "optimizer_groups_sha256"
        ],
    }
    for name, expected in expected_attrs.items():
        if frozen_trial.user_attrs.get(name) != expected:
            raise ValueError(
                "complete trial {} binding differs from its receipt".format(
                    name
                )
            )
    return candidate


def _valid_complete_candidates(args, study, baseline_receipt):
    candidates = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        try:
            candidates.append(
                _candidate_for_frozen_trial(args, baseline_receipt, trial)
            )
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError) as error:
            raise ValueError(
                "COMPLETE trial {} has an invalid bound receipt: {}".format(
                    trial.number, error
                )
            )
    return candidates


def _write_reports(args, study, candidates):
    output_root = _output_root(args)
    rows = []
    candidates_by_number = {
        candidate["trial_number"]: candidate for candidate in candidates
    }
    for trial in study.trials:
        candidate = candidates_by_number.get(trial.number)
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "receipt": trial.user_attrs.get("receipt", ""),
            "feasible": candidate["feasible"] if candidate else "",
        }
        row.update(trial.params)
        rows.append(row)
    csv_path = output_root / "trials.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else [
        "trial_number", "state"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(output_root / "study_summary.json", {
        "study_name": study.study_name,
        "trial_count": len(study.trials),
        "valid_complete_count": len(candidates),
        "state_counts": {
            state.name: sum(trial.state == state for trial in study.trials)
            for state in TrialState
        },
    })


def _build_study_contract(args):
    provenance = _provenance_binding(args)
    split_metadata = dict(AUTHORITATIVE_SCANREFER_SPLIT_METADATA)
    contract = {
        "schema": STUDY_CONTRACT_SCHEMA,
        "study_name": args.study_name,
        "storage_identity": _sqlite_storage_identity(args.storage),
        "model": "MCLN+source-choice-selector",
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "base_checkpoint_sha256": args.base_sha256,
        "pointnet_checkpoint": os.path.abspath(args.pp_checkpoint),
        "pointnet_checkpoint_sha256": provenance[
            "pointnet_checkpoint_sha256"
        ],
        "target_successful_trials": args.target_successful_trials,
        "trial_epochs": [55, 56],
        "long_epochs": [55, 100],
        "sampler": {
            "name": "TPESampler",
            "seed": 0,
            "n_startup_trials": 5,
        },
        "seed_presets": [dict(preset) for preset in seed_presets()],
        "fit_calibration": split_metadata,
        "split_metadata_sha256": canonical_json_sha256(split_metadata),
        "data_digests": {
            "fit_scene_sha256": split_metadata["fit_scene_sha256"],
            "calibration_scene_sha256": split_metadata[
                "calibration_scene_sha256"
            ],
            "mapping_sha256": split_metadata["mapping_sha256"],
        },
        "official_validation_used_for_tuning": False,
        "failed_trials_count_toward_target": False,
        "short_checkpoint_used_for_long_initialization": False,
        "source_pair": SOURCE_PAIR,
        "fixed": {
            "batch_size": 18,
            "num_workers": 4,
            "weight_decay": 5e-4,
            "clip_norm": 0.1,
            "selector_hidden_dim": 288,
            "continuation_horizon": 46,
        },
        "search_space": {
            "decoder_lr": {"low": 5e-6, "high": 4e-5, "log": True},
            "mask_head_lr_multiplier": [1.0, 2.0, 4.0],
            "selector_lr": {"low": 2e-4, "high": 2e-3, "log": True},
            "mask_loss_scale": {"low": 0.5, "high": 4.0, "log": True},
            "consistency_loss_scale": {
                "low": 0.1, "high": 2.0, "log": True,
            },
            "selector_loss_weight": {
                "low": 0.1, "high": 1.0, "log": True,
            },
            "selector_min_iou_gap": [0.02, 0.03, 0.05, 0.08],
        },
    }
    contract.update(provenance)
    contract["contract_digest"] = canonical_json_sha256(
        canonical_study_contract_payload(contract)
    )
    return contract


def write_study_contract(args):
    proposed = _build_study_contract(args)
    path = _output_root(args) / "study_contract.json"
    if path.is_file():
        existing = _validated_study_contract(_load_json(path))
        if (
            existing["contract_digest"] != proposed["contract_digest"]
            or canonical_study_contract_payload(existing)
            != canonical_study_contract_payload(proposed)
        ):
            raise ValueError(
                "existing study contract differs from the requested contract"
            )
        return existing

    if _write_json_no_replace(path, proposed):
        return proposed
    existing = _validated_study_contract(_load_json(path))
    if (
        existing["contract_digest"] != proposed["contract_digest"]
        or canonical_study_contract_payload(existing)
        != canonical_study_contract_payload(proposed)
    ):
        raise ValueError(
            "concurrent study contract differs from the requested contract"
        )
    return existing


def publish_baseline_metrics(args, baseline_receipt):
    if (
        not isinstance(baseline_receipt, dict)
        or baseline_receipt.get("schema") != TRIAL_RECEIPT_SCHEMA
        or baseline_receipt.get("mode") != "baseline"
        or baseline_receipt.get("selection_epoch") != 54
    ):
        raise ValueError("baseline receipt contract is invalid")
    expected_binding = _expected_study_binding(args)
    _validate_receipt_study_binding(
        baseline_receipt, expected=expected_binding
    )
    if baseline_receipt.get("checkpoint") is not None:
        raise ValueError("baseline receipt checkpoint is invalid")
    if baseline_receipt.get("trial_params") is not None:
        raise ValueError("baseline receipt trial params are invalid")
    if baseline_receipt.get("optimizer_groups") not in ([], ()):
        raise ValueError("baseline receipt optimizer groups are invalid")
    if "losses" in baseline_receipt:
        raise ValueError("baseline receipt must not contain training losses")
    metrics_by_epoch = baseline_receipt.get("metrics")
    if not isinstance(metrics_by_epoch, dict) or set(metrics_by_epoch) != {
        "epoch_54"
    }:
        raise ValueError("baseline receipt metrics are invalid")
    metrics = validate_metrics_receipt(metrics_by_epoch["epoch_54"])
    artifact = {
        "schema": BASELINE_ARTIFACT_SCHEMA,
        "study_binding": expected_binding,
        "baseline_receipt_sha256": canonical_json_sha256(baseline_receipt),
        "metrics": metrics,
    }
    path = _output_root(args) / "baseline_metrics.json"
    if path.is_file():
        existing = _load_json(path)
        if existing != artifact:
            raise ValueError(
                "existing baseline artifact differs from the study binding"
            )
    elif not _write_json_no_replace(path, artifact):
        existing = _load_json(path)
        if existing != artifact:
            raise ValueError(
                "concurrent baseline artifact differs from the study binding"
            )
    return metrics


def _subprocess_environment(args):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONPATH"] = os.pathsep.join([
        args.repo_root,
        str(Path(args.repo_root) / "pointnet2"),
        environment.get("PYTHONPATH", ""),
    ])
    return environment


def _process_start_time(pid):
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
    ):
        return None
    try:
        payload = Path("/proc/{}/stat".format(pid)).read_text(
            encoding="utf-8"
        )
    except (OSError, ValueError):
        return None
    closing_parenthesis = payload.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = payload[closing_parenthesis + 2:].split()
    if len(fields) <= 19:
        return None
    return fields[19]


def _record_trial_subprocess(trial, process):
    pid = int(process.pid)
    start_time = _process_start_time(pid)
    if start_time is None:
        raise RuntimeError("trial subprocess identity could not be recorded")
    trial.set_user_attr("subprocess_pid", pid)
    trial.set_user_attr("subprocess_start_time", start_time)


def _run_command(args, command, stdout_path, on_started=None):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8", errors="replace") as stdout:
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            env=_subprocess_environment(args),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if on_started is not None:
                on_started(process)
            process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        return process


def _ensure_baseline(args):
    receipt_path = _output_root(args) / "baseline" / "receipt.json"
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        if (
            receipt.get("schema") != TRIAL_RECEIPT_SCHEMA
            or receipt.get("mode") != "baseline"
            or receipt.get("selection_epoch") != 54
        ):
            raise ValueError("existing baseline receipt contract is invalid")
        try:
            _validate_receipt_study_binding(
                receipt, expected=_expected_study_binding(args)
            )
        except ValueError as error:
            raise ValueError(
                "existing baseline study binding is invalid: {}".format(error)
            )
        if receipt.get("checkpoint") is not None:
            raise ValueError("existing baseline checkpoint binding is invalid")
        if receipt.get("trial_params") is not None:
            raise ValueError("existing baseline trial params binding is invalid")
        if receipt.get("optimizer_groups") not in ([], ()):
            raise ValueError("existing baseline optimizer binding is invalid")
        if "losses" in receipt:
            raise ValueError(
                "existing baseline must not contain training losses"
            )
        metrics = receipt.get("metrics", {})
        if not isinstance(metrics, dict) or set(metrics) != {"epoch_54"}:
            raise ValueError("existing baseline metrics are invalid")
        validate_metrics_receipt(metrics["epoch_54"])
        return receipt
    result = _run_command(
        args,
        command_for_baseline(args),
        _output_root(args) / "baseline" / "stdout.log",
    )
    if result.returncode != 0:
        raise RuntimeError("epoch-54 calibration baseline subprocess failed")
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema") != TRIAL_RECEIPT_SCHEMA
        or receipt.get("mode") != "baseline"
        or receipt.get("selection_epoch") != 54
    ):
        raise ValueError("baseline subprocess wrote an invalid receipt")
    _validate_receipt_study_binding(
        receipt, expected=_expected_study_binding(args)
    )
    if receipt.get("checkpoint") is not None:
        raise ValueError("baseline subprocess wrote a checkpoint binding")
    if receipt.get("trial_params") is not None:
        raise ValueError("baseline subprocess wrote trial params")
    if receipt.get("optimizer_groups") not in ([], ()):
        raise ValueError("baseline subprocess wrote optimizer groups")
    if "losses" in receipt:
        raise ValueError("baseline subprocess wrote training losses")
    validate_metrics_receipt(receipt["metrics"]["epoch_54"])
    return receipt


def _infeasible_penalty(assessment):
    deficit = sum(max(0.0, -value) for value in assessment["deltas"].values())
    return -1000.0 - 100.0 * deficit - len(
        assessment["constraint_failures"]
    )


def _objective(args, baseline_receipt, trial):
    require_minimum_free_space(_output_root(args))
    params = suggest_trial_params(trial)
    trial_dir = _trial_directory(args, trial.number)
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = command_for_trial(args, params, trial.number)
    (trial_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    result = _run_command(
        args,
        command,
        trial_dir / "stdout.log",
        on_started=lambda process: _record_trial_subprocess(
            trial, process
        ),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "trial {} subprocess failed with code {}".format(
                trial.number, result.returncode
            )
        )
    receipt_path = trial_dir / "receipt.json"
    candidate = candidate_from_trial_receipt(
        trial.number,
        baseline_receipt,
        _load_json(receipt_path),
        receipt_path=os.path.relpath(
            str(receipt_path), str(_output_root(args))
        ),
        expected_study_binding=_expected_study_binding(args),
        expected_trial_params=params,
    )
    trial.set_user_attr("receipt", candidate["receipt"])
    trial.set_user_attr("feasible", candidate["feasible"])
    trial.set_user_attr(
        "study_contract_digest",
        _expected_study_binding(args)["study_contract_digest"],
    )
    trial.set_user_attr("receipt_sha256", file_sha256(receipt_path))
    trial.set_user_attr(
        "optimizer_groups_sha256",
        candidate["optimizer_groups_sha256"],
    )
    if candidate["feasible"]:
        return float(candidate["objective"])
    return _infeasible_penalty(candidate)


def fail_stale_running_trials(study):
    active = []
    stale = []
    for trial in study.trials:
        if trial.state != TrialState.RUNNING:
            continue
        pid = trial.user_attrs.get("subprocess_pid")
        expected_start = trial.user_attrs.get("subprocess_start_time")
        actual_start = _process_start_time(pid)
        if (
            actual_start is not None
            and isinstance(expected_start, str)
            and actual_start == expected_start
        ):
            active.append(trial.number)
        else:
            stale.append(trial.number)
    if active:
        raise RuntimeError(
            "active RUNNING trial subprocesses still exist: {}".format(
                ", ".join(str(number) for number in active)
            )
        )
    for trial_number in stale:
        study.tell(trial_number, state=TrialState.FAIL)
    return len(stale)


@contextmanager
def _study_owner_lock(args):
    lock_path = _output_root(args) / "control" / "study_owner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "another orchestrator holds the study owner lock"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _study_exists(storage, study_name):
    return any(
        summary.study_name == study_name
        for summary in optuna.get_all_study_summaries(storage=storage)
    )


def _open_owned_study(args, sampler, contract, contract_existed):
    existed = _study_exists(args.storage, args.study_name)
    if not existed and contract_existed:
        raise ValueError(
            "study ownership is missing for the existing output contract"
        )
    if existed:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
            sampler=sampler,
        )
    else:
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                direction="maximize",
                sampler=sampler,
                load_if_exists=False,
            )
        except optuna.exceptions.DuplicatedStudyError as error:
            raise RuntimeError(
                "study appeared concurrently outside the owner lock"
            ) from error

    if study.direction.name != "MAXIMIZE":
        raise ValueError("existing study direction is not maximize")
    expected_attrs = {
        "schema": STUDY_BINDING_SCHEMA,
        "contract_digest": contract["contract_digest"],
    }
    if existed:
        if study.user_attrs != expected_attrs:
            raise ValueError("existing study binding ownership differs")
    else:
        if study.trials or study.user_attrs:
            raise ValueError("new study is not empty and unowned")
        for name, value in expected_attrs.items():
            study.set_user_attr(name, value)
    return study


def _cleanup_all_trial_weights(args):
    trials_root = _output_root(args) / "trials"
    if not trials_root.is_dir():
        return
    for trial_dir in sorted(trials_root.glob("trial_*")):
        if trial_dir.is_dir():
            cleanup_trial_checkpoints(trial_dir)


def run_study(args):
    output_root = _output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    with _study_owner_lock(args):
        require_minimum_free_space(output_root)
        contract_path = output_root / "study_contract.json"
        contract_existed = contract_path.is_file()
        contract = write_study_contract(args)
        sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=5)
        study = _open_owned_study(
            args, sampler, contract, contract_existed
        )
        fail_stale_running_trials(study)
        baseline_receipt = _ensure_baseline(args)
        publish_baseline_metrics(args, baseline_receipt)
        enqueue_seed_presets_if_new(study)

        attempts = 0
        while attempts < args.max_process_attempts:
            candidates = _valid_complete_candidates(
                args, study, baseline_receipt
            )
            valid_numbers = {
                candidate["trial_number"] for candidate in candidates
            }
            remaining = remaining_successful_trials(
                study,
                args.target_successful_trials,
                receipt_is_valid=lambda trial: trial.number in valid_numbers,
            )
            _write_reports(args, study, candidates)
            if remaining == 0:
                break
            study.optimize(
                lambda trial: _objective(args, baseline_receipt, trial),
                n_trials=1,
                catch=(Exception,),
            )
            attempts += 1
            candidates = _valid_complete_candidates(
                args, study, baseline_receipt
            )
            publish_final_selection(
                args, candidates, dispatch_long=False
            )
            _cleanup_all_trial_weights(args)

        candidates = _valid_complete_candidates(
            args, study, baseline_receipt
        )
        valid_numbers = {
            candidate["trial_number"] for candidate in candidates
        }
        remaining = remaining_successful_trials(
            study,
            args.target_successful_trials,
            receipt_is_valid=lambda trial: trial.number in valid_numbers,
        )
        _write_reports(args, study, candidates)
        if remaining != 0:
            raise RuntimeError(
                "study stopped after {} attempts with {} successful trials missing".format(
                    attempts, remaining
                )
            )
        result = publish_final_selection(
            args, candidates, dispatch_long=True
        )
        _cleanup_all_trial_weights(args)
        return result


def parse_args(argv=None):
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pp-checkpoint", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=str(root / "pretained model" / "ckpt_epoch_54.pth"),
    )
    parser.add_argument("--base-sha256", default=DEFAULT_BASE_SHA256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=29600)
    parser.add_argument("--target-successful-trials", type=int, default=20)
    parser.add_argument("--max-process-attempts", type=int, default=MAX_PROCESS_ATTEMPTS)
    parser.add_argument(
        "--python-bin",
        default="/root/miniconda3/envs/bdetr/bin/python",
    )
    parser.add_argument("--repo-root", default=str(root))
    parser.add_argument("--provenance-manifest", required=True)
    args = parser.parse_args(argv)
    if args.storage is None:
        args.storage = "sqlite:///{}".format(
            Path(args.output_root) / "optuna.db"
        )
    if args.target_successful_trials != 20:
        raise ValueError("formal study requires exactly 20 successful trials")
    if args.max_process_attempts != MAX_PROCESS_ATTEMPTS:
        raise ValueError("formal study requires the 60-attempt safety cap")
    return args


def main():
    args = parse_args()
    result = run_study(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
