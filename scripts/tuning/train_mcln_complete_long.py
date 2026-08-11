#!/usr/bin/env python
"""Full-train epoch-55-to-100 continuation with five-metric Pareto retention."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import shutil
from pathlib import Path

import optuna
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from main_utils import load_checkpoint, parse_option
from scripts.tuning.mcln_optuna_contract import (
    suggest_trial_params,
    validate_metrics_receipt,
)
from scripts.tuning.optuna_mcln_complete_retrain import (
    LONG_COMPLETION_SCHEMA,
    LONG_DISPATCH_SCHEMA,
    LONG_STARTUP_ACK_SCHEMA,
    _process_start_time,
    _valid_dispatch_token,
    _validated_dispatch_state,
    _validated_startup_ack,
    long_dispatch_binding,
)
from scripts.tuning.train_mcln_optuna_trial import (
    DEFAULT_CONTINUATION_HORIZON,
    SELECTOR_TARGET,
    SOURCE_PAIR,
    atomic_write_json,
    build_cosine_scheduler,
    file_sha256,
    seed_everything,
    verify_base_checkpoint,
)
from train_dist_mod import TrainTester


OFFICIAL_SAMPLE_COUNT = 9508
FINAL_EPOCH = 100
VALIDATION_EPOCHS = (60, 65, 70, 75, 80, 85, 90, 95, 100)

POSITION025_REQUIRED_HITS = 5610
POSITION050_REQUIRED_HITS = 4621
MASK025_REQUIRED_HITS = 5582
MASK050_REQUIRED_HITS = 4821
MIOU_STRICT_THRESHOLD = 0.4472

GIB = 1024 ** 3
INITIAL_MIN_FREE_BYTES = 8 * GIB
LONG_RUN_SAFETY_RESERVE_BYTES = 512 * 1024 ** 2
LONG_SUMMARY_SCHEMA = "mcln-complete-long-summary-v2"
VALIDATION_RECEIPT_SCHEMA = "mcln-complete-long-validation-v1"

TARGET_VECTOR = (
    0.59,
    POSITION050_REQUIRED_HITS / float(OFFICIAL_SAMPLE_COUNT),
    MASK025_REQUIRED_HITS / float(OFFICIAL_SAMPLE_COUNT),
    MASK050_REQUIRED_HITS / float(OFFICIAL_SAMPLE_COUNT),
    MIOU_STRICT_THRESHOLD,
)


def long_train_epochs(base_epoch=54, final_epoch=FINAL_EPOCH):
    if (
        not isinstance(base_epoch, int)
        or isinstance(base_epoch, bool)
        or not isinstance(final_epoch, int)
        or isinstance(final_epoch, bool)
        or base_epoch < 0
        or final_epoch <= base_epoch
    ):
        raise ValueError("long training epoch range is invalid")
    return tuple(range(base_epoch + 1, final_epoch + 1))


def validation_epochs():
    return VALIDATION_EPOCHS


def _validated_checkpoint_size(checkpoint_size):
    if (
        not isinstance(checkpoint_size, int)
        or isinstance(checkpoint_size, bool)
        or checkpoint_size <= 0
    ):
        raise ValueError("checkpoint size must be a positive integer")
    return checkpoint_size


def projected_peak_capacity_bytes(checkpoint_size):
    """Worst fresh-run peak: three Pareto, latest, and one temp inode."""
    checkpoint_size = _validated_checkpoint_size(checkpoint_size)
    return 5 * checkpoint_size + LONG_RUN_SAFETY_RESERVE_BYTES


def required_initial_free_bytes(checkpoint_size):
    return max(
        INITIAL_MIN_FREE_BYTES,
        projected_peak_capacity_bytes(checkpoint_size),
    )


def required_resume_free_bytes(checkpoint_size):
    """Incremental resume need; existing retained artifacts are already used."""
    checkpoint_size = _validated_checkpoint_size(checkpoint_size)
    return 2 * checkpoint_size + LONG_RUN_SAFETY_RESERVE_BYTES


def require_long_run_capacity(
        checkpoint_size, resume, reported_free_bytes):
    checkpoint_size = _validated_checkpoint_size(checkpoint_size)
    if not isinstance(resume, bool):
        raise ValueError("resume capacity mode must be boolean")
    if (
        not isinstance(reported_free_bytes, int)
        or isinstance(reported_free_bytes, bool)
        or reported_free_bytes < 0
    ):
        raise ValueError("reported free space must be a non-negative integer")
    required = (
        required_resume_free_bytes(checkpoint_size)
        if resume
        else required_initial_free_bytes(checkpoint_size)
    )
    if reported_free_bytes < required:
        raise ValueError(
            "free space {} is below required long-run capacity {}".format(
                reported_free_bytes, required
            )
        )
    return reported_free_bytes


def require_long_run_filesystem_capacity(
        path, checkpoint_size, resume, reported_free_bytes=None):
    path = Path(path)
    if reported_free_bytes is None:
        reported_free_bytes = shutil.disk_usage(str(path)).free
    return require_long_run_capacity(
        checkpoint_size,
        resume=resume,
        reported_free_bytes=reported_free_bytes,
    )


def _validated_official_metrics(metrics):
    return validate_metrics_receipt(
        metrics, expected_sample_count=OFFICIAL_SAMPLE_COUNT
    )


def _metric_vector(metrics):
    metrics = _validated_official_metrics(metrics)
    count = float(OFFICIAL_SAMPLE_COUNT)
    return (
        metrics["position"]["learned_selector"]["hits025"] / count,
        metrics["position"]["learned_selector"]["hits050"] / count,
        metrics["mask"]["hits025"] / count,
        metrics["mask"]["hits050"] / count,
        metrics["mask"]["miou"],
    )


def dominates(first, second):
    first_vector = _metric_vector(first)
    second_vector = _metric_vector(second)
    return (
        all(left >= right for left, right in zip(first_vector, second_vector))
        and any(left > right for left, right in zip(first_vector, second_vector))
    )


def target_distance(metrics):
    vector = _metric_vector(metrics)
    return sum(
        max(0.0, target - value)
        for target, value in zip(TARGET_VECTOR, vector)
    )


def mask_balance(metrics):
    vector = _metric_vector(metrics)
    mask050_ratio = vector[3] / TARGET_VECTOR[3]
    miou_ratio = vector[4] / TARGET_VECTOR[4]
    return min(mask050_ratio, miou_ratio)


def release_gate_status(metrics):
    metrics = _validated_official_metrics(metrics)
    learned = metrics["position"]["learned_selector"]
    mask = metrics["mask"]
    checks = {
        "position025": learned["hits025"] >= POSITION025_REQUIRED_HITS,
        "position050": learned["hits050"] >= POSITION050_REQUIRED_HITS,
        "mask025": mask["hits025"] >= MASK025_REQUIRED_HITS,
        "mask050": mask["hits050"] >= MASK050_REQUIRED_HITS,
        "mask_miou": mask["miou"] > MIOU_STRICT_THRESHOLD,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "exact": {
            "position025_hits": learned["hits025"],
            "position050_hits": learned["hits050"],
            "mask025_hits": mask["hits025"],
            "mask050_hits": mask["hits050"],
            "mask_miou": mask["miou"],
            "sample_count": OFFICIAL_SAMPLE_COUNT,
        },
    }


def _normalized_path(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _checkpoint_identity(path):
    path = Path(path)
    try:
        stat_result = path.stat()
    except OSError:
        return ("path", _normalized_path(path))
    return ("inode", int(stat_result.st_dev), int(stat_result.st_ino))


def _validated_candidate(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("Pareto candidate must be a mapping")
    epoch = candidate.get("epoch")
    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch not in VALIDATION_EPOCHS
    ):
        raise ValueError("Pareto candidate epoch is invalid")
    path = candidate.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("Pareto candidate path is invalid")
    if not Path(path).is_file():
        raise ValueError("Pareto candidate checkpoint is missing")
    result = copy.deepcopy(candidate)
    result["metrics"] = _validated_official_metrics(candidate.get("metrics"))
    return result


def _selected_roles(nondominated, max_keep):
    remaining = list(nondominated)
    selected = []
    selectors = (
        (
            "target_distance",
            lambda item: (
                -target_distance(item["metrics"]),
                item["epoch"],
            ),
        ),
        (
            "position025",
            lambda item: (
                item["metrics"]["position"]["learned_selector"]["hits025"],
                -item["epoch"],
            ),
        ),
        (
            "mask_balance",
            lambda item: (mask_balance(item["metrics"]), -item["epoch"]),
        ),
    )
    for role, key in selectors:
        if not remaining or len(selected) >= max_keep:
            break
        chosen = max(remaining, key=key)
        chosen = copy.deepcopy(chosen)
        chosen["retention_role"] = role
        selected.append(chosen)
        remaining = [
            item for item in remaining
            if _checkpoint_identity(item["path"])
            != _checkpoint_identity(chosen["path"])
        ]
    return selected


def plan_pareto_checkpoints(candidates, max_keep=3, protected_paths=()):
    if (
        not isinstance(max_keep, int)
        or isinstance(max_keep, bool)
        or max_keep <= 0
        or max_keep > 3
    ):
        raise ValueError("max_keep must be between one and three")
    validated = [_validated_candidate(item) for item in candidates]
    validated.sort(key=lambda item: (item["epoch"], item["path"]))

    unique = []
    duplicate_paths = []
    seen_identities = set()
    for candidate in validated:
        identity = _checkpoint_identity(candidate["path"])
        if identity in seen_identities:
            duplicate_paths.append(candidate["path"])
            continue
        seen_identities.add(identity)
        unique.append(candidate)

    nondominated = []
    for candidate in unique:
        if any(
            dominates(other["metrics"], candidate["metrics"])
            for other in unique
            if other is not candidate
        ):
            continue
        nondominated.append(candidate)

    selected = _selected_roles(nondominated, max_keep)
    selected_paths = {
        _normalized_path(candidate["path"]) for candidate in selected
    }
    deletion_paths = list(duplicate_paths)
    deletion_paths.extend(
        candidate["path"] for candidate in unique
        if _normalized_path(candidate["path"]) not in selected_paths
    )
    protected = {_normalized_path(path) for path in protected_paths}
    conflicts = [
        path for path in deletion_paths if _normalized_path(path) in protected
    ]
    if conflicts:
        raise ValueError("Pareto cleanup targets a protected checkpoint")

    return selected, deletion_paths


def cleanup_checkpoint_paths(paths, protected_paths=()):
    protected = {_normalized_path(path) for path in protected_paths}
    normalized = []
    for path_string in paths:
        path = Path(path_string)
        if _normalized_path(path) in protected:
            raise ValueError("checkpoint cleanup targets a protected path")
        normalized.append(path)
    for path in normalized:
        if path.exists():
            path.unlink()


def select_pareto_checkpoints(
        candidates, max_keep=3, protected_paths=()):
    selected, deletion_paths = plan_pareto_checkpoints(
        candidates,
        max_keep=max_keep,
        protected_paths=protected_paths,
    )
    cleanup_checkpoint_paths(
        deletion_paths, protected_paths=protected_paths
    )
    return selected


def publish_long_summary(
        summary_path, payload, cleanup_paths=(), protected_paths=()):
    retained = payload.get("retained")
    if not isinstance(retained, list):
        raise ValueError("long summary retained candidates are invalid")
    for candidate in retained:
        path = candidate.get("path") if isinstance(candidate, dict) else None
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError("long summary references a missing checkpoint")
    atomic_write_json(summary_path, payload)
    cleanup_checkpoint_paths(
        cleanup_paths, protected_paths=protected_paths
    )


def atomic_copy_checkpoint(source, destination, protected_paths=()):
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise ValueError("checkpoint source does not exist")
    protected = {_normalized_path(path) for path in protected_paths}
    if _normalized_path(destination) in protected:
        raise ValueError("checkpoint destination is protected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".{}.tmp".format(destination.name))
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(str(source), str(temporary))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _atomic_save_checkpoint(
        path, args, epoch, model, optimizer, scheduler, protected_paths=()):
    path = Path(path)
    protected = {_normalized_path(item) for item in protected_paths}
    if _normalized_path(path) in protected:
        raise ValueError("long checkpoint destination is protected")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}-{}.tmp".format(path.name, os.getpid()))
    state = {
        "config": args,
        "save_path": str(path),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    try:
        torch.save(state, str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _validate_best_json(best):
    if not isinstance(best, dict):
        raise ValueError("best.json must be a mapping")
    if best.get("selection_status") != "feasible_best":
        raise ValueError("long training requires a feasible best trial")
    params = best.get("trial_params")
    if not isinstance(params, dict):
        raise ValueError("best trial parameters are missing")
    fixed = optuna.trial.FixedTrial(dict(params))
    resolved = suggest_trial_params(fixed)
    if resolved != params:
        raise ValueError("best trial parameters are invalid")
    checkpoint = best.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        raise ValueError("published Optuna best checkpoint is missing")
    return dict(resolved)


def _apply_long_contract(shared_args, custom_args, params):
    shared_args.batch_size = 18
    shared_args.num_workers = 4
    shared_args.weight_decay = 5e-4
    shared_args.clip_norm = 0.1
    shared_args.rng_seed = 0
    shared_args.model = "MCLN"
    shared_args.num_decoder_layers = 6
    shared_args.dataset = ["scanrefer"]
    shared_args.test_dataset = "scanrefer"
    shared_args.joint_det = True
    shared_args.detect_intermediate = True
    shared_args.use_color = True
    shared_args.use_soft_token_loss = True
    shared_args.use_contrastive_align = True
    shared_args.butd = True
    shared_args.butd_gt = False
    shared_args.butd_cls = False
    shared_args.self_attend = True
    shared_args.augment_det = True
    shared_args.eval = False
    shared_args.eval_train = False
    shared_args.debug = False
    shared_args.frozen = False
    shared_args.small_lr = False
    shared_args.source_choice_selector_train_only = False
    shared_args.use_source_choice_selector = True
    shared_args.source_choice_selector_sources = SOURCE_PAIR
    shared_args.source_choice_selector_default_source = "default"
    shared_args.source_choice_selector_choice_target = SELECTOR_TARGET
    shared_args.source_choice_selector_hidden_dim = 288
    shared_args.eval_use_selector_choice_scores = True
    shared_args.lr = params["decoder_lr"]
    shared_args.lr_backbone = params["decoder_lr"] * 10.0
    shared_args.source_choice_selector_lr = params["selector_lr"]
    shared_args.mask_head_lr_multiplier = params["mask_head_lr_multiplier"]
    shared_args.mask_loss_scale = params["mask_loss_scale"]
    shared_args.consistency_loss_scale = params["consistency_loss_scale"]
    shared_args.source_choice_selector_loss_weight = (
        params["selector_loss_weight"]
    )
    shared_args.source_choice_selector_min_iou_gap = (
        params["selector_min_iou_gap"]
    )
    shared_args.expected_eval_sample_count = OFFICIAL_SAMPLE_COUNT
    shared_args.max_epoch = FINAL_EPOCH
    shared_args.log_dir = str(Path(custom_args.output_root) / "logs")
    shared_args.exp = "mcln_complete_long"
    shared_args.data_root = custom_args.data_root
    shared_args.pp_checkpoint = custom_args.pp_checkpoint
    shared_args.local_rank = 0
    return shared_args


def _initialize_distributed(custom_args):
    if not torch.cuda.is_available():
        raise RuntimeError("formal long runner requires CUDA")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:{}".format(custom_args.master_port),
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(seconds=5400),
        )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def _build_runtime(shared_args, custom_args):
    trainer = TrainTester(shared_args)
    train_loader, official_loader = trainer.get_loaders(shared_args)
    if len(train_loader) <= 0 or len(official_loader.dataset) != OFFICIAL_SAMPLE_COUNT:
        raise ValueError("full train or official validation loader is invalid")
    model = trainer.get_model(shared_args)
    criterion, set_criterion = trainer.get_criterion(shared_args)
    optimizer = trainer.get_optimizer(shared_args, model)
    scheduler = build_cosine_scheduler(
        optimizer,
        len(train_loader),
        continuation_horizon=DEFAULT_CONTINUATION_HORIZON,
    )
    model = model.cuda()
    model = DistributedDataParallel(
        model,
        device_ids=[0],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    latest = Path(custom_args.output_root) / "latest.pth"
    if latest.is_file():
        shared_args.checkpoint_path = str(latest)
        shared_args.reduce_lr = False
    else:
        shared_args.checkpoint_path = custom_args.base_checkpoint
        shared_args.reduce_lr = True
    load_checkpoint(shared_args, model, optimizer, scheduler)
    if shared_args.start_epoch < 55 or shared_args.start_epoch > 101:
        raise ValueError("long-run checkpoint resolved an invalid start epoch")
    return {
        "trainer": trainer,
        "train_loader": train_loader,
        "official_loader": official_loader,
        "model": model,
        "criterion": criterion,
        "set_criterion": set_criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "latest": latest,
    }


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validation_receipt_path(output_root, epoch):
    return (
        Path(output_root)
        / "validation_receipts"
        / "epoch_{:03d}.json".format(epoch)
    )


def _validated_validation_receipt(receipt, expected_epoch=None):
    if not isinstance(receipt, dict):
        raise ValueError("validation receipt must be a mapping")
    if receipt.get("schema") != VALIDATION_RECEIPT_SCHEMA:
        raise ValueError("validation receipt schema is invalid")
    epoch = receipt.get("epoch")
    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch not in VALIDATION_EPOCHS
        or (expected_epoch is not None and epoch != expected_epoch)
    ):
        raise ValueError("validation receipt epoch is invalid")
    digest = receipt.get("checkpoint_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("validation receipt checkpoint digest is invalid")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("validation receipt candidate path is invalid")
    result = dict(receipt)
    result["metrics"] = _validated_official_metrics(receipt.get("metrics"))
    return result


def _load_validation_receipts(output_root):
    receipts = {}
    root = Path(output_root) / "validation_receipts"
    if not root.is_dir():
        return receipts
    for path in sorted(root.glob("epoch_*.json")):
        receipt = _validated_validation_receipt(_load_json(path))
        epoch = receipt["epoch"]
        if path != _validation_receipt_path(output_root, epoch):
            raise ValueError("validation receipt filename is invalid")
        if epoch in receipts:
            raise ValueError("duplicate validation receipt epoch")
        receipts[epoch] = receipt
    return receipts


def _write_validation_receipt(output_root, epoch, metrics, candidate_path):
    candidate_path = Path(candidate_path)
    if not candidate_path.is_file():
        raise ValueError("validation candidate checkpoint is missing")
    receipt = {
        "schema": VALIDATION_RECEIPT_SCHEMA,
        "epoch": epoch,
        "metrics": _validated_official_metrics(metrics),
        "candidate": str(candidate_path),
        "checkpoint_sha256": file_sha256(candidate_path),
    }
    path = _validation_receipt_path(output_root, epoch)
    if path.is_file():
        existing = _validated_validation_receipt(
            _load_json(path), expected_epoch=epoch
        )
        if existing != receipt:
            raise ValueError("existing validation receipt differs")
        return existing
    atomic_write_json(path, receipt)
    return receipt


def _candidate_from_validation_receipt(receipt):
    receipt = _validated_validation_receipt(receipt)
    path = Path(receipt["candidate"])
    if not path.is_file():
        return None
    if file_sha256(path) != receipt["checkpoint_sha256"]:
        raise ValueError("validation candidate checkpoint digest differs")
    return {
        "epoch": receipt["epoch"],
        "path": str(path),
        "metrics": receipt["metrics"],
        "release_gate": release_gate_status(receipt["metrics"]),
    }


def _load_long_summary(summary_path):
    if not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("long summary must be a mapping")
    schema = summary.get("schema")
    if schema not in (LONG_SUMMARY_SCHEMA, "mcln-complete-long-summary-v1"):
        raise ValueError("long summary schema is invalid")
    latest_epoch = summary.get("latest_epoch")
    if (
        not isinstance(latest_epoch, int)
        or isinstance(latest_epoch, bool)
        or latest_epoch < 55
        or latest_epoch > FINAL_EPOCH
    ):
        raise ValueError("long summary latest epoch is invalid")
    candidates = summary.get("retained")
    if not isinstance(candidates, list):
        raise ValueError("long summary retained candidates are invalid")
    result = dict(summary)
    result["retained"] = [
        _validated_candidate(candidate) for candidate in candidates
    ]
    if schema == LONG_SUMMARY_SCHEMA:
        if result.get("phase") not in ("checkpointed", "epoch_complete"):
            raise ValueError("long summary phase is invalid")
        completed_epochs = result.get("completed_validation_epochs")
        if (
            not isinstance(completed_epochs, list)
            or completed_epochs != sorted(set(completed_epochs))
            or any(epoch not in VALIDATION_EPOCHS for epoch in completed_epochs)
        ):
            raise ValueError(
                "long summary completed validation epochs are invalid"
            )
    else:
        result["phase"] = "epoch_complete"
        result["completed_validation_epochs"] = []
    return result


def _summary_payload(
        latest, epoch, phase, retained, completed_validation_epochs):
    return {
        "schema": LONG_SUMMARY_SCHEMA,
        "latest_epoch": epoch,
        "latest": str(latest),
        "phase": phase,
        "retained": retained,
        "validation_epochs": list(VALIDATION_EPOCHS),
        "completed_validation_epochs": sorted(completed_validation_epochs),
        "completed": (
            epoch == FINAL_EPOCH
            and phase == "epoch_complete"
            and sorted(completed_validation_epochs) == list(VALIDATION_EPOCHS)
        ),
    }


def _candidate_pool(retained, receipts):
    candidates = list(retained)
    retained_identities = {
        _checkpoint_identity(candidate["path"]) for candidate in retained
    }
    for receipt in receipts.values():
        candidate = _candidate_from_validation_receipt(receipt)
        if candidate is None:
            continue
        identity = _checkpoint_identity(candidate["path"])
        if identity not in retained_identities:
            candidates.append(candidate)
            retained_identities.add(identity)
    return candidates


def _reconstruct_legacy_receipts(output_root, summary, receipts):
    if summary is None or summary.get("schema") == LONG_SUMMARY_SCHEMA:
        return receipts
    for candidate in summary["retained"]:
        epoch = candidate["epoch"]
        if epoch in receipts:
            continue
        receipts[epoch] = _write_validation_receipt(
            output_root,
            epoch,
            candidate["metrics"],
            candidate["path"],
        )
    return receipts


def _hardlink_latest(latest, candidate_path):
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        candidate_path.unlink()
    try:
        os.link(str(latest), str(candidate_path))
    except OSError:
        shutil.copy2(str(latest), str(candidate_path))
    return candidate_path


def _evaluate_validation_epoch(
        epoch, runtime, shared_args, output_root, receipts):
    metrics = runtime["trainer"].evaluate_one_epoch(
        epoch,
        runtime["official_loader"],
        runtime["model"],
        runtime["criterion"],
        runtime["set_criterion"],
        shared_args,
    )
    metrics = _validated_official_metrics(metrics)
    candidate_path = Path(output_root) / "pareto" / (
        "epoch_{:03d}.pth".format(epoch)
    )
    _hardlink_latest(runtime["latest"], candidate_path)
    receipts[epoch] = _write_validation_receipt(
        output_root, epoch, metrics, candidate_path
    )
    return receipts


def _same_retained_selection(first, second):
    def normalized(items):
        return sorted(
            (
                item["epoch"],
                _normalized_path(item["path"]),
                item.get("retention_role"),
            )
            for item in items
        )
    return normalized(first) == normalized(second)


def _reconcile_loaded_checkpoint(
        runtime, shared_args, output_root, summary_path, protected_paths):
    latest = runtime["latest"]
    if not latest.is_file():
        if shared_args.start_epoch != 55:
            raise ValueError("fresh long run resolved an invalid start epoch")
        if summary_path.is_file():
            raise ValueError("long summary exists without latest checkpoint")
        return [], {}

    loaded_epoch = shared_args.start_epoch - 1
    if loaded_epoch < 55 or loaded_epoch > FINAL_EPOCH:
        raise ValueError("loaded latest checkpoint epoch is invalid")
    summary = _load_long_summary(summary_path)
    if summary is not None:
        if summary["latest_epoch"] > loaded_epoch:
            raise ValueError("long summary is ahead of latest checkpoint")
        summary_latest = summary.get("latest")
        if (
            isinstance(summary_latest, str)
            and _normalized_path(summary_latest) != _normalized_path(latest)
        ):
            raise ValueError("long summary latest checkpoint path differs")

    receipts = _load_validation_receipts(output_root)
    receipts = _reconstruct_legacy_receipts(
        output_root, summary, receipts
    )
    if any(epoch > loaded_epoch for epoch in receipts):
        raise ValueError("validation receipt is ahead of latest checkpoint")
    if summary is not None:
        for epoch in summary["completed_validation_epochs"]:
            if epoch not in receipts:
                raise ValueError(
                    "long summary references a missing validation receipt"
                )

    expected_completed = [
        epoch for epoch in VALIDATION_EPOCHS if epoch <= loaded_epoch
    ]
    missing = [epoch for epoch in expected_completed if epoch not in receipts]
    unreconstructable = [epoch for epoch in missing if epoch != loaded_epoch]
    if unreconstructable:
        raise ValueError(
            "validation epoch {} cannot be reconstructed from latest".format(
                unreconstructable[0]
            )
        )

    retained = [] if summary is None else summary["retained"]
    summary_is_complete_for_loaded = (
        summary is not None
        and summary.get("schema") == LONG_SUMMARY_SCHEMA
        and summary["latest_epoch"] == loaded_epoch
        and summary["phase"] == "epoch_complete"
        and not missing
    )
    if summary_is_complete_for_loaded:
        selected, cleanup_paths = plan_pareto_checkpoints(
            _candidate_pool(retained, receipts),
            max_keep=3,
            protected_paths=protected_paths,
        )
        if not _same_retained_selection(selected, retained):
            raise ValueError("published retained checkpoint selection differs")
        cleanup_checkpoint_paths(
            cleanup_paths, protected_paths=protected_paths
        )
        return retained, receipts

    if loaded_epoch in VALIDATION_EPOCHS:
        if loaded_epoch not in receipts:
            receipts = _evaluate_validation_epoch(
                loaded_epoch,
                runtime,
                shared_args,
                output_root,
                receipts,
            )
        elif _candidate_from_validation_receipt(receipts[loaded_epoch]) is None:
            raise ValueError(
                "checkpointed validation candidate cannot be reconstructed"
            )

    selected, cleanup_paths = plan_pareto_checkpoints(
        _candidate_pool(retained, receipts),
        max_keep=3,
        protected_paths=protected_paths,
    )
    payload = _summary_payload(
        latest,
        loaded_epoch,
        "epoch_complete",
        selected,
        receipts.keys(),
    )
    publish_long_summary(
        summary_path,
        payload,
        cleanup_paths=cleanup_paths,
        protected_paths=protected_paths,
    )
    return selected, receipts


def _validated_completed_long_state(output_root):
    output_root = Path(output_root)
    summary_path = output_root / "long_summary.json"
    summary = _load_long_summary(summary_path)
    if summary is None:
        raise ValueError("completed long summary is missing")
    if (
        summary.get("schema") != LONG_SUMMARY_SCHEMA
        or summary["latest_epoch"] != FINAL_EPOCH
        or summary["phase"] != "epoch_complete"
        or summary.get("completed") is not True
        or summary["completed_validation_epochs"] != list(VALIDATION_EPOCHS)
    ):
        raise ValueError("long summary is not durably complete")
    latest = output_root / "latest.pth"
    if not latest.is_file():
        raise ValueError("completed latest checkpoint is missing")
    receipts = _load_validation_receipts(output_root)
    if sorted(receipts) != list(VALIDATION_EPOCHS):
        raise ValueError("exact completed validation receipts are missing")
    for candidate in summary["retained"]:
        if not Path(candidate["path"]).is_file():
            raise ValueError("completed retained checkpoint is missing")
    return summary


def _write_sidecar_handoff(custom_args, best, retained):
    artifacts = []
    for candidate in retained:
        artifacts.append({
            "path": candidate["path"],
            "sha256": file_sha256(candidate["path"]),
            "epoch": candidate["epoch"],
            "metrics": candidate["metrics"],
            "retention_role": candidate["retention_role"],
            "release_gate": release_gate_status(candidate["metrics"]),
        })
    handoff = {
        "schema": "mcln-sidecar-handoff-v1",
        "source_snapshot_digest": best.get("source_snapshot_digest"),
        "backbones": artifacts,
        "entry_points": {
            "candidate_cache": "scripts/cache_scanrefer_rec_candidates.py",
            "parent_reranker": "scripts/train_rec_reranker.py",
            "geometry_cache": "scripts/rec_geometry_cache.py",
            "geometry_reranker": "scripts/train_rec_geometry_reranker.py",
            "joint_cache": "scripts/cache_scanrefer_joint_box_mask.py",
            "joint_selector": "scripts/train_scanrefer_joint_box_mask.py",
        },
        "inference_uses_ground_truth": False,
    }
    atomic_write_json(
        Path(custom_args.output_root) / "sidecar_handoff.json", handoff
    )
    return handoff


def _validated_dispatch_paths(custom_args):
    output_root = Path(custom_args.output_root)
    ack_path = Path(custom_args.startup_ack)
    completion_path = Path(custom_args.completion_receipt)
    if ack_path != output_root / "startup_ack.json":
        raise ValueError("startup ack path differs from long output contract")
    if completion_path != output_root / "completion.json":
        raise ValueError(
            "completion receipt path differs from long output contract"
        )
    if not _valid_dispatch_token(custom_args.dispatch_token):
        raise ValueError("long runner dispatch token is invalid")
    return output_root, ack_path, completion_path


def write_startup_ack(custom_args, binding):
    output_root, ack_path, _completion_path = _validated_dispatch_paths(
        custom_args
    )
    dispatch_path = output_root / "dispatch.json"
    if not dispatch_path.is_file():
        raise ValueError("long runner dispatch state is missing")
    dispatch = _validated_dispatch_state(
        _load_json(dispatch_path), binding
    )
    if dispatch["token"] != custom_args.dispatch_token:
        raise ValueError("long runner dispatch token differs")
    pid = os.getpid()
    start_time = _process_start_time(pid)
    if start_time is None:
        raise RuntimeError("long runner process identity is unavailable")
    ack = {
        "schema": LONG_STARTUP_ACK_SCHEMA,
        "token": custom_args.dispatch_token,
        "binding": binding,
        "pid": pid,
        "process_start_time": start_time,
    }
    atomic_write_json(ack_path, ack)
    return ack


def write_completion_receipt(custom_args, binding):
    output_root, ack_path, completion_path = _validated_dispatch_paths(
        custom_args
    )
    dispatch = _validated_dispatch_state(
        _load_json(output_root / "dispatch.json"), binding
    )
    if dispatch["token"] != custom_args.dispatch_token:
        raise ValueError("completion dispatch token differs")
    if not ack_path.is_file():
        raise ValueError("completion startup ack is missing")
    ack = _validated_startup_ack(_load_json(ack_path), binding)
    if ack["token"] != custom_args.dispatch_token:
        raise ValueError("completion startup ack token differs")
    pid = os.getpid()
    start_time = _process_start_time(pid)
    if ack["pid"] != pid or ack["process_start_time"] != start_time:
        raise ValueError("completion process identity differs from startup ack")
    summary_path = output_root / "long_summary.json"
    handoff_path = output_root / "sidecar_handoff.json"
    if not summary_path.is_file() or not handoff_path.is_file():
        raise ValueError("completion artifacts are missing")
    completion = {
        "schema": LONG_COMPLETION_SCHEMA,
        "token": custom_args.dispatch_token,
        "binding": binding,
        "pid": pid,
        "process_start_time": start_time,
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "handoff": str(handoff_path),
        "handoff_sha256": file_sha256(handoff_path),
    }
    atomic_write_json(completion_path, completion)
    return completion


def run_long_training(shared_args, custom_args, best):
    output_root = Path(custom_args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_size = Path(custom_args.base_checkpoint).stat().st_size
    resume = (output_root / "latest.pth").is_file()
    require_long_run_filesystem_capacity(
        output_root, checkpoint_size, resume=resume
    )
    runtime = _build_runtime(shared_args, custom_args)
    protected_paths = (
        custom_args.base_checkpoint,
        best["checkpoint"],
        runtime["latest"],
    )
    summary_path = output_root / "long_summary.json"
    retained, receipts = _reconcile_loaded_checkpoint(
        runtime,
        shared_args,
        output_root,
        summary_path,
        protected_paths,
    )

    for epoch in range(shared_args.start_epoch, FINAL_EPOCH + 1):
        require_long_run_filesystem_capacity(
            output_root, checkpoint_size, resume=True
        )
        runtime["train_loader"].sampler.set_epoch(epoch)
        runtime["trainer"].train_one_epoch(
            epoch,
            runtime["train_loader"],
            runtime["model"],
            runtime["criterion"],
            runtime["set_criterion"],
            runtime["optimizer"],
            runtime["scheduler"],
            shared_args,
        )
        _atomic_save_checkpoint(
            runtime["latest"],
            shared_args,
            epoch,
            runtime["model"],
            runtime["optimizer"],
            runtime["scheduler"],
            protected_paths=(custom_args.base_checkpoint, best["checkpoint"]),
        )

        publish_long_summary(
            summary_path,
            _summary_payload(
                runtime["latest"],
                epoch,
                "checkpointed",
                retained,
                receipts.keys(),
            ),
            protected_paths=protected_paths,
        )

        if epoch in VALIDATION_EPOCHS:
            receipts = _evaluate_validation_epoch(
                epoch,
                runtime,
                shared_args,
                output_root,
                receipts,
            )
        retained, cleanup_paths = plan_pareto_checkpoints(
                _candidate_pool(retained, receipts),
                max_keep=3,
                protected_paths=protected_paths,
            )
        publish_long_summary(
            summary_path,
            _summary_payload(
                runtime["latest"],
                epoch,
                "epoch_complete",
                retained,
                receipts.keys(),
            ),
            cleanup_paths=cleanup_paths,
            protected_paths=protected_paths,
        )

    completed = _validated_completed_long_state(output_root)
    return _write_sidecar_handoff(custom_args, best, completed["retained"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pp-checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=29780)
    parser.add_argument("--dispatch-token", required=True)
    parser.add_argument("--startup-ack", required=True)
    parser.add_argument("--completion-receipt", required=True)
    args, _unknown = parser.parse_known_args(argv)
    return args


def main():
    custom_args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(custom_args.gpu))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with open(custom_args.best_json, "r", encoding="utf-8") as handle:
        best = json.load(handle)
    dispatch_binding = long_dispatch_binding(best)
    write_startup_ack(custom_args, dispatch_binding)
    params = _validate_best_json(best)
    verify_base_checkpoint(
        custom_args.base_checkpoint, custom_args.expected_base_sha256
    )
    shared_args = parse_option()
    _apply_long_contract(shared_args, custom_args, params)
    _initialize_distributed(custom_args)
    seed_everything(0)
    handoff = run_long_training(shared_args, custom_args, best)
    write_completion_receipt(custom_args, dispatch_binding)
    if dist.get_rank() == 0:
        print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
