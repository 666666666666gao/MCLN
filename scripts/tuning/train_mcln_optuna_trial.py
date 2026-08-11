#!/usr/bin/env python
"""Run the epoch-54 calibration baseline or one strict two-epoch trial."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from main_utils import (
    TRAIN_LOSS_RECEIPT_SCHEMA,
    load_checkpoint,
    parse_option,
)
from scripts.tuning.mcln_optuna_contract import (
    EXPECTED_CALIBRATION_COUNT,
    seed_presets,
    validate_metrics_receipt,
)
from scripts.tuning.scanrefer_train_only import (
    AUTHORITATIVE_SCANREFER_SPLIT_METADATA,
    build_train_only_data,
)
from train_dist_mod import TrainTester


TRIAL_RECEIPT_SCHEMA = "mcln-optuna-trial-v1"
BASE_GLOBAL_EPOCH = 54
TRIAL_GLOBAL_EPOCHS = (55, 56)
DEFAULT_CONTINUATION_HORIZON = 46
SOURCE_PAIR = "default,default_rank_blend_contrastive010"
SELECTOR_TARGET = "precision_gain_default_sourcewise_focal_bce"

TRIAL_PARAMETER_NAMES = (
    "decoder_lr",
    "mask_head_lr_multiplier",
    "selector_lr",
    "mask_loss_scale",
    "consistency_loss_scale",
    "selector_loss_weight",
    "selector_min_iou_gap",
)
STUDY_BINDING_FIELDS = frozenset((
    "study_contract_digest",
    "source_snapshot_digest",
    "base_checkpoint_sha256",
    "pointnet_checkpoint_sha256",
    "repo_root",
    "data_root",
    "python_bin",
    "run_manifest_sha256",
    "environment_sha256",
    "inputs_sha256",
    "study_name",
    "storage_identity",
    "split_metadata",
    "split_metadata_sha256",
    "data_digests",
    "continuation_horizon",
))


def canonical_json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("{} must be a positive integer".format(label))
    return value


def cosine_horizon_steps(
        steps_per_epoch,
        continuation_horizon=DEFAULT_CONTINUATION_HORIZON):
    """Return the fixed full-continuation scheduler horizon in steps."""
    steps_per_epoch = _positive_integer(steps_per_epoch, "steps_per_epoch")
    continuation_horizon = _positive_integer(
        continuation_horizon, "continuation_horizon"
    )
    return steps_per_epoch * continuation_horizon


def cosine_factor(step, total_steps):
    """Return the shared short/long-run cosine multiplier."""
    total_steps = _positive_integer(total_steps, "total_steps")
    if (
        not isinstance(step, int)
        or isinstance(step, bool)
        or step < 0
        or step > total_steps
    ):
        raise ValueError("cosine step is outside the fixed horizon")
    return 0.5 * (
        1.0 + math.cos(math.pi * step / float(total_steps))
    )


def build_cosine_scheduler(
        optimizer, steps_per_epoch,
        continuation_horizon=DEFAULT_CONTINUATION_HORIZON):
    """Build the batch-stepped scheduler shared with the long continuation."""
    total_steps = cosine_horizon_steps(
        steps_per_epoch, continuation_horizon=continuation_horizon
    )
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_factor(step, total_steps),
    )


def atomic_write_json(path, payload):
    """Write JSON through a temporary sibling and atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _optimizer_groups(context):
    if not hasattr(context, "optimizer_group_receipt"):
        raise ValueError("trial context lacks optimizer group receipt")
    groups = context.optimizer_group_receipt()
    if not isinstance(groups, (list, tuple)):
        raise ValueError("optimizer group receipt must be a sequence")
    expected_names = ("decoder", "backbone", "mask_head", "selector")
    actual_names = tuple(
        group.get("name") if isinstance(group, dict) else None
        for group in groups
    )
    if actual_names != expected_names:
        raise ValueError("optimizer group receipt is not the strict layout")
    result = []
    all_parameter_names = []
    for group in groups:
        learning_rate = group.get("initial_lr")
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
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


def validate_loss_receipt(receipt):
    """Validate and normalize one finite training-epoch loss receipt."""
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "batch_count", "loss_means"
    }:
        raise ValueError("training loss receipt fields are invalid")
    if receipt["schema"] != TRAIN_LOSS_RECEIPT_SCHEMA:
        raise ValueError("training loss receipt schema is invalid")
    batch_count = receipt["batch_count"]
    if (
        not isinstance(batch_count, int)
        or isinstance(batch_count, bool)
        or batch_count <= 0
    ):
        raise ValueError("training loss receipt batch_count is invalid")
    means = receipt["loss_means"]
    if not isinstance(means, dict) or not means:
        raise ValueError("training loss means must be a nonempty mapping")
    keys = tuple(means.keys())
    if (
        any(not isinstance(key, str) or not key for key in keys)
        or keys != tuple(sorted(keys))
        or "total_loss" not in means
    ):
        raise ValueError("training loss mean keys are invalid")
    normalized = {}
    for key in keys:
        value = means[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("training loss mean {} is invalid".format(key))
        normalized[key] = float(value)
    return {
        "schema": TRAIN_LOSS_RECEIPT_SCHEMA,
        "batch_count": batch_count,
        "loss_means": normalized,
    }


def run_trial_core(context, mode):
    """Execute the strict baseline or two-epoch loop through a small context."""
    if mode not in ("baseline", "trial"):
        raise ValueError("mode must be baseline or trial")
    if mode == "baseline":
        metrics = validate_metrics_receipt(
            context.evaluate_epoch(BASE_GLOBAL_EPOCH)
        )
        return {
            "schema": TRIAL_RECEIPT_SCHEMA,
            "mode": "baseline",
            "selection_epoch": BASE_GLOBAL_EPOCH,
            "metrics": {"epoch_54": metrics},
            "checkpoint": None,
            "optimizer_groups": (),
        }

    metrics_by_epoch = {}
    losses_by_epoch = {}
    for epoch in TRIAL_GLOBAL_EPOCHS:
        epoch_key = "epoch_{}".format(epoch)
        losses_by_epoch[epoch_key] = validate_loss_receipt(
            context.train_epoch(epoch)
        )
        metrics_by_epoch[epoch_key] = (
            validate_metrics_receipt(context.evaluate_epoch(epoch))
        )
    checkpoint = context.publish_checkpoint(TRIAL_GLOBAL_EPOCHS[-1])
    if not isinstance(checkpoint, (str, os.PathLike)):
        raise ValueError("published checkpoint path is invalid")
    return {
        "schema": TRIAL_RECEIPT_SCHEMA,
        "mode": "trial",
        "selection_epoch": TRIAL_GLOBAL_EPOCHS[-1],
        "metrics": metrics_by_epoch,
        "losses": losses_by_epoch,
        "checkpoint": os.fspath(checkpoint),
        "optimizer_groups": _optimizer_groups(context),
    }


def _validated_gradient_groups(context):
    if not hasattr(context, "optimizer_gradient_receipt"):
        raise ValueError("smoke context lacks optimizer gradient receipt")
    gradients = context.optimizer_gradient_receipt()
    if not isinstance(gradients, (list, tuple)):
        raise ValueError("optimizer gradient receipt must be a sequence")
    expected_names = ("decoder", "backbone", "mask_head", "selector")
    actual_names = tuple(
        group.get("name") if isinstance(group, dict) else None
        for group in gradients
    )
    if actual_names != expected_names:
        raise ValueError("optimizer gradient receipt has an invalid layout")
    result = []
    for group in gradients:
        count = group.get("gradient_tensor_count")
        total_norm = group.get("total_norm")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or group.get("all_finite") is not True
            or not isinstance(total_norm, (int, float))
            or isinstance(total_norm, bool)
            or not math.isfinite(float(total_norm))
        ):
            raise ValueError("optimizer gradient group is not finite")
        result.append({
            "name": group["name"],
            "gradient_tensor_count": count,
            "all_finite": True,
            "total_norm": float(total_norm),
        })
    return tuple(result)


def run_smoke_core(context, expected_sample_count):
    """Run one fit batch and one calibration batch without a checkpoint."""
    if (
        not isinstance(expected_sample_count, int)
        or isinstance(expected_sample_count, bool)
        or expected_sample_count <= 0
    ):
        raise ValueError("smoke expected_sample_count must be positive")
    losses = validate_loss_receipt(
        context.train_epoch(TRIAL_GLOBAL_EPOCHS[0])
    )
    gradients = _validated_gradient_groups(context)
    metrics = validate_metrics_receipt(
        context.evaluate_epoch(TRIAL_GLOBAL_EPOCHS[0]),
        expected_sample_count=expected_sample_count,
    )
    return {
        "schema": TRIAL_RECEIPT_SCHEMA,
        "mode": "smoke",
        "selection_epoch": TRIAL_GLOBAL_EPOCHS[0],
        "metrics": {"epoch_55": metrics},
        "losses": {"epoch_55": losses},
        "checkpoint": None,
        "optimizer_groups": _optimizer_groups(context),
        "gradients": gradients,
    }


def _runner_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--mode", choices=("baseline", "trial", "smoke"), required=True
    )
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--checkpoint-output", default=None)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument(
        "--continuation-horizon", type=int,
        default=DEFAULT_CONTINUATION_HORIZON,
    )
    parser.add_argument("--study-binding-json", required=True)
    parser.add_argument(
        "--decoder-lr", "--decoder_lr", dest="decoder_lr", type=float
    )
    parser.add_argument(
        "--mask-head-lr-multiplier", "--mask_head_lr_multiplier",
        dest="mask_head_lr_multiplier", type=float,
    )
    parser.add_argument(
        "--selector-lr", "--selector_lr", "--source_choice_selector_lr",
        dest="selector_lr", type=float,
    )
    parser.add_argument(
        "--mask-loss-scale", "--mask_loss_scale",
        dest="mask_loss_scale", type=float,
    )
    parser.add_argument(
        "--consistency-loss-scale", "--consistency_loss_scale",
        dest="consistency_loss_scale", type=float,
    )
    parser.add_argument(
        "--selector-loss-weight", "--selector_loss_weight",
        "--source_choice_selector_loss_weight",
        dest="selector_loss_weight", type=float,
    )
    parser.add_argument(
        "--selector-min-iou-gap", "--selector_min_iou_gap",
        "--source_choice_selector_min_iou_gap",
        dest="selector_min_iou_gap", type=float,
    )
    return parser


def _valid_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def validate_study_binding(binding):
    if not isinstance(binding, dict) or set(binding) != STUDY_BINDING_FIELDS:
        raise ValueError("study binding fields are invalid")
    for field in (
            "study_contract_digest", "source_snapshot_digest",
            "base_checkpoint_sha256", "pointnet_checkpoint_sha256",
            "run_manifest_sha256", "environment_sha256", "inputs_sha256",
            "split_metadata_sha256"):
        if not _valid_sha256(binding[field]):
            raise ValueError("study binding {} is invalid".format(field))

    for field in (
            "repo_root", "data_root", "python_bin", "storage_identity"):
        value = binding[field]
        if not isinstance(value, str) or not value or not os.path.isabs(value):
            raise ValueError("study binding {} is invalid".format(field))
    if (
        not isinstance(binding["study_name"], str)
        or not binding["study_name"]
    ):
        raise ValueError("study binding study_name is invalid")

    split_metadata = binding["split_metadata"]
    expected_split = dict(AUTHORITATIVE_SCANREFER_SPLIT_METADATA)
    if split_metadata != expected_split:
        raise ValueError("study binding split metadata is invalid")
    if canonical_json_sha256(split_metadata) != binding[
            "split_metadata_sha256"]:
        raise ValueError("study binding split metadata digest is invalid")
    data_digests = binding["data_digests"]
    expected_data_digests = {
        "fit_scene_sha256": expected_split["fit_scene_sha256"],
        "calibration_scene_sha256": expected_split[
            "calibration_scene_sha256"
        ],
        "mapping_sha256": expected_split["mapping_sha256"],
    }
    if data_digests != expected_data_digests:
        raise ValueError("study binding data digests are invalid")
    if binding["continuation_horizon"] != DEFAULT_CONTINUATION_HORIZON:
        raise ValueError("study binding continuation horizon is invalid")
    return json.loads(json.dumps(binding, sort_keys=True))


def study_binding_from_args(args):
    try:
        binding = json.loads(args.study_binding_json)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("study binding JSON is invalid")
    return validate_study_binding(binding)


def _validate_resolved_params(params):
    if set(params) != set(TRIAL_PARAMETER_NAMES):
        raise ValueError("trial parameters are incomplete")
    finite = {}
    for name, value in params.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("{} is invalid".format(name))
        finite[name] = float(value)
    if not 5e-6 <= finite["decoder_lr"] <= 4e-5:
        raise ValueError("decoder_lr is outside the approved space")
    if finite["mask_head_lr_multiplier"] not in (1.0, 2.0, 4.0):
        raise ValueError("mask_head_lr_multiplier is outside the approved space")
    if not 2e-4 <= finite["selector_lr"] <= 2e-3:
        raise ValueError("selector_lr is outside the approved space")
    if not 0.5 <= finite["mask_loss_scale"] <= 4.0:
        raise ValueError("mask_loss_scale is outside the approved space")
    if not 0.1 <= finite["consistency_loss_scale"] <= 2.0:
        raise ValueError("consistency_loss_scale is outside the approved space")
    if not 0.1 <= finite["selector_loss_weight"] <= 1.0:
        raise ValueError("selector_loss_weight is outside the approved space")
    if finite["selector_min_iou_gap"] not in (0.02, 0.03, 0.05, 0.08):
        raise ValueError("selector_min_iou_gap is outside the approved space")
    return finite


def resolved_trial_params(args):
    if args.mode == "baseline":
        return None
    params = {
        name: getattr(args, name, None) for name in TRIAL_PARAMETER_NAMES
    }
    if any(value is None for value in params.values()):
        raise ValueError("trial mode requires all seven resolved parameters")
    return _validate_resolved_params(params)


def parse_runner_args(argv=None):
    args, _unknown = _runner_parser().parse_known_args(argv)
    if not _valid_sha256(args.expected_base_sha256):
        raise ValueError("expected base SHA-256 must be 64 hexadecimal characters")
    if args.split_seed != 0:
        raise ValueError("formal search requires split seed 0")
    if not math.isclose(args.calibration_fraction, 0.10, abs_tol=1e-15):
        raise ValueError("formal search requires calibration fraction 0.10")
    if args.continuation_horizon != DEFAULT_CONTINUATION_HORIZON:
        raise ValueError("formal search requires a 46-epoch continuation horizon")
    if args.mode == "trial" and not args.checkpoint_output:
        raise ValueError("trial mode requires --checkpoint-output")
    study_binding = study_binding_from_args(args)
    if study_binding["base_checkpoint_sha256"].lower() != (
            args.expected_base_sha256.lower()):
        raise ValueError("study binding base checkpoint digest differs")
    if study_binding["continuation_horizon"] != args.continuation_horizon:
        raise ValueError("study binding continuation horizon differs")
    resolved_trial_params(args)
    return args


def apply_runner_contract(shared_args, runner_args):
    """Overwrite mutable shared CLI values with the approved run contract."""
    params = resolved_trial_params(runner_args)
    if params is None:
        params = seed_presets()[0]

    shared_args.checkpoint_path = runner_args.base_checkpoint
    shared_args.reduce_lr = True
    shared_args.start_epoch = TRIAL_GLOBAL_EPOCHS[0]
    shared_args.max_epoch = TRIAL_GLOBAL_EPOCHS[-1]
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
    shared_args.expected_eval_sample_count = (
        18 if runner_args.mode == "smoke" else EXPECTED_CALIBRATION_COUNT
    )
    shared_args.train_only_seed = runner_args.split_seed
    shared_args.train_only_calibration_fraction = (
        runner_args.calibration_fraction
    )
    shared_args.continuation_horizon = runner_args.continuation_horizon
    return shared_args


class TrainOnlyTrainTester(TrainTester):
    """Use only scene-disjoint views of the ScanRefer training split."""

    @staticmethod
    def get_datasets(args):
        data = build_train_only_data(
            seed=args.train_only_seed,
            calibration_fraction=args.train_only_calibration_fraction,
            formal_run=True,
            split="train",
            dataset_dict={"scanrefer": 1, "scannet": 10},
            test_dataset=args.test_dataset,
            data_path=args.data_root,
            use_color=args.use_color,
            use_height=args.use_height,
            use_multiview=args.use_multiview,
            detect_intermediate=args.detect_intermediate,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            augment_det=args.augment_det,
            wo_obj_name=args.wo_obj_name,
            skip_missing_superpoints=args.skip_missing_superpoints,
        )
        return data["fit_dataset"], data["calibration_dataset"]


class LimitedLoader:
    """Expose a deterministic prefix of an existing data loader."""

    def __init__(self, loader, max_batches):
        if (
            not isinstance(max_batches, int)
            or isinstance(max_batches, bool)
            or max_batches <= 0
            or max_batches > len(loader)
        ):
            raise ValueError("max_batches is outside the loader range")
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset
        self.sampler = loader.sampler

    def __len__(self):
        return self.max_batches

    def __iter__(self):
        iterator = iter(self.loader)
        for _index in range(self.max_batches):
            try:
                yield next(iterator)
            except StopIteration:
                raise RuntimeError(
                    "underlying loader ended before max_batches"
                )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_checkpoint(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise ValueError("base checkpoint does not exist")
    actual = file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError("base checkpoint SHA-256 mismatch")
    checkpoint = torch.load(str(path), map_location="cpu")
    try:
        epoch = int(checkpoint["epoch"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("base checkpoint epoch is invalid")
    finally:
        del checkpoint
    if epoch != BASE_GLOBAL_EPOCH:
        raise ValueError("base checkpoint must be epoch 54")
    return actual


def seed_everything(seed):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _RuntimeContext:
    def __init__(
            self, args, trainer, train_loader, calibration_loader, model,
            criterion, set_criterion, optimizer, scheduler, checkpoint_output):
        self.args = args
        self.trainer = trainer
        self.train_loader = train_loader
        self.calibration_loader = calibration_loader
        self.model = model
        self.criterion = criterion
        self.set_criterion = set_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_output = checkpoint_output
        self.steps_per_epoch = len(train_loader)
        self._initial_groups = tuple({
            "name": group.get("name"),
            "initial_lr": float(group["lr"]),
            "parameter_names": tuple(group.get("parameter_names", ())),
        } for group in optimizer.param_groups)

    def train_epoch(self, epoch):
        self.train_loader.sampler.set_epoch(epoch)
        return self.trainer.train_one_epoch(
            epoch, self.train_loader, self.model,
            self.criterion, self.set_criterion,
            self.optimizer, self.scheduler, self.args,
        )

    def evaluate_epoch(self, epoch):
        metrics = self.trainer.evaluate_one_epoch(
            epoch, self.calibration_loader, self.model,
            self.criterion, self.set_criterion, self.args,
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        return metrics

    def publish_checkpoint(self, epoch):
        if self.checkpoint_output is None:
            raise ValueError("trial checkpoint output is missing")
        output = Path(self.checkpoint_output)
        if dist.get_rank() == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(".{}-{}.tmp".format(
                output.name, os.getpid()
            ))
            state = {
                "config": self.args,
                "save_path": str(output),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
            }
            try:
                torch.save(state, str(temporary))
                os.replace(str(temporary), str(output))
            finally:
                if temporary.exists():
                    temporary.unlink()
        if dist.is_initialized():
            dist.barrier()
        return str(output)

    def optimizer_group_receipt(self):
        return self._initial_groups

    def optimizer_gradient_receipt(self):
        result = []
        for group in self.optimizer.param_groups:
            gradient_count = 0
            squared_norm = 0.0
            all_finite = True
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                gradient_count += 1
                finite = bool(torch.isfinite(gradient).all().item())
                all_finite = all_finite and finite
                if finite:
                    squared_norm += float(
                        gradient.detach().float().norm().cpu().item()
                    ) ** 2
            result.append({
                "name": group.get("name"),
                "gradient_tensor_count": gradient_count,
                "all_finite": all_finite,
                "total_norm": math.sqrt(squared_norm),
            })
        return tuple(result)


def build_runtime_context(args, runner_args):
    trainer = TrainOnlyTrainTester(args)
    train_loader, calibration_loader = trainer.get_loaders(args)
    if len(train_loader) <= 0:
        raise ValueError("fit loader has no batches")

    model = trainer.get_model(args)
    criterion, set_criterion = trainer.get_criterion(args)
    optimizer = trainer.get_optimizer(args, model)
    scheduler = build_cosine_scheduler(
        optimizer,
        len(train_loader),
        continuation_horizon=runner_args.continuation_horizon,
    )

    if runner_args.mode == "smoke":
        train_loader = LimitedLoader(train_loader, max_batches=1)
        calibration_loader = LimitedLoader(
            calibration_loader, max_batches=1
        )

    if not torch.cuda.is_available():
        raise RuntimeError("formal trial runner requires CUDA")
    model = model.cuda()
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    load_checkpoint(args, model, optimizer, scheduler)
    if args.start_epoch != TRIAL_GLOBAL_EPOCHS[0]:
        raise ValueError("epoch-54 load did not resolve start epoch 55")
    return _RuntimeContext(
        args=args,
        trainer=trainer,
        train_loader=train_loader,
        calibration_loader=calibration_loader,
        model=model,
        criterion=criterion,
        set_criterion=set_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_output=runner_args.checkpoint_output,
    )


def initialize_distributed(local_rank):
    if not torch.cuda.is_available():
        raise RuntimeError("formal trial runner requires CUDA")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=5400),
        )
    if dist.get_world_size() != 1:
        raise ValueError("formal Optuna trials require exactly one GPU process")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    shared_args = parse_option()
    runner_args = parse_runner_args()
    apply_runner_contract(shared_args, runner_args)
    initialize_distributed(shared_args.local_rank)
    seed_everything(shared_args.rng_seed)
    base_sha256 = verify_base_checkpoint(
        runner_args.base_checkpoint, runner_args.expected_base_sha256
    )
    study_binding = study_binding_from_args(runner_args)
    if study_binding["base_checkpoint_sha256"].lower() != (
            base_sha256.lower()):
        raise ValueError("verified base checkpoint differs from study binding")
    pointnet_sha256 = file_sha256(shared_args.pp_checkpoint)
    if pointnet_sha256.lower() != study_binding[
            "pointnet_checkpoint_sha256"].lower():
        raise ValueError("PointNet checkpoint differs from study binding")
    context = build_runtime_context(shared_args, runner_args)
    if runner_args.mode == "smoke":
        result = run_smoke_core(
            context,
            expected_sample_count=shared_args.expected_eval_sample_count,
        )
    else:
        result = run_trial_core(context, runner_args.mode)
    result.update({
        "study_binding": study_binding,
        "base_checkpoint": os.path.abspath(runner_args.base_checkpoint),
        "base_sha256": base_sha256,
        "split_seed": runner_args.split_seed,
        "calibration_fraction": runner_args.calibration_fraction,
        "continuation_horizon": runner_args.continuation_horizon,
        "trial_params": resolved_trial_params(runner_args),
    })
    if result.get("checkpoint") is not None:
        result["checkpoint_sha256"] = file_sha256(result["checkpoint"])
    if dist.get_rank() == 0:
        atomic_write_json(runner_args.receipt_path, result)
    dist.barrier()


if __name__ == "__main__":
    main()
