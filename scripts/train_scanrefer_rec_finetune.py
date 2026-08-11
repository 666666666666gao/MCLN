#!/usr/bin/env python
"""Initialize the train-only ScanRefer REC fine-tuning contract."""

import argparse
from contextlib import redirect_stdout
import copy
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
from fractions import Fraction
import gc
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Dataset

from models import (
    HungarianMatcher,
    SetCriterion,
    compute_hungarian_loss,
    rec_finetune,
)
from models.rec_reranker import compute_query_ious
from scripts.cache_scanrefer_rec_candidates import (
    _load_frozen_model,
    _move_batch_to_device,
    _normalized_data_root,
    _prepare_model_config,
    _seed_worker,
    strip_module_prefix,
)
from scripts.train_rec_geometry_reranker import (
    load_geometry_reranker_artifact,
    load_parent_reranker_snapshot,
    validate_geometry_artifact,
)


EXPECTED_BACKBONE_SHA256 = (
    rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
)
EXPECTED_PARENT_SHA256 = (
    rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256
)
EXPECTED_GEOMETRY_SHA256 = (
    rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256
)
EXPECTED_BACKBONE_EPOCH = 71
PRODUCTION_BATCH_SIZE = rec_finetune.PRODUCTION_BATCH_SIZE
PRODUCTION_NUM_WORKERS = 2
PRODUCTION_SEED = 0
PRODUCTION_MAX_STEPS = rec_finetune.PRODUCTION_MAX_STEPS
PRODUCTION_CALIBRATION_INTERVAL = (
    rec_finetune.PRODUCTION_CALIBRATION_INTERVAL
)
PRODUCTION_DEVICE = "cuda:0"
PRODUCTION_PROGRESS_INTERVAL = 25
RUNTIME_ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
)
PUBLICATION_ORDER = ("backbone", "parent", "geometry", "selection")
_PUBLICATION_FILENAMES = {
    "backbone": "backbone.pth",
    "parent": "parent.pth",
    "geometry": "geometry.pth",
    "selection": "selection.json",
}
REC_TARGET_ONLY_FIELDS = frozenset((
    "center_label",
    "size_gts",
    "sem_cls_label",
    "box_label_mask",
    "gt_masks",
    "point_instance_label",
    "all_bboxes",
    "candidate_ious",
    "geometry_ious",
    "threshold_labels",
))

_LEGACY_ROOT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16"
)
PROTECTED_LEGACY_PATHS = (
    _LEGACY_ROOT / "train",
    _LEGACY_ROOT / "val",
    _LEGACY_ROOT / "geometry_train",
    _LEGACY_ROOT / "geometry_val",
    _LEGACY_ROOT / "artifacts",
    _LEGACY_ROOT / "geometry_artifacts",
    _LEGACY_ROOT / "geometry_official_val",
)


@dataclass(frozen=True)
class RecFinetuneRuntimePaths:
    """Normalized inputs and a still-nonexistent final output path."""

    data_root: Path
    backbone_checkpoint: Path
    parent_reranker: Path
    geometry_reranker: Path
    output_dir: Path


class IndexedDatasetView(Dataset):
    """An annotation-index view carrying the source index in every sample."""

    def __init__(self, dataset, indices, augment, augment_det):
        if not isinstance(dataset, Dataset) and not (
                hasattr(dataset, "__len__")
                and hasattr(dataset, "__getitem__")):
            raise ValueError("indexed view requires a dataset")
        values = tuple(indices)
        if (any(not isinstance(index, int) or isinstance(index, bool)
                for index in values)
                or any(index < 0 or index >= len(dataset) for index in values)
                or len(set(values)) != len(values)):
            raise ValueError("indexed view contains invalid dataset indices")
        self.dataset = copy.copy(dataset)
        self.dataset.augment = bool(augment)
        self.dataset.augment_det = bool(augment_det)
        self.indices = values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_index = self.indices[index]
        item = self.dataset[dataset_index]
        if not isinstance(item, dict):
            raise ValueError("Joint3DDataset samples must be mappings")
        result = dict(item)
        result["dataset_index"] = int(dataset_index)
        return result


def _positive_smoke_steps(value):
    try:
        steps = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("smoke steps must be an integer")
    if not 1 <= steps <= PRODUCTION_MAX_STEPS:
        raise argparse.ArgumentTypeError(
            "smoke steps must lie in [1, {}]".format(PRODUCTION_MAX_STEPS)
        )
    return steps


def parse_args(argv=None):
    """Parse the path-only production runner command line."""
    parser = argparse.ArgumentParser(
        description="Initialize train-only ScanRefer REC fine-tuning."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--parent-reranker", required=True)
    parser.add_argument("--geometry-reranker", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=PRODUCTION_DEVICE)
    parser.add_argument("--smoke-steps", type=_positive_smoke_steps)
    args = parser.parse_args(argv)
    if args.device != PRODUCTION_DEVICE:
        parser.error("--device must be cuda:0")
    return args


def _logical_absolute(path, label):
    if (not isinstance(path, (str, os.PathLike))
            or isinstance(path, bytes)):
        raise ValueError("{} must be path-like".format(label))
    try:
        return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("{} is invalid: {}".format(label, error))


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


def _require_input_file(path, label):
    logical = _logical_absolute(path, label)
    if logical.is_symlink():
        raise ValueError("{} must not be a symlink".format(label))
    if not logical.is_file():
        raise ValueError("{} does not exist as a regular file".format(label))
    return logical


def validate_runtime_paths(args):
    """Validate every input and output collision without creating output."""
    if args is None:
        raise ValueError("runner arguments are required")
    data_root = _logical_absolute(
        getattr(args, "data_root", None), "data root"
    )
    if data_root.is_symlink() or not data_root.is_dir():
        raise ValueError("data root must be an existing non-symlink directory")
    data_root = data_root.resolve()
    backbone = _require_input_file(
        getattr(args, "backbone_checkpoint", None), "backbone checkpoint"
    )
    parent = _require_input_file(
        getattr(args, "parent_reranker", None), "parent reranker"
    )
    geometry = _require_input_file(
        getattr(args, "geometry_reranker", None), "geometry reranker"
    )
    input_files = (backbone, parent, geometry)
    for index, first in enumerate(input_files):
        for second in input_files[index + 1:]:
            if _same_existing_file(first, second):
                raise ValueError("runner input files must be distinct")

    output_logical = _logical_absolute(
        getattr(args, "output_dir", None), "output directory"
    )
    if output_logical.exists() or output_logical.is_symlink():
        raise FileExistsError(
            "final output directory must not exist: {}".format(output_logical)
        )
    output = _comparison_path(output_logical)
    input_directories = tuple(path.parent for path in input_files)
    protected_paths = (
        input_files + input_directories + tuple(PROTECTED_LEGACY_PATHS)
    )
    for protected in protected_paths:
        if _paths_overlap(output, protected):
            raise ValueError(
                "output directory collides with protected input: {}".format(
                    protected
                )
            )
    return RecFinetuneRuntimePaths(
        data_root=data_root,
        backbone_checkpoint=backbone,
        parent_reranker=parent,
        geometry_reranker=geometry,
        output_dir=output,
    )


def checkpoint_sha256(path):
    """Hash the same stable checkpoint bytes used for deserialization."""
    _resolved, _snapshot, digest = rec_finetune._stable_artifact_snapshot(
        path, "REC fine-tune backbone checkpoint"
    )
    return digest


def _load_backbone_checkpoint(path):
    resolved, snapshot, digest = rec_finetune._stable_artifact_snapshot(
        path, "REC fine-tune backbone checkpoint"
    )
    if digest != EXPECTED_BACKBONE_SHA256:
        raise ValueError("backbone checkpoint SHA-256 is not authoritative")
    try:
        checkpoint = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError(
            "could not deserialize backbone checkpoint: {}".format(error)
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("backbone checkpoint must contain a mapping")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch != EXPECTED_BACKBONE_EPOCH:
        raise ValueError("backbone checkpoint must be exact epoch 71")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("backbone checkpoint has no model state dict")
    return resolved, checkpoint, digest


def _prepare_training_config(checkpoint, data_root):
    config = _prepare_model_config(
        checkpoint, _normalized_data_root(data_root)
    )
    config.eval = False
    config.dataset = ["scanrefer"]
    config.test_dataset = "scanrefer"
    config.joint_det = False
    config.butd = True
    config.butd_gt = False
    config.butd_cls = False
    config.augment = True
    config.augment_det = True
    config.debug = False
    config.source_choice_selector_loss_weight = 0.0
    config.batch_size = PRODUCTION_BATCH_SIZE
    config.num_workers = PRODUCTION_NUM_WORKERS
    config.seed = PRODUCTION_SEED
    config.max_steps = PRODUCTION_MAX_STEPS
    config.calibration_interval = PRODUCTION_CALIBRATION_INTERVAL
    return config


def _default_model_factory(checkpoint, config, device):
    return _load_frozen_model(checkpoint, config, device)


def _load_testable_model(checkpoint, config, device, model_factory):
    try:
        if model_factory is None:
            return _default_model_factory(checkpoint, config, device)
        model = model_factory(config)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("model factory must return a torch module")
        state_dict = checkpoint.get("model")
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "backbone strict model state load failed: {}".format(error)
        )
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def _assert_model_matches_artifact(model, artifact, label):
    expected = artifact.get("model_state_dict")
    actual = model.state_dict()
    if (not isinstance(expected, dict)
            or set(actual) != set(expected)
            or any(not isinstance(value, torch.Tensor)
                   or not torch.equal(
                       actual[name].detach().cpu(), value.detach().cpu()
                   )
                   for name, value in expected.items())):
        raise ValueError("{} live state differs from its artifact".format(label))


def _config_model_inputs(config):
    return {
        key: bool(getattr(config, key))
        for key in (
            "use_color", "use_height", "use_multiview",
            "butd", "butd_gt", "butd_cls",
        )
    }


def _config_backbone(config):
    return {
        "model": str(config.model),
        "num_target": int(config.num_target),
        "num_decoder_layers": int(config.num_decoder_layers),
        "self_position_embedding": str(config.self_position_embedding),
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
    }


def _validate_initial_lineage(parent, parent_artifact, geometry,
                              geometry_artifact, checkpoint_sha256_value,
                              checkpoint_epoch, geometry_validator=None):
    if getattr(parent, "_artifact_sha256", None) != EXPECTED_PARENT_SHA256:
        raise ValueError("parent reranker SHA-256 is not authoritative")
    if getattr(geometry, "_artifact_sha256", None) != EXPECTED_GEOMETRY_SHA256:
        raise ValueError("geometry reranker SHA-256 is not authoritative")
    if parent_artifact.get("checkpoint_sha256") != checkpoint_sha256_value:
        raise ValueError("parent reranker backbone lineage is invalid")
    if (geometry_artifact.get("checkpoint_sha256")
            != checkpoint_sha256_value
            or geometry_artifact.get("checkpoint_epoch") != checkpoint_epoch):
        raise ValueError("geometry reranker backbone lineage is invalid")
    if (geometry_artifact.get("parent_artifact_sha256")
            != EXPECTED_PARENT_SHA256):
        raise ValueError("geometry reranker parent lineage is invalid")
    _assert_model_matches_artifact(parent, parent_artifact, "parent reranker")
    _assert_model_matches_artifact(
        geometry, geometry_artifact, "geometry reranker"
    )
    validator = geometry_validator or validate_geometry_artifact
    validator(
        geometry_artifact, parent=(parent, parent_artifact)
    )


def _validate_config_binding(config, parent_artifact, geometry_artifact):
    expected_inputs = _config_model_inputs(config)
    expected_backbone = _config_backbone(config)
    for label, artifact in (
            ("parent", parent_artifact),
            ("geometry", geometry_artifact)):
        if ("model_inputs" in artifact
                and artifact["model_inputs"] != expected_inputs):
            raise ValueError(
                "{} reranker model inputs differ from checkpoint".format(label)
            )
        if ("backbone_config" in artifact
                and artifact["backbone_config"] != expected_backbone):
            raise ValueError(
                "{} reranker backbone config differs from checkpoint".format(
                    label
                )
            )


def load_rec_finetune_initial_state(
        backbone_checkpoint, parent_reranker, geometry_reranker, data_root,
        *, device="cpu", model_factory=None, parent_loader=None,
        geometry_loader=None, geometry_validator=None):
    """Strict-load weights and build the fresh three-group optimizer."""
    device = torch.device(device)
    resolved_checkpoint, checkpoint, fingerprint = _load_backbone_checkpoint(
        backbone_checkpoint
    )
    config = _prepare_training_config(checkpoint, data_root)
    mcln = _load_testable_model(
        checkpoint, config, device, model_factory
    )

    parent_path = _require_input_file(parent_reranker, "parent reranker")
    geometry_path = _require_input_file(
        geometry_reranker, "geometry reranker"
    )
    parent_loader = parent_loader or load_parent_reranker_snapshot
    geometry_loader = geometry_loader or load_geometry_reranker_artifact
    parent, parent_artifact = parent_loader(
        parent_path, device=device
    )
    geometry, geometry_artifact = geometry_loader(
        geometry_path,
        device=device,
        parent_artifact_path=parent_path,
    )
    _validate_initial_lineage(
        parent,
        parent_artifact,
        geometry,
        geometry_artifact,
        fingerprint,
        checkpoint["epoch"],
        geometry_validator=geometry_validator,
    )
    _validate_config_binding(config, parent_artifact, geometry_artifact)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = rec_finetune.build_rec_finetune_optimizer(groups)
    if optimizer.state:
        raise AssertionError("fresh REC fine-tuning optimizer has state")
    _assert_model_matches_artifact(parent, parent_artifact, "parent reranker")
    _assert_model_matches_artifact(
        geometry, geometry_artifact, "geometry reranker"
    )
    return {
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_sha256": fingerprint,
        "checkpoint_epoch": checkpoint["epoch"],
        "config": config,
        "mcln": mcln,
        "parent": parent,
        "parent_artifact": parent_artifact,
        "geometry": geometry,
        "geometry_artifact": geometry_artifact,
        "groups": groups,
        "optimizer": optimizer,
    }


def _default_dataset_factory(**kwargs):
    from src.joint_det_dataset import Joint3DDataset

    return Joint3DDataset(**kwargs)


def _new_generator():
    generator = torch.Generator()
    generator.manual_seed(PRODUCTION_SEED)
    return generator


def _build_contract_loader(loader_factory, dataset, shuffle, device):
    return loader_factory(
        dataset,
        batch_size=PRODUCTION_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=PRODUCTION_NUM_WORKERS,
        pin_memory=(torch.device(device).type == "cuda"),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=_new_generator(),
    )


def build_train_only_data(
        config, device, *, dataset_factory=None, loader_factory=None,
        expected_split_metadata=None):
    """Construct the sole train dataset and its fit/calibration views."""
    if bool(getattr(config, "eval", True)):
        raise ValueError("fine-tuning config must use training semantics")
    if (getattr(config, "dataset", ["scanrefer"]) != ["scanrefer"]
            or bool(getattr(config, "joint_det", False))
            or not bool(getattr(config, "butd", False))
            or bool(getattr(config, "butd_gt", True))
            or bool(getattr(config, "butd_cls", True))):
        raise ValueError("fine-tuning config must be ScanRefer-only and no-GT")
    factory = dataset_factory or _default_dataset_factory
    loader_factory = loader_factory or DataLoader
    with redirect_stdout(io.StringIO()):
        dataset = factory(
            dataset_dict={"scanrefer": 1},
            test_dataset="scanrefer",
            split="train",
            use_color=bool(config.use_color),
            use_height=bool(config.use_height),
            overfit=False,
            data_path=config.data_root,
            detect_intermediate=bool(config.detect_intermediate),
            use_multiview=bool(config.use_multiview),
            butd=True,
            butd_gt=False,
            butd_cls=False,
            augment_det=True,
            wo_obj_name=config.wo_obj_name,
            skip_missing_superpoints=bool(config.skip_missing_superpoints),
        )
    if (getattr(dataset, "augment", None) is not True
            or getattr(dataset, "augment_det", None) is not True):
        raise ValueError("fit dataset must retain both augmentation modes")
    if getattr(dataset, "joint_det", False):
        raise ValueError("fit dataset must not include joint detection data")
    annos = getattr(dataset, "annos", None)
    if not isinstance(annos, (list, tuple)) or not annos:
        raise ValueError("train dataset annotations are unavailable")
    scan_ids = []
    for annotation in annos:
        scan_id = annotation.get("scan_id") if isinstance(annotation, dict) else None
        if not isinstance(scan_id, str) or not scan_id.strip():
            raise ValueError("train annotation scan_id is invalid")
        scan_ids.append(scan_id)
    split = rec_finetune.build_rec_finetune_scene_split(scan_ids)
    authoritative = (
        rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
        if expected_split_metadata is None else expected_split_metadata
    )
    if split["metadata"] != authoritative:
        raise ValueError("train scene split is not authoritative")

    fit_view = IndexedDatasetView(
        dataset, split["fit_indices"], augment=True, augment_det=True
    )
    calibration_view = IndexedDatasetView(
        dataset,
        split["calibration_indices"],
        augment=False,
        augment_det=False,
    )
    fit_loader = _build_contract_loader(
        loader_factory, fit_view, shuffle=True, device=device
    )
    calibration_loader = _build_contract_loader(
        loader_factory, calibration_view, shuffle=False, device=device
    )
    if expected_split_metadata is None:
        if (rec_finetune.natural_batch_count(
                len(fit_view), PRODUCTION_BATCH_SIZE)
                != PRODUCTION_MAX_STEPS
                or len(fit_loader) != PRODUCTION_MAX_STEPS):
            raise AssertionError("fit loader must contain exactly 1,836 batches")
    return {
        "dataset": dataset,
        "split": split,
        "fit_view": fit_view,
        "calibration_view": calibration_view,
        "fit_loader": fit_loader,
        "calibration_loader": calibration_loader,
    }


def build_rec_finetune_train_data_contract(initialized):
    """Bind the actual sole train dataset, views, split, and loaders."""
    if not isinstance(initialized, dict):
        raise ValueError("initialized train-data state must be a mapping")
    state = initialized.get("initial_state")
    data = initialized.get("data")
    if not isinstance(state, dict) or not isinstance(data, dict):
        raise ValueError("initialized train-data state is incomplete")
    expected_data_keys = {
        "dataset", "split", "fit_view", "calibration_view",
        "fit_loader", "calibration_loader",
    }
    if set(data) != expected_data_keys:
        prohibited = {
            "validation_loader", "test_loader", "val_dataset",
            "test_dataset",
        }
        if set(data).intersection(prohibited):
            raise ValueError(
                "train-only data contains validation or test objects"
            )
        raise ValueError("train-only data must contain the sole data contract")

    config = state.get("config")
    dataset = data["dataset"]
    split = data["split"]
    fit_view = data["fit_view"]
    calibration_view = data["calibration_view"]
    fit_loader = data["fit_loader"]
    calibration_loader = data["calibration_loader"]
    if (config is None or not isinstance(fit_view, IndexedDatasetView)
            or not isinstance(calibration_view, IndexedDatasetView)
            or not isinstance(split, dict)
            or set(split) != {
                "fit_indices", "calibration_indices", "fit_scenes",
                "calibration_scenes", "metadata",
            }):
        raise ValueError("live train-only views or split are invalid")

    config_values = {
        "dataset": getattr(config, "dataset", None),
        "joint_det": getattr(config, "joint_det", None),
        "butd": getattr(config, "butd", None),
        "butd_gt": getattr(config, "butd_gt", None),
        "butd_cls": getattr(config, "butd_cls", None),
        "eval": getattr(config, "eval", None),
    }
    if config_values != {
            "dataset": ["scanrefer"],
            "joint_det": False,
            "butd": True,
            "butd_gt": False,
            "butd_cls": False,
            "eval": False}:
        raise ValueError("live config is not ScanRefer train-only no-GT")
    if (getattr(dataset, "dataset_dict", None) != {"scanrefer": 1}
            or getattr(dataset, "test_dataset", None) != "scanrefer"
            or getattr(dataset, "split", None) != "train"
            or getattr(dataset, "joint_det", None) is not False
            or getattr(dataset, "butd", None) is not True
            or getattr(dataset, "butd_gt", None) is not False
            or getattr(dataset, "butd_cls", None) is not False
            or getattr(dataset, "augment", None) is not True
            or getattr(dataset, "augment_det", None) is not True):
        raise ValueError("live dataset is not ScanRefer train-only no-GT")

    if (fit_view.dataset is dataset
            or calibration_view.dataset is dataset
            or fit_view.dataset is calibration_view.dataset
            or fit_view.dataset.__class__ is not dataset.__class__
            or calibration_view.dataset.__class__ is not dataset.__class__):
        raise ValueError("fit/calibration views must be distinct shallow views")
    source_annotations = getattr(dataset, "annos", None)
    if (source_annotations is None
            or getattr(fit_view.dataset, "annos", None) is not source_annotations
            or getattr(calibration_view.dataset, "annos", None)
            is not source_annotations):
        raise ValueError(
            "fit and calibration views must share source annotations"
        )
    if (getattr(fit_view.dataset, "augment", None) is not True
            or getattr(fit_view.dataset, "augment_det", None) is not True
            or getattr(calibration_view.dataset, "augment", None) is not False
            or getattr(calibration_view.dataset, "augment_det", None)
            is not False):
        raise ValueError("live fit/calibration augmentation contract changed")

    fit_indices = tuple(split["fit_indices"])
    calibration_indices = tuple(split["calibration_indices"])
    if (fit_view.indices != fit_indices
            or calibration_view.indices != calibration_indices
            or fit_loader.dataset is not fit_view
            or calibration_loader.dataset is not calibration_view):
        raise ValueError("live train-only views or loaders changed")
    if (getattr(fit_loader, "batch_size", None) != PRODUCTION_BATCH_SIZE
            or getattr(calibration_loader, "batch_size", None)
            != PRODUCTION_BATCH_SIZE
            or getattr(fit_loader, "drop_last", None) is not False
            or getattr(calibration_loader, "drop_last", None) is not False):
        raise ValueError("live train-only loader contract changed")

    metadata = split["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("authoritative split metadata is invalid")
    mapping_sha256 = metadata.get("mapping_sha256")
    if (not isinstance(mapping_sha256, str) or len(mapping_sha256) != 64
            or metadata.get("fit_sample_count") != len(fit_view)
            or metadata.get("calibration_sample_count")
            != len(calibration_view)
            or len(fit_loader) != rec_finetune.natural_batch_count(
                len(fit_view), PRODUCTION_BATCH_SIZE
            )
            or len(calibration_loader) != rec_finetune.natural_batch_count(
                len(calibration_view), PRODUCTION_BATCH_SIZE
            )):
        raise ValueError("authoritative split or loader counts changed")
    dataset_class = "{}.{}".format(
        dataset.__class__.__module__, dataset.__class__.__qualname__
    )
    return {
        "schema": "scanrefer-rec-finetune-train-data-v1",
        "dataset_split": "train",
        "datasets": ["scanrefer"],
        "joint_det": False,
        "butd": True,
        "butd_gt": False,
        "butd_cls": False,
        "fit_augment": True,
        "fit_augment_det": True,
        "calibration_augment": False,
        "calibration_augment_det": False,
        "authoritative_split_metadata": copy.deepcopy(metadata),
        "authoritative_split_mapping_sha256": mapping_sha256,
        "fit_sample_count": len(fit_view),
        "calibration_sample_count": len(calibration_view),
        "fit_loader_batch_count": len(fit_loader),
        "calibration_loader_batch_count": len(calibration_loader),
        "batch_size": PRODUCTION_BATCH_SIZE,
        "drop_last": False,
        "validation_data_accessed": False,
        "dataset_class": dataset_class,
        "dataset_instance_count": 1,
        "fit_and_calibration_share_source_annotations": True,
        "validation_data_objects_present": False,
    }


def _require_production_device(args):
    if getattr(args, "device", None) != PRODUCTION_DEVICE:
        raise ValueError("production REC fine-tuning requires cuda:0")


def initialize_rec_finetune_run(
        args, *, model_factory=None, dataset_factory=None,
        loader_factory=None, expected_split_metadata=None,
        parent_loader=None, geometry_loader=None,
        geometry_validator=None):
    """Validate inputs, initialize models, then build train-only data."""
    _require_production_device(args)
    paths = validate_runtime_paths(args)
    initial_state = load_rec_finetune_initial_state(
        paths.backbone_checkpoint,
        paths.parent_reranker,
        paths.geometry_reranker,
        paths.data_root,
        device=args.device,
        model_factory=model_factory,
        parent_loader=parent_loader,
        geometry_loader=geometry_loader,
        geometry_validator=geometry_validator,
    )
    data = build_train_only_data(
        initial_state["config"],
        device=args.device,
        dataset_factory=dataset_factory,
        loader_factory=loader_factory,
        expected_split_metadata=expected_split_metadata,
    )
    if paths.output_dir.exists() or paths.output_dir.is_symlink():
        raise RuntimeError("output appeared during train-only initialization")
    initialized = {
        "args": args,
        "paths": paths,
        "initial_state": initial_state,
        "data": data,
        "smoke_steps": getattr(args, "smoke_steps", None),
    }
    initialized["train_data_contract"] = (
        build_rec_finetune_train_data_contract(initialized)
    )
    return initialized


def build_rec_finetune_criterion(config):
    """Build the exact production Hungarian matcher and loss set."""
    losses = ["boxes", "labels", "masks"]
    if bool(getattr(config, "use_contrastive_align", False)):
        losses.append("contrastive_align")
    matcher = HungarianMatcher(
        cost_class=1,
        cost_bbox=5,
        cost_giou=2,
        soft_token=bool(getattr(config, "use_soft_token_loss", False)),
    )
    return SetCriterion(
        matcher=matcher,
        losses=losses,
        eos_coef=0.1,
        temperature=0.07,
    )


def _clone_model_state(model, label):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("{} must be a torch module".format(label))
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def snapshot_rec_finetune_state(mcln, parent, geometry):
    """Clone the complete three-model state, including buffers, to CPU."""
    return {
        "mcln": _clone_model_state(mcln, "MCLN"),
        "parent": _clone_model_state(parent, "parent reranker"),
        "geometry": _clone_model_state(geometry, "geometry reranker"),
    }


def restore_rec_finetune_state(mcln, parent, geometry, snapshot):
    """Strictly restore one complete three-model snapshot."""
    if not isinstance(snapshot, dict) or set(snapshot) != {
            "mcln", "parent", "geometry"}:
        raise ValueError("REC fine-tune snapshot fields are invalid")
    for label, model in (
            ("mcln", mcln), ("parent", parent), ("geometry", geometry)):
        state = snapshot[label]
        if not isinstance(state, dict):
            raise ValueError("{} snapshot state is invalid".format(label))
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                "{} strict snapshot restore failed: {}".format(label, error)
            )


def build_rec_finetune_inputs(batch):
    """Return the exact GT-free input mapping used by TrainTester."""
    if not isinstance(batch, dict):
        raise ValueError("REC fine-tune batch must be a mapping")
    try:
        inputs = {
            "point_clouds": batch["point_clouds"].float(),
            "text": batch["utterances"],
            "det_boxes": batch["all_detected_boxes"],
            "det_bbox_label_mask": batch["all_detected_bbox_label_mask"],
            "det_class_ids": batch["all_detected_class_ids"],
            "superpoint": batch["superpoint"],
        }
    except KeyError as error:
        raise ValueError(
            "REC fine-tune batch is missing model input {}".format(error)
        )
    for key in (
            "positive_map", "modify_positive_map", "pron_positive_map",
            "other_entity_map", "rel_positive_map"):
        if key in batch:
            inputs[key] = batch[key]
    return inputs


def _reject_rec_target_only_fields(value, label):
    if not isinstance(value, dict):
        raise ValueError("{} must be a mapping".format(label))
    leaked = REC_TARGET_ONLY_FIELDS.intersection(value)
    if leaked:
        raise ValueError(
            "{} contains target-only ground-truth fields: {}".format(
                label, ", ".join(sorted(leaked))
            )
        )


def _selected_calibration_ious(forward_state):
    if not isinstance(forward_state, dict):
        raise ValueError("REC fine-tune forward state must be a mapping")
    runtime = forward_state.get("runtime_outputs")
    if not isinstance(runtime, dict):
        raise ValueError("REC fine-tune runtime outputs must be a mapping")
    mode = runtime.get("rec_geometry_runtime_mode")
    if mode == "flat_geometry_axis":
        scores = runtime.get("rec_geometry_scores")
        valid = runtime.get("rec_geometry_valid_mask")
        ious = forward_state.get("geometry_candidate_ious")
        if (not isinstance(scores, torch.Tensor)
                or not isinstance(valid, torch.Tensor)
                or not isinstance(ious, torch.Tensor)
                or scores.dim() != 2
                or scores.shape != valid.shape
                or scores.shape != ious.shape
                or valid.dtype != torch.bool
                or not bool(valid.any(dim=1).all().item())
                or not bool(torch.isfinite(scores[valid]).all().item())):
            raise ValueError("flat geometry calibration outputs are malformed")
        stable_scores = scores.float().masked_fill(~valid, -float("inf"))
        top1 = stable_scores.argmax(dim=1, keepdim=True)
        return ious.gather(1, top1).squeeze(1)
    if mode == "parent_query_axis":
        query_scores = runtime.get("rec_reranker_scores")
        parent_state = forward_state.get("parent_state")
        ious = forward_state.get("parent_candidate_ious")
        if (not isinstance(query_scores, torch.Tensor)
                or query_scores.dim() != 2
                or not isinstance(parent_state, dict)
                or not isinstance(ious, torch.Tensor)
                or ious.dim() != 2):
            raise ValueError("parent calibration outputs are malformed")
        top1_query = parent_state.get("top1_query_index")
        top1_mask = parent_state.get("parent_top1_mask")
        if (not isinstance(top1_query, torch.Tensor)
                or top1_query.shape != (query_scores.shape[0],)
                or top1_query.dtype != torch.long
                or not torch.equal(query_scores.argmax(dim=1), top1_query)
                or not isinstance(top1_mask, torch.Tensor)
                or top1_mask.dtype != torch.bool
                or top1_mask.shape != ious.shape
                or not bool((top1_mask.sum(dim=1) == 1).all().item())):
            raise ValueError("deployed parent Top-1 calibration is malformed")
        return ious.masked_select(top1_mask).reshape(ious.shape[0])
    raise ValueError("REC fine-tune runtime output mode is invalid")


_CALIBRATION_BRANCH_NAMES = (
    "default_top1",
    "source_selector_top1",
    "parent_top1",
    "geometry_top1",
    "raw_query_oracle",
    "parent_candidate_oracle",
    "geometry_candidate_oracle",
)


def _calibration_float_tensor(value, label, shape, device):
    if (not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(shape)
            or not torch.is_floating_point(value)):
        raise ValueError("{} shape or dtype is invalid".format(label))
    if value.device != device:
        raise ValueError("{} device is invalid".format(label))
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("{} must be finite".format(label))
    return value


def _calibration_candidate_ious(value, valid, label, batch_size, device):
    if (not isinstance(value, torch.Tensor) or value.dim() != 2
            or value.shape[0] != batch_size
            or value.shape[1] <= 0
            or not torch.is_floating_point(value)):
        raise ValueError("{} IoUs shape or dtype is invalid".format(label))
    if (value.device != device
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != value.shape
            or valid.device != device):
        raise ValueError("{} valid mask is invalid".format(label))
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("{} valid mask has an empty row".format(label))
    if (not bool(torch.isfinite(value).all().item())
            or bool(((value < 0.0) | (value > 1.0)).any().item())):
        raise ValueError("{} IoUs must be finite and lie in [0, 1]".format(
            label
        ))
    oracle = value.masked_fill(~valid, -float("inf")).max(dim=1).values
    return value, valid, oracle


def _build_calibration_branch_ious(end_points, forward_state, targets):
    """Build the seven train-only diagnostic branches after scorer forward."""
    if (not isinstance(end_points, dict)
            or not isinstance(forward_state, dict)
            or not isinstance(targets, dict)):
        raise ValueError("calibration diagnostic inputs must be mappings")
    centers = end_points.get("last_center")
    sizes = end_points.get("last_pred_size")
    if (not isinstance(centers, torch.Tensor) or centers.dim() != 3
            or centers.shape[-1] != 3 or not all(centers.shape)
            or not torch.is_floating_point(centers)
            or not isinstance(sizes, torch.Tensor)
            or sizes.shape != centers.shape
            or sizes.dtype != centers.dtype
            or sizes.device != centers.device):
        raise ValueError("calibration query boxes shape, dtype, or device is invalid")
    if (not bool(torch.isfinite(centers).all().item())
            or not bool(torch.isfinite(sizes).all().item())):
        raise ValueError("calibration query boxes must be finite")
    # Match the deployed evaluator: degenerate decoder sizes are clamped before
    # any IoU or ranking decision is made.
    effective_sizes = sizes.clamp(min=1e-6)
    batch_size, num_queries, _ = centers.shape
    device = centers.device

    gt_centers = targets.get("center_label")
    gt_sizes = targets.get("size_gts")
    gt_mask = targets.get("box_label_mask")
    if (not isinstance(gt_centers, torch.Tensor) or gt_centers.dim() != 3
            or gt_centers.shape[0] != batch_size
            or gt_centers.shape[1] <= 0 or gt_centers.shape[2] < 3
            or gt_centers.dtype != torch.float32
            or not isinstance(gt_sizes, torch.Tensor)
            or gt_sizes.shape != gt_centers.shape[:2] + (3,)
            or gt_sizes.dtype != torch.float32
            or gt_centers.device != device
            or gt_sizes.device != device):
        raise ValueError("calibration root GT shape, dtype, or device is invalid")
    if (not isinstance(gt_mask, torch.Tensor)
            or gt_mask.shape != gt_centers.shape[:2]):
        raise ValueError("calibration root GT mask shape is invalid")
    if gt_mask.device != device:
        raise ValueError("calibration root GT mask device is invalid")
    if torch.is_complex(gt_mask):
        raise ValueError("calibration root GT mask dtype is invalid")
    if gt_mask.dtype != torch.bool:
        if not bool(torch.isfinite(gt_mask).all().item()):
            raise ValueError("calibration root GT mask must be finite")
        if not bool(((gt_mask == 0) | (gt_mask == 1)).all().item()):
            raise ValueError("calibration root GT mask must contain only 0/1")
    gt_mask = gt_mask.bool()
    root_centers = gt_centers[:, :1, :3]
    root_sizes = gt_sizes[:, :1]
    root_mask = gt_mask[:, :1]
    if not bool(root_mask.all().item()):
        raise ValueError("every calibration sample needs a valid root GT")
    if (not bool(torch.isfinite(root_centers).all().item())
            or not bool(torch.isfinite(root_sizes).all().item())
            or not bool((root_sizes > 0.0).all().item())):
        raise ValueError("calibration root GT must be finite and positive")
    raw_query_ious = compute_query_ious(
        torch.cat([centers, effective_sizes], dim=-1),
        torch.cat([root_centers, root_sizes], dim=-1),
        root_mask,
    )
    if (raw_query_ious.shape != (batch_size, num_queries)
            or not bool(torch.isfinite(raw_query_ious).all().item())
            or bool(((raw_query_ious < 0.0)
                     | (raw_query_ious > 1.0)).any().item())):
        raise ValueError("raw query IoUs are invalid")

    source_scores = end_points.get("source_choice_source_scores")
    source_names = end_points.get("selector_choice_source_names")
    if (not isinstance(source_scores, dict) or "default" not in source_scores):
        raise ValueError("source choice scores must contain default")
    if (not isinstance(source_names, (list, tuple))
            or tuple(source_names) != tuple(source_scores)
            or len(source_names) != len(set(source_names))
            or any(not isinstance(name, str) or not name
                   for name in source_names)):
        raise ValueError("source names differ from source choice scores")
    normalized_source_scores = []
    for name in source_names:
        normalized_source_scores.append(_calibration_float_tensor(
            source_scores[name], "source choice {} scores".format(name),
            (batch_size, num_queries), device,
        ))
    selector_choice_scores = _calibration_float_tensor(
        end_points.get("selector_choice_scores"),
        "selector choice scores", (batch_size, len(source_names)), device,
    )
    selected_source_scores = _calibration_float_tensor(
        end_points.get("selected_source_scores"),
        "selected_source_scores", (batch_size, num_queries), device,
    )
    selected_source_id = end_points.get("selected_source_id")
    expected_source_id = selector_choice_scores.argmax(dim=1)
    if (not isinstance(selected_source_id, torch.Tensor)
            or selected_source_id.dtype != torch.long
            or selected_source_id.shape != (batch_size,)
            or selected_source_id.device != device
            or not torch.equal(selected_source_id, expected_source_id)):
        raise ValueError("selected_source_id differs from selector choice")
    stacked_source_scores = torch.stack(
        normalized_source_scores, dim=1
    )
    batch_indices = torch.arange(batch_size, device=device)
    expected_selected_scores = stacked_source_scores[
        batch_indices, selected_source_id
    ]
    if not torch.equal(selected_source_scores, expected_selected_scores):
        raise ValueError("selected_source_scores differ from selected source")
    default_top1 = source_scores["default"].argmax(dim=1, keepdim=True)
    source_top1 = selected_source_scores.argmax(dim=1, keepdim=True)
    default_top1_ious = raw_query_ious.gather(1, default_top1).squeeze(1)
    source_top1_ious = raw_query_ious.gather(1, source_top1).squeeze(1)

    parent_inputs = forward_state.get("parent_model_inputs")
    parent_state = forward_state.get("parent_state")
    if not isinstance(parent_inputs, dict) or not isinstance(parent_state, dict):
        raise ValueError("parent calibration state is invalid")
    parent_ious, parent_valid, parent_oracle = _calibration_candidate_ious(
        forward_state.get("parent_candidate_ious"),
        parent_inputs.get("valid_mask"), "parent candidate", batch_size,
        device,
    )
    runtime = forward_state.get("runtime_outputs")
    parent_query_scores = parent_state.get("query_scores")
    parent_query_indices = parent_state.get("query_indices")
    parent_top1_query = parent_state.get("top1_query_index")
    runtime_parent_scores = (
        runtime.get("rec_reranker_scores")
        if isinstance(runtime, dict) else None
    )
    if (not isinstance(parent_query_scores, torch.Tensor)
            or not torch.is_floating_point(parent_query_scores)
            or parent_query_scores.shape != (batch_size, num_queries)
            or parent_query_scores.device != device
            or not isinstance(parent_query_indices, torch.Tensor)
            or parent_query_indices.dtype != torch.long
            or parent_query_indices.shape != parent_ious.shape
            or parent_query_indices.device != device
            or not isinstance(parent_top1_query, torch.Tensor)
            or parent_top1_query.dtype != torch.long
            or parent_top1_query.shape != (batch_size,)
            or parent_top1_query.device != device
            or not isinstance(runtime_parent_scores, torch.Tensor)
            or runtime_parent_scores.dtype != parent_query_scores.dtype
            or runtime_parent_scores.shape != parent_query_scores.shape
            or runtime_parent_scores.device != device
            or not torch.equal(runtime_parent_scores, parent_query_scores)
            or bool(torch.isnan(parent_query_scores).any().item())
            or bool(torch.isposinf(parent_query_scores).any().item())):
        raise ValueError("parent deployed Top-1 state is invalid")
    expected_finite_query_scores = torch.zeros_like(
        parent_query_scores, dtype=torch.bool
    )
    for batch_index in range(batch_size):
        row_indices = parent_query_indices[
            batch_index, parent_valid[batch_index]
        ]
        if (bool((row_indices < 0).any().item())
                or bool((row_indices >= num_queries).any().item())
                or int(torch.unique(row_indices).numel())
                != int(row_indices.numel())):
            raise ValueError("parent deployed Top-1 state is invalid")
        expected_finite_query_scores[batch_index, row_indices] = True
    if (not torch.equal(
            torch.isfinite(parent_query_scores), expected_finite_query_scores)
            or not torch.equal(
                parent_query_scores.argmax(dim=1), parent_top1_query
            )):
        raise ValueError("parent deployed Top-1 state is invalid")
    parent_top1_mask = parent_state.get("parent_top1_mask")
    expected_parent_top1_mask = (
        parent_query_indices == parent_top1_query.unsqueeze(1)
    ) & parent_valid
    if (not isinstance(parent_top1_mask, torch.Tensor)
            or parent_top1_mask.dtype != torch.bool
            or parent_top1_mask.shape != parent_ious.shape
            or parent_top1_mask.device != device
            or not bool((parent_top1_mask.sum(dim=1) == 1).all().item())
            or not torch.equal(
                parent_top1_mask, expected_parent_top1_mask
            )):
        raise ValueError("parent Top-1 mask is invalid")
    parent_top1_ious = parent_ious.masked_select(
        parent_top1_mask
    ).reshape(batch_size)

    geometry_inputs = forward_state.get("geometry_model_inputs")
    if not isinstance(geometry_inputs, dict):
        raise ValueError("geometry calibration state is invalid")
    geometry_ious, geometry_valid, geometry_oracle = (
        _calibration_candidate_ious(
            forward_state.get("geometry_candidate_ious"),
            geometry_inputs.get("valid_mask"), "geometry candidate",
            batch_size, device,
        )
    )
    if (isinstance(runtime, dict)
            and runtime.get("rec_geometry_runtime_mode")
            == "flat_geometry_axis"):
        runtime_valid = runtime.get("rec_geometry_valid_mask")
        if (not isinstance(runtime_valid, torch.Tensor)
                or runtime_valid.dtype != torch.bool
                or runtime_valid.shape != geometry_valid.shape
                or runtime_valid.device != geometry_valid.device):
            raise ValueError("geometry runtime valid mask is invalid")
        if not torch.equal(runtime_valid, geometry_valid):
            raise ValueError("geometry runtime and model valid masks differ")
    geometry_top1_ious = _selected_calibration_ious(forward_state)
    _calibration_float_tensor(
        geometry_top1_ious, "geometry selected IoUs", (batch_size,), device
    )
    if bool(((geometry_top1_ious < 0.0)
             | (geometry_top1_ious > 1.0)).any().item()):
        raise ValueError("geometry selected IoUs must lie in [0, 1]")

    return {
        "default_top1": default_top1_ious.detach(),
        "source_selector_top1": source_top1_ious.detach(),
        "parent_top1": parent_top1_ious.detach(),
        "geometry_top1": geometry_top1_ious.detach(),
        "raw_query_oracle": raw_query_ious.max(dim=1).values.detach(),
        "parent_candidate_oracle": parent_oracle.detach(),
        "geometry_candidate_oracle": geometry_oracle.detach(),
    }


@dataclass(frozen=True)
class CalibrationObservation:
    """Frozen observation container with defensively copied inputs."""

    selection_metrics: dict
    diagnostics_result: rec_finetune.CalibrationDiagnosticsResult

    def __post_init__(self):
        object.__setattr__(
            self, "selection_metrics", copy.deepcopy(self.selection_metrics)
        )
        object.__setattr__(
            self, "diagnostics_result", copy.deepcopy(self.diagnostics_result)
        )


def calibrate_rec_finetune(
        mcln, parent, geometry, parent_artifact, geometry_artifact,
        calibration_loader, expected_indices, device, *,
        input_builder=None, forward_fn=None, diagnostic_builder=None):
    """Run one ordered, unaugmented, train-only calibration pass."""
    input_builder = input_builder or build_rec_finetune_inputs
    forward_fn = forward_fn or rec_finetune.build_rec_finetune_forward
    diagnostic_builder = diagnostic_builder or _build_calibration_branch_ious
    if not callable(diagnostic_builder):
        raise ValueError("calibration diagnostic builder must be callable")
    accumulator = rec_finetune.CalibrationAccumulator(expected_indices)
    diagnostics_accumulator = rec_finetune.CalibrationDiagnosticsAccumulator(
        expected_indices
    )
    rec_finetune.set_rec_finetune_eval_mode(mcln, parent, geometry)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.no_grad():
            for batch in calibration_loader:
                target_batch = _move_batch_to_device(batch, device)
                inputs = input_builder(target_batch)
                _reject_rec_target_only_fields(inputs, "calibration inputs")
                inputs["train"] = False
                end_points = mcln(inputs)
                if not isinstance(end_points, dict):
                    raise ValueError("MCLN output must be a mapping")
                _reject_rec_target_only_fields(
                    inputs, "post-MCLN calibration inputs"
                )
                _reject_rec_target_only_fields(
                    end_points, "calibration MCLN outputs"
                )
                forward_state = forward_fn(
                    end_points,
                    inputs,
                    target_batch,
                    parent,
                    parent_artifact,
                    geometry,
                    geometry_artifact,
                )
                indices = target_batch.get("dataset_index")
                if indices is None:
                    raise ValueError("calibration batch has no dataset_index")
                selected_ious = _selected_calibration_ious(forward_state)
                branch_ious = diagnostic_builder(
                    end_points, forward_state, target_batch
                )
                if (not isinstance(branch_ious, dict)
                        or set(branch_ious) != set(_CALIBRATION_BRANCH_NAMES)
                        or not isinstance(branch_ious.get("geometry_top1"),
                                          torch.Tensor)
                        or not torch.equal(
                            branch_ious["geometry_top1"], selected_ious
                        )):
                    raise ValueError(
                        "calibration diagnostic geometry Top-1 differs from selection"
                    )
                diagnostics_accumulator.update(indices, branch_ious)
                accumulator.update(indices, selected_ious)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return CalibrationObservation(
        selection_metrics=accumulator.finalize(),
        diagnostics_result=diagnostics_accumulator.finalize(),
    )


def _live_rec_finetune_state(mcln, parent, geometry):
    return {
        "mcln": mcln.state_dict(),
        "parent": parent.state_dict(),
        "geometry": geometry.state_dict(),
    }


def _validated_loop_contract(fit_loader, max_steps, contract_steps):
    if (not isinstance(max_steps, int) or isinstance(max_steps, bool)
            or max_steps <= 0):
        raise ValueError("maximum REC fine-tune steps must be positive")
    if (not isinstance(contract_steps, tuple) or not contract_steps
            or contract_steps[0] != 0
            or contract_steps[-1] != max_steps
            or any(not isinstance(step, int) or isinstance(step, bool)
                   for step in contract_steps)
            or any(left >= right for left, right in zip(
                contract_steps, contract_steps[1:]
            ))):
        raise ValueError("calibration steps must span zero through max_steps")
    try:
        natural_steps = len(fit_loader)
    except TypeError:
        natural_steps = None
    if natural_steps is not None and max_steps > natural_steps:
        raise ValueError("max_steps exceeds the natural fit loader")


_DIAGNOSTIC_GROUP_SPECS = (
    ("mcln_decoder_box", "mcln_names", "mcln_parameters", 0.1),
    ("parent_reranker", "parent_names", "parent_parameters", 1.0),
    ("geometry_reranker", "geometry_names", "geometry_parameters", 1.0),
)
_DIAGNOSTIC_LOSS_NAMES = ("hungarian", "parent", "geometry", "total")
_DIAGNOSTIC_RERANKER_NAMES = ("parent", "geometry")
_DIAGNOSTIC_RERANKER_LOSS_NAMES = (
    "loss_listwise",
    "loss_best_tier_pairwise",
    "loss_ranking",
    "loss_threshold",
    "loss_iou",
    "loss_total",
)
_DIAGNOSTIC_RERANKER_COUNT_NAMES = (
    "tier_pairwise_informative_rows",
    "tier_pairwise_pair_count",
    "tier_pairwise_positive_count",
    "tier_pairwise_negative_count",
)


def _canonical_parameter_name_sha256(names):
    payload = json.dumps(
        sorted(names), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_finite_primitive(label, value):
    if (not isinstance(value, numbers.Real) or isinstance(value, bool)):
        raise ValueError("{} must be a numeric primitive".format(label))
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("{} must be finite".format(label))
    return result


def validate_rec_finetune_losses(
        hungarian_loss, parent_loss, geometry_loss):
    """Validate scalar component losses and return total plus detached values."""
    components = (
        ("hungarian", hungarian_loss),
        ("parent", parent_loss),
        ("geometry", geometry_loss),
    )
    for name, value in components:
        if not isinstance(value, torch.Tensor):
            raise ValueError("{} loss must be a tensor".format(name))
        if value.ndim != 0:
            raise ValueError("{} loss must be scalar".format(name))
    total = hungarian_loss + parent_loss + geometry_loss
    if total.ndim != 0:
        raise ValueError("total loss must be scalar")
    names = tuple(name for name, _value in components) + ("total",)
    detached_values = torch.stack(
        tuple(value for _name, value in components) + (total,)
    ).detach().to(device="cpu")
    finite_flags = torch.isfinite(detached_values)
    if not bool(finite_flags.all().item()):
        bad_name = names[
            next(
                index for index, finite in enumerate(finite_flags.tolist())
                if not finite
            )
        ]
        raise FloatingPointError("{} loss is non-finite".format(bad_name))
    detached = {
        name: float(value)
        for name, value in zip(names, detached_values.tolist())
    }
    return total, detached


def _validated_group_inventory(groups):
    if not isinstance(groups, dict):
        raise ValueError("REC fine-tune groups must be a mapping")
    inventories = {}
    identities = set()
    for group_name, names_key, parameters_key, clip_limit in (
            _DIAGNOSTIC_GROUP_SPECS):
        names = groups.get(names_key)
        parameters = groups.get(parameters_key)
        if (not isinstance(names, tuple)
                or not isinstance(parameters, tuple)
                or not names
                or len(names) != len(parameters)
                or any(not isinstance(name, str) or not name
                       for name in names)
                or len(set(names)) != len(names)
                or any(not isinstance(parameter, torch.nn.Parameter)
                       for parameter in parameters)):
            raise ValueError(
                "{} diagnostic parameter group is invalid".format(group_name)
            )
        group_identities = {id(parameter) for parameter in parameters}
        if (len(group_identities) != len(parameters)
                or not identities.isdisjoint(group_identities)):
            raise ValueError("diagnostic parameter groups overlap")
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError(
                "{} contains a frozen parameter".format(group_name)
            )
        identities.update(group_identities)
        inventories[group_name] = {
            "names": names,
            "parameters": parameters,
            "parameter_tensor_count": len(parameters),
            "parameter_element_count": sum(
                parameter.numel() for parameter in parameters
            ),
            "parameter_names_sha256": _canonical_parameter_name_sha256(
                names
            ),
            "clip_limit": clip_limit,
        }
    return inventories


def _frozen_mcln_inventory(mcln):
    if not isinstance(mcln, torch.nn.Module):
        raise ValueError("MCLN diagnostics require a torch module")
    frozen = tuple(
        (name, parameter)
        for name, parameter in mcln.named_parameters()
        if not parameter.requires_grad
    )
    names = tuple(name for name, _parameter in frozen)
    return {
        "named_parameters": frozen,
        "parameter_tensor_count": len(frozen),
        "parameter_element_count": sum(
            parameter.numel() for _name, parameter in frozen
        ),
        "parameter_names_sha256": _canonical_parameter_name_sha256(names),
    }


def _validated_detached_losses(losses):
    if not isinstance(losses, dict) or set(losses) != set(
            _DIAGNOSTIC_LOSS_NAMES):
        raise ValueError("detached loss diagnostics are invalid")
    return {
        name: _require_finite_primitive("{} loss".format(name), losses[name])
        for name in _DIAGNOSTIC_LOSS_NAMES
    }


def _validated_reranker_objective_stats(reranker_stats, losses):
    if (not isinstance(reranker_stats, dict)
            or set(reranker_stats) != set(_DIAGNOSTIC_RERANKER_NAMES)):
        raise ValueError("reranker objective diagnostics are invalid")
    normalized = {}
    expected_fields = set(
        _DIAGNOSTIC_RERANKER_LOSS_NAMES
        + _DIAGNOSTIC_RERANKER_COUNT_NAMES
    )
    for name in _DIAGNOSTIC_RERANKER_NAMES:
        stats = reranker_stats[name]
        if not isinstance(stats, dict) or set(stats) != expected_fields:
            raise ValueError("{} objective diagnostics are invalid".format(name))
        normalized[name] = {}
        for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES:
            value = stats[field]
            if (not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                    or not torch.is_floating_point(value)):
                raise ValueError(
                    "{} {} diagnostic must be a float scalar tensor".format(
                        name, field
                    )
                )
            normalized[name][field] = _require_finite_primitive(
                "{} {} diagnostic".format(name, field),
                value.detach().to(device="cpu").item(),
            )
            if normalized[name][field] < 0.0:
                raise ValueError("reranker loss diagnostics must be nonnegative")
        for field in _DIAGNOSTIC_RERANKER_COUNT_NAMES:
            value = stats[field]
            if (not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                    or value.dtype != torch.long):
                raise ValueError(
                    "{} {} diagnostic must be a long scalar tensor".format(
                        name, field
                    )
                )
            count = int(value.detach().to(device="cpu").item())
            if count < 0:
                raise ValueError("reranker coverage counts must be nonnegative")
            normalized[name][field] = count
        if normalized[name]["loss_total"] != losses[name]:
            raise ValueError("{} objective total differs from loss".format(name))
        expected_ranking = (
            normalized[name]["loss_listwise"]
            if name == "parent"
            else normalized[name]["loss_best_tier_pairwise"]
        )
        if normalized[name]["loss_ranking"] != expected_ranking:
            raise ValueError("{} ranking objective route is invalid".format(name))
        informative = normalized[name]["tier_pairwise_informative_rows"]
        pairs = normalized[name]["tier_pairwise_pair_count"]
        positives = normalized[name]["tier_pairwise_positive_count"]
        negatives = normalized[name]["tier_pairwise_negative_count"]
        if (positives <= 0
                or informative > positives
                or informative > negatives
                or informative > pairs
                or pairs > positives * negatives
                or (pairs > 0) != (informative > 0)
                or (pairs > 0) != (negatives > 0)
                or (pairs == 0
                    and normalized[name]["loss_best_tier_pairwise"] != 0.0)):
            raise ValueError("{} pairwise coverage is invalid".format(name))
    return normalized


def _validated_reranker_objective_record(reranker_stats, losses):
    if (not isinstance(reranker_stats, dict)
            or set(reranker_stats) != set(_DIAGNOSTIC_RERANKER_NAMES)):
        raise ValueError("reranker objective record is invalid")
    expected_fields = set(
        _DIAGNOSTIC_RERANKER_LOSS_NAMES
        + _DIAGNOSTIC_RERANKER_COUNT_NAMES
    )
    normalized = {}
    for name in _DIAGNOSTIC_RERANKER_NAMES:
        stats = reranker_stats[name]
        if not isinstance(stats, dict) or set(stats) != expected_fields:
            raise ValueError("reranker objective record fields are invalid")
        normalized[name] = {
            field: _require_finite_primitive(
                "{} {} diagnostic".format(name, field), stats[field]
            )
            for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES
        }
        if any(normalized[name][field] < 0.0
               for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES):
            raise ValueError("reranker loss diagnostics must be nonnegative")
        for field in _DIAGNOSTIC_RERANKER_COUNT_NAMES:
            count = stats[field]
            if (not isinstance(count, int) or isinstance(count, bool)
                    or count < 0):
                raise ValueError("reranker coverage count is invalid")
            normalized[name][field] = count
        if normalized[name]["loss_total"] != losses[name]:
            raise ValueError("reranker objective total differs from loss")
        expected_ranking = (
            normalized[name]["loss_listwise"]
            if name == "parent"
            else normalized[name]["loss_best_tier_pairwise"]
        )
        informative = normalized[name]["tier_pairwise_informative_rows"]
        pairs = normalized[name]["tier_pairwise_pair_count"]
        positives = normalized[name]["tier_pairwise_positive_count"]
        negatives = normalized[name]["tier_pairwise_negative_count"]
        if (normalized[name]["loss_ranking"] != expected_ranking
                or positives <= 0
                or informative > positives
                or informative > negatives
                or informative > pairs
                or pairs > positives * negatives
                or (pairs > 0) != (informative > 0)
                or (pairs > 0) != (negatives > 0)
                or (pairs == 0
                    and normalized[name]["loss_best_tier_pairwise"] != 0.0)):
            raise ValueError("reranker objective record is inconsistent")
    return normalized


def _gradient_finite_flags(gradients):
    flags = []
    for gradient in gradients:
        values = (
            gradient.coalesce().values()
            if gradient.is_sparse else gradient
        )
        flags.append(torch.isfinite(values.detach()).all())
    return tuple(
        bool(value)
        for value in torch.stack(flags).detach().to(device="cpu").tolist()
    )


def collect_rec_finetune_update_diagnostics(
        mcln, groups, losses, reranker_stats, *, step):
    """Gate gradients, clip them, and return one detached update record."""
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("diagnostic update step must be positive")
    loss_values = _validated_detached_losses(losses)
    objective_stats = _validated_reranker_objective_stats(
        reranker_stats, loss_values
    )
    inventories = _validated_group_inventory(groups)
    group_records = {}
    for group_name, inventory in inventories.items():
        present = tuple(
            parameter.grad
            for parameter in inventory["parameters"]
            if parameter.grad is not None
        )
        if not present:
            raise RuntimeError(
                "{} has no gradients".format(group_name)
            )
        finite = tuple(
            gradient for gradient, is_finite in zip(
                present, _gradient_finite_flags(present)
            )
            if is_finite
        )
        if len(finite) != len(present):
            raise FloatingPointError(
                "{} has a non-finite gradient".format(group_name)
            )
        group_records[group_name] = {
            "parameter_tensor_count": inventory[
                "parameter_tensor_count"
            ],
            "parameter_element_count": inventory[
                "parameter_element_count"
            ],
            "parameter_names_sha256": inventory[
                "parameter_names_sha256"
            ],
            "gradient_tensor_count": len(present),
            "gradient_element_count": sum(
                gradient.numel() for gradient in present
            ),
            "finite_gradient_tensor_count": len(finite),
            "finite_gradient_element_count": sum(
                gradient.numel() for gradient in finite
            ),
            "all_present_finite": True,
            "clip_limit": inventory["clip_limit"],
        }

    frozen = _frozen_mcln_inventory(mcln)
    frozen_gradients = tuple(
        parameter.grad
        for _name, parameter in frozen["named_parameters"]
        if parameter.grad is not None
    )
    if frozen_gradients:
        raise RuntimeError("frozen MCLN parameter has a gradient")
    frozen_record = {
        "parameter_tensor_count": frozen["parameter_tensor_count"],
        "parameter_element_count": frozen["parameter_element_count"],
        "parameter_names_sha256": frozen["parameter_names_sha256"],
        "gradient_tensor_count": 0,
        "gradient_element_count": 0,
    }

    norms = rec_finetune.clip_rec_finetune_gradients(groups)
    expected_names = {spec[0] for spec in _DIAGNOSTIC_GROUP_SPECS}
    if not isinstance(norms, dict) or set(norms) != expected_names:
        raise ValueError("pre-clip gradient norm result is invalid")
    for group_name in expected_names:
        group_records[group_name]["preclip_gradient_norm"] = (
            _require_finite_primitive(
                "{} pre-clip gradient norm".format(group_name),
                norms[group_name],
            )
        )
    return {
        "schema": "rec-finetune-update-diagnostics-v2",
        "step": step,
        "losses": loss_values,
        "ranking_objectives": objective_stats,
        "groups": group_records,
        "frozen_mcln": frozen_record,
    }


def create_rec_finetune_training_diagnostics(mcln, groups):
    """Create a constant-size aggregate without retaining parameter tensors."""
    inventories = _validated_group_inventory(groups)
    frozen = _frozen_mcln_inventory(mcln)
    return {
        "schema": "rec-finetune-training-diagnostics-v2",
        "update_count": 0,
        "trainable_groups": {
            name: {
                "parameter_tensor_count": inventory[
                    "parameter_tensor_count"
                ],
                "parameter_element_count": inventory[
                    "parameter_element_count"
                ],
                "parameter_names_sha256": inventory[
                    "parameter_names_sha256"
                ],
                "gradient_tensor_count": {
                    "min": None, "max": None, "last": None,
                },
                "gradient_element_count": {
                    "min": None, "max": None, "last": None,
                },
                "preclip_gradient_norm": {
                    "min": None, "max": None, "last": None,
                },
                "clip_limit": inventory["clip_limit"],
                "all_present_finite": True,
            }
            for name, inventory in inventories.items()
        },
        "frozen_mcln": {
            "parameter_tensor_count": frozen["parameter_tensor_count"],
            "parameter_element_count": frozen["parameter_element_count"],
            "parameter_names_sha256": frozen["parameter_names_sha256"],
        },
        "losses": {
            name: {
                "min": None, "max": None, "last": None,
                "all_finite": True,
            }
            for name in _DIAGNOSTIC_LOSS_NAMES
        },
        "ranking_objectives": {
            name: {
                **{
                    field: {
                        "min": None, "max": None, "last": None,
                        "all_finite": True,
                    }
                    for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES
                },
                **{
                    field: {
                        "min": None, "max": None, "last": None,
                        "total": 0,
                    }
                    for field in _DIAGNOSTIC_RERANKER_COUNT_NAMES
                },
            }
            for name in _DIAGNOSTIC_RERANKER_NAMES
        },
        "all_present_finite": True,
        "frozen_gradient_tensors_seen": 0,
        "last_update": None,
    }


def _update_min_max_last(summary, value):
    summary["min"] = value if summary["min"] is None else min(
        summary["min"], value
    )
    summary["max"] = value if summary["max"] is None else max(
        summary["max"], value
    )
    summary["last"] = value


def update_rec_finetune_training_diagnostics(aggregate, record):
    """Fold one validated primitive update record into the run summary."""
    if (not isinstance(aggregate, dict)
            or aggregate.get("schema")
            != "rec-finetune-training-diagnostics-v2"):
        raise ValueError("training diagnostics aggregate is invalid")
    if (not isinstance(record, dict)
            or record.get("schema")
            != "rec-finetune-update-diagnostics-v2"
            or record.get("step") != aggregate["update_count"] + 1):
        raise ValueError("update diagnostic record is invalid or out of order")
    losses = _validated_detached_losses(record.get("losses"))
    objective_stats = _validated_reranker_objective_record(
        record.get("ranking_objectives"), losses
    )
    if set(record.get("groups", {})) != set(
            aggregate["trainable_groups"]):
        raise ValueError("update diagnostic groups differ from aggregate")
    for name, value in losses.items():
        _update_min_max_last(aggregate["losses"][name], value)
    for name, stats in objective_stats.items():
        summary = aggregate["ranking_objectives"][name]
        for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES:
            _update_min_max_last(summary[field], stats[field])
        for field in _DIAGNOSTIC_RERANKER_COUNT_NAMES:
            _update_min_max_last(summary[field], stats[field])
            summary[field]["total"] += stats[field]
    for group_name, static in aggregate["trainable_groups"].items():
        update = record["groups"][group_name]
        for field in (
                "parameter_tensor_count", "parameter_element_count",
                "parameter_names_sha256", "clip_limit"):
            if update.get(field) != static[field]:
                raise ValueError(
                    "{} static diagnostics changed".format(group_name)
                )
        if (update.get("all_present_finite") is not True
                or update.get("gradient_tensor_count", 0) <= 0
                or update.get("finite_gradient_tensor_count")
                != update.get("gradient_tensor_count")
                or update.get("finite_gradient_element_count")
                != update.get("gradient_element_count")):
            raise ValueError("update gradient diagnostics did not pass")
        for field in (
                "gradient_tensor_count", "gradient_element_count",
                "preclip_gradient_norm"):
            value = update.get(field)
            if field == "preclip_gradient_norm":
                value = _require_finite_primitive(
                    "pre-clip gradient norm", value
                )
            elif (not isinstance(value, int) or isinstance(value, bool)
                  or value <= 0):
                raise ValueError("gradient count is invalid")
            _update_min_max_last(static[field], value)
    frozen = record.get("frozen_mcln", {})
    for field in (
            "parameter_tensor_count", "parameter_element_count",
            "parameter_names_sha256"):
        if frozen.get(field) != aggregate["frozen_mcln"][field]:
            raise ValueError("frozen MCLN inventory changed")
    seen = frozen.get("gradient_tensor_count")
    if (seen != 0 or frozen.get("gradient_element_count") != 0):
        raise ValueError("frozen MCLN gradient diagnostics did not pass")
    aggregate["frozen_gradient_tensors_seen"] += seen
    aggregate["update_count"] += 1
    aggregate["last_update"] = copy.deepcopy(record)


def _progress_cuda_memory(device):
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
        "reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
    }


def _emit_rec_finetune_progress(
        event, phase, step, metrics, training_diagnostics, device):
    last_update = training_diagnostics.get("last_update")
    if last_update is None:
        loss_summary = None
        grad_norms = None
        ranking_objectives = None
    else:
        loss_summary = copy.deepcopy(last_update["losses"])
        grad_norms = {
            name: group["preclip_gradient_norm"]
            for name, group in last_update["groups"].items()
        }
        ranking_objectives = copy.deepcopy(
            last_update["ranking_objectives"]
        )
    payload = {
        "schema": "rec-finetune-progress-v2",
        "event": event,
        "phase": phase,
        "step": int(step),
        "loss_summary": loss_summary,
        "ranking_objectives": ranking_objectives,
        "grad_norms": grad_norms,
        "cuda_memory": _progress_cuda_memory(device),
        "metrics": copy.deepcopy(metrics),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    print(encoded, file=sys.stderr, flush=True)


_CALIBRATION_SELECTION_METRIC_FIELDS = frozenset((
    "sample_count", "hits025", "hits050", "acc025", "acc050", "score",
))


def _unpack_calibration_observation(value):
    if isinstance(value, CalibrationObservation):
        metrics = value.selection_metrics
        diagnostics_result = value.diagnostics_result
        if (not isinstance(metrics, dict)
                or set(metrics) != _CALIBRATION_SELECTION_METRIC_FIELDS
                or not isinstance(
                    diagnostics_result,
                    rec_finetune.CalibrationDiagnosticsResult,
                )):
            raise ValueError("diagnostic calibration observation is invalid")
        return (
            "diagnostic",
            copy.deepcopy(metrics),
            copy.deepcopy(diagnostics_result),
        )
    if (isinstance(value, dict)
            and set(value) == _CALIBRATION_SELECTION_METRIC_FIELDS):
        return "legacy", copy.deepcopy(value), None
    raise ValueError("calibration observation fields are invalid")


def fit_rec_finetune_one_epoch(
        mcln, parent, geometry, parent_artifact, geometry_artifact, groups,
        optimizer, set_criterion, fit_loader, calibration_loader,
        calibration_indices, device, *, max_steps=PRODUCTION_MAX_STEPS,
        calibration_steps=rec_finetune.CALIBRATION_STEPS,
        input_builder=None, forward_fn=None, hungarian_loss_fn=None,
        calibration_fn=None, smoke_run=False):
    """Fit at most one natural loader epoch and restore the selected state."""
    _validated_loop_contract(fit_loader, max_steps, calibration_steps)
    input_builder = input_builder or build_rec_finetune_inputs
    forward_fn = forward_fn or rec_finetune.build_rec_finetune_forward
    hungarian_loss_fn = hungarian_loss_fn or compute_hungarian_loss
    calibration_fn = calibration_fn or calibrate_rec_finetune
    if not isinstance(smoke_run, bool):
        raise ValueError("smoke progress mode must be boolean")
    selector = rec_finetune.CalibrationSelector(
        contract_steps=calibration_steps,
        expected_sample_count=len(tuple(calibration_indices)),
    )
    training_diagnostics = create_rec_finetune_training_diagnostics(
        mcln, groups
    )
    calibration_mode = None
    calibration_diagnostics_history = []
    diagnostic_results_by_step = {}
    previous_diagnostic_step = None
    previous_diagnostics_result = None

    def observe(step):
        nonlocal calibration_mode
        nonlocal previous_diagnostic_step
        nonlocal previous_diagnostics_result
        raw_observation = calibration_fn(
            mcln,
            parent,
            geometry,
            parent_artifact,
            geometry_artifact,
            calibration_loader,
            calibration_indices,
            device,
            input_builder=input_builder,
            forward_fn=forward_fn,
        )
        mode, metrics, diagnostics_result = _unpack_calibration_observation(
            raw_observation
        )
        if calibration_mode is None:
            calibration_mode = mode
        elif mode != calibration_mode:
            raise ValueError(
                "cannot mix legacy and diagnostic calibration observations"
            )
        transition = None
        if mode == "diagnostic" and previous_diagnostics_result is not None:
            transition = rec_finetune.build_calibration_step_transition(
                previous_diagnostics_result.transition_state,
                diagnostics_result.transition_state,
                previous_diagnostic_step,
                step,
            )
        decision = selector.observe(
            step, metrics, _live_rec_finetune_state(mcln, parent, geometry)
        )
        if mode == "diagnostic":
            calibration_diagnostics_history.append({
                "step": step,
                "diagnostics": copy.deepcopy(
                    diagnostics_result.diagnostics
                ),
                "transition_from_previous": copy.deepcopy(transition),
            })
            diagnostic_results_by_step[step] = diagnostics_result
            previous_diagnostic_step = step
            previous_diagnostics_result = diagnostics_result
        _emit_rec_finetune_progress(
            "calibration", "contract", step, metrics,
            training_diagnostics, device,
        )
        return decision

    def fit_one_update(batch, step):
        rec_finetune.set_rec_finetune_train_mode(mcln, parent, geometry)
        optimizer.zero_grad(set_to_none=True)
        target_batch = _move_batch_to_device(batch, device)
        inputs = input_builder(target_batch)
        _reject_rec_target_only_fields(inputs, "fit inputs")
        end_points = mcln(inputs)
        if not isinstance(end_points, dict):
            raise ValueError("MCLN output must be a mapping")
        _reject_rec_target_only_fields(inputs, "post-MCLN fit inputs")
        _reject_rec_target_only_fields(end_points, "fit MCLN outputs")
        collisions = set(end_points).intersection(target_batch)
        if collisions:
            raise ValueError(
                "MCLN/target key collision: {}".format(
                    ", ".join(sorted(collisions))
                )
            )
        forward_state = forward_fn(
            end_points,
            inputs,
            target_batch,
            parent,
            parent_artifact,
            geometry,
            geometry_artifact,
        )
        end_points.update(target_batch)
        hungarian_loss, _end_points = hungarian_loss_fn(
            end_points,
            6,
            set_criterion,
            query_points_obj_topk=4,
            source_choice_selector_loss_weight=0.0,
            mask_loss_scale=0.1,
            consistency_loss_scale=0.1,
        )
        if not isinstance(forward_state, dict):
            raise ValueError("REC fine-tune forward state must be a mapping")
        total, loss_values = validate_rec_finetune_losses(
            hungarian_loss,
            forward_state.get("parent_loss"),
            forward_state.get("geometry_loss"),
        )
        total.backward()
        update_diagnostics = collect_rec_finetune_update_diagnostics(
            mcln,
            groups,
            loss_values,
            {
                "parent": forward_state.get("parent_loss_stats"),
                "geometry": forward_state.get("geometry_loss_stats"),
            },
            step=step,
        )
        update_rec_finetune_training_diagnostics(
            training_diagnostics, update_diagnostics
        )
        optimizer.step()
        if ((smoke_run and step == 1)
                or (not smoke_run
                    and step % PRODUCTION_PROGRESS_INTERVAL == 0)):
            _emit_rec_finetune_progress(
                "update", "fit", step, None,
                training_diagnostics, device,
            )

    observe(0)
    completed_updates = 0
    stopped = False
    for batch in fit_loader:
        if completed_updates >= max_steps:
            break
        fit_one_update(batch, completed_updates + 1)
        completed_updates += 1
        if completed_updates in calibration_steps[1:]:
            optimizer.zero_grad(set_to_none=True)
            decision = observe(completed_updates)
            if decision.regression:
                stopped = True
                break

    if completed_updates not in calibration_steps:
        raise RuntimeError("fit loader ended outside the calibration contract")
    best_snapshot = selector.best_snapshot
    if best_snapshot is None or selector.best_step is None:
        raise RuntimeError("calibration selector produced no best snapshot")
    restore_rec_finetune_state(mcln, parent, geometry, best_snapshot)
    reproduced_observation = calibration_fn(
        mcln,
        parent,
        geometry,
        parent_artifact,
        geometry_artifact,
        calibration_loader,
        calibration_indices,
        device,
        input_builder=input_builder,
        forward_fn=forward_fn,
    )
    reproduced_mode, reproduced, reproduced_diagnostics_result = (
        _unpack_calibration_observation(reproduced_observation)
    )
    if reproduced_mode != calibration_mode:
        raise ValueError(
            "cannot mix legacy and diagnostic calibration observations"
        )
    _emit_rec_finetune_progress(
        "calibration", "selected_reproduction", selector.best_step,
        reproduced, training_diagnostics, device,
    )
    if reproduced != selector.best_metrics:
        raise RuntimeError("selected calibration metrics did not reproduce exactly")
    if calibration_mode == "diagnostic":
        selected_diagnostics_result = diagnostic_results_by_step.get(
            selector.best_step
        )
        if selected_diagnostics_result is None:
            raise RuntimeError("selected calibration diagnostics are missing")
        selected_calibration_output_sha256 = (
            rec_finetune.calibration_selected_output_sha256(
                selected_diagnostics_result.transition_state
            )
        )
        reproduced_calibration_output_sha256 = (
            rec_finetune.calibration_selected_output_sha256(
                reproduced_diagnostics_result.transition_state
            )
        )
        if (reproduced_calibration_output_sha256
                != selected_calibration_output_sha256):
            raise RuntimeError(
                "selected calibration output did not reproduce exactly"
            )
        selected_calibration_diagnostics = copy.deepcopy(
            selected_diagnostics_result.diagnostics
        )
        reproduced_calibration_diagnostics = copy.deepcopy(
            reproduced_diagnostics_result.diagnostics
        )
    else:
        selected_calibration_output_sha256 = None
        reproduced_calibration_output_sha256 = None
        selected_calibration_diagnostics = None
        reproduced_calibration_diagnostics = None
    return {
        "completed_updates": completed_updates,
        "stopped_early": stopped,
        "selected_step": selector.best_step,
        "selected_metrics": selector.best_metrics,
        "reproduced_metrics": reproduced,
        "calibration_history": list(selector.history),
        "calibration_diagnostics_history": copy.deepcopy(
            calibration_diagnostics_history
        ),
        "selected_calibration_diagnostics": (
            selected_calibration_diagnostics
        ),
        "reproduced_calibration_diagnostics": (
            reproduced_calibration_diagnostics
        ),
        "selected_calibration_output_sha256": (
            selected_calibration_output_sha256
        ),
        "reproduced_calibration_output_sha256": (
            reproduced_calibration_output_sha256
        ),
        "training_diagnostics": training_diagnostics,
    }


def run_rec_finetune(
        initialized, *, max_steps=None, calibration_steps=None,
        input_builder=None, forward_fn=None, hungarian_loss_fn=None,
        calibration_fn=None):
    """Run the train-only loop from an initialized Task 6 state."""
    if not isinstance(initialized, dict):
        raise ValueError("initialized REC fine-tune run must be a mapping")
    state = initialized["initial_state"]
    data = initialized["data"]
    smoke_steps = initialized.get("smoke_steps")
    if max_steps is None:
        max_steps = smoke_steps or PRODUCTION_MAX_STEPS
    if calibration_steps is None:
        calibration_steps = (
            (0, max_steps)
            if smoke_steps is not None
            else rec_finetune.CALIBRATION_STEPS
        )
    set_criterion = state.get("set_criterion")
    if set_criterion is None:
        set_criterion = build_rec_finetune_criterion(state["config"])
    train_data_contract = initialized.get("train_data_contract")
    if (not isinstance(train_data_contract, dict)
            or train_data_contract.get("schema")
            != "scanrefer-rec-finetune-train-data-v1"):
        raise ValueError("initialized train-data contract is invalid")
    result = fit_rec_finetune_one_epoch(
        state["mcln"],
        state["parent"],
        state["geometry"],
        state["parent_artifact"],
        state["geometry_artifact"],
        state["groups"],
        state["optimizer"],
        set_criterion,
        data["fit_loader"],
        data["calibration_loader"],
        data["calibration_view"].indices,
        next(state["mcln"].parameters()).device,
        max_steps=max_steps,
        calibration_steps=calibration_steps,
        input_builder=input_builder,
        forward_fn=forward_fn,
        hungarian_loss_fn=hungarian_loss_fn,
        calibration_fn=calibration_fn,
        smoke_run=smoke_steps is not None,
    )
    result["train_data_contract"] = copy.deepcopy(train_data_contract)
    return result


def _fsync_directory(path):
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path, payload):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("publication file already exists: {}".format(path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _atomic_canonical_json(path, payload):
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("publication file already exists: {}".format(path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _immutable_file_identity(path, label):
    resolved, _snapshot, digest = rec_finetune._stable_artifact_snapshot(
        path, label
    )
    metadata = os.stat(str(resolved), follow_symlinks=False)
    return {
        "path": resolved,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "size": metadata.st_size,
        "sha256": digest,
    }


def _assert_immutable_inputs(identities):
    for label, expected in identities.items():
        actual = _immutable_file_identity(expected["path"], label)
        if actual != expected:
            raise RuntimeError("{} changed during publication".format(label))


def _assert_staged_publication_files(
        staging_paths, expected_sha256, *, require_read_only=False):
    if set(staging_paths) != set(expected_sha256):
        raise ValueError("staged publication file set is invalid")
    for stage, expected in expected_sha256.items():
        path = Path(staging_paths[stage])
        resolved, _snapshot, actual = rec_finetune._stable_artifact_snapshot(
            path, "staged " + stage
        )
        if resolved != path.resolve() or actual != expected:
            raise RuntimeError("staged {} SHA-256 changed".format(stage))
        if require_read_only:
            mode = os.stat(str(path), follow_symlinks=False).st_mode & 0o777
            if mode != 0o444:
                raise RuntimeError("staged {} is not read-only".format(stage))


def _optimizer_publication_contract():
    return [
        {
            "name": "mcln_decoder_box",
            "lr": 2e-5,
            "weight_decay": 5e-4,
            "grad_clip": 0.1,
        },
        {
            "name": "parent_reranker",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
        },
        {
            "name": "geometry_reranker",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
        },
    ]


def _loss_publication_contract():
    return {
        "matcher_costs": {"class": 1.0, "bbox": 5.0, "giou": 2.0},
        "num_decoder_layers": 6,
        "query_points_obj_topk": 4,
        "scales": {
            "mask": 0.1,
            "consistency": 0.1,
            "source_choice": 0.0,
            "parent": 1.0,
            "geometry": 1.0,
        },
        "reranker_loss_weights": {
            "ranking": 1.0,
            "threshold": 1.0,
            "iou": 0.5,
        },
        "reranker_ranking_objectives": {
            "parent": {
                "name": "single-best-iou-listwise-v1",
                "tier_pairwise_alpha": 0.0,
            },
            "geometry": {
                "name": "best-tier-pairwise-v1",
                "tier_pairwise_alpha": 1.0,
                "thresholds": [0.25, 0.50],
                "threshold_operator": "strict_gt",
                "positive_policy": "all_valid_candidates_in_best_tier",
                "negative_policy": "all_valid_candidates_below_best_tier",
                "loss": "softplus(negative_logit-positive_logit)",
                "pair_reduction": "mean_within_row",
                "row_reduction": "mean_over_informative_rows",
                "no_pair_policy": "differentiable_zero",
            },
        },
    }


def _online_provenance(groups, run_result):
    losses = _loss_publication_contract()
    return {
        "initial_backbone_sha256": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
        ),
        "initial_parent_artifact_sha256": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256
        ),
        "initial_geometry_artifact_sha256": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256
        ),
        "authoritative_split_mapping_sha256": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0[
                "mapping_sha256"
            ]
        ),
        "selected_step": run_result["selected_step"],
        "validation_data_accessed": False,
        "normalization_policy": "fixed-initial-artifact-v1",
        "parent_reranker_weight": 0.9,
        "geometry_reranker_weight": 1.0,
        "matcher_costs": copy.deepcopy(losses["matcher_costs"]),
        "loss_scales": copy.deepcopy(losses["scales"]),
        "reranker_loss_weights": copy.deepcopy(
            losses["reranker_loss_weights"]
        ),
        "reranker_ranking_objectives": copy.deepcopy(
            losses["reranker_ranking_objectives"]
        ),
        "optimizer_groups": _optimizer_publication_contract(),
        "max_steps": PRODUCTION_MAX_STEPS,
        "calibration_steps": list(rec_finetune.CALIBRATION_STEPS),
        "mcln_trainable_parameter_names": sorted(groups["mcln_names"]),
        "calibration_history": copy.deepcopy(
            run_result["calibration_history"]
        ),
    }


def _replay_calibration_history(history, contract_steps):
    if not isinstance(history, list) or not history:
        raise ValueError("calibration history is invalid")
    first_metrics = history[0].get("metrics") if isinstance(history[0], dict) else None
    sample_count = (
        first_metrics.get("sample_count")
        if isinstance(first_metrics, dict) else None
    )
    selector = rec_finetune.CalibrationSelector(
        contract_steps=contract_steps,
        expected_sample_count=sample_count,
    )
    try:
        for record in history:
            if not isinstance(record, dict):
                raise ValueError("calibration history record is invalid")
            selector.observe(
                record.get("step"),
                record.get("metrics"),
                {"state": torch.tensor(0.0)},
            )
            if selector.history[-1] != record:
                raise ValueError("calibration history is internally inconsistent")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "calibration history is invalid: {}".format(error)
        )
    return selector


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_min_max_last(summary, label, *, integer=False):
    if not isinstance(summary, dict) or set(summary) != {
            "min", "max", "last"}:
        raise ValueError("{} summary is invalid".format(label))
    values = tuple(summary[name] for name in ("min", "max", "last"))
    if integer:
        if any(not isinstance(value, int) or isinstance(value, bool)
               or value <= 0 for value in values):
            raise ValueError("{} count summary is invalid".format(label))
    else:
        values = tuple(
            _require_finite_primitive(label, value) for value in values
        )
        if any(value < 0.0 for value in values):
            raise ValueError("{} summary must be nonnegative".format(label))
    if values[0] > values[2] or values[2] > values[1]:
        raise ValueError("{} summary bounds are invalid".format(label))
    return values


def _validate_count_summary(summary, label, expected_updates):
    if not isinstance(summary, dict) or set(summary) != {
            "min", "max", "last", "total"}:
        raise ValueError("{} count summary is invalid".format(label))
    values = tuple(summary[name] for name in ("min", "max", "last", "total"))
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in values):
        raise ValueError("{} count summary is invalid".format(label))
    minimum, maximum, last, total = values
    if (minimum > last or last > maximum
            or total < minimum * expected_updates
            or total > maximum * expected_updates):
        raise ValueError("{} count summary bounds are invalid".format(label))
    return values


def _validate_training_diagnostics(
        diagnostics, expected_updates, *, mcln=None, groups=None):
    expected_fields = {
        "schema", "update_count", "trainable_groups", "frozen_mcln",
        "losses", "ranking_objectives", "all_present_finite",
        "frozen_gradient_tensors_seen", "last_update",
    }
    if (not isinstance(diagnostics, dict)
            or set(diagnostics) != expected_fields
            or diagnostics.get("schema")
            != "rec-finetune-training-diagnostics-v2"
            or diagnostics.get("update_count") != expected_updates
            or diagnostics.get("all_present_finite") is not True
            or diagnostics.get("frozen_gradient_tensors_seen") != 0):
        raise ValueError(
            "selected loop training diagnostics are invalid"
        )
    losses = diagnostics["losses"]
    if not isinstance(losses, dict) or set(losses) != set(
            _DIAGNOSTIC_LOSS_NAMES):
        raise ValueError("selected loop loss diagnostics are invalid")
    for name, summary in losses.items():
        if (not isinstance(summary, dict)
                or set(summary) != {"min", "max", "last", "all_finite"}
                or summary.get("all_finite") is not True):
            raise ValueError("selected loop loss diagnostics are invalid")
        _validate_min_max_last(
            {key: summary[key] for key in ("min", "max", "last")},
            "{} loss".format(name),
        )

    objective_summaries = diagnostics["ranking_objectives"]
    if (not isinstance(objective_summaries, dict)
            or set(objective_summaries)
            != set(_DIAGNOSTIC_RERANKER_NAMES)):
        raise ValueError("selected loop ranking diagnostics are invalid")
    expected_objective_fields = set(
        _DIAGNOSTIC_RERANKER_LOSS_NAMES
        + _DIAGNOSTIC_RERANKER_COUNT_NAMES
    )
    for name, summary in objective_summaries.items():
        if not isinstance(summary, dict) or set(summary) != expected_objective_fields:
            raise ValueError("selected loop ranking diagnostics are invalid")
        for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES:
            values = summary[field]
            if (not isinstance(values, dict)
                    or set(values) != {"min", "max", "last", "all_finite"}
                    or values.get("all_finite") is not True):
                raise ValueError("selected loop ranking loss is invalid")
            _validate_min_max_last(
                {key: values[key] for key in ("min", "max", "last")},
                "{} {}".format(name, field),
            )
        count_values = {
            field: _validate_count_summary(
                summary[field], "{} {}".format(name, field), expected_updates
            )
            for field in _DIAGNOSTIC_RERANKER_COUNT_NAMES
        }
        informative = count_values[
            "tier_pairwise_informative_rows"
        ][3]
        pairs = count_values["tier_pairwise_pair_count"][3]
        positives = count_values["tier_pairwise_positive_count"][3]
        negatives = count_values["tier_pairwise_negative_count"][3]
        if (positives <= 0
                or informative > positives
                or informative > negatives
                or informative > pairs
                or pairs > positives * negatives
                or (pairs > 0) != (informative > 0)
                or (pairs > 0) != (negatives > 0)
                or (pairs == 0 and any(
                    summary["loss_best_tier_pairwise"][field] != 0.0
                    for field in ("min", "max", "last")
                ))):
            raise ValueError("selected loop pairwise coverage is invalid")
        expected_ranking_field = (
            "loss_listwise"
            if name == "parent" else "loss_best_tier_pairwise"
        )
        if (summary["loss_ranking"] != summary[expected_ranking_field]
                or summary["loss_total"] != losses[name]):
            raise ValueError("selected loop ranking objective is inconsistent")
    geometry_coverage = objective_summaries["geometry"]
    if (geometry_coverage["tier_pairwise_informative_rows"]["total"] <= 0
            or geometry_coverage["tier_pairwise_pair_count"]["total"] <= 0):
        raise ValueError("geometry pairwise objective had no training signal")
    live_inventory = None
    live_frozen = None
    if mcln is not None or groups is not None:
        if not isinstance(mcln, torch.nn.Module) or not isinstance(groups, dict):
            raise ValueError("live diagnostic inventory is invalid")
        live_inventory = _validated_group_inventory(groups)
        live_frozen = _frozen_mcln_inventory(mcln)
    group_summaries = diagnostics["trainable_groups"]
    expected_group_names = {spec[0] for spec in _DIAGNOSTIC_GROUP_SPECS}
    if (not isinstance(group_summaries, dict)
            or set(group_summaries) != expected_group_names):
        raise ValueError("selected loop diagnostic groups are invalid")
    for group_name, summary in group_summaries.items():
        if not isinstance(summary, dict) or set(summary) != {
                "parameter_tensor_count", "parameter_element_count",
                "parameter_names_sha256", "gradient_tensor_count",
                "gradient_element_count", "preclip_gradient_norm",
                "clip_limit", "all_present_finite"}:
            raise ValueError("selected loop diagnostic group is invalid")
        if (not isinstance(summary["parameter_tensor_count"], int)
                or isinstance(summary["parameter_tensor_count"], bool)
                or summary["parameter_tensor_count"] <= 0
                or not isinstance(summary["parameter_element_count"], int)
                or isinstance(summary["parameter_element_count"], bool)
                or summary["parameter_element_count"] <= 0
                or not _is_sha256(summary["parameter_names_sha256"])
                or summary["all_present_finite"] is not True):
            raise ValueError("selected loop diagnostic inventory is invalid")
        tensor_values = _validate_min_max_last(
            summary["gradient_tensor_count"],
            "{} gradient tensors".format(group_name), integer=True,
        )
        element_values = _validate_min_max_last(
            summary["gradient_element_count"],
            "{} gradient elements".format(group_name), integer=True,
        )
        _validate_min_max_last(
            summary["preclip_gradient_norm"],
            "{} pre-clip gradient norm".format(group_name),
        )
        if (tensor_values[1] > summary["parameter_tensor_count"]
                or element_values[1] > summary["parameter_element_count"]):
            raise ValueError("selected loop gradient coverage is impossible")
        expected_clip = next(
            spec[3] for spec in _DIAGNOSTIC_GROUP_SPECS
            if spec[0] == group_name
        )
        if summary["clip_limit"] != expected_clip:
            raise ValueError("selected loop gradient clip differs")
        if live_inventory is not None:
            live = live_inventory[group_name]
            for field in (
                    "parameter_tensor_count", "parameter_element_count",
                    "parameter_names_sha256", "clip_limit"):
                if summary[field] != live[field]:
                    raise ValueError(
                        "selected loop diagnostic inventory differs from live"
                    )

    frozen = diagnostics["frozen_mcln"]
    if (not isinstance(frozen, dict)
            or set(frozen) != {
                "parameter_tensor_count", "parameter_element_count",
                "parameter_names_sha256",
            }
            or not isinstance(frozen["parameter_tensor_count"], int)
            or isinstance(frozen["parameter_tensor_count"], bool)
            or frozen["parameter_tensor_count"] <= 0
            or not isinstance(frozen["parameter_element_count"], int)
            or isinstance(frozen["parameter_element_count"], bool)
            or frozen["parameter_element_count"] <= 0
            or not _is_sha256(frozen["parameter_names_sha256"])):
        raise ValueError("selected loop frozen MCLN inventory is invalid")
    if live_frozen is not None:
        for field in (
                "parameter_tensor_count", "parameter_element_count",
                "parameter_names_sha256"):
            if frozen[field] != live_frozen[field]:
                raise ValueError(
                    "selected loop frozen inventory differs from live"
                )

    last = diagnostics["last_update"]
    if (not isinstance(last, dict)
            or set(last) != {
                "schema", "step", "losses", "ranking_objectives",
                "groups", "frozen_mcln",
            }
            or last.get("schema")
            != "rec-finetune-update-diagnostics-v2"
            or last.get("step") != expected_updates
            or last.get("losses") != {
                name: losses[name]["last"]
                for name in _DIAGNOSTIC_LOSS_NAMES
            }
            or not isinstance(last.get("groups"), dict)
            or set(last["groups"]) != expected_group_names):
        raise ValueError("selected loop last update diagnostics are invalid")
    last_objectives = _validated_reranker_objective_record(
        last.get("ranking_objectives"), last["losses"]
    )
    for name, stats in last_objectives.items():
        summary = objective_summaries[name]
        for field in _DIAGNOSTIC_RERANKER_LOSS_NAMES:
            if stats[field] != summary[field]["last"]:
                raise ValueError("selected loop last ranking loss differs")
        for field in _DIAGNOSTIC_RERANKER_COUNT_NAMES:
            if stats[field] != summary[field]["last"]:
                raise ValueError("selected loop last ranking count differs")
    for group_name, update in last["groups"].items():
        summary = group_summaries[group_name]
        if (not isinstance(update, dict)
                or set(update) != {
                    "parameter_tensor_count", "parameter_element_count",
                    "parameter_names_sha256", "gradient_tensor_count",
                    "gradient_element_count", "finite_gradient_tensor_count",
                    "finite_gradient_element_count", "all_present_finite",
                    "preclip_gradient_norm", "clip_limit",
                }
                or update["parameter_tensor_count"]
                != summary["parameter_tensor_count"]
                or update["parameter_element_count"]
                != summary["parameter_element_count"]
                or update["parameter_names_sha256"]
                != summary["parameter_names_sha256"]
                or update["gradient_tensor_count"]
                != summary["gradient_tensor_count"]["last"]
                or update["gradient_element_count"]
                != summary["gradient_element_count"]["last"]
                or update["finite_gradient_tensor_count"]
                != update["gradient_tensor_count"]
                or update["finite_gradient_element_count"]
                != update["gradient_element_count"]
                or update["all_present_finite"] is not True
                or update["preclip_gradient_norm"]
                != summary["preclip_gradient_norm"]["last"]
                or update["clip_limit"] != summary["clip_limit"]):
            raise ValueError("selected loop last gradient diagnostics are invalid")
    last_frozen = last["frozen_mcln"]
    if (not isinstance(last_frozen, dict)
            or set(last_frozen) != {
                "parameter_tensor_count", "parameter_element_count",
                "parameter_names_sha256", "gradient_tensor_count",
                "gradient_element_count",
            }
            or any(last_frozen[field] != frozen[field] for field in (
                "parameter_tensor_count", "parameter_element_count",
                "parameter_names_sha256",
            ))
            or last_frozen["gradient_tensor_count"] != 0
            or last_frozen["gradient_element_count"] != 0):
        raise ValueError("selected loop frozen gradient diagnostics are invalid")
    try:
        json.dumps(diagnostics, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "selected loop diagnostics are not JSON-safe: {}".format(error)
        )


def _validate_runtime_provenance(runtime):
    if (not isinstance(runtime, dict)
            or set(runtime) != {
                "schema", "started_utc", "finished_utc", "elapsed_seconds",
                "completed_successfully", "oom_detected", "command",
                "interpreter", "versions", "device", "peak_cuda_memory",
                "environment",
            }
            or runtime.get("schema") != "rec-finetune-runtime-v1"
            or runtime.get("completed_successfully") is not True
            or runtime.get("oom_detected") is not False):
        raise ValueError("selected loop runtime provenance is invalid")
    _require_utc_timestamp("runtime start", runtime["started_utc"])
    _require_utc_timestamp("runtime finish", runtime["finished_utc"])
    if (_utc_timestamp_datetime(runtime["finished_utc"])
            < _utc_timestamp_datetime(runtime["started_utc"])):
        raise ValueError("runtime finish timestamp precedes runtime start")
    elapsed = _require_finite_primitive(
        "runtime elapsed seconds", runtime["elapsed_seconds"]
    )
    if elapsed < 0.0:
        raise ValueError("runtime elapsed seconds must be nonnegative")
    if (not isinstance(runtime["command"], list)
            or not runtime["command"]
            or any(not isinstance(value, str) or not value.strip()
                   for value in runtime["command"])):
        raise ValueError("selected loop runtime command is invalid")
    interpreter = runtime["interpreter"]
    versions = runtime["versions"]
    device = runtime["device"]
    peak = runtime["peak_cuda_memory"]
    environment = runtime["environment"]
    if (not isinstance(interpreter, dict)
            or set(interpreter) != {"logical_path", "resolved_path"}
            or any(not isinstance(value, str) or not value.strip()
                   for value in interpreter.values())
            or not isinstance(versions, dict)
            or set(versions) != {"python", "torch", "cuda", "cudnn"}
            or any(not isinstance(versions[name], str)
                   or not versions[name].strip()
                   for name in ("python", "torch", "cuda"))
            or not isinstance(versions["cudnn"], int)
            or isinstance(versions["cudnn"], bool)
            or versions["cudnn"] <= 0):
        raise ValueError("selected loop runtime identity is invalid")
    logical_interpreter = interpreter["logical_path"]
    resolved_interpreter = interpreter["resolved_path"]
    if runtime["command"][0] != logical_interpreter:
        raise ValueError("runtime command interpreter is incoherent")
    try:
        actual_resolved = Path(logical_interpreter).resolve(strict=True)
        recorded_resolved = Path(resolved_interpreter)
        if (not recorded_resolved.is_absolute()
                or recorded_resolved != actual_resolved
                or recorded_resolved.resolve(strict=True) != actual_resolved):
            raise ValueError("runtime interpreter paths are incoherent")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "runtime interpreter paths are incoherent: {}".format(error)
        )
    if (not isinstance(device, dict)
            or set(device) != {
                "type", "index", "name", "total_memory_bytes",
            }
            or device.get("type") != "cuda" or device.get("index") != 0
            or not isinstance(device.get("name"), str)
            or not device["name"].strip()
            or not isinstance(device.get("total_memory_bytes"), int)
            or isinstance(device.get("total_memory_bytes"), bool)
            or device["total_memory_bytes"] <= 0
            or not isinstance(peak, dict)
            or set(peak) != {"allocated_bytes", "reserved_bytes"}
            or any(not isinstance(peak[name], int)
                   or isinstance(peak[name], bool) or peak[name] < 0
                   for name in peak)):
        raise ValueError("selected loop CUDA runtime provenance is invalid")
    if (not isinstance(environment, dict)
            or set(environment) != set(RUNTIME_ENVIRONMENT_ALLOWLIST)
            or any(value is not None and not isinstance(value, str)
                   for value in environment.values())):
        raise ValueError("selected loop runtime environment is invalid")
    expected_interpreter = {
        "logical_path": str(sys.executable),
        "resolved_path": str(Path(sys.executable).resolve(strict=True)),
    }
    live_cuda_version = torch.version.cuda
    live_cudnn_version = torch.backends.cudnn.version()
    if (not isinstance(live_cuda_version, str)
            or not live_cuda_version.strip()
            or not isinstance(live_cudnn_version, int)
            or isinstance(live_cudnn_version, bool)
            or live_cudnn_version <= 0):
        raise ValueError("live runtime versions are invalid")
    expected_versions = {
        "python": str(platform.python_version()),
        "torch": str(torch.__version__),
        "cuda": live_cuda_version,
        "cudnn": live_cudnn_version,
    }
    expected_environment = {
        name: os.environ.get(name)
        for name in RUNTIME_ENVIRONMENT_ALLOWLIST
    }
    live_cuda = _runtime_cuda_snapshot()
    if (not isinstance(live_cuda, dict)
            or set(live_cuda) != {"device", "peak_cuda_memory"}
            or not isinstance(live_cuda["device"], dict)
            or not isinstance(live_cuda["peak_cuda_memory"], dict)):
        raise ValueError("live CUDA runtime snapshot is invalid")
    if (interpreter != expected_interpreter
            or runtime["command"][0]
            != expected_interpreter["logical_path"]):
        raise ValueError("runtime interpreter differs from live identity")
    if versions != expected_versions:
        raise ValueError("runtime versions differ from live identity")
    if device != live_cuda["device"]:
        raise ValueError("runtime device differs from live CUDA device")
    if peak != live_cuda["peak_cuda_memory"]:
        raise ValueError("runtime peak differs from live CUDA peak")
    if environment != expected_environment:
        raise ValueError("runtime environment differs from live environment")
    try:
        json.dumps(runtime, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "selected loop runtime is not JSON-safe: {}".format(error)
        )


def _validate_train_data_contract(contract, initialized_contract=None):
    expected_fields = {
        "schema", "dataset_split", "datasets", "joint_det", "butd",
        "butd_gt", "butd_cls", "fit_augment", "fit_augment_det",
        "calibration_augment", "calibration_augment_det",
        "authoritative_split_metadata",
        "authoritative_split_mapping_sha256", "fit_sample_count",
        "calibration_sample_count", "fit_loader_batch_count",
        "calibration_loader_batch_count", "batch_size", "drop_last",
        "validation_data_accessed", "dataset_class",
        "dataset_instance_count",
        "fit_and_calibration_share_source_annotations",
        "validation_data_objects_present",
    }
    if (not isinstance(contract, dict) or set(contract) != expected_fields
            or contract.get("schema")
            != "scanrefer-rec-finetune-train-data-v1"
            or contract.get("dataset_split") != "train"
            or contract.get("datasets") != ["scanrefer"]
            or contract.get("joint_det") is not False
            or contract.get("butd") is not True
            or contract.get("butd_gt") is not False
            or contract.get("butd_cls") is not False
            or contract.get("fit_augment") is not True
            or contract.get("fit_augment_det") is not True
            or contract.get("calibration_augment") is not False
            or contract.get("calibration_augment_det") is not False
            or contract.get("batch_size") != PRODUCTION_BATCH_SIZE
            or contract.get("drop_last") is not False
            or contract.get("validation_data_accessed") is not False
            or not isinstance(contract.get("dataset_class"), str)
            or not contract["dataset_class"]
            or contract.get("dataset_instance_count") != 1
            or contract.get(
                "fit_and_calibration_share_source_annotations") is not True
            or contract.get("validation_data_objects_present") is not False):
        raise ValueError("selected loop train-data contract is invalid")
    metadata = contract["authoritative_split_metadata"]
    if (metadata != rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
            or contract["authoritative_split_mapping_sha256"]
            != metadata["mapping_sha256"]
            or contract["fit_sample_count"] != metadata["fit_sample_count"]
            or contract["calibration_sample_count"]
            != metadata["calibration_sample_count"]
            or contract["fit_loader_batch_count"]
            != rec_finetune.natural_batch_count(
                contract["fit_sample_count"], PRODUCTION_BATCH_SIZE
            )
            or contract["calibration_loader_batch_count"]
            != rec_finetune.natural_batch_count(
                contract["calibration_sample_count"], PRODUCTION_BATCH_SIZE
            )):
        raise ValueError("selected loop train-data split is not authoritative")
    if initialized_contract is not None and contract != initialized_contract:
        raise ValueError("selected loop train-data contract changed")
    try:
        json.dumps(contract, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "selected loop train-data contract is not JSON-safe: {}".format(
                error
            )
        )


def _validate_initialized_train_data_binding(initialized, contract):
    if not isinstance(initialized, dict):
        raise ValueError("initialized train-data binding is invalid")
    data = initialized.get("data")
    if not isinstance(data, dict):
        raise ValueError("initialized train-data binding is invalid")
    full_data_keys = {
        "dataset", "split", "fit_view", "calibration_view",
        "fit_loader", "calibration_loader",
    }
    if set(data) != full_data_keys:
        raise ValueError(
            "initialized train-data keys must match the exact live contract"
        )
    _validate_train_data_contract(
        contract, initialized.get("train_data_contract")
    )
    split = data.get("split")
    if (not isinstance(split, dict)
            or split.get("metadata")
            != contract["authoritative_split_metadata"]):
        raise ValueError("initialized train-data split differs from contract")
    rebuilt = build_rec_finetune_train_data_contract(initialized)
    if rebuilt != contract:
        raise ValueError("live train-data contract changed")


_CALIBRATION_DIAGNOSTIC_TOP_FIELDS = frozenset((
    "schema", "sample_count", "candidate_oracle", "stages", "effects",
    "selected_iou", "geometry_oracle_selected_regret",
    "recoverable_misses", "selected_oracle_regret_cells",
))
_CALIBRATION_DIAGNOSTIC_THRESHOLD_FIELDS = frozenset((
    "hits025", "hits050", "acc025", "acc050",
))
_CALIBRATION_DIAGNOSTIC_BIN_FIELDS = tuple(
    spec[0] for spec in rec_finetune.CALIBRATION_SELECTED_IOU_BIN_SPECS
)
_CALIBRATION_DIAGNOSTIC_MAX_JSON_BYTES = 256 * 1024


def _diagnostic_finite_number(value, label, minimum, maximum):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum):
        raise ValueError("calibration diagnostic {} is invalid".format(label))
    return float(value)


def _diagnostic_count(value, sample_count, label):
    if (not isinstance(value, int) or isinstance(value, bool)
            or not 0 <= value <= sample_count):
        raise ValueError("calibration diagnostic {} is invalid".format(label))
    return value


def _diagnostic_nested_transition_is_feasible(
        previous_middle, lower_upward, lower_downward,
        upper_upward, upper_downward):
    return (
        min(lower_upward, upper_upward)
        + min(lower_downward, upper_downward)
        >= lower_downward + upper_upward - previous_middle
    )


def _diagnostic_three_bin_transition_is_feasible(
        previous_metrics, upward025, downward025, upward050, downward050):
    return _diagnostic_nested_transition_is_feasible(
        previous_metrics["hits025"] - previous_metrics["hits050"],
        upward025, downward025, upward050, downward050,
    )


def _reject_private_calibration_diagnostic_values(value, ancestors=None):
    if isinstance(value, torch.Tensor):
        raise ValueError("calibration diagnostics cannot contain tensors")
    if ancestors is None:
        ancestors = set()
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("calibration diagnostics cannot contain cycles")
        ancestors.add(identity)
        try:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"selected_ious", "geometry_oracle_ious"}:
                        raise ValueError(
                            "calibration diagnostics contain per-sample IoUs"
                        )
                    _reject_private_calibration_diagnostic_values(
                        item, ancestors
                    )
            else:
                for item in value:
                    _reject_private_calibration_diagnostic_values(
                        item, ancestors
                    )
        finally:
            ancestors.remove(identity)


def _validate_calibration_diagnostic_json(payload):
    try:
        _reject_private_calibration_diagnostic_values(payload)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError(
            "calibration diagnostics are not finite JSON: {}".format(error)
        )
    if len(encoded) > _CALIBRATION_DIAGNOSTIC_MAX_JSON_BYTES:
        raise ValueError("calibration diagnostics JSON is unreasonably large")


def _validate_diagnostic_threshold_metrics(value, sample_count, label):
    if (not isinstance(value, dict)
            or set(value) != _CALIBRATION_DIAGNOSTIC_THRESHOLD_FIELDS):
        raise ValueError(
            "calibration diagnostic {} fields do not match schema".format(
                label
            )
        )
    hits025 = _diagnostic_count(
        value["hits025"], sample_count, label + " hits025"
    )
    hits050 = _diagnostic_count(
        value["hits050"], sample_count, label + " hits050"
    )
    acc025 = _diagnostic_finite_number(
        value["acc025"], label + " acc025", 0.0, 1.0
    )
    acc050 = _diagnostic_finite_number(
        value["acc050"], label + " acc050", 0.0, 1.0
    )
    if (hits050 > hits025
            or not math.isclose(
                acc025, hits025 / float(sample_count),
                rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                acc050, hits050 / float(sample_count),
                rel_tol=0.0, abs_tol=1e-12,
            )):
        raise ValueError(
            "calibration diagnostic {} is internally inconsistent".format(
                label
            )
        )
    return {
        "hits025": hits025,
        "hits050": hits050,
        "acc025": acc025,
        "acc050": acc050,
    }


def _intersect_diagnostic_intervals(first, second):
    first_lower, first_upper, first_lower_closed, first_upper_closed = first
    second_lower, second_upper, second_lower_closed, second_upper_closed = second
    lower = max(first_lower, second_lower)
    upper = min(first_upper, second_upper)
    lower_closed = (
        (first_lower != lower or first_lower_closed)
        and (second_lower != lower or second_lower_closed)
    )
    upper_closed = (
        (first_upper != upper or first_upper_closed)
        and (second_upper != upper or second_upper_closed)
    )
    return lower, upper, lower_closed, upper_closed


def _diagnostic_endpoint_percent_units(value):
    scaled = Fraction(str(value)) * 100
    if scaled.denominator != 1:
        raise ValueError(
            "calibration diagnostic interval endpoint is not percent-scaled"
        )
    return scaled.numerator


def _validate_selected_oracle_regret_cells(
        value, sample_count, selected_bins, geometry_oracle_metrics,
        positive_count, ge005_count, ge010_count, recoverable_counts):
    selected_specs = rec_finetune.CALIBRATION_SELECTED_IOU_BIN_SPECS
    oracle_specs = rec_finetune.CALIBRATION_ORACLE_TIER_SPECS
    regret_specs = rec_finetune.CALIBRATION_REGRET_BAND_SPECS
    selected_names = tuple(spec[0] for spec in selected_specs)
    oracle_names = tuple(spec[0] for spec in oracle_specs)
    regret_names = tuple(spec[0] for spec in regret_specs)
    if not isinstance(value, dict) or set(value) != set(selected_names):
        raise ValueError("calibration diagnostic regret cell schema is invalid")

    selected_totals = {name: 0 for name in selected_names}
    oracle_totals = {name: 0 for name in oracle_names}
    regret_totals = {name: 0 for name in regret_names}
    derived_recoverable = {"025": 0, "050": 0}
    total_count = 0
    selected_spec_by_name = {spec[0]: spec for spec in selected_specs}
    oracle_spec_by_name = {spec[0]: spec for spec in oracle_specs}
    regret_spec_by_name = {spec[0]: spec for spec in regret_specs}

    for selected_name in selected_names:
        selected_row = value[selected_name]
        if (not isinstance(selected_row, dict)
                or set(selected_row) != set(oracle_names)):
            raise ValueError(
                "calibration diagnostic regret cell selected row is invalid"
            )
        _name, selected_lower, selected_upper, \
            selected_lower_closed, selected_tier = (
                selected_spec_by_name[selected_name]
            )
        selected_lower = _diagnostic_endpoint_percent_units(selected_lower)
        selected_upper = _diagnostic_endpoint_percent_units(selected_upper)
        for oracle_name in oracle_names:
            oracle_row = selected_row[oracle_name]
            if (not isinstance(oracle_row, dict)
                    or set(oracle_row) != set(regret_names)):
                raise ValueError(
                    "calibration diagnostic regret cell oracle row is invalid"
                )
            _name, oracle_lower, oracle_upper, \
                oracle_lower_closed, oracle_tier = (
                    oracle_spec_by_name[oracle_name]
                )
            oracle_lower = _diagnostic_endpoint_percent_units(oracle_lower)
            oracle_upper = _diagnostic_endpoint_percent_units(oracle_upper)
            gap_lower = max(0, oracle_lower - selected_upper)
            gap_lower_closed = (
                selected_tier == oracle_tier
                if gap_lower == 0 else oracle_lower_closed
            )
            gap_upper = max(0, oracle_upper - selected_lower)
            gap_upper_closed = selected_lower_closed
            gap_interval = (
                gap_lower, gap_upper,
                gap_lower_closed, gap_upper_closed,
            )
            for regret_name in regret_names:
                cell = oracle_row[regret_name]
                if (not isinstance(cell, dict)
                        or set(cell) != {"count"}):
                    raise ValueError(
                        "calibration diagnostic regret cell fields are invalid"
                    )
                count = _diagnostic_count(
                    cell["count"], sample_count,
                    "regret cell count",
                )
                _name, band_lower, band_upper, band_lower_closed, \
                    band_upper_closed = regret_spec_by_name[regret_name]
                band_lower = _diagnostic_endpoint_percent_units(band_lower)
                band_upper = _diagnostic_endpoint_percent_units(band_upper)
                interval = _intersect_diagnostic_intervals(
                    gap_interval,
                    (
                        band_lower, band_upper,
                        band_lower_closed, band_upper_closed,
                    ),
                )
                lower, upper, lower_closed, upper_closed = interval
                interval_nonempty = (
                    lower < upper
                    or (lower == upper and lower_closed and upper_closed)
                )
                if count > 0 and (
                        selected_tier > oracle_tier
                        or not interval_nonempty):
                    raise ValueError(
                        "calibration diagnostic regret cell is impossible"
                    )
                selected_totals[selected_name] += count
                oracle_totals[oracle_name] += count
                regret_totals[regret_name] += count
                total_count += count
                if selected_tier == 0 and oracle_tier >= 1:
                    derived_recoverable["025"] += count
                if selected_tier <= 1 and oracle_tier == 2:
                    derived_recoverable["050"] += count

    expected_oracle_totals = {
        "o0": sample_count - geometry_oracle_metrics["hits025"],
        "o1": (
            geometry_oracle_metrics["hits025"]
            - geometry_oracle_metrics["hits050"]
        ),
        "o2": geometry_oracle_metrics["hits050"],
    }
    expected_regret_totals = {
        "zero": sample_count - positive_count,
        "gt_000_lt_005": positive_count - ge005_count,
        "ge_005_lt_010": ge005_count - ge010_count,
        "ge_010": ge010_count,
    }
    if (total_count != sample_count
            or selected_totals != selected_bins
            or oracle_totals != expected_oracle_totals
            or regret_totals != expected_regret_totals
            or derived_recoverable != recoverable_counts):
        raise ValueError(
            "calibration diagnostic regret cells differ from aggregates"
        )


def _validate_one_calibration_diagnostic(value, expected_sample_count, label):
    if (not isinstance(value, dict)
            or set(value) != _CALIBRATION_DIAGNOSTIC_TOP_FIELDS
            or value.get("schema")
            != "rec-finetune-calibration-diagnostics-v3"):
        raise ValueError(
            "calibration diagnostic {} fields do not match exact schema".format(
                label
            )
        )
    sample_count = value.get("sample_count")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count <= 0
            or sample_count != expected_sample_count):
        raise ValueError(
            "calibration diagnostic {} sample_count is invalid".format(label)
        )

    oracle_value = value.get("candidate_oracle")
    oracle_names = {
        "raw_query", "parent_candidate", "geometry_candidate",
    }
    if not isinstance(oracle_value, dict) or set(oracle_value) != oracle_names:
        raise ValueError("calibration diagnostic candidate oracle schema is invalid")
    candidate_oracle = {
        name: _validate_diagnostic_threshold_metrics(
            oracle_value[name], sample_count,
            "{} candidate_oracle {}".format(label, name),
        )
        for name in sorted(oracle_names)
    }

    stages_value = value.get("stages")
    stage_names = {
        "default_top1", "source_selector_top1", "parent_top1",
        "geometry_top1",
    }
    if not isinstance(stages_value, dict) or set(stages_value) != stage_names:
        raise ValueError("calibration diagnostic stage schema is invalid")
    stages = {
        name: _validate_diagnostic_threshold_metrics(
            stages_value[name], sample_count,
            "{} stage {}".format(label, name),
        )
        for name in sorted(stage_names)
    }

    oracle_requirements = {
        "raw_query": (
            "default_top1", "source_selector_top1", "parent_top1",
        ),
        "parent_candidate": ("default_top1", "parent_top1"),
        "geometry_candidate": (
            "default_top1", "parent_top1", "geometry_top1",
        ),
    }
    for oracle_name, required_stages in oracle_requirements.items():
        for stage_name in required_stages:
            for suffix in ("025", "050"):
                if (candidate_oracle[oracle_name]["hits" + suffix]
                        < stages[stage_name]["hits" + suffix]):
                    raise ValueError(
                        "calibration diagnostic oracle aggregate is lower than stage"
                    )
    parent_oracle = candidate_oracle["parent_candidate"]
    for outer_oracle_name in ("raw_query", "geometry_candidate"):
        outer_oracle = candidate_oracle[outer_oracle_name]
        if any(
                outer_oracle["hits" + suffix]
                < parent_oracle["hits" + suffix]
                for suffix in ("025", "050")):
            raise ValueError(
                "calibration diagnostic parent oracle nesting is invalid"
            )

    selected_iou = value.get("selected_iou")
    if (not isinstance(selected_iou, dict)
            or set(selected_iou) != {"bins"}):
        raise ValueError("calibration diagnostic selected IoU schema is invalid")
    bins_value = selected_iou.get("bins")
    if (not isinstance(bins_value, dict)
            or set(bins_value) != set(_CALIBRATION_DIAGNOSTIC_BIN_FIELDS)):
        raise ValueError("calibration diagnostic selected IoU bins are invalid")
    bins = {
        name: _diagnostic_count(
            bins_value[name], sample_count, "selected IoU bin " + name
        )
        for name in _CALIBRATION_DIAGNOSTIC_BIN_FIELDS
    }
    if sum(bins.values()) != sample_count:
        raise ValueError("calibration diagnostic selected IoU bins do not sum")
    selected_hits025 = sum(
        bins[name] for name in _CALIBRATION_DIAGNOSTIC_BIN_FIELDS[3:]
    )
    selected_hits050 = sum(
        bins[name] for name in _CALIBRATION_DIAGNOSTIC_BIN_FIELDS[5:]
    )
    if (selected_hits025 != stages["geometry_top1"]["hits025"]
            or selected_hits050 != stages["geometry_top1"]["hits050"]):
        raise ValueError(
            "calibration diagnostic selected IoU bins differ from geometry hits"
        )

    effects_value = value.get("effects")
    effect_specs = {
        "source_selector_vs_default": (
            "default_top1", "source_selector_top1",
        ),
        "parent_vs_default": ("default_top1", "parent_top1"),
        "geometry_vs_parent": ("parent_top1", "geometry_top1"),
    }
    effect_fields = {
        "fixes025", "breaks025", "fixes050", "breaks050",
    }
    if (not isinstance(effects_value, dict)
            or set(effects_value) != set(effect_specs)):
        raise ValueError("calibration diagnostic effect schema is invalid")
    for effect_name, (old_name, new_name) in effect_specs.items():
        effect = effects_value[effect_name]
        if not isinstance(effect, dict) or set(effect) != effect_fields:
            raise ValueError("calibration diagnostic effect fields are invalid")
        for suffix in ("025", "050"):
            fixes = _diagnostic_count(
                effect["fixes" + suffix], sample_count,
                effect_name + " fixes" + suffix,
            )
            breaks = _diagnostic_count(
                effect["breaks" + suffix], sample_count,
                effect_name + " breaks" + suffix,
            )
            old_hits = stages[old_name]["hits" + suffix]
            new_hits = stages[new_name]["hits" + suffix]
            if (new_hits - old_hits != fixes - breaks
                    or fixes > min(sample_count - old_hits, new_hits)
                    or breaks > min(old_hits, sample_count - new_hits)):
                raise ValueError(
                    "calibration diagnostic effect counts are inconsistent"
                )
        if not _diagnostic_three_bin_transition_is_feasible(
                stages[old_name],
                effect["fixes025"], effect["breaks025"],
                effect["fixes050"], effect["breaks050"]):
            raise ValueError(
                "calibration diagnostic effect thresholds are jointly impossible"
            )
    oracle_union_requirements = {
        "raw_query": ("source_selector_vs_default", "parent_vs_default"),
        "parent_candidate": ("parent_vs_default",),
        "geometry_candidate": ("parent_vs_default", "geometry_vs_parent"),
    }
    for oracle_name, effect_names in oracle_union_requirements.items():
        oracle = candidate_oracle[oracle_name]
        for effect_name in effect_names:
            old_name, _new_name = effect_specs[effect_name]
            effect = effects_value[effect_name]
            for suffix in ("025", "050"):
                stage_union = (
                    stages[old_name]["hits" + suffix]
                    + effect["fixes" + suffix]
                )
                if oracle["hits" + suffix] < stage_union:
                    raise ValueError(
                        "calibration diagnostic oracle stage union is invalid"
                    )

    regret = value.get("geometry_oracle_selected_regret")
    regret_fields = {
        "positive_count", "ge005_count", "ge010_count",
    }
    if not isinstance(regret, dict) or set(regret) != regret_fields:
        raise ValueError("calibration diagnostic regret schema is invalid")
    positive_count = _diagnostic_count(
        regret["positive_count"], sample_count, "regret positive_count"
    )
    ge005_count = _diagnostic_count(
        regret["ge005_count"], sample_count, "regret ge005_count"
    )
    ge010_count = _diagnostic_count(
        regret["ge010_count"], sample_count, "regret ge010_count"
    )
    if not 0 <= ge010_count <= ge005_count <= positive_count:
        raise ValueError("calibration diagnostic regret is inconsistent")

    recoverable = value.get("recoverable_misses")
    if (not isinstance(recoverable, dict)
            or set(recoverable) != {"at025", "at050"}):
        raise ValueError("calibration diagnostic recoverable schema is invalid")
    recoverable_counts = {}
    for suffix in ("025", "050"):
        count = _diagnostic_count(
            recoverable["at" + suffix], sample_count,
            "recoverable at" + suffix,
        )
        recoverable_counts[suffix] = count
        expected = (
            candidate_oracle["geometry_candidate"]["hits" + suffix]
            - stages["geometry_top1"]["hits" + suffix]
        )
        if count != expected:
            raise ValueError(
                "calibration diagnostic recoverable misses are inconsistent"
            )
    if max(recoverable_counts.values()) > positive_count:
        raise ValueError(
            "calibration diagnostic recoverable misses exceed positive regret"
        )
    _validate_selected_oracle_regret_cells(
        value.get("selected_oracle_regret_cells"),
        sample_count,
        bins,
        candidate_oracle["geometry_candidate"],
        positive_count,
        ge005_count,
        ge010_count,
        recoverable_counts,
    )
    return {
        "sample_count": sample_count,
        "candidate_oracle": candidate_oracle,
        "stages": stages,
    }


def _validate_calibration_joint_transition_witness(
        value, transition_value, sample_count,
        previous_diagnostic, current_diagnostic):
    state_specs = rec_finetune.CALIBRATION_JOINT_STATE_TIERS
    state_names = tuple(name for name, _selected, _oracle in state_specs)
    state_tiers = {
        name: (selected, oracle)
        for name, selected, oracle in state_specs
    }
    if (not isinstance(value, dict)
            or set(value) != set(state_names)):
        raise ValueError(
            "calibration diagnostic joint transition schema is invalid"
        )
    counts = {}
    total = 0
    for previous_name in state_names:
        row = value[previous_name]
        if not isinstance(row, dict) or set(row) != set(state_names):
            raise ValueError(
                "calibration diagnostic joint transition row is invalid"
            )
        counts[previous_name] = {}
        for current_name in state_names:
            count = _diagnostic_count(
                row[current_name], sample_count,
                "joint transition {} to {}".format(
                    previous_name, current_name
                ),
            )
            counts[previous_name][current_name] = count
            total += count
    if total != sample_count:
        raise ValueError(
            "calibration diagnostic joint transition count is invalid"
        )

    sources = {
        "selected": ("stages", "geometry_top1", 0),
        "geometry_oracle": (
            "candidate_oracle", "geometry_candidate", 1,
        ),
    }
    threshold_specs = (("025", 1), ("050", 2))
    for source_name, (section_name, metric_name, tier_index) in sources.items():
        previous_metrics = previous_diagnostic[section_name][metric_name]
        current_metrics = current_diagnostic[section_name][metric_name]
        derived_previous = {suffix: 0 for suffix, _tier in threshold_specs}
        derived_current = {suffix: 0 for suffix, _tier in threshold_specs}
        derived_transition = {
            "gained" + suffix: 0
            for suffix, _tier in threshold_specs
        }
        derived_transition.update({
            "lost" + suffix: 0
            for suffix, _tier in threshold_specs
        })
        for previous_name in state_names:
            previous_tier = state_tiers[previous_name][tier_index]
            row_total = sum(counts[previous_name].values())
            for suffix, minimum_tier in threshold_specs:
                if previous_tier >= minimum_tier:
                    derived_previous[suffix] += row_total
            for current_name in state_names:
                count = counts[previous_name][current_name]
                current_tier = state_tiers[current_name][tier_index]
                for suffix, minimum_tier in threshold_specs:
                    previous_hit = previous_tier >= minimum_tier
                    current_hit = current_tier >= minimum_tier
                    if current_hit:
                        derived_current[suffix] += count
                    if not previous_hit and current_hit:
                        derived_transition["gained" + suffix] += count
                    elif previous_hit and not current_hit:
                        derived_transition["lost" + suffix] += count
        if any(
                derived_previous[suffix]
                != previous_metrics["hits" + suffix]
                or derived_current[suffix]
                != current_metrics["hits" + suffix]
                for suffix, _tier in threshold_specs):
            raise ValueError(
                "calibration diagnostic joint transition marginals differ"
            )
        if derived_transition != transition_value[source_name]:
            raise ValueError(
                "calibration diagnostic joint transition effects differ"
            )


def _validate_calibration_diagnostic_transition(
        value, previous_step, current_step, sample_count,
        previous_diagnostic, current_diagnostic):
    fields = {
        "schema", "previous_step", "current_step", "sample_count",
        "selected", "geometry_oracle", "selected_oracle_joint",
    }
    transition_previous_step = (
        value.get("previous_step") if isinstance(value, dict) else None
    )
    transition_current_step = (
        value.get("current_step") if isinstance(value, dict) else None
    )
    transition_sample_count = (
        value.get("sample_count") if isinstance(value, dict) else None
    )
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("schema")
            != "rec-finetune-calibration-step-transition-v2"
            or not isinstance(transition_previous_step, int)
            or isinstance(transition_previous_step, bool)
            or transition_previous_step != previous_step
            or not isinstance(transition_current_step, int)
            or isinstance(transition_current_step, bool)
            or transition_current_step != current_step
            or not isinstance(transition_sample_count, int)
            or isinstance(transition_sample_count, bool)
            or transition_sample_count != sample_count):
        raise ValueError(
            "calibration diagnostic transition fields are invalid"
        )
    transition_fields = {
        "gained025", "lost025", "gained050", "lost050",
    }
    sources = {
        "selected": ("stages", "geometry_top1"),
        "geometry_oracle": ("candidate_oracle", "geometry_candidate"),
    }
    for name, (section_name, metric_name) in sources.items():
        transition = value.get(name)
        if (not isinstance(transition, dict)
                or set(transition) != transition_fields):
            raise ValueError(
                "calibration diagnostic transition count schema is invalid"
            )
        previous_metrics = previous_diagnostic[section_name][metric_name]
        current_metrics = current_diagnostic[section_name][metric_name]
        for suffix in ("025", "050"):
            gained = _diagnostic_count(
                transition["gained" + suffix], sample_count,
                name + " transition gained" + suffix,
            )
            lost = _diagnostic_count(
                transition["lost" + suffix], sample_count,
                name + " transition lost" + suffix,
            )
            previous_hits = previous_metrics["hits" + suffix]
            current_hits = current_metrics["hits" + suffix]
            if (current_hits - previous_hits != gained - lost
                    or gained + lost > sample_count
                    or gained > min(sample_count - previous_hits, current_hits)
                    or lost > min(previous_hits, sample_count - current_hits)):
                raise ValueError(
                    "calibration diagnostic transition counts are inconsistent"
                )
        if not _diagnostic_three_bin_transition_is_feasible(
                previous_metrics,
                transition["gained025"], transition["lost025"],
                transition["gained050"], transition["lost050"]):
            raise ValueError(
                "calibration diagnostic transition thresholds are jointly impossible"
            )
    previous_selected = previous_diagnostic["stages"]["geometry_top1"]
    previous_oracle = previous_diagnostic[
        "candidate_oracle"
    ]["geometry_candidate"]
    for suffix in ("025", "050"):
        if not _diagnostic_nested_transition_is_feasible(
                previous_oracle["hits" + suffix]
                - previous_selected["hits" + suffix],
                value["geometry_oracle"]["gained" + suffix],
                value["geometry_oracle"]["lost" + suffix],
                value["selected"]["gained" + suffix],
                value["selected"]["lost" + suffix]):
            raise ValueError(
                "calibration diagnostic selected/oracle transition is impossible"
            )
    _validate_calibration_joint_transition_witness(
        value["selected_oracle_joint"], value, sample_count,
        previous_diagnostic, current_diagnostic,
    )


def _validate_calibration_diagnostics_history(run_result, calibration_history):
    payload = {
        "calibration_diagnostics_history": run_result.get(
            "calibration_diagnostics_history"
        ),
        "selected_calibration_diagnostics": run_result.get(
            "selected_calibration_diagnostics"
        ),
        "reproduced_calibration_diagnostics": run_result.get(
            "reproduced_calibration_diagnostics"
        ),
    }
    _validate_calibration_diagnostic_json(payload)
    history = payload["calibration_diagnostics_history"]
    if (not isinstance(history, list) or not history
            or len(history) != len(calibration_history)):
        raise ValueError("calibration diagnostic history length is invalid")
    normalized = []
    for index, (record, selection_record) in enumerate(zip(
            history, calibration_history)):
        record_step = record.get("step") if isinstance(record, dict) else None
        if (not isinstance(record, dict)
                or set(record) != {
                    "step", "diagnostics", "transition_from_previous",
                }
                or not isinstance(record_step, int)
                or isinstance(record_step, bool)
                or record_step != selection_record["step"]):
            raise ValueError("calibration diagnostic history record is invalid")
        selection_metrics = selection_record["metrics"]
        diagnostic = _validate_one_calibration_diagnostic(
            record.get("diagnostics"), selection_metrics["sample_count"],
            "step {}".format(record["step"]),
        )
        geometry = diagnostic["stages"]["geometry_top1"]
        if any(geometry[name] != selection_metrics[name] for name in (
                "hits025", "hits050", "acc025", "acc050")):
            raise ValueError(
                "calibration diagnostic geometry selection binding is invalid"
            )
        transition = record["transition_from_previous"]
        if index == 0:
            if transition is not None:
                raise ValueError(
                    "first calibration diagnostic transition must be null"
                )
        else:
            _validate_calibration_diagnostic_transition(
                transition,
                history[index - 1]["step"],
                record["step"],
                selection_metrics["sample_count"],
                normalized[index - 1],
                diagnostic,
            )
        normalized.append(diagnostic)

    selected_matches = [
        (record["diagnostics"], selection_record["metrics"])
        for record, selection_record in zip(history, calibration_history)
        if record["step"] == run_result["selected_step"]
    ]
    if (len(selected_matches) != 1
            or payload["selected_calibration_diagnostics"]
            != selected_matches[0][0]):
        raise ValueError("selected calibration diagnostics are invalid")
    selected_metrics = selected_matches[0][1]
    reproduced = _validate_one_calibration_diagnostic(
        payload["reproduced_calibration_diagnostics"],
        selected_metrics["sample_count"],
        "reproduced selected step",
    )
    reproduced_geometry = reproduced["stages"]["geometry_top1"]
    if any(reproduced_geometry[name] != selected_metrics[name] for name in (
            "hits025", "hits050", "acc025", "acc050")):
        raise ValueError(
            "reproduced calibration diagnostic selection binding is invalid"
        )


def _validate_selected_run_common(run_result, contract_steps):
    required = {
        "completed_updates", "stopped_early", "selected_step",
        "selected_metrics", "reproduced_metrics", "calibration_history",
        "training_diagnostics", "runtime", "train_data_contract",
    }
    if not isinstance(run_result, dict) or not required.issubset(run_result):
        raise ValueError("selected loop result fields are invalid")
    diagnostic_required = {
        "calibration_diagnostics_history",
        "selected_calibration_diagnostics",
        "reproduced_calibration_diagnostics",
    }
    if not diagnostic_required.issubset(run_result):
        raise ValueError("selected loop diagnostic fields are invalid")
    output_digest_required = {
        "selected_calibration_output_sha256",
        "reproduced_calibration_output_sha256",
    }
    if not output_digest_required.issubset(run_result):
        raise ValueError(
            "selected calibration output digest fields are invalid"
        )
    if (not isinstance(run_result.get("selected_step"), int)
            or isinstance(run_result["selected_step"], bool)):
        raise ValueError("selected loop selected step is invalid")
    completed = run_result["completed_updates"]
    selected_output_sha256 = run_result.get(
        "selected_calibration_output_sha256"
    )
    reproduced_output_sha256 = run_result.get(
        "reproduced_calibration_output_sha256"
    )
    if (not isinstance(completed, int) or isinstance(completed, bool)
            or completed <= 0
            or not isinstance(run_result["stopped_early"], bool)
            or run_result["reproduced_metrics"]
            != run_result["selected_metrics"]):
        raise ValueError("selected loop result did not reproduce")
    if (not _is_sha256(selected_output_sha256)
            or not _is_sha256(reproduced_output_sha256)
            or reproduced_output_sha256 != selected_output_sha256):
        raise ValueError(
            "selected calibration output digest did not reproduce exactly"
        )
    _validate_training_diagnostics(
        run_result["training_diagnostics"], completed
    )
    _validate_runtime_provenance(run_result["runtime"])
    _validate_train_data_contract(run_result["train_data_contract"])
    if (run_result["selected_metrics"].get("sample_count")
            != run_result["train_data_contract"][
                "calibration_sample_count"
            ]):
        raise ValueError(
            "selected loop calibration sample count differs from train-data"
        )
    selector = _replay_calibration_history(
        run_result["calibration_history"], contract_steps
    )
    if (selector.best_step != run_result["selected_step"]
            or selector.best_metrics != run_result["selected_metrics"]):
        raise ValueError("selected loop metrics differ from calibration history")
    _validate_calibration_diagnostics_history(
        run_result, run_result["calibration_history"]
    )
    return selector


def _validate_production_run_result(run_result):
    selector = _validate_selected_run_common(
        run_result, rec_finetune.CALIBRATION_STEPS
    )
    history = run_result["calibration_history"]
    last = history[-1]
    completed = run_result["completed_updates"]
    if run_result["stopped_early"]:
        if (last["regression"] is not True
                or last["action"] != "stop"
                or completed != last["step"]):
            raise ValueError(
                "production loop must stop exactly at its first regression"
            )
    elif (completed != PRODUCTION_MAX_STEPS
          or last["step"] != PRODUCTION_MAX_STEPS
          or last["regression"] is not False
          or len(history) != len(rec_finetune.CALIBRATION_STEPS)):
        raise ValueError(
            "production loop without regression must complete 1,836 updates"
        )
    return selector


def _validate_smoke_run_result(run_result, smoke_steps):
    if (not isinstance(smoke_steps, int) or isinstance(smoke_steps, bool)
            or not 1 <= smoke_steps <= PRODUCTION_MAX_STEPS):
        raise ValueError("smoke step contract is invalid")
    selector = _validate_selected_run_common(
        run_result, (0, smoke_steps)
    )
    last = run_result["calibration_history"][-1]
    if (run_result["completed_updates"] != smoke_steps
            or last["step"] != smoke_steps
            or run_result["stopped_early"] is not last["regression"]):
        raise ValueError("smoke loop result is internally inconsistent")
    return selector


def _publication_code_hashes():
    root = Path(__file__).resolve().parents[1]
    models_root = Path(rec_finetune.__file__).resolve().parent
    paths = {
        "runner": Path(__file__).resolve(),
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
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            paths["source/" + relative] = path
    for filename in ("main_utils.py", "train_dist_mod.py"):
        path = root / filename
        paths["source/" + filename] = path
    for path in sorted((root / "pointnet2").glob("_ext*.so")):
        relative = path.relative_to(root).as_posix()
        paths["binary/" + relative] = path
    return {
        name: {
            "path": str(path),
            "sha256": rec_finetune.sha256_file(path),
        }
        for name, path in paths.items()
    }


def _require_unchanged_publication_code_hashes(initial_hashes):
    current_hashes = _publication_code_hashes()
    if initial_hashes != current_hashes:
        raise RuntimeError("publication source code changed during training")
    return copy.deepcopy(current_hashes)


def _require_equal_model_state(expected_model, actual_model, label):
    expected = expected_model.state_dict()
    actual = actual_model.state_dict()
    if (set(expected) != set(actual)
            or any(not torch.equal(
                expected[name].detach().cpu(), actual[name].detach().cpu()
            ) for name in expected)):
        raise RuntimeError("{} strict reload parity failed".format(label))


def _release_live_training_state(state):
    models = (
        state.get("mcln"), state.get("parent"), state.get("geometry")
    )
    if not all(isinstance(model, torch.nn.Module) for model in models):
        raise ValueError("live publication models are invalid")
    parameters = tuple(
        parameter for model in models for parameter in model.parameters()
    )
    if not parameters:
        raise ValueError("live publication models have no parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("live publication models must share one device")
    original_device = next(iter(devices))
    optimizer = state.get("optimizer")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("live publication optimizer is invalid")

    for model in models:
        model.zero_grad(set_to_none=True)
    for model in models:
        model.to(torch.device("cpu"))
    optimizer.state.clear()
    gc.collect()
    if original_device.type == "cuda":
        torch.cuda.empty_cache()
    return original_device


def _rename_directory_noreplace(source, destination):
    """Atomically publish a directory without replacing any destination."""
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            "final output directory already exists: {}".format(destination)
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("atomic RENAME_NOREPLACE is unavailable")
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(destination),
    )


def _validate_calibration_reproduction_observation(
        observation, run_result, label):
    mode, metrics, diagnostics_result = _unpack_calibration_observation(
        observation
    )
    if mode != "diagnostic":
        raise ValueError(
            "{} calibration must reproduce diagnostic mode".format(label)
        )
    if metrics != run_result["selected_metrics"]:
        raise RuntimeError(
            "{} calibration metrics did not reproduce exactly".format(label)
        )
    output_sha256 = rec_finetune.calibration_selected_output_sha256(
        diagnostics_result.transition_state
    )
    if output_sha256 != run_result["selected_calibration_output_sha256"]:
        raise RuntimeError(
            "{} calibration output did not reproduce exactly".format(label)
        )
    diagnostics = copy.deepcopy(diagnostics_result.diagnostics)
    _validate_calibration_diagnostic_json(diagnostics)
    normalized = _validate_one_calibration_diagnostic(
        diagnostics,
        metrics["sample_count"],
        label + " selected step",
    )
    geometry = normalized["stages"]["geometry_top1"]
    if any(geometry[name] != metrics[name] for name in (
            "hits025", "hits050", "acc025", "acc050")):
        raise RuntimeError(
            "{} calibration diagnostic binding is invalid".format(label)
        )
    return {
        "metrics": copy.deepcopy(metrics),
        "diagnostics": diagnostics,
        "output_sha256": output_sha256,
    }


def _verify_staged_publication(
        staging_paths, state, data, run_result, *, model_factory,
        input_builder, forward_fn, calibration_fn, calibration_device,
        expected_sha256):
    if (not isinstance(expected_sha256, dict)
            or set(expected_sha256) != {"backbone", "parent", "geometry"}
            or any(not _is_sha256(value)
                   for value in expected_sha256.values())):
        raise ValueError("staged publication SHA-256 contract is invalid")
    _resolved, checkpoint_bytes, checkpoint_digest = (
        rec_finetune._stable_artifact_snapshot(
            staging_paths["backbone"], "published REC checkpoint"
        )
    )
    if checkpoint_digest != expected_sha256["backbone"]:
        raise RuntimeError("published REC checkpoint SHA-256 changed")
    try:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    except Exception as error:
        raise ValueError(
            "published REC checkpoint could not reload: {}".format(error)
        )
    expected_metadata = {
        "schema": "rec-finetune-selected-step-v2",
        "selected_step": run_result["selected_step"],
        "selected_metrics": copy.deepcopy(run_result["selected_metrics"]),
        "completed_updates": run_result["completed_updates"],
        "stopped_early": run_result["stopped_early"],
    }
    if (not isinstance(checkpoint, dict)
            or set(checkpoint) != {"model", "config", "epoch", "rec_finetune"}
            or checkpoint.get("epoch") != EXPECTED_BACKBONE_EPOCH
            or checkpoint.get("rec_finetune") != expected_metadata):
        raise ValueError("published REC checkpoint fields are invalid")
    cpu_device = torch.device("cpu")
    reloaded_mcln = _load_testable_model(
        checkpoint, checkpoint["config"], cpu_device, model_factory
    )
    _require_equal_model_state(state["mcln"], reloaded_mcln, "MCLN")
    reloaded_parent, parent_artifact, reloaded_geometry, geometry_artifact = (
        rec_finetune.load_rec_finetune_runtime_artifacts(
            staging_paths["parent"], staging_paths["geometry"],
            device=cpu_device,
        )
    )
    if (getattr(reloaded_parent, "_artifact_sha256", None)
            != expected_sha256["parent"]
            or getattr(reloaded_geometry, "_artifact_sha256", None)
            != expected_sha256["geometry"]
            or parent_artifact.get("checkpoint_sha256")
            != checkpoint_digest
            or geometry_artifact.get("checkpoint_sha256")
            != checkpoint_digest
            or geometry_artifact.get("parent_artifact_sha256")
            != expected_sha256["parent"]):
        raise RuntimeError("published REC artifact SHA-256 binding is invalid")
    _require_equal_model_state(state["parent"], reloaded_parent, "parent")
    _require_equal_model_state(
        state["geometry"], reloaded_geometry, "geometry"
    )
    try:
        reloaded_mcln.to(calibration_device)
        reloaded_parent.to(calibration_device)
        reloaded_geometry.to(calibration_device)
        reproduced_observation = calibration_fn(
            reloaded_mcln,
            reloaded_parent,
            reloaded_geometry,
            parent_artifact,
            geometry_artifact,
            data["calibration_loader"],
            data["calibration_view"].indices,
            calibration_device,
            input_builder=input_builder,
            forward_fn=forward_fn,
        )
        return _validate_calibration_reproduction_observation(
            reproduced_observation, run_result, "published REC"
        )
    finally:
        reloaded_mcln.to(cpu_device)
        reloaded_parent.to(cpu_device)
        reloaded_geometry.to(cpu_device)
        gc.collect()
        if calibration_device.type == "cuda":
            torch.cuda.empty_cache()


def publish_rec_finetune_run(
        initialized, run_result, *, failure_injector=None,
        model_factory=None, input_builder=None, forward_fn=None,
        calibration_fn=None):
    """Atomically publish and seal one strictly reproduced selected run."""
    if not isinstance(initialized, dict) or not isinstance(run_result, dict):
        raise ValueError("publication inputs must be mappings")
    if initialized.get("smoke_steps") is not None:
        raise ValueError("smoke runs cannot publish deployable artifacts")
    _validate_production_run_result(run_result)
    failure_injector = failure_injector or (lambda _stage: None)
    if not callable(failure_injector):
        raise ValueError("failure injector must be callable")
    input_builder = input_builder or build_rec_finetune_inputs
    forward_fn = forward_fn or rec_finetune.build_rec_finetune_forward
    calibration_fn = calibration_fn or calibrate_rec_finetune
    paths = initialized["paths"]
    state = initialized["initial_state"]
    data = initialized["data"]
    _validate_training_diagnostics(
        run_result["training_diagnostics"],
        run_result["completed_updates"],
        mcln=state.get("mcln"),
        groups=state.get("groups"),
    )
    _validate_initialized_train_data_binding(
        initialized, run_result["train_data_contract"]
    )
    initial_code_hashes = initialized.get("publication_code_hashes")
    code_hashes = _require_unchanged_publication_code_hashes(
        initial_code_hashes
    )
    output_dir = Path(paths.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            "final output directory already exists: {}".format(output_dir)
        )
    if (not output_dir.parent.is_dir()
            or output_dir.parent.is_symlink()):
        raise ValueError("final output parent must be an existing directory")
    if (state.get("checkpoint_epoch") != EXPECTED_BACKBONE_EPOCH
            or run_result.get("reproduced_metrics")
            != run_result.get("selected_metrics")):
        raise ValueError("selected run is not strictly reproduced from epoch 71")

    input_identities = {
        "initial backbone": _immutable_file_identity(
            paths.backbone_checkpoint, "initial backbone"
        ),
        "initial parent": _immutable_file_identity(
            paths.parent_reranker, "initial parent"
        ),
        "initial geometry": _immutable_file_identity(
            paths.geometry_reranker, "initial geometry"
        ),
    }
    expected_initial_shas = {
        "initial backbone": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
        ),
        "initial parent": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256
        ),
        "initial geometry": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256
        ),
    }
    if any(input_identities[label]["sha256"] != digest
           for label, digest in expected_initial_shas.items()):
        raise ValueError("publication inputs are not authoritative")
    if state.get("checkpoint_sha256") != expected_initial_shas[
            "initial backbone"]:
        raise ValueError("initialized backbone SHA differs from publication")

    staging = Path(tempfile.mkdtemp(
        prefix=".{}.staging-".format(output_dir.name),
        dir=str(output_dir.parent),
    ))
    staging_paths = {
        stage: staging / filename
        for stage, filename in _PUBLICATION_FILENAMES.items()
    }
    final_paths = {
        stage: output_dir / filename
        for stage, filename in _PUBLICATION_FILENAMES.items()
    }
    committed = False
    try:
        checkpoint = {
            "model": _clone_model_state(state["mcln"], "MCLN"),
            "config": state["config"],
            "epoch": EXPECTED_BACKBONE_EPOCH,
            "rec_finetune": {
                "schema": "rec-finetune-selected-step-v2",
                "selected_step": run_result["selected_step"],
                "selected_metrics": copy.deepcopy(
                    run_result["selected_metrics"]
                ),
                "completed_updates": run_result["completed_updates"],
                "stopped_early": run_result["stopped_early"],
            },
        }
        _atomic_torch_save(staging_paths["backbone"], checkpoint)
        checkpoint_sha = rec_finetune.sha256_file(
            staging_paths["backbone"]
        )
        failure_injector("backbone")

        provenance = _online_provenance(state["groups"], run_result)
        parent_artifact = rec_finetune.build_rec_finetune_parent_artifact(
            state["parent"],
            paths.parent_reranker,
            checkpoint_sha,
            EXPECTED_BACKBONE_EPOCH,
            provenance,
            run_result["selected_metrics"],
        )
        rec_finetune.save_rec_finetune_artifact(
            staging_paths["parent"], parent_artifact
        )
        parent_sha = rec_finetune.sha256_file(staging_paths["parent"])
        failure_injector("parent")

        geometry_artifact = rec_finetune.build_rec_finetune_geometry_artifact(
            state["geometry"],
            paths.geometry_reranker,
            parent_artifact,
            parent_sha,
            checkpoint_sha,
            EXPECTED_BACKBONE_EPOCH,
            provenance,
            run_result["selected_metrics"],
        )
        rec_finetune.save_rec_finetune_artifact(
            staging_paths["geometry"], geometry_artifact
        )
        geometry_sha = rec_finetune.sha256_file(staging_paths["geometry"])
        failure_injector("geometry")

        calibration_device = _release_live_training_state(state)
        reloaded_calibration = _verify_staged_publication(
            staging_paths,
            state,
            data,
            run_result,
            model_factory=model_factory,
            input_builder=input_builder,
            forward_fn=forward_fn,
            calibration_fn=calibration_fn,
            calibration_device=calibration_device,
            expected_sha256={
                "backbone": checkpoint_sha,
                "parent": parent_sha,
                "geometry": geometry_sha,
            },
        )

        split_metadata = copy.deepcopy(data["split"]["metadata"])
        if split_metadata != rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0:
            raise ValueError("publication split metadata is not authoritative")
        selection = {
            "schema": "rec-finetune-selection-v3",
            "version": 3,
            "files": {
                "initial_backbone": {
                    "path": str(Path(paths.backbone_checkpoint).resolve()),
                    "sha256": expected_initial_shas["initial backbone"],
                },
                "initial_parent": {
                    "path": str(Path(paths.parent_reranker).resolve()),
                    "sha256": expected_initial_shas["initial parent"],
                },
                "initial_geometry": {
                    "path": str(Path(paths.geometry_reranker).resolve()),
                    "sha256": expected_initial_shas["initial geometry"],
                },
                "final_backbone": {
                    "path": str(final_paths["backbone"].resolve()),
                    "sha256": checkpoint_sha,
                },
                "final_parent": {
                    "path": str(final_paths["parent"].resolve()),
                    "sha256": parent_sha,
                },
                "final_geometry": {
                    "path": str(final_paths["geometry"].resolve()),
                    "sha256": geometry_sha,
                },
            },
            "authoritative_split": split_metadata,
            "mcln_trainable_parameter_names": sorted(
                state["groups"]["mcln_names"]
            ),
            "losses": _loss_publication_contract(),
            "optimizer_groups": _optimizer_publication_contract(),
            "calibration_history": copy.deepcopy(
                run_result["calibration_history"]
            ),
            "calibration_diagnostics_history": copy.deepcopy(
                run_result["calibration_diagnostics_history"]
            ),
            "selected_calibration_diagnostics": copy.deepcopy(
                run_result["selected_calibration_diagnostics"]
            ),
            "reproduced_calibration_diagnostics": copy.deepcopy(
                run_result["reproduced_calibration_diagnostics"]
            ),
            "selected_calibration_output_sha256": run_result[
                "selected_calibration_output_sha256"
            ],
            "reproduced_calibration_output_sha256": run_result[
                "reproduced_calibration_output_sha256"
            ],
            "reloaded_calibration_metrics": copy.deepcopy(
                reloaded_calibration["metrics"]
            ),
            "reloaded_calibration_diagnostics": copy.deepcopy(
                reloaded_calibration["diagnostics"]
            ),
            "reloaded_calibration_output_sha256": reloaded_calibration[
                "output_sha256"
            ],
            "selected_step": run_result["selected_step"],
            "completed_updates": run_result["completed_updates"],
            "stopped_early": run_result["stopped_early"],
            "selected_metrics": copy.deepcopy(
                run_result["selected_metrics"]
            ),
            "validation_data_accessed": False,
            "no_validation_data_declaration": (
                "scanrefer-train-only-no-validation-v1"
            ),
            "code_hashes": copy.deepcopy(code_hashes),
            "publication_order": list(PUBLICATION_ORDER),
            "training_diagnostics": copy.deepcopy(
                run_result["training_diagnostics"]
            ),
            "runtime": copy.deepcopy(run_result["runtime"]),
            "train_data_contract": copy.deepcopy(
                run_result["train_data_contract"]
            ),
        }
        _atomic_canonical_json(staging_paths["selection"], selection)
        selection_sha = rec_finetune.sha256_file(staging_paths["selection"])
        failure_injector("selection")

        staged_sha256 = {
            "backbone": checkpoint_sha,
            "parent": parent_sha,
            "geometry": geometry_sha,
            "selection": selection_sha,
        }
        _assert_staged_publication_files(staging_paths, staged_sha256)
        _assert_immutable_inputs(input_identities)
        _require_unchanged_publication_code_hashes(initial_code_hashes)
        for path in staging_paths.values():
            path.chmod(0o444)
            _fsync_file(path)
        _fsync_directory(staging)
        _assert_staged_publication_files(
            staging_paths, staged_sha256, require_read_only=True
        )
        failure_injector("finalize")
        _assert_staged_publication_files(
            staging_paths, staged_sha256, require_read_only=True
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                "final output directory already exists: {}".format(output_dir)
            )
        _rename_directory_noreplace(staging, output_dir)
        committed = True
        _fsync_directory(output_dir.parent)
        _assert_immutable_inputs(input_identities)
        _assert_staged_publication_files(
            final_paths, staged_sha256, require_read_only=True
        )
        return {
            "output_dir": output_dir,
            "paths": final_paths,
            "sha256": {
                "backbone": checkpoint_sha,
                "parent": parent_sha,
                "geometry": geometry_sha,
                "selection": selection_sha,
            },
            "publication_order": list(PUBLICATION_ORDER),
            "selection": selection,
        }
    finally:
        if not committed and (staging.exists() or staging.is_symlink()):
            shutil.rmtree(staging)


def publish_rec_finetune_smoke_receipt(
        initialized, run_result, *, failure_injector=None,
        input_builder=None, forward_fn=None, calibration_fn=None):
    """Atomically publish a nondeployable receipt for a bounded smoke run."""
    if not isinstance(initialized, dict) or not isinstance(run_result, dict):
        raise ValueError("smoke receipt inputs must be mappings")
    smoke_steps = initialized.get("smoke_steps")
    _validate_smoke_run_result(run_result, smoke_steps)
    failure_injector = failure_injector or (lambda _stage: None)
    if not callable(failure_injector):
        raise ValueError("failure injector must be callable")
    input_builder = input_builder or build_rec_finetune_inputs
    forward_fn = forward_fn or rec_finetune.build_rec_finetune_forward
    calibration_fn = calibration_fn or calibrate_rec_finetune
    if (not callable(input_builder) or not callable(forward_fn)
            or not callable(calibration_fn)):
        raise ValueError("smoke calibration hooks must be callable")
    paths = initialized["paths"]
    state = initialized["initial_state"]
    data = initialized["data"]
    _validate_training_diagnostics(
        run_result["training_diagnostics"],
        run_result["completed_updates"],
        mcln=state.get("mcln"),
        groups=state.get("groups"),
    )
    _validate_initialized_train_data_binding(
        initialized, run_result["train_data_contract"]
    )
    initial_code_hashes = initialized.get("publication_code_hashes")
    code_hashes = _require_unchanged_publication_code_hashes(
        initial_code_hashes
    )
    output_dir = Path(paths.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            "final output directory already exists: {}".format(output_dir)
        )
    if (not output_dir.parent.is_dir()
            or output_dir.parent.is_symlink()):
        raise ValueError("final output parent must be an existing directory")

    input_identities = {
        "initial backbone": _immutable_file_identity(
            paths.backbone_checkpoint, "initial backbone"
        ),
        "initial parent": _immutable_file_identity(
            paths.parent_reranker, "initial parent"
        ),
        "initial geometry": _immutable_file_identity(
            paths.geometry_reranker, "initial geometry"
        ),
    }
    expected_shas = {
        "initial backbone": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
        ),
        "initial parent": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256
        ),
        "initial geometry": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256
        ),
    }
    if (state.get("checkpoint_epoch") != EXPECTED_BACKBONE_EPOCH
            or state.get("checkpoint_sha256")
            != expected_shas["initial backbone"]
            or any(input_identities[label]["sha256"] != digest
                   for label, digest in expected_shas.items())):
        raise ValueError("smoke receipt inputs are not authoritative")

    models = (state.get("mcln"), state.get("parent"), state.get("geometry"))
    if not all(isinstance(model, torch.nn.Module) for model in models):
        raise ValueError("smoke calibration models are invalid")
    devices = {
        parameter.device
        for model in models
        for parameter in model.parameters()
    }
    if len(devices) != 1:
        raise ValueError("smoke calibration models must share one device")
    calibration_device = next(iter(devices))
    _assert_immutable_inputs(input_identities)
    verified_observation = calibration_fn(
        state["mcln"], state["parent"], state["geometry"],
        state["parent_artifact"], state["geometry_artifact"],
        data["calibration_loader"], data["calibration_view"].indices,
        calibration_device,
        input_builder=input_builder,
        forward_fn=forward_fn,
    )
    verified_calibration = _validate_calibration_reproduction_observation(
        verified_observation, run_result, "smoke receipt"
    )
    _assert_immutable_inputs(input_identities)

    input_paths = {
        "initial backbone": Path(paths.backbone_checkpoint),
        "initial parent": Path(paths.parent_reranker),
        "initial geometry": Path(paths.geometry_reranker),
    }
    receipt = {
        "schema": "rec-finetune-smoke-receipt-v3",
        "version": 3,
        "deployable": False,
        "files": {
            label.replace(" ", "_"): {
                "path": str(input_paths[label].resolve()),
                "sha256": expected_shas[label],
            }
            for label in (
                "initial backbone", "initial parent", "initial geometry"
            )
        },
        "smoke_steps": smoke_steps,
        "completed_updates": run_result["completed_updates"],
        "stopped_early": run_result["stopped_early"],
        "selected_step": run_result["selected_step"],
        "selected_metrics": copy.deepcopy(run_result["selected_metrics"]),
        "calibration_history": copy.deepcopy(
            run_result["calibration_history"]
        ),
        "calibration_diagnostics_history": copy.deepcopy(
            run_result["calibration_diagnostics_history"]
        ),
        "selected_calibration_diagnostics": copy.deepcopy(
            run_result["selected_calibration_diagnostics"]
        ),
        "reproduced_calibration_diagnostics": copy.deepcopy(
            run_result["reproduced_calibration_diagnostics"]
        ),
        "selected_calibration_output_sha256": run_result[
            "selected_calibration_output_sha256"
        ],
        "reproduced_calibration_output_sha256": run_result[
            "reproduced_calibration_output_sha256"
        ],
        "verified_calibration_metrics": copy.deepcopy(
            verified_calibration["metrics"]
        ),
        "verified_calibration_diagnostics": copy.deepcopy(
            verified_calibration["diagnostics"]
        ),
        "verified_calibration_output_sha256": verified_calibration[
            "output_sha256"
        ],
        "validation_data_accessed": False,
        "no_validation_data_declaration": (
            "scanrefer-train-only-no-validation-v1"
        ),
        "losses": _loss_publication_contract(),
        "code_hashes": copy.deepcopy(code_hashes),
        "training_diagnostics": copy.deepcopy(
            run_result["training_diagnostics"]
        ),
        "runtime": copy.deepcopy(run_result["runtime"]),
        "train_data_contract": copy.deepcopy(
            run_result["train_data_contract"]
        ),
    }

    staging = Path(tempfile.mkdtemp(
        prefix=".{}.staging-".format(output_dir.name),
        dir=str(output_dir.parent),
    ))
    staged_receipt = staging / "smoke-receipt.json"
    committed = False
    try:
        _atomic_canonical_json(staged_receipt, receipt)
        receipt_sha = rec_finetune.sha256_file(staged_receipt)
        failure_injector("receipt")
        staged_receipt_paths = {"receipt": staged_receipt}
        staged_receipt_sha256 = {"receipt": receipt_sha}
        _assert_staged_publication_files(
            staged_receipt_paths, staged_receipt_sha256
        )
        _assert_immutable_inputs(input_identities)
        _require_unchanged_publication_code_hashes(initial_code_hashes)
        staged_receipt.chmod(0o444)
        _fsync_file(staged_receipt)
        _fsync_directory(staging)
        _assert_staged_publication_files(
            staged_receipt_paths,
            staged_receipt_sha256,
            require_read_only=True,
        )
        failure_injector("finalize")
        _assert_staged_publication_files(
            staged_receipt_paths,
            staged_receipt_sha256,
            require_read_only=True,
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                "final output directory already exists: {}".format(output_dir)
            )
        _rename_directory_noreplace(staging, output_dir)
        committed = True
        _fsync_directory(output_dir.parent)
        _assert_immutable_inputs(input_identities)
        _assert_staged_publication_files(
            {"receipt": output_dir / "smoke-receipt.json"},
            staged_receipt_sha256,
            require_read_only=True,
        )
        return {
            "output_dir": output_dir,
            "path": output_dir / "smoke-receipt.json",
            "sha256": receipt_sha,
            "receipt": receipt,
        }
    finally:
        if not committed and (staging.exists() or staging.is_symlink()):
            shutil.rmtree(staging)


def _require_utc_timestamp(label, value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("{} must be a UTC timestamp".format(label))
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("{} is invalid: {}".format(label, error))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("{} must be a UTC timestamp".format(label))
    return value


def _utc_timestamp_datetime(value):
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _runtime_cuda_snapshot():
    properties = torch.cuda.get_device_properties(0)
    return {
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
    }


def build_rec_finetune_runtime_provenance(
        *, started_utc, finished_utc, elapsed_seconds, command):
    """Build the successful-run runtime receipt from JSON-safe primitives."""
    started = _require_utc_timestamp("runtime start", started_utc)
    finished = _require_utc_timestamp("runtime finish", finished_utc)
    if _utc_timestamp_datetime(finished) < _utc_timestamp_datetime(started):
        raise ValueError("runtime finish timestamp precedes runtime start")
    elapsed = _require_finite_primitive(
        "runtime elapsed seconds", elapsed_seconds
    )
    if elapsed < 0.0:
        raise ValueError("runtime elapsed seconds must be nonnegative")
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(value, str) for value in command)):
        raise ValueError("runtime command must be a non-empty string sequence")
    cuda = _runtime_cuda_snapshot()
    if (not isinstance(cuda, dict)
            or set(cuda) != {"device", "peak_cuda_memory"}):
        raise ValueError("CUDA runtime snapshot is invalid")
    device = cuda["device"]
    peak = cuda["peak_cuda_memory"]
    if (not isinstance(device, dict)
            or set(device) != {
                "type", "index", "name", "total_memory_bytes",
            }
            or device.get("type") != "cuda"
            or device.get("index") != 0
            or not isinstance(device.get("name"), str)
            or not isinstance(device.get("total_memory_bytes"), int)
            or isinstance(device.get("total_memory_bytes"), bool)
            or device["total_memory_bytes"] <= 0):
        raise ValueError("CUDA device provenance is invalid")
    if (not isinstance(peak, dict)
            or set(peak) != {"allocated_bytes", "reserved_bytes"}
            or any(not isinstance(peak.get(name), int)
                   or isinstance(peak.get(name), bool)
                   or peak[name] < 0
                   for name in ("allocated_bytes", "reserved_bytes"))):
        raise ValueError("peak CUDA memory provenance is invalid")
    logical_interpreter = str(sys.executable)
    cudnn_version = torch.backends.cudnn.version()
    runtime = {
        "schema": "rec-finetune-runtime-v1",
        "started_utc": started,
        "finished_utc": finished,
        "elapsed_seconds": elapsed,
        "completed_successfully": True,
        "oom_detected": False,
        "command": list(command),
        "interpreter": {
            "logical_path": logical_interpreter,
            "resolved_path": str(Path(logical_interpreter).resolve()),
        },
        "versions": {
            "python": str(platform.python_version()),
            "torch": str(torch.__version__),
            "cuda": (
                None if torch.version.cuda is None
                else str(torch.version.cuda)
            ),
            "cudnn": (
                None if cudnn_version is None else int(cudnn_version)
            ),
        },
        "device": copy.deepcopy(device),
        "peak_cuda_memory": copy.deepcopy(peak),
        "environment": {
            name: os.environ.get(name)
            for name in RUNTIME_ENVIRONMENT_ALLOWLIST
        },
    }
    try:
        json.dumps(runtime, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "runtime provenance is not JSON-safe: {}".format(error)
        )
    _validate_runtime_provenance(runtime)
    return runtime


def _runtime_utc_now():
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _runtime_monotonic():
    return time.monotonic()


def _reset_cuda_peak_memory_stats():
    torch.cuda.reset_peak_memory_stats(0)


def _runtime_command(argv):
    if argv is None:
        arguments = [str(value) for value in sys.argv]
    else:
        arguments = [str(Path(__file__).resolve())] + [
            str(value) for value in argv
        ]
    return [str(sys.executable)] + arguments


def _set_production_determinism():
    random.seed(PRODUCTION_SEED)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(PRODUCTION_SEED)
    torch.manual_seed(PRODUCTION_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("production REC fine-tuning requires CUDA")
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(PRODUCTION_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def main(argv=None):
    """Initialize, fit, select, and atomically publish the production run."""
    args = parse_args(argv)
    command = _runtime_command(argv)
    started_utc = _runtime_utc_now()
    started_monotonic = _runtime_monotonic()
    _set_production_determinism()
    _reset_cuda_peak_memory_stats()
    publication_code_hashes = _publication_code_hashes()
    initialized = initialize_rec_finetune_run(args)
    if not isinstance(initialized, dict):
        raise ValueError("initialized REC fine-tune state is invalid")
    initialized["publication_code_hashes"] = publication_code_hashes
    run_result = run_rec_finetune(initialized)
    finished_utc = _runtime_utc_now()
    finished_monotonic = _runtime_monotonic()
    run_result["runtime"] = build_rec_finetune_runtime_provenance(
        started_utc=started_utc,
        finished_utc=finished_utc,
        elapsed_seconds=finished_monotonic - started_monotonic,
        command=command,
    )
    if args.smoke_steps is None:
        publication = publish_rec_finetune_run(initialized, run_result)
    else:
        publication = publish_rec_finetune_smoke_receipt(
            initialized, run_result
        )
    summary = {
        "output_dir": str(publication["output_dir"]),
        "selected_step": run_result["selected_step"],
        "completed_updates": run_result["completed_updates"],
        "stopped_early": run_result["stopped_early"],
        "acc025": run_result["selected_metrics"]["acc025"],
        "acc050": run_result["selected_metrics"]["acc050"],
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return summary


if __name__ == "__main__":
    main()
