"""Build scene-disjoint ScanRefer fit/calibration views from train data."""

import copy
import os
from collections.abc import Mapping

from models import rec_finetune
from src.joint_det_dataset import Joint3DDataset


AUTHORITATIVE_SCANREFER_SPLIT_METADATA = dict(
    rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
)
_SUPPORTED_DATASETS = frozenset(("scanrefer", "scannet"))
_REQUIRED_DATASET_ATTRIBUTES = (
    "annos",
    "augment",
    "augment_det",
    "joint_det",
    "scans",
    "superpoints",
)


def _annotation_fields(annotation, index):
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation {} must be a mapping".format(index))
    try:
        dataset_key_count = sum(
            key == "dataset" for key in annotation.keys()
        )
    except Exception as error:
        raise ValueError(
            "annotation {} has invalid fields: {}".format(index, error)
        )
    if dataset_key_count != 1:
        raise ValueError(
            "annotation {} must contain exactly one dataset field".format(
                index
            )
        )
    dataset = annotation["dataset"]
    if not isinstance(dataset, str) or dataset not in _SUPPORTED_DATASETS:
        raise ValueError(
            "annotation {} dataset must be scanrefer or scannet".format(
                index
            )
        )
    scan_id = annotation.get("scan_id")
    if not isinstance(scan_id, str) or not scan_id.strip():
        raise ValueError(
            "annotation {} scan_id must be a non-empty string".format(index)
        )
    return dataset, scan_id


def partition_train_annotations(
        annos, seed=0, calibration_fraction=0.10):
    """Partition mixed train annotations by ScanRefer scene.

    ScanRefer rows determine the authoritative scene split. Scannet rows from
    fit scenes and Scannet-only scenes remain in fit; Scannet rows from a
    calibration scene are omitted so calibration stays ScanRefer-only.
    """
    if not isinstance(annos, (list, tuple)) or not annos:
        raise ValueError("annos must be a non-empty annotation sequence")

    validated = []
    scanrefer_scan_ids = []
    for index, annotation in enumerate(annos):
        dataset, scan_id = _annotation_fields(annotation, index)
        validated.append((annotation, dataset, scan_id))
        if dataset == "scanrefer":
            scanrefer_scan_ids.append(scan_id)

    if len(set(scanrefer_scan_ids)) < 2:
        raise ValueError(
            "ScanRefer annotations must cover at least two scenes"
        )

    scene_split = rec_finetune.build_rec_finetune_scene_split(
        scanrefer_scan_ids,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    calibration_scene_set = set(scene_split["calibration_scenes"])
    fit_annos = [
        annotation
        for annotation, _dataset, scan_id in validated
        if scan_id not in calibration_scene_set
    ]
    calibration_annos = [
        annotation
        for annotation, dataset, scan_id in validated
        if dataset == "scanrefer" and scan_id in calibration_scene_set
    ]

    return {
        "fit_annos": fit_annos,
        "calibration_annos": calibration_annos,
        "fit_scenes": tuple(scene_split["fit_scenes"]),
        "calibration_scenes": tuple(scene_split["calibration_scenes"]),
        "metadata": dict(scene_split["metadata"]),
    }


def _require_dataset_attributes(base_dataset):
    missing = [
        name for name in _REQUIRED_DATASET_ATTRIBUTES
        if not hasattr(base_dataset, name)
    ]
    if missing:
        raise ValueError(
            "base_dataset is missing required attributes: {}".format(
                ", ".join(missing)
            )
        )
    if not isinstance(base_dataset.annos, (list, tuple)):
        raise ValueError("base_dataset.annos must be an annotation sequence")


def make_dataset_views(base_dataset, split):
    """Return shallow fit/calibration copies sharing heavy dataset state."""
    _require_dataset_attributes(base_dataset)
    if not isinstance(split, Mapping):
        raise ValueError("split must be a mapping")
    fit_annos = split.get("fit_annos")
    calibration_annos = split.get("calibration_annos")
    if not isinstance(fit_annos, (list, tuple)):
        raise ValueError("split.fit_annos must be an annotation sequence")
    if not isinstance(calibration_annos, (list, tuple)):
        raise ValueError(
            "split.calibration_annos must be an annotation sequence"
        )

    fit_dataset = copy.copy(base_dataset)
    calibration_dataset = copy.copy(base_dataset)
    if (fit_dataset is base_dataset
            or calibration_dataset is base_dataset
            or fit_dataset is calibration_dataset):
        raise ValueError("base_dataset must support distinct shallow copies")

    fit_dataset.annos = list(fit_annos)
    calibration_dataset.annos = list(calibration_annos)
    calibration_dataset.augment = False
    calibration_dataset.augment_det = False
    calibration_dataset.joint_det = False
    calibration_dataset.random_utt = False
    return fit_dataset, calibration_dataset


def validate_formal_run_metadata(metadata):
    """Require the exact seed-0 authoritative real-train split metadata."""
    if not isinstance(metadata, Mapping):
        raise ValueError("formal-run metadata must be a mapping")
    actual = dict(metadata)
    expected = dict(rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0)
    if actual != expected:
        raise ValueError(
            "formal run requires the authoritative ScanRefer train split"
        )
    return actual


def _forbidden_validation_path(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    raw = os.fspath(value)
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    normalized = raw.replace("\\", "/").lower()
    basename = normalized.rsplit("/", 1)[-1]
    return basename == "val_v3scans.pkl" or basename.endswith("_val.json")


def _contains_forbidden_validation_path(value):
    if _forbidden_validation_path(value):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_validation_path(key)
            or _contains_forbidden_validation_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_forbidden_validation_path(item) for item in value)
    return False


def build_train_only_data(
        *, seed=0, calibration_fraction=0.10, formal_run=False,
        dataset_factory=None, **dataset_kwargs):
    """Construct one train ``Joint3DDataset`` and its two train views."""
    if not isinstance(formal_run, bool):
        raise ValueError("formal_run must be a boolean")
    requested_split = dataset_kwargs.pop("split", "train")
    if requested_split != "train":
        raise ValueError("train-only data must use split='train'")
    if _contains_forbidden_validation_path(dataset_kwargs):
        raise ValueError("official validation paths are forbidden")

    factory = Joint3DDataset if dataset_factory is None else dataset_factory
    if not callable(factory):
        raise ValueError("dataset_factory must be callable")
    kwargs = dict(dataset_kwargs)
    kwargs.setdefault(
        "dataset_dict", {"scanrefer": 1, "scannet": 10}
    )
    kwargs.setdefault("test_dataset", "scanrefer")
    kwargs["split"] = "train"
    if _contains_forbidden_validation_path(kwargs):
        raise ValueError("official validation paths are forbidden")

    base_dataset = factory(**kwargs)
    if getattr(base_dataset, "split", None) != "train":
        raise ValueError("dataset factory did not honor split='train'")
    split = partition_train_annotations(
        getattr(base_dataset, "annos", None),
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    if formal_run:
        validate_formal_run_metadata(split["metadata"])
    fit_dataset, calibration_dataset = make_dataset_views(
        base_dataset, split
    )
    return {
        "base_dataset": base_dataset,
        "fit_dataset": fit_dataset,
        "calibration_dataset": calibration_dataset,
        "split": split,
        "metadata": dict(split["metadata"]),
    }


__all__ = (
    "AUTHORITATIVE_SCANREFER_SPLIT_METADATA",
    "Joint3DDataset",
    "build_train_only_data",
    "make_dataset_views",
    "partition_train_annotations",
    "validate_formal_run_metadata",
)
