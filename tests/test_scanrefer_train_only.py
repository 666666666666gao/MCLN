import builtins
import math
import os

import pytest

from models import rec_finetune
from scripts.tuning import scanrefer_train_only as train_only


def _row(dataset, scan_id, value):
    return {
        "dataset": dataset,
        "scan_id": scan_id,
        "value": value,
    }


def _small_annotations():
    scanrefer_rows = [
        _row("scanrefer", "scene_a", "a0"),
        _row("scanrefer", "scene_b", "b0"),
        _row("scanrefer", "scene_a", "a1"),
        _row("scanrefer", "scene_c", "c0"),
        _row("scanrefer", "scene_d", "d0"),
    ]
    scannet_a = _row("scannet", "scene_a", "det-a")
    scannet_b = _row("scannet", "scene_b", "det-b")
    scannet_c = _row("scannet", "scene_c", "det-c")
    scannet_d = _row("scannet", "scene_d", "det-d")
    scannet_extra = _row("scannet", "scene_extra", "det-extra")
    return [
        scanrefer_rows[0],
        scannet_a,
        scanrefer_rows[1],
        scannet_b,
        scanrefer_rows[2],
        scannet_a,
        scanrefer_rows[3],
        scannet_c,
        scanrefer_rows[4],
        scannet_d,
        scannet_extra,
        scannet_extra,
    ]


def test_partition_is_scene_disjoint_scanrefer_only_calibration_and_ordered():
    annos = _small_annotations()

    split = train_only.partition_train_annotations(
        annos, seed=7, calibration_fraction=0.25
    )

    calibration_scenes = {
        row["scan_id"] for row in split["calibration_annos"]
    }
    assert calibration_scenes
    assert {
        row["dataset"] for row in split["calibration_annos"]
    } == {"scanrefer"}
    assert all(
        row["scan_id"] not in calibration_scenes
        for row in split["fit_annos"]
    )
    assert split["fit_annos"] == [
        row for row in annos if row["scan_id"] not in calibration_scenes
    ]
    assert split["calibration_annos"] == [
        row for row in annos
        if row["dataset"] == "scanrefer"
        and row["scan_id"] in calibration_scenes
    ]
    assert sum(
        row["scan_id"] == "scene_extra" for row in split["fit_annos"]
    ) == 2
    assert set(split["fit_scenes"]).isdisjoint(split["calibration_scenes"])


def test_partition_reuses_rec_finetune_split_metadata_and_is_deterministic():
    annos = _small_annotations()
    scanrefer_scan_ids = [
        row["scan_id"] for row in annos if row["dataset"] == "scanrefer"
    ]
    expected = rec_finetune.build_rec_finetune_scene_split(
        scanrefer_scan_ids, seed=3, calibration_fraction=0.40
    )

    first = train_only.partition_train_annotations(
        annos, seed=3, calibration_fraction=0.40
    )
    second = train_only.partition_train_annotations(
        list(annos), seed=3, calibration_fraction=0.40
    )

    assert first == second
    assert first["metadata"] == expected["metadata"]
    assert first["fit_scenes"] == expected["fit_scenes"]
    assert first["calibration_scenes"] == expected["calibration_scenes"]
    assert first["metadata"]["scene_count"] == 4
    assert first["metadata"]["sample_count"] == 5
    assert (
        first["metadata"]["fit_sample_count"]
        + first["metadata"]["calibration_sample_count"]
        == 5
    )


class _DuplicateDatasetKeys(dict):
    def keys(self):
        return list(super().keys()) + ["dataset"]


@pytest.mark.parametrize(
    "bad_annos",
    [
        [],
        [_row("scanrefer", "scene_a", 0)],
        [
            {"scan_id": "scene_a"},
            _row("scanrefer", "scene_b", 1),
        ],
        [
            _DuplicateDatasetKeys(
                dataset="scanrefer", scan_id="scene_a", value=0
            ),
            _row("scanrefer", "scene_b", 1),
        ],
        [
            _row(("scanrefer", "scannet"), "scene_a", 0),
            _row("scanrefer", "scene_b", 1),
        ],
        [
            _row("other", "scene_a", 0),
            _row("scanrefer", "scene_b", 1),
        ],
        [
            {"dataset": "scanrefer", "value": 0},
            _row("scanrefer", "scene_b", 1),
        ],
        [
            _row("scanrefer", "", 0),
            _row("scanrefer", "scene_b", 1),
        ],
        [
            _row("scanrefer", "scene_a", 0),
            _row("scanrefer", "scene_a", 1),
            _row("scannet", "scene_extra", 2),
        ],
    ],
)
def test_partition_rejects_invalid_rows_and_fewer_than_two_scanrefer_scenes(
        bad_annos):
    with pytest.raises(ValueError):
        train_only.partition_train_annotations(bad_annos)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": 1.5},
        {"calibration_fraction": 0.0},
        {"calibration_fraction": 1.0},
        {"calibration_fraction": math.nan},
    ],
)
def test_partition_rejects_invalid_seed_and_fraction(kwargs):
    with pytest.raises(ValueError):
        train_only.partition_train_annotations(_small_annotations(), **kwargs)


class _Dataset:
    def __init__(self, annos, split="train"):
        self.annos = annos
        self.split = split
        self.augment = True
        self.augment_det = True
        self.joint_det = True
        self.random_utt = True
        self.scans = {"heavy": object()}
        self.superpoints = {"heavy": object()}
        self.tokenizer = object()


def test_make_dataset_views_shares_heavy_state_but_isolates_annos_and_flags():
    annos = _small_annotations()
    split = train_only.partition_train_annotations(annos)
    base = _Dataset(annos)

    fit, calibration = train_only.make_dataset_views(base, split)

    assert fit is not base
    assert calibration is not base
    assert fit is not calibration
    assert fit.annos == split["fit_annos"]
    assert calibration.annos == split["calibration_annos"]
    assert fit.annos is not split["fit_annos"]
    assert calibration.annos is not split["calibration_annos"]
    assert fit.annos is not base.annos
    assert calibration.annos is not base.annos
    for attribute in ("scans", "superpoints", "tokenizer"):
        assert getattr(fit, attribute) is getattr(base, attribute)
        assert getattr(calibration, attribute) is getattr(base, attribute)

    assert fit.augment is True
    assert fit.augment_det is True
    assert fit.joint_det is True
    assert fit.random_utt is True
    assert calibration.augment is False
    assert calibration.augment_det is False
    assert calibration.joint_det is False
    assert calibration.random_utt is False
    assert base.augment is True
    assert base.augment_det is True
    assert base.joint_det is True
    assert base.random_utt is True

    calibration.annos.append(_row("scanrefer", "scene_z", "z0"))
    calibration.augment = True
    assert fit.annos == split["fit_annos"]
    assert fit.augment is True
    assert base.annos == annos


@pytest.mark.parametrize(
    "missing_attribute",
    ["annos", "augment", "augment_det", "joint_det", "scans", "superpoints"],
)
def test_make_dataset_views_rejects_incomplete_dataset_objects(
        missing_attribute):
    base = _Dataset(_small_annotations())
    delattr(base, missing_attribute)
    split = train_only.partition_train_annotations(_small_annotations())

    with pytest.raises(ValueError):
        train_only.make_dataset_views(base, split)


def test_formal_validator_requires_the_exact_authoritative_metadata():
    expected = dict(rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0)

    validated = train_only.validate_formal_run_metadata(expected)

    assert validated == expected
    assert validated is not expected
    assert validated["scene_count"] == 562
    assert validated["sample_count"] == 36665
    assert validated["fit_scene_count"] == 506
    assert validated["fit_sample_count"] == 33040
    assert validated["calibration_scene_count"] == 56
    assert validated["calibration_sample_count"] == 3625

    for key in (
            "scene_count", "sample_count", "fit_scene_count",
            "fit_sample_count", "calibration_scene_count",
            "calibration_sample_count"):
        wrong = dict(expected)
        wrong[key] += 1
        with pytest.raises(ValueError):
            train_only.validate_formal_run_metadata(wrong)

    extra = dict(expected)
    extra["unexpected"] = True
    with pytest.raises(ValueError):
        train_only.validate_formal_run_metadata(extra)


def _is_forbidden_validation_path(value):
    try:
        path = os.fspath(value).replace("\\", "/").lower()
    except TypeError:
        return False
    return path.endswith("/val_v3scans.pkl") or path.endswith("_val.json")


def test_builder_constructs_only_joint_train_and_never_opens_official_val(
        monkeypatch):
    calls = []
    opened = []

    class FakeJoint3DDataset(_Dataset):
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            super().__init__(_small_annotations(), split=kwargs["split"])

    real_open = builtins.open

    def guarded_open(path, *args, **kwargs):
        opened.append(os.fspath(path))
        if _is_forbidden_validation_path(path):
            raise AssertionError("official validation path was opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(train_only, "Joint3DDataset", FakeJoint3DDataset)
    monkeypatch.setattr(builtins, "open", guarded_open)

    data = train_only.build_train_only_data(
        data_path="/synthetic/train-root",
        use_color=False,
        use_height=False,
    )

    assert len(calls) == 1
    assert calls[0]["split"] == "train"
    assert calls[0]["dataset_dict"] == {"scanrefer": 1, "scannet": 10}
    assert calls[0]["test_dataset"] == "scanrefer"
    assert not any(
        _is_forbidden_validation_path(value)
        for value in calls[0].values()
    )
    assert not any(_is_forbidden_validation_path(path) for path in opened)
    assert data["base_dataset"].split == "train"
    assert data["fit_dataset"].split == "train"
    assert data["calibration_dataset"].split == "train"
    assert data["split"]["metadata"] == data["metadata"]
    assert data["calibration_dataset"].augment is False
    assert data["calibration_dataset"].joint_det is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "val"},
        {"scan_cache": "/data/val_v3scans.pkl"},
        {"annotation_path": "/data/ScanRefer_filtered_val.json"},
    ],
)
def test_builder_rejects_validation_inputs_before_dataset_construction(
        monkeypatch, kwargs):
    calls = []

    class FakeJoint3DDataset(_Dataset):
        def __init__(self, **factory_kwargs):
            calls.append(factory_kwargs)
            super().__init__(_small_annotations(), split="train")

    monkeypatch.setattr(train_only, "Joint3DDataset", FakeJoint3DDataset)

    with pytest.raises(ValueError):
        train_only.build_train_only_data(**kwargs)

    assert calls == []


def test_builder_enforces_formal_metadata_only_when_requested():
    def factory(**kwargs):
        return _Dataset(_small_annotations(), split=kwargs["split"])

    data = train_only.build_train_only_data(dataset_factory=factory)
    assert data["metadata"]["scene_count"] == 4

    with pytest.raises(ValueError):
        train_only.build_train_only_data(
            dataset_factory=factory,
            formal_run=True,
        )


def test_builder_rejects_a_factory_that_ignores_the_train_split():
    def factory(**kwargs):
        return _Dataset(_small_annotations(), split="val")

    with pytest.raises(ValueError):
        train_only.build_train_only_data(dataset_factory=factory)
