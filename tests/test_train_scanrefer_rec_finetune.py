import json
import builtins
import copy
from decimal import localcontext
import gc
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

import models.rec_finetune as rec_finetune
from models.rec_reranker import QueryReranker, compute_query_ious
import scripts.train_scanrefer_rec_finetune as finetune_runner
import src.grounding_evaluator as grounding_evaluator_module
import src.joint_det_dataset as joint_det_dataset_module


TRAIN_ANNOTATIONS = Path(
    "/root/autodl-tmp/DATA_ROOT/scanrefer/ScanRefer_filtered_train.json"
)


def test_production_import_graph_keeps_wandb_visualization_dependency_lazy():
    assert "wandb" not in vars(joint_det_dataset_module)
    assert "wandb" not in vars(grounding_evaluator_module)


def test_production_import_graph_does_not_spawn_git_or_ldconfig():
    repo = Path(finetune_runner.__file__).resolve().parents[1]
    child = r'''
import json
import subprocess
import sys

real_popen = subprocess.Popen
events = []

class SpyPopen(real_popen):
    def __init__(self, *args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        if isinstance(command, (list, tuple)):
            command = [str(value) for value in command]
        else:
            command = str(command)
        events.append({"command": command, "cwd": kwargs.get("cwd")})
        real_popen.__init__(self, *args, **kwargs)

subprocess.Popen = SpyPopen
import src.joint_det_dataset as joint
import src.grounding_evaluator as evaluator
print(json.dumps({
    "events": events,
    "joint_wandb": "wandb" in vars(joint),
    "evaluator_wandb": "wandb" in vars(evaluator),
}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=str(repo),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["joint_wandb"] is False
    assert payload["evaluator_wandb"] is False
    forbidden = {"git", "ldconfig", "ldconfig.real"}
    for event in payload["events"]:
        command = event["command"]
        argv = command if isinstance(command, list) else [command]
        executable = Path(argv[0]).name if argv else ""
        assert executable not in forbidden
        assert not (
            executable == "git" and len(argv) > 1 and argv[1] == "version"
        )
INITIAL_BACKBONE_CHECKPOINT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/"
    "mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
)
INITIAL_PARENT_ARTIFACT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/"
    "reranker_h256_d010_lr1e3_seed0_final_contract.pth"
)
INITIAL_GEOMETRY_ARTIFACT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_artifacts/selected_geometry_reranker.pth"
)
SEALED_OFFICIAL_EVIDENCE = (
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_artifacts/epoch71_rec_geometry_official_validation_once."
        "claim.json"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_artifacts/epoch71_rec_geometry_official_validation_once."
        "receipt.json"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_artifacts/geometry_val_sidecar_once.claim"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_artifacts/selection.json"
    ),
    INITIAL_GEOMETRY_ARTIFACT,
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_official_val/geometry_val_sidecar_once.json"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_official_val/official_result.json"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_official_val/official_sidecar_comparison.json"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_official_val/official_stdout.log"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_official_val/scanrefer/epoch71_geometry_official/"
        "1784133929/config.json"
    ),
    Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_official_val/scanrefer/epoch71_geometry_official/"
        "1784133929/log.txt"
    ),
)

EXPECTED_SEED0_SPLIT = {
    "split_seed": 0,
    "calibration_fraction": 0.10,
    "scene_count": 562,
    "fit_scene_count": 506,
    "calibration_scene_count": 56,
    "sample_count": 36665,
    "fit_sample_count": 33040,
    "calibration_sample_count": 3625,
    "fit_scene_sha256": (
        "790264c59d4e4f5937b49b0440c020d485c0929a843176a3a434f2ce8d797a17"
    ),
    "calibration_scene_sha256": (
        "f58524379488c4bd061849167f537ba3a10671317b30c89dd580ba147e8e5cdc"
    ),
    "mapping_sha256": (
        "72685aa01285dbe72b9e0331acd5f10457f773e9e158ae4f884b9c4176cf95bd"
    ),
}


def test_rec_finetune_runner_initialization_api_is_public():
    assert callable(finetune_runner.parse_args)
    assert callable(finetune_runner.validate_runtime_paths)
    assert callable(finetune_runner.load_rec_finetune_initial_state)
    assert callable(finetune_runner.build_train_only_data)
    assert callable(finetune_runner.initialize_rec_finetune_run)


def test_runner_reexports_the_candidate_cache_batch_device_helper():
    from scripts import cache_scanrefer_rec_candidates as candidate_cache

    assert (
        finetune_runner._move_batch_to_device
        is candidate_cache._move_batch_to_device
    )
    labels = ["unchanged"]
    moved = finetune_runner._move_batch_to_device(
        {"tensor": torch.tensor([1.0]), "labels": labels},
        torch.device("cpu"),
    )
    assert moved["tensor"].device.type == "cpu"
    assert torch.equal(moved["tensor"], torch.tensor([1.0]))
    assert moved["labels"] is labels


def test_low_level_initialization_test_seams_are_keyword_only():
    load_signature = inspect.signature(
        finetune_runner.load_rec_finetune_initial_state
    )
    for name in (
            "device", "model_factory", "parent_loader",
            "geometry_loader", "geometry_validator"):
        assert load_signature.parameters[name].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )
    data_signature = inspect.signature(finetune_runner.build_train_only_data)
    for name in (
            "dataset_factory", "loader_factory", "expected_split_metadata"):
        assert data_signature.parameters[name].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )

    with pytest.raises(TypeError):
        finetune_runner.load_rec_finetune_initial_state(
            "backbone", "parent", "geometry", "data-root", "cpu"
        )
    with pytest.raises(TypeError):
        finetune_runner.build_train_only_data(object(), "cpu", object())


def _runner_argv(tmp_path, **overrides):
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    values = {
        "backbone_checkpoint": (
            tmp_path / "inputs" / "backbone" / "backbone.pth"
        ),
        "parent_reranker": (
            tmp_path / "inputs" / "parent" / "parent.pth"
        ),
        "geometry_reranker": (
            tmp_path / "inputs" / "geometry" / "geometry.pth"
        ),
        "output_dir": tmp_path / "new-output",
    }
    values.update(overrides)
    for name in ("backbone_checkpoint", "parent_reranker",
                 "geometry_reranker"):
        path = Path(values[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(name.encode("ascii"))
    return [
        "--data-root", str(data_root),
        "--backbone-checkpoint", str(values["backbone_checkpoint"]),
        "--parent-reranker", str(values["parent_reranker"]),
        "--geometry-reranker", str(values["geometry_reranker"]),
        "--output-dir", str(values["output_dir"]),
    ]


def test_runner_cli_exposes_only_paths_device_and_bounded_smoke_steps(tmp_path):
    argv = _runner_argv(tmp_path)
    args = finetune_runner.parse_args(argv)

    assert set(vars(args)) == {
        "data_root", "backbone_checkpoint", "parent_reranker",
        "geometry_reranker", "output_dir", "device", "smoke_steps",
    }
    assert args.device == "cuda:0"
    assert args.smoke_steps is None
    assert finetune_runner.parse_args(argv + [
        "--smoke-steps", "1836"
    ]).smoke_steps == 1836

    for extra in (
            ["--device", "cpu"],
            ["--device", "cuda:1"],
            ["--smoke-steps", "0"],
            ["--smoke-steps", "1837"],
            ["--batch-size", "1"],
            ["--num-workers", "0"],
            ["--seed", "1"],
            ["--max-steps", "1"],
            ["--calibration-interval", "1"]):
        with pytest.raises(SystemExit):
            finetune_runner.parse_args(argv + extra)


def test_runtime_path_validation_is_read_only_and_rejects_collisions(
        tmp_path, monkeypatch):
    protected = tmp_path / "legacy" / "geometry_val"
    protected.mkdir(parents=True)
    monkeypatch.setattr(
        finetune_runner, "PROTECTED_LEGACY_PATHS", (protected,)
    )
    args = finetune_runner.parse_args(_runner_argv(tmp_path))

    paths = finetune_runner.validate_runtime_paths(args)

    assert paths.data_root == (tmp_path / "data").resolve()
    assert paths.output_dir == (tmp_path / "new-output").resolve()
    assert not paths.output_dir.exists()

    for output in (
            args.backbone_checkpoint,
            str(Path(args.parent_reranker).parent),
            str(protected / "nested"),
            str(protected.parent)):
        bad = SimpleNamespace(**vars(args))
        bad.output_dir = output
        with pytest.raises((ValueError, FileExistsError)):
            finetune_runner.validate_runtime_paths(bad)

    for input_path in (
            args.backbone_checkpoint,
            args.parent_reranker,
            args.geometry_reranker):
        nested = SimpleNamespace(**vars(args))
        nested.output_dir = str(Path(input_path).parent / "nested-output")
        with pytest.raises(ValueError, match="parent|protected|collid"):
            finetune_runner.validate_runtime_paths(nested)

    missing = SimpleNamespace(**vars(args))
    missing.backbone_checkpoint = str(tmp_path / "missing.pth")
    with pytest.raises(ValueError, match="backbone"):
        finetune_runner.validate_runtime_paths(missing)
    assert not paths.output_dir.exists()


class _TinyMcln(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.decoder_query_proj = torch.nn.Linear(1, 1)
        self.proposal_head = torch.nn.Linear(1, 1)
        self.prediction_heads = torch.nn.Linear(1, 1)


class _DeviceNeutralTinyMcln(_TinyMcln):
    def to(self, device):
        self.requested_device = str(device)
        return self


def _checkpoint_config():
    return SimpleNamespace(
        use_color=True,
        use_height=False,
        use_multiview=False,
        detect_intermediate=True,
        num_decoder_layers=6,
        num_target=256,
        self_attend=True,
        self_position_embedding="loc_learned",
        use_soft_token_loss=True,
        use_contrastive_align=True,
        use_source_choice_selector=True,
        source_choice_selector_sources=(
            "default,default_rank_blend_contrastive010"
        ),
        source_choice_selector_hidden_dim=288,
        wo_obj_name="None",
        skip_missing_superpoints=True,
    )


def test_initial_state_loads_only_epoch71_weights_and_builds_fresh_adamw(
        tmp_path, monkeypatch):
    source_model = _TinyMcln()
    checkpoint = tmp_path / "epoch71.pth"
    torch.save({
        "epoch": 71,
        "config": _checkpoint_config(),
        "model": {
            "module." + name: value.detach().clone()
            for name, value in source_model.state_dict().items()
        },
        "optimizer": {"poison": torch.tensor(float("nan"))},
        "scheduler": {"poison": object()},
    }, checkpoint)
    parent_path = tmp_path / "parent.pth"
    geometry_path = tmp_path / "geometry.pth"
    parent_path.write_bytes(b"parent")
    geometry_path.write_bytes(b"geometry")
    actual_sha = finetune_runner.checkpoint_sha256(checkpoint)
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_BACKBONE_SHA256", actual_sha
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_PARENT_SHA256", "1" * 64
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_GEOMETRY_SHA256", "2" * 64
    )

    parent = torch.nn.Linear(1, 1)
    parent._artifact_sha256 = "1" * 64
    parent_artifact = {
        "checkpoint_sha256": actual_sha,
        "model_state_dict": {
            name: value.detach().clone()
            for name, value in parent.state_dict().items()
        },
    }
    geometry = torch.nn.Linear(1, 1)
    geometry._artifact_sha256 = "2" * 64
    geometry_artifact = {
        "checkpoint_sha256": actual_sha,
        "checkpoint_epoch": 71,
        "parent_artifact_sha256": "1" * 64,
        "model_state_dict": {
            name: value.detach().clone()
            for name, value in geometry.state_dict().items()
        },
    }
    calls = []

    def load_parent(path, device):
        calls.append(("parent", Path(path), str(device)))
        return parent, parent_artifact

    def load_geometry(path, device, parent_artifact_path=None):
        calls.append((
            "geometry", Path(path), str(device), Path(parent_artifact_path)
        ))
        return geometry, geometry_artifact

    def validate_geometry(artifact, parent=None, **_kwargs):
        calls.append(("validate", artifact, parent))

    monkeypatch.setattr(
        finetune_runner, "load_parent_reranker_snapshot", load_parent
    )
    monkeypatch.setattr(
        finetune_runner, "load_geometry_reranker_artifact", load_geometry
    )
    monkeypatch.setattr(
        finetune_runner, "validate_geometry_artifact", validate_geometry
    )
    seen_config = []

    def model_factory(config):
        seen_config.append(config)
        return _TinyMcln()

    state = finetune_runner.load_rec_finetune_initial_state(
        checkpoint, parent_path, geometry_path, tmp_path,
        device="cpu", model_factory=model_factory,
    )

    assert state["checkpoint_epoch"] == 71
    assert state["checkpoint_sha256"] == actual_sha
    assert torch.optim.AdamW is type(state["optimizer"])
    assert len(state["optimizer"].param_groups) == 3
    assert not state["optimizer"].state
    assert "scheduler" not in state
    assert state["mcln"].state_dict().keys() == source_model.state_dict().keys()
    for name, value in source_model.state_dict().items():
        assert torch.equal(state["mcln"].state_dict()[name], value)
    for model, artifact in (
            (state["parent"], parent_artifact),
            (state["geometry"], geometry_artifact)):
        for name, value in artifact["model_state_dict"].items():
            assert torch.equal(model.state_dict()[name], value)
    assert calls[0] == ("parent", parent_path.resolve(), "cpu")
    assert calls[1] == (
        "geometry", geometry_path.resolve(), "cpu", parent_path.resolve()
    )
    assert calls[2][0] == "validate"
    assert calls[2][2] == (parent, parent_artifact)
    config = seen_config[0]
    assert config.data_root.endswith("/")
    assert config.eval is False
    assert config.dataset == ["scanrefer"]
    assert config.test_dataset == "scanrefer"
    assert config.joint_det is False
    assert config.butd is True
    assert config.butd_gt is False
    assert config.butd_cls is False
    assert config.augment_det is True
    assert config.source_choice_selector_loss_weight == 0.0


@pytest.mark.parametrize("mutation", ("missing", "unexpected", "double"))
def test_backbone_model_state_is_loaded_strictly_after_one_prefix_strip(
        mutation):
    source = _TinyMcln()
    state = {
        "module." + name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    first_name = next(iter(source.state_dict()))
    prefixed_name = "module." + first_name
    if mutation == "missing":
        del state[prefixed_name]
    elif mutation == "unexpected":
        state["module.unexpected"] = torch.zeros(1)
    else:
        state["module.module." + first_name] = state.pop(prefixed_name)

    with pytest.raises(ValueError, match="strict model state"):
        finetune_runner._load_testable_model(
            {"model": state},
            _checkpoint_config(),
            torch.device("cpu"),
            lambda _config: _TinyMcln(),
        )


@pytest.mark.parametrize("failure", ("epoch70", "wrong-sha"))
def test_backbone_snapshot_rejects_non_authoritative_epoch_or_sha(
        tmp_path, monkeypatch, failure):
    source = _TinyMcln()
    checkpoint = tmp_path / "backbone.pth"
    torch.save({
        "epoch": 70 if failure == "epoch70" else 71,
        "config": _checkpoint_config(),
        "model": {
            "module." + name: value.detach().clone()
            for name, value in source.state_dict().items()
        },
        "optimizer": {"poison": object()},
        "scheduler": {"poison": object()},
    }, checkpoint)
    actual_sha = finetune_runner.checkpoint_sha256(checkpoint)
    expected = actual_sha if failure == "epoch70" else "0" * 64
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_BACKBONE_SHA256", expected
    )
    model_factory_called = []

    def forbidden_model_load(*_args, **_kwargs):
        model_factory_called.append(True)
        raise AssertionError("model construction must follow snapshot checks")

    monkeypatch.setattr(
        finetune_runner, "_load_testable_model", forbidden_model_load
    )

    with pytest.raises(ValueError, match="epoch 71|SHA-256"):
        finetune_runner.load_rec_finetune_initial_state(
            checkpoint,
            tmp_path / "unused-parent.pth",
            tmp_path / "unused-geometry.pth",
            tmp_path,
            device="cpu",
        )

    assert model_factory_called == []


class _DatasetSpy:
    calls = []

    def __init__(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        self.dataset_dict = kwargs["dataset_dict"]
        self.test_dataset = kwargs["test_dataset"]
        self.split = kwargs["split"]
        self.butd = kwargs.get("butd", False)
        self.butd_gt = kwargs.get("butd_gt", False)
        self.butd_cls = kwargs.get("butd_cls", False)
        self.joint_det = False
        self.annos = [
            {"scan_id": "scene_{:02d}".format(index // 2), "value": index}
            for index in range(26)
        ]
        self.scans = {row["scan_id"]: object() for row in self.annos}
        self.augment = True
        self.augment_det = kwargs["augment_det"]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        return {"value": self.annos[index]["value"]}


class _PrintingDatasetSpy(_DatasetSpy):
    def __init__(self, **kwargs):
        print("Joint3DDataset synthetic initialization detail")
        super().__init__(**kwargs)


def _workers_zero_loader(dataset, **kwargs):
    kwargs["num_workers"] = 0
    return torch.utils.data.DataLoader(dataset, **kwargs)


def test_train_data_uses_one_scanrefer_train_dataset_and_two_indexed_views():
    _DatasetSpy.calls = []
    probe = _DatasetSpy(
        dataset_dict={"scanrefer": 1}, test_dataset="scanrefer",
        split="train", augment_det=True,
    )
    split = rec_finetune.build_rec_finetune_scene_split([
        row["scan_id"] for row in probe.annos
    ])
    _DatasetSpy.calls = []
    config = _checkpoint_config()
    config.data_root = "/synthetic/train-root/"
    config.butd = True
    config.butd_gt = False
    config.butd_cls = False
    config.augment_det = True
    config.eval = False

    data = finetune_runner.build_train_only_data(
        config,
        device="cpu",
        dataset_factory=_DatasetSpy,
        loader_factory=_workers_zero_loader,
        expected_split_metadata=split["metadata"],
    )

    assert len(_DatasetSpy.calls) == 1
    call = _DatasetSpy.calls[0]
    assert call["dataset_dict"] == {"scanrefer": 1}
    assert call["test_dataset"] == "scanrefer"
    assert call["split"] == "train"
    assert call["butd"] is True
    assert call["butd_gt"] is False
    assert call["butd_cls"] is False
    assert call["augment_det"] is True
    source = data["dataset"]
    annos_before = [dict(row) for row in source.annos]
    scans_before = dict(source.scans)
    assert source.augment is True
    assert source.augment_det is True
    assert data["fit_view"].dataset is not source
    assert data["fit_view"].dataset.annos is source.annos
    assert data["fit_view"].dataset.scans is source.scans
    assert data["fit_view"].dataset.augment is True
    assert data["fit_view"].dataset.augment_det is True
    assert data["calibration_view"].dataset is not source
    assert data["calibration_view"].dataset.annos is source.annos
    assert data["calibration_view"].dataset.scans is source.scans
    assert data["calibration_view"].dataset.augment is False
    assert data["calibration_view"].dataset.augment_det is False
    assert source.annos == annos_before
    assert source.scans == scans_before

    calibration_indices = []
    calibration_batch_sizes = []
    for batch in data["calibration_loader"]:
        calibration_batch_sizes.append(len(batch["dataset_index"]))
        calibration_indices.extend(batch["dataset_index"].tolist())
    assert calibration_indices == list(split["calibration_indices"])
    assert calibration_batch_sizes[-1] <= 18

    fit_indices = []
    fit_batch_sizes = []
    for batch in data["fit_loader"]:
        fit_batch_sizes.append(len(batch["dataset_index"]))
        fit_indices.extend(batch["dataset_index"].tolist())
    assert sorted(fit_indices) == sorted(split["fit_indices"])
    assert fit_indices != sorted(fit_indices)
    assert fit_batch_sizes[-1] == len(split["fit_indices"]) % 18
    assert data["fit_loader"].batch_size == 18
    assert data["fit_loader"].drop_last is False
    assert data["calibration_loader"].batch_size == 18
    assert data["calibration_loader"].drop_last is False


def test_printing_dataset_factory_never_leaks_to_build_or_main_stdout(
        tmp_path, monkeypatch, capsys):
    probe = _DatasetSpy(
        dataset_dict={"scanrefer": 1}, test_dataset="scanrefer",
        split="train", augment_det=True,
    )
    split = rec_finetune.build_rec_finetune_scene_split([
        row["scan_id"] for row in probe.annos
    ])
    config = _checkpoint_config()
    config.data_root = "/synthetic/train-root/"
    config.dataset = ["scanrefer"]
    config.joint_det = False
    config.butd = True
    config.butd_gt = False
    config.butd_cls = False
    config.augment_det = True
    config.eval = False

    data = finetune_runner.build_train_only_data(
        config,
        device="cpu",
        dataset_factory=_PrintingDatasetSpy,
        loader_factory=_workers_zero_loader,
        expected_split_metadata=split["metadata"],
    )

    assert data["dataset"].__class__ is _PrintingDatasetSpy
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    args = finetune_runner.parse_args(_runner_argv(tmp_path))
    monkeypatch.setattr(
        finetune_runner,
        "load_rec_finetune_initial_state",
        lambda *_args, **_kwargs: {"config": config},
    )
    initialized = finetune_runner.initialize_rec_finetune_run(
        args,
        dataset_factory=_PrintingDatasetSpy,
        loader_factory=_workers_zero_loader,
        expected_split_metadata=split["metadata"],
    )

    assert initialized["train_data_contract"]["validation_data_accessed"] is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_train_data_contract_is_derived_from_live_train_only_objects():
    _DatasetSpy.calls = []
    probe = _DatasetSpy(
        dataset_dict={"scanrefer": 1}, test_dataset="scanrefer",
        split="train", augment_det=True,
    )
    split = rec_finetune.build_rec_finetune_scene_split([
        row["scan_id"] for row in probe.annos
    ])
    config = _checkpoint_config()
    config.data_root = "/synthetic/train-root/"
    config.dataset = ["scanrefer"]
    config.joint_det = False
    config.butd = True
    config.butd_gt = False
    config.butd_cls = False
    config.augment_det = True
    config.eval = False
    data = finetune_runner.build_train_only_data(
        config,
        device="cpu",
        dataset_factory=_DatasetSpy,
        loader_factory=_workers_zero_loader,
        expected_split_metadata=split["metadata"],
    )
    initialized = {"initial_state": {"config": config}, "data": data}

    contract = finetune_runner.build_rec_finetune_train_data_contract(
        initialized
    )

    assert set(contract) == {
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
    assert contract == {
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
        "authoritative_split_metadata": split["metadata"],
        "authoritative_split_mapping_sha256": split["metadata"][
            "mapping_sha256"
        ],
        "fit_sample_count": len(split["fit_indices"]),
        "calibration_sample_count": len(split["calibration_indices"]),
        "fit_loader_batch_count": len(data["fit_loader"]),
        "calibration_loader_batch_count": len(data["calibration_loader"]),
        "batch_size": 18,
        "drop_last": False,
        "validation_data_accessed": False,
        "dataset_class": (
            "test_train_scanrefer_rec_finetune._DatasetSpy"
        ),
        "dataset_instance_count": 1,
        "fit_and_calibration_share_source_annotations": True,
        "validation_data_objects_present": False,
    }
    json.dumps(contract, allow_nan=False)

    data["calibration_view"].dataset.augment = True
    with pytest.raises(ValueError, match="calibration|augment|live"):
        finetune_runner.build_rec_finetune_train_data_contract(initialized)
    data["calibration_view"].dataset.augment = False
    data["validation_loader"] = _NeverAccess()
    with pytest.raises(ValueError, match="sole|validation|train-only"):
        finetune_runner.build_rec_finetune_train_data_contract(initialized)


def test_initializer_validates_all_inputs_before_data_and_never_creates_output(
        tmp_path, monkeypatch):
    args = finetune_runner.parse_args(_runner_argv(tmp_path))
    events = []

    def load_state(*_args, **_kwargs):
        events.append("weights")
        return {"config": _checkpoint_config()}

    def build_data(*_args, **_kwargs):
        events.append("train-data")
        return {"split": {"metadata": {}}}

    train_data_contract = {
        "schema": "scanrefer-rec-finetune-train-data-v1"
    }

    def bind_train_data(initialized):
        events.append("train-data-contract")
        assert initialized["initial_state"]["config"] is not None
        assert initialized["data"] == {"split": {"metadata": {}}}
        return train_data_contract

    monkeypatch.setattr(
        finetune_runner, "load_rec_finetune_initial_state", load_state
    )
    monkeypatch.setattr(
        finetune_runner, "build_train_only_data", build_data
    )
    monkeypatch.setattr(
        finetune_runner,
        "build_rec_finetune_train_data_contract",
        bind_train_data,
    )

    initialized = finetune_runner.initialize_rec_finetune_run(args)

    assert events == ["weights", "train-data", "train-data-contract"]
    assert initialized["train_data_contract"] is train_data_contract
    assert initialized["paths"].output_dir == Path(args.output_dir).resolve()
    assert not Path(args.output_dir).exists()

    bad = SimpleNamespace(**vars(args))
    bad.geometry_reranker = str(tmp_path / "missing-geometry.pth")
    events[:] = []
    with pytest.raises(ValueError, match="geometry"):
        finetune_runner.initialize_rec_finetune_run(bad)
    assert events == []
    assert not Path(args.output_dir).exists()


def test_real_initializer_orchestration_never_accesses_validation_or_caches(
        tmp_path, monkeypatch):
    import scripts.cache_scanrefer_rec_candidates as candidate_cache
    import scripts.rec_geometry_cache as geometry_cache

    args = finetune_runner.parse_args(_runner_argv(tmp_path))
    source_model = _TinyMcln()
    torch.save({
        "epoch": 71,
        "config": _checkpoint_config(),
        "model": {
            "module." + name: value.detach().clone()
            for name, value in source_model.state_dict().items()
        },
        "optimizer": {"poison": torch.tensor(float("nan"))},
        "scheduler": {"poison": object()},
    }, args.backbone_checkpoint)
    actual_sha = finetune_runner.checkpoint_sha256(
        args.backbone_checkpoint
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_BACKBONE_SHA256", actual_sha
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_PARENT_SHA256", "1" * 64
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_GEOMETRY_SHA256", "2" * 64
    )

    parent = torch.nn.Linear(1, 1)
    parent._artifact_sha256 = "1" * 64
    parent_artifact = {
        "checkpoint_sha256": actual_sha,
        "model_state_dict": {
            name: value.detach().clone()
            for name, value in parent.state_dict().items()
        },
    }
    geometry = torch.nn.Linear(1, 1)
    geometry._artifact_sha256 = "2" * 64
    geometry_artifact = {
        "checkpoint_sha256": actual_sha,
        "checkpoint_epoch": 71,
        "parent_artifact_sha256": "1" * 64,
        "model_state_dict": {
            name: value.detach().clone()
            for name, value in geometry.state_dict().items()
        },
    }
    loader_calls = []

    def load_parent(path, device):
        loader_calls.append(("parent", Path(path), str(device)))
        return parent, parent_artifact

    def load_geometry(path, device, parent_artifact_path=None):
        loader_calls.append((
            "geometry", Path(path), str(device), Path(parent_artifact_path)
        ))
        return geometry, geometry_artifact

    def validate_geometry(artifact, parent=None, **_kwargs):
        loader_calls.append(("validate", artifact, parent))

    probe = _DatasetSpy(
        dataset_dict={"scanrefer": 1}, test_dataset="scanrefer",
        split="train", augment_det=True,
    )
    expected_split = rec_finetune.build_rec_finetune_scene_split([
        row["scan_id"] for row in probe.annos
    ])["metadata"]
    _DatasetSpy.calls = []

    def train_dataset_only(**kwargs):
        assert kwargs["split"] == "train"
        assert kwargs["dataset_dict"] == {"scanrefer": 1}
        assert kwargs["test_dataset"] == "scanrefer"
        return _DatasetSpy(**kwargs)

    def forbidden_cache_loader(*_args, **_kwargs):
        raise AssertionError("validation/cache loader must not run")

    for module, names in (
            (candidate_cache, ("_build_dataset", "_build_loader")),
            (geometry_cache, (
                "load_bound_candidate_cache", "load_geometry_cache",
            ))):
        for name in names:
            monkeypatch.setattr(
                module, name, forbidden_cache_loader, raising=False
            )

    accessed = []
    original_builtin_open = builtins.open
    original_os_open = os.open
    original_torch_load = torch.load

    def record_path(value):
        if isinstance(value, (str, os.PathLike)):
            path = os.fspath(value)
            lowered = path.lower().replace("\\", "/")
            accessed.append(lowered)
            assert "/val/" not in lowered
            assert "geometry_val" not in lowered
            assert "official_result" not in lowered

    def builtin_open_spy(path, *open_args, **open_kwargs):
        record_path(path)
        return original_builtin_open(path, *open_args, **open_kwargs)

    def os_open_spy(path, *open_args, **open_kwargs):
        record_path(path)
        return original_os_open(path, *open_args, **open_kwargs)

    def torch_load_spy(source, *load_args, **load_kwargs):
        record_path(source)
        return original_torch_load(source, *load_args, **load_kwargs)

    monkeypatch.setattr(builtins, "open", builtin_open_spy)
    monkeypatch.setattr(os, "open", os_open_spy)
    monkeypatch.setattr(torch, "load", torch_load_spy)

    initialized = finetune_runner.initialize_rec_finetune_run(
        args,
        model_factory=lambda _config: _DeviceNeutralTinyMcln(),
        dataset_factory=train_dataset_only,
        loader_factory=_workers_zero_loader,
        expected_split_metadata=expected_split,
        parent_loader=load_parent,
        geometry_loader=load_geometry,
        geometry_validator=validate_geometry,
    )

    assert len(_DatasetSpy.calls) == 1
    assert _DatasetSpy.calls[0]["split"] == "train"
    assert loader_calls[0] == (
        "parent", Path(args.parent_reranker).resolve(), "cuda:0"
    )
    assert loader_calls[1] == (
        "geometry", Path(args.geometry_reranker).resolve(), "cuda:0",
        Path(args.parent_reranker).resolve(),
    )
    assert loader_calls[2][0] == "validate"
    assert initialized["initial_state"]["mcln"].requested_device == "cuda:0"
    assert not initialized["initial_state"]["optimizer"].state
    assert not Path(args.output_dir).exists()
    assert not any(
        token in path
        for path in accessed
        for token in ("/val/", "geometry_val", "official_result")
    )

def test_scene_split_and_calibration_contracts_are_public():
    assert callable(rec_finetune.build_rec_finetune_scene_split)
    assert callable(rec_finetune.CalibrationAccumulator)
    assert callable(rec_finetune.CalibrationSelector)


def test_real_train_scene_split_matches_authoritative_seed0_metadata():
    if not TRAIN_ANNOTATIONS.is_file():
        pytest.skip("ScanRefer train annotation is unavailable")
    rows = json.loads(TRAIN_ANNOTATIONS.read_text(encoding="utf-8"))

    split = rec_finetune.build_rec_finetune_scene_split([
        row["scene_id"] for row in rows
    ])

    assert split["metadata"] == EXPECTED_SEED0_SPLIT
    assert (
        rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
        == EXPECTED_SEED0_SPLIT
    )
    fit_indices = split["fit_indices"]
    calibration_indices = split["calibration_indices"]
    assert set(fit_indices).isdisjoint(calibration_indices)
    assert sorted(fit_indices + calibration_indices) == list(range(len(rows)))
    assert list(fit_indices) == sorted(fit_indices)
    assert list(calibration_indices) == sorted(calibration_indices)
    assert set(split["fit_scenes"]).isdisjoint(split["calibration_scenes"])


def test_scene_split_is_scene_disjoint_and_preserves_annotation_order():
    scan_ids = ("scene_b", "scene_a", "scene_b", "scene_c", "scene_a")

    split = rec_finetune.build_rec_finetune_scene_split(
        scan_ids, seed=7, calibration_fraction=0.34
    )

    fit = split["fit_indices"]
    calibration = split["calibration_indices"]
    assert list(fit) == sorted(fit)
    assert list(calibration) == sorted(calibration)
    assert set(fit).isdisjoint(calibration)
    assert sorted(fit + calibration) == list(range(len(scan_ids)))
    assert {scan_ids[index] for index in fit} == set(split["fit_scenes"])
    assert {
        scan_ids[index] for index in calibration
    } == set(split["calibration_scenes"])
    assert tuple(split["fit_scenes"]) == tuple(sorted(split["fit_scenes"]))
    assert tuple(split["calibration_scenes"]) == tuple(
        sorted(split["calibration_scenes"])
    )


def test_scene_split_rejects_bad_inputs_and_keeps_a_single_scene_in_fit():
    one_scene = rec_finetune.build_rec_finetune_scene_split(
        ["scene_a", "scene_a"]
    )
    assert one_scene["fit_indices"] == (0, 1)
    assert one_scene["calibration_indices"] == ()

    with pytest.raises(ValueError):
        rec_finetune.build_rec_finetune_scene_split([])
    with pytest.raises(ValueError):
        rec_finetune.build_rec_finetune_scene_split(["scene_a", ""])
    with pytest.raises(ValueError):
        rec_finetune.build_rec_finetune_scene_split(["scene_a"], seed=True)
    with pytest.raises(ValueError):
        rec_finetune.build_rec_finetune_scene_split(
            ["scene_a", "scene_b"], calibration_fraction=1.0
        )


def test_calibration_accumulator_uses_order_and_strict_iou_thresholds():
    accumulator = rec_finetune.CalibrationAccumulator((9, 3, 7, 4, 8))
    accumulator.update((9, 3), torch.tensor([0.25, 0.50]))
    accumulator.update((7, 4, 8), torch.tensor([0.2501, 0.5001, 1.0]))

    metrics = accumulator.finalize()

    assert metrics["sample_count"] == 5
    assert metrics["hits025"] == 4
    assert metrics["hits050"] == 2
    assert metrics["acc025"] == pytest.approx(0.8)
    assert metrics["acc050"] == pytest.approx(0.4)
    assert metrics["score"] == pytest.approx(
        min(0.8 / 0.60, 0.4 / 0.47) + 0.1 * (0.8 + 0.4)
    )


def test_calibration_accumulator_rejects_incomplete_and_out_of_order_batches():
    accumulator = rec_finetune.CalibrationAccumulator((2, 5, 1))
    with pytest.raises(ValueError, match="order"):
        accumulator.update((5,), torch.tensor([0.8]))

    accumulator.update((2, 5), torch.tensor([0.2, 0.7]))
    with pytest.raises(ValueError, match="incomplete"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="length"):
        accumulator.update((1,), torch.tensor([0.8, 0.9]))


def test_calibration_accumulator_rejects_malformed_iou_tensors():
    accumulator = rec_finetune.CalibrationAccumulator((0,))
    with pytest.raises(ValueError, match="float tensor"):
        accumulator.update((0,), torch.tensor([1]))
    with pytest.raises(ValueError, match="finite"):
        accumulator.update((0,), torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        accumulator.update((0,), torch.tensor([1.01]))


DIAGNOSTIC_BRANCH_NAMES = (
    "default_top1",
    "source_selector_top1",
    "parent_top1",
    "geometry_top1",
    "raw_query_oracle",
    "parent_candidate_oracle",
    "geometry_candidate_oracle",
)


def _diagnostic_branch_ious():
    return {
        "default_top1": torch.tensor([
            0.10, 0.30, 0.20, 0.40, 0.60, 0.20, 0.70, 0.80, 0.10,
        ]),
        "source_selector_top1": torch.tensor([
            0.30, 0.20, 0.20, 0.60, 0.40, 0.70, 0.70, 0.10, 0.90,
        ]),
        "parent_top1": torch.tensor([
            0.20, 0.40, 0.60, 0.10, 0.70, 0.80, 0.20, 0.90, 0.30,
        ]),
        "geometry_top1": torch.tensor([
            0.10, 0.15, 0.25, 0.26, 0.30, 0.50, 0.60, 0.75, 1.00,
        ]),
        "raw_query_oracle": torch.tensor([
            0.30, 0.60, 0.60, 0.70, 0.80, 0.80, 0.90, 0.90, 1.00,
        ]),
        "parent_candidate_oracle": torch.tensor([
            0.20, 0.40, 0.60, 0.60, 0.70, 0.80, 0.80, 0.90, 0.90,
        ]),
        "geometry_candidate_oracle": torch.tensor([
            0.225, 0.40, 0.625, 0.60, 0.70, 0.80, 0.80, 0.90, 1.00,
        ]),
    }


def _build_diagnostic_result(branch_ious=None, expected_indices=None):
    branch_ious = branch_ious or _diagnostic_branch_ious()
    sample_count = len(next(iter(branch_ious.values())))
    expected_indices = expected_indices or tuple(range(10, 10 + sample_count))
    accumulator = rec_finetune.CalibrationDiagnosticsAccumulator(
        expected_indices
    )
    split = min(4, sample_count)
    accumulator.update(
        expected_indices[:split],
        {name: values[:split] for name, values in branch_ious.items()},
    )
    if split < sample_count:
        accumulator.update(
            expected_indices[split:],
            {name: values[split:] for name, values in branch_ious.items()},
        )
    return accumulator.finalize()


def test_calibration_diagnostics_accumulates_strict_metrics_and_fixed_bins():
    result = _build_diagnostic_result()
    diagnostics = result.diagnostics

    assert set(diagnostics) == {
        "schema", "sample_count", "candidate_oracle", "stages",
        "effects", "selected_iou", "geometry_oracle_selected_regret",
        "recoverable_misses", "selected_oracle_regret_cells",
    }
    assert diagnostics["schema"] == "rec-finetune-calibration-diagnostics-v3"
    assert diagnostics["sample_count"] == 9
    assert diagnostics["candidate_oracle"] == {
        "raw_query": {
            "hits025": 9, "hits050": 8,
            "acc025": pytest.approx(1.0),
            "acc050": pytest.approx(8 / 9),
        },
        "parent_candidate": {
            "hits025": 8, "hits050": 7,
            "acc025": pytest.approx(8 / 9),
            "acc050": pytest.approx(7 / 9),
        },
        "geometry_candidate": {
            "hits025": 8, "hits050": 7,
            "acc025": pytest.approx(8 / 9),
            "acc050": pytest.approx(7 / 9),
        },
    }
    assert diagnostics["stages"]["geometry_top1"] == {
        "hits025": 6,
        "hits050": 3,
        "acc025": pytest.approx(6 / 9),
        "acc050": pytest.approx(3 / 9),
    }
    assert diagnostics["selected_iou"] == {
        "bins": {
            "le_010": 1,
            "gt_010_le_020": 1,
            "gt_020_le_025": 1,
            "gt_025_le_030": 2,
            "gt_030_le_050": 1,
            "gt_050_le_075": 2,
            "gt_075_le_100": 1,
        },
    }
    assert sum(diagnostics["selected_iou"]["bins"].values()) == 9
    cells = diagnostics["selected_oracle_regret_cells"]
    selected_names = tuple(
        spec[0] for spec in rec_finetune.CALIBRATION_SELECTED_IOU_BIN_SPECS
    )
    oracle_names = tuple(
        spec[0] for spec in rec_finetune.CALIBRATION_ORACLE_TIER_SPECS
    )
    regret_names = tuple(
        spec[0] for spec in rec_finetune.CALIBRATION_REGRET_BAND_SPECS
    )
    assert set(cells) == set(selected_names)
    assert all(set(cells[name]) == set(oracle_names) for name in selected_names)
    assert all(
        set(cells[selected][oracle]) == set(regret_names)
        for selected in selected_names for oracle in oracle_names
    )
    assert all(
        set(cell) == {"count"}
        for selected in cells.values()
        for oracle in selected.values()
        for cell in oracle.values()
    )
    assert sum(
        cell["count"]
        for selected in cells.values()
        for oracle in selected.values()
        for cell in oracle.values()
    ) == 9
    for selected_name in selected_names:
        assert sum(
            cell["count"]
            for oracle in cells[selected_name].values()
            for cell in oracle.values()
        ) == diagnostics["selected_iou"]["bins"][selected_name]
    assert {
        oracle_name: sum(
            cell["count"]
            for selected in cells.values()
            for cell in selected[oracle_name].values()
        )
        for oracle_name in oracle_names
    } == {"o0": 1, "o1": 1, "o2": 7}
    regret = diagnostics["geometry_oracle_selected_regret"]
    assert {
        regret_name: sum(
            oracle[regret_name]["count"]
            for selected in cells.values()
            for oracle in selected.values()
        )
        for regret_name in regret_names
    } == {
        "zero": 9 - regret["positive_count"],
        "gt_000_lt_005": (
            regret["positive_count"] - regret["ge005_count"]
        ),
        "ge_005_lt_010": (
            regret["ge005_count"] - regret["ge010_count"]
        ),
        "ge_010": regret["ge010_count"],
    }


def test_calibration_diagnostics_reports_effects_regret_and_recoverable_misses():
    diagnostics = _build_diagnostic_result().diagnostics

    assert diagnostics["effects"] == {
        "source_selector_vs_default": {
            "fixes025": 3, "breaks025": 2,
            "fixes050": 3, "breaks050": 2,
        },
        "parent_vs_default": {
            "fixes025": 3, "breaks025": 2,
            "fixes050": 2, "breaks050": 1,
        },
        "geometry_vs_parent": {
            "fixes025": 2, "breaks025": 2,
            "fixes050": 2, "breaks050": 3,
        },
    }
    for effect_name, (old_name, new_name) in {
            "source_selector_vs_default": (
                "default_top1", "source_selector_top1"),
            "parent_vs_default": ("default_top1", "parent_top1"),
            "geometry_vs_parent": ("parent_top1", "geometry_top1"),
            }.items():
        effect = diagnostics["effects"][effect_name]
        for suffix in ("025", "050"):
            assert (
                diagnostics["stages"][new_name]["hits" + suffix]
                - diagnostics["stages"][old_name]["hits" + suffix]
            ) == effect["fixes" + suffix] - effect["breaks" + suffix]

    assert diagnostics["geometry_oracle_selected_regret"] == {
        "positive_count": 8,
        "ge005_count": 8,
        "ge010_count": 8,
    }
    assert diagnostics["recoverable_misses"] == {
        "at025": 2,
        "at050": 4,
    }


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda values: values.pop("geometry_top1"), "fields"),
        (lambda values: values.update(extra=torch.zeros(9)), "fields"),
        (
            lambda values: values.update(
                geometry_top1=torch.zeros(9, 1)
            ),
            "one-dimensional float tensor",
        ),
        (
            lambda values: values.update(
                geometry_top1=torch.zeros(9, dtype=torch.long)
            ),
            "one-dimensional float tensor",
        ),
        (
            lambda values: values.update(
                geometry_top1=torch.zeros(8)
            ),
            "length",
        ),
        (
            lambda values: values["geometry_top1"].__setitem__(
                0, float("nan")
            ),
            "finite",
        ),
        (
            lambda values: values["geometry_top1"].__setitem__(0, 1.01),
            r"\[0, 1\]",
        ),
    ],
)
def test_calibration_diagnostics_rejects_malformed_branch_ious(
        mutation, error):
    values = _diagnostic_branch_ious()
    mutation(values)
    accumulator = rec_finetune.CalibrationDiagnosticsAccumulator(tuple(range(9)))

    with pytest.raises(ValueError, match=error):
        accumulator.update(tuple(range(9)), values)


def test_calibration_diagnostics_rejects_order_and_incomplete_passes():
    values = _diagnostic_branch_ious()
    accumulator = rec_finetune.CalibrationDiagnosticsAccumulator(tuple(range(9)))

    with pytest.raises(ValueError, match="order"):
        accumulator.update(
            (1,), {name: branch[:1] for name, branch in values.items()}
        )
    accumulator.update(
        tuple(range(4)), {name: branch[:4] for name, branch in values.items()}
    )
    with pytest.raises(ValueError, match="incomplete"):
        accumulator.finalize()


@pytest.mark.parametrize(
    "oracle_name,stage_name,row",
    [
        ("raw_query_oracle", "default_top1", 0),
        ("raw_query_oracle", "source_selector_top1", 8),
        ("raw_query_oracle", "parent_top1", 2),
        ("parent_candidate_oracle", "default_top1", 6),
        ("parent_candidate_oracle", "parent_top1", 7),
        ("geometry_candidate_oracle", "default_top1", 6),
        ("geometry_candidate_oracle", "parent_top1", 7),
        ("geometry_candidate_oracle", "geometry_top1", 8),
    ],
)
def test_calibration_diagnostics_oracles_fail_closed_and_update_is_atomic(
        oracle_name, stage_name, row):
    valid = _diagnostic_branch_ious()
    invalid = {
        name: values.clone() for name, values in valid.items()
    }
    invalid[oracle_name][row] = invalid[stage_name][row] - 0.01
    indices = tuple(range(9))
    accumulator = rec_finetune.CalibrationDiagnosticsAccumulator(indices)

    with pytest.raises(ValueError, match="oracle.*lower"):
        accumulator.update(indices, invalid)

    accumulator.update(indices, valid)
    result = accumulator.finalize()
    assert result.diagnostics["sample_count"] == 9
    assert result.transition_state.expected_indices == indices


@pytest.mark.parametrize(
    "outer_oracle",
    ["raw_query_oracle", "geometry_candidate_oracle"],
)
def test_calibration_diagnostics_parent_oracle_is_nested_and_update_is_atomic(
        outer_oracle):
    indices = (0,)
    invalid = {
        name: torch.tensor([0.50], dtype=torch.float64)
        for name in DIAGNOSTIC_BRANCH_NAMES
    }
    for name in (
            "raw_query_oracle", "parent_candidate_oracle",
            "geometry_candidate_oracle"):
        invalid[name] = torch.tensor([0.90], dtype=torch.float64)
    invalid[outer_oracle] = torch.tensor([0.80], dtype=torch.float64)
    accumulator = rec_finetune.CalibrationDiagnosticsAccumulator(indices)

    with pytest.raises(ValueError, match="oracle.*lower"):
        accumulator.update(indices, invalid)

    valid = {name: values.clone() for name, values in invalid.items()}
    valid[outer_oracle] = torch.tensor([0.90], dtype=torch.float64)
    accumulator.update(indices, valid)
    assert accumulator.finalize().diagnostics["sample_count"] == 1


def test_calibration_diagnostics_canonicalizes_only_tiny_oracle_roundoff():
    selected = torch.tensor([0.90], dtype=torch.float64)
    within_tolerance = selected - 0.0000005
    branches = {
        name: selected.clone() for name in DIAGNOSTIC_BRANCH_NAMES
    }
    for name in (
            "raw_query_oracle", "parent_candidate_oracle",
            "geometry_candidate_oracle"):
        branches[name] = within_tolerance.clone()

    result = _build_diagnostic_result(branches, (4,))

    assert result.diagnostics["geometry_oracle_selected_regret"] == {
        "positive_count": 0,
        "ge005_count": 0,
        "ge010_count": 0,
    }
    assert result.transition_state.geometry_oracle_ious == pytest.approx(
        result.transition_state.selected_ious
    )


def test_calibration_diagnostics_rejects_oracle_gap_beyond_tolerance():
    branches = {
        name: torch.tensor([0.90], dtype=torch.float64)
        for name in DIAGNOSTIC_BRANCH_NAMES
    }
    branches["geometry_candidate_oracle"] = torch.tensor(
        [0.90 - 0.000002], dtype=torch.float64
    )
    accumulator = rec_finetune.CalibrationDiagnosticsAccumulator((0,))

    with pytest.raises(ValueError, match="geometry.*oracle.*lower"):
        accumulator.update((0,), branches)


def _transition_result(selected_ious, oracle_ious, indices):
    selected = torch.as_tensor(selected_ious).clone()
    oracle = torch.as_tensor(oracle_ious).clone()
    branches = {
        name: selected.clone() for name in DIAGNOSTIC_BRANCH_NAMES
    }
    branches["raw_query_oracle"] = torch.maximum(selected, oracle)
    branches["parent_candidate_oracle"] = torch.maximum(selected, oracle)
    branches["geometry_candidate_oracle"] = oracle
    return _build_diagnostic_result(branches, tuple(indices))


def _expected_joint_transition(*nonzero_entries):
    state_names = tuple(
        name for name, _selected, _oracle
        in rec_finetune.CALIBRATION_JOINT_STATE_TIERS
    )
    result = {
        previous: {current: 0 for current in state_names}
        for previous in state_names
    }
    for previous, current, count in nonzero_entries:
        result[previous][current] = count
    return result


def test_calibration_step_transition_reports_selected_and_oracle_gain_loss():
    previous = _transition_result(
        [0.10, 0.30, 0.60, 0.20, 0.80],
        [0.20, 0.40, 0.70, 0.80, 0.90],
        (5, 1, 8, 3, 2),
    )
    current = _transition_result(
        [0.60, 0.20, 0.70, 0.40, 0.50],
        [0.60, 0.20, 0.80, 0.50, 0.90],
        (5, 1, 8, 3, 2),
    )

    transition = rec_finetune.build_calibration_step_transition(
        previous.transition_state,
        current.transition_state,
        previous_step=0,
        current_step=306,
    )

    assert transition == {
        "schema": "rec-finetune-calibration-step-transition-v2",
        "previous_step": 0,
        "current_step": 306,
        "sample_count": 5,
        "selected": {
            "gained025": 2, "lost025": 1,
            "gained050": 1, "lost050": 1,
        },
        "geometry_oracle": {
            "gained025": 1, "lost025": 1,
            "gained050": 1, "lost050": 1,
        },
        "selected_oracle_joint": _expected_joint_transition(
            ("s0_o0", "s2_o2", 1),
            ("s1_o1", "s0_o0", 1),
            ("s2_o2", "s2_o2", 1),
            ("s0_o2", "s1_o1", 1),
            ("s2_o2", "s1_o2", 1),
        ),
    }


def test_calibration_step_transition_rejects_different_index_order():
    previous = _transition_result(
        [0.2, 0.3], [0.4, 0.5], (2, 7)
    )
    current = _transition_result(
        [0.3, 0.4], [0.5, 0.6], (7, 2)
    )

    with pytest.raises(ValueError, match="indices"):
        rec_finetune.build_calibration_step_transition(
            previous.transition_state,
            current.transition_state,
            previous_step=0,
            current_step=1,
        )


def test_calibration_step_transition_rejects_forged_oracle_state():
    forged = rec_finetune.CalibrationDiagnosticsTransitionState(
        expected_indices=(0,),
        selected_ious=(0.90,),
        geometry_oracle_ious=(0.10,),
    )
    valid = rec_finetune.CalibrationDiagnosticsTransitionState(
        expected_indices=(0,),
        selected_ious=(0.90,),
        geometry_oracle_ious=(0.90,),
    )

    with pytest.raises(ValueError, match="geometry.*oracle.*lower"):
        rec_finetune.build_calibration_step_transition(
            forged, valid, previous_step=0, current_step=1
        )


def test_calibration_diagnostics_public_payload_is_finite_json_without_rows():
    diagnostics = _build_diagnostic_result().diagnostics

    encoded = json.dumps(diagnostics, allow_nan=False, sort_keys=True)

    assert "selected_ious" not in encoded
    assert "geometry_oracle_ious" not in encoded
    assert not any(
        isinstance(value, torch.Tensor)
        for section in diagnostics.values()
        if isinstance(section, dict)
        for value in section.values()
    )


def test_public_diagnostic_validator_accepts_nontrivial_json_roundtrip():
    indices = tuple(range(9))
    branch_ious = _diagnostic_branch_ious()
    result = _build_diagnostic_result(branch_ious, indices)
    selection = rec_finetune.CalibrationAccumulator(indices)
    selection.update(indices, branch_ious["geometry_top1"])
    metrics = selection.finalize()
    transition = rec_finetune.build_calibration_step_transition(
        result.transition_state, result.transition_state, 0, 1
    )
    diagnostics = copy.deepcopy(result.diagnostics)
    run_result = {
        "selected_step": 0,
        "calibration_diagnostics_history": [
            {
                "step": 0,
                "diagnostics": diagnostics,
                "transition_from_previous": None,
            },
            {
                "step": 1,
                "diagnostics": copy.deepcopy(diagnostics),
                "transition_from_previous": transition,
            },
        ],
        "selected_calibration_diagnostics": copy.deepcopy(diagnostics),
        "reproduced_calibration_diagnostics": copy.deepcopy(diagnostics),
    }
    calibration_history = [
        {"step": 0, "metrics": metrics},
        {"step": 1, "metrics": copy.deepcopy(metrics)},
    ]
    roundtripped = json.loads(json.dumps(run_result, allow_nan=False))

    finetune_runner._validate_calibration_diagnostics_history(
        roundtripped, calibration_history
    )


def test_public_diagnostic_validator_accepts_valid_reproduction_oracle_drift():
    indices = (4, 7, 9, 12)
    selected_observation = _synthetic_calibration_observation(
        [0.20, 0.30, 0.60, 0.80],
        [0.40, 0.40, 0.70, 0.90],
        indices,
    )
    reproduced_observation = _synthetic_calibration_observation(
        [0.20, 0.30, 0.60, 0.80],
        [0.90, 0.90, 0.90, 0.90],
        indices,
    )
    metrics = selected_observation.selection_metrics
    selected = selected_observation.diagnostics_result.diagnostics
    reproduced = reproduced_observation.diagnostics_result.diagnostics
    run_result = {
        "selected_step": 0,
        "calibration_diagnostics_history": [{
            "step": 0,
            "diagnostics": copy.deepcopy(selected),
            "transition_from_previous": None,
        }],
        "selected_calibration_diagnostics": copy.deepcopy(selected),
        "reproduced_calibration_diagnostics": copy.deepcopy(reproduced),
    }

    finetune_runner._validate_calibration_diagnostics_history(
        json.loads(json.dumps(run_result, allow_nan=False)),
        [{"step": 0, "metrics": metrics}],
    )


def test_public_diagnostic_validator_rejects_exact_boundary_regret_cell():
    observation = _synthetic_calibration_observation(
        [0.20], [0.26], (0,)
    )
    diagnostic = copy.deepcopy(observation.diagnostics_result.diagnostics)
    regret = diagnostic["geometry_oracle_selected_regret"]
    cells = diagnostic["selected_oracle_regret_cells"][
        "gt_010_le_020"
    ]["o1"]
    assert cells["ge_005_lt_010"]["count"] == 1

    cells["ge_005_lt_010"]["count"] = 0
    cells["gt_000_lt_005"]["count"] = 1
    regret["ge005_count"] = 0

    with pytest.raises(ValueError, match="diagnostic.*cell.*impossible"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 1, "exact-boundary-cell-test"
        )


def test_diagnostic_endpoint_units_are_decimal_context_independent():
    with localcontext() as context:
        context.prec = 1
        assert finetune_runner._diagnostic_endpoint_percent_units(0.25) == 25

    with localcontext() as context:
        context.prec = 2
        with pytest.raises(ValueError, match="endpoint.*percent-scaled"):
            finetune_runner._diagnostic_endpoint_percent_units(0.251)


@pytest.mark.parametrize("suffix", ["025", "050"])
def test_public_diagnostic_validator_binds_recoverable_to_positive_regret(
        suffix):
    metrics = _metrics(0.50, 0.40, sample_count=10)
    diagnostic = copy.deepcopy(
        _synthetic_observation_for_metrics(metrics)
        .diagnostics_result.diagnostics
    )
    oracle = diagnostic["candidate_oracle"]["geometry_candidate"]
    _set_diagnostic_threshold_hits(
        oracle, suffix, oracle["hits" + suffix] + 1, 10
    )
    diagnostic["recoverable_misses"]["at" + suffix] = 1

    with pytest.raises(ValueError, match="diagnostic.*regret|recoverable"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 10, "recoverable-test"
        )


def test_public_diagnostic_validator_rejects_wide_gap_small_regret_cell():
    observation = _synthetic_calibration_observation(
        [0.10], [0.60], (0,)
    )
    diagnostic = copy.deepcopy(observation.diagnostics_result.diagnostics)
    regret = diagnostic["geometry_oracle_selected_regret"]
    assert diagnostic["recoverable_misses"] == {"at025": 1, "at050": 1}
    cells = diagnostic["selected_oracle_regret_cells"]["le_010"]["o2"]
    assert cells["ge_010"]["count"] == 1

    cells["ge_010"]["count"] = 0
    cells["ge_005_lt_010"]["count"] = 1
    regret["ge010_count"] = 0

    with pytest.raises(ValueError, match="diagnostic.*cell.*impossible"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 1, "wide-gap-cell-test"
        )


def test_public_diagnostic_validator_rejects_extra_continuous_field():
    observation = _synthetic_calibration_observation([0.10], [0.10], (0,))
    diagnostic = copy.deepcopy(observation.diagnostics_result.diagnostics)
    diagnostic["selected_iou"]["mean"] = 0.10

    with pytest.raises(ValueError, match="diagnostic.*selected IoU schema"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 1, "extra-continuous-field-test"
        )


@pytest.mark.parametrize("lower_oracle", ["raw_query", "geometry_candidate"])
def test_public_diagnostic_validator_requires_parent_oracle_nesting(
        lower_oracle):
    selected = torch.tensor([0.10])
    branches = {
        name: selected.clone() for name in DIAGNOSTIC_BRANCH_NAMES
    }
    if lower_oracle == "raw_query":
        branches["geometry_candidate_oracle"] = torch.tensor([0.30])
    else:
        branches["raw_query_oracle"] = torch.tensor([0.30])
    diagnostic = copy.deepcopy(
        _build_diagnostic_result(branches, (0,)).diagnostics
    )
    parent = diagnostic["candidate_oracle"]["parent_candidate"]
    parent["hits025"] = 1
    parent["acc025"] = 1.0

    with pytest.raises(ValueError, match="oracle.*parent|parent.*oracle"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 1, "oracle-nesting-test"
        )


def test_public_diagnostic_validator_requires_oracle_to_cover_stage_union():
    branches = {
        name: torch.tensor([0.10, 0.10])
        for name in DIAGNOSTIC_BRANCH_NAMES
    }
    branches["default_top1"] = torch.tensor([0.80, 0.10])
    branches["source_selector_top1"] = branches["default_top1"].clone()
    branches["parent_top1"] = torch.tensor([0.10, 0.80])
    branches["geometry_top1"] = branches["parent_top1"].clone()
    for name in (
            "raw_query_oracle", "parent_candidate_oracle",
            "geometry_candidate_oracle"):
        branches[name] = torch.tensor([0.80, 0.80])
    diagnostic = copy.deepcopy(
        _build_diagnostic_result(branches, (0, 1)).diagnostics
    )
    effect = diagnostic["effects"]["parent_vs_default"]
    assert effect["fixes025"] == effect["breaks025"] == 1
    parent = diagnostic["candidate_oracle"]["parent_candidate"]
    parent.update({
        "hits025": 1, "hits050": 1,
        "acc025": 0.5, "acc050": 0.5,
    })

    with pytest.raises(ValueError, match="diagnostic.*oracle"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 2, "oracle-union-test"
        )


def test_public_diagnostic_validator_rejects_jointly_impossible_effects():
    observation = _synthetic_calibration_observation(
        [0.10, 0.80], [0.10, 0.80], (0, 1)
    )
    diagnostic = copy.deepcopy(observation.diagnostics_result.diagnostics)
    diagnostic["effects"]["parent_vs_default"] = {
        "fixes025": 1,
        "breaks025": 1,
        "fixes050": 0,
        "breaks050": 0,
    }

    with pytest.raises(ValueError, match="diagnostic.*effect"):
        finetune_runner._validate_one_calibration_diagnostic(
            diagnostic, 2, "effect-test"
        )


@pytest.mark.parametrize("section", ["selected", "geometry_oracle"])
def test_public_diagnostic_validator_rejects_jointly_impossible_transition(
        section):
    observation = _synthetic_calibration_observation(
        [0.10, 0.80], [0.10, 0.80], (0, 1)
    )
    result = observation.diagnostics_result
    previous = finetune_runner._validate_one_calibration_diagnostic(
        result.diagnostics, 2, "previous"
    )
    current = finetune_runner._validate_one_calibration_diagnostic(
        result.diagnostics, 2, "current"
    )
    transition = rec_finetune.build_calibration_step_transition(
        result.transition_state, result.transition_state, 0, 1
    )
    transition[section] = {
        "gained025": 1,
        "lost025": 1,
        "gained050": 0,
        "lost050": 0,
    }

    with pytest.raises(ValueError, match="diagnostic.*transition"):
        finetune_runner._validate_calibration_diagnostic_transition(
            transition, 0, 1, 2, previous, current
        )


def test_public_diagnostic_validator_rejects_selected_oracle_transition_conflict():
    observation = _synthetic_calibration_observation(
        [0.10, 0.80], [0.10, 0.80], (0, 1)
    )
    result = observation.diagnostics_result
    previous = finetune_runner._validate_one_calibration_diagnostic(
        result.diagnostics, 2, "previous"
    )
    current = finetune_runner._validate_one_calibration_diagnostic(
        result.diagnostics, 2, "current"
    )
    transition = rec_finetune.build_calibration_step_transition(
        result.transition_state, result.transition_state, 0, 1
    )
    transition["selected"] = {
        "gained025": 1,
        "lost025": 1,
        "gained050": 1,
        "lost050": 1,
    }

    with pytest.raises(ValueError, match="diagnostic.*transition"):
        finetune_runner._validate_calibration_diagnostic_transition(
            transition, 0, 1, 2, previous, current
        )


def test_public_diagnostic_validator_requires_exact_joint_transition_witness():
    indices = (0, 1)
    previous_result = _transition_result(
        [0.10, 0.80], [0.30, 0.80], indices
    )
    current_result = _transition_result(
        [0.10, 0.30], [0.10, 0.80], indices
    )
    previous = finetune_runner._validate_one_calibration_diagnostic(
        previous_result.diagnostics, 2, "previous"
    )
    current = finetune_runner._validate_one_calibration_diagnostic(
        current_result.diagnostics, 2, "current"
    )
    transition = rec_finetune.build_calibration_step_transition(
        previous_result.transition_state,
        current_result.transition_state,
        0,
        1,
    )
    transition["selected"] = {
        "gained025": 0, "lost025": 0,
        "gained050": 0, "lost050": 1,
    }
    transition["geometry_oracle"] = {
        "gained025": 0, "lost025": 1,
        "gained050": 1, "lost050": 1,
    }

    with pytest.raises(ValueError, match="diagnostic.*transition"):
        finetune_runner._validate_calibration_diagnostic_transition(
            transition, 0, 1, 2, previous, current
        )


def test_public_diagnostic_validator_accepts_jointly_feasible_nonzero_changes():
    indices = (0, 1, 2)
    previous_selected = torch.tensor([0.10, 0.30, 0.80])
    current_selected = torch.tensor([0.30, 0.80, 0.10])
    previous = _transition_result(
        previous_selected, previous_selected, indices
    )
    current = _transition_result(
        current_selected, current_selected, indices
    )
    previous_public = finetune_runner._validate_one_calibration_diagnostic(
        previous.diagnostics, 3, "previous"
    )
    current_public = finetune_runner._validate_one_calibration_diagnostic(
        current.diagnostics, 3, "current"
    )
    transition = rec_finetune.build_calibration_step_transition(
        previous.transition_state, current.transition_state, 0, 1
    )

    finetune_runner._validate_calibration_diagnostic_transition(
        transition, 0, 1, 3, previous_public, current_public
    )


def _metrics(acc025, acc050, sample_count=100):
    hits025 = int(round(acc025 * sample_count))
    hits050 = int(round(acc050 * sample_count))
    actual025 = hits025 / sample_count
    actual050 = hits050 / sample_count
    return {
        "sample_count": sample_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": actual025,
        "acc050": actual050,
        "score": min(actual025 / 0.60, actual050 / 0.47)
        + 0.1 * (actual025 + actual050),
    }


def test_selector_keeps_earliest_tie_and_restores_best_snapshot_on_regression(
        monkeypatch):
    clone_calls = []
    original_clone = rec_finetune._clone_cpu_snapshot

    def tracked_clone(snapshot):
        if isinstance(snapshot, dict):
            clone_calls.append(snapshot)
        return original_clone(snapshot)

    monkeypatch.setattr(rec_finetune, "_clone_cpu_snapshot", tracked_clone)
    selector = rec_finetune.CalibrationSelector(
        contract_steps=(0, 306, 612, 918), expected_sample_count=100
    )
    baseline_snapshot = {
        "model": [torch.tensor([1.0]), (torch.tensor([2.0]),)]
    }
    selector.observe(0, _metrics(0.50, 0.45), baseline_snapshot)
    assert len(clone_calls) == 1
    baseline_snapshot["model"][0].add_(20.0)

    best_snapshot = {"model": [torch.tensor([3.0])]}
    decision = selector.observe(
        306, _metrics(0.55, 0.46), best_snapshot
    )
    assert decision.action == "continue"
    assert decision.best_step == 306
    assert len(clone_calls) == 2

    selector.observe(612, _metrics(0.55, 0.46), {"model": []})
    assert selector.best_step == 306

    decision = selector.observe(
        918, _metrics(0.58, 0.44), {"model": [torch.tensor([99.0])]}
    )
    assert decision.action == "stop"
    assert selector.best_step == 306
    assert len(clone_calls) == 2
    assert torch.equal(selector.best_snapshot["model"][0], torch.tensor([3.0]))

    best_snapshot["model"][0].add_(30.0)
    exposed = selector.best_snapshot
    exposed["model"][0].add_(40.0)
    assert selector.best_snapshot["model"][0].item() == 3.0
    assert len(selector.history) == 4
    exposed_metrics = selector.best_metrics
    exposed_metrics["score"] = -1.0
    exposed_history = selector.history
    exposed_history[0]["metrics"]["score"] = -1.0
    assert selector.best_metrics["score"] >= 0.0
    assert selector.history[0]["metrics"]["score"] >= 0.0


def test_selector_stops_on_composite_regression_above_baseline_thresholds():
    selector = rec_finetune.CalibrationSelector(
        contract_steps=(0, 1, 2), expected_sample_count=100
    )
    selector.observe(0, _metrics(0.50, 0.40), {"w": torch.tensor(0.0)})
    selector.observe(1, _metrics(0.60, 0.50), {"w": torch.tensor(1.0)})

    decision = selector.observe(
        2, _metrics(0.55, 0.45), {"w": torch.tensor(2.0)}
    )

    assert decision.action == "stop"
    assert selector.best_step == 1


def test_selector_rejects_noncontract_steps_bad_counts_and_nonfinite_snapshots():
    selector = rec_finetune.CalibrationSelector(
        contract_steps=(0, 2), expected_sample_count=10
    )
    with pytest.raises(ValueError, match="step"):
        selector.observe(2, _metrics(0.5, 0.4, 10), {})
    with pytest.raises(ValueError, match="sample_count"):
        selector.observe(0, _metrics(0.5, 0.4, 100), {})
    with pytest.raises(ValueError, match="finite"):
        selector.observe(
            0, _metrics(0.5, 0.4, 10), {"w": torch.tensor(float("nan"))}
        )
    assert selector.history == ()
    selector.observe(0, _metrics(0.5, 0.4, 10), {"w": torch.tensor(0.0)})

    with pytest.raises(ValueError, match="finite"):
        selector.observe(
            2, _metrics(0.6, 0.5, 10),
            {"w": torch.tensor(float("nan"))},
        )
    assert selector.best_step == 0
    assert len(selector.history) == 1
    decision = selector.observe(
        2, _metrics(0.6, 0.5, 10), {"w": torch.tensor(2.0)}
    )
    assert decision.best_step == 2


class _LoopMcln(torch.nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.decoder_query_proj = torch.nn.Linear(1, 1)
        self.proposal_head = torch.nn.Linear(1, 1)
        self.prediction_heads = torch.nn.Linear(1, 1)
        self.register_buffer("completed_updates", torch.tensor(0))
        self.events = events

    def forward(self, inputs):
        if self.events is not None:
            is_fit = self.decoder.training
            self.events.append((
                "mcln_fit" if is_fit else "mcln_calibration",
                int(self.completed_updates.item()) + (1 if is_fit else 0),
            ))
            if is_fit:
                assert not self.training
                assert not self.encoder.training
                assert self.decoder_query_proj.training
                assert self.proposal_head.training
                assert self.prediction_heads.training
        assert inputs["point_clouds"].dtype == torch.float32
        assert "center_label" not in inputs
        signal = sum(
            parameter.sum()
            for name, parameter in self.named_parameters()
            if name.startswith((
                "decoder.", "decoder_query_proj.", "proposal_head.",
                "prediction_heads.",
            ))
        )
        return {"model_signal": signal}


class _CountingSgd:
    def __init__(self, mcln, groups, events):
        parameters = (
            groups["mcln_parameters"]
            + groups["parent_parameters"]
            + groups["geometry_parameters"]
        )
        self._optimizer = torch.optim.SGD(parameters, lr=1e-3)
        self._mcln = mcln
        self._events = events
        self.zero_grad_calls = 0
        self.zero_grad_records = []
        self.step_calls = 0

    def zero_grad(self, **kwargs):
        self.zero_grad_calls += 1
        record = {
            "completed_updates": int(self._mcln.completed_updates.item()),
            "kwargs": dict(kwargs),
        }
        self.zero_grad_records.append(record)
        self._events.append(("zero_grad", copy.deepcopy(record)))
        self._optimizer.zero_grad(**kwargs)

    def step(self):
        self._optimizer.step()
        self.step_calls += 1
        self._mcln.completed_updates.add_(1)
        self._events.append(("optimizer_step", self.step_calls))


class _TrackingAdamW:
    def __init__(self, mcln, parent, groups):
        self._optimizer = rec_finetune.build_rec_finetune_optimizer(groups)
        self._mcln = mcln
        self._parent = parent
        self.zero_grad_records = []
        self.parent_gradient_present = []
        self.parent_after_steps = []

    def zero_grad(self, **kwargs):
        self.zero_grad_records.append({
            "completed_updates": int(self._mcln.completed_updates.item()),
            "kwargs": dict(kwargs),
        })
        self._optimizer.zero_grad(**kwargs)

    def step(self):
        self.parent_gradient_present.append({
            name: parameter.grad is not None
            for name, parameter in self._parent.named_parameters()
        })
        self._optimizer.step()
        self.parent_after_steps.append({
            name: parameter.detach().clone()
            for name, parameter in self._parent.named_parameters()
        })
        self._mcln.completed_updates.add_(1)


class _NeverAccess:
    def __iter__(self):
        raise AssertionError("validation/test sentinel was accessed")

    def __len__(self):
        raise AssertionError("validation/test sentinel was accessed")

    def __getitem__(self, _key):
        raise AssertionError("validation/test sentinel was accessed")


def _loop_batch(indices):
    count = len(indices)
    return {
        "point_clouds": torch.arange(
            count, dtype=torch.float64
        ).reshape(count, 1, 1),
        "utterances": ["sample-{}".format(index) for index in indices],
        "all_detected_boxes": torch.zeros(count, 1, 6),
        "all_detected_bbox_label_mask": torch.ones(
            count, 1, dtype=torch.bool
        ),
        "all_detected_class_ids": torch.zeros(count, 1, dtype=torch.long),
        "superpoint": torch.zeros(count, 1, dtype=torch.long),
        "center_label": torch.zeros(count, 1, 3),
        "size_gts": torch.ones(count, 1, 3),
        "box_label_mask": torch.ones(count, 1, dtype=torch.bool),
        "dataset_index": torch.tensor(indices, dtype=torch.long),
    }


def _assert_state_equal(actual, expected):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        assert torch.equal(value.detach().cpu(), expected[name]), name


def _canonical_parameter_name_sha256(names):
    payload = json.dumps(
        sorted(names), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _diagnostic_loss_tensors(mcln, parent, geometry):
    hungarian = sum(
        parameter.square().sum()
        for parameter in groups_for_model(mcln, "mcln")
    )
    parent_loss = sum(parameter.square().sum() for parameter in parent.parameters())
    geometry_loss = sum(
        parameter.square().sum() for parameter in geometry.parameters()
    )
    return hungarian, parent_loss, geometry_loss


def groups_for_model(model, prefix):
    if prefix == "mcln":
        return tuple(
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("encoder.")
        )
    raise AssertionError("unsupported synthetic diagnostic model")


def _diagnostic_reranker_stats(parent_total, geometry_total, step=1):
    def stats(total, ranking, informative_rows, pair_count, positives,
              negatives):
        return {
            "loss_listwise": torch.tensor(0.7 * step),
            "loss_best_tier_pairwise": torch.tensor(0.5 * step),
            "loss_ranking": torch.tensor(ranking * step),
            "loss_threshold": torch.tensor(0.2 * step),
            "loss_iou": torch.tensor(0.1 * step),
            "loss_total": total.detach(),
            "tier_pairwise_informative_rows": torch.tensor(
                informative_rows, dtype=torch.long
            ),
            "tier_pairwise_pair_count": torch.tensor(
                pair_count, dtype=torch.long
            ),
            "tier_pairwise_positive_count": torch.tensor(
                positives, dtype=torch.long
            ),
            "tier_pairwise_negative_count": torch.tensor(
                negatives, dtype=torch.long
            ),
        }

    return {
        "parent": stats(parent_total, 0.7, 2, 6, 5, 3),
        "geometry": stats(geometry_total, 0.5, 3, 9, 6, 4),
    }


def _forward_state_with_reranker_stats(parent_loss, geometry_loss):
    stats = _diagnostic_reranker_stats(parent_loss, geometry_loss)
    return {
        "parent_loss": parent_loss,
        "geometry_loss": geometry_loss,
        "parent_loss_stats": stats["parent"],
        "geometry_loss_stats": stats["geometry"],
    }


def test_update_diagnostics_publishes_pairwise_signal_coverage(monkeypatch):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 2)
    geometry = torch.nn.Linear(2, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    losses = _diagnostic_loss_tensors(mcln, parent, geometry)
    total, detached_losses = finetune_runner.validate_rec_finetune_losses(
        *losses
    )
    total.backward()
    monkeypatch.setattr(
        rec_finetune,
        "clip_rec_finetune_gradients",
        lambda _groups: {
            "mcln_decoder_box": 0.25,
            "parent_reranker": 0.5,
            "geometry_reranker": 0.75,
        },
    )

    record = finetune_runner.collect_rec_finetune_update_diagnostics(
        mcln,
        groups,
        detached_losses,
        reranker_stats=_diagnostic_reranker_stats(losses[1], losses[2]),
        step=1,
    )

    assert record["ranking_objectives"]["geometry"] == {
        "loss_listwise": pytest.approx(0.7),
        "loss_best_tier_pairwise": pytest.approx(0.5),
        "loss_ranking": pytest.approx(0.5),
        "loss_threshold": pytest.approx(0.2),
        "loss_iou": pytest.approx(0.1),
        "loss_total": detached_losses["geometry"],
        "tier_pairwise_informative_rows": 3,
        "tier_pairwise_pair_count": 9,
        "tier_pairwise_positive_count": 6,
        "tier_pairwise_negative_count": 4,
    }


@pytest.mark.parametrize("mutation", ("too_few_pairs", "too_few_negatives", "zero_pair_loss"))
def test_update_diagnostics_reject_impossible_pairwise_coverage(mutation):
    parent_total = torch.tensor(2.0)
    geometry_total = torch.tensor(3.0)
    stats = _diagnostic_reranker_stats(parent_total, geometry_total)
    geometry = stats["geometry"]
    if mutation == "too_few_pairs":
        geometry["tier_pairwise_informative_rows"] = torch.tensor(10)
        geometry["tier_pairwise_pair_count"] = torch.tensor(1)
        geometry["tier_pairwise_positive_count"] = torch.tensor(10)
        geometry["tier_pairwise_negative_count"] = torch.tensor(10)
    elif mutation == "too_few_negatives":
        geometry["tier_pairwise_informative_rows"] = torch.tensor(2)
        geometry["tier_pairwise_pair_count"] = torch.tensor(2)
        geometry["tier_pairwise_positive_count"] = torch.tensor(2)
        geometry["tier_pairwise_negative_count"] = torch.tensor(1)
    else:
        geometry["tier_pairwise_informative_rows"] = torch.tensor(0)
        geometry["tier_pairwise_pair_count"] = torch.tensor(0)
        geometry["tier_pairwise_negative_count"] = torch.tensor(0)

    with pytest.raises(ValueError, match="pairwise|coverage"):
        finetune_runner._validated_reranker_objective_stats(
            stats, {"parent": 2.0, "geometry": 3.0}
        )


def test_update_diagnostics_record_exact_dynamic_counts_names_and_clipping(
        monkeypatch):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 2)
    geometry = torch.nn.Linear(2, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    losses = _diagnostic_loss_tensors(mcln, parent, geometry)
    total, detached_losses = finetune_runner.validate_rec_finetune_losses(
        *losses
    )
    total.backward()
    monkeypatch.setattr(
        rec_finetune,
        "clip_rec_finetune_gradients",
        lambda actual: {
            "mcln_decoder_box": 0.25,
            "parent_reranker": 0.5,
            "geometry_reranker": 0.75,
        } if actual is groups else pytest.fail("wrong groups"),
    )

    record = finetune_runner.collect_rec_finetune_update_diagnostics(
        mcln, groups, detached_losses,
        _diagnostic_reranker_stats(losses[1], losses[2]), step=1
    )

    assert record["schema"] == "rec-finetune-update-diagnostics-v2"
    assert record["step"] == 1
    assert record["losses"] == {
        "hungarian": float(losses[0].detach()),
        "parent": float(losses[1].detach()),
        "geometry": float(losses[2].detach()),
        "total": float(sum(losses).detach()),
    }
    expected = {
        "mcln_decoder_box": (
            groups["mcln_names"], groups["mcln_parameters"], 0.25, 0.1
        ),
        "parent_reranker": (
            groups["parent_names"], groups["parent_parameters"], 0.5, 1.0
        ),
        "geometry_reranker": (
            groups["geometry_names"], groups["geometry_parameters"], 0.75, 1.0
        ),
    }
    assert set(record["groups"]) == set(expected)
    for name, (names, parameters, norm, limit) in expected.items():
        diagnostic = record["groups"][name]
        tensor_count = len(parameters)
        element_count = sum(parameter.numel() for parameter in parameters)
        assert diagnostic == {
            "parameter_tensor_count": tensor_count,
            "parameter_element_count": element_count,
            "parameter_names_sha256": _canonical_parameter_name_sha256(names),
            "gradient_tensor_count": tensor_count,
            "gradient_element_count": element_count,
            "finite_gradient_tensor_count": tensor_count,
            "finite_gradient_element_count": element_count,
            "all_present_finite": True,
            "preclip_gradient_norm": norm,
            "clip_limit": limit,
        }
    frozen_names = tuple(
        name for name, parameter in mcln.named_parameters()
        if not parameter.requires_grad
    )
    frozen_parameters = tuple(
        parameter for parameter in mcln.parameters()
        if not parameter.requires_grad
    )
    assert record["frozen_mcln"] == {
        "parameter_tensor_count": len(frozen_parameters),
        "parameter_element_count": sum(
            parameter.numel() for parameter in frozen_parameters
        ),
        "parameter_names_sha256": _canonical_parameter_name_sha256(
            frozen_names
        ),
        "gradient_tensor_count": 0,
        "gradient_element_count": 0,
    }
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "bad_losses, error",
    [
        ((None, torch.tensor(1.0), torch.tensor(1.0)), ValueError),
        ((torch.tensor([1.0]), torch.tensor(1.0), torch.tensor(1.0)), ValueError),
        ((torch.tensor(float("nan")), torch.tensor(1.0), torch.tensor(1.0)),
         FloatingPointError),
        ((torch.tensor(1.0), torch.tensor(float("inf")), torch.tensor(1.0)),
         FloatingPointError),
    ],
)
def test_loss_diagnostics_reject_missing_nonscalar_and_nonfinite_components(
        bad_losses, error):
    with pytest.raises(error, match="loss|finite|scalar|tensor"):
        finetune_runner.validate_rec_finetune_losses(*bad_losses)


@pytest.mark.parametrize("failure", ("all-none", "nonfinite", "frozen"))
def test_update_diagnostics_fail_closed_before_clipping_and_optimizer_step(
        monkeypatch, failure):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    hungarian, parent_loss, geometry_loss = _diagnostic_loss_tensors(
        mcln, parent, geometry
    )
    total, detached_losses = finetune_runner.validate_rec_finetune_losses(
        hungarian, parent_loss, geometry_loss
    )
    total.backward()
    if failure == "all-none":
        for parameter in groups["geometry_parameters"]:
            parameter.grad = None
    elif failure == "nonfinite":
        groups["parent_parameters"][0].grad.flatten()[0] = float("nan")
    else:
        mcln.encoder.weight.grad = torch.ones_like(mcln.encoder.weight)
    clip_calls = []
    monkeypatch.setattr(
        rec_finetune,
        "clip_rec_finetune_gradients",
        lambda _groups: clip_calls.append(True),
    )

    with pytest.raises((RuntimeError, FloatingPointError), match=(
            "gradient|frozen|finite")):
        finetune_runner.collect_rec_finetune_update_diagnostics(
            mcln, groups, detached_losses,
            _diagnostic_reranker_stats(parent_loss, geometry_loss), step=1
        )
    assert clip_calls == []


@pytest.mark.parametrize(
    "clip_result",
    [
        {"mcln_decoder_box": 1.0, "parent_reranker": 1.0},
        {"mcln_decoder_box": 1.0, "parent_reranker": float("inf"),
         "geometry_reranker": 1.0},
    ],
)
def test_update_diagnostics_reject_invalid_preclip_norms(
        monkeypatch, clip_result):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    losses = _diagnostic_loss_tensors(mcln, parent, geometry)
    total, detached_losses = finetune_runner.validate_rec_finetune_losses(
        *losses
    )
    total.backward()
    monkeypatch.setattr(
        rec_finetune, "clip_rec_finetune_gradients",
        lambda _groups: clip_result,
    )

    with pytest.raises((ValueError, FloatingPointError), match="clip|norm|finite"):
        finetune_runner.collect_rec_finetune_update_diagnostics(
            mcln, groups, detached_losses,
            _diagnostic_reranker_stats(losses[1], losses[2]), step=1
        )


def test_training_diagnostics_aggregate_is_exact_constant_size_and_json_safe(
        monkeypatch):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    aggregate = finetune_runner.create_rec_finetune_training_diagnostics(
        mcln, groups
    )
    clip_results = iter((
        {"mcln_decoder_box": 3.0, "parent_reranker": 2.0,
         "geometry_reranker": 1.0},
        {"mcln_decoder_box": 1.0, "parent_reranker": 4.0,
         "geometry_reranker": 2.0},
    ))
    monkeypatch.setattr(
        rec_finetune, "clip_rec_finetune_gradients",
        lambda _groups: next(clip_results),
    )
    records = []
    for step in (1, 2):
        for parameter in (
                groups["mcln_parameters"] + groups["parent_parameters"]
                + groups["geometry_parameters"]):
            parameter.grad = None
        losses = tuple(
            loss * step for loss in _diagnostic_loss_tensors(
                mcln, parent, geometry
            )
        )
        total, detached = finetune_runner.validate_rec_finetune_losses(*losses)
        total.backward()
        record = finetune_runner.collect_rec_finetune_update_diagnostics(
            mcln, groups, detached,
            _diagnostic_reranker_stats(losses[1], losses[2], step=step),
            step=step,
        )
        records.append(record)
        finetune_runner.update_rec_finetune_training_diagnostics(
            aggregate, record
        )

    assert set(aggregate) == {
        "schema", "update_count", "trainable_groups", "frozen_mcln",
        "losses", "ranking_objectives", "all_present_finite",
        "frozen_gradient_tensors_seen", "last_update",
    }
    assert aggregate["schema"] == "rec-finetune-training-diagnostics-v2"
    assert aggregate["update_count"] == 2
    assert aggregate["all_present_finite"] is True
    assert aggregate["frozen_gradient_tensors_seen"] == 0
    assert aggregate["last_update"] == records[-1]
    assert "updates" not in aggregate
    for loss_name in ("hungarian", "parent", "geometry", "total"):
        values = [record["losses"][loss_name] for record in records]
        assert aggregate["losses"][loss_name] == {
            "min": min(values), "max": max(values), "last": values[-1],
            "all_finite": True,
        }
    for name in ("parent", "geometry"):
        objective = aggregate["ranking_objectives"][name]
        for field in finetune_runner._DIAGNOSTIC_RERANKER_LOSS_NAMES:
            values = [
                record["ranking_objectives"][name][field]
                for record in records
            ]
            assert objective[field] == {
                "min": min(values), "max": max(values),
                "last": values[-1], "all_finite": True,
            }
        for field in finetune_runner._DIAGNOSTIC_RERANKER_COUNT_NAMES:
            values = [
                record["ranking_objectives"][name][field]
                for record in records
            ]
            assert objective[field] == {
                "min": min(values), "max": max(values),
                "last": values[-1], "total": sum(values),
            }
    expected_norms = {
        "mcln_decoder_box": [3.0, 1.0],
        "parent_reranker": [2.0, 4.0],
        "geometry_reranker": [1.0, 2.0],
    }
    for name, values in expected_norms.items():
        group = aggregate["trainable_groups"][name]
        assert group["preclip_gradient_norm"] == {
            "min": min(values), "max": max(values), "last": values[-1],
        }
        for field in ("gradient_tensor_count", "gradient_element_count"):
            expected_value = records[-1]["groups"][name][field]
            assert group[field] == {
                "min": expected_value,
                "max": expected_value,
                "last": expected_value,
            }
        assert group["all_present_finite"] is True
    encoded = json.dumps(aggregate, allow_nan=False)
    assert len(encoded) < 10000


def test_authoritative_production_parameter_inventory_and_name_digests():
    from train_dist_mod import TrainTester

    checkpoint = torch.load(INITIAL_BACKBONE_CHECKPOINT, map_location="cpu")
    config = finetune_runner._prepare_training_config(
        checkpoint, "/root/autodl-tmp/DATA_ROOT"
    )
    del checkpoint
    gc.collect()
    mcln = TrainTester.get_model(config)
    parent, _parent_artifact = (
        finetune_runner.load_parent_reranker_snapshot(
            INITIAL_PARENT_ARTIFACT, "cpu"
        )
    )
    geometry, _geometry_artifact = (
        finetune_runner.load_geometry_reranker_artifact(
            INITIAL_GEOMETRY_ARTIFACT,
            "cpu",
            parent_artifact_path=INITIAL_PARENT_ARTIFACT,
        )
    )
    try:
        groups = rec_finetune.configure_rec_finetune_trainability(
            mcln, parent, geometry
        )
        diagnostics = (
            finetune_runner.create_rec_finetune_training_diagnostics(
                mcln, groups
            )
        )

        expected_groups = {
            "mcln_decoder_box": (
                434,
                13_618_282,
                "ea6053d8c172309810a2d12dbfc5c863"
                "3da2e6ece48d10db56b94ab836199236",
                groups["mcln_names"],
            ),
            "parent_reranker": (
                10,
                150_404,
                "6abd4c763ec29d444147dc20760e4599"
                "86cd431a07e3d0227aad11ec9e2ae910",
                groups["parent_names"],
            ),
            "geometry_reranker": (
                10,
                171_140,
                "6abd4c763ec29d444147dc20760e4599"
                "86cd431a07e3d0227aad11ec9e2ae910",
                groups["geometry_names"],
            ),
        }
        for name, (tensor_count, element_count, digest, names) in (
                expected_groups.items()):
            actual = diagnostics["trainable_groups"][name]
            assert actual["parameter_tensor_count"] == tensor_count
            assert actual["parameter_element_count"] == element_count
            assert actual["parameter_names_sha256"] == digest
            assert actual["parameter_names_sha256"] == (
                _canonical_parameter_name_sha256(names)
            )
        frozen_names = tuple(
            name for name, parameter in mcln.named_parameters()
            if not parameter.requires_grad
        )
        assert diagnostics["frozen_mcln"] == {
            "parameter_tensor_count": 505,
            "parameter_element_count": 136_016_659,
            "parameter_names_sha256": (
                "e112a6e9944ede4a32230f3c32676c8"
                "1e9b0b33ed57641d1f2772f965c94af6e"
            ),
        }
        assert diagnostics["frozen_mcln"]["parameter_names_sha256"] == (
            _canonical_parameter_name_sha256(frozen_names)
        )
        json.dumps(diagnostics, allow_nan=False)
    finally:
        del mcln, parent, geometry
        gc.collect()


def test_loop_criterion_and_full_snapshot_restore_are_public_and_exact():
    assert finetune_runner.REC_TARGET_ONLY_FIELDS == frozenset({
        "center_label", "size_gts", "sem_cls_label", "box_label_mask",
        "gt_masks", "point_instance_label", "all_bboxes",
        "candidate_ious", "geometry_ious", "threshold_labels",
    })
    for name in (
            "build_rec_finetune_criterion",
            "snapshot_rec_finetune_state",
            "restore_rec_finetune_state",
            "calibrate_rec_finetune",
            "fit_rec_finetune_one_epoch",
            "publish_rec_finetune_run",
            "run_rec_finetune",
            "main"):
        assert callable(getattr(finetune_runner, name))

    criterion = finetune_runner.build_rec_finetune_criterion(
        SimpleNamespace(
            use_soft_token_loss=True,
            use_contrastive_align=True,
        )
    )
    assert criterion.losses == [
        "boxes", "labels", "masks", "contrastive_align",
    ]
    assert criterion.matcher.cost_class == 1
    assert criterion.matcher.cost_bbox == 5
    assert criterion.matcher.cost_giou == 2
    assert criterion.matcher.soft_token is True

    models = (
        _LoopMcln(),
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
    )
    snapshot = finetune_runner.snapshot_rec_finetune_state(*models)
    assert set(snapshot) == {"mcln", "parent", "geometry"}
    expected = copy.deepcopy(snapshot)
    with torch.no_grad():
        for model in models:
            for parameter in model.parameters():
                parameter.add_(10.0)
        models[0].completed_updates.fill_(9)

    finetune_runner.restore_rec_finetune_state(*models, snapshot)

    for key, model in zip(("mcln", "parent", "geometry"), models):
        _assert_state_equal(model.state_dict(), expected[key])
    bad = copy.deepcopy(snapshot)
    bad["parent"]["unexpected"] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError), match="parent|strict|state"):
        finetune_runner.restore_rec_finetune_state(*models, bad)


def _real_calibration_branch_fixture():
    centers = torch.tensor([[[0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.0],
                             [0.5, 0.0, 0.0]]])
    sizes = torch.full((1, 3, 3), 2.0)
    default_scores = torch.tensor([[0.0, 2.0, 1.0]])
    alternate_scores = torch.tensor([[2.0, 0.0, 1.0]])
    end_points = {
        "last_center": centers,
        "last_pred_size": sizes,
        "source_choice_source_scores": {
            "default": default_scores,
            "alternate": alternate_scores,
        },
        "selector_choice_source_names": ["default", "alternate"],
        "selector_choice_scores": torch.tensor([[0.0, 1.0]]),
        "selected_source_scores": alternate_scores.clone(),
        "selected_source_id": torch.tensor([1], dtype=torch.long),
    }
    forward_state = {
        "parent_model_inputs": {
            "valid_mask": torch.tensor([[True, True]]),
        },
        "parent_candidate_ious": torch.tensor([[0.0, 0.60]]),
        "parent_state": {
            "query_scores": torch.tensor([[0.10, -float("inf"), 0.90]]),
            "query_indices": torch.tensor([[0, 2]], dtype=torch.long),
            "top1_query_index": torch.tensor([2], dtype=torch.long),
            "parent_top1_mask": torch.tensor([[False, True]]),
        },
        "geometry_model_inputs": {
            "valid_mask": torch.tensor([[True, True, True]]),
        },
        "geometry_candidate_ious": torch.tensor([[0.0, 0.80, 1.0]]),
        "runtime_outputs": {
            "rec_geometry_runtime_mode": "flat_geometry_axis",
            "rec_reranker_scores": torch.tensor([
                [0.10, -float("inf"), 0.90]
            ]),
            "rec_geometry_scores": torch.tensor([[0.0, 2.0, 1.0]]),
            "rec_geometry_valid_mask": torch.tensor(
                [[True, True, True]]
            ),
        },
    }
    targets = {
        "center_label": torch.tensor([[
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        ]]),
        "size_gts": torch.full((1, 2, 3), 2.0),
        "box_label_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    }
    return end_points, forward_state, targets


def test_build_calibration_branch_ious_maps_real_tensors_and_root_gt_only():
    end_points, forward_state, targets = _real_calibration_branch_fixture()

    branches = finetune_runner._build_calibration_branch_ious(
        end_points, forward_state, targets
    )

    assert set(branches) == set(DIAGNOSTIC_BRANCH_NAMES)
    expected = {
        "default_top1": 0.0,
        "source_selector_top1": 1.0,
        "parent_top1": 0.60,
        "geometry_top1": 0.80,
        "raw_query_oracle": 1.0,
        "parent_candidate_oracle": 0.60,
        "geometry_candidate_oracle": 1.0,
    }
    for name, value in expected.items():
        assert branches[name].shape == (1,)
        assert branches[name].device == end_points["last_center"].device
        assert torch.is_floating_point(branches[name])
        assert float(branches[name].item()) == pytest.approx(value)
    # Query 1 exactly matches the second GT, but default_top1 remains a root miss.
    assert end_points["source_choice_source_scores"]["default"].argmax(1).item() == 1


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
def test_build_calibration_branch_ious_accepts_binary_root_mask_dtypes(dtype):
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    targets["box_label_mask"] = targets["box_label_mask"].to(dtype=dtype)

    branches = finetune_runner._build_calibration_branch_ious(
        end_points, forward_state, targets
    )

    assert branches["raw_query_oracle"].item() == pytest.approx(1.0)


def test_build_calibration_branch_ious_accepts_deployable_zero_query_size():
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    end_points["last_pred_size"][0, 1, 0] = 0.0

    branches = finetune_runner._build_calibration_branch_ious(
        end_points, forward_state, targets
    )

    assert all(bool(torch.isfinite(value).all()) for value in branches.values())
    assert branches["raw_query_oracle"].item() == pytest.approx(1.0)
    expected_raw = compute_query_ious(
        torch.cat([
            end_points["last_center"],
            end_points["last_pred_size"].clamp(min=1e-6),
        ], dim=-1),
        torch.cat([
            targets["center_label"][:, :1], targets["size_gts"][:, :1]
        ], dim=-1),
        targets["box_label_mask"][:, :1],
    )
    assert branches["default_top1"].item() == pytest.approx(
        expected_raw[0, 1].item()
    )


def test_build_calibration_branch_ious_clamps_negative_raw_query_size():
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    end_points["last_pred_size"][0, 1, 0] = -1.0

    branches = finetune_runner._build_calibration_branch_ious(
        end_points, forward_state, targets
    )
    expected_raw = compute_query_ious(
        torch.cat([
            end_points["last_center"],
            end_points["last_pred_size"].clamp(min=1e-6),
        ], dim=-1),
        torch.cat([
            targets["center_label"][:, :1], targets["size_gts"][:, :1]
        ], dim=-1),
        targets["box_label_mask"][:, :1],
    )

    assert branches["default_top1"].item() == pytest.approx(
        expected_raw[0, 1].item()
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda end_points: end_points[
                "source_choice_source_scores"
            ].pop("default"),
            "default",
        ),
        (
            lambda end_points: end_points.update(
                selected_source_id=torch.tensor([0], dtype=torch.long)
            ),
            "selected_source_id",
        ),
        (
            lambda end_points: end_points.update(
                selected_source_scores=end_points[
                    "source_choice_source_scores"
                ]["default"].clone()
            ),
            "selected_source_scores",
        ),
        (
            lambda end_points: end_points.update(
                selector_choice_source_names=["alternate", "default"]
            ),
            "source names",
        ),
        (
            lambda end_points: end_points[
                "selector_choice_scores"
            ].fill_(float("nan")),
            "finite",
        ),
    ],
)
def test_build_calibration_branch_ious_rejects_source_choice_tampering(
        mutation, error):
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    mutation(end_points)

    with pytest.raises(ValueError, match=error):
        finetune_runner._build_calibration_branch_ious(
            end_points, forward_state, targets
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda _end_points, state, _targets: state[
                "parent_model_inputs"
            ]["valid_mask"].fill_(False),
            "parent.*valid",
        ),
        (
            lambda _end_points, state, _targets: state[
                "geometry_model_inputs"
            ]["valid_mask"].fill_(False),
            "geometry.*valid",
        ),
        (
            lambda _end_points, _state, targets: targets[
                "box_label_mask"
            ][:, 0].fill_(False),
            "root",
        ),
        (
            lambda end_points, _state, _targets: end_points[
                "last_center"
            ].fill_(float("inf")),
            "finite",
        ),
        (
            lambda _end_points, _state, targets: targets.update(
                box_label_mask=torch.tensor([[1.0, float("nan")]])
            ),
            "finite",
        ),
        (
            lambda _end_points, _state, targets: targets.update(
                box_label_mask=torch.tensor([[1.0, 0.5]])
            ),
            "0/1",
        ),
        (
            lambda _end_points, _state, targets: targets.update(
                box_label_mask=torch.tensor(
                    [[1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex64
                )
            ),
            "dtype",
        ),
        (
            lambda _end_points, _state, targets: targets.update(
                box_label_mask=torch.ones(1, 2, 1)
            ),
            "shape",
        ),
        (
            lambda _end_points, _state, targets: targets.update(
                center_label=targets["center_label"].double()
            ),
            "root GT.*dtype",
        ),
        (
            lambda _end_points, _state, targets: targets.update(
                size_gts=targets["size_gts"].double()
            ),
            "root GT.*dtype",
        ),
    ],
)
def test_build_calibration_branch_ious_rejects_invalid_masks_and_boxes(
        mutation, error):
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    mutation(end_points, forward_state, targets)

    with pytest.raises(ValueError, match=error):
        finetune_runner._build_calibration_branch_ious(
            end_points, forward_state, targets
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runtime: runtime.pop("rec_geometry_valid_mask"),
        lambda runtime: runtime.update(
            rec_geometry_valid_mask=torch.ones(1, 3)
        ),
    ],
)
def test_build_calibration_branch_ious_rejects_bad_runtime_geometry_valid(
        mutation):
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    mutation(forward_state["runtime_outputs"])

    with pytest.raises(ValueError, match="geometry runtime valid mask"):
        finetune_runner._build_calibration_branch_ious(
            end_points, forward_state, targets
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state["parent_state"].update(
            parent_top1_mask=torch.tensor([[True, False]])
        ),
        lambda state: state["parent_state"].update(
            top1_query_index=torch.tensor([0], dtype=torch.long)
        ),
        lambda state: state["parent_state"].update(
            query_scores=torch.tensor([[0.90, -float("inf"), 0.10]])
        ),
        lambda state: state["parent_state"].update(
            query_indices=torch.tensor([[2, 0]], dtype=torch.long)
        ),
        lambda state: state["runtime_outputs"].update(
            rec_reranker_scores=torch.tensor([
                [0.90, -float("inf"), 0.10]
            ])
        ),
    ],
)
def test_build_calibration_branch_ious_binds_parent_top1_state(mutation):
    end_points, forward_state, targets = _real_calibration_branch_fixture()
    mutation(forward_state)

    with pytest.raises(ValueError, match="parent.*Top-1|parent.*state"):
        finetune_runner._build_calibration_branch_ious(
            end_points, forward_state, targets
        )


def test_calibration_uses_stable_runtime_axis_top1_and_dataset_order():
    events = []
    mcln = _LoopMcln(events)
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    loader = [_loop_batch([8]), _loop_batch([11])]
    calls = []
    diagnostic_order = []

    def forward_fn(end_points, inputs, targets, live_parent,
                   parent_artifact, live_geometry, geometry_artifact):
        assert not torch.is_grad_enabled()
        assert not mcln.training
        assert not mcln.decoder.training
        assert not live_parent.training
        assert not live_geometry.training
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert "center_label" not in inputs
        assert inputs["train"] is False
        assert set(end_points) == {"model_signal"}
        index = int(targets["dataset_index"].item())
        calls.append(index)
        diagnostic_order.append(("forward", index))
        loss = end_points["model_signal"] * 0.0
        if index == 8:
            return {
                "parent_loss": loss,
                "geometry_loss": loss,
                "runtime_outputs": {
                    "rec_geometry_runtime_mode": "flat_geometry_axis",
                    "rec_geometry_scores": torch.tensor([[0.7, 0.7, -1.0]]),
                    "rec_geometry_valid_mask": torch.tensor(
                        [[True, True, False]]
                    ),
                },
                "geometry_candidate_ious": torch.tensor([[0.30, 0.90, 0.99]]),
                "parent_candidate_ious": torch.tensor([[0.99, 0.99]]),
                "parent_state": {
                    "parent_top1_mask": torch.tensor([[False, True]])
                },
            }
        return {
            "parent_loss": loss,
            "geometry_loss": loss,
            "runtime_outputs": {
                "rec_geometry_runtime_mode": "parent_query_axis",
                "rec_reranker_scores": torch.tensor([[0.2, 0.9, 0.9]]),
            },
            "geometry_candidate_ious": torch.tensor([[0.99, 0.99]]),
            "parent_candidate_ious": torch.tensor([[0.10, 0.80]]),
            "parent_state": {
                "top1_query_index": torch.tensor([1]),
                "parent_top1_mask": torch.tensor([[False, True]]),
            },
        }

    def diagnostic_builder(end_points, forward_state, targets):
        assert set(end_points) == {"model_signal"}
        assert "center_label" in targets
        index = int(targets["dataset_index"].item())
        diagnostic_order.append(("diagnostic", index))
        selected = finetune_runner._selected_calibration_ious(
            forward_state
        )
        return {
            name: selected.clone() for name in DIAGNOSTIC_BRANCH_NAMES
        }

    observation = finetune_runner.calibrate_rec_finetune(
        mcln,
        parent,
        geometry,
        {},
        {},
        loader,
        (8, 11),
        torch.device("cpu"),
        forward_fn=forward_fn,
        diagnostic_builder=diagnostic_builder,
    )

    assert calls == [8, 11]
    assert finetune_runner.CalibrationObservation.__dataclass_params__.frozen
    assert isinstance(observation, finetune_runner.CalibrationObservation)
    assert observation.selection_metrics == _metrics(
        1.0, 0.5, sample_count=2
    )
    assert set(observation.selection_metrics) == {
        "sample_count", "hits025", "hits050", "acc025", "acc050", "score",
    }
    assert observation.diagnostics_result.diagnostics["sample_count"] == 2
    assert diagnostic_order == [
        ("forward", 8), ("diagnostic", 8),
        ("forward", 11), ("diagnostic", 11),
    ]
    assert [event[0] for event in events] == [
        "mcln_calibration", "mcln_calibration",
    ]


def _synthetic_calibration_observation(selected, oracle, indices):
    selected = torch.tensor(selected, dtype=torch.float32)
    oracle = torch.tensor(oracle, dtype=torch.float32)
    selection = rec_finetune.CalibrationAccumulator(indices)
    selection.update(indices, selected)
    diagnostics = rec_finetune.CalibrationDiagnosticsAccumulator(indices)
    branches = {
        name: selected.clone() for name in DIAGNOSTIC_BRANCH_NAMES
    }
    for name in (
            "raw_query_oracle", "parent_candidate_oracle",
            "geometry_candidate_oracle"):
        branches[name] = oracle.clone()
    diagnostics.update(indices, branches)
    return finetune_runner.CalibrationObservation(
        selection_metrics=selection.finalize(),
        diagnostics_result=diagnostics.finalize(),
    )


def _synthetic_observation_for_metrics(metrics):
    sample_count = metrics["sample_count"]
    selected = (
        [0.75] * metrics["hits050"]
        + [0.30] * (metrics["hits025"] - metrics["hits050"])
        + [0.10] * (sample_count - metrics["hits025"])
    )
    observation = _synthetic_calibration_observation(
        selected, selected, tuple(range(sample_count))
    )
    assert observation.selection_metrics == metrics
    return observation


def _synthetic_diagnostic_run_fields(calibration_history, selected_step):
    observations = {
        record["step"]: _synthetic_observation_for_metrics(record["metrics"])
        for record in calibration_history
    }
    diagnostic_history = []
    previous_step = None
    previous_result = None
    for record in calibration_history:
        step = record["step"]
        result = observations[step].diagnostics_result
        transition = None
        if previous_result is not None:
            transition = rec_finetune.build_calibration_step_transition(
                previous_result.transition_state,
                result.transition_state,
                previous_step,
                step,
            )
        diagnostic_history.append({
            "step": step,
            "diagnostics": copy.deepcopy(result.diagnostics),
            "transition_from_previous": copy.deepcopy(transition),
        })
        previous_step = step
        previous_result = result
    selected = copy.deepcopy(
        observations[selected_step].diagnostics_result.diagnostics
    )
    selected_output_sha256 = rec_finetune.calibration_selected_output_sha256(
        observations[selected_step].diagnostics_result.transition_state
    )
    return {
        "calibration_diagnostics_history": diagnostic_history,
        "selected_calibration_diagnostics": selected,
        "reproduced_calibration_diagnostics": copy.deepcopy(selected),
        "selected_calibration_output_sha256": selected_output_sha256,
        "reproduced_calibration_output_sha256": selected_output_sha256,
    }


def test_unpack_calibration_observation_defensively_copies_diagnostics():
    source = _synthetic_observation_for_metrics(
        _metrics(0.50, 0.25, sample_count=4)
    )
    source_metrics = source.selection_metrics
    source_result = source.diagnostics_result
    observation = finetune_runner.CalibrationObservation(
        selection_metrics=source_metrics,
        diagnostics_result=source_result,
    )
    source_metrics["hits025"] = 0
    source_result.diagnostics["sample_count"] = 99

    assert observation.selection_metrics["hits025"] == 2
    assert observation.diagnostics_result.diagnostics["sample_count"] == 4

    mode, metrics, diagnostics_result = (
        finetune_runner._unpack_calibration_observation(observation)
    )
    observation.selection_metrics["hits025"] = 0
    observation.diagnostics_result.diagnostics["sample_count"] = 99

    assert mode == "diagnostic"
    assert metrics["hits025"] == 2
    assert diagnostics_result.diagnostics["sample_count"] == 4
    diagnostics_result.diagnostics["sample_count"] = 1
    assert observation.diagnostics_result.diagnostics["sample_count"] == 99


def _simple_fit_forward(end_points, _inputs, _targets, parent,
                        _parent_artifact, geometry, _geometry_artifact):
    loss = (
        end_points["model_signal"].square()
        + sum(parameter.square().sum() for parameter in parent.parameters())
        + sum(parameter.square().sum() for parameter in geometry.parameters())
    )
    return _forward_state_with_reranker_stats(loss * 0.2, loss * 0.3)


def _simple_hungarian_loss(end_points, *_args, **_kwargs):
    return end_points["model_signal"].square() * 0.1, end_points


def test_diagnostic_loop_tracks_transitions_and_reproduces_selected_summary():
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    indices = (20, 21, 22, 23)
    calibration_calls = []

    def calibration_fn(*_args, **_kwargs):
        step = int(mcln.completed_updates.item())
        calibration_calls.append(step)
        selected = {
            0: [0.20, 0.30, 0.60, 0.80],
            1: [0.40, 0.20, 0.70, 0.50],
        }[step]
        oracle = {
            0: [0.40, 0.40, 0.70, 0.90],
            1: [0.50, 0.20, 0.80, 0.90],
        }[step]
        return _synthetic_calibration_observation(
            selected, oracle, indices
        )

    result = finetune_runner.fit_rec_finetune_one_epoch(
        mcln, parent, geometry, {}, {}, groups, optimizer, object(),
        [_loop_batch([0]), _loop_batch([1])], [_loop_batch(indices)], indices,
        torch.device("cpu"), max_steps=2, calibration_steps=(0, 1, 2),
        forward_fn=_simple_fit_forward,
        hungarian_loss_fn=_simple_hungarian_loss,
        calibration_fn=calibration_fn,
    )

    assert result["completed_updates"] == 1
    assert result["stopped_early"] is True
    assert result["selected_step"] == 0
    assert calibration_calls == [0, 1, 0]
    history = result["calibration_diagnostics_history"]
    assert len(history) == 2
    assert [set(record) for record in history] == [
        {"step", "diagnostics", "transition_from_previous"},
        {"step", "diagnostics", "transition_from_previous"},
    ]
    assert history[0]["step"] == 0
    assert history[0]["transition_from_previous"] is None
    assert history[1]["step"] == 1
    assert history[1]["transition_from_previous"] == {
        "schema": "rec-finetune-calibration-step-transition-v2",
        "previous_step": 0,
        "current_step": 1,
        "sample_count": 4,
        "selected": {
            "gained025": 1, "lost025": 1,
            "gained050": 0, "lost050": 1,
        },
        "geometry_oracle": {
            "gained025": 0, "lost025": 1,
            "gained050": 0, "lost050": 0,
        },
        "selected_oracle_joint": _expected_joint_transition(
            ("s0_o1", "s1_o1", 1),
            ("s1_o1", "s0_o0", 1),
            ("s2_o2", "s2_o2", 1),
            ("s2_o2", "s1_o2", 1),
        ),
    }
    assert result["selected_calibration_diagnostics"] == history[0][
        "diagnostics"
    ]
    assert result["reproduced_calibration_diagnostics"] == history[0][
        "diagnostics"
    ]
    preserved_history = copy.deepcopy(history)
    preserved_reproduced = copy.deepcopy(
        result["reproduced_calibration_diagnostics"]
    )
    result["selected_calibration_diagnostics"]["sample_count"] = 0
    assert history == preserved_history
    assert result["reproduced_calibration_diagnostics"] == preserved_reproduced
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "selected_ious" not in encoded
    assert "geometry_oracle_ious" not in encoded


def test_diagnostic_loop_accepts_oracle_only_reproduction_drift():
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    indices = (20, 21, 22, 23)
    calibration_calls = []

    def calibration_fn(*_args, **_kwargs):
        step = int(mcln.completed_updates.item())
        is_reproduction = step == 0 and calibration_calls.count(0) == 1
        calibration_calls.append(step)
        if step == 0:
            selected = [0.20, 0.30, 0.60, 0.80]
            oracle = (
                [0.90, 0.90, 0.90, 0.90]
                if is_reproduction
                else [0.40, 0.40, 0.70, 0.90]
            )
        else:
            selected = [0.10, 0.20, 0.40, 0.60]
            oracle = [0.30, 0.30, 0.50, 0.70]
        return _synthetic_calibration_observation(
            selected, oracle, indices
        )

    result = finetune_runner.fit_rec_finetune_one_epoch(
        mcln, parent, geometry, {}, {}, groups, optimizer, object(),
        [_loop_batch([0])], [_loop_batch(indices)], indices,
        torch.device("cpu"), max_steps=1, calibration_steps=(0, 1),
        forward_fn=_simple_fit_forward,
        hungarian_loss_fn=_simple_hungarian_loss,
        calibration_fn=calibration_fn,
    )

    assert calibration_calls == [0, 1, 0]
    assert result["selected_metrics"] == result["reproduced_metrics"]
    assert result["selected_calibration_output_sha256"] == result[
        "reproduced_calibration_output_sha256"
    ]
    assert result["selected_calibration_diagnostics"] != result[
        "reproduced_calibration_diagnostics"
    ]


def test_diagnostic_loop_rejects_changed_selected_output_with_same_metrics():
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    indices = (20, 21, 22, 23)
    calibration_calls = []

    def calibration_fn(*_args, **_kwargs):
        step = int(mcln.completed_updates.item())
        is_reproduction = step == 0 and calibration_calls.count(0) == 1
        calibration_calls.append(step)
        if step == 0:
            selected = (
                [0.30, 0.20, 0.70, 0.60]
                if is_reproduction
                else [0.20, 0.30, 0.60, 0.70]
            )
        else:
            selected = [0.10, 0.20, 0.40, 0.60]
        return _synthetic_calibration_observation(
            selected, selected, indices
        )

    with pytest.raises(RuntimeError, match="selected calibration output"):
        finetune_runner.fit_rec_finetune_one_epoch(
            mcln, parent, geometry, {}, {}, groups, optimizer, object(),
            [_loop_batch([0])], [_loop_batch(indices)], indices,
            torch.device("cpu"), max_steps=1, calibration_steps=(0, 1),
            forward_fn=_simple_fit_forward,
            hungarian_loss_fn=_simple_hungarian_loss,
            calibration_fn=calibration_fn,
        )


@pytest.mark.parametrize("diagnostic_first", [False, True])
def test_fit_loop_rejects_mixed_legacy_and_diagnostic_calibration_modes(
        diagnostic_first):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    indices = (7, 8)

    def calibration_fn(*_args, **_kwargs):
        step = int(mcln.completed_updates.item())
        diagnostic = _synthetic_calibration_observation(
            [0.40, 0.60], [0.50, 0.70], indices
        )
        legacy = _metrics(0.5, 0.5, sample_count=2)
        return diagnostic if (step == 0) is diagnostic_first else legacy

    with pytest.raises(ValueError, match="mix"):
        finetune_runner.fit_rec_finetune_one_epoch(
            mcln, parent, geometry, {}, {}, groups, optimizer, object(),
            [_loop_batch([0])], [_loop_batch(indices)], indices,
            torch.device("cpu"), max_steps=1, calibration_steps=(0, 1),
            forward_fn=_simple_fit_forward,
            hungarian_loss_fn=_simple_hungarian_loss,
            calibration_fn=calibration_fn,
        )


def test_six_update_loop_uses_natural_remainder_cadence_and_earliest_best(
        monkeypatch):
    events = []
    calibration_updates = []
    mcln = _LoopMcln(events)
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, events)
    fit_loader = [
        _loop_batch([0, 1]),
        _loop_batch([2, 3]),
        _loop_batch([4, 5]),
        _loop_batch([6, 7]),
        _loop_batch([8, 9]),
        _loop_batch([10]),
    ]
    calibration_loader = [_loop_batch([20, 21, 22, 23])]
    selected_snapshot = {}

    original_train_mode = rec_finetune.set_rec_finetune_train_mode

    def train_mode(*models):
        original_train_mode(*models)
        assert mcln.decoder.training
        assert not mcln.encoder.training
        events.append((
            "train_mode", int(mcln.completed_updates.item()) + 1
        ))

    monkeypatch.setattr(
        rec_finetune, "set_rec_finetune_train_mode", train_mode
    )

    original_move = finetune_runner._move_batch_to_device

    def move_batch(batch, device):
        if mcln.decoder.training:
            events.append(("move", int(batch["dataset_index"].numel())))
        return original_move(batch, device)

    monkeypatch.setattr(finetune_runner, "_move_batch_to_device", move_batch)

    def input_builder(batch):
        if mcln.decoder.training:
            events.append((
                "input_builder", int(mcln.completed_updates.item()) + 1
            ))
        return finetune_runner.build_rec_finetune_inputs(batch)

    original_clip = rec_finetune.clip_rec_finetune_gradients

    def clip(groups_value):
        events.append(("clip", int(mcln.completed_updates.item()) + 1))
        return original_clip(groups_value)

    monkeypatch.setattr(rec_finetune, "clip_rec_finetune_gradients", clip)
    mcln.decoder.weight.register_hook(
        lambda gradient: events.append((
            "backward", int(mcln.completed_updates.item()) + 1
        )) or gradient
    )

    def forward_fn(end_points, inputs, targets, live_parent,
                   parent_artifact, live_geometry, geometry_artifact):
        update = int(mcln.completed_updates.item())
        if mcln.decoder.training:
            events.append(("rec_forward", update + 1))
            assert set(end_points) == {"model_signal"}
            assert "center_label" not in inputs
            loss = (
                end_points["model_signal"].square()
                + sum(value.square().sum()
                      for value in live_parent.parameters())
                + sum(value.square().sum()
                      for value in live_geometry.parameters())
            )
            result = _forward_state_with_reranker_stats(
                loss * 0.2, loss * 0.3
            )
            result.update({
                "runtime_outputs": {},
                "geometry_candidate_ious": torch.zeros(1, 1),
                "parent_candidate_ious": torch.zeros(1, 1),
                "parent_state": {},
            })
            return result

        calibration_updates.append(update)
        if update == 2 and not selected_snapshot:
            selected_snapshot.update({
                "mcln": copy.deepcopy(mcln.state_dict()),
                "parent": copy.deepcopy(live_parent.state_dict()),
                "geometry": copy.deepcopy(live_geometry.state_dict()),
            })
        selected = {
            0: [0.60, 0.60, 0.20, 0.20],
            2: [0.80, 0.80, 0.80, 0.20],
            4: [0.80, 0.80, 0.80, 0.20],
            6: [0.80, 0.80, 0.80, 0.20],
        }[update]
        count = len(selected)
        scores = torch.tensor([[1.0, 0.0]]).expand(count, -1).clone()
        ious = torch.tensor(selected).unsqueeze(1).repeat(1, 2)
        zero = end_points["model_signal"] * 0.0
        return {
            "parent_loss": zero,
            "geometry_loss": zero,
            "runtime_outputs": {
                "rec_geometry_runtime_mode": "flat_geometry_axis",
                "rec_geometry_scores": scores,
                "rec_geometry_valid_mask": torch.ones_like(
                    scores, dtype=torch.bool
                ),
            },
            "geometry_candidate_ious": ious,
            "parent_candidate_ious": ious,
            "parent_state": {
                "parent_top1_mask": torch.tensor(
                    [[True, False]]
                ).expand(count, -1).clone()
            },
        }

    def diagnostic_builder(_end_points, forward_state, _targets):
        selected = finetune_runner._selected_calibration_ious(
            forward_state
        )
        return {
            name: selected.clone() for name in DIAGNOSTIC_BRANCH_NAMES
        }

    def calibration_fn(*args, **kwargs):
        kwargs["diagnostic_builder"] = diagnostic_builder
        return finetune_runner.calibrate_rec_finetune(*args, **kwargs)

    def hungarian_loss(end_points, layers, set_criterion, **kwargs):
        events.append((
            "hungarian_gt_attached",
            int(mcln.completed_updates.item()) + 1,
        ))
        assert layers == 6
        assert set_criterion is criterion
        assert kwargs == {
            "query_points_obj_topk": 4,
            "source_choice_selector_loss_weight": 0.0,
            "mask_loss_scale": 0.1,
            "consistency_loss_scale": 0.1,
        }
        assert "center_label" in end_points
        assert "dataset_index" in end_points
        return end_points["model_signal"].square() * 0.1, end_points

    criterion = object()
    initialized = {
        "initial_state": {
            "mcln": mcln,
            "parent": parent,
            "geometry": geometry,
            "parent_artifact": {},
            "geometry_artifact": {},
            "groups": groups,
            "optimizer": optimizer,
            "set_criterion": criterion,
        },
        "data": {
            "fit_loader": fit_loader,
            "calibration_loader": calibration_loader,
            "calibration_view": SimpleNamespace(indices=(20, 21, 22, 23)),
            "validation_loader": _NeverAccess(),
            "test_loader": _NeverAccess(),
        },
        "smoke_steps": None,
        "train_data_contract": {
            "schema": "scanrefer-rec-finetune-train-data-v1",
            "synthetic": True,
        },
    }

    result = finetune_runner.run_rec_finetune(
        initialized,
        max_steps=6,
        calibration_steps=(0, 2, 4, 6),
        input_builder=input_builder,
        forward_fn=forward_fn,
        hungarian_loss_fn=hungarian_loss,
        calibration_fn=calibration_fn,
    )

    assert optimizer.zero_grad_calls == 9
    assert optimizer.zero_grad_records == [
        {"completed_updates": update, "kwargs": {"set_to_none": True}}
        for update in (0, 1, 2, 2, 3, 4, 4, 5, 6)
    ]
    assert optimizer.step_calls == 6
    assert result["completed_updates"] == 6
    assert result["stopped_early"] is False
    assert result["selected_step"] == 2
    assert [record["step"] for record in result["calibration_history"]] == [
        0, 2, 4, 6,
    ]
    assert calibration_updates == [0, 2, 4, 6, 2]
    assert result["selected_metrics"] == _metrics(
        0.75, 0.75, sample_count=4
    )
    assert result["reproduced_metrics"] == result["selected_metrics"]
    assert len(result["calibration_diagnostics_history"]) == 4
    assert result["selected_calibration_diagnostics"] == result[
        "reproduced_calibration_diagnostics"
    ]
    assert result["train_data_contract"] == initialized[
        "train_data_contract"
    ]
    assert result["train_data_contract"] is not initialized[
        "train_data_contract"
    ]
    diagnostics = result["training_diagnostics"]
    assert diagnostics["schema"] == "rec-finetune-training-diagnostics-v2"
    assert diagnostics["update_count"] == 6
    assert diagnostics["last_update"]["step"] == 6
    assert diagnostics["frozen_gradient_tensors_seen"] == 0
    assert diagnostics["all_present_finite"] is True
    assert all(
        loss["all_finite"] is True
        for loss in diagnostics["losses"].values()
    )
    assert all(
        group["all_present_finite"] is True
        for group in diagnostics["trainable_groups"].values()
    )
    json.dumps(diagnostics, allow_nan=False)
    assert fit_loader[-1]["dataset_index"].numel() == 1
    for key, model in (
            ("mcln", mcln), ("parent", parent), ("geometry", geometry)):
        _assert_state_equal(model.state_dict(), selected_snapshot[key])

    calibration_events = [
        event for event in events if event[0] == "mcln_calibration"
    ]
    assert calibration_events == [
        ("mcln_calibration", 0),
        ("mcln_calibration", 2),
        ("mcln_calibration", 4),
        ("mcln_calibration", 6),
        ("mcln_calibration", 2),
    ]
    fit_events = [
        event[0] for event in events
        if event[0] != "mcln_calibration"
    ]
    expected_fit_events = []
    for step in range(1, 7):
        expected_fit_events.extend((
            "train_mode", "zero_grad", "move", "input_builder",
            "mcln_fit", "rec_forward", "hungarian_gt_attached",
            "backward", "clip", "optimizer_step",
        ))
        if step in (2, 4, 6):
            expected_fit_events.append("zero_grad")
    assert fit_events == expected_fit_events


def test_post_update_calibration_releases_graph_and_clears_gradients():
    class GraphMcln(_LoopMcln):
        def forward(self, inputs):
            output = super().forward(inputs)
            output["fit_ephemeral"] = self.decoder.weight.square().sum()
            return output

    mcln = GraphMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    ephemeral_refs = []
    calibration_entries = []
    all_parameters = (
        groups["mcln_parameters"]
        + groups["parent_parameters"]
        + groups["geometry_parameters"]
    )

    def forward_fn(end_points, *_args, **_kwargs):
        ephemeral = end_points["fit_ephemeral"]
        ephemeral_refs.append(weakref.ref(ephemeral))
        loss = (
            ephemeral.square()
            + sum(value.square().sum() for value in parent.parameters())
            + sum(value.square().sum() for value in geometry.parameters())
        )
        return _forward_state_with_reranker_stats(loss * 0.2, loss * 0.3)

    def calibration_fn(*_args, **_kwargs):
        update = int(mcln.completed_updates.item())
        calibration_entries.append(update)
        if update:
            gc.collect()
            assert ephemeral_refs[-1]() is None
            assert all(parameter.grad is None for parameter in all_parameters)
        return _metrics(1.0, 1.0, sample_count=1)

    result = finetune_runner.fit_rec_finetune_one_epoch(
        mcln, parent, geometry, {}, {}, groups, optimizer, object(),
        [_loop_batch([0]), _loop_batch([1])], [_loop_batch([3])], (3,),
        torch.device("cpu"), max_steps=2, calibration_steps=(0, 1, 2),
        forward_fn=forward_fn,
        hungarian_loss_fn=lambda end_points, *_args, **_kwargs: (
            end_points["model_signal"].square() * 0.1, end_points
        ),
        calibration_fn=calibration_fn,
    )

    assert result["completed_updates"] == 2
    assert result["calibration_diagnostics_history"] == []
    assert result["selected_calibration_diagnostics"] is None
    assert result["reproduced_calibration_diagnostics"] is None
    assert calibration_entries == [0, 1, 2, 0]
    assert optimizer.zero_grad_records == [
        {"completed_updates": update, "kwargs": {"set_to_none": True}}
        for update in (0, 1, 1, 2)
    ]


def test_inactive_parameter_stays_none_uncovered_and_unmodified_next_update():
    class AlternatingParent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.tensor(1.0))
            self.second = torch.nn.Parameter(torch.tensor(2.0))

    mcln = _LoopMcln()
    parent = AlternatingParent()
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _TrackingAdamW(mcln, parent, groups)
    initial_parent = {
        name: parameter.detach().clone()
        for name, parameter in parent.named_parameters()
    }

    def forward_fn(end_points, *_args, **_kwargs):
        update = int(mcln.completed_updates.item())
        active = parent.first if update == 0 else parent.second
        geometry_loss = sum(
            parameter.square().sum()
            for parameter in geometry.parameters()
        )
        return _forward_state_with_reranker_stats(
            active.square(), geometry_loss
        )

    def calibration_fn(*_args, **_kwargs):
        update = int(mcln.completed_updates.item())
        value = 0.5 if update == 0 else 1.0
        return _metrics(value, value, sample_count=1)

    result = finetune_runner.fit_rec_finetune_one_epoch(
        mcln, parent, geometry, {}, {}, groups, optimizer, object(),
        [_loop_batch([0]), _loop_batch([1])], [_loop_batch([3])], (3,),
        torch.device("cpu"), max_steps=2, calibration_steps=(0, 2),
        forward_fn=forward_fn,
        hungarian_loss_fn=lambda end_points, *_args, **_kwargs: (
            end_points["model_signal"].square() * 0.1, end_points
        ),
        calibration_fn=calibration_fn,
    )

    assert len(optimizer.parent_after_steps) == 2
    assert optimizer.parent_gradient_present == [
        {"first": True, "second": False},
        {"first": False, "second": True},
    ]
    after_first, after_second = optimizer.parent_after_steps
    assert torch.equal(after_first["second"], initial_parent["second"])
    assert torch.equal(after_second["first"], after_first["first"])
    assert not torch.equal(after_second["second"], after_first["second"])
    parent_diagnostics = result["training_diagnostics"][
        "trainable_groups"
    ]["parent_reranker"]
    assert parent_diagnostics["parameter_tensor_count"] == 2
    assert parent_diagnostics["parameter_element_count"] == 2
    assert parent_diagnostics["gradient_tensor_count"] == {
        "min": 1, "max": 1, "last": 1,
    }
    assert parent_diagnostics["gradient_element_count"] == {
        "min": 1, "max": 1, "last": 1,
    }
    last_parent = result["training_diagnostics"]["last_update"]["groups"][
        "parent_reranker"
    ]
    assert last_parent["gradient_tensor_count"] == 1
    assert last_parent["finite_gradient_tensor_count"] == 1
    assert optimizer.zero_grad_records == [
        {"completed_updates": update, "kwargs": {"set_to_none": True}}
        for update in (0, 1, 2)
    ]
    assert all(
        parameter.grad is None
        for parameter in groups["parent_parameters"]
    )


def test_progress_is_canonical_stderr_only_for_calibrations_and_smoke_update(
        monkeypatch, capsys):
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    monkeypatch.setattr(
        finetune_runner,
        "_progress_cuda_memory",
        lambda _device: {"allocated_bytes": 111, "reserved_bytes": 222},
    )

    def forward_fn(end_points, *_args, **_kwargs):
        loss = (
            end_points["model_signal"].square()
            + sum(parameter.square().sum() for parameter in parent.parameters())
            + sum(
                parameter.square().sum()
                for parameter in geometry.parameters()
            )
        )
        return _forward_state_with_reranker_stats(loss * 0.2, loss * 0.3)

    def calibration_fn(*_args, **_kwargs):
        value = 0.5 if int(mcln.completed_updates.item()) == 0 else 1.0
        return _metrics(value, value, sample_count=1)

    result = finetune_runner.fit_rec_finetune_one_epoch(
        mcln, parent, geometry, {}, {}, groups, optimizer, object(),
        [_loop_batch([0])], [_loop_batch([3])], (3,),
        torch.device("cpu"), max_steps=1, calibration_steps=(0, 1),
        forward_fn=forward_fn,
        hungarian_loss_fn=lambda end_points, *_args, **_kwargs: (
            end_points["model_signal"].square() * 0.1, end_points
        ),
        calibration_fn=calibration_fn,
        smoke_run=True,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    raw_lines = captured.err.splitlines()
    records = [json.loads(line) for line in raw_lines]
    assert [(record["event"], record["phase"], record["step"])
            for record in records] == [
        ("calibration", "contract", 0),
        ("update", "fit", 1),
        ("calibration", "contract", 1),
        ("calibration", "selected_reproduction", 1),
    ]
    for raw, record in zip(raw_lines, records):
        assert raw == json.dumps(
            record, sort_keys=True, separators=(",", ":")
        )
        assert set(record) == {
            "schema", "event", "phase", "step", "loss_summary",
            "ranking_objectives", "grad_norms", "cuda_memory", "metrics",
        }
        assert record["schema"] == "rec-finetune-progress-v2"
        assert record["cuda_memory"] == {
            "allocated_bytes": 111, "reserved_bytes": 222,
        }
    assert records[0]["loss_summary"] is None
    assert records[0]["grad_norms"] is None
    assert records[0]["ranking_objectives"] is None
    assert records[0]["metrics"] == _metrics(0.5, 0.5, sample_count=1)
    for record in records[1:]:
        assert record["loss_summary"] == result[
            "training_diagnostics"
        ]["last_update"]["losses"]
        assert record["grad_norms"] == {
            name: group["preclip_gradient_norm"]
            for name, group in result["training_diagnostics"][
                "last_update"
            ]["groups"].items()
        }
        assert record["ranking_objectives"] == result[
            "training_diagnostics"
        ]["last_update"]["ranking_objectives"]
    assert records[1]["metrics"] is None
    assert finetune_runner.PRODUCTION_PROGRESS_INTERVAL == 25


def test_first_regression_stops_before_remainder_and_restores_best():
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    fit_loader = [_loop_batch([index]) for index in range(6)]
    calibration_updates = []
    selected_snapshot = {}

    def calibration_fn(*args, **kwargs):
        update = int(mcln.completed_updates.item())
        calibration_updates.append(update)
        if update == 2 and not selected_snapshot:
            selected_snapshot.update({
                "mcln": copy.deepcopy(mcln.state_dict()),
                "parent": copy.deepcopy(parent.state_dict()),
                "geometry": copy.deepcopy(geometry.state_dict()),
            })
        return {
            0: _metrics(0.50, 0.50, sample_count=2),
            2: _metrics(1.00, 1.00, sample_count=2),
            4: _metrics(0.50, 0.00, sample_count=2),
        }[update]

    def forward_fn(end_points, *_args, **_kwargs):
        loss = (
            end_points["model_signal"].square()
            + sum(value.square().sum() for value in parent.parameters())
            + sum(value.square().sum() for value in geometry.parameters())
        )
        return _forward_state_with_reranker_stats(loss * 0.2, loss * 0.3)

    result = finetune_runner.fit_rec_finetune_one_epoch(
        mcln, parent, geometry, {}, {}, groups, optimizer, object(),
        fit_loader, [_loop_batch([20, 21])], (20, 21),
        torch.device("cpu"), max_steps=6, calibration_steps=(0, 2, 4, 6),
        forward_fn=forward_fn,
        hungarian_loss_fn=lambda end_points, *_args, **_kwargs: (
            end_points["model_signal"].square() * 0.1, end_points
        ),
        calibration_fn=calibration_fn,
    )

    assert optimizer.zero_grad_calls == 6
    assert optimizer.zero_grad_records == [
        {"completed_updates": update, "kwargs": {"set_to_none": True}}
        for update in (0, 1, 2, 2, 3, 4)
    ]
    assert optimizer.step_calls == 4
    assert result["completed_updates"] == 4
    assert result["stopped_early"] is True
    assert result["selected_step"] == 2
    assert [record["step"] for record in result["calibration_history"]] == [
        0, 2, 4,
    ]
    assert calibration_updates == [0, 2, 4, 2]
    assert result["reproduced_metrics"] == result["selected_metrics"]
    for key, model in (
            ("mcln", mcln), ("parent", parent), ("geometry", geometry)):
        _assert_state_equal(model.state_dict(), selected_snapshot[key])


def test_fit_loop_rejects_target_collisions_and_nonfinite_total():
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    batch = _loop_batch([0])
    collision_forward_calls = []

    def collision_forward(*_args, **_kwargs):
        collision_forward_calls.append(True)
        zero = next(parent.parameters()).sum() * 0.0
        return _forward_state_with_reranker_stats(zero, zero)

    class CollisionMcln(_LoopMcln):
        def forward(self, inputs):
            output = super().forward(inputs)
            output["dataset_index"] = torch.zeros(1)
            return output

    colliding = CollisionMcln()
    collision_groups = rec_finetune.configure_rec_finetune_trainability(
        colliding, parent, geometry
    )
    collision_optimizer = _CountingSgd(colliding, collision_groups, [])
    with pytest.raises(ValueError, match="collision"):
        finetune_runner.fit_rec_finetune_one_epoch(
            colliding, parent, geometry, {}, {}, collision_groups,
            collision_optimizer, object(), [batch], [_loop_batch([3])],
            (3,), torch.device("cpu"), max_steps=1,
            calibration_steps=(0, 1), forward_fn=collision_forward,
            hungarian_loss_fn=lambda *_args, **_kwargs: (torch.tensor(0.0), {}),
            calibration_fn=lambda *_args, **_kwargs: _metrics(
                1.0, 1.0, sample_count=1
            ),
        )
    assert collision_optimizer.step_calls == 0
    assert collision_forward_calls == []

    def ordinary_forward(end_points, *_args, **_kwargs):
        zero = end_points["model_signal"] * 0.0
        return _forward_state_with_reranker_stats(zero, zero)

    with pytest.raises(FloatingPointError, match="finite"):
        finetune_runner.fit_rec_finetune_one_epoch(
            mcln, parent, geometry, {}, {}, groups, optimizer, object(),
            [batch], [_loop_batch([3])], (3,), torch.device("cpu"),
            max_steps=1, calibration_steps=(0, 1),
            forward_fn=ordinary_forward,
            hungarian_loss_fn=lambda *_args, **_kwargs: (
                next(mcln.parameters()).sum() * float("nan"), {}
            ),
            calibration_fn=lambda *_args, **_kwargs: _metrics(
                1.0, 1.0, sample_count=1
            ),
        )
    assert optimizer.step_calls == 0


def test_fit_loop_rejects_target_only_input_before_rec_forward():
    class PermissiveMcln(_LoopMcln):
        def forward(self, inputs):
            signal = sum(
                parameter.sum()
                for name, parameter in self.named_parameters()
                if name.startswith((
                    "decoder.", "decoder_query_proj.", "proposal_head.",
                    "prediction_heads.",
                ))
            )
            return {"model_signal": signal}

    mcln = PermissiveMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = _CountingSgd(mcln, groups, [])
    forward_calls = []

    def leaking_input_builder(batch):
        inputs = finetune_runner.build_rec_finetune_inputs(batch)
        inputs["center_label"] = batch["center_label"]
        return inputs

    def forward_spy(*_args, **_kwargs):
        forward_calls.append(True)
        raise AssertionError("REC forward observed target-only input")

    with pytest.raises(ValueError, match="target-only|ground-truth|GT"):
        finetune_runner.fit_rec_finetune_one_epoch(
            mcln, parent, geometry, {}, {}, groups, optimizer, object(),
            [_loop_batch([0])], [_loop_batch([3])], (3,),
            torch.device("cpu"), max_steps=1, calibration_steps=(0, 1),
            input_builder=leaking_input_builder,
            forward_fn=forward_spy,
            hungarian_loss_fn=lambda *_args, **_kwargs: (torch.tensor(0.0), {}),
            calibration_fn=lambda *_args, **_kwargs: _metrics(
                1.0, 1.0, sample_count=1
            ),
        )
    assert forward_calls == []
    assert optimizer.step_calls == 0

    class TargetOutputMcln(PermissiveMcln):
        def forward(self, inputs):
            output = super().forward(inputs)
            output["threshold_labels"] = torch.zeros(1)
            return output

    target_output_mcln = TargetOutputMcln()
    output_groups = rec_finetune.configure_rec_finetune_trainability(
        target_output_mcln, parent, geometry
    )
    output_optimizer = _CountingSgd(
        target_output_mcln, output_groups, []
    )
    with pytest.raises(ValueError, match="target-only|ground-truth|GT"):
        finetune_runner.fit_rec_finetune_one_epoch(
            target_output_mcln, parent, geometry, {}, {}, output_groups,
            output_optimizer, object(), [_loop_batch([0])],
            [_loop_batch([3])], (3,), torch.device("cpu"), max_steps=1,
            calibration_steps=(0, 1), forward_fn=forward_spy,
            hungarian_loss_fn=lambda *_args, **_kwargs: (torch.tensor(0.0), {}),
            calibration_fn=lambda *_args, **_kwargs: _metrics(
                1.0, 1.0, sample_count=1
            ),
        )
    assert forward_calls == []
    assert output_optimizer.step_calls == 0

    class MutatingInputMcln(PermissiveMcln):
        def forward(self, inputs):
            output = super().forward(inputs)
            inputs["geometry_ious"] = torch.zeros(1)
            return output

    mutating_mcln = MutatingInputMcln()
    mutating_groups = rec_finetune.configure_rec_finetune_trainability(
        mutating_mcln, parent, geometry
    )
    mutating_optimizer = _CountingSgd(mutating_mcln, mutating_groups, [])
    with pytest.raises(ValueError, match="target-only|ground-truth|GT"):
        finetune_runner.fit_rec_finetune_one_epoch(
            mutating_mcln, parent, geometry, {}, {}, mutating_groups,
            mutating_optimizer, object(), [_loop_batch([0])],
            [_loop_batch([3])], (3,), torch.device("cpu"), max_steps=1,
            calibration_steps=(0, 1), forward_fn=forward_spy,
            hungarian_loss_fn=lambda *_args, **_kwargs: (torch.tensor(0.0), {}),
            calibration_fn=lambda *_args, **_kwargs: _metrics(
                1.0, 1.0, sample_count=1
            ),
        )
    assert forward_calls == []
    assert mutating_optimizer.step_calls == 0


def _file_identity(path):
    path = Path(path)
    metadata = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "size": metadata.st_size,
        "sha256": digest.hexdigest(),
    }


def _fixed_runtime_cuda_snapshot():
    return {
        "device": {
            "type": "cuda", "index": 0, "name": "Synthetic GPU",
            "total_memory_bytes": 24_000_000_000,
        },
        "peak_cuda_memory": {
            "allocated_bytes": 1234, "reserved_bytes": 5678,
        },
    }


def test_runtime_provenance_is_exact_json_safe_and_uses_allowlisted_env(
        tmp_path, monkeypatch):
    interpreter = tmp_path / "python-real"
    interpreter.write_bytes(b"synthetic interpreter")
    logical = tmp_path / "python"
    logical.symlink_to(interpreter.name)
    monkeypatch.setattr(finetune_runner.sys, "executable", str(logical))
    monkeypatch.setattr(
        finetune_runner,
        "_runtime_cuda_snapshot",
        _fixed_runtime_cuda_snapshot,
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "/workspace:/workspace/pointnet2",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SECRET_NOT_ALLOWLISTED", "must-not-appear")

    runtime = finetune_runner.build_rec_finetune_runtime_provenance(
        started_utc="2026-07-16T10:00:00.000000Z",
        finished_utc="2026-07-16T10:00:02.500000Z",
        elapsed_seconds=2.5,
        command=[str(logical), "runner.py", "--smoke-steps", "1"],
    )

    assert set(runtime) == {
        "schema", "started_utc", "finished_utc", "elapsed_seconds",
        "completed_successfully", "oom_detected", "command",
        "interpreter", "versions", "device", "peak_cuda_memory",
        "environment",
    }
    assert runtime["schema"] == "rec-finetune-runtime-v1"
    assert runtime["started_utc"] == "2026-07-16T10:00:00.000000Z"
    assert runtime["finished_utc"] == "2026-07-16T10:00:02.500000Z"
    assert runtime["elapsed_seconds"] == 2.5
    assert runtime["completed_successfully"] is True
    assert runtime["oom_detected"] is False
    assert runtime["command"] == [
        str(logical), "runner.py", "--smoke-steps", "1",
    ]
    assert runtime["interpreter"] == {
        "logical_path": str(logical),
        "resolved_path": str(interpreter.resolve()),
    }
    assert set(runtime["versions"]) == {"python", "torch", "cuda", "cudnn"}
    assert isinstance(runtime["versions"]["python"], str)
    assert isinstance(runtime["versions"]["torch"], str)
    assert runtime["device"] == {
        "type": "cuda", "index": 0, "name": "Synthetic GPU",
        "total_memory_bytes": 24_000_000_000,
    }
    assert runtime["peak_cuda_memory"] == {
        "allocated_bytes": 1234, "reserved_bytes": 5678,
    }
    assert runtime["environment"] == environment
    assert "SECRET_NOT_ALLOWLISTED" not in json.dumps(runtime)
    json.dumps(runtime, allow_nan=False)

    with pytest.raises((ValueError, FloatingPointError), match="elapsed|finite"):
        finetune_runner.build_rec_finetune_runtime_provenance(
            started_utc="2026-07-16T10:00:00.000000Z",
            finished_utc="2026-07-16T10:00:02.500000Z",
            elapsed_seconds=float("nan"),
            command=["runner.py"],
        )
    with pytest.raises(ValueError, match="timestamp|finish|chronolog"):
        finetune_runner.build_rec_finetune_runtime_provenance(
            started_utc="2026-07-16T10:00:03.000000Z",
            finished_utc="2026-07-16T10:00:02.500000Z",
            elapsed_seconds=0.5,
            command=["runner.py"],
        )


def _live_synthetic_runtime(monkeypatch):
    monkeypatch.setattr(
        finetune_runner,
        "_runtime_cuda_snapshot",
        _fixed_runtime_cuda_snapshot,
    )
    return finetune_runner.build_rec_finetune_runtime_provenance(
        started_utc="2026-07-16T10:00:00.000000Z",
        finished_utc="2026-07-16T10:00:01.000000Z",
        elapsed_seconds=1.0,
        command=[finetune_runner.sys.executable, "runner.py"],
    )


def test_runtime_validator_rejects_fully_nonempty_forged_live_identity(
        monkeypatch):
    runtime = _live_synthetic_runtime(monkeypatch)
    runtime["command"][0] = "/bin/sh"
    runtime["interpreter"] = {
        "logical_path": "/bin/sh",
        "resolved_path": str(Path("/bin/sh").resolve()),
    }
    runtime["versions"] = {
        "python": "99.0.0",
        "torch": "99.0.0+forged",
        "cuda": "99.0",
        "cudnn": 1,
    }
    runtime["device"] = {
        "type": "cuda", "index": 0, "name": "Bogus GPU",
        "total_memory_bytes": 1,
    }
    runtime["peak_cuda_memory"] = {
        "allocated_bytes": 1, "reserved_bytes": 2,
    }

    with pytest.raises(ValueError, match="runtime|live|identity|version|device"):
        finetune_runner._validate_runtime_provenance(runtime)


@pytest.mark.parametrize(
    "mutation",
    (
        "interpreter", "python", "torch", "cuda", "cudnn", "device",
        "environment", "peak",
    ),
)
def test_runtime_validator_binds_every_recorded_identity_to_live_state(
        monkeypatch, mutation):
    runtime = _live_synthetic_runtime(monkeypatch)
    if mutation == "interpreter":
        runtime["command"][0] = "/bin/sh"
        runtime["interpreter"] = {
            "logical_path": "/bin/sh",
            "resolved_path": str(Path("/bin/sh").resolve()),
        }
    elif mutation in ("python", "torch", "cuda"):
        runtime["versions"][mutation] = "99.0-forged"
    elif mutation == "cudnn":
        runtime["versions"]["cudnn"] = 1
    elif mutation == "device":
        runtime["device"]["name"] = "Bogus GPU"
    elif mutation == "environment":
        current = runtime["environment"]["OMP_NUM_THREADS"]
        runtime["environment"]["OMP_NUM_THREADS"] = (
            "forged" if current != "forged" else "different"
        )
    else:
        runtime["peak_cuda_memory"]["allocated_bytes"] += 1

    with pytest.raises(ValueError, match=(
            "runtime|live|identity|version|device|environment|peak")):
        finetune_runner._validate_runtime_provenance(runtime)


def _assert_immutable_evidence(snapshot):
    for path, identity in snapshot.items():
        assert _file_identity(path) == identity, path


def _synthetic_training_diagnostics(mcln, groups, update_count):
    diagnostics = finetune_runner.create_rec_finetune_training_diagnostics(
        mcln, groups
    )
    losses = {
        "hungarian": 1.0, "parent": 2.0, "geometry": 3.0,
        "total": 6.0,
    }
    group_records = {}
    for name, group in diagnostics["trainable_groups"].items():
        tensor_count = group["parameter_tensor_count"]
        element_count = group["parameter_element_count"]
        norm = float(tensor_count + 1)
        group["gradient_tensor_count"] = {
            "min": tensor_count, "max": tensor_count, "last": tensor_count,
        }
        group["gradient_element_count"] = {
            "min": element_count, "max": element_count,
            "last": element_count,
        }
        group["preclip_gradient_norm"] = {
            "min": norm, "max": norm, "last": norm,
        }
        group_records[name] = {
            "parameter_tensor_count": tensor_count,
            "parameter_element_count": element_count,
            "parameter_names_sha256": group["parameter_names_sha256"],
            "gradient_tensor_count": tensor_count,
            "gradient_element_count": element_count,
            "finite_gradient_tensor_count": tensor_count,
            "finite_gradient_element_count": element_count,
            "all_present_finite": True,
            "preclip_gradient_norm": norm,
            "clip_limit": group["clip_limit"],
        }
    for name, value in losses.items():
        diagnostics["losses"][name] = {
            "min": value, "max": value, "last": value,
            "all_finite": True,
        }
    objective_records = {
        "parent": {
            "loss_listwise": 0.7,
            "loss_best_tier_pairwise": 0.5,
            "loss_ranking": 0.7,
            "loss_threshold": 0.2,
            "loss_iou": 0.1,
            "loss_total": losses["parent"],
            "tier_pairwise_informative_rows": 2,
            "tier_pairwise_pair_count": 6,
            "tier_pairwise_positive_count": 5,
            "tier_pairwise_negative_count": 3,
        },
        "geometry": {
            "loss_listwise": 0.7,
            "loss_best_tier_pairwise": 0.5,
            "loss_ranking": 0.5,
            "loss_threshold": 0.2,
            "loss_iou": 0.1,
            "loss_total": losses["geometry"],
            "tier_pairwise_informative_rows": 3,
            "tier_pairwise_pair_count": 9,
            "tier_pairwise_positive_count": 6,
            "tier_pairwise_negative_count": 4,
        },
    }
    for name, record in objective_records.items():
        for field in finetune_runner._DIAGNOSTIC_RERANKER_LOSS_NAMES:
            value = record[field]
            diagnostics["ranking_objectives"][name][field] = {
                "min": value, "max": value, "last": value,
                "all_finite": True,
            }
        for field in finetune_runner._DIAGNOSTIC_RERANKER_COUNT_NAMES:
            value = record[field]
            diagnostics["ranking_objectives"][name][field] = {
                "min": value, "max": value, "last": value,
                "total": value * update_count,
            }
    diagnostics["update_count"] = update_count
    diagnostics["last_update"] = {
        "schema": "rec-finetune-update-diagnostics-v2",
        "step": update_count,
        "losses": losses,
        "ranking_objectives": objective_records,
        "groups": group_records,
        "frozen_mcln": {
            **copy.deepcopy(diagnostics["frozen_mcln"]),
            "gradient_tensor_count": 0,
            "gradient_element_count": 0,
        },
    }
    return diagnostics


def _synthetic_runtime_provenance():
    logical_interpreter = str(finetune_runner.sys.executable)
    cuda = finetune_runner._runtime_cuda_snapshot()
    return {
        "schema": "rec-finetune-runtime-v1",
        "started_utc": "2026-07-16T10:00:00.000000Z",
        "finished_utc": "2026-07-16T10:00:01.000000Z",
        "elapsed_seconds": 1.0,
        "completed_successfully": True,
        "oom_detected": False,
        "command": [logical_interpreter, "runner.py"],
        "interpreter": {
            "logical_path": logical_interpreter,
            "resolved_path": str(Path(logical_interpreter).resolve()),
        },
        "versions": {
            "python": str(finetune_runner.platform.python_version()),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version()),
        },
        "device": copy.deepcopy(cuda["device"]),
        "peak_cuda_memory": copy.deepcopy(cuda["peak_cuda_memory"]),
        "environment": {
            name: os.environ.get(name)
            for name in finetune_runner.RUNTIME_ENVIRONMENT_ALLOWLIST
        },
    }


def test_training_diagnostics_reject_geometry_without_pairwise_signal():
    mcln = _LoopMcln()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    diagnostics = _synthetic_training_diagnostics(mcln, groups, 1)
    for field in (
            "tier_pairwise_informative_rows",
            "tier_pairwise_pair_count",
            "tier_pairwise_negative_count"):
        diagnostics["ranking_objectives"]["geometry"][field] = {
            "min": 0, "max": 0, "last": 0, "total": 0,
        }
        diagnostics["last_update"]["ranking_objectives"]["geometry"][
            field
        ] = 0

    with pytest.raises(ValueError, match="pairwise|training signal"):
        finetune_runner._validate_training_diagnostics(
            diagnostics, 1, mcln=mcln, groups=groups
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "command-empty", "command-whitespace", "command-arg-whitespace",
        "command-interpreter-mismatch", "logical-empty",
        "logical-whitespace", "resolved-empty", "resolved-mismatch",
        "python-empty", "torch-whitespace", "cuda-none", "cuda-empty",
        "cudnn-none", "cudnn-zero", "gpu-empty",
    ),
)
def test_runtime_validator_rejects_blank_or_incoherent_identity_fields(
        monkeypatch, mutation):
    monkeypatch.setattr(
        finetune_runner,
        "_runtime_cuda_snapshot",
        _fixed_runtime_cuda_snapshot,
    )
    runtime = _synthetic_runtime_provenance()
    if mutation == "command-empty":
        runtime["command"] = [""]
    elif mutation == "command-whitespace":
        runtime["command"] = ["   "]
    elif mutation == "command-arg-whitespace":
        runtime["command"][1] = "\t"
    elif mutation == "command-interpreter-mismatch":
        runtime["command"][0] = "/different/python"
    elif mutation == "logical-empty":
        runtime["interpreter"]["logical_path"] = ""
    elif mutation == "logical-whitespace":
        runtime["interpreter"]["logical_path"] = "  "
    elif mutation == "resolved-empty":
        runtime["interpreter"]["resolved_path"] = ""
    elif mutation == "resolved-mismatch":
        runtime["interpreter"]["resolved_path"] = "/different/python"
    elif mutation == "python-empty":
        runtime["versions"]["python"] = ""
    elif mutation == "torch-whitespace":
        runtime["versions"]["torch"] = "\t"
    elif mutation == "cuda-none":
        runtime["versions"]["cuda"] = None
    elif mutation == "cuda-empty":
        runtime["versions"]["cuda"] = ""
    elif mutation == "cudnn-none":
        runtime["versions"]["cudnn"] = None
    elif mutation == "cudnn-zero":
        runtime["versions"]["cudnn"] = 0
    else:
        runtime["device"]["name"] = " "

    with pytest.raises(ValueError, match=(
            "runtime|command|interpreter|identity|CUDA|version|device")):
        finetune_runner._validate_runtime_provenance(runtime)


def _synthetic_train_data_contract():
    metadata = copy.deepcopy(
        rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
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
        "authoritative_split_metadata": metadata,
        "authoritative_split_mapping_sha256": metadata["mapping_sha256"],
        "fit_sample_count": metadata["fit_sample_count"],
        "calibration_sample_count": metadata["calibration_sample_count"],
        "fit_loader_batch_count": rec_finetune.natural_batch_count(
            metadata["fit_sample_count"], 18
        ),
        "calibration_loader_batch_count": rec_finetune.natural_batch_count(
            metadata["calibration_sample_count"], 18
        ),
        "batch_size": 18,
        "drop_last": False,
        "validation_data_accessed": False,
        "dataset_class": "src.joint_det_dataset.Joint3DDataset",
        "dataset_instance_count": 1,
        "fit_and_calibration_share_source_annotations": True,
        "validation_data_objects_present": False,
    }


def _publication_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        finetune_runner,
        "_runtime_cuda_snapshot",
        _fixed_runtime_cuda_snapshot,
    )
    inputs = tmp_path / "immutable-inputs"
    inputs.mkdir()
    backbone_path = inputs / "epoch71.pth"
    backbone_path.write_bytes(b"synthetic authoritative epoch-71 checkpoint")
    backbone_sha = _file_identity(backbone_path)["sha256"]

    initial_parent = torch.load(INITIAL_PARENT_ARTIFACT, map_location="cpu")
    initial_parent["checkpoint_sha256"] = backbone_sha
    parent_path = inputs / "initial-parent.pth"
    torch.save(initial_parent, parent_path)
    parent_sha = _file_identity(parent_path)["sha256"]

    initial_geometry = torch.load(
        INITIAL_GEOMETRY_ARTIFACT, map_location="cpu"
    )
    initial_geometry["checkpoint_sha256"] = backbone_sha
    initial_geometry["parent_artifact_sha256"] = parent_sha
    geometry_path = inputs / "initial-geometry.pth"
    torch.save(initial_geometry, geometry_path)
    geometry_sha = _file_identity(geometry_path)["sha256"]
    for path in (backbone_path, parent_path, geometry_path):
        path.chmod(0o444)

    monkeypatch.setattr(
        rec_finetune,
        "AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256",
        backbone_sha,
    )
    monkeypatch.setattr(
        rec_finetune,
        "AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256",
        parent_sha,
    )
    monkeypatch.setattr(
        rec_finetune,
        "AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256",
        geometry_sha,
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_BACKBONE_SHA256", backbone_sha
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_PARENT_SHA256", parent_sha
    )
    monkeypatch.setattr(
        finetune_runner, "EXPECTED_GEOMETRY_SHA256", geometry_sha
    )

    torch.manual_seed(71)
    mcln = _LoopMcln()
    parent = QueryReranker(152, hidden_dim=8, dropout=0.0)
    geometry = QueryReranker(179, hidden_dim=8, dropout=0.0)
    groups = rec_finetune.configure_rec_finetune_trainability(
        mcln, parent, geometry
    )
    optimizer = rec_finetune.build_rec_finetune_optimizer(groups)
    optimizer.state[groups["mcln_parameters"][0]] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.ones_like(groups["mcln_parameters"][0]),
        "exp_avg_sq": torch.ones_like(groups["mcln_parameters"][0]),
    }
    calibration_count = (
        rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0[
            "calibration_sample_count"
        ]
    )
    baseline = _metrics(0.50, 0.40, sample_count=calibration_count)
    selected = _metrics(0.70, 0.50, sample_count=calibration_count)
    regressed = _metrics(0.60, 0.30, sample_count=calibration_count)
    history = [
        {
            "step": 0,
            "metrics": baseline,
            "eligible": True,
            "regression": False,
            "action": "continue",
            "best_step": 0,
        },
        {
            "step": 306,
            "metrics": selected,
            "eligible": True,
            "regression": False,
            "action": "continue",
            "best_step": 306,
        },
        {
            "step": 612,
            "metrics": regressed,
            "eligible": False,
            "regression": True,
            "action": "stop",
            "best_step": 306,
        },
    ]
    output_dir = tmp_path / "published"
    paths = finetune_runner.RecFinetuneRuntimePaths(
        data_root=tmp_path,
        backbone_checkpoint=backbone_path,
        parent_reranker=parent_path,
        geometry_reranker=geometry_path,
        output_dir=output_dir,
    )
    config = SimpleNamespace(
        use_soft_token_loss=True,
        use_contrastive_align=True,
    )
    train_data_contract = _synthetic_train_data_contract()
    initialized = {
        "paths": paths,
        "initial_state": {
            "checkpoint_path": backbone_path,
            "checkpoint_sha256": backbone_sha,
            "checkpoint_epoch": 71,
            "config": config,
            "mcln": mcln,
            "parent": parent,
            "parent_artifact": initial_parent,
            "geometry": geometry,
            "geometry_artifact": initial_geometry,
            "groups": groups,
            "optimizer": optimizer,
        },
        "data": {
            "dataset": object(),
            "split": {
                "metadata": copy.deepcopy(
                    rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
                )
            },
            "fit_view": object(),
            "calibration_loader": [_loop_batch(list(range(10)))],
            "calibration_view": SimpleNamespace(indices=tuple(range(10))),
            "fit_loader": object(),
        },
        "train_data_contract": train_data_contract,
        "publication_code_hashes": (
            finetune_runner._publication_code_hashes()
        ),
    }
    monkeypatch.setattr(
        finetune_runner,
        "build_rec_finetune_train_data_contract",
        lambda live: copy.deepcopy(live["train_data_contract"]),
    )
    result = {
        "completed_updates": 612,
        "stopped_early": True,
        "selected_step": 306,
        "selected_metrics": selected,
        "reproduced_metrics": copy.deepcopy(selected),
        "calibration_history": history,
        "training_diagnostics": _synthetic_training_diagnostics(
            mcln, groups, 612
        ),
        "runtime": _synthetic_runtime_provenance(),
        "train_data_contract": copy.deepcopy(
            initialized["train_data_contract"]
        ),
        **_synthetic_diagnostic_run_fields(history, 306),
    }
    immutable_evidence = {
        str(path): _file_identity(path)
        for path in (
            (INITIAL_PARENT_ARTIFACT,)
            + SEALED_OFFICIAL_EVIDENCE
            + (backbone_path, parent_path, geometry_path)
        )
    }
    parity_calls = []

    def calibration_fn(reloaded_mcln, reloaded_parent, reloaded_geometry,
                       *_args, **_kwargs):
        parity_calls.append(True)
        for live, reloaded in (
                (mcln, reloaded_mcln),
                (parent, reloaded_parent),
                (geometry, reloaded_geometry)):
            _assert_state_equal(reloaded.state_dict(), live.state_dict())
        return _synthetic_observation_for_metrics(selected)

    return {
        "initialized": initialized,
        "result": result,
        "output_dir": output_dir,
        "initial_shas": {
            "backbone": backbone_sha,
            "parent": parent_sha,
            "geometry": geometry_sha,
        },
        "immutable_evidence": immutable_evidence,
        "model_factory": lambda _config: _LoopMcln(),
        "calibration_fn": calibration_fn,
        "parity_calls": parity_calls,
    }


class _InjectedPublicationFailure(RuntimeError):
    pass


def test_noreplace_rename_fails_closed_without_atomic_primitive(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "marker"
    marker.write_text("source remains", encoding="ascii")
    destination = tmp_path / "destination"
    rename_calls = []
    monkeypatch.setattr(
        finetune_runner.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        finetune_runner.os,
        "rename",
        lambda *_args, **_kwargs: rename_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="RENAME_NOREPLACE|unavailable"):
        finetune_runner._rename_directory_noreplace(source, destination)

    assert rename_calls == []
    assert marker.read_text("ascii") == "source remains"
    assert not destination.exists()


def test_publication_code_hash_snapshot_covers_ranking_target_implementation():
    hashes = finetune_runner._publication_code_hashes()

    assert {
        "runner",
        "rec_finetune",
        "rec_reranker",
        "rec_candidate_adapter",
        "rec_mask_geometry",
        "rec_geometry_reranker",
        "source_choice_selector",
        "losses",
        "train_dist_mod",
        "source/models/mcln.py",
        "source/models/mcln_attention.py",
        "source/models/encoder_decoder_layers.py",
        "source/models/backbone_module.py",
        "source/src/joint_det_dataset.py",
        "source/scripts/train_rec_reranker.py",
        "source/scripts/rec_geometry_cache.py",
        "binary/pointnet2/_ext.cpython-37m-x86_64-linux-gnu.so",
    }.issubset(hashes)


def test_publication_rejects_code_hash_changes_after_training_start(
        monkeypatch):
    initial = finetune_runner._publication_code_hashes()
    changed = copy.deepcopy(initial)
    changed["rec_reranker"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        finetune_runner, "_publication_code_hashes", lambda: changed
    )

    with pytest.raises(RuntimeError, match="code.*changed|source.*changed"):
        finetune_runner._require_unchanged_publication_code_hashes(initial)


def test_publication_objective_contract_is_independent_of_mutable_helper(
        monkeypatch):
    expected = finetune_runner._loss_publication_contract()
    monkeypatch.setattr(
        rec_finetune,
        "rec_finetune_ranking_objective_contract",
        lambda: {
            "parent": {"name": "forged", "tier_pairwise_alpha": 1.0},
            "geometry": {"name": "forged", "tier_pairwise_alpha": 0.0},
        },
    )

    assert finetune_runner._loss_publication_contract() == expected


def test_noreplace_rename_fails_closed_on_enosys_during_destination_race(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "marker"
    marker.write_text("source remains", encoding="ascii")
    destination = tmp_path / "destination"

    class MissingRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, _source_fd, _source, _destination_fd,
                     destination_bytes, _flags):
            Path(os.fsdecode(destination_bytes)).mkdir()
            finetune_runner.ctypes.set_errno(finetune_runner.errno.ENOSYS)
            return -1

    monkeypatch.setattr(
        finetune_runner.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(
            renameat2=MissingRenameAt2()
        ),
    )

    with pytest.raises(RuntimeError, match="RENAME_NOREPLACE|unavailable"):
        finetune_runner._rename_directory_noreplace(source, destination)

    assert marker.read_text("ascii") == "source remains"
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize(
    "failure_stage",
    ["backbone", "parent", "geometry", "selection", "finalize"],
)
def test_publication_interruptions_leave_no_final_or_staging_output(
        tmp_path, monkeypatch, failure_stage):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    events = []

    def inject(stage):
        events.append(stage)
        if stage == failure_stage:
            raise _InjectedPublicationFailure(stage)

    with pytest.raises(_InjectedPublicationFailure, match=failure_stage):
        finetune_runner.publish_rec_finetune_run(
            fixture["initialized"],
            fixture["result"],
            failure_injector=inject,
            model_factory=fixture["model_factory"],
            calibration_fn=fixture["calibration_fn"],
        )

    ordered = ["backbone", "parent", "geometry", "selection", "finalize"]
    assert events == ordered[:ordered.index(failure_stage) + 1]
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []
    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_publication_rejects_backbone_metadata_mutated_by_returning_hook(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    events = []

    def inject(stage):
        events.append(stage)
        if stage == "backbone":
            staged = list(
                tmp_path.glob(".published.staging-*/backbone.pth")
            )
            assert len(staged) == 1
            checkpoint = torch.load(staged[0], map_location="cpu")
            checkpoint["rec_finetune"]["selected_step"] += 1
            torch.save(checkpoint, staged[0])

    with pytest.raises(
            RuntimeError, match="published REC checkpoint SHA-256 changed"):
        finetune_runner.publish_rec_finetune_run(
            fixture["initialized"],
            fixture["result"],
            failure_injector=inject,
            model_factory=fixture["model_factory"],
            calibration_fn=fixture["calibration_fn"],
        )

    assert events == ["backbone", "parent", "geometry"]
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []
    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_publication_rejects_selection_mutated_by_returning_hook(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    events = []

    def inject(stage):
        events.append(stage)
        if stage == "selection":
            staged = list(
                tmp_path.glob(".published.staging-*/selection.json")
            )
            assert len(staged) == 1
            selection = json.loads(staged[0].read_text("utf-8"))
            selection["selected_step"] += 1
            staged[0].write_text(
                json.dumps(
                    selection, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

    with pytest.raises(
            RuntimeError, match="staged selection SHA-256 changed"):
        finetune_runner.publish_rec_finetune_run(
            fixture["initialized"],
            fixture["result"],
            failure_injector=inject,
            model_factory=fixture["model_factory"],
            calibration_fn=fixture["calibration_fn"],
        )

    assert events == ["backbone", "parent", "geometry", "selection"]
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []
    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_publication_releases_live_models_and_optimizer_before_reload(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    state = fixture["initialized"]["initial_state"]
    live_models = (
        ("mcln", state["mcln"]),
        ("parent", state["parent"]),
        ("geometry", state["geometry"]),
    )
    for _label, model in live_models:
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
    assert state["optimizer"].state

    lifecycle = []
    for label, model in live_models:
        original_to = model.to

        def tracked_to(device, *args, _label=label, _to=original_to,
                       **kwargs):
            lifecycle.append(("live_to", _label, str(torch.device(device))))
            return _to(device, *args, **kwargs)

        monkeypatch.setattr(model, "to", tracked_to)

    base_factory = fixture["model_factory"]

    def lifecycle_factory(config):
        lifecycle.append(("reload_factory",))
        assert lifecycle[:3] == [
            ("live_to", "mcln", "cpu"),
            ("live_to", "parent", "cpu"),
            ("live_to", "geometry", "cpu"),
        ]
        assert all(
            parameter.device.type == "cpu" and parameter.grad is None
            for _label, model in live_models
            for parameter in model.parameters()
        )
        assert not state["optimizer"].state
        return base_factory(config)

    finetune_runner.publish_rec_finetune_run(
        fixture["initialized"],
        fixture["result"],
        model_factory=lifecycle_factory,
        calibration_fn=fixture["calibration_fn"],
    )

    assert lifecycle[:4] == [
        ("live_to", "mcln", "cpu"),
        ("live_to", "parent", "cpu"),
        ("live_to", "geometry", "cpu"),
        ("reload_factory",),
    ]
    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_publication_binds_reloads_and_seals_all_outputs(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    events = []

    publication = finetune_runner.publish_rec_finetune_run(
        fixture["initialized"],
        fixture["result"],
        failure_injector=events.append,
        model_factory=fixture["model_factory"],
        calibration_fn=fixture["calibration_fn"],
    )

    output_dir = fixture["output_dir"]
    assert events == [
        "backbone", "parent", "geometry", "selection", "finalize",
    ]
    assert publication["publication_order"] == [
        "backbone", "parent", "geometry", "selection",
    ]
    assert fixture["parity_calls"] == [True]
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "backbone.pth", "geometry.pth", "parent.pth", "selection.json",
    ]
    assert list(tmp_path.glob(".published.staging-*")) == []

    output_paths = {
        "backbone": output_dir / "backbone.pth",
        "parent": output_dir / "parent.pth",
        "geometry": output_dir / "geometry.pth",
        "selection": output_dir / "selection.json",
    }
    for path in output_paths.values():
        assert path.stat().st_mode & 0o777 == 0o444

    checkpoint = torch.load(output_paths["backbone"], map_location="cpu")
    assert set(checkpoint) == {"model", "config", "epoch", "rec_finetune"}
    assert checkpoint["epoch"] == 71
    assert checkpoint["rec_finetune"] == {
        "schema": "rec-finetune-selected-step-v2",
        "selected_step": 306,
        "selected_metrics": fixture["result"]["selected_metrics"],
        "completed_updates": 612,
        "stopped_early": True,
    }
    assert "optimizer" not in checkpoint
    assert "scheduler" not in checkpoint
    reloaded_mcln = _LoopMcln()
    reloaded_mcln.load_state_dict(checkpoint["model"], strict=True)
    _assert_state_equal(
        reloaded_mcln.state_dict(),
        fixture["initialized"]["initial_state"]["mcln"].state_dict(),
    )

    parent_model, parent_artifact, geometry_model, geometry_artifact = (
        rec_finetune.load_rec_finetune_runtime_artifacts(
            output_paths["parent"], output_paths["geometry"], device="cpu"
        )
    )
    _assert_state_equal(
        parent_model.state_dict(),
        fixture["initialized"]["initial_state"]["parent"].state_dict(),
    )
    _assert_state_equal(
        geometry_model.state_dict(),
        fixture["initialized"]["initial_state"]["geometry"].state_dict(),
    )
    backbone_sha = rec_finetune.sha256_file(output_paths["backbone"])
    parent_sha = rec_finetune.sha256_file(output_paths["parent"])
    geometry_sha = rec_finetune.sha256_file(output_paths["geometry"])
    assert parent_artifact["checkpoint_sha256"] == backbone_sha
    assert geometry_artifact["checkpoint_sha256"] == backbone_sha
    assert geometry_artifact["parent_artifact_sha256"] == parent_sha
    assert publication["sha256"] == {
        "backbone": backbone_sha,
        "parent": parent_sha,
        "geometry": geometry_sha,
        "selection": rec_finetune.sha256_file(output_paths["selection"]),
    }

    selection = json.loads(output_paths["selection"].read_text("utf-8"))
    assert set(selection) == {
        "schema", "version", "files", "authoritative_split",
        "mcln_trainable_parameter_names", "losses", "optimizer_groups",
        "calibration_history", "selected_step", "completed_updates",
        "stopped_early", "selected_metrics", "validation_data_accessed",
        "no_validation_data_declaration", "code_hashes",
        "publication_order", "training_diagnostics", "runtime",
        "train_data_contract", "calibration_diagnostics_history",
        "selected_calibration_diagnostics",
        "reproduced_calibration_diagnostics",
        "selected_calibration_output_sha256",
        "reproduced_calibration_output_sha256",
        "reloaded_calibration_metrics",
        "reloaded_calibration_diagnostics",
        "reloaded_calibration_output_sha256",
    }
    assert selection["schema"] == "rec-finetune-selection-v3"
    assert selection["version"] == 3
    assert selection["selected_step"] == 306
    assert selection["completed_updates"] == 612
    assert selection["stopped_early"] is True
    assert selection["validation_data_accessed"] is False
    assert selection["no_validation_data_declaration"] == (
        "scanrefer-train-only-no-validation-v1"
    )
    assert selection["publication_order"] == [
        "backbone", "parent", "geometry", "selection",
    ]
    assert selection["authoritative_split"] == (
        rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
    )
    assert selection["mcln_trainable_parameter_names"] == sorted(
        fixture["initialized"]["initial_state"]["groups"]["mcln_names"]
    )
    assert selection["losses"] == {
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
    expected_groups = [
        {"name": "mcln_decoder_box", "lr": 2e-5,
         "weight_decay": 5e-4, "grad_clip": 0.1},
        {"name": "parent_reranker", "lr": 1e-3,
         "weight_decay": 1e-4, "grad_clip": 1.0},
        {"name": "geometry_reranker", "lr": 3e-4,
         "weight_decay": 1e-4, "grad_clip": 1.0},
    ]
    assert selection["optimizer_groups"] == expected_groups
    assert parent_artifact["provenance"]["optimizer_groups"] == expected_groups
    assert selection["calibration_history"] == (
        fixture["result"]["calibration_history"]
    )
    assert selection["calibration_diagnostics_history"] == fixture[
        "result"
    ]["calibration_diagnostics_history"]
    assert selection["selected_calibration_diagnostics"] == fixture[
        "result"
    ]["selected_calibration_diagnostics"]
    assert selection["reproduced_calibration_diagnostics"] == fixture[
        "result"
    ]["reproduced_calibration_diagnostics"]
    assert selection["selected_calibration_output_sha256"] == fixture[
        "result"
    ]["selected_calibration_output_sha256"]
    assert selection["reproduced_calibration_output_sha256"] == fixture[
        "result"
    ]["reproduced_calibration_output_sha256"]
    assert selection["reloaded_calibration_metrics"] == fixture["result"][
        "selected_metrics"
    ]
    assert selection["reloaded_calibration_diagnostics"] == fixture[
        "result"
    ]["selected_calibration_diagnostics"]
    assert selection["reloaded_calibration_output_sha256"] == fixture[
        "result"
    ]["selected_calibration_output_sha256"]
    assert selection["training_diagnostics"] == fixture["result"][
        "training_diagnostics"
    ]
    assert selection["runtime"] == fixture["result"]["runtime"]
    assert selection["train_data_contract"] == fixture["result"][
        "train_data_contract"
    ]
    assert selection["selected_metrics"]["sample_count"] == selection[
        "train_data_contract"
    ]["calibration_sample_count"]
    assert parent_artifact["provenance"]["calibration_history"] == (
        fixture["result"]["calibration_history"]
    )
    diagnostic_fields = {
        "calibration_diagnostics_history",
        "selected_calibration_diagnostics",
        "reproduced_calibration_diagnostics",
        "selected_calibration_output_sha256",
        "reproduced_calibration_output_sha256",
        "verified_calibration_metrics",
        "verified_calibration_diagnostics",
        "verified_calibration_output_sha256",
        "reloaded_calibration_metrics",
        "reloaded_calibration_diagnostics",
        "reloaded_calibration_output_sha256",
    }
    assert diagnostic_fields.isdisjoint(parent_artifact)
    assert diagnostic_fields.isdisjoint(geometry_artifact)
    assert diagnostic_fields.isdisjoint(parent_artifact["provenance"])
    assert diagnostic_fields.isdisjoint(geometry_artifact["provenance"])
    assert set(parent_artifact["calibration_metrics"]) == {
        "sample_count", "hits025", "hits050", "acc025", "acc050", "score",
    }
    assert set(geometry_artifact["calibration_metrics"]) == {
        "sample_count", "hits025", "hits050", "acc025", "acc050", "score",
    }

    expected_files = {
        "initial_backbone": (
            fixture["initialized"]["paths"].backbone_checkpoint,
            fixture["initial_shas"]["backbone"],
        ),
        "initial_parent": (
            fixture["initialized"]["paths"].parent_reranker,
            fixture["initial_shas"]["parent"],
        ),
        "initial_geometry": (
            fixture["initialized"]["paths"].geometry_reranker,
            fixture["initial_shas"]["geometry"],
        ),
        "final_backbone": (output_paths["backbone"], backbone_sha),
        "final_parent": (output_paths["parent"], parent_sha),
        "final_geometry": (output_paths["geometry"], geometry_sha),
    }
    assert set(selection["files"]) == set(expected_files)
    for key, (path, digest) in expected_files.items():
        assert selection["files"][key] == {
            "path": str(path.resolve()), "sha256": digest,
        }

    code_paths = {
        "runner": Path(finetune_runner.__file__).resolve(),
        "rec_finetune": Path(rec_finetune.__file__).resolve(),
        "rec_reranker": (
            Path(rec_finetune.__file__).with_name("rec_reranker.py").resolve()
        ),
        "rec_candidate_adapter": (
            Path(rec_finetune.__file__)
            .with_name("rec_candidate_adapter.py").resolve()
        ),
        "rec_mask_geometry": (
            Path(rec_finetune.__file__)
            .with_name("rec_mask_geometry.py").resolve()
        ),
        "rec_geometry_reranker": (
            Path(rec_finetune.__file__)
            .with_name("rec_geometry_reranker.py").resolve()
        ),
        "source_choice_selector": (
            Path(rec_finetune.__file__)
            .with_name("source_choice_selector.py").resolve()
        ),
        "losses": Path(rec_finetune.__file__).with_name("losses.py").resolve(),
        "train_dist_mod": (
            Path(finetune_runner.__file__).resolve().parents[1]
            / "train_dist_mod.py"
        ),
    }
    assert set(code_paths).issubset(selection["code_hashes"])
    for key, path in code_paths.items():
        assert selection["code_hashes"][key] == {
            "path": str(path),
            "sha256": rec_finetune.sha256_file(path),
        }

    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_publication_preserves_reloaded_oracle_drift_as_third_observation(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    base_calibration = fixture["calibration_fn"]

    def oracle_drift_calibration(*args, **kwargs):
        observation = base_calibration(*args, **kwargs)
        transition = observation.diagnostics_result.transition_state
        return _synthetic_calibration_observation(
            transition.selected_ious,
            tuple(max(value, 0.90) for value in transition.selected_ious),
            transition.expected_indices,
        )

    publication = finetune_runner.publish_rec_finetune_run(
        fixture["initialized"],
        fixture["result"],
        model_factory=fixture["model_factory"],
        calibration_fn=oracle_drift_calibration,
    )

    selection = publication["selection"]
    selected_digest = fixture["result"][
        "selected_calibration_output_sha256"
    ]
    assert selection["schema"] == "rec-finetune-selection-v3"
    assert selection["version"] == 3
    assert selection["selected_calibration_output_sha256"] == selected_digest
    assert selection[
        "reproduced_calibration_output_sha256"
    ] == selected_digest
    assert selection["reloaded_calibration_output_sha256"] == selected_digest
    assert selection["reloaded_calibration_metrics"] == fixture["result"][
        "selected_metrics"
    ]
    assert selection["reloaded_calibration_diagnostics"] != selection[
        "selected_calibration_diagnostics"
    ]


def test_publication_rejects_invalid_reloaded_diagnostics_with_exact_digest(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    base_calibration = fixture["calibration_fn"]

    def invalid_diagnostics_calibration(*args, **kwargs):
        observation = base_calibration(*args, **kwargs)
        diagnostics_result = copy.deepcopy(observation.diagnostics_result)
        diagnostics_result.diagnostics["schema"] = "wrong"
        return finetune_runner.CalibrationObservation(
            selection_metrics=observation.selection_metrics,
            diagnostics_result=diagnostics_result,
        )

    with pytest.raises(ValueError, match="diagnostic"):
        finetune_runner.publish_rec_finetune_run(
            fixture["initialized"],
            fixture["result"],
            model_factory=fixture["model_factory"],
            calibration_fn=invalid_diagnostics_calibration,
        )
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []


def test_production_publication_rejects_impossible_loop_results(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    incomplete = copy.deepcopy(fixture["result"])
    incomplete["calibration_history"] = incomplete[
        "calibration_history"
    ][:2]
    incomplete["completed_updates"] = 306
    incomplete["stopped_early"] = False
    mismatched_selection = copy.deepcopy(fixture["result"])
    mismatched_selection["selected_metrics"] = _metrics(
        0.60, 0.40, sample_count=10
    )
    mismatched_selection["reproduced_metrics"] = copy.deepcopy(
        mismatched_selection["selected_metrics"]
    )
    mismatched_diagnostics = copy.deepcopy(fixture["result"])
    mismatched_diagnostics["training_diagnostics"]["update_count"] = 611
    missing_runtime = copy.deepcopy(fixture["result"])
    missing_runtime.pop("runtime")

    for bad in (
            incomplete, mismatched_selection, mismatched_diagnostics,
            missing_runtime):
        with pytest.raises(ValueError, match="loop|history|selected|production"):
            finetune_runner.publish_rec_finetune_run(
                fixture["initialized"],
                bad,
                model_factory=fixture["model_factory"],
                calibration_fn=fixture["calibration_fn"],
            )
        assert not fixture["output_dir"].exists()
        assert list(tmp_path.glob(".published.staging-*")) == []
        _assert_immutable_evidence(fixture["immutable_evidence"])


def test_staged_publication_rejects_legacy_calibration_reproduction(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)

    def legacy_calibration(*_args, **_kwargs):
        return copy.deepcopy(fixture["result"]["selected_metrics"])

    with pytest.raises(ValueError, match="diagnostic"):
        finetune_runner.publish_rec_finetune_run(
            fixture["initialized"],
            fixture["result"],
            model_factory=fixture["model_factory"],
            calibration_fn=legacy_calibration,
        )
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []


def test_staged_publication_rejects_intrinsic_diagnostic_mismatch(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)

    def mismatched_calibration(*_args, **_kwargs):
        metrics = fixture["result"]["selected_metrics"]
        sample_count = metrics["sample_count"]
        selected = (
            [0.80] * metrics["hits050"]
            + [0.40] * (metrics["hits025"] - metrics["hits050"])
            + [0.15] * (sample_count - metrics["hits025"])
        )
        return _synthetic_calibration_observation(
            selected, selected, tuple(range(sample_count))
        )

    with pytest.raises(RuntimeError, match="output|diagnostic"):
        finetune_runner.publish_rec_finetune_run(
            fixture["initialized"],
            fixture["result"],
            model_factory=fixture["model_factory"],
            calibration_fn=mismatched_calibration,
        )
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []


def _smoke_result(fixture):
    calibration_count = fixture["initialized"]["train_data_contract"][
        "calibration_sample_count"
    ]
    baseline = _metrics(0.50, 0.40, sample_count=calibration_count)
    selected = _metrics(0.60, 0.50, sample_count=calibration_count)
    history = [
        {
            "step": 0,
            "metrics": baseline,
            "eligible": True,
            "regression": False,
            "action": "continue",
            "best_step": 0,
        },
        {
            "step": 1,
            "metrics": selected,
            "eligible": True,
            "regression": False,
            "action": "continue",
            "best_step": 1,
        },
    ]
    result = {
        "completed_updates": 1,
        "stopped_early": False,
        "selected_step": 1,
        "selected_metrics": selected,
        "reproduced_metrics": copy.deepcopy(selected),
        "calibration_history": history,
        "training_diagnostics": _synthetic_training_diagnostics(
            fixture["initialized"]["initial_state"]["mcln"],
            fixture["initialized"]["initial_state"]["groups"],
            1,
        ),
        "runtime": _synthetic_runtime_provenance(),
        "train_data_contract": copy.deepcopy(
            fixture["initialized"]["train_data_contract"]
        ),
    }
    result.update(_synthetic_diagnostic_run_fields(history, 1))
    return result


def _synthetic_calibration_fn_for_result(result):
    observation = _synthetic_observation_for_metrics(
        result["selected_metrics"]
    )

    def calibration_fn(*_args, **_kwargs):
        return copy.deepcopy(observation)

    return calibration_fn


@pytest.mark.parametrize("run_kind", ["production", "smoke"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing-history",
        "missing-selected",
        "missing-reproduced",
        "legacy",
    ],
)
def test_selected_run_validators_require_diagnostic_mode(
        tmp_path, monkeypatch, run_kind, mutation):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    if run_kind == "production":
        result = copy.deepcopy(fixture["result"])
        validate = finetune_runner._validate_production_run_result
    else:
        result = _smoke_result(fixture)
        validate = lambda value: finetune_runner._validate_smoke_run_result(
            value, 1
        )
    if mutation == "missing-history":
        result.pop("calibration_diagnostics_history")
    elif mutation == "missing-selected":
        result.pop("selected_calibration_diagnostics")
    elif mutation == "missing-reproduced":
        result.pop("reproduced_calibration_diagnostics")
    else:
        result["calibration_diagnostics_history"] = []
        result["selected_calibration_diagnostics"] = None
        result["reproduced_calibration_diagnostics"] = None

    with pytest.raises(ValueError, match="diagnostic"):
        validate(result)


@pytest.mark.parametrize("run_kind", ["production", "smoke"])
@pytest.mark.parametrize(
    "mutation",
    ["missing-selected", "missing-reproduced", "malformed", "mismatch"],
)
def test_selected_run_validators_require_exact_selected_output_digests(
        tmp_path, monkeypatch, run_kind, mutation):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    if run_kind == "production":
        result = copy.deepcopy(fixture["result"])
        validate = finetune_runner._validate_production_run_result
    else:
        result = _smoke_result(fixture)
        validate = lambda value: finetune_runner._validate_smoke_run_result(
            value, 1
        )
    if mutation == "missing-selected":
        result.pop("selected_calibration_output_sha256")
    elif mutation == "missing-reproduced":
        result.pop("reproduced_calibration_output_sha256")
    elif mutation == "malformed":
        result["selected_calibration_output_sha256"] = "not-a-sha256"
    else:
        result["reproduced_calibration_output_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="output|digest|reproduc"):
        validate(result)


def test_smoke_receipt_rejects_jointly_forged_selected_output_digests(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    result = _smoke_result(fixture)
    result["selected_calibration_output_sha256"] = "0" * 64
    result["reproduced_calibration_output_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="calibration output"):
        finetune_runner.publish_rec_finetune_smoke_receipt(
            fixture["initialized"], result,
            calibration_fn=_synthetic_calibration_fn_for_result(result),
        )
    assert not fixture["output_dir"].exists()


@pytest.mark.parametrize(
    "mutation",
    ["selected-step", "history-step", "previous-step", "current-step"],
)
def test_smoke_validator_rejects_boolean_diagnostic_steps(
        tmp_path, monkeypatch, mutation):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    result = _smoke_result(fixture)
    transition = result["calibration_diagnostics_history"][1][
        "transition_from_previous"
    ]
    if mutation == "selected-step":
        result["selected_step"] = True
    elif mutation == "history-step":
        result["calibration_diagnostics_history"][1]["step"] = True
    elif mutation == "previous-step":
        transition["previous_step"] = False
    else:
        transition["current_step"] = True

    with pytest.raises(ValueError, match="diagnostic|selected"):
        finetune_runner._validate_smoke_run_result(result, 1)


def _set_diagnostic_threshold_hits(section, suffix, hits, sample_count):
    section["hits" + suffix] = hits
    section["acc" + suffix] = hits / float(sample_count)


@pytest.mark.parametrize(
    "mutation",
    [
        "history-length",
        "diagnostic-fields",
        "diagnostic-schema",
        "sample-count",
        "threshold-accuracy",
        "threshold-nesting",
        "oracle-containment",
        "bin-fields",
        "bin-derived-hits",
        "effect-range",
        "effect-conservation",
        "regret-count-nesting",
        "cell-fields",
        "cell-count",
        "recoverable-range",
        "recoverable-consistency",
        "history-step",
        "selection-binding",
        "first-transition",
        "transition-fields",
        "transition-schema",
        "transition-steps",
        "transition-count",
        "transition-range",
        "joint-row-fields",
        "joint-bool-count",
        "joint-total-count",
        "selected-transition-conservation",
        "oracle-transition-conservation",
        "selected-summary",
        "reproduced-summary",
        "tensor",
        "per-sample-key",
        "oversized",
    ],
)
def test_production_validator_rejects_calibration_diagnostic_tampering(
        tmp_path, monkeypatch, mutation):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    result = copy.deepcopy(fixture["result"])
    history = result["calibration_diagnostics_history"]
    record = history[0]
    diagnostics = record["diagnostics"]
    sample_count = diagnostics["sample_count"]

    if mutation == "history-length":
        history.pop()
    elif mutation == "diagnostic-fields":
        diagnostics["extra"] = 0
    elif mutation == "diagnostic-schema":
        diagnostics["schema"] = "wrong"
    elif mutation == "sample-count":
        diagnostics["sample_count"] += 1
    elif mutation == "threshold-accuracy":
        diagnostics["stages"]["default_top1"]["acc025"] += 0.01
    elif mutation == "threshold-nesting":
        stage = diagnostics["stages"]["default_top1"]
        _set_diagnostic_threshold_hits(
            stage, "050", stage["hits025"] + 1, sample_count
        )
    elif mutation == "oracle-containment":
        stage = diagnostics["stages"]["default_top1"]
        oracle = diagnostics["candidate_oracle"]["raw_query"]
        _set_diagnostic_threshold_hits(
            oracle, "025", stage["hits025"] - 1, sample_count
        )
    elif mutation == "bin-fields":
        diagnostics["selected_iou"]["bins"]["extra"] = 0
    elif mutation == "bin-derived-hits":
        bins = diagnostics["selected_iou"]["bins"]
        bins["le_010"] -= 1
        bins["gt_025_le_030"] += 1
    elif mutation == "effect-range":
        diagnostics["effects"]["geometry_vs_parent"]["fixes025"] = (
            sample_count + 1
        )
    elif mutation == "effect-conservation":
        diagnostics["effects"]["geometry_vs_parent"]["fixes025"] = 1
    elif mutation == "regret-count-nesting":
        regret = diagnostics["geometry_oracle_selected_regret"]
        regret["positive_count"] = 0
        regret["ge005_count"] = 1
    elif mutation in {"cell-fields", "cell-count"}:
        cells = diagnostics["selected_oracle_regret_cells"]
        cell = next(
            cell
            for selected in cells.values()
            for oracle in selected.values()
            for cell in oracle.values()
            if cell["count"] > 0
        )
        if mutation == "cell-fields":
            cell["extra"] = 0
        else:
            cell["count"] += 1
    elif mutation == "recoverable-range":
        diagnostics["recoverable_misses"]["at025"] = sample_count + 1
    elif mutation == "recoverable-consistency":
        diagnostics["recoverable_misses"]["at025"] = 1
    elif mutation == "history-step":
        record["step"] = 1
    elif mutation == "selection-binding":
        stage = diagnostics["stages"]["geometry_top1"]
        _set_diagnostic_threshold_hits(
            stage, "025", stage["hits025"] + 1, sample_count
        )
    elif mutation == "first-transition":
        record["transition_from_previous"] = copy.deepcopy(
            history[1]["transition_from_previous"]
        )
    else:
        transition = history[1]["transition_from_previous"]
        if mutation == "transition-fields":
            transition["extra"] = 0
        elif mutation == "transition-schema":
            transition["schema"] = "wrong"
        elif mutation == "transition-steps":
            transition["previous_step"] = 1
        elif mutation == "transition-count":
            transition["sample_count"] += 1
        elif mutation == "transition-range":
            transition["selected"]["gained025"] = sample_count + 1
        elif mutation == "joint-row-fields":
            first_row = next(iter(transition["selected_oracle_joint"].values()))
            first_row["extra"] = 0
        elif mutation == "joint-bool-count":
            first_row = next(iter(transition["selected_oracle_joint"].values()))
            first_key = next(iter(first_row))
            first_row[first_key] = False
        elif mutation == "joint-total-count":
            joint = transition["selected_oracle_joint"]
            previous_name, current_name = next(
                (previous_name, current_name)
                for previous_name, row in joint.items()
                for current_name, count in row.items()
                if count > 0
            )
            joint[previous_name][current_name] += 1
        elif mutation == "selected-transition-conservation":
            transition["selected"]["gained025"] += 1
        elif mutation == "oracle-transition-conservation":
            transition["geometry_oracle"]["gained050"] += 1
        elif mutation == "selected-summary":
            result["selected_calibration_diagnostics"] = copy.deepcopy(
                result["selected_calibration_diagnostics"]
            )
            result["selected_calibration_diagnostics"]["schema"] = "wrong"
        elif mutation == "reproduced-summary":
            result["reproduced_calibration_diagnostics"] = copy.deepcopy(
                result["reproduced_calibration_diagnostics"]
            )
            result["reproduced_calibration_diagnostics"]["schema"] = "wrong"
        elif mutation == "tensor":
            cells = diagnostics["selected_oracle_regret_cells"]
            cell = next(
                cell
                for selected in cells.values()
                for oracle in selected.values()
                for cell in oracle.values()
                if cell["count"] > 0
            )
            cell["count"] = torch.tensor(cell["count"])
        elif mutation == "per-sample-key":
            diagnostics["selected_iou"]["selected_ious"] = []
        elif mutation == "oversized":
            diagnostics["schema"] = "x" * 300_000
        else:
            raise AssertionError("unknown diagnostic mutation")

    with pytest.raises(ValueError, match="diagnostic"):
        finetune_runner._validate_production_run_result(result)


def test_smoke_receipt_rejects_nonempty_live_runtime_forgery_at_publication(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    result = _smoke_result(fixture)
    result["runtime"]["command"][0] = "/bin/sh"
    result["runtime"]["interpreter"] = {
        "logical_path": "/bin/sh",
        "resolved_path": str(Path("/bin/sh").resolve()),
    }
    result["runtime"]["versions"] = {
        "python": "99.0.0", "torch": "99.0.0", "cuda": "99.0",
        "cudnn": 1,
    }
    result["runtime"]["device"] = {
        "type": "cuda", "index": 0, "name": "Bogus GPU",
        "total_memory_bytes": 1,
    }

    with pytest.raises(ValueError, match="runtime|live|identity|version|device"):
        finetune_runner.publish_rec_finetune_smoke_receipt(
            fixture["initialized"], result
        )
    assert not fixture["output_dir"].exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-fit-loader", "validation_loader", "validation_dataset",
        "val_data", "test_loader", "test_dataset", "test_data",
        "arbitrary_extra",
    ),
)
def test_smoke_receipt_requires_the_exact_live_train_data_key_set(
        tmp_path, monkeypatch, mutation):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    data = fixture["initialized"]["data"]
    if mutation == "missing-fit-loader":
        data.pop("fit_loader")
    else:
        data[mutation] = object()

    with pytest.raises(ValueError, match="train-data|data|key|exact"):
        finetune_runner.publish_rec_finetune_smoke_receipt(
            fixture["initialized"], _smoke_result(fixture)
        )
    assert not fixture["output_dir"].exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing", "nonfinite", "count", "frozen-gradient",
        "validation-object", "calibration-metrics-count",
    ),
)
def test_smoke_receipt_rejects_invalid_training_diagnostics(
        tmp_path, monkeypatch, mutation):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    result = _smoke_result(fixture)
    if mutation == "missing":
        result.pop("training_diagnostics")
    elif mutation == "nonfinite":
        result["training_diagnostics"]["losses"]["total"]["last"] = (
            float("nan")
        )
    elif mutation == "count":
        result["training_diagnostics"]["update_count"] = 2
    elif mutation == "frozen-gradient":
        result["training_diagnostics"]["last_update"]["frozen_mcln"][
            "gradient_tensor_count"
        ] = 1
    elif mutation == "validation-object":
        fixture["initialized"]["data"]["validation_loader"] = _NeverAccess()
    else:
        wrong_baseline = _metrics(0.50, 0.40, sample_count=10)
        wrong_selected = _metrics(0.60, 0.50, sample_count=10)
        result["calibration_history"][0]["metrics"] = wrong_baseline
        result["calibration_history"][1]["metrics"] = wrong_selected
        result["selected_metrics"] = wrong_selected
        result["reproduced_metrics"] = copy.deepcopy(wrong_selected)

    with pytest.raises(
            (ValueError, FloatingPointError),
            match=(
                "loop|diagnostic|finite|frozen|validation|train-data|sample"
            )):
        finetune_runner.publish_rec_finetune_smoke_receipt(
            fixture["initialized"], result
        )
    assert not fixture["output_dir"].exists()


@pytest.mark.parametrize("failure_stage", ["receipt", "finalize"])
def test_smoke_receipt_is_atomic_and_never_builds_deployable_artifacts(
        tmp_path, monkeypatch, failure_stage):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("smoke attempted to build a deployable artifact")
    )
    monkeypatch.setattr(
        rec_finetune, "build_rec_finetune_parent_artifact", forbidden
    )
    monkeypatch.setattr(
        rec_finetune, "build_rec_finetune_geometry_artifact", forbidden
    )
    events = []

    def inject(stage):
        events.append(stage)
        if stage == failure_stage:
            raise _InjectedPublicationFailure(stage)

    with pytest.raises(_InjectedPublicationFailure, match=failure_stage):
        finetune_runner.publish_rec_finetune_smoke_receipt(
            fixture["initialized"],
            _smoke_result(fixture),
            failure_injector=inject,
            calibration_fn=_synthetic_calibration_fn_for_result(
                _smoke_result(fixture)
            ),
        )
    assert events == (
        ["receipt"] if failure_stage == "receipt"
        else ["receipt", "finalize"]
    )
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []
    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_smoke_receipt_rejects_digest_field_removed_by_returning_hook(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    result = _smoke_result(fixture)
    events = []

    def inject(stage):
        events.append(stage)
        if stage == "receipt":
            staged = list(
                tmp_path.glob(".published.staging-*/smoke-receipt.json")
            )
            assert len(staged) == 1
            receipt = json.loads(staged[0].read_text("utf-8"))
            del receipt["verified_calibration_output_sha256"]
            staged[0].write_text(
                json.dumps(
                    receipt, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

    with pytest.raises(RuntimeError, match="staged receipt SHA-256 changed"):
        finetune_runner.publish_rec_finetune_smoke_receipt(
            fixture["initialized"],
            result,
            failure_injector=inject,
            calibration_fn=_synthetic_calibration_fn_for_result(result),
        )

    assert events == ["receipt"]
    assert not fixture["output_dir"].exists()
    assert list(tmp_path.glob(".published.staging-*")) == []
    _assert_immutable_evidence(fixture["immutable_evidence"])


def test_smoke_receipt_preserves_valid_oracle_only_reproduction_drift(
        tmp_path, monkeypatch):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    fixture["initialized"]["smoke_steps"] = 1
    result = _smoke_result(fixture)
    selected_observation = _synthetic_observation_for_metrics(
        result["selected_metrics"]
    )
    transition = selected_observation.diagnostics_result.transition_state
    reproduced_observation = _synthetic_calibration_observation(
        transition.selected_ious,
        tuple(max(value, 0.90) for value in transition.selected_ious),
        transition.expected_indices,
    )
    result["reproduced_calibration_diagnostics"] = copy.deepcopy(
        reproduced_observation.diagnostics_result.diagnostics
    )

    finetune_runner.publish_rec_finetune_smoke_receipt(
        fixture["initialized"], result,
        calibration_fn=_synthetic_calibration_fn_for_result(result),
    )

    receipt = json.loads(
        (fixture["output_dir"] / "smoke-receipt.json").read_text("utf-8")
    )
    assert receipt["selected_calibration_diagnostics"] != receipt[
        "reproduced_calibration_diagnostics"
    ]
    assert receipt["selected_calibration_output_sha256"] == receipt[
        "reproduced_calibration_output_sha256"
    ]


def test_smoke_receipt_is_read_only_nondeployable_and_main_uses_it(
        tmp_path, monkeypatch, capsys):
    fixture = _publication_fixture(tmp_path, monkeypatch)
    initialized = fixture["initialized"]
    initialized["smoke_steps"] = 1
    smoke_result = _smoke_result(fixture)
    builder_calls = []

    def forbidden(*_args, **_kwargs):
        builder_calls.append(True)
        raise AssertionError("smoke attempted deployable artifact publication")

    monkeypatch.setattr(
        rec_finetune, "build_rec_finetune_parent_artifact", forbidden
    )
    monkeypatch.setattr(
        rec_finetune, "build_rec_finetune_geometry_artifact", forbidden
    )
    publication = finetune_runner.publish_rec_finetune_smoke_receipt(
        initialized, smoke_result,
        calibration_fn=_synthetic_calibration_fn_for_result(smoke_result),
    )

    receipt_path = fixture["output_dir"] / "smoke-receipt.json"
    assert builder_calls == []
    assert sorted(path.name for path in fixture["output_dir"].iterdir()) == [
        "smoke-receipt.json"
    ]
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    assert publication["sha256"] == rec_finetune.sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text("utf-8"))
    assert set(receipt) == {
        "schema", "version", "deployable", "files", "smoke_steps",
        "completed_updates", "stopped_early", "selected_step",
        "selected_metrics", "calibration_history",
        "validation_data_accessed", "no_validation_data_declaration",
        "code_hashes", "losses", "training_diagnostics", "runtime",
        "train_data_contract", "calibration_diagnostics_history",
        "selected_calibration_diagnostics",
        "reproduced_calibration_diagnostics",
        "selected_calibration_output_sha256",
        "reproduced_calibration_output_sha256",
        "verified_calibration_metrics",
        "verified_calibration_diagnostics",
        "verified_calibration_output_sha256",
    }
    assert receipt["schema"] == "rec-finetune-smoke-receipt-v3"
    assert receipt["version"] == 3
    assert receipt["deployable"] is False
    assert receipt["smoke_steps"] == 1
    assert receipt["completed_updates"] == 1
    assert receipt["selected_step"] == 1
    assert receipt["validation_data_accessed"] is False
    assert receipt["no_validation_data_declaration"] == (
        "scanrefer-train-only-no-validation-v1"
    )
    assert receipt["losses"] == finetune_runner._loss_publication_contract()
    assert receipt["training_diagnostics"] == smoke_result[
        "training_diagnostics"
    ]
    assert receipt["training_diagnostics"]["update_count"] == 1
    assert receipt["runtime"] == smoke_result["runtime"]
    assert receipt["train_data_contract"] == smoke_result[
        "train_data_contract"
    ]
    assert receipt["calibration_diagnostics_history"] == smoke_result[
        "calibration_diagnostics_history"
    ]
    assert receipt["selected_calibration_diagnostics"] == smoke_result[
        "selected_calibration_diagnostics"
    ]
    assert receipt["reproduced_calibration_diagnostics"] == smoke_result[
        "reproduced_calibration_diagnostics"
    ]
    assert receipt["selected_calibration_output_sha256"] == smoke_result[
        "selected_calibration_output_sha256"
    ]
    assert receipt["reproduced_calibration_output_sha256"] == smoke_result[
        "reproduced_calibration_output_sha256"
    ]
    assert receipt["verified_calibration_metrics"] == smoke_result[
        "selected_metrics"
    ]
    assert receipt["verified_calibration_diagnostics"] == smoke_result[
        "selected_calibration_diagnostics"
    ]
    assert receipt["verified_calibration_output_sha256"] == smoke_result[
        "selected_calibration_output_sha256"
    ]
    assert set(receipt["files"]) == {
        "initial_backbone", "initial_parent", "initial_geometry",
    }
    for key, label in (
            ("initial_backbone", "backbone"),
            ("initial_parent", "parent"),
            ("initial_geometry", "geometry")):
        assert receipt["files"][key]["sha256"] == (
            fixture["initial_shas"][label]
        )
    _assert_immutable_evidence(fixture["immutable_evidence"])

    second_output = tmp_path / "main-smoke"
    initialized["paths"] = finetune_runner.RecFinetuneRuntimePaths(
        data_root=initialized["paths"].data_root,
        backbone_checkpoint=initialized["paths"].backbone_checkpoint,
        parent_reranker=initialized["paths"].parent_reranker,
        geometry_reranker=initialized["paths"].geometry_reranker,
        output_dir=second_output,
    )
    args = SimpleNamespace(smoke_steps=1)
    lifecycle = []
    main_result = copy.deepcopy(smoke_result)
    main_result.pop("runtime")
    monkeypatch.setattr(finetune_runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        finetune_runner, "_set_production_determinism",
        lambda: lifecycle.append("determinism"),
    )
    monkeypatch.setattr(
        finetune_runner, "_reset_cuda_peak_memory_stats",
        lambda: lifecycle.append("reset-peaks"),
    )
    utc_values = iter((
        "2026-07-16T11:00:00.000000Z",
        "2026-07-16T11:00:02.500000Z",
    ))
    monotonic_values = iter((100.0, 102.5))
    monkeypatch.setattr(
        finetune_runner, "_runtime_utc_now", lambda: next(utc_values)
    )
    monkeypatch.setattr(
        finetune_runner, "_runtime_monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        finetune_runner,
        "_runtime_cuda_snapshot",
        lambda: {
            "device": {
                "type": "cuda", "index": 0, "name": "Synthetic GPU",
                "total_memory_bytes": 24_000_000_000,
            },
            "peak_cuda_memory": {
                "allocated_bytes": 1234, "reserved_bytes": 5678,
            },
        },
    )

    def initialize(_args):
        lifecycle.append("initialize")
        return initialized

    def run(_initialized):
        lifecycle.append("run")
        return main_result

    monkeypatch.setattr(
        finetune_runner,
        "initialize_rec_finetune_run",
        initialize,
    )
    monkeypatch.setattr(
        finetune_runner, "run_rec_finetune", run
    )
    monkeypatch.setattr(
        finetune_runner,
        "publish_rec_finetune_run",
        forbidden,
    )
    monkeypatch.setattr(
        finetune_runner,
        "calibrate_rec_finetune",
        _synthetic_calibration_fn_for_result(smoke_result),
    )

    summary = finetune_runner.main([])

    assert lifecycle == ["determinism", "reset-peaks", "initialize", "run"]
    assert summary["output_dir"] == str(second_output)
    assert (second_output / "smoke-receipt.json").is_file()
    main_receipt = json.loads(
        (second_output / "smoke-receipt.json").read_text("utf-8")
    )
    assert main_receipt["runtime"]["started_utc"] == (
        "2026-07-16T11:00:00.000000Z"
    )
    assert main_receipt["runtime"]["finished_utc"] == (
        "2026-07-16T11:00:02.500000Z"
    )
    assert main_receipt["runtime"]["elapsed_seconds"] == 2.5
    assert main_receipt["runtime"]["command"] == [
        finetune_runner.sys.executable,
        str(Path(finetune_runner.__file__).resolve()),
    ]
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["selected_step"] == 1
    _assert_immutable_evidence(fixture["immutable_evidence"])
