import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys

import pytest
import torch

from models import rec_finetune
from scripts import probe_scanrefer_rec_source_gate as probe


def _cli_paths(tmp_path):
    data_root = tmp_path / "data"
    inputs_root = tmp_path / "inputs"
    output_parent = tmp_path / "outputs"
    data_root.mkdir()
    inputs_root.mkdir()
    output_parent.mkdir()
    paths = []
    for name in ("backbone.pth", "parent.pth", "geometry.pth"):
        path = inputs_root / name
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o444)
        paths.append(path)
    return data_root, paths, output_parent / "probe"


def _argv(data_root, inputs, output_dir, *extra):
    return [
        "--data-root", str(data_root),
        "--backbone-checkpoint", str(inputs[0]),
        "--parent-reranker", str(inputs[1]),
        "--geometry-reranker", str(inputs[2]),
        "--output-dir", str(output_dir),
    ] + list(extra)


def _exact_data_contract():
    metadata = copy.deepcopy(
        rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
    )
    return {
        "schema": "scanrefer-rec-source-gate-train-data-v1",
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
        "fit_sample_count": 33040,
        "calibration_sample_count": 3625,
        "fit_loader_batch_count": 1836,
        "calibration_loader_batch_count": 202,
        "batch_size": 18,
        "drop_last": False,
        "validation_data_accessed": False,
        "dataset_class": "tests.SyntheticTrainDataset",
        "dataset_instance_count": 1,
        "fit_and_calibration_share_source_annotations": True,
        "validation_data_objects_present": False,
        "loader_execution": {
            "fit": {"num_workers": 2, "pin_memory": False},
            "calibration": {"num_workers": 2, "pin_memory": False},
        },
    }


def test_parse_args_accepts_only_exact_probe_cli(tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)

    default = probe.parse_args(_argv(data_root, inputs, output_dir))
    one = probe.parse_args(
        _argv(data_root, inputs, output_dir, "--probe-steps", "1")
    )
    full = probe.parse_args(
        _argv(
            data_root, inputs, output_dir,
            "--device", "cuda:0", "--probe-steps", "306",
        )
    )

    assert default.device == "cuda:0"
    assert default.probe_steps == 306
    assert one.probe_steps == 1
    assert full.probe_steps == 306
    for invalid in ("0", "2", "305", "307", "-1", "1.0", "true"):
        with pytest.raises(SystemExit):
            probe.parse_args(
                _argv(
                    data_root, inputs, output_dir,
                    "--probe-steps", invalid,
                )
            )
    with pytest.raises(SystemExit):
        probe.parse_args(
            _argv(data_root, inputs, output_dir, "--device", "cpu")
        )
    with pytest.raises(SystemExit):
        probe.parse_args(
            _argv(data_root, inputs, output_dir, "--smoke-steps", "1")
        )
    abbreviated = _argv(data_root, inputs, output_dir)
    abbreviated[0] = "--data-r"
    with pytest.raises(SystemExit):
        probe.parse_args(abbreviated)
    with pytest.raises(SystemExit):
        probe.parse_args(
            _argv(data_root, inputs, output_dir)
            + ["--device", "cuda:0", "--device", "cuda:0"]
        )


def test_validate_runtime_paths_requires_readonly_distinct_inputs_and_new_output(
        tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    args = probe.parse_args(_argv(data_root, inputs, output_dir))

    paths = probe.validate_runtime_paths(args)

    assert paths.data_root == data_root.resolve()
    assert paths.output_dir == output_dir.resolve()
    output_parent_metadata = output_dir.parent.stat()
    assert paths.output_parent == output_dir.resolve().parent
    assert paths.output_parent_device == output_parent_metadata.st_dev
    assert paths.output_parent_inode == output_parent_metadata.st_ino
    assert not paths.output_dir.exists()
    assert all(path.is_file() for path in (
        paths.backbone_checkpoint,
        paths.parent_reranker,
        paths.geometry_reranker,
    ))

    inputs[0].chmod(0o644)
    with pytest.raises(ValueError, match="read-only"):
        probe.validate_runtime_paths(args)
    inputs[0].chmod(0o444)

    duplicate_args = probe.parse_args(_argv(
        data_root, (inputs[0], inputs[0], inputs[2]), output_dir
    ))
    with pytest.raises(ValueError, match="distinct"):
        probe.validate_runtime_paths(duplicate_args)

    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        probe.validate_runtime_paths(args)


@pytest.mark.parametrize(
    "collision",
    ["data-child", "input-parent-child", "input-file-child"],
)
def test_validate_runtime_paths_rejects_protected_tree_overlap(
        tmp_path, collision):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    if collision == "data-child":
        output_dir = data_root / "probe"
    elif collision == "input-parent-child":
        output_dir = inputs[0].parent / "probe"
    else:
        output_dir = Path(str(inputs[0]) + "/probe")
    args = probe.parse_args(_argv(data_root, inputs, output_dir))

    with pytest.raises(ValueError, match="overlap|collid"):
        probe.validate_runtime_paths(args)


@pytest.mark.parametrize("location", ["data", "input", "output-parent"])
def test_validate_runtime_paths_rejects_symlink_ancestors(
        tmp_path, location):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    data_root, inputs, output_dir = _cli_paths(real)
    if location == "data":
        data_root = linked / "data"
    elif location == "input":
        inputs[0] = linked / "inputs" / inputs[0].name
    else:
        output_dir = linked / "outputs" / "probe"
    args = probe.parse_args(_argv(data_root, inputs, output_dir))

    with pytest.raises(ValueError, match="symlink"):
        probe.validate_runtime_paths(args)


def test_validate_runtime_paths_rejects_symlink_leaf_and_missing_output_parent(
        tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    linked_input = inputs[0].parent / "linked.pth"
    linked_input.symlink_to(inputs[0])
    args = probe.parse_args(_argv(
        data_root, (linked_input, inputs[1], inputs[2]), output_dir
    ))
    with pytest.raises(ValueError, match="symlink"):
        probe.validate_runtime_paths(args)

    missing_parent = tmp_path / "missing" / "probe"
    args = probe.parse_args(_argv(data_root, inputs, missing_parent))
    with pytest.raises(ValueError, match="parent"):
        probe.validate_runtime_paths(args)


class _PoisonLegacyOptimizer:
    def __init__(self):
        self.state = {}
        self.param_groups = [{"params": []}]
        self.calls = []

    def _poison(self, name):
        self.calls.append(name)
        raise AssertionError("legacy optimizer method called: " + name)

    def zero_grad(self, *args, **kwargs):
        return self._poison("zero_grad")

    def step(self, *args, **kwargs):
        return self._poison("step")

    def state_dict(self, *args, **kwargs):
        return self._poison("state_dict")

    def load_state_dict(self, *args, **kwargs):
        return self._poison("load_state_dict")


def test_source_gate_manifest_walk_prunes_cache_directories(tmp_path):
    safe = tmp_path / "models" / "safe.py"
    pytest_cache = tmp_path / ".pytest_cache" / "cached.py"
    pycache = tmp_path / "models" / "__pycache__" / "cached.py"
    for path in (safe, pytest_cache, pycache):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="ascii")

    discovered = set(probe._iter_source_gate_python_files(tmp_path))

    assert discovered == {safe}


def test_source_gate_code_manifest_excludes_unrelated_test_and_cache_scripts():
    project_root = Path(probe.__file__).resolve().parents[1]

    manifest = probe.build_source_gate_code_manifest()
    paths = {Path(record["path"]) for record in manifest.values()}

    assert project_root / "scripts" / "cache_scanrefer_rec_candidates.py" in paths
    assert project_root / "scripts" / "rec_geometry_cache.py" in paths
    assert project_root / "pointnet2" / "pointnet2_test.py" not in paths
    assert (
        project_root / "scripts" / "cache_scanrefer_rec_mask_geometry.py"
        not in paths
    )


def test_initialize_captures_manifest_before_models_and_data_and_drops_legacy_optimizer(
        tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    args = probe.parse_args(
        _argv(data_root, inputs, output_dir, "--probe-steps", "1")
    )
    calls = []
    legacy_optimizer = _PoisonLegacyOptimizer()
    mcln = torch.nn.Linear(2, 2)
    parent = torch.nn.Linear(2, 1)
    geometry = torch.nn.Linear(2, 1)
    parameter_contract = object()
    source_optimizer = object()

    def manifest_builder():
        calls.append("manifest")
        return {
            "source_gate_runner": {
                "path": str(Path(probe.__file__).resolve()),
                "sha256": "a" * 64,
            },
            "rec_source_gate": {
                "path": str(Path("models/rec_source_gate.py").resolve()),
                "sha256": "b" * 64,
            },
        }

    def initial_state_loader(*loader_args, **loader_kwargs):
        calls.append("models")
        assert loader_kwargs["device"] == "cpu"
        return {
            "config": object(),
            "mcln": mcln,
            "parent": parent,
            "parent_artifact": {"kind": "parent"},
            "geometry": geometry,
            "geometry_artifact": {"kind": "geometry"},
            "groups": {"legacy": object()},
            "optimizer": legacy_optimizer,
            "checkpoint_path": inputs[0],
            "checkpoint_sha256": "c" * 64,
            "checkpoint_epoch": 71,
        }

    synthetic_data = {
        "dataset": object(),
        "split": {"metadata": copy.deepcopy(
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0
        )},
        "fit_view": object(),
        "calibration_view": argparse.Namespace(indices=tuple(range(3625))),
        "fit_loader": object(),
        "calibration_loader": object(),
    }

    def data_builder(config, device):
        calls.append("data")
        assert device == "cpu"
        return synthetic_data

    def contract_builder(initialized):
        calls.append("contract")
        assert initialized["data"] is synthetic_data
        return _exact_data_contract()

    def configure(mcln_arg, parent_arg, geometry_arg):
        calls.append("configure")
        assert (mcln_arg, parent_arg, geometry_arg) == (
            mcln, parent, geometry
        )
        return parameter_contract

    def optimizer_builder(contract):
        calls.append("source-optimizer")
        assert contract is parameter_contract
        return source_optimizer

    initialized = probe.initialize_source_gate_probe(
        args,
        device="cpu",
        manifest_builder=manifest_builder,
        initial_state_loader=initial_state_loader,
        data_builder=data_builder,
        data_contract_builder=contract_builder,
        trainability_configurer=configure,
        optimizer_builder=optimizer_builder,
    )

    assert calls == [
        "manifest", "models", "configure", "source-optimizer",
        "data", "contract",
    ]
    assert legacy_optimizer.calls == []
    assert "groups" not in initialized["initial_state"]
    assert "optimizer" not in initialized["initial_state"]
    assert initialized["source_parameters"] is parameter_contract
    assert initialized["source_optimizer"] is source_optimizer
    assert initialized["train_data_contract"] == _exact_data_contract()
    assert initialized["probe_steps"] == 1
    assert initialized["legacy_joint_optimizer_updates"] == 0
    assert initialized["publication_code_hashes"]["source_gate_runner"][
        "sha256"
    ] == "a" * 64


def test_source_gate_default_data_builder_preserves_workers_and_disables_pinning(
        monkeypatch):
    config = object()
    built = object()
    loader_calls = []
    worker_initializer = object()
    generator = object()

    def loader_factory(dataset, **kwargs):
        loader_calls.append((dataset, dict(kwargs)))
        return object()

    def legacy_builder(config_arg, device_arg, *, loader_factory):
        assert config_arg is config
        assert device_arg == "cuda:0"
        for name, shuffle in (("fit", True), ("calibration", False)):
            loader_factory(
                name,
                batch_size=18,
                shuffle=shuffle,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=worker_initializer,
                generator=generator,
            )
        return built

    monkeypatch.setattr(probe.legacy, "DataLoader", loader_factory)
    monkeypatch.setattr(
        probe.legacy, "build_train_only_data", legacy_builder
    )

    result = probe.build_source_gate_train_only_data(config, "cuda:0")

    assert result is built
    assert [dataset for dataset, _kwargs in loader_calls] == [
        "fit", "calibration",
    ]
    for _dataset, kwargs in loader_calls:
        assert kwargs["num_workers"] == 2
        assert kwargs["pin_memory"] is False
        assert kwargs["worker_init_fn"] is worker_initializer
        assert kwargs["generator"] is generator


def test_source_gate_data_contract_is_derived_from_both_live_loaders(
        monkeypatch):
    base_contract = _exact_data_contract()
    del base_contract["loader_execution"]
    base_contract["schema"] = "scanrefer-rec-finetune-train-data-v1"
    fit_loader = argparse.Namespace(num_workers=2, pin_memory=False)
    calibration_loader = argparse.Namespace(
        num_workers=2, pin_memory=False
    )
    initialized = {
        "data": {
            "fit_loader": fit_loader,
            "calibration_loader": calibration_loader,
        },
    }
    monkeypatch.setattr(
        probe.legacy,
        "build_rec_finetune_train_data_contract",
        lambda value: copy.deepcopy(base_contract),
    )

    contract = probe.build_source_gate_train_data_contract(initialized)

    assert contract == _exact_data_contract()
    fit_loader.pin_memory = True
    with pytest.raises(ValueError, match="loader|pin"):
        probe.build_source_gate_train_data_contract(initialized)


@pytest.mark.parametrize(
    "loader_name,field,value",
    [
        ("fit", "num_workers", 0),
        ("calibration", "num_workers", 1),
        ("fit", "pin_memory", True),
        ("calibration", "pin_memory", True),
    ],
)
def test_source_gate_data_contract_rejects_loader_execution_drift(
        loader_name, field, value):
    contract = _exact_data_contract()
    contract["loader_execution"][loader_name][field] = value

    with pytest.raises(ValueError, match="loader|contract"):
        probe._validate_train_data_contract(contract)


def test_initialize_rejects_nonempty_legacy_optimizer_before_source_configuration(
        tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    args = probe.parse_args(_argv(data_root, inputs, output_dir))
    poison = _PoisonLegacyOptimizer()
    poison.state[object()] = {"step": 1}
    configured = []

    def loader(*args, **kwargs):
        return {
            "config": object(),
            "mcln": torch.nn.Linear(1, 1),
            "parent": torch.nn.Linear(1, 1),
            "parent_artifact": {},
            "geometry": torch.nn.Linear(1, 1),
            "geometry_artifact": {},
            "groups": {},
            "optimizer": poison,
        }

    with pytest.raises(ValueError, match="fresh|state"):
        probe.initialize_source_gate_probe(
            args,
            device="cpu",
            manifest_builder=lambda: {
                "source_gate_runner": {
                    "path": str(Path(probe.__file__).resolve()),
                    "sha256": "a" * 64,
                }
            },
            initial_state_loader=loader,
            trainability_configurer=lambda *models: configured.append(models),
        )
    assert configured == []
    assert poison.calls == []


def test_legacy_optimizer_with_no_parameter_groups_is_not_fresh():
    optimizer = _PoisonLegacyOptimizer()
    optimizer.param_groups = []
    state = {"groups": {}, "optimizer": optimizer}

    with pytest.raises(ValueError, match="fresh|group"):
        probe._require_fresh_legacy_optimizer(state)

    assert optimizer.calls == []
    assert set(state) == {"groups", "optimizer"}


def test_initialize_rejects_validation_cache_or_inexact_train_contract(
        tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    args = probe.parse_args(_argv(data_root, inputs, output_dir))
    contract = _exact_data_contract()
    contract["validation_data_accessed"] = True

    state = {
        "config": object(),
        "mcln": torch.nn.Linear(1, 1),
        "parent": torch.nn.Linear(1, 1),
        "parent_artifact": {},
        "geometry": torch.nn.Linear(1, 1),
        "geometry_artifact": {},
        "groups": {},
        "optimizer": _PoisonLegacyOptimizer(),
    }
    data = {
        "dataset": object(), "split": {}, "fit_view": object(),
        "calibration_view": object(), "fit_loader": object(),
        "calibration_loader": object(), "validation_cache": object(),
    }

    with pytest.raises(ValueError, match="validation|cache|train-only"):
        probe.initialize_source_gate_probe(
            args,
            device="cpu",
            manifest_builder=lambda: {"runner": {
                "path": str(Path(probe.__file__).resolve()),
                "sha256": "a" * 64,
            }},
            initial_state_loader=lambda *a, **k: state,
            data_builder=lambda *a, **k: data,
            data_contract_builder=lambda initialized: contract,
            trainability_configurer=lambda *models: object(),
            optimizer_builder=lambda parameters: object(),
        )


def test_publication_manifest_contains_runner_gate_and_runtime_dependencies():
    manifest = probe.build_source_gate_code_manifest()

    assert set((
        "source_gate_runner", "rec_source_gate", "rec_finetune",
        "rec_candidate_adapter", "rec_mask_geometry",
        "rec_geometry_reranker", "train_rec_finetune",
    )).issubset(manifest)
    for record in manifest.values():
        assert set(record) == {"path", "sha256"}
        assert Path(record["path"]).is_file()
        assert len(record["sha256"]) == 64
        int(record["sha256"], 16)


def _frame(digest, label, payload):
    label = label.encode("ascii")
    digest.update(struct.pack("<Q", len(label)))
    digest.update(label)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def test_canonical_state_digest_binds_key_dtype_shape_and_raw_bits():
    import hashlib

    state = {
        "z": torch.tensor([1.0, -0.0], dtype=torch.float32),
        "a": torch.tensor([[1, 2]], dtype=torch.int16),
    }
    expected = hashlib.sha256()
    _frame(expected, "schema", b"rec-source-gate-state-digest-v1")
    for name in ("a", "z"):
        value = state[name].detach().cpu().contiguous()
        _frame(expected, "tensor", b"")
        _frame(expected, "key", name.encode("utf-8"))
        _frame(expected, "dtype", str(value.dtype).encode("ascii"))
        _frame(
            expected,
            "shape",
            json.dumps(list(value.shape), separators=(",", ":")).encode(
                "ascii"
            ),
        )
        _frame(
            expected,
            "bytes",
            value.numpy().tobytes(),
        )

    assert probe.canonical_state_digest(state) == expected.hexdigest()
    reordered = {"a": state["a"], "z": state["z"]}
    assert probe.canonical_state_digest(reordered) == expected.hexdigest()
    changed_bits = copy.deepcopy(state)
    changed_bits["z"][1] = 0.0
    assert probe.canonical_state_digest(changed_bits) != expected.hexdigest()
    changed_dtype = copy.deepcopy(state)
    changed_dtype["a"] = changed_dtype["a"].to(torch.int32)
    assert probe.canonical_state_digest(changed_dtype) != expected.hexdigest()
    changed_key = {"b": state["a"], "z": state["z"]}
    assert probe.canonical_state_digest(changed_key) != expected.hexdigest()


class _CaptureAccumulator:
    instances = []

    def __init__(self, expected_indices, baseline=None):
        self.expected_indices = tuple(expected_indices)
        self.baseline = baseline
        self.updates = []
        self.__class__.instances.append(self)

    def update(self, indices, observation):
        self.updates.append((tuple(indices.tolist()), observation))

    def finalize(self, expected_sample_count):
        assert expected_sample_count == len(self.expected_indices)
        return {"captured": expected_sample_count}


def test_calibrate_normalizes_string_device_before_default_batch_mover(
        monkeypatch):
    model = torch.nn.Linear(1, 1)
    initialized = {
        "device": "cpu",
        "initial_state": {
            "mcln": model,
            "parent": copy.deepcopy(model),
            "parent_artifact": {},
            "geometry": copy.deepcopy(model),
            "geometry_artifact": {},
        },
        "data": {
            "calibration_loader": [{
                "dataset_index": torch.tensor([1]),
            }],
            "calibration_view": argparse.Namespace(indices=(1,)),
        },
        "train_data_contract": {"calibration_sample_count": 1},
    }
    real_move_batch = probe.legacy._move_batch_to_device
    observed_devices = []

    def capture_default_move(batch, device):
        observed_devices.append(device)
        real_move_batch(batch, device)
        raise RuntimeError("default calibration mover reached")

    monkeypatch.setattr(
        probe.legacy, "_move_batch_to_device", capture_default_move
    )

    with pytest.raises(RuntimeError, match="default calibration mover reached"):
        probe.calibrate_source_gate_probe(initialized)

    assert observed_devices == [torch.device("cpu")]


def test_calibrate_builds_exact_observation_and_uses_compact_parent_top1_axis(
        monkeypatch):
    _CaptureAccumulator.instances = []
    mcln = torch.nn.Linear(1, 1)
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    default_scores = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    contrastive_scores = default_scores.flip(1).clone()
    compact_indices = torch.arange(16, dtype=torch.long).reshape(1, 16)
    compact_valid = torch.ones(1, 16, dtype=torch.bool)
    parent_top1_mask = torch.zeros(1, 16, dtype=torch.bool)
    parent_top1_mask[0, 3] = True
    geometry_valid = torch.ones(1, 7, dtype=torch.bool)
    full = {
        "default_scores": default_scores,
        "contrastive_scores": contrastive_scores,
        "boxes": torch.zeros(1, 16, 6),
        "num_queries": 16,
    }
    compact = {
        "query_indices": compact_indices,
        "valid_mask": compact_valid,
    }
    forward = {
        "parent_model_inputs": {"valid_mask": compact_valid.clone()},
        "parent_state": {
            "query_indices": compact_indices.clone(),
            "candidate_valid": compact_valid.clone(),
            "parent_top1_mask": parent_top1_mask,
            # This is a full-query index and must never be used as position 15.
            "top1_query_index": torch.tensor([15], dtype=torch.long),
        },
        "parent_candidate_ious": torch.linspace(0, 1, 16).reshape(1, 16),
        "geometry_model_inputs": {"valid_mask": geometry_valid},
        "geometry_candidate_ious": torch.linspace(0, 1, 7).reshape(1, 7),
    }
    calls = []
    autocast_calls = []

    class AutocastContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def autocast(enabled=True):
        autocast_calls.append(enabled)
        return AutocastContext()

    monkeypatch.setattr(torch.cuda.amp, "autocast", autocast)

    def input_builder(batch):
        calls.append("inputs")
        return {"feature": batch["feature"]}

    def model_forward(inputs):
        calls.append("mcln")
        assert inputs["train"] is False
        return {"deployable": torch.tensor(1.0)}

    mcln.forward = model_forward

    def forward_fn(
            end_points, inputs, moved_batch, parent_arg, parent_artifact,
            geometry_arg, geometry_artifact):
        calls.append("frozen-forward")
        assert "center_label" not in inputs
        assert (parent_arg, geometry_arg) == (parent, geometry)
        return forward

    initialized = {
        "device": "cpu",
        "initial_state": {
            "mcln": mcln,
            "parent": parent,
            "parent_artifact": {},
            "geometry": geometry,
            "geometry_artifact": {},
        },
        "data": {
            "calibration_loader": [{
                "dataset_index": torch.tensor([41], dtype=torch.long),
                "feature": torch.ones(1, 1),
                "center_label": torch.zeros(1, 1, 3),
            }],
            "calibration_view": argparse.Namespace(indices=(41,)),
        },
        "train_data_contract": {"calibration_sample_count": 1},
    }

    calibrated = probe.calibrate_source_gate_probe(
        initialized,
        move_batch=lambda batch, device: batch,
        input_builder=input_builder,
        full_state_builder=lambda end_points, inputs: full,
        compact_state_builder=lambda state: compact,
        target_attacher=lambda state, batch, root_only: torch.full(
            (1, 16), 0.75, dtype=torch.float32
        ),
        forward_fn=forward_fn,
        selected_iou_builder=lambda state: torch.tensor([0.6]),
        eval_mode_setter=lambda *models: calls.append("eval"),
        accumulator_factory=_CaptureAccumulator,
    )

    assert calls == ["eval", "inputs", "mcln", "frozen-forward"]
    assert autocast_calls == [False]
    assert calibrated["report"] == {"captured": 1}
    accumulator = calibrated["accumulator"]
    assert accumulator.expected_indices == (41,)
    observation = accumulator.updates[0][1]
    assert set(observation) == {
        "full_query_ious", "default_scores", "contrastive_scores",
        "compact_query_indices", "compact_valid_mask",
        "parent_candidate_ious", "parent_valid_mask",
        "parent_top1_positions", "geometry_candidate_ious",
        "geometry_valid_mask", "geometry_selected_ious",
    }
    assert observation["parent_top1_positions"].tolist() == [3]
    assert torch.equal(
        observation["compact_query_indices"], compact_indices
    )
    assert torch.equal(observation["compact_valid_mask"], compact_valid)


def test_calibrate_rejects_independent_compact_mismatch():
    mcln = torch.nn.Linear(1, 1)
    mcln.forward = lambda inputs: {}
    model = torch.nn.Linear(1, 1)
    valid = torch.ones(1, 16, dtype=torch.bool)
    indices = torch.arange(16).reshape(1, 16)
    initialized = {
        "device": "cpu",
        "initial_state": {
            "mcln": mcln, "parent": model, "parent_artifact": {},
            "geometry": copy.deepcopy(model), "geometry_artifact": {},
        },
        "data": {
            "calibration_loader": [{"dataset_index": torch.tensor([1])}],
            "calibration_view": argparse.Namespace(indices=(1,)),
        },
        "train_data_contract": {"calibration_sample_count": 1},
    }
    forward = {
        "parent_model_inputs": {"valid_mask": valid},
        "parent_state": {
            "query_indices": indices + 1,
            "candidate_valid": valid,
            "parent_top1_mask": torch.nn.functional.one_hot(
                torch.tensor([0]), 16
            ).bool(),
        },
    }
    with pytest.raises(ValueError, match="compact"):
        probe.calibrate_source_gate_probe(
            initialized,
            move_batch=lambda batch, device: batch,
            input_builder=lambda batch: {},
            full_state_builder=lambda end_points, inputs: {
                "default_scores": torch.zeros(1, 16),
                "contrastive_scores": torch.zeros(1, 16),
                "boxes": torch.zeros(1, 16, 6),
            },
            compact_state_builder=lambda state: {
                "query_indices": indices, "valid_mask": valid,
            },
            target_attacher=lambda state, batch, root_only: torch.zeros(
                1, 16
            ),
            forward_fn=lambda *args: forward,
            eval_mode_setter=lambda *models: None,
            accumulator_factory=_CaptureAccumulator,
        )


_REPORT_GROUPS = {
    "membership": ("default_top8", "contrastive_top8", "union_top16"),
    "candidate_oracle": (
        "raw_query", "union_query", "parent_candidate",
        "geometry_candidate",
    ),
    "top1": ("default", "parent", "geometry"),
}


def _report_metrics(sample_count=10):
    result = {}
    for group, branches in _REPORT_GROUPS.items():
        result[group] = {}
        for branch in branches:
            result[group][branch] = {
                "hits025": 5,
                "hits050": 4,
                "acc025": 0.5,
                "acc050": 0.4,
            }
    result["candidate_oracle"]["raw_query"].update({
        "hits025": 10, "hits050": 9, "acc025": 1.0, "acc050": 0.9,
    })
    result["candidate_oracle"]["parent_candidate"].update({
        "hits025": 8, "hits050": 7, "acc025": 0.8, "acc050": 0.7,
    })
    result["candidate_oracle"]["geometry_candidate"].update({
        "hits025": 9, "hits050": 8, "acc025": 0.9, "acc050": 0.8,
    })
    return result


def _calibration_report(metrics=None, baseline=None, raw_digest="1" * 64):
    metrics = copy.deepcopy(metrics if metrics is not None else _report_metrics())
    for branches in metrics.values():
        for metric in branches.values():
            for suffix in ("025", "050"):
                metric["acc" + suffix] = metric["hits" + suffix] / 10.0
    if baseline is None:
        transitions = {
            group: {
                branch: {
                    "gained025": 0, "lost025": 0,
                    "gained050": 0, "lost050": 0,
                }
                for branch in branches
            }
            for group, branches in _REPORT_GROUPS.items()
        }
    else:
        transitions = {}
        for group, branches in _REPORT_GROUPS.items():
            transitions[group] = {}
            for branch in branches:
                current = metrics[group][branch]
                previous = baseline["metrics"][group][branch]
                record = {}
                for suffix in ("025", "050"):
                    difference = (
                        current["hits" + suffix]
                        - previous["hits" + suffix]
                    )
                    record["gained" + suffix] = max(difference, 0)
                    record["lost" + suffix] = max(-difference, 0)
                transitions[group][branch] = record
    return {
        "schema": "rec-source-gate-calibration-v1",
        "sample_count": 10,
        "baseline_present": baseline is not None,
        "metrics": metrics,
        "transitions": transitions,
        "digests": {
            "canonical_format": (
                "rec-source-gate-calibration-float32-sha256-v1"
            ),
            "raw_query_ious_sha256": raw_digest,
            "geometry_selected_ious_sha256": "2" * 64,
        },
    }


def _eligible_reports():
    step0 = _calibration_report()
    final_metrics = copy.deepcopy(step0["metrics"])
    final_metrics["membership"]["default_top8"].update({
        "hits025": 6, "acc025": 0.6,
    })
    final_metrics["candidate_oracle"]["parent_candidate"].update({
        "hits025": 9, "acc025": 0.9,
    })
    final_metrics["top1"]["geometry"].update({
        "hits025": 6, "acc025": 0.6,
    })
    final = _calibration_report(final_metrics, baseline=step0)
    return step0, final


_PUBLIC_TO_INTERNAL_BRANCH = {
    "membership": {
        "default_top8": "default_top8",
        "contrastive_top8": "contrastive_top8",
        "union_top16": "union_query",
    },
    "candidate_oracle": {
        "raw_query": "raw_query",
        "union_query": "union_query",
        "parent_candidate": "parent_candidate",
        "geometry_candidate": "geometry_candidate",
    },
    "top1": {
        "default": "default_top1",
        "parent": "parent_top1",
        "geometry": "geometry_top1",
    },
}


def _accumulator_for_report(report):
    accumulator = object.__new__(
        probe.rec_source_gate.RecSourceGateCalibrationAccumulator
    )
    sample_count = report["sample_count"]
    accumulator._expected_indices = tuple(range(sample_count))
    accumulator._hit_bits = {}
    for group, branches in _PUBLIC_TO_INTERNAL_BRANCH.items():
        for public_name, internal_name in branches.items():
            metric = report["metrics"][group][public_name]
            bits = {
                suffix: tuple(
                    index < metric["hits" + suffix]
                    for index in range(sample_count)
                )
                for suffix in ("025", "050")
            }
            if internal_name in accumulator._hit_bits:
                assert accumulator._hit_bits[internal_name] == bits
            else:
                accumulator._hit_bits[internal_name] = bits
    accumulator._finalized = True
    accumulator._report = copy.deepcopy(report)
    return accumulator


def test_calibration_hit_bit_digest_binds_private_order_without_arrays():
    step0, _final = _eligible_reports()
    accumulator = _accumulator_for_report(step0)

    digest = probe._source_gate_calibration_hit_bits_sha256(
        accumulator, step0
    )

    reordered = _accumulator_for_report(step0)
    for suffix in ("025", "050"):
        bits = list(reordered._hit_bits["default_top8"][suffix])
        bits[0], bits[5] = bits[5], bits[0]
        reordered._hit_bits["default_top8"][suffix] = tuple(bits)
    reordered_digest = probe._source_gate_calibration_hit_bits_sha256(
        reordered, step0
    )
    assert probe._is_sha256(digest)
    assert probe._is_sha256(reordered_digest)
    assert reordered_digest != digest
    assert not isinstance(digest, (dict, list, tuple))

    accumulator._finalized = False
    with pytest.raises(ValueError, match="final|accumulator"):
        probe._source_gate_calibration_hit_bits_sha256(accumulator, step0)


def _state_digests(mcln_full="a", mcln_frozen="b", parent="c", geometry="d"):
    return {
        "mcln_full": mcln_full * 64,
        "mcln_frozen": mcln_frozen * 64,
        "parent": parent * 64,
        "geometry": geometry * 64,
    }


def test_evaluate_gate_recomputes_all_checks_from_hits_and_digests():
    step0, final = _eligible_reports()
    baseline_state = _state_digests()
    final_state = _state_digests(mcln_full="e")

    decision = probe.evaluate_source_gate_gate(
        step0,
        final,
        baseline_state_digests=baseline_state,
        final_state_digests=final_state,
        training_diagnostics={"informative_rows_total": 7},
        final_step=306,
    )

    assert decision["eligible"] is True
    assert decision["selected_step"] == 306
    assert decision["reasons"] == []
    assert decision["checks"]["strict_improvement_present"] is True
    assert all(decision["checks"].values())


@pytest.mark.parametrize(
    "failure",
    [
        "geometry025", "geometry050", "parent025", "parent050",
        "geometry-candidate025", "geometry-candidate050", "raw025",
        "raw050", "raw-digest", "no-improvement", "mcln-frozen",
        "parent-state", "geometry-state", "zero-informative",
    ],
)
def test_evaluate_gate_fails_each_required_check(failure):
    step0, final = _eligible_reports()
    baseline_state = _state_digests()
    final_state = _state_digests(mcln_full="e")
    diagnostics = {"informative_rows_total": 7}
    if failure == "geometry025":
        final["metrics"]["top1"]["geometry"]["hits025"] = 4
    elif failure == "geometry050":
        final["metrics"]["top1"]["geometry"]["hits050"] = 3
    elif failure == "parent025":
        final["metrics"]["candidate_oracle"]["parent_candidate"]["hits025"] = 7
    elif failure == "parent050":
        final["metrics"]["candidate_oracle"]["parent_candidate"]["hits050"] = 6
    elif failure == "geometry-candidate025":
        final["metrics"]["candidate_oracle"]["geometry_candidate"]["hits025"] = 8
    elif failure == "geometry-candidate050":
        final["metrics"]["candidate_oracle"]["geometry_candidate"]["hits050"] = 7
    elif failure == "raw025":
        final["metrics"]["candidate_oracle"]["raw_query"]["hits025"] = 9
    elif failure == "raw050":
        final["metrics"]["candidate_oracle"]["raw_query"]["hits050"] = 8
    elif failure == "raw-digest":
        final["digests"]["raw_query_ious_sha256"] = "f" * 64
    elif failure == "no-improvement":
        for group, branch in (
                ("membership", "default_top8"),
                ("candidate_oracle", "parent_candidate"),
                ("top1", "geometry")):
            final["metrics"][group][branch]["hits025"] = (
                step0["metrics"][group][branch]["hits025"]
            )
    elif failure == "mcln-frozen":
        final_state["mcln_frozen"] = "f" * 64
    elif failure == "parent-state":
        final_state["parent"] = "f" * 64
    elif failure == "geometry-state":
        final_state["geometry"] = "f" * 64
    else:
        diagnostics["informative_rows_total"] = 0
    # Keep the report structurally coherent after deliberately changing hits.
    if failure not in {
            "raw-digest", "no-improvement", "mcln-frozen", "parent-state",
            "geometry-state", "zero-informative"}:
        metric = final["metrics"]
        for group, branches in metric.items():
            for branch, record in branches.items():
                for suffix in ("025", "050"):
                    record["acc" + suffix] = (
                        record["hits" + suffix] / 10.0
                    )
        final = _calibration_report(metric, baseline=step0,
                                    raw_digest=final["digests"][
                                        "raw_query_ious_sha256"
                                    ])
    elif failure == "no-improvement":
        final = _calibration_report(final["metrics"], baseline=step0)

    decision = probe.evaluate_source_gate_gate(
        step0,
        final,
        baseline_state_digests=baseline_state,
        final_state_digests=final_state,
        training_diagnostics=diagnostics,
        final_step=306,
    )

    assert decision["eligible"] is False
    assert decision["selected_step"] == 0
    assert decision["reasons"]


@pytest.mark.parametrize("tamper", ["acc", "transition", "union"])
def test_evaluate_gate_rejects_invalid_report_instead_of_trusting_it(tamper):
    step0, final = _eligible_reports()
    if tamper == "acc":
        final["metrics"]["top1"]["geometry"]["acc025"] = 0.123
    elif tamper == "transition":
        final["transitions"]["top1"]["geometry"]["gained025"] += 1
    else:
        final["metrics"]["membership"]["union_top16"]["hits025"] += 1

    with pytest.raises(ValueError):
        probe.evaluate_source_gate_gate(
            step0,
            final,
            baseline_state_digests=_state_digests(),
            final_state_digests=_state_digests(mcln_full="e"),
            training_diagnostics={"informative_rows_total": 1},
            final_step=306,
        )


class _AllowedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))


class _ProbeMCLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        heads = [torch.nn.Identity() for _ in range(5)]
        final = torch.nn.Module()
        final.sem_cls_scores_head = _AllowedHead()
        heads.append(final)
        self.prediction_heads = torch.nn.ModuleList(heads)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        score = self.prediction_heads[5].sem_cls_scores_head.weight
        return {"score": score}


class _FitPoisonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, *args, **kwargs):
        raise AssertionError("frozen reranker called during fit")


class _TrackingOptimizer:
    def __init__(self, parameter):
        self.inner = torch.optim.SGD([parameter], lr=0.1)
        self.state = self.inner.state
        self.param_groups = self.inner.param_groups
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return self.inner.zero_grad(*args, **kwargs)

    def step(self):
        self.step_calls += 1
        return self.inner.step()


def _run_initialized(tmp_path, probe_steps=1):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    mcln = _ProbeMCLN()
    parent = _FitPoisonModel()
    geometry = _FitPoisonModel()
    parameter = mcln.prediction_heads[5].sem_cls_scores_head.weight
    optimizer = _TrackingOptimizer(parameter)
    return {
        "paths": argparse.Namespace(output_dir=output_dir),
        "device": "cpu",
        "probe_steps": probe_steps,
        "seed": 0,
        "legacy_joint_optimizer_updates": 0,
        "initial_state": {
            "mcln": mcln, "parent": parent, "parent_artifact": {},
            "geometry": geometry, "geometry_artifact": {},
        },
        "source_parameters": argparse.Namespace(
            names=("prediction_heads.5.sem_cls_scores_head.weight",),
            parameters=(parameter,),
        ),
        "source_optimizer": optimizer,
        "data": {
            "fit_loader": [{
                "target": torch.tensor([[1.0] + [0.0] * 8]),
            }] * probe_steps,
            "calibration_loader": [],
            "calibration_view": argparse.Namespace(indices=tuple(range(10))),
        },
        "train_data_contract": _exact_data_contract(),
        "publication_code_hashes": {"runner": {
            "path": str(Path(probe.__file__).resolve()),
            "sha256": "a" * 64,
        }},
        "publication_code_manifest_sha256": "b" * 64,
    }


def _calibration_sequence(eligible=True, fail_call=None):
    step0, final = _eligible_reports()
    if not eligible:
        final_metrics = copy.deepcopy(step0["metrics"])
        final = _calibration_report(final_metrics, baseline=step0)
    calls = []
    baseline_accumulator = _accumulator_for_report(step0)
    final_accumulator = _accumulator_for_report(final)

    def calibrate(initialized, *, baseline=None, **kwargs):
        call_number = len(calls)
        calls.append(baseline)
        if fail_call == call_number:
            raise RuntimeError("calibration failed")
        if call_number == 0:
            assert baseline is None
            return {"accumulator": baseline_accumulator, "report": step0}
        if call_number == 1:
            assert baseline is baseline_accumulator
            return {"accumulator": final_accumulator, "report": final}
        if eligible:
            assert baseline is baseline_accumulator
            return {
                "accumulator": _accumulator_for_report(final),
                "report": copy.deepcopy(final),
            }
        assert baseline is None
        return {
            "accumulator": _accumulator_for_report(step0),
            "report": copy.deepcopy(step0),
        }

    return calibrate, calls


def test_run_probe_normalizes_string_device_before_default_batch_mover(
        tmp_path, monkeypatch):
    initialized = _run_initialized(tmp_path)
    calibrate, _calls = _calibration_sequence(eligible=True)
    real_move_batch = probe.legacy._move_batch_to_device
    observed_devices = []

    def capture_default_move(batch, device):
        observed_devices.append(device)
        real_move_batch(batch, device)
        raise RuntimeError("default fit mover reached")

    monkeypatch.setattr(
        probe.legacy, "_move_batch_to_device", capture_default_move
    )

    with pytest.raises(RuntimeError, match="default fit mover reached"):
        probe.run_source_gate_probe(
            initialized,
            calibration_fn=calibrate,
            live_contract_validator=lambda value: None,
        )

    assert observed_devices == [torch.device("cpu")]


def test_run_probe_updates_source_only_then_restores_and_reproduces_final(
        tmp_path, monkeypatch):
    initialized = _run_initialized(tmp_path)
    mcln = initialized["initial_state"]["mcln"]
    parameter = initialized["source_parameters"].parameters[0]
    before = parameter.detach().clone()
    calibrate, calibration_calls = _calibration_sequence(eligible=True)
    autocast_calls = []
    contract_checks = []

    class AutocastContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def autocast(enabled=True):
        autocast_calls.append(enabled)
        return AutocastContext()

    monkeypatch.setattr(torch.cuda.amp, "autocast", autocast)

    result = probe.run_source_gate_probe(
        initialized,
        calibration_fn=calibrate,
        move_batch=lambda batch, device: batch,
        input_builder=lambda batch: {},
        full_state_builder=lambda end_points, inputs: {
            "default_scores": torch.cat((
                end_points["score"].reshape(1, 1), torch.zeros(1, 8)
            ), dim=1),
            "boxes": torch.zeros(1, 9, 6),
        },
        target_attacher=lambda full, batch, root_only: batch["target"],
        train_mode_setter=lambda *models: None,
        eval_mode_setter=lambda *models: [model.eval() for model in models],
        gradient_clipper=lambda contract: {
            "source_gate_semantic_classifier": float(
                contract.parameters[0].grad.detach().abs().item()
            )
        },
        live_contract_validator=lambda value: contract_checks.append(value),
    )

    assert result["completed_steps"] == 1
    assert result["decision"]["eligible"] is True
    assert result["decision"]["selected_step"] == 1
    assert result["reproduction_matches"] is True
    assert len(calibration_calls) == 3
    assert initialized["source_optimizer"].step_calls == 1
    assert initialized["source_optimizer"].zero_calls >= 2
    assert not torch.equal(parameter.detach(), before)
    assert mcln.forward_calls == 1
    assert result["training_diagnostics"]["informative_rows_total"] > 0
    evidence = result["calibration_evidence"]
    assert set(evidence) == {"schema", "canonical_format", "step0", "final",
                             "reproduced"}
    assert evidence["reproduced"]["hit_bits_sha256"] == (
        evidence["final"]["hit_bits_sha256"]
    )
    assert all(
        set(evidence[name]) == {"hit_bits_sha256", "binding_sha256"}
        for name in ("step0", "final", "reproduced")
    )
    assert autocast_calls == [False]
    assert contract_checks == [initialized, initialized, initialized]
    assert all(not model.training for model in (
        initialized["initial_state"]["mcln"],
        initialized["initial_state"]["parent"],
        initialized["initial_state"]["geometry"],
    ))


def test_run_probe_ineligible_restores_earliest_step0_and_reproduces_it(
        tmp_path):
    initialized = _run_initialized(tmp_path)
    parameter = initialized["source_parameters"].parameters[0]
    before = parameter.detach().clone()
    calibrate, calls = _calibration_sequence(eligible=False)

    result = probe.run_source_gate_probe(
        initialized,
        calibration_fn=calibrate,
        move_batch=lambda batch, device: batch,
        input_builder=lambda batch: {},
        full_state_builder=lambda end_points, inputs: {
            "default_scores": torch.cat((
                end_points["score"].reshape(1, 1), torch.zeros(1, 8)
            ), dim=1),
            "boxes": torch.zeros(1, 9, 6),
        },
        target_attacher=lambda full, batch, root_only: batch["target"],
        train_mode_setter=lambda *models: None,
        eval_mode_setter=lambda *models: [model.eval() for model in models],
        gradient_clipper=lambda contract: {
            "source_gate_semantic_classifier": 1.0
        },
        live_contract_validator=lambda value: None,
    )

    assert result["decision"]["eligible"] is False
    assert result["decision"]["selected_step"] == 0
    assert torch.equal(parameter.detach(), before)
    assert calls[2] is None
    assert result["reproduced_report"] == result["step0_report"]
    assert result["calibration_evidence"]["reproduced"][
        "hit_bits_sha256"
    ] == result["calibration_evidence"]["step0"]["hit_bits_sha256"]


@pytest.mark.parametrize("failure", ["short-loader", "calibration"])
def test_run_probe_any_exception_restores_step0_clears_optimizer_and_eval(
        tmp_path, failure):
    initialized = _run_initialized(tmp_path, probe_steps=1)
    parameter = initialized["source_parameters"].parameters[0]
    before = parameter.detach().clone()
    if failure == "short-loader":
        initialized["data"]["fit_loader"] = []
        calibrate, _calls = _calibration_sequence()
    else:
        calibrate, _calls = _calibration_sequence(fail_call=1)

    with pytest.raises((RuntimeError, ValueError), match="fit loader|calibration"):
        probe.run_source_gate_probe(
            initialized,
            calibration_fn=calibrate,
            move_batch=lambda batch, device: batch,
            input_builder=lambda batch: {},
            full_state_builder=lambda end_points, inputs: {
                "default_scores": torch.cat((
                    end_points["score"].reshape(1, 1), torch.zeros(1, 8)
                ), dim=1),
                "boxes": torch.zeros(1, 9, 6),
            },
            target_attacher=lambda full, batch, root_only: batch["target"],
            train_mode_setter=lambda *models: None,
            eval_mode_setter=lambda *models: [
                model.eval() for model in models
            ],
            gradient_clipper=lambda contract: {
                "source_gate_semantic_classifier": 1.0
            },
            live_contract_validator=lambda value: None,
        )

    assert torch.equal(parameter.detach(), before)
    assert initialized["source_optimizer"].state == {}
    assert parameter.grad is None
    assert all(not model.training for model in (
        initialized["initial_state"]["mcln"],
        initialized["initial_state"]["parent"],
        initialized["initial_state"]["geometry"],
    ))


def test_run_probe_rejects_final_calibration_model_mutation_and_rolls_back(
        tmp_path):
    initialized = _run_initialized(tmp_path)
    parameter = initialized["source_parameters"].parameters[0]
    before = parameter.detach().clone()
    base_calibrate, calls = _calibration_sequence(eligible=True)

    def calibrate(value, *, baseline=None, **kwargs):
        call_index = len(calls)
        result = base_calibrate(value, baseline=baseline, **kwargs)
        if call_index == 1:
            with torch.no_grad():
                parameter.add_(5.0)
        return result

    with pytest.raises(RuntimeError, match="calibration.*mutat"):
        probe.run_source_gate_probe(
            initialized,
            calibration_fn=calibrate,
            move_batch=lambda batch, device: batch,
            input_builder=lambda batch: {},
            full_state_builder=lambda end_points, inputs: {
                "default_scores": torch.cat((
                    end_points["score"].reshape(1, 1), torch.zeros(1, 8)
                ), dim=1),
                "boxes": torch.zeros(1, 9, 6),
            },
            target_attacher=lambda full, batch, root_only: batch["target"],
            train_mode_setter=lambda *models: None,
            eval_mode_setter=lambda *models: [
                model.eval() for model in models
            ],
            gradient_clipper=lambda contract: {
                "source_gate_semantic_classifier": 1.0
            },
            live_contract_validator=lambda value: None,
        )

    assert torch.equal(parameter.detach(), before)
    assert initialized["source_optimizer"].state == {}


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _public_identity(path):
    metadata = os.stat(str(path), follow_symlinks=False)
    return {
        "path": str(Path(path).resolve()),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "size": metadata.st_size,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def _production_reports():
    step0, final = _eligible_reports()
    step0["sample_count"] = 3625
    final["sample_count"] = 3625
    for report in (step0, final):
        for branches in report["metrics"].values():
            for metric in branches.values():
                for suffix in ("025", "050"):
                    metric["acc" + suffix] = (
                        metric["hits" + suffix] / 3625.0
                    )
    return step0, final


def _training_diagnostics(update_count=1, informative=2):
    return {
        "schema": "rec-source-gate-training-diagnostics-v1",
        "update_count": update_count,
        "loss": {
            "count": update_count, "min": 0.5, "max": 0.5,
            "mean": 0.5, "last": 0.5,
        },
        "gradient_norm": {
            "count": update_count, "min": 0.25, "max": 0.25,
            "mean": 0.25, "last": 0.25,
        },
        "informative_rows025": informative // 2,
        "informative_rows050": informative - informative // 2,
        "active_violations025": 1,
        "active_violations050": 1,
        "no_positive_rows025": 0,
        "no_positive_rows050": 0,
        "too_few_negative_rows025": 0,
        "too_few_negative_rows050": 0,
        "positive_count025": 1,
        "positive_count050": 1,
        "mean_positive_cutoff_gap025": (
            0.125 if informative // 2 else 0.0
        ),
        "mean_positive_cutoff_gap050": (
            0.25 if informative - informative // 2 else 0.0
        ),
        "informative_rows_total": informative,
    }


def _fit_threshold_stats(informative_rows, mean_gap):
    return {
        "informative_rows": informative_rows,
        "active_violations": 0,
        "no_positive_rows": 0,
        "too_few_negative_rows": 0,
        "positive_count": informative_rows,
        "mean_positive_cutoff_gap": mean_gap,
    }


def test_training_diagnostics_weight_cutoff_gaps_by_informative_rows():
    aggregate = probe._new_training_diagnostics()
    updates = (
        (1, 2.0, 3, 10.0),
        (3, 6.0, 1, 2.0),
    )
    for rows025, gap025, rows050, gap050 in updates:
        probe._update_training_diagnostics(
            aggregate,
            0.5,
            {
                "loss025": 0.25,
                "loss050": 0.25,
                "loss_total": 0.5,
                "threshold025": _fit_threshold_stats(rows025, gap025),
                "threshold050": _fit_threshold_stats(rows050, gap050),
            },
            0.25,
        )

    diagnostics = probe._finalize_training_diagnostics(aggregate, 2)

    assert diagnostics["mean_positive_cutoff_gap025"] == 5.0
    assert diagnostics["mean_positive_cutoff_gap050"] == 8.0
    assert probe.validate_source_gate_training_diagnostics(
        diagnostics, 2
    ) == diagnostics

    zero_aggregate = probe._new_training_diagnostics()
    probe._update_training_diagnostics(
        zero_aggregate,
        0.5,
        {
            "loss025": 0.25,
            "loss050": 0.25,
            "loss_total": 0.5,
            "threshold025": _fit_threshold_stats(1, 3.0),
            "threshold050": _fit_threshold_stats(0, 99.0),
        },
        0.25,
    )
    zero_diagnostics = probe._finalize_training_diagnostics(
        zero_aggregate, 1
    )
    assert zero_diagnostics["mean_positive_cutoff_gap050"] == 0.0
    invalid = copy.deepcopy(diagnostics)
    invalid["mean_positive_cutoff_gap025"] = float("nan")
    with pytest.raises(ValueError, match="finite|diagnostic"):
        probe.validate_source_gate_training_diagnostics(invalid, 2)
    invalid = copy.deepcopy(diagnostics)
    del invalid["mean_positive_cutoff_gap050"]
    with pytest.raises(ValueError, match="schema|diagnostic"):
        probe.validate_source_gate_training_diagnostics(invalid, 2)
    zero_diagnostics["mean_positive_cutoff_gap050"] = 1.0
    with pytest.raises(ValueError, match="zero|diagnostic|coherent"):
        probe.validate_source_gate_training_diagnostics(zero_diagnostics, 1)


def test_calibration_report_rejects_impossible_threshold_hit_order():
    report = _calibration_report()
    metric = report["metrics"]["top1"]["default"]
    metric["hits025"] = 3
    metric["hits050"] = 4
    metric["acc025"] = 0.3
    metric["acc050"] = 0.4

    with pytest.raises(ValueError, match="threshold|hits"):
        probe.validate_source_gate_calibration_report(report)


def test_receipt_rejects_coordinated_transition_tamper(tmp_path):
    receipt = _exact_receipt(tmp_path)
    for report_name in ("final", "reproduced"):
        transition = receipt["calibration"][report_name]["transitions"][
            "top1"
        ]["geometry"]
        transition["gained025"] += 1
        transition["lost025"] += 1

    with pytest.raises(ValueError, match="calibration|binding|transition"):
        probe.validate_source_gate_receipt(receipt)


def _runtime_record(command=None):
    return {
        "schema": "rec-source-gate-runtime-v1",
        "started_utc": "2026-07-17T12:00:00.000000Z",
        "finished_utc": "2026-07-17T12:00:01.000000Z",
        "elapsed_seconds": 1.0,
        "command": (
            [str(sys.executable), str(Path(probe.__file__).resolve())]
            if command is None else list(command)
        ),
        "interpreter": {
            "logical_path": str(sys.executable),
            "resolved_path": str(Path(sys.executable).resolve()),
        },
        "versions": {
            "python": "3.7.16", "torch": "1.10.2",
            "cuda": "11.1", "cudnn": 8005,
        },
        "device": {
            "type": "cuda", "index": 0, "name": "Synthetic CUDA",
            "total_memory_bytes": 1024,
        },
        "peak_cuda_memory": {
            "allocated_bytes": 128, "reserved_bytes": 256,
        },
        "environment": {
            name: None
            for name in probe.legacy.RUNTIME_ENVIRONMENT_ALLOWLIST
        },
    }


def _exact_receipt(tmp_path, *, eligible=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    identities = {
        name: _public_identity(path)
        for name, path in zip((
            "backbone_checkpoint", "parent_reranker", "geometry_reranker",
        ), inputs)
    }
    dependency_paths = {
        "source_gate_runner": Path(probe.__file__),
        "rec_source_gate": Path("models/rec_source_gate.py"),
        "rec_finetune": Path("models/rec_finetune.py"),
        "rec_reranker": Path("models/rec_reranker.py"),
        "rec_candidate_adapter": Path("models/rec_candidate_adapter.py"),
        "rec_mask_geometry": Path("models/rec_mask_geometry.py"),
        "rec_geometry_reranker": Path("models/rec_geometry_reranker.py"),
        "source_choice_adapter": Path("models/source_choice_adapter.py"),
        "source_choice_selector": Path("models/source_choice_selector.py"),
        "train_rec_finetune": Path(
            "scripts/train_scanrefer_rec_finetune.py"
        ),
    }
    manifest = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in dependency_paths.items()
    }
    step0, final = _production_reports()
    if not eligible:
        final_metrics = copy.deepcopy(step0["metrics"])
        final = _calibration_report(final_metrics, baseline=step0)
        final["sample_count"] = 3625
        for branches in final["metrics"].values():
            for metric in branches.values():
                for suffix in ("025", "050"):
                    metric["acc" + suffix] = (
                        metric["hits" + suffix] / 3625.0
                    )
    diagnostics = _training_diagnostics()
    step0_state = _state_digests()
    final_state = _state_digests(mcln_full="e")
    decision = probe.evaluate_source_gate_gate(
        step0,
        final,
        baseline_state_digests=step0_state,
        final_state_digests=final_state,
        training_diagnostics=diagnostics,
        final_step=1,
    )
    selected_state = final_state if decision["eligible"] else step0_state
    reproduced = final if decision["eligible"] else step0
    step0_hit_bits_sha256 = "3" * 64
    final_hit_bits_sha256 = "4" * 64
    reproduced_hit_bits_sha256 = (
        final_hit_bits_sha256 if decision["eligible"]
        else step0_hit_bits_sha256
    )
    calibration_evidence = {
        "schema": "rec-source-gate-calibration-evidence-v1",
        "canonical_format": (
            "rec-source-gate-calibration-transition-binding-sha256-v1"
        ),
        "step0": probe._calibration_evidence_record(
            step0_hit_bits_sha256, step0_hit_bits_sha256, step0
        ),
        "final": probe._calibration_evidence_record(
            step0_hit_bits_sha256, final_hit_bits_sha256, final
        ),
        "reproduced": probe._calibration_evidence_record(
            step0_hit_bits_sha256,
            reproduced_hit_bits_sha256,
            reproduced,
        ),
    }
    parameter_names = [
        "prediction_heads.5.sem_cls_scores_head.weight",
        "prediction_heads.5.sem_cls_scores_head.bias",
    ]
    receipt = {
        "schema": "rec-source-gate-probe-receipt-v1",
        "version": 1,
        "deployable": False,
        "checkpoint_written": False,
        "output_dir": str(output_dir.resolve()),
        "output_files": ["smoke-receipt.json"],
        "validation_data_accessed": False,
        "validation_data_objects_present": False,
        "inputs": {
            "data_root": str(data_root.resolve()),
            **{
                name: {
                    "before": copy.deepcopy(identity),
                    "after": copy.deepcopy(identity),
                }
                for name, identity in identities.items()
            },
        },
        "code": {
            "hashes": manifest,
            "manifest_sha256": _canonical_sha(manifest),
        },
        "runtime": _runtime_record(),
        "data_contract": _exact_data_contract(),
        "trainability": {
            "mode": "final-semantic-classifier-only-v1",
            "allowed_prefix": (
                "prediction_heads.5.sem_cls_scores_head."
            ),
            "parameter_names": parameter_names,
            "parameter_count": 2,
            "parameter_elements": 34,
            "parameter_names_sha256": _canonical_sha(parameter_names),
        },
        "loss_contract": {
            "name": "strict-top8-membership-v1",
            "topk": 8,
            "strict_gt": True,
            "thresholds": [0.25, 0.5],
            "threshold_weights": [2.0, 1.0],
            "margin": 0.0,
            "temperature": 1.0,
            "reduction": "per-threshold-informative-row-mean-sum",
        },
        "optimizer_contract": {
            "type": "AdamW",
            "group_count": 1,
            "group": {
                "name": "source_gate_semantic_classifier",
                "lr": 1e-4,
                "weight_decay": 1e-4,
            },
            "gradient_clip_max_norm": 1.0,
            "scheduler": None,
            "legacy_joint_optimizer_updates": 0,
        },
        "probe": {
            "requested_steps": 1,
            "completed_steps": 1,
            "seed": 0,
            "batch_size": 18,
            "dataset_split": "train",
            "fit_sample_count": 33040,
            "calibration_sample_count": 3625,
            "training_diagnostics": diagnostics,
        },
        "calibration": {
            "step0": step0,
            "final": final,
            "reproduced": copy.deepcopy(reproduced),
        },
        "calibration_evidence": calibration_evidence,
        "decision": decision,
        "state": {
            "canonical_format": "rec-source-gate-state-digest-v1",
            "step0": step0_state,
            "final": final_state,
            "selected": copy.deepcopy(selected_state),
            "restored": copy.deepcopy(selected_state),
            "reproduced": copy.deepcopy(selected_state),
        },
        "restore": {
            "target_step": decision["selected_step"],
            "target_digests": copy.deepcopy(selected_state),
            "actual_digests": copy.deepcopy(selected_state),
            "bitwise_verified": True,
            "reproduction_matches": True,
        },
    }
    receipt["runtime"]["command"] = [
        str(sys.executable),
        str(Path(probe.__file__).resolve()),
    ] + _argv(
        data_root, inputs, output_dir, "--probe-steps", "1"
    )
    return receipt


@pytest.mark.parametrize(
    "tamper",
    [
        "schema", "bool-version", "deployable", "output", "validation",
        "input-identity", "input-sha", "code-digest", "runtime-time",
        "runtime-device", "data", "data-loader-workers",
        "data-loader-pin", "trainability-count",
        "trainability-digest", "loss", "optimizer", "probe-steps",
        "diagnostic-total", "diagnostic-nan", "calibration-acc",
        "calibration-transition", "reproduction", "decision",
        "state", "restore-claim", "code-missing-dependency",
        "runtime-command", "output-overlap", "calibration-binding",
        "calibration-evidence-schema", "calibration-reproduction-bits",
    ],
)
def test_validate_receipt_rejects_every_major_tamper(tmp_path, tamper):
    receipt = _exact_receipt(tmp_path)
    if tamper == "schema":
        receipt["schema"] = "wrong"
    elif tamper == "bool-version":
        receipt["version"] = True
    elif tamper == "deployable":
        receipt["deployable"] = True
    elif tamper == "output":
        receipt["output_files"].append("model.pth")
    elif tamper == "validation":
        receipt["validation_data_accessed"] = True
    elif tamper == "input-identity":
        receipt["inputs"]["parent_reranker"]["after"]["inode"] += 1
    elif tamper == "input-sha":
        receipt["inputs"]["geometry_reranker"]["before"]["sha256"] = "x" * 64
    elif tamper == "code-digest":
        receipt["code"]["manifest_sha256"] = "f" * 64
    elif tamper == "code-missing-dependency":
        del receipt["code"]["hashes"]["rec_candidate_adapter"]
        receipt["code"]["manifest_sha256"] = _canonical_sha(
            receipt["code"]["hashes"]
        )
    elif tamper == "runtime-time":
        receipt["runtime"]["elapsed_seconds"] = -1.0
    elif tamper == "runtime-device":
        receipt["runtime"]["device"]["type"] = "cpu"
    elif tamper == "runtime-command":
        output_index = receipt["runtime"]["command"].index("--output-dir")
        receipt["runtime"]["command"][output_index + 1] = str(
            tmp_path / "different-output"
        )
    elif tamper == "output-overlap":
        receipt["output_dir"] = receipt["inputs"]["data_root"]
    elif tamper == "calibration-binding":
        receipt["calibration_evidence"]["final"][
            "binding_sha256"
        ] = "f" * 64
    elif tamper == "calibration-evidence-schema":
        del receipt["calibration_evidence"]["step0"][
            "hit_bits_sha256"
        ]
    elif tamper == "calibration-reproduction-bits":
        record = receipt["calibration_evidence"]["reproduced"]
        record["hit_bits_sha256"] = "e" * 64
        record["binding_sha256"] = (
            probe._source_gate_calibration_binding_sha256(
                receipt["calibration_evidence"]["step0"][
                    "hit_bits_sha256"
                ],
                record["hit_bits_sha256"],
                receipt["calibration"]["reproduced"],
            )
        )
    elif tamper == "data":
        receipt["data_contract"]["calibration_sample_count"] = 3624
    elif tamper == "data-loader-workers":
        receipt["data_contract"]["loader_execution"]["fit"][
            "num_workers"
        ] = 0
    elif tamper == "data-loader-pin":
        receipt["data_contract"]["loader_execution"]["calibration"][
            "pin_memory"
        ] = True
    elif tamper == "trainability-count":
        receipt["trainability"]["parameter_count"] = True
    elif tamper == "trainability-digest":
        receipt["trainability"]["parameter_names_sha256"] = "f" * 64
    elif tamper == "loss":
        receipt["loss_contract"]["strict_gt"] = False
    elif tamper == "optimizer":
        receipt["optimizer_contract"]["group"]["lr"] = 2e-4
    elif tamper == "probe-steps":
        receipt["probe"]["completed_steps"] = 0
    elif tamper == "diagnostic-total":
        receipt["probe"]["training_diagnostics"][
            "informative_rows_total"
        ] += 1
    elif tamper == "diagnostic-nan":
        receipt["probe"]["training_diagnostics"]["loss"]["last"] = float("nan")
    elif tamper == "calibration-acc":
        receipt["calibration"]["final"]["metrics"]["top1"][
            "geometry"
        ]["acc025"] = 0.0
    elif tamper == "calibration-transition":
        receipt["calibration"]["final"]["transitions"]["top1"][
            "geometry"
        ]["gained025"] += 1
    elif tamper == "reproduction":
        receipt["calibration"]["reproduced"]["digests"][
            "geometry_selected_ious_sha256"
        ] = "f" * 64
    elif tamper == "decision":
        receipt["decision"]["eligible"] = False
    elif tamper == "state":
        receipt["state"]["restored"]["parent"] = "f" * 64
    else:
        receipt["restore"]["bitwise_verified"] = False

    with pytest.raises(ValueError):
        probe.validate_source_gate_receipt(receipt)


def test_validate_receipt_accepts_eligible_and_ineligible_exact_records(tmp_path):
    eligible = _exact_receipt(tmp_path / "eligible")
    ineligible = _exact_receipt(tmp_path / "ineligible", eligible=False)

    assert probe.validate_source_gate_receipt(eligible) == eligible
    assert probe.validate_source_gate_receipt(ineligible) == ineligible
    assert ineligible["decision"]["selected_step"] == 0
    assert ineligible["calibration"]["reproduced"] == (
        ineligible["calibration"]["step0"]
    )


def _publication_initialized(receipt):
    output_dir = Path(receipt["output_dir"])
    output_parent = output_dir.parent
    output_parent_metadata = os.stat(
        str(output_parent), follow_symlinks=False
    )
    paths = argparse.Namespace(
        output_dir=output_dir,
        output_parent=output_parent,
        output_parent_device=output_parent_metadata.st_dev,
        output_parent_inode=output_parent_metadata.st_ino,
        data_root=Path(receipt["inputs"]["data_root"]),
        backbone_checkpoint=Path(
            receipt["inputs"]["backbone_checkpoint"]["before"]["path"]
        ),
        parent_reranker=Path(
            receipt["inputs"]["parent_reranker"]["before"]["path"]
        ),
        geometry_reranker=Path(
            receipt["inputs"]["geometry_reranker"]["before"]["path"]
        ),
    )
    before = {
        name: copy.deepcopy(receipt["inputs"][name]["before"])
        for name in (
            "backbone_checkpoint", "parent_reranker", "geometry_reranker",
        )
    }
    return {
        "paths": paths,
        "input_identities_before": before,
        "publication_code_hashes": copy.deepcopy(receipt["code"]["hashes"]),
    }


def _identity_reader_from_receipt(receipt):
    by_path = {
        record["before"]["path"]: copy.deepcopy(record["before"])
        for name, record in receipt["inputs"].items()
        if name != "data_root"
    }
    return lambda path, label: copy.deepcopy(by_path[str(Path(path).resolve())])


def _runtime_reader_from_receipt(receipt):
    runtime = receipt["runtime"]
    snapshot = {
        name: copy.deepcopy(runtime[name])
        for name in (
            "interpreter", "versions", "device", "peak_cuda_memory",
            "environment",
        )
    }
    return lambda: copy.deepcopy(snapshot)


@pytest.mark.parametrize("drift", ["requires-grad", "mode"])
def test_trainability_receipt_revalidates_live_sealed_contract(drift):
    mcln = _ProbeMCLN()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    contract = probe.rec_source_gate.configure_rec_source_gate_trainability(
        mcln, parent, geometry
    )
    probe.rec_source_gate.set_rec_source_gate_eval_mode(
        mcln, parent, geometry
    )
    if drift == "requires-grad":
        parent.weight.requires_grad_(True)
    else:
        parent.train()

    with pytest.raises((ValueError, RuntimeError), match="train|mode|allowlist"):
        probe._build_trainability_receipt({"source_parameters": contract})


def test_publish_receipt_is_canonical_readonly_exact_and_checkpoint_free(tmp_path):
    receipt = _exact_receipt(tmp_path)
    initialized = _publication_initialized(receipt)

    published = probe.publish_source_gate_receipt(
        initialized,
        receipt,
        manifest_builder=lambda: copy.deepcopy(receipt["code"]["hashes"]),
        identity_reader=_identity_reader_from_receipt(receipt),
        runtime_snapshot_reader=_runtime_reader_from_receipt(receipt),
    )

    output_dir = Path(receipt["output_dir"])
    output = output_dir / "smoke-receipt.json"
    assert published["path"] == output
    assert [path.name for path in output_dir.iterdir()] == [
        "smoke-receipt.json"
    ]
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.read_bytes() == probe._canonical_json_bytes(receipt)
    assert probe.load_strict_source_gate_receipt(output) == receipt
    assert not list(output_dir.rglob("*.pth"))


@pytest.mark.parametrize("drift", ["version", "peak-cuda-memory"])
def test_publish_rejects_runtime_identity_drift_without_output(
        tmp_path, drift):
    receipt = _exact_receipt(tmp_path)
    initialized = _publication_initialized(receipt)
    runtime_reader = _runtime_reader_from_receipt(receipt)

    def drifted_runtime():
        snapshot = runtime_reader()
        if drift == "version":
            snapshot["versions"]["torch"] = "different"
        else:
            snapshot["peak_cuda_memory"]["allocated_bytes"] += 1
        return snapshot

    with pytest.raises(RuntimeError, match="runtime"):
        probe.publish_source_gate_receipt(
            initialized,
            receipt,
            manifest_builder=lambda: copy.deepcopy(
                receipt["code"]["hashes"]
            ),
            identity_reader=_identity_reader_from_receipt(receipt),
            runtime_snapshot_reader=drifted_runtime,
        )

    assert not Path(receipt["output_dir"]).exists()


def test_publish_binds_parent_before_staging_creation(tmp_path, monkeypatch):
    receipt = _exact_receipt(tmp_path)
    initialized = _publication_initialized(receipt)
    output_dir = Path(receipt["output_dir"])
    output_parent = output_dir.parent
    original_parent = tmp_path / "original-output-parent"
    original_mkdir = os.mkdir
    replacement_happened = []

    def replace_parent_during_staging_mkdir(path, mode=0o777, *, dir_fd=None):
        if (not replacement_happened
                and Path(os.fspath(path)).name.startswith(
                    ".probe.staging-"
                )):
            output_parent.rename(original_parent)
            original_mkdir(str(output_parent), 0o777)
            replacement_happened.append(True)
        if dir_fd is None:
            return original_mkdir(path, mode)
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(probe.os, "mkdir", replace_parent_during_staging_mkdir)

    with pytest.raises(RuntimeError, match="parent.*identity|identity.*parent"):
        probe.publish_source_gate_receipt(
            initialized,
            receipt,
            manifest_builder=lambda: copy.deepcopy(
                receipt["code"]["hashes"]
            ),
            identity_reader=_identity_reader_from_receipt(receipt),
            runtime_snapshot_reader=_runtime_reader_from_receipt(receipt),
        )

    assert replacement_happened == [True]
    assert not output_dir.exists()
    assert not list(output_parent.glob(".probe.staging-*"))
    assert not (original_parent / output_dir.name).exists()
    assert not list(original_parent.glob(".probe.staging-*"))


def test_publish_removes_bound_output_when_parent_changes_before_rename(
        tmp_path):
    receipt = _exact_receipt(tmp_path)
    initialized = _publication_initialized(receipt)
    output_dir = Path(receipt["output_dir"])
    output_parent = output_dir.parent
    original_parent = tmp_path / "original-output-parent"
    replacement_happened = []

    def rename_after_replacing_parent(*args):
        output_parent.rename(original_parent)
        output_parent.mkdir()
        replacement_happened.append(True)
        if len(args) == 2:
            return probe._rename_directory_noreplace(*args)
        return probe._rename_directory_noreplace_at(*args)

    with pytest.raises(RuntimeError, match="parent.*identity|identity.*parent"):
        probe.publish_source_gate_receipt(
            initialized,
            receipt,
            rename_fn=rename_after_replacing_parent,
            manifest_builder=lambda: copy.deepcopy(
                receipt["code"]["hashes"]
            ),
            identity_reader=_identity_reader_from_receipt(receipt),
            runtime_snapshot_reader=_runtime_reader_from_receipt(receipt),
        )

    assert replacement_happened == [True]
    assert not output_dir.exists()
    assert not list(output_parent.glob(".probe.staging-*"))
    assert not (original_parent / output_dir.name).exists()
    assert not list(original_parent.glob(".probe.staging-*"))


def test_bound_cleanup_opens_and_fstat_checks_child_before_deleting(
        tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    owned = parent / "owned"
    moved_owned = parent / "moved-owned"
    parent.mkdir()
    owned.mkdir()
    (owned / "owned.txt").write_text("owned", encoding="ascii")
    owned_metadata = owned.stat()
    owned_identity = (owned_metadata.st_dev, owned_metadata.st_ino)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(str(parent), flags)
    original_open = os.open
    replacement_happened = []

    def replace_before_child_open(path, open_flags, mode=0o777, *, dir_fd=None):
        if (not replacement_happened and path == "owned"
                and dir_fd == parent_fd):
            os.rename(
                "owned", "moved-owned",
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            )
            os.mkdir("owned", 0o700, dir_fd=parent_fd)
            (owned / "replacement.txt").write_text(
                "replacement", encoding="ascii"
            )
            replacement_happened.append(True)
        if dir_fd is None:
            return original_open(path, open_flags, mode)
        return original_open(
            path, open_flags, mode, dir_fd=dir_fd
        )

    monkeypatch.setattr(probe.os, "open", replace_before_child_open)
    try:
        removed = probe._remove_bound_directory_if_owned(
            parent_fd, "owned", owned_identity
        )
    finally:
        os.close(parent_fd)

    assert replacement_happened == [True]
    assert removed is False
    assert (owned / "replacement.txt").read_text(encoding="ascii") == (
        "replacement"
    )
    assert (moved_owned / "owned.txt").read_text(encoding="ascii") == (
        "owned"
    )


def test_bound_cleanup_recurses_via_child_fds(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    owned = parent / "owned"
    nested = owned / "nested"
    parent.mkdir()
    nested.mkdir(parents=True)
    (owned / "receipt.json").write_text("receipt", encoding="ascii")
    (nested / "nested.txt").write_text("nested", encoding="ascii")
    (owned / "linked").symlink_to(nested, target_is_directory=True)
    owned_metadata = owned.stat()
    owned_identity = (owned_metadata.st_dev, owned_metadata.st_ino)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(str(parent), flags)
    original_unlink = os.unlink
    original_rmdir = os.rmdir
    unlink_dir_fds = []
    rmdir_dir_fds = []

    def unlink(path, *, dir_fd=None):
        unlink_dir_fds.append(dir_fd)
        return original_unlink(path, dir_fd=dir_fd)

    def rmdir(path, *, dir_fd=None):
        rmdir_dir_fds.append(dir_fd)
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(probe.shutil, "rmtree", lambda path: (_ for _ in ()).throw(
        AssertionError("path-based rmtree used")
    ))
    monkeypatch.setattr(probe.os, "unlink", unlink)
    monkeypatch.setattr(probe.os, "rmdir", rmdir)
    try:
        removed = probe._remove_bound_directory_if_owned(
            parent_fd, "owned", owned_identity
        )
    finally:
        os.close(parent_fd)

    assert removed is True
    assert not owned.exists()
    assert unlink_dir_fds and all(value is not None for value in unlink_dir_fds)
    assert rmdir_dir_fds and all(value is not None for value in rmdir_dir_fds)


def test_staging_creation_strictly_cleans_after_post_mkdir_failure(
        tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(str(parent), flags)
    original_fsync = os.fsync
    original_rmdir = os.rmdir
    interrupted = []

    def fail_parent_fsync(descriptor):
        if descriptor == parent_fd:
            raise OSError("injected parent fsync failure")
        return original_fsync(descriptor)

    def interrupt_first_staging_rmdir(path, *, dir_fd=None):
        if (not interrupted and dir_fd == parent_fd
                and str(path).startswith(".probe.staging-")):
            interrupted.append(True)
            raise InterruptedError(probe.errno.EINTR, "interrupted rmdir")
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(probe.os, "fsync", fail_parent_fsync)
    monkeypatch.setattr(probe.os, "rmdir", interrupt_first_staging_rmdir)
    try:
        with pytest.raises(OSError, match="fsync failure"):
            probe._create_unique_staging_directory(parent_fd, "probe")
    finally:
        os.close(parent_fd)

    assert interrupted == [True]
    assert not list(parent.glob(".probe.staging-*"))


@pytest.mark.parametrize(
    ("helper_name", "operation_name"),
    [
        ("_unlink_at_retry", "unlink"),
        ("_rmdir_at_retry", "rmdir"),
    ],
)
def test_bound_cleanup_limits_eintr_retries(
        tmp_path, monkeypatch, helper_name, operation_name):
    directory = tmp_path / "directory"
    directory.mkdir()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(str(directory), flags)
    calls = []

    def always_interrupted(path, *, dir_fd=None):
        calls.append((path, dir_fd))
        if len(calls) > 8:
            raise AssertionError("filesystem EINTR retry was unbounded")
        raise InterruptedError(probe.errno.EINTR, "interrupted operation")

    monkeypatch.setattr(probe.os, operation_name, always_interrupted)
    try:
        with pytest.raises(InterruptedError):
            getattr(probe, helper_name)(directory_fd, "entry")
    finally:
        os.close(directory_fd)

    assert len(calls) == 8
    assert all(dir_fd == directory_fd for _path, dir_fd in calls)


def test_publish_rechecks_all_output_ancestors_before_staging(tmp_path):
    receipt = _exact_receipt(tmp_path)
    safe_ancestor = tmp_path / "safe-output"
    safe_parent = safe_ancestor / "nested"
    safe_parent.mkdir(parents=True)
    output_dir = safe_parent / "probe"
    receipt["output_dir"] = str(output_dir.resolve())
    output_index = receipt["runtime"]["command"].index("--output-dir")
    receipt["runtime"]["command"][output_index + 1] = receipt["output_dir"]
    initialized = _publication_initialized(receipt)
    original = tmp_path / "original-output"
    redirected = tmp_path / "redirected"
    (redirected / "nested").mkdir(parents=True)
    manifest_calls = []

    def manifest_builder():
        if not manifest_calls:
            safe_ancestor.rename(original)
            safe_ancestor.symlink_to(redirected, target_is_directory=True)
        manifest_calls.append(True)
        return copy.deepcopy(receipt["code"]["hashes"])

    with pytest.raises(ValueError, match="symlink"):
        probe.publish_source_gate_receipt(
            initialized,
            receipt,
            manifest_builder=manifest_builder,
            identity_reader=_identity_reader_from_receipt(receipt),
            runtime_snapshot_reader=_runtime_reader_from_receipt(receipt),
        )

    assert not (redirected / "nested" / "probe").exists()
    assert not list((redirected / "nested").glob(".probe.staging-*"))


@pytest.mark.parametrize(
    "failure", [
        "writer", "mutation", "written", "validated", "finalize",
        "rename", "rename-after-move", "committed",
    ]
)
def test_publish_failure_or_mutation_leaves_no_output_or_staging(
        tmp_path, failure):
    receipt = _exact_receipt(tmp_path)
    initialized = _publication_initialized(receipt)

    def writer(path, payload):
        if failure == "writer":
            raise RuntimeError("writer failed")
        value = copy.deepcopy(payload)
        if failure == "mutation":
            value["deployable"] = True
        path.write_bytes(probe._canonical_json_bytes(value))

    def inject(stage):
        if stage == failure:
            raise RuntimeError("injected " + stage)

    def rename(parent_fd, source_name, destination_name):
        if failure == "rename":
            raise RuntimeError("rename failed")
        result = probe._rename_directory_noreplace_at(
            parent_fd, source_name, destination_name
        )
        if failure == "rename-after-move":
            raise RuntimeError("rename failed after move")
        return result

    with pytest.raises((RuntimeError, ValueError)):
        probe.publish_source_gate_receipt(
            initialized,
            receipt,
            writer=writer,
            failure_injector=inject,
            rename_fn=rename,
            manifest_builder=lambda: copy.deepcopy(
                receipt["code"]["hashes"]
            ),
            identity_reader=_identity_reader_from_receipt(receipt),
            runtime_snapshot_reader=_runtime_reader_from_receipt(receipt),
        )

    output_dir = Path(receipt["output_dir"])
    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(
        ".{}.staging-*".format(output_dir.name)
    ))


@pytest.mark.parametrize("corruption", ["duplicate", "nan", "noncanonical"])
def test_strict_receipt_reload_rejects_duplicate_nan_and_noncanonical_bytes(
        tmp_path, corruption):
    receipt = _exact_receipt(tmp_path / "fixture")
    path = tmp_path / "receipt.json"
    encoded = probe._canonical_json_bytes(receipt)
    if corruption == "duplicate":
        encoded = b'{"version":1,' + encoded[1:]
    elif corruption == "nan":
        encoded = encoded.replace(
            b'"elapsed_seconds":1.0', b'"elapsed_seconds":NaN'
        )
    else:
        encoded += b"\n"
    path.write_bytes(encoded)

    with pytest.raises(ValueError):
        probe.load_strict_source_gate_receipt(path)


def test_build_receipt_binds_live_contracts_and_rechecks_code_and_inputs(
        tmp_path):
    expected = _exact_receipt(tmp_path)
    mcln = _ProbeMCLN()
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    parameter_contract = (
        probe.rec_source_gate.configure_rec_source_gate_trainability(
            mcln, parent, geometry
        )
    )
    probe.rec_source_gate.set_rec_source_gate_eval_mode(
        mcln, parent, geometry
    )
    optimizer = probe.rec_source_gate.build_rec_source_gate_optimizer(
        parameter_contract
    )
    initialized = _publication_initialized(expected)
    initialized.update({
        "train_data_contract": _exact_data_contract(),
        "data": {
            "fit_loader": argparse.Namespace(
                num_workers=2, pin_memory=False
            ),
            "calibration_loader": argparse.Namespace(
                num_workers=2, pin_memory=False
            ),
        },
        "source_parameters": parameter_contract,
        "source_optimizer": optimizer,
        "probe_steps": 1,
        "seed": 0,
        "legacy_joint_optimizer_updates": 0,
    })
    step0, final = _production_reports()
    diagnostics = _training_diagnostics()
    step0_state = _state_digests()
    final_state = _state_digests(mcln_full="e")
    decision = probe.evaluate_source_gate_gate(
        step0, final,
        baseline_state_digests=step0_state,
        final_state_digests=final_state,
        training_diagnostics=diagnostics,
        final_step=1,
    )
    run_result = {
        "requested_steps": 1,
        "completed_steps": 1,
        "step0_report": step0,
        "final_report": final,
        "reproduced_report": copy.deepcopy(final),
        "decision": decision,
        "training_diagnostics": diagnostics,
        "calibration_evidence": copy.deepcopy(
            expected["calibration_evidence"]
        ),
        "state_digests": {
            "step0": step0_state, "final": final_state,
            "selected": final_state, "restored": final_state,
            "reproduced": final_state,
        },
        "restore": {
            "target_step": 1, "target_digests": final_state,
            "actual_digests": final_state, "bitwise_verified": True,
        },
        "reproduction_matches": True,
        "legacy_joint_optimizer_updates": 0,
    }

    receipt = probe.build_source_gate_receipt(
        initialized,
        run_result,
        _runtime_record(command=expected["runtime"]["command"]),
        manifest_builder=lambda: copy.deepcopy(expected["code"]["hashes"]),
        identity_reader=_identity_reader_from_receipt(expected),
        runtime_snapshot_reader=_runtime_reader_from_receipt(expected),
    )

    assert probe.validate_source_gate_receipt(receipt) == receipt
    assert receipt["trainability"]["parameter_count"] == 1
    assert receipt["trainability"]["parameter_elements"] == 1
    assert receipt["optimizer_contract"]["legacy_joint_optimizer_updates"] == 0

    initialized["data"]["fit_loader"].pin_memory = True
    with pytest.raises(RuntimeError, match="loader contract"):
        probe.build_source_gate_receipt(
            initialized,
            run_result,
            _runtime_record(command=expected["runtime"]["command"]),
            manifest_builder=lambda: copy.deepcopy(
                expected["code"]["hashes"]
            ),
            identity_reader=_identity_reader_from_receipt(expected),
            runtime_snapshot_reader=_runtime_reader_from_receipt(expected),
        )
    initialized["data"]["fit_loader"].pin_memory = False

    def drifted_peak_memory():
        snapshot = _runtime_reader_from_receipt(expected)()
        snapshot["peak_cuda_memory"]["reserved_bytes"] += 1
        return snapshot

    with pytest.raises(RuntimeError, match="runtime"):
        probe.build_source_gate_receipt(
            initialized,
            run_result,
            _runtime_record(command=expected["runtime"]["command"]),
            manifest_builder=lambda: copy.deepcopy(
                expected["code"]["hashes"]
            ),
            identity_reader=_identity_reader_from_receipt(expected),
            runtime_snapshot_reader=drifted_peak_memory,
        )


def test_main_uses_clock_runtime_seams_and_rolls_back_on_writer_failure(
        tmp_path):
    data_root, inputs, output_dir = _cli_paths(tmp_path)
    argv = _argv(
        data_root, inputs, output_dir, "--probe-steps", "1"
    )
    mcln = torch.nn.Linear(1, 1)
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(mcln.parameters(), lr=0.1)
    initialized = {
        "initial_state": {
            "mcln": mcln, "parent": parent, "geometry": geometry,
        },
        "source_optimizer": optimizer,
    }
    before = legacy_state = probe.legacy.snapshot_rec_finetune_state(
        mcln, parent, geometry
    )
    calls = []
    utc_values = iter((
        "2026-07-17T12:00:00.000000Z",
        "2026-07-17T12:00:01.000000Z",
    ))
    monotonic_values = iter((10.0, 11.5))

    def runner(value):
        calls.append("run")
        with torch.no_grad():
            value["initial_state"]["mcln"].weight.add_(10.0)
        return {"_step0_snapshot": legacy_state}

    def runtime_builder(**kwargs):
        calls.append(("runtime", kwargs))
        return _runtime_record()

    with pytest.raises(RuntimeError, match="publish failed"):
        probe.main(
            argv,
            utc_now=lambda: next(utc_values),
            monotonic=lambda: next(monotonic_values),
            determinism_setter=lambda: calls.append("determinism"),
            peak_memory_resetter=lambda: calls.append("peak-reset"),
            initializer=lambda args: initialized,
            runner=runner,
            runtime_builder=runtime_builder,
            receipt_builder=lambda init, result, runtime: {},
            publisher=lambda init, receipt: (_ for _ in ()).throw(
                RuntimeError("publish failed")
            ),
        )

    assert torch.equal(mcln.state_dict()["weight"], before["mcln"]["weight"])
    assert all(not model.training for model in (mcln, parent, geometry))
    assert calls[:3] == ["determinism", "peak-reset", "run"]
    runtime_call = calls[3][1]
    assert runtime_call["elapsed_seconds"] == 1.5
    assert runtime_call["started_utc"].endswith("Z")
    assert runtime_call["finished_utc"].endswith("Z")
