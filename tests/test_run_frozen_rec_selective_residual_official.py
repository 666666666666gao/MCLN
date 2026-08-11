import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_frozen_rec_selective_residual_official as official


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bind_paths(monkeypatch, tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    paths = {
        "selection": root / "selection.json",
        "residual": root / "selected_residual.pth",
        "residual_receipt": root / "result-receipt.json",
        "parent": root / "parent.pth",
        "backbone": root / "backbone.pth",
        "geometry": root / "geometry.pth",
    }
    paths["residual"].write_bytes(b"residual-artifact")
    for name in ("parent", "backbone", "geometry"):
        paths[name].write_bytes(name.encode("ascii"))
    artifact = {
        "name": paths["residual"].name,
        "sha256": _sha256(paths["residual"]),
    }
    input_sha256 = {
        "backbone": _sha256(paths["backbone"]),
        "parent": _sha256(paths["parent"]),
        "geometry": _sha256(paths["geometry"]),
    }
    online = {
        "sample_count": 3625,
        "baseline_hits025": 3461,
        "baseline_hits050": 3316,
        "candidate_hits025": 3524,
        "candidate_hits050": 3316,
    }
    receipt = {
        "schema": "rec-selective-residual-promotion-receipt-v1",
        "version": 1,
        "selected": "deployable_residual",
        "deployable": True,
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "artifact": artifact,
        "input_sha256": input_sha256,
        "online_calibration": online,
    }
    paths["residual_receipt"].write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    selection = {
        "schema": "rec-selective-residual-deployment-selection-v1",
        "version": 1,
        "selected": "deployable_residual",
        "deployable": True,
        "validation_data_accessed": False,
        "artifact": artifact,
        "input_sha256": input_sha256,
        "online_calibration": online,
        "promotion_receipt_sha256": _sha256(paths["residual_receipt"]),
    }
    paths["selection"].write_text(
        json.dumps(selection, sort_keys=True), encoding="utf-8"
    )
    for path in paths.values():
        path.chmod(0o444)
    run_root = tmp_path / "official"
    registry = tmp_path / "claims"
    registry.mkdir()
    monkeypatch.setattr(official, "OFFICIAL_CLAIM_REGISTRY", registry)
    monkeypatch.setattr(official, "OFFICIAL_SELECTION_RECORD_PATH", paths["selection"])
    monkeypatch.setattr(official, "OFFICIAL_SELECTED_RESIDUAL_ARTIFACT_PATH", paths["residual"])
    monkeypatch.setattr(official, "OFFICIAL_RESIDUAL_RECEIPT_PATH", paths["residual_receipt"])
    monkeypatch.setattr(official, "OFFICIAL_PARENT_ARTIFACT_PATH", paths["parent"])
    monkeypatch.setattr(official, "OFFICIAL_CHECKPOINT_PATH", paths["backbone"])
    monkeypatch.setattr(official, "OFFICIAL_GEOMETRY_ARTIFACT_PATH", paths["geometry"])
    monkeypatch.setattr(official, "OFFICIAL_RUN_ROOT", run_root)
    monkeypatch.setattr(official, "OFFICIAL_CODE_ROOT", Path(__file__).parents[1])
    python_executable = Path(os.sys.executable)
    monkeypatch.setattr(official, "OFFICIAL_PYTHON_EXECUTABLE", python_executable)
    python_target = python_executable.resolve()
    monkeypatch.setattr(official, "OFFICIAL_PYTHON_LINK_TARGET", os.readlink(str(python_executable)))
    monkeypatch.setattr(official, "OFFICIAL_PYTHON_TARGET_SHA256", _sha256(python_target))
    monkeypatch.setattr(official, "OFFICIAL_PYTHON_TARGET_SIZE", python_target.stat().st_size)
    monkeypatch.setattr(
        official, "OFFICIAL_PYTHON_TARGET_MODE", python_target.stat().st_mode & 0o777
    )
    for constant, path in (
        ("OFFICIAL_SELECTION_RECORD_SHA256", paths["selection"]),
        ("OFFICIAL_SELECTED_RESIDUAL_ARTIFACT_SHA256", paths["residual"]),
        ("OFFICIAL_RESIDUAL_RECEIPT_SHA256", paths["residual_receipt"]),
        ("OFFICIAL_PARENT_ARTIFACT_SHA256", paths["parent"]),
        ("OFFICIAL_CHECKPOINT_SHA256", paths["backbone"]),
        ("OFFICIAL_GEOMETRY_ARTIFACT_SHA256", paths["geometry"]),
    ):
        monkeypatch.setattr(official, constant, _sha256(path))
    return paths, run_root


def _metric_text(hits025=5705, hits050=4469):
    acc025 = "%.5f" % (hits025 / float(official.OFFICIAL_SAMPLE_COUNT))
    acc050 = "%.5f" % (hits050 / float(official.OFFICIAL_SAMPLE_COUNT))
    return (
        "length of testing dataset: 9508\n"
        "last_ position alignment Acc0.25: Top-1: {}, Top-5: {}\n"
        "last_ position alignment Acc0.50: Top-1: {}, Top-5: {}\n"
    ).format(acc025, acc025, acc050, acc050)


def _full_config(paths, run_path):
    config = dict(official._CONFIG_VALUES)
    config.update({
        "log_dir": str(run_path),
        "checkpoint_path": str(paths["backbone"]),
        "rec_reranker_checkpoint": str(paths["parent"]),
        "rec_geometry_reranker_checkpoint": str(paths["geometry"]),
        "rec_selective_residual_checkpoint": str(paths["residual"]),
        "data_root": str(official.OFFICIAL_DATA_ROOT) + os.sep,
    })
    return config


def _launch_with_metrics(monkeypatch, tmp_path, hits025=5705, hits050=4469):
    paths, run_root = _bind_paths(monkeypatch, tmp_path)
    run_path = official.timestamp_run_root(run_root) / "1700000001"

    def fake_run(_command, **kwargs):
        run_path.mkdir(parents=True)
        (run_path / "config.json").write_text(
            json.dumps(_full_config(paths, run_path)), encoding="utf-8"
        )
        evidence = _metric_text(hits025, hits050)
        (run_path / "log.txt").write_text(evidence, encoding="utf-8")
        kwargs["stdout"].write(evidence.encode("utf-8"))
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    launched = official.run_official_launch()
    return paths, run_path, launched


def test_authoritative_command_binds_fixed_residual_and_no_gt_flags():
    command = official.build_authoritative_command()
    assert "--eval_use_rec_selective_residual_scores" in command
    assert "--rec_selective_residual_checkpoint" in command
    assert "--butd" in command
    assert "--butd_gt" not in command
    assert "--butd_cls" not in command
    assert command[command.index("--nproc_per_node") + 1] == "1"


def test_metric_parser_recovers_exact_official_hits_and_rejects_duplicates():
    text = _metric_text()
    assert official.parse_official_metrics(text, text) == {
        "printed_acc025": "0.60002",
        "printed_acc050": "0.47003",
        "hits025": 5705,
        "hits050": 4469,
    }
    with pytest.raises(ValueError, match="exactly once"):
        official.parse_official_metrics(text + text, text)


def test_official_result_validator_requires_9508_no_gt_and_gate():
    with pytest.raises(ValueError):
        official.validate_official_result({
            "schema": official.OFFICIAL_RESULT_SCHEMA,
            "version": official.OFFICIAL_RESULT_VERSION,
            "sample_count": 9507,
        })


def test_launcher_is_one_shot_and_preserves_best_inputs(monkeypatch, tmp_path):
    paths, run_root = _bind_paths(monkeypatch, tmp_path)
    run_path = official.timestamp_run_root(run_root) / "1700000001"

    def fake_run(command, **kwargs):
        assert "--eval_use_rec_selective_residual_scores" in command
        run_path.mkdir(parents=True)
        (run_path / "config.json").write_text(json.dumps({
            "eval": True,
            "dataset": ["scanrefer"],
            "test_dataset": "scanrefer",
            "butd": True,
            "butd_gt": False,
            "butd_cls": False,
            "eval_use_rec_selective_residual_scores": True,
            "batch_size": 12,
            "local_rank": 0,
        }), encoding="utf-8")
        (run_path / "log.txt").write_text(_metric_text(), encoding="utf-8")
        kwargs["stdout"].write(_metric_text().encode("utf-8"))
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    before = {name: path.read_bytes() for name, path in paths.items()}
    launched = official.run_official_launch()
    assert launched["success"] is True
    assert launched["inference_uses_ground_truth"] is False
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    with pytest.raises(FileExistsError, match="claim"):
        official.run_official_launch()


def test_sealer_sets_acceptance_gate_only_for_both_metrics(monkeypatch, tmp_path):
    paths, run_root = _bind_paths(monkeypatch, tmp_path)
    run_path = official.timestamp_run_root(run_root) / "1700000001"

    def fake_run(_command, **kwargs):
        run_path.mkdir(parents=True)
        config = dict(official._CONFIG_VALUES)
        config.update({
            "log_dir": str(run_path),
            "checkpoint_path": str(paths["backbone"]),
            "rec_reranker_checkpoint": str(paths["parent"]),
            "rec_geometry_reranker_checkpoint": str(paths["geometry"]),
            "rec_selective_residual_checkpoint": str(paths["residual"]),
            "data_root": str(official.OFFICIAL_DATA_ROOT) + os.sep,
        })
        (run_path / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        evidence = _metric_text(5705, 4468)
        (run_path / "log.txt").write_text(evidence, encoding="utf-8")
        kwargs["stdout"].write(evidence.encode("utf-8"))
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    official.run_official_launch()
    record = official.seal_official_result()
    assert record["hits025"] == 5705
    assert record["hits050"] == 4468
    assert record["acceptance_gate_pass"] is False
    assert Path(official.official_result_path()).stat().st_mode & 0o777 == 0o444


def test_official_result_gate_boundaries_are_strict():
    assert official.acceptance_gate_pass(5705, 4469, 9508, False)
    assert not official.acceptance_gate_pass(5704, 4469, 9508, False)
    assert not official.acceptance_gate_pass(5705, 4468, 9508, False)
    assert not official.acceptance_gate_pass(5705, 4469, 9507, False)
    assert not official.acceptance_gate_pass(5705, 4469, 9508, True)


def test_cli_exposes_only_launch_and_seal_operations():
    parser = official._build_argument_parser()
    assert parser.parse_args(["launch"]).operation == "launch"
    assert parser.parse_args(["seal"]).operation == "seal"
    with pytest.raises(SystemExit):
        parser.parse_args(["compare"])


@pytest.mark.parametrize("constant", [
    "OFFICIAL_SELECTION_RECORD_SHA256",
    "OFFICIAL_SELECTED_RESIDUAL_ARTIFACT_SHA256",
    "OFFICIAL_RESIDUAL_RECEIPT_SHA256",
    "OFFICIAL_GEOMETRY_ARTIFACT_SHA256",
    "OFFICIAL_PARENT_ARTIFACT_SHA256",
    "OFFICIAL_CHECKPOINT_SHA256",
])
def test_wrong_frozen_sha_aborts_before_subprocess(monkeypatch, tmp_path, constant):
    _bind_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(official, constant, "0" * 64)
    monkeypatch.setattr(
        official.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("official subprocess must not run"),
    )
    with pytest.raises(ValueError, match="SHA"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


def test_sealed_result_contains_all_frozen_artifact_bindings_and_immutable_logs(
        monkeypatch, tmp_path):
    paths, run_path, _launched = _launch_with_metrics(monkeypatch, tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}
    record = official.seal_official_result()
    assert set(record["artifacts"]) == {
        "selection_record", "residual_artifact", "residual_receipt",
        "geometry_artifact", "parent_artifact", "checkpoint",
    }
    assert record["acceptance_gate_pass"] is True
    for path in (
            Path(record["files"]["stdout"]["path"]),
            Path(record["files"]["log"]["path"]),
            Path(record["files"]["config"]["path"]),
            Path(official.official_result_path())):
        assert path.stat().st_mode & 0o777 == 0o444
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert run_path.joinpath("log.txt").is_file()


def test_failed_metric_gate_still_seals_result_and_keeps_best_weights(
        monkeypatch, tmp_path):
    paths, _run_path, _launched = _launch_with_metrics(
        monkeypatch, tmp_path, hits025=5705, hits050=4468
    )
    before = {name: path.read_bytes() for name, path in paths.items()}
    record = official.seal_official_result()
    assert record["acceptance_gate_pass"] is False
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_acceptance_gate_rejects_ground_truth_and_sample_count_drift():
    assert not official.acceptance_gate_pass(5705, 4469, 9507, False)
    assert not official.acceptance_gate_pass(5705, 4469, 9508, True)
