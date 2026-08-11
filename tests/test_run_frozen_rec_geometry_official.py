import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.evaluate_frozen_rec_geometry_sidecar as frozen_sidecar
import scripts.run_frozen_rec_geometry_official as official


@pytest.fixture(autouse=True)
def _isolate_official_claim_registry(monkeypatch, tmp_path):
    registry = tmp_path / "authoritative-official-claims"
    registry.mkdir()
    monkeypatch.setattr(
        official,
        "OFFICIAL_CLAIM_REGISTRY",
        registry,
        raising=False,
    )


def _launch_inputs(tmp_path):
    selected = tmp_path / "selected.pth"
    parent = tmp_path / "parent.pth"
    checkpoint = tmp_path / "backbone.pth"
    selection = tmp_path / "selection.json"
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (code_root / "train_dist_mod.py").write_text(
        "# synthetic entrypoint\n", encoding="utf-8"
    )
    pointnet2 = code_root / "pointnet2"
    pointnet2.mkdir()
    native_module = pointnet2 / "_ext.synthetic.so"
    native_module.write_bytes(b"synthetic native module")
    python_bin = tmp_path / "python-env" / "bin"
    python_bin.mkdir(parents=True)
    python_target = python_bin / "python3.7"
    python_target.write_bytes(b"synthetic python executable")
    os.chmod(str(python_target), 0o755)
    python_executable = python_bin / "python"
    python_executable.symlink_to("python3.7")
    selected.write_bytes(b"selected artifact")
    parent.write_bytes(b"parent artifact")
    checkpoint.write_bytes(b"backbone checkpoint")
    selection.write_text(json.dumps({
        "selection_uses_validation": False,
        "winner": {
            "selected_filename": selected.name,
            "selected_sha256": hashlib.sha256(
                selected.read_bytes()
            ).hexdigest(),
        },
        "common_train_provenance": {
            "parent_artifact_sha256": hashlib.sha256(
                parent.read_bytes()
            ).hexdigest(),
        },
    }, sort_keys=True), encoding="utf-8")
    return {
        "selection_record": selection,
        "selected_artifact": selected,
        "parent_artifact": parent,
        "checkpoint": checkpoint,
        "code_root": code_root,
        "native_module": native_module,
        "python_executable": python_executable,
        "python_target": python_target,
        "sidecar_record": tmp_path / "sidecar_record.json",
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bind_authoritative(monkeypatch, inputs, run_root):
    bindings = {
        "OFFICIAL_SELECTION_RECORD_PATH": inputs["selection_record"],
        "OFFICIAL_SELECTED_ARTIFACT_PATH": inputs["selected_artifact"],
        "OFFICIAL_PARENT_ARTIFACT_PATH": inputs["parent_artifact"],
        "OFFICIAL_CHECKPOINT_PATH": inputs["checkpoint"],
        "OFFICIAL_RUN_ROOT": Path(run_root),
        "OFFICIAL_CODE_ROOT": inputs["code_root"],
        "OFFICIAL_SIDECAR_RECORD_PATH": inputs["sidecar_record"],
        "OFFICIAL_PYTHON_EXECUTABLE": inputs["python_executable"],
        "OFFICIAL_PYTHON_LINK_TARGET": "python3.7",
        "OFFICIAL_PYTHON_TARGET_SHA256": _sha256(
            inputs["python_target"]
        ),
        "OFFICIAL_PYTHON_TARGET_SIZE": inputs["python_target"].stat().st_size,
        "OFFICIAL_PYTHON_TARGET_MODE": 0o755,
        "OFFICIAL_SELECTION_RECORD_SHA256": _sha256(
            inputs["selection_record"]
        ),
        "OFFICIAL_SELECTED_ARTIFACT_SHA256": _sha256(
            inputs["selected_artifact"]
        ),
        "OFFICIAL_PARENT_ARTIFACT_SHA256": _sha256(
            inputs["parent_artifact"]
        ),
        "OFFICIAL_CHECKPOINT_SHA256": _sha256(inputs["checkpoint"]),
        "OFFICIAL_SIDECAR_RECORD_SHA256": "0" * 64,
    }
    for name, value in bindings.items():
        monkeypatch.setattr(official, name, value, raising=False)

    calls = []

    def synthetic_preflight(selection, selected, parent, device="cuda:0"):
        calls.append((selection, selected, parent, device))
        return {
            "selection_path": Path(selection).resolve(),
            "selected_path": Path(selected).resolve(),
            "parent_path": Path(parent).resolve(),
            "selection_record_sha256": _sha256(selection),
            "selected_artifact_sha256": _sha256(selected),
            "parent_artifact_sha256": _sha256(parent),
            "selection_uses_validation": False,
            "geometry_artifact": {
                "checkpoint_sha256": _sha256(inputs["checkpoint"]),
                "model_inputs": {
                    "butd": True,
                    "butd_gt": False,
                    "butd_cls": False,
                },
                "filter_non_gt_boxes": False,
                "target_iou_policy": "root_only",
            },
        }

    monkeypatch.setattr(
        official, "preflight_frozen_inputs", synthetic_preflight,
        raising=False,
    )
    inputs["preflight_calls"] = calls
    inputs["preflight_function"] = synthetic_preflight
    return bindings


def _bind_sidecar(monkeypatch, path):
    monkeypatch.setattr(
        official, "OFFICIAL_SIDECAR_RECORD_PATH", Path(path), raising=False
    )
    monkeypatch.setattr(
        official, "OFFICIAL_SIDECAR_RECORD_SHA256", _sha256(path),
        raising=False,
    )


def _official_command():
    return official.build_authoritative_command()


def _official_environment():
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "/synthetic/repo",
    }


def _metric_lines(hits025=6000, hits050=5000):
    acc025 = "%.5f" % (hits025 / float(official.OFFICIAL_SAMPLE_COUNT))
    acc050 = "%.5f" % (hits050 / float(official.OFFICIAL_SAMPLE_COUNT))
    return (
        "last_ position alignment Acc0.25: Top-1: {}, Top-5: {}\n"
        "last_ position alignment Acc0.50: Top-1: {}, Top-5: {}\n"
    ).format(acc025, acc025, acc050, acc050)


def _valid_config(inputs, run_root, run_path):
    return {
        "eval": True,
        "eval_train": False,
        "dataset": ["scanrefer"],
        "test_dataset": "scanrefer",
        "exp": "epoch71_geometry_official",
        "log_dir": str(run_path),
        "batch_size": 12,
        "num_workers": 2,
        "print_freq": 100,
        "local_rank": 0,
        "num_target": 256,
        "num_decoder_layers": 6,
        "model": "MCLN",
        "butd": True,
        "butd_gt": False,
        "butd_cls": False,
        "use_color": True,
        "use_height": False,
        "use_multiview": False,
        "joint_det": True,
        "self_attend": True,
        "detect_intermediate": True,
        "use_soft_token_loss": True,
        "use_contrastive_align": True,
        "use_source_choice_selector": True,
        "source_choice_selector_sources": (
            "default,default_rank_blend_contrastive010"
        ),
        "source_choice_selector_hidden_dim": 288,
        "skip_missing_superpoints": True,
        "eval_use_rec_reranker_scores": True,
        "eval_use_rec_geometry_reranker_scores": True,
        "checkpoint_path": str(inputs["checkpoint"]),
        "data_root": "/root/autodl-tmp/DATA_ROOT/",
        "rec_reranker_checkpoint": str(inputs["parent_artifact"]),
        "rec_geometry_reranker_checkpoint": str(inputs["selected_artifact"]),
    }


def _launch_synthetic_official(monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    run_path = official.timestamp_run_root(run_root) / "1700000001"

    def fake_run(_command, **kwargs):
        run_path.mkdir(parents=True)
        (run_path / "config.json").write_text(
            json.dumps(_valid_config(inputs, run_root, run_path)),
            encoding="utf-8",
        )
        (run_path / "log.txt").write_text(
            "length of testing dataset: 9508\n" + _metric_lines(),
            encoding="utf-8",
        )
        kwargs["stdout"].write(
            ("synthetic launcher\n" + _metric_lines()).encode("utf-8")
        )
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    launched = official.run_official_launch()
    return inputs, run_root, run_path, launched


def test_recover_exact_hits_is_unique_for_the_frozen_sample_count():
    for hits in (0, 1, 4469, 5705, official.OFFICIAL_SAMPLE_COUNT):
        token = "%.5f" % (hits / float(official.OFFICIAL_SAMPLE_COUNT))
        assert official.recover_exact_hits(token) == hits

    with pytest.raises(ValueError, match="unique"):
        official.recover_exact_hits("0.123456")


def test_claim_path_is_fixed_by_authoritative_registry_and_goal(tmp_path):
    first_selection = tmp_path / "first" / "selection.json"
    second_selection = tmp_path / "second" / "selection-copy.json"
    first_selection.parent.mkdir()
    second_selection.parent.mkdir()
    first_selection.write_bytes(b'{"winner":"first"}\n')
    second_selection.write_bytes(b'{"winner":"different"}\n')

    expected = (
        Path(official.OFFICIAL_CLAIM_REGISTRY).expanduser().resolve()
        / (official.OFFICIAL_GOAL_NAME + ".claim.json")
    )
    assert official.claim_path_for() == expected


def test_unique_timestamp_run_is_the_only_new_numeric_directory(tmp_path):
    run_root = tmp_path / "official"
    timestamp_root = official.timestamp_run_root(run_root)
    timestamp_root.mkdir(parents=True)
    (timestamp_root / "1700000000").mkdir()
    before = official.snapshot_timestamp_runs(run_root)
    (timestamp_root / "notes").mkdir()
    expected = timestamp_root / "1700000001"
    expected.mkdir()

    assert official.discover_unique_timestamp_run(run_root, before) == expected

    (timestamp_root / "1700000002").mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        official.discover_unique_timestamp_run(run_root, before)


def test_low_level_one_shot_surfaces_take_no_caller_paths_or_command():
    for function in (
            official.build_authoritative_command,
            official.run_official_launch,
            official.seal_official_result,
            official.seal_sidecar_comparison):
        assert list(inspect.signature(function).parameters) == []


@pytest.mark.parametrize(
    "sha_constant",
    [
        "OFFICIAL_SELECTION_RECORD_SHA256",
        "OFFICIAL_SELECTED_ARTIFACT_SHA256",
        "OFFICIAL_PARENT_ARTIFACT_SHA256",
        "OFFICIAL_CHECKPOINT_SHA256",
    ],
)
def test_launcher_rejects_wrong_authoritative_sha_before_claim(
        monkeypatch, tmp_path, sha_constant):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    monkeypatch.setattr(official, sha_constant, "0" * 64)
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="authoritative.*SHA"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "validation_selection",
        "checkpoint",
        "butd",
        "butd_gt",
        "butd_cls",
        "filter_non_gt_boxes",
        "target_iou_policy",
    ],
)
def test_launcher_requires_full_no_gt_artifact_preflight_before_claim(
        monkeypatch, tmp_path, invalid):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    valid_preflight = inputs["preflight_function"]

    def invalid_preflight(*args, **kwargs):
        result = valid_preflight(*args, **kwargs)
        if invalid == "validation_selection":
            result["selection_uses_validation"] = True
        elif invalid == "checkpoint":
            result["geometry_artifact"]["checkpoint_sha256"] = "0" * 64
        elif invalid == "butd":
            result["geometry_artifact"]["model_inputs"][invalid] = False
        elif invalid in ("butd_gt", "butd_cls"):
            result["geometry_artifact"]["model_inputs"][invalid] = True
        elif invalid == "filter_non_gt_boxes":
            result["geometry_artifact"][invalid] = True
        else:
            result["geometry_artifact"][invalid] = "all_candidates"
        return result

    monkeypatch.setattr(official, "preflight_frozen_inputs", invalid_preflight)
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="authoritative preflight"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


def test_launcher_rejects_wrong_authoritative_interpreter_sha_before_claim(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    _bind_authoritative(monkeypatch, inputs, tmp_path / "official")
    monkeypatch.setattr(
        official, "OFFICIAL_PYTHON_TARGET_SHA256", "0" * 64
    )
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="interpreter.*SHA"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


@pytest.mark.parametrize(
    "substitution", ["parent_symlink", "retarget_live", "retarget_dangling"]
)
def test_launcher_rejects_authoritative_interpreter_substitution(
        monkeypatch, tmp_path, substitution):
    inputs = _launch_inputs(tmp_path)
    _bind_authoritative(monkeypatch, inputs, tmp_path / "official")
    logical = inputs["python_executable"]
    if substitution == "parent_symlink":
        real_environment = logical.parents[1]
        alias_environment = tmp_path / "python-alias"
        alias_environment.symlink_to(
            real_environment, target_is_directory=True
        )
        monkeypatch.setattr(
            official,
            "OFFICIAL_PYTHON_EXECUTABLE",
            alias_environment / "bin" / "python",
        )
    else:
        logical.unlink()
        target_name = (
            "python3.8" if substitution == "retarget_live" else "missing"
        )
        if substitution == "retarget_live":
            alternate = logical.parent / target_name
            alternate.write_bytes(inputs["python_target"].read_bytes())
            os.chmod(str(alternate), 0o755)
        logical.symlink_to(target_name)
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="interpreter.*symlink|link target"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


@pytest.mark.parametrize("phase", ["seal", "compare"])
@pytest.mark.parametrize("mutation", ["retarget", "bytes"])
def test_interpreter_evidence_is_rehashed_after_claim(
        monkeypatch, tmp_path, phase, mutation):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    if phase == "compare":
        official_record = official.seal_official_result()
        _publish_sidecar(
            monkeypatch,
            inputs["sidecar_record"],
            _sidecar_record(official_record),
        )
    if mutation == "retarget":
        alternate = inputs["python_target"].with_name("python3.8")
        alternate.write_bytes(inputs["python_target"].read_bytes())
        os.chmod(str(alternate), 0o755)
        inputs["python_executable"].unlink()
        inputs["python_executable"].symlink_to(alternate.name)
    else:
        inputs["python_target"].write_bytes(b"mutated python executable")

    with pytest.raises(
            ValueError, match="interpreter.*(changed|SHA)|link target"):
        if phase == "seal":
            official.seal_official_result()
        else:
            official.seal_sidecar_comparison()


@pytest.mark.parametrize("binding", ["interpreter", "native"])
def test_launch_detects_transient_substitute_then_restore(
        monkeypatch, tmp_path, binding):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(b"synthetic stdout\n")
        if binding == "interpreter":
            target = inputs["python_target"]
            original = target.read_bytes()
            target.write_bytes(b"temporary interpreter substitute")
            target.write_bytes(original)
        else:
            target = inputs["native_module"]
            backup = target.with_suffix(".backup")
            original = target.read_bytes()
            target.rename(backup)
            target.write_bytes(original)
            target.unlink()
            backup.rename(target)
        run = official.timestamp_run_root(run_root) / "1700000001"
        run.mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="identity|changed"):
        official.run_official_launch()

    receipt_path = official.receipt_path_for_claim(official.claim_path_for())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["success"] is False
    assert "identity" in receipt["launcher_error"]["message"]


def test_launch_detects_transient_interpreter_symlink_retarget_restore(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    logical = inputs["python_executable"]
    alternate = inputs["python_target"].with_name("python3.8")
    alternate.write_bytes(b"malicious alternate interpreter")
    os.chmod(str(alternate), 0o755)

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(b"synthetic stdout\n")
        logical.unlink()
        logical.symlink_to(alternate.name)
        logical.unlink()
        logical.symlink_to("python3.7")
        run = official.timestamp_run_root(run_root) / "1700000001"
        run.mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="interpreter.*identity"):
        official.run_official_launch()

    receipt_path = official.receipt_path_for_claim(official.claim_path_for())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["success"] is False
    assert "identity" in receipt["launcher_error"]["message"]


def test_sealer_detects_transient_frozen_input_restore(
        monkeypatch, tmp_path):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    selected = inputs["selected_artifact"]
    original = selected.read_bytes()
    selected.write_bytes(b"temporary selected substitute")
    selected.write_bytes(original)

    with pytest.raises(ValueError, match="identity|changed"):
        official.seal_official_result()


def test_launcher_uses_exclusive_stdout_and_selection_fixed_claim(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        assert official.claim_path_for().is_file()
        assert kwargs["stderr"] is official.subprocess.STDOUT
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == str(inputs["code_root"].resolve())
        kwargs["stdout"].write(b"synthetic official stdout\n")
        kwargs["stdout"].flush()
        run = official.timestamp_run_root(run_root) / "1700000001"
        run.mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)

    launched = official.run_official_launch()

    assert len(calls) == 1
    assert launched["run_path"] == str(
        official.timestamp_run_root(run_root) / "1700000001"
    )
    stdout = Path(launched["stdout_path"])
    assert stdout.read_bytes() == b"synthetic official stdout\n"
    assert launched["stdout_sha256"] == hashlib.sha256(
        stdout.read_bytes()
    ).hexdigest()
    assert Path(launched["claim_path"]).is_file()
    assert Path(launched["receipt_path"]).is_file()
    claim = json.loads(
        Path(launched["claim_path"]).read_text(encoding="utf-8")
    )
    assert claim["goal"] == official.OFFICIAL_GOAL_NAME
    assert claim["cwd"] == str(inputs["code_root"].resolve())
    assert claim["inputs"]["selection_record"]["path"] == str(
        inputs["selection_record"].resolve()
    )
    assert claim["inputs"]["selection_record"]["sha256"] == hashlib.sha256(
        inputs["selection_record"].read_bytes()
    ).hexdigest()
    receipt = json.loads(
        Path(launched["receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["cwd"] == str(inputs["code_root"].resolve())
    assert receipt["success"] is True
    assert receipt["status"] == "success"
    assert receipt["launcher_error"] is None
    assert receipt["returncode"] == 0
    assert receipt["stdout_status"] == "captured"

    alias_root = tmp_path / "different-official-root"
    copied_inputs = tmp_path / "copied-inputs"
    copied_inputs.mkdir()
    copied_selection = copied_inputs / "selection-copy.json"
    copied_selection.write_bytes(inputs["selection_record"].read_bytes())
    alias_selected = copied_inputs / inputs["selected_artifact"].name
    alias_selected.write_bytes(inputs["selected_artifact"].read_bytes())
    monkeypatch.setattr(
        official, "OFFICIAL_SELECTION_RECORD_PATH", copied_selection
    )
    monkeypatch.setattr(
        official, "OFFICIAL_SELECTED_ARTIFACT_PATH", alias_selected
    )
    monkeypatch.setattr(official, "OFFICIAL_RUN_ROOT", alias_root)
    with pytest.raises(FileExistsError, match="claim"):
        official.run_official_launch()
    assert len(calls) == 1


def test_authoritative_environment_removes_loader_injection(tmp_path):
    code_root = tmp_path / "code"
    environment = official._authoritative_environment(
        code_root,
        base_environment={
            "KEEP_ME": "ordinary",
            "PYTHONPATH": "/hostile/python",
            "PYTHONHOME": "/hostile/home",
            "LD_PRELOAD": "/hostile/preload.so",
            "LD_LIBRARY_PATH": "/hostile/lib",
        },
    )

    assert environment["KEEP_ME"] == "ordinary"
    assert environment["PYTHONPATH"] == os.pathsep.join((
        str(code_root.resolve()),
        str(code_root.resolve() / "pointnet2"),
    ))
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["OMP_NUM_THREADS"] == "1"
    for forbidden in ("PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        assert forbidden not in environment


def test_launcher_records_sanitized_loader_environment(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    monkeypatch.setenv("PYTHONPATH", "/hostile/python")
    monkeypatch.setenv("PYTHONHOME", "/hostile/home")
    monkeypatch.setenv("LD_PRELOAD", "/hostile/preload.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/hostile/lib")

    def fake_run(_command, **kwargs):
        environment = kwargs["env"]
        assert environment["PYTHONPATH"] == os.pathsep.join((
            str(inputs["code_root"].resolve()),
            str(inputs["code_root"].resolve() / "pointnet2"),
        ))
        for forbidden in ("PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
            assert forbidden not in environment
        run = official.timestamp_run_root(run_root) / "1700000001"
        run.mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    launched = official.run_official_launch()
    claim = json.loads(
        Path(launched["claim_path"]).read_text(encoding="utf-8")
    )
    assert claim["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join((
            str(inputs["code_root"].resolve()),
            str(inputs["code_root"].resolve() / "pointnet2"),
        )),
        "PYTHONHOME": None,
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
    }


def test_launcher_refuses_a_preexisting_stdout_without_running(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    stdout = official.official_stdout_path()
    stdout.parent.mkdir(parents=True)
    stdout.write_bytes(b"pre-existing")
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(FileExistsError, match="stdout"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


def test_launcher_requires_entrypoint_before_claim(monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    (inputs["code_root"] / "train_dist_mod.py").unlink()
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="entrypoint"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


@pytest.mark.parametrize(
    "failure",
    ["stdout_open", "subprocess_exception", "nonzero", "discovery"],
)
def test_every_post_claim_failure_writes_an_exact_failure_receipt(
        monkeypatch, tmp_path, failure):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    original_open = official.os.open

    if failure == "stdout_open":
        def failing_open(path, flags, *args, **kwargs):
            if (os.fspath(path) == str(official.official_stdout_path())
                    and flags & os.O_CREAT):
                raise OSError("synthetic stdout open failure")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(official.os, "open", failing_open)
    else:
        def fake_run(_command, **kwargs):
            kwargs["stdout"].write(b"synthetic partial stdout\n")
            kwargs["stdout"].flush()
            if failure == "subprocess_exception":
                raise OSError("synthetic subprocess failure")
            if failure == "nonzero":
                run = official.timestamp_run_root(run_root) / "1700000001"
                run.mkdir(parents=True)
                return SimpleNamespace(returncode=7)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(official.subprocess, "run", fake_run)

    expected_error = {
        "stdout_open": OSError,
        "subprocess_exception": OSError,
        "nonzero": subprocess.CalledProcessError,
        "discovery": ValueError,
    }[failure]
    with pytest.raises(expected_error):
        official.run_official_launch()

    receipt_path = official.receipt_path_for_claim(official.claim_path_for())
    assert receipt_path.is_file()
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt_bytes == official._canonical_json_bytes(receipt)
    assert receipt["success"] is False
    assert receipt["status"] == "failure"
    assert set(receipt["launcher_error"]) == {"type", "message"}
    assert receipt["launcher_error"]["message"]
    assert receipt["returncode"] == {
        "nonzero": 7,
        "discovery": 0,
    }.get(failure)
    assert receipt["stdout_path"] == str(official.official_stdout_path())
    if failure == "stdout_open":
        assert receipt["stdout_status"] == "not-created"
        assert receipt["stdout_sha256"] is None
    else:
        assert receipt["stdout_status"] == "captured"
        assert official._is_sha256(receipt["stdout_sha256"])
    if failure in ("stdout_open", "subprocess_exception", "discovery"):
        assert receipt["run_discovery_error"] is not None


@pytest.mark.parametrize("tamper", ["unlink", "replace", "symlink"])
def test_stdout_capture_is_bound_to_the_open_exclusive_fd(
        monkeypatch, tmp_path, tamper):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    captured = b"stdout bound to exclusive fd\n"

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(captured)
        kwargs["stdout"].flush()
        stdout_path = official.official_stdout_path()
        stdout_path.unlink()
        if tamper == "replace":
            stdout_path.write_bytes(b"replacement pathname bytes\n")
        elif tamper == "symlink":
            replacement = tmp_path / "replacement.log"
            replacement.write_bytes(b"symlink target bytes\n")
            stdout_path.symlink_to(replacement)
        run = official.timestamp_run_root(run_root) / "1700000001"
        run.mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    with pytest.raises((ValueError, OSError)):
        official.run_official_launch()

    receipt_path = official.receipt_path_for_claim(official.claim_path_for())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["success"] is False
    assert receipt["stdout_status"] == "identity-failed"
    assert receipt["stdout_sha256"] == hashlib.sha256(captured).hexdigest()
    assert receipt["launcher_error"]["type"]


def test_keyboard_interrupt_after_claim_writes_receipt_then_reraises(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)

    def interrupted_run(_command, **kwargs):
        kwargs["stdout"].write(b"stdout before interrupt\n")
        kwargs["stdout"].flush()
        raise KeyboardInterrupt("synthetic interrupt")

    monkeypatch.setattr(official.subprocess, "run", interrupted_run)
    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        official.run_official_launch()

    receipt_path = official.receipt_path_for_claim(official.claim_path_for())
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt_bytes == official._canonical_json_bytes(receipt)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert receipt["success"] is False
    assert receipt["launcher_error"] == {
        "type": "KeyboardInterrupt",
        "message": "synthetic interrupt",
    }
    assert receipt["returncode"] is None
    assert receipt["stdout_status"] == "captured"
    assert official._is_sha256(receipt["stdout_sha256"])


def test_discovery_keyboard_interrupt_preserves_receipt_and_exception(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    interrupt = KeyboardInterrupt("synthetic discovery interrupt")

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(b"stdout before discovery interrupt\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    def interrupted_discovery(_run_root, _preexisting):
        raise interrupt

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    monkeypatch.setattr(
        official, "discover_unique_timestamp_run", interrupted_discovery
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        official.run_official_launch()

    assert caught.value is interrupt
    receipt_path = official.receipt_path_for_claim(official.claim_path_for())
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt_bytes == official._canonical_json_bytes(receipt)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert receipt["success"] is False
    assert receipt["launcher_error"] == {
        "type": "KeyboardInterrupt",
        "message": "synthetic discovery interrupt",
    }
    assert receipt["returncode"] == 0
    assert receipt["run_discovery_error"] == (
        "KeyboardInterrupt: synthetic discovery interrupt"
    )


def test_receipt_is_read_back_exactly_before_launcher_returns(
        monkeypatch, tmp_path):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)

    def fake_run(_command, **kwargs):
        run = official.timestamp_run_root(run_root) / "1700000001"
        run.mkdir(parents=True)
        kwargs["stdout"].write(b"synthetic stdout\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    original_write = official._write_registry_json

    def tampering_write(name, payload, label):
        path = original_write(name, payload, label)
        if label == "official receipt":
            os.chmod(str(path), 0o600)
        return path

    monkeypatch.setattr(official, "_write_registry_json", tampering_write)
    with pytest.raises(ValueError, match="canonical|immutable"):
        official.run_official_launch()


def test_exclusive_writer_never_follows_a_final_symlink(tmp_path):
    target = tmp_path / "redirected-claim.json"
    link = tmp_path / "official.claim.json"
    link.symlink_to(target)

    with pytest.raises(FileExistsError, match="claim"):
        official._write_exclusive_file(link, b"claim\n", "official claim")

    assert link.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("kind", ["python_file", "directory"])
def test_snapshot_code_tree_rejects_runtime_symlinks_and_target_changes(
        tmp_path, kind):
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "train_dist_mod.py").write_text(
        "# entrypoint\n", encoding="utf-8"
    )
    external = tmp_path / "external"
    if kind == "python_file":
        external.write_text("VALUE = 1\n", encoding="utf-8")
        (code_root / "runtime.py").symlink_to(external)
    else:
        external.mkdir()
        (external / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
        (code_root / "modules").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        official.snapshot_code_tree(code_root)

    if kind == "python_file":
        external.write_text("VALUE = 2\n", encoding="utf-8")
    else:
        (external / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        official.snapshot_code_tree(code_root)


def test_snapshot_code_tree_rejects_symlinked_root_parent(tmp_path):
    real_parent = tmp_path / "real-parent"
    code_root = real_parent / "code"
    code_root.mkdir(parents=True)
    (code_root / "train_dist_mod.py").write_text(
        "# entrypoint\n", encoding="utf-8"
    )
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        official.snapshot_code_tree(alias_parent / "code")


def test_snapshot_code_tree_binds_local_native_modules(tmp_path):
    inputs = _launch_inputs(tmp_path)
    manifest = official.snapshot_code_tree(inputs["code_root"])

    native = inputs["native_module"]
    metadata = native.stat()
    relative = native.relative_to(inputs["code_root"]).as_posix()
    assert manifest["files"][relative] == {
        "sha256": _sha256(native),
        "size": metadata.st_size,
        "identity": [
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ],
    }


@pytest.mark.parametrize("component", ["root", "ancestor"])
def test_launcher_rejects_logical_code_root_symlinks_before_claim(
        monkeypatch, tmp_path, component):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    if component == "root":
        logical_root = tmp_path / "code-alias"
        logical_root.symlink_to(inputs["code_root"], target_is_directory=True)
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        real_code_root = real_parent / "code"
        inputs["code_root"].rename(real_code_root)
        alias_parent = tmp_path / "alias-parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        logical_root = alias_parent / "code"
        monkeypatch.setattr(
            official, "OFFICIAL_CODE_ROOT", logical_root
        )
        inputs["code_root"] = real_code_root
    if component == "root":
        monkeypatch.setattr(official, "OFFICIAL_CODE_ROOT", logical_root)
    monkeypatch.setattr(
        official.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="runtime code tree.*symlink"):
        official.run_official_launch()
    assert not official.claim_path_for().exists()


def test_symlinked_launcher_import_retains_and_rejects_logical_code_root(
        tmp_path):
    source = Path(official.__file__).resolve()
    logical_root = tmp_path / "logical-code-root"
    logical_scripts = logical_root / "scripts"
    logical_scripts.mkdir(parents=True)
    launcher = logical_scripts / source.name
    launcher.symlink_to(source)
    probe = inspect.cleandoc(
        """
        import importlib.util
        from pathlib import Path
        import sys

        launcher = Path(sys.argv[1])
        expected_root = Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location(
            "symlinked_official_launcher", str(launcher)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.OFFICIAL_CODE_ROOT == expected_root, (
            module.OFFICIAL_CODE_ROOT, expected_root
        )
        try:
            module.snapshot_code_tree(module.OFFICIAL_CODE_ROOT)
        except ValueError as error:
            assert str(error) == (
                "runtime code tree must not contain symlinks"
            )
        else:
            raise AssertionError("symlinked launcher code root was accepted")
        """
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            probe,
            str(launcher),
            str(logical_root),
        ],
        cwd=str(tmp_path),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("component", ["registry", "parent"])
@pytest.mark.parametrize("target_exists", [True, False])
def test_claim_registry_rejects_live_and_dangling_component_symlinks(
        monkeypatch, tmp_path, component, target_exists):
    configured = tmp_path / "configured"
    configured.mkdir()
    target = tmp_path / "target"
    if target_exists:
        target.mkdir()

    if component == "registry":
        configured.rmdir()
        configured.symlink_to(target, target_is_directory=True)
        registry = configured
    else:
        configured.rmdir()
        configured.symlink_to(target, target_is_directory=True)
        registry = configured / "claims"
        if target_exists:
            registry.mkdir()
    monkeypatch.setattr(official, "OFFICIAL_CLAIM_REGISTRY", registry)

    with pytest.raises(ValueError, match="registry.*symlink"):
        official.claim_path_for()


def test_metric_parser_requires_one_matching_line_in_log_and_stdout():
    lines = _metric_lines()
    parsed = official.parse_official_metrics(lines, lines)
    assert parsed == {
        "printed_acc025": "%.5f" % (
            6000 / float(official.OFFICIAL_SAMPLE_COUNT)
        ),
        "printed_acc050": "%.5f" % (
            5000 / float(official.OFFICIAL_SAMPLE_COUNT)
        ),
        "hits025": 6000,
        "hits050": 5000,
    }

    with pytest.raises(ValueError, match="exactly once"):
        official.parse_official_metrics(lines + lines, lines)
    conflicting = lines.replace("Top-1: 0.63105", "Top-1: 0.63106")
    with pytest.raises(ValueError, match="agree"):
        official.parse_official_metrics(lines, conflicting)


def test_sealer_publishes_one_exact_immutable_official_result(
        monkeypatch, tmp_path):
    inputs, run_root, run_path, launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )

    record = official.seal_official_result()

    output = official.official_result_path()
    assert output.is_file()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert set(record) == set(official.OFFICIAL_RESULT_FIELDS)
    assert record["schema"] == official.OFFICIAL_RESULT_SCHEMA
    assert record["sample_count"] == official.OFFICIAL_SAMPLE_COUNT
    assert record["hits025"] == 6000
    assert record["hits050"] == 5000
    assert record["inference_uses_ground_truth"] is False
    assert record["acceptance_gate_pass"] is True
    assert record["run"]["path"] == str(run_path)
    assert record["launch"]["claim_sha256"] == launched["claim_sha256"]
    assert record["launch"]["cwd"] == str(inputs["code_root"].resolve())
    assert record["files"]["stdout"]["sha256"] == launched[
        "stdout_sha256"
    ]

    with pytest.raises(FileExistsError, match="result"):
        official.seal_official_result()


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("butd", False),
        ("butd_gt", True),
        ("butd_cls", True),
        ("eval_use_rec_geometry_reranker_scores", False),
        ("batch_size", 8),
        ("local_rank", 1),
        ("print_freq", 50),
        ("data_root", "/wrong/data"),
        ("checkpoint_path", "/wrong/backbone.pth"),
    ],
)
def test_sealer_rejects_every_critical_config_mismatch(
        monkeypatch, tmp_path, field, bad_value):
    inputs, _run_root, run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    config_path = run_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = bad_value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="config"):
        official.seal_official_result()


@pytest.mark.parametrize("changed", ["selected", "code", "native"])
def test_sealer_rehashes_frozen_inputs_and_code(
        monkeypatch, tmp_path, changed):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    if changed == "selected":
        inputs["selected_artifact"].write_bytes(b"changed selected")
    elif changed == "code":
        (inputs["code_root"] / "runtime.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
    else:
        inputs["native_module"].write_bytes(b"changed native module")

    with pytest.raises(ValueError, match="changed"):
        official.seal_official_result()


def test_official_result_validator_requires_integer_schema_numbers(
        monkeypatch, tmp_path):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    record = official.seal_official_result()

    for field in ("version", "sample_count"):
        malformed = json.loads(json.dumps(record))
        malformed[field] = float(malformed[field])
        with pytest.raises(ValueError, match="exact schema"):
            official.validate_official_result(malformed)
    malformed = json.loads(json.dumps(record))
    malformed["launch"]["cwd"] = str(tmp_path / "other-code")
    with pytest.raises(ValueError, match="provenance"):
        official.validate_official_result(malformed)


@pytest.mark.parametrize(
    "malformed_value",
    [
        "run_path",
        "run_timestamp",
        "run_dataset",
        "command",
        "environment",
        "artifact_path",
        "code_root",
        "code_files",
    ],
)
def test_official_result_validator_rejects_nested_value_mismatches(
        monkeypatch, tmp_path, malformed_value):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    record = official.seal_official_result()
    malformed = json.loads(json.dumps(record))
    if malformed_value == "run_path":
        malformed["run"]["path"] = str(tmp_path / "other-run")
    elif malformed_value == "run_timestamp":
        malformed["run"]["timestamp"] += 1
    elif malformed_value == "run_dataset":
        malformed["run"]["dataset"] = "not-scanrefer"
    elif malformed_value == "command":
        malformed["launch"]["command"].append("--butd_gt")
    elif malformed_value == "environment":
        malformed["launch"]["environment"]["LD_PRELOAD"] = "/bad.so"
    elif malformed_value == "artifact_path":
        malformed["artifacts"]["selected_artifact"]["path"] = str(
            tmp_path / "other-selected.pth"
        )
    elif malformed_value == "code_root":
        malformed["code"]["root"] = str(tmp_path / "other-code")
        malformed["launch"]["cwd"] = malformed["code"]["root"]
    else:
        malformed["code"]["files"]["runtime.py"]["size"] = "invalid"

    with pytest.raises(ValueError):
        official.validate_official_result(malformed)


@pytest.mark.parametrize("tamper", ["bytes", "mode"])
def test_official_result_requires_exact_immutable_post_write_readback(
        monkeypatch, tmp_path, tamper):
    inputs, run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    original_write = official._write_exclusive_json

    def tampering_write(path, payload, label):
        published = original_write(path, payload, label)
        if label == "official result":
            os.chmod(str(published), 0o600)
            if tamper == "bytes":
                published.write_text(
                    json.dumps(payload, sort_keys=True), encoding="utf-8"
                )
                os.chmod(str(published), 0o444)
        return published

    monkeypatch.setattr(official, "_write_exclusive_json", tampering_write)
    with pytest.raises(ValueError, match="canonical|immutable"):
        official.seal_official_result()
    assert official.official_result_path().exists()


def _sidecar_record(official_record, hits025=None, hits050=None):
    hits025 = (
        official_record["hits025"] if hits025 is None else int(hits025)
    )
    hits050 = (
        official_record["hits050"] if hits050 is None else int(hits050)
    )
    artifacts = official_record["artifacts"]
    sample_count = official.OFFICIAL_SAMPLE_COUNT
    score_sha = "f" * 64
    return {
        "schema": frozen_sidecar.FROZEN_RECORD_SCHEMA,
        "version": frozen_sidecar.FROZEN_RECORD_VERSION,
        "sample_count": sample_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": hits025 / float(sample_count),
        "acc050": hits050 / float(sample_count),
        "parent_hits025": hits025,
        "parent_hits050": hits050,
        "parent_acc025": hits025 / float(sample_count),
        "parent_acc050": hits050 / float(sample_count),
        "fixes025": 0,
        "breaks025": 0,
        "fixes050": 0,
        "breaks050": 0,
        "geometry_weight": 0.35,
        "selected_artifact_sha256": artifacts[
            "selected_artifact"
        ]["sha256"],
        "parent_artifact_sha256": artifacts["parent_artifact"]["sha256"],
        "backbone_checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "selection_record_sha256": artifacts["selection_record"]["sha256"],
        "sidecar_evaluator_sha256": "d" * 64,
        "record_schema_sha256": (
            frozen_sidecar.FROZEN_RECORD_SCHEMA_SHA256
        ),
        "base_cache_content_sha256": "e" * 64,
        "base_cache_manifest_sha256": score_sha,
        "geometry_cache_content_sha256": "a" * 64,
        "geometry_cache_manifest_sha256": "b" * 64,
        "geometry_cache_immutable_metadata_sha256": "c" * 64,
        "val_parent_score_content_sha256": score_sha,
        "parent_inference_contract": {
            "schema": "rec-parent-inference-contract",
            "version": 1,
            "device_type": "cuda",
            "device_index": 0,
            "local_batch_size": 12,
            "world_size": 1,
            "row_order": "dataset-index-contiguous",
            "remainder_policy": "natural-remainder",
            "feature_source": "bound-base-cache-features",
            "dtype": "float32",
            "autocast": False,
            "allow_tf32": True,
            "eval": True,
            "no_grad": True,
            "score_builder": "normalized-query-reranker-rank-blend",
            "score_builder_version": 1,
            "canonical_query_tie_policy": (
                "score-desc-query-index-asc-v1"
            ),
            "content_digest_version": (
                "ordered-identity-raw-float32-sha256-v1"
            ),
            "row_count": sample_count,
            "score_content_sha256": score_sha,
        },
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
    }


def _publish_sidecar(monkeypatch, path, record):
    path.write_bytes(official._canonical_json_bytes(record))
    os.chmod(str(path), 0o444)
    _bind_sidecar(monkeypatch, path)
    return path


@pytest.mark.parametrize(
    "case",
    [
        "float_version",
        "float_sample_count",
        "boolean_hits",
        "reversed_hits",
        "inexact_accuracy",
        "invalid_sha",
        "extra_extension",
    ],
)
def test_sidecar_core_validator_is_strict_but_allows_extensions(case):
    hits025 = 6000
    hits050 = 5000
    record = {
        "schema": "rec-geometry-frozen-sidecar-evaluation",
        "version": 1,
        "sample_count": official.OFFICIAL_SAMPLE_COUNT,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": hits025 / float(official.OFFICIAL_SAMPLE_COUNT),
        "acc050": hits050 / float(official.OFFICIAL_SAMPLE_COUNT),
        "selected_artifact_sha256": "a" * 64,
        "parent_artifact_sha256": "b" * 64,
        "backbone_checkpoint_sha256": "c" * 64,
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
    }
    if case == "float_version":
        record["version"] = 1.0
    elif case == "float_sample_count":
        record["sample_count"] = float(official.OFFICIAL_SAMPLE_COUNT)
    elif case == "boolean_hits":
        record["hits025"] = True
    elif case == "reversed_hits":
        record["hits050"] = hits025 + 1
    elif case == "inexact_accuracy":
        record["acc025"] += 0.00001
    elif case == "extra_extension":
        record["future_extension"] = {"not": "allowed"}
    else:
        record["selected_artifact_sha256"] = "A" * 64

    with pytest.raises(ValueError, match="sidecar"):
        official._validate_sidecar_core(record)


@pytest.mark.parametrize("delta", [0, -1])
def test_comparator_immutably_seals_match_and_hit_mismatch(
        monkeypatch, tmp_path, delta):
    inputs, run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    official_record = official.seal_official_result()
    sidecar_path = inputs["sidecar_record"]
    sidecar = _sidecar_record(
        official_record,
        hits025=official_record["hits025"] + delta,
    )
    _publish_sidecar(monkeypatch, sidecar_path, sidecar)

    comparison = official.seal_sidecar_comparison()

    output = official.comparison_result_path()
    assert output.is_file()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert json.loads(output.read_text(encoding="utf-8")) == comparison
    assert set(comparison) == set(official.COMPARISON_RESULT_FIELDS)
    assert comparison["delta025"] == -delta
    assert comparison["delta050"] == 0
    assert comparison["hit_counts_match"] is (delta == 0)
    assert comparison["acceptance"] is (delta == 0)
    with pytest.raises(FileExistsError, match="comparison"):
        official.seal_sidecar_comparison()


def test_comparison_validator_derives_schema_numbers_and_gate(
        monkeypatch, tmp_path):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    official_record = official.seal_official_result()
    sidecar_path = inputs["sidecar_record"]
    _publish_sidecar(
        monkeypatch,
        sidecar_path,
        _sidecar_record(official_record),
    )
    comparison = official.seal_sidecar_comparison()

    for field in ("version", "sample_count"):
        malformed = json.loads(json.dumps(comparison))
        malformed[field] = float(malformed[field])
        with pytest.raises(ValueError, match="exact schema"):
            official.validate_comparison_result(malformed)
    malformed = json.loads(json.dumps(comparison))
    malformed["official_acceptance_gate_pass"] = False
    malformed["acceptance"] = False
    with pytest.raises(ValueError, match="official gate"):
        official.validate_comparison_result(malformed)


@pytest.mark.parametrize(
    "forgery", ["official_path", "sidecar_path", "artifact_shas"]
)
def test_comparison_validator_rejects_forged_nested_provenance(
        monkeypatch, tmp_path, forgery):
    inputs, _run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    official_record = official.seal_official_result()
    _publish_sidecar(
        monkeypatch,
        inputs["sidecar_record"],
        _sidecar_record(official_record),
    )
    comparison = official.seal_sidecar_comparison()
    malformed = json.loads(json.dumps(comparison))
    if forgery == "official_path":
        malformed["files"]["official_result"]["path"] = str(
            tmp_path / "forged-official-result.json"
        )
    elif forgery == "sidecar_path":
        malformed["files"]["sidecar_record"]["path"] = str(
            tmp_path / "forged-sidecar-record.json"
        )
    else:
        malformed["artifacts"] = {
            name: "0" * 64 for name in malformed["artifacts"]
        }

    with pytest.raises(ValueError, match="path|artifact|provenance"):
        official.validate_comparison_result(malformed)


def test_comparator_rejects_invalid_sidecar_provenance_without_publishing(
        monkeypatch, tmp_path):
    inputs, run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    official_record = official.seal_official_result()
    sidecar = _sidecar_record(official_record)
    sidecar["inference_uses_ground_truth"] = True
    sidecar_path = inputs["sidecar_record"]
    _publish_sidecar(monkeypatch, sidecar_path, sidecar)

    with pytest.raises(ValueError, match="sidecar"):
        official.seal_sidecar_comparison()
    assert not official.comparison_result_path().exists()


def test_comparator_requires_the_official_selection_record_sha(
        monkeypatch, tmp_path):
    inputs, run_root, _run_path, _launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    official_record = official.seal_official_result()
    sidecar = _sidecar_record(official_record)
    sidecar["selection_record_sha256"] = "0" * 64
    sidecar_path = inputs["sidecar_record"]
    _publish_sidecar(monkeypatch, sidecar_path, sidecar)

    with pytest.raises(ValueError, match="sidecar"):
        official.seal_sidecar_comparison()
    assert not official.comparison_result_path().exists()


def _prepare_comparison(monkeypatch, tmp_path):
    inputs, _run_root, run_path, launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    official_record = official.seal_official_result()
    sidecar = _sidecar_record(official_record)
    _publish_sidecar(
        monkeypatch, inputs["sidecar_record"], sidecar
    )
    return inputs, run_path, launched, official_record, sidecar


@pytest.mark.parametrize("source", ["official", "sidecar"])
@pytest.mark.parametrize("tamper", ["mode", "bytes"])
def test_comparator_requires_canonical_immutable_source_records(
        monkeypatch, tmp_path, source, tamper):
    inputs, _run_path, _launched, official_record, sidecar = (
        _prepare_comparison(monkeypatch, tmp_path)
    )
    path = (
        official.official_result_path()
        if source == "official" else inputs["sidecar_record"]
    )
    record = official_record if source == "official" else sidecar
    os.chmod(str(path), 0o600)
    if tamper == "bytes":
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        os.chmod(str(path), 0o444)

    with pytest.raises(ValueError, match="canonical|immutable"):
        official.seal_sidecar_comparison()
    assert not official.comparison_result_path().exists()


@pytest.mark.parametrize("source", ["claim", "receipt"])
def test_sealer_requires_immutable_registry_evidence(
        monkeypatch, tmp_path, source):
    _inputs, _run_root, _run_path, launched = _launch_synthetic_official(
        monkeypatch, tmp_path
    )
    path = Path(launched[source + "_path"])
    os.chmod(str(path), 0o600)

    with pytest.raises(ValueError, match="canonical|immutable"):
        official.seal_official_result()


@pytest.mark.parametrize(
    "changed", [
        "selected", "code", "native", "config", "log", "stdout"
    ]
)
def test_comparator_rehashes_every_frozen_official_input(
        monkeypatch, tmp_path, changed):
    inputs, run_path, launched, _official_record, _sidecar = (
        _prepare_comparison(monkeypatch, tmp_path)
    )
    path = {
        "selected": inputs["selected_artifact"],
        "code": inputs["code_root"] / "runtime.py",
        "native": inputs["native_module"],
        "config": run_path / "config.json",
        "log": run_path / "log.txt",
        "stdout": Path(launched["stdout_path"]),
    }[changed]
    os.chmod(str(path), 0o600)
    with path.open("ab") as handle:
        handle.write(b"\nchanged after official seal\n")

    with pytest.raises(ValueError, match="changed|evidence|config"):
        official.seal_sidecar_comparison()
    assert not official.comparison_result_path().exists()


@pytest.mark.parametrize("source", ["official_result", "receipt"])
def test_comparator_rebinds_published_official_provenance(
        monkeypatch, tmp_path, source):
    _inputs, _run_path, launched, official_record, _sidecar = (
        _prepare_comparison(monkeypatch, tmp_path)
    )
    if source == "official_result":
        path = official.official_result_path()
        record = json.loads(json.dumps(official_record))
        record["launch"]["environment"]["OMP_NUM_THREADS"] = "9"
    else:
        path = Path(launched["receipt_path"])
        record = json.loads(path.read_text(encoding="utf-8"))
        record["cwd"] = str(tmp_path / "wrong-code")
    os.chmod(str(path), 0o600)
    path.write_bytes(official._canonical_json_bytes(record))
    os.chmod(str(path), 0o444)

    with pytest.raises(
            ValueError, match="provenance|receipt|claim|environment"):
        official.seal_sidecar_comparison()
    assert not official.comparison_result_path().exists()


def test_comparator_requires_fixed_sidecar_sha(monkeypatch, tmp_path):
    _prepare_comparison(monkeypatch, tmp_path)
    monkeypatch.setattr(
        official, "OFFICIAL_SIDECAR_RECORD_SHA256", "0" * 64
    )

    with pytest.raises(ValueError, match="sidecar.*SHA"):
        official.seal_sidecar_comparison()
    assert not official.comparison_result_path().exists()


def test_cli_script_help_runs_without_preconfigured_pythonpath(tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(official.__file__).resolve()), "--help"],
        cwd=str(tmp_path),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "{launch,seal,compare}" in completed.stdout
    for forbidden in ("--overwrite", "--output", "--hits025", "--run-path"):
        assert forbidden not in completed.stdout


def test_cli_launch_constructs_the_authoritative_command(
        monkeypatch, tmp_path, capsys):
    inputs = _launch_inputs(tmp_path)
    run_root = tmp_path / "official"
    _bind_authoritative(monkeypatch, inputs, run_root)
    calls = []

    def fake_launch():
        calls.append(True)
        return {"schema": "synthetic-launch-receipt"}

    monkeypatch.setattr(official, "run_official_launch", fake_launch)
    assert official.main(["launch"]) == 0

    assert calls == [True]
    command = official.build_authoritative_command()
    assert str(inputs["selected_artifact"].resolve()) in command
    assert str(inputs["parent_artifact"].resolve()) in command
    assert str(inputs["checkpoint"].resolve()) in command
    assert str(run_root.resolve()) in command
    assert json.loads(capsys.readouterr().out) == {
        "schema": "synthetic-launch-receipt"
    }


def test_cli_seal_and_compare_use_only_fixed_inputs(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        official,
        "seal_official_result",
        lambda: calls.append("seal") or {"sealed": True},
    )
    monkeypatch.setattr(
        official,
        "seal_sidecar_comparison",
        lambda: calls.append("compare") or {"compared": True},
    )

    assert official.main(["seal"]) == 0
    assert official.main(["compare"]) == 0
    assert calls == ["seal", "compare"]
    rendered = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rendered == [{"sealed": True}, {"compared": True}]


@pytest.mark.parametrize(
    "extra",
    [
        ["--overwrite"],
        ["--output", "other.json"],
        ["--hits025", "6000"],
        ["--run-path", "1700000001"],
        ["--selection-record", "selection.json"],
        ["--selected-artifact", "selected.pth"],
        ["--parent-artifact", "parent.pth"],
        ["--checkpoint", "backbone.pth"],
        ["--run-root", "official"],
        ["--code-root", "code"],
        ["--", "python", "train_dist_mod.py"],
    ],
)
def test_cli_launch_exposes_no_claim_or_metric_escape_hatches(
        tmp_path, extra):
    with pytest.raises(SystemExit):
        official.main(["launch"] + extra)
