import json
import hashlib
import os
import fcntl
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState, create_trial

from scripts.tuning.mcln_optuna_contract import (
    EXPECTED_CALIBRATION_COUNT,
    METRICS_SCHEMA,
    assess_trial_metrics,
    seed_presets,
    suggest_trial_params,
)
from scripts.tuning import optuna_mcln_complete_retrain as orchestrator


def _args(tmp_path):
    output_root = tmp_path / "study"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    repo_root.mkdir()
    data_root.mkdir()
    python_bin = tmp_path / "venv" / "bin" / "python"
    provenance_path = output_root / "provenance" / "run_manifest.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path = provenance_path.parent / "environment.json"
    environment_path.write_text(json.dumps({
        "python_executable": str(python_bin.resolve()),
        "python_version": "3.7.test",
    }, sort_keys=True))
    inputs_path = provenance_path.parent / "inputs.json"
    inputs = {
        "schema": "mcln-retrain-inputs-v1",
        "data_root": str(data_root.resolve()),
        "base_checkpoint": {"sha256": "a" * 64},
        "pointnet_checkpoint": {"sha256": "c" * 64},
    }
    inputs_path.write_text(json.dumps(inputs, sort_keys=True))
    provenance_path.write_text(json.dumps({
        "schema": "mcln-retrain-run-provenance-v1",
        "repo_root": str(repo_root.resolve()),
        "output_root": str(output_root.resolve()),
        "data_root": str(data_root.resolve()),
        "base_checkpoint": {"sha256": "a" * 64},
        "pointnet_checkpoint": {"sha256": "c" * 64},
        "source_snapshot": {"manifest_sha256": "b" * 64},
        "environment": str(environment_path.resolve()),
        "environment_sha256": hashlib.sha256(
            environment_path.read_bytes()
        ).hexdigest(),
        "inputs": str(inputs_path.resolve()),
        "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
    }, sort_keys=True))
    return SimpleNamespace(
        python_bin=str(python_bin),
        repo_root=str(repo_root),
        output_root=str(output_root),
        data_root=str(data_root),
        pp_checkpoint=str(data_root / "checkpoints" / "pointnet.pth"),
        base_checkpoint=str(repo_root / "pretained model" / "ckpt_epoch_54.pth"),
        base_sha256="a" * 64,
        gpu=0,
        master_port_base=29600,
        target_successful_trials=20,
        study_name="mcln-complete-test",
        storage="sqlite:///{}".format(output_root / "optuna.db"),
        provenance_manifest=str(provenance_path),
        max_process_attempts=60,
    )


def _command_value(command, flag):
    index = command.index(flag)
    return command[index + 1]


APPROVED_PARAMS = seed_presets()[1]


def _optimizer_groups(params=APPROVED_PARAMS):
    return [
        {
            "name": "decoder",
            "initial_lr": params["decoder_lr"],
            "parameter_names": ["decoder.weight"],
        },
        {
            "name": "backbone",
            "initial_lr": params["decoder_lr"] * 10.0,
            "parameter_names": ["backbone_net.weight"],
        },
        {
            "name": "mask_head",
            "initial_lr": params["decoder_lr"]
            * params["mask_head_lr_multiplier"],
            "parameter_names": ["x_mask.weight"],
        },
        {
            "name": "selector",
            "initial_lr": params["selector_lr"],
            "parameter_names": ["source_choice_selector.weight"],
        },
    ]


def _loss_receipt(total_loss=1.0):
    return {
        "schema": "mcln-train-loss-epoch-v1",
        "batch_count": 7,
        "loss_means": {
            "mask_loss": 0.5,
            "total_loss": total_loss,
        },
    }


def test_trial_command_fixes_training_contract_and_all_parameters(tmp_path):
    args = _args(tmp_path)

    command = orchestrator.command_for_trial(
        args, APPROVED_PARAMS, trial_number=7
    )

    assert command[:5] == [
        args.python_bin,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        "1",
    ]
    assert _command_value(command, "--mode") == "trial"
    assert _command_value(command, "--batch_size") == "18"
    assert _command_value(command, "--num_workers") == "4"
    assert _command_value(command, "--source_choice_selector_sources") == (
        "default,default_rank_blend_contrastive010"
    )
    assert _command_value(command, "--mask_loss_scale") == str(
        APPROVED_PARAMS["mask_loss_scale"]
    )
    assert _command_value(command, "--consistency_loss_scale") == str(
        APPROVED_PARAMS["consistency_loss_scale"]
    )
    assert _command_value(command, "--source_choice_selector_lr") == str(
        APPROVED_PARAMS["selector_lr"]
    )
    assert _command_value(command, "--decoder-lr") == str(
        APPROVED_PARAMS["decoder_lr"]
    )
    assert "--reduce_lr" in command
    assert "--eval" not in command
    assert _command_value(command, "--max_epoch") == "56"


def test_baseline_command_has_no_trial_parameters_and_uses_same_anchor(
        tmp_path):
    args = _args(tmp_path)

    command = orchestrator.command_for_baseline(args)

    assert _command_value(command, "--mode") == "baseline"
    assert _command_value(command, "--base-checkpoint") == args.base_checkpoint
    assert _command_value(command, "--expected-base-sha256") == args.base_sha256
    assert "--decoder-lr" not in command
    assert "--checkpoint-output" not in command


def test_study_contract_and_baseline_metrics_are_stable_artifacts(tmp_path):
    args = _args(tmp_path)
    baseline = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
    }

    contract = orchestrator.write_study_contract(args)
    baseline.update({
        "study_binding": orchestrator.study_receipt_binding(contract),
        "trial_params": None,
        "optimizer_groups": [],
    })
    metrics = orchestrator.publish_baseline_metrics(args, baseline)

    assert contract["target_successful_trials"] == 20
    assert contract["trial_epochs"] == [55, 56]
    assert contract["sampler"] == {
        "name": "TPESampler",
        "seed": 0,
        "n_startup_trials": 5,
    }
    assert contract["official_validation_used_for_tuning"] is False
    assert contract["base_checkpoint_sha256"] == args.base_sha256
    assert contract["seed_presets"] == list(seed_presets())
    assert json.loads(
        (Path(args.output_root) / "study_contract.json").read_text()
    ) == contract
    assert metrics == _metrics()
    artifact = json.loads(
        (Path(args.output_root) / "baseline_metrics.json").read_text()
    )
    assert artifact["metrics"] == _metrics()
    assert artifact["study_binding"] == orchestrator.study_receipt_binding(
        contract
    )


def test_existing_baseline_recovery_rejects_training_loss_receipts(tmp_path):
    args = _args(tmp_path)
    contract = orchestrator.write_study_contract(args)
    receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "losses": {"epoch_54": _loss_receipt()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": orchestrator.study_receipt_binding(contract),
    }
    receipt_path = Path(args.output_root) / "baseline" / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))

    with pytest.raises(ValueError, match="training loss"):
        orchestrator._ensure_baseline(args)


def test_baseline_publication_rejects_training_loss_receipts(tmp_path):
    args = _args(tmp_path)
    contract = orchestrator.write_study_contract(args)
    receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "losses": {"epoch_54": _loss_receipt()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": orchestrator.study_receipt_binding(contract),
    }

    with pytest.raises(ValueError, match="training loss"):
        orchestrator.publish_baseline_metrics(args, receipt)


def test_contract_and_receipts_bind_full_provenance_and_storage(tmp_path):
    args = _args(tmp_path)

    contract = orchestrator.write_study_contract(args)
    binding = orchestrator.study_receipt_binding(contract)

    expected = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "python_bin": str(Path(args.python_bin).resolve()),
        "run_manifest_sha256": orchestrator.file_sha256(
            args.provenance_manifest
        ),
        "environment_sha256": json.loads(
            Path(args.provenance_manifest).read_text()
        )["environment_sha256"],
        "inputs_sha256": json.loads(
            Path(args.provenance_manifest).read_text()
        )["inputs_sha256"],
        "study_name": args.study_name,
        "storage_identity": str(
            (Path(args.output_root) / "optuna.db").resolve()
        ),
    }
    for name, value in expected.items():
        assert contract[name] == value
        assert binding[name] == value


@pytest.mark.parametrize(
    "mutation",
    [
        "environment_hash",
        "inputs_hash",
        "repo_root",
        "data_root",
        "python_bin",
    ],
)
def test_provenance_binding_rejects_referenced_hash_and_path_drift(
        tmp_path, mutation):
    args = _args(tmp_path)
    manifest_path = Path(args.provenance_manifest)
    manifest = json.loads(manifest_path.read_text())
    if mutation == "environment_hash":
        Path(manifest["environment"]).write_text('{"tampered": true}\n')
    elif mutation == "inputs_hash":
        Path(manifest["inputs"]).write_text('{"tampered": true}\n')
    elif mutation == "repo_root":
        manifest["repo_root"] = str((tmp_path / "other-repo").resolve())
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    elif mutation == "data_root":
        manifest["data_root"] = str((tmp_path / "other-data").resolve())
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    else:
        environment_path = Path(manifest["environment"])
        environment = json.loads(environment_path.read_text())
        environment["python_executable"] = str(
            (tmp_path / "other-python").resolve()
        )
        environment_path.write_text(json.dumps(environment, sort_keys=True))
        manifest["environment_sha256"] = orchestrator.file_sha256(
            environment_path
        )
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(ValueError, match="provenance|environment|inputs"):
        orchestrator.write_study_contract(args)


def test_study_contract_is_canonical_and_never_replaced_on_mismatch(
        tmp_path):
    args = _args(tmp_path)

    first = orchestrator.write_study_contract(args)
    path = Path(args.output_root) / "study_contract.json"
    original = path.read_bytes()

    assert first["contract_digest"] == orchestrator.canonical_json_sha256(
        orchestrator.canonical_study_contract_payload(first)
    )

    args.target_successful_trials = 19
    with pytest.raises(ValueError, match="study contract"):
        orchestrator.write_study_contract(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "field", ["base_checkpoint", "pp_checkpoint", "provenance_manifest"]
)
def test_study_contract_rejects_relevant_path_changes_with_same_hashes(
        tmp_path, field):
    args = _args(tmp_path)
    orchestrator.write_study_contract(args)
    contract_path = Path(args.output_root) / "study_contract.json"
    original = contract_path.read_bytes()

    if field == "provenance_manifest":
        alternate = tmp_path / "alternate" / "run_manifest.json"
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes(Path(args.provenance_manifest).read_bytes())
        setattr(args, field, str(alternate))
    else:
        setattr(args, field, str(tmp_path / "alternate" / field))

    with pytest.raises(ValueError, match="study contract"):
        orchestrator.write_study_contract(args)
    assert contract_path.read_bytes() == original


def test_study_contract_binds_canonical_sqlite_storage_identity(tmp_path):
    args = _args(tmp_path)
    contract = orchestrator.write_study_contract(args)

    assert contract["storage_identity"] == str(
        (Path(args.output_root) / "optuna.db").resolve()
    )

    args.storage = "sqlite:///{}".format(tmp_path / "other.db")
    with pytest.raises(ValueError, match="study contract"):
        orchestrator.write_study_contract(args)


def test_existing_baseline_with_different_study_binding_fails_closed(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    orchestrator.write_study_contract(args)
    receipt_path = Path(args.output_root) / "baseline" / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
        "study_binding": {"study_contract_digest": "0" * 64},
    }))
    monkeypatch.setattr(
        orchestrator,
        "_run_command",
        lambda *_args, **_kwargs: pytest.fail(
            "a mismatched baseline must not be overwritten"
        ),
    )

    with pytest.raises(ValueError, match="baseline.*binding|binding.*baseline"):
        orchestrator._ensure_baseline(args)


def test_remaining_successful_trials_ignores_failed_running_and_bad_receipts():
    study = optuna.create_study(direction="maximize")
    study.add_trial(create_trial(
        state=TrialState.COMPLETE, value=1.0,
        user_attrs={"receipt": "valid.json"},
    ))
    study.add_trial(create_trial(
        state=TrialState.COMPLETE, value=0.5,
        user_attrs={"receipt": "invalid.json"},
    ))
    study.add_trial(create_trial(state=TrialState.FAIL))
    study.add_trial(create_trial(state=TrialState.RUNNING))

    remaining = orchestrator.remaining_successful_trials(
        study,
        20,
        receipt_is_valid=lambda trial: (
            trial.user_attrs.get("receipt") == "valid.json"
        ),
    )

    assert remaining == 19


def test_remaining_successful_trials_never_returns_negative():
    study = optuna.create_study(direction="maximize")
    for index in range(3):
        study.add_trial(create_trial(
            state=TrialState.COMPLETE,
            value=float(index),
            user_attrs={"receipt": "{}.json".format(index)},
        ))

    assert orchestrator.remaining_successful_trials(
        study, 2, receipt_is_valid=lambda _trial: True
    ) == 0


def test_seed_presets_are_enqueued_only_for_a_new_study():
    study = optuna.create_study(direction="maximize")

    assert orchestrator.enqueue_seed_presets_if_new(study) == 3
    assert len(study.trials) == 3
    assert [trial.state for trial in study.trials] == [
        TrialState.WAITING, TrialState.WAITING, TrialState.WAITING
    ]
    assert [trial.system_attrs["fixed_params"] for trial in study.trials] == (
        list(seed_presets())
    )
    assert orchestrator.enqueue_seed_presets_if_new(study) == 0
    assert len(study.trials) == 3


def _metrics(
        fixed025=2200,
        fixed050=1700,
        learned025=2210,
        learned050=1710,
        mask025=2250,
        mask050=1850,
        iou_sum=1600.0):
    return {
        "schema": METRICS_SCHEMA,
        "sample_count": EXPECTED_CALIBRATION_COUNT,
        "position": {
            "fixed_default": {"hits025": fixed025, "hits050": fixed050},
            "learned_selector": {
                "hits025": learned025,
                "hits050": learned050,
            },
        },
        "mask": {
            "hits025": mask025,
            "hits050": mask050,
            "iou_sum": iou_sum,
            "miou": iou_sum / float(EXPECTED_CALIBRATION_COUNT),
        },
    }


def _candidate(tmp_path, trial_number=4, feasible=True):
    baseline = _metrics()
    trial = _metrics(
        fixed025=2201,
        fixed050=1701,
        learned025=2220,
        learned050=1720,
        mask025=2260,
        mask050=1860 if feasible else 1849,
        iou_sum=1610.0,
    )
    assessment = assess_trial_metrics(baseline, trial)
    checkpoint = tmp_path / "trial_{}.pth".format(trial_number)
    checkpoint.write_bytes(b"short trial checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "trial_number": trial_number,
        "metrics": trial,
        "feasible": assessment["feasible"],
        "objective": assessment["objective"],
        "deltas": assessment["deltas"],
        "constraint_failures": assessment["constraint_failures"],
        "trial_params": dict(APPROVED_PARAMS),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "receipt": "trials/trial_{}/receipt.json".format(trial_number),
    }


class _FakeProcess:
    def __init__(self, pid=4242):
        self.pid = pid


def test_no_feasible_trial_publishes_receipt_and_never_dispatches_long(
        tmp_path):
    args = _args(tmp_path)
    popen_calls = []

    result = orchestrator.publish_final_selection(
        args,
        [_candidate(tmp_path, feasible=False)],
        popen_factory=lambda *a, **k: popen_calls.append((a, k)),
    )

    assert result["selection_status"] == "no_feasible_trial"
    assert popen_calls == []
    best_path = Path(args.output_root) / "best.json"
    assert json.loads(best_path.read_text())["selection_status"] == (
        "no_feasible_trial"
    )
    assert not (Path(args.output_root) / "long_run" / "dispatch.json").exists()


def test_feasible_best_is_hardlinked_and_dispatches_long_once(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    candidate = _candidate(tmp_path, trial_number=4, feasible=True)
    calls = []
    monkeypatch.setattr(
        orchestrator, "_process_start_time", lambda _pid: "fake-start"
    )

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return _FakeProcess()

    result = orchestrator.publish_final_selection(
        args, [candidate], popen_factory=fake_popen
    )

    assert result["selection_status"] == "feasible_best"
    assert result["trial_number"] == 4
    stable = Path(args.output_root) / "checkpoints" / "optuna_best_trial.pth"
    assert stable.exists()
    assert os.stat(stable).st_ino == os.stat(candidate["checkpoint"]).st_ino
    assert result["checkpoint_sha256"] == hashlib.sha256(
        stable.read_bytes()
    ).hexdigest()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == args.python_bin
    assert command[1].endswith("scripts/tuning/train_mcln_complete_long.py")
    assert _command_value(command, "--best-json") == str(
        Path(args.output_root) / "best.json"
    )
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == args.repo_root
    dispatch = json.loads(
        (Path(args.output_root) / "long_run" / "dispatch.json").read_text()
    )
    assert dispatch["pid"] == 4242

    second = orchestrator.publish_final_selection(
        args, [candidate], popen_factory=fake_popen
    )
    assert second["long_dispatch"]["already_dispatched"] is True
    assert len(calls) == 1


def _published_best_for_dispatch(tmp_path):
    args = _args(tmp_path)
    candidate = _candidate(tmp_path, trial_number=4, feasible=True)
    orchestrator.publish_final_selection(
        args, [candidate], dispatch_long=False
    )
    best_path = Path(args.output_root) / "best.json"
    return args, best_path, json.loads(best_path.read_text())


def test_dispatch_publishes_starting_state_before_popen(
        tmp_path, monkeypatch):
    args, best_path, best = _published_best_for_dispatch(tmp_path)
    dispatch_path = Path(args.output_root) / "long_run" / "dispatch.json"
    observed = {}
    monkeypatch.setattr(
        orchestrator, "_process_start_time", lambda _pid: "start-4242"
    )

    def fake_popen(command, **kwargs):
        observed["starting"] = json.loads(dispatch_path.read_text())
        observed["command"] = command
        return _FakeProcess()

    result = orchestrator._dispatch_long_run(args, best_path, fake_popen)

    starting = observed["starting"]
    assert starting["schema"] == orchestrator.LONG_DISPATCH_SCHEMA
    assert starting["status"] == "starting"
    assert starting["attempt"] == 1
    assert starting["binding"] == orchestrator.long_dispatch_binding(best)
    assert _command_value(observed["command"], "--dispatch-token") == (
        starting["token"]
    )
    assert _command_value(observed["command"], "--startup-ack") == str(
        Path(args.output_root) / "long_run" / "startup_ack.json"
    )
    assert _command_value(observed["command"], "--completion-receipt") == str(
        Path(args.output_root) / "long_run" / "completion.json"
    )
    assert result["status"] == "running"
    assert result["process_start_time"] == "start-4242"


def test_startup_ack_recovers_crash_before_final_dispatch_write(
        tmp_path, monkeypatch):
    args, best_path, _best = _published_best_for_dispatch(tmp_path)
    root = Path(args.output_root) / "long_run"
    dispatch_path = root / "dispatch.json"
    startup_state = {}
    actual_start = orchestrator._process_start_time(os.getpid())

    def fake_popen(command, **kwargs):
        startup_state.update(json.loads(dispatch_path.read_text()))
        ack_path = Path(_command_value(command, "--startup-ack"))
        ack_path.write_text(json.dumps({
            "schema": orchestrator.LONG_STARTUP_ACK_SCHEMA,
            "token": startup_state["token"],
            "binding": startup_state["binding"],
            "pid": os.getpid(),
            "process_start_time": actual_start,
        }))
        return _FakeProcess(pid=os.getpid())

    orchestrator._dispatch_long_run(args, best_path, fake_popen)
    dispatch_path.write_text(json.dumps(startup_state))

    result = orchestrator._dispatch_long_run(
        args,
        best_path,
        lambda *args, **kwargs: pytest.fail("must not duplicate live ack"),
    )

    assert result["already_dispatched"] is True
    assert result["status"] == "running"
    assert result["pid"] == os.getpid()
    assert result["process_start_time"] == actual_start


def test_dead_dispatched_process_relaunches_with_incremented_attempt(
        tmp_path, monkeypatch):
    args, best_path, _best = _published_best_for_dispatch(tmp_path)
    pids = iter((81001, 81002))
    calls = []
    monkeypatch.setattr(
        orchestrator, "_process_start_time", lambda _pid: None
    )

    def fake_popen(command, **kwargs):
        calls.append(command)
        return _FakeProcess(pid=next(pids))

    first = orchestrator._dispatch_long_run(args, best_path, fake_popen)
    second = orchestrator._dispatch_long_run(args, best_path, fake_popen)

    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert second["already_dispatched"] is False
    assert len(calls) == 2
    assert first["token"] != second["token"]


def test_dead_legacy_launched_state_is_recoverable(tmp_path, monkeypatch):
    args, best_path, _best = _published_best_for_dispatch(tmp_path)
    root = Path(args.output_root) / "long_run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dispatch.json").write_text(json.dumps({
        "status": "launched",
        "pid": 99999999,
        "command": ["legacy"],
    }))
    monkeypatch.setattr(
        orchestrator, "_process_start_time", lambda _pid: None
    )

    result = orchestrator._dispatch_long_run(
        args, best_path, lambda *args, **kwargs: _FakeProcess(pid=81003)
    )

    assert result["status"] == "failed"
    assert result["attempt"] == 2
    assert result["token"]


def test_valid_completion_receipt_prevents_relaunch_even_with_stale_dispatch(
        tmp_path):
    args, best_path, best = _published_best_for_dispatch(tmp_path)
    root = Path(args.output_root) / "long_run"
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "long_summary.json"
    handoff = root / "sidecar_handoff.json"
    summary.write_text('{"completed": true}\n')
    handoff.write_text('{"schema": "mcln-sidecar-handoff-v1"}\n')
    binding = orchestrator.long_dispatch_binding(best)
    token = "a" * 32
    (root / "completion.json").write_text(json.dumps({
        "schema": orchestrator.LONG_COMPLETION_SCHEMA,
        "token": token,
        "binding": binding,
        "pid": 99999999,
        "process_start_time": "dead",
        "summary": str(summary),
        "summary_sha256": orchestrator.file_sha256(summary),
        "handoff": str(handoff),
        "handoff_sha256": orchestrator.file_sha256(handoff),
    }))
    (root / "dispatch.json").write_text(json.dumps({
        "schema": orchestrator.LONG_DISPATCH_SCHEMA,
        "status": "failed",
        "token": "b" * 32,
        "binding": binding,
        "attempt": 7,
        "created_at": 0.0,
    }))

    result = orchestrator._dispatch_long_run(
        args,
        best_path,
        lambda *args, **kwargs: pytest.fail("completed run must not relaunch"),
    )

    assert result["status"] == "completed"
    assert result["already_dispatched"] is True
    assert result["token"] == token


def test_dispatch_rejects_tampered_best_study_binding(tmp_path):
    args, best_path, best = _published_best_for_dispatch(tmp_path)
    root = Path(args.output_root) / "long_run"
    root.mkdir(parents=True, exist_ok=True)
    binding = orchestrator.long_dispatch_binding(best)
    binding["study_binding"] = dict(
        binding["study_binding"], study_name="other-study"
    )
    (root / "dispatch.json").write_text(json.dumps({
        "schema": orchestrator.LONG_DISPATCH_SCHEMA,
        "status": "starting",
        "token": "c" * 32,
        "binding": binding,
        "attempt": 1,
        "created_at": 0.0,
    }))

    with pytest.raises(ValueError, match="binding"):
        orchestrator._dispatch_long_run(
            args, best_path, lambda *args, **kwargs: _FakeProcess()
        )


@pytest.mark.parametrize("mismatch", ["stable_sha", "receipt", "study"])
def test_missing_source_reuse_requires_exact_stable_and_best_bindings(
        tmp_path, mismatch):
    args = _args(tmp_path)
    binding = orchestrator.study_receipt_binding(
        orchestrator.write_study_contract(args)
    )
    candidate = _candidate(tmp_path)
    source = Path(candidate["checkpoint"])
    stable = Path(args.output_root) / "checkpoints" / "optuna_best_trial.pth"
    stable.parent.mkdir(parents=True)
    stable.write_bytes(source.read_bytes())
    source.unlink()
    best = {
        "trial_number": candidate["trial_number"],
        "receipt": candidate["receipt"],
        "study_binding": binding,
    }
    if mismatch == "stable_sha":
        stable.write_bytes(b"corrupt")
    elif mismatch == "receipt":
        best["receipt"] = "other.json"
    else:
        best["study_binding"] = dict(
            binding, study_contract_digest="0" * 64
        )
    (Path(args.output_root) / "best.json").write_text(json.dumps(best))

    with pytest.raises(ValueError, match="checkpoint|binding|receipt"):
        orchestrator._publish_best_checkpoint(args, candidate)


def test_staged_checkpoint_digest_mismatch_preserves_existing_stable(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    orchestrator.write_study_contract(args)
    candidate = _candidate(tmp_path)
    stable = Path(args.output_root) / "checkpoints" / "optuna_best_trial.pth"
    stable.parent.mkdir(parents=True)
    stable.write_bytes(b"previous stable")

    def corrupt_link(_source, destination):
        Path(destination).write_bytes(b"corrupt staged bytes")

    monkeypatch.setattr(orchestrator.os, "link", corrupt_link)

    with pytest.raises(ValueError, match="SHA-256|digest"):
        orchestrator._publish_best_checkpoint(args, candidate)
    assert stable.read_bytes() == b"previous stable"


def test_candidate_from_trial_receipt_uses_epoch56_and_strict_baseline(
        tmp_path):
    args = _args(tmp_path)
    binding = orchestrator.study_receipt_binding(
        orchestrator.write_study_contract(args)
    )
    baseline_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": binding,
    }
    checkpoint = tmp_path / "epoch56.pth"
    checkpoint.write_bytes(b"checkpoint")
    trial_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "trial",
        "selection_epoch": 56,
        "metrics": {
            "epoch_55": _metrics(),
            "epoch_56": _metrics(
                fixed025=2201,
                fixed050=1701,
                learned025=2220,
                learned050=1720,
                mask025=2260,
                mask050=1860,
                iou_sum=1610.0,
            ),
        },
        "losses": {
            "epoch_55": _loss_receipt(1.1),
            "epoch_56": _loss_receipt(1.0),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "trial_params": dict(APPROVED_PARAMS),
        "optimizer_groups": _optimizer_groups(),
        "study_binding": binding,
    }

    candidate = orchestrator.candidate_from_trial_receipt(
        8, baseline_receipt, trial_receipt, receipt_path="trial.json"
    )

    assert candidate["trial_number"] == 8
    assert candidate["feasible"] is True
    assert candidate["metrics"] == trial_receipt["metrics"]["epoch_56"]
    assert candidate["checkpoint"] == str(checkpoint)
    assert candidate["receipt"] == "trial.json"

    trial_receipt["selection_epoch"] = 55
    with pytest.raises(ValueError):
        orchestrator.candidate_from_trial_receipt(
            8, baseline_receipt, trial_receipt, receipt_path="trial.json"
        )


@pytest.mark.parametrize("bad_losses", [
    None,
    {
        "epoch_55": _loss_receipt(float("nan")),
        "epoch_56": _loss_receipt(1.0),
    },
])
def test_candidate_rejects_missing_or_nonfinite_training_losses(
        tmp_path, bad_losses):
    args = _args(tmp_path)
    binding = orchestrator.study_receipt_binding(
        orchestrator.write_study_contract(args)
    )
    baseline_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": binding,
    }
    checkpoint = tmp_path / "epoch56.pth"
    checkpoint.write_bytes(b"checkpoint")
    trial_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "trial",
        "selection_epoch": 56,
        "metrics": {
            "epoch_55": _metrics(),
            "epoch_56": _metrics(),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "trial_params": dict(APPROVED_PARAMS),
        "optimizer_groups": _optimizer_groups(),
        "study_binding": binding,
    }
    if bad_losses is not None:
        trial_receipt["losses"] = bad_losses

    with pytest.raises(ValueError, match="loss"):
        orchestrator.candidate_from_trial_receipt(
            8, baseline_receipt, trial_receipt, receipt_path="trial.json"
        )


def test_candidate_rejects_cross_contract_receipt_binding(tmp_path):
    args = _args(tmp_path)
    binding = orchestrator.study_receipt_binding(
        orchestrator.write_study_contract(args)
    )
    baseline_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": binding,
    }
    checkpoint = tmp_path / "epoch56.pth"
    checkpoint.write_bytes(b"checkpoint")
    trial_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "trial",
        "selection_epoch": 56,
        "metrics": {
            "epoch_55": _metrics(),
            "epoch_56": _metrics(),
        },
        "losses": {
            "epoch_55": _loss_receipt(1.1),
            "epoch_56": _loss_receipt(1.0),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "trial_params": dict(APPROVED_PARAMS),
        "optimizer_groups": _optimizer_groups(),
        "study_binding": dict(binding, study_contract_digest="0" * 64),
    }

    with pytest.raises(ValueError, match="study binding"):
        orchestrator.candidate_from_trial_receipt(
            8, baseline_receipt, trial_receipt, receipt_path="trial.json"
        )


def test_candidate_rejects_optimizer_groups_without_parameter_names(
        tmp_path):
    args = _args(tmp_path)
    binding = orchestrator.study_receipt_binding(
        orchestrator.write_study_contract(args)
    )
    baseline_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": binding,
    }
    checkpoint = tmp_path / "epoch56.pth"
    checkpoint.write_bytes(b"checkpoint")
    trial_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "trial",
        "selection_epoch": 56,
        "metrics": {
            "epoch_55": _metrics(),
            "epoch_56": _metrics(),
        },
        "losses": {
            "epoch_55": _loss_receipt(1.1),
            "epoch_56": _loss_receipt(1.0),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "trial_params": dict(APPROVED_PARAMS),
        "optimizer_groups": [
            {"name": "decoder", "initial_lr": APPROVED_PARAMS["decoder_lr"]},
            {
                "name": "backbone",
                "initial_lr": APPROVED_PARAMS["decoder_lr"] * 10.0,
            },
            {
                "name": "mask_head",
                "initial_lr": APPROVED_PARAMS["decoder_lr"]
                * APPROVED_PARAMS["mask_head_lr_multiplier"],
            },
            {"name": "selector", "initial_lr": APPROVED_PARAMS["selector_lr"]},
        ],
        "study_binding": binding,
    }

    with pytest.raises(ValueError, match="parameter_names"):
        orchestrator.candidate_from_trial_receipt(
            8, baseline_receipt, trial_receipt, receipt_path="trial.json"
        )


@pytest.mark.parametrize(
    "case", [
        "valid", "missing_attrs", "different_params", "out_of_range",
    ]
)
def test_complete_trial_recovery_validates_all_bindings_before_counting(
        tmp_path, case):
    args = _args(tmp_path)
    binding = orchestrator.study_receipt_binding(
        orchestrator.write_study_contract(args)
    )
    baseline_receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "baseline",
        "selection_epoch": 54,
        "metrics": {"epoch_54": _metrics()},
        "checkpoint": None,
        "trial_params": None,
        "optimizer_groups": [],
        "study_binding": binding,
    }
    study_params = dict(APPROVED_PARAMS)
    if case == "out_of_range":
        study_params["decoder_lr"] = 1e-3
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(study_params)
    live_trial = study.ask()
    if case == "out_of_range":
        with pytest.warns(UserWarning, match="out of range"):
            assert suggest_trial_params(live_trial) == study_params
    else:
        assert suggest_trial_params(live_trial) == study_params

    receipt_params = (
        seed_presets()[0] if case == "different_params" else study_params
    )
    checkpoint = tmp_path / "epoch56.pth"
    checkpoint.write_bytes(b"checkpoint")
    receipt = {
        "schema": "mcln-optuna-trial-v1",
        "mode": "trial",
        "selection_epoch": 56,
        "metrics": {
            "epoch_55": _metrics(),
            "epoch_56": _metrics(),
        },
        "losses": {
            "epoch_55": _loss_receipt(1.1),
            "epoch_56": _loss_receipt(1.0),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "trial_params": dict(receipt_params),
        "optimizer_groups": _optimizer_groups(receipt_params),
        "study_binding": binding,
    }
    receipt_path = (
        Path(args.output_root) / "trials" / "trial_0000" / "receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    live_trial.set_user_attr(
        "receipt", "trials/trial_0000/receipt.json"
    )
    if case != "missing_attrs":
        live_trial.set_user_attr(
            "study_contract_digest", binding["study_contract_digest"]
        )
        live_trial.set_user_attr(
            "receipt_sha256", orchestrator.file_sha256(receipt_path)
        )
        live_trial.set_user_attr(
            "optimizer_groups_sha256",
            orchestrator.canonical_json_sha256(
                _optimizer_groups(receipt_params)
            ),
        )
    study.tell(live_trial, 0.0)

    if case == "valid":
        candidates = orchestrator._valid_complete_candidates(
            args, study, baseline_receipt
        )
        assert [candidate["trial_number"] for candidate in candidates] == [0]
    else:
        with pytest.raises(ValueError, match="COMPLETE trial"):
            orchestrator._valid_complete_candidates(
                args, study, baseline_receipt
            )


@pytest.mark.parametrize(
    "name,value",
    [
        ("decoder_lr", True),
        ("decoder_lr", float("nan")),
        ("decoder_lr", 4e-6),
        ("mask_head_lr_multiplier", 3.0),
        ("selector_lr", 2.1e-3),
        ("mask_loss_scale", 0.4),
        ("consistency_loss_scale", 2.1),
        ("selector_loss_weight", 1.1),
        ("selector_min_iou_gap", 0.04),
    ],
)
def test_trial_params_reject_invalid_type_finite_bounds_and_categories(
        name, value):
    params = dict(APPROVED_PARAMS)
    params[name] = value

    with pytest.raises(ValueError, match=name):
        orchestrator._validate_trial_params(params)


def _proc_start_time_for_test(pid):
    payload = Path("/proc/{}/stat".format(pid)).read_text()
    return payload[payload.rfind(")") + 2:].split()[19]


def test_live_running_trial_fails_closed_instead_of_being_marked_failed():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr("subprocess_pid", os.getpid())
    trial.set_user_attr(
        "subprocess_start_time", _proc_start_time_for_test(os.getpid())
    )

    with pytest.raises(RuntimeError, match="active|RUNNING"):
        orchestrator.fail_stale_running_trials(study)
    assert study.trials[0].state == TrialState.RUNNING


def test_dead_or_missing_running_trial_is_marked_failed():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr("subprocess_pid", 99999999)
    trial.set_user_attr("subprocess_start_time", "1")

    assert orchestrator.fail_stale_running_trials(study) == 1
    assert study.trials[0].state == TrialState.FAIL


def test_second_orchestrator_owner_lock_fails_before_study_mutation(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    lock_path = Path(args.output_root) / "control" / "study_owner.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(
            orchestrator, "require_minimum_free_space", lambda _path: None
        )
        monkeypatch.setattr(
            orchestrator,
            "_ensure_baseline",
            lambda _args: pytest.fail("lock must be checked first"),
        )
        with pytest.raises(RuntimeError, match="owner|lock|orchestrator"):
            orchestrator.run_study(args)


def test_new_empty_study_is_claimed_before_baseline_mutation(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(
        orchestrator, "require_minimum_free_space", lambda _path: None
    )
    monkeypatch.setattr(
        orchestrator,
        "_ensure_baseline",
        lambda _args: (_ for _ in ()).throw(RuntimeError("stop-after-claim")),
    )

    with pytest.raises(RuntimeError, match="stop-after-claim"):
        orchestrator.run_study(args)

    study = optuna.load_study(
        study_name=args.study_name, storage=args.storage
    )
    contract = json.loads(
        (Path(args.output_root) / "study_contract.json").read_text()
    )
    assert study.user_attrs == {
        "schema": "mcln-optuna-study-binding-v1",
        "contract_digest": contract["contract_digest"],
    }


@pytest.mark.parametrize("attrs", [{}, {"schema": "other"}])
def test_existing_unowned_study_with_output_contract_is_rejected(
        tmp_path, monkeypatch, attrs):
    args = _args(tmp_path)
    orchestrator.write_study_contract(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
    )
    for name, value in attrs.items():
        study.set_user_attr(name, value)
    monkeypatch.setattr(
        orchestrator, "require_minimum_free_space", lambda _path: None
    )
    monkeypatch.setattr(orchestrator, "_ensure_baseline", lambda _args: {})
    monkeypatch.setattr(
        orchestrator, "publish_baseline_metrics", lambda *_args: None
    )
    monkeypatch.setattr(
        optuna.study.Study,
        "optimize",
        lambda *_args, **_kwargs: pytest.fail(
            "unowned study must be rejected before optimize"
        ),
    )

    with pytest.raises(ValueError, match="study.*binding|ownership"):
        orchestrator.run_study(args)
