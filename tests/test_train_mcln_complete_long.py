import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tuning.mcln_optuna_contract import METRICS_SCHEMA
from scripts.tuning import train_mcln_complete_long as long_runner


OFFICIAL_COUNT = 9508
GIB = 1024 ** 3


def _receipt(pos025, pos050, mask025, mask050, miou):
    iou_sum = miou * OFFICIAL_COUNT
    return {
        "schema": METRICS_SCHEMA,
        "sample_count": OFFICIAL_COUNT,
        "position": {
            "fixed_default": {
                "hits025": pos025,
                "hits050": pos050,
            },
            "learned_selector": {
                "hits025": pos025,
                "hits050": pos050,
            },
        },
        "mask": {
            "hits025": mask025,
            "hits050": mask050,
            "iou_sum": iou_sum,
            "miou": miou,
        },
    }


def _candidate(tmp_path, name, epoch, metrics):
    path = tmp_path / "{}.pth".format(name)
    path.write_bytes(name.encode("ascii"))
    return {
        "epoch": epoch,
        "path": str(path),
        "metrics": metrics,
    }


def test_long_training_and_validation_epochs_are_exact():
    assert long_runner.long_train_epochs(
        base_epoch=54, final_epoch=100
    ) == tuple(range(55, 101))
    assert long_runner.validation_epochs() == (
        60, 65, 70, 75, 80, 85, 90, 95, 100
    )


@pytest.mark.parametrize(
    "base_epoch,final_epoch",
    [(54, 54), (55, 54), (54.0, 100), (54, True)],
)
def test_long_training_epochs_reject_invalid_ranges(base_epoch, final_epoch):
    with pytest.raises(ValueError):
        long_runner.long_train_epochs(base_epoch, final_epoch)


def test_dominance_uses_all_five_metrics_and_requires_one_strict_gain():
    stronger = _receipt(5700, 4700, 5700, 4900, 0.46)
    weaker = _receipt(5600, 4600, 5600, 4800, 0.45)

    assert long_runner.dominates(stronger, weaker) is True
    assert long_runner.dominates(stronger, stronger) is False

    tradeoff = _receipt(5800, 4500, 5700, 4900, 0.46)
    assert long_runner.dominates(tradeoff, weaker) is False
    assert long_runner.dominates(weaker, tradeoff) is False


def test_target_distance_uses_exact_declared_targets():
    exact = _receipt(
        5610,
        4621,
        5582,
        4821,
        0.4472,
    )
    below = _receipt(5609, 4620, 5581, 4820, 0.4471)

    assert long_runner.target_distance(exact) == 0.0
    assert long_runner.target_distance(below) > 0.0


def test_release_gate_preserves_strict_mask_and_miou_inequalities():
    boundary = _receipt(5610, 4621, 5581, 4820, 0.4472)
    passed = _receipt(5610, 4621, 5582, 4821, 0.4472001)

    assert long_runner.release_gate_status(boundary)["passed"] is False
    assert long_runner.release_gate_status(passed)["passed"] is True


def test_pareto_selection_removes_dominated_and_keeps_three_roles(tmp_path):
    target = _candidate(
        tmp_path, "target", 60,
        _receipt(5609, 4620, 5581, 4820, 0.4471),
    )
    position = _candidate(
        tmp_path, "position", 65,
        _receipt(6000, 4500, 5400, 4600, 0.42),
    )
    mask = _candidate(
        tmp_path, "mask", 70,
        _receipt(5400, 4400, 5700, 5100, 0.47),
    )
    extra = _candidate(
        tmp_path, "extra", 75,
        _receipt(5700, 4700, 5500, 4700, 0.44),
    )
    dominated = _candidate(
        tmp_path, "dominated", 80,
        _receipt(5300, 4300, 5300, 4300, 0.40),
    )

    selected = long_runner.select_pareto_checkpoints(
        [target, position, mask, extra, dominated], max_keep=3
    )

    assert [item["retention_role"] for item in selected] == [
        "target_distance", "position025", "mask_balance"
    ]
    assert [Path(item["path"]).stem for item in selected] == [
        "target", "position", "mask"
    ]
    assert all(Path(item["path"]).exists() for item in selected)
    assert not Path(extra["path"]).exists()
    assert not Path(dominated["path"]).exists()


def test_pareto_selection_deduplicates_identical_checkpoint_identity(tmp_path):
    first = _candidate(
        tmp_path, "first", 60,
        _receipt(5600, 4600, 5600, 4800, 0.45),
    )
    alias_path = tmp_path / "alias.pth"
    os.link(first["path"], alias_path)
    alias = dict(first, epoch=65, path=str(alias_path))

    selected = long_runner.select_pareto_checkpoints([first, alias])

    assert len(selected) == 1
    assert selected[0]["epoch"] == 60
    assert Path(first["path"]).exists()
    assert not alias_path.exists()


def test_pareto_selection_rejects_protected_deletion_before_mutation(tmp_path):
    kept = _candidate(
        tmp_path, "kept", 60,
        _receipt(5700, 4700, 5700, 4900, 0.46),
    )
    protected = _candidate(
        tmp_path, "protected", 65,
        _receipt(5300, 4300, 5300, 4300, 0.40),
    )

    with pytest.raises(ValueError):
        long_runner.select_pareto_checkpoints(
            [kept, protected],
            protected_paths=[protected["path"]],
        )

    assert Path(kept["path"]).exists()
    assert Path(protected["path"]).exists()


def test_atomic_checkpoint_copy_replaces_latest_without_touching_source(
        tmp_path):
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    latest = tmp_path / "latest.pth"
    first.write_bytes(b"epoch55")
    second.write_bytes(b"epoch56")

    long_runner.atomic_copy_checkpoint(first, latest)
    assert latest.read_bytes() == b"epoch55"
    long_runner.atomic_copy_checkpoint(second, latest)

    assert latest.read_bytes() == b"epoch56"
    assert first.read_bytes() == b"epoch55"
    assert second.read_bytes() == b"epoch56"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_checkpoint_copy_rejects_protected_destination(tmp_path):
    source = tmp_path / "source.pth"
    protected = tmp_path / "epoch54.pth"
    source.write_bytes(b"new")
    protected.write_bytes(b"protected")

    with pytest.raises(ValueError):
        long_runner.atomic_copy_checkpoint(
            source, protected, protected_paths=[protected]
        )

    assert protected.read_bytes() == b"protected"


def test_fresh_capacity_projects_five_checkpoint_inodes_plus_reserve():
    checkpoint_size = 793041121

    projected = long_runner.projected_peak_capacity_bytes(checkpoint_size)

    assert projected == (
        5 * checkpoint_size + long_runner.LONG_RUN_SAFETY_RESERVE_BYTES
    )
    assert long_runner.required_initial_free_bytes(checkpoint_size) == max(
        8 * GIB, projected
    )


def test_resume_capacity_only_charges_next_latest_candidate_and_reserve():
    checkpoint_size = 793041121
    current_free = int(11.8 * GIB)

    required = long_runner.required_resume_free_bytes(checkpoint_size)

    assert required == (
        2 * checkpoint_size + long_runner.LONG_RUN_SAFETY_RESERVE_BYTES
    )
    assert long_runner.require_long_run_capacity(
        checkpoint_size,
        resume=True,
        reported_free_bytes=current_free,
    ) == current_free


def test_resume_capacity_rejects_insufficient_next_atomic_write():
    checkpoint_size = 793041121
    required = long_runner.required_resume_free_bytes(checkpoint_size)

    with pytest.raises(ValueError, match="capacity|free space"):
        long_runner.require_long_run_capacity(
            checkpoint_size,
            resume=True,
            reported_free_bytes=required - 1,
        )


def test_pareto_plan_does_not_delete_before_summary_publication(tmp_path):
    stronger = _candidate(
        tmp_path, "stronger", 60,
        _receipt(5700, 4700, 5700, 4900, 0.46),
    )
    weaker = _candidate(
        tmp_path, "weaker", 65,
        _receipt(5300, 4300, 5300, 4300, 0.40),
    )

    selected, cleanup_paths = long_runner.plan_pareto_checkpoints(
        [stronger, weaker]
    )

    assert [item["path"] for item in selected] == [stronger["path"]]
    assert cleanup_paths == [weaker["path"]]
    assert Path(stronger["path"]).is_file()
    assert Path(weaker["path"]).is_file()


def test_long_summary_is_published_before_unreferenced_cleanup(
        tmp_path, monkeypatch):
    retained = _candidate(
        tmp_path, "retained", 60,
        _receipt(5700, 4700, 5700, 4900, 0.46),
    )
    retained["retention_role"] = "target_distance"
    extra = tmp_path / "extra.pth"
    extra.write_bytes(b"extra")
    summary_path = tmp_path / "long_summary.json"
    events = []
    real_atomic_write = long_runner.atomic_write_json

    def recording_write(path, payload):
        events.append("summary")
        real_atomic_write(path, payload)

    def recording_cleanup(paths, protected_paths=()):
        events.append("cleanup")
        for path in paths:
            Path(path).unlink()

    monkeypatch.setattr(long_runner, "atomic_write_json", recording_write)
    monkeypatch.setattr(
        long_runner, "cleanup_checkpoint_paths", recording_cleanup
    )

    long_runner.publish_long_summary(
        summary_path,
        {
            "schema": long_runner.LONG_SUMMARY_SCHEMA,
            "latest_epoch": 60,
            "phase": "epoch_complete",
            "retained": [retained],
        },
        cleanup_paths=[extra],
    )

    assert events == ["summary", "cleanup"]
    assert summary_path.is_file()
    assert Path(retained["path"]).is_file()
    assert not extra.exists()


class _FakeSampler:
    def set_epoch(self, epoch):
        self.epoch = epoch


class _FakeLoader:
    def __init__(self):
        self.sampler = _FakeSampler()


class _FakeLongTrainer:
    def __init__(self, train_events, validation_events):
        self.train_events = train_events
        self.validation_events = validation_events

    def train_one_epoch(self, epoch, *args):
        self.train_events.append(epoch)

    def evaluate_one_epoch(self, epoch, *args):
        self.validation_events.append(epoch)
        return _receipt(
            5600 + epoch,
            4600 + epoch,
            5500 + epoch,
            4700 + epoch,
            0.44 + epoch / 100000.0,
        )


def _fake_long_inputs(tmp_path):
    output_root = tmp_path / "long_run"
    base_checkpoint = tmp_path / "base.pth"
    best_checkpoint = tmp_path / "best.pth"
    base_checkpoint.write_bytes(b"base")
    best_checkpoint.write_bytes(b"best")
    custom_args = SimpleNamespace(
        output_root=str(output_root),
        base_checkpoint=str(base_checkpoint),
        data_root=str(tmp_path / "data"),
        pp_checkpoint=str(tmp_path / "pointnet.pth"),
    )
    shared_args = SimpleNamespace(start_epoch=55)
    best = {
        "checkpoint": str(best_checkpoint),
        "source_snapshot_digest": "b" * 64,
    }
    return shared_args, custom_args, best


def _install_fake_long_runtime(monkeypatch, train_events, validation_events):
    def fake_build_runtime(shared_args, custom_args):
        latest = Path(custom_args.output_root) / "latest.pth"
        if latest.is_file():
            shared_args.start_epoch = int(latest.read_text()) + 1
        else:
            shared_args.start_epoch = 55
        return {
            "trainer": _FakeLongTrainer(train_events, validation_events),
            "train_loader": _FakeLoader(),
            "official_loader": object(),
            "model": object(),
            "criterion": object(),
            "set_criterion": object(),
            "optimizer": object(),
            "scheduler": object(),
            "latest": latest,
        }

    def fake_atomic_save(path, args, epoch, *unused, **kwargs):
        path = Path(path)
        temporary = path.with_name(".latest-test.tmp")
        temporary.write_text(str(epoch))
        os.replace(str(temporary), str(path))
        return path

    monkeypatch.setattr(long_runner, "_build_runtime", fake_build_runtime)
    monkeypatch.setattr(
        long_runner, "_atomic_save_checkpoint", fake_atomic_save
    )
    monkeypatch.setattr(
        long_runner, "require_long_run_filesystem_capacity",
        lambda *args, **kwargs: None,
        raising=False,
    )


@pytest.mark.parametrize("crash_epoch", [60, 100])
def test_resume_recovers_checkpointed_validation_without_retraining(
        tmp_path, monkeypatch, crash_epoch):
    train_events = []
    validation_events = []
    shared_args, custom_args, best = _fake_long_inputs(tmp_path)
    _install_fake_long_runtime(monkeypatch, train_events, validation_events)
    real_atomic_write = long_runner.atomic_write_json

    def crash_before_checkpointed_summary(path, payload):
        if (
            Path(path).name == "long_summary.json"
            and payload.get("phase") == "checkpointed"
            and payload.get("latest_epoch") == crash_epoch
        ):
            raise RuntimeError("simulated crash after latest")
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        long_runner, "atomic_write_json", crash_before_checkpointed_summary
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        long_runner.run_long_training(shared_args, custom_args, best)

    output_root = Path(custom_args.output_root)
    assert not (output_root / "sidecar_handoff.json").exists()
    assert train_events.count(crash_epoch) == 1

    monkeypatch.setattr(long_runner, "atomic_write_json", real_atomic_write)
    resumed_shared = SimpleNamespace(start_epoch=55)
    handoff = long_runner.run_long_training(
        resumed_shared, custom_args, best
    )

    assert train_events.count(crash_epoch) == 1
    assert validation_events == list(long_runner.VALIDATION_EPOCHS)
    summary = json.loads((output_root / "long_summary.json").read_text())
    assert summary["latest_epoch"] == 100
    assert summary["phase"] == "epoch_complete"
    assert summary["completed"] is True
    assert summary["completed_validation_epochs"] == list(
        long_runner.VALIDATION_EPOCHS
    )
    assert handoff["schema"] == "mcln-sidecar-handoff-v1"

    train_count = len(train_events)
    validation_count = len(validation_events)
    long_runner.run_long_training(
        SimpleNamespace(start_epoch=55), custom_args, best
    )
    assert len(train_events) == train_count
    assert len(validation_events) == validation_count


def test_resume_fails_closed_when_earlier_validation_cannot_be_reconstructed(
        tmp_path, monkeypatch):
    train_events = []
    validation_events = []
    shared_args, custom_args, best = _fake_long_inputs(tmp_path)
    output_root = Path(custom_args.output_root)
    output_root.mkdir(parents=True)
    (output_root / "latest.pth").write_text("65")
    (output_root / "long_summary.json").write_text(json.dumps({
        "schema": "mcln-complete-long-summary-v1",
        "latest_epoch": 65,
        "latest": str(output_root / "latest.pth"),
        "retained": [],
        "validation_epochs": list(long_runner.VALIDATION_EPOCHS),
        "completed": False,
    }))
    _install_fake_long_runtime(monkeypatch, train_events, validation_events)

    with pytest.raises(ValueError, match="validation.*60|reconstruct"):
        long_runner.run_long_training(shared_args, custom_args, best)

    assert train_events == []
    assert validation_events == []
    assert not (output_root / "sidecar_handoff.json").exists()


def _dispatch_custom_args(tmp_path, token="d" * 32):
    output_root = tmp_path / "long_run"
    output_root.mkdir()
    return SimpleNamespace(
        output_root=str(output_root),
        dispatch_token=token,
        startup_ack=str(output_root / "startup_ack.json"),
        completion_receipt=str(output_root / "completion.json"),
    )


def test_long_runner_writes_bound_startup_ack(tmp_path, monkeypatch):
    custom_args = _dispatch_custom_args(tmp_path)
    output_root = Path(custom_args.output_root)
    binding = {"schema": "test-binding", "study": "owned"}
    (output_root / "dispatch.json").write_text(json.dumps({
        "schema": long_runner.LONG_DISPATCH_SCHEMA,
        "status": "starting",
        "token": custom_args.dispatch_token,
        "binding": binding,
        "attempt": 1,
        "created_at": 1.0,
    }))
    monkeypatch.setattr(
        long_runner, "_process_start_time", lambda _pid: "proc-start"
    )

    ack = long_runner.write_startup_ack(custom_args, binding)

    assert ack == json.loads(Path(custom_args.startup_ack).read_text())
    assert ack["schema"] == long_runner.LONG_STARTUP_ACK_SCHEMA
    assert ack["token"] == custom_args.dispatch_token
    assert ack["binding"] == binding
    assert ack["pid"] == os.getpid()
    assert ack["process_start_time"] == "proc-start"


def test_long_runner_completion_receipt_hashes_summary_and_handoff(
        tmp_path, monkeypatch):
    custom_args = _dispatch_custom_args(tmp_path)
    output_root = Path(custom_args.output_root)
    binding = {"schema": "test-binding", "study": "owned"}
    process_start = "proc-start"
    monkeypatch.setattr(
        long_runner, "_process_start_time", lambda _pid: process_start
    )
    dispatch = {
        "schema": long_runner.LONG_DISPATCH_SCHEMA,
        "status": "running",
        "token": custom_args.dispatch_token,
        "binding": binding,
        "attempt": 1,
        "created_at": 1.0,
        "pid": os.getpid(),
        "process_start_time": process_start,
    }
    (output_root / "dispatch.json").write_text(json.dumps(dispatch))
    long_runner.write_startup_ack(custom_args, binding)
    summary = output_root / "long_summary.json"
    handoff = output_root / "sidecar_handoff.json"
    summary.write_text('{"completed": true}\n')
    handoff.write_text('{"schema": "mcln-sidecar-handoff-v1"}\n')

    completion = long_runner.write_completion_receipt(custom_args, binding)

    assert completion == json.loads(
        Path(custom_args.completion_receipt).read_text()
    )
    assert completion["schema"] == long_runner.LONG_COMPLETION_SCHEMA
    assert completion["summary_sha256"] == long_runner.file_sha256(summary)
    assert completion["handoff_sha256"] == long_runner.file_sha256(handoff)
