import hashlib
import importlib
import json
import os
import stat

import pytest
import torch
from torch import nn


def _trainer():
    return importlib.import_module(
        "scripts.train_scanrefer_joint_mask_head"
    )


class _ArtifactToyMCLN(nn.Module):
    def __init__(self):
        super(_ArtifactToyMCLN, self).__init__()
        self.detector = nn.Linear(3, 3)
        self.x_query = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )
        self.x_query.register_buffer(
            "calibration_counter", torch.tensor([7], dtype=torch.long)
        )


def _base_checkpoint_binding():
    return {
        "path": "/protected/backbone.pth",
        "sha256": "a" * 64,
        "size": 123456,
        "mode": "0444",
    }


def _run_record():
    return {
        "run_id": "20260723_stage2_xquery_seed0",
        "step": 3,
        "epoch": 0,
        "config": {
            "batch_size": 18,
            "gradient_clip": 0.1,
            "lr": 2e-5,
            "seed": 0,
            "weight_decay": 5e-4,
        },
        "source_sha256": {
            "scripts/train_scanrefer_joint_mask_head.py": "b" * 64,
        },
        "split_digest": "c" * 64,
        "protected_artifacts": {
            "parent_reranker": {
                "path": "/protected/parent.pth",
                "sha256": "d" * 64,
                "size": 2345,
                "mode": "0444",
            },
            "geometry_reranker": {
                "path": "/protected/geometry.pth",
                "sha256": "e" * 64,
                "size": 3456,
                "mode": "0444",
            },
        },
        "train_metrics": {
            "mask_dice_loss": 0.25,
            "mask_focal_loss": 0.5,
            "total_loss": 0.75,
        },
    }


def _assert_nested_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert tuple(actual.shape) == tuple(expected.shape)
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(actual, dict):
        assert list(actual) == list(expected)
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "api_name",
    ["publish_query_mask_checkpoint", "load_query_mask_checkpoint"],
)
def test_checkpoint_api_documents_read_only_seal_without_immutability_claim(
        api_name):
    trainer = _trainer()
    documentation = " ".join(
        getattr(trainer, api_name).__doc__.lower().split()
    )

    assert "immutable" not in documentation
    assert "sealed read-only at publication" in documentation
    assert "same-owner" in documentation
    assert "root" in documentation
    assert "modify or unlink" in documentation


def test_build_query_mask_checkpoint_contains_only_owned_x_query_state():
    trainer = _trainer()
    torch.manual_seed(41)
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(
        group["parameters"], lr=2e-5, weight_decay=5e-4
    )

    loss = sum(parameter.square().sum() for parameter in group["parameters"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )

    assert artifact["schema"] == "mcln-x-query-checkpoint-v1"
    assert artifact["validation_data_accessed"] is False
    assert artifact["inference_uses_ground_truth"] is False
    assert artifact["base_checkpoint"] == _base_checkpoint_binding()
    assert artifact["run_record"] == _run_record()
    assert tuple(artifact["x_query_state_dict"]) == (
        "x_query.0.weight",
        "x_query.0.bias",
        "x_query.2.weight",
        "x_query.2.bias",
        "x_query.calibration_counter",
    )
    assert all(
        value.device.type == "cpu"
        for value in artifact["x_query_state_dict"].values()
    )
    assert not any(
        name.startswith("detector.")
        for name in artifact["x_query_state_dict"]
    )
    assert set(artifact["optimizer_state_dict"]) == {
        "state", "param_groups"
    }

    saved_weight = artifact["x_query_state_dict"][
        "x_query.0.weight"
    ].clone()
    with torch.no_grad():
        model.x_query[0].weight.add_(10.0)
    assert torch.equal(
        artifact["x_query_state_dict"]["x_query.0.weight"],
        saved_weight,
    )


def test_publish_query_mask_checkpoint_is_read_only_hashed_and_no_clobber(
        tmp_path):
    trainer = _trainer()
    torch.manual_seed(43)
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(
        group["parameters"], lr=2e-5, weight_decay=5e-4
    )
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "x_query_step_000003.pth"

    receipt = trainer.publish_query_mask_checkpoint(destination, artifact)

    receipt_path = tmp_path / "x_query_step_000003.pth.receipt.json"
    assert destination.is_file()
    assert receipt_path.is_file()
    assert destination.stat().st_mode & 0o777 == 0o444
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    checkpoint_bytes = destination.read_bytes()
    assert receipt["schema"] == "mcln-x-query-checkpoint-receipt-v1"
    assert receipt["checkpoint_name"] == destination.name
    assert receipt["checkpoint_sha256"] == hashlib.sha256(
        checkpoint_bytes
    ).hexdigest()
    assert receipt["checkpoint_size"] == len(checkpoint_bytes)
    assert receipt["checkpoint_mode"] == "0444"
    assert receipt["base_checkpoint_sha256"] == "a" * 64
    assert receipt["run_id"] == _run_record()["run_id"]
    assert receipt["step"] == _run_record()["step"]
    assert receipt["validation_data_accessed"] is False
    assert receipt["inference_uses_ground_truth"] is False
    with receipt_path.open("r") as handle:
        assert json.load(handle) == receipt

    original_checkpoint = destination.read_bytes()
    original_receipt = receipt_path.read_bytes()
    artifact["x_query_state_dict"]["x_query.0.weight"].add_(100.0)
    try:
        trainer.publish_query_mask_checkpoint(destination, artifact)
    except FileExistsError:
        pass
    else:
        raise AssertionError("checkpoint publisher overwrote an existing best")
    assert destination.read_bytes() == original_checkpoint
    assert receipt_path.read_bytes() == original_receipt
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.staging.*"))


def test_publish_query_mask_checkpoint_rejects_preoccupied_receipt_before_staging(
        tmp_path, monkeypatch):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    competitor_bytes = b"competitor receipt\n"
    receipt_path.write_bytes(competitor_bytes)
    competitor_inode = receipt_path.stat().st_ino
    staging_calls = []
    original_create_staging = trainer._create_checkpoint_staging_file

    def record_staging_creation(*args, **kwargs):
        staging_calls.append((args, kwargs))
        return original_create_staging(*args, **kwargs)

    monkeypatch.setattr(
        trainer,
        "_create_checkpoint_staging_file",
        record_staging_creation,
    )

    with pytest.raises(FileExistsError, match="receipt.json"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    assert staging_calls == []
    assert not destination.exists()
    assert receipt_path.read_bytes() == competitor_bytes
    assert receipt_path.stat().st_ino == competitor_inode
    assert not list(tmp_path.glob("*.staging.*"))


def test_publish_query_mask_checkpoint_rejects_replaced_staging_identity(
        tmp_path, monkeypatch):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    original_publish = trainer._publish_staged_file_noreplace
    competitor_staging = {}

    def publish_after_replacing_staging(
            parent_fd, staged_name, destination_name, destination_path,
            *expected_signature):
        if destination_name == destination.name:
            os.unlink(staged_name, dir_fd=parent_fd)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                staged_name, flags, 0o444, dir_fd=parent_fd
            )
            try:
                replacement = b"replacement staging\n"
                os.write(descriptor, replacement)
            finally:
                os.close(descriptor)
            staged_stat = os.stat(
                staged_name, dir_fd=parent_fd, follow_symlinks=False
            )
            competitor_staging["name"] = staged_name
            competitor_staging["bytes"] = replacement
            competitor_staging["identity"] = (
                staged_stat.st_dev, staged_stat.st_ino
            )
        return original_publish(
            parent_fd,
            staged_name,
            destination_name,
            destination_path,
            *expected_signature
        )

    monkeypatch.setattr(
        trainer,
        "_publish_staged_file_noreplace",
        publish_after_replacing_staging,
    )

    with pytest.raises(RuntimeError, match="staging.*identity"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    assert not destination.exists()
    assert not receipt_path.exists()
    competitor_path = tmp_path / competitor_staging["name"]
    assert competitor_path.read_bytes() == competitor_staging["bytes"]
    competitor_stat = competitor_path.stat()
    assert (competitor_stat.st_dev, competitor_stat.st_ino) == (
        competitor_staging["identity"]
    )
    assert list(tmp_path.glob("*.staging.*")) == [competitor_path]


def test_publish_query_mask_checkpoint_revalidates_destination_before_return(
        tmp_path, monkeypatch):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    replacement = b"replacement published checkpoint\n"
    original_publish = trainer._publish_staged_file_noreplace

    def publish_then_replace_destination(
            parent_fd, staged_name, destination_name, destination_path,
            *expected_signature):
        identity = original_publish(
            parent_fd,
            staged_name,
            destination_name,
            destination_path,
            *expected_signature
        )
        if destination_name == receipt_path.name:
            os.unlink(destination.name, dir_fd=parent_fd)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                destination.name, flags, 0o444, dir_fd=parent_fd
            )
            try:
                os.write(descriptor, replacement)
            finally:
                os.close(descriptor)
        return identity

    monkeypatch.setattr(
        trainer,
        "_publish_staged_file_noreplace",
        publish_then_replace_destination,
    )

    with pytest.raises(RuntimeError, match="published checkpoint"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    assert destination.read_bytes() == replacement
    assert receipt_path.is_file()
    assert not list(tmp_path.glob("*.staging.*"))


@pytest.mark.parametrize("replace_checkpoint", [False, True])
def test_publish_query_mask_checkpoint_receipt_race_preserves_checkpoint(
        tmp_path, monkeypatch, replace_checkpoint):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    competitor_checkpoint = b"competitor checkpoint\n"
    competitor_receipt = b'{"owner": "competitor"}\n'
    original_publish = trainer._publish_staged_file_noreplace
    owned_checkpoint = {}

    def write_claimed_file(parent_fd, name, content):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o444, dir_fd=parent_fd)
        try:
            os.write(descriptor, content)
        finally:
            os.close(descriptor)

    def publish_with_receipt_race(
            parent_fd, staged_name, destination_name, destination_path,
            *expected_signature):
        if destination_name == destination.name:
            identity = original_publish(
                parent_fd,
                staged_name,
                destination_name,
                destination_path,
                *expected_signature
            )
            owned_checkpoint["identity"] = identity
            owned_checkpoint["bytes"] = destination.read_bytes()
            owned_checkpoint["mode"] = destination.stat().st_mode & 0o777
            return identity

        if replace_checkpoint:
            os.unlink(destination.name, dir_fd=parent_fd)
            write_claimed_file(
                parent_fd, destination.name, competitor_checkpoint
            )
        write_claimed_file(
            parent_fd, receipt_path.name, competitor_receipt
        )
        raise FileExistsError("receipt path was claimed")

    monkeypatch.setattr(
        trainer,
        "_publish_staged_file_noreplace",
        publish_with_receipt_race,
    )

    with pytest.raises(FileExistsError, match="receipt path was claimed"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    assert receipt_path.read_bytes() == competitor_receipt
    if replace_checkpoint:
        assert destination.read_bytes() == competitor_checkpoint
        destination_stat = destination.stat()
        assert (destination_stat.st_dev, destination_stat.st_ino) != (
            owned_checkpoint["identity"]
        )
    else:
        assert destination.read_bytes() == owned_checkpoint["bytes"]
        destination_stat = destination.stat()
        assert (destination_stat.st_dev, destination_stat.st_ino) == (
            owned_checkpoint["identity"]
        )
        assert destination_stat.st_mode & 0o777 == 0o444
        assert owned_checkpoint["mode"] == 0o444
    assert not list(tmp_path.glob("*.staging.*"))


@pytest.mark.parametrize("primary_type", [FileExistsError, KeyboardInterrupt])
def test_publish_query_mask_checkpoint_preserves_primary_cleanup_exception(
        tmp_path, monkeypatch, primary_type):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    original_publish = trainer._publish_staged_file_noreplace
    original_close = trainer._os.close
    original_fstat = trainer._os.fstat
    primary_error = primary_type("primary receipt publication failure")
    primary_started = []
    cleanup_attempts = []
    fsync_attempts = []
    close_attempts = []

    def fail_receipt_publication(
            parent_fd, staged_name, destination_name, destination_path,
            expected_signature):
        if destination_name == destination.name:
            return original_publish(
                parent_fd,
                staged_name,
                destination_name,
                destination_path,
                expected_signature,
            )
        primary_started.append(True)
        raise primary_error

    def fail_staging_cleanup(name, *args, **kwargs):
        cleanup_attempts.append(name)
        raise KeyboardInterrupt("staging cleanup failure")

    def fail_directory_fsync(parent_fd):
        fsync_attempts.append(parent_fd)
        raise SystemExit("directory fsync failure")

    def fail_parent_close(descriptor):
        if (primary_started
                and trainer._stat.S_ISDIR(
                    original_fstat(descriptor).st_mode
                )):
            close_attempts.append(descriptor)
            original_close(descriptor)
            raise RuntimeError("parent close failure")
        return original_close(descriptor)

    monkeypatch.setattr(
        trainer,
        "_publish_staged_file_noreplace",
        fail_receipt_publication,
    )
    monkeypatch.setattr(trainer._os, "unlink", fail_staging_cleanup)
    monkeypatch.setattr(
        trainer, "_fsync_checkpoint_directory", fail_directory_fsync
    )
    monkeypatch.setattr(trainer._os, "close", fail_parent_close)

    caught = None
    try:
        trainer.publish_query_mask_checkpoint(destination, artifact)
    except BaseException as error:
        caught = error

    assert caught is primary_error
    assert len(cleanup_attempts) == 2
    assert len(fsync_attempts) == 1
    assert len(close_attempts) == 1


def test_publish_query_mask_checkpoint_raises_first_cleanup_error(
        tmp_path, monkeypatch):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    original_validate = trainer._validate_published_checkpoint_against_receipt
    original_close = trainer._os.close
    original_fstat = trainer._os.fstat
    original_fsync = trainer._fsync_checkpoint_directory
    first_cleanup_error = KeyboardInterrupt("first staging cleanup failure")
    cleanup_phase = []
    cleanup_attempts = []
    fsync_attempts = []
    close_attempts = []

    def mark_cleanup_phase(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        cleanup_phase.append(True)
        return result

    def fail_staging_cleanup(name, *args, **kwargs):
        cleanup_attempts.append(name)
        if len(cleanup_attempts) == 1:
            raise first_cleanup_error
        raise RuntimeError("second staging cleanup failure")

    def fail_final_fsync(parent_fd):
        fsync_attempts.append(parent_fd)
        if len(fsync_attempts) == 1:
            return original_fsync(parent_fd)
        raise SystemExit("final directory fsync failure")

    def fail_parent_close(descriptor):
        if (cleanup_phase
                and trainer._stat.S_ISDIR(
                    original_fstat(descriptor).st_mode
                )):
            close_attempts.append(descriptor)
            original_close(descriptor)
            raise RuntimeError("parent close failure")
        return original_close(descriptor)

    monkeypatch.setattr(
        trainer,
        "_validate_published_checkpoint_against_receipt",
        mark_cleanup_phase,
    )
    monkeypatch.setattr(trainer._os, "unlink", fail_staging_cleanup)
    monkeypatch.setattr(
        trainer, "_fsync_checkpoint_directory", fail_final_fsync
    )
    monkeypatch.setattr(trainer._os, "close", fail_parent_close)

    caught = None
    try:
        trainer.publish_query_mask_checkpoint(destination, artifact)
    except BaseException as error:
        caught = error

    assert caught is first_cleanup_error
    assert len(cleanup_attempts) == 2
    assert len(fsync_attempts) == 2
    assert len(close_attempts) == 1


@pytest.mark.parametrize("failed_fdopen_call", [1, 2, 3])
def test_publish_query_mask_checkpoint_closes_fd_when_fdopen_fails(
        tmp_path, monkeypatch, failed_fdopen_call):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    original_fdopen = trainer._os.fdopen
    original_fstat = trainer._os.fstat
    fdopen_calls = []
    failed_descriptor = {}

    def fail_selected_fdopen(descriptor, *args, **kwargs):
        fdopen_calls.append(descriptor)
        if len(fdopen_calls) == failed_fdopen_call:
            failed_descriptor["value"] = descriptor
            raise OSError("injected fdopen construction failure")
        return original_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(trainer._os, "fdopen", fail_selected_fdopen)

    with pytest.raises(
            (OSError, RuntimeError), match="fdopen construction failure"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    with pytest.raises(OSError) as closed_error:
        original_fstat(failed_descriptor["value"])
    assert closed_error.value.errno == 9


@pytest.mark.parametrize("failed_staging", [1, 2])
def test_publish_query_mask_checkpoint_owns_staging_before_identity_fstat(
        tmp_path, monkeypatch, failed_staging):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    original_open = trainer._os.open
    original_fstat = trainer._os.fstat
    staging_creations = []
    target = {}
    primary_error = OSError("injected staging identity fstat failure")

    def record_staging_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if ".staging." in str(path):
            staging_creations.append((descriptor, path))
        if (len(staging_creations) == failed_staging
                and not target):
            target.update({
                "descriptor": descriptor,
                "name": path,
                "armed": True,
            })
        return descriptor

    def fail_selected_identity_fstat(descriptor):
        if target.get("armed") and descriptor == target["descriptor"]:
            target["armed"] = False
            raise primary_error
        return original_fstat(descriptor)

    monkeypatch.setattr(trainer._os, "open", record_staging_open)
    monkeypatch.setattr(trainer._os, "fstat", fail_selected_identity_fstat)

    caught = None
    try:
        trainer.publish_query_mask_checkpoint(destination, artifact)
    except BaseException as error:
        caught = error

    assert caught is primary_error
    closed_errno = None
    try:
        original_fstat(target["descriptor"])
    except OSError as error:
        closed_errno = error.errno
    remaining_staging = tuple(tmp_path.glob("*.staging.*"))
    assert (closed_errno, remaining_staging) == (9, ())
    assert not destination.exists()
    assert not receipt_path.exists()


@pytest.mark.parametrize("failed_staging", [1, 2])
def test_publish_identity_fstat_cleanup_preserves_replaced_staging(
        tmp_path, monkeypatch, failed_staging):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    original_open = trainer._os.open
    original_fstat = trainer._os.fstat
    target = {}
    staging_count = []
    competitor_bytes = b"replacement staging owner\n"
    primary_error = OSError("injected staging identity fstat failure")

    def record_staging_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if ".staging." in str(path):
            staging_count.append(path)
        if len(staging_count) == failed_staging and not target:
            target.update({
                "parent_fd": kwargs["dir_fd"],
                "descriptor": descriptor,
                "name": path,
                "armed": True,
            })
        return descriptor

    def replace_name_then_fail_fstat(descriptor):
        if target.get("armed") and descriptor == target["descriptor"]:
            target["armed"] = False
            os.unlink(target["name"], dir_fd=target["parent_fd"])
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            competitor_fd = os.open(
                target["name"], flags, 0o600, dir_fd=target["parent_fd"]
            )
            try:
                os.write(competitor_fd, competitor_bytes)
            finally:
                os.close(competitor_fd)
            competitor_stat = os.stat(
                target["name"],
                dir_fd=target["parent_fd"],
                follow_symlinks=False,
            )
            target["competitor_identity"] = (
                competitor_stat.st_dev, competitor_stat.st_ino
            )
            raise primary_error
        return original_fstat(descriptor)

    monkeypatch.setattr(trainer._os, "open", record_staging_open)
    monkeypatch.setattr(trainer._os, "fstat", replace_name_then_fail_fstat)

    caught = None
    try:
        trainer.publish_query_mask_checkpoint(destination, artifact)
    except BaseException as error:
        caught = error

    assert caught is primary_error
    with pytest.raises(OSError) as closed_error:
        original_fstat(target["descriptor"])
    assert closed_error.value.errno == 9
    competitor_path = tmp_path / target["name"]
    assert competitor_path.read_bytes() == competitor_bytes
    competitor_stat = competitor_path.stat()
    assert (competitor_stat.st_dev, competitor_stat.st_ino) == target[
        "competitor_identity"
    ]
    assert tuple(tmp_path.glob("*.staging.*")) == (competitor_path,)
    assert not destination.exists()
    assert not receipt_path.exists()


@pytest.mark.parametrize("primary_type", [KeyboardInterrupt, SystemExit])
def test_open_checkpoint_parent_closes_all_directories_on_base_exception(
        tmp_path, monkeypatch, primary_type):
    trainer = _trainer()
    original_open = trainer._os.open
    original_close = trainer._os.close
    original_fstat = trainer._os.fstat
    opened_directories = []
    close_attempts = []
    close_errors = []
    primary_error = primary_type("injected child identity failure")
    primary_traceback = {}
    primary_started = []

    def record_directory_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags == trainer._CHECKPOINT_DIRECTORY_FLAGS:
            opened_directories.append(descriptor)
        return descriptor

    def fail_child_identity_fstat(descriptor):
        if (not primary_started
                and len(opened_directories) >= 2
                and descriptor == opened_directories[1]):
            primary_started.append(True)
            try:
                raise primary_error
            except BaseException as error:
                primary_traceback["value"] = error.__traceback__
                raise
        return original_fstat(descriptor)

    def close_directory_then_fail(descriptor):
        if primary_started and descriptor in opened_directories[:2]:
            close_attempts.append(descriptor)
            original_close(descriptor)
            close_error = RuntimeError(
                "injected directory close failure {}".format(descriptor)
            )
            close_errors.append(close_error)
            raise close_error
        return original_close(descriptor)

    monkeypatch.setattr(trainer._os, "open", record_directory_open)
    monkeypatch.setattr(trainer._os, "fstat", fail_child_identity_fstat)
    monkeypatch.setattr(trainer._os, "close", close_directory_then_fail)

    caught = None
    try:
        trainer._open_checkpoint_parent(
            tmp_path / "checkpoint.pth", "test checkpoint", create=False
        )
    except BaseException as error:
        caught = error

    descriptor_errnos = []
    for descriptor in opened_directories[:2]:
        try:
            original_fstat(descriptor)
        except OSError as error:
            descriptor_errnos.append(error.errno)
        else:
            descriptor_errnos.append(None)
            original_close(descriptor)

    assert caught is primary_error
    traceback_cursor = caught.__traceback__
    while (traceback_cursor is not None
            and traceback_cursor is not primary_traceback["value"]):
        traceback_cursor = traceback_cursor.tb_next
    assert traceback_cursor is primary_traceback["value"]
    assert close_attempts == [
        opened_directories[1], opened_directories[0]
    ]
    assert descriptor_errnos == [9, 9]
    assert len(close_errors) == 2


def test_open_checkpoint_parent_closes_child_when_parent_close_fails(
        tmp_path, monkeypatch):
    trainer = _trainer()
    original_open = trainer._os.open
    original_close = trainer._os.close
    original_fstat = trainer._os.fstat
    opened_directories = []
    close_attempts = []
    primary_error = RuntimeError("injected parent transition close failure")
    primary_traceback = {}

    def record_directory_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags == trainer._CHECKPOINT_DIRECTORY_FLAGS:
            opened_directories.append(descriptor)
        return descriptor

    def fail_first_parent_close(descriptor):
        if (len(opened_directories) >= 2
                and descriptor in opened_directories[:2]):
            close_attempts.append(descriptor)
        if ("value" not in primary_traceback
                and len(opened_directories) >= 2
                and descriptor == opened_directories[0]):
            original_close(descriptor)
            try:
                raise primary_error
            except BaseException as error:
                primary_traceback["value"] = error.__traceback__
                raise
        return original_close(descriptor)

    monkeypatch.setattr(trainer._os, "open", record_directory_open)
    monkeypatch.setattr(trainer._os, "close", fail_first_parent_close)

    caught = None
    try:
        trainer._open_checkpoint_parent(
            tmp_path / "checkpoint.pth", "test checkpoint", create=False
        )
    except BaseException as error:
        caught = error

    descriptor_errnos = []
    for descriptor in opened_directories[:2]:
        try:
            original_fstat(descriptor)
        except OSError as error:
            descriptor_errnos.append(error.errno)
        else:
            descriptor_errnos.append(None)
            original_close(descriptor)

    assert caught is primary_error
    traceback_cursor = caught.__traceback__
    while (traceback_cursor is not None
            and traceback_cursor is not primary_traceback["value"]):
        traceback_cursor = traceback_cursor.tb_next
    assert traceback_cursor is primary_traceback["value"]
    assert close_attempts == [
        opened_directories[0], opened_directories[1]
    ]
    assert descriptor_errnos == [9, 9]


def test_open_checkpoint_child_preserves_identity_error_when_close_fails(
        tmp_path, monkeypatch):
    trainer = _trainer()
    child_path = tmp_path / "child"
    child_path.mkdir()
    original_open = trainer._os.open
    original_close = trainer._os.close
    original_fstat = trainer._os.fstat
    original_identity = trainer._checkpoint_inode_identity
    parent_fd = original_open(
        str(tmp_path), trainer._CHECKPOINT_DIRECTORY_FLAGS
    )
    child_descriptor = {}
    identity_calls = []
    close_attempts = []
    close_error = RuntimeError("injected child close failure")

    def record_child_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        child_descriptor["value"] = descriptor
        return descriptor

    def report_mismatched_identity(file_stat):
        identity = original_identity(file_stat)
        identity_calls.append(identity)
        if len(identity_calls) == 2:
            return (identity[0], identity[1] + 1)
        return identity

    def close_child_then_fail(descriptor):
        if descriptor == child_descriptor.get("value"):
            close_attempts.append(descriptor)
            original_close(descriptor)
            raise close_error
        return original_close(descriptor)

    monkeypatch.setattr(trainer._os, "open", record_child_open)
    monkeypatch.setattr(
        trainer, "_checkpoint_inode_identity", report_mismatched_identity
    )
    monkeypatch.setattr(trainer._os, "close", close_child_then_fail)

    caught = None
    try:
        trainer._open_checkpoint_child_directory(
            parent_fd, child_path.name, "test checkpoint"
        )
    except BaseException as error:
        caught = error
    finally:
        original_close(parent_fd)

    assert caught is not close_error
    assert isinstance(caught, RuntimeError)
    assert "ancestor changed while being opened" in str(caught)
    assert close_attempts == [child_descriptor["value"]]
    with pytest.raises(OSError) as closed_error:
        original_fstat(child_descriptor["value"])
    assert closed_error.value.errno == 9


def test_load_query_mask_checkpoint_closes_fd_when_initial_fstat_fails(
        tmp_path, monkeypatch):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=2e-5
    )
    original_fstat = trainer._os.fstat
    failed_descriptor = {}

    def fail_first_regular_file_fstat(descriptor):
        descriptor_stat = original_fstat(descriptor)
        if (not failed_descriptor
                and stat.S_ISREG(descriptor_stat.st_mode)):
            failed_descriptor["value"] = descriptor
            raise OSError("injected initial fstat failure")
        return descriptor_stat

    monkeypatch.setattr(trainer._os, "fstat", fail_first_regular_file_fstat)

    with pytest.raises(
            (OSError, ValueError), match="injected initial fstat failure"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    with pytest.raises(OSError) as closed_error:
        original_fstat(failed_descriptor["value"])
    assert closed_error.value.errno == 9


def test_load_query_mask_checkpoint_rejects_missing_receipt_before_copy(
        tmp_path, monkeypatch):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)
    receipt_path = tmp_path / "checkpoint.pth.receipt.json"
    receipt_path.unlink()

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=2e-5
    )
    copy_calls = []

    def reject_patch_copy(*args, **kwargs):
        copy_calls.append((args, kwargs))
        raise AssertionError("patch copy must not run without a receipt")

    monkeypatch.setattr(
        trainer, "_restore_query_mask_patch_state", reject_patch_copy
    )

    with pytest.raises(ValueError, match="receipt.*unavailable"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    assert copy_calls == []


def test_load_query_mask_checkpoint_restores_patch_and_optimizer_only(
        tmp_path):
    trainer = _trainer()
    torch.manual_seed(47)
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5, weight_decay=5e-4
    )
    loss = sum(
        (index + 1) * parameter.square().sum()
        for index, parameter in enumerate(source_group["parameters"])
    )
    loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "x_query_step_000003.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    torch.manual_seed(53)
    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=9e-4, weight_decay=0.0
    )
    frozen_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
        if not name.startswith("x_query.")
    }

    loaded = trainer.load_query_mask_checkpoint(
        destination,
        restored,
        restored_optimizer,
        expected_base_checkpoint=_base_checkpoint_binding(),
    )

    assert loaded["schema"] == "mcln-x-query-checkpoint-v1"
    restored_state = restored.state_dict()
    for name, expected in artifact["x_query_state_dict"].items():
        assert torch.equal(restored_state[name].cpu(), expected)
    for name, expected in frozen_before.items():
        assert torch.equal(restored_state[name], expected)
    _assert_nested_equal(
        restored_optimizer.state_dict(), artifact["optimizer_state_dict"]
    )
    assert restored.training is False
    assert tuple(
        name for name, parameter in restored.named_parameters()
        if parameter.requires_grad
    ) == restored_group["names"]


def test_build_query_mask_checkpoint_binds_optimizer_type_and_state_contract():
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    loss = sum(parameter.square().sum() for parameter in group["parameters"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )

    contract = artifact["optimizer_contract"]
    assert contract["type"] == {
        "module": type(optimizer).__module__,
        "qualname": type(optimizer).__qualname__,
    }
    assert contract["parameter_groups"] == (group["names"],)
    assert tuple(contract["state"]) == group["names"]


def test_load_query_mask_checkpoint_rejects_different_optimizer_unchanged(
        tmp_path):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    loss = sum(
        parameter.square().sum() for parameter in source_group["parameters"]
    )
    loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.SGD(
        restored_group["parameters"], lr=0.1
    )
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = restored_optimizer.state_dict()

    with pytest.raises(ValueError, match="optimizer type"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    for name, expected in model_before.items():
        assert torch.equal(restored.state_dict()[name], expected)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


def test_load_query_mask_checkpoint_rejects_wrong_optimizer_tensor_shape(
        tmp_path):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    loss = sum(
        parameter.square().sum() for parameter in source_group["parameters"]
    )
    loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    first_state = next(iter(artifact["optimizer_state_dict"]["state"].values()))
    first_state["exp_avg"] = torch.zeros(1)
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=9e-4
    )
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = restored_optimizer.state_dict()

    with pytest.raises(ValueError, match="optimizer state tensor.*shape"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    for name, expected in model_before.items():
        assert torch.equal(restored.state_dict()[name], expected)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


def test_load_query_mask_checkpoint_rejects_optimizer_dtype_before_copy(
        tmp_path, monkeypatch):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    loss = sum(
        parameter.square().sum() for parameter in source_group["parameters"]
    )
    loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    first_state = next(iter(artifact["optimizer_state_dict"]["state"].values()))
    first_state["exp_avg"] = first_state["exp_avg"].to(torch.float64)
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=9e-4
    )
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = restored_optimizer.state_dict()

    def reject_patch_copy(*_args, **_kwargs):
        raise AssertionError("patch copy ran before optimizer dtype rejection")

    monkeypatch.setattr(
        trainer, "_restore_query_mask_patch_state", reject_patch_copy
    )
    with pytest.raises(ValueError, match="optimizer state mismatch"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    for name, expected in model_before.items():
        assert torch.equal(restored.state_dict()[name], expected)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


def test_load_query_mask_checkpoint_rolls_back_post_load_contract_failure(
        tmp_path, monkeypatch):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    source_loss = sum(
        parameter.square().sum() for parameter in source_group["parameters"]
    )
    source_loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=9e-4, weight_decay=0.0
    )
    restored_loss = sum(
        (index + 1) * parameter.square().sum()
        for index, parameter in enumerate(restored_group["parameters"])
    )
    restored_loss.backward()
    restored_optimizer.step()
    restored_optimizer.zero_grad()
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = trainer._cpu_checkpoint_clone(
        restored_optimizer.state_dict(), "test optimizer state"
    )
    original_load_state_dict = restored_optimizer.load_state_dict
    load_calls = []

    def load_then_corrupt_once(state_dict):
        result = original_load_state_dict(state_dict)
        load_calls.append(state_dict)
        if len(load_calls) == 1:
            first_state = next(iter(restored_optimizer.state.values()))
            first_state["exp_avg"] = first_state["exp_avg"].to(torch.float64)
        return result

    monkeypatch.setattr(
        restored_optimizer, "load_state_dict", load_then_corrupt_once
    )
    with pytest.raises(ValueError, match="optimizer state mismatch"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    assert len(load_calls) == 2
    for name, expected in model_before.items():
        assert torch.equal(restored.state_dict()[name], expected)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


@pytest.mark.parametrize(
    "failed_actions",
    [
        ("patch",),
        ("optimizer",),
        ("eval",),
        ("patch", "optimizer", "eval"),
    ],
)
def test_load_query_mask_checkpoint_reports_all_rollback_failures(
        tmp_path, monkeypatch, failed_actions):
    trainer = _trainer()
    torch.manual_seed(71)
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    source_loss = sum(
        parameter.square().sum() for parameter in source_group["parameters"]
    )
    source_loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    torch.manual_seed(73)
    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=9e-4, weight_decay=0.0
    )
    restored_loss = sum(
        (index + 1) * parameter.square().sum()
        for index, parameter in enumerate(restored_group["parameters"])
    )
    restored_loss.backward()
    restored_optimizer.step()
    restored_optimizer.zero_grad()
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = trainer._cpu_checkpoint_clone(
        restored_optimizer.state_dict(), "test optimizer state"
    )

    primary_error = RuntimeError("post-load contract failure")
    rollback_error_by_action = {
        action: RuntimeError("{} rollback failure".format(action))
        for action in failed_actions
    }
    validate_calls = []
    patch_calls = []
    optimizer_calls = []
    eval_calls = []
    original_validate = trainer._validate_optimizer_checkpoint_contract
    original_restore_patch = trainer._restore_query_mask_patch_state
    original_load_optimizer = restored_optimizer.load_state_dict
    original_eval = restored.eval

    def fail_post_load_validation(*args, **kwargs):
        validate_calls.append(True)
        if len(validate_calls) == 2:
            raise primary_error
        return original_validate(*args, **kwargs)

    def restore_patch_with_failure(*args, **kwargs):
        patch_calls.append(True)
        if len(patch_calls) == 2 and "patch" in failed_actions:
            raise rollback_error_by_action["patch"]
        return original_restore_patch(*args, **kwargs)

    def load_optimizer_with_failure(state_dict):
        optimizer_calls.append(True)
        if len(optimizer_calls) == 2 and "optimizer" in failed_actions:
            raise rollback_error_by_action["optimizer"]
        return original_load_optimizer(state_dict)

    def eval_with_failure():
        eval_calls.append(True)
        if "eval" in failed_actions:
            raise rollback_error_by_action["eval"]
        return original_eval()

    monkeypatch.setattr(
        trainer,
        "_validate_optimizer_checkpoint_contract",
        fail_post_load_validation,
    )
    monkeypatch.setattr(
        trainer, "_restore_query_mask_patch_state", restore_patch_with_failure
    )
    monkeypatch.setattr(
        restored_optimizer, "load_state_dict", load_optimizer_with_failure
    )
    monkeypatch.setattr(restored, "eval", eval_with_failure)

    caught = None
    try:
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )
    except RuntimeError as error:
        caught = error

    assert type(caught).__name__ == "QueryMaskCheckpointRollbackError"
    assert caught.original_error is primary_error
    assert caught.__cause__ is primary_error
    expected_rollback_errors = tuple(
        (action, rollback_error_by_action[action])
        for action in ("patch", "optimizer", "eval")
        if action in failed_actions
    )
    assert caught.rollback_errors == expected_rollback_errors
    assert "post-load contract failure" in str(caught)
    for action, rollback_error in expected_rollback_errors:
        assert action in str(caught)
        assert str(rollback_error) in str(caught)
    assert len(patch_calls) == 2
    assert len(optimizer_calls) == 2
    assert len(eval_calls) == 1
    if "patch" not in failed_actions:
        for name, expected in model_before.items():
            assert torch.equal(restored.state_dict()[name], expected)
    if "optimizer" not in failed_actions:
        _assert_nested_equal(
            restored_optimizer.state_dict(), optimizer_before
        )
    if "eval" not in failed_actions:
        assert restored.training is False


@pytest.mark.parametrize("frozen_kind", ["parameter", "buffer"])
def test_load_query_mask_checkpoint_rejects_patch_buffer_frozen_storage_alias(
        tmp_path, frozen_kind):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source.x_query.register_buffer("storage_probe", torch.zeros(3))
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=2e-5
    )
    if frozen_kind == "parameter":
        frozen_tensor = restored.detector.weight.view(-1)[:3]
    else:
        restored.detector.register_buffer(
            "frozen_probe", torch.tensor([19.0, 23.0, 29.0, 31.0])
        )
        frozen_tensor = restored.detector.frozen_probe[:3]
    restored.x_query.register_buffer("storage_probe", frozen_tensor)
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = restored_optimizer.state_dict()

    with pytest.raises(ValueError, match="storage alias"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    for name, expected in model_before.items():
        assert torch.equal(restored.state_dict()[name], expected)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


@pytest.mark.parametrize("frozen_kind", ["parameter", "buffer"])
def test_load_query_mask_checkpoint_rejects_patch_parameter_frozen_alias(
        tmp_path, frozen_kind):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    destination = tmp_path / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=2e-5
    )
    if frozen_kind == "parameter":
        frozen_tensor = restored.detector.weight.view(-1)[:2]
    else:
        restored.detector.register_buffer(
            "frozen_probe", torch.tensor([19.0, 23.0, 29.0])
        )
        frozen_tensor = restored.detector.frozen_probe[:2]
    restored.x_query[2].bias.data = frozen_tensor
    model_before = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
    }
    optimizer_before = restored_optimizer.state_dict()

    with pytest.raises(ValueError, match="storage alias"):
        trainer.load_query_mask_checkpoint(
            destination,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )

    for name, expected in model_before.items():
        assert torch.equal(restored.state_dict()[name], expected)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


def test_publish_query_mask_checkpoint_rejects_dangling_symlink(tmp_path):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    protected_target = tmp_path / "protected_best.pth"
    destination = tmp_path / "x_query_step_000003.pth"
    destination.symlink_to(protected_target)

    try:
        trainer.publish_query_mask_checkpoint(destination, artifact)
    except ValueError as error:
        assert "symlink" in str(error)
    else:
        raise AssertionError("publisher followed a dangling output symlink")

    assert destination.is_symlink()
    assert not protected_target.exists()
    assert not (tmp_path / "protected_best.pth.receipt.json").exists()


def test_publish_query_mask_checkpoint_rejects_symlink_ancestor_without_mkdir(
        tmp_path):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    protected_parent = tmp_path / "protected"
    protected_parent.mkdir()
    redirected_parent = tmp_path / "redirected"
    redirected_parent.symlink_to(protected_parent, target_is_directory=True)
    destination = redirected_parent / "created" / "checkpoint.pth"

    with pytest.raises(ValueError, match="symlink"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    assert not (protected_parent / "created").exists()


def test_publish_query_mask_checkpoint_does_not_normalize_away_symlink(
        tmp_path):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    artifact = trainer.build_query_mask_checkpoint(
        model,
        optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    protected_parent = tmp_path / "protected"
    protected_parent.mkdir()
    redirected_parent = tmp_path / "redirected"
    redirected_parent.symlink_to(protected_parent, target_is_directory=True)
    destination = redirected_parent / ".." / "escaped.pth"

    with pytest.raises(ValueError, match="canonical"):
        trainer.publish_query_mask_checkpoint(destination, artifact)

    assert not (tmp_path / "escaped.pth").exists()


def test_load_query_mask_checkpoint_rejects_symlink_ancestor(tmp_path):
    trainer = _trainer()
    source = _ArtifactToyMCLN()
    source_group = trainer.configure_query_mask_head_trainability(source)
    source_optimizer = torch.optim.AdamW(
        source_group["parameters"], lr=2e-5
    )
    artifact = trainer.build_query_mask_checkpoint(
        source,
        source_optimizer,
        base_checkpoint=_base_checkpoint_binding(),
        run_record=_run_record(),
    )
    protected_parent = tmp_path / "protected"
    protected_parent.mkdir()
    destination = protected_parent / "checkpoint.pth"
    trainer.publish_query_mask_checkpoint(destination, artifact)
    redirected_parent = tmp_path / "redirected"
    redirected_parent.symlink_to(protected_parent, target_is_directory=True)

    restored = _ArtifactToyMCLN()
    restored_group = trainer.configure_query_mask_head_trainability(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_group["parameters"], lr=2e-5
    )
    with pytest.raises(ValueError, match="symlink"):
        trainer.load_query_mask_checkpoint(
            redirected_parent / destination.name,
            restored,
            restored_optimizer,
            expected_base_checkpoint=_base_checkpoint_binding(),
        )


@pytest.mark.parametrize(
    "missing",
    [
        "run_id",
        "step",
        "epoch",
        "config",
        "source_sha256",
        "split_digest",
        "protected_artifacts",
        "train_metrics",
    ],
)
def test_build_query_mask_checkpoint_requires_reproduction_record(missing):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    run_record = _run_record()
    del run_record[missing]

    with pytest.raises(ValueError, match="run record"):
        trainer.build_query_mask_checkpoint(
            model,
            optimizer,
            base_checkpoint=_base_checkpoint_binding(),
            run_record=run_record,
        )


def test_build_query_mask_checkpoint_rejects_boolean_train_metric():
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    run_record = _run_record()
    run_record["train_metrics"]["total_loss"] = True

    with pytest.raises(ValueError, match="train_metrics"):
        trainer.build_query_mask_checkpoint(
            model,
            optimizer,
            base_checkpoint=_base_checkpoint_binding(),
            run_record=run_record,
        )


def test_build_query_mask_checkpoint_rejects_non_finite_optimizer_state():
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    loss = sum(parameter.square().sum() for parameter in group["parameters"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"].fill_(float("nan"))

    with pytest.raises(ValueError, match="optimizer.*finite"):
        trainer.build_query_mask_checkpoint(
            model,
            optimizer,
            base_checkpoint=_base_checkpoint_binding(),
            run_record=_run_record(),
        )


def test_build_query_mask_checkpoint_rejects_non_finite_patch_weight():
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)
    with torch.no_grad():
        model.x_query[0].weight[0, 0] = float("nan")

    with pytest.raises(ValueError, match="patch tensor.*finite"):
        trainer.build_query_mask_checkpoint(
            model,
            optimizer,
            base_checkpoint=_base_checkpoint_binding(),
            run_record=_run_record(),
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_build_query_mask_checkpoint_rejects_non_finite_complex_patch(
        non_finite):
    trainer = _trainer()
    model = _ArtifactToyMCLN()
    model.x_query.register_buffer(
        "complex_probe",
        torch.tensor([complex(non_finite, 0.0)], dtype=torch.complex64),
    )
    group = trainer.configure_query_mask_head_trainability(model)
    optimizer = torch.optim.AdamW(group["parameters"], lr=2e-5)

    with pytest.raises(ValueError, match="patch tensor.*finite"):
        trainer.build_query_mask_checkpoint(
            model,
            optimizer,
            base_checkpoint=_base_checkpoint_binding(),
            run_record=_run_record(),
        )
