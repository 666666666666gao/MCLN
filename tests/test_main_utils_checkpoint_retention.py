import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from main_utils import (
    CHECKPOINT_RETENTION_METRICS,
    load_checkpoint,
    parse_option,
    save_checkpoint,
    update_checkpoint_retention,
)


def _receipt(
    rec025,
    rec050,
    mask025,
    mask050,
    mask_miou,
    sample_count=100,
):
    return {
        "schema": "mcln-retrain-metrics-v1",
        "sample_count": sample_count,
        "position": {
            "fixed_default": {"hits025": 0, "hits050": 0},
            "learned_selector": {
                "hits025": rec025,
                "hits050": rec050,
            },
        },
        "mask": {
            "hits025": mask025,
            "hits050": mask050,
            "iou_sum": mask_miou * sample_count,
            "miou": mask_miou,
        },
    }


def _write_dummy_checkpoint(directory, epoch):
    path = directory / "ckpt_epoch_{}.pth".format(epoch)
    path.write_bytes("checkpoint {}".format(epoch).encode("ascii"))
    return path


def _assert_same_file(first, second):
    first_stat = os.stat(str(first))
    second_stat = os.stat(str(second))
    assert (first_stat.st_dev, first_stat.st_ino) == (
        second_stat.st_dev,
        second_stat.st_ino,
    )


def test_checkpoint_cli_supports_explicit_start_and_metric_retention(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dist_mod.py",
            "--checkpoint_start_epoch",
            "72",
            "--checkpoint_metric_retention",
        ],
    )

    args = parse_option()

    assert args.checkpoint_start_epoch == 72
    assert args.checkpoint_metric_retention is True


def test_non_numeric_initialization_checkpoint_honors_explicit_start(tmp_path):
    source_model = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "gate_last.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": {},
            "scheduler": {},
            "epoch": "last",
        },
        checkpoint_path,
    )
    target_model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        checkpoint_start_epoch=72,
        start_epoch=1,
        eval=False,
        reduce_lr=True,
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=False,
        use_source_choice_selector=False,
        use_source_moe=False,
        frozen=False,
        small_lr=False,
    )

    load_checkpoint(
        args,
        target_model,
        torch.optim.Adam(target_model.parameters()),
        None,
    )

    assert args.start_epoch == 72
    for expected, actual in zip(
        source_model.parameters(), target_model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_save_checkpoint_is_atomic_and_numeric_epoch_is_resumable(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10], gamma=0.1
    )
    args = SimpleNamespace(log_dir=str(tmp_path), save_freq=1)

    path = save_checkpoint(args, 72, model, optimizer, scheduler)

    assert path == str(tmp_path / "ckpt_epoch_72.pth")
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["epoch"] == 72
    assert checkpoint["optimizer"] == optimizer.state_dict()
    assert checkpoint["scheduler"] == scheduler.state_dict()
    assert not list(tmp_path.glob("*.tmp.*"))


def test_metric_retention_keeps_latest_and_five_independent_bests(tmp_path):
    checkpoint72 = _write_dummy_checkpoint(tmp_path, 72)
    update_checkpoint_retention(
        str(tmp_path), 72, _receipt(60, 50, 70, 55, 0.40)
    )
    _assert_same_file(checkpoint72, tmp_path / "ckpt_epoch_last.pth")
    for metric_name in CHECKPOINT_RETENTION_METRICS:
        _assert_same_file(
            checkpoint72,
            tmp_path / "ckpt_best_{}.pth".format(metric_name),
        )

    checkpoint73 = _write_dummy_checkpoint(tmp_path, 73)
    update_checkpoint_retention(
        str(tmp_path), 73, _receipt(61, 49, 69, 56, 0.41)
    )
    assert checkpoint72.exists()
    assert checkpoint73.exists()
    _assert_same_file(
        checkpoint73, tmp_path / "ckpt_best_rec_acc025.pth"
    )
    _assert_same_file(
        checkpoint72, tmp_path / "ckpt_best_rec_acc050.pth"
    )
    _assert_same_file(
        checkpoint72, tmp_path / "ckpt_best_mask_acc025.pth"
    )
    _assert_same_file(
        checkpoint73, tmp_path / "ckpt_best_mask_acc050.pth"
    )
    _assert_same_file(checkpoint73, tmp_path / "ckpt_best_mask_miou.pth")

    checkpoint74 = _write_dummy_checkpoint(tmp_path, 74)
    update_checkpoint_retention(
        str(tmp_path), 74, _receipt(59, 48, 68, 54, 0.39)
    )
    assert {path.name for path in tmp_path.glob("ckpt_epoch_[0-9]*.pth")} == {
        "ckpt_epoch_72.pth",
        "ckpt_epoch_73.pth",
        "ckpt_epoch_74.pth",
    }
    _assert_same_file(checkpoint74, tmp_path / "ckpt_epoch_last.pth")

    checkpoint75 = _write_dummy_checkpoint(tmp_path, 75)
    report = update_checkpoint_retention(
        str(tmp_path), 75, _receipt(62, 52, 72, 57, 0.42)
    )

    assert {path.name for path in tmp_path.glob("ckpt_epoch_[0-9]*.pth")} == {
        "ckpt_epoch_75.pth"
    }
    assert {os.path.basename(path) for path in report["removed"]} == {
        "ckpt_epoch_72.pth",
        "ckpt_epoch_73.pth",
        "ckpt_epoch_74.pth",
    }
    for alias in [tmp_path / "ckpt_epoch_last.pth"] + [
        tmp_path / "ckpt_best_{}.pth".format(metric_name)
        for metric_name in CHECKPOINT_RETENTION_METRICS
    ]:
        _assert_same_file(checkpoint75, alias)
    manifest = json.loads(
        (tmp_path / "checkpoint_retention.json").read_text()
    )
    assert manifest["latest_epoch"] == 75
    assert set(manifest["records"]) == {"72", "73", "74", "75"}
    assert {
        metric_name: record["epoch"]
        for metric_name, record in manifest["best"].items()
    } == {metric_name: 75 for metric_name in CHECKPOINT_RETENTION_METRICS}
    assert not list(tmp_path.glob("*.tmp.*"))


def test_metric_retention_fails_closed_on_invalid_receipt(tmp_path):
    checkpoint72 = _write_dummy_checkpoint(tmp_path, 72)
    update_checkpoint_retention(
        str(tmp_path), 72, _receipt(60, 50, 70, 55, 0.40)
    )
    _write_dummy_checkpoint(tmp_path, 73)

    with pytest.raises(ValueError, match="hits are invalid"):
        update_checkpoint_retention(
            str(tmp_path), 73, _receipt(101, 50, 70, 55, 0.40)
        )

    manifest = json.loads(
        (tmp_path / "checkpoint_retention.json").read_text()
    )
    assert manifest["latest_epoch"] == 72
    _assert_same_file(checkpoint72, tmp_path / "ckpt_epoch_last.pth")

