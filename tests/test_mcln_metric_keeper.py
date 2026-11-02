import importlib.util
import json
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "keeper", Path(__file__).parents[1] / "tools" / "mcln_metric_keeper.py"
)
keeper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keeper)


def log_epoch(epoch, hits25, hits50, count=7899):
    return (
        "epoch {}, total time 10\n"
        "last_ semantic alignment Acc0.25: Top-1: {:.5f}\n"
        "last_ semantic alignment Acc0.50: Top-1: {:.5f}\n"
        "[time]: multi {}\n[time]: multi50 {}\n"
    ).format(epoch, hits25 / count, hits50 / count,
             hits25 / count, hits50 / count)


def save_epoch(folder, epoch):
    path = folder / "ckpt_epoch_{}.pth".format(epoch)
    torch.save({"epoch": epoch, "model": {"weight": torch.ones(2)},
                "optimizer": {}, "scheduler": {}}, path)
    return path


def test_keeper_survives_trainer_deletion_and_ignores_worse_epoch(tmp_path):
    log = tmp_path / "train.log"
    first = save_epoch(tmp_path, 1)
    log.write_text(log_epoch(1, 2000, 1000))
    keeper.preserve_best(tmp_path, log, "nr3d")
    first.unlink()
    save_epoch(tmp_path, 2)
    log.write_text(log_epoch(1, 2000, 1000) + log_epoch(2, 1900, 1500))
    receipt = keeper.preserve_best(tmp_path, log, "nr3d")
    assert receipt["epoch"] == 1
    assert torch.load(tmp_path / "semantic_best.pth")["epoch"] == 1
    save_epoch(tmp_path, 3)
    log.write_text(log.read_text() + log_epoch(3, 2100, 900))
    receipt = keeper.preserve_best(tmp_path, log, "nr3d")
    assert receipt["epoch"] == 3
    assert receipt["hits050"] == 900
    assert torch.load(tmp_path / "semantic_best.pth")["epoch"] == 3
    assert json.loads((tmp_path / "semantic_best.json").read_text())["epoch"] == 3


def test_scan_semantic_first_groups_ignore_mask_duplicates(tmp_path):
    count = 9508
    log = tmp_path / "train.log"
    text = (
        "epoch 8, total time 10\n"
        "last_ semantic alignment Acc0.25: Top-1: {:.5f}\n"
        "last_ semantic alignment Acc0.50: Top-1: {:.5f}\n"
        "[x]: unique {}\n[x]: multi {}\n"
        "[x]: unique50 {}\n[x]: multi50 {}\n"
        "[x]: unique50 1.0\n[x]: multi50 1.0\n"
    ).format(5422 / count, 4505 / count, 1234 / 1419, 4188 / 8089,
             1058 / 1419, 3447 / 8089)
    log.write_text(text)
    save_epoch(tmp_path, 8)
    receipt = keeper.preserve_best(tmp_path, log, "scanrefer")
    assert receipt["hits025"] == 5422
    assert receipt["hits050"] == 4505
    assert receipt["primary_metric"] == "hits050"


def test_missing_best_weight_is_not_replaced_with_lower_metric(tmp_path):
    save_epoch(tmp_path, 1)
    log = tmp_path / "train.log"
    log.write_text(log_epoch(1, 2000, 1000) + log_epoch(2, 2500, 1400))
    with pytest.raises(FileNotFoundError, match="no surviving weight"):
        keeper.preserve_best(tmp_path, log, "nr3d")


def test_partial_validation_is_not_selected():
    text = log_epoch(1, 2000, 1000)
    text += "epoch 2, total time 10\n"
    text += "last_ semantic alignment Acc0.25: Top-1: 0.90000\n"
    assert [row["epoch"] for row in keeper.completed_rows(text, "nr3d")] == [1]
