import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "stage_a_collector", TOOLS / "collect_scanrefer_stage_a_epochs.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def prepare(stage, checkpoint_epoch=1):
    run = stage / "stage_a/scanrefer/scanrefer_stage_a_20261102/fixture"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "pvground_train_only": True, "expected_eval_sample_count": 9508,
    }))
    (stage / "e0_gate.json").write_text(json.dumps({
        "passed": True, "official": {"hits": {"0.25": 5435, "0.5": 4327}},
    }))
    (stage / "stage_a_train.log").write_text(
        "epoch 1, total time 10\n"
        "last_ semantic alignment Acc0.25: Top-1: {:.5f}\n"
        "last_ semantic alignment Acc0.50: Top-1: {:.5f}\n"
        "[x]: unique {}\n[x]: multi {}\n"
        "[x]: unique50 {}\n[x]: multi50 {}\n"
        "epoch 2, total time 10\n"
        "last_ semantic alignment Acc0.25: Top-1: 0.99999\n".format(
            5425 / 9508, 4328 / 9508,
            1200 / 1419, 4225 / 8089, 1000 / 1419, 3328 / 8089,
        )
    )
    torch.save({
        "epoch": checkpoint_epoch,
        "model": {str(index): torch.tensor(0) for index in range(1168)},
        "config": argparse.Namespace(pvground_train_only=True),
        "optimizer": {}, "scheduler": {},
    }, run / "ckpt_epoch_1.pth")
    return run


def test_complete_same_epoch_transition_is_not_final_success(tmp_path):
    run = prepare(tmp_path)
    receipt = collector.collect(tmp_path)
    row, = receipt["rows"]
    assert row["checkpoint_epoch"] == row["epoch"] == 1
    assert (row["hits025"], row["hits050"]) == (5425, 4328)
    assert row["stage_b_eligible"] and not row["final_target_met"]
    (run / "ckpt_epoch_1.pth").unlink()
    assert collector.collect(tmp_path)["rows"] == receipt["rows"]


def test_wrong_checkpoint_epoch_is_rejected(tmp_path):
    prepare(tmp_path, checkpoint_epoch=2)
    with pytest.raises(AssertionError):
        collector.collect(tmp_path)
