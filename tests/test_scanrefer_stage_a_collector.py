import argparse
import hashlib
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


def terminal_pipeline(stage, exit_code=0):
    (stage / "state.json").write_text(json.dumps({"phase": "stage_a_complete"}))
    (stage / "pipeline_exit_code").write_text(str(exit_code))


def decision_rows(stage, hits):
    rows = []
    for epoch, (hits025, hits050) in enumerate(hits, start=1):
        checkpoint = stage / "fixture_epoch_{}.pth".format(epoch)
        checkpoint.write_bytes("checkpoint{}".format(epoch).encode())
        rows.append({
            "epoch": epoch, "hits025": hits025, "hits050": hits050,
            "checkpoint": str(checkpoint),
        })
    return rows


def test_transition_waits_for_successful_three_epoch_completion(tmp_path):
    rows = decision_rows(tmp_path, [(5705, 4945)])
    assert collector.stage_b_decision(tmp_path, rows)["status"] == "pending"
    terminal_pipeline(tmp_path)
    with pytest.raises(AssertionError, match="three complete audited epochs"):
        collector.stage_b_decision(tmp_path, rows)
    terminal_pipeline(tmp_path, exit_code=1)
    assert collector.stage_b_decision(tmp_path, rows)["status"] == "pipeline_failed"


def test_transition_never_combines_hits_from_different_checkpoints(tmp_path):
    rows = decision_rows(tmp_path, [(5424, 5000), (6000, 4327), (5424, 4327)])
    terminal_pipeline(tmp_path)
    result = collector.stage_b_decision(tmp_path, rows)
    assert result["status"] == "gate_failed" and "selected" not in result
    assert not list(tmp_path.glob("stage_b_initialization*"))


@pytest.mark.parametrize("hits, selected_index", [
    ([(5424, 5000), (5425, 4328), (5435, 4328)], 2),
    ([(5424, 5000), (5425, 4329), (5435, 4328)], 1),
    ([(5424, 5000), (5435, 4328), (5435, 4328)], 1),
])
def test_transition_selects_eligible_pair_preserves_and_hashes_it(
        tmp_path, hits, selected_index):
    rows = decision_rows(tmp_path, hits)
    terminal_pipeline(tmp_path)
    result = collector.stage_b_decision(tmp_path, rows)
    assert result["status"] == "ready" and not result["stage_b_started"]
    assert result["selected"] == rows[selected_index]
    saved = Path(result["initialization_checkpoint"])
    assert saved.samefile(rows[selected_index]["checkpoint"])
    assert result["initialization_sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert collector.stage_b_decision(tmp_path, rows) == result
