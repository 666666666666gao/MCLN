"""Export complete Stage A semantic rows with their own checkpoint metadata."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from mcln_metric_keeper import completed_rows


def stage_b_decision(stage, rows):
    state_path = stage / "state.json"
    exit_path = stage / "pipeline_exit_code"
    phase = json.loads(state_path.read_text())["phase"] if state_path.exists() else None
    exit_code = int(exit_path.read_text()) if exit_path.exists() else None
    epochs = [row["epoch"] for row in rows]
    decision = {
        "status": "pending",
        "complete_epochs": epochs,
        "pipeline_phase": phase,
        "pipeline_exit_code": exit_code,
    }
    if exit_code is not None and exit_code != 0:
        return dict(decision, status="pipeline_failed")
    if phase != "stage_a_complete" or exit_code != 0:
        return decision
    assert epochs == [1, 2, 3], "Stage A requires three complete audited epochs"
    eligible = [
        row for row in rows
        if row["hits050"] >= 4328 and row["hits025"] >= 5425
    ]
    if not eligible:
        return dict(decision, status="gate_failed")
    selected = max(
        eligible, key=lambda row: (row["hits050"], row["hits025"], -row["epoch"])
    )
    source = Path(selected["checkpoint"])
    preserved = stage / "stage_b_initialization_epoch_{}.pth".format(selected["epoch"])
    if not preserved.exists():
        os.link(source, preserved)
    assert os.path.samefile(source, preserved)
    digest = hashlib.sha256()
    with preserved.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return dict(
        decision,
        status="ready",
        selected=selected,
        initialization_checkpoint=str(preserved),
        initialization_sha256=digest.hexdigest(),
        stage_b_started=False,
    )


def collect(stage):
    import torch

    torch.set_num_threads(1)
    run_dir, = [
        path.parent for path in stage.glob(
            "stage_a/scanrefer/scanrefer_stage_a_20261102/*/config.json"
        )
    ]
    config = json.loads((run_dir / "config.json").read_text())
    assert config["pvground_train_only"]
    assert config["expected_eval_sample_count"] == 9508
    gate = json.loads((stage / "e0_gate.json").read_text())
    assert gate["passed"]
    rows = completed_rows(
        (stage / "stage_a_train.log").read_text(errors="replace"), "scanrefer"
    )
    preserved = stage / "preserved_checkpoints"
    preserved.mkdir(exist_ok=True)
    receipts = []
    for row in sorted(rows, key=lambda item: item["epoch"]):
        filename = "ckpt_epoch_{}.pth".format(row["epoch"])
        checkpoint_path = preserved / filename
        if not checkpoint_path.exists():
            os.link(run_dir / filename, checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        assert checkpoint["epoch"] == row["epoch"]
        assert len(checkpoint["model"]) == 1168
        assert "optimizer" in checkpoint and "scheduler" in checkpoint
        assert checkpoint["config"].pvground_train_only
        receipts.append(dict(
            row,
            run_dir=str(run_dir),
            phase="stage_a",
            metric_branch="last_semantic_alignment_top1",
            metric_source="complete_log_semantic_subgroups",
            checkpoint=str(checkpoint_path),
            checkpoint_epoch=checkpoint["epoch"],
            model_tensors=len(checkpoint["model"]),
            has_optimizer=True,
            has_scheduler=True,
            stage_b_eligible=(
                row["hits050"] >= 4328 and row["hits025"] >= 5425
            ),
            final_target_met=(
                row["hits025"] >= 5705 and row["hits050"] >= 4945
            ),
        ))
    result = {
        "run_dir": str(run_dir),
        "e0_gate_passed": True,
        "e0_hits": gate["official"]["hits"],
        "rows": receipts,
        "stage_b_decision": stage_b_decision(stage, receipts),
        "cuda_initialized": torch.cuda.is_initialized(),
        "collector_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    assert not result["cuda_initialized"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.stage)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
