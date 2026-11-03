#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
sys.path.insert(0, "/root/autodl-tmp/mcln_pvground_ft_20261101")
import retain_mcln_best_latest as retention
with tempfile.TemporaryDirectory(prefix="mcln-retention-fixture-") as temporary:
    retention.ROOT = Path(temporary)
    run = retention.ROOT / "scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/fixture"
    run.mkdir(parents=True)
    assert retention.retain("scanrefer", False)["state"] == "waiting_for_stage_b_validation"
    preserved = run / "preserved_checkpoints"
    preserved.mkdir()
    rows = [
        {"epoch": 1, "hits025": 5400, "hits050": 4600},
        {"epoch": 2, "hits025": 5500, "hits050": 4500},
        {"epoch": 3, "hits025": 5450, "hits050": 4550},
    ]
    (run / "stage_b_semantic_history.json").write_text(json.dumps({"rows": rows}))
    (run / "semantic_best.json").write_text(json.dumps(rows[0]))
    (run / "config.json").write_text(json.dumps({
        "model_only_initialization": True, "use_pvground_innovations": True,
        "pvground_train_only": False,
    }))
    for epoch in (1, 2, 3):
        path = run / f"ckpt_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch, "model": {str(i): torch.zeros(1) for i in range(1168)},
            "optimizer": {"state": {0: {"step": 1}}}, "scheduler": {"last_epoch": 1},
        }, path)
        os.link(path, preserved / path.name)
    os.link(run / "ckpt_epoch_1.pth", run / "semantic_best.pth")
    plan = retention.retain("scanrefer", False)
    assert plan["keep_epochs"] == [1, 3] and len(plan["candidates"]) == 2
    assert (preserved / "ckpt_epoch_2.pth").exists()
    applied = retention.retain("scanrefer", True)
    assert applied["keep_epochs"] == [1, 3]
    assert sorted(path.name for path in preserved.glob("*.pth")) == [
        "ckpt_epoch_1.pth", "ckpt_epoch_3.pth",
    ]
    assert not (run / "ckpt_epoch_2.pth").exists()
    assert retention.retain("scanrefer", True)["state"] == "unchanged"
    rows[1]["hits050"] = 4700
    (run / "stage_b_semantic_history.json").write_text(json.dumps({"rows": rows}))
    assert retention.retain("scanrefer", True)["state"] == "waiting_for_metric_keeper"
assert not torch.cuda.is_initialized()
print(json.dumps({"passed": True, "checks": [
    "wait_for_complete_validation", "scanrefer_acc050_selection",
    "dry_run_preserves_files", "delete_only_older_preserved_epoch",
    "best_and_latest_resumable_metadata", "unchanged_no_reaudit",
    "wait_for_best_manifest", "cuda_uninitialized",
]}))
PY
