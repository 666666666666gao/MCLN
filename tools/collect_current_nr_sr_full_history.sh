#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
sys.path.insert(0, str(root))
from mcln_metric_keeper import completed_rows

cases = (
    ("nr3d", "1793553113", 240, 4, 96960),
    ("sr3d", "1793552465", 140, 2, 92354),
)

for dataset, run_id, budget, accumulation, trainer_pid in cases:
    if not (Path("/proc") / str(trainer_pid)).exists():
        continue
    directory = root / "runs" / dataset
    run = directory / dataset / f"mcln_pvground_{dataset}_20261101" / run_id
    if not run.exists():
        continue

    config = json.loads((run / "config.json").read_text())
    assert config["checkpoint_path"] is None
    assert config["pp_checkpoint"] is None
    assert not config["use_pvground_innovations"]
    assert config["max_epoch"] == budget
    assert config["batch_size"] == 12
    assert config["gradient_accumulation_steps"] == accumulation

    log = directory / "train.log"
    rows = completed_rows(log.read_text(errors="replace"), dataset)
    order = ("hits025", "hits050")
    best = max(rows, key=lambda row: (row[order[0]], row[order[1]], -row["epoch"]))
    latest_epoch = max(row["epoch"] for row in rows)
    latest = next(row for row in rows if row["epoch"] == latest_epoch)
    manifest = json.loads((run / "semantic_best.json").read_text())
    assert (manifest["epoch"], manifest["hits025"], manifest["hits050"]) == (
        best["epoch"], best["hits025"], best["hits050"])

    preserved = directory / "preserved_checkpoints"
    audits = []
    for epoch in sorted({best["epoch"], latest_epoch}):
        checkpoint = run / (
            "semantic_best.pth" if epoch == best["epoch"]
            else "ckpt_epoch_last.pth"
        )
        retained_path = preserved / f"ckpt_epoch_{epoch}.pth"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["epoch"] == epoch
        assert len(saved["model"]) == 1134
        assert saved["optimizer"]["state"]
        assert saved["scheduler"]
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        audits.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "preserved_checkpoint": str(retained_path),
            "preserved_exists": retained_path.is_file(),
            "sha256": digest.hexdigest(),
            "bytes": checkpoint.stat().st_size,
            "links": checkpoint.stat().st_nlink,
            "own_cpu_epoch": saved["epoch"],
            "model_tensors": len(saved["model"]),
            "nonempty_optimizer": True,
            "has_scheduler": True,
        })
        del saved

    for row in rows:
        checkpoint = preserved / f"ckpt_epoch_{row['epoch']}.pth"
        row["checkpoint_exists_at_collection"] = checkpoint.exists()
        row["checkpoint"] = str(checkpoint)

    retention_path = directory / "best_latest_retention.json"
    retention_history_path = directory / "best_latest_retention_history.jsonl"
    retained_history = [
        json.loads(line)
        for line in retention_history_path.read_text().splitlines()
        if line.strip()
    ] if retention_history_path.exists() else []
    retention = (
        json.loads(retention_path.read_text()) if retention_path.exists() else None
    )
    receipt = {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset,
        "run": run_id,
        "rows": rows,
        "best": best,
        "latest_epoch": latest_epoch,
        "selection_manifest": manifest,
        "config": config,
        "retained_checkpoint_audits": audits,
        "retention": retention,
        "native_checkpoint_retention": json.loads(
            (run / "checkpoint_retention.json").read_text()
        ),
        "retention_history": retained_history,
        "trainer_pid": trainer_pid,
        "trainer_live": (Path("/proc") / str(trainer_pid)).exists(),
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader",
        ], text=True).strip(),
        "screen": subprocess.run(
            ["screen", "-ls"], capture_output=True, text=True
        ).stdout,
        "cuda_initialized": torch.cuda.is_initialized(),
        "scope": (
            "Exact registered NR3D/SR3D runs only. Completed semantic Top1 "
            "log rows and current best/latest own-CPU checkpoints audited. "
            "No ScanRefer, model construction, forward, evaluation or training."
        ),
    }
    assert not receipt["cuda_initialized"]
    destination = run / "current_full_history_20261103.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "dataset": dataset,
        "epochs": len(rows),
        "best": best,
        "latest": latest,
        "audits": audits,
        "retention_latest": (
            receipt["retention"]["latest_epoch"] if receipt["retention"] else None
        ),
        "destination": str(destination),
        "trainer_live": receipt["trainer_live"],
    }))
PY
