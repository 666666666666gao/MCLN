#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
sys.path.insert(0, str(root))
from mcln_metric_keeper import completed_rows

for dataset, run_id, pids in (
    ("nr3d", "1793553113", (96960, 135075, 156800)),
    ("sr3d", "1793552465", (92354, 129032, 148725)),
):
    if not (Path("/proc") / str(pids[0])).exists():
        continue
    directory = root / "runs" / dataset
    run = directory / dataset / f"mcln_pvground_{dataset}_20261101" / run_id
    if not run.exists():
        continue
    rows = completed_rows((directory / "train.log").read_text(errors="replace"), dataset)
    best_path = run / "semantic_best.json"
    retention_path = directory / "best_latest_retention.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    retention = (
        json.loads(retention_path.read_text()) if retention_path.exists() else None
    )
    checkpoint_dir = directory / "preserved_checkpoints"
    checkpoints = sorted(path.name for path in checkpoint_dir.glob("ckpt_epoch_*.pth"))
    latest = max(rows, key=lambda row: row["epoch"]) if rows else None
    tracked_epochs = set()
    if best is not None:
        tracked_epochs.add(best["epoch"])
    if latest is not None:
        tracked_epochs.add(latest["epoch"])
    retained = []
    for epoch in sorted(tracked_epochs):
        path = checkpoint_dir / f"ckpt_epoch_{epoch}.pth"
        retained.append({"epoch": epoch, "path": str(path), "exists": path.exists()})
    history_path = directory / "best_latest_retention_history.jsonl"
    history = [
        json.loads(line)
        for line in history_path.read_text().splitlines()
        if line.strip()
    ] if history_path.exists() else None
    print(json.dumps({
        "dataset": dataset,
        "run": run_id,
        "processes": {
            str(pid): (Path("/proc") / str(pid)).exists() for pid in pids
        },
        "completed_rows": len(rows),
        "latest_completed_row": latest,
        "best_manifest": best,
        "retention": retention,
        "native_retention": json.loads(
            (run / "checkpoint_retention.json").read_text()
        ),
        "run_files": sorted(path.name for path in run.iterdir()),
        "best_latest_files": retained,
        "preserved_epoch_files": checkpoints,
        "retention_history_tail": history[-6:] if history is not None else None,
        "screen": subprocess.run(
            ["screen", "-ls"], capture_output=True, text=True
        ).stdout,
    }))
PY
