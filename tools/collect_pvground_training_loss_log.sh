#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import re
from pathlib import Path

base = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/runs")
for dataset in ("scanrefer", "nr3d", "sr3d"):
    root = base / dataset
    path = root / "train.log"
    if not path.exists():
        continue
    rows = {}
    current = None
    waiting = False
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"Train: \[(\d+)\]\[(\d+)/(\d+)\]", line)
        if match:
            epoch, batch, total = map(int, match.groups())
            current = rows.setdefault(epoch, {"epoch": epoch})
            current.update(last_print_batch=batch, total_train_batches=total)
            waiting = True
        if waiting and re.search(r"\]: .*?\bloss\b", line):
            current["periodic_loss_line"] = line
            current["periodic_running_means"] = {
                key: float(value) for key, value in
                re.findall(r"([A-Za-z_]+loss[A-Za-z_]*|loss[A-Za-z_]*)\s+([0-9.eE+-]+)", line)
            }
            waiting = False
        match = re.search(
            r"epoch\s+(\d+), total time ([0-9.]+), lr_base ([0-9.eE+-]+), lr_pointnet ([0-9.eE+-]+)",
            line,
        )
        if match:
            epoch = int(match[1])
            rows.setdefault(epoch, {"epoch": epoch}).update(
                completed_training=True,
                train_seconds=float(match[2]),
                printed_lr_base=match[3],
                printed_lr_pointnet=match[4],
            )
    receipt = {
        "dataset": dataset,
        "source_log": str(path),
        "limitation": "Last periodic running mean per epoch, not exact final epoch mean; printed LR precision is limited.",
        "epochs": list(rows.values()),
    }
    destination = root / "periodic_training_loss_observations.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "dataset": dataset,
        "receipt": str(destination),
        "limitation": receipt["limitation"],
        "epochs": [
            {
                "epoch": row["epoch"],
                "last_print_batch": row["last_print_batch"],
                "total_train_batches": row["total_train_batches"],
                "losses": {
                    key: row["periodic_running_means"][key]
                    for key in (
                        "loss", "loss_bbox", "loss_giou", "loss_sem_align",
                        "sp_loss_dice", "sp_loss_mask",
                    )
                },
            }
            for row in rows.values()
        ],
    }, sort_keys=True))
PY
