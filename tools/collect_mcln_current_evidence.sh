#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import os
import re
from pathlib import Path
import torch

base = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/runs")
for dataset, total in (("scanrefer", 9508), ("nr3d", 7899), ("sr3d", 17726)):
    root = base / dataset
    log = root / "train.log"
    if not log.exists():
        continue
    rows = {}
    epoch = None
    for line in log.read_text(errors="replace").splitlines():
        match = re.search(r"epoch\s+(\d+), total time", line)
        if match:
            epoch = int(match.group(1))
        if epoch is None:
            continue
        match = re.search(
            r"last_ semantic alignment Acc0\.(25|50): Top-1: ([0-9.]+)", line
        )
        if match:
            row = rows.setdefault(epoch, {"epoch": epoch, "subgroups": {}})
            row["acc" + match[1]] = float(match[2])
            row["log_time"] = line.split("]:", 1)[0].lstrip("[")
        match = re.search(r"\]: (unique|multi)(50)? ([0-9.eE+-]+)", line)
        if match and epoch in rows:
            key = match[1] + (match[2] or "25")
            # The first group values are semantic; later repeats are mask metrics.
            rows[epoch]["subgroups"].setdefault(key, float(match[3]))
    checkpoints = {}
    for path in sorted(root.rglob("ckpt_epoch_*.pth")):
        if path.name == "ckpt_epoch_last.pth":
            continue
        own = torch.load(path, map_location="cpu", weights_only=False)
        own_epoch = own["epoch"]
        if own_epoch not in rows or "acc50" not in rows[own_epoch]:
            continue
        target = root / "preserved_checkpoints" / f"ckpt_epoch_{own_epoch}.pth"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            os.link(path, target)
        checkpoints[own_epoch] = {
            "epoch": own_epoch, "model_tensors": len(own["model"]),
            "optimizer": "optimizer" in own, "scheduler": "scheduler" in own,
            "preserved_checkpoint": str(target.relative_to(base.parent)),
        }
    for number, row in sorted(rows.items()):
        if "acc25" not in row or "acc50" not in row:
            continue
        for threshold in ("25", "50"):
            groups = row["subgroups"]
            if dataset == "scanrefer":
                hits = round(groups["unique" + threshold] * 1419)
                hits += round(groups["multi" + threshold] * 8089)
            else:
                hits = round(groups["multi" + threshold] * total)
            assert abs(hits / total - row["acc" + threshold]) < 0.000006
            row["hits" + threshold] = hits
        row.update(dataset=dataset, val_examples=total)
        row["checkpoint"] = checkpoints.get(number)
        print(json.dumps(row, sort_keys=True))
PY
