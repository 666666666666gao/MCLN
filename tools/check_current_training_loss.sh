#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
import math
from pathlib import Path
import re
import subprocess

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
stage = root / "scanrefer_restart_20261102"
cases = [
    ("nr3d", root / "runs/nr3d/nr3d/mcln_pvground_nr3d_20261101/1793553113",
     root / "runs/nr3d/train.log", 96960),
    ("sr3d", root / "runs/sr3d/sr3d/mcln_pvground_sr3d_20261101/1793552465",
     root / "runs/sr3d/train.log", 92354),
    ("scanrefer_stage_b", stage / "stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354",
     stage / "stage_b_train.log", 207977),
]
for dataset, directory, log, pid in cases:
    if not directory.exists():
        continue
    config = json.loads((directory / "config.json").read_text())
    offset = max(0, log.stat().st_size - 2 * 1024 * 1024)
    with log.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
    losses = []
    train_position = None
    pending_train_position = None
    progress = None
    for line in lines:
        header = re.search(r"Train: \[(\d+)\]\[(\d+)/(\d+)\]", line)
        if header:
            train_position = dict(zip(("epoch", "batch", "total_batches"), map(int, header.groups())))
            pending_train_position = train_position
        if "Eval: [" in line:
            pending_train_position = None
        match = re.search(r"(?<!\w)loss\s+([-+]?(?:\d+(?:\.\d*)?(?:e[-+]?\d+)?|nan|inf))(?=\s|$)",
                          line, re.IGNORECASE)
        if match and pending_train_position is not None:
            losses.append({"position": pending_train_position, "value": float(match[1]),
                           "finite": math.isfinite(float(match[1]))})
            pending_train_position = None
        if re.search(r"\d+/\d+ \[", line):
            progress = line.strip()
    process = Path("/proc") / str(pid) / "cmdline"
    receipt = {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset, "run": directory.name, "pid": pid,
        "process_live": process.exists(),
        "command": process.read_bytes().replace(b"\0", b" ").decode() if process.exists() else None,
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader",
        ], text=True).strip(),
        "config": {key: config.get(key) for key in (
            "checkpoint_path", "pp_checkpoint", "model_only_initialization",
            "use_pvground_innovations", "batch_size", "gradient_accumulation_steps",
            "lr", "lr_backbone", "candidate_local_visual_lr", "max_epoch",
        )},
        "log_tail_offset": offset, "logged_total_loss_count": len(losses),
        "finite_logged_total_losses_in_tail": all(row["finite"] for row in losses) if losses else None,
        "latest_logged_total_losses": losses[-3:], "latest_training_position": train_position,
        "latest_progress": progress,
        "error_lines_in_tail": [line for line in lines if any(token in line for token in (
            "Traceback (most recent call last):", "torch.cuda.OutOfMemoryError", "RuntimeError:",
        ))][-5:],
        "scope": "Current runs only; recent cumulative logged training total-loss averages paired with their Train headers, excluding Eval losses. Not every batch or optimizer update. No model import, forward, evaluation or training.",
    }
    (directory / "current_training_loss_health.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
PY
