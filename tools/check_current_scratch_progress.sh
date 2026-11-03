#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
import math
from pathlib import Path
import re
import subprocess

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/runs")
for dataset, run_id, pids, budget, accumulation in (
    ("nr3d", "1793553113", (96960, 135075, 156800), 240, 4),
    ("sr3d", "1793552465", (92354, 129032, 148725), 140, 2),
):
    directory = root / dataset
    run = directory / dataset / f"mcln_pvground_{dataset}_20261101" / run_id
    if not run.exists():
        continue
    config = json.loads((run / "config.json").read_text())
    assert config["checkpoint_path"] is None and config["pp_checkpoint"] is None
    assert not config["use_pvground_innovations"]
    assert config["max_epoch"] == budget and config["batch_size"] == 12
    assert config["gradient_accumulation_steps"] == accumulation
    processes = []
    for pid in pids:
        process = Path("/proc") / str(pid)
        processes.append({
            "pid": pid, "exists": process.exists(),
            "command": (process / "cmdline").read_bytes().replace(
                b"\0", b" ").decode() if process.exists() else None,
        })
    log = directory / "train.log"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 2 * 1024 * 1024))
        lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
    losses = [
        float(match.group(1))
        for line in lines
        for match in re.finditer(
            r"(?<!\w)loss\s+([-+]?(?:\d+(?:\.\d*)?(?:e[-+]?\d+)?|nan|inf))(?=\s|$)",
            line, re.IGNORECASE)
    ]
    screen = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    receipt = {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset, "run": run_id, "processes": processes,
        "screen": [line for line in screen.stdout.splitlines()
                   if f"pvg_{dataset}_mcln_scratch" in line],
        "config": {key: config[key] for key in (
            "checkpoint_path", "pp_checkpoint", "use_pvground_innovations",
            "batch_size", "gradient_accumulation_steps", "lr", "lr_backbone",
            "max_epoch", "lr_decay_epochs")},
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True).strip(),
        "tail": [line for line in lines if len(line) < 1400][-10:],
        "recent_logged_loss_count": len(losses),
        "recent_logged_losses_finite": all(math.isfinite(value) for value in losses) if losses else None,
        "latest_logged_losses": losses[-3:],
        "recent_errors": [line for line in lines if any(token in line for token in (
            "Traceback (most recent call last):", "RuntimeError:", "Error:"))][-5:],
        "scope": "Only registered current NR/SR scratch runs; source/log/config/process reads. No model import, checkpoint load, forward, evaluation, training or run restart.",
    }
    (directory / "current_scratch_progress.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
PY
