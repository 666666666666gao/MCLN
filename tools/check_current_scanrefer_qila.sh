#!/usr/bin/env bash
set -euo pipefail

/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_qila_20261105"
run = trial / "run"
checkpoint = (
    root
    / "scanrefer_restart_20261102/stage_b/scanrefer/"
    "scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/"
    "ckpt_epoch_2.pth"
)

screen = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
workers = subprocess.run(
    ["ps", "-eo", "pid,ppid,etime,stat,args"], capture_output=True, text=True
)
worker_lines = [
    line[:400]
    for line in workers.stdout.splitlines()
    if "train_scanrefer_qila.py" in line
    and "/scanrefer_qila_20261105/" in line
]
gpu = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ],
    text=True,
).strip()
compute = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ],
    text=True,
).strip()
log = trial / "train.log"
text = log.read_text(errors="replace") if log.exists() else ""
lines = [line for line in text.replace("\r", "\n").splitlines() if line]
progress = [
    line.strip()[:300]
    for line in lines
    if re.search(r"\d+/\d+|Epoch|epoch|loss|Loading|validation", line, re.I)
][-30:]
errors = [
    line.strip()[:300]
    for line in lines
    if any(
        token in line
        for token in (
            "Traceback (most recent call last):",
            "RuntimeError:",
            "CUDA out of memory",
            "AssertionError:",
        )
    )
][-8:]
witnesses = {}
for name in ("pvground_first_update.json", "pvground_second_update.json"):
    matches = list(run.rglob(name))
    if matches:
        witnesses[name] = [
            {
                "path": str(path.relative_to(run)),
                "witness": json.loads(path.read_text()),
            }
            for path in matches
        ]
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": (
        "Current isolated ScanRefer QILA trial only; reads process/GPU/log/"
        "witness/completion state and immutable B2 identity. No model load,"
        "forward,evaluation,training,restart or deletion."
    ),
    "screen": [
        line.strip()
        for line in screen.stdout.splitlines()
        if "pvg_scanrefer_qila_20261105" in line
    ],
    "workers": worker_lines,
    "gpu": gpu,
    "compute_processes": compute,
    "trial_exists": trial.is_dir(),
    "run_exists": run.is_dir(),
    "started": (
        json.loads((run / "started.json").read_text())
        if (run / "started.json").exists()
        else None
    ),
    "first_update_witnesses": witnesses,
    "progress_tail": progress,
    "log_tail": [line.strip()[:500] for line in lines[-80:]],
    "recent_errors": errors,
    "exit_code": (
        (trial / "exit_code").read_text().strip()
        if (trial / "exit_code").exists()
        else None
    ),
    "complete": (
        json.loads((run / "complete.json").read_text())
        if (run / "complete.json").exists()
        else None
    ),
    "b2_checkpoint_exists": checkpoint.is_file(),
    "b2_sha256": (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint.is_file()
        else None
    ),
}
print(json.dumps(result))
PY
