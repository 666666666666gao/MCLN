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
trial = root / "scanrefer_bbf_quality_20261104"
run = trial / "run"
completion_path = run / "complete.json"
log = trial / "train.log"
checkpoint = root / "scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
screen = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
workers = subprocess.run(
    ["ps", "-eo", "pid,ppid,etime,stat,args"], capture_output=True, text=True
)
worker_lines = [
    line[:400] for line in workers.stdout.splitlines()
    if "train_scanrefer_bbf_quality.py" in line
    and "/scanrefer_bbf_quality_20261104/" in line
]
gpu = subprocess.check_output([
    "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader",
], text=True).strip()
compute = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader",
], text=True).strip()
text = log.read_text(errors="replace") if log.exists() else ""
lines = [line for line in text.replace("\r", "\n").splitlines() if line]
progress = [
    line.strip()[:300] for line in lines
    if re.search(r"\d+/\d+|Epoch|epoch|loss|Data loading|Loading", line, re.I)
][-24:]
errors = [
    line.strip()[:300] for line in lines
    if any(token in line for token in (
        "Traceback (most recent call last):", "RuntimeError:", "CUDA out of memory",
    ))
][-8:]
updates = {}
for name in ("pvground_first_update.json", "pvground_second_update.json"):
    matches = list(run.rglob(name))
    if matches:
        updates[name] = [
            {"path": str(path.relative_to(run)),
             "witness": json.loads(path.read_text())}
            for path in matches
        ]
history_paths = list(run.rglob("full10_semantic_history.json"))
best_paths = list(run.rglob("semantic_best.json"))
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Current isolated ScanRefer BBF-quality trial only; reads existing metrics and process/GPU state without model loading, checkpoint deserialization, forward, evaluation, training, restart, or deletion.",
    "screen": [line.strip() for line in screen.stdout.splitlines()
               if "pvg_scanrefer_bbf_quality_20261104" in line],
    "workers": worker_lines,
    "gpu": gpu,
    "compute_processes": compute,
    "trial_files": sorted(path.name for path in trial.iterdir()),
    "run_path_is_dir": run.is_dir(),
    "run_entries": sorted(path.name for path in run.iterdir()) if run.is_dir() else None,
    "started_path_exists": (run / "started.json").exists(),
    "started_path_is_symlink": (run / "started.json").is_symlink(),
    "started": json.loads((run / "started.json").read_text())
        if (run / "started.json").exists() else None,
    "first_update_witnesses": updates,
    "full10_semantic_history": [
        {"path": str(path.relative_to(run)), "history": json.loads(path.read_text())}
        for path in history_paths
    ],
    "semantic_best": [
        {"path": str(path.relative_to(run)), "best": json.loads(path.read_text())}
        for path in best_paths
    ],
    "progress_tail": progress,
    "log_tail": [line.strip()[:500] for line in lines[-60:]],
    "recent_errors": errors,
    "exit_code": (trial / "exit_code").read_text().strip()
        if (trial / "exit_code").exists() else None,
    "complete": json.loads(completion_path.read_text())
        if completion_path.exists() else None,
    "b2_checkpoint_exists": checkpoint.exists(),
    "b2_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint.exists() else None,
}
print(json.dumps(result))
PY
