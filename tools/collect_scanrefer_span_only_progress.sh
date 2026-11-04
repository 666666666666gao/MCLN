#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
import math
from pathlib import Path
import re
import subprocess

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104")
result = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scope": "Only registered Span-only trial; no model import or training restart.",
          "conditions": {}}
for name in ("preparation.json", "started.json", "preflight_complete.json",
             "real_batch_gate_gradient.json", "control_start.json", "span_start.json",
             "control_result.json", "span_result.json", "control_frozen_state.json",
             "span_frozen_state.json", "control_complete.json", "span_complete.json",
             "complete.json"):
    path = trial / name
    if path.is_file():
        result[name] = json.loads(path.read_text())
for name in ("pipeline_exit_code", "current_condition"):
    path = trial / name
    if path.is_file():
        result[name] = path.read_text().strip()
pid = result["started.json"]["pid"]
result["controller_process"] = subprocess.run(
    ["ps", "-p", str(pid), "-o", "pid,ppid,etime,stat,args"],
    text=True, capture_output=True).stdout
result["children"] = subprocess.run(
    ["ps", "--ppid", str(pid), "-o", "pid,ppid,etime,stat,args"],
    text=True, capture_output=True).stdout
result["gpu_processes"] = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
    text=True).strip()
for condition in ("preflight", "control", "span"):
    records = {}
    for name in ("pvground_first_update.json", "pvground_second_update.json"):
        paths = list((trial / condition).glob("scanrefer/*/*/" + name))
        assert len(paths) <= 1
        if paths:
            records[name] = json.loads(paths[0].read_text())
    log = trial / (condition + ".log")
    if log.is_file():
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
        records["log_tail"] = [line for line in lines if len(line) < 1000][-6:]
        records["recent_logged_loss_count"] = len(losses)
        records["recent_logged_losses_finite"] = (
            all(math.isfinite(value) for value in losses) if losses else None
        )
        records["latest_logged_losses"] = losses[-3:]
        stage_losses = {"train": [], "eval": []}
        stage = None
        for line in lines:
            if "Train: [" in line:
                stage = "train"
            elif "Eval: [" in line:
                stage = "eval"
            match = re.search(
                r"(?<!\w)loss\s+([-+]?(?:\d+(?:\.\d*)?(?:e[-+]?\d+)?|nan|inf))(?=\s|$)",
                line, re.IGNORECASE)
            if stage is not None and match:
                stage_losses[stage].append(float(match.group(1)))
        records["logged_losses_by_stage"] = {
            name: {"count": len(values), "latest": values[-3:],
                   "finite": all(math.isfinite(value) for value in values) if values else None}
            for name, values in stage_losses.items()
        }
        records["recent_errors"] = [line for line in lines
                                   if "Traceback (most recent call last)" in line or "RuntimeError:" in line]
    if records:
        result["conditions"][condition] = records
output = trial / "progress.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
