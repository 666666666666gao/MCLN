#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
from pathlib import Path
import subprocess

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104")
result = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scope": "Only registered Span-only trial; no model import or training restart.",
          "conditions": {}}
for name in ("preparation.json", "started.json", "preflight_complete.json",
             "real_batch_gate_gradient.json", "control_start.json", "span_start.json",
             "control_result.json", "span_result.json", "control_frozen_state.json",
             "span_frozen_state.json", "complete.json"):
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
            stream.seek(max(0, log.stat().st_size - 8000))
            lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
        records["log_tail"] = [line for line in lines if len(line) < 1000][-6:]
        records["recent_errors"] = [line for line in lines
                                   if "Traceback (most recent call last)" in line or "RuntimeError:" in line]
    if records:
        result["conditions"][condition] = records
output = trial / "progress.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
