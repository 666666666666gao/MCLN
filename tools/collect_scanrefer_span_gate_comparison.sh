#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
from pathlib import Path
import subprocess

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103")
result = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in ("started.json", "current_condition", "pipeline_exit_code",
             "preflight_start.json", "preflight_complete.json",
             "real_batch_gate_gradient.json", "control_start.json",
             "control_complete.json", "control_result.json",
             "span_start.json", "span_complete.json", "span_result.json", "complete.json"):
    path = trial / name
    if path.exists():
        result[name] = json.loads(path.read_text()) if name.endswith(".json") else path.read_text().strip()
for condition in ("preflight", "control", "span"):
    log = trial / f"{condition}.log"
    if not log.exists():
        continue
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 22000))
        lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
    result[f"{condition}_tail"] = [line for line in lines if len(line) < 1400][-12:]
    result[f"{condition}_recent_errors"] = [
        line for line in lines if any(token in line for token in (
            "Traceback (most recent call last):", "RuntimeError:", "Error:"))][-5:]
    for start_name in (f"{condition}_complete.json",):
        if (trial / start_name).exists():
            run = Path(json.loads((trial / start_name).read_text())["run_dir"])
            witness = run / "pvground_first_update.json"
            if witness.exists():
                result[f"{condition}_first_update"] = json.loads(witness.read_text())
result["gpu"] = subprocess.check_output([
    "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
    "--format=csv,noheader"], text=True).strip()
result["compute_processes"] = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader"], text=True).strip()
(trial / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
