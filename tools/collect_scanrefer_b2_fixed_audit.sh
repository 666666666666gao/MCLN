#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
from pathlib import Path
import subprocess

audit = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive")
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "audit_directory": str(audit),
    "started": json.loads((audit / "started.json").read_text()),
    "current_condition": (audit / "current_condition").read_text().strip(),
    "gpu": subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader",
    ], text=True).strip(),
    "compute_processes": subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ], text=True).strip(),
    "conditions": {},
}
for name in ("pipeline_exit_code", "complete.json"):
    path = audit / name
    if path.exists():
        receipt[name] = path.read_text().strip()
for condition in ("none", "reader", "support", "span"):
    path = audit / f"{condition}.jsonl"
    if not path.exists():
        continue
    counts = {}
    detail = {}
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            kind = row["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind in ("metadata", "forward_complete"):
                detail[kind] = row
            if kind == "semantic_sample":
                detail["last_semantic_sample"] = {
                    key: row[key] for key in (
                        "sample_index", "scan_id", "target_id", "point_sha256",
                        "selected_query", "selected_iou",
                    )
                }
    detail["record_counts"] = counts
    log = audit / f"{condition}.log"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 20000))
        lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
    detail["recent_errors"] = [
        line for line in lines if any(token in line for token in (
            "Traceback (most recent call last):", "RuntimeError:", "Error:",
        ))
    ][-5:]
    detail["log_tail"] = [line for line in lines if len(line) < 1000][-10:]
    receipt["conditions"][condition] = detail
output = audit / "progress.json"
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
