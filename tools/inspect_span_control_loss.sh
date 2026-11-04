#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
from pathlib import Path
import re

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104")
lines = (trial / "control.log").read_text(errors="replace").replace("\r", "\n").splitlines()
records = []
stage = None
for index, line in enumerate(lines):
    if "Train: [" in line or "Eval: [" in line:
        stage = line
    match = re.search(r"(?<!\w)loss\s+(\d+(?:\.\d*)?)(?=\s|$)", line)
    if match and float(match.group(1)) > 100:
        records.append({
            "line": index + 1, "stage": stage,
            "context": [value for value in lines[max(0, index - 3):index + 3] if value.strip()],
        })
receipts = {}
for path in (trial / "control").glob("scanrefer/*/*/train_audit_receipt_epoch_1.json"):
    receipts[str(path)] = json.loads(path.read_text())
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Current control log and train receipt only; no model import or execution.",
    "large_logged_loss_records": records,
    "train_receipts": receipts,
}
(trial / "control_loss_context.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
