#!/usr/bin/env bash
set -euo pipefail
audit=/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/reader_result_20261104
date -u -Is
screen -ls | grep '[.]pvg_scanrefer_reader_result_audit[[:space:]]' || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
for name in preparation.json started.json pipeline_exit_code complete.json; do
    if test -f "$audit/$name"; then
        printf '%s: ' "$name"
        cat "$audit/$name"
        printf '\n'
    fi
done
if test -f "$audit/started.json"; then
    AUDIT="$audit" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import os
from pathlib import Path
import subprocess

audit = Path(os.environ["AUDIT"])
pid = json.loads((audit / "started.json").read_text())["pid"]
print(subprocess.run(["ps", "-p", str(pid), "-o", "pid,etime,stat,args"],
                     text=True, capture_output=True).stdout)
path = audit / "none.jsonl"
counts = {}
footer = None
if path.exists():
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            kind = row["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "forward_complete":
                footer = row
print(json.dumps({"record_counts": counts, "footer": footer}))
PY
fi
if test -f "$audit/none.log"; then
    tail -c 7000 "$audit/none.log" | tr '\r' '\n' | tail -n 8
fi
