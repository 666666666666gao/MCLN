#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104
date -u -Is
screen -ls | grep '[.]pvg_scanrefer_span_only[[:space:]]' || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
if test -f "$trial/started.json"; then
    TRIAL="$trial" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
pid = json.loads((Path(os.environ["TRIAL"]) / "started.json").read_text())["pid"]
for arguments in (["-p", str(pid)], ["--ppid", str(pid)]):
    print(subprocess.run(["ps", *arguments, "-o", "pid,ppid,etime,stat,args"],
                         text=True, capture_output=True).stdout)
PY
fi
for name in current_condition started.json pipeline_exit_code complete.json real_batch_gate_gradient.json preflight_complete.json control_result.json span_result.json control_frozen_state.json span_frozen_state.json; do
    if test -f "$trial/$name"; then
        printf '%s: ' "$name"
        cat "$trial/$name"
        printf '\n'
    fi
done
for condition in preflight control span; do
    if test -f "$trial/$condition.log"; then
        printf '%s log:\n' "$condition"
        tail -c 3500 "$trial/$condition.log" | tr '\r' '\n' | tail -n 5
    fi
    if test -d "$trial/$condition"; then
        find "$trial/$condition" -name 'pvground_*update.json' -print -exec cat {} \;
    fi
done
