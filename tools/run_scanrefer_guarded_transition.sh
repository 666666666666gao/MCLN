#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
audit="$stage/protocol_audit"
python=/root/autodl-tmp/PV-Ground/env/bin/python
trap 'status=$?; printf "%s\n" "$status" > "$stage/transition_exit_code"' EXIT
test ! -e "$stage/transition_started.json"
printf '{"pid":%d,"started_utc":"%s","waiting_for_run":"1793644724"}\n' \
    "$$" "$(date -u -Is)" > "$stage/transition_started.json"

while [ ! -f "$stage/pipeline_exit_code" ]; do
    date -u -Is
    ps -p 189152 -o pid,etime,args | cut -c1-260
    sleep 240
done
test "$(cat "$stage/pipeline_exit_code")" -eq 0
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$(dirname "$stage")" \
    "$python" "$stage/collect_scanrefer_stage_a_epochs.py" \
    --stage "$stage" --output "$stage/stage_a_complete_epochs.json"
selection="$(STAGE="$stage" "$python" - <<'PY'
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
decision = json.loads((stage / "stage_a_complete_epochs.json").read_text())["stage_b_decision"]
assert decision["status"] == "ready", decision
assert not (stage / "stage_b_start.json").exists()
print(decision["selected"]["checkpoint"])
print(decision["selected"]["epoch"])
PY
)"
mapfile -t selected <<< "$selection"
test "${#selected[@]}" -eq 2
test ! -e "$audit/pipeline_exit_code"
(
    trap 'status=$?; printf "%s\n" "$status" > "$audit/pipeline_exit_code"' EXIT
    bash "$audit/tools/run_scanrefer_matched_validation.sh" \
        "${selected[0]}" "${selected[1]}"
) >> "$audit/pipeline.log" 2>&1

test ! -e "$stage/stage_b_pipeline_exit_code"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
(
    trap 'status=$?; printf "%s\n" "$status" > "$stage/stage_b_pipeline_exit_code"' EXIT
    bash "$stage/run_scanrefer_stage_b.sh"
) >> "$stage/stage_b_pipeline.log" 2>&1
