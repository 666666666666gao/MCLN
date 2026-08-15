#!/usr/bin/env bash
set -uo pipefail

CANDIDATE_SESSION=mcln_v108_currentcode_oldsp_control_20260814
GEOMETRY_SESSION=mcln_v108_meshsp_geometry_wait_20260814
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
CONTROL_BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_currentcode_oldsp
CANDIDATE_EXIT="$CONTROL_BASE/candidate_train_exitcode.txt"
CANDIDATE_RECEIPT="$CONTROL_BASE/candidate_train_receipt.json"
LOG="$BASE/single_gpu_serial_geometry_wait.log"
EXIT_FILE="$BASE/single_gpu_serial_geometry_wait_exitcode.txt"

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    chmod 0444 "$LOG" "$EXIT_FILE"
    exit "$rc"
}

if [[ -e "$LOG" || -e "$EXIT_FILE" ]]; then
    echo "single-GPU serial watcher output already exists" >&2
    exit 64
fi

exec >"$LOG" 2>&1
echo "session=$GEOMETRY_SESSION"
echo "policy=wait-for-current-code-old-sp-control-then-run-causal-ab-and-geometry-on-cuda0"
date -Is

while screen -ls 2>/dev/null | grep -Fq ".$CANDIDATE_SESSION"; do
    sleep 30
done

if [[ ! -f "$CANDIDATE_EXIT" ]]; then
    echo "candidate session ended without exit code"
    finish 70
fi
candidate_rc="$(tr -d '[:space:]' <"$CANDIDATE_EXIT")"
if [[ "$candidate_rc" != "0" ]]; then
    echo "candidate pipeline failed: rc=$candidate_rc"
    finish 71
fi
if [[ ! -f "$CANDIDATE_RECEIPT" ]]; then
    echo "candidate pipeline ended without sealed receipt"
    finish 72
fi

echo "candidate pipeline passed; starting geometry pipeline"
date -Is
bash /tmp/mcln_repo/scripts/run_v108_meshsp_train_geometry.sh
geometry_rc=$?
echo "geometry pipeline rc=$geometry_rc"
date -Is
finish "$geometry_rc"
