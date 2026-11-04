#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_reader_competitor_20261104
date -u -Is
screen -ls | grep '[.]pvg_scanrefer_reader_competitor[[:space:]]' || true
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
for name in current_condition pipeline_exit_code preparation.json preflight_complete.json reader_start.json reader_complete.json reader_result.json; do
    if test -f "$trial/$name"; then
        printf '%s: ' "$name"
        cat "$trial/$name"
        printf '\n'
    fi
done
for name in preflight reader; do
    if test -f "$trial/$name.log"; then
        printf '%s log tail:\n' "$name"
        tail -c 9000 "$trial/$name.log" | tr '\r' '\n' | tail -n 12
    fi
done
find "$trial/reader" -name pvground_first_update.json -print -exec cat {} \; 2>/dev/null || true
find "$trial/reader" -name train_audit_receipt_epoch_1.json -print -exec cat {} \; 2>/dev/null || true
