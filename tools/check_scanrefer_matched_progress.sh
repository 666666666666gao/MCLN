#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
date -u -Is
ps -eo pid,ppid,etime,args \
    | grep -E '[r]un_scanrefer_guarded_transition.sh|[p]ython -u .*eval_scanrefer_restart_e0.py|[p]ython -u .*train_scanrefer_stage_b.py' \
    | cut -c1-300
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for name in protocol_audit/official.json protocol_audit/selected.json \
    protocol_audit/matched_validation.json protocol_audit/pipeline_exit_code \
    stage_b_start.json stage_b_pipeline_exit_code transition_exit_code; do
    if [ -f "$stage/$name" ]; then
        printf '\n%s\n' "$name"
        head -c 12000 "$stage/$name"
        printf '\n'
    fi
done
tail -c 6500 "$stage/protocol_audit/pipeline.log" | tr '\r' '\n' | tail -n 12
