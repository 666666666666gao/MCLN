#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
date -u -Is
sha256sum "$stage/run_scanrefer_guarded_transition.sh"
ps -eo pid,ppid,etime,args \
    | grep -E '[r]un_scanrefer_guarded_transition.sh|[e]val_scanrefer_restart_e0.py|[t]rain_scanrefer_stage_b.py' \
    | cut -c1-300
ps -p 189152,188533,188537 -o pid,etime,args | cut -c1-240
for name in transition_started.json transition_exit_code state.json \
    protocol_audit/pipeline_exit_code stage_b_start.json stage_b_complete.json; do
    if [ -f "$stage/$name" ]; then
        printf '%s ' "$name"
        head -c 2500 "$stage/$name"
        printf '\n'
    fi
done
tail -n 8 "$stage/transition.log"
