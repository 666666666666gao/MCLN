#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
bash -n "$stage/run_scanrefer_guarded_transition.sh"
test ! -e "$stage/transition_started.json"
test ! -e "$stage/pipeline_exit_code"
test ! -e "$stage/stage_b_start.json"
kill -0 189152
screen -dmS pvg_scanrefer_restart_transition bash -c \
    'exec bash "$1/run_scanrefer_guarded_transition.sh" >> "$1/transition.log" 2>&1' \
    transition "$stage"
date -u -Is
screen -ls | grep pvg_scanrefer_restart_transition
