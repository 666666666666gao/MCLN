#!/usr/bin/env bash
set -euo pipefail
audit=/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive
test -f "$audit/preparation.json"
test ! -e "$audit/started.json"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
screen -dmS pvg_scanrefer_b2_fixed_audit bash "$audit/run_conditions.sh"
date -u -Is
screen -ls | grep pvg_scanrefer_b2_fixed_audit
