#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103
audit=/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive
test -f "$trial/preparation.json"
test -f "$audit/paired-summary.json"
test "$(cat "$audit/pipeline_exit_code")" -eq 0
test ! -e "$trial/started.json"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test -z "$(screen -ls | grep '[.]pvg_scanrefer_span_gate_comparison[[:space:]]' || true)"
screen -dmS pvg_scanrefer_span_gate_comparison bash "$trial/run.sh"
screen -ls | grep pvg_scanrefer_span_gate_comparison
