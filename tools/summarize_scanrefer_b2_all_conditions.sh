#!/usr/bin/env bash
set -euo pipefail
audit=/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive
test -f "$audit/complete.json"
test "$(cat "$audit/pipeline_exit_code")" -eq 0
/root/autodl-tmp/PV-Ground/env/bin/python "$audit/summarize_pvground_fixed_audit.py" \
    "$audit" --output "$audit/paired-summary.json"
tar -czf "$audit/completed_evidence.tar.gz" -C "$audit" \
    preparation.json started.json complete.json pipeline_exit_code paired-summary.json \
    none.log reader.log support.log span.log none.command reader.command support.command span.command
