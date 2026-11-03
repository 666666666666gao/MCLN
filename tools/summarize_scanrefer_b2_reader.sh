#!/usr/bin/env bash
set -euo pipefail
audit=/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive
/root/autodl-tmp/PV-Ground/env/bin/python "$audit/summarize_pvground_fixed_audit.py" \
    "$audit" --conditions none reader --output "$audit/reader-paired-summary.json"
