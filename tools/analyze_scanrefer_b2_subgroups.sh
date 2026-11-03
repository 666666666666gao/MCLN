#!/usr/bin/env bash
set -euo pipefail
audit=/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python \
    "$audit/analyze_pvground_decision_subgroups.py" "$audit" \
    --output "$audit/decision-subgroups.json"
