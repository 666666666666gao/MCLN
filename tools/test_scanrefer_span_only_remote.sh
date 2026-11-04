#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104/source
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python \
    /tmp/test_span_only_freezing_20261104.py
