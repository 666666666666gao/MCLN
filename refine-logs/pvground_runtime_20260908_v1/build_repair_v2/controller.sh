#!/bin/bash
cd /root/autodl-tmp/mcln_pvground_runtime_20260908_v1
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u /root/autodl-tmp/mcln_pvground_runtime_20260908_v1/resume_compile_v2.py
result=$?
printf "%s\n" "$result" > /root/autodl-tmp/mcln_pvground_runtime_20260908_v1/build_repair_v2/controller.exit
exit "$result"
