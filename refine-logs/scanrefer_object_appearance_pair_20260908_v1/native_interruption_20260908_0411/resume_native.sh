#!/bin/bash
cd /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1
/root/miniconda3/envs/bdetr/bin/python -u scripts/queue_object_appearance_native_evaluation.py --root /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1 > native_queue.log 2>&1
code=$?
printf "%s\n" "$code" > native_queue.exit
exit "$code"
