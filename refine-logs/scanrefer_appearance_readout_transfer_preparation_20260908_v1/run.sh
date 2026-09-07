#!/bin/bash
cd /root/autodl-tmp/mcln_scanrefer_appearance_readout_transfer_preparation_20260908_v1
while [ ! -f /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/native_queue.exit ]; do sleep 240; done
code=$(cat /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/native_queue.exit)
if [ "$code" -ne 0 ]; then
  printf 'Native evaluation did not complete successfully: %s
' "$code"
  printf '%s
' "$code" > queue.exit
  exit "$code"
fi
printf 'af93f970ba6c88bf679a05cd86618608a4d9475dbf0c610ee56e4f0ea09131aa  analyze.py
' | sha256sum -c - || exit 1
/root/miniconda3/envs/bdetr/bin/python analyze.py --root /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1 --output /root/autodl-tmp/mcln_scanrefer_appearance_readout_transfer_preparation_20260908_v1/analysis > analysis.log 2>&1
code=$?
printf '%s
' "$code" > queue.exit
exit "$code"
