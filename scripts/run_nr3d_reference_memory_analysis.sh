#!/usr/bin/env bash
set -euo pipefail
analysis=/root/autodl-tmp/mcln_reference_memory_analysis_20260905_v1
run=/root/autodl-tmp/mcln_reference_memory_train_20260905_v1
python=/root/miniconda3/envs/bdetr/bin/python
delay=${1:?expected initial delay in seconds}
trap 'status=$?; printf "%s\n" "$status" > "$analysis/controller.exit"' EXIT
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
sleep "$delay"
while [ ! -f "$run/train.exit" ]; do sleep 240; done
read -r training_status < "$run/train.exit"
test "$training_status" = 0
"$python" "$analysis/scripts/summarize_nr3d_reference_memory.py" \
  --run "$run/results" --output "$analysis/analysis.json" > "$analysis/analysis.log" 2>&1
"$python" "$analysis/scripts/verify_nr3d_reference_memory.py" \
  --addon "$run" --analysis "$analysis/analysis.json" --output "$analysis/verification.json" \
  > "$analysis/verification.log" 2>&1
