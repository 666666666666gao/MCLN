#!/usr/bin/env bash
set -euo pipefail

addon=/root/autodl-tmp/mcln_reference_memory_train_20260905_v1
source=/root/autodl-tmp/mcln_g0_view_pair_20260905/inputs_v3/fixed_source
checkpoint=/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth
python=/root/miniconda3/envs/bdetr/bin/python
role=${1:?expected preflight or train}
case "$role" in
  preflight) output="$addon/preflight"; options=(--preflight-only) ;;
  train) output="$addon/results"; options=() ;;
  *) exit 2 ;;
esac
exec 9>/root/autodl-tmp/mcln_v99_backbone_gpu0.lock
flock -n 9
trap 'status=$?; printf "%s\n" "$status" > "$addon/${role}.exit"' EXIT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$source"
cd "$source"
"$python" "$addon/scripts/run_nr3d_reference_memory.py" \
  --checkpoint "$checkpoint" --output "$output" "${options[@]}"
