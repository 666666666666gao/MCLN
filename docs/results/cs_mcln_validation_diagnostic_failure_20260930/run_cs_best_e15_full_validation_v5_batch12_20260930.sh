#!/usr/bin/env bash
cd /root/autodl-tmp/cs_mcln_source_20260923_v1
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/bdetr/bin/python -u /root/cs_mcln_scanrefer_cs_20260923/diagnostics/diagnose_cs_mcln_support_validation_v5.py --parent /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth --checkpoint /root/autodl-tmp/cs_mcln_scanrefer_best_20260923/best.pth --data-root /root/autodl-tmp/DATA_ROOT --output /root/cs_mcln_scanrefer_cs_20260923/diagnostics/cs_best_e15_full_validation_v5_batch12_20260930.json --split validation --batch-size 12 2>&1 | tee /root/cs_mcln_scanrefer_cs_20260923/diagnostics/cs_best_e15_full_validation_v5_batch12_20260930.log
exit_code=${PIPESTATUS[0]}
printf '%s\n' "$exit_code" > /root/cs_mcln_scanrefer_cs_20260923/diagnostics/cs_best_e15_full_validation_v5_batch12_20260930.exit.txt
exit "$exit_code"
