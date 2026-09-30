#!/usr/bin/env bash
cd /root/autodl-tmp/cs_mcln_source_20260923_v1
export PYTHONPATH=/root/autodl-tmp/cs_v99_internal_draft_20260928:/root/autodl-tmp/cs_mcln_source_20260923_v1
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/bdetr/bin/python -u /root/autodl-tmp/cs_v99_internal_draft_20260928/scripts/train_cs_mcln_scanrefer.py --arm cs_readback --mode preflight --checkpoint /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth --data-root /root/autodl-tmp/DATA_ROOT --output /root/cs_mcln_readback_preflight_b12_v2_20260930 --batch-size 12 2>&1 | tee /root/cs_mcln_readback_preflight_b12_v2_20260930.log
exit_code=${PIPESTATUS[0]}
printf '%s\n' "$exit_code" > /root/cs_mcln_readback_preflight_b12_v2_20260930.exit.txt
exit "$exit_code"
