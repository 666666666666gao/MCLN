#!/usr/bin/env bash
cd /root/autodl-tmp/cs_mcln_source_20260923_v1
export PYTHONPATH=/root/autodl-tmp/cs_mcln_readback_source_20260930_v1:/root/autodl-tmp/cs_mcln_source_20260923_v1
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/bdetr/bin/python -u /root/autodl-tmp/cs_mcln_readback_source_20260930_v1/scripts/train_cs_mcln_scanrefer.py --arm cs --mode train --checkpoint /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth --data-root /root/autodl-tmp/DATA_ROOT --output /root/cs_mcln_scanrefer_cs_numeric_control_v1 --best-dir /root/cs_mcln_scanrefer_cs_numeric_control_best_v1 --batch-size 12 2>&1 | tee /root/cs_mcln_scanrefer_cs_numeric_control_v1.log
exit_code=${PIPESTATUS[0]}
printf '%s\n' "$exit_code" > /root/cs_mcln_scanrefer_cs_numeric_control_v1.exit.txt
exit "$exit_code"
