#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/bdetr/bin/python -u /root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1/run_cs_mcln_readback_posttrain_diagnostic.py --training-pid 187859 --not-before 2026-10-04T01:45:00+08:00 --run-dir /root/cs_mcln_scanrefer_readback_20260930 --best-dir /root/cs_mcln_scanrefer_readback_best_20260930 --source /root/autodl-tmp/cs_mcln_readback_source_20260930_v1 --base /root/autodl-tmp/cs_mcln_source_20260923_v1 --diagnostic /root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1/diagnose_cs_mcln_readback.py --parent /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth --data-root /root/autodl-tmp/DATA_ROOT > /root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1/controller.log 2>&1
exit_code=$?
printf '%s\n' "$exit_code" > /root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1/controller.exit.txt
exit "$exit_code"
