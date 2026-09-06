#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1
run_queue() {
set -e
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_queue_native_candidate_range_preflight.py > unit_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u scripts/queue_native_candidate_range_preflight.py --manifest input_manifest.json
}
(run_queue)
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
