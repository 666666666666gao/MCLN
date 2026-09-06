#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
run_checks() {
set -e
/root/miniconda3/envs/bdetr/bin/python /root/autodl-tmp/mcln_native_range_preparation_20260907_v2/prepare_source.py /root/autodl-tmp/mcln_native_range_preparation_20260907_v2
cd /root/autodl-tmp/mcln_native_range_preparation_20260907_v2/model_source
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_candidate_local_visual_training.py tests/test_candidate_range_visual.py tests/test_mcln_training_groups.py tests/test_main_utils_source_choice_checkpoint.py
/root/miniconda3/envs/bdetr/bin/python /root/autodl-tmp/mcln_native_range_preparation_20260907_v2/check_native.py /root/autodl-tmp/mcln_native_range_preparation_20260907_v2
}
(run_checks)
status=$?
printf "%s\n" "$status" > /root/autodl-tmp/mcln_native_range_preparation_20260907_v2/controller.exit
exit "$status"
