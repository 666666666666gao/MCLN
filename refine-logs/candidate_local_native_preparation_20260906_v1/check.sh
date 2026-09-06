set -e
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
/root/miniconda3/envs/bdetr/bin/python /root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/prepare_source.py /root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1
cd /root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/model_source
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_candidate_local_visual_training.py tests/test_mcln_training_groups.py tests/test_main_utils_source_choice_checkpoint.py
/root/miniconda3/envs/bdetr/bin/python -m py_compile main_utils.py train_dist_mod.py models/mcln_training_groups.py models/candidate_local_visual_training.py
/root/miniconda3/envs/bdetr/bin/python /root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/check_native.py /root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1
