set -e
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/root/autodl-tmp/mcln_native_box_initialization_preparation_20260907_v1
cd /root/autodl-tmp/mcln_native_box_initialization_preparation_20260907_v1
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_native_box_transfer_initialization.py > cpu_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u check_native_box_initialization_cpu.py > cpu_load_stdout.txt 2> cpu_load_stderr.txt
