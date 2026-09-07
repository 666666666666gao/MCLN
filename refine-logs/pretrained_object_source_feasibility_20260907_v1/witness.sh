set -eu
cd /root/autodl-tmp/mcln_openshape_object_source_20260907_v1
export PYTHONPATH=/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/vendor:/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/deps
export DGLBACKEND=pytorch
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/bdetr/bin/python -u probe_objects.py --root /root/autodl-tmp/mcln_openshape_object_source_20260907_v1 --witness-only
