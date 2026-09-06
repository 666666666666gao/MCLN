set -e
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_scanrefer_local_visual_audit_preparation_20260906_v1
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_audit_scanrefer_local_visual_pair.py
/root/miniconda3/envs/bdetr/bin/python -m py_compile scripts/audit_scanrefer_local_visual_pair.py
/root/miniconda3/envs/bdetr/bin/python -m scripts.audit_scanrefer_local_visual_pair --help
/root/miniconda3/envs/bdetr/bin/python check.py
