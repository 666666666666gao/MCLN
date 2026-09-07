cd /root/autodl-tmp/mcln_scanrefer_object_appearance_formal_preparation_20260908_v1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/root/miniconda3/envs/bdetr/bin/python -u scripts/queue_scanrefer_object_appearance_official.py --plan /root/autodl-tmp/mcln_scanrefer_object_appearance_formal_preparation_20260908_v1/plan.json > queue.log 2>&1
code=$?
printf "%s\n" "$code" > queue.exit
exit "$code"
