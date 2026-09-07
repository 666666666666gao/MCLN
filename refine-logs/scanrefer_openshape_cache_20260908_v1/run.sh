cd /root/autodl-tmp/mcln_scanrefer_openshape_cache_20260908_v1
export PYTHONPATH=/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/vendor:/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/deps DGLBACKEND=pytorch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u cache.py --manifest /root/autodl-tmp/mcln_scanrefer_openshape_cache_20260908_v1/manifest.json > cache.log 2>&1
code=$?
printf "%s\n" "$code" > cache.exit
exit "$code"
