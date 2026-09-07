cd /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_object_appearance_pair.py --manifest /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json > controller.log 2>&1
code=$?
printf "%s\n" "$code" > controller.exit
exit "$code"
