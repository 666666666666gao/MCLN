cd /root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
flock /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u probe.py --manifest /root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1/manifest.json > probe.log 2>&1
code=$?
printf "%s\n" "$code" > probe.exit
exit "$code"
