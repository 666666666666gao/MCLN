cd /root/autodl-tmp/mcln_referit3d_joint_detection_slots_20260908_v1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 /root/miniconda3/envs/bdetr/bin/python -u export.py --manifest /root/autodl-tmp/mcln_referit3d_joint_detection_slots_20260908_v1/manifest.json > export.log 2>&1
code=$?
printf "%s\n" "$code" > export.exit
exit "$code"
