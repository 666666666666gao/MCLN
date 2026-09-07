cd /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/root/miniconda3/envs/bdetr/bin/python -u scripts/queue_object_appearance_audit.py --root /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1 > audit_queue.log 2>&1
code=$?
printf "%s\n" "$code" > audit_queue.exit
exit "$code"
