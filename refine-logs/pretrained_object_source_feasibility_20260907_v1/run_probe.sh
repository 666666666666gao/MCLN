cd /root/autodl-tmp/mcln_openshape_object_source_20260907_v1
export PYTHONPATH=/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/vendor:/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/deps
export DGLBACKEND=pytorch
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u probe_objects.py --root /root/autodl-tmp/mcln_openshape_object_source_20260907_v1 > probe.log 2>&1
code=$?
printf "%s\n" "$code" > probe.exit
if [ "$code" -ne 0 ]; then exit "$code"; fi
/root/miniconda3/envs/bdetr/bin/python -u analyze_probe.py --root /root/autodl-tmp/mcln_openshape_object_source_20260907_v1 > analysis.log 2>&1
code=$?
printf "%s\n" "$code" > analysis.exit
exit "$code"
