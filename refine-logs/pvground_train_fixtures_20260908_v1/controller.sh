#!/bin/bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
/root/miniconda3/envs/bdetr/bin/python -u /root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v1/export_fixtures.py --manifest /root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json --output /root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v1/fixtures
result=$?
printf "%s\n" "$result" > /root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v1/controller.exit
exit "$result"
