#!/bin/bash
# Usage: bash scripts/launch_bn_probe.sh <run_name> <extra args...>
#   lr0:       bash scripts/launch_bn_probe.sh bn_probe_lr0 --lr 0 --lr_backbone 0 --lr_modules 0 --lr_span 0 --max_epoch 1
#   freeze_bn: bash scripts/launch_bn_probe.sh bn_probe_freeze_0p1 --lr 1e-7 --lr_backbone 1e-7 --lr_modules 1e-6 --lr_span 1e-7 --freeze_bn
# Same recipe as lr_scale_0p1_retry2; later arguments override earlier ones.
set -e
RUN=$1; shift
SRC=$(cd "$(dirname "$0")/.." && pwd)
RUNTIME=/root/autodl-tmp/mcln_pvground_runtime_20260908_v1
OUT=/root/autodl-tmp/pvground_bn_probe_20261020/$RUN
mkdir -p "$OUT/tmp"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29691 TMPDIR=$OUT/tmp
export PYTHONPATH=$RUNTIME/PV-Ground/pointnet2:$RUNTIME/PV-Ground:$RUNTIME/OpenPCDet:$SRC
cd "$SRC"
nohup $RUNTIME/venv/bin/python -B -u train_dist_mod.py --model PVGround --dataset scanrefer --test_dataset scanrefer \
  --data_root /root/autodl-tmp/DATA_ROOT_mcln_meshsp/ --use_color --use_soft_token_loss --use_contrastive_align \
  --self_attend --butd --rng_seed 2027 --batch_size 8 --num_workers 4 --num_target 256 \
  --num_decoder_layers 6 --num_encoder_layers 3 --self_position_embedding loc_learned \
  --lr 1e-7 --lr_backbone 1e-7 --lr_modules 1e-6 --weight_decay 0.0005 --clip_norm 0.1 \
  --max_epoch 8 --start_epoch 1 --val_freq 1 --lr-scheduler step --lr_decay_epochs 280 340 \
  --warmup-epoch -1 --lr_span 1e-7 \
  --native_init_spec /root/autodl-tmp/pvground_matched_endpoint_20261014/initialization_reference.json \
  --log_dir "$OUT/logs" --exp "$RUN" --print_freq 128 "$@" > "$OUT/train.log" 2>&1 &
echo $! > "$OUT/pid"
echo "started $RUN pid $(cat $OUT/pid) log $OUT/train.log"
