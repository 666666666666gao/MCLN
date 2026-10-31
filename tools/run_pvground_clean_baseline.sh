#!/usr/bin/env bash
set -euo pipefail
dataset=$1
cd /root/autodl-tmp/pvground_clean_baseline_20261031
export PYTHONPATH="$PWD/OpenPCDet:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
case "$dataset" in
  scanrefer)
    checkpoint=PV-Ground_ScanRefer.pth
    flags=(--butd --augment_det)
    max_epoch=100
    ;;
  nr3d)
    checkpoint=PV-Ground_NR3D.pth
    flags=(--butd_cls)
    max_epoch=100
    ;;
  sr3d)
    checkpoint=PV-Ground_SR3D.pth
    flags=(--butd_cls)
    max_epoch=140
    ;;
  *) exit 2 ;;
esac
exec /root/autodl-tmp/PV-Ground/env/bin/python -u -m torch.distributed.run \
  --nproc_per_node 1 --master_port 29641 train_pvground_clean_baseline.py \
  --num_decoder_layers 6 --num_target 256 --sampling kps --model PVGround \
  --use_color --self_attend --use_soft_token_loss --use_contrastive_align \
  --detect_intermediate --joint_det "${flags[@]}" \
  --dataset "$dataset" --test_dataset "$dataset" \
  --data_root /root/autodl-tmp/PV-Ground/data/ \
  --checkpoint_path "/root/autodl-tmp/PV-Ground/checkpoints/$checkpoint" \
  --batch_size 10 --num_workers 4 --rng_seed 111 \
  --lr 1e-5 --lr_backbone 2e-5 --lr_decay_epochs 10 30 --warmup-epoch 0 \
  --weight_decay 0.0005 --max_epoch "$max_epoch" --val_freq 1 \
  --print_freq 200 --save_freq 1 --log_dir "$PWD/results" --exp clean_baseline
