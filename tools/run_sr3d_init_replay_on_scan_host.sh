set -e
OUTPUT="$1"
ROOT=/root/autodl-tmp/pvground_sr3d_replay_20261031
test -n "$OUTPUT"
test ! -e "$OUTPUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
cd "$ROOT/code"
export PYTHONPATH=/tmp/PV-Ground/OpenPCDet:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
exec /root/autodl-tmp/PV-Ground/env/bin/python -m torch.distributed.run \
  --nproc_per_node 1 --master_port 29619 /tmp/pvground_geometry_replay.py \
  --native_init_spec "$ROOT/code/init_sr3d.json" \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root /root/autodl-tmp/PV-Ground/data/ \
  --val_freq 1 --batch_size 10 --num_workers 4 --save_freq 1 \
  --print_freq 500 --lr_backbone 2e-5 --lr 1e-5 --lr_modules 1e-5 \
  --dataset sr3d --test_dataset sr3d --detect_intermediate \
  --joint_det --use_soft_token_loss --use_contrastive_align \
  --lr_decay_epochs 10 30 --butd_cls --self_attend \
  --max_epoch 140 --rng_seed 111 --model PVGround \
  --exp sr3d_init_geometry_replay --log_dir "$ROOT/logs" --eval \
  --reference-metrics "$ROOT/code/native_metrics.jsonl" --replay-output "$OUTPUT"
