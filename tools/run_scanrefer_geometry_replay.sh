set -e
OUTPUT="$1"
RUN_ROOT=/root/autodl-tmp/pvground_scanrefer_module_ft_20261030
RUN="$RUN_ROOT/logs_semantic/scanrefer/scanrefer_module_ft_semantic/1793345477"
test -n "$OUTPUT"
test ! -e "$OUTPUT"
test -f "$RUN/best_trained_epoch17.pth"
test -f /tmp/pvground_geometry_replay.py
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
cd "$RUN_ROOT/code"
export PYTHONPATH=/tmp/PV-Ground/OpenPCDet:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
exec /root/autodl-tmp/PV-Ground/env/bin/python -m torch.distributed.run \
  --nproc_per_node 1 --master_port 29619 /tmp/pvground_geometry_replay.py \
  --native_init_spec "$RUN_ROOT/code/init_scanrefer.json" \
  --checkpoint_path "$RUN/best_trained_epoch17.pth" \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root /root/autodl-tmp/PV-Ground/data/ \
  --val_freq 1 --batch_size 10 --num_workers 4 --save_freq 1 \
  --print_freq 500 --lr_backbone 2e-5 --lr 1e-5 --lr_modules 1e-5 \
  --dataset scanrefer --test_dataset scanrefer --detect_intermediate \
  --joint_det --use_soft_token_loss --use_contrastive_align \
  --lr_decay_epochs 10 30 --butd --self_attend --augment_det \
  --max_epoch 100 --rng_seed 111 --model PVGround \
  --exp scanrefer_epoch17_geometry_replay \
  --log_dir "$RUN_ROOT/logs_geometry_replay" --eval \
  --reference-metrics "$RUN/native_metrics.jsonl" --replay-output "$OUTPUT"
