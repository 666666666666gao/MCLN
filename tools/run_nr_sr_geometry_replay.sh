set -e
DATASET="$1"
CHECKPOINT="$2"
OUTPUT="$3"
case "$DATASET" in
  nr3d) RUN_ID=1793419199; MAX_EPOCH=100 ;;
  sr3d) RUN_ID=1793419200; MAX_EPOCH=140 ;;
  *) printf 'Expected nr3d or sr3d\n' >&2; exit 1 ;;
esac
RUN_ROOT="/root/autodl-tmp/pvground_${DATASET}_module_training_20261030"
RUN="$RUN_ROOT/logs/$DATASET/${DATASET}_module_ft/$RUN_ID"
test -n "$OUTPUT"
test ! -e "$OUTPUT"
test -f "$RUN/$CHECKPOINT"
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
  --native_init_spec "$RUN_ROOT/code/init_${DATASET}.json" \
  --checkpoint_path "$RUN/$CHECKPOINT" \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root /root/autodl-tmp/PV-Ground/data/ \
  --val_freq 1 --batch_size 10 --num_workers 4 --save_freq 1 \
  --print_freq 500 --lr_backbone 2e-5 --lr 1e-5 --lr_modules 1e-5 \
  --dataset "$DATASET" --test_dataset "$DATASET" --detect_intermediate \
  --joint_det --use_soft_token_loss --use_contrastive_align \
  --lr_decay_epochs 10 30 --butd_cls --self_attend \
  --max_epoch "$MAX_EPOCH" --rng_seed 111 --model PVGround \
  --exp "${DATASET}_geometry_replay" \
  --log_dir "$RUN_ROOT/logs_geometry_replay" --eval \
  --reference-metrics "$RUN/native_metrics.jsonl" --replay-output "$OUTPUT"
