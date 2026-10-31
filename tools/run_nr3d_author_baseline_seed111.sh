set -e
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
printf '%s  %s\n' \
  d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d \
  /root/autodl-tmp/PV-Ground/checkpoints/PV-Ground_NR3D.pth | sha256sum -c -
cd /tmp/PV-Ground
export PYTHONPATH=/tmp/PV-Ground/OpenPCDet:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
exec /root/autodl-tmp/PV-Ground/env/bin/python -m torch.distributed.run \
  --nproc_per_node 1 --master_port 29617 train_dist_mod.py \
  --num_target 256 --sampling kps --num_encoder_layers 3 \
  --num_decoder_layers 6 --self_position_embedding loc_learned \
  --self_attend --model PVGround --query_points_obj_topk 4 \
  --use_contrastive_align --use_soft_token_loss --detect_intermediate \
  --joint_det --batch_size 10 --dataset nr3d --test_dataset nr3d \
  --data_root /root/autodl-tmp/PV-Ground/data/ --use_color \
  --butd_cls --num_workers 4 --checkpoint_path \
  /root/autodl-tmp/PV-Ground/checkpoints/PV-Ground_NR3D.pth \
  --log_dir /root/autodl-tmp/pvground_author_baseline_seed111_20261031 \
  --exp author_nr3d_seed111 --print_freq 100 --eval --rng_seed 111
