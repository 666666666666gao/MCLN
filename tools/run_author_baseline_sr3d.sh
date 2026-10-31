set -e
cd /tmp/PV-Ground
export PYTHONPATH=/tmp/PV-Ground/OpenPCDet:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
mkdir -p /root/autodl-tmp/pvground_author_baseline_20261030/sr3d
/root/autodl-tmp/PV-Ground/env/bin/python -m torch.distributed.run \
  --nproc_per_node 1 --master_port 29618 train_dist_mod.py \
  --num_target 256 --sampling kps --num_encoder_layers 3 \
  --num_decoder_layers 6 --self_position_embedding loc_learned \
  --self_attend --model MCLN --query_points_obj_topk 4 \
  --use_contrastive_align --use_soft_token_loss --detect_intermediate \
  --joint_det --batch_size 3 --dataset sr3d --test_dataset sr3d \
  --data_root /root/autodl-tmp/PV-Ground/data/ --use_color \
  --butd_cls --num_workers 4 --checkpoint_path \
  /root/autodl-tmp/PV-Ground/checkpoints/PV-Ground_SR3D.pth \
  --log_dir /root/autodl-tmp/pvground_author_baseline_20261030 \
  --exp author_sr3d_baseline --print_freq 100 --eval --rng_seed 111
