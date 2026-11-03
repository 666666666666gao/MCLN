#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
stage="$root/scanrefer_restart_20261102"
staging="$root/offline_audits/observations_v2"
audit="$root/offline_audits/b2_fixed_audit_20261103_lrpositive"
checkpoint="$stage/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
python=/root/autodl-tmp/PV-Ground/env/bin/python
trap 'status=$?; printf "%s\n" "$status" > "$audit/pipeline_exit_code"' EXIT
test ! -e "$audit/started.json"
printf '{"pid":%d,"started_utc":"%s"}\n' "$$" "$(date -u -Is)" > "$audit/started.json"
cd "$stage/source"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$staging:$PWD/pointnet2:$PWD"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
common=(
    --local_rank 0 --num_decoder_layers 6 --use_color --weight_decay 0.0005
    --data_root /root/autodl-tmp/PV-Ground/data/ --batch_size 8 --num_workers 4
    --lr 2e-6 --lr_backbone 4e-6 --text_encoder_lr 0 --candidate_local_visual_lr 2e-5
    --dataset scanrefer --test_dataset scanrefer --detect_intermediate --joint_det
    --use_soft_token_loss --use_contrastive_align --butd --self_attend --augment_det
    --model MCLN --rng_seed 0 --use_pvground_innovations --eval
    --expected_eval_sample_count 9508 --checkpoint_path "$checkpoint" --print_freq 100
)
for condition in none reader support span; do
    test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
    test ! -e "$audit/$condition.jsonl"
    printf '%s condition=%s\n' "$(date -u -Is)" "$condition" > "$audit/current_condition"
    printf '%q ' "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29531 \
        "$staging/tools/eval_pvground_audit.py" "${common[@]}" \
        --audit-bypass "$condition" --audit-jsonl "$audit/$condition.jsonl" \
        --log_dir "$audit/logs/$condition" --exp "b2_fixed_audit_$condition" \
        > "$audit/$condition.command"
    printf '\n' >> "$audit/$condition.command"
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29531 \
        "$staging/tools/eval_pvground_audit.py" "${common[@]}" \
        --audit-bypass "$condition" --audit-jsonl "$audit/$condition.jsonl" \
        --log_dir "$audit/logs/$condition" --exp "b2_fixed_audit_$condition" \
        2>&1 | tee "$audit/$condition.log"
done
printf '{"completed_utc":"%s","conditions":4}\n' "$(date -u -Is)" > "$audit/complete.json"
