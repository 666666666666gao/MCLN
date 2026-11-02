#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
stage="$root/scanrefer_restart_20261102"
python=/root/autodl-tmp/PV-Ground/env/bin/python
cd "$stage/source"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$stage/source/pointnet2:$stage/source"
export TOKENIZERS_PARALLELISM=false
checkpoint="$root/checkpoints/ckpt_epoch_54.pth"
sha=a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d
echo "$sha  $checkpoint" | sha256sum -c -
common=(
    --local_rank 0 --num_decoder_layers 6 --use_color --weight_decay 0.0005
    --data_root /root/autodl-tmp/PV-Ground/data/ --batch_size 8
    --lr_backbone 0 --lr 0 --text_encoder_lr 0 --candidate_local_visual_lr 1e-4
    --dataset scanrefer --test_dataset scanrefer --detect_intermediate --joint_det
    --use_soft_token_loss --use_contrastive_align --butd --self_attend --augment_det
    --model MCLN --rng_seed 0 --checkpoint_path "$checkpoint"
    --pp_checkpoint "$checkpoint" --pp_checkpoint_sha256 "$sha"
    --model_only_initialization --checkpoint_start_epoch 1
    --expected_eval_sample_count 9508 --print_freq 100
)
for condition in official identity; do
    module_flags=()
    if [ "$condition" = identity ]; then
        module_flags=(--use_pvground_innovations)
    fi
    printf '{"phase":"e0_%s","started_utc":"%s"}\n' "$condition" "$(date -u -Is)" > "$stage/state.json"
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29517 \
        tools/eval_scanrefer_restart_e0.py \
        --e0-receipt "$stage/e0_${condition}.json" \
        "${common[@]}" --eval "${module_flags[@]}" \
        --log_dir "$stage/e0_$condition" --exp "scanrefer_e0_$condition" \
        2>&1 | tee "$stage/e0_${condition}.log"
done
STAGE="$stage" "$python" - <<'PY'
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
official = json.loads((stage / "e0_official.json").read_text())
identity = json.loads((stage / "e0_identity.json").read_text())
keys = ("samples", "hits", "point_sha256", "identity_sha256", "seed", "checkpoint_sha256")
assert all(official[key] == identity[key] for key in keys), (official, identity)
receipt = {"passed": True, "compared_fields": keys, "official": official, "identity": identity}
(stage / "e0_gate.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("E0_GATE_PASSED", json.dumps(receipt))
PY
printf '{"phase":"stage_a","started_utc":"%s"}\n' "$(date -u -Is)" > "$stage/state.json"
"$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29517 \
    train_dist_mod.py "${common[@]}" \
    --use_pvground_innovations --pvground_train_only \
    --gradient_accumulation_steps 6 --max_epoch 3 --lr_decay_epochs 100 \
    --val_freq 1 --save_freq 1 --checkpoint_metric_retention \
    --checkpoint_retention_metrics rec_acc025 rec_acc050 \
    --log_dir "$stage/stage_a" --exp scanrefer_stage_a_20261102 \
    2>&1 | tee "$stage/stage_a_train.log"
printf '{"phase":"stage_a_complete","completed_utc":"%s"}\n' "$(date -u -Is)" > "$stage/state.json"
