#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
audit="$stage/protocol_audit"
python=/root/autodl-tmp/PV-Ground/env/bin/python
selected_checkpoint="$1"
selected_epoch="$2"
test ! -e "$audit/official.json"
test ! -e "$audit/selected.json"
STAGE="$stage" SELECTED="$selected_checkpoint" EPOCH="$selected_epoch" "$python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
assert json.loads((stage / "state.json").read_text())["phase"] == "stage_a_complete"
assert int((stage / "pipeline_exit_code").read_text()) == 0
result = json.loads((stage / "stage_a_complete_epochs.json").read_text())
assert [row["epoch"] for row in result["rows"]] == [1, 2, 3]
row, = [row for row in result["rows"] if row["checkpoint"] == os.environ["SELECTED"]]
assert row["epoch"] == row["checkpoint_epoch"] == int(os.environ["EPOCH"])
assert row["sample_count"] == 9508 and row["model_tensors"] == 1168
digest = hashlib.sha256()
with Path(row["checkpoint"]).open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
plan = {"selected": row, "selected_sha256": digest.hexdigest(),
        "stage_b_started": (stage / "stage_b_start.json").exists()}
assert not plan["stage_b_started"], "Do matched validation before launching Stage B"
(stage / "protocol_audit/plan.json").write_text(json.dumps(plan, indent=2) + "\n")
PY
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
official="$(dirname "$stage")/checkpoints/ckpt_epoch_54.pth"
echo "a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d  $official" | sha256sum -c -
cd "$stage/source"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/pointnet2:$PWD"
export TOKENIZERS_PARALLELISM=false
common=(
    --local_rank 0 --num_decoder_layers 6 --use_color --weight_decay 0.0005
    --data_root /root/autodl-tmp/PV-Ground/data/ --batch_size 8
    --lr_backbone 0 --lr 0 --text_encoder_lr 0 --candidate_local_visual_lr 1e-4
    --dataset scanrefer --test_dataset scanrefer --detect_intermediate --joint_det
    --use_soft_token_loss --use_contrastive_align --butd --self_attend --augment_det
    --model MCLN --rng_seed 0 --model_only_initialization --checkpoint_start_epoch 1
    --expected_eval_sample_count 9508 --use_pvground_innovations --eval
    --cudnn-benchmark --print_freq 100
)
for condition in official selected; do
    if [ "$condition" = official ]; then
        checkpoint="$official"
        epoch=54
        tensors=1134
    else
        checkpoint="$selected_checkpoint"
        epoch="$selected_epoch"
        tensors=1168
    fi
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29521 \
        "$audit/tools/eval_scanrefer_restart_e0.py" "${common[@]}" \
        --checkpoint_path "$checkpoint" --checkpoint-epoch "$epoch" \
        --checkpoint-model-tensors "$tensors" --e0-receipt "$audit/$condition.json" \
        --log_dir "$audit/$condition" --exp "scanrefer_matched_$condition" \
        2>&1 | tee "$audit/$condition.log"
done
STAGE="$stage" "$python" - <<'PY'
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
audit = stage / "protocol_audit"
official = json.loads((audit / "official.json").read_text())
selected = json.loads((audit / "selected.json").read_text())
plan = json.loads((audit / "plan.json").read_text())
keys = ("samples", "point_sha256", "identity_sha256", "seed", "backend", "innovations")
assert all(official[key] == selected[key] for key in keys)
assert official["samples"] == 9508 and official["backend"]["cudnn_benchmark"]
assert selected["checkpoint_sha256"] == plan["selected_sha256"]
old_e0 = json.loads((stage / "e0_official.json").read_text())
receipt = {
    "matched_fields": keys, "official": official, "selected": selected,
    "selected_epoch": plan["selected"]["epoch"],
    "selected_minus_matched_official_hits": {
        key: selected["hits"][key] - official["hits"][key] for key in ("0.25", "0.5")
    },
    "matched_official_minus_original_e0_hits": {
        key: official["hits"][key] - old_e0["hits"][key] for key in ("0.25", "0.5")
    },
    "original_e0_preserved": True,
    "declared_stage_a_gate_changed": False,
    "final_target_met": selected["hits"]["0.25"] >= 5705 and selected["hits"]["0.5"] >= 4945,
}
(audit / "matched_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
