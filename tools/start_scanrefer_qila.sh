#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_qila_20261105"
source="$root/scanrefer_restart_20261102/stage_b_source"
checkpoint="$root/scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
python=/root/autodl-tmp/PV-Ground/env/bin/python

test -f "$trial/preparation.json"
test ! -e "$trial/started.json"
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"
test -z "$(screen -ls | grep '[.]pvg_scanrefer_qila_20261105[[:space:]]' || true)"
mkdir -p "$trial/run"
screen -dmS pvg_scanrefer_qila_20261105 bash -c '
    set -o pipefail
    trial="$1"
    source="$2"
    checkpoint="$3"
    python="$4"
    cd "$source"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$source/pointnet2:$source"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    "$python" -m torch.distributed.run --standalone --nproc_per_node 1 \
        --master_port 29573 \
        "$trial/train_scanrefer_qila.py" \
        "$trial/run" "$checkpoint" 2>&1 | tee "$trial/train.log"
    status=$?
    printf "%s\n" "$status" > "$trial/exit_code"
    if [ "$status" -ne 0 ]; then exit "$status"; fi
' launch "$trial" "$source" "$checkpoint" "$python"
screen -ls | grep '[.]pvg_scanrefer_qila_20261105[[:space:]]'
