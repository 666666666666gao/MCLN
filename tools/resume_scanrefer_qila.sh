#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_qila_20261105"
run_root="$trial/run"
source="$root/scanrefer_restart_20261102/stage_b_source"
checkpoint="$run_root/scanrefer/scanrefer_qila_20261105/1793909615/ckpt_epoch_2.pth"
last_checkpoint="${checkpoint%/ckpt_epoch_2.pth}/ckpt_epoch_last.pth"
python=/root/autodl-tmp/PV-Ground/env/bin/python
expected_sha=fe2182575fc136004df5dee005058d19a627c817b1fbf48bc670db6c1804c5cf

test -d "$run_root"
test -f "$run_root/started.json"
test ! -e "$run_root/complete.json"
test "$(cat "$trial/exit_code")" = "1"
test "$(cat "$trial/exit_code_resume_epoch1")" = "1"
test "$(cat "$trial/exit_code_resume_epoch2_attempt3")" = "1"
test ! -e "$trial/exit_code_resume_epoch2_attempt4"
test -f "$checkpoint"
test -f "$last_checkpoint"
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = "$expected_sha"
test "$(sha256sum "$last_checkpoint" | cut -d' ' -f1)" = "$expected_sha"
test -f "$run_root/scanrefer/scanrefer_qila_20261105/1793909615/pvground_first_update.json"
test -z "$(screen -ls | grep '[.]pvg_scanrefer_qila_20261105[[:space:]]' || true)"
if pgrep -af '[t]rain_scanrefer_qila.py' | grep -F "$trial/" >/dev/null; then
    echo "A QILA trainer process already exists" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="" "$python" -m py_compile "$trial/train_scanrefer_qila.py"
cd "$source"
screen -dmS pvg_scanrefer_qila_20261105 bash -c '
    set -o pipefail
    trial="$1"
    run_root="$2"
    checkpoint="$3"
    python="$4"
    cd "$5"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$5/pointnet2:$5"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    "$python" -m torch.distributed.run --standalone --nproc_per_node 1 \
        --master_port 29573 \
        "$trial/train_scanrefer_qila.py" \
        "$run_root" "$checkpoint" resume 2>&1 | tee -a "$trial/train.log"
    status=$?
    printf "%s\n" "$status" > "$trial/exit_code_resume_epoch2_attempt4"
    exit "$status"
' launch "$trial" "$run_root" "$checkpoint" "$python" "$source"
screen -ls | grep '[.]pvg_scanrefer_qila_20261105[[:space:]]'
