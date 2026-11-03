#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_span_gate_comparison_20261103"
audit="$root/offline_audits/b2_fixed_audit_20261103_lrpositive"
python=/root/autodl-tmp/PV-Ground/env/bin/python
test -f "$audit/paired-summary.json"
test "$(cat "$audit/pipeline_exit_code")" -eq 0
test ! -e "$trial/started.json"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
trap 'status=$?; printf "%s\n" "$status" > "$trial/pipeline_exit_code"' EXIT
printf '{"pid":%d,"started_utc":"%s"}\n' "$$" "$(date -u -Is)" > "$trial/started.json"
cd "$trial/source"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/pointnet2:$PWD"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
for condition in preflight control span; do
    printf '%s condition=%s\n' "$(date -u -Is)" "$condition" > "$trial/current_condition"
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29537 \
        tools/train_scanrefer_span_gate_comparison.py \
        --trial "$trial" --condition "$condition" 2>&1 | tee "$trial/$condition.log"
    if [ "$condition" = preflight ]; then
        CUDA_VISIBLE_DEVICES="" "$python" - <<'PY'
import json
from pathlib import Path
trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103")
run = Path(json.loads((trial / "preflight_complete.json").read_text())["run_dir"])
update = json.loads((run / "pvground_first_update.json").read_text())
assert update["epoch"] == 1 and update["batch"] == 6
assert update["finite_loss"] and update["optimizer_step_complete"]
assert (trial / "real_batch_gate_gradient.json").exists()
print(json.dumps(update))
PY
    fi
done
printf '{"completed_utc":"%s","conditions":2,"full_epochs_each":1}\n' "$(date -u -Is)" > "$trial/complete.json"
