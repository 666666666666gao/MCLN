#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_reader_competitor_20261104
python=/root/autodl-tmp/PV-Ground/env/bin/python
test -f "$trial/preparation.json"
test ! -e "$trial/started.json"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
trap 'status=$?; printf "%s\n" "$status" > "$trial/pipeline_exit_code"' EXIT
printf '{"pid":%d,"started_utc":"%s"}\n' "$$" "$(date -u -Is)" > "$trial/started.json"
cd "$trial/source"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/pointnet2:$PWD"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
for condition in preflight reader; do
    printf '%s condition=%s\n' "$(date -u -Is)" "$condition" > "$trial/current_condition"
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29539 \
        tools/train_scanrefer_reader_competitor.py --trial "$trial" \
        --condition "$condition" 2>&1 | tee "$trial/$condition.log"
    if [ "$condition" = preflight ]; then
        TRIAL="$trial" CUDA_VISIBLE_DEVICES="" "$python" - <<'PY'
import json
import os
from pathlib import Path
trial = Path(os.environ["TRIAL"])
run = Path(json.loads((trial / "preflight_complete.json").read_text())["run_dir"])
update = json.loads((run / "pvground_first_update.json").read_text())
assert update["epoch"] == 1 and update["batch"] == 6
assert update["finite_loss"] and update["optimizer_step_complete"]
print(json.dumps(update))
PY
    fi
done
printf '{"completed_utc":"%s"}\n' "$(date -u -Is)" > "$trial/complete.json"
