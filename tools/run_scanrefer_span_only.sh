#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104
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
for condition in preflight control span; do
    printf '%s condition=%s\n' "$(date -u -Is)" "$condition" > "$trial/current_condition"
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29543 \
        tools/train_scanrefer_span_gate_comparison.py --trial "$trial" \
        --condition "$condition" --span-only 2>&1 | tee "$trial/$condition.log"
    if [ "$condition" = preflight ]; then
        CUDA_VISIBLE_DEVICES="" "$python" - <<'PY'
import json
from pathlib import Path
trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104")
run = Path(json.loads((trial / "preflight_complete.json").read_text())["run_dir"])
for name in ("pvground_first_update.json", "pvground_second_update.json"):
    receipt = json.loads((run / name).read_text())
    assert receipt["finite_loss"] and receipt["optimizer_step_complete"]
    assert receipt["frozen_parameters_unchanged"] and receipt["frozen_buffers_unchanged"]
    assert set(receipt["modules"]) == {"pvground_span_mixer."}
    assert receipt["modules"]["pvground_span_mixer."]["parameter_delta_l2"] > 0
    print(json.dumps(receipt))
gradient = json.loads((trial / "real_batch_gate_gradient.json").read_text())
assert gradient["valid_matched_axes"] > 0 and gradient["raw_gate_gradient_l1"] > 0
PY
    fi
done
printf '{"completed_utc":"%s","conditions":2}\n' "$(date -u -Is)" > "$trial/complete.json"
