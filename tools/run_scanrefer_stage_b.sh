#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
python=/root/autodl-tmp/PV-Ground/env/bin/python
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$(dirname "$stage")" \
    "$python" "$stage/collect_scanrefer_stage_a_epochs.py" \
    --stage "$stage" --output "$stage/stage_a_complete_epochs.json"
STAGE="$stage" "$python" - <<'PY'
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
result = json.loads((stage / "stage_a_complete_epochs.json").read_text())
assert result["stage_b_decision"]["status"] == "ready", result["stage_b_decision"]
assert not (stage / "stage_b_start.json").exists(), "Stage B was already launched"
PY
cd "$stage/stage_b_source"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/pointnet2:$PWD"
export TOKENIZERS_PARALLELISM=false
"$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29519 \
    tools/train_scanrefer_stage_b.py --stage "$stage" \
    2>&1 | tee "$stage/stage_b_train.log"
