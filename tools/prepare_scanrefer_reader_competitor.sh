#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_reader_competitor_20261104"
checkpoint="$root/scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
test ! -e "$trial"
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = 46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8
mkdir "$trial"
cp -a "$root/scanrefer_restart_20261102/stage_b_source" "$trial/source"
tar -xf "$root/scanrefer_reader_competitor_bundle.tar" -C "$trial/source"
cd "$trial/source"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/pointnet2:$PWD"
/root/autodl-tmp/PV-Ground/env/bin/python tools/test_scanrefer_reader_competitor.py -v 2>&1 | tee "$trial/remote_cpu_tests.log"
CHECKPOINT="$checkpoint" TRIAL="$trial" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import torch

checkpoint = Path(os.environ["CHECKPOINT"])
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 2 and len(saved["model"]) == 1168
assert saved["optimizer"]["state"] and saved["scheduler"]
files = ["models/candidate_local_visual.py", "tools/train_scanrefer_reader_competitor.py",
         "tools/test_scanrefer_reader_competitor.py"]
receipt = {
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files},
    "own_cpu_epoch": 2, "model_tensors": 1168,
    "experiment": "zero-initialized four nearest predicted-candidate local-summary contrast",
    "comparison": "existing one-epoch B2 control and immutable B2",
    "cuda_initialized": torch.cuda.is_initialized(),
}
assert not receipt["cuda_initialized"]
(Path(os.environ["TRIAL"]) / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
