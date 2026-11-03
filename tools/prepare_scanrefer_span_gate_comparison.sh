#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_span_gate_comparison_20261103"
test ! -e "$trial"
mkdir "$trial"
cp -a "$root/scanrefer_restart_20261102/stage_b_source" "$trial/source"
tar -xf "$root/scanrefer_span_gate_bundle.tar" -C "$trial/source"
cd "$trial/source"
git apply --check span_gate.patch
git apply span_gate.patch
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/pointnet2:$PWD"
/root/autodl-tmp/PV-Ground/env/bin/python -m unittest discover \
    -s tests -p test_pvground_span_supervision.py -v 2>&1 | tee "$trial/remote_cpu_tests.log"
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import py_compile
import torch

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_span_gate_comparison_20261103"
checkpoint = root / "scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha == "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 2 and len(saved["model"]) == 1168
assert saved["optimizer"]["state"] and saved["scheduler"]
files = ["main_utils.py", "models/mcln.py", "models/losses.py",
         "models/pvground_span_supervision.py",
         "tools/train_scanrefer_span_gate_comparison.py"]
for filename in files:
    py_compile.compile(str(trial / "source" / filename), doraise=True)
assert not torch.cuda.is_initialized()
result = {
    "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
    "own_cpu_epoch": 2, "model_tensors": 1168,
    "parent": str(root / "scanrefer_restart_20261102/stage_b_source"),
    "source_files_sha256": {
        filename: hashlib.sha256((trial / "source" / filename).read_bytes()).hexdigest()
        for filename in files
    },
    "conditions": {"control": 0.0, "span": 0.1},
    "seed": 0, "full_epochs_each": 1, "full_train_augmented_entries": 48655,
    "full_validation_each": 9508, "remote_cpu_tests_passed": 6,
    "cuda_initialized": False,
    "scope": "Separate source; no overwrite of B or active fixed-B2 diagnostics. Raw gate SmoothL1 only; equal center/size projection proxy, not IoU-optimal supervision. One-seed one-epoch matched pilot, not final main result.",
}
(trial / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
