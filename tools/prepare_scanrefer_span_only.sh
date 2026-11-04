#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_span_only_20261104"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
if [ "${1:-prepare}" = finish ]; then
    test -d "$trial/source"
    test ! -e "$trial/preparation.json"
    test ! -e "$trial/started.json"
else
    test ! -e "$trial"
    mkdir "$trial"
    cp -a "$root/scanrefer_span_gate_comparison_20261103/source" "$trial/source"
    tar -xf "$root/scanrefer_span_only_bundle_20261104.tar" -C "$trial/source"
fi
cd "$trial/source"
git apply --ignore-space-change --check span_only.patch
git apply --ignore-space-change span_only.patch
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import py_compile
import torch

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_span_only_20261104"
checkpoint = root / "scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha == "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 2 and len(saved["model"]) == 1168
assert saved["optimizer"]["state"] and saved["scheduler"]
files = ["main_utils.py", "models/candidate_local_visual.py",
         "models/pvground_span_supervision.py",
         "tools/train_scanrefer_span_gate_comparison.py"]
for filename in files:
    py_compile.compile(str(trial / "source" / filename), doraise=True)
assert "competitor_scale" not in (trial / "source/models/candidate_local_visual.py").read_text()
assert not torch.cuda.is_initialized()
receipt = {
    "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
    "own_cpu_epoch": 2, "model_tensors": 1168,
    "source_files_sha256": {
        name: hashlib.sha256((trial / "source" / name).read_bytes()).hexdigest()
        for name in files},
    "conditions": {"control": 0., "span": .1},
    "span_only": True, "seed": 0, "batch_size": 8, "accumulation": 6,
    "module_lr": 2e-5, "full_epochs_each": 1, "full_validation_each": 9508,
    "preflight_batches": 12, "cuda_initialized": False,
    "scope": "Fixed B2 upstream parameters and BN; train existing Span only. Same matched-object projection proxy; no network/module replacement.",
}
(trial / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
