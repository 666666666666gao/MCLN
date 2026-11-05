#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_qila_20261105"
source="$root/scanrefer_restart_20261102/stage_b_source"
checkpoint="$root/scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
python=/root/autodl-tmp/PV-Ground/env/bin/python

test -d "$trial"
test ! -e "$trial/started.json"
test -f "$trial/train_scanrefer_qila.py"
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test -z "$(screen -ls | grep '[.]pvg_scanrefer_qila_20261105[[:space:]]' || true)"

cd "$source"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$source/pointnet2:$source" "$python" - <<'PY'
import hashlib
import json
import py_compile
from datetime import datetime, timezone
from pathlib import Path
import torch

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_qila_20261105"
source = root / "scanrefer_restart_20261102/stage_b_source"
checkpoint = root / "scanrefer_restart_20261102/stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
runner = trial / "train_scanrefer_qila.py"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 2 and len(saved["model"]) == 1168
assert saved["optimizer"]["state"] and saved["scheduler"]
assert not torch.cuda.is_initialized()
py_compile.compile(str(runner), doraise=True)
payload = {
    "prepared_utc": datetime.now(timezone.utc).isoformat(),
    "run_dir": str(trial / "run"),
    "source": str(source),
    "runner": str(runner),
    "source_sha256": {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "main_utils.py", "models/losses.py",
            "models/mcln.py", "train_dist_mod.py",
        )
    },
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "initialization_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "initialization_epoch": saved["epoch"],
    "initialization_model_tensors": len(saved["model"]),
    "optimizer_restored": False,
    "scheduler_restored": False,
    "budget_epochs": 8,
    "early_stopping": False,
    "qila_rank_loss_weight": 0.3,
    "cuda_initialized": False,
}
(trial / "preparation.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY
