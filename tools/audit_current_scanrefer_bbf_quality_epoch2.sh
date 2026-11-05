#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import torch

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_bbf_quality_20261104"
run = trial / "run/scanrefer/scanrefer_bbf_quality_20261104/1793849982"
history = json.loads((run / "full10_semantic_history.json").read_text())
row, = [item for item in history["rows"] if item["epoch"] == 2]
best = json.loads((run / "semantic_best.json").read_text())
checkpoint = Path(row["checkpoint"])
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

assert row["sample_count"] == 9508
assert row["metric_branch"] == "last_semantic_alignment_top1"
assert (row["hits025"], row["hits050"]) == (5150, 4203)
assert saved["epoch"] == row["epoch"] == 2
assert len(saved["model"]) == row["model_tensors"] == 1169
assert saved["optimizer"]["state"] and saved["scheduler"]
assert digest == row["sha256"]
assert saved["config"].use_bbf_quality_adapter
assert saved["config"].bbf_quality_adapter_train_only
assert saved["config"].model_only_initialization
assert saved["config"].checkpoint_path.endswith(
    "1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
)
assert best["epoch"] == 1
assert (best["hits025"], best["hits050"]) == (5383, 4518)
assert history["early_stopping"] is False
assert history["stop_requested"] is False

result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run": str(run),
    "metric": row,
    "semantic_best_epoch": best["epoch"],
    "checkpoint_sha256": digest,
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_links": checkpoint.stat().st_nlink,
    "checkpoint_own_epoch": saved["epoch"],
    "model_tensors": len(saved["model"]),
    "optimizer_nonempty": bool(saved["optimizer"]["state"]),
    "scheduler_nonempty": bool(saved["scheduler"]),
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": (
        "Current ScanRefer BBF-quality epoch2 checkpoint audited on CPU; "
        "no model construction, forward, evaluation, training, restart, "
        "or deletion."
    ),
}
assert not result["cuda_initialized"]
(run / "epoch2_cpu_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
