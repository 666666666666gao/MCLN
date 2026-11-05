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
row, = [item for item in history["rows"] if item["epoch"] == 1]
checkpoint = Path(row["checkpoint"])
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert row["sample_count"] == 9508
assert saved["epoch"] == row["epoch"] == 1
assert len(saved["model"]) == row["model_tensors"] == 1169
assert saved["optimizer"]["state"] and saved["scheduler"]
assert digest == row["sha256"]
assert saved["config"].use_bbf_quality_adapter
assert saved["config"].bbf_quality_adapter_train_only
assert saved["config"].model_only_initialization
assert saved["config"].checkpoint_path.endswith("preserved_checkpoints/ckpt_epoch_2.pth")
assert (row["hits025"], row["hits050"]) == (5383, 4518)
assert row["metric_branch"] == "last_semantic_alignment_top1"
log = (trial / "train.log").read_text(errors="replace")
assert "Test evaluation" in log
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run": str(run),
    "metric": row,
    "checkpoint_sha256": digest,
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_links": checkpoint.stat().st_nlink,
    "checkpoint_own_epoch": saved["epoch"],
    "model_tensors": len(saved["model"]),
    "optimizer_nonempty": bool(saved["optimizer"]["state"]),
    "scheduler_nonempty": bool(saved["scheduler"]),
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": "Current BBF-quality epoch1 metric and its own checkpoint audited on CPU; no model construction, forward, evaluation, training, restart, or deletion.",
}
assert not result["cuda_initialized"]
(run / "epoch1_cpu_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
