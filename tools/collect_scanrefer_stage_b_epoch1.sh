#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
sys.path.insert(0, str(root))
from mcln_metric_keeper import completed_rows

stage = root / "scanrefer_restart_20261102"
run = stage / "stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354"
history = json.loads((run / "stage_b_semantic_history.json").read_text())
row, = [item for item in history["rows"] if item["epoch"] == 1]
assert row["sample_count"] == 9508 and row["checkpoint_epoch"] == 1
text = (stage / "stage_b_train.log").read_text(errors="replace")
log_row, = [item for item in completed_rows(text, "scanrefer") if item["epoch"] == 1]
assert (row["hits025"], row["hits050"]) == (
    log_row["hits025"], log_row["hits050"])
checkpoint = Path(row["checkpoint"])
assert checkpoint.parent == run / "preserved_checkpoints"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 1 and len(saved["model"]) == 1168
assert saved["optimizer"]["state"] and saved["scheduler"]
config = saved["config"]
assert config.model_only_initialization and not config.pvground_train_only
assert config.use_pvground_innovations and config.pp_checkpoint is None
assert Path(config.checkpoint_path) == stage / "stage_b_initialization_epoch_3.pth"
assert config.batch_size == 8 and config.gradient_accumulation_steps == 6
groups = [(item["name"], item["lr"])
          for item in saved["optimizer"]["param_groups"]]
assert groups == [("native", 2e-6), ("backbone", 4e-6), ("pvground", 2e-5)]
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
markers = []
active = False
for line in text.splitlines():
    if "epoch 1, total time" in line:
        active = True
        markers.append(line)
    elif "epoch 2, total time" in line:
        break
    elif active and "last_ semantic alignment Acc0." in line:
        markers.append(line)
assert len(markers) == 3
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run": run.name, "phase": "stage_b",
    "row": row, "independent_log_row": log_row,
    "log_markers": markers,
    "checkpoint_sha256": digest.hexdigest(),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_links": checkpoint.stat().st_nlink,
    "checkpoint_epoch_cpu": saved["epoch"],
    "model_tensors": len(saved["model"]),
    "nonempty_optimizer": True, "has_scheduler": True,
    "optimizer_groups": groups,
    "semantic_best": json.loads((run / "semantic_best.json").read_text()),
    "decrease_streak": history["decrease_streak"],
    "stop_requested": history["stop_requested"],
    "final_target_met": history["final_target_met"],
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": "Full Stage B epoch1 semantic counters crosschecked against complete log and own CPU checkpoint. No model import, construction, forward or evaluation.",
}
assert not receipt["cuda_initialized"]
(stage / "stage_b_epoch1_audited.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
