#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
STAGE="$stage" CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
import os
from pathlib import Path
import shutil
stage = Path(os.environ["STAGE"]).resolve()
start = json.loads((stage / "stage_b_start.json").read_text())
assert start["parent_decision"]["selected"]["epoch"] == 3
assert int((stage / "protocol_audit/pipeline_exit_code").read_text()) == 0
update, = stage.glob("stage_b/scanrefer/scanrefer_stage_b_20261102/*/pvground_first_update.json")
receipt = json.loads(update.read_text())
assert receipt["finite_loss"] and receipt["optimizer_step_complete"]
assert all(row["gradient_norm"] > 0 and row["parameter_delta_l2"] > 0
           for row in receipt["modules"].values())
preserved = stage / "preserved_checkpoints"
keep = preserved / "ckpt_epoch_3.pth"
assert keep.is_file()
before = shutil.disk_usage(stage).free
removed = []
for epoch in (1, 2):
    path = preserved / f"ckpt_epoch_{epoch}.pth"
    assert path.resolve().parent == preserved.resolve() and not path.is_symlink()
    assert path.stat().st_nlink == 1
    path.unlink()
    removed.append(str(path))
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "removed": removed, "kept_epoch": 3,
    "role": "Stage A best/latest and Stage B initialization input",
    "free_bytes_before": before, "free_bytes_after": shutil.disk_usage(stage).free,
    "stage_b_first_update": receipt,
}
(stage / "stage_a_weight_cleanup.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
