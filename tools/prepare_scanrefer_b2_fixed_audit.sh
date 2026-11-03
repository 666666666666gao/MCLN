#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
audit="$root/offline_audits/b2_fixed_audit_20261103_lrpositive"
test ! -e "$audit"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
stage = root / "scanrefer_restart_20261102"
staging = root / "offline_audits/observations_v2"
audit = root / "offline_audits/b2_fixed_audit_20261103_lrpositive"
checkpoint = stage / "stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354/preserved_checkpoints/ckpt_epoch_2.pth"
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
sha = digest.hexdigest()
assert sha == "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 2 and len(saved["model"]) == 1168
assert saved["optimizer"]["state"] and saved["scheduler"]
assert saved["config"].use_pvground_innovations
assert not torch.cuda.is_initialized()
paths = [
    staging / "tools/eval_pvground_audit.py",
    staging / "tools/pvground_forward_audit.py",
    staging / "tools/pvground_semantic_audit.py",
    stage / "source/train_dist_mod.py",
    stage / "source/main_utils.py",
]
receipt = {
    "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint": str(checkpoint), "checkpoint_epoch_cpu": saved["epoch"],
    "checkpoint_sha256": sha, "model_tensors": len(saved["model"]),
    "nonempty_optimizer": True, "has_scheduler": True,
    "conditions": ["none", "reader", "support", "span"],
    "samples_required_per_condition": 9508, "batch_size": 8, "seed": 0,
    "recording_repository_commit": "fcc1e7e",
    "source_files_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    },
    "cuda_initialized": False,
    "scope": "Fixed trained B2 inference interventions only. No training, no old-run inspection, no local model operations; no individual-module gain claim.",
}
audit.mkdir()
(audit / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
