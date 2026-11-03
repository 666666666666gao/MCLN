#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import shutil

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
directory = root / "runs/sr3d"
audit = json.loads((directory / "latest_complete_epoch_14.json").read_text())
assert audit["epoch"] == audit["checkpoint"]["epoch"] == 14
assert audit["val_examples"] == 17726
assert audit["checkpoint"]["model_tensors"] == 1134
assert audit["checkpoint"]["optimizer"] and audit["checkpoint"]["scheduler"]
checkpoint = root / audit["checkpoint"]["preserved_checkpoint"]
digest = hashlib.sha256()
with checkpoint.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "dataset": "sr3d", "run": "1793552465",
    "checkpoint": str(checkpoint.relative_to(root)),
    "checkpoint_sha256": digest.hexdigest(),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "hardlink_count_at_hash": checkpoint.stat().st_nlink,
    "free_bytes": shutil.disk_usage(directory).free,
    "live_pids": [pid for pid in (92354, 129032, 148725)
                  if (Path("/proc") / str(pid) / "cmdline").exists()],
    "scope": "Remote file identity; own CPU checkpoint epoch/model/optimizer/scheduler audit is in mcln-sr3d-epoch14.json.",
}
(directory / "checkpoint_identity_epoch_14.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
