#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
STAGE="$stage" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path

stage = Path(os.environ["STAGE"])
data = Path("/root/autodl-tmp/PV-Ground/data")

def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(data)), "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

files = sorted((data / "superpoints/val").glob("*_superpoint.pth"))
assert len(files) == 312, len(files)
rows = [identity(path) for path in files]
manifest_digest = hashlib.sha256()
for row in rows:
    manifest_digest.update(
        (row["path"] + "\t" + row["sha256"] + "\n").encode("utf-8")
    )
annotations = [
    identity(data / "ScanRefer/ScanRefer_filtered_val.json"),
    identity(data / "ScanRefer/ScanRefer_filtered_train.json"),
]
sources = {}
for name in ("src/joint_det_dataset.py", "src/visual_data_handlers.py"):
    sources[name] = hashlib.sha256((stage / "source" / name).read_bytes()).hexdigest()
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "data_root": str(data), "validation_superpoint_files": rows,
    "validation_superpoint_count": len(rows),
    "validation_superpoint_total_bytes": sum(row["bytes"] for row in rows),
    "manifest_sha256": manifest_digest.hexdigest(),
    "annotations": annotations, "dataset_sources": sources,
    "historical_mesh_equivalence": "unverified; no reference manifest compared",
    "scope": "Read-only file fingerprints; no tensor load, model import or evaluation.",
}
target = stage / "current_data_fingerprint.json"
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items()
                  if key != "validation_superpoint_files"}))
PY
