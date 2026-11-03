#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
CUDA_VISIBLE_DEVICES="" STAGE="$stage" \
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
stage = Path(os.environ["STAGE"])
data = Path("/root/autodl-tmp/PV-Ground/data")
manifest = json.loads((stage / "current_data_fingerprint.json").read_text())
rows = []
for item in manifest["validation_superpoint_files"]:
    path = data / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    labels = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(labels, np.ndarray), type(labels)
    assert labels.ndim == 1 and len(labels) == 50000, (path.name, labels.shape)
    values, sizes = np.unique(labels, return_counts=True)
    rows.append({
        "scene": path.name.removesuffix("_superpoint.pth"),
        "points": int(len(labels)), "superpoints": int(len(values)),
        "min_label": int(values.min()), "max_label": int(values.max()),
        "largest_superpoint_points": int(sizes.max()),
        "singleton_superpoints": int((sizes == 1).sum()),
        "sha256": item["sha256"],
    })
counts = np.asarray([row["superpoints"] for row in rows])
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest_sha256": manifest["manifest_sha256"],
    "scene_count": len(rows), "points_per_scene": 50000,
    "superpoints_min": int(counts.min()), "superpoints_max": int(counts.max()),
    "superpoints_quantiles": dict(zip(
        ("p10", "p25", "p50", "p75", "p90"),
        np.quantile(counts, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist(),
    )),
    "single_superpoint_scenes": [row["scene"] for row in rows if row["superpoints"] == 1],
    "negative_label_scenes": [row["scene"] for row in rows if row["min_label"] < 0],
    "rows": rows, "cuda_initialized": torch.cuda.is_initialized(),
    "historical_mesh_equivalence": "unverified; no reference manifest compared",
    "scope": "Remote CPU data-only audit; no model import, forward, evaluation or training.",
}
assert not result["cuda_initialized"]
(stage / "current_superpoint_statistics.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
PY
