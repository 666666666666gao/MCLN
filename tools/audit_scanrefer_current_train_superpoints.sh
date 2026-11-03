#!/usr/bin/env bash
set -euo pipefail
date -u -Is
ps -p 207977,209444 -o pid,etime,args | cut -c1-260
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

torch.set_num_threads(1)
stage = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102")
data = Path("/root/autodl-tmp/PV-Ground/data")
annotations = data / "ScanRefer/ScanRefer_filtered_train.json"
annotation_bytes = annotations.read_bytes()
annotation_sha = hashlib.sha256(annotation_bytes).hexdigest()
fingerprint = json.loads((stage / "current_data_fingerprint.json").read_text())
expected, = [item for item in fingerprint["annotations"]
             if item["path"] == str(annotations.relative_to(data))]
assert annotation_sha == expected["sha256"]
examples = json.loads(annotation_bytes)
scanrefer_ids = set((annotations.parent / "ScanRefer_filtered_train.txt").read_text().splitlines())
examples = [item for item in examples if item["scene_id"] in scanrefer_ids]
annotation_scenes = {item["scene_id"] for item in examples}
paths = sorted((data / "superpoints/train").glob("*_superpoint.pth"))
scenes = {path.name.removesuffix("_superpoint.pth") for path in paths}
metadata = stage / "source/data/meta_data/scannetv2_train.txt"
metadata_bytes = metadata.read_bytes()
metadata_scenes = set(metadata_bytes.decode().splitlines())
assert scenes == metadata_scenes
assert annotation_scenes <= scenes
validation = json.loads((data / "ScanRefer/ScanRefer_filtered_val.json").read_text())
validation_scenes = {item["scene_id"] for item in validation}
overlap = sorted(scenes & validation_scenes)
run, = (stage / "stage_b/scanrefer/scanrefer_stage_b_20261102").iterdir()
config = json.loads((run / "config.json").read_text())
assert config["joint_det"] and config["dataset"] == ["scanrefer"]
train_log = (stage / "stage_b_train.log").read_text()
training_examples = int(re.findall(r"length of training dataset: (\d+)", train_log)[-1])
assert (training_examples - len(examples)) % 10 == 0
rows = []
manifest = hashlib.sha256()
for path in paths:
    scene = path.name.removesuffix("_superpoint.pth")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    labels = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(labels, np.ndarray), (scene, type(labels))
    assert labels.ndim == 1 and len(labels) == 50000, (scene, labels.shape)
    assert labels.dtype.kind in "iu", (scene, str(labels.dtype))
    values, sizes = np.unique(labels, return_counts=True)
    relative = str(path.relative_to(data))
    manifest.update((relative + "\t" + digest + "\n").encode("utf-8"))
    rows.append({
        "scene": scene, "path": relative, "sha256": digest,
        "bytes": len(content), "points": len(labels), "dtype": str(labels.dtype),
        "superpoints": len(values), "min_label": int(values.min()),
        "max_label": int(values.max()), "largest_superpoint_points": int(sizes.max()),
        "singleton_superpoints": int((sizes == 1).sum()),
        "scanrefer_annotated": scene in annotation_scenes,
    })
counts = np.asarray([row["superpoints"] for row in rows])
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "training_annotation_examples": len(examples),
    "training_annotation_scene_count": len(annotation_scenes),
    "training_annotation_sha256": annotation_sha,
    "training_annotation_matches_prior_fingerprint": True,
    "scannet_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    "all_train_files_match_scannet_metadata": True,
    "joint_det": True, "scannet_repetitions": 10,
    "observed_training_dataset_examples": training_examples,
    "scannet_examples_inferred_from_training_length": (training_examples - len(examples)) // 10,
    "scene_count": len(rows), "validation_scene_count": len(validation_scenes),
    "training_validation_overlap": overlap, "points_per_scene": 50000,
    "total_bytes": sum(row["bytes"] for row in rows),
    "manifest_sha256": manifest.hexdigest(),
    "superpoints_min": int(counts.min()), "superpoints_max": int(counts.max()),
    "superpoints_quantiles": dict(zip(
        ("p10", "p25", "p50", "p75", "p90"),
        np.quantile(counts, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist(),
    )),
    "single_superpoint_scenes": [row["scene"] for row in rows if row["superpoints"] == 1],
    "negative_label_scenes": [row["scene"] for row in rows if row["min_label"] < 0],
    "rows": rows, "cuda_initialized": torch.cuda.is_initialized(),
    "historical_mesh_equivalence": "unverified; no reference manifest compared",
    "scope": "Remote CPU data-only audit of all training-directory superpoint files, matching ScanNet train metadata; no model import, forward, evaluation or training. Detection example count is inferred from source recipe and observed length, not reconstructed object vocabulary filtering.",
}
assert not result["cuda_initialized"]
(stage / "current_all_train_superpoint_statistics.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
PY
ps -p 207977,209444 -o pid,etime,args | cut -c1-260
