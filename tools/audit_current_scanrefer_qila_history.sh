#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path

import torch

torch.set_num_threads(1)
trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_qila_20261105")
run = trial / "run"
history_path = run / "qila_semantic_history.json"
history = json.loads(history_path.read_text())
rows = history["rows"]
assert rows
assert [row["epoch"] for row in rows] == list(range(1, len(rows) + 1))
assert all(row["sample_count"] == 9508 for row in rows)
assert all(row["metric_branch"] == "last_semantic_alignment_top1" for row in rows)
best = max(rows, key=lambda row: (row["hits050"], row["hits025"], -row["epoch"]))
latest = rows[-1]

audits = []
for name, row in {"best": best, "latest": latest}.items():
    checkpoint = run / f"semantic_{name}.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["epoch"] == row["epoch"]
    assert len(saved["model"]) == row["model_tensors"] == 1168
    assert saved["optimizer"]["state"]
    assert saved["scheduler"]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == row["sha256"]
    audits.append({
        "role": name,
        "epoch": row["epoch"],
        "hits025": row["hits025"],
        "hits050": row["hits050"],
        "sample_count": row["sample_count"],
        "metric_branch": row["metric_branch"],
        "checkpoint": str(checkpoint),
        "sha256": digest.hexdigest(),
        "own_cpu_epoch": saved["epoch"],
        "model_tensors": len(saved["model"]),
        "optimizer_nonempty": bool(saved["optimizer"]["state"]),
        "scheduler_nonempty": bool(saved["scheduler"]),
        "model_only_initialization_config": bool(
            saved["config"].model_only_initialization
        ),
    })
    del saved

assert not torch.cuda.is_initialized()
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "trial": str(trial),
    "history_path": str(history_path),
    "rows": rows,
    "best": best,
    "latest": latest,
    "retained_checkpoint_audits": audits,
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": (
        "Current isolated ScanRefer QILA trial only. Reads canonical "
        "semantic history and CPU-audits semantic_best/semantic_latest; "
        "no model construction, forward, evaluation, training, restart "
        "or deletion."
    ),
}
destination = run / "current_history_cpu_audit.json"
destination.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({
    "observed_utc": receipt["observed_utc"],
    "epochs": len(rows),
    "best": best,
    "latest": latest,
    "retained_checkpoint_audits": audits,
    "destination": str(destination),
    "cuda_initialized": receipt["cuda_initialized"],
}))
PY
