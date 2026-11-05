#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_bbf_quality_20261104")
run = trial / "run/scanrefer/scanrefer_bbf_quality_20261104/1793849982"
history = json.loads((run / "full10_semantic_history.json").read_text())
rows = history["rows"]
assert history["early_stopping"] is False
assert history["stop_requested"] is False
assert [row["epoch"] for row in rows] == list(range(1, len(rows) + 1))
assert all(row["sample_count"] == 9508 for row in rows)
assert all(row["metric_branch"] == "last_semantic_alignment_top1" for row in rows)
best = max(rows, key=lambda row: (row["hits050"], row["hits025"], -row["epoch"]))
manifest = json.loads((run / "semantic_best.json").read_text())
assert manifest == best
retention = json.loads((run / "best_latest_retention.json").read_text())
latest = rows[-1]
assert retention["best_epoch"] == best["epoch"]
assert retention["latest_epoch"] == latest["epoch"]
assert retention["deletion_applied"]

audits = []
for epoch in sorted({best["epoch"], latest["epoch"]}):
    row, = [row for row in rows if row["epoch"] == epoch]
    checkpoint = Path(row["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["epoch"] == epoch
    assert len(saved["model"]) == row["model_tensors"] == 1169
    assert saved["optimizer"]["state"] and saved["scheduler"]
    config = saved["config"]
    assert config.use_bbf_quality_adapter and config.bbf_quality_adapter_train_only
    assert config.model_only_initialization and config.max_epoch == 10
    assert config.checkpoint_path.endswith("1793674354/preserved_checkpoints/ckpt_epoch_2.pth")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == row["sha256"]
    audits.append({
        "epoch": epoch, "sha256": digest.hexdigest(),
        "checkpoint": str(checkpoint), "own_cpu_epoch": saved["epoch"],
        "model_tensors": len(saved["model"]),
        "optimizer_nonempty": bool(saved["optimizer"]["state"]),
        "scheduler_nonempty": bool(saved["scheduler"]),
        "bytes": checkpoint.stat().st_size, "links": checkpoint.stat().st_nlink,
    })
    del saved

for row in rows:
    row["checkpoint_exists_at_collection"] = Path(row["checkpoint"]).is_file()
lines = (trial / "train.log").read_text(errors="replace").replace("\r", "\n").splitlines()
summary = [
    line.strip()[:1400] for line in lines
    if any(token in line for token in (
        "Test evaluation", "semantic alignment", "last_", "overall25",
        "overall50", "total time",
    ))
]
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run": str(run), "history": history, "best": best,
    "latest": latest, "retained_checkpoint_audits": audits,
    "retention": retention, "log_metric_summary": summary,
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": "Only registered current ScanRefer BBF-quality run; full metric history and best/latest CPU checkpoint audits. No model construction, forward, evaluation, training, restart, or deletion.",
}
assert not result["cuda_initialized"]
destination = run / "current_monitor_history.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "observed_utc": result["observed_utc"], "epochs": len(rows),
    "best": best, "latest": latest, "audits": audits,
    "log_metric_summary_tail": summary[-24:], "destination": str(destination),
    "cuda_initialized": result["cuda_initialized"],
}))
PY
