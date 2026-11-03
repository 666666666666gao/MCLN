#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
stage = root / "scanrefer_restart_20261102"
sys.path.insert(0, str(root))
from mcln_metric_keeper import completed_rows

cases = [
    ("nr3d", root / "runs/nr3d/nr3d/mcln_pvground_nr3d_20261101/1793553113",
     root / "runs/nr3d", root / "runs/nr3d/train.log", 96960),
    ("sr3d", root / "runs/sr3d/sr3d/mcln_pvground_sr3d_20261101/1793552465",
     root / "runs/sr3d", root / "runs/sr3d/train.log", 92354),
    ("scanrefer", stage / "stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354",
     stage / "stage_b/scanrefer/scanrefer_stage_b_20261102/1793674354",
     stage / "stage_b_train.log", 207977),
]
for dataset, run, directory, log, pid in cases:
    if not run.exists():
        continue
    text = log.read_text(errors="replace")
    rows = completed_rows(text, dataset)
    order = ("hits050", "hits025") if dataset == "scanrefer" else ("hits025", "hits050")
    best = max(rows, key=lambda item: (item[order[0]], item[order[1]], -item["epoch"]))
    latest = max(item["epoch"] for item in rows)
    manifest = json.loads((run / "semantic_best.json").read_text())
    assert (manifest["epoch"], manifest["hits025"], manifest["hits050"]) == (
        best["epoch"], best["hits025"], best["hits050"])
    config = json.loads((run / "config.json").read_text())
    if dataset == "scanrefer":
        assert config["model_only_initialization"] and config["use_pvground_innovations"]
        assert config["pp_checkpoint"] is None and not config["pvground_train_only"]
        history = json.loads((run / "stage_b_semantic_history.json").read_text())
        assert [(r["epoch"], r["hits025"], r["hits050"]) for r in rows] == [
            (r["epoch"], r["hits025"], r["hits050"]) for r in history["rows"]]
    else:
        assert config["checkpoint_path"] is None and config["pp_checkpoint"] is None
        assert not config["use_pvground_innovations"]
        history = None
    audits = []
    for epoch in sorted({best["epoch"], latest}):
        path = directory / "preserved_checkpoints" / f"ckpt_epoch_{epoch}.pth"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["epoch"] == epoch
        assert len(saved["model"]) == (1168 if dataset == "scanrefer" else 1134)
        assert saved["optimizer"]["state"] and saved["scheduler"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        audits.append({
            "epoch": epoch, "checkpoint": str(path), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size, "links": path.stat().st_nlink,
            "own_epoch_cpu": saved["epoch"], "model_tensors": len(saved["model"]),
            "nonempty_optimizer": True, "has_scheduler": True,
        })
        del saved
    for row in rows:
        path = directory / "preserved_checkpoints" / f"ckpt_epoch_{row['epoch']}.pth"
        row["checkpoint_exists_at_collection"] = path.exists()
        row["checkpoint"] = str(path)
    retained_history = [
        json.loads(line)
        for line in (directory / "best_latest_retention_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    receipt = {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset, "run": run.name, "rows": rows,
        "best": best, "latest_epoch": latest, "selection_manifest": manifest,
        "config": config, "retained_checkpoint_audits": audits,
        "retention": json.loads((directory / "best_latest_retention.json").read_text()),
        "retention_history": retained_history,
        "stage_b_history": history,
        "trainer_pid": pid, "trainer_live": (Path("/proc") / str(pid)).exists(),
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True).strip(),
        "screen": subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout,
        "cuda_initialized": torch.cuda.is_initialized(),
        "scope": "Exact current runs only. All completed last semantic Top1 log rows; current best/latest own CPU checkpoints audited and hashed. Deleted epochs have log and historical retention evidence only. No model construction, forward, evaluation or training.",
    }
    if dataset == "scanrefer":
        receipt["terminal_evidence"] = {
            "complete": json.loads((stage / "stage_b_complete.json").read_text()),
            "pipeline_exit_code": int((stage / "stage_b_pipeline_exit_code").read_text()),
            "transition_exit_code": int((stage / "transition_exit_code").read_text()),
        }
    assert not receipt["cuda_initialized"]
    destination = directory / "current_full_history_20261103.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "dataset": dataset, "epochs": len(rows), "best": best,
        "latest": rows[-1], "audits": audits, "retention_latest": receipt["retention"]["latest_epoch"],
        "destination": str(destination), "trainer_live": receipt["trainer_live"],
    }))
PY
