#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
date -u -Is
screen -ls | grep -E 'pvg_scanrefer_(restart_transition|best_latest_retention)' || true
ps -eo pid,ppid,etime,args \
    | grep -E '[t]rain_scanrefer_stage_b.py|[r]etain_mcln_best_latest.py --dataset scanrefer' \
    | cut -c1-300 || true
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
STAGE="$stage" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
for name in ("stage_b_start.json", "stage_b_complete.json", "stage_b_pipeline_exit_code"):
    path = stage / name
    if path.exists():
        if name == "stage_b_start.json":
            payload = json.loads(path.read_text())
            decision = payload["parent_decision"]
            print(name, json.dumps({
                "phase": payload["phase"],
                "initialization_checkpoint": decision["initialization_checkpoint"],
                "initialization_sha256": decision["initialization_sha256"],
                "selected_epoch": decision["selected"]["epoch"],
                "optimizer_restored": payload["optimizer_restored"],
                "scheduler_restored": payload["scheduler_restored"],
                "matched_selected_hits": payload["matched_validation"]["selected"]["hits"],
            }))
        else:
            print(name, path.read_text())
for directory in stage.glob("stage_b/scanrefer/scanrefer_stage_b_20261102/*"):
    for name in ("config.json", "pvground_first_update.json", "semantic_best.json",
                 "stage_b_semantic_history.json"):
        path = directory / name
        if path.exists():
            payload = json.loads(path.read_text())
            if name == "config.json":
                payload = {key: payload.get(key) for key in (
                    "checkpoint_path", "model_only_initialization", "pp_checkpoint",
                    "pvground_train_only", "batch_size", "gradient_accumulation_steps",
                    "lr", "lr_backbone", "text_encoder_lr",
                    "candidate_local_visual_lr", "max_epoch", "rng_seed",
                )}
            elif name == "stage_b_semantic_history.json":
                rows = payload["rows"]
                payload = {
                    "complete_epochs": len(rows), "latest": rows[-1],
                    "decrease_streak": payload["decrease_streak"],
                    "stop_requested": payload["stop_requested"],
                    "final_target_met": payload["final_target_met"],
                }
            print(str(path), json.dumps(payload))
path = stage / "best_latest_retention.log"
if path.exists():
    print("RETENTION_TAIL", "\n".join(path.read_text().splitlines()[-2:]))
path = stage / "stage_b_train.log"
if path.exists():
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 12000))
        lines = source.read().decode(errors="replace").replace("\r", "\n").splitlines()
    print("LOG_TAIL", "\n".join(line for line in lines[-15:] if len(line) < 1800))
PY
