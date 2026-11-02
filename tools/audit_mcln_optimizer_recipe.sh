#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
for dataset, epoch in (("scanrefer", 9), ("nr3d", 22), ("sr3d", 8)):
    path = root / "runs" / dataset / "preserved_checkpoints" / f"ckpt_epoch_{epoch}.pth"
    if not path.exists():
        continue
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == epoch
    config = vars(saved["config"])
    optimizer = saved["optimizer"]
    groups = [
        {
            "group_index": index,
            "parameter_count": len(group["params"]),
            "lr": group["lr"],
            "initial_lr": group.get("initial_lr"),
            "weight_decay": group["weight_decay"],
        }
        for index, group in enumerate(optimizer["param_groups"])
    ]
    steps = [
        int(state["step"]) for state in optimizer["state"].values()
        if "step" in state
    ]
    receipt = {
        "dataset": dataset,
        "epoch": epoch,
        "checkpoint": str(path),
        "model_tensors": len(saved["model"]),
        "optimizer_groups": groups,
        "optimizer_state_count": len(optimizer["state"]),
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
        "scheduler": saved["scheduler"],
        "config": {
            key: config[key] for key in (
                "batch_size", "gradient_accumulation_steps", "lr", "lr_backbone",
                "lr_decay_epochs", "max_epoch", "checkpoint_path", "pp_checkpoint",
                "model_only_initialization", "use_pvground_innovations",
            )
        },
        "cuda_initialized": torch.cuda.is_initialized(),
        "limitation": "Saved checkpoint optimizer/scheduler audit; no live forward or performance attribution.",
    }
    destination = root / "offline_audits" / "optimizer_recipe"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{dataset}-e{epoch}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
PY
