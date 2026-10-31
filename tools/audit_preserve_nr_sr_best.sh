set -e
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
import os
from pathlib import Path
import torch

for dataset, run, expected in (("nr3d", "1793419199", 7899),
                               ("sr3d", "1793419200", 17726)):
    root = Path(f"/root/autodl-tmp/pvground_{dataset}_module_training_20261030/logs/{dataset}/{dataset}_module_ft/{run}")
    if not (root / "config.json").exists():
        continue
    metrics_path = root / "native_metrics.jsonl"
    if not metrics_path.exists():
        print(json.dumps(dict(dataset=dataset, run=run, evaluation="pending")))
        continue
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert all(row["rows"] == expected and row["primary_score"] == "last/bbf"
               and row["same_complete_model"] for row in metrics)
    by_epoch = {row["epoch"]: row for row in metrics}
    assert len(by_epoch) == len(metrics)
    trained = [row for row in metrics if row["epoch"] > 0]
    selected = max(trained, key=lambda row: (row["hits025"], row["hits050"])) if trained else None
    print(json.dumps(dict(dataset=dataset, run=run, selected_trained_acc025_first=selected)))
    for name in ("best.pth", "latest.pth"):
        path = root / name
        if not path.exists():
            print(json.dumps(dict(checkpoint=name, state="pending")))
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert all(key in checkpoint for key in
                   ("model", "optimizer", "scheduler", "rng", "architecture", "epoch"))
        epoch = checkpoint["epoch"]
        own = by_epoch[epoch]
        print(json.dumps(dict(checkpoint=str(path), epoch=epoch,
                              evaluation_at_checkpoint_epoch=own,
                              retained_metrics_is_historical_best=checkpoint["retained_metrics"])))
        if name == "latest.pth" and selected is not None and epoch == selected["epoch"]:
            alias = root / f"best_trained_acc025_epoch{epoch}.pth"
            created = not alias.exists()
            if created:
                os.link(path, alias)
            assert os.path.samefile(path, alias)
            print(json.dumps(dict(preserved=str(alias), newly_created=created,
                                  epoch=epoch, evaluation=own,
                                  selection="hits025_then_hits050",
                                  link_count=alias.stat().st_nlink)))
        del checkpoint
    if selected is not None:
        alias = root / f"best_trained_acc025_epoch{selected['epoch']}.pth"
        print(json.dumps(dict(selected_checkpoint_exists=alias.exists(),
                              selected_checkpoint=str(alias))))
PY
