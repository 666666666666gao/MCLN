set -e
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

current = {"scanrefer": "1793345477", "nr3d": "1793419199", "sr3d": "1793419200"}
result = {"observed_at": datetime.now(timezone.utc).isoformat(), "runs": []}
for dataset, expected in (("scanrefer", 9508), ("nr3d", 7899), ("sr3d", 17726)):
    root = Path("/root/autodl-tmp/pvground_scanrefer_module_ft_20261030/logs_semantic/scanrefer/scanrefer_module_ft_semantic") if dataset == "scanrefer" else Path(f"/root/autodl-tmp/pvground_{dataset}_module_training_20261030/logs/{dataset}/{dataset}_module_ft")
    if not (root / current[dataset] / "config.json").exists():
        continue
    paths = [root / current[dataset]] if dataset == "scanrefer" else sorted(root.iterdir())
    for run in paths:
        if not run.is_dir():
            continue
        path = run / "native_metrics.jsonl"
        raw = path.read_bytes() if path.exists() else b""
        metrics = [json.loads(line) for line in raw.splitlines()]
        assert all(row["rows"] == expected and row["primary_score"] == "last/bbf"
                   and row["same_complete_model"] for row in metrics)
        assert len({row["epoch"] for row in metrics}) == len(metrics)
        result["runs"].append(dict(
            dataset=dataset, run=run.name,
            current_run=run.name == current[dataset],
            source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
            metrics=metrics,
        ))
output = Path("/tmp/pvground_full_epoch_history.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
