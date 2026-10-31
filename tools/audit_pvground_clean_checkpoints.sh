set -euo pipefail
cd /root/autodl-tmp/pvground_clean_baseline_20261031
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
directory = max(Path("results").glob("*/*/*/config.json"), key=lambda p: int(p.parent.name)).parent
rows = [json.loads(line) for line in (directory / "native_metrics.jsonl").read_text().splitlines()]
checkpoints = {}
for name in ("best.pth", "latest.pth"):
    payload = torch.load(directory / name, map_location="cpu", weights_only=False)
    row = next(row for row in rows if row["epoch"] == payload["epoch"])
    assert payload["metrics"] == row
    assert payload["architecture"] == "PointNet++_TopK_noPVI_noTGS"
    assert row["primary_score"] == "last/bbf" and row["same_complete_model"]
    assert {"model", "optimizer", "scheduler", "torch_rng", "cuda_rng", "numpy_rng", "python_rng"} <= payload.keys()
    if payload["epoch"] == 0:
        assert not payload["optimizer"]["state"]
    checkpoints[name] = {
        "epoch": payload["epoch"], "metrics": row,
        "model_tensor_count": len(payload["model"]),
        "optimizer_state_count": len(payload["optimizer"]["state"]),
        "architecture": payload["architecture"], "own_row_matches": True,
    }
    del payload
report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "result_directory": str(directory.resolve()), "device": "cpu", "checkpoints": checkpoints}
Path("checkpoint_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
PY
