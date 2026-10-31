set -euo pipefail
cd /root/autodl-tmp/pvground_clean_baseline_20261031
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import re

paths = list(Path("results").glob("*/*/*/config.json"))
directory = max(paths, key=lambda path: int(path.parent.name)).parent
current = directory / "load_audit.json"
audit = json.loads(current.read_text())
report = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "result_directory": str(directory.resolve()),
    "load_audit": audit,
    "config": json.loads((directory / "config.json").read_text()),
    "source_hashes": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in map(Path, ["models/pv_ground.py", "models/losses.py",
                               "models/backbone_module.py", "src/grounding_evaluator.py",
                               "src/joint_det_dataset.py", "main_utils.py",
                               "train_pvground_clean_baseline.py"])
    },
    "backbone_kernel_witness": json.loads(Path("backbone_kernel_witness.json").read_text()),
    "first_update": None,
    "latest_eval_progress": None,
    "native_metrics": [
        json.loads(line) for line in (directory / "native_metrics.jsonl").read_text().splitlines()
    ] if (directory / "native_metrics.jsonl").exists() else [],
}
if (directory / "first_update.json").exists():
    report["first_update"] = json.loads((directory / "first_update.json").read_text())
matches = re.findall(r"CLEAN_VAL epoch=(\d+) batch=(\d+)/(\d+)", Path("train_clean.log").read_text())
if matches:
    epoch, batch, total = map(int, matches[-1])
    report["latest_eval_progress"] = {"epoch": epoch, "batch": batch, "total_batches": total}
Path("startup_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({
    "result_directory": report["result_directory"],
    "source_sha256": audit["source_sha256"],
    "loaded_tensors": len(audit["loaded_tensors"]),
    "excluded_tensors": len(audit["excluded_tensors"]),
    "new_tensors": len(audit["new_tensors"]),
    "first_update": report["first_update"],
    "latest_eval_progress": report["latest_eval_progress"],
}))
PY
