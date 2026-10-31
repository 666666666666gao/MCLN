set -e
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "checkpoint_audit", "/tmp/pvground_checkpoint_update_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for dataset, run_id in (("nr3d", "1793419199"),):
    root = Path(f"/root/autodl-tmp/pvground_{dataset}_module_training_20261030")
    run = root / f"logs/{dataset}/{dataset}_module_ft/{run_id}"
    if not (run / "config.json").exists():
        continue
    history = [json.loads(line) for line in
               (run / "native_metrics.jsonl").read_text().splitlines()]
    selected = max((row for row in history if row["epoch"] > 0),
                   key=lambda row: (row["hits025"], row["hits050"]))
    epoch = selected["epoch"]
    checkpoint = run / f"best_trained_acc025_epoch{epoch}.pth"
    output = root / f"diagnostics/selected_epoch{epoch}_cpu_update_audit.json"
    assert not output.exists()
    result = module.audit(run / "best.pth", checkpoint,
                          run / "trainable_parameters.json",
                          run / "native_metrics.jsonl")
    assert result["epoch"] == epoch and result["own_epoch_metrics"] == selected
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "parameter_deltas"}))
    print("SMALL_PARAMETERS", json.dumps([
        row for row in result["parameter_deltas"] if row["elements"] <= 3]))
    print("OUTPUT", str(output), hashlib.sha256(output.read_bytes()).hexdigest())
PY
