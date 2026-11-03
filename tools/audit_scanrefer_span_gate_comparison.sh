#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_span_gate_comparison_20261103"
sys.path.insert(0, str(root))
from mcln_metric_keeper import completed_rows

preparation = json.loads((trial / "preparation.json").read_text())
assert preparation["checkpoint_sha256"] == (
    "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8")
for name, expected in preparation["source_files_sha256"].items():
    assert hashlib.sha256((trial / "source" / name).read_bytes()).hexdigest() == expected

receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "initialization_sha256": preparation["checkpoint_sha256"],
    "source_files_sha256": preparation["source_files_sha256"],
    "conditions": {},
    "scope": "Only this registered pilot. Independent log/receipt and own CPU checkpoint audit; no model construction, forward, evaluation or training.",
}
for condition, weight in (("control", 0.0), ("span", 0.1)):
    complete_path = trial / f"{condition}_complete.json"
    if not complete_path.exists():
        receipt["conditions"][condition] = {"state": "not_completed"}
        continue
    complete = json.loads(complete_path.read_text())
    start = json.loads((trial / f"{condition}_start.json").read_text())
    result = json.loads((trial / f"{condition}_result.json").read_text())
    run = Path(complete["run_dir"])
    assert run.parent == trial / condition / "scanrefer" / f"scanrefer_span_gate_{condition}_20261103"
    config = json.loads((run / "config.json").read_text())
    assert start["condition"] == complete["condition"] == result["condition"] == condition
    assert complete["full_validation_expected"]
    assert start["checkpoint_sha256"] == preparation["checkpoint_sha256"]
    assert not start["optimizer_restored"] and not start["scheduler_restored"]
    assert config["checkpoint_path"] == preparation["checkpoint"]
    assert config["pp_checkpoint"] is None and config["model_only_initialization"]
    assert config["pvground_span_gate_loss_weight"] == weight == result["gate_loss_weight"]
    assert config["max_epoch"] == 1 and config["max_train_batches"] == 0
    assert config["batch_size"] == 8 and config["gradient_accumulation_steps"] == 6
    assert config["rng_seed"] == 0 and config["expected_eval_sample_count"] == 9508
    assert config["use_pvground_innovations"] and not config["pvground_train_only"]
    rows = completed_rows((trial / f"{condition}.log").read_text(errors="replace"), "scanrefer")
    assert len(rows) == 1
    for key in ("epoch", "sample_count", "hits025", "hits050"):
        assert rows[0][key] == result[key]
    assert result["epoch"] == 1 and result["sample_count"] == 9508
    assert result["metric_branch"] == "last_semantic_alignment_top1"
    checkpoint = run / "preserved_checkpoints" / "ckpt_epoch_1.pth"
    assert Path(result["checkpoint"]) == checkpoint
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 1 and len(saved["model"]) == 1168
    assert saved["optimizer"]["state"] and saved["scheduler"]
    assert saved["config"].pvground_span_gate_loss_weight == weight
    groups = saved["optimizer"]["param_groups"]
    assert [(group["name"], group["lr"]) for group in groups] == [
        ("native", 2e-6), ("backbone", 4e-6), ("pvground", 2e-5)]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == result["checkpoint_sha256"]
    receipt["conditions"][condition] = {
        "state": "audited", "result": result, "config": config,
        "log_row": rows[0], "own_cpu_epoch": saved["epoch"],
        "model_tensors": len(saved["model"]), "nonempty_optimizer": True,
        "has_scheduler": True, "checkpoint_sha256": digest.hexdigest(),
        "bytes": checkpoint.stat().st_size, "links": checkpoint.stat().st_nlink,
        "delta_vs_b2": {
            "hits025": result["hits025"] - 5483,
            "hits050": result["hits050"] - 4656,
        },
        "target_met": result["hits025"] >= 5705 and result["hits050"] >= 4945,
    }
    del saved
control, span = (receipt["conditions"][name] for name in ("control", "span"))
if control["state"] == span["state"] == "audited":
    receipt["span_minus_control"] = {
        key: span["result"][key] - control["result"][key]
        for key in ("hits025", "hits050")
    }
    assert int((trial / "pipeline_exit_code").read_text()) == 0
    receipt["pipeline_complete"] = json.loads((trial / "complete.json").read_text())
receipt["cuda_initialized"] = torch.cuda.is_initialized()
assert not receipt["cuda_initialized"]
(trial / "independent_checkpoint_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
