#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
from pathlib import Path
import tarfile

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103")
complete = json.loads((trial / "preflight_complete.json").read_text())
run = Path(complete["run_dir"])
config = json.loads((run / "config.json").read_text())
update = json.loads((run / "pvground_first_update.json").read_text())
gradient = json.loads((trial / "real_batch_gate_gradient.json").read_text())
assert config["max_train_batches"] == 6 and config["pvground_span_gate_loss_weight"] == 0.1
assert config["model_only_initialization"] and config["pp_checkpoint"] is None
assert update["batch"] == 6 and update["batch_size"] == 8 and update["optimizer_step"] == 1
assert update["finite_loss"] and update["optimizer_step_complete"]
assert gradient["valid_matched_axes"] == 48 and gradient["raw_gate_gradient_l1"] > 0
for module in update["modules"].values():
    assert module["gradient_norm"] > 0 and module["parameter_delta_l2"] > 0
    assert module["updated_tensors"] == module["parameter_tensors"]
files = [
    (trial / "preparation.json", "preparation.json"),
    (trial / "started.json", "started.json"),
    (trial / "progress.json", "progress.json"),
    (trial / "preflight_start.json", "preflight_start.json"),
    (trial / "preflight_complete.json", "preflight_complete.json"),
    (trial / "real_batch_gate_gradient.json", "real_batch_gate_gradient.json"),
    (trial / "control_start.json", "control_start.json"),
    (trial / "remote_cpu_tests.log", "remote_cpu_tests.log"),
    (trial / "preflight.log", "preflight.log"),
    (run / "config.json", "preflight_config.json"),
    (run / "pvground_first_update.json", "preflight_first_update.json"),
    (run / "train_audit_receipt_epoch_1.json", "preflight_train_audit.json"),
]
with tarfile.open(trial / "preflight_evidence.tar.gz", "w:gz") as archive:
    for path, name in files:
        archive.add(path, arcname=name)
print(json.dumps({
    "preflight_complete_utc": complete["completed_utc"],
    "first_update": update, "gate_gradient": gradient,
    "current_condition": (trial / "current_condition").read_text().strip(),
    "archive": str(trial / "preflight_evidence.tar.gz"),
}))
PY
