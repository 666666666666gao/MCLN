set -e
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/pvground_sr3d_module_training_20261030")
run = root / "logs/sr3d/sr3d_module_ft/1793419200"
spec = json.loads((root / "code/init_sr3d.json").read_text())
author_path = Path(spec["pretrained_checkpoint"])
assert hashlib.sha256(author_path.read_bytes()).hexdigest() == spec["pretrained_checkpoint_sha256"]
author = torch.load(author_path, map_location="cpu", weights_only=False)["model"]
trained_path = run / "best_trained_acc025_epoch2.pth"
checkpoint = torch.load(trained_path, map_location="cpu", weights_only=False)
assert checkpoint["epoch"] == 2
trainable = json.loads((run / "trainable_parameters.json").read_text())
state = checkpoint["model"]
assert all(name.startswith("module.") for name in state)
frozen = {name: value for name, value in state.items() if name[7:] not in trainable}
assert len(frozen) == 1234
assert set(author) - set(frozen) == {"module.text_encoder.embeddings.position_ids"}
assert set(frozen).issubset(author)
changed = [name for name, tensor in frozen.items()
           if tensor.shape != author[name].shape or tensor.dtype != author[name].dtype
           or not torch.equal(tensor, author[name])]
metrics = [json.loads(line) for line in (run / "native_metrics.jsonl").read_text().splitlines()]
own = next(row for row in metrics if row["epoch"] == checkpoint["epoch"])
assert own["rows"] == 17726 and own["primary_score"] == "last/bbf" and own["same_complete_model"]
result = dict(
    diagnostic="author_to_trained_frozen_state_comparison", gpu_used=False,
    author_checkpoint=str(author_path), author_sha256=spec["pretrained_checkpoint_sha256"],
    trained_checkpoint=str(trained_path), epoch=checkpoint["epoch"], own_epoch_metrics=own,
    frozen_tensors_compared=len(frozen), changed_frozen_tensors=changed,
    frozen_state_unchanged=not changed,
    author_only_tensor="module.text_encoder.embeddings.position_ids",
    trainable_tensors_excluded=len(trainable),
    limits="Checks frozen state against author weights; does not test forward identity, module initialization deltas or causal accuracy contribution.")
output = root / "diagnostics/author_to_epoch2_frozen_cpu_audit.json"
assert not output.exists()
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
print("SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
PY
