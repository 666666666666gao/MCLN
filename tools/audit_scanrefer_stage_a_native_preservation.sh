#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
stage = root / "scanrefer_restart_20261102"
official = root / "checkpoints/ckpt_epoch_54.pth"
selected = stage / "stage_b_initialization_epoch_3.pth"
prefixes = ("decoder.5.local_visual.", "pvground_support_corrector.",
            "pvground_span_mixer.")
expected_hashes = (
    "a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d",
    "f4e98cce830eb474a94402df00dbd160c379b963e7d285c759b86c4a60b0a898",
)
identities = []
for path, expected in zip((official, selected), expected_hashes):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == expected
    identities.append({"path": str(path), "sha256": digest.hexdigest()})
original = torch.load(official, map_location="cpu", weights_only=False)
trained = torch.load(selected, map_location="cpu", weights_only=False)
assert original["epoch"] == 54 and trained["epoch"] == 3
assert trained["config"].pvground_train_only
native = {name.removeprefix("module."): value
          for name, value in original["model"].items()
          if not name.endswith("embeddings.position_ids")}
actual = {name.removeprefix("module."): value
          for name, value in trained["model"].items()}
assert len(native) == 1134 and len(actual) == 1168
assert not (set(native) - set(actual))
assert all(name.startswith(prefixes) for name in set(actual) - set(native))
changed = []
for name, value in native.items():
    other = actual[name]
    assert value.shape == other.shape and value.dtype == other.dtype
    if not torch.equal(value, other):
        changed.append(name)
bn_buffer_names = [name for name in native if name.endswith(
    ("running_mean", "running_var", "num_batches_tracked"))]
assert bn_buffer_names
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoints": identities,
    "official_epoch": 54, "selected_epoch": 3,
    "native_tensors": len(native),
    "additional_module_tensors": len(actual) - len(native),
    "native_bn_buffer_count": len(bn_buffer_names),
    "changed_native_tensor_names": changed,
    "native_exactly_unchanged": not changed,
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": "Exact serialized tensor equality for the complete Stage A selected checkpoint versus official initialization. Includes native parameters and BatchNorm buffers. No model import, construction, forward or evaluation; no claim about Stage B or individual module causality.",
}
assert not receipt["cuda_initialized"]
destination = stage / "stage_a_native_preservation.json"
destination.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
