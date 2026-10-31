set -e
printf '%s  %s\n' \
  3a27dd986f55e02ced18d4e1f0c233071174f6d016385e1ff450c20970af2aad \
  /tmp/sr3d_replay_source_delta.tgz | sha256sum -c -
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess

source = Path("/root/autodl-tmp/pvground_sr3d_module_training_20261030/code")
destination = Path("/root/autodl-tmp/pvground_sr3d_replay_20261031/code")
manifest = json.loads(Path("/tmp/sr_host_sr3d_code_manifest.json").read_text())
assert len(manifest["files"]) == 33
replaced = {"native_model_initialization.py", "train_dist_mod.py"}
for name, digest in manifest["files"].items():
    if name not in replaced:
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
destination.mkdir(parents=True)
for path in source.iterdir():
    if path.name not in replaced | {"init_sr3d.json", "native_metrics.jsonl", "__pycache__"}:
        (destination / path.name).symlink_to(path, target_is_directory=path.is_dir())
subprocess.run(["tar", "-xzf", "/tmp/sr3d_replay_source_delta.tgz",
                "-C", str(destination)], check=True)
for name, digest in manifest["files"].items():
    assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest, name
spec = json.loads((destination / "init_sr3d.json").read_text())
assert hashlib.sha256(Path(spec["pretrained_checkpoint"]).read_bytes()).hexdigest() == spec["pretrained_checkpoint_sha256"]
rows = [json.loads(line) for line in (destination / "native_metrics.jsonl").read_text().splitlines()]
assert all(row["rows"] == 17726 and row["primary_score"] == "last/bbf"
           and row["same_complete_model"] for row in rows)
print(json.dumps(dict(
    diagnostic_code=str(destination), verified_python_files=len(manifest["files"]),
    checkpoint_sha_verified=True, reference_epochs=[row["epoch"] for row in rows],
    gpu_used=False, training_sources_modified=False)))
PY
