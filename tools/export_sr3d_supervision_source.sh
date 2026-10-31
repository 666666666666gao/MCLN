set -e
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("/root/autodl-tmp/pvground_sr3d_module_training_20261030/code")
ranges = {
    "selected_query_mask_objective.py": None,
    "native_root_bbs.py": None,
    "pvground_semantic_assignment.py": None,
    "native_mask_geometry.py": None,
    "extremal_span_mixer.py": None,
    "pvground_task_observation_query.py": None,
    "native_model_initialization.py": None,
    "models/losses.py": (850, 990),
    "src/grounding_evaluator.py": (315, 405),
}
files = []
for name, extent in ranges.items():
    raw = (root / name).read_bytes()
    lines = raw.decode().splitlines()
    start, end = extent if extent else (1, len(lines))
    files.append(dict(path=str(root / name), sha256=hashlib.sha256(raw).hexdigest(),
                      start_line=start, end_line=min(end, len(lines)),
                      excerpt="\n".join(lines[start - 1:end])))
report = dict(observed_at=datetime.now(timezone.utc).isoformat(),
              dataset="sr3d", run="1793419200", diagnostic="read_only_supervision_source",
              gpu_used=False, files=files)
output = Path("/tmp/sr3d_supervision_source_audit.json")
output.write_text(json.dumps(report, indent=2) + "\n")
print("SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
PY
