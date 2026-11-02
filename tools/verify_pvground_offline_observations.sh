#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
archive="$root/pvground_offline_observations_v2.tar.gz"
staging="$root/offline_audits/observations_v2"
sha256sum "$archive"
mkdir -p "$staging"
tar -xzf "$archive" -C "$staging"
cd "$staging"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$staging/test_dependencies:$staging" \
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import pytest
import torch

torch.set_num_threads(2)
result = pytest.main([
    "-q", "tests/test_pvground_forward_audit.py",
    "tests/test_pvground_semantic_audit.py",
])
receipt = {
    "status": "passed" if result == 0 else "failed",
    "pytest_exit_code": int(result),
    "torch": torch.__version__,
    "pytest": pytest.__version__,
    "cuda_initialized": torch.cuda.is_initialized(),
    "test_scope": "CPU fixtures; not actual model forward or formal accuracy",
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path("tools/pvground_forward_audit.py"),
            Path("tools/pvground_semantic_audit.py"),
            Path("tools/eval_pvground_audit.py"),
            Path("tests/test_pvground_forward_audit.py"),
            Path("tests/test_pvground_semantic_audit.py"),
            Path("models/losses.py"),
            Path("src/grounding_evaluator.py"),
        )
    },
}
Path("remote-tests.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True))
raise SystemExit(int(result))
PY
