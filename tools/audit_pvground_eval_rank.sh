#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
cd "$root/source"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$root/source/pointnet2:$root/source" \
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
from pathlib import Path
import sys
import torch
from main_utils import parse_option

base = [
    "eval_pvground_audit.py", "--local-rank=0",
    "--eval", "--use_pvground_innovations",
    "--dataset", "scanrefer", "--test_dataset", "scanrefer",
]
sys.argv = base
before = parse_option()
sys.argv = base + ["--local_rank", "0"]
after = parse_option()
assert before.local_rank == 1
assert after.local_rank == 0
assert not torch.cuda.is_initialized()
receipt = {
    "status": "confirmed",
    "launcher_hyphenated_flag": "--local-rank=0",
    "without_explicit_underscore_flag": before.local_rank,
    "with_explicit_underscore_flag": after.local_rank,
    "cuda_initialized": torch.cuda.is_initialized(),
    "limitation": "Argument parsing only; no CUDA device selection or model forward.",
}
destination = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/observations_v2")
(destination / "rank-parser-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY
