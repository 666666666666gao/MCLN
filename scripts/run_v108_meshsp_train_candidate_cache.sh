#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
DATA_VIEW=/root/autodl-tmp/DATA_ROOT_mcln_meshsp
CHECKPOINT=/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
OUT="$BASE/train"
LOG="$BASE/candidate_train.log"
EXIT_FILE="$BASE/candidate_train_exitcode.txt"
RECEIPT="$BASE/candidate_train_receipt.json"
PY=/root/miniconda3/envs/bdetr/bin/python

mkdir -p "$BASE"
if [[ -e "$OUT" || -e "$LOG" || -e "$EXIT_FILE" || -e "$RECEIPT" ]]; then
    echo "V108 meshSP train candidate output already exists" >&2
    exit 64
fi
if [[ "$(readlink "$DATA_VIEW/superpoints/train")" != "/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train" ]]; then
    echo "mesh-derived train superpoint view is not active" >&2
    exit 65
fi
if [[ "$(sha256sum /root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_audit.json | cut -d' ' -f1)" != "a118d311a3f7f1a434f06ff61582142178d9f4e740b3e5a6a8b529b4239b9215" ]]; then
    echo "train superpoint audit identity changed" >&2
    exit 66
fi
if [[ "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" != "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208" ]]; then
    echo "epoch71 checkpoint identity changed" >&2
    exit 67
fi
free_bytes=$(df -B1 /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ ! "$free_bytes" =~ ^[0-9]+$ || "$free_bytes" -lt 4000000000 ]]; then
    echo "insufficient free disk before V108 candidate cache: $free_bytes" >&2
    exit 68
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"

"$PY" scripts/cache_scanrefer_rec_candidates.py \
    --split train \
    --data-root "$DATA_VIEW/" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUT" \
    --batch-size 12 \
    --num-workers 4 \
    --shard-size 256 \
    --max-candidates 16 \
    --device cuda:0 >"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
if [[ "$rc" -ne 0 ]]; then
    chmod 0444 "$LOG" "$EXIT_FILE"
    exit "$rc"
fi

"$PY" - "$OUT" "$RECEIPT" "$LOG" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1]).resolve()
receipt = Path(sys.argv[2]).resolve()
log = Path(sys.argv[3]).resolve()
manifest_path = out / "manifest.json"
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
expected = {
    "split": "train",
    "dataset_size": 36665,
    "source_dataset_size": 36665,
    "sample_count": 36665,
    "checkpoint_epoch": 71,
    "checkpoint_sha256": "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208",
    "feature_schema_version": "rec-query-v1",
    "feature_dim": 152,
    "deterministic": True,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"candidate manifest {key} mismatch")
if Path(manifest.get("data_root", "")).resolve() != Path(
        "/root/autodl-tmp/DATA_ROOT_mcln_meshsp").resolve():
    raise RuntimeError("candidate manifest data root mismatch")
if manifest.get("candidate_rule") != {"topk_per_source": 8, "max_candidates": 16}:
    raise RuntimeError("candidate rule mismatch")
shards = manifest.get("shards")
if not isinstance(shards, list) or len(shards) != 144:
    raise RuntimeError("candidate shard count mismatch")
sequence = hashlib.sha256()
total_bytes = 0
total_rows = 0
expected_files = {"manifest.json"}
for index, descriptor in enumerate(shards):
    name = f"shard_{index:06d}.pt"
    if descriptor != name:
        raise RuntimeError("candidate shard sequence mismatch")
    path = out / name
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    decoded = torch.load(path, map_location="cpu")
    rows = decoded.get("rows") if isinstance(decoded, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"candidate shard rows missing: {name}")
    total_bytes += len(payload)
    total_rows += len(rows)
    sequence.update(name.encode("ascii") + b"\0")
    sequence.update(str(len(payload)).encode("ascii") + b"\0")
    sequence.update(digest.encode("ascii") + b"\n")
    expected_files.add(name)
if total_rows != 36665:
    raise RuntimeError("candidate shard row total mismatch")
actual_files = {path.name for path in out.iterdir() if path.is_file()}
if actual_files != expected_files:
    raise RuntimeError("candidate output contains unexpected files")
log_text = log.read_text(encoding="utf-8", errors="replace")
if "Cached samples: 36665" not in log_text:
    raise RuntimeError("candidate completion line missing")
payload = {
    "schema": "mcln-v108-meshsp-train-candidate-cache-receipt-v1",
    "version": 1,
    "validation_data_accessed": False,
    "split": "train",
    "sample_count": 36665,
    "scene_count": 562,
    "extraction_batch_size": 12,
    "data_root": "/root/autodl-tmp/DATA_ROOT_mcln_meshsp",
    "train_superpoint_manifest_sha256": "95c11c2714c2d67d3059b3de0e9d57a9eb717273ee66d2c98d35f18d4218869f",
    "train_superpoint_audit_sha256": "a118d311a3f7f1a434f06ff61582142178d9f4e740b3e5a6a8b529b4239b9215",
    "checkpoint_sha256": expected["checkpoint_sha256"],
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "shard_count": len(shards),
    "shard_bytes": total_bytes,
    "shard_sequence_sha256": sequence.hexdigest(),
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
fd = os.open(str(receipt), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
try:
    os.write(fd, raw)
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps(payload, sort_keys=True))
PY
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
if [[ "$rc" -ne 0 ]]; then
    chmod 0444 "$LOG" "$EXIT_FILE"
    exit "$rc"
fi

find "$OUT" -maxdepth 1 -type f -exec chmod 0444 {} +
chmod 0555 "$OUT"
chmod 0444 "$LOG" "$EXIT_FILE" "$RECEIPT"
exit 0
