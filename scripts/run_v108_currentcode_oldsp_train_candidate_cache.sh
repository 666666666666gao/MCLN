#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
DATA_ROOT=/root/autodl-tmp/DATA_ROOT
CHECKPOINT="$DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
BASE="$DATA_ROOT/output/rec_reranker/e71_top16_currentcode_oldsp"
OUT="$BASE/train"
LOG="$BASE/candidate_train.log"
EXIT_FILE="$BASE/candidate_train_exitcode.txt"
RECEIPT="$BASE/candidate_train_receipt.json"
PY=/root/miniconda3/envs/bdetr/bin/python
OLD_SP="$DATA_ROOT/superpoints/train"
OLD_SP_MANIFEST_SHA=365aa6a6dd2184fdf581b650bedf4996e4a16b450d1ca027b3a9274134fc59e6

if [[ -e "$BASE" ]]; then
    echo "V108 current-code old-SP control output already exists" >&2
    exit 64
fi
if [[ "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" != "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208" ]]; then
    echo "epoch71 checkpoint identity changed" >&2
    exit 65
fi
declare -A expected_sources=(
    [scripts/cache_scanrefer_rec_candidates.py]=b1db28f1ea3d036231b22643890033e96c175bc7a44eb914c2e6f564beef1932
    [train_dist_mod.py]=34a6ed34ffc09979479deb4b5b4c72cf0c6a98ef6c768e9a5630d652bb754078
    [models/mcln.py]=3bcd2387bb75e7d18666ebfe74af785947121f590a44dd157e88b2ff1d2c0c3c
    [models/rec_candidate_adapter.py]=dfc5afaa6ca4feabc67417707660f4f881594f9fad11663a14f56b9be26c10a3
    [models/mask_fusion.py]=1d0d5026d2c5e63d68408488cbdcf1036819f457fc00482620248d6fda0f2a97
    [src/joint_det_dataset.py]=567fda4b19b90f4e183524c19edcaa66c5a155ccb09835b78c7c5110c9fa386a
)
for path in "${!expected_sources[@]}"; do
    if [[ "$(sha256sum "$ROOT/$path" | cut -d' ' -f1)" != "${expected_sources[$path]}" ]]; then
        echo "V108 control source identity changed: $path" >&2
        exit 66
    fi
done
if [[ "$(find "$OLD_SP" -maxdepth 1 -type f | wc -l)" != "1201" ]]; then
    echo "old train superpoint file count changed" >&2
    exit 67
fi
old_sp_digest=$(
    cd "$OLD_SP" &&
    find . -maxdepth 1 -type f -printf '%f\0' |
        sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
)
if [[ "$old_sp_digest" != "$OLD_SP_MANIFEST_SHA" ]]; then
    echo "old train superpoint content identity changed" >&2
    exit 68
fi
free_bytes=$(df -B1 /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ ! "$free_bytes" =~ ^[0-9]+$ || "$free_bytes" -lt 4000000000 ]]; then
    echo "insufficient free disk before V108 control cache: $free_bytes" >&2
    exit 69
fi
compute_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$compute_count" -ne 0 ]]; then
    echo "GPU0 already has a compute process" >&2
    exit 70
fi

mkdir -p "$BASE"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"

"$PY" scripts/cache_scanrefer_rec_candidates.py \
    --split train \
    --data-root "$DATA_ROOT/" \
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

"$PY" - "$OUT" "$RECEIPT" "$LOG" "$OLD_SP_MANIFEST_SHA" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1]).resolve()
receipt = Path(sys.argv[2]).resolve()
log = Path(sys.argv[3]).resolve()
old_sp_manifest_sha = sys.argv[4]
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
        raise RuntimeError(f"control candidate manifest {key} mismatch")
if Path(manifest.get("data_root", "")).resolve() != Path(
        "/root/autodl-tmp/DATA_ROOT").resolve():
    raise RuntimeError("control candidate data root mismatch")
if manifest.get("candidate_rule") != {"topk_per_source": 8, "max_candidates": 16}:
    raise RuntimeError("control candidate rule mismatch")
shards = manifest.get("shards")
if not isinstance(shards, list) or len(shards) != 144:
    raise RuntimeError("control candidate shard count mismatch")
sequence = hashlib.sha256()
total_bytes = 0
total_rows = 0
counts = {
    "default_hits025": 0,
    "default_hits050": 0,
    "oracle_hits025": 0,
    "oracle_hits050": 0,
}
expected_files = {"manifest.json"}
for index, descriptor in enumerate(shards):
    name = f"shard_{index:06d}.pt"
    if descriptor != name:
        raise RuntimeError("control candidate shard sequence mismatch")
    path = out / name
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    decoded = torch.load(path, map_location="cpu")
    rows = decoded.get("rows") if isinstance(decoded, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"control candidate shard rows missing: {name}")
    for row in rows:
        valid = row["valid_mask"].bool()
        ious = row["candidate_ious"].float()
        queries = row["query_indices"].long()
        default = queries.eq(int(row["default_top1_query_index"])) & valid
        if not bool(default.any().item()):
            raise RuntimeError("control row lacks a default candidate")
        default_iou = float(ious.masked_fill(~default, -1.0).max().item())
        oracle_iou = float(ious.masked_fill(~valid, -1.0).max().item())
        counts["default_hits025"] += int(default_iou > 0.25)
        counts["default_hits050"] += int(default_iou > 0.50)
        counts["oracle_hits025"] += int(oracle_iou > 0.25)
        counts["oracle_hits050"] += int(oracle_iou > 0.50)
    total_bytes += len(payload)
    total_rows += len(rows)
    sequence.update(name.encode("ascii") + b"\0")
    sequence.update(str(len(payload)).encode("ascii") + b"\0")
    sequence.update(digest.encode("ascii") + b"\n")
    expected_files.add(name)
if total_rows != 36665:
    raise RuntimeError("control candidate shard row total mismatch")
expected_counts = {
    "default_hits025": 34892,
    "default_hits050": 31870,
    "oracle_hits025": 36405,
    "oracle_hits050": 35409,
}
if counts != expected_counts:
    raise RuntimeError(f"control candidate metric counts changed: {counts}")
if {path.name for path in out.iterdir() if path.is_file()} != expected_files:
    raise RuntimeError("control candidate output contains unexpected files")
log_text = log.read_text(encoding="utf-8", errors="replace")
if "Cached samples: 36665" not in log_text:
    raise RuntimeError("control candidate completion line missing")
sources = {
    "scripts/cache_scanrefer_rec_candidates.py": "b1db28f1ea3d036231b22643890033e96c175bc7a44eb914c2e6f564beef1932",
    "train_dist_mod.py": "34a6ed34ffc09979479deb4b5b4c72cf0c6a98ef6c768e9a5630d652bb754078",
    "models/mcln.py": "3bcd2387bb75e7d18666ebfe74af785947121f590a44dd157e88b2ff1d2c0c3c",
    "models/rec_candidate_adapter.py": "dfc5afaa6ca4feabc67417707660f4f881594f9fad11663a14f56b9be26c10a3",
    "models/mask_fusion.py": "1d0d5026d2c5e63d68408488cbdcf1036819f457fc00482620248d6fda0f2a97",
    "src/joint_det_dataset.py": "567fda4b19b90f4e183524c19edcaa66c5a155ccb09835b78c7c5110c9fa386a",
}
payload = {
    "schema": "mcln-v108-currentcode-oldsp-train-candidate-cache-receipt-v1",
    "version": 1,
    "validation_data_accessed": False,
    "control_role": "same_current_code_old_superpoints_causal_baseline",
    "sample_count": 36665,
    "scene_count": 562,
    "extraction_batch_size": 12,
    "data_root": "/root/autodl-tmp/DATA_ROOT",
    "old_train_superpoint_manifest_sha256": old_sp_manifest_sha,
    "checkpoint_sha256": expected["checkpoint_sha256"],
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "shard_count": len(shards),
    "shard_bytes": total_bytes,
    "shard_sequence_sha256": sequence.hexdigest(),
    "exact_candidate_metric_counts": counts,
    "source_sha256": sources,
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
fd = os.open(str(receipt), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
try:
    written = 0
    while written < len(raw):
        count = os.write(fd, raw[written:])
        if count <= 0:
            raise OSError("V108 control receipt write made no progress")
        written += count
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
